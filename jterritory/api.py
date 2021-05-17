# FIXME: Pydantic seems broken under `from __future__ import
# annotations` when using a `NamedTuple` as a field in a Pydantic model
# https://github.com/samuelcolvin/pydantic/issues/2760

from dataclasses import dataclass, field
from functools import wraps
import json
from pydantic import ValidationError
import re
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.future import Connection, Engine
from traceback import print_exc
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Type, TypeVar
from . import exceptions, models
from .exceptions import method, request
from .types import BaseModel, Id, JSONPointer, ObjectId, String


class Invocation(NamedTuple):
    "https://tools.ietf.org/html/rfc8620#section-3.2"
    name: String
    arguments: dict
    call_id: String


class Request(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-3.3"
    using: Set[String]
    method_calls: List[Invocation]
    created_ids: Optional[Dict[Id, ObjectId]]


class Response(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-3.4"
    method_responses: List[Invocation]
    created_ids: Optional[Dict[Id, ObjectId]]
    session_state: String


@dataclass
class Context:
    "Stores any number of responses from a single method call."
    connection: Connection
    account_cache: Dict[Id, int] = field(default_factory=dict)
    created_ids: Dict[Id, ObjectId] = field(default_factory=dict)
    method_responses: List[Invocation] = field(default_factory=list)
    call_id: String = String("")

    def use_account(
        self,
        account_id: Id,
        error: Type[exceptions.MethodError] = method.AccountNotFound,
    ) -> int:
        """
        Returns the internal database ID for the specified account, or
        raises a method error if the account does not exist. By default
        the error raised is accountNotFound, but for example
        fromAccountNotFound can be specified instead.

        The result is cached within this Context object, so subsequent
        methods within the same request won't query the database for
        this account again.
        """
        try:
            result = self.account_cache[account_id]
        except KeyError:
            result = self.connection.scalar(
                select(models.accounts.c.id).where(
                    models.accounts.c.account == account_id
                )
            )
            if result is None:
                raise error().exception()
            self.account_cache[account_id] = result
        return result

    def add_response(self, name: str, arguments: BaseModel) -> None:
        self.method_responses.append(
            Invocation(
                name=String(name),
                arguments=arguments.dict(
                    by_alias=True,
                    exclude_none=True,
                ),
                call_id=self.call_id,
            )
        )


class StrictJSONPointer(JSONPointer):
    regex = re.compile("^/")


class ResultReference(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-3.7"
    result_of: String
    name: String
    path: StrictJSONPointer

    class Config:
        extra = "forbid"

    def resolve(self, ctx: Context) -> object:
        for response in ctx.method_responses:
            if response.call_id == self.result_of:
                break
        else:
            raise method.InvalidResultReference().exception()

        # XXX: why not look for first response where name matches too?
        if response.name != self.name:
            raise method.InvalidResultReference().exception()

        try:
            return self.resolve_path(response.arguments, self.path.reference_tokens())
        except (IndexError, KeyError) as exc:
            raise method.InvalidResultReference().exception() from exc

    @classmethod
    def resolve_path(cls, within: object, tokens: List[str]) -> object:
        it = iter(tokens)
        for token in it:
            if isinstance(within, list):
                if token == "*":
                    # Wildcards are special.
                    break
                if not re.fullmatch(r"0|[1-9][0-9]*", token):
                    raise IndexError(token)
                within = within[int(token)]
            elif isinstance(within, dict):
                within = within[token]
            else:
                raise KeyError(token)
        else:
            # No wildcards found, so return the last selected value.
            return within

        # We've navigated down to some list, where we've been asked to
        # evaluate the remaining part of the path on all elements of the
        # list. The above loop consumed the iterator up through the "*".
        tokens = list(it)
        result = []
        for e in within:
            v = cls.resolve_path(e, tokens)
            if isinstance(v, list):
                result.extend(v)
            else:
                result.append(v)
        return result


RequestModel = TypeVar("RequestModel", bound=BaseModel)
MethodHandler = Callable[[Context, RequestModel], None]
GenericMethod = Callable[[Context, Any], None]


def make_method(
    model: Type[RequestModel], handler: MethodHandler[RequestModel]
) -> GenericMethod:
    @wraps(handler)
    def call_method(ctx: Context, raw_arguments: Any) -> None:
        try:
            arguments = model.parse_obj(raw_arguments)
        except ValidationError as exc:
            raise method.InvalidArguments(description=str(exc)).exception() from exc

        handler(ctx, arguments)

    return call_method


def serializable(f: "MethodHandler[RequestModel]") -> "MethodHandler[RequestModel]":
    @wraps(f)
    def wrapper(ctx: Context, request: RequestModel) -> None:
        dialect = ctx.connection.engine.name
        if dialect == "sqlite":
            begin = "BEGIN IMMEDIATE"
        elif dialect == "postgresql":
            begin = "BEGIN ISOLATION LEVEL SERIALIZABLE"
        else:
            raise NotImplementedError(f"unsupported database: {dialect!r}")

        while True:
            ctx.connection.exec_driver_sql(begin)
            try:
                f(ctx, request)
            except Exception as exc:
                # Retry if this is a Postgres "serialization_failure" or
                # "deadlock_detected", assuming the backend is psycopg2.
                if dialect == "postgresql" and isinstance(exc, OperationalError):
                    if exc.orig.pgcode in ("40001", "40P01"):
                        continue
                ctx.connection.exec_driver_sql("ROLLBACK")
                raise exc
            else:
                ctx.connection.exec_driver_sql("COMMIT")
            break

    return wrapper


@dataclass
class Endpoint:
    capabilities: Set[String]
    methods: Dict[str, GenericMethod]
    engine: Engine

    def __post_init__(self) -> None:
        self.engine = self.engine.execution_options(isolation_level="AUTOCOMMIT")

    def request(self, body: bytes) -> BaseModel:
        try:
            raw_request = json.loads(body.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return request.NotJSON(detail=str(exc))

        # Pydantic tries to interpret lists as dictionaries, which I
        # don't want. That behavior also allows clients to trigger
        # https://github.com/samuelcolvin/pydantic/issues/2762, though
        # this check isn't sufficient to block all cases of that issue.
        if not isinstance(raw_request, dict):
            return request.NotRequest(detail="JSON root is not an object")

        try:
            parsed = Request.parse_obj(raw_request)
        except (ValidationError, TypeError) as exc:
            return request.NotRequest(detail=str(exc))

        unknown = parsed.using - self.capabilities
        if unknown:
            return request.UnknownCapability(detail=str(unknown))

        with self.engine.connect() as connection:
            ctx = Context(connection=connection)

            if parsed.created_ids:
                ctx.created_ids = parsed.created_ids

            for invocation in parsed.method_calls:
                ctx.call_id = invocation.call_id

                try:
                    try:
                        handler = self.methods[invocation.name]
                    except KeyError:
                        raise method.UnknownMethod().exception()

                    raw_arguments = {}
                    for k, v in invocation.arguments.items():
                        if k.startswith("#"):
                            k = k[1:]
                            try:
                                ref = ResultReference.parse_obj(v)
                            except ValidationError as exc:
                                raise method.InvalidResultReference().exception() from exc
                            v = ref.resolve(ctx)

                        if k in raw_arguments:
                            raise method.InvalidArguments(
                                description=f"found both {k} and #{k}"
                            ).exception()
                        raw_arguments[k] = v

                    handler(ctx, raw_arguments)
                except exceptions.MethodException as exc:
                    ctx.add_response("error", exc.args[0])
                except Exception:
                    # TODO: log these exceptions
                    print_exc()
                    ctx.add_response("error", method.ServerFail())

        created_ids = None
        if parsed.created_ids is not None:
            created_ids = ctx.created_ids

        return Response(
            method_responses=ctx.method_responses,
            created_ids=created_ids,
            session_state=String("TODO"),
        )
