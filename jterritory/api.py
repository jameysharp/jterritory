from dataclasses import dataclass, field
import json
from pydantic import ValidationError
import re
from sqlalchemy import select
from sqlalchemy.future import Connection, Engine
import typing
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Type
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
    created_ids: Dict[Id, Optional[ObjectId]] = field(default_factory=dict)
    method_responses: List[Invocation] = field(default_factory=list)
    call_id: String = ""

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

    def add_response(self, name: String, arguments: BaseModel) -> None:
        self.method_responses.append(
            Invocation(
                name=name,
                arguments=arguments.dict(
                    by_alias=True,
                    exclude_defaults=True,
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

    def resolve(self, ctx: Context) -> Any:
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
    def resolve_path(cls, within: Any, tokens: List[str]) -> Any:
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


@dataclass
class Endpoint:
    capabilities: Set[str]
    methods: Dict[str, Callable[[Context, BaseModel], None]]
    engine: Engine

    def request(self, body: bytes) -> BaseModel:
        try:
            raw_request = json.loads(body.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return request.NotJSON(detail=str(exc))

        try:
            parsed = Request.parse_obj(raw_request)
        except ValidationError as exc:
            return request.NotRequest(detail=str(exc))

        unknown = parsed.using - self.capabilities
        if unknown:
            return request.UnknownCapability(detail=str(unknown))

        with self.engine.connect() as connection:
            ctx = Context(connection=connection)

            if parsed.created_ids:
                for name, object_id in parsed.created_ids.items():
                    ctx.created_ids[name] = object_id

            for invocation in parsed.method_calls:
                ctx.call_id = invocation.call_id

                try:
                    try:
                        handler = self.methods[invocation.name]
                    except KeyError:
                        raise method.UnknownMethod().exception()
                    hints = typing.get_type_hints(handler)

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

                    try:
                        arguments = hints["request"].parse_obj(raw_arguments)
                    except ValidationError as exc:
                        raise method.InvalidArguments(
                            description=str(exc)
                        ).exception() from exc

                    handler(ctx, arguments)
                except exceptions.MethodException as exc:
                    ctx.add_response("error", exc.args[0])
                except Exception as exc:
                    # TODO: log these exceptions
                    print(exc)
                    ctx.add_response("error", method.ServerFail())

        created_ids = {}
        if parsed.created_ids is not None:
            created_ids = {k: v for k, v in ctx.created_ids.items() if v is not None}

        return Response(
            method_responses=ctx.method_responses,
            created_ids=created_ids,
            session_state="TODO",
        )
