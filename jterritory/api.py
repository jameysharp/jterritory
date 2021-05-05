from dataclasses import dataclass, field
from sqlalchemy import select
from sqlalchemy.future import Connection
from typing import Dict, List, NamedTuple, Optional, Set, Type
from . import exceptions, models
from .exceptions import method
from .types import BaseModel, Id, String


class Invocation(NamedTuple):
    "https://tools.ietf.org/html/rfc8620#section-3.2"
    name: String
    arguments: dict
    call_id: String


class Request(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-3.3"
    using: Set[String]
    method_calls: List[Invocation]
    created_ids: Optional[Dict[Id, Id]]


class Response(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-3.4"
    method_responses: List[Invocation]
    created_ids: Optional[Dict[Id, Id]]
    session_state: String


@dataclass
class Context:
    "Stores any number of responses from a single method call."
    connection: Connection
    call_id: String
    account_cache: Dict[Id, int] = field(default_factory=dict)
    created_ids: Dict[Id, Optional[Id]] = field(default_factory=dict)
    method_responses: List[Invocation] = field(default_factory=list)

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
