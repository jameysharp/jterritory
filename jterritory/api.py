from dataclasses import dataclass, field
from sqlalchemy.future import Connection
from typing import Dict, List, NamedTuple, Optional, Set
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
    created_ids: Dict[Id, Optional[Id]]
    call_id: String
    method_responses: List[Invocation] = field(default_factory=list)

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
