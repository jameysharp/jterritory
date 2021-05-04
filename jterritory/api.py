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
