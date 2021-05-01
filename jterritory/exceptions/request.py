from typing import Any
from . import RequestError


class UnknownCapability(RequestError):
    name = "urn:ietf:params:jmap:error:unknownCapability"


class NotJSON(RequestError):
    name = "urn:ietf:params:jmap:error:notJSON"


class NotRequest(RequestError):
    name = "urn:ietf:params:jmap:error:notRequest"


class Limit(RequestError):
    name = "urn:ietf:params:jmap:error:limit"

    def __init__(self, limit: str, **kwargs: Any) -> None:
        super().__init__(limit=limit, **kwargs)
