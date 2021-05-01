from typing import Any, List
from . import SetError


class Forbidden(SetError):
    name = "forbidden"


class OverQuota(SetError):
    name = "overQuota"


class TooLarge(SetError):
    name = "tooLarge"


class RateLimit(SetError):
    name = "rateLimit"


class NotFound(SetError):
    name = "notFound"


class InvalidPatch(SetError):
    name = "invalidPatch"


class WillDestroy(SetError):
    name = "willDestroy"


class InvalidProperties(SetError):
    name = "invalidProperties"

    def __init__(self, properties: List[str] = [], **kwargs: Any) -> None:
        if properties:
            kwargs["properties"] = properties
        super().__init__(**kwargs)


class Singleton(SetError):
    name = "singleton"


class AlreadyExists(SetError):
    name = "alreadyExists"

    def __init__(self, existingId: str, **kwargs: Any) -> None:
        super().__init__(existingId=existingId, **kwargs)
