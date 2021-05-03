from pydantic import Field
from typing import List
from . import SetError


class Forbidden(SetError):
    pass


class OverQuota(SetError):
    pass


class TooLarge(SetError):
    pass


class RateLimit(SetError):
    pass


class NotFound(SetError):
    pass


class InvalidPatch(SetError):
    pass


class WillDestroy(SetError):
    pass


class InvalidProperties(SetError):
    properties: List[str] = Field(default_factory=list)


class Singleton(SetError):
    pass


class AlreadyExists(SetError):
    existing_id: str
