from __future__ import annotations

from . import RequestError


class UnknownCapability(RequestError):
    pass


class NotJSON(RequestError):
    pass


class NotRequest(RequestError):
    pass


class Limit(RequestError):
    limit: str
