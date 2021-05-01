from typing import Any
from . import MethodError


class ServerUnavailable(MethodError):
    name = "serverUnavailable"


class ServerFail(MethodError):
    name = "serverFail"

    def __init__(self, description: str, **kwargs: Any) -> None:
        super().__init__(description=description, **kwargs)


class ServerPartialFail(MethodError):
    name = "serverPartialFail"


class UnknownMethod(MethodError):
    name = "unknownMethod"


class InvalidArguments(MethodError):
    name = "invalidArguments"

    def __init__(self, description: str, **kwargs: Any) -> None:
        super().__init__(description=description, **kwargs)


class InvalidResultReference(MethodError):
    name = "invalidResultReference"


class Forbidden(MethodError):
    name = "forbidden"


class AccountNotFound(MethodError):
    name = "accountNotFound"


class AccountNotSupportedByMethod(MethodError):
    name = "accountNotSupportedByMethod"


class AccountReadOnly(MethodError):
    name = "accountReadOnly"


class RequestTooLarge(MethodError):
    name = "requestTooLarge"


class CannotCalculateChanges(MethodError):
    name = "cannotCalculateChanges"


class StateMismatch(MethodError):
    name = "stateMismatch"


class FromAccountNotFound(MethodError):
    name = "fromAccountNotFound"


class FromAccountNotSupportedByMethod(MethodError):
    name = "fromAccountNotSupportedByMethod"


class AnchorNotFound(MethodError):
    name = "anchorNotFound"


class UnsupportedSort(MethodError):
    name = "unsupportedSort"


class UnsupportedFilter(MethodError):
    name = "unsupportedFilter"


class TooManyChanges(MethodError):
    name = "tooManyChanges"
