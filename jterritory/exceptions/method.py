from . import MethodError


class ServerUnavailable(MethodError):
    pass


class ServerFail(MethodError):
    description: str


class ServerPartialFail(MethodError):
    pass


class UnknownMethod(MethodError):
    pass


class InvalidArguments(MethodError):
    description: str


class InvalidResultReference(MethodError):
    pass


class Forbidden(MethodError):
    pass


class AccountNotFound(MethodError):
    pass


class AccountNotSupportedByMethod(MethodError):
    pass


class AccountReadOnly(MethodError):
    pass


class RequestTooLarge(MethodError):
    pass


class CannotCalculateChanges(MethodError):
    pass


class StateMismatch(MethodError):
    pass


class FromAccountNotFound(MethodError):
    pass


class FromAccountNotSupportedByMethod(MethodError):
    pass


class AnchorNotFound(MethodError):
    pass


class UnsupportedSort(MethodError):
    pass


class UnsupportedFilter(MethodError):
    pass


class TooManyChanges(MethodError):
    pass
