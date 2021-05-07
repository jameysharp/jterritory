from __future__ import annotations

from sqlalchemy.sql import ClauseElement
from typing import Any, Callable, NamedTuple, Optional, Tuple, TypeVar
from .. import models
from ..exceptions import method
from ..types import BaseModel, String, Boolean


class TypedKey(NamedTuple):
    compatible: Tuple[type, ...]
    obj: Any

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, TypedKey):
            raise method.UnsupportedSort().exception()
        if not isinstance(other.obj, self.compatible):
            raise method.UnsupportedSort().exception()
        return self.obj < other.obj

    def descending(self) -> Reverse:
        return Reverse(self.compatible, self.obj)


class Reverse(TypedKey):
    __slots__ = ()

    def __lt__(self, other: Any) -> bool:
        return TypedKey.__lt__(other, self)


def stringKey(x: Any) -> TypedKey:
    return TypedKey((str,), x.casefold())


def booleanKey(x: Any) -> TypedKey:
    return TypedKey((bool,), x)


def numberKey(x: Any) -> TypedKey:
    return TypedKey((int, float), x)


def autoKey(x: Any) -> TypedKey:
    if isinstance(x, str):
        return stringKey(x)
    elif isinstance(x, bool):
        return booleanKey(x)
    elif isinstance(x, (int, float)):
        return numberKey(x)
    else:
        raise method.UnsupportedSort().exception()


class SortKey(NamedTuple):
    column: ClauseElement
    key: Callable[[Any], TypedKey] = autoKey

    def descending(self) -> SortKey:
        key = self.key
        return self._replace(key=lambda x: key(x).descending())


class Comparator(BaseModel):
    property: String
    is_ascending: Boolean = True
    collation: Optional[String]

    def compile(self) -> SortKey:
        if self.property == "id":
            key = SortKey(column=models.objects.c.id, key=numberKey)
        else:
            key = SortKey(column=models.objects.c.contents[self.property])

        if not self.is_ascending:
            key = key.descending()

        return key


ComparatorImpl = TypeVar("ComparatorImpl", bound=Comparator)
