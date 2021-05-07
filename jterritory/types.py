from __future__ import annotations

from pydantic import ConstrainedInt, ConstrainedStr, StrictBool, StrictStr
from pydantic import BaseModel as PydanticBaseModel
from pydantic.generics import GenericModel as PydanticGenericModel
import re
from typing import Any, Dict, List, NamedTuple, Optional, Type


class BaseModel(PydanticBaseModel):
    class Config:
        allow_population_by_field_name = True

        @staticmethod
        def alias_generator(name: str) -> str:
            words = name.split("_")
            return "".join(words[:1] + [word.capitalize() for word in words[1:]])

        @staticmethod
        def schema_extra(schema: Dict[str, Any], model: Type[BaseModel]) -> None:
            props = schema["properties"]
            for field in model.__fields__.values():
                props[field.alias]["title"] = field.name.replace("_", " ").capitalize()

    def json(self, **kwargs: Any) -> str:
        # Change some defaults but allow callers to override:
        kwargs = {
            "by_alias": True,
            "exclude_none": True,
            **kwargs,
        }
        return super().json(**kwargs)


class GenericModel(PydanticGenericModel):
    class Config(BaseModel.Config):
        pass


# https://tools.ietf.org/html/rfc8620#section-1.1
class String(StrictStr):
    @classmethod
    def validate(cls, value: str) -> String:
        return cls(value)


Number = float  # XXX: want to allow int or float but not string
Boolean = StrictBool


class Id(ConstrainedStr):
    "https://tools.ietf.org/html/rfc8620#section-1.2"
    strict = True
    min_length = 1
    max_length = 255
    regex = re.compile(r"^[A-Za-z0-9_-]*$")


class Int(ConstrainedInt):
    "https://tools.ietf.org/html/rfc8620#section-1.3"
    strict = True
    le = (1 << 53) - 1
    ge = -le


class UnsignedInt(Int):
    "https://tools.ietf.org/html/rfc8620#section-1.3"
    ge = 0


class PositiveInt(Int):
    # Not specifically named in the RFC, but specified in prose sometimes.
    ge = 1


class JSONPointer(String):
    "https://tools.ietf.org/html/rfc6901"

    def reference_tokens(self) -> List[str]:
        v: str = self
        if v.startswith("/"):
            v = v[1:]
        return [token.replace("~1", "/").replace("~0", "~") for token in v.split("/")]


class ObjectId(Id):
    """
    An Id which is expected to name a particular object, though the
    object might not exist.
    """

    @classmethod
    def from_int(cls, value: int) -> ObjectId:
        return cls(f"o{value}")

    def to_int(self) -> Optional[int]:
        if self[0] == "o":
            try:
                return int(self[1:])
            except ValueError:
                pass
        return None


class ObjectPosition(NamedTuple):
    position: int
    objectId: ObjectId

    def offset(self, offset: int) -> ObjectPosition:
        if not offset:
            return self
        return self._replace(position=self.position + offset)
