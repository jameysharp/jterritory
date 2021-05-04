from pydantic import ConstrainedInt, ConstrainedStr, StrictBool, StrictStr
from pydantic import BaseModel as PydanticBaseModel
from pydantic.generics import GenericModel as PydanticGenericModel
import re
from typing import Any, Dict, NamedTuple, Type, TypedDict


class BaseModel(PydanticBaseModel):
    class Config:
        allow_population_by_field_name = True

        @staticmethod
        def alias_generator(name: str) -> str:
            words = name.split("_")
            return "".join(words[:1] + [word.capitalize() for word in words[1:]])

        @staticmethod
        def schema_extra(schema: Dict[str, Any], model: Type["BaseModel"]) -> None:
            props = schema["properties"]
            for field in model.__fields__.values():
                props[field.alias]["title"] = field.name.replace("_", " ").capitalize()

    def json(self, **kwargs: Any) -> str:
        # Change some defaults but allow callers to override:
        kwargs = {
            "by_alias": True,
            "exclude_defaults": True,
            **kwargs,
        }
        return super().json(**kwargs)


class GenericModel(PydanticGenericModel):
    class Config(BaseModel.Config):
        pass


# https://tools.ietf.org/html/rfc8620#section-1.1
String = StrictStr
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


class ObjectId(int):
    def __str__(self) -> str:
        return "o" + super().__str__()


class AddedItem(TypedDict):
    index: int
    id: str


class ObjectPosition(NamedTuple):
    position: int
    objectId: ObjectId

    def offset(self, offset: int) -> "ObjectPosition":
        if not offset:
            return self
        return self._replace(position=self.position + offset)

    def as_dict(self) -> AddedItem:
        return {
            "index": self.position,
            "id": str(self.objectId),
        }
