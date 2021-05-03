from pydantic.main import ModelMetaclass
from typing import ClassVar, Literal, Optional, no_type_check
from jterritory.types import BaseModel


class NameMeta(ModelMetaclass):
    "Derive the 'type' field from the name of the model."

    @no_type_check
    def __new__(mcs, name, bases, namespace, **kwargs):
        if "type" not in namespace:
            prefix = namespace.get("_prefix", "")
            for base in bases:
                try:
                    prefix = base._prefix
                    break
                except AttributeError:
                    pass

            namespace["type"] = prefix + name[:1].lower() + name[1:]

        annotations = namespace.setdefault("__annotations__", {})
        if "type" not in annotations:
            annotations["type"] = Literal[namespace["type"]]

        return super().__new__(mcs, name, bases, namespace, **kwargs)


class MethodException(Exception):
    pass


class MethodError(BaseModel, metaclass=NameMeta):
    type: str = ""

    def exception(self) -> MethodException:
        return MethodException(self)

    class Config:
        allow_mutation = False


class SetException(Exception):
    pass


class SetError(BaseModel, metaclass=NameMeta):
    type: str = ""
    description: Optional[str]

    def exception(self) -> SetException:
        return SetException(self)

    class Config:
        allow_mutation = False


class RequestError(BaseModel, metaclass=NameMeta):
    "https://tools.ietf.org/html/rfc7807"
    type: str = "about:blank"
    status: int = 400
    title: Optional[str]
    detail: Optional[str]
    instance: Optional[str]

    _prefix: ClassVar[str] = "urn:ietf:params:jmap:error:"

    class Config:
        allow_mutation = False
        extra = "allow"
