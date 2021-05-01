from abc import abstractmethod
from typing import Any, Dict, Optional


class JMAPError(Exception):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, **self.kwargs}


class RequestError(JMAPError):
    pass


class MethodError(JMAPError):
    pass


class SetError(JMAPError):
    def __init__(self, description: Optional[str] = None, **kwargs: Any) -> None:
        if description is not None:
            kwargs["description"] = description
        super().__init__(**kwargs)
