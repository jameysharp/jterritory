"""
These methods should be supported by all JMAP servers and do not depend
on which datatypes the server understands.
"""

from typing import Any, Dict, Optional, Set
from .. import exceptions
from ..api import Context
from ..types import BaseModel, Id, String, UnsignedInt


class Echo(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-4"

    class Config:
        extra = "allow"

    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        # Core/echo is specified to return exactly what was passed to
        # it, so in this specific case we must not exclude anything.
        kwargs.update(exclude_none=False)
        return super().dict(**kwargs)


def echo(ctx: Context, request: Echo) -> None:
    ctx.add_response("Core/echo", request)


class UploadResponse(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-6.1"
    account_id: Id
    blob_id: Id
    type: String
    size: UnsignedInt


class CopyRequest(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-6.3"
    from_account_id: Id
    account_id: Id
    blob_ids: Set[Id]


class CopyResponse(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-6.3"
    from_account_id: Id
    account_id: Id
    copied: Optional[Dict[Id, Id]]
    not_copied: Optional[Dict[Id, exceptions.SetError]]
