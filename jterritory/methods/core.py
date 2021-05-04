"""
These methods should be supported by all JMAP servers and do not depend
on which datatypes the server understands.
"""

from typing import Dict, Optional, Set
from .. import exceptions
from ..types import BaseModel, Id, String, UnsignedInt


class Echo(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-4"

    class Config:
        extra = "allow"


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
