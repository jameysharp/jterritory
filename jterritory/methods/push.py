"""
https://tools.ietf.org/html/rfc8620#section-7
"""

from __future__ import annotations

from pydantic import Field
from typing import Any, Dict, List, Literal, Optional, Set
from .. import exceptions
from ..types import BaseModel, Id, String


UTCDate = String  # TODO: define properly
TypeState = Dict[String, String]
PatchObject = Dict[String, Any]


class StateChange(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-7.1"
    type: Literal["StateChange"] = Field("StateChange", alias="@type")
    changed: Dict[Id, TypeState]


class PushVerification(BaseModel):
    type: Literal["PushVerification"] = Field("PushVerification", alias="@type")
    push_subscription_id: String
    verification_code: String


class PushSecrets(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-7.2"
    p256dh: String
    auth: String


class PushSubscription(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-7.2"
    id: Id
    device_client_id: String
    url: String
    keys: Optional[PushSecrets]
    verification_code: Optional[String]
    expires: Optional[UTCDate]
    types: Optional[Set[String]]


class GetRequest(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-7.2.1"
    ids: Optional[Set[Id]]
    properties: Optional[Set[String]]


class GetResponse(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-7.2.1"
    list: List[PushSubscription]
    not_found: Set[Id]


class SetRequest(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-7.2.2"
    create: Optional[Dict[Id, PushSubscription]]
    update: Optional[Dict[Id, PatchObject]]
    destroy: Optional[Set[Id]]


class SetResponse(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-7.2.2"
    created: Optional[Dict[Id, dict]]
    updated: Optional[Dict[Id, Optional[dict]]]
    destroyed: Optional[Set[Id]]
    not_created: Optional[Dict[Id, exceptions.SetError]]
    not_updated: Optional[Dict[Id, exceptions.SetError]]
    not_destroyed: Optional[Dict[Id, exceptions.SetError]]
