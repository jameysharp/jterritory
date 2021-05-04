"""
Generic implementations of the standard methods specified in RFC8620
section 5, parameterized over a concrete datatype.

The spec says: 'JMAP provides a uniform interface for creating,
retrieving, updating, and deleting objects of a particular type. For a
"Foo" data type, records of that type would be fetched via a "Foo/get"
call and modified via a "Foo/set" call. Delta updates may be fetched via
a "Foo/changes" call. These methods all follow a standard format ...
Some types may not have all these methods. Specifications defining types
MUST specify which methods are available for the type.'

https://tools.ietf.org/html/rfc8620#section-5
"""

from typing import Any, Dict, Generic, List, Optional, Set, TypeVar, Union
from .. import exceptions
from ..query.filter import FilterImpl, FilterOperator
from ..query.sort import ComparatorImpl
from ..types import BaseModel, GenericModel
from ..types import Boolean, Id, Int, PositiveInt, String, UnsignedInt


class BaseDatatype(BaseModel):
    id: Id

    class Config:
        extra = "allow"


Datatype = TypeVar("Datatype", bound=BaseDatatype)


class GetRequest(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-5.1"
    account_id: Id
    ids: Optional[Set[Id]]
    properties: Optional[Set[String]]


class GetResponse(GenericModel, Generic[Datatype]):
    "https://tools.ietf.org/html/rfc8620#section-5.1"
    account_id: Id
    state: String
    list: List[Datatype]
    not_found: Set[Id]


class ChangesRequest(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-5.2"
    account_id: Id
    since_state: String
    max_changes: Optional[PositiveInt]


class ChangesResponse(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-5.2"
    account_id: Id
    old_state: String
    new_state: String
    has_more_changes: Boolean
    created: Set[Id]
    updated: Set[Id]
    destroyed: Set[Id]


PatchObject = Dict[String, Any]


class SetRequest(GenericModel, Generic[Datatype]):
    "https://tools.ietf.org/html/rfc8620#section-5.3"
    account_id: Id
    if_in_state: Optional[String]
    create: Optional[Dict[Id, Datatype]]
    update: Optional[Dict[Id, PatchObject]]
    destroy: Optional[Set[Id]]


class SetResponse(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-5.3"
    account_id: Id
    old_state: Optional[String]
    new_state: String
    created: Optional[Dict[Id, BaseDatatype]]
    updated: Optional[Dict[Id, Optional[dict]]]
    destroyed: Optional[Set[Id]]
    not_created: Optional[Dict[Id, exceptions.SetError]]
    not_updated: Optional[Dict[Id, exceptions.SetError]]
    not_destroyed: Optional[Dict[Id, exceptions.SetError]]


class CopyRequest(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-5.4"
    from_account_id: Id
    if_from_in_state: Optional[String]
    account_id: Id
    if_in_state: Optional[String]
    create: Dict[Id, BaseDatatype]
    on_success_destroy_original: Boolean
    destroy_from_if_in_state: Optional[String]


class CopyResponse(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-5.4"
    from_account_id: Id
    account_id: Id
    old_state: Optional[String]
    new_state: String
    created: Optional[Dict[Id, BaseDatatype]]
    not_created: Optional[Dict[Id, exceptions.SetError]]


class QueryRequest(GenericModel, Generic[FilterImpl, ComparatorImpl]):
    "https://tools.ietf.org/html/rfc8620#section-5.5"
    account_id: Id
    filter: Union[FilterOperator[FilterImpl], FilterImpl, None]
    sort: Optional[List[ComparatorImpl]]
    position: Int = Int(0)
    anchor: Optional[Id]
    anchor_offset: Int = Int(0)
    limit: Optional[UnsignedInt]  # XXX: shouldn't 0 be prohibited too?
    calculate_total: Boolean = False


class QueryResponse(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-5.5"
    account_id: Id
    query_state: String
    can_calculate_changes: Boolean
    position: UnsignedInt
    ids: List[Id]
    total: Optional[UnsignedInt]
    limit: Optional[UnsignedInt]


class QueryChangesRequest(GenericModel, Generic[FilterImpl, ComparatorImpl]):
    "https://tools.ietf.org/html/rfc8620#section-5.6"
    account_id: Id
    filter: Union[FilterOperator[FilterImpl], FilterImpl, None]
    sort: Optional[List[ComparatorImpl]]
    since_query_state: String
    max_changes: Optional[UnsignedInt]  # XXX: shouldn't 0 be prohibited too?
    up_to_id: Optional[Id]
    calculate_total: Boolean = False


class AddedItem(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-5.6"
    index: UnsignedInt
    id: Id


class QueryChangesResponse(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-5.6"
    account_id: Id
    old_query_state: String
    new_query_state: String
    total: Optional[UnsignedInt]
    removed: Set[Id]
    added: List[AddedItem]
