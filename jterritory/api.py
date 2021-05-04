from pydantic import Field
from typing import Any, Dict, Generic, List, NamedTuple, Optional, Set, TypeVar, Union
from . import exceptions
from .query.filter import FilterImpl, FilterOperator
from .query.sort import ComparatorImpl
from .types import BaseModel, GenericModel
from .types import Boolean, Id, Int, PositiveInt, String, UnsignedInt


class Invocation(NamedTuple):
    "https://tools.ietf.org/html/rfc8620#section-3.2"
    name: String
    arguments: dict
    call_id: String


class Request(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-3.3"
    using: Set[String]
    method_calls: List[Invocation] = Field(title="Method calls")
    created_ids: Optional[Dict[Id, Id]] = Field(title="Created Ids")


class Response(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-3.4"
    method_responses: List[Invocation] = Field(title="Method responses")
    created_ids: Optional[Dict[Id, Id]] = Field(title="Created Ids")
    session_state: String = Field(title="Session state")


class BaseDatatype(BaseModel):
    id: Id

    class Config:
        extra = "allow"


Datatype = TypeVar("Datatype", bound=BaseDatatype)


class GetRequest(BaseModel):
    account_id: Id
    ids: Optional[Set[Id]]
    properties: Optional[Set[String]]


class GetResponse(GenericModel, Generic[Datatype]):
    account_id: Id
    state: String
    list: List[Datatype]
    not_found: Set[Id]


class ChangesRequest(BaseModel):
    account_id: Id
    since_state: String
    max_changes: Optional[PositiveInt]


class ChangesResponse(BaseModel):
    account_id: Id
    old_state: String
    new_state: String
    has_more_changes: Boolean
    created: Set[Id]
    updated: Set[Id]
    destroyed: Set[Id]


PatchObject = Dict[String, Any]


class SetRequest(GenericModel, Generic[Datatype]):
    account_id: Id
    if_in_state: Optional[String]
    create: Optional[Dict[Id, Datatype]]
    update: Optional[Dict[Id, PatchObject]]
    destroy: Optional[Set[Id]]


class SetResponse(BaseModel):
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
    from_account_id: Id
    if_from_in_state: Optional[String]
    account_id: Id
    if_in_state: Optional[String]
    create: Dict[Id, BaseDatatype]
    on_success_destroy_original: Boolean
    destroy_from_if_in_state: Optional[String]


class CopyResponse(BaseModel):
    from_account_id: Id
    account_id: Id
    old_state: Optional[String]
    new_state: String
    created: Optional[Dict[Id, BaseDatatype]]
    not_created: Optional[Dict[Id, exceptions.SetError]]


class QueryRequest(GenericModel, Generic[FilterImpl, ComparatorImpl]):
    account_id: Id
    filter: Union[FilterOperator[FilterImpl], FilterImpl, None]
    sort: Optional[List[ComparatorImpl]]
    position: Int = Int(0)
    anchor: Optional[Id]
    anchor_offset: Int = Int(0)
    limit: Optional[UnsignedInt]  # XXX: shouldn't 0 be prohibited too?
    calculate_total: Boolean = False


class QueryResponse(BaseModel):
    account_id: Id
    query_state: String
    can_calculate_changes: Boolean
    position: UnsignedInt
    ids: List[Id]
    total: Optional[UnsignedInt]
    limit: Optional[UnsignedInt]


class QueryChangesRequest(GenericModel, Generic[FilterImpl, ComparatorImpl]):
    account_id: Id
    filter: Union[FilterOperator[FilterImpl], FilterImpl, None]
    sort: Optional[List[ComparatorImpl]]
    since_query_state: String
    max_changes: Optional[UnsignedInt]  # XXX: shouldn't 0 be prohibited too?
    up_to_id: Optional[Id]
    calculate_total: Boolean = False


class AddedItem(BaseModel):
    index: UnsignedInt
    id: Id


class QueryChangesResponse(BaseModel):
    account_id: Id
    old_query_state: String
    new_query_state: String
    total: Optional[UnsignedInt]
    removed: Set[Id]
    added: List[AddedItem]
