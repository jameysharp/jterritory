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

from itertools import islice
from sqlalchemy import func, select, and_, or_
import typing
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, Type, TypeVar, Union
from zlib import crc32
from .. import exceptions, models
from ..api import Context
from ..exceptions import method
from ..query.filter import FilterImpl, FilterOperator
from ..query.sort import ComparatorImpl
from ..types import BaseModel, GenericModel, ObjectId
from ..types import Boolean, Id, Int, PositiveInt, String, UnsignedInt


class BaseDatatype(BaseModel):
    id: Id

    class Config:
        extra = "allow"


Datatype = TypeVar("Datatype", bound=BaseDatatype)


class GetRequest(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-5.1"
    account_id: Id
    ids: Optional[Set[ObjectId]]
    properties: Optional[Set[String]]


class GetResponse(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-5.1"
    account_id: Id
    state: String
    list: List[dict]
    not_found: Set[ObjectId]


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
    created: Set[ObjectId]
    updated: Set[ObjectId]
    destroyed: Set[ObjectId]


PatchObject = Dict[String, Any]


class SetRequest(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-5.3"
    account_id: Id
    if_in_state: Optional[String]
    create: Optional[Dict[Id, dict]]
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


class StandardMethods(Generic[FilterImpl, ComparatorImpl]):
    def __init__(self, datatype: Type[Datatype]) -> None:
        self.datatype = datatype

    @property
    def query_types(self) -> Tuple[Type[FilterImpl], Type[ComparatorImpl]]:
        try:
            return typing.get_args(self.__orig_class__)  # type: ignore
        except AttributeError as exc:
            raise TypeError(
                "StandardMethods must be instantiated with concrete types"
            ) from exc

    @property
    def type_name(self) -> str:
        return self.datatype.__name__

    @property
    def internal_id(self) -> int:
        return crc32(self.type_name.encode())

    def last_changed(self, ctx: Context, account_id: String) -> int:
        result = ctx.connection.scalar(
            select(func.max(models.objects.c.changed))
            .join(models.accounts)
            .where(models.accounts.c.account == account_id)
            .where(models.objects.c.datatype == self.internal_id)
        )
        return result or 0

    def get(self, ctx: Context, request: GetRequest) -> None:
        query = (
            select(
                models.objects.c.id, models.objects.c.changed, models.objects.c.contents
            )
            .join(models.accounts)
            .where(models.accounts.c.account == request.account_id)
            .where(models.objects.c.datatype == self.internal_id)
            .where(~models.objects.c.destroyed)
        )

        last_changed = 0
        if request.ids is not None:
            last_changed = self.last_changed(ctx, request.account_id)
            ids = [id for id in map(ObjectId.to_int, request.ids) if id is not None]
            query = query.where(models.objects.c.id.in_(ids))

        objects = []
        not_found = request.ids or set()
        for row in ctx.connection.execute(query):
            object_id = ObjectId.from_int(row.id)
            not_found.discard(object_id)
            row.contents["id"] = object_id
            objects.append(row.contents)
            if last_changed < row.changed:
                last_changed = row.changed

        if request.properties is not None:
            request.properties.add("id")
            objects = [
                {name: contents[name] for name in request.properties}
                for contents in objects
            ]

        response = GetResponse(
            account_id=request.account_id,
            state=str(last_changed),
            list=objects,
            not_found=not_found,
        )
        ctx.add_response(f"{self.type_name}/get", response)

    def changes(self, ctx: Context, request: ChangesRequest) -> None:
        try:
            states = [int(state) for state in request.since_state.split("-")]
        except ValueError as exc:
            raise method.CannotCalculateChanges().exception() from exc

        # The number of past states must be odd. The first state is
        # exactly as used by /get; followed by pairs describing ranges
        # of previously reported changes from past calls to /changes
        # that exceeded the maxChanges limit.
        if len(states) % 2 == 0:
            raise method.CannotCalculateChanges().exception()

        # The past states must be in strictly ascending order.
        if any(a >= b for a, b in zip(states, islice(states, 1, None))):
            raise method.CannotCalculateChanges().exception()

        spans = []
        it = iter(states)
        for seen, upto in zip(it, it):
            # We need to return any changes <= upto but not <= seen. But
            # if the last change was to destroy the object and it was
            # created within the same span, then the client never saw it
            # and we don't need to return it at all.
            spans.append(
                and_(
                    models.objects.c.changed > seen,
                    models.objects.c.changed <= upto,
                    or_(
                        models.objects.c.created <= seen,
                        ~models.objects.c.destroyed,
                    ),
                )
            )

        lastSeen = states[-1]
        spans.append(
            and_(
                models.objects.c.changed > lastSeen,
                or_(
                    models.objects.c.created <= lastSeen,
                    ~models.objects.c.destroyed,
                ),
            )
        )

        created = set()
        updated = set()
        destroyed = set()

        query = (
            select(
                models.objects.c.id,
                models.objects.c.changed,
                models.objects.c.created,
                models.objects.c.destroyed,
            )
            .order_by(models.objects.c.changed.desc())
            .join(models.accounts)
            .where(models.accounts.c.account == request.account_id)
            .where(models.objects.c.datatype == self.internal_id)
            .where(or_(*spans))
        )

        if request.max_changes is not None:
            query = query.limit(request.max_changes + 1)

        for idx, row in enumerate(ctx.connection.execute(query)):
            # Find the span containing this object; delete all later
            # spans and the end of this span.
            while row.changed <= states[-1]:
                states.pop()

            if idx == request.max_changes:
                # We've hit the client's response-size limit. Set this
                # span's endpoint to this object so we can pick up here
                # next time.
                states.append(row.changed)
                break

            object_id = ObjectId.from_int(row.id)

            if row.destroyed:
                destroyed.add(object_id)
            elif row.created > states[-1]:
                # This object was created within the current span and
                # not changed after this span, so report its creation
                # now even if it was also updated in this span.
                created.add(object_id)
            else:
                updated.add(object_id)

            # Since this query is sorted by `changed` in descending
            # order, the following condition can only be true the first
            # time through this loop.
            if lastSeen < row.changed:
                lastSeen = row.changed
        else:
            # We did not hit the client's response-size limit, so all
            # changes which committed before the start of this query
            # have been reported.
            states.clear()

        # The last change that committed before the start of this query
        # is the beginning of a new span for the next request; the end
        # of that span will be the last-committed change at the
        # beginning of the next request.
        states.append(lastSeen)

        response = ChangesResponse(
            account_id=request.account_id,
            old_state=request.since_state,
            new_state="-".join(str(state) for state in states),
            has_more_changes=len(states) > 1,
            created=created,
            updated=updated,
            destroyed=destroyed,
        )
        ctx.add_response(f"{self.type_name}/changes", response)
