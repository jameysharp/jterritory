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

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice
from pydantic import parse_obj_as, ValidationError
import re
from sqlalchemy import false, func, null, select, and_, or_
from sqlalchemy.sql import ClauseElement
import typing
from typing import Any, Dict, Generic, List, Mapping, Optional, Set, Tuple, Type, Union
from zlib import crc32
from .. import exceptions, models
from ..api import Context, Method, serializable
from ..exceptions import method, seterror
from ..query.filter import FilterImpl, FilterOperator
from ..query.sort import ComparatorImpl, SortKey, numberKey
from ..types import BaseModel, GenericModel, ObjectId
from ..types import Boolean, Id, Int, JSONPointer, PositiveInt, String, UnsignedInt


class BaseDatatype(BaseModel):
    class Config:
        extra = "allow"


PartialObject = Dict[String, Any]


class GetRequest(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-5.1"
    account_id: Id
    ids: Optional[Set[ObjectId]]
    properties: Optional[Set[String]]


class GetResponse(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-5.1"
    account_id: Id
    state: String
    list: List[PartialObject]
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


class PatchPointer(JSONPointer):
    regex = re.compile("^[^/]")


PatchObject = Dict[PatchPointer, Any]


class SetRequest(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-5.3"
    account_id: Id
    if_in_state: Optional[String]
    create: Optional[Dict[Id, PartialObject]]
    update: Optional[Dict[ObjectId, PatchObject]]
    destroy: Optional[Set[ObjectId]]


class SetResponse(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-5.3"
    account_id: Id
    old_state: Optional[String]
    new_state: String
    created: Optional[Dict[Id, PartialObject]]
    updated: Optional[Dict[ObjectId, Optional[PartialObject]]]
    destroyed: Optional[Set[ObjectId]]
    not_created: Optional[Dict[Id, exceptions.SetError]]
    not_updated: Optional[Dict[ObjectId, exceptions.SetError]]
    not_destroyed: Optional[Dict[ObjectId, exceptions.SetError]]


class CopyRequest(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-5.4"
    from_account_id: Id
    if_from_in_state: Optional[String]
    account_id: Id
    if_in_state: Optional[String]
    create: Dict[Id, PartialObject]
    on_success_destroy_original: Boolean
    destroy_from_if_in_state: Optional[String]


class CopyResponse(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-5.4"
    from_account_id: Id
    account_id: Id
    old_state: Optional[String]
    new_state: String
    created: Optional[Dict[Id, PartialObject]]
    not_created: Optional[Dict[Id, exceptions.SetError]]


class QueryRequest(GenericModel, Generic[FilterImpl, ComparatorImpl]):
    "https://tools.ietf.org/html/rfc8620#section-5.5"
    account_id: Id
    filter: Union[FilterOperator[FilterImpl], FilterImpl, None]
    sort: Optional[List[ComparatorImpl]]
    position: Int = Int(0)
    anchor: Optional[ObjectId]
    anchor_offset: Int = Int(0)
    limit: Optional[UnsignedInt]  # XXX: shouldn't 0 be prohibited too?
    calculate_total: Boolean = False


class QueryResponse(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-5.5"
    account_id: Id
    query_state: String
    can_calculate_changes: Boolean
    position: UnsignedInt
    ids: List[ObjectId]
    total: Optional[UnsignedInt]
    limit: Optional[UnsignedInt]


class QueryChangesRequest(GenericModel, Generic[FilterImpl, ComparatorImpl]):
    "https://tools.ietf.org/html/rfc8620#section-5.6"
    account_id: Id
    filter: Union[FilterOperator[FilterImpl], FilterImpl, None]
    sort: Optional[List[ComparatorImpl]]
    since_query_state: String
    max_changes: Optional[UnsignedInt]  # XXX: shouldn't 0 be prohibited too?
    up_to_id: Optional[ObjectId]
    calculate_total: Boolean = False


class AddedItem(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-5.6"
    index: UnsignedInt
    id: ObjectId


class QueryChangesResponse(BaseModel):
    "https://tools.ietf.org/html/rfc8620#section-5.6"
    account_id: Id
    old_query_state: String
    new_query_state: String
    total: Optional[UnsignedInt]
    removed: Set[ObjectId]
    added: List[AddedItem]


class StandardMethods(Generic[FilterImpl, ComparatorImpl]):
    datatype: Type[BaseDatatype]

    def methods(self) -> Dict[str, Method]:
        base = self.type_name
        return {
            base + "/get": self.get,
            base + "/changes": self.changes,
            base + "/set": serializable(self.set),
            base + "/query": self.query,
        }

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

    def last_changed(self, ctx: Context, account: int) -> int:
        result = ctx.connection.scalar(
            select(func.max(models.objects.c.changed))
            .where(models.objects.c.account == account)
            .where(models.objects.c.datatype == self.internal_id)
        )
        return result or 0

    def get(self, ctx: Context, request: GetRequest) -> None:
        account = ctx.use_account(request.account_id)
        query = (
            select(
                models.objects.c.id, models.objects.c.changed, models.objects.c.contents
            )
            .where(models.objects.c.account == account)
            .where(models.objects.c.datatype == self.internal_id)
            .where(~models.objects.c.destroyed)
        )

        last_changed = self.last_changed(ctx, account)
        if request.ids is not None:
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
            request.properties.add(String("id"))
            objects = [
                {name: contents[name] for name in request.properties}
                for contents in objects
            ]

        response = GetResponse(
            account_id=request.account_id,
            state=String(last_changed),
            list=objects,
            not_found=not_found,
        )
        ctx.add_response(f"{self.type_name}/get", response)

    def changes(self, ctx: Context, request: ChangesRequest) -> None:
        account = ctx.use_account(request.account_id)

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

        lastSeen = max(lastSeen, self.last_changed(ctx, account))

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
            .where(models.objects.c.account == account)
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
            new_state=String("-".join(str(state) for state in states)),
            has_more_changes=len(states) > 1,
            created=created,
            updated=updated,
            destroyed=destroyed,
        )
        ctx.add_response(f"{self.type_name}/changes", response)

    def set(self, ctx: Context, request: SetRequest) -> None:
        account = ctx.use_account(request.account_id)
        old_state = self.last_changed(ctx, account)
        if request.if_in_state is not None and request.if_in_state != str(old_state):
            raise method.StateMismatch().exception()

        update = request.update or {}
        destroy = request.destroy or set()

        helper = SetHelper(
            ctx=ctx,
            datatype=self.datatype,
            internal_id=self.internal_id,
            create=request.create or {},
            account=account,
            last_changed=old_state,
            not_updated=dict.fromkeys(update, seterror.NotFound()),
            not_destroyed=dict.fromkeys(destroy, seterror.NotFound()),
        )

        # "In the case of records with references to the same type, the
        # server MUST order the creates and updates within a single
        # method call so that creates happen before their creation ids
        # are referenced by another create/update/destroy in the same
        # call."
        # XXX: are circular creation dependencies supposed to be allowed?
        #
        # Note that destroy can't reference a newly-created object and
        # once all the creates are done the updates can happen in any
        # order, so the only tricky part is ordering creates.

        # Create requested objects
        while helper.create_some():
            pass

        # Apply changes to existing objects in order of object id, so
        # that a database like PostgreSQL which locks individual rows
        # can't deadlock between concurrent changes. If the client
        # requests to both update and destroy the same object, skip the
        # update.
        changes: Mapping[ObjectId, Optional[PatchObject]] = {
            **update,
            **dict.fromkeys(destroy),
        }

        for object_id, change in sorted(changes.items()):
            if change is not None:
                helper.update(object_id, change)
            elif helper.destroy(object_id) and object_id in update:
                helper.not_updated[object_id] = seterror.WillDestroy()

        # TODO: let caller validate that any datatype-specific
        # multi-object invariants still hold after this series of
        # changes

        response = SetResponse(
            account_id=request.account_id,
            old_state=String(old_state),
            new_state=String(helper.last_changed),
            created=helper.created or None,
            updated=helper.updated or None,
            destroyed=helper.destroyed or None,
            not_created=helper.not_created or None,
            not_updated=helper.not_updated or None,
            not_destroyed=helper.not_destroyed or None,
        )
        ctx.add_response(f"{self.type_name}/set", response)

    def query(
        self, ctx: Context, request: QueryRequest[FilterImpl, ComparatorImpl]
    ) -> None:
        account = ctx.use_account(request.account_id)

        # Since object IDs are unique, if the sort criteria include the
        # "id" property then we can discard all later criteria.
        # Otherwise, implicitly add the "id" property to ensure a stable
        # sort order. This also means that the last selected column is
        # always the object ID, which is what we need to return.
        order = []
        for comparator in request.sort or []:
            key = comparator.compile()
            order.append(key)
            if key.column is models.objects.c.id:
                break
        else:
            order.append(SortKey(models.objects.c.id, key=numberKey))

        # Queries are always implicitly filtered to exclude objects
        # which have been destroyed, in addition to any filter the
        # client specified.
        criteria = ~models.objects.c.destroyed
        if request.filter is not None:
            criteria = and_(criteria, request.filter.compile())

        columns: List[ClauseElement] = [models.objects.c.changed]
        columns.extend(key.column for key in order)
        query = select(*columns).where(
            models.objects.c.account == account,
            models.objects.c.datatype == self.internal_id,
            criteria,
        )

        last_changed = self.last_changed(ctx, account)
        rows = []
        for row in ctx.connection.execute(query):
            it = iter(row)

            changed = next(it)
            if last_changed < changed:
                last_changed = changed

            rows.append(tuple(k.key(x) for k, x in zip(order, it)))

        rows.sort()
        objects = [ObjectId.from_int(row[-1].obj) for row in rows]

        start: int = request.position
        if request.anchor is not None:
            try:
                start = objects.index(request.anchor) + request.anchor_offset
            except ValueError as exc:
                raise method.AnchorNotFound().exception() from exc
        elif start < 0:
            # Negative anchorOffsets don't wrap around the end of the
            # list, so this adjustment must not be applied when an
            # anchor is being used.
            start += len(objects)

        if start < 0:
            start = 0

        if request.limit is None:
            end = len(objects)
        else:
            end = start + request.limit

        response = QueryResponse(
            account_id=request.account_id,
            query_state=String(last_changed),
            can_calculate_changes=False,
            position=UnsignedInt(start),
            ids=objects[start:end],
            total=UnsignedInt(len(objects)) if request.calculate_total else None,
            limit=None,
        )
        ctx.add_response(f"{self.type_name}/query", response)


@dataclass
class SetHelper:
    ctx: Context
    datatype: Type[BaseDatatype]
    internal_id: int
    create: Dict[Id, PartialObject]
    account: int
    last_changed: int

    created: Dict[Id, PartialObject] = field(default_factory=dict)
    updated: Dict[ObjectId, Optional[PartialObject]] = field(default_factory=dict)
    destroyed: Set[ObjectId] = field(default_factory=set)
    not_created: Dict[Id, exceptions.SetError] = field(default_factory=dict)
    not_updated: Dict[ObjectId, exceptions.SetError] = field(default_factory=dict)
    not_destroyed: Dict[ObjectId, exceptions.SetError] = field(default_factory=dict)

    def next_change(self) -> int:
        self.last_changed += 1
        return self.last_changed

    def criteria(self, object_id: ObjectId) -> ClauseElement:
        id = object_id.to_int()
        if id is None:
            return false()
        return and_(
            models.objects.c.id == id,
            models.objects.c.account == self.account,
            models.objects.c.datatype == self.internal_id,
            ~models.objects.c.destroyed,
        )

    def create_some(self) -> bool:
        """
        Creates one or more objects. Returns False when there's nothing
        left to do.
        """
        try:
            creation_id, contents = self.create.popitem()
        except KeyError:
            return False
        self.create_one(creation_id, contents)
        return True

    def create_one(self, creation_id: Id, new: PartialObject) -> None:
        # If a previous method invocation used the same creation id,
        # ensure we don't use that older object during this invocation.
        self.ctx.created_ids.pop(creation_id, None)

        self.subst(new)

        # TODO: reject "id" property if presented by the client
        try:
            model = self.datatype.parse_obj(new)
        except ValidationError as exc:
            self.not_created[creation_id] = seterror.InvalidProperties(
                properties=["/".join(map(str, error["loc"])) for error in exc.errors()]
            )
            return

        validated = model.dict()

        self.next_change()
        inserted = self.ctx.connection.execute(
            models.objects.insert().values(
                account=self.account,
                datatype=self.internal_id,
                contents=validated,
                changed=self.last_changed,
                created=self.last_changed,
                destroyed=False,
            )
        )
        object_id = ObjectId.from_int(inserted.inserted_primary_key[0])

        # TODO: diff new against validated and report any
        # server-computed changes
        self.created[creation_id] = {String("id"): object_id}
        self.ctx.created_ids[creation_id] = object_id

    def update(self, object_id: ObjectId, patch: PatchObject) -> None:
        old: Optional[PartialObject] = self.ctx.connection.scalar(
            select(models.objects.c.contents)
            .where(self.criteria(object_id))
            .with_for_update()
        )

        # Note that `old` could be None if either there is no row with
        # that ID, or the object used to exist but has been destroyed.
        # Either way we should report it as not found.
        if old is None:
            return

        old[String("id")] = object_id
        self.subst(patch)

        try:
            new = self.patch(old, patch)

            try:
                model = self.datatype.parse_obj(new)
            except ValidationError as exc:
                raise seterror.InvalidProperties(
                    properties=[
                        "/".join(map(str, error["loc"])) for error in exc.errors()
                    ]
                ).exception()

            # TODO: let caller validate new compared to old
            if new.pop(String("id"), None) != object_id:
                raise seterror.InvalidProperties(properties=["id"]).exception()
        except exceptions.SetException as exc:
            self.not_updated[object_id] = exc.args[0]
            return

        validated = model.dict()

        # TODO: diff new against validated and report any
        # server-computed changes
        self.updated[object_id] = None

        del validated["id"]
        updated = self.ctx.connection.execute(
            models.objects.update()
            .where(self.criteria(object_id))
            .values(contents=validated, changed=self.next_change())
        )
        assert updated.rowcount == 1
        del self.not_updated[object_id]

    def destroy(self, object_id: ObjectId) -> bool:
        destroyed = self.ctx.connection.execute(
            models.objects.update()
            .where(self.criteria(object_id))
            .values(contents=null(), destroyed=True, changed=self.next_change())
        )
        if destroyed.rowcount != 1:
            self.last_changed -= 1
            return False

        self.destroyed.add(object_id)
        del self.not_destroyed[object_id]
        return True

    def subst(self, new: Any) -> None:
        if isinstance(new, dict):
            items = iter(new.items())
        elif isinstance(new, list):
            items = enumerate(new)
        else:
            return

        for k, v in items:
            if not isinstance(v, str):
                self.subst(v)
                continue

            if not v.startswith("#"):
                continue

            try:
                creation_id = parse_obj_as(Id, v[1:])
            except ValidationError:
                continue

            # This looks like it might be a reference to an object we're
            # supposed to create. If there's an object we haven't
            # created yet bearing this creation id, then we have to
            # recursively create that one first before we can continue
            # with this one.
            try:
                dependency = self.create.pop(creation_id)
            except KeyError:
                pass
            else:
                self.create_one(creation_id, dependency)

            # At this point we don't know whether this property is
            # supposed to be an ObjectId or just a random string. If
            # there's a successfully-created object with this creation
            # ID, then go ahead and substitute it. If this doesn't refer
            # to a creation ID, then this is some random string that
            # happens to begin with "#", and we shouldn't change it. The
            # final possibility is that the creation failed. Note that
            # "#" can't be in a valid Id, so assuming the datatype
            # correctly declared the type of this property, validation
            # will reject this field for us later.
            try:
                new[k] = self.ctx.created_ids[creation_id]
            except KeyError:
                pass

    @classmethod
    def patch(cls, old: PartialObject, patches: PatchObject) -> Any:
        # Don't modify the caller's copy of the pre-patch object, in
        # case it needs to validate that certain properties are
        # unchanged.
        old = old.copy()
        lastpath = ""
        for path, value in sorted(patches.items()):
            # "There MUST NOT be two patches in the PatchObject where
            # the pointer of one is the prefix of the pointer of the
            # other, e.g., “alerts/1/offset” and “alerts”." By walking
            # the patches in sorted order, some such pair will always
            # appear next to each other, with the shorter one first.
            if path.startswith(lastpath) and path[len(lastpath) :][:1] in ("", "/"):
                raise seterror.InvalidPatch().exception()

            # str.split always returns at least one item even if the
            # input is empty, so calling next(tokens) once can never
            # raise StopIteration.
            tokens = iter(path.reference_tokens())
            current: Any = old
            last = cls.key(current, next(tokens))
            try:
                for token in tokens:
                    current = current[last] = current[last].copy()
                    last = cls.key(current, token)
            except (KeyError, IndexError):
                # "All parts prior to the last (i.e., the value after
                # the final slash) MUST already exist on the object
                # being patched."
                raise seterror.InvalidPatch().exception()

            # "The pointer MUST NOT reference inside an array (i.e., you
            # MUST NOT insert/delete from an array; the array MUST be
            # replaced in its entirety instead)."
            if not isinstance(current, dict):
                raise seterror.InvalidPatch().exception()

            if value is not None:
                current[last] = value
            else:
                try:
                    del current[last]
                except KeyError:
                    # "If the key is not present in the parent, this a
                    # no-op."
                    pass

        return old

    @staticmethod
    def key(current: Any, token: str) -> Union[int, str]:
        """
        Evaluate one reference token of a JSON Pointer according to
        <https://tools.ietf.org/html/rfc6901#section-4>.
        """

        if isinstance(current, dict):
            return token
        if isinstance(current, list):
            # JMAP requires that the tokens before the last must already
            # exist and the last must not refer to an array element. So
            # JSON Pointer's "-" token (to refer to the element after
            # the last of an array) is never valid in this application.
            try:
                return int(token)
            except ValueError:
                pass
        raise seterror.InvalidPatch().exception()
