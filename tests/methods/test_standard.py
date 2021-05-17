from __future__ import annotations

from hypothesis import assume, settings, stateful, strategies as st
import itertools
import json
from jterritory import models
from jterritory.api import Endpoint, Response
from jterritory.exceptions import method
from jterritory.methods.standard import BaseDatatype, SetRequest, StandardMethods
from jterritory.query.filter import FilterCondition
from jterritory.query.sort import autoKey, Comparator, TypedKey
from jterritory.types import Id, Number, ObjectId, String
from pydantic.json import pydantic_encoder
from sqlalchemy.future import create_engine
from sqlalchemy.sql import ClauseElement
from typing import (
    AbstractSet,
    Any,
    cast,
    Dict,
    FrozenSet,
    Generator,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)


class Sample(BaseDatatype):
    number: Number
    text: String


class SampleComparator(Comparator):
    @classmethod
    def id(cls) -> SampleComparator:
        return cls(property=String("id"))

    def key(self, obj: dict) -> TypedKey:
        value = obj[self.property]
        if self.property == "id":
            value = ObjectId(value).to_int()
        key = autoKey(value)
        if not self.is_ascending:
            key = key.descending()
        return key


class SampleFilter(FilterCondition):
    number_is: Literal["<", "=", ">"]
    value: Number

    def compile(self) -> ClauseElement:
        column = models.objects.c.contents
        column = column["number"].as_float()  # type: ignore
        if self.number_is == "<":
            return column < self.value
        if self.number_is == "=":
            return column == self.value
        if self.number_is == ">":
            return column > self.value
        raise method.UnsupportedFilter().exception()

    def matches(self, obj: dict) -> bool:
        number = obj["number"]
        if self.number_is == "<":
            return number < self.value
        if self.number_is == "=":
            return number == self.value
        if self.number_is == ">":
            return number > self.value
        raise AssertionError(f"unknown operator {self.number_is!r}")


class SampleMethods(StandardMethods):
    datatype = Sample
    filter = SampleFilter
    comparator = SampleComparator


st.register_type_strategy(
    Sample,
    st.builds(
        Sample,
        number=st.floats(allow_infinity=False, allow_nan=False),
        text=st.text(),
    ),
)

st.register_type_strategy(
    SampleComparator,
    st.builds(
        SampleComparator,
        property=st.sampled_from(["id", "number", "text"]),
        is_ascending=st.booleans(),
    ),
)


st.register_type_strategy(
    SampleFilter,
    st.builds(
        SampleFilter,
        number_is=st.sampled_from(["<", ">"]),
        value=st.floats(allow_infinity=False, allow_nan=False),
    ),
)


class PastState(NamedTuple):
    state: str
    query_state: str
    created: FrozenSet[str] = frozenset()
    updated: FrozenSet[str] = frozenset()
    destroyed: FrozenSet[str] = frozenset()
    query: Sequence[str] = []


MethodCallTests = Generator[Tuple[str, dict], Tuple[Tuple[str, dict], ...], None]
T = TypeVar("T")


class DrawProtocol(Protocol):
    def __call__(self, base: st.SearchStrategy[T]) -> T:
        ...


class ConsistentHistory(stateful.RuleBasedStateMachine):
    CAPABILITY = String("urn:example:sample")
    ACCOUNT_ID = "some-account"
    sort: List[SampleComparator]
    filter: Optional[SampleFilter]
    base_query: Mapping[str, Any]

    def __init__(self) -> None:
        super().__init__()
        self.states = [PastState("0", "0")]  # FIXME: implementation details
        self.live: Dict[str, dict] = {}
        self.dead: Set[str] = set()

        sample = SampleMethods()
        self.endpoint = Endpoint(
            capabilities={self.CAPABILITY},
            methods=sample.methods(),
            engine=create_engine("sqlite://", echo=False),
        )

        models.metadata.create_all(self.endpoint.engine)
        with self.endpoint.engine.begin() as connection:
            connection.execute(models.accounts.insert().values(account=self.ACCOUNT_ID))

    @stateful.initialize(
        sort=st.lists(st.from_type(SampleComparator)),
        filter=st.none() | st.from_type(SampleFilter),
    )  # type: ignore
    def make_query(
        self, sort: List[SampleComparator], filter: Optional[SampleFilter]
    ) -> None:
        self.sort = sort
        self.filter = filter
        base_query: dict = {"accountId": self.ACCOUNT_ID}
        if sort:
            base_query["sort"] = [cmp.dict() for cmp in sort]
        if filter:
            base_query["filter"] = filter.dict()
        self.base_query = base_query

    def submit(self, tests: Iterable[MethodCallTests]) -> None:
        # The first call to generator.send has to be given None in order
        # to start the generator. I don't feel like convincing mypy that
        # this is okay.
        results: Iterable[Tuple[Tuple[str, dict], ...]]
        results = itertools.repeat(None)  # type: ignore
        while True:
            calls: List[Tuple[str, dict, str]] = []
            next_tests = []
            for test, result in zip(tests, results):
                try:
                    name, arguments = test.send(result)
                except StopIteration:
                    continue
                calls.append((name, arguments, f"call-{len(calls)}"))
                next_tests.append(test)
            tests = next_tests

            if not calls:
                return

            body = json.dumps(
                {"using": [self.CAPABILITY], "method_calls": calls},
                default=pydantic_encoder,
            ).encode()
            response = self.endpoint.request(body)
            assert isinstance(response, Response)
            call_ids, results = zip(
                *(
                    (call_id, tuple((call.name, call.arguments) for call in group))
                    for call_id, group in itertools.groupby(
                        response.method_responses, key=lambda call: call.call_id
                    )
                )
            )
            assert tuple(c[2] for c in calls) == call_ids

    @staticmethod
    def exact(
        call_name: str, call: dict, response_name: str, response: dict
    ) -> MethodCallTests:
        result = yield (call_name, call)
        assert result == ((response_name, response),)

    def check_live(self) -> MethodCallTests:
        ((name, arguments),) = yield ("Sample/get", {"accountId": self.ACCOUNT_ID})
        assert name == "Sample/get"
        objects = arguments.pop("list")
        assert arguments == {
            "accountId": self.ACCOUNT_ID,
            "state": self.states[-1].state,
            "notFound": set(),
        }
        assert self.live == {obj["id"]: obj for obj in objects}

    def check_dead(self) -> MethodCallTests:
        result = yield (
            "Sample/get",
            {"accountId": self.ACCOUNT_ID, "ids": self.dead},
        )
        assert result == (
            (
                "Sample/get",
                {
                    "accountId": self.ACCOUNT_ID,
                    "state": self.states[-1].state,
                    "list": [],
                    "notFound": self.dead,
                },
            ),
        )

    def check_changes(
        self,
        past: PastState,
        created: AbstractSet[str],
        updated: AbstractSet[str],
        destroyed: AbstractSet[str],
        limit: int = 2,
    ) -> MethodCallTests:
        actual_created: Set[str] = set()
        actual_updated: Set[str] = set()
        actual_destroyed: Set[str] = set()

        state = past.state
        while True:
            ((name, arguments),) = yield (
                "Sample/changes",
                {
                    "accountId": self.ACCOUNT_ID,
                    "sinceState": state,
                    "maxChanges": limit,
                },
            )
            if state == past.state and name == "error":
                assert arguments["type"] == "CannotCalculateChanges"
                return
            assert name == "Sample/changes"
            assert arguments["accountId"] == self.ACCOUNT_ID
            assert arguments["oldState"] == state

            # 'If a "maxChanges" is supplied, or set automatically by
            # the server, the server MUST ensure the number of ids
            # returned across "created", "updated", and "destroyed" does
            # not exceed this limit.'
            assert (
                sum(len(arguments[s]) for s in ("created", "updated", "destroyed"))
                <= limit
            )

            # "Where multiple changes to a record are split across
            # different intermediate states, the server MUST NOT return
            # a record as created after a response that deems it as
            # updated or destroyed, and it MUST NOT return a record as
            # destroyed before a response that deems it as created or
            # updated."
            assert arguments["created"].isdisjoint(actual_updated)
            assert arguments["created"].isdisjoint(actual_destroyed)
            assert arguments["updated"].isdisjoint(actual_destroyed)

            actual_created.update(arguments["created"])
            actual_updated.update(arguments["updated"])
            actual_destroyed.update(arguments["destroyed"])

            if not arguments["hasMoreChanges"]:
                break

            assert state != arguments["newState"]
            state = arguments["newState"]

        assert arguments["newState"] == self.states[-1].state

        # This assertion is stricter than the spec requires, because the
        # same object _may_ be returned in multiple sets, but this
        # enforces these SHOULDs: 'If a record has been created AND
        # updated since the old state, the server SHOULD just return the
        # id in the "created" list'; 'If a record has been updated AND
        # destroyed since the old state, the server SHOULD just return
        # the id in the "destroyed" list'; 'If a record has been created
        # AND destroyed since the old state, the server SHOULD remove
        # the id from the response entirely.'
        expected = (created, updated, destroyed)
        actual = (actual_created, actual_updated, actual_destroyed)
        assert expected == actual

    @stateful.invariant()  # type: ignore
    def check_history(self) -> None:
        tests = [self.check_live(), self.check_dead()]

        created: Set[str] = set()
        updated: Set[str] = set()
        destroyed: Set[str] = set()
        for past in reversed(self.states):
            tests.append(
                self.check_changes(
                    past=past,
                    created=frozenset(created),
                    updated=frozenset(updated),
                    destroyed=frozenset(destroyed),
                )
            )

            # If the same object has had multiple state transitions
            # since a given point, the rule is that destroyed wins over
            # created, which wins over updated; except that if it's both
            # created and destroyed, it shouldn't appear in any set.
            created.update(past.created)
            updated.update(past.updated)
            destroyed.update(past.destroyed)

            updated.difference_update(destroyed)
            updated.difference_update(created)
            created.difference_update(destroyed)
            destroyed.difference_update(past.created)

        assert updated == set() and destroyed == set() and created == self.live.keys()
        self.submit(tests)

    @st.composite
    def make_specific_query(draw: DrawProtocol) -> Tuple[dict, str, dict]:
        self: ConsistentHistory = draw(st.runner())
        current = self.states[-1]

        call = dict(self.base_query)
        total = len(current.query)
        max_int = (1 << 53) - 1

        if draw(st.booleans()):
            call["limit"] = draw(st.integers(min_value=0, max_value=max_int))

        if draw(st.booleans()):
            call["calculateTotal"] = True

        position = draw(st.integers(min_value=-max_int, max_value=max_int))
        if position != 0:
            call["position"] = position
            if position < 0:
                position += total

        offset = draw(st.integers(min_value=-max_int, max_value=max_int))
        if offset != 0:
            call["anchorOffset"] = offset
            # ignored unless "anchor" is also specified

        if draw(st.booleans()):
            call["anchor"] = draw(
                st.from_type(Id).filter(lambda x: x not in current.query)
            )
            return (call, "error", {"type": "anchorNotFound"})

        # Even if position is included in the call, anchor overrides it.
        if total > 0 and draw(st.booleans()):
            call["anchor"] = anchor = draw(st.sampled_from(current.query))
            position = current.query.index(anchor) + offset

        if position < 0:
            position = 0
        elif position >= total:
            # XXX: what should the server return in this case?
            # FIXME: probably should allow any value
            pass

        response: dict = {
            "accountId": ConsistentHistory.ACCOUNT_ID,
            "queryState": current.query_state,
            "canCalculateChanges": False,  # FIXME: implementation detail
            "position": position,
        }

        limit = call.get("limit", total)

        if call.get("calculateTotal", False):
            response["total"] = total

        response["ids"] = current.query[position : position + limit]
        return (call, "Sample/query", response)

    @stateful.rule(queries=st.lists(make_specific_query(), min_size=1))  # type: ignore
    def check_query(self, queries: List[Tuple[dict, str, dict]]) -> None:
        self.submit(self.exact("Sample/query", *query) for query in queries)

    @st.composite
    def make_set_request(draw: DrawProtocol) -> SetRequest:
        self: ConsistentHistory = draw(st.runner())
        ids = self.live.keys() | self.dead | {draw(st.from_type(Id))}

        # Roughly equivalent to the following, except Hypothesis throws
        # out a lot fewer random inputs as "invalid" this way:
        # >>> created = stateful.bundle("created")
        # >>> create = st.lists(st.from_type(Sample), max_size=5)
        # >>> update = st.lists(st.tuples(created, st.from_type(Sample)))
        # >>> destroy = st.lists(stateful.consumes(created))
        create = {
            Id(idx): cast(dict, obj.dict())
            for idx, obj in enumerate(draw(st.lists(st.from_type(Sample), max_size=5)))
        }
        update = {
            ObjectId(object_id): cast(dict, draw(st.from_type(Sample)).dict())
            for object_id in ids
            if draw(st.booleans())
        }
        destroy = {ObjectId(object_id) for object_id in ids if draw(st.booleans())}

        return SetRequest(
            account_id=Id(self.ACCOUNT_ID),
            create=create or None,
            update=update or None,
            destroy=destroy or None,
        )

    @stateful.rule(request=make_set_request())  # type: ignore
    def make_changes(self, request: SetRequest) -> None:
        self.submit([self.changes(request)])

    def changes(self, request: SetRequest) -> MethodCallTests:
        create = request.create or {}
        update = request.update or {}
        destroy = request.destroy or set()

        ((name, result),) = yield ("Sample/set", request.dict())
        assert name == "Sample/set"

        assert result["oldState"] in (None, self.states[-1].state)

        created = []
        assert "notCreated" not in result
        if "created" in result:
            for creation_id, server_set in result["created"].items():
                requested_create = create.pop(creation_id)
                requested_create.update(server_set)
                object_id = requested_create[String("id")]
                assert object_id not in self.live and object_id not in self.dead
                self.live[object_id] = requested_create
                created.append(object_id)
        assert create == {}

        updated = []
        if "updated" in result:
            for object_id, changes in result["updated"].items():
                requested_update = update.pop(object_id)
                if changes is not None:
                    requested_update.update(changes)
                self.live[object_id].update(requested_update)
                updated.append(object_id)
        if "notUpdated" in result:
            for object_id, error in result["notUpdated"].items():
                del update[object_id]
                if object_id not in self.live:
                    assert error["type"] == "notFound"
                elif object_id in destroy:
                    assert error["type"] == "willDestroy"
                else:
                    raise AssertionError(f"why wasn't {object_id} updated? {error!r}")
        assert update == {}

        destroyed = []
        if "destroyed" in result:
            for object_id in result["destroyed"]:
                destroy.remove(object_id)
                del self.live[object_id]
                self.dead.add(object_id)
                destroyed.append(object_id)
        if "notDestroyed" in result:
            for object_id, error in result["notDestroyed"].items():
                destroy.remove(object_id)
                if object_id not in self.live:
                    assert error["type"] == "notFound"
                else:
                    raise AssertionError(f"why wasn't {object_id} destroyed? {error!r}")
        assert destroy == set()

        by_id = SampleComparator.id()
        matching: List[Tuple[List[TypedKey], str]] = [
            ([cmp.key(obj) for cmp in self.sort] + [by_id.key(obj)], obj["id"])
            for obj in self.live.values()
            if self.filter is None or self.filter.matches(obj)
        ]
        matching.sort()
        query_ids = [row[1] for row in matching]

        query_all = dict(self.base_query, calculateTotal=True)
        ((name, query_result),) = yield ("Sample/query", query_all)
        assert name == "Sample/query"

        query_state = query_result.pop("queryState")
        del query_result["canCalculateChanges"]  # skip: either way is valid
        limit = query_result.pop("limit", len(query_ids))
        assert query_result == {
            "accountId": self.ACCOUNT_ID,
            "position": 0,
            "ids": query_ids[:limit],
            "total": len(query_ids),
        }

        # Compared to every past query we've done, either the queryState
        # has changed, or the query results must not have changed. "This
        # string MUST change if the results of the query (i.e., the
        # matching ids and their sort order) have changed. ... There is
        # no requirement for it to change if a property on an object
        # matching the query changes but the query results are
        # unaffected..."
        for past in reversed(self.states):
            assert past.query_state != query_state or past.query == query_ids

        self.states.append(
            PastState(
                state=result["newState"],
                query_state=query_state,
                created=frozenset(created),
                updated=frozenset(updated),
                destroyed=frozenset(destroyed),
                query=query_ids,
            )
        )

    @stateful.rule(bad_state=st.text())  # type: ignore
    def state_mismatch(self, bad_state: str) -> None:
        assume(bad_state != self.states[-1].state)
        test = self.exact(
            "Sample/set",
            {"accountId": self.ACCOUNT_ID, "ifInState": bad_state},
            "error",
            {"type": "stateMismatch"},
        )
        self.submit([test])


# This test takes time quadratic in the number of steps but tests a lot
# in each step, so let's keep the number of steps pretty small.
ConsistentHistory.TestCase.settings = settings(
    stateful_step_count=5,
    deadline=None,
)
ConsistentHistoryTestCase = ConsistentHistory.TestCase
