from hypothesis import settings, stateful, strategies as st
import json
from jterritory import models
from jterritory.api import Endpoint, Response
from jterritory.exceptions import method
from jterritory.methods.standard import BaseDatatype, SetRequest, StandardMethods
from jterritory.query.filter import FilterCondition
from jterritory.query.sort import Comparator
from jterritory.types import Id, Number, ObjectId, String
from pydantic.json import pydantic_encoder
from sqlalchemy.future import create_engine
from sqlalchemy.sql import ClauseElement
from typing import (
    cast,
    Dict,
    FrozenSet,
    List,
    Literal,
    NamedTuple,
    Protocol,
    Set,
    Tuple,
    TypeVar,
)


class Sample(BaseDatatype):
    number: Number
    text: String


st.register_type_strategy(
    Sample,
    st.builds(
        Sample,
        number=st.floats(allow_infinity=False, allow_nan=False),
        text=st.text(),
    ),
)


class PastState(NamedTuple):
    state: str
    created: FrozenSet[str] = frozenset()
    updated: FrozenSet[str] = frozenset()
    destroyed: FrozenSet[str] = frozenset()


T = TypeVar("T")


class DrawProtocol(Protocol):
    def __call__(self, base: st.SearchStrategy[T]) -> T:
        ...


class ConsistentHistory(stateful.RuleBasedStateMachine):
    CAPABILITY = String("urn:example:sample")
    ACCOUNT_ID = "some-account"

    def __init__(self) -> None:
        super().__init__()
        self.states: List[PastState] = [PastState("0")]
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

    def submit(self, calls: List[Tuple[str, dict, str]]) -> Response:
        body = json.dumps(
            {"using": [self.CAPABILITY], "method_calls": calls},
            default=pydantic_encoder,
        ).encode()
        response = self.endpoint.request(body)
        assert isinstance(response, Response)
        assert set(call[2] for call in calls) == set(
            call.call_id for call in response.method_responses
        )
        return response

    @stateful.invariant()  # type: ignore
    def check_history(self) -> None:
        calls: List[Tuple[str, dict, str]] = [
            ("Sample/get", {"accountId": self.ACCOUNT_ID}, "live"),
            (
                "Sample/get",
                {"accountId": self.ACCOUNT_ID, "ids": list(self.dead)},
                "dead",
            ),
        ]
        calls.extend(
            (
                "Sample/changes",
                {"accountId": self.ACCOUNT_ID, "since_state": past.state},
                f"since-{past.state}",
            )
            for past in reversed(self.states)
        )
        results = iter(self.submit(calls).method_responses)
        current_state = self.states[-1].state

        result = next(results)
        assert (result.name, result.call_id) == ("Sample/get", "live")
        assert set() == result.arguments["notFound"]
        assert self.live == {obj.pop("id"): obj for obj in result.arguments["list"]}
        assert current_state == result.arguments["state"]

        result = next(results)
        assert (result.name, result.call_id) == ("Sample/get", "dead")
        assert self.dead == set(result.arguments["notFound"])
        assert [] == result.arguments["list"]
        assert current_state == result.arguments["state"]

        created: Set[str] = set()
        updated: Set[str] = set()
        destroyed: Set[str] = set()
        for result, past in zip(results, reversed(self.states)):
            assert result.call_id == f"since-{past.state}"
            if result.name == "error":
                assert result.arguments["type"] == "CannotCalculateChanges"
                continue
            assert result.name == "Sample/changes"

            # This assertion is stricter than the spec requires, because
            # the same object _may_ be returned in multiple sets, but
            # this enforces these statements: 'If a record has been
            # created AND updated since the old state, the server SHOULD
            # just return the id in the "created" list'; 'If a record
            # has been updated AND destroyed since the old state, the
            # server SHOULD just return the id in the "destroyed" list';
            # 'If a record has been created AND destroyed since the old
            # state, the server SHOULD remove the id from the response
            # entirely.'
            assert result.arguments == {
                "accountId": self.ACCOUNT_ID,
                "oldState": past.state,
                "newState": current_state,
                "hasMoreChanges": False,  # TODO: test pagination
                "created": created,
                "updated": updated,
                "destroyed": destroyed,
            }

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
        create = request.create or {}
        update = request.update or {}
        destroy = request.destroy or set()

        response = self.submit([("Sample/set", request.dict(), "set-call")])
        result = response.method_responses[0]
        assert (result.name, result.call_id) == ("Sample/set", "set-call")
        assert response.method_responses[1:] == []
        assert result.arguments["oldState"] in (None, self.states[-1].state)

        created = []
        assert "notCreated" not in result.arguments
        if "created" in result.arguments:
            for creation_id, server_set in result.arguments["created"].items():
                requested_create = create.pop(creation_id)
                requested_create.update(server_set)
                object_id = requested_create.pop(String("id"))
                assert object_id not in self.live and object_id not in self.dead
                self.live[object_id] = requested_create
                created.append(object_id)
        assert create == {}

        updated = []
        if "updated" in result.arguments:
            for object_id, changes in result.arguments["updated"].items():
                requested_update = update.pop(object_id)
                if changes is not None:
                    requested_update.update(changes)
                self.live[object_id] = requested_update
                updated.append(object_id)
        if "notUpdated" in result.arguments:
            for object_id, error in result.arguments["notUpdated"].items():
                del update[object_id]
                if object_id not in self.live:
                    assert error["type"] == "notFound"
                elif object_id in destroy:
                    assert error["type"] == "willDestroy"
                else:
                    raise AssertionError(f"why wasn't {object_id} updated? {error!r}")
        assert update == {}

        destroyed = []
        if "destroyed" in result.arguments:
            for object_id in result.arguments["destroyed"]:
                destroy.remove(object_id)
                del self.live[object_id]
                self.dead.add(object_id)
                destroyed.append(object_id)
        if "notDestroyed" in result.arguments:
            for object_id, error in result.arguments["notDestroyed"].items():
                destroy.remove(object_id)
                if object_id not in self.live:
                    assert error["type"] == "notFound"
                else:
                    raise AssertionError(f"why wasn't {object_id} destroyed? {error!r}")
        assert destroy == set()

        self.states.append(
            PastState(
                state=result.arguments["newState"],
                created=frozenset(created),
                updated=frozenset(updated),
                destroyed=frozenset(destroyed),
            )
        )


# This test takes time quadratic in the number of steps but tests a lot
# in each step, so let's keep the number of steps pretty small.
ConsistentHistory.TestCase.settings = settings(
    stateful_step_count=5,
    deadline=None,
)
ConsistentHistoryTestCase = ConsistentHistory.TestCase


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


class SampleMethods(StandardMethods[SampleFilter, Comparator]):
    datatype = Sample
