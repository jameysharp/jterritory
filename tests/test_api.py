from hypothesis import assume, event, given, strategies as st
import json
import pytest
from sqlalchemy.future import create_engine
from typing import Any, NamedTuple
from jterritory.api import Endpoint, Invocation
from jterritory.methods.core import echo

st_json = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.floats(allow_infinity=False, allow_nan=False),
        st.text(),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(), children, max_size=5),
    ),
)
st_id = st.text(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_",
    min_size=1,
    max_size=255,
)
st_invocations = st.lists(
    st.builds(
        Invocation,
        st.just("Core/echo"),
        st.dictionaries(st.text().filter(lambda k: not k.startswith("#")), st_json),
        st.text(),
    ),
    max_size=5,
)


@pytest.fixture(scope="module")
def endpoint():
    return Endpoint(
        capabilities=set(),
        methods={"Core/echo": echo},
        engine=create_engine("sqlite://"),
    )


@given(st.binary())
def test_not_json(endpoint, body):
    try:
        json.loads(body.decode())
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    else:
        event("accidentally generated valid JSON")
        assume(False)

    assert endpoint.request(body).dict()["type"] == "urn:ietf:params:jmap:error:notJSON"


@given(st_json)
def test_not_request(endpoint, data):
    body = json.dumps(data).encode()
    assert (
        endpoint.request(body).dict()["type"] == "urn:ietf:params:jmap:error:notRequest"
    )


@given(st_invocations, st.none() | st.just(2))
def test_echo(endpoint, calls, indent):
    body = json.dumps({"using": [], "methodCalls": calls}, indent=indent).encode()
    assert endpoint.request(body).method_responses == calls


@given(st.none() | st.dictionaries(st_id, st_id))
def test_preserves_created_ids(endpoint, created_ids):
    body = json.dumps(
        {
            "using": [],
            "methodCalls": [],
            "createdIds": created_ids,
        }
    ).encode()
    assert endpoint.request(body).created_ids == created_ids


class RandomPointer(NamedTuple):
    sample_json: Any
    path: str
    expect_list: bool

    @st.composite
    def strategy(draw) -> "RandomPointer":
        path = draw(
            st.lists(
                st.integers(min_value=0, max_value=9) | st.just("*"),
                min_size=1,
                max_size=5,
            )
        )

        strategy = draw(st.sampled_from((st.booleans(), st.integers())))

        expect_list = False
        if draw(st.booleans()):
            strategy = st.lists(strategy)
            expect_list = True

        for token in reversed(path[1:]):
            if draw(st.booleans()):
                strategy = st.fixed_dictionaries({str(token): strategy})
            elif isinstance(token, int):
                strategy = RandomPointer.list_strategy(token, strategy)
            else:
                strategy = st.lists(strategy)
                expect_list = True

        strategy = st.fixed_dictionaries({str(path[0]): strategy})

        return RandomPointer(
            sample_json=draw(strategy),
            path="/" + "/".join(map(str, path)),
            expect_list=expect_list,
        )

    @staticmethod
    def list_strategy(token: int, child):
        def make(v):
            def setter(l):
                l[token] = v
                return l

            return st.lists(st.none(), min_size=token + 1).map(setter)

        return child.flatmap(make)

    def expected(self) -> Any:
        def go(within: Any) -> None:
            if isinstance(within, list):
                for e in within:
                    if e is not None:
                        go(e)
            elif isinstance(within, dict):
                for e in within.values():
                    go(e)
            else:
                leaves.append(within)

        leaves = []
        go(self.sample_json)
        if self.expect_list:
            return leaves
        assert len(leaves) == 1
        return leaves[0]


@given(RandomPointer.strategy())
def test_result_reference(endpoint, ptr: RandomPointer):
    first_call = Invocation("Core/echo", ptr.sample_json, "first_call")
    second_call = Invocation(
        "Core/echo",
        {"#result": {"result_of": "first_call", "name": "Core/echo", "path": ptr.path}},
        "second_call",
    )

    body = json.dumps({"using": [], "methodCalls": [first_call, second_call]}).encode()
    assert endpoint.request(body).method_responses == [
        first_call,
        Invocation("Core/echo", {"result": ptr.expected()}, "second_call"),
    ]
