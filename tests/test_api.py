from __future__ import annotations

from hypothesis import assume, event, example, given, infer, strategies as st
import json
import pytest
from sqlalchemy.future import create_engine
from typing import Dict, List, NamedTuple, Optional, Protocol, TypeVar
from jterritory.api import Endpoint, Invocation, Response
from jterritory.types import Id, String
from jterritory.methods import core


def st_json(max_size: int) -> st.SearchStrategy[object]:
    return st.recursive(
        st.one_of(
            st.none(),
            st.booleans(),
            st.floats(allow_infinity=False, allow_nan=False),
            st.text(),
        ),
        lambda children: st.one_of(
            st.lists(children, max_size=max_size),
            st.dictionaries(st.text(), children, max_size=max_size),
        ),
    )


st_invocations = st.lists(
    st.builds(
        Invocation,
        st.just("Core/echo"),
        st.dictionaries(st.text().filter(lambda k: not k.startswith("#")), st_json(3)),
        st.text(),
    ),
    max_size=5,
)


@pytest.fixture(scope="module")
def endpoint() -> Endpoint:
    return Endpoint(
        capabilities=set(),
        methods=core.methods,
        engine=create_engine("sqlite://"),
    )


@given(body=infer)
def test_not_json(endpoint: Endpoint, body: bytes) -> None:
    try:
        json.loads(body.decode())
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    else:
        event("accidentally generated valid JSON")
        assume(False)

    assert endpoint.request(body).dict()["type"] == "urn:ietf:params:jmap:error:notJSON"


@given(st_json(6))
# https://github.com/samuelcolvin/pydantic/issues/2762:
@example([[None, None]])
@example({"__pydantic_self__": None})
def test_not_request(endpoint: Endpoint, data: object) -> None:
    body = json.dumps(data).encode()
    assert (
        endpoint.request(body).dict()["type"] == "urn:ietf:params:jmap:error:notRequest"
    )


@given(st_invocations, st.none() | st.just(2))
def test_echo(
    endpoint: Endpoint, calls: List[Invocation], indent: Optional[int]
) -> None:
    body = json.dumps({"using": [], "methodCalls": calls}, indent=indent).encode()
    response = endpoint.request(body)
    assert isinstance(response, Response)
    assert response.method_responses == calls


@given(created_ids=infer)
def test_preserves_created_ids(
    endpoint: Endpoint, created_ids: Optional[Dict[Id, Id]]
) -> None:
    body = json.dumps(
        {
            "using": [],
            "methodCalls": [],
            "createdIds": created_ids,
        }
    ).encode()
    response = endpoint.request(body)
    assert isinstance(response, Response)
    assert response.created_ids == created_ids


T = TypeVar("T")


class DrawProtocol(Protocol):
    def __call__(self, base: st.SearchStrategy[T]) -> T:
        ...


class RandomPointer(NamedTuple):
    sample_json: object
    path: str
    expect_list: bool

    @st.composite
    @staticmethod
    def strategy(draw: DrawProtocol) -> RandomPointer:
        path = draw(
            st.lists(
                st.integers(min_value=0, max_value=9) | st.just("*"),
                min_size=1,
                max_size=5,
            )
        )

        strategy: st.SearchStrategy[object] = draw(
            st.sampled_from((st.booleans(), st.integers()))
        )

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
    def list_strategy(
        token: int, child: st.SearchStrategy[T]
    ) -> st.SearchStrategy[List[Optional[T]]]:
        def make(v: T) -> List[Optional[T]]:
            l: List[Optional[T]] = [None] * token
            l.append(v)
            return l

        return child.map(make)

    def expected(self) -> object:
        def go(within: object) -> None:
            if isinstance(within, list):
                for e in within:
                    if e is not None:
                        go(e)
            elif isinstance(within, dict):
                for e in within.values():
                    go(e)
            else:
                leaves.append(within)

        leaves: List[object] = []
        go(self.sample_json)
        if self.expect_list:
            return leaves
        assert len(leaves) == 1
        return leaves[0]


st.register_type_strategy(RandomPointer, RandomPointer.strategy())


@given(ptr=infer)
def test_result_reference(endpoint: Endpoint, ptr: RandomPointer) -> None:
    assert isinstance(ptr.sample_json, dict)
    first_call = Invocation(String("Core/echo"), ptr.sample_json, String("first_call"))
    second_call = Invocation(
        String("Core/echo"),
        {"#result": {"result_of": "first_call", "name": "Core/echo", "path": ptr.path}},
        String("second_call"),
    )

    body = json.dumps({"using": [], "methodCalls": [first_call, second_call]}).encode()
    response = endpoint.request(body)
    assert isinstance(response, Response)
    assert response.method_responses == [
        first_call,
        Invocation(
            String("Core/echo"), {"result": ptr.expected()}, String("second_call")
        ),
    ]
