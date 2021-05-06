from hypothesis import assume, event, given, strategies as st
import json
import pytest
from sqlalchemy.future import create_engine
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
