from hypothesis import given, strategies as st
from jterritory.api import Comparator


@given(
    l=st.one_of(
        st.lists(st.booleans()),
        st.lists(st.integers() | st.floats(allow_nan=False)),
    ),
    reverse=st.booleans(),
)
def test_sortkey(l, reverse):
    sort = Comparator(property="foo", is_ascending=not reverse).compile()
    assert sorted(l, reverse=reverse) == sorted(l, key=sort.key)


@given(l=st.lists(st.text()), reverse=st.booleans())
def test_sortkey_strings(l, reverse):
    sort = Comparator(property="foo", is_ascending=not reverse).compile()
    assert sorted(l, key=str.casefold, reverse=reverse) == sorted(l, key=sort.key)
