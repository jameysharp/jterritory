from __future__ import annotations

from hypothesis import given, infer, strategies as st
from typing import List, Union
from jterritory.types import String
from jterritory.query.sort import Comparator


@given(
    l=st.one_of(
        st.lists(st.booleans()),
        st.lists(st.integers() | st.floats(allow_nan=False)),
    ),
    reverse=infer,
)
def test_sortkey(l: Union[List[bool], List[Union[int, float]]], reverse: bool) -> None:
    sort = Comparator(property=String("foo"), is_ascending=not reverse).compile()
    assert sorted(l, reverse=reverse) == sorted(l, key=sort.key)


@given(l=infer, reverse=infer)
def test_sortkey_strings(l: List[str], reverse: bool) -> None:
    sort = Comparator(property=String("foo"), is_ascending=not reverse).compile()
    assert sorted(l, key=str.casefold, reverse=reverse) == sorted(l, key=sort.key)
