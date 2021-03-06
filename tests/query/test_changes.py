from __future__ import annotations

from hypothesis import given, note, strategies as st
from typing import List
from jterritory.query.changes import Changes
from jterritory.types import ObjectId


def objectId(size: int = 10) -> st.SearchStrategy[ObjectId]:
    return st.integers(0, size - 1).map(ObjectId)


def query(size: int = 10) -> st.SearchStrategy[List[ObjectId]]:
    return st.lists(objectId(), max_size=size, unique=True)


def patch(old: List[ObjectId], diff: Changes) -> List[ObjectId]:
    """
    Naive implementation of the specification for applying /queryChanges
    results. Do not use this on large lists, because it's O(n^2).
    """
    new = list(old)

    assert diff.removed == sorted(diff.removed)
    for rem in reversed(diff.removed):
        assert 0 <= rem.position < len(new)
        assert new[rem.position] == rem.objectId
        del new[rem.position]

    assert diff.added == sorted(diff.added)
    for ins in diff.added:
        assert 0 <= ins.position <= len(new)
        new.insert(ins.position, ins.objectId)

    return new


@given(query(), query())
def test_diff(old: List[ObjectId], new: List[ObjectId]) -> None:
    diff = Changes.diff(old, new)
    assert patch(old, diff) == new


@given(query(), query())
def test_diff_reverse(old: List[ObjectId], new: List[ObjectId]) -> None:
    diff = Changes.diff(new, old).reverse()
    assert patch(old, diff) == new


@given(query(), query())
def test_merge_left_identity(old: List[ObjectId], new: List[ObjectId]) -> None:
    diff = Changes.diff(old, new)
    empty = Changes([], [])
    assert empty.merge(diff) == diff


@given(query(), query())
def test_merge_right_identity(old: List[ObjectId], new: List[ObjectId]) -> None:
    diff = Changes.diff(old, new)
    empty = Changes([], [])
    assert diff.merge(empty) == diff


@given(query(), query(), query())
def test_merge(a: List[ObjectId], b: List[ObjectId], c: List[ObjectId]) -> None:
    diff1 = Changes.diff(a, b)
    note(f"diff(a, b) == {diff1}")
    diff2 = Changes.diff(b, c)
    note(f"diff(b, c) == {diff2}")
    merged = diff1.merge(diff2)
    note(f"diff(a, c) == {merged}")
    best = Changes.diff(a, c)
    note(f"correct    == {best}")
    assert patch(a, merged) == c
    # It would be nice if this were true too but it isn't necessary:
    # assert merged == best


@given(query(), query(), query(), query())
def test_merge_associative(
    a: List[ObjectId], b: List[ObjectId], c: List[ObjectId], d: List[ObjectId]
) -> None:
    diff1 = Changes.diff(a, b)
    note(f"diff(a, b) == {diff1}")
    diff2 = Changes.diff(b, c)
    note(f"diff(b, c) == {diff2}")
    diff3 = Changes.diff(c, d)
    note(f"diff(c, d) == {diff3}")

    l = diff1.merge(diff2).merge(diff3)
    r = diff1.merge(diff2.merge(diff3))
    assert l == r
