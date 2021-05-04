from bisect import bisect_left
from typing import Iterable, List, NamedTuple, Sequence
from ..types import ObjectId, ObjectPosition


class ChangeValidityError(Exception):
    pass


class Changes(NamedTuple):
    # A /queryChanges response only needs the set of removed objects,
    # but internally we also need to know what index they're removed
    # from for cache purposes.
    removed: Sequence[ObjectPosition]
    added: Sequence[ObjectPosition]

    @classmethod
    def diff(cls, old: Sequence[ObjectId], new: Sequence[ObjectId]) -> "Changes":
        byId = {objectId: idx for idx, objectId in enumerate(old)}

        # If the sort order changed for this query, we have to remove
        # some items and re-add them in different spots, but we have
        # many choices for how to do that. The optimal choice is to find
        # any longest subsequence of old items which are in the same
        # order in the new list.
        # https://en.wikipedia.org/wiki/Longest_increasing_subsequence
        matching = [(-1, -1)]
        prior = []
        for new_idx, objectId in enumerate(new):
            try:
                old_idx = byId[objectId]
            except KeyError:
                prior.append(-1)
                continue

            match = (old_idx, new_idx)
            at = bisect_left(matching, match, 1)

            prior.append(matching[at - 1][1])
            if at == len(matching):
                matching.append(match)
            else:
                matching[at] = match

        _, new_idx = matching.pop()
        del matching

        added = []
        for position in reversed(range(len(new))):
            if new_idx < position:
                # This new item is not part of a longest increasing
                # subsequence of the old items, so it needs to be
                # inserted appropriately.
                added.append(ObjectPosition(position, new[position]))
            else:
                # This item is part of a longest increasing subsequence,
                # so don't remove/re-add it.
                assert new_idx == position
                new_idx = prior[position]
                byId.pop(new[position], None)

        added.reverse()
        removed = sorted(ObjectPosition(v, k) for k, v in byId.items())
        return cls(added=added, removed=removed)

    def merge(self, other: "Changes") -> "Changes":
        # Initially we have four sequences of changes that are logically
        # supposed to be applied in this order:
        # 1. self.removed
        # 2. self.added
        # 3. other.removed
        # 4. other.added
        #
        # First, swap the order of application of self.added and
        # other.removed. `MergeHelper` allows concisely comparing the
        # next element from each list; if one list is empty then it
        # always sorts after the other one.
        oldadd = MergeHelper(self.added)
        newrem = MergeHelper(other.removed)
        tmprem: List[ObjectPosition] = []
        tmpadd: List[ObjectPosition] = []
        while oldadd or newrem:
            if newrem < oldadd:
                # For each removed object, decrement its position once
                # for each object that had originally been added at an
                # earlier position, because after this swap we're adding
                # those after the removals, not before.
                tmprem.append(newrem.next().offset(-len(tmpadd)))
            elif oldadd < newrem:
                # Added objects are the same, in the other direction.
                tmpadd.append(oldadd.next().offset(-len(tmprem)))
            elif oldadd.next() != newrem.next():
                # If neither the add nor the remove sort before the
                # other, then they're at the same position. In that case
                # we know the only object that other should be able to
                # remove there is the one that self added. If they are
                # not the same object, then these two changes can't be
                # merged.
                raise ChangeValidityError()
            else:
                # Adding and then removing an object cancels out, so we
                # don't do anything with this object. But subsequent
                # positions still need to change as if we did both of
                # the above branches.
                oldadd.offset -= 1
                newrem.offset -= 1

        # Now we have four sequences of changes that are logically
        # supposed to be applied in this order:
        # 1. self.removed
        # 2. tmprem (from other.removed)
        # 3. tmpadd (from self.added)
        # 4. other.added
        #
        # Note that the positions in step 1 must be correct for the
        # original sequence of objects, and in step 4 must be correct
        # for the final sequence of objects, so those "outer" changes
        # need to appear in the output unchanged. But the "inner"
        # changes (2 and 3) have positions that need to be modified to
        # reflect the outer positions. So we have to merge 2 outward
        # into 1, and 3 outward into 4.
        return Changes(
            removed=MergeHelper.merge(self.removed, tmprem),
            added=MergeHelper.merge(other.added, tmpadd),
        )

    def reverse(self) -> "Changes":
        return Changes(removed=self.added, added=self.removed)

    def upTo(self, length: int) -> "Changes":
        removeTo = bisect_left(
            self.removed,
            ObjectPosition(length, ObjectId(0)),
        )
        # XXX: spec says 'any ids that were added but have a higher
        # index than "upToId" SHOULD be omitted,' but shouldn't that
        # interact with the number of items removed before that
        # position, somehow?
        addTo = bisect_left(
            self.added,
            ObjectPosition(length, ObjectId(0)),
        )
        return Changes(
            self.removed[:removeTo],
            self.added[:addTo],
        )


class MergeHelper:
    def __init__(self, objects: Iterable[ObjectPosition]) -> None:
        self.it = iter(objects)
        self.obj = next(self.it, None)
        self.offset = 0

    def __bool__(self) -> bool:
        return self.obj is not None

    def __lt__(self, other: "MergeHelper") -> bool:
        if self.obj is None:
            return False
        if other.obj is None:
            return True
        return self.obj.position + self.offset < other.obj.position + other.offset

    def next(self) -> ObjectPosition:
        assert self.obj is not None
        done = self.obj.offset(self.offset)
        self.obj = next(self.it, None)
        return done

    @classmethod
    def merge(
        cls, outer_iter: Iterable[ObjectPosition], inner_iter: Iterable[ObjectPosition]
    ) -> List[ObjectPosition]:
        inner = cls(inner_iter)
        outer = cls(outer_iter)
        result = []
        while inner or outer:
            if inner < outer:
                result.append(inner.next())
            else:
                result.append(outer.next())
                inner.offset += 1
        return result
