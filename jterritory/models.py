from __future__ import annotations

from sqlalchemy import Column, MetaData, Table
from sqlalchemy import Boolean, Integer, JSON, Text
from sqlalchemy import ForeignKey, UniqueConstraint


metadata = MetaData()

accounts = Table(
    "accounts",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("account", Text, nullable=False, unique=True),
    # Not used by jterritory but if a particular application wants to
    # store data with an account then they can use this column.
    Column("details", JSON),
)

# To support the `/changes` endpoint, we need to be able to find the
# objects which have changed state since a specified point in the past,
# but note that we always want to find changes up to the current state.
# So we only need to know the change sequence number for when the object
# was created and when it was last updated or destroyed, not any updates
# in between.
#
# If the object was destroyed after the client's last-seen state, then
# we need to report that it was destroyed, unless it was also
# newly-created since then, in which case we don't need to report it at
# all.
#
# Otherwise, if the object was newly-created, that's what we need to
# report, regardless of how many times it may have been updated since.
#
# Otherwise, if it changed at all then we need to report it was updated;
# again, regardless of how many times it was updated.
#
# In any case, we need to find all objects that underwent any change
# since the specified state, so keep a single "last changed" column to
# index and query against; then sort out how to report the changes using
# the other columns.
#
# Supporting `/queryChanges` is more involved, as is making `/query`
# efficient.
objects = Table(
    "objects",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("account", ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False),
    # Application-assigned number representing a single datatype, such
    # as "Email" or "Todo". This allows applications to add functional
    # indexes to this metadata object to speed up common queries.
    Column("datatype", Integer, nullable=False),
    Column("contents", JSON),
    # When this object is created, updated, or destroyed, set `changed`
    # higher than any existing object of this type in this account.
    Column("changed", Integer, nullable=False),
    # When this object is created, set `created` to the new value of
    # `changed`.
    Column("created", Integer, nullable=False),
    Column("destroyed", Boolean, nullable=False),
    UniqueConstraint("account", "datatype", "changed"),
)

# An application using this project can speed up common queries by
# evaluating statements like these before calling metadata.create_all().
# In this example, "5" is the datatype ID that the application has
# assigned to "Email" objects.
#
# Index(
#     "ix_email_thread",
#     models.objects.c.account,
#     models.objects.c.contents["thread"],
#     postgresql_where=models.objects.c.datatype == 5,
#     sqlite_where=models.objects.c.datatype == 5,
# )
#
# See:
# https://docs.sqlalchemy.org/en/14/core/constraints.html#functional-indexes
# https://docs.sqlalchemy.org/en/14/dialects/postgresql.html#partial-indexes
# https://docs.sqlalchemy.org/en/14/dialects/sqlite.html#partial-indexes
