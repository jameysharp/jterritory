from __future__ import annotations

from hypothesis import strategies as st
from jterritory.types import Id


# FIXME: the Pydantic Hypothesis plugin should have registered ConstrainedStr
st.register_type_strategy(
    Id,
    st.text(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_",
        min_size=1,
        max_size=255,
    ).map(Id),
)
