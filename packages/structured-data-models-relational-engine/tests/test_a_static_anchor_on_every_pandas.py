# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A static question's anchor on every pandas the client supports (T3.8).

A static question has no window: its step is ``pd.DateOffset(0)`` and its
anchor ``pd.Timestamp.min``. pandas 3.0 gives that anchor back unchanged when
the empty step is taken from it, and pandas 3.1 raises ``OutOfBoundsDatetime``
there, which refused every static question on the old route.
"""

import pandas as pd
from sdm_relational_engine.rfm.rfm import _before


def test_nothing_taken_from_the_smallest_time_is_that_time() -> None:
    assert _before(pd.Timestamp.min, pd.DateOffset(0)) == pd.Timestamp.min


def test_a_step_is_still_taken() -> None:
    assert _before(pd.Timestamp('2024-03-01'), pd.DateOffset(days=30)) == (
        pd.Timestamp('2024-01-31')
    )
    assert _before(pd.Timestamp('2024-03-31'), pd.DateOffset(months=1) * 2) == (
        pd.Timestamp('2024-01-31')
    )
