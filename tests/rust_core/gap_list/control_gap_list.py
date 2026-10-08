# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Two known tests the Phase 7.8 checks measure, to show the measure
works (``CONTRACT.md``, "Measuring tests"). Not collected by the suites:
the checks name this file to pytest themselves."""

from __future__ import annotations


def test_takes_the_naive_utc_branch_and_calls_the_core() -> None:
    import pandas as pd
    from sdm_core import _native
    from sdm_relational_engine.rfm.base.utils import to_naive_utc

    zoned = pd.Series(pd.to_datetime(['2024-01-01T00:00:00+01:00']))
    assert str(to_naive_utc(zoned).iloc[0]) == '2023-12-31 23:00:00'
    assert _native.version()


def test_takes_nothing_and_calls_nothing() -> None:
    assert sum(range(4)) == 6
