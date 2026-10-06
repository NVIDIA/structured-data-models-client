# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Equivalence tests for the vectorized gathers in the request builder.

Both functions used to index row-at-a-time (``df.iloc[i][col]`` per edge, and a
full ``batch`` scan per instance). These pin the edge-case semantics the loops
had, so the faster versions cannot drift from them.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from sdm_relational_engine.rfm.inference.payload import _entity_values


def _context(
    batch: np.ndarray,
    df: pd.DataFrame,
    batch_size: int,
    row: np.ndarray | None = None,
) -> SimpleNamespace:
    table = SimpleNamespace(df=df, batch=batch, row=row, primary_key='pk')
    return SimpleNamespace(
        sampled_subgraph=SimpleNamespace(
            tables={'t': table},
            batch_size=batch_size,
        )
    )


def _reference(context, table_name: str):
    r"""The original row-at-a-time implementation, kept as the oracle."""
    table = context.sampled_subgraph.tables.get(table_name)
    if table is None or table.primary_key is None:
        return None
    if table.primary_key not in table.df:
        return None
    values: list = []
    batch = np.asarray(table.batch)
    df = table.df.reset_index(drop=True)
    row = np.asarray(table.row) if table.row is not None else None
    for instance_id in range(context.sampled_subgraph.batch_size):
        rows = np.flatnonzero(batch == instance_id)
        if len(rows) == 0:
            values.append(None)
            continue
        row_index = int(rows[0])
        if row is not None:
            row_index = int(row[row_index])
        if row_index >= len(df):
            values.append(None)
            continue
        values.append(df.iloc[row_index][table.primary_key])
    return values


def _frame(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            'pk': np.arange(100, 100 + n),
            'f': rng.random(n),
            's': rng.choice(list('abc'), n),
        }
    )


def test_entity_values_matches_the_row_at_a_time_reference() -> None:
    df = _frame(12)
    cases = [
        # (batch, batch_size, row), sorted, unsorted, gaps, repeats, empty,
        # a row remap, and a remap pointing past the frame.
        (np.array([0, 0, 1, 1, 2]), 3, None),
        (np.array([2, 0, 1, 0, 2]), 3, None),
        (np.array([0, 0, 3, 3]), 5, None),
        (np.array([], dtype=np.int64), 3, None),
        (np.array([0, 1, 2]), 3, np.array([11, 5, 0])),
        (np.array([0, 1, 2]), 3, np.array([99, 1, 2])),
    ]
    for batch, batch_size, row in cases:
        context = _context(batch, df, batch_size, row)
        assert _entity_values(context, 't') == _reference(context, 't'), batch


def test_entity_values_takes_the_first_occurrence_of_each_instance() -> None:
    df = _frame(6)
    context = _context(np.array([1, 0, 1, 0, 2, 2]), df, 3)
    # First occurrence of instance 0 is position 1 -> pk 101; of 1, position 0
    # -> pk 100; of 2, position 4 -> pk 104.
    assert _entity_values(context, 't') == [101, 100, 104]


def test_entity_values_is_none_for_instances_absent_from_the_batch() -> None:
    context = _context(np.array([0, 0, 2]), _frame(3), 4)
    assert _entity_values(context, 't') == [100, None, 102, None]
