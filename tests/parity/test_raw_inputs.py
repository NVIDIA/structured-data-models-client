# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source identity checks must not repair inputs or excuse changed values."""

import json

import pandas as pd
import pytest
from python_runner import Incomplete, check_snapshot, tables_of

from contract import HERE


def test_raw_source_verification_keeps_reader_values_and_rejects_corruption():
    fixture = json.loads((HERE / 'fixtures/shop.json').read_text())
    tables = tables_of(fixture)
    tables['users']['class'] = tables['users']['class'].astype(int)
    tables['users']['flag'] = tables['users']['flag'].astype(int)
    tables['events']['event_time'] = tables['events']['event_time'].dt.strftime(
        '%Y-%m-%d'
    )
    before = {name: frame.copy(deep=True) for name, frame in tables.items()}
    check_snapshot(tables, fixture, raw_reader=True)
    for name in tables:
        pd.testing.assert_frame_equal(tables[name], before[name])
    tables['users'].loc[0, 'flag'] = 2
    with pytest.raises(Incomplete, match='source values differ'):
        check_snapshot(tables, fixture, raw_reader=True)
    tables['users'].loc[0, 'flag'] = 1
    tables['users'].loc[0, 'class'] = 7
    with pytest.raises(Incomplete, match='source values differ'):
        check_snapshot(tables, fixture, raw_reader=True)


def test_missing_slot_proof_does_not_erase_nonmissing_nan_or_values():
    from python_runner import fixture_nulls

    assert fixture_nulls([3.0, {'float': 'nan'}], [3.0, None]) == [3.0, None]
    assert fixture_nulls({'float': 'nan'}, 1.0) == {'float': 'nan'}
    assert fixture_nulls(2.0, None) == 2.0
    assert fixture_nulls([1.0], [1.0, None]) == [1.0]


@pytest.mark.parametrize(
    'fixture_name', ['shop-tags.json', 'shop-sequence.json']
)
def test_nested_parquet_fixture_is_readable_without_pandas_extension_metadata(
    fixture_name,
):
    import contextlib

    from python_runner import source_tables

    fixture = json.loads((HERE / 'fixtures' / fixture_name).read_text())
    with contextlib.ExitStack() as stack:
        _, tables, _ = source_tables(stack, 'parquet', {}, fixture)
        before = {name: frame.copy(deep=True) for name, frame in tables.items()}
        check_snapshot(tables, fixture, raw_reader=True)
        for name in tables:
            pd.testing.assert_frame_equal(tables[name], before[name])
