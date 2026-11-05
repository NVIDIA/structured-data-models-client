# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source identity checks must not repair inputs or excuse changed values."""

import json

import pandas as pd
import pytest
from python_runner import Incomplete, check_snapshot, tables_of
from shop_fixtures import load_fixture

from contract import HERE


def test_raw_source_verification_keeps_reader_values_and_rejects_corruption():
    fixture = load_fixture(HERE / 'fixtures/shop.json')
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

    fixture = load_fixture(HERE / 'fixtures' / fixture_name)
    with contextlib.ExitStack() as stack:
        _, tables, _ = source_tables(stack, 'parquet', {}, fixture)
        before = {name: frame.copy(deep=True) for name, frame in tables.items()}
        check_snapshot(tables, fixture, raw_reader=True)
        for name in tables:
            pd.testing.assert_frame_equal(tables[name], before[name])


def test_all_shop_variants_keep_their_independent_fixture_identity():
    from contract import digest, manifest

    hashes = {}
    for capability in manifest()['capabilities']:
        for required in capability['requirements']:
            for path, expected in required['fixture_hashes'].items():
                assert hashes.setdefault(path, expected) == expected
    assert set(hashes) == {
        'fixtures/' + path.name for path in (HERE / 'fixtures').glob('*.json')
    }
    for path, expected in hashes.items():
        assert digest(load_fixture(HERE / path)) == expected, path


def test_shop_variants_preserve_literal_null_types_order_and_isolation():
    def users(name):
        return load_fixture(HERE / 'fixtures' / name)['tables']['users']

    numeric = users('shop-numeric.json')
    assert numeric['columns'] == [
        'id',
        'value',
        'class',
        'flag',
        'narrow',
        'wide',
        'fraction',
        'single',
    ]
    assert numeric['rows'][0][4:] == [None, 9007199254740993, None, 0.0]
    assert type(numeric['rows'][0][5]) is int
    assert type(numeric['rows'][0][7]) is float
    assert numeric['rows'][1][6] == '1.123456'
    assert numeric['types']['fraction'] == 'decimal128(20,6)'
    tags = users('shop-tags.json')
    assert [row[-1] for row in tags['rows'][:4]] == [
        ['red', 'blue'],
        [],
        None,
        ['green', None],
    ]
    nulls = users('shop-null.json')
    assert nulls['rows'][1][-2] is True
    assert all(row[-1] is None for row in nulls['rows'])
    assert nulls['types']['all_null'] == 'string'
    times = users('shop-temporal.json')
    assert times['rows'][1][-1] == '2024-03-10T03:00:00.987654321-07:00'
    assert times['types']['observed_at'] == 'timestamp[ns,America/Los_Angeles]'
    upper = load_fixture(HERE / 'fixtures/shop-snowflake-semantic.json')
    assert upper['primary_keys']['users'] == 'ID'
    assert upper['links'] == [
        ['events.ID', 'users.ID'],
        ['events.ITEM', 'items.ITEM'],
    ]
    numeric['rows'][0][0] = -123
    assert users('shop-numeric.json')['rows'][0][0] == 0
    assert len(users('shop.json')['columns']) == 4


def test_invalid_shop_variant_cannot_silently_truncate_rows(tmp_path):
    import shutil

    shutil.copy(HERE / 'fixtures/shop.json', tmp_path / 'shop.json')
    variant = tmp_path / 'variant.json'
    variant.write_text(
        json.dumps(
            {
                'base': 'shop.json',
                'user_columns': [{'name': 'extra', 'values': [1]}],
            }
        )
    )
    with pytest.raises(ValueError, match='one value per row'):
        load_fixture(variant)
