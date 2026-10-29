# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete same-read Reader data precedes native captured-time ownership.

The explicit Arrow products below supply facts, not a Python time parser or
condition evaluator. Actual SQL/Host journeys separately own those products.
"""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from sdm_client._rust import native_sql
from sdm_client._rust.codec import CodecError
from sdm_client._rust.native_products import Product
from sdm_client._rust.native_source_completion import NativeSourceTable

ENTITY = {'source': ['users']}
OTHER = {'source': ['other']}


def view(
    source, read, values, *, role='windowed', columns=('ts',), positions=None
):
    rows = len(values)
    metadata = {
        'source_role': role,
        'source_read': read,
        'source': source,
        'columns': list(columns),
        'rows': rows,
    }
    arrays = [
        pa.array(
            range(rows) if positions is None else positions, type=pa.uint64()
        )
    ]
    arrays.extend(pa.array(values, type=pa.timestamp('ns')) for _ in columns)
    return json.dumps(metadata).encode(), pa.Table.from_arrays(
        arrays, names=['source_row', *columns]
    )


class Owner:
    def __init__(self, views):
        self.views = views
        self.calls = 0

    def temporal_sources(self):
        self.calls += 1
        return tuple(self.views)

    def rows(self):
        raise AssertionError(
            'a selected/repeated candidate view is not a complete source'
        )


class Preparation:
    def __init__(self):
        self.failures = []

    def native_failure(self, failure):
        self.failures.append(failure.error)
        raise failure.error


def complete(views, entity, windowed, preparation=None):
    owner = Owner(views)
    product = Product('label_inputs', {'owner': 5}, {5: owner})
    preparation = preparation or Preparation()
    native_sql.complete_source_times(
        product,
        SimpleNamespace(tables=entity),
        SimpleNamespace(tables=windowed),
        preparation,
    )
    assert owner.calls == 1
    return preparation


def test_complete_read_rows_include_entity_and_other_table_before_native_capture(
    monkeypatch,
):
    entity = NativeSourceTable(
        ENTITY,
        pd.DataFrame({'ts': pd.Series(['raw'] * 4, dtype='string')}),
        'ts',
    )
    other = NativeSourceTable(
        OTHER,
        pd.DataFrame(
            {
                'ts': pd.Series(['raw'] * 3, dtype='string'),
                'condition_time': pd.Series(['raw'] * 3, dtype='string'),
            }
        ),
        'ts',
    )
    original = [
        view(ENTITY, 7, [1, None, 3, 4], role='entity'),
        view(OTHER, 9, [11, 12, 13], columns=('ts', 'condition_time')),
    ]
    snapshots = [(metadata, table.to_pydict()) for metadata, table in original]

    def no_parser(*args, **kwargs):
        raise AssertionError('canonical Reader data must not be parsed again')

    monkeypatch.setattr(pd, 'to_datetime', no_parser)
    complete(original, {7: entity}, {9: other})
    assert entity.frame['ts'].dtype == 'datetime64[ns]'
    assert entity.frame['ts'].astype('int64').tolist() == [1, -(2**63), 3, 4]
    assert other.frame['ts'].astype('int64').tolist() == [11, 12, 13]
    assert other.frame['condition_time'].astype('int64').tolist() == [
        11,
        12,
        13,
    ]
    assert [
        (metadata, table.to_pydict()) for metadata, table in original
    ] == snapshots
    captured = native_sql._FetchedColumns(other, Preparation())
    times = captured.time_series({'source': OTHER, 'column': 'ts'})
    times.iloc[0] = pd.Timestamp(21)
    assert captured.time_series({'source': OTHER, 'column': 'ts'}).iloc[
        0
    ] == pd.Timestamp(21)
    other.frame['ts'] = pd.Series([pd.Timestamp(99)] * 3)
    assert captured.time_series({'source': OTHER, 'column': 'ts'}) is times
    assert times.iloc[0] == pd.Timestamp(21)
    assert [
        (metadata, table.to_pydict()) for metadata, table in original
    ] == snapshots


@pytest.mark.parametrize('count', [0, 2])
def test_all_missing_source_storage_uses_complete_typed_reader_values(
    count, monkeypatch
):
    table = NativeSourceTable(
        ENTITY,
        pd.DataFrame({'ts': pd.Series([None] * count, dtype='Int64')}),
        'ts',
    )
    monkeypatch.setattr(
        pd, 'to_datetime', lambda *args, **kwargs: pytest.fail('not a parser')
    )
    complete([view(ENTITY, 7, [None] * count, role='entity')], {7: table}, {})
    assert str(table.frame['ts'].dtype) == 'datetime64[ns]'
    assert table.frame['ts'].isna().tolist() == [True] * count


def test_already_typed_native_backing_survives_without_canonical_overwrite():
    frame = pd.DataFrame({'ts': pd.Series([pd.Timestamp(7), pd.Timestamp(8)])})
    table = NativeSourceTable(ENTITY, frame, 'ts')
    captured = frame['ts']
    complete([view(ENTITY, 7, [100, 200], role='entity')], {7: table}, {})
    assert np.shares_memory(
        frame['ts'].to_numpy(copy=False), captured.to_numpy(copy=False)
    )
    assert captured.astype('int64').tolist() == [7, 8]


@pytest.mark.parametrize(
    'changed',
    [
        'read',
        'source',
        'role',
        'positions',
        'short',
        'column',
        'type',
        'extra',
        'duplicate',
    ],
)
def test_full_read_binding_is_validated_before_any_native_assignment(changed):
    first = NativeSourceTable(
        ENTITY,
        pd.DataFrame({'ts': pd.Series(['raw'] * 2, dtype='string')}),
        'ts',
    )
    second = NativeSourceTable(
        OTHER,
        pd.DataFrame({'ts': pd.Series(['raw'] * 2, dtype='string')}),
        'ts',
    )
    good = view(ENTITY, 7, [1, 2], role='entity')
    metadata, data = view(OTHER, 9, [3, 4])
    meaning = json.loads(metadata)
    if changed == 'read':
        meaning['source_read'] = 7
    elif changed == 'source':
        meaning['source'] = ENTITY
    elif changed == 'role':
        meaning['source_role'] = 'entity'
    elif changed == 'positions':
        data = data.set_column(
            0, 'source_row', pa.array([1, 0], type=pa.uint64())
        )
    elif changed == 'short':
        data = data.slice(0, 1)
    elif changed == 'column':
        meaning['columns'] = ['unread']
        data = data.rename_columns(['source_row', 'unread'])
    elif changed == 'type':
        data = data.set_column(1, 'ts', pa.array(['raw', 'raw']))
    elif changed == 'extra':
        meaning['trusted'] = True
    bad = (json.dumps(meaning).encode(), data)
    values = [good, bad] if changed != 'duplicate' else [good, good]
    before = [copy.deepcopy(first.frame), copy.deepcopy(second.frame)]
    preparation = Preparation()
    with pytest.raises(CodecError):
        complete(values, {7: first}, {9: second}, preparation)
    pd.testing.assert_frame_equal(first.frame, before[0])
    pd.testing.assert_frame_equal(second.frame, before[1])
    assert preparation.failures == []


def test_actual_typed_construction_failure_keeps_native_owner_and_stops(
    monkeypatch,
):
    selected = ValueError('native typed timestamp construction')
    table = NativeSourceTable(
        ENTITY, pd.DataFrame({'ts': pd.Series(['raw'], dtype='string')}), 'ts'
    )
    preparation = Preparation()
    actual_series = pd.Series

    def series(data=None, *args, **kwargs):
        if getattr(getattr(data, 'dtype', None), 'kind', None) == 'M':
            raise selected
        return actual_series(data, *args, **kwargs)

    monkeypatch.setattr(pd, 'Series', series)
    with pytest.raises(ValueError) as raised:
        complete(
            [view(ENTITY, 7, [1], role='entity')], {7: table}, {}, preparation
        )
    assert raised.value is selected
    assert preparation.failures == [selected]
    assert table.frame['ts'].tolist() == ['raw']


def test_retained_native_failure_releases_extra_read_views_but_not_caller_product(
    monkeypatch,
):
    import weakref
    from sdm_client._rust import native_sql_sources

    source_rows = []

    class FreshOwner:
        def temporal_sources(self):
            descriptor, rows = view(ENTITY, 7, [1], role='entity')
            source_rows.append(weakref.ref(rows))
            return ((descriptor, rows),)

    owner = FreshOwner()
    product = Product('label_inputs', {'owner': 5}, {5: owner})
    table = NativeSourceTable(
        ENTITY, pd.DataFrame({'ts': pd.Series(['raw'], dtype='string')}), 'ts'
    )
    selected = ValueError('native timestamp finishing')

    def fail(values):
        assert str(values.dtype) == 'datetime64[ns]'
        raise selected

    monkeypatch.setattr(native_sql_sources, 'finish_timestamp', fail)
    preparation = Preparation()
    with pytest.raises(ValueError) as raised:
        native_sql.complete_source_times(
            product,
            SimpleNamespace(tables={7: table}),
            SimpleNamespace(tables={}),
            preparation,
        )
    assert raised.value is selected
    assert selected.__traceback__ is not None
    assert len(source_rows) == 1 and source_rows[0]() is None
    assert product.owners == {5: owner}
    assert table.frame['ts'].tolist() == ['raw']
