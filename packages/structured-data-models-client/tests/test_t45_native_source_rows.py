# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw source data keeps actual cursor lifetimes and native construction order."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pytest

from sdm_client._rust import native_source_rows as native
from sdm_client._rust.providers.postgres import ScalarRows
from sdm_client._rust.run import FetchRefused


def _steps(kind='neighbors', parts=1):
    return [
        {
            'id': 7 + part,
            'request': {'request': kind},
            'partition': {'first': 7, 'ordinal': part, 'parts': parts},
        }
        for part in range(parts)
    ]


class Provider:
    def __init__(self, values, events, *, failed=None, cleanup=None):
        self.values = values
        self.events = events
        self.failed = failed or {}
        self.cleanup = cleanup or {}
        self.bound = []
        self.active = set()

    def __call__(self, *args):
        pytest.fail('ordinary Arrow boxing must not be used')

    @contextmanager
    def source_rows(self, step, keys):
        number = step['id']
        self.bound.append((step, keys))
        self.events.append(('enter', number))
        if number in self.failed:
            raise self.failed[number]
        self.active.add(number)
        try:
            yield self.values[number]
        except BaseException as error:
            self.events.append(('unwind', number, error))
            raise
        finally:
            self.active.remove(number)
            self.events.append(('exit', number))
            if number in self.cleanup:
                raise self.cleanup[number]


def _scalars(values, names=('value', '__kumo_batch__')):
    return ScalarRows(values, [SimpleNamespace(name=name) for name in names])


def test_single_arrow_result_keeps_identity_helpers_and_closes_scope():
    events = []
    table = pa.table({'value': [2**53 + 1, None], '__KUMO_BATCH__': [4, 9]})
    steps = _steps()
    keys = {7: {12: object()}}
    provider = Provider({7: table}, events)
    result = native.read_source_rows(provider, steps, keys)
    assert isinstance(result, native.SourceRowsBatch)
    assert result.rows is table
    assert result.row_counts == (2,)
    assert result.storage == 'typed_columns'
    assert result.rows.column_names == ['value', '__KUMO_BATCH__']
    assert result.rows['__KUMO_BATCH__'].to_pylist() == [4, 9]
    assert provider.bound[0][0] is steps[0]
    assert provider.bound[0][1] is keys[7]
    assert provider.active == set()
    assert events == [('enter', 7), ('exit', 7)]


def test_multiple_arrow_parts_concat_once_after_all_io_without_helper_offsets(
    monkeypatch,
):
    events = []
    tables = [
        pa.table({'value': [None, 2**53 + 1], '__KUMO_BATCH__': [3, 4]}),
        pa.table({'value': [2**53 + 2], '__KUMO_BATCH__': [91]}),
    ]
    provider = Provider(dict(zip((7, 8), tables, strict=True)), events)
    concat = pa.concat_tables

    def joined(actual, *args, **kwargs):
        assert provider.active == set()
        assert len(actual) == 2
        assert all(
            left is right for left, right in zip(actual, tables, strict=True)
        )
        events.append('arrow_concat')
        return concat(actual, *args, **kwargs)

    monkeypatch.setattr(pa, 'concat_tables', joined)
    result = native.read_source_rows(provider, _steps(parts=2), {7: {}, 8: {}})
    assert isinstance(result, native.SourceRowsBatch)
    assert result.row_counts == (2, 1)
    assert result.rows['value'].to_pylist() == [None, 2**53 + 1, 2**53 + 2]
    assert result.rows['__KUMO_BATCH__'].to_pylist() == [3, 4, 91]
    assert events == [
        ('enter', 7),
        ('exit', 7),
        ('enter', 8),
        ('exit', 8),
        'arrow_concat',
    ]


@pytest.mark.parametrize('kind', ['lookup', 'neighbors', 'scan'])
def test_keyed_scalar_frame_constructs_inside_scope_then_concats_even_one_part(
    monkeypatch, kind
):
    events = []
    rows = [[2**53 + 1, None, 17, 8, 2]]
    names = [
        'value',
        'nullable',
        '__kumo_batch__',
        '__kumo_row__',
        '__kumo_matches__',
    ]
    provider = Provider({7: _scalars(rows, names)}, events)
    constructor, concat = pd.DataFrame, pd.concat
    made = []

    def frame(*args, **kwargs):
        assert provider.active == {7}
        assert args[0] is rows
        assert kwargs == {
            'columns': [
                'value',
                'nullable',
                '__KUMO_BATCH__',
                '__KUMO_ROW__',
                '__KUMO_MATCHES__',
            ],
            'dtype': object,
        }
        events.append('frame')
        value = constructor(*args, **kwargs)
        made.append(value)
        return value

    def joined(actual, *args, **kwargs):
        assert provider.active == set()
        assert actual == made
        assert kwargs == {'ignore_index': True}
        events.append('pandas_concat')
        return concat(actual, *args, **kwargs)

    monkeypatch.setattr(pd, 'DataFrame', frame)
    monkeypatch.setattr(pd, 'concat', joined)
    result = native.read_source_rows(provider, _steps(kind), {7: {}})
    assert isinstance(result, native.SourceRowsBatch)
    assert result.storage == 'cell_rows'
    assert result.row_counts == (1,)
    assert result.rows.iloc[0, 0] == 2**53 + 1
    assert result.rows.iloc[0, 1] is None
    assert result.rows.iloc[0, 2:].tolist() == [17, 8, 2]
    assert all(dtype.name == 'object' for dtype in result.rows.dtypes)
    assert events == [('enter', 7), 'frame', ('exit', 7), 'pandas_concat']


def test_unkeyed_scalar_sample_returns_the_actual_in_scope_frame_without_concat(
    monkeypatch,
):
    events = []
    provider = Provider({7: _scalars([[2**53 + 1, 0]])}, events)
    constructor = pd.DataFrame
    made = []

    def frame(*args, **kwargs):
        assert provider.active == {7}
        value = constructor(*args, **kwargs)
        made.append(value)
        return value

    def forbidden(*args, **kwargs):
        pytest.fail('ordinary scalar sampling has no concat or Arrow boxing')

    monkeypatch.setattr(pd, 'DataFrame', frame)
    monkeypatch.setattr(pd, 'concat', forbidden)
    monkeypatch.setattr(pa, 'table', forbidden)
    result = native.read_source_rows(
        provider, _steps('sample_entities'), {7: {}}
    )
    assert isinstance(result, native.SourceRowsBatch)
    assert result.rows is made[0]
    assert result.storage == 'cell_rows'
    assert events == [('enter', 7), ('exit', 7)]


def test_scalar_description_names_are_read_in_scope_with_exact_old_helper_remaps():
    events = []
    provider = Provider({}, events)

    class Description:
        @property
        def name(self):
            assert provider.active == {7}
            events.append('name')
            return '__kumo_batch__'

    provider.values[7] = ScalarRows([[31]], [Description()])
    result = native.read_source_rows(
        provider, _steps('sample_entities'), {7: {}}
    )
    assert isinstance(result, native.SourceRowsBatch)
    assert result.rows.columns.tolist() == ['__KUMO_BATCH__']
    assert events == [('enter', 7), 'name', 'name', ('exit', 7)]


def test_scalar_frames_are_built_before_the_next_io_and_concat_after_all_scopes(
    monkeypatch,
):
    events = []
    provider = Provider(
        {7: _scalars([[2**53 + 1, 10]]), 8: _scalars([[None, 22]])}, events
    )
    constructor, concat = pd.DataFrame, pd.concat

    def frame(*args, **kwargs):
        number = next(iter(provider.active))
        events.append(('frame', number))
        return constructor(*args, **kwargs)

    def joined(*args, **kwargs):
        assert provider.active == set()
        events.append('pandas_concat')
        return concat(*args, **kwargs)

    monkeypatch.setattr(pd, 'DataFrame', frame)
    monkeypatch.setattr(pd, 'concat', joined)
    result = native.read_source_rows(provider, _steps(parts=2), {7: {}, 8: {}})
    assert isinstance(result, native.SourceRowsBatch)
    assert result.row_counts == (1, 1)
    assert result.rows['value'].tolist() == [2**53 + 1, None]
    assert result.rows['__KUMO_BATCH__'].tolist() == [10, 22]
    assert events == [
        ('enter', 7),
        ('frame', 7),
        ('exit', 7),
        ('enter', 8),
        ('frame', 8),
        ('exit', 8),
        'pandas_concat',
    ]


@pytest.mark.parametrize('scalar', [False, True])
def test_later_io_failure_preserves_exact_step_and_owner_without_final_concat(
    monkeypatch, scalar
):
    events = []
    error = OSError('later I/O')
    first = _scalars([[2**53 + 1, 0]]) if scalar else pa.table({'value': [1]})
    provider = Provider({7: first}, events, failed={8: error})
    constructor = pd.DataFrame

    def frame(*args, **kwargs):
        events.append('frame')
        return constructor(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail('no final concat or scalar boxing after incomplete I/O')

    monkeypatch.setattr(pd, 'DataFrame', frame)
    monkeypatch.setattr(pd, 'concat', forbidden)
    monkeypatch.setattr(pa, 'concat_tables', forbidden)
    monkeypatch.setattr(pa, 'table', forbidden)
    steps = _steps(parts=2)
    result = native.read_source_rows(provider, steps, {7: {}, 8: {}})
    assert isinstance(result, native.SourceReadFailure)
    assert result.step is steps[1]
    assert result.error is error
    assert events == [
        ('enter', 7),
        *(['frame'] if scalar else []),
        ('exit', 7),
        ('enter', 8),
    ]


def test_native_frame_failure_unwinds_the_actual_scope_and_prevents_later_io(
    monkeypatch,
):
    events = []
    error = TypeError('actual native constructor')
    provider = Provider({7: _scalars([[1, 0]])}, events)

    def frame(*args, **kwargs):
        assert provider.active == {7}
        raise error

    monkeypatch.setattr(pd, 'DataFrame', frame)
    result = native.read_source_rows(provider, _steps(parts=2), {7: {}, 8: {}})
    assert isinstance(result, FetchRefused)
    assert result.raised is error
    assert error.__context__ is None
    assert events == [('enter', 7), ('unwind', 7, error), ('exit', 7)]


def test_io_cleanup_failure_is_distinct_from_the_native_body_failure(
    monkeypatch,
):
    events = []
    original = TypeError('actual native constructor')
    cleanup = OSError('cursor close')
    provider = Provider({7: _scalars([[1, 0]])}, events, cleanup={7: cleanup})

    def frame(*args, **kwargs):
        raise original

    monkeypatch.setattr(pd, 'DataFrame', frame)
    steps = _steps()
    result = native.read_source_rows(provider, steps, {7: {}})
    assert isinstance(result, native.SourceReadFailure)
    assert result.step is steps[0]
    assert result.error is cleanup
    assert cleanup.__context__ is original


@pytest.mark.parametrize('scalar', [False, True])
def test_actual_concat_failure_remains_native_and_retains_its_exact_owner(
    monkeypatch, scalar
):
    events = []
    original = ValueError('native concat')
    parts = 1 if scalar else 2
    values = (
        {7: _scalars([[1, 0]])}
        if scalar
        else {
            7: pa.table({'value': [1]}),
            8: pa.table({'value': [2]}),
        }
    )
    provider = Provider(values, events)

    def failed(*args, **kwargs):
        assert provider.active == set()
        raise original

    monkeypatch.setattr(
        pd if scalar else pa, 'concat' if scalar else 'concat_tables', failed
    )
    result = native.read_source_rows(
        provider, _steps(parts=parts), {7: {}, 8: {}}
    )
    assert isinstance(result, FetchRefused)
    assert result.raised is original
    assert provider.active == set()


@pytest.mark.parametrize('error_type', [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize('position', ['io', 'native'])
def test_process_control_is_never_converted_to_a_failure_carrier(
    monkeypatch, error_type, position
):
    events = []
    original = error_type()
    provider = Provider(
        {7: _scalars([[1, 0]])},
        events,
        failed={7: original} if position == 'io' else None,
    )

    def frame(*args, **kwargs):
        raise original

    if position == 'native':
        monkeypatch.setattr(pd, 'DataFrame', frame)
    with pytest.raises(error_type) as caught:
        native.read_source_rows(provider, _steps(), {7: {}})
    assert caught.value is original


def test_missing_scoped_port_never_falls_back_to_the_ordinary_provider():
    def forbidden(*args, **kwargs):
        pytest.fail('ordinary provider fallback would lose native scalar data')

    with pytest.raises(AttributeError):
        native.read_source_rows(forbidden, _steps(), {7: {}})


def test_positional_descriptions_keep_business_labels_and_only_remap_exact_helpers():
    source = ScalarRows(
        [[7, 8, 9]],
        [('Value',), ('__kumo_matches__',), ('__KUMO_ROW__',)],
    )
    provider = Provider({7: source}, [])
    result = native.read_source_rows(
        provider, _steps('sample_entities'), {7: {}}
    )
    assert isinstance(result, native.SourceRowsBatch)
    assert result.rows.columns.tolist() == [
        'Value',
        '__KUMO_MATCHES__',
        '__KUMO_ROW__',
    ]
    assert result.rows.iloc[0].tolist() == [7, 8, 9]


def test_description_getter_failure_remains_native_and_stops_later_io():
    events = []
    original = ValueError('actual native description')

    class Description:
        @property
        def name(self):
            raise original

    provider = Provider({7: ScalarRows([[1]], [Description()])}, events)
    result = native.read_source_rows(provider, _steps(parts=2), {7: {}, 8: {}})
    assert isinstance(result, FetchRefused)
    assert result.raised is original
    assert events == [('enter', 7), ('unwind', 7, original), ('exit', 7)]


def test_empty_arrow_result_retains_exact_schema_metadata_and_zero_part_count():
    table = pa.table(
        {'value': pa.array([], type=pa.int64())}
    ).replace_schema_metadata({b'raw': b'kept'})
    provider = Provider({7: table}, [])
    result = native.read_source_rows(provider, _steps(), {7: {}})
    assert isinstance(result, native.SourceRowsBatch)
    assert result.rows is table
    assert result.row_counts == (0,)


def test_scope_exit_io_failure_is_retained_without_reading_error_diagnostics():
    class UnreadError(OSError):
        @property
        def code(self):
            pytest.fail('diagnostic code must not be observed in raw assembly')

        def __str__(self):
            pytest.fail('diagnostic text must not be observed in raw assembly')

    events = []
    original = UnreadError()
    provider = Provider(
        {7: pa.table({'value': [1]})}, events, cleanup={7: original}
    )
    steps = _steps()
    result = native.read_source_rows(provider, steps, {7: {}})
    assert isinstance(result, native.SourceReadFailure)
    assert result.step is steps[0]
    assert result.error is original
    assert events == [('enter', 7), ('exit', 7)]
