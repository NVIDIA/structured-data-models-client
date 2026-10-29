# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source completion keeps Core authority outside fixed native conversion."""

from __future__ import annotations

import copy
import json
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pytest
from test_t44_fetch_native_handover import _CoreError, _Handle
from test_t45_native_source_rows import Provider, _scalars

from sdm_client._rust import native_source_completion as completion
from sdm_client._rust import run as run_module
from sdm_client._rust.codec import CodecError
from sdm_client._rust.native_products import Product
from sdm_client._rust.run import FetchRefused
from sdm_client.errors import StructuredDataError

LOGICAL = {'source': ['logical']}
READ = {
    'source': ['catalog', 'schema', 'actual'],
    'found': True,
    'columns': [{'name': 'value', 'column': 'value'}],
}


def _source(columns=('value',), *, temporal_columns=()):
    return Product(
        'source',
        {
            'tables': [
                {
                    'source': LOGICAL,
                    'read': READ,
                    'columns': list(columns),
                    'temporal_columns': list(temporal_columns),
                }
            ]
        },
        {},
    )


def _steps(parts=1, *, lookup=True, conversions=()):
    value = {
        'owner': 91,
        'read': 7,
        'source': READ,
        'key': 'id',
        'range': {'start': 0, 'stop': 1},
        'context_rows': 1,
        'scored_rows': 1,
    }
    return [
        {
            'id': 7 + part,
            'purpose': 'source_rows',
            'partition': {'first': 7, 'ordinal': part, 'parts': parts},
            'request': {
                'request': 'lookup' if lookup else 'neighbors',
                'table': READ,
            },
            'plan': {
                'plan': 'sql',
                'statements': [],
                'result_conversions': list(conversions),
            },
            **({'lookup': copy.deepcopy(value)} if lookup else {}),
        }
        for part in range(parts)
    ]


class Service:
    def __init__(self, *, recipes=None, refusal=None, positions=None):
        self.recipes = recipes or {}
        self.refusal = refusal
        self.positions = positions
        self.calls = []

    def lookup(self, lookup, helpers, owner):
        self.calls.append(('lookup', lookup, helpers, owner))
        if self.refusal is not None:
            raise self.refusal
        values = (
            self.positions
            if self.positions is not None
            else helpers['__kumo_batch__']
        )
        positions = (
            values.cast(pa.uint64())
            if isinstance(values, (pa.Array, pa.ChunkedArray))
            else pa.array(values, type=pa.uint64())
        )
        return pa.table({'position': positions})

    def column(self, binding, facts):
        self.calls.append(('column', binding, facts))
        return {
            'source': 'source identity',
            'column': binding,
            'input': facts,
            'scalar': self.recipes.get(
                binding['column'], {'meaning': 'original'}
            ),
        }

    def fallback(self, meaning, failure):
        self.calls.append(('fallback', meaning, failure))
        return {**meaning, 'scalar': {'meaning': 'original'}}

    def read_failure(self, failure):
        self.calls.append(('read_failure', failure))
        raise self.refusal


def _keys(steps, owner):
    return {step['id']: {91: owner} for step in steps}


def test_completed_lookup_precedes_native_nullable_conversion_and_keeps_full_width_values():
    events = []
    owner = object()
    raw = _scalars(
        [[2**53 + 1, 1, 0], [None, 1, 1]],
        ['value', '__kumo_matches__', '__kumo_batch__'],
    )
    provider = Provider({7: raw}, events)
    service = Service(
        recipes={
            'value': {'meaning': 'nullable', 'physical': {'type': 'int64'}}
        },
        positions=[0, 1],
    )
    adapter = completion.SourceCompletion(provider, _source(), service)
    steps = _steps()
    answer = adapter(steps, _keys(steps, owner))
    assert isinstance(answer, dict)
    assert answer[7].column_names == [
        'value',
        '__kumo_matches__',
        '__kumo_batch__',
    ]
    assert answer[7]['value'].to_pylist() == [2**53 + 1, None]
    assert answer[7]['__kumo_batch__'].to_pylist() == [0, 1]
    assert answer[7]['__kumo_matches__'].to_pylist() == [1, 1]
    assert service.calls[0][0] == 'lookup'
    assert service.calls[0][3] is owner
    assert service.calls[0][2].to_pydict() == {
        '__kumo_batch__': [0, 1],
        '__kumo_matches__': [1, 1],
    }
    assert service.calls[1] == (
        'column',
        {'source': LOGICAL, 'column': 'value'},
        {'storage': 'cell_rows'},
    )
    assert adapter.native_error is None
    retained = adapter.tables[7]
    assert retained.source == LOGICAL
    assert retained.frame.columns.tolist() == ['value']
    assert retained.frame['value'].dtype == pd.Int64Dtype()
    assert retained.frame['value'].iloc[0] == 2**53 + 1
    assert provider.active == set()


def test_core_lookup_refusal_cancels_pending_run_before_reply_or_native_conversion():
    events = []
    owner = object()
    refusal = StructuredDataError(
        'selected duplicate refusal', code='INVALID_REQUEST'
    )
    service = Service(refusal=refusal)
    provider = Provider(
        {
            7: _scalars(
                [[object(), 2, 1]],
                ['value', '__kumo_matches__', '__kumo_batch__'],
            )
        },
        events,
    )
    adapter = completion.SourceCompletion(provider, _source(), service)
    requests = _steps()

    class Handle(_Handle):
        def next(self):
            return json.dumps(
                {'message': 'step', 'step': 'fetch', 'requests': requests}
            ).encode(), {}

        def take_attachment(self, attachment):
            assert attachment == 91
            return owner

    handle = Handle(events)
    with pytest.raises(StructuredDataError) as caught:
        run_module._steps(
            SimpleNamespace(CoreError=_CoreError),
            handle,
            provider,
            None,
            frozenset(),
            None,
            source_provider=adapter,
        )
    assert caught.value is refusal
    assert adapter.native_error is None
    assert [call[0] for call in service.calls] == ['lookup']
    assert handle.replies == []
    assert events[-1] == 'cancel'
    assert events.count('cancel') == 1
    assert adapter.tables == {}


def test_real_io_failure_uses_actual_service_diagnostic_without_native_attribution():
    events = []
    original = OSError('real I/O')
    selected = StructuredDataError(
        'core-selected diagnostic', code='INVALID_REQUEST'
    )
    provider = Provider({}, events, failed={7: original})
    service = Service(refusal=selected)
    adapter = completion.SourceCompletion(provider, _source(), service)
    steps = _steps()
    answer = adapter(steps, _keys(steps, object()))
    assert isinstance(answer, FetchRefused)
    assert answer.raised is selected
    assert adapter.native_error is None
    assert len(service.calls) == 1
    assert service.calls[0][0] == 'read_failure'
    assert service.calls[0][1].step is steps[0]
    assert service.calls[0][1].error is original
    assert adapter.tables == {}


def test_nullable_casts_follow_original_declaration_order_and_actual_changed_strings():
    events = []

    class Changed:
        def __str__(self):
            events.append('actual stringify')
            return 'changed value'

    provider = Provider(
        {
            7: _scalars(
                [[2**53 + 1, Changed(), 0]],
                ['number', 'text', '__kumo_batch__'],
            )
        },
        events,
    )
    service = Service(
        recipes={
            'number': {'meaning': 'nullable', 'physical': {'type': 'int64'}},
            'text': {'meaning': 'nullable', 'physical': {'type': 'utf8'}},
        }
    )
    adapter = completion.SourceCompletion(
        provider, _source(('text', 'absent', 'number')), service
    )
    steps = _steps(lookup=False)
    answer = adapter(steps, _keys(steps, object()))
    assert isinstance(answer, dict)
    assert answer[7].column_names == ['number', 'text', '__kumo_batch__']
    assert answer[7]['number'].to_pylist() == [2**53 + 1]
    assert answer[7]['text'].to_pylist() == ['changed value']
    columns = [call for call in service.calls if call[0] == 'column']
    assert [call[1]['column'] for call in columns] == ['text', 'number']
    assert all(call[1]['source'] == LOGICAL for call in columns)
    assert events[-1] == 'actual stringify'


def test_arrow_decimal_recipe_changes_actual_retained_values_and_native_declaration():
    raw = pa.table(
        {
            'value': pa.array([Decimal('3'), None], type=pa.decimal128(20, 0)),
            '__KUMO_BATCH__': [0, 1],
        }
    )
    service = Service(
        recipes={'value': {'meaning': 'decimal', 'physical': {'type': 'utf8'}}}
    )
    adapter = completion.SourceCompletion(
        Provider({7: raw}, []), _source(), service
    )
    steps = _steps(lookup=False)
    answer = adapter(steps, _keys(steps, object()))
    assert isinstance(answer, dict)
    assert answer[7]['value'].to_pylist() == ['3', None]
    assert adapter.tables[7].frame['value'].dtype == pd.ArrowDtype(pa.string())
    assert service.calls == [
        (
            'column',
            {'source': LOGICAL, 'column': 'value'},
            {
                'storage': 'typed_columns',
                'declaration': {
                    'declaration': 'decimal',
                    'native': {
                        'type': 'decimal128',
                        'precision': 20,
                        'scale': 0,
                    },
                },
            },
        )
    ]


@pytest.mark.parametrize('error_type', [TypeError, ValueError, ArithmeticError])
def test_actual_decimal_failure_is_assessed_once_and_core_fallback_retains_original(
    monkeypatch, error_type
):
    original = error_type('actual Decimal cast')
    raw = pa.table(
        {
            'value': pa.array([Decimal('3')], type=pa.decimal128(20, 0)),
            '__kumo_batch__': [0],
        }
    )
    scalar_column = completion.scalar_column
    calls = []

    def cast(values, meaning):
        calls.append(meaning['meaning'])
        if meaning['meaning'] == 'decimal':
            raise original
        return scalar_column(values, meaning)

    monkeypatch.setattr(completion, 'scalar_column', cast)
    service = Service(
        recipes={'value': {'meaning': 'decimal', 'physical': {'type': 'int64'}}}
    )
    adapter = completion.SourceCompletion(
        Provider({7: raw}, []), _source(), service
    )
    steps = _steps(lookup=False)
    answer = adapter(steps, _keys(steps, object()))
    assert isinstance(answer, dict)
    fallback = service.calls[1]
    assert fallback[0] == 'fallback'
    assert fallback[2].error is original
    assert calls == ['decimal', 'original']
    assert adapter.native_error is None
    assert adapter.tables[7].frame['value'].dtype == pd.ArrowDtype(
        pa.decimal128(20, 0)
    )


def test_nullable_failure_is_native_and_never_uses_decimal_fallback(
    monkeypatch,
):
    original = TypeError('actual nullable failure')
    provider = Provider({7: _scalars([[1, 0]])}, [])
    service = Service(
        recipes={
            'value': {'meaning': 'nullable', 'physical': {'type': 'int64'}}
        }
    )

    def failed(*args):
        raise original

    monkeypatch.setattr(completion, 'scalar_column', failed)
    adapter = completion.SourceCompletion(provider, _source(), service)
    steps = _steps(lookup=False)
    answer = adapter(steps, _keys(steps, object()))
    assert isinstance(answer, FetchRefused)
    assert answer.raised is original
    assert adapter.native_error is original
    assert [call[0] for call in service.calls] == ['column']
    assert adapter.tables == {}


def test_timestamp_transport_keeps_raw_ordinals_then_removes_only_helpers():
    raw = pa.table(
        {
            '__KUMO_BATCH__': [0, 1],
            'value': [2**53 + 1, None],
            '__KUMO_ROW__': [9, 4],
        }
    )
    service = Service()
    adapter = completion.SourceCompletion(
        Provider({7: raw}, []), _source(), service
    )
    steps = _steps(
        lookup=False,
        conversions=[
            {
                'conversion': 'timestamp_micros',
                'column': 1,
                'name': 'value',
                'source_kind': 'wall_time',
            }
        ],
    )
    answer = adapter(steps, _keys(steps, object()))
    assert isinstance(answer, dict)
    assert answer[7]['value'].type == pa.timestamp('us')
    assert answer[7]['value'].cast(pa.int64()).to_pylist() == [2**53 + 1, None]
    assert adapter.tables[7].frame.columns.tolist() == ['value']
    assert answer[7].column_names == ['__kumo_batch__', 'value', '__kumo_row__']
    assert answer[7]['__kumo_row__'].to_pylist() == [9, 4]


def test_malformed_timestamp_descriptor_is_a_boundary_failure_without_native_owner():
    raw = pa.table({'value': [1], '__kumo_batch__': [0]})
    adapter = completion.SourceCompletion(
        Provider({7: raw}, []), _source(), Service()
    )
    steps = _steps(
        lookup=False,
        conversions=[
            {
                'conversion': 'timestamp_micros',
                'column': 1,
                'name': 'value',
                'source_kind': 'wall_time',
            }
        ],
    )
    answer = adapter(steps, _keys(steps, object()))
    assert isinstance(answer, FetchRefused)
    assert isinstance(answer.raised, ValueError)
    assert adapter.native_error is None
    assert adapter.tables == {}


def test_complete_replies_split_by_actual_part_rows_and_keep_one_reached_native_frame():
    provider = Provider(
        {
            7: pa.table({'value': [7, 8], '__kumo_batch__': [0, 1]}),
            8: pa.table({'value': [9], '__kumo_batch__': [2]}),
        },
        [],
    )
    adapter = completion.SourceCompletion(provider, _source(), Service())
    steps = _steps(parts=2, lookup=False)
    answer = adapter(steps, _keys(steps, object()))
    assert isinstance(answer, dict)
    assert answer[7].to_pydict() == {'value': [7, 8], '__kumo_batch__': [0, 1]}
    assert answer[8].to_pydict() == {'value': [9], '__kumo_batch__': [2]}
    assert adapter.tables[7].frame['value'].tolist() == [7, 8, 9]


def test_exact_read_binding_rejects_same_name_other_relation_before_io():
    events = []
    adapter = completion.SourceCompletion(
        Provider({}, events), _source(), Service()
    )
    steps = _steps()
    steps[0]['request']['table'] = {'source': ['actual']}
    answer = adapter(steps, _keys(steps, object()))
    assert isinstance(answer, FetchRefused)
    assert isinstance(answer.raised, CodecError)
    assert adapter.native_error is None
    assert events == []


def test_first_reply_ids_are_local_to_their_actual_completion_instance():
    adapters = [
        completion.SourceCompletion(
            Provider(
                {7: pa.table({'value': [value], '__kumo_batch__': [0]})}, []
            ),
            _source(),
            Service(),
        )
        for value in (1, 2)
    ]
    steps = _steps(lookup=False)
    for adapter in adapters:
        assert isinstance(adapter(steps, _keys(steps, object())), dict)
    assert adapters[0].tables[7].frame['value'].tolist() == [1]
    assert adapters[1].tables[7].frame['value'].tolist() == [2]
    repeated = adapters[0](steps, _keys(steps, object()))
    assert isinstance(repeated, FetchRefused)
    assert isinstance(repeated.raised, CodecError)
    assert adapters[0].tables[7].frame['value'].tolist() == [1]


def test_scalar_helper_pops_keep_matches_before_lookup_and_batch_row_after(
    monkeypatch,
):
    from sdm_client._rust.native_source_rows import SourceRowsBatch

    events = []

    class Frame(pd.DataFrame):
        def pop(self, item):
            events.append(('pop', item))
            return super().pop(item)

    class OrderedService(Service):
        def lookup(self, *args):
            events.append('lookup')
            return super().lookup(*args)

    frame = Frame(
        {
            'value': [3],
            '__KUMO_BATCH__': [0],
            '__KUMO_MATCHES__': [1],
            '__KUMO_ROW__': [9],
        },
        dtype=object,
    )
    monkeypatch.setattr(
        completion,
        'read_source_rows',
        lambda *args: SourceRowsBatch(frame, (1,)),
    )
    adapter = completion.SourceCompletion(
        Provider({}, []), _source(), OrderedService()
    )
    steps = _steps()
    assert isinstance(adapter(steps, _keys(steps, object())), dict)
    assert events == [
        ('pop', '__KUMO_MATCHES__'),
        'lookup',
        ('pop', '__KUMO_BATCH__'),
        ('pop', '__KUMO_ROW__'),
    ]


def test_core_column_refusal_remains_distinct_from_native_construction():
    selected = StructuredDataError(
        'core column refusal', code='INVALID_REQUEST'
    )

    class FailedService(Service):
        def column(self, *args):
            raise selected

    adapter = completion.SourceCompletion(
        Provider({7: _scalars([[1, 0]])}, []), _source(), FailedService()
    )
    steps = _steps(lookup=False)
    answer = adapter(steps, _keys(steps, object()))
    assert isinstance(answer, FetchRefused)
    assert answer.raised is selected
    assert adapter.native_error is None
    assert adapter.tables == {}


def test_reached_arrow_transport_cast_error_retains_exact_native_owner(
    monkeypatch,
):
    selected = pa.ArrowInvalid('actual Arrow cast')

    def failed(*args):
        raise selected

    monkeypatch.setattr(completion, 'result_columns', failed)
    adapter = completion.SourceCompletion(
        Provider({7: pa.table({'value': [1], '__kumo_batch__': [0]})}, []),
        _source(),
        Service(),
    )
    steps = _steps(lookup=False)
    answer = adapter(steps, _keys(steps, object()))
    assert isinstance(answer, FetchRefused)
    assert answer.raised is selected
    assert adapter.native_error is selected
    assert adapter.tables == {}


def test_replies_preserve_all_actual_helpers_in_original_slots_after_business_conversion():
    raw = pa.table(
        {
            '__KUMO_BATCH__': [1, 0],
            'value': [3, 4],
            '__KUMO_MATCHES__': [1, 1],
            '__KUMO_ROW__': [8, 5],
            '__KUMO_ID__': [12, 9],
        }
    )
    adapter = completion.SourceCompletion(
        Provider({7: raw}, []), _source(), Service()
    )
    steps = _steps()
    reply = adapter(steps, _keys(steps, object()))
    assert isinstance(reply, dict)
    assert reply[7].column_names == [
        '__kumo_batch__',
        'value',
        '__kumo_matches__',
        '__kumo_row__',
        '__kumo_id__',
    ]
    assert reply[7].to_pydict() == {
        '__kumo_batch__': [1, 0],
        'value': [3, 4],
        '__kumo_matches__': [1, 1],
        '__kumo_row__': [8, 5],
        '__kumo_id__': [12, 9],
    }
    assert adapter.tables[7].frame.columns.tolist() == ['value']


def test_public_request_reuses_actual_early_lookup_and_late_shared_helper_reader(
    monkeypatch,
):
    """Controlled native group completion reaches both real Rust lookup sites.

    Passing through Rust's already computed group values avoids coupling this
    helper-retention control to extension-dtype restoration behavior.
    This control makes no frozen-client native-restoration parity claim.
    """
    import warnings

    import numpy as np
    from sdm_connectors import DuckDBSource
    from test_t45_sql_link import QUERY, connection, route

    from sdm_client import PredictiveQuery, Schema
    from sdm_client._rust import native_sql
    from sdm_client._rust.native_target import NativeTargetColumn
    from sdm_client.materializers.relational import RelationalMaterializer

    route(monkeypatch, True)
    stages, early = [], []
    prepare = native_sql._Preparation.prepare
    lookup = native_sql._Preparation.lookup

    def prepared(self, value, *args, **kwargs):
        stages.append(value['stage'])
        result = prepare(self, value, *args, **kwargs)
        stages.append(value['stage'] + ':done')
        return result

    def counted(self, value, helpers, owner):
        actual = lookup(self, value, helpers, owner)
        early.append((helpers, actual))
        return actual

    def completed_groups(self, groups):
        actual = np.empty(len(groups), dtype=object)
        for ordinal, group in enumerate(groups):
            actual[ordinal] = group
        return actual

    def forbidden(*args, **kwargs):
        pytest.fail('the public core request entered the old engine')

    monkeypatch.setattr(native_sql._Preparation, 'prepare', prepared)
    monkeypatch.setattr(native_sql._Preparation, 'lookup', counted)
    monkeypatch.setattr(NativeTargetColumn, 'group_result', completed_groups)
    monkeypatch.setattr(RelationalMaterializer, '_get_model', forbidden)
    with connection() as db, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(db))
        actual = PredictiveQuery(QUERY, schema).materialize(
            anchor_time=pd.Timestamp('2023-03-01'), indices=[0, 1]
        )
        assert actual.payloads
        assert actual.predict_rows == 2
    assert 'lookup_rows' in stages
    assert stages[-1] == 'request:done'
    assert early
    assert any(
        '__kumo_matches__' in helpers.column_names for helpers, _ in early
    )
    assert all(answer.column_names == ['position'] for _, answer in early)


@pytest.mark.parametrize('width', [128, 256])
@pytest.mark.parametrize('fallback', [False, True])
def test_exact_decimal_reply_and_actual_native_frame_are_independent(
    monkeypatch, width, fallback
):
    dtype = (pa.decimal128 if width == 128 else pa.decimal256)(38, 3)
    values = [
        Decimal('9007199254740992.001'),
        None,
        Decimal('9007199254740992.002'),
    ]
    raw = pa.table(
        {'value': pa.array(values, type=dtype), '__kumo_batch__': [0, 1, 2]}
    )
    events = []
    actual = completion.scalar_column

    def observed(series, meaning):
        events.append(meaning['meaning'])
        if fallback and meaning['meaning'] == 'exact_decimal':
            raise ValueError('actual reached cast failure')
        return actual(series, meaning)

    monkeypatch.setattr(completion, 'scalar_column', observed)
    service = Service(
        recipes={
            'value': {
                'meaning': 'exact_decimal',
                'physical': {'type': 'float64'},
            }
        }
    )
    adapter = completion.SourceCompletion(
        Provider({7: raw}, []), _source(), service
    )
    steps = _steps(lookup=False)
    answer = adapter(steps, _keys(steps, object()))
    assert isinstance(answer, dict)
    assert answer[7]['value'].type == dtype
    assert answer[7]['value'].to_pylist() == values
    native = adapter.tables[7].frame['value']
    assert native.dtype == pd.ArrowDtype(dtype if fallback else pa.float64())
    if not fallback:
        assert native.iloc[0] == native.iloc[2]
    assert events == (
        ['exact_decimal', 'original'] if fallback else ['exact_decimal']
    )
    assert [call[0] for call in service.calls] == (
        ['column', 'fallback'] if fallback else ['column']
    )
    assert adapter.native_error is None


def test_typed_timestamp_is_assigned_in_physical_order_before_time_capture(
    monkeypatch,
):
    raw = pa.table(
        {
            'before': pa.array([Decimal('1')], type=pa.decimal128(10, 0)),
            'ts': pa.array([123], type=pa.timestamp('us', tz='UTC')),
            'after': pa.array([Decimal('2')], type=pa.decimal128(10, 0)),
        }
    )
    events = []
    provider = Provider({7: raw}, events)
    service = Service(
        recipes={
            'before': {
                'meaning': 'exact_decimal',
                'physical': {'type': 'float64'},
            },
            'ts': {
                'meaning': 'nullable',
                'physical': {'type': 'timestamp', 'unit': 'ns', 'zone': None},
            },
            'after': {
                'meaning': 'exact_decimal',
                'physical': {'type': 'float64'},
            },
        }
    )
    actual = completion.scalar_column
    converted = []

    def observed(values, meaning):
        assert provider.active == set()
        converted.append(values.name)
        return actual(values, meaning)

    timestamp = completion.SourceCompletion._timestamp

    def observed_timestamp(self, values, meaning):
        assert provider.active == set()
        converted.append(values.name)
        return timestamp(self, values, meaning)

    monkeypatch.setattr(completion, 'scalar_column', observed)
    monkeypatch.setattr(
        completion.SourceCompletion, '_timestamp', observed_timestamp
    )
    adapter = completion.SourceCompletion(
        provider, _source(raw.column_names, temporal_columns=['ts']), service
    )
    steps = _steps(lookup=False)
    result = adapter(steps, {})
    assert isinstance(result, dict)
    assert converted == ['before', 'ts', 'after']
    assert adapter.tables[7].frame['ts'].dtype == 'datetime64[ns]'
    assert result[7]['ts'].type == pa.timestamp('ns')
    assert result[7]['ts'].to_pylist() == [pd.Timestamp(123000)]
    calls = [call for call in service.calls if call[0] == 'column']
    assert calls[1][2] == {
        'storage': 'typed_columns',
        'declaration': {
            'declaration': 'physical',
            'physical': {'type': 'timestamp', 'unit': 'us', 'zone': 'UTC'},
        },
    }
    assert result[7]['before'].type == raw['before'].type
    assert result[7]['after'].type == raw['after'].type


@pytest.mark.parametrize('kind', ['datetime', 'date', 'null', 'empty', 'text'])
def test_boxed_source_content_is_observed_after_scalar_construction(kind):
    from datetime import date, datetime, timezone

    values = {
        'datetime': [datetime(2023, 1, 2, tzinfo=timezone.utc), None],
        'date': [date(2023, 1, 2), None],
        'null': [None, None],
        'empty': [],
        'text': ['not a native datetime', None],
    }[kind]
    events = []
    provider = Provider(
        {7: _scalars([[value] for value in values], ['value'])}, events
    )
    service = Service(
        recipes={
            'value': {
                'meaning': 'nullable',
                'physical': {
                    'type': 'timestamp',
                    'unit': 'ns',
                    'zone': None,
                },
            }
        }
        if kind != 'text'
        else {}
    )
    # The scalar constructor is deliberately Original; only the later
    # content completion carries the time meaning chosen by Core.
    original = service.column

    def column(binding, facts):
        result = original(binding, facts)
        if facts['storage'] == 'cell_rows':
            result['scalar'] = {'meaning': 'original'}
        return result

    service.column = column
    adapter = completion.SourceCompletion(
        provider, _source(temporal_columns=['value']), service
    )
    result = adapter(_steps(lookup=False), {})
    assert isinstance(result, dict)
    facts = [call[2] for call in service.calls if call[0] == 'column']
    assert facts == [
        {'storage': 'cell_rows'},
        {
            'storage': 'column_values',
            'inferred': 'temporal'
            if kind in ('datetime', 'date')
            else 'text'
            if kind == 'text'
            else 'other',
            'inferred_empty': kind in ('null', 'empty'),
        },
    ]
    if kind == 'text':
        assert adapter.tables[7].frame['value'].tolist() == values
        assert adapter.tables[7].frame['value'].dtype == object
    else:
        assert adapter.tables[7].frame['value'].dtype == 'datetime64[ns]'
        assert result[7]['value'].null_count == (
            len(values) if kind in ('null', 'empty') else 1
        )


@pytest.mark.parametrize('handler_error', [False, True])
def test_actual_boxed_time_coercion_notice_precedes_finishing_and_keeps_native_failure(
    monkeypatch, handler_error
):
    import warnings
    from datetime import datetime

    failure = ValueError('actual native time notice handler failed')
    service = Service(
        recipes={
            'value': {
                'meaning': 'nullable',
                'physical': {
                    'type': 'timestamp',
                    'unit': 'ns',
                    'zone': None,
                },
            }
        }
    )
    original = service.column

    def column(binding, facts):
        result = original(binding, facts)
        if facts['storage'] == 'cell_rows':
            result['scalar'] = {'meaning': 'original'}
        elif facts['storage'] == 'time_conversion':
            assert facts == {
                'storage': 'time_conversion',
                'rows': 2,
                'unread': 1,
                'examples': [
                    ord(point)
                    for point in '[datetime.datetime(3000, 1, 1, 0, 0)]'
                ],
            }
            result['scalar'] = {'meaning': 'original'}
            result['notice'] = {
                'code': 'TIME_VALUES_UNREAD',
                'sentence': 'native_time_values_unread',
                'message': 'native time conversion',
                'fields': {
                    'column': {'t': 'utf8', 'v': 'value'},
                    'unread': {'t': 'u64', 'v': '1'},
                    'rows': {'t': 'u64', 'v': '2'},
                    'examples_points': [
                        {'t': 'u32', 'v': str(point)}
                        for point in facts['examples']
                    ],
                },
            }
        return result

    service.column = column
    calls = []
    present = completion.errors.warn_held

    def warn_held(notices):
        calls.append(notices)
        if handler_error:
            raise failure
        present(notices)

    monkeypatch.setattr(completion.errors, 'warn_held', warn_held)
    provider = Provider(
        {7: _scalars([[datetime(3000, 1, 1)], [None]], ['value'])}, []
    )
    adapter = completion.SourceCompletion(
        provider, _source(temporal_columns=['value']), service
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = adapter(_steps(lookup=False), {})
    assert len(calls) == 1
    if handler_error:
        assert isinstance(result, FetchRefused)
        assert result.raised is failure
        assert adapter.native_error is failure
        assert adapter.tables == {}
    else:
        assert isinstance(result, dict)
        assert result[7]['value'].null_count == 2
        assert [str(item.message) for item in caught] == [
            "[TIME_VALUES_UNREAD] Could not parse 1 of 2 value(s) of time column 'value' (for example [datetime.datetime(3000, 1, 1, 0, 0)]); they were dropped as 'NaT'"
        ]
        assert adapter.native_error is None


@pytest.mark.parametrize('native_failure', [False, True])
def test_public_scalar_row_source_uses_actual_core_selected_boxed_time_recipe(
    monkeypatch, native_failure
):
    from contextlib import contextmanager

    from test_t45_native_sql_labels import _observe

    from sdm_client._rust import native_sql
    from sdm_client._rust.providers.duckdb import DuckDBProvider

    source_rows = DuckDBProvider.source_rows
    column = native_sql._Preparation.column
    to_datetime = pd.to_datetime
    selected = []
    converted = []
    failure = ValueError('actual boxed source datetime conversion failed')

    @contextmanager
    def scalar_rows(self, step, keys=None):
        with source_rows(self, step, keys) as table:
            if step.get('purpose') != 'source_rows':
                yield table
            else:
                names = table.column_names
                yield _scalars(
                    [
                        [row[name] for name in names]
                        for row in table.to_pylist()
                    ],
                    names,
                )

    def observed(self, binding, facts):
        result = column(self, binding, facts)
        if binding['column'] == 'ts' and facts['storage'] == 'column_values':
            selected.append(
                (copy.deepcopy(facts), copy.deepcopy(result['scalar']))
            )
        return result

    def actual_conversion(values, *args, **kwargs):
        if (
            selected
            and isinstance(values, pd.Series)
            and values.name == 'ts'
            and values.dtype == object
        ):
            converted.append((values.name, len(values), kwargs.copy()))
            if native_failure:
                raise failure
        return to_datetime(values, *args, **kwargs)

    monkeypatch.setattr(DuckDBProvider, 'source_rows', scalar_rows)
    monkeypatch.setattr(native_sql._Preparation, 'column', observed)
    monkeypatch.setattr(pd, 'to_datetime', actual_conversion)
    outcome = _observe(monkeypatch, 'ordinary', True)
    assert selected
    assert all(
        facts
        == {
            'storage': 'column_values',
            'inferred': 'temporal',
            'inferred_empty': False,
        }
        and recipe
        == {
            'meaning': 'nullable',
            'physical': {'type': 'timestamp', 'unit': 'ns', 'zone': None},
        }
        for facts, recipe in selected
    )
    assert converted
    assert all(
        options == {'unit': 'ns', 'errors': 'coerce'}
        for _, _, options in converted
    )
    if native_failure:
        assert outcome['outcome'] == 'error'
        assert outcome['stage'] == 'materialize'
        assert outcome['error']['cause']['type'] == 'builtins.ValueError'
        assert outcome['error']['cause']['text'] == str(failure)
    elif outcome['outcome'] == 'error':
        assert (
            outcome['error']['cause']['type']
            == 'pyarrow.lib.ArrowNotImplementedError'
        )
        assert (
            outcome['error']['cause']['text']
            == 'Unsupported cast from list<item: int64> to int64 using function cast_int64'
        )
    else:
        assert outcome['request']['predict_rows'] == 2
        assert outcome['request']['context_rows'] > 0


def test_unrelated_object_source_has_no_new_native_time_content_reads(
    monkeypatch,
):
    observed = []
    content = completion.observed_content

    def inferred(values):
        observed.append(values)
        return content(values)

    # Observe only this source-construction fact port; Arrow's independent
    # final export may perform its own legitimate dtype inference.
    monkeypatch.setattr(completion, 'observed_content', inferred)
    provider = Provider(
        {7: _scalars([['ordinary'], ['unread']], ['value'])}, []
    )
    service = Service()
    adapter = completion.SourceCompletion(provider, _source(), service)
    actual = adapter(_steps(lookup=False), {})
    assert isinstance(actual, dict)
    assert observed == []
    assert [call[2] for call in service.calls if call[0] == 'column'] == [
        {'storage': 'cell_rows'}
    ]
    assert actual[7]['value'].to_pylist() == ['ordinary', 'unread']


@pytest.mark.parametrize('repr_failure', [False, True])
def test_public_boxed_time_native_repr_keeps_surrogates_and_actual_failure(
    monkeypatch, repr_failure
):
    from contextlib import contextmanager
    from datetime import datetime

    from test_t45_native_sql_labels import _observe

    from sdm_client._rust import errors, native_sql
    from sdm_client._rust.providers.duckdb import DuckDBProvider

    failure = RuntimeError('actual native date example repr failed')
    renders, notices = [], []

    class NativeDatetime(datetime):
        def __repr__(self):
            renders.append('repr')
            if repr_failure:
                raise failure
            return '\ud800'

    source_rows = DuckDBProvider.source_rows
    column = native_sql._Preparation.column

    @contextmanager
    def scalar_rows(self, step, keys=None):
        with source_rows(self, step, keys) as table:
            if step.get('purpose') != 'source_rows':
                yield table
            else:
                names = table.column_names
                rows = [
                    [row[name] for name in names] for row in table.to_pylist()
                ]
                if rows and 'ts' in names:
                    rows[0][names.index('ts')] = NativeDatetime(3000, 1, 1)
                yield _scalars(rows, names)

    def observed(self, binding, facts):
        value = column(self, binding, facts)
        if facts['storage'] == 'time_conversion':
            assert facts['examples'] == [91, 0xD800, 93]
            assert (
                errors.decoded(value['notice']['fields'])['examples_points']
                == facts['examples']
            )
            notices.append(value['notice'])
        return value

    monkeypatch.setattr(DuckDBProvider, 'source_rows', scalar_rows)
    monkeypatch.setattr(native_sql._Preparation, 'column', observed)
    outcome = _observe(monkeypatch, 'ordinary', True)
    assert renders
    assert all(event == 'repr' for event in renders)
    if repr_failure:
        assert notices == []
        assert outcome['outcome'] == 'error'
        assert outcome['stage'] == 'materialize'
        assert outcome['error']['cause']['type'] == 'builtins.RuntimeError'
        assert outcome['error']['cause']['text'] == str(failure)
    else:
        assert len(notices) == len(renders)
        actual = [
            notice
            for notice in outcome['warnings']
            if '[TIME_VALUES_UNREAD]' in notice['message']
        ]
        assert len(actual) == len(notices)
        assert all(
            '(for example [\ud800])' in notice['message']
            and notice['category'] == 'sdm_client._rust.errors.UserWarning'
            and notice['stage'] == 'materialize'
            for notice in actual
        )
        if outcome['outcome'] == 'error':
            assert (
                outcome['error']['cause']['type']
                == 'pyarrow.lib.ArrowNotImplementedError'
            )
            assert (
                outcome['error']['cause']['text']
                == 'Unsupported cast from list<item: int64> to int64 using function cast_int64'
            )
        else:
            assert outcome['request']['predict_rows'] == 2
            assert outcome['request']['context_rows'] > 0
