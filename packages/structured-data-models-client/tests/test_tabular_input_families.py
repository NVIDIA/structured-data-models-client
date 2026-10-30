# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct native input families must preserve the complete model call."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
import warnings

import numpy as np
import pandas as pd
import pytest

from sdm_client import KumoTabular, Request, SDMWarning, StructuredDataError
from test_tabular_direct_core import DIRECT, Endpoint, client


def feature(family: str) -> pd.Series:
    if family == 'float32':
        return pd.Series([1.5, np.nan, -0.0, 2.5], dtype='float32')
    if family == 'float64':
        return pd.Series([1.5, np.nan, -0.0, 2.5], dtype='float64')
    if family == 'nullable_integer':
        return pd.Series([1, None, 3, 4], dtype='Int64')
    if family == 'nullable_float':
        return pd.Series([1.5, None, -0.0, 2.5], dtype='Float64')
    if family == 'nullable_boolean':
        return pd.Series([True, None, False, True], dtype='boolean')
    if family == 'nullable_present_nan':
        return pd.Series(pd.arrays.FloatingArray(
            np.array([1.5, np.nan, -0.0, 2.5]), np.array([False] * 4),
        ))
    if family.startswith('arrow_'):
        import pyarrow as pa

        kinds = {
            'arrow_integer': (pa.int64(), [1, None, 3, 4]),
            'arrow_unsigned': (pa.uint64(), [2**63, None, 2**63 + 1, 2**63 + 2]),
            'arrow_boolean': (pa.bool_(), [True, None, False, True]),
            'arrow_float32': (pa.float32(), [1.5, None, -0.0, 2.5]),
            'arrow_present_nan': (pa.float64(), [1.5, np.nan, -0.0, 2.5]),
            'arrow_decimal_integer': (pa.decimal128(20, 0), [Decimal('1'), None, Decimal('3'), Decimal('4')]),
            'arrow_decimal_fraction': (pa.decimal128(20, 2), [Decimal('1.25'), None, Decimal('3.50'), Decimal('4.75')]),
            'arrow_decimal256': (pa.decimal256(50, 0), [Decimal(2**100), None, Decimal(2**100 + 1), Decimal(2**100 + 2)]),
        }
        dtype, values = kinds[family]
        return pd.Series(pa.array(values, type=dtype, from_pandas=False), dtype=pd.ArrowDtype(dtype))
    if family == 'string':
        return pd.Series(['alpha', None, '', 'beta'], dtype='string')
    if family == 'categorical_integer':
        return pd.Series(pd.Categorical([3, 1, None, 2], categories=[2, 3, 1]))
    if family == 'categorical_string':
        return pd.Series(pd.Categorical(['b', 'a', None, 'c']))
    if family == 'timestamp':
        return pd.Series(pd.to_datetime(['2020-01-01', None, '2020-01-03', '2020-01-04']))
    if family == 'timestamp_zone':
        return feature('timestamp').dt.tz_localize('America/New_York')
    if family == 'decimal_integer':
        return pd.Series([Decimal('1'), None, Decimal('3'), Decimal('4')], dtype=object)
    if family == 'decimal_fraction':
        return pd.Series([Decimal('1.25'), None, Decimal('3.5'), Decimal('4.75')], dtype=object)
    if family == 'wide_integer':
        return pd.Series([2**70, None, 2**70 + 1, 2**70 + 2], dtype=object)
    if family == 'object_integer':
        return pd.Series([1, None, 3, 4], dtype=object)
    if family == 'object_mixed':
        return pd.Series([1, 'two', None, True], dtype=object)
    if family == 'object_null':
        return pd.Series([None] * 4, dtype=object)
    raise AssertionError(family)


@pytest.mark.parametrize('family', [
    'float32', 'float64', 'nullable_integer', 'nullable_float',
    'nullable_boolean', 'nullable_present_nan', 'arrow_integer',
    'arrow_unsigned', 'arrow_boolean', 'arrow_float32', 'arrow_present_nan',
    'arrow_decimal_integer', 'arrow_decimal_fraction', 'arrow_decimal256',
    'string', 'categorical_integer', 'categorical_string',
    'timestamp', 'timestamp_zone', 'decimal_integer', 'decimal_fraction',
    'wide_integer', 'object_integer', 'object_mixed', 'object_null',
])
@pytest.mark.parametrize('mixed', [False, True], ids=['passive', 'mixed-native'])
def test_direct_family_preserves_wire_and_result(
    family: str, mixed: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = []
    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        context = pd.DataFrame({'feature': feature(family), 'target': [0, 1, 0, 1]})
        scoring = pd.DataFrame({'feature': feature(family).iloc[:2].reset_index(drop=True)})
        if mixed:
            # An adjacent object column selects native observation. The
            # feature must retain its semantics when encoded as a column
            # batch instead of as part of the passive frame batch.
            context['native'] = pd.Series(['a', 'b', 'c', 'd'], dtype=object)
            scoring['native'] = pd.Series(['e', 'f'], dtype=object)
        request = Request(context, scoring, target='target', task='classification')
        endpoint = Endpoint({
            'model': 'kumo-tabular', 'id': 'answer', 'metadata': {},
            'predictions': [{'row_index': 0, 'prediction': 0}, {'row_index': 1, 'prediction': 1}],
        })
        result = client(endpoint).predict(request, model=KumoTabular())
        payload: dict[str, Any] = endpoint.calls[0]
        payload['metadata'].pop('request_id')
        outcomes.append((payload, result.items, request.task, request.schema[request.entity_table].time_column))
        assert request.context.instance_table is context
        assert request.predict.instance_table is scoring
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize('names', [
    ('event_at', 'other_at'), ('created_at', 'event_at'),
    ('loaded_at', 'updated_at'), ('birth_date', 'event_at'),
])
def test_native_time_selection_preserves_nanosecond_order(
    names: tuple[str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = []
    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        frame = pd.DataFrame({
            names[0]: pd.to_datetime([
                '2020-01-01T00:00:00.000000002',
                '2020-01-02T00:00:00.000000002',
                '2020-01-03T00:00:00.000000002',
            ]),
            names[1]: pd.to_datetime([
                '2020-01-01T00:00:00.000000001',
                '2020-01-02T00:00:00.000000001',
                '2020-01-03T00:00:00.000000001',
            ]),
            'native': pd.Series(['a', 'b', 'c'], dtype=object),
            'target': [0, 1, 0],
        })
        request = Request(frame, frame.drop(columns='target').iloc[:1], target='target', task='classification')
        selected.append(request.schema[request.entity_table].time_column)
    assert selected[0] == selected[1]


@pytest.mark.parametrize('companion', ['integer', 'timestamp', 'none'])
def test_inferred_duplicate_key_advice(companion: str, monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = []
    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        context = pd.DataFrame({
            'row_id': [*range(39), 0],
            'native': pd.Series(['label'] * 40, dtype=object),
            'target': [0] * 20 + [1] * 19 + [0],
        })
        if companion == 'integer':
            context['version'] = range(40)
        elif companion == 'timestamp':
            context['event_at'] = pd.date_range('2020-01-01', periods=40)
        scoring = context.drop(columns='target').iloc[:1].copy()
        scoring['row_id'] = 99
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', SDMWarning)
            request = Request(context, scoring, target='target', task='classification')
        outcomes.append((
            request.schema[request.entity_table].duplicate_keys,
            [str(item.message) for item in caught if issubclass(item.category, SDMWarning)],
        ))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0] == 1


@pytest.mark.parametrize('values', [
    ['2020-01-01', '2020-02-01', '2020-03-01'],
    ['15/01/2020', '16/01/2020', '17/01/2020'],
    ['1800-01-01', '1800-02-01', '1800-03-01'],
    ['2200-01-01', '2200-02-01', '2200-03-01'],
    ['2020-01-01', '2020-13-01', '2020-03-01'],
])
@pytest.mark.parametrize('dtype', ['object', 'string'])
def test_native_date_advice_before_scoring(
    values: list[str], dtype: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = []
    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        context = pd.DataFrame({'event': pd.Series(values, dtype=dtype), 'target': [0, 1, 0]})
        scoring = context.drop(columns='target').iloc[:1].copy()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', SDMWarning)
            request = Request(context, scoring, target='target', task='classification')
        outcomes.append((
            request.schema[request.entity_table].time_column,
            [str(item.message) for item in caught if issubclass(item.category, SDMWarning)],
        ))
    assert outcomes[0] == outcomes[1]


def test_invalid_task_precedes_target_cell_inspection(monkeypatch: pytest.MonkeyPatch) -> None:
    import sdm_client.request as request_module

    def unexpected_inspection(series: Any) -> bool:
        raise AssertionError('invalid task must be refused before target cells are read')

    monkeypatch.setattr(request_module, '_has_container_cells', unexpected_inspection)
    outcomes = []
    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        with pytest.raises(StructuredDataError) as caught:
            Request(
                pd.DataFrame({'feature': [1, 2], 'target': [0, 1]}),
                pd.DataFrame({'feature': [3]}), target='target', task='not-a-task',
            )
        outcomes.append((caught.value.code, str(caught.value)))
    assert outcomes[0] == outcomes[1]


def test_custom_nullable_producer_never_takes_arrow_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    class NativeFloatArray(pd.arrays.FloatingArray):
        def __arrow_array__(self, type: Any = None) -> Any:
            raise AssertionError('custom Arrow snapshot is not a native scalar read')

    outcomes = []
    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        values = NativeFloatArray(np.array([1.5, 2.5, 3.5, 4.5]), np.zeros(4, dtype=bool))
        context = pd.DataFrame({'feature': pd.Series(values), 'target': [0, 1, 0, 1]})
        assert type(context['feature'].array) is NativeFloatArray
        request = Request(context, pd.DataFrame({'feature': [5.5]}), target='target', task='classification')
        endpoint = Endpoint({'model': 'kumo-tabular', 'id': 'answer', 'metadata': {}, 'predictions': [{'row_index': 0, 'prediction': 1}]})
        result = client(endpoint).predict(request, model=KumoTabular())
        payload = endpoint.calls[0]
        payload['metadata'].pop('request_id')
        outcomes.append((payload, result.items))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize('value', [2**53 + 1, Decimal('9007199254740991.5')])
def test_native_float_refusal_keeps_specific_diagnostic(value: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = []
    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        context = pd.DataFrame({'feature': pd.Series([value, value], dtype=object), 'target': [0, 1]})
        request = Request(context, pd.DataFrame({'feature': [0.5]}), target='target', task='classification')
        endpoint = Endpoint()
        with pytest.raises(StructuredDataError) as caught:
            client(endpoint).predict(request, model=KumoTabular())
        assert endpoint.calls == []
        outcomes.append((caught.value.code, str(caught.value)))
    assert outcomes[0] == outcomes[1]


def test_native_integer_range_after_earlier_column_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = []
    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        context = pd.DataFrame({'first': pd.Series(['a', 'b'], dtype=object), 'feature': [1, 2], 'target': [0, 1]})

        class MutatingText:
            def __str__(self) -> str:
                context['feature'] = pd.Series([2**70, 2], dtype=object)
                return 'a'

        context['first'] = pd.Series([MutatingText(), 'b'], dtype=object)
        request = Request(context, pd.DataFrame({'first': ['c'], 'feature': [3]}), target='target', task='classification')
        endpoint = Endpoint()
        with pytest.raises(StructuredDataError) as caught:
            client(endpoint).predict(request, model=KumoTabular())
        assert endpoint.calls == []
        outcomes.append((caught.value.code, str(caught.value)))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize('fail', [False, True])
def test_native_conversion_order_and_nested_options(
    fail: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = []
    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        events: list[str] = []
        context = pd.DataFrame({'feature': ['a', 'b', 'c', 'd'], 'later': [1, 2, 3, 4], 'target': [0, 1, 0, 1]})
        scoring = pd.DataFrame({'feature': ['e', 'f'], 'later': [5, 6]})

        class NativeText:
            def __init__(self, label: str) -> None:
                self.label = label

            def __str__(self) -> str:
                events.append(self.label)
                if self.label == 'a':
                    context['later'] = [9, 8, 7, 6]
                    scoring['later'] = [12, 13]
                if fail and self.label == 'b':
                    raise RuntimeError('native conversion failed')
                return self.label

        context['feature'] = pd.Series([NativeText(label) for label in 'abcd'], dtype=object)
        scoring['feature'] = pd.Series([NativeText(label) for label in 'ef'], dtype=object)
        request = Request(context, scoring, target='target', task='classification', inference={
            'nested': {'seed': np.int64(7), 'flags': [True, None], 'weights': (np.float32(0.5), 2**53)},
        })
        endpoint = Endpoint({
            'model': 'kumo-tabular', 'id': 'answer', 'metadata': {},
            'predictions': [{'row_index': 0, 'prediction': 0}, {'row_index': 1, 'prediction': 1}],
        })
        if fail:
            with pytest.raises(RuntimeError, match='native conversion failed'):
                client(endpoint).predict(request, model=KumoTabular())
            assert endpoint.calls == []
            outcomes.append((events, None))
        else:
            result = client(endpoint).predict(request, model=KumoTabular())
            payload = endpoint.calls[0]
            payload['metadata'].pop('request_id')
            outcomes.append((events, payload, result.items))
    assert outcomes[0] == outcomes[1]
