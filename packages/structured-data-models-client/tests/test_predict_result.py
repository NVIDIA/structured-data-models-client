# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prediction results, malformed responses and documented numeric policies.

Authored inputs and scripted replies cover output layout, types and options.
Generated responses cover malformed-response and transport boundaries.
Explicit native policies take precedence where stricter validation differs from historical fixture outcomes.
These tests execute the current client and do not consult migration approvals."""

from __future__ import annotations

import importlib.util
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent


def _recorder() -> Any:
    name = 'predict_result_record'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, HERE / 'predict_result_record.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = _recorder()
CASES = [c for c in R.CASES if c['covers'] != R.BOUNDARY]
CASES.append(
    {
        'name': 'size.one-past-the-limit',
        'covers': R.SIZE,
        'query': R.REGRESSION,
        'setup': [
            *R.shop(),
            "CREATE OR REPLACE TABLE users AS SELECT *, repeat('a', 275690) AS note FROM users",
        ],
        'materialize': {'indices': [0, 1], 'anchor_time': R.ANCHOR},
        'predict': {},
        'script': 'order:given',
    }
)
BOUNDARIES = [c for c in R.CASES if c['covers'] == R.BOUNDARY]


def _decode(cell):
    if cell is None:
        return None
    if 'int' in cell:
        return int(cell['int'])
    if 'f64_bits' in cell:
        return struct.unpack('>d', bytes.fromhex(cell['f64_bits']))[0]
    if 'f64' in cell:
        return float(cell['f64'])
    if 'bool' in cell:
        return cell['bool']
    if 'str' in cell:
        return cell['str']
    raise AssertionError(cell)


REFUSALS = {
    'generated-31.binary-count': 'UNSUPPORTED_REQUEST',
    'generated-32.binary-count': 'UNSUPPORTED_REQUEST',
    'context.empty-static-all-scored': 'UNSUPPORTED_REQUEST',
    'context.constant-multiclass': 'UNSUPPORTED_REQUEST',
    'context.constant-binary': 'UNSUPPORTED_REQUEST',
    'context.eleven-classes': 'UNSUPPORTED_REQUEST',
    'context.no-such-entity': 'INVALID_REQUEST',
    'prepare.regression.model-tabular': 'UNSUPPORTED_REQUEST',
    'prepare.regression.model-unknown': 'UNKNOWN_MODEL',
    'prepare.regression.outputs-empty': 'INVALID_REQUEST',
    'prepare.regression.outputs-string': 'INVALID_REQUEST',
    'prepare.regression.outputs-not-names': 'INVALID_REQUEST',
    'prepare.regression.outputs-unknown': 'UNSUPPORTED_REQUEST',
    'prepare.not-a-request': 'INVALID_REQUEST',
    'carry.longdouble-num-context': 'INVALID_REQUEST',
    'carry.ndarray-target-transforms': 'UNSUPPORTED_REQUEST',
    'size.one-past-the-limit': 'INVALID_REQUEST',
}
VALID_NUMERIC_TEXT = {
    'digit-separator',
    'arabic-indic-digits',
    'fullwidth-digits',
    'surrounding-spaces',
    'tab-and-newline',
    'plain',
    'exponent',
}


@pytest.mark.parametrize('spec', CASES, ids=[c['name'] for c in CASES])
def test_public_prediction_obeys_scripted_values_and_input_contract(
    spec, monkeypatch
):
    calls = []
    answer = R.answer

    def capture(script, body):
        calls.append(json.loads(body))
        return answer(script, body)

    monkeypatch.setattr(R, 'answer', capture)
    got = R.observe({**spec, 'items': True})
    name = spec['name']
    code = REFUSALS.get(name)
    if (
        name.startswith('numeric-text.')
        and name.split('.', 1)[1] not in VALID_NUMERIC_TEXT
    ):
        code = 'INVALID_RESPONSE'
    if name.startswith('databricks.malformed.'):
        code = 'TRANSPORT_ERROR'
    if spec.get('core_outcome'):
        code = spec['core_outcome']['code']
    if code:
        assert got['outcome'] == 'refused', (name, got)
        assert got['code'] == code, (name, got)
        assert 'endpoint' not in got and got['status_code'] is None
        if code not in {'INVALID_RESPONSE', 'TRANSPORT_ERROR'}:
            assert not calls, name
        for key in ('field', 'reason'):
            if key in spec.get('core_outcome', {}):
                assert got['fields'][key] == spec['core_outcome'][key]
        if name == 'size.one-past-the-limit':
            assert re.search(r'30 ?MB|30 ?MiB|31,?457,?280', got['message'])
        return
    assert got['outcome'] == 'result', (name, got)
    assert calls and got['model'] == 'kumo-relational'
    assert got['metadata'] == {'served_by': 'scripted', 'took_ms': 12}
    requested = spec['materialize']['indices']
    rows = [[_decode(c) for c in row] for row in got.get('rows', [])]
    if rows:
        assert [row[:2] for row in rows] == [
            [i, key] for i, key in enumerate(requested)
        ]
    assert got.get('columns', [])[:2] == ['row_index', 'id']
    items = sorted(got['items'], key=lambda item: _decode(item['row_index']))
    assert [_decode(item['id']) for item in items] == requested
    script = spec['script']
    call = calls[0]
    instance_table = call['predict']['instance_table']
    instance_at = instance_table['columns'].index('instance_id')
    classes = call['task'].get('target', {}).get('classes', [])
    import pandas as pd

    is_binary = call['task']['kind'] == 'binary_classification'
    probability_classes = (
        [str(c.lower() == 'true') for c in classes] if is_binary else classes
    )
    has_probabilities = bool(classes) and script != 'prediction-only'
    tokens = (
        [
            None if t == 'MISSING' else json.loads(t)
            for t in script.partition(':')[2].split(',')
        ]
        if script.startswith('predictions:')
        else None
    )
    has_prediction_column = tokens is None or any(t is not None for t in tokens)
    dtype = (
        (
            'bool'
            if is_binary
            else ('str' if int(pd.__version__.split('.')[0]) >= 3 else 'object')
        )
        if classes
        else 'float64'
    )
    if tokens is not None and all(type(t) is int for t in tokens):
        dtype = 'int64'
    expected_columns = (
        ['row_index', 'id']
        + (['prediction'] if has_prediction_column else [])
        + (
            [f'probability_{c}' for c in probability_classes]
            if has_probabilities
            else []
        )
    )
    expected_dtypes = (
        ['int64', 'int64']
        + ([dtype] if has_prediction_column else [])
        + (['float64'] * len(classes) if has_probabilities else [])
    )
    assert got['columns'] == expected_columns
    assert got['dtypes'] == expected_dtypes
    assert len(rows) == len(requested)
    expected_outputs = spec['predict'].get(
        'outputs',
        ['prediction', 'probabilities'] if classes else ['prediction'],
    )
    assert got['outputs'] == expected_outputs
    assert all(c['output']['fields'] == expected_outputs for c in calls)

    for index, item in enumerate(items):
        value = _decode(item['prediction'])
        if script.startswith('predictions:'):
            token = script.partition(':')[2].split(',')[index]
            expected = None if token == 'MISSING' else json.loads(token)
            if isinstance(expected, float) and math.isnan(expected):
                assert isinstance(value, float) and math.isnan(value)
            else:
                assert value == expected and type(value) is type(expected)
        elif script == 'mutate:no-prediction' and index == 0:
            assert value is None
        elif call['task']['kind'] == 'regression':
            expected = (
                0.5
                + 1.25 * index
                + int(instance_table['rows'][index][instance_at]) / 1024
            )
            assert type(value) is float and value == expected
        else:
            expected = classes[(index + 1) % len(classes)]
            if call['task']['kind'] == 'binary_classification':
                expected = expected.lower() == 'true'
            assert type(value) is type(expected) and value == expected
        assert item['prediction_type'] == type(value).__name__
        if 'prediction' in got['columns']:
            at = got['columns'].index('prediction')
            frame_value = rows[index][at]
            if value is None or (
                isinstance(value, float) and math.isnan(value)
            ):
                assert frame_value is None or (
                    isinstance(frame_value, float) and math.isnan(frame_value)
                )
            else:
                want = float(value) if got['dtypes'][at] == 'float64' else value
                assert frame_value == want and type(frame_value) is type(want)
        probabilities = item['probabilities']
        assert (probabilities is not None) == has_probabilities
        if probabilities is not None:
            actual = {entry[0]: _decode(entry[2]) for entry in probabilities}
            probability_classes = (
                [str(c.lower() == 'true') for c in classes]
                if call['task']['kind'] == 'binary_classification'
                else classes
            )
            expected = {
                c: (k + 2) / sum(range(2, len(classes) + 2))
                for k, c in enumerate(probability_classes)
            }
            if script.startswith('probabilities:') and index == 0:
                expected = dict.fromkeys(probability_classes, 0.5)
            assert actual == expected
    if spec['covers'] == R.INTEGERS:
        tokens = [json.loads(t) for t in script.partition(':')[2].split(',')]
        dtype = 'float64' if any(type(t) is float for t in tokens) else 'int64'
        assert got['dtypes'][2] == dtype
        assert [row[2] for row in rows] == [
            float(t) if dtype == 'float64' else t for t in tokens
        ]
    if name.startswith('absent.probabilities'):
        assert all(item['probabilities'] is None for item in items)
        assert any(
            'answered without' in n for n in got['notices']
        ) == name.endswith('named')
    if name == 'context.constant-regression':
        assert any(
            'every context row has the same' in n for n in got['notices']
        )
    if 'explanation' in spec['predict'].get('outputs', []):
        assert all('explanation' in c['output']['fields'] for c in calls)
    if name == 'carry.longdouble-num-estimators':
        assert call['inference']['inference_config']['num_estimators'] == 4


# Keep public validation distinct from the low-level binding reader: the
# latter can admit optional IDs that cannot identify a requested prediction.
INVALID_BOUNDARIES = {
    'deep.metadata-200',
    'deep.metadata-5000',
    'deep.item-5000',
    'finite.integer-past-int64',
    'finite.integer-past-double',
    'overflow.row-index-past-uint64',
    'overflow.row-index-1e400',
    'provenance.local-empty-body',
    'provenance.local-id-other-row',
    'provenance.local-duplicate-row',
    'provenance.local-missing-row',
    'provenance.local-row-index-out-of-range',
    'provenance.local-forecast-step',
    'evidence.no-predictions',
    'evidence.predictions-text',
    'evidence.no-response-id',
    'evidence.response-id-number',
    'evidence.no-model',
    'evidence.model-number',
    'evidence.no-metadata',
    'evidence.metadata-list',
    'evidence.no-row-index',
    'evidence.row-index-text',
    'evidence.row-index-null',
    'evidence.no-item-id',
    'evidence.item-id-null',
    'evidence.item-id-number',
    'evidence.prediction-text',
    'evidence.prediction-bool',
    'evidence.prediction-list',
    'evidence.item-not-object',
}
TRANSPORT_BOUNDARIES = {
    'provenance.local-not-json',
    'provenance.local-array',
    'provenance.local-truncated',
    'provenance.local-null',
}


def _same_float(actual, expected):
    assert type(actual) is float
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        # Includes the sign of zero and exact subnormal/rounded bits.
        assert struct.pack('>d', actual) == struct.pack('>d', expected)


@pytest.mark.parametrize(
    'spec', BOUNDARIES, ids=[c['name'] for c in BOUNDARIES]
)
def test_public_response_boundaries_have_explicit_results(spec, monkeypatch):
    calls = []
    answer = R.answer

    def capture(script, body):
        calls.append(json.loads(body))
        return answer(script, body)

    monkeypatch.setattr(R, 'answer', capture)
    # These authored accepted metadata cases are deliberately inspected whole;
    # their width is generated, never stored in an expected-output file.
    monkeypatch.setattr(R, '_shallow', lambda value: True)
    got = R.observe({**spec, 'items': True})
    name = spec['name'].removeprefix('boundary.')
    script = spec['script']
    assert got['stage'] == 'predict'
    assert len(calls) == 1
    assert calls[0]['output'] == {
        'fields': ['prediction'],
        'prediction_statistic': 'median',
    }
    if script.startswith('status:'):
        status = int(script.split(':')[1].split('/')[0])
        assert got['outcome'] == 'refused'
        assert 'NimRequestError' in got['mro']
        assert got['status_code'] == status
        assert got['code'] == ('UNSUPPORTED_FEATURE' if status == 422 else None)
        details = (
            {'diagnosis': {'stage': 'encode', 'rows': 2}}
            if status == 422
            else (
                {'error': {'message': 'model busy', 'type': 'overload'}}
                if status == 503
                else {}
            )
        )
        assert got['endpoint']['details'] == details
        messages = {
            422: 'categorical cardinality 13812 exceeds limit 10000',
            500: 'Unexpected server error',
            503: '{"error": {"message": "model busy", "type": "overload"}}',
            400: 'upstream connect error',
            404: 'NIM request failed with status 404',
        }
        assert got['endpoint']['message'] == messages[status]
        return
    code = (
        'INVALID_RESPONSE'
        if name in INVALID_BOUNDARIES
        else 'TRANSPORT_ERROR'
        if name in TRANSPORT_BOUNDARIES
        else None
    )
    if code:
        assert got['outcome'] == 'refused', name
        assert got['code'] == code, (name, got.get('code'), got.get('message'))
        assert got['status_code'] is None and 'endpoint' not in got
        assert 'StructuredDataError' in got['mro']
        if name.startswith('deep.'):
            assert got['fields'] == {'field': 'response', 'reason': 'too_deep'}
        return
    assert got['outcome'] == 'result', (
        name,
        got.get('code'),
        got.get('message'),
    )
    assert got['model'] == 'kumo-relational'
    assert got['outputs'] == ['prediction']
    assert got.get('frame_error') is None
    assert got['columns'] == ['row_index', 'id', 'prediction']
    assert got['dtypes'] == ['int64', 'int64', 'float64']
    rows = [[_decode(cell) for cell in row] for row in got['rows']]
    assert [row[:2] for row in rows] == [[0, 0], [1, 1]]
    items = sorted(got['items'], key=lambda item: _decode(item['row_index']))
    assert [_decode(item['row_index']) for item in items] == [0, 1]
    assert [_decode(item['id']) for item in items] == [0, 1]
    instance_table = calls[0]['predict']['instance_table']
    at = instance_table['columns'].index('instance_id')
    values = [
        0.5 + 1.25 * i + row[at] / 1024
        for i, row in enumerate(instance_table['rows'])
    ]
    if script.startswith('token:'):
        values[0] = float(script.partition(':')[2])
    elif name == 'evidence.prediction-null':
        values[0] = None
    elif name == 'evidence.prediction-integer':
        values[0] = 3
    for item, row, value in zip(items, rows, values, strict=True):
        actual = _decode(item['prediction'])
        if value is None:
            assert actual is None and item['prediction_type'] == 'NoneType'
            assert math.isnan(row[2])
        elif type(value) is int:
            assert type(actual) is int and actual == value
            assert item['prediction_type'] == 'int'
            _same_float(row[2], float(value))
        else:
            assert item['prediction_type'] == 'float'
            _same_float(actual, value)
            _same_float(row[2], value)
        assert item['probabilities'] is None
    metadata = got['metadata']
    if name == 'wide.metadata-20000':
        assert len(metadata) == 20000
        assert all(metadata[f'k{i:06d}'] == i for i in range(20000))
    elif name == 'deep.metadata-100':
        for _ in range(100):
            assert isinstance(metadata, dict) and list(metadata) == ['d']
            metadata = metadata['d']
        assert metadata == 'bottom'
    elif script.startswith('metadata-token:'):
        assert list(metadata) == ['score']
        _same_float(_decode(metadata['score']), float(script.partition(':')[2]))
    else:
        assert metadata == {'served_by': 'scripted', 'took_ms': 12}
