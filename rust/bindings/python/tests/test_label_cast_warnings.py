# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import warnings

import numpy as np
import pytest
from sdm_core import _native


def aggregation(operation='MIN', values=None):
    values = [1e100, 1.0] if values is None else values

    def timestamp(value):
        return {'$timestamp': value}

    return {
        'aggregation': {
            'target': 'fact.value',
            'aggr': operation,
            'start': 0,
            'end': 2,
            'unit': 'days',
        },
        'tables': {
            'fact': {
                'columns': {'value': {'dtype': 'float64', 'values': values}}
            }
        },
        'times': {
            'fact': {
                'dtype': 'datetime64[ns]',
                'values': [timestamp('2024-01-02T00:00:00')] * len(values),
            }
        },
        'batches': {'fact': [0] * len(values)},
        'anchor_time': {
            'dtype': 'datetime64[ns]',
            'values': [timestamp('2024-01-01T00:00:00')],
        },
        'filter_na': False,
    }


def execute(given):
    return json.loads(
        _native.labels_op('labels.aggregation', json.dumps(given))
    )


def host_cast(mode, warning_error=False):
    with np.errstate(over=mode), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('error' if warning_error else 'always')
        try:
            np.array([1e100, 1.0]).astype('float32')
        except (FloatingPointError, RuntimeWarning) as error:
            return {'code': type(error).__name__, 'message': str(error)}, []
    return None, [(item.category, str(item.message)) for item in caught]


@pytest.mark.parametrize('operation', ['MIN', 'COUNT'])
def test_float32_source_cast_warns_before_aggregation(operation):
    error, expected = host_cast('warn')
    assert error is None
    with (
        np.errstate(over='warn'),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter('always')
        result = execute(aggregation(operation))
    assert [(item.category, str(item.message)) for item in caught] == expected
    assert result['output']['out']['values'] == (
        [1.0] if operation == 'MIN' else [2.0]
    )
    assert result['output']['mask'] == [True]


def test_float32_source_cast_respects_numpy_ignore():
    with (
        np.errstate(over='ignore'),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter('always')
        result = execute(aggregation())
    assert not caught
    assert result['output']['out']['values'] == [1.0]


def test_float32_source_cast_respects_numpy_raise():
    expected, _ = host_cast('raise')
    with np.errstate(over='raise'):
        result = execute(aggregation())
    assert result == (
        {'error': expected}
        if expected is not None
        else {
            'output': {
                'out': {'dtype': 'float32', 'values': [1.0]},
                'mask': [True],
            }
        }
    )


def test_float32_source_cast_respects_warning_error():
    expected, _ = host_cast('warn', warning_error=True)
    with np.errstate(over='warn'), warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        result = execute(aggregation())
    assert result == (
        {'error': expected}
        if expected is not None
        else {
            'output': {
                'out': {'dtype': 'float32', 'values': [1.0]},
                'mask': [True],
            }
        }
    )


def test_float32_source_cast_does_not_warn_for_infinity_or_finite_values():
    with (
        np.errstate(over='warn'),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter('always')
        result = execute(aggregation(values=[{'$float': 'inf'}, 1.0]))
    assert not caught
    assert result['output']['out']['values'] == [1.0]
