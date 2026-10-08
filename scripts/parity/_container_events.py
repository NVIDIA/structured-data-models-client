# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Symbolic host inputs that JSON cannot represent literally."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

CASES = (
    'list-cycle',
    'dict-cycle',
    'array-cycle',
    'array-list-cycle',
    'deep-list',
    'deep-dict',
    'deep-tuple',
    'deep-array',
    'shared-list',
    'shared-array',
    'ordinary-list',
)


def value_for(case: str) -> Any:
    if case == 'list-cycle':
        value: Any = []
        value.append(value)
        return value
    if case == 'dict-cycle':
        value = {}
        value['self'] = value
        return value
    if case in ('array-cycle', 'array-list-cycle'):
        value = np.empty(1, dtype=object)
        value[0] = value if case == 'array-cycle' else [value]
        return value
    if case.startswith('deep-'):
        value = 1
        for _ in range(1500):
            if case == 'deep-dict':
                value = {'child': value}
            elif case == 'deep-tuple':
                value = (value,)
            elif case == 'deep-array':
                parent = np.empty(1, dtype=object)
                parent[0] = value
                value = parent
            else:
                value = [value]
        return value
    if case.startswith('shared-'):
        shared = np.array([1, 2]) if case == 'shared-array' else [1, 2]
        return [shared, shared]
    if case == 'ordinary-list':
        return [{'values': [1, None, 2]}]
    if case == 'ordinary-array':
        return np.array([1, 2])
    raise ValueError(case)


def capture(case: str, entry: str) -> dict[str, Any]:
    from sdm_client import PredictiveQuery, Schema

    value = value_for(case)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            if entry == 'schema':
                Schema.from_data(
                    {
                        'users': pd.DataFrame(
                            {'user_id': [1], 'feature': [value]}
                        )
                    },
                    primary_keys={'users': 'user_id'},
                )
            else:
                query = PredictiveQuery(
                    'PREDICT users.target FOR EACH users.user_id',
                    {
                        'users': pd.DataFrame(
                            {
                                'user_id': [1, 2, 3],
                                'target': [3.0, 4.0, 5.0],
                            }
                        )
                    },
                )
                query.materialize(indices=value, random_seed=7)
        except Exception as error:
            outcome = {
                'class': f'{type(error).__module__}.{type(error).__qualname__}',
                'code': getattr(error, 'code', None),
                'message': str(error),
            }
        else:
            outcome = {'status': 'accepted'}
    outcome['warnings'] = [
        {
            'category': f'{warning.category.__module__}.{warning.category.__qualname__}',
            'message': str(warning.message),
        }
        for warning in caught
    ]
    return outcome


def capture_container_events() -> dict[str, Any]:
    events = {}
    for entry in ('schema', 'indices'):
        for case in CASES:
            outcome = capture(case, entry)
            assert (
                outcome['class'] == 'sdm_client.errors.StructuredDataError'
            ), outcome
            assert outcome['code'] == (
                'UNSUPPORTED_REQUEST'
                if entry == 'schema'
                else 'INVALID_REQUEST'
            ), outcome
            events[f'{entry}/{case}'] = outcome
    return events
