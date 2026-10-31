# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite ordinary object-label journeys, independent of candidate policy."""

import warnings
from datetime import timedelta
from decimal import Decimal, FloatOperation, InvalidOperation, localcontext

import numpy as np
import pandas as pd
from tabular_arrow_index_cases import digest
from tabular_numeric_pure_cases import Endpoint

CASES = [
    f'{family}/{entry}/{shape}/clear'
    for family in ('bytes', 'decimal_infinity', 'timedelta')
    for entry in ('indices', 'predict')
    for shape in ('forward', 'reverse', 'multi')
] + [
    f'decimal_float/{entry}/{shape}/{mode}'
    for entry in ('indices', 'predict')
    for shape in ('forward', 'reverse')
    for mode in ('clear', 'trapped')
]


def object_index(values):
    owners = np.empty(len(values), dtype=object)
    for position, value in enumerate(values):
        owners[position] = value
    index = pd.Index(owners, dtype=object)
    assert all(index[i] is owners[i] for i in range(len(owners)))
    return index


def frame_for(family, n, shape):
    patterns = {
        'bytes': [b'', b'a', b'a\x00', b'\x00', b'\xff', b'a\xff'],
        'decimal_infinity': [
            Decimal('Infinity'),
            Decimal('-Infinity'),
            Decimal('1.1'),
            0,
            2,
        ],
        'decimal_float': [
            Decimal('Infinity'),
            Decimal('-Infinity'),
            Decimal('1.1'),
            1.1,
            float('inf'),
            float('-inf'),
        ],
        'timedelta': [
            timedelta.min,
            timedelta.max,
            timedelta(microseconds=-1),
            timedelta(0),
            timedelta(days=-2, seconds=7, microseconds=3),
        ],
    }
    pattern = patterns[family]
    if shape == 'reverse':
        pattern = pattern[::-1]
    labels = [
        None if i % 11 == 0 else pattern[(i * 17) % len(pattern)]
        for i in range(n)
    ]
    index = object_index(labels)
    if shape == 'multi':
        level = object_index(pattern)
        index = pd.MultiIndex(
            levels=[pd.Index([0, 1]), level],
            codes=[
                np.arange(n) % 2,
                [
                    -1 if i % 11 == 0 else (i * 17) % len(pattern)
                    for i in range(n)
                ],
            ],
            verify_integrity=False,
        )
        assert all(
            index.levels[1][i] is pattern[i] for i in range(len(pattern))
        )
    return pd.DataFrame(
        {
            'id': range(n),
            'feature': np.arange(n) * 0.5,
            'target': np.arange(n) + 100.0,
        },
        index=index,
    )


def observe_case(name):
    from sdm_client import KumoTabular, PredictiveQuery, SDMClient

    family, entry, shape, mode = name.split('/')
    n = 81 if entry == 'indices' else 10002
    with localcontext() as context:
        context.prec = 2
        context.traps[FloatOperation] = False
        context.traps[InvalidOperation] = False
        frame = frame_for(family, n, shape)
        original_index = frame.index
        context.clear_flags()
        context.traps[FloatOperation] = mode == 'trapped'
        endpoint = Endpoint()
        client = SDMClient.__new__(SDMClient)
        client._start(endpoint)
        try:
            with warnings.catch_warnings(record=True):
                query = PredictiveQuery(
                    'PREDICT data.target FOR EACH data.id', {'data': frame}
                )
                args = (
                    {'indices': [n - 1], 'num_context_samples': 40}
                    if entry == 'indices'
                    else {'predict': frame.iloc[[-1]].drop(columns=['target'])}
                )
                request = query.materialize(**args, task='regression')
                result = client.predict(request, model=KumoTabular())
                observed_flags = sorted(
                    signal.__name__
                    for signal, set_ in context.flags.items()
                    if set_
                )
                selected = request.context.instance_table
                ids = selected['id'].tolist()
                encoded = [
                    row[0]
                    for row in endpoint.calls[0]['context']['instance_table'][
                        'rows'
                    ]
                ]
                assert frame.index is original_index
                if not isinstance(original_index, pd.MultiIndex):
                    assert all(
                        selected.index[i] is original_index[row]
                        for i, row in enumerate(ids)
                    )
                return {
                    'context_count': len(ids),
                    'context_ids_sha256': digest(ids),
                    'encoded_ids_sha256': digest(encoded),
                    'index_dtype': str(selected.index.dtype),
                    'predictions': result.to_pandas().to_dict(orient='records'),
                    'flags': observed_flags,
                }
        except Exception as error:  # noqa: BLE001 - frozen public outcome
            return {
                'error': {'type': type(error).__name__, 'message': str(error)},
                'flags': sorted(
                    signal.__name__
                    for signal, set_ in context.flags.items()
                    if set_
                ),
            }
        finally:
            client.close()
