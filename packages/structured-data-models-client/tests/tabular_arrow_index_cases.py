# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite public Arrow index journeys shared with the frozen recorder."""

import hashlib
import json
import warnings
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
from tabular_numeric_pure_cases import Endpoint

FAMILIES = (
    'decimal128',
    'decimal256',
    'decimal_negative_scale',
    'date32',
    'date64',
    'binary',
    'large_binary',
    'fixed_binary',
)
CASES = [
    f'{family}/{entry}/{shape}'
    for family in FAMILIES
    for entry in ('indices', 'predict')
    for shape in ('forward', 'reverse', 'all_null', 'one_present')
]
CASES += [
    f'multi_{family}/{entry}/forward'
    for family in ('decimal128', 'date32', 'binary')
    for entry in ('indices', 'predict')
]


def frame_for(family, n, shape):
    multi = family.startswith('multi_')
    family = family.removeprefix('multi_')
    keys = [None if i % 11 == 0 else (i * 17) % 29 - 14 for i in range(n)]
    if shape == 'reverse':
        keys.reverse()
    elif shape == 'all_null':
        keys = [None] * n
    elif shape == 'one_present':
        keys = [None] * n
        keys[5] = 3
    if family.startswith('decimal'):
        scale = -2 if family == 'decimal_negative_scale' else 2
        dtype = (
            pa.decimal256(60, scale)
            if family == 'decimal256'
            else pa.decimal128(20, scale)
        )

        # Tuples are exact even with a low ambient Decimal precision.
        def decimal(key):
            coefficient = (
                abs(key) * (10**40 + 1) if family == 'decimal256' else abs(key)
            )
            return Decimal(
                (int(key < 0), tuple(map(int, str(coefficient))), -scale)
            )

        array = pa.array(
            [None if k is None else decimal(k) for k in keys], type=dtype
        )
    elif family.startswith('date32'):
        dtype = pa.date32()
        offset = 10000000 if family == 'date32_wide' else 0
        array = pa.array(
            [None if k is None else k + offset for k in keys], type=pa.int32()
        ).cast(dtype)
    elif family == 'date64':
        dtype = pa.date64()
        array = pa.array(
            [None if k is None else k * 86400000 for k in keys], type=pa.int64()
        ).cast(dtype)
    else:
        dtype = (
            pa.binary(2)
            if family == 'fixed_binary'
            else pa.large_binary()
            if family == 'large_binary'
            else pa.binary()
        )
        labels = [
            b'',
            b'\x00',
            b'\x00\xff',
            b'a',
            b'a\x00',
            b'a\xff',
            b'\xff',
            b'\xff\x00',
        ]
        array = pa.array(
            [
                None
                if k is None
                else bytes([k + 14, 255 - (k + 14)])
                if family == 'fixed_binary'
                else labels[(k + 14) % len(labels)]
                for k in keys
            ],
            type=dtype,
        )
    split = n // 3
    chunks = pa.chunked_array(
        [array.slice(0, split), array.slice(split)], type=dtype
    )
    index = pd.Index(pd.array(chunks, dtype=pd.ArrowDtype(dtype)))
    if multi:
        index = pd.MultiIndex.from_arrays([np.arange(n) % 3, index])
    return pd.DataFrame(
        {
            'id': range(n),
            'feature': np.arange(n) * 0.5,
            'target': np.arange(n) + 100.0,
        },
        index=index,
    )


def digest(values):
    return hashlib.sha256(
        json.dumps(values, separators=(',', ':')).encode()
    ).hexdigest()


def observe_case(name):
    from sdm_client import KumoTabular, PredictiveQuery, SDMClient

    family, entry, shape = name.split('/')
    n = 81 if entry == 'indices' else 10002
    frame = frame_for(family, n, shape)
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
            context = request.context.instance_table
            ids = context['id'].tolist()
            encoded = [
                row[0]
                for row in endpoint.calls[0]['context']['instance_table'][
                    'rows'
                ]
            ]
            return {
                'context_count': len(ids),
                'context_ids_sha256': digest(ids),
                'encoded_ids_sha256': digest(encoded),
                'index_matches_source': context.index.equals(
                    frame.index.take(ids)
                ),
                'index_dtype': str(context.index.dtype),
                'predictions': result.to_pandas().to_dict(orient='records'),
            }
    except Exception as error:  # noqa: BLE001 - frozen public outcome
        return {'error': {'type': type(error).__name__, 'message': str(error)}}
    finally:
        client.close()
