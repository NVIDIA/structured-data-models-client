# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from sdm_client import KumoRelational, PredictiveQuery, Schema


@pytest.mark.parametrize(
    'kind', ['binary', 's', 'ms', 'us', 'ns', 'inferred_ns']
)
@pytest.mark.parametrize('missing', [False, True])
def test_public_duration_and_binary_targets_preserve_values(kind, missing):
    values = [1, 2, 3, 4]
    frame = pd.DataFrame(
        {'product_id': values, 'category': ['a', 'b', 'c', 'd']}
    )
    if kind == 'binary':
        raw = [b'one', b'two', b'\xff\x00', b'']
        if missing:
            raw[2] = None
        target = pd.Series(raw, dtype=pd.ArrowDtype(pa.binary()))
        expected = [
            None if value is None else base64.b64encode(value).decode('ascii')
            for value in raw
        ]
        expected_dtype = 'string'
    else:
        unit = 'ns' if kind == 'inferred_ns' else kind
        raw = np.array([1, -2, 3, 4], dtype=f'timedelta64[{unit}]')
        if missing:
            raw[2] = np.timedelta64('NaT', unit)
        target = pd.Series(raw)
        expected = [
            None if pd.isna(value) else pd.Timedelta(value).value
            for value in target
        ]
        expected_dtype = 'int64'
    if kind == 'inferred_ns':
        frame['category'] = target
    schema = Schema.from_data(
        {'products': frame}, primary_keys={'products': 'product_id'}
    )
    schema['products'].data = frame.assign(category=target)
    request = PredictiveQuery(
        'PREDICT products.category FOR EACH products.product_id', schema
    ).materialize(model=KumoRelational(), indices=[1, 2], random_seed=7)
    assert request.payloads
    seen = {}
    for payload in request.payloads:
        assert payload['task']['target']['dtype'] == expected_dtype
        if kind in ('ns', 'inferred_ns'):
            classes = payload['task']['target']['classes']
            expected_classes = [
                str(value) for value in expected if value is not None
            ]
            assert len(classes) == len(expected_classes)
            assert set(classes) == set(expected_classes)
        rows = payload['context']['instance_table']
        entity = rows['columns'].index('__kumo_entity_ref_0')
        label = rows['columns'].index('TARGET')
        for row in rows['rows']:
            identity = int(row[entity])
            assert row[label] == expected[identity - 1]
            seen[identity] = row[label]
    assert seen
    assert all(value is not None for value in seen.values())
