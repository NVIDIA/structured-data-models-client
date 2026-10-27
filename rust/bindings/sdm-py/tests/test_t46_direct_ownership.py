# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct numeric preparation detaches producer buffers at the reached stage."""

import gc
import json

import messages
import numpy as np
import pyarrow as pa
from sdm_client import _native
from sdm_client._rust import prepare


def test_direct_schema_start_owns_numeric_buffers_before_next():
    arguments = prepare._call(
        {
            'stage': 'arguments',
            'context': {'kind': 'frame'},
            'predict': {'kind': 'frame'},
            'related': [],
            'task': {'value': 'regression'},
        }
    )
    context = {
        'rows': 3,
        'columns': [
            {
                'ordinal': 0,
                'name': {'kind': 'text', 'value': 'feature_id'},
                'physical': {'type': 'uint64'},
            },
            {
                'ordinal': 1,
                'name': {'kind': 'text', 'value': 'target'},
                'physical': {'type': 'int64'},
            },
        ],
        'duplicates': [],
    }
    scoring = {
        'rows': 2,
        'columns': [
            {
                'ordinal': 0,
                'name': {'kind': 'text', 'value': 'feature_id'},
                'physical': {'type': 'uint64'},
            }
        ],
        'duplicates': [],
    }
    headers = prepare._call(
        {
            'stage': 'headers',
            'arguments': arguments,
            'context': context,
            'predict': scoring,
            'target_context': {'positions': [1]},
            'target_predict': {'positions': []},
            'predict_context': [[0]],
        }
    )
    feature = np.array([1, 2, 3], dtype='uint64')
    target = np.array([2, 4, 6], dtype='int64')
    run = _native.Run.start(
        messages.start({'op': 'prepare_request', 'input': 1}),
        {
            1: messages.encoded(
                {
                    'stage': 'schema',
                    'headers': headers,
                    'context': 2,
                    'predict': 3,
                    'entity_table': 'instance',
                }
            ),
            2: pa.Table.from_arrays(
                [pa.array(feature), pa.array(target)], names=['0', '1']
            ),
            3: pa.table({'0': pa.array([4, 5], type=pa.uint64())}),
        },
    )
    feature[:] = 2**63
    target[:] = 2
    del feature, target
    gc.collect()
    warnings = []
    while True:
        reply = json.loads(run.next()[0])
        if reply['message'] == 'warnings':
            warnings.extend(reply['warnings'])
            continue
        assert reply['step'] == 'done', reply
        prepared = json.loads(run.take_attachment(reply['result']['value']))
        break
    del run
    gc.collect()
    assert not warnings
    table = prepared['description']['tables'][0]
    assert table['primary_key'] == ['feature_id']
    assert table['columns'][0]['dtype'] == 'int64'
    assert prepared['target'] == 'target'
