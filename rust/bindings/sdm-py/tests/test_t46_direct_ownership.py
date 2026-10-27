# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct numeric preparation detaches producer buffers at the reached stage."""

import gc
import json

import messages
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from sdm_client import _native
from sdm_client._rust import prepare


@pytest.mark.parametrize('rows', [0, 2, 5])
def test_zero_column_scoring_rows_survive_native_admission_and_owner_release(
    rows,
):
    context = pd.DataFrame({'target': [2, 4, 6]})
    scoring = pd.DataFrame(index=range(rows))
    context_header, scoring_header = (
        prepare._header(context),
        prepare._header(scoring),
    )
    arguments = prepare._call(
        {
            'stage': 'arguments',
            'context': {'kind': 'frame'},
            'predict': {'kind': 'frame'},
            'related': [],
            'task': {'value': 'regression'},
        }
    )
    headers = prepare._call(
        {
            'stage': 'headers',
            'arguments': arguments,
            'context': context_header,
            'predict': scoring_header,
            'target_context': {'positions': [0]},
            'target_predict': {'positions': []},
            'predict_context': [],
        }
    )
    producer = prepare._buffers(scoring, scoring_header)
    run = _native.Run.start(
        messages.start({'op': 'prepare_request', 'input': 1}),
        {
            1: messages.encoded(
                {
                    'stage': 'context',
                    'headers': headers,
                    'context': 2,
                    'predict': 3,
                    'entity_table': 'instance',
                }
            ),
            2: prepare._buffers(context, context_header),
            3: producer,
        },
    )
    del producer, scoring, context
    gc.collect()
    reply = json.loads(run.next()[0])
    assert reply['step'] == 'done', reply
    prepared = json.loads(run.take_attachment(reply['result']['value']))
    assert prepared['target'] == 'target'
    assert prepared['entity_table'] == 'instance'
    # Context verifies the native Arrow row count against the original header.
    # Reaching this useful result proves zero-column rows survived ownership.
    del run
    gc.collect()


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
