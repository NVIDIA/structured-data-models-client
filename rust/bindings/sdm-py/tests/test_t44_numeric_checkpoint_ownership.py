# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The numeric consumer owns native inputs before a later next can read them."""

import gc
import json
import weakref

import messages
import numpy as np
import pyarrow as pa
from sdm_client import _native


def column(name, stype):
    return {
        'name': name,
        'dtype': 'int64',
        'stype': stype,
        'stype_before_key': None,
    }


def test_materialize_frame_start_detaches_and_releases_foreign_producers():
    amounts = np.array(list(range(1, 19)), dtype=np.int64)
    producer = weakref.ref(amounts)
    orders = pa.table(
        {
            'order_id': pa.array(list(range(18)), type=pa.int64()),
            'user_id': pa.array([1] * 6 + [2] * 6 + [3] * 6, type=pa.int64()),
            'amount': pa.Array.from_buffers(
                pa.int64(), len(amounts), [None, pa.py_buffer(amounts)]
            ),
        }
    )
    tables = [
        {
            'name': 'users',
            'source': {'source': ['users']},
            'columns': [column('user_id', 'ID'), column('age', 'numerical')],
            'primary_key': ['user_id'],
            'frame_key_form': 'scalar',
            'time_column': None,
        },
        {
            'name': 'orders',
            'source': {'source': ['orders']},
            'columns': [
                column('order_id', 'ID'),
                column('user_id', 'ID'),
                column('amount', 'numerical'),
            ],
            'primary_key': ['order_id'],
            'frame_key_form': 'scalar',
            'time_column': None,
        },
    ]
    attached = {
        1: messages.encoded(
            {
                'tables': tables,
                'links': [
                    [
                        {'table': 'orders', 'column': 'user_id'},
                        {'table': 'users', 'column': 'user_id'},
                    ]
                ],
            }
        ),
        2: messages.encoded('PREDICT users.age FOR EACH users.user_id'),
        3: pa.table({'index': pa.array([3], type=pa.int64())}),
        5: messages.encoded(
            {
                'num_context_samples': 1,
                'num_neighbors': [2],
                'num_hops': 1,
                'random_seed': 7,
            }
        ),
        7: messages.encoded(
            {
                'tables': [
                    {'source': {'source': ['users']}, 'rows': 8},
                    {'source': {'source': ['orders']}, 'rows': 9},
                ]
            }
        ),
        8: pa.table(
            {
                'user_id': pa.array([1, 2, 3], type=pa.int64()),
                'age': pa.array([20, 30, 40], type=pa.int64()),
            }
        ),
        9: orders,
    }
    run = _native.Run.start(
        messages.start(
            {
                'op': 'materialize',
                'provider': {'provider': 'frames', 'manifest': 7},
                'schema': 1,
                'query': 2,
                'indices': 3,
                'seed': '0x7',
                'values': 5,
                'options': [],
            }
        ),
        attached,
    )
    amounts[:] = 999
    del amounts, orders, attached
    gc.collect()
    assert producer() is None

    while True:
        step = json.loads(run.next()[0])
        if step['message'] == 'warnings':
            continue
        assert step['message'] == 'step' and step['step'] == 'done', step
        break
    state = json.loads(run.take_sampler_state()[0])['state']
    checkpoint = state['frame_checkpoint']
    manifest = json.loads(run.take_attachment(checkpoint['manifest']))
    owned = {
        table['source']['source'][0]: pa.table(
            run.take_attachment(table['rows'])
        )
        for table in manifest['tables']
    }
    del run
    gc.collect()
    assert owned['orders']['amount'].to_pylist() == list(range(1, 19))
    assert owned['users']['age'].to_pylist() == [20, 30, 40]
