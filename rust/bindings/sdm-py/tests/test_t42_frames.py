# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Call-owned frame data through the genuine installed native Run."""

from __future__ import annotations

import gc
import json

import messages
import numpy as np
import pyarrow as pa
import pytest

from sdm_client import _native


@pytest.mark.parametrize('kind', ['numbers', 'validity', 'sliced'])
def test_frame_start_detaches_numpy_producer_before_later_steps(kind):
    count = 100
    values = np.arange(count, dtype=np.int64)
    validity = bytearray([255] * 13)
    if kind == 'validity':
        array = pa.Array.from_buffers(
            pa.int64(), count, [pa.py_buffer(validity), pa.py_buffer(values)]
        )
    else:
        array = pa.array(values)
        if kind == 'sliced':
            array = array.slice(2, count - 4)
    table = pa.table({'user_id': array})
    operation = {
        'op': 'describe_source',
        'provider': {'provider': 'frames', 'manifest': 2},
        'source': 1,
        'check_keys': True,
    }
    attached = {
        1: messages.encoded(
            {
                'tables': [{'name': 'users', 'source': {'source': ['users']}}],
                'primary_keys': {},
                'time_columns': {},
            }
        ),
        2: messages.encoded(
            {
                'stage': 'schema',
                'tables': [{'source': {'source': ['users']}, 'rows': 3}],
            }
        ),
        3: table,
    }
    run = _native.Run.start(messages.start(operation), attached)
    # These writes affect the exported Arrow objects too. The frame run must
    # already own independent values and validity, not just an Arc.
    values[:] = 1
    validity[:] = b'\0' * len(validity)
    del attached, table, array
    gc.collect()
    while True:
        outgoing = json.loads(run.next()[0])
        if outgoing['message'] == 'warnings':
            continue
        assert outgoing['message'] == 'step'
        assert outgoing['step'] == 'done', outgoing
        result = outgoing['result']
        description = json.loads(run.take_attachment(result['description']))
        break
    assert description['tables'][0]['primary_key'] == ['user_id']
    # Returned metadata is ordinary bytes and stays valid after the Run dies.
    del run
    gc.collect()
    assert description['tables'][0]['name'] == 'users'


def test_table_output_is_owned_after_run_drop_and_rebinds_under_fresh_ids():
    values = np.arange(100, dtype=np.int64)
    operation = {
        'op': 'describe_source',
        'provider': {'provider': 'frames', 'manifest': 2},
        'source': 1,
        'check_keys': True,
    }
    spec = {
        'tables': [{'name': 'users', 'source': {'source': ['users']}}],
        'primary_keys': {},
        'time_columns': {},
    }
    run = _native.Run.start(
        messages.start(operation),
        {
            1: messages.encoded(spec),
            2: messages.encoded(
                {
                    'stage': 'table',
                    'tables': [{'source': {'source': ['users']}, 'rows': 3}],
                }
            ),
            3: pa.table({'user_id': pa.array(values)}),
        },
    )
    while True:
        reply = json.loads(run.next()[0])
        if reply['message'] == 'warnings':
            continue
        assert reply['step'] == 'done'
        snapshot = run.take_attachment(reply['result']['frame_rows'])
        break
    del run
    values[:] = 1
    gc.collect()
    second = _native.Run.start(
        messages.start(
            {
                **operation,
                'source': 101,
                'provider': {'provider': 'frames', 'manifest': 102},
            }
        ),
        {
            101: messages.encoded(spec),
            102: messages.encoded(
                {
                    'stage': 'schema',
                    'tables': [{'source': {'source': ['users']}, 'rows': 103}],
                }
            ),
            103: snapshot,
        },
    )
    del snapshot
    gc.collect()
    while True:
        reply = json.loads(second.next()[0])
        if reply['message'] == 'warnings':
            continue
        assert reply['step'] == 'done'
        assert 'frame_rows' not in reply['result']
        description = json.loads(
            second.take_attachment(reply['result']['description'])
        )
        assert description['tables'][0]['primary_key'] == ['user_id']
        break
