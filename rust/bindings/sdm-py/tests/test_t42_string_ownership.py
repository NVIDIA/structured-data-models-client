# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text data, offsets and validity detach at the existing frame boundary."""

import gc
import json

import messages
import numpy as np
import pyarrow as pa
import pytest
from sdm_client import _native


def done(run):
    while True:
        reply = json.loads(run.next()[0])
        if reply['message'] == 'warnings':
            continue
        assert reply['message'] == 'step', reply
        assert reply['step'] == 'done', reply
        return reply['result']


@pytest.mark.parametrize('large', [False, True])
@pytest.mark.parametrize('sliced', [False, True])
def test_string_table_owns_buffers_across_drop_and_schema_rebinding(
    large, sliced
):
    values = bytearray(b'aabbccdd')
    offsets = np.array([0, 2, 4, 6, 8], dtype=np.int64 if large else np.int32)
    validity = bytearray([0b1011])
    array = pa.Array.from_buffers(
        pa.large_string() if large else pa.string(),
        4,
        [pa.py_buffer(validity), pa.py_buffer(offsets), pa.py_buffer(values)],
    )
    if sliced:
        array = array.slice(1, 3)
    expected = ['bb', None, 'dd'] if sliced else ['aa', 'bb', None, 'dd']
    operation = {
        'op': 'describe_source',
        'provider': {'provider': 'frames', 'manifest': 2},
        'source': 1,
        'check_keys': True,
    }
    spec = {
        'tables': [{'name': 'rows', 'source': {'source': ['rows']}}],
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
                    'tables': [{'source': {'source': ['rows']}, 'rows': 3}],
                }
            ),
            3: pa.table({'value': array}),
        },
    )
    # Every foreign Arrow buffer is writable and remains live here.
    # Mutating all three after start must not alter the detached Table.
    values[:] = b'xxxxxxxx'
    offsets[:] = 0
    validity[:] = b'\0'
    del array
    result = done(run)
    snapshot = run.take_attachment(result['frame_rows'])
    del run
    gc.collect()
    owned = pa.RecordBatchReader._import_from_c_capsule(
        snapshot.__arrow_c_stream__()
    ).read_all()
    assert owned['value'].to_pylist() == expected
    assert owned.schema.field('value').type == pa.string()
    del owned
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
                    'tables': [{'source': {'source': ['rows']}, 'rows': 103}],
                }
            ),
            103: snapshot,
        },
    )
    del snapshot
    gc.collect()
    description = json.loads(
        second.take_attachment(done(second)['description'])
    )
    assert description['tables'][0]['columns'][0]['dtype'] == 'string'
    assert description['tables'][0]['columns'][0]['stype'] == 'categorical'
    del second
    gc.collect()
    assert description['tables'][0]['name'] == 'rows'
