# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native temporal counts and validity across both owned constructor stages."""

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


@pytest.mark.parametrize('unit', ['us', 'ns'])
@pytest.mark.parametrize('zone', [None, 'America/Los_Angeles'])
def test_temporal_table_owns_exact_unit_zone_counts_and_validity_after_run_drop(
    unit, zone
):
    counts = np.array([-2, -1, 0, 1], dtype=np.int64)
    validity = bytearray([0b1011])
    kind = pa.timestamp(unit, tz=zone)
    array = pa.Array.from_buffers(
        kind, 4, [pa.py_buffer(validity), pa.py_buffer(counts)]
    )
    operation = {
        'op': 'describe_source',
        'provider': {'provider': 'frames', 'manifest': 2},
        'source': 1,
        'check_keys': True,
    }
    spec = {
        'tables': [{'name': 'events', 'source': {'source': ['events']}}],
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
                    'tables': [{'source': {'source': ['events']}, 'rows': 3}],
                }
            ),
            3: pa.table({'observed_at': array}),
        },
    )
    counts[:] = 0
    validity[:] = b'\0'
    del array
    result = done(run)
    snapshot = run.take_attachment(result['frame_rows'])
    del run
    gc.collect()
    owned = pa.RecordBatchReader.from_stream(snapshot).read_all()
    assert owned.schema.field('observed_at').type == kind
    assert owned['observed_at'].cast(pa.int64()).to_pylist() == [
        -2,
        -1,
        None,
        1,
    ]
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
                    'tables': [{'source': {'source': ['events']}, 'rows': 103}],
                }
            ),
            103: snapshot,
        },
    )
    del snapshot
    gc.collect()
    result = done(second)
    assert 'frame_rows' not in result
    described = json.loads(second.take_attachment(result['description']))
    assert described['tables'][0]['time_column'] == 'observed_at'
    assert described['tables'][0]['primary_key'] == []
