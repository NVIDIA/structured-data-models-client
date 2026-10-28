# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constructor float snapshots retain actual raw bits and validity owners."""

from __future__ import annotations

import gc
import json

import messages
import numpy as np
import pyarrow as pa
import pytest
from sdm_client import _native


def inference(stage, table, *, base=0):
    operation = {
        'op': 'describe_source',
        'provider': {'provider': 'frames', 'manifest': base + 2},
        'source': base + 1,
        'check_keys': True,
    }
    return _native.Run.start(
        messages.start(operation),
        {
            base + 1: messages.encoded(
                {
                    'tables': [
                        {'name': 'rows', 'source': {'source': ['rows']}}
                    ],
                    'primary_keys': {},
                    'time_columns': {},
                }
            ),
            base + 2: messages.encoded(
                {
                    'stage': stage,
                    'tables': [
                        {'source': {'source': ['rows']}, 'rows': base + 3}
                    ],
                }
            ),
            base + 3: table,
        },
    )


def finished(run):
    while True:
        outgoing = json.loads(run.next()[0])
        if outgoing['message'] == 'warnings':
            continue
        assert outgoing['message'] == 'step'
        assert outgoing['step'] == 'done', outgoing
        return outgoing['result']


def imported(stream):
    return pa.RecordBatchReader._import_from_c_capsule(
        stream.__arrow_c_stream__()
    ).read_all()


def raw_bits(array, width):
    return np.frombuffer(array.buffers()[1], dtype=f'uint{width}')[
        array.offset : array.offset + len(array)
    ].tolist()


@pytest.mark.parametrize('width', [32, 64])
@pytest.mark.parametrize('offset', [0, 1, 3, 7])
@pytest.mark.parametrize('chunked', [False, True])
def test_constructor_float_buffers_detach_before_producer_mutation_and_rebind(
    width, offset, chunked
):
    quiet = 0x7FC00000 if width == 32 else 0x7FF8000000000000
    signaling = 0x7F800001 if width == 32 else 0x7FF0000000000001
    sign = 1 << (width - 1)
    bits = np.array(
        [
            0,
            sign,
            quiet,
            quiet | sign,
            quiet | 1,
            signaling,
            quiet | 2,
            1,
            sign | 1,
            0,
            sign,
            quiet,
        ],
        dtype=f'uint{width}',
    )
    validity = bytearray([0b10101101, 0b00001101])
    array = pa.Array.from_buffers(
        pa.float32() if width == 32 else pa.float64(),
        len(bits),
        [pa.py_buffer(validity), pa.py_buffer(bits)],
    ).slice(offset)
    chunks = (
        [
            array.slice(0, 0),
            array.slice(0, 2),
            array.slice(2),
            array.slice(0, 0),
        ]
        if chunked
        else [array]
    )
    producer_addresses = {
        buffer.address
        for chunk in chunks
        for buffer in chunk.buffers()
        if buffer is not None and buffer.size
    }
    table = pa.table({'value': pa.chunked_array(chunks)})
    # Observe the producer's C-stream batch layout, including whatever it
    # exports for empty chunks, before independent SDK admission/detachment.
    exported = imported(table)
    expected = [
        (chunk.offset, raw_bits(chunk, width), chunk.is_valid().to_pylist())
        for chunk in exported.column('value').chunks
    ]
    run = inference('table', table)
    bits[:] = 0
    validity[:] = b'\0' * len(validity)
    assert all(value == 0 for value in raw_bits(array, width))
    assert array.is_valid().to_pylist() == [False] * len(array)
    del table, array, chunks, exported
    gc.collect()
    result = finished(run)
    description = json.loads(run.take_attachment(result['description']))
    assert description['tables'][0]['columns'][0]['dtype'] == f'float{width}'
    assert description['tables'][0]['columns'][0]['stype'] == 'numerical'
    stream = run.take_attachment(result['frame_rows'])
    del run
    gc.collect()
    snapshot = imported(stream)
    actual = snapshot.column('value').chunks
    assert [
        (chunk.offset, raw_bits(chunk, width), chunk.is_valid().to_pylist())
        for chunk in actual
    ] == expected
    assert all(
        buffer.address not in producer_addresses
        for chunk in actual
        for buffer in chunk.buffers()
        if buffer is not None and buffer.size
    )
    rebound = inference('table', snapshot, base=100)
    del stream, snapshot
    gc.collect()
    rebound_result = finished(rebound)
    assert (
        json.loads(rebound.take_attachment(rebound_result['description']))
        == description
    )
    again = imported(rebound.take_attachment(rebound_result['frame_rows']))
    assert [
        (chunk.offset, raw_bits(chunk, width), chunk.is_valid().to_pylist())
        for chunk in again.column('value').chunks
    ] == expected
    fresh = inference(
        'table',
        pa.table(
            {
                'value': pa.Array.from_buffers(
                    pa.float32() if width == 32 else pa.float64(),
                    len(bits),
                    [pa.py_buffer(validity), pa.py_buffer(bits)],
                ).slice(offset)
            }
        ),
    )
    fresh_result = finished(fresh)
    changed = imported(fresh.take_attachment(fresh_result['frame_rows']))
    assert changed.column('value').null_count == len(bits) - offset
    assert all(
        value == 0
        for chunk in changed.column('value').chunks
        for value in raw_bits(chunk, width)
    )
