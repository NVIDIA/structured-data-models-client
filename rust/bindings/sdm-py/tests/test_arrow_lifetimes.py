# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Arrow through the installed binding, for the wheel smoke (T1.17).

What the clean containers of ``wheel-smoke`` check through
``sdm_client._native`` at each lane's pandas, NumPy and pyarrow, over the
test-only fetch that echoes the data it admitted:

- a canonical input, one column of every type of the closed set, crosses
  sharing its buffers: each buffer of the data the core gives back is the
  input's own, at the input's address;
- the run holds the producer's memory while it holds the data, and lets go
  of it exactly: pyarrow's default pool, which allocates the input's
  buffers and each exported array's private data, keeps every byte while
  the run or an output holds them, and returns to its count before the
  input once nothing does;
- an input in a form the admission table converts comes back in the
  canonical type the request expects, check (b) passing on it, with its
  values and its null masks;
- an output stays valid, over the PyCapsule path, after its input is
  collected, after the run has errored, and after the run is destroyed;
- busy and cancel behave through the binding: a cancel while another call
  is in the run is ``RUN_BUSY``, a cancel releases what the run held, and
  every later step, reply or cancel is ``RUN_CANCELLED``.

pyarrow's pool counts only what pyarrow allocates, so a binding that
copied the input into memory of its own would let the pool fall as soon as
the input was collected, and one that released the producer's memory while
an output still pointed into it would let it fall early; both fail here, as
does an output whose buffers are not the input's.
"""

from __future__ import annotations

import datetime
import decimal
import gc
import logging
import threading
from typing import Any

import messages
import numpy as np
import pyarrow as pa
import pytest
from sdm_client import _native


@pytest.fixture(autouse=True)
def _test_ops_build(native_build: str) -> None:
    assert native_build == 'test-ops', 'these tests need the test-ops build'
    assert _native.TEST_OPS, 'the installed _native has no test operations'


def _held() -> int:
    """The bytes pyarrow's default pool holds, once nothing unreachable
    keeps any."""
    gc.collect()
    return pa.total_allocated_bytes()


def _canonical() -> dict[str, pa.Array]:
    """One column of every type of the closed set (``sdm_contract``'s
    ``ColumnType``), each with a null, in the canonical form the core
    keeps."""
    return {
        'null': pa.array([None, None, None], pa.null()),
        'bool': pa.array([True, None, False]),
        'int8': pa.array([1, None, -128], pa.int8()),
        'int16': pa.array([1, None, -3], pa.int16()),
        'int32': pa.array([1, None, -3], pa.int32()),
        'int64': pa.array([1, None, -(2**63)], pa.int64()),
        'uint8': pa.array([1, None, 255], pa.uint8()),
        'uint16': pa.array([1, None, 3], pa.uint16()),
        'uint32': pa.array([1, None, 3], pa.uint32()),
        'uint64': pa.array([1, None, 2**64 - 1], pa.uint64()),
        'float32': pa.array([1.5, None, -2.0], pa.float32()),
        'float64': pa.array([1.5, None, -0.0], pa.float64()),
        'decimal128': pa.array(
            [decimal.Decimal('1.25'), None, decimal.Decimal('-3.50')],
            pa.decimal128(10, 2),
        ),
        'decimal256': pa.array(
            [decimal.Decimal('1.25'), None, decimal.Decimal('-3.50')],
            pa.decimal256(50, 2),
        ),
        'utf8': pa.array(['a', None, 'ccc']),
        'binary': pa.array([b'a', None, b'\x00c'], pa.binary()),
        'date32': pa.array(
            [datetime.date(2024, 1, 1), None, datetime.date(1960, 2, 29)],
            pa.date32(),
        ),
        'timestamp_us': pa.array([1, None, -1], pa.timestamp('us')),
        'timestamp_ns_utc': pa.array(
            [1, None, -1], pa.timestamp('ns', tz='UTC')
        ),
        'duration_us': pa.array([1, None, -1], pa.duration('us')),
        'duration_ns': pa.array([1, None, -1], pa.duration('ns')),
        'time64_us': pa.array([1, None, 86_399_999_999], pa.time64('us')),
        'time64_ns': pa.array([1, None, 5], pa.time64('ns')),
        'list': pa.array([[1, None], None, []], pa.list_(pa.int64())),
        'struct': pa.array(
            [{'x': 1, 'y': 'a'}, None, {'x': None, 'y': None}],
            pa.struct([('x', pa.int64()), ('y', pa.utf8())]),
        ),
        'map': pa.array(
            [[('k', 1)], None, [('j', None)]], pa.map_(pa.utf8(), pa.int64())
        ),
        'dictionary_utf8': pa.array(['a', None, 'a']).dictionary_encode(),
    }


def _addresses(array: pa.Array) -> list[int | None]:
    """Where each buffer of `array` is, its children's and its
    dictionary's included; ``None`` for a buffer it does not have."""
    found = [None if b is None else b.address for b in array.buffers()]
    if pa.types.is_dictionary(array.type):
        found += _addresses(array.dictionary)
    return found


def _only_chunk(table: pa.Table, name: str) -> pa.Array:
    column = table.column(name)
    assert column.num_chunks == 1, (name, column.num_chunks)
    return column.chunk(0)


def _echoed(
    data: pa.Table, expect: dict[str, Any] | None = None
) -> tuple[_native.Run, _native.ArrowStream]:
    """`data` through the test-only fetch that echoes what it admitted,
    checked against `expect` (check (b)), or discovered; the run, finished,
    and the attachment it gave back."""
    run = _native.Run.start(
        *messages.fetch(expect or {'expect': 'discover'}, echo=True)
    )
    assert messages.read(run.next()[0])['step'] == 'fetch'
    run.reply(messages.fetch_reply(), {1: data})
    done = messages.read(run.next()[0])
    assert done['result'] == {
        'op': 'test',
        'rows': data.num_rows,
        'data': 1,
    }
    stream = run.take_attachment(1)
    assert isinstance(stream, _native.ArrowStream)
    return run, stream


def _imported(exportable: Any) -> pa.Table:
    """What a consumer reads through the PyCapsule interface."""
    capsule = exportable.__arrow_c_stream__()
    return pa.RecordBatchReader._import_from_c_capsule(capsule).read_all()


def test_the_canonical_set_is_every_type_of_the_closed_set() -> None:
    # The columns below stand for the closed set; a type left out would
    # make every test here silent about it.
    names = set(_canonical())
    assert len(names) == 27
    for name, array in _canonical().items():
        assert array.null_count >= 1, name


def test_canonical_inputs_cross_sharing_their_buffers() -> None:
    sent = pa.table(_canonical())
    run, stream = _echoed(sent)
    back = _imported(stream)
    assert back.schema.equals(sent.schema, check_metadata=True)
    compared = 0
    for name in sent.column_names:
        given, taken = _only_chunk(sent, name), _only_chunk(back, name)
        assert taken.equals(given), name
        assert _addresses(taken) == _addresses(given), name
        present = [a for a in _addresses(given) if a is not None]
        # Every column but null's has buffers, so none passes vacuously.
        assert present or name == 'null', name
        compared += len(present)
    assert compared >= 2 * (len(sent.column_names) - 1)
    del run


def _unaligned_values() -> pa.Array:
    values = np.ndarray((3,), dtype=np.int64, buffer=bytearray(25), offset=1)
    values[:] = [1, 2, 3]
    array = pa.Array.from_buffers(
        pa.int64(),
        3,
        [pa.py_buffer(b'\x05'), pa.py_buffer(values)],
        null_count=1,
    )
    assert array.buffers()[1].address % 8 == 1
    return array


def _unaligned_text() -> pa.Array:
    offsets = np.ndarray((4,), dtype=np.int32, buffer=bytearray(17), offset=1)
    offsets[:] = [0, 1, 1, 3]
    array = pa.Array.from_buffers(
        pa.utf8(),
        3,
        [pa.py_buffer(b'\x05'), pa.py_buffer(offsets), pa.py_buffer(b'acc')],
        null_count=1,
    )
    assert array.buffers()[1].address % 4 == 1
    return array


@pytest.mark.parametrize(
    'shape',
    [
        'primitive',
        'utf8-offsets',
        'nested',
        'dictionary-values',
        'dictionary-indices',
    ],
)
def test_unaligned_buffers_are_refused_by_name_before_typed_import(
    shape: str,
) -> None:
    if shape == 'primitive':
        array = _unaligned_values()
    elif shape == 'utf8-offsets':
        array = _unaligned_text()
    elif shape == 'nested':
        array = pa.ListArray.from_arrays([0, 1, 2, 3], _unaligned_values())
    elif shape == 'dictionary-values':
        array = pa.DictionaryArray.from_arrays(
            pa.array([0, None, 2], pa.int8()), _unaligned_text()
        )
    else:
        array = pa.DictionaryArray.from_arrays(
            _unaligned_values(), pa.array(['a', 'b', 'c', 'd'])
        )
    run = _native.Run.start(*messages.fetch({'expect': 'discover'}))
    assert messages.read(run.next()[0])['step'] == 'fetch'
    with pytest.raises(_native.CoreError) as raised:
        run.reply(messages.fetch_reply(), {1: pa.table({'a': array})})
    error = raised.value
    assert (error.kind, error.code) == (
        'data_source',
        'PROVIDER_SCHEMA_MISMATCH',
    )
    fields = messages.read(error.control)['error']['fields']
    assert fields['check'] == {'t': 'utf8', 'v': 'declared'}
    assert fields['attachment'] == {'t': 'uint32', 'v': 1}
    with pytest.raises(_native.CoreError) as finished:
        run.next()
    assert finished.value.code == 'RUN_FINISHED'


def test_empty_primitive_dangling_buffer_remains_admissible() -> None:
    # A zero-byte C buffer may use a dangling address. It is never read and
    # Arrow's importer replaces it with its own aligned empty allocation.
    array = pa.Array.from_buffers(
        pa.int64(), 0, [None, pa.foreign_buffer(1, 0)]
    )
    batch = pa.record_batch([array], names=['a'])
    # Tables may omit empty batches entirely. An explicit reader makes the
    # native preflight see this batch and its zero-byte data buffer.
    reader = pa.RecordBatchReader.from_batches(batch.schema, [batch])
    run = _native.Run.start(*messages.fetch({'expect': 'discover'}, echo=True))
    run.next()
    run.reply(messages.fetch_reply(), {1: reader})
    assert messages.read(run.next()[0])['result']['rows'] == 0
    assert _imported(run.take_attachment(1)).num_rows == 0


def test_the_run_holds_the_producer_s_memory_and_releases_it_exactly() -> None:
    before = _held()
    sent = pa.table(_canonical())
    with_input = _held()
    assert with_input > before
    run = _native.Run.start(*messages.fetch({'expect': 'discover'}))
    run.next()
    run.reply(messages.fetch_reply(), {1: sent})
    del sent
    # The input collected on this side: the run still holds its buffers,
    # and the private data of each array pyarrow exported for it.
    assert _held() > with_input, 'the input was copied or released'
    done = messages.read(run.next()[0])
    assert done['result'] == {'op': 'test', 'rows': 3}
    # The run counted the rows and dropped the data: every byte of the
    # input, and of its export, is back, and nothing more.
    assert _held() == before
    del run
    assert _held() == before


def test_an_output_stays_valid_after_its_input_its_run_s_error_and_its_run() -> (
    None
):
    before = _held()
    expected = pa.table(_canonical())
    with_expected = _held()
    sent = pa.table(_canonical())
    run, stream = _echoed(sent)
    early = stream.__arrow_c_stream__()
    del sent
    holding = _held()
    assert holding > with_expected, 'the input was copied or released'
    # The run errors: a call after it finished, and a reply it never asked.
    with pytest.raises(_native.CoreError) as finished:
        run.next()
    assert finished.value.code == 'RUN_FINISHED'
    with pytest.raises(_native.CoreError):
        run.reply(messages.fetch_reply(), {1: pa.table(_canonical())})
    assert _held() == holding
    del run
    # The run destroyed: the output still holds the producer's memory.
    assert _held() == holding, 'an output was freed with its run'
    _check_values_and_nulls(_imported(stream), expected)
    # A capsule taken before the run went is read after it.
    _check_values_and_nulls(
        pa.RecordBatchReader._import_from_c_capsule(early).read_all(), expected
    )
    del early
    assert _held() == holding
    del stream
    assert _held() == with_expected
    del expected
    assert _held() == before


def _check_values_and_nulls(back: pa.Table, expected: pa.Table) -> None:
    """`back` holds `expected`'s values, with its null masks."""
    assert back.schema.equals(expected.schema)
    for name in expected.column_names:
        given = _only_chunk(expected, name)
        taken = _only_chunk(back, name)
        assert taken.is_null().equals(given.is_null()), name
        assert taken.equals(given), name


def _admitted() -> dict[str, tuple[pa.Array, dict[str, Any], pa.Array]]:
    """Forms the admission table converts, each as it is sent, the
    canonical type the request expects back (check (b)), and what it holds
    there, written apart from the input."""
    utc = 'UTC'
    return {
        'large_utf8': (
            pa.array(['a', None, 'cc'], pa.large_utf8()),
            {'type': 'utf8'},
            pa.array(['a', None, 'cc'], pa.utf8()),
        ),
        'large_binary': (
            pa.array([b'a', None, b'cc'], pa.large_binary()),
            {'type': 'binary'},
            pa.array([b'a', None, b'cc'], pa.binary()),
        ),
        'fixed_size_binary': (
            pa.array([b'ab', None, b'cd'], pa.binary(2)),
            {'type': 'binary'},
            pa.array([b'ab', None, b'cd'], pa.binary()),
        ),
        'float16': (
            pa.array(
                np.array([1.5, 0.0, -2.0], dtype=np.float16),
                mask=np.array([False, True, False]),
            ),
            {'type': 'float32'},
            pa.array([1.5, None, -2.0], pa.float32()),
        ),
        'timestamp_s_utc': (
            pa.array([1, None, -1], pa.timestamp('s', tz=utc)),
            {'type': 'timestamp', 'unit': 'us', 'zone': utc},
            pa.array([1_000_000, None, -1_000_000], pa.timestamp('us', utc)),
        ),
        'timestamp_ms': (
            pa.array([1, None, -1], pa.timestamp('ms')),
            {'type': 'timestamp', 'unit': 'us', 'zone': None},
            pa.array([1_000, None, -1_000], pa.timestamp('us')),
        ),
        'duration_s': (
            pa.array([1, None, -1], pa.duration('s')),
            {'type': 'duration', 'unit': 'us'},
            pa.array([1_000_000, None, -1_000_000], pa.duration('us')),
        ),
        'duration_ms': (
            pa.array([1, None, -1], pa.duration('ms')),
            {'type': 'duration', 'unit': 'us'},
            pa.array([1_000, None, -1_000], pa.duration('us')),
        ),
        'time32_s': (
            pa.array([1, None, 86_399], pa.time32('s')),
            {'type': 'time64', 'unit': 'us'},
            pa.array([1_000_000, None, 86_399_000_000], pa.time64('us')),
        ),
        'time32_ms': (
            pa.array([1, None, 86_399_999], pa.time32('ms')),
            {'type': 'time64', 'unit': 'us'},
            pa.array([1_000, None, 86_399_999_000], pa.time64('us')),
        ),
        'date64': (
            pa.array([86_400_000, None, 0], pa.date64()),
            {'type': 'timestamp', 'unit': 'us', 'zone': None},
            pa.array([86_400_000_000, None, 0], pa.timestamp('us')),
        ),
        'large_list': (
            pa.array([[1, None], None, []], pa.large_list(pa.int64())),
            {'type': 'list', 'item': _field('item', {'type': 'int64'})},
            pa.array([[1, None], None, []], pa.list_(pa.int64())),
        ),
        'fixed_size_list': (
            pa.array([[1, None], None, [3, 4]], pa.list_(pa.int64(), 2)),
            {'type': 'list', 'item': _field('item', {'type': 'int64'})},
            pa.array([[1, None], None, [3, 4]], pa.list_(pa.int64())),
        ),
        'dictionary_large_utf8': (
            pa.DictionaryArray.from_arrays(
                pa.array([0, None, 1], pa.int8()),
                pa.array(['x', 'y'], pa.large_utf8()),
            ),
            {'type': 'dictionary_utf8', 'index': 'int8'},
            pa.DictionaryArray.from_arrays(
                pa.array([0, None, 1], pa.int8()), pa.array(['x', 'y'])
            ),
        ),
        'dictionary_int64': (
            pa.DictionaryArray.from_arrays(
                pa.array([1, None, 0], pa.int16()),
                pa.array([10, 20], pa.int64()),
            ),
            {'type': 'int64'},
            pa.array([20, None, 10], pa.int64()),
        ),
        'run_end_encoded': (
            # pyarrow 14 builds one from lists alone: int64 run ends.
            pa.RunEndEncodedArray.from_arrays([1, 2, 3], ['a', None, 'b']),
            {'type': 'utf8'},
            pa.array(['a', None, 'b'], pa.utf8()),
        ),
    }


def _field(name: str, column_type: dict[str, Any]) -> dict[str, Any]:
    return {'name': name, 'type': column_type, 'nullable': True}


def test_admitted_types_match_the_expected_canonical_schema() -> None:
    cases = _admitted()
    sent = pa.table({name: case[0] for name, case in cases.items()})
    expect = {
        'expect': 'schema',
        'fields': [_field(name, case[1]) for name, case in cases.items()],
    }
    run, stream = _echoed(sent, expect)
    back = _imported(stream)
    wanted = pa.schema(
        [pa.field(name, case[2].type) for name, case in cases.items()]
    )
    assert back.schema.equals(wanted), (back.schema, wanted)
    for name, (given, _, expected) in cases.items():
        taken = _only_chunk(back, name)
        # The input's nulls, where its form keeps them (a run-end array in
        # its values, a dictionary in its keys), are the expected ones.
        assert [v is None for v in given.to_pylist()] == [
            v is None for v in expected.to_pylist()
        ], name
        assert taken.null_count == expected.null_count == 1, name
        assert taken.is_null().equals(expected.is_null()), name
        assert taken.equals(expected), (name, taken, expected)
    del run


def test_a_cancel_releases_what_the_run_held_and_refuses_what_follows() -> None:
    before = _held()
    sent = pa.table(_canonical())
    run = _native.Run.start(*messages.fetch({'expect': 'discover'}))
    run.next()
    run.reply(messages.fetch_reply(), {1: sent})
    del sent
    assert _held() > before, 'the input was copied or released'
    run.cancel()
    # The reply the run held is dropped with it, every byte at once.
    assert _held() == before
    calls = {
        'next': run.next,
        'cancel': run.cancel,
        'reply': lambda: run.reply(
            messages.fetch_reply(), {1: pa.table({'a': [1]})}
        ),
    }
    for call, made in calls.items():
        with pytest.raises(_native.CoreError) as refused:
            made()
        assert (refused.value.kind, refused.value.code) == (
            'cancelled',
            'RUN_CANCELLED',
        ), call
    del refused
    # What it held is gone: nothing is left to take.
    assert run.take_attachment(1) is None


def test_a_cancel_drops_an_output_the_run_still_held() -> None:
    before = _held()
    sent = pa.table(_canonical())
    run = _native.Run.start(*messages.fetch({'expect': 'discover'}, echo=True))
    run.next()
    run.reply(messages.fetch_reply(), {1: sent})
    run.next()
    del sent
    # Finished, its output untaken: a finished run is not cancelled, and
    # its output is the host's to take until the run goes.
    with pytest.raises(_native.CoreError) as finished:
        run.cancel()
    assert finished.value.code == 'RUN_FINISHED'
    assert _held() > before
    del run
    assert _held() == before


def test_a_cancel_while_another_call_is_in_the_run_is_busy() -> None:
    run = _native.Run.start(
        messages.operation_for_tests('log', hold_ms=2000), {}
    )
    first: list[object] = []
    in_call = threading.Event()

    class Signal(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            in_call.set()

    logger = logging.getLogger('sdm_core.test_ops')
    handler = Signal()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    _native.reset_log_cache()
    try:
        worker = threading.Thread(target=lambda: first.append(run.next()))
        worker.start()
        assert in_call.wait(30), 'the held call never logged'
        with pytest.raises(_native.CoreError) as busy:
            run.cancel()
        assert (busy.value.kind, busy.value.code) == ('protocol', 'RUN_BUSY')
        worker.join(30)
    finally:
        logger.removeHandler(handler)
    assert len(first) == 1
    # The held call finished the run; a cancel then meets that, not busy.
    with pytest.raises(_native.CoreError) as finished:
        run.cancel()
    assert finished.value.code == 'RUN_FINISHED'
