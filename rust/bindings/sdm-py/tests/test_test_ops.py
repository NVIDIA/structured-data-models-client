# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The binding over the test-only operations, which only a ``test-ops``
build has: logging and the GIL (decision H8), an abort, a contained panic,
busy runs, the schema-match checks of T1.7c (decision H16) and Arrow back
out through the PyCapsule interface."""

from __future__ import annotations

import json
import logging
import pathlib
import signal
import subprocess
import sys
import textwrap
import threading
from typing import Any

import messages
import pyarrow as pa
import pytest
from sdm_client import _native

CANARY = 'CANARY-91c4'
HERE = pathlib.Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def _test_ops_build(native_build: str) -> None:
    assert native_build == 'test-ops', 'these tests need the test-ops build'
    assert _native.TEST_OPS, 'the installed _native has no test operations'


def _child(
    script: str, timeout: float = 120
) -> subprocess.CompletedProcess[str]:
    """Runs `script` in a fresh interpreter, beside `messages.py`."""
    return subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=HERE,
    )


#: The deadlock regression of decision H8, in a fresh interpreter so the
#: logging bridge installs at this run's first call. While the core holds
#: one call for a second with the GIL released, another thread logs, each
#: time holding the handler's lock that every record takes. A binding that
#: kept the GIL would stop that thread for the whole call, and one that
#: blocked on the GIL while logging would hang: `faulthandler` then dumps
#: every thread and exits, and the parent's timeout is the last resort.
_LONG_CALL = """
    import faulthandler, json, logging, threading, time
    faulthandler.dump_traceback_later(60, exit=True)

    import messages
    from sdm_client import _native

    progress = [0]
    records = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(
                (record.name, record.levelno, record.getMessage(), progress[0])
            )

    handler = Collect()
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    other = logging.getLogger('other')
    other.propagate = False
    other.addHandler(logging.NullHandler())
    stop = threading.Event()

    def work():
        while not stop.is_set():
            with handler.lock:
                other.info('progress')
                progress[0] += 1
            time.sleep(0.001)

    worker = threading.Thread(target=work)
    worker.start()
    run = _native.Run.start(messages.operation_for_tests('log', hold_ms=1000), {})
    control, _ = run.next()
    after = progress[0]
    stop.set()
    worker.join()
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps({
        'done': json.loads(control)['step'],
        'records': records,
        'after': after,
    }))
"""


def test_another_thread_runs_and_logs_while_the_core_holds_a_call() -> None:
    result = _child(_LONG_CALL)
    assert result.returncode == 0, result.stderr
    found: dict[str, Any] = json.loads(result.stdout)
    assert found['done'] == 'done'
    ours = [r for r in found['records'] if r[0].startswith('sdm_core')]
    # Exactly one record, at the event's level and target.
    assert [r[:3] for r in ours] == [
        ['sdm_core.test_ops', 20, 'a test-only event']
    ]
    # The event comes first, then a second's hold: the other thread kept
    # going, under the handler's lock, for most of it.
    progress_during_hold = found['after'] - ours[0][3]
    assert progress_during_hold > 100, progress_during_hold


def test_a_second_call_while_one_is_in_the_run_is_refused_as_busy() -> None:
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
        with pytest.raises(_native.CoreError) as raised:
            run.next()
        assert (raised.value.kind, raised.value.code) == (
            'protocol',
            'RUN_BUSY',
        )
        worker.join(30)
    finally:
        logger.removeHandler(handler)
    assert len(first) == 1


def test_a_forced_abort_is_reported_and_never_recovered() -> None:
    result = _child(
        """
        import messages
        from sdm_client import _native
        run = _native.Run.start(messages.operation_for_tests('abort'), {})
        try:
            run.next()
        except BaseException as caught:
            print('recovered', type(caught).__name__)
        print('survived')
        """
    )
    assert result.returncode == -signal.SIGABRT, (
        result.returncode,
        result.stderr,
    )
    assert 'recovered' not in result.stdout
    assert 'survived' not in result.stdout


def test_a_panic_is_contained_and_quotes_nothing() -> None:
    run = _native.Run.start(
        messages.operation_for_tests('panic', at='body', payload=CANARY), {}
    )
    with pytest.raises(_native.CoreError) as raised:
        run.next()
    error = raised.value
    assert (error.kind, error.code) == ('internal', 'INTERNAL_ERROR')
    fields = messages.read(error.control)['error']['fields']
    assert fields['panic'] == {'t': 'bool', 'v': True}
    assert CANARY not in f'{error} {error!r} {error.control!r}'
    with pytest.raises(_native.CoreError) as again:
        run.next()
    assert again.value.code == 'RUN_POISONED'


def test_an_error_past_the_cap_is_raised_whole_from_its_attachment() -> None:
    error = {
        'kind': 'data_source',
        'code': 'PROVIDER_ERROR',
        'message': 'x' * 70_000,
        'retryable': True,
    }
    run = _native.Run.start(
        messages.operation_for_tests('fail', error=1),
        {1: messages.encoded(error)},
    )
    with pytest.raises(_native.CoreError) as raised:
        run.next()
    found = raised.value
    assert (found.kind, found.code) == ('data_source', 'PROVIDER_ERROR')
    attached = messages.read(found.control)
    assert attached['message'] == 'attached'
    whole = found.attachments[attached['attachment']]
    assert isinstance(whole, bytes)
    assert messages.read(whole)['error'] == error


def _fetching(expect: dict[str, Any], echo: bool = False) -> _native.Run:
    run = _native.Run.start(*messages.fetch(expect, echo))
    step = messages.read(run.next()[0])
    assert step['step'] == 'fetch'
    return run


def _ints(*names: str, nullable: bool = True) -> pa.Table:
    schema = pa.schema(
        [pa.field(name, pa.int64(), nullable=nullable) for name in names]
    )
    return pa.table({name: [1, 2] for name in names}, schema=schema)


def _refused(
    check: str, expect: dict[str, Any], data: pa.Table, fault: str
) -> None:
    run = _fetching(expect)
    with pytest.raises(_native.CoreError) as raised:
        run.reply(messages.fetch_reply(), {1: data})
    error = raised.value
    assert (error.kind, error.code) == (
        'data_source',
        'PROVIDER_SCHEMA_MISMATCH',
    ), fault
    fields = messages.read(error.control)['error']['fields']
    assert fields['check'] == {'t': 'utf8', 'v': check}, fault
    with pytest.raises(_native.CoreError) as finished:
        run.next()
    assert finished.value.code == 'RUN_FINISHED', f'{fault}: never admitted'


def _expected(*fields: dict[str, Any]) -> dict[str, Any]:
    return {'expect': 'schema', 'fields': list(fields)}


def test_data_that_does_not_match_its_request_is_refused_by_check_b() -> None:
    a, b = messages.int64_field('a'), messages.int64_field('b')
    for names, fault in [
        (('a',), 'lacks a column'),
        (('a', 'b', 'c'), 'adds a column'),
        (('a', 'z'), 'renames a column'),
        (('b', 'a'), 'orders the columns otherwise'),
    ]:
        _refused('expected', _expected(a, b), _ints(*names), fault)
    _refused(
        'expected',
        _expected(a),
        pa.table({'a': pa.array([1], pa.int32())}),
        'another type',
    )
    _refused(
        'expected',
        _expected(messages.int64_field('a', metadata={'role': 'key'})),
        _ints('a'),
        'required metadata',
    )
    _refused(
        'expected',
        _expected(messages.int64_field('a', nullable=False)),
        pa.table({'a': [1, None]}),
        'nulls the request does not take',
    )


def test_a_stream_that_breaks_its_own_schema_is_refused_before_admission() -> (
    None
):
    # Through the binding an Arrow attachment is one stream, whose batches
    # carry its schema, so check (a) can fail only on nulls in a column the
    # stream declares without them; Arrow's import refuses those first, so
    # nothing is admitted and the run still awaits its fetch.
    declared = pa.schema([pa.field('a', pa.int64(), nullable=False)])
    run = _fetching({'expect': 'discover'})
    with pytest.raises(
        ValueError, match='attachment 1: the Arrow stream failed'
    ):
        run.reply(
            messages.fetch_reply(),
            {1: pa.table({'a': [1, None]}, schema=declared)},
        )
    run.reply(
        messages.fetch_reply(), {1: pa.table({'a': [1, 2]}, schema=declared)}
    )
    assert messages.read(run.next()[0])['result'] == {'op': 'test', 'rows': 2}


def test_the_data_asked_for_is_taken_and_comes_back_out() -> None:
    a, b = messages.int64_field('a'), messages.int64_field('b')
    sent = pa.table(
        {'a': [1, 2, None], 'b': [3, 4, 5]},
        schema=pa.schema([('a', pa.int64()), ('b', pa.int64())]),
    )
    run = _fetching(_expected(a, b), echo=True)
    run.reply(messages.fetch_reply(), {1: sent})
    done = messages.read(run.next()[0])
    assert done['result'] == {'op': 'test', 'rows': 3, 'data': 1}
    stream = run.take_attachment(1)
    assert isinstance(stream, _native.ArrowStream)
    back = pa.RecordBatchReader._import_from_c_capsule(
        stream.__arrow_c_stream__()
    ).read_all()
    assert back.equals(sent)
    # Exported again, the same data; and its schema on its own.
    again = pa.RecordBatchReader._import_from_c_capsule(
        stream.__arrow_c_stream__()
    ).read_all()
    assert again.equals(sent)
    assert pa.Schema._import_from_c_capsule(stream.__arrow_c_schema__()).equals(
        sent.schema
    )
    # A capsule never read releases its stream when it goes.
    stream.__arrow_c_stream__()
    # The attachment was taken once; the run holds it no more.
    assert run.take_attachment(1) is None


def test_a_stream_is_exported_only_in_its_own_schema() -> None:
    run = _fetching({'expect': 'discover'}, echo=True)
    run.reply(messages.fetch_reply(), {1: _ints('a')})
    run.next()
    stream = run.take_attachment(1)
    assert isinstance(stream, _native.ArrowStream)
    with pytest.raises(ValueError, match='none can be requested'):
        stream.__arrow_c_stream__(
            pa.schema([('a', pa.int32())]).__arrow_c_schema__()
        )
