# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A paged answer is held to its request's schema page by page (T2.1b,
review round 1, P1).

Contract v26 lets a fetch's answer arrive in pages: a ``data`` reply with a
``continuation`` is followed by the same request asked again, until a reply
with none. Each page is a reply like any other, so each must pass the
schema-match check an unpaged reply passes (check (b), decision H16): a
column the request declares non-nullable holding a null is
``PROVIDER_SCHEMA_MISMATCH`` whether it arrives in the first page, a page in
the middle, or the only reply. A valid paged answer still reads whole.
"""

from __future__ import annotations

from typing import Any

import messages
import pyarrow as pa
import pytest
from sdm_client import _native


@pytest.fixture(autouse=True)
def _test_ops_build(native_build: str) -> None:
    assert native_build == 'test-ops', 'these tests need the test-ops build'
    assert _native.TEST_OPS, 'the installed _native has no test operations'


#: A request for one non-nullable int64 column.
EXPECT = {
    'expect': 'schema',
    'fields': [messages.int64_field('a', nullable=False)],
}

#: The provider's pages: a nullable int64 column, as a driver hands one
#: over, whatever the request declares.
NULLABLE = pa.schema([pa.field('a', pa.int64(), nullable=True)])


def _page(values: list[int | None]) -> pa.Table:
    return pa.table({'a': pa.array(values, pa.int64())}, schema=NULLABLE)


def _reply(request_id: int, continuation: str | None) -> bytes:
    outcome: dict[str, Any] = {'outcome': 'data', 'data': 1}
    if continuation is not None:
        outcome['continuation'] = continuation
    return messages.encoded(
        {'message': 'fetch_reply', 'id': request_id, 'outcome': outcome}
    )


def _fetch_step(run: _native.Run) -> dict[str, Any]:
    step = messages.read(run.next()[0])
    assert step['step'] == 'fetch', step
    [request] = step['requests']
    return request


def _answer(
    pages: list[tuple[list[int | None], str | None]],
) -> tuple[_native.Run, Exception | None, list[str | None]]:
    r"""The test-only fetch of ``EXPECT`` answered with ``pages`` in turn,
    each its values and its continuation: the run, the error the core
    raised on a reply (none if every reply was taken), and the continuation
    each request asked with."""
    run = _native.Run.start(*messages.fetch(EXPECT, echo=True))
    asked: list[str | None] = []
    for values, continuation in pages:
        request = _fetch_step(run)
        asked.append(request.get('continuation'))
        try:
            run.reply(_reply(request['id'], continuation), {1: _page(values)})
        except _native.CoreError as error:
            return run, error, asked
    return run, None, asked


def _refused_by_check_b(error: Exception | None, where: str) -> None:
    assert isinstance(error, _native.CoreError), (
        f'{where}: the null in a non-nullable column was taken'
    )
    assert (error.kind, error.code) == (
        'data_source',
        'PROVIDER_SCHEMA_MISMATCH',
    ), where
    fields = messages.read(error.control)['error']['fields']
    assert fields['check'] == {'t': 'utf8', 'v': 'expected'}, where


def test_the_unpaged_reply_with_a_null_is_refused() -> None:
    # The control the pages are held to: one reply, no continuation.
    _, error, _ = _answer([([None], None)])
    _refused_by_check_b(error, 'unpaged')


def test_a_null_in_the_first_page_is_refused() -> None:
    run, error, asked = _answer([([None], 'more'), ([1], None)])
    _refused_by_check_b(error, 'first page')
    assert asked == [None]
    with pytest.raises(_native.CoreError) as finished:
        run.next()
    assert finished.value.code == 'RUN_FINISHED', 'first page: never admitted'


def test_a_null_in_a_middle_page_is_refused() -> None:
    run, error, asked = _answer([([1], 'more'), ([None], 'again'), ([2], None)])
    _refused_by_check_b(error, 'middle page')
    assert asked == [None, 'more']
    with pytest.raises(_native.CoreError) as finished:
        run.next()
    assert finished.value.code == 'RUN_FINISHED', 'middle page: never admitted'


def test_a_valid_paged_answer_reads_whole() -> None:
    # The control: the same pages with no null are asked for in turn, each
    # with the continuation before it, and read as one answer.
    run, error, asked = _answer([([1], 'more'), ([2], 'again'), ([3], None)])
    assert error is None, error
    assert asked == [None, 'more', 'again']
    done = messages.read(run.next()[0])
    assert done['step'] == 'done', done
    assert done['result']['rows'] == 3
    stream = run.take_attachment(done['result']['data'])
    assert isinstance(stream, _native.ArrowStream)
    table = pa.RecordBatchReader._import_from_c_capsule(
        stream.__arrow_c_stream__()
    ).read_all()
    assert table.column('a').to_pylist() == [1, 2, 3]
