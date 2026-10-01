# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A metadata statement must fall through to the row-based fetch.

Snowflake answers ``SHOW``, ``DESC`` and ``LIST`` with a result set it cannot
render as Arrow or pandas, and raises ``NotSupportedError`` instead of
returning ``None``. Only that one exception means "this strategy does not
apply"; anything else is a real failure and must reach the caller, or a broken
connection would look like an empty result.
"""

from __future__ import annotations

from typing import Any

import pytest

from sdm_connectors.reader import _columnar_fetch


class _NotSupportedError(Exception):
    r"""Named to match what the driver raises, which is how it is recognised.

    The connector matches on the class name rather than the type so it does
    not have to import an optional driver just to catch its exception.
    """


_NotSupportedError.__name__ = 'NotSupportedError'


class _Cursor:
    def __init__(
        self, error: BaseException | None, value: Any = 'rows'
    ) -> None:
        self._error = error
        self._value = value

    def fetch_arrow_all(self) -> Any:
        if self._error is not None:
            raise self._error
        return self._value


def test_a_metadata_statement_falls_through_to_the_row_fetch() -> None:
    cursor = _Cursor(_NotSupportedError('cannot render SHOW as Arrow'))
    assert _columnar_fetch(cursor, 'fetch_arrow_all') is None


def test_a_supported_statement_returns_its_columnar_result() -> None:
    cursor = _Cursor(None, value='arrow-table')
    assert _columnar_fetch(cursor, 'fetch_arrow_all') == 'arrow-table'


@pytest.mark.parametrize(
    'error',
    [
        RuntimeError('connection reset'),
        ValueError('bad cursor state'),
        MemoryError('out of memory'),
    ],
    ids=['runtime', 'value', 'memory'],
)
def test_any_other_failure_propagates(error: BaseException) -> None:
    r"""Swallowing these would turn a broken read into an empty result."""
    cursor = _Cursor(error)
    with pytest.raises(type(error)):
        _columnar_fetch(cursor, 'fetch_arrow_all')


def test_a_none_result_is_passed_through_unchanged() -> None:
    r"""A driver that returns ``None`` already means the same thing."""
    cursor = _Cursor(None, value=None)
    assert _columnar_fetch(cursor, 'fetch_arrow_all') is None
