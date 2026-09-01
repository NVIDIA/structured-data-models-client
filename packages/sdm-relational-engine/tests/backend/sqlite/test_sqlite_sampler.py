# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

pytest.importorskip(
    'adbc_driver_sqlite', reason="'sqlite' extension not installed"
)

from sdm_relational_engine.rfm.backend.sqlite import SQLiteSampler


class _Cursor:
    def __init__(self, row: Any, fail: bool) -> None:
        self._row = row
        self._fail = fail

    def execute(self, sql: str) -> None:
        if self._fail:
            raise RuntimeError('AdbcConnectionInit not called')

    def fetchone(self) -> Any:
        return self._row


class _Connection:
    def __init__(self, row: Any, fail: bool = False) -> None:
        self._cursor = _Cursor(row, fail)

    @contextmanager
    def cursor(self) -> Iterator[_Cursor]:
        yield self._cursor


class _StubSampler(SQLiteSampler):
    r"""Overrides the three lookups ``_time_separator`` reads."""

    @property
    def time_column_dict(self) -> dict[str, str]:
        return {'t': 'ts'}

    @property
    def table_column_ref_dict(self) -> dict[str, dict[str, str]]:
        return {'t': {'ts': '"ts"'}}

    @property
    def source_name_dict(self) -> dict[str, str]:
        return {'t': '"t"'}


def _sampler(row: Any, fail: bool = False) -> SQLiteSampler:
    r"""A sampler wired to a stub connection, with no driver involved.

    ``_time_separator`` reads one stored value and decides how to spell a
    window bound. Driving it through a real ADBC connection would test that
    driver's initialisation order rather than the decision, so the connection
    is stubbed and only the decision is asserted. The end-to-end behaviour is
    covered by the live suites.
    """
    sampler = _StubSampler.__new__(_StubSampler)
    sampler._connection = _Connection(row, fail)
    # `__new__` skips `__init__`, so the per-table memo it installs has to be
    # supplied here. It is per-instance rather than a `functools.cache` on the
    # method, which would key on `self` and keep every sampler -- and its
    # graph and its open connection -- alive for the life of the process.
    sampler._time_separator_dict = {}
    return sampler


@pytest.mark.parametrize(
    ('stored', 'expected'),
    [
        ('2024-05-01 00:00:00', ' '),
        ('2024-05-01T00:00:00', 'T'),
        ('2024-05-01T00:00:00.123456', 'T'),
    ],
)
def test_a_window_bound_is_spelled_the_way_the_column_is(
    stored: str, expected: str
) -> None:
    r"""An ISO-8601 timestamp column must not lose its window-edge rows.

    SQLite has no datetime type, so a window bound is compared as text. Both
    bounds were written ``YYYY-MM-DD HH:MM:SS`` unconditionally, and
    ``'...T00:00:00' <= '... 00:00:00'`` is false for the same instant because
    ``T`` sorts after a space. A column written by an ISO-8601 writer therefore
    dropped every row landing exactly on a window edge -- no error, just a
    quietly smaller answer.
    """
    assert _sampler((stored,))._time_separator('t') == expected


@pytest.mark.parametrize(
    'row',
    [None, (None,), (12345,), ('short',)],
    ids=['no-row', 'null', 'not-text', 'too-short'],
)
def test_an_unreadable_value_falls_back_to_the_space_form(row: Any) -> None:
    r"""Anything undecidable keeps the spelling this code always assumed."""
    assert _sampler(row)._time_separator('t') == ' '


def test_a_failing_probe_falls_back_to_the_space_form() -> None:
    r"""A probe that cannot run must not take the prediction down with it."""
    sampler = _sampler(('2024-05-01T00:00:00',), fail=True)
    assert sampler._time_separator('t') == ' '


def test_a_table_without_a_time_column_falls_back() -> None:
    assert _sampler(('2024-05-01T00:00:00',))._time_separator('other') == ' '
