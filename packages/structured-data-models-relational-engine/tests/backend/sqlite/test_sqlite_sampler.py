# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pandas as pd
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
    r"""Overrides the three lookups ``_time_spelling`` reads."""

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

    ``_time_spelling`` reads one stored value and decides how to spell a
    window bound. Driving it through a real ADBC connection would test that
    driver's initialisation order rather than the decision, so the connection
    is stubbed and only the decision is asserted. The end-to-end behaviour is
    covered by the client's tests over a real SQLite file.
    """
    sampler = _StubSampler.__new__(_StubSampler)
    sampler._connection = _Connection(row, fail)
    # `__new__` skips `__init__`, so the per-table memo it installs has to be
    # supplied here. It is per-instance rather than a `functools.cache` on the
    # method, which would key on `self` and keep every sampler, and its
    # graph and its open connection, alive for the life of the process.
    sampler._time_spelling_dict = {}
    return sampler


_INSTANT = pd.Series(pd.to_datetime(['2024-12-30 14:00:00.250000']))


@pytest.mark.parametrize(
    ('stored', 'spelled'),
    [
        ('2024-05-01 00:00:00', '2024-12-30 14:00:00'),
        ('2024-05-01T00:00:00', '2024-12-30T14:00:00'),
        ('2024-05-01T00:00:00.000000', '2024-12-30T14:00:00.250000'),
        ('2024-05-01 00:00:00.000', '2024-12-30 14:00:00.250'),
        ('2024-05-01T00:00:00.000000000', '2024-12-30T14:00:00.250000000'),
        ('2024-05-01T00:00:00Z', '2024-12-30T14:00:00Z'),
        ('2024-05-01T00:00:00.000+00:00', '2024-12-30T14:00:00.250+00:00'),
        ('2024-05-01T00:00:00+05:30', '2024-12-30T14:00:00'),
    ],
)
def test_a_window_bound_is_spelled_the_way_the_column_is(
    stored: str, spelled: str
) -> None:
    r"""A window edge must compare equal to a stored value at that instant.

    SQLite has no datetime type, so a window bound is compared as text. The
    bounds were written ``YYYY-MM-DD HH:MM:SS`` whatever the column held, and
    ``'...T00:00:00' <= '... 00:00:00'`` is false for the same instant, as is
    ``'...14:00:00.000000' <= '...14:00:00'``: a shorter string with the same
    prefix sorts first. A column written by an ISO-8601 writer, or by pandas
    through ADBC, therefore dropped every row landing exactly on a window's
    end and counted every row landing exactly on its anchor. There was no
    error, just a different answer. A non-UTC offset spells another instant,
    so it is not copied onto a UTC bound.
    """
    got = _sampler((stored,))._time_spelling('t').spell(_INSTANT)
    assert got.tolist() == [spelled]


@pytest.mark.parametrize(
    'row',
    [None, (None,), (12345,), ('short',)],
    ids=['no-row', 'null', 'not-text', 'too-short'],
)
def test_an_unreadable_value_falls_back_to_the_space_form(row: Any) -> None:
    r"""Anything undecidable keeps the spelling this code always assumed."""
    assert _sampler(row)._time_spelling('t').spell(_INSTANT).tolist() == [
        '2024-12-30 14:00:00'
    ]


def test_a_failing_probe_falls_back_to_the_space_form() -> None:
    r"""A probe that cannot run must not take the prediction down with it."""
    sampler = _sampler(('2024-05-01T00:00:00',), fail=True)
    assert sampler._time_spelling('t').separator == ' '


def test_a_table_without_a_time_column_falls_back() -> None:
    spelling = _sampler(('2024-05-01T00:00:00',))._time_spelling('other')
    assert spelling.separator == ' ' and spelling.digits == 0


@pytest.mark.parametrize(
    ('stored', 'spelled'),
    [
        ('2024-05-01 00:00:00.000', ['14:00:00.123', '14:00:00.123']),
        ('2024-05-01 00:00:00.000000', ['14:00:00.123456', '14:00:00.123456']),
        (
            '2024-05-01 00:00:00.000000000',
            ['14:00:00.123456000', '14:00:00.123456007'],
        ),
    ],
)
def test_a_missing_time_leaves_the_others_spelled_as_they_are(
    stored: str, spelled: list[str]
) -> None:
    r"""A NaT among the times, as an entity's own time column with NULLs
    gives, turns the nanoseconds to floats, which printed as ``0.0``.
    """
    times = pd.Series(
        pd.to_datetime(
            [
                '2024-12-30 14:00:00.123456',
                '2024-12-30 14:00:00.123456007',
                None,
            ]
        )
    )
    got = _sampler((stored,))._time_spelling('t').spell(times)
    assert got[:2].tolist() == [f'2024-12-30 {time}' for time in spelled]
    assert pd.isna(got[2])


def test_a_window_edge_keeps_its_nanoseconds(tmp_path: Any) -> None:
    r"""``%f`` stops at microseconds, so a bound on a nanosecond edge was
    spelled ``.123456000``, which sorts before the stored ``.123456789``: the
    row on the end of the window was dropped and the one on its start kept.
    """
    import adbc_driver_sqlite.dbapi as adbc
    from sdm_relational_engine.rfm import Graph

    connection = adbc.connect(str(tmp_path / 'nanos.db'))
    with connection.cursor() as cursor:
        cursor.execute(
            'CREATE TABLE events (event_id INTEGER PRIMARY KEY, '
            'user_id INTEGER, ts TEXT)'
        )
        cursor.execute(
            "INSERT INTO events VALUES (1, 1, '2024-01-01T00:00:00.123456789'),"
            " (2, 1, '2024-01-02T00:00:00.123456789')"
        )
    connection.commit()
    graph = Graph.from_sqlite(
        connection,
        tables=[dict(name='events', primary_key='event_id', time_column='ts')],
        verbose=False,
    )
    sampler = SQLiteSampler(graph, verbose=False)

    rows, _ = sampler._by_time(
        'events',
        'user_id',
        pd.Series([1]),
        anchor_time=pd.Series([pd.Timestamp('2024-01-02 00:00:00.123456789')]),
        min_offset=pd.DateOffset(days=-1),
        max_offset=pd.DateOffset(days=0),
        columns={'event_id', 'ts'},
    )
    connection.close()

    assert rows['event_id'].tolist() == [2]
