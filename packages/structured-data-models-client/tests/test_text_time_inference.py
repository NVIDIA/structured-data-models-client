# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public warehouse text/time inference, including offsets and year bounds."""

from __future__ import annotations

import warnings
from typing import Any

import duckdb
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import Schema, StructuredDataError


def _assert_text(schema: Any, table: str, column: str) -> None:
    r"""``column`` described as text: no refusal, not the time column, no
    timestamp ( part 3, the owner's item 24)."""
    assert schema[table].time_column != column
    described = schema[table].columns[column]
    assert str(described.dtype) == 'string'
    assert str(described.stype) != 'timestamp'


def _offset_events(zone: str) -> Any:
    connection = duckdb.connect()
    connection.execute(
        'CREATE TABLE events (id BIGINT PRIMARY KEY, event_date VARCHAR)'
    )
    connection.execute(
        'INSERT INTO events VALUES '
        + ', '.join(
            f"({i}, '2024-01-0{i}T01:02:03{zone}')" for i in range(1, 5)
        )
    )
    return connection


@pytest.mark.parametrize(
    ('zone', 'core_reads'),
    [
        ('+24:00', False),
        ('+99:00', False),
        ('+2360', False),
        ('-25:00', False),
        ('+01:99', False),
        ('+0160', False),
        ('+23:59', True),
        ('-23:59', True),
        ('+00', True),
        ('-00', True),
        ('Z', True),
    ],
)
def test_supported_offsets_infer_timestamps_and_others_remain_text(
    zone: str, core_reads: bool
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if core_reads:
            core = Schema.from_source(DuckDBSource(_offset_events(zone)))
            assert core['events'].time_column == 'event_date'
            assert (
                str(core['events'].columns['event_date'].stype) == 'timestamp'
            )
        else:

            def check() -> None:
                core = Schema.from_source(DuckDBSource(_offset_events(zone)))
                _assert_text(core, 'events', 'event_date')

            check()


def _text_events(values: list[str]) -> Any:
    connection = duckdb.connect()
    connection.execute(
        'CREATE TABLE events (id BIGINT PRIMARY KEY, event_time VARCHAR)'
    )
    connection.executemany(
        'INSERT INTO events VALUES (?, ?)', list(enumerate(values))
    )
    return connection


@pytest.mark.parametrize(
    'values',
    [
        ['0000-12-31T23:00-02:00'] * 3,
        ['9999-12-31T23:00-02:00'] * 3,
    ],
    ids=['year-0', 'past-9999'],
)
def test_text_the_list_does_not_read_is_text(values: list[str]) -> None:

    def check() -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            core = Schema.from_source(DuckDBSource(_text_events(values)))
        _assert_text(core, 'events', 'event_time')

    check()


@pytest.mark.parametrize(
    ('values', 'rule'),
    [
        (['03/15/99', '15/11/00'], 'text_time_order'),
    ],
)
def test_text_the_list_does_not_decide_is_refused_under_its_rule(
    values: list[str], rule: str
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(StructuredDataError) as refused:
            Schema.from_source(DuckDBSource(_text_events(values)))
    assert refused.value.details['rule'] == rule


def test_a_year_the_list_holds_after_an_offset_reads() -> None:
    values = ['0001-01-01T00:30-01:00', '9999-12-31T22:00-01:00'] * 2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        core = Schema.from_source(DuckDBSource(_text_events(values)))
    assert str(core['events'].columns['event_time'].stype) == 'timestamp'
