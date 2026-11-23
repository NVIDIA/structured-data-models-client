# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A selective query still gathers the context it asks for.

The client draws its context examples from a seeded sample of the entity
table, ten times the context's size. When the query's ``WHERE`` keeps one
entity in sixty, or most entities have no value for a static target, a draw
of every entity keeps a sixtieth, or a twentieth, of what was asked for,
though the table holds many more that qualify. The conditions on the
entity's own row go to the source with the sample, so the draw is of the
entities that may qualify.
"""

from __future__ import annotations

import datetime
import warnings
from typing import Any

import duckdb
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import (
    ContextDraw,
    ContextReport,
    PredictiveQuery,
    Schema,
    StructuredDataError,
)

USERS = 120_000
ASKED = 1_000


def _connection() -> Any:
    connection = duckdb.connect()
    # One user in sixty is older than 76; one in twenty has a label.
    connection.execute(
        f"""
        CREATE TABLE users AS
        SELECT u AS id,
               18 + u % 60 AS age,
               u % 2 = 0 AS churned,
               CASE WHEN u % 20 = 0 THEN u % 3 = 0 END AS labelled,
               TIMESTAMP '2023-01-01' + (u % 60) * INTERVAL 1 DAY AS seen,
               TIMESTAMP '2024-01-01' AS joined
        FROM range({USERS}) t(u)
        """
    )
    connection.execute('ALTER TABLE users ADD PRIMARY KEY (id)')
    return connection


def _request(query: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(_connection()))
        return PredictiveQuery(query, schema).materialize(
            indices=[1, 2],
            anchor_time=pd.Timestamp('2024-03-01'),
            num_context_samples=ASKED,
        )


def _context_rows(query: str) -> int:
    return _request(query).context_rows


@pytest.mark.parametrize(
    'query',
    [
        'PREDICT users.churned FOR EACH users.id WHERE users.age > 76',
        'PREDICT users.churned FOR EACH users.id WHERE users.age IN (77, 200)',
        'PREDICT users.churned FOR EACH users.id WHERE NOT users.age <= 76',
    ],
)
def test_a_selective_filter_on_the_entity_fills_the_context(query: str) -> None:
    # 2,000 users qualify; a draw of every user would keep about 167.
    assert _context_rows(query) == ASKED


def test_a_sparse_static_target_fills_the_context() -> None:
    # 6,000 users have a label; a draw of every user would keep about 500.
    assert _context_rows('PREDICT users.labelled FOR EACH users.id') == ASKED


def _shop() -> Any:
    """120,000 users; one in sixty orders every five days through 2023 and
    2024, the rest never."""
    connection = duckdb.connect()
    connection.execute('CREATE TABLE users (id BIGINT PRIMARY KEY, age BIGINT)')
    connection.execute(
        f'INSERT INTO users SELECT u, 18 + u % 60 FROM range({USERS}) t(u)'
    )
    connection.execute(
        'CREATE TABLE orders (oid BIGINT PRIMARY KEY, '
        'uid BIGINT REFERENCES users(id), placed TIMESTAMP, amount DOUBLE)'
    )
    connection.execute(
        f"""
        INSERT INTO orders
        SELECT row_number() OVER (), u,
               TIMESTAMP '2023-01-01' + k * INTERVAL 5 DAY, (u % 7) + k % 3
        FROM range(0, {USERS}, 60) a(u), range(146) b(k)
        """
    )
    return connection


def _shop_context_rows(query: str) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(_shop()))
        request = PredictiveQuery(query, schema).materialize(
            indices=[0, 60],
            anchor_time=pd.Timestamp('2024-12-01'),
            num_context_samples=ASKED,
        )
    return request.context_rows


@pytest.mark.parametrize(
    'query',
    [
        # 2,000 users have orders in any 30 days; a draw of every user
        # would keep about 167.
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.id '
        'WHERE COUNT(orders.*, -30, 0, days) > 1',
        # An average over no orders is no answer.
        'PREDICT AVG(orders.amount, 0, 30, days) FOR EACH users.id',
    ],
)
def test_a_window_most_entities_have_no_rows_in_fills_the_context(
    query: str,
) -> None:
    assert _shop_context_rows(query) == ASKED


def test_the_report_says_the_filter_went_to_the_source_and_one_draw_did() -> (
    None
):
    request = _request(
        'PREDICT users.churned FOR EACH users.id WHERE users.age > 76'
    )
    assert request.context_report == ContextReport(
        'available',
        (
            ContextDraw(
                # Every one of the 2,000 users older than 76: fewer than
                # the first draw's 10,000 rows, all of them answered.
                draws=1,
                rows_drawn=2_000,
                candidates=2_000,
                kept=2_000,
                asked=ASKED,
                stopped='full',
                filtered_at_source=True,
            ),
        ),
    )


def test_a_condition_the_source_is_not_sent_is_drawn_again_larger() -> None:
    # A time comparison stays with the client, which may read a time
    # otherwise than the source; one user in sixty was seen after 28
    # February 2023.
    request = _request(
        'PREDICT users.churned FOR EACH users.id WHERE users.seen > 2023-02-28'
    )
    assert request.context_rows == ASKED
    [part] = request.context_report.parts
    assert part.draws == 2
    assert part.stopped == 'full'
    assert part.filtered_at_source is False
    assert part.kept >= ASKED
    assert part.rows_drawn > 10_000


def test_a_smaller_context_asked_for_is_not_drawn_again_for_more() -> None:
    # The first draw keeps about one in sixty of its 10,000 rows: short of
    # the 1,000 the route gathers by default, more than the 50 asked for.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(_connection()))
        request = PredictiveQuery(
            'PREDICT users.churned FOR EACH users.id '
            'WHERE users.seen > 2023-02-28',
            schema,
        ).materialize(
            indices=[1, 2],
            anchor_time=pd.Timestamp('2024-03-01'),
            num_context_samples=50,
        )
    assert request.context_rows == 50
    [part] = request.context_report.parts
    assert (part.draws, part.asked, part.stopped) == (1, 50, 'full')
    assert part.kept >= 50


def test_a_source_holding_too_few_says_so() -> None:
    # 2,000 users are older than 76 and 120 of them are 77 with id under
    # 7,200: fewer than asked, and the source has no more.
    request = _request(
        'PREDICT users.churned FOR EACH users.id '
        'WHERE users.age > 76 AND users.id < 7200'
    )
    [part] = request.context_report.parts
    assert part.stopped == 'source_exhausted'
    assert part.kept == request.context_rows == 120


def _seen_after_february(**options: Any) -> Any:
    # One user in sixty was seen after 28 February 2023: the first draw
    # of 10,000 rows keeps about 166 of the 1,000 asked for.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(_connection()))
        return PredictiveQuery(
            'PREDICT users.churned FOR EACH users.id '
            'WHERE users.seen > 2023-02-28',
            schema,
        ).materialize(
            indices=[1, 2],
            anchor_time=pd.Timestamp('2024-03-01'),
            num_context_samples=ASKED,
            **options,
        )


@pytest.mark.parametrize(
    'limit', [0, datetime.timedelta(0)], ids=['seconds', 'timedelta']
)
def test_drawing_stops_at_the_time_limit_with_the_examples_kept(
    limit: Any,
) -> None:
    request = _seen_after_february(context_time_limit=limit)
    [part] = request.context_report.parts
    assert (part.draws, part.stopped) == (1, 'time_limit')
    assert 0 < part.kept < ASKED
    assert request.context_rows == part.kept


def test_a_time_limit_that_is_not_seconds_is_refused() -> None:
    with pytest.raises(StructuredDataError) as raised:
        _seen_after_february(context_time_limit=-1)
    assert 'context_time_limit' in str(raised.value)


def test_the_hosts_clock_past_the_default_five_minutes_stops_drawing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The host reads its clock for each reply; ten minutes pass between
    # every two readings, so the first draw already took longer than the
    # default limit.
    from sdm_client._rust import run

    ticks = iter(range(0, 10**15, 600 * 10**9))
    monkeypatch.setattr(run.time, 'monotonic_ns', lambda: next(ticks))
    request = _seen_after_february()
    [part] = request.context_report.parts
    assert (part.draws, part.stopped) == (1, 'time_limit')
    assert request.context_rows == part.kept < ASKED
