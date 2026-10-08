# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P55.caps, PostgreSQL: the unkeyed inference sample reads a block sample.

``DESIGN-scan-caps.md`` A1-A5 and A8 for PostgreSQL. The thresholds are the
class attributes ``CONTRACT.md`` names; the checks shrink them (256 and 32
blocks) so that a 300,000-row table, about 2,000 blocks, is "large":

* A1: its sample is a ``Sample Scan`` reading at most 128 blocks, found by
  ``EXPLAIN (ANALYZE, BUFFERS)`` of the statement the engine sent, on a table
  never analyzed;
* A2: two tables over separate connections sample the same rows, and the
  expression sample lines up with the source sample row for row;
* A3: a bloated table (most rows deleted, no vacuum) falls back to today's
  statement and still samples 1,000 rows;
* A4: a partitioned table takes the prefilter; a view never does; a keyed
  table sends exactly today's statements, and an unkeyed one sends one probe
  more. (A materialized view is left out: ``information_schema.columns``
  doesn't list one, so the engine can't read one today.)
* A5: below the threshold the sample statement is byte for byte today's;
* A8: the fraction is the power of two at or above ``target / blocks``, the
  prefilter applies only when that is at most 1/8 and the table reaches the
  threshold, and its literal is ``repr(100 * fraction)``.

Today's statements come from running ``BASE``'s code in a child process.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from _fixes import (
    assert_imports_base,
    base_env,
    flat,
    postgres_dsn,
    run_child,
    statements,
)

MIN_BLOCKS = '_SAMPLE_PREFILTER_MIN_BLOCKS'
TARGET_BLOCKS = '_SAMPLE_PREFILTER_TARGET_BLOCKS'
ROWS = 300_000


def _table_class() -> Any:
    from sdm_relational_engine.rfm.backend.postgres import PostgresTable

    return PostgresTable


@pytest.fixture(scope='module')
def database() -> Any:
    import psycopg

    dsn = postgres_dsn()
    schema = f'fix_caps_{uuid.uuid4().hex[:10]}'
    connection = psycopg.connect(dsn, autocommit=True)
    connection.execute(f'CREATE SCHEMA "{schema}"')
    fill = f"SELECT n, 'm' || n, n / 7.0 FROM generate_series(1, {ROWS}) AS n"
    connection.execute(
        f'CREATE TABLE "{schema}".big (n BIGINT, m TEXT, f DOUBLE PRECISION)'
    )
    connection.execute(f'INSERT INTO "{schema}".big {fill}')
    connection.execute(
        f'CREATE TABLE "{schema}".keyed '
        '(n BIGINT PRIMARY KEY, m TEXT, f DOUBLE PRECISION)'
    )
    connection.execute(f'INSERT INTO "{schema}".keyed {fill}')
    connection.execute(
        f'CREATE TABLE "{schema}".parts (n BIGINT, m TEXT, f DOUBLE PRECISION) '
        'PARTITION BY RANGE (n)'
    )
    for index, (low, high) in enumerate(
        ((0, 100_001), (100_001, 200_001), (200_001, 400_000))
    ):
        connection.execute(
            f'CREATE TABLE "{schema}".parts_{index} PARTITION OF '
            f'"{schema}".parts FOR VALUES FROM ({low}) TO ({high})'
        )
    connection.execute(f'INSERT INTO "{schema}".parts {fill}')
    connection.execute(
        f'CREATE VIEW "{schema}".big_view AS SELECT * FROM "{schema}".big'
    )
    connection.execute(
        f'CREATE TABLE "{schema}".small (n BIGINT, m TEXT, f DOUBLE PRECISION)'
    )
    connection.execute(
        f'INSERT INTO "{schema}".small '
        "SELECT n, 'm' || n, n / 7.0 FROM generate_series(1, 5000) AS n"
    )
    # Most rows deleted, nothing vacuumed: the blocks stay, the rows don't.
    connection.execute(
        f'CREATE TABLE "{schema}".bloated '
        '(n BIGINT, m TEXT, f DOUBLE PRECISION) '
        'WITH (autovacuum_enabled = false)'
    )
    connection.execute(f'INSERT INTO "{schema}".bloated {fill}')
    connection.execute(f'DELETE FROM "{schema}".bloated WHERE n % 150 <> 0')
    try:
        yield dsn, schema
    finally:
        connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
        connection.close()


def _blocks(dsn: str, schema: str, table: str) -> int:
    import psycopg

    connection = psycopg.connect(dsn, autocommit=True)
    try:
        row = connection.execute(
            "SELECT CASE WHEN c.relkind = 'p' THEN (SELECT "
            'COALESCE(SUM(pg_relation_size(p.relid)), 0)::bigint '
            'FROM pg_partition_tree(c.oid) AS p) '
            'ELSE pg_relation_size(c.oid) END '
            "/ current_setting('block_size')::bigint "
            'FROM pg_catalog.pg_class AS c WHERE c.oid = to_regclass(%s)',
            [f'"{schema}"."{table}"'],
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    return int(row[0])


def _shrink(monkeypatch: pytest.MonkeyPatch, minimum: int, target: int) -> None:
    table = _table_class()
    monkeypatch.setattr(table, MIN_BLOCKS, minimum)
    monkeypatch.setattr(table, TARGET_BLOCKS, target)


def _shrink_if_present(
    monkeypatch: pytest.MonkeyPatch, minimum: int, target: int
) -> None:
    table = _table_class()
    monkeypatch.setattr(table, MIN_BLOCKS, minimum, raising=False)
    monkeypatch.setattr(table, TARGET_BLOCKS, target, raising=False)


def _sample(
    dsn: str, schema: str, table: str, **kwargs: Any
) -> tuple[Any, list[tuple[str, Any]], Any]:
    r"""A table's source sample, every statement building it sent, and the
    table."""
    import psycopg

    connection = psycopg.connect(dsn, autocommit=True)
    try:
        with statements() as log:
            built = _table_class()(connection, table, schema=schema, **kwargs)
            sample = built._source_sample_df
        return sample, list(log), built
    finally:
        connection.close()


def _samples(log: list[tuple[str, Any]]) -> list[str]:
    return [sql for sql, _ in log if '__kumo_order_' in sql]


def _extra(log: list[tuple[str, Any]], today: list[list[Any]]) -> list[str]:
    r"""The statements other than samples that today's code doesn't send:
    the probe of the table's size."""
    left = [json.dumps(entry) for entry in today]
    out = []
    for sql, params in log:
        if '__kumo_order_' in sql:
            continue
        key = json.dumps([sql, params])
        if key in left:
            left.remove(key)
        else:
            out.append(sql)
    return out


@pytest.fixture(scope='module')
def base_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp('base')
    assert_imports_base(base_env(root), root)
    return root


_base_cache: dict[str, Any] = {}


def _base(base_root: Path, dsn: str, schema: str, table: str) -> Any:
    if table not in _base_cache:
        _base_cache[table] = run_child(
            'postgres-sample', dsn, schema, table, env=base_env(base_root)
        )
    return _base_cache[table]


def _base_sample_sql(base_root: Path, dsn: str, schema: str, table: str) -> str:
    found = [
        sql
        for sql, _ in _base(base_root, dsn, schema, table)['statements']
        if '__kumo_order_' in sql
    ]
    assert len(found) == 1, found
    return found[0]


def _expected_literal(blocks: int, target: int) -> str:
    r"""``repr(100 * f)``, ``f`` the least power of two at or above
    ``target / blocks``."""
    ratio = Fraction(target, blocks)
    exponent = math.ceil(math.log2(ratio)) if ratio > 0 else 0
    while Fraction(2) ** (exponent - 1) >= ratio:
        exponent -= 1
    while Fraction(2) ** exponent < ratio:
        exponent += 1
    return repr(100 * 2.0**exponent)


def _with_prefilter(sql: str, source: str, literal: str) -> str:
    return sql.replace(
        f'FROM {source} ',
        f'FROM {source} TABLESAMPLE SYSTEM ({literal}) REPEATABLE (0) ',
        1,
    )


def _source(schema: str, table: str) -> str:
    return f'"{schema}"."{table}"'


# The defaults ################################################################


def test_the_thresholds_are_the_designs() -> None:
    table = _table_class()
    assert getattr(table, MIN_BLOCKS) == 16_384
    assert getattr(table, TARGET_BLOCKS) == 2_048


# A1 ##########################################################################


def test_a1_a_large_unkeyed_table_reads_a_block_sample(
    database: Any, base_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import psycopg

    dsn, schema = database
    _shrink(monkeypatch, 256, 32)
    blocks = _blocks(dsn, schema, 'big')
    assert blocks >= 1024, blocks
    sample, log, _ = _sample(dsn, schema, 'big')
    assert len(sample) == 1000
    sent = _samples(log)
    assert len(sent) == 1, sent
    literal = _expected_literal(blocks, 32)
    today = _base_sample_sql(base_root, dsn, schema, 'big')
    assert sent[0] == _with_prefilter(today, _source(schema, 'big'), literal)
    probes = _extra(log, _base(base_root, dsn, schema, 'big')['statements'])
    assert len(probes) == 1, probes

    connection = psycopg.connect(dsn, autocommit=True)
    try:
        row = connection.execute(
            f'EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sent[0]}'
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    plan = row[0] if isinstance(row[0], list) else json.loads(row[0])
    nodes = list(_nodes(plan[0]['Plan']))
    kinds = [node['Node Type'] for node in nodes]
    assert 'Seq Scan' not in kinds, kinds
    scans = [node for node in nodes if node['Node Type'] == 'Sample Scan']
    assert scans, kinds
    read = sum(
        node.get('Shared Hit Blocks', 0) + node.get('Shared Read Blocks', 0)
        for node in scans
    )
    assert 0 < read <= 128, f'the sample read {read} of {blocks} blocks'


def _nodes(node: dict[str, Any]) -> Any:
    yield node
    for child in node.get('Plans', []):
        yield from _nodes(child)


# A2 ##########################################################################


def test_a2_the_block_sample_repeats_and_lines_up(
    database: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import psycopg

    dsn, schema = database
    _shrink(monkeypatch, 256, 32)
    samples = []
    for _ in range(2):
        connection = psycopg.connect(dsn, autocommit=True)
        try:
            with statements() as log:
                table = _table_class()(
                    connection,
                    'big',
                    schema=schema,
                    columns=[
                        'n',
                        dict(name='m', expr='CAST(n AS TEXT)', dtype='string'),
                        dict(name='double', expr='2 * n', dtype='int'),
                    ],
                    primary_key=None,
                )
                source = table._source_sample_df
                expr = table._get_expr_sample_df(
                    [table['m'], table['double']]
                ).reset_index(drop=True)
        finally:
            connection.close()
        sent = _samples(log)
        assert len(sent) >= 2, sent
        assert all('TABLESAMPLE SYSTEM' in sql for sql in sent), sent
        assert len(source) == len(expr) == 1000
        assert expr['m'].astype(str).tolist() == [str(n) for n in source['n']]
        assert (expr['double'] == 2 * source['n']).all()
        samples.append(source['n'].tolist())
    assert samples[0] == samples[1]
    assert len(set(samples[0])) == 1000


# A3 ##########################################################################


def test_a3_a_bloated_table_falls_back_to_the_full_read(
    database: Any, base_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import psycopg

    dsn, schema = database
    _shrink(monkeypatch, 256, 32)
    today = _base_sample_sql(base_root, dsn, schema, 'bloated')
    connection = psycopg.connect(dsn, autocommit=True)
    try:
        with statements() as log:
            table = _table_class()(
                connection,
                'bloated',
                schema=schema,
                columns=[
                    'n',
                    'm',
                    'f',
                    dict(name='half', expr='n / 2', dtype='int'),
                ],
                primary_key=None,
            )
            source = table._source_sample_df
            expr = table._get_expr_sample_df([table['half']]).reset_index(
                drop=True
            )
    finally:
        connection.close()
    sent = _samples(log)
    # The first read tries the prefilter; it holds too few rows, so it is
    # read again in full, and every later read, of columns or of
    # expressions, reads what that settled on.
    assert 'TABLESAMPLE SYSTEM' in sent[0], sent
    assert all('TABLESAMPLE' not in sql for sql in sent[1:]), sent
    assert sent.count(today) == 1, sent
    assert len(source) == 1000
    assert (expr['half'] == source['n'] // 2).all()


# A4 ##########################################################################


def test_a4_a_partitioned_table_takes_the_prefilter(
    database: Any, base_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dsn, schema = database
    table = 'parts'
    _shrink(monkeypatch, 256, 32)
    blocks = _blocks(dsn, schema, table)
    sample, log, _ = _sample(dsn, schema, table)
    assert len(sample) == 1000
    sent = _samples(log)
    assert len(sent) == 1, sent
    literal = _expected_literal(blocks, 32)
    today = _base_sample_sql(base_root, dsn, schema, table)
    assert sent[0] == _with_prefilter(today, _source(schema, table), literal)


def test_a4_a_view_is_never_prefiltered(
    database: Any, base_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dsn, schema = database
    _shrink_if_present(monkeypatch, 256, 32)
    sample, log, _ = _sample(dsn, schema, 'big_view')
    assert len(sample) == 1000
    assert _samples(log) == [
        _base_sample_sql(base_root, dsn, schema, 'big_view')
    ]


def test_a4_a_keyed_table_sends_todays_statements(
    database: Any, base_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dsn, schema = database
    _shrink_if_present(monkeypatch, 256, 32)
    _, log, _ = _sample(dsn, schema, 'keyed')
    today = _base(base_root, dsn, schema, 'keyed')['statements']
    assert [[sql, params] for sql, params in log] == today


def test_a4_an_unkeyed_table_is_probed_once(
    database: Any, base_root: Path
) -> None:
    dsn, schema = database
    _, log, _ = _sample(dsn, schema, 'small')
    probes = _extra(log, _base(base_root, dsn, schema, 'small')['statements'])
    assert len(probes) == 1, [flat(sql) for sql in probes]
    assert re.search(r'pg_relation_size|relpages', probes[0]), probes[0]


# A5 ##########################################################################


def test_a5_below_the_threshold_the_statement_is_todays(
    database: Any, base_root: Path
) -> None:
    dsn, schema = database
    sample, log, _ = _sample(dsn, schema, 'small')
    assert len(sample) == 1000
    assert _samples(log) == [_base_sample_sql(base_root, dsn, schema, 'small')]
    today = _base(base_root, dsn, schema, 'small')['statements']
    probes = _extra(log, today)
    assert len(probes) <= 1, probes
    assert [[sql, params] for sql, params in log if sql not in probes] == today


def test_a5_the_real_thresholds_leave_a_large_table_alone(
    database: Any, base_root: Path
) -> None:
    r"""300,000 rows are about 2,000 blocks, far below 16,384."""
    dsn, schema = database
    _, log, _ = _sample(dsn, schema, 'big')
    assert _samples(log) == [_base_sample_sql(base_root, dsn, schema, 'big')]


# A8 ##########################################################################


@pytest.mark.parametrize('divisor', (8, 9, 15, 16, 17, 33, 64, 100, 1000))
def test_a8_the_fraction_is_the_power_of_two_at_or_above(
    database: Any,
    base_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    divisor: int,
) -> None:
    dsn, schema = database
    blocks = _blocks(dsn, schema, 'big')
    target = max(1, blocks // divisor)
    _shrink(monkeypatch, 8 * target, target)
    _, log, _ = _sample(dsn, schema, 'big')
    sent = _samples(log)
    literal = _expected_literal(blocks, target)
    today = _base_sample_sql(base_root, dsn, schema, 'big')
    assert sent[0] == _with_prefilter(today, _source(schema, 'big'), literal)
    assert float(literal) <= 12.5


@pytest.mark.parametrize('divisor', (2, 3, 7))
def test_a8_no_prefilter_above_one_eighth(
    database: Any,
    base_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    divisor: int,
) -> None:
    dsn, schema = database
    blocks = _blocks(dsn, schema, 'big')
    target = blocks // divisor
    _shrink(monkeypatch, 1, target)
    _, log, _ = _sample(dsn, schema, 'big')
    assert _samples(log) == [_base_sample_sql(base_root, dsn, schema, 'big')]


def test_a8_the_threshold_is_inclusive(
    database: Any, base_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dsn, schema = database
    blocks = _blocks(dsn, schema, 'big')
    target = blocks // 16
    today = _base_sample_sql(base_root, dsn, schema, 'big')

    _shrink(monkeypatch, blocks, target)
    _, log, _ = _sample(dsn, schema, 'big')
    assert _samples(log)[0] == _with_prefilter(
        today, _source(schema, 'big'), _expected_literal(blocks, target)
    )

    _shrink(monkeypatch, blocks + 1, target)
    _, log, _ = _sample(dsn, schema, 'big')
    assert _samples(log) == [today]


def test_a8_the_literal_is_exact() -> None:
    assert _expected_literal(16_384, 2_048) == '12.5'
    assert _expected_literal(16_385, 2_048) == '12.5'
    assert _expected_literal(32_767, 2_048) == '12.5'
    assert _expected_literal(32_768, 2_048) == '6.25'
    assert _expected_literal(142_720, 2_048) == '1.5625'
    assert _expected_literal(2_000_000, 2_048) == '0.1953125'
