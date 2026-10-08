# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How often a valid multi-column link is refused, on large PostgreSQL tables.

``Graph.link`` checks that a multi-column reference meets the parent's
identity before it builds the edge. It used to compare the child's sample
with the parent's own 1,000-row sample, and two samples drawn independently
from a large parent mostly share no parent, so most valid links were refused
(25-90% at one to ten million parents). It now asks the parent whether it
holds any of the child's sampled identities, so a valid link is never
refused. This rebuilds that measurement as a user meets it: tables built in
PostgreSQL, ``Graph.from_postgres``, a composite primary key set on the
parent, ``link`` called.

Every combination of the options below is tried ``--trials`` times, each with
other key values. A parent holds ``N`` rows and each child row references one
of them, so every link tried is valid. Each trial prints one JSON line, and
the run ends with a summary; it exits 1 if any valid link was refused or any
trial failed otherwise.

Not part of the test suite: it builds tables of millions of rows, which takes
minutes. Run it by hand against a scratch database::

    python qa/composite_link_matrix.py \\
        --dsn postgresql://postgres:postgres@localhost:55432/sdm_test

The defaults are the matrix ``DESIGN-composite-link.md`` measured: 1M and 5M
parents, 2 and 3 key columns, a child with and without a declared primary
key, stored clustered by parent or at random.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import uuid
import warnings
from typing import Any

#: Child rows per parent.
CHILDREN = 5
#: A multiplier that spreads consecutive ids over the parents.
_MULTIPLIER = 2654435761


def _key_expressions(columns: int, row: str, salt: int) -> list[str]:
    r"""The key parts of parent ``row``, an SQL expression: a text part and
    one or two numbers, unique per row, different for each ``salt``."""
    if columns == 2:
        return [
            f"'R' || CAST(({row}) % 50 AS TEXT)",
            f'(({row}) / 50) + {salt * 10_000_000}',
        ]
    return [
        f"'R' || CAST(({row}) % 10 AS TEXT)",
        f'((({row}) / 10) % 20)',
        f'(({row}) / 200) + {salt * 10_000_000}',
    ]


def _parent_of(order: str, child: str, parents: int, salt: int) -> str:
    r"""The parent child row ``child`` references: its own block of
    ``CHILDREN`` rows when ``clustered``, any parent when ``random``."""
    if order == 'clustered':
        return f'(({child}) / {CHILDREN})'
    # Squaring breaks the lattice a multiplicative hash of consecutive ids
    # leaves, so the first child rows reference parents at random.
    spread = f'((({child}) * {_MULTIPLIER} + {salt * 1000003}) % 2147483647)'
    return f'(((({spread}) * ({spread}) + {salt}) % 2147483647) % {parents})'


def _build(
    connection: Any,
    schema: str,
    *,
    parents: int,
    columns: int,
    parent_key: str,
    child_key: str,
    order: str,
    salt: int,
) -> list[str]:
    r"""The parent and child tables of one trial, in ``schema``; the names
    of the key columns."""
    keys = [f'k{index}' for index in range(1, columns + 1)]
    types = ['TEXT'] + ['BIGINT'] * (columns - 1)
    declared = ', '.join(
        f'{key} {kind} NOT NULL' for key, kind in zip(keys, types, strict=True)
    )
    surrogate = parent_key == 'surrogate'
    connection.execute(f'CREATE SCHEMA "{schema}"')
    connection.execute(f'SET search_path TO "{schema}"')
    connection.execute(
        'CREATE TABLE parent ('
        + ('parent_id BIGINT NOT NULL PRIMARY KEY, ' if surrogate else '')
        + f'{declared}, name TEXT)'
    )
    connection.execute(
        'CREATE TABLE child (child_id BIGINT NOT NULL'
        + (' PRIMARY KEY' if child_key == 'pk' else '')
        + f', {declared}, amount BIGINT)'
    )
    parent_parts = ', '.join(_key_expressions(columns, 'i', salt))
    connection.execute(
        'INSERT INTO parent SELECT '
        + ('i, ' if surrogate else '')
        + f"{parent_parts}, 'p' || i "
        f'FROM generate_series(0, {parents - 1}) AS s(i) ORDER BY i'
    )
    referenced = _parent_of(order, 'j', parents, salt)
    child_parts = ', '.join(_key_expressions(columns, referenced, salt))
    connection.execute(
        f'INSERT INTO child SELECT j, {child_parts}, j % 997 '
        f'FROM generate_series(0, {parents * CHILDREN - 1}) AS s(j) '
        'ORDER BY j'
    )
    connection.execute('ANALYZE parent')
    connection.execute('ANALYZE child')
    return keys


def _link(dsn: str, schema: str, keys: list[str]) -> tuple[bool, str | None]:
    r"""Whether linking the child to the parent was refused as matching no
    rows, and any other error."""
    import sdm_relational_engine.rfm as rfm

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        graph = rfm.Graph.from_postgres(
            dsn,
            tables=['parent', 'child'],
            schema=schema,
            infer_metadata=False,
            verbose=False,
        )
        try:
            graph['parent'].primary_key = tuple(keys)
            graph['child'].primary_key = 'child_id'
            graph.link('child', tuple(keys), 'parent')
        except ValueError as error:
            if 'matches no rows' in str(error):
                return True, None
            return False, str(error)[:500]
    return False, None


def trial(dsn: str, **case: Any) -> dict[str, Any]:
    r"""One link, built and checked in a schema of its own, then dropped."""
    import psycopg

    schema = f'qa_link_{os.getpid()}_{uuid.uuid4().hex[:8]}'
    started = time.perf_counter()
    try:
        with psycopg.connect(dsn, autocommit=True) as connection:
            keys = _build(connection, schema, **case)
        built = time.perf_counter()
        refused, error = _link(dsn, schema, keys)
        linked = time.perf_counter()
    except Exception as error:  # noqa: BLE001 - recorded, and the run goes on
        return {**case, 'refused': None, 'error': repr(error)[:500]}
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    return {
        **case,
        'refused': refused,
        'error': error,
        'build_s': round(built - started, 1),
        'link_s': round(linked - built, 2),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--dsn',
        default=os.getenv('SDM_POSTGRES_TEST_DSN'),
        help='a scratch PostgreSQL database (default: SDM_POSTGRES_TEST_DSN)',
    )
    parser.add_argument(
        '--parents', type=int, nargs='+', default=[1_000_000, 5_000_000]
    )
    parser.add_argument('--columns', type=int, nargs='+', default=[2, 3])
    parser.add_argument(
        '--parent-keys',
        nargs='+',
        default=['none', 'surrogate'],
        choices=['none', 'surrogate'],
        help='no primary key declared on the parent, or a surrogate one '
        '(PostgreSQL discovery refuses a declared composite key)',
    )
    parser.add_argument(
        '--child-keys',
        nargs='+',
        default=['pk', 'none'],
        choices=['pk', 'none'],
    )
    parser.add_argument(
        '--orders',
        nargs='+',
        default=['clustered', 'random'],
        choices=['clustered', 'random'],
    )
    parser.add_argument('--trials', type=int, default=2)
    options = parser.parse_args(argv)
    if not options.dsn:
        parser.error('--dsn or SDM_POSTGRES_TEST_DSN is required')

    results = []
    for parents, columns, parent_key, child_key, order in itertools.product(
        options.parents,
        options.columns,
        options.parent_keys,
        options.child_keys,
        options.orders,
    ):
        for salt in range(options.trials):
            result = trial(
                options.dsn,
                parents=parents,
                columns=columns,
                parent_key=parent_key,
                child_key=child_key,
                order=order,
                salt=salt,
            )
            results.append(result)
            print(json.dumps(result), flush=True)

    refused = sum(result['refused'] is True for result in results)
    failed = sum(
        result['refused'] is None or result['error'] is not None
        for result in results
    )
    print(
        f'{len(results)} valid link(s) tried: {refused} refused, '
        f'{failed} failed otherwise',
        flush=True,
    )
    return 0 if refused == 0 and failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
