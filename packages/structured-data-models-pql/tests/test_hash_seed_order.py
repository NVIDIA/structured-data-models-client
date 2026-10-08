# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The same query gives the same answer in every process.

Five places took their order from a set of strings, so the order followed
Python's string hashing, which is seeded afresh in each process unless
``PYTHONHASHSEED`` is set:

- the columns of a query (``all_query_columns`` and
  ``all_query_columns_with_locations``), and so the order of the unknown
  table and column errors;
- the ``AND``-joined entity filters of ``PQL_TARGET_FILTER_ONLY_ENTITY_COLUMNS``;
- the column list of ``PQL_TIME_POTENTIAL_LEAKAGE``;
- the entity-table column that ``PQL_TARGET_FILTER_REFERENCES_ENTITY`` points at;
- the columns of ``get_exclude_cols_dict``.

Each now keeps the order the query is written in, each item once, with the
entity first, then the target, then ``ASSUMING``.

Run as a script, this file prints its answers for every query in ``QUERIES``;
the test runs it under two hash seeds and wants the same output.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

import pytest

QUERIES = {
    'unknown-names': (
        'PREDICT COUNT(orders.* WHERE orders.a1 > 1 AND orders.a2 > 1 '
        'AND orders.a3 > 1 AND orders.a4 > 1 AND gone.x = 1 '
        'AND lost.y = 1, 0, 30, days) > 0 '
        'FOR EACH users.user_id WHERE users.a5 > 1'
    ),
    'entity-filters': (
        "PREDICT COUNT(orders.* WHERE users.age > 1 AND users.country = 'x', "
        '0, 30, days) > 0 AND COUNT(orders.* WHERE users.vip = True, 0, 30, '
        'days) > 0 AND COUNT(orders.* WHERE users.age > 2, 0, 30, days) > 0 '
        'FOR EACH users.user_id'
    ),
    'leakage': (
        'PREDICT users.country FOR EACH users.user_id WHERE '
        'COUNT(orders.* WHERE orders.amount > 1 AND orders.quantity > 1 AND '
        "orders.status = 'a', -INF, 0, days) > 0 AND "
        'COUNT(reviews.rating, -INF, 0, days) > 0'
    ),
    'filter-references-entity': (
        'PREDICT LIST_DISTINCT(orders.product_id WHERE users.age > 3 AND '
        "users.country = 'x' AND users.vip = True AND users.age < 9, "
        '0, 30, days) RANK TOP 3 FOR EACH users.user_id'
    ),
    'exclude-cols': (
        "PREDICT users.age > 1 AND users.country = 'x' AND users.vip = True "
        'AND users.signup > 2024-01-01 AND users.age < 99 '
        'FOR EACH users.user_id'
    ),
}

SEEDS = ('1', '2')


def _schema() -> Any:
    from predictive_query.schema.definition import (
        ColumnKey,
        ColumnKeyGroup,
        SchemaDefinition,
    )
    from predictive_query.schema.table import Column, TableDefinition
    from predictive_query.schema.typing import Dtype, Stype

    def table(
        cols: list[tuple[str, Stype, Dtype]],
        pkey: str,
        time_col: str | None = None,
    ) -> TableDefinition:
        return TableDefinition(
            cols=[Column(name=n, stype=s, dtype=d) for n, s, d in cols],
            pkey=pkey,
            time_col=time_col,
        )

    ident = (Stype.ID, Dtype.int)
    tables = {
        'users': table(
            [
                ('user_id', *ident),
                ('age', Stype.numerical, Dtype.int),
                ('country', Stype.categorical, Dtype.string),
                ('vip', Stype.categorical, Dtype.bool),
                ('signup', Stype.timestamp, Dtype.time),
            ],
            'user_id',
        ),
        'orders': table(
            [
                ('order_id', *ident),
                ('user_id', *ident),
                ('product_id', *ident),
                ('quantity', Stype.numerical, Dtype.int),
                ('amount', Stype.numerical, Dtype.float),
                ('status', Stype.categorical, Dtype.string),
                ('ordered_at', Stype.timestamp, Dtype.time),
            ],
            'order_id',
            'ordered_at',
        ),
        'products': table([('product_id', *ident)], 'product_id'),
        'reviews': table(
            [
                ('review_id', *ident),
                ('user_id', *ident),
                ('rating', Stype.numerical, Dtype.int),
                ('created_at', Stype.timestamp, Dtype.time),
            ],
            'review_id',
            'created_at',
        ),
    }
    groups = [
        ColumnKeyGroup(
            columns=(
                ColumnKey('orders', 'user_id'),
                ColumnKey('reviews', 'user_id'),
                ColumnKey('users', 'user_id'),
            )
        ),
        ColumnKeyGroup(
            columns=(
                ColumnKey('orders', 'product_id'),
                ColumnKey('products', 'product_id'),
            )
        ),
    ]
    return SchemaDefinition(tables=tables, col_groups=groups)


def answer(query: str) -> dict[str, Any]:
    r"""Everything a caller can see of *query*, in a form JSON keeps in order."""
    from predictive_query import RELATIONAL, PQLParser, validate
    from predictive_query.schema import json_serde

    resolved, response = validate(query, _schema(), RELATIONAL)
    parsed = PQLParser(mode=RELATIONAL.mode).to_parsed_predictive_query(query)
    return {
        'response': json_serde.to_json_dict(response),
        'tree': None if resolved is None else json_serde.to_json_dict(resolved),
        'columns': parsed.all_query_columns,
        'columns_with_locations': [
            [fqn, [loc.start_row, loc.start_col, loc.end_row, loc.end_col]]
            for fqn, loc in parsed.all_query_columns_with_locations
        ],
        'exclude_cols': None
        if resolved is None
        else resolved.get_exclude_cols_dict(),
    }


@pytest.fixture(scope='module')
def answers_by_seed() -> dict[str, dict[str, Any]]:
    answers = {}
    for seed in SEEDS:
        env = {**os.environ, 'PYTHONHASHSEED': seed}
        run = subprocess.run(
            [sys.executable, __file__],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        answers[seed] = json.loads(run.stdout)
    return answers


@pytest.mark.parametrize('case', sorted(QUERIES))
def test_two_hash_seeds_give_the_same_answer(answers_by_seed, case):
    first, second = (answers_by_seed[seed][case] for seed in SEEDS)
    assert first == second


def test_columns_come_in_the_order_written_each_once():
    """Entity first, then the target, then ``ASSUMING``; a repeat is dropped."""
    got = answer(QUERIES['exclude-cols'])
    assert got['columns'] == [
        'users.user_id',
        'users.age',
        'users.country',
        'users.vip',
        'users.signup',
    ]
    # `users.age` is written twice, at two places, so it is listed twice here.
    assert [fqn for fqn, _ in got['columns_with_locations']] == [
        'users.user_id',
        'users.age',
        'users.country',
        'users.vip',
        'users.signup',
        'users.age',
    ]
    assert got['exclude_cols'] == {'users': ['age', 'country', 'vip', 'signup']}


def test_messages_list_names_in_the_order_written():
    filters = answer(QUERIES['entity-filters'])['response']['warnings']
    assert (
        'FOR EACH users.user_id WHERE (users.age > 1) AND '
        "(users.country = 'x') AND users.vip = True AND users.age > 2\""
    ) in filters[0]['message']

    leakage = answer(QUERIES['leakage'])['response']['warnings'][0]['message']
    assert (
        "['orders.*', 'orders.amount', 'orders.quantity', 'orders.status', "
        "'reviews.rating']"
    ) in leakage

    errors = answer(QUERIES['unknown-names'])['response']['errors']
    assert [e['message'].split(': ', 1)[1].split("'")[1] for e in errors] == [
        'gone',
        'lost',
        'a5',
        'a1',
        'a2',
        'a3',
        'a4',
    ]

    refusal = answer(QUERIES['filter-references-entity'])['response']['errors']
    assert refusal[0]['code'] == 'PQL_TARGET_FILTER_REFERENCES_ENTITY'
    assert refusal[0]['message'].startswith('row 1, column 46:')


if __name__ == '__main__':
    print(json.dumps({case: answer(q) for case, q in QUERIES.items()}))
