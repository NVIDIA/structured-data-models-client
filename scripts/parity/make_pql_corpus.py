#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Write the PQL corpus: queries, their schemas, and how today's PQL reads them.

    make_pql_corpus.py [--check]

The queries cover the grammar a clause at a time: every aggregation over
every kind of column, windows, conditions, filters on the target and on the
entity, entity lists, static targets, forecasts, rankings, assumptions,
keyword spellings, quoted names and literals. Each is resolved against a
schema by ``predictive_query.validate`` under a profile, and the case records
the resolved tree, or the diagnostics with their codes and spans. Where a
query is written to be refused, the case says so (``refused=True``) and
writing stops if today's code accepts it, and the other way round.

``--check`` writes nothing and fails if the stored corpus differs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PARITY = REPO / 'tests' / 'parity'
sys.path.insert(0, str(PARITY))

from test_pql_corpus import resolve  # noqa: E402

CORPUS = PARITY / 'pql'


def _schema(tables: dict[str, Any], links: list[list[str]]) -> dict[str, Any]:
    r"""A ``SchemaDefinition`` as JSON: tables of ``(name, stype, dtype)``
    columns with their keys, and groups of columns that join."""
    from predictive_query.schema import json_serde
    from predictive_query.schema.definition import (
        ColumnKey,
        ColumnKeyGroup,
        SchemaDefinition,
    )
    from predictive_query.schema.table import Column, TableDefinition
    from predictive_query.schema.typing import Dtype, Stype

    definition = SchemaDefinition(
        tables={
            name: TableDefinition(
                cols=[
                    Column(column, Stype(stype), Dtype(dtype))
                    for column, stype, dtype in spec['cols']
                ],
                pkey=spec.get('pkey'),
                time_col=spec.get('time_col'),
            )
            for name, spec in tables.items()
        },
        col_groups=[
            ColumnKeyGroup(
                tuple(ColumnKey(*key.split('.')) for key in sorted(group))
            )
            for group in links
        ],
    )
    return json_serde.to_json_dict(definition)


SHOP = _schema(
    {
        'users': {
            'cols': [
                ('user_id', 'ID', 'int'),
                ('age', 'numerical', 'int'),
                ('country', 'categorical', 'string'),
                ('vip', 'categorical', 'bool'),
                ('bio', 'text', 'string'),
                ('signup', 'timestamp', 'time'),
            ],
            'pkey': 'user_id',
        },
        'products': {
            'cols': [
                ('product_id', 'ID', 'int'),
                ('category', 'categorical', 'string'),
                ('price', 'numerical', 'float'),
                ('tags', 'multicategorical', 'stringlist'),
            ],
            'pkey': 'product_id',
        },
        'orders': {
            'cols': [
                ('order_id', 'ID', 'int'),
                ('user_id', 'ID', 'int'),
                ('product_id', 'ID', 'int'),
                ('quantity', 'numerical', 'int'),
                ('amount', 'numerical', 'float'),
                ('status', 'categorical', 'string'),
                ('ordered_at', 'timestamp', 'time'),
            ],
            'pkey': 'order_id',
            'time_col': 'ordered_at',
        },
        'reviews': {
            'cols': [
                ('review_id', 'ID', 'int'),
                ('user_id', 'ID', 'int'),
                ('product_id', 'ID', 'int'),
                ('rating', 'numerical', 'int'),
                ('created_at', 'timestamp', 'time'),
            ],
            'pkey': 'review_id',
            'time_col': 'created_at',
        },
    },
    [
        ['users.user_id', 'orders.user_id', 'reviews.user_id'],
        ['products.product_id', 'orders.product_id', 'reviews.product_id'],
    ],
)

PEOPLE = _schema(
    {
        'people': {
            'cols': [
                ('person_id', 'ID', 'int'),
                ('age', 'numerical', 'float'),
                ('churn', 'numerical', 'int'),
                ('segment', 'categorical', 'string'),
                ('joined', 'timestamp', 'time'),
            ],
            'pkey': 'person_id',
        }
    },
    [],
)

QUOTED = _schema(
    {
        'Order Lines': {
            'cols': [
                ('Line ID', 'ID', 'int'),
                ('Customer ID', 'ID', 'int'),
                ('Unit Price', 'numerical', 'float'),
                ('Order Date', 'timestamp', 'time'),
            ],
            'pkey': 'Line ID',
            'time_col': 'Order Date',
        },
        'Customers': {
            'cols': [
                ('Customer ID', 'ID', 'int'),
                ('Region', 'categorical', 'string'),
            ],
            'pkey': 'Customer ID',
        },
    },
    [['Order Lines.Customer ID', 'Customers.Customer ID']],
)

# Shapes the three schemas above lack, for the codes only they reach: an
# aggregation nested in another's filter, a link with no time column, a
# table that points at itself, and two foreign keys into one table.
NESTED = _schema(
    {
        'customers': {
            'cols': [
                ('customer_id', 'ID', 'int'),
                ('region', 'categorical', 'string'),
            ],
            'pkey': 'customer_id',
        },
        'orders': {
            'cols': [
                ('order_id', 'ID', 'int'),
                ('customer_id', 'ID', 'int'),
                ('total', 'numerical', 'float'),
                ('placed_at', 'timestamp', 'time'),
            ],
            'pkey': 'order_id',
            'time_col': 'placed_at',
        },
        'items': {
            'cols': [
                ('item_id', 'ID', 'int'),
                ('order_id', 'ID', 'int'),
                ('price', 'numerical', 'float'),
                ('added_at', 'timestamp', 'time'),
            ],
            'pkey': 'item_id',
            'time_col': 'added_at',
        },
    },
    [
        ['customers.customer_id', 'orders.customer_id'],
        ['orders.order_id', 'items.order_id'],
    ],
)

LINKS = _schema(
    {
        'members': {
            'cols': [('member_id', 'ID', 'int'), ('age', 'numerical', 'int')],
            'pkey': 'member_id',
        },
        'likes': {
            'cols': [
                ('like_id', 'ID', 'int'),
                ('member_id', 'ID', 'int'),
                ('item_id', 'ID', 'int'),
            ],
            'pkey': 'like_id',
        },
        'items': {
            'cols': [('item_id', 'ID', 'int'), ('name', 'text', 'string')],
            'pkey': 'item_id',
        },
    },
    [
        ['members.member_id', 'likes.member_id'],
        ['items.item_id', 'likes.item_id'],
    ],
)

STAFF = _schema(
    {
        'employees': {
            'cols': [
                ('employee_id', 'ID', 'int'),
                ('manager_id', 'ID', 'int'),
                ('salary', 'numerical', 'float'),
                ('hired_at', 'timestamp', 'time'),
            ],
            'pkey': 'employee_id',
            'time_col': 'hired_at',
        },
    },
    [['employees.employee_id', 'employees.manager_id']],
)

TRIPS = _schema(
    {
        'people': {'cols': [('person_id', 'ID', 'int')], 'pkey': 'person_id'},
        'trips': {
            'cols': [
                ('trip_id', 'ID', 'int'),
                ('rider_id', 'ID', 'int'),
                ('driver_id', 'ID', 'int'),
                ('fare', 'numerical', 'float'),
                ('started_at', 'timestamp', 'time'),
            ],
            'pkey': 'trip_id',
            'time_col': 'started_at',
        },
    },
    [
        ['people.person_id', 'trips.rider_id'],
        ['people.person_id', 'trips.driver_id'],
    ],
)

SCHEMAS = {
    'shop': SHOP,
    'people': PEOPLE,
    'quoted': QUOTED,
    'nested': NESTED,
    'links': LINKS,
    'staff': STAFF,
    'trips': TRIPS,
}


class Corpus:
    def __init__(self) -> None:
        self.cases: list[dict[str, Any]] = []
        self._seen: dict[str, str] = {}

    def add(
        self,
        group: str,
        query: str,
        schema: str = 'shop',
        profile: str = 'RELATIONAL',
        refused: bool | None = None,
    ) -> None:
        key = json.dumps([query, schema, profile])
        if key in self._seen:
            return
        label = re.sub(r'[^A-Za-z0-9]+', '-', query).strip('-').lower()[:70]
        case_id = f'{group}/{label or "empty"}'
        suffix = 2
        while case_id in self._seen.values():
            case_id = f'{group}/{label}-{suffix}'
            suffix += 1
        self._seen[key] = case_id
        case: dict[str, Any] = {
            'id': case_id,
            'query': query,
            'schema': SCHEMAS[schema],
            'profile': profile,
        }
        got = json.loads(json.dumps(resolve(case)))
        if 'tree' in got:
            if refused:
                raise SystemExit(
                    f'{case_id}: resolved, and was written to be refused'
                )
            case['expected_tree'] = got['tree']
        else:
            if refused is False:
                raise SystemExit(f'{case_id}: refused {got["errors"]}')
            case['expected_errors'] = got['errors']
        if got['warnings']:
            case['expected_warnings'] = got['warnings']
        self.cases.append(case)


AGGREGATIONS = (
    'SUM',
    'AVG',
    'MIN',
    'MAX',
    'COUNT',
    'COUNT_DISTINCT',
    'FIRST',
    'LAST',
    'LIST_DISTINCT',
)
TARGETS = (
    'orders.*',
    'orders.quantity',
    'orders.amount',
    'orders.status',
    'orders.product_id',
    'orders.ordered_at',
)
WINDOWS = (
    '0, 30, days',
    '0, 7, days',
    '0, 12, hours',
    '0, 2, weeks',
    '0, 3, months',
    '-30, 0, days',
    '10, 40, days',
    '-INF, 0, days',
)


def build() -> Corpus:
    c = Corpus()
    each = 'FOR EACH users.user_id'
    # Every aggregation over every kind of column, over 30 days.
    for aggr in AGGREGATIONS:
        for target in TARGETS:
            c.add(
                'aggregation', f'PREDICT {aggr}({target}, 0, 30, days) {each}'
            )
    # The aggregations a question usually asks, of users and of products,
    # over windows of every unit.
    for aggr, target in [
        ('COUNT', 'orders.*'),
        ('SUM', 'orders.quantity'),
        ('SUM', 'orders.amount'),
        ('AVG', 'orders.amount'),
        ('MAX', 'orders.amount'),
        ('MIN', 'orders.quantity'),
        ('COUNT_DISTINCT', 'orders.product_id'),
        ('LIST_DISTINCT', 'orders.product_id'),
        ('COUNT', 'reviews.*'),
        ('AVG', 'reviews.rating'),
    ]:
        for entity in ('users.user_id', 'products.product_id'):
            for window in (
                '0, 7, days',
                '0, 90, days',
                '0, 24, hours',
                '0, 1, months',
            ):
                c.add(
                    'question',
                    f'PREDICT {aggr}({target}, {window}) FOR EACH {entity}',
                )
    # Windows, and windows written without a unit.
    for window in WINDOWS:
        c.add('window', f'PREDICT COUNT(orders.*, {window}) {each}')
        c.add('window', f'PREDICT SUM(orders.amount, {window}) {each}')
    for window in (
        '0, 30',
        '0, 0, days',
        '30, 0, days',
        '0, 30, years',
        '0, -5, days',
    ):
        c.add('window', f'PREDICT COUNT(orders.*, {window}) {each}')
    c.add('window', f'PREDICT COUNT(orders.*) {each}')
    # Conditions on an aggregation, as classifications.
    for op, value in [
        ('>', '0'),
        ('=', '0'),
        ('>=', '3'),
        ('<', '2'),
        ('<=', '1'),
        ('!=', '0'),
        ('>', '2.5'),
        ('=', "'x'"),
        ('IS', 'NULL'),
        ('IS NOT', 'NULL'),
    ]:
        c.add(
            'condition',
            f'PREDICT COUNT(orders.*, 0, 30, days) {op} {value} {each}',
        )
        c.add(
            'condition',
            f'PREDICT SUM(orders.amount, 0, 30, days) {op} {value} {each}',
        )
    for text in [
        'COUNT(orders.*, 0, 30, days) > 0 AND SUM(orders.amount, 0, 30, days) > 100',
        'COUNT(orders.*, 0, 30, days) = 0 OR COUNT(reviews.*, 0, 30, days) = 0',
        'NOT COUNT(orders.*, 0, 30, days) > 0',
        '(COUNT(orders.*, 0, 30, days) > 0 OR COUNT(reviews.*, 0, 30, days) > 0) '
        'AND SUM(orders.quantity, 0, 30, days) >= 5',
        'COUNT(orders.*, 0, 30, days) > 0 AND NOT COUNT(reviews.*, 0, 30, days) > 0',
        'COUNT(orders.*, 0, 30, days) > 0 OR COUNT(orders.*, 0, 30, days) > 0 AND '
        'COUNT(reviews.*, 0, 7, days) > 1',
        'MAX(orders.amount, 0, 30, days) > 50 AND MIN(orders.amount, 0, 30, days) < 5',
    ]:
        c.add('logical', f'PREDICT {text} {each}')
    # Filters on the target's rows.
    for where in [
        "orders.status = 'shipped'",
        "orders.status != 'returned'",
        "orders.status IN ('shipped', 'delivered')",
        'orders.amount > 10.5',
        'orders.quantity >= 2',
        "orders.status LIKE 'ship%'",
        "orders.status NOT LIKE 'ret%'",
        "orders.status CONTAINS 'hip'",
        "orders.status NOT CONTAINS 'x'",
        "orders.status STARTS WITH 's'",
        "orders.status ENDS WITH 'd'",
        'orders.status IS NULL',
        'orders.status IS NOT NULL',
        'orders.ordered_at > 2024-01-01',
        'orders.ordered_at <= 2024-06-30 12:00:00',
        "orders.amount > 10 AND orders.status = 'shipped'",
        "NOT orders.status = 'returned'",
        'orders.quantity IN (1, 2, 3)',
        'orders.amount IN (1.5, 2.5)',
        'orders.quantity = 1.5',
        "orders.amount > 'ten'",
        'orders.missing = 1',
        'users.age > 30',
    ]:
        c.add(
            'filter',
            f'PREDICT COUNT(orders.* WHERE {where}, 0, 30, days) {each}',
        )
    c.add(
        'filter',
        "PREDICT SUM(orders.amount WHERE orders.status = 'shipped', 0, 30, days) > 100 "
        f'{each}',
    )
    # Entities: plain, filtered, listed.
    for entity in [
        'FOR EACH users.user_id',
        'FOR users.user_id = 1',
        'FOR users.user_id IN (1, 2, 3)',
        'FOR EACH users.user_id WHERE users.age > 30',
        "FOR EACH users.user_id WHERE users.country = 'DE'",
        'FOR EACH users.user_id WHERE COUNT(orders.*, -30, 0, days) > 0',
        'FOR EACH users.user_id WHERE users.vip = true',
        'FOR users.user_id IN (1, 2) WHERE users.age > 30',
        'FOR EACH products.product_id',
        'FOR EACH orders.order_id',
        'FOR EACH users.age',
        'FOR EACH users.*',
        'FOR users.user_id = 1.5',
        "FOR users.user_id = 'one'",
        'FOR users.user_id IN (1)',
    ]:
        c.add('entity', f'PREDICT COUNT(orders.*, 0, 30, days) {entity}')
    # Targets that are not aggregations.
    for target in [
        'users.country',
        'users.age',
        'users.vip',
        'users.age > 30',
        "users.country = 'DE'",
        'users.bio',
        'users.signup',
        'users.user_id',
        'users.*',
        'products.category',
        'products.price',
        'products.tags',
    ]:
        entity = (
            'products.product_id'
            if target.startswith('products')
            else 'users.user_id'
        )
        c.add('static', f'PREDICT {target} FOR EACH {entity}')
    # Forecasts, rankings, classification, assumptions.
    for n in (1, 3, 4, 12, 0):
        c.add(
            'forecast',
            f'PREDICT SUM(orders.quantity, 0, 30, days) FORECAST {n} TIMEFRAMES {each}',
        )
    c.add(
        'forecast',
        'PREDICT COUNT(orders.*, 0, 7, days) FORECAST 3 TIMEFRAMES FOR EACH products.product_id',
    )
    c.add(
        'forecast',
        f'PREDICT COUNT(orders.*, 0, 7, days) > 0 FORECAST 3 TIMEFRAMES {each}',
    )
    c.add(
        'forecast',
        'PREDICT users.age FORECAST 3 TIMEFRAMES FOR EACH users.user_id',
    )
    for k in (1, 3, 5, 10, 0):
        c.add(
            'ranking',
            f'PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) RANK TOP {k} {each}',
        )
    c.add(
        'ranking',
        f'PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) {each}',
    )
    c.add(
        'ranking',
        f'PREDICT LIST_DISTINCT(orders.status, 0, 30, days) RANK TOP 3 {each}',
    )
    c.add('ranking', f'PREDICT COUNT(orders.*, 0, 30, days) RANK TOP 3 {each}')
    c.add(
        'ranking',
        f'PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) TOP 3 {each}',
    )
    c.add(
        'ranking',
        f'PREDICT LIST_DISTINCT(reviews.product_id, 0, 30, days) RANK TOP 2 {each}',
    )
    c.add(
        'classify',
        f'PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) CLASSIFY {each}',
    )
    c.add(
        'classify', f'PREDICT COUNT(orders.*, 0, 30, days) > 0 CLASSIFY {each}'
    )
    for assumption in [
        "orders.status = 'shipped'",
        'COUNT(reviews.*, 0, 30, days) > 0',
        'users.vip = true',
    ]:
        c.add(
            'assuming',
            f'PREDICT COUNT(orders.*, 0, 30, days) {each} ASSUMING {assumption}',
        )
    # Keyword spellings: exact lower or upper case, one space.
    for query in [
        'predict count(orders.*, 0, 30, days) for each users.user_id',
        'PREDICT count(orders.*, 0, 30, DAYS) FOR EACH users.user_id',
        'Predict COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
        'PREDICT Count(orders.*, 0, 30, days) FOR EACH users.user_id',
        'PREDICT COUNT(orders.*, 0, 30, days) FOR  EACH users.user_id',
        'PREDICT COUNT(orders.*, 0, 30, days) For Each users.user_id',
        'PREDICT COUNT(orders.*, 0, 30, days)\nFOR EACH users.user_id',
        'PREDICT COUNT(orders.*, 0, 30, days) # a comment\nFOR EACH users.user_id',
        '  PREDICT   COUNT( orders.* , 0 , 30 , days )   FOR EACH   users.user_id  ',
        "PREDICT COUNT(orders.* where orders.status = 'x', 0, 30, days) for each users.user_id",
        "PREDICT COUNT(orders.* WHERE orders.status not like 'x%', 0, 30, days) FOR EACH users.user_id",
        'PREDICT COUNT(orders.*, 0, 30, days) = 0 and COUNT(reviews.*, 0, 30, days) = 0 FOR EACH users.user_id',
        'PREDICT sum(orders.amount, -inf, 0, days) FOR EACH users.user_id',
        'PREDICT SUM(orders.amount, -INFINITY, 0, days) FOR EACH users.user_id',
    ]:
        c.add('spelling', query)
    # Syntax the grammar refuses.
    for query in [
        ' ',
        'PREDICT',
        'PREDICT COUNT(orders.*, 0, 30, days)',
        'COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
        'PREDICT COUNT(orders.*, 0, 30, days FOR EACH users.user_id',
        'PREDICT COUNT orders.*, 0, 30, days) FOR EACH users.user_id',
        'PREDICT COUNT(orders.*, 0, days) FOR EACH users.user_id',
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH',
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users',
        'PREDICT COUNT(orders, 0, 30, days) FOR EACH users.user_id',
        'PREDICT TOTAL(orders.amount, 0, 30, days) FOR EACH users.user_id',
        'PREDICT COUNT(orders.*, 0, 30, days) > FOR EACH users.user_id',
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id extra',
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id WHERE',
        'PREDICT COUNT(orders.*, a, 30, days) FOR EACH users.user_id',
        'PREDICT COUNT(orders.*, 0.5, 30, days) FOR EACH users.user_id',
        'PREDICT COUNT(orders.* WHERE orders.quantity IN (1), 0, 30, days) FOR EACH users.user_id',
        'PREDICT COUNT(orders.* WHERE orders.quantity IN (1,), 0, 30, days) FOR EACH users.user_id',
        'PREDICT COUNT(orders.*, 0, 30, days) FORECAST TIMEFRAMES FOR EACH users.user_id',
        'PREDICT COUNT(orders.*, 0, 30, days) RANK TOP FOR EACH users.user_id',
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id ASSUMING',
        "PREDICT COUNT(orders.* WHERE orders.status = 'unterminated, 0, 30, days) FOR EACH users.user_id",
        'PREDICT COUNT(orders.*, 0, 30, days) >> 0 FOR EACH users.user_id',
        'PREDICT (COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id',
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id;',
        'SELECT * FROM orders',
    ]:
        c.add('syntax', query, refused=True)
    # Names the schema does not have.
    for query in [
        'PREDICT COUNT(order.*, 0, 30, days) FOR EACH users.user_id',
        'PREDICT SUM(orders.amt, 0, 30, days) FOR EACH users.user_id',
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH user.user_id',
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.id',
        'PREDICT COUNT(Orders.*, 0, 30, days) FOR EACH users.user_id',
        'PREDICT SUM(orders.Amount, 0, 30, days) FOR EACH users.user_id',
        'PREDICT users.nope FOR EACH users.user_id',
    ]:
        c.add('names', query, refused=True)
    # Quoted names.
    for query in [
        'PREDICT SUM(`Order Lines`.`Unit Price`, 0, 30, days) FOR EACH Customers.`Customer ID`',
        'PREDICT COUNT(`Order Lines`.*, 0, 30, days) FOR EACH `Customers`.`Customer ID`',
        'PREDICT COUNT(`Order Lines`.*, 0, 30, days) > 0 FOR EACH Customers.`Customer ID`',
        'PREDICT Customers.Region FOR EACH Customers.`Customer ID`',
        'PREDICT COUNT(Order Lines.*, 0, 30, days) FOR EACH Customers.`Customer ID`',
        'PREDICT COUNT(`Order Lines`.*, 0, 30, days) FOR EACH Customers.Customer ID',
        'PREDICT COUNT(`Order Lines`.*, 0, 30, days) FOR EACH Customers.`customer id`',
    ]:
        c.add('quoted', query, 'quoted')
    # One table: the tabular profile, and the relational one.
    for query in [
        'PREDICT people.churn FOR EACH people.person_id',
        'PREDICT people.age FOR EACH people.person_id',
        'PREDICT people.segment FOR EACH people.person_id',
        "PREDICT people.segment = 'gold' FOR EACH people.person_id",
        'PREDICT people.age > 40 FOR EACH people.person_id',
        'PREDICT people.churn FOR EACH people.person_id WHERE people.age > 30',
        'PREDICT people.churn FOR people.person_id IN (1, 2)',
        'PREDICT people.joined FOR EACH people.person_id',
        'PREDICT people.person_id FOR EACH people.person_id',
        'PREDICT COUNT(people.*, 0, 30, days) FOR EACH people.person_id',
        'PREDICT people.churn RANK TOP 2 FOR EACH people.person_id',
        'PREDICT people.churn FORECAST 2 TIMEFRAMES FOR EACH people.person_id',
        "PREDICT people.churn FOR EACH people.person_id ASSUMING people.segment = 'gold'",
    ]:
        for profile in ('TABULAR', 'RELATIONAL', 'TIMESERIES'):
            c.add(f'single-table-{profile.lower()}', query, 'people', profile)
    # Every profile over the shop.
    for profile in ('TABULAR', 'TIMESERIES', 'ANOMALY', 'FINE_TUNING'):
        for query in [
            f'PREDICT COUNT(orders.*, 0, 30, days) {each}',
            f'PREDICT COUNT(orders.*, 0, 30, days) > 0 {each}',
            'PREDICT users.country FOR EACH users.user_id',
            f'PREDICT SUM(orders.quantity, 0, 30, days) FORECAST 3 TIMEFRAMES {each}',
            f'PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) RANK TOP 3 {each}',
        ]:
            c.add(
                f'profile-{profile.lower().replace("_", "-")}',
                query,
                'shop',
                profile,
            )
    # Literals.
    for literal in [
        "'shipped'",
        '"shipped"',
        "'it''s'",
        "'with \\' quote'",
        '1e3',
        '-4.5',
        '1.35E-9',
        'true',
        'FALSE',
        'null',
        '2024-02-29',
        '2023-02-29',
        '2024/02/29',
        '29.02.2024',
        '2024-01-01 25:00:00',
        '2024-13-01',
    ]:
        column = (
            'orders.ordered_at'
            if literal[0].isdigit()
            and '-' in literal
            or '/' in literal
            or literal.count('.') == 2
            else 'orders.status'
        )
        c.add(
            'literal',
            f'PREDICT COUNT(orders.* WHERE {column} = {literal}, 0, 30, days) {each}',
        )
    coverage(c, each)
    return c


def coverage(c: Corpus, each: str) -> None:
    r"""A refusal for each code the cases above leave out, so every code
    today's Python can give is in the corpus (Phase 5, task 6). The codes
    no query can reach are listed in ``rust/crates/pql/UNREACHABLE_CODES.md``."""
    where = 'PREDICT COUNT(orders.* WHERE {}, 0, 30, days) ' + each
    ids = ', '.join(str(i) for i in range(1001))
    for schema, profile, query in [
        (
            'trips',
            'FINE_TUNING',
            'PREDICT COUNT(trips.*, 0, 7, days) FOR EACH people.person_id',
        ),
        ('shop', 'RELATIONAL', f'PREDICT orders.amount {each}'),
        (
            'shop',
            'RELATIONAL',
            f'PREDICT COUNT(orders.*, 0, 30, days) {each} WHERE orders.amount > 1',
        ),
        (
            'shop',
            'RELATIONAL',
            f'PREDICT COUNT(orders.*, 0, 30, days) > 0 {each} WHERE '
            "LIST_DISTINCT(orders.product_id, -30, 0, days) = 'x'",
        ),
        (
            'shop',
            'FINE_TUNING',
            'PREDICT COUNT(orders.*, 0, 30, days) FOR users.user_id = 1',
        ),
        (
            'shop',
            'FINE_TUNING',
            'PREDICT COUNT(orders.*, 0, 30, days) FOR users.user_id',
        ),
        (
            'shop',
            'RELATIONAL',
            f'PREDICT COUNT(orders.*, 0, 30, days) {each} = 1',
        ),
        (
            'shop',
            'RELATIONAL',
            'PREDICT COUNT(orders.*, 0, 30, days) FOR users.user_id > 1',
        ),
        (
            'shop',
            'RELATIONAL',
            'PREDICT COUNT(orders.*, 0, 30, days) FOR users.user_id',
        ),
        (
            'shop',
            'RELATIONAL',
            f'PREDICT COUNT(orders.*, 0, 30, days) FOR users.user_id IN ({ids})',
        ),
        (
            'nested',
            'RELATIONAL',
            'PREDICT COUNT(orders.* WHERE COUNT(items.*, -7, 0, days) > 0, 0, 30, days) '
            'FOR EACH customers.customer_id',
        ),
        (
            'staff',
            'TIMESERIES',
            'PREDICT COUNT(employees.*, 0, 30, days) FOR EACH employees.employee_id',
        ),
        (
            'links',
            'RELATIONAL',
            'PREDICT LIST_DISTINCT(likes.item_id) RANK TOP 3 FOR EACH members.member_id',
        ),
        (
            'shop',
            'RELATIONAL',
            'PREDICT products.tags CLASSIFY FOR EACH products.product_id',
        ),
        (
            'shop',
            'RELATIONAL',
            f'PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) RANK TOP 21 {each}',
        ),
        (
            'shop',
            'RELATIONAL',
            f"PREDICT COUNT(orders.*, 0, 30, days) {each} WHERE users.bio = 'x'",
        ),
        (
            'shop',
            'FINE_TUNING',
            f'PREDICT LIST_DISTINCT(orders.order_id, 0, 30, days) RANK TOP 3 {each}',
        ),
        (
            'shop',
            'FINE_TUNING',
            'PREDICT LIST_DISTINCT(orders.product_id WHERE users.age > 3, 0, 30, days) '
            f'RANK TOP 3 {each}',
        ),
        ('shop', 'FINE_TUNING', f'PREDICT users.user_id = 1 {each}'),
        (
            'shop',
            'FINE_TUNING',
            f'PREDICT LIST_DISTINCT(orders.product_id) RANK TOP 3 {each}',
        ),
        (
            'shop',
            'FINE_TUNING',
            f'PREDICT SUM(orders.quantity, 0, 30, days) FORECAST 10001 TIMEFRAMES {each}',
        ),
        (
            'shop',
            'FINE_TUNING',
            'PREDICT products.tags FOR EACH products.product_id',
        ),
        (
            'shop',
            'FINE_TUNING',
            f'PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) RANK {each}',
        ),
        (
            'shop',
            'RELATIONAL',
            f'PREDICT COUNT(orders.*, 0, 30, days) {each} WHERE COUNT(orders.*, 0, 30, days) > 0',
        ),
        (
            'shop',
            'FINE_TUNING',
            f'PREDICT LIST_DISTINCT(orders.product_id) CLASSIFY {each}',
        ),
        (
            'shop',
            'FINE_TUNING',
            f'PREDICT LIST_DISTINCT(orders.status) RANK TOP 3 {each}',
        ),
        (
            'shop',
            'FINE_TUNING',
            f'PREDICT LIST_DISTINCT(orders.amount) RANK TOP 3 {each}',
        ),
        (
            'shop',
            'FINE_TUNING',
            f'PREDICT COUNT(orders.*, 0, 30, days) > 0 AND users.age > 3 {each}',
        ),
        (
            'shop',
            'FINE_TUNING',
            f'PREDICT users.age {each} WHERE COUNT(orders.*, -30, 0, days) > 0',
        ),
        ('shop', 'FINE_TUNING', where.format("orders.quantity IN ('a', 'b')")),
        ('shop', 'FINE_TUNING', where.format('orders.quantity > true')),
        ('shop', 'FINE_TUNING', where.format('orders.quantity > NULL')),
        ('shop', 'FINE_TUNING', where.format('orders.quantity IS IN (1, 2)')),
        ('shop', 'FINE_TUNING', where.format('orders.quantity IN 1')),
        ('shop', 'FINE_TUNING', where.format("orders.amount CONTAINS 'x'")),
        ('shop', 'FINE_TUNING', where.format('orders.quantity = (1, 2)')),
        (
            'shop',
            'FINE_TUNING',
            where.format('orders.quantity > 9223372036854775808'),
        ),
        (
            'shop',
            'FINE_TUNING',
            f'PREDICT LIST_DISTINCT(orders.user_id, 0, 30, days) RANK TOP 3 {each}',
        ),
    ]:
        c.add('coverage', query, schema, profile, refused=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    corpus = build()
    valid = [c for c in corpus.cases if 'expected_tree' in c]
    invalid = [c for c in corpus.cases if 'expected_errors' in c]
    spans = [c for c in invalid if any(e['span'] for e in c['expected_errors'])]
    print(
        f'{len(valid)} valid, {len(invalid)} invalid ({len(spans)} with a span)'
    )
    files = {}
    for kind, cases in (('valid', valid), ('invalid', invalid)):
        files[CORPUS / f'{kind}.json'] = (
            json.dumps({'cases': cases}, indent=1, ensure_ascii=False) + '\n'
        )
    if args.check:
        stale = [
            p for p, t in files.items() if not p.exists() or p.read_text() != t
        ]
        for path in stale:
            print(f'differs: {path.relative_to(REPO)}', file=sys.stderr)
        return 1 if stale else 0
    CORPUS.mkdir(parents=True, exist_ok=True)
    for path, text in files.items():
        path.write_text(text, encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
