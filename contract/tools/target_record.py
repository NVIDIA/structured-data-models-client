#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference's labels for T2.10's PQL targets into
``contract/t210-target-vectors.json``.

Written by T2.10's test author, before the implementation. Run with the
reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root, a database at a time or all three::

    ~/code/sdmc-reference/.venv/bin/python -I contract/tools/target_record.py \
        --reference ~/code/sdmc-reference [--database duckdb sqlite postgres]

Postgres needs ``SDM_POSTGRES_TEST_DSN``, as ``run_python.py`` does.

The core refuses a target it does not take yet, ``UNSUPPORTED_REQUEST``
naming the rule (``sdm-core``'s ``materialize.rs``, ``shape``):
``aggregation`` (an aggregation other than ``SUM``, ``COUNT``, ``AVG``,
``MIN`` and ``MAX``), ``target_filter`` (a filter inside the target's
aggregation), ``open_window`` (an aggregation with no window start),
``static_target`` (a static column of another table than the entity's),
``condition`` (a condition other than a comparison with a number or
``NULL``, of a numeric column or an aggregation) and ``logical_target``
(``AND``, ``OR`` and ``NOT``). Each :data:`SCENARIOS` entry is one target
under one rule over the contract's ``shop`` base (users and their orders,
a few columns added by its setup), anchored early in the data so its
context is a few rows: the reference's labels for it are what the core
must give once the rule is lifted. ``open_window`` and ``static_target``
are refused by the query validator on both routes before the rule is
reached from the public calls (the refusal census's probes say so), so
theirs are held where the reference holds them.

Each scenario runs on each database as ``run_python.py`` runs a case (its
``_open``, ``_source`` and arguments; Schema.from_source, then
``materialize``, and through ``predict`` against the scripted endpoint
where the scenario's ``until`` says so, answering probabilities where it
says so). What is kept, per scenario and database (``observe``):

- ``outcome``: the result, the call it ended in, the class and
  ``StructuredDataError.code`` (no message text, H17);
- ``payloads``: each request ``materialize`` built (one per question; a
  ranking's, one per candidate), its ``task`` whole and its context's and
  predicted rows' instance tables (``context`` and ``predict``: the
  columns and the rows, the labels in ``TARGET``), the related tables left
  out: the labels, and what they are labels of;
- ``calls``: how many rows each call the endpoint received scores;
- ``result``: the answer through ``to_pandas()`` (its columns and rows).

A scenario with a ``twin`` names a column the reference cannot say (a
dot): the reference leaves the column out and refuses the query; its twin
is the same data and query with the column named ``i_d``. Column names
should not matter (KO2-36), so the core's answer for it is the twin's.

:func:`observe` is imported by ``contract/tests/test_t210_targets.py``, so
``python-core`` is observed by the same code.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/t210-target-vectors.json'
CASES = ROOT / 'contract/cases/s1'
FORMAT = 'sdm-t210-target-vectors/1'
DATABASES = ('duckdb', 'sqlite', 'postgres')
#: The core's rules this package lifts, as ``materialize.rs`` names them.
RULES = (
    'aggregation',
    'open_window',
    'condition',
    'logical_target',
    'target_filter',
    'static_target',
)
#: The anchor: two months into the orders, so a context is a few rows.
ANCHOR = '2023-03-01'
#: The shop's anchor in the contract's other cases (base.regression's).
SHOP_ANCHOR = '2024-03-01'
#: A case's routes: every call's on DuckDB (``base.regression``'s).
ROUTES_FROM = CASES / 'base.regression.json'

ITEMS = (
    'CREATE TABLE items AS SELECT i::{kind} AS {key}, '
    "['a', 'b', 'c', 'd'][1 + i] AS kind FROM range(4) t(i)"
)
BUYS = (
    'CREATE TABLE buys AS SELECT i::BIGINT AS buy_id, (i % 8)::BIGINT AS '
    'user_id, ((i * 3) % 4)::{kind} AS {key}, '
    "TIMESTAMP '2023-01-01' + to_days((i * 5)::INTEGER) AS ts "
    'FROM range(120) t(i)'
)


def bought(key: str = 'item_id', kind: str = 'BIGINT') -> list[str]:
    r"""Items and the users' buys of them, each buy naming its item by
    ``key`` (items' primary key by name)."""
    return [ITEMS.format(key=key, kind=kind), BUYS.format(key=key, kind=kind)]


def declared(key: str) -> list[str]:
    r"""The same items and buys with items' primary key, ``key``, and
    buys' reference to it declared, so a name the key inference does not
    read as a key's is one."""
    return [
        f'CREATE TABLE items ({key} BIGINT PRIMARY KEY, kind VARCHAR)',
        "INSERT INTO items SELECT i, ['a', 'b', 'c', 'd'][1 + i] "
        'FROM range(4) t(i)',
        'CREATE TABLE buys (buy_id BIGINT PRIMARY KEY, user_id BIGINT, '
        f'{key} BIGINT REFERENCES items ({key}), ts TIMESTAMP)',
        'INSERT INTO buys SELECT i, i % 8, (i * 3) % 4, '
        "TIMESTAMP '2023-01-01' + to_days((i * 5)::INTEGER) "
        'FROM range(120) t(i)',
    ]


CHANNEL = (
    'CREATE OR REPLACE TABLE orders AS SELECT *, '
    "['web', 'app', 'shop'][1 + order_id % 3] AS channel FROM orders"
)
VIP = (
    'CREATE OR REPLACE TABLE users AS SELECT *, '
    '(user_id % 2 = 0) AS vip FROM users'
)
AMOUNT = (
    'CREATE OR REPLACE TABLE orders AS SELECT *, '
    '(price + 0.25)::DECIMAL(10,2) AS amount FROM orders'
)
EACH = 'FOR EACH users.user_id'
WINDOW = '0, 30, days'


def _scenario(
    rule: str,
    pql: str,
    setup: list[str] | None = None,
    anchor: bool = True,
    **more: Any,
) -> dict[str, Any]:
    materialize: dict[str, Any] = {'indices': [0, 1]}
    if anchor:
        materialize['anchor_time'] = ANCHOR
    materialize.update(more.pop('materialize', {}))
    return {
        'rule': rule,
        'setup': list(setup or []),
        'pql': pql,
        'materialize': materialize,
        'until': more.pop('until', 'materialize'),
        **more,
    }


#: The scenarios, by id. Those whose case is named in :data:`CASE_OF` are
#: also S1 cases (``contract/cases/s1/<case>.json``), run through
#: ``predict``.
SCENARIOS: dict[str, dict[str, Any]] = {
    # aggregation: LIST_DISTINCT of a foreign key, a temporal link
    # prediction (FIRST, LAST and COUNT_DISTINCT the validator refuses).
    'aggregation.list-distinct': _scenario(
        'aggregation',
        f'PREDICT LIST_DISTINCT(buys.item_id, {WINDOW}) {EACH}',
        bought(),
    ),
    'aggregation.list-distinct.users': _scenario(
        'aggregation',
        f'PREDICT LIST_DISTINCT(buys.user_id, {WINDOW}) FOR EACH items.item_id',
        bought(),
    ),
    # condition: a text or boolean column, membership (IN), an aggregation's
    # membership.
    'condition.text': _scenario(
        'condition', f"PREDICT users.country = 'US' {EACH}", anchor=False
    ),
    'condition.text-not-equal': _scenario(
        'condition', f"PREDICT users.country != 'US' {EACH}", anchor=False
    ),
    'condition.text-in': _scenario(
        'condition',
        f"PREDICT users.country IN ('US', 'DE') {EACH}",
        anchor=False,
    ),
    'condition.in': _scenario(
        'condition', f'PREDICT users.age IN (22, 23) {EACH}', anchor=False
    ),
    'condition.bool': _scenario(
        'condition', f'PREDICT users.vip = True {EACH}', [VIP], anchor=False
    ),
    'condition.aggregation-in': _scenario(
        'condition', f'PREDICT COUNT(orders.*, {WINDOW}) IN (1, 2) {EACH}'
    ),
    # logical_target: AND, OR and NOT, of windowed and of static conditions.
    # The three of these that are S1 cases are written in the canonical
    # text the Python host hands the core (each operand of AND and NOT
    # parenthesized), which the gate's inputs rule holds to the case's pql.
    'logical.and': _scenario(
        'logical_target',
        f'PREDICT (COUNT(orders.*, {WINDOW}) > 0) AND '
        f'(SUM(orders.price, {WINDOW}) > 20) {EACH}',
    ),
    'logical.or': _scenario(
        'logical_target',
        f'PREDICT SUM(orders.price, {WINDOW}) > 21 OR '
        f'SUM(orders.price, {WINDOW}) < 18 {EACH}',
    ),
    'logical.not': _scenario(
        'logical_target', f'PREDICT NOT (COUNT(orders.*, {WINDOW}) > 0) {EACH}'
    ),
    'logical.static-and': _scenario(
        'logical_target',
        f'PREDICT users.age > 22 AND users.age < 26 {EACH}',
        anchor=False,
    ),
    'logical.and-or': _scenario(
        'logical_target',
        f'PREDICT COUNT(orders.*, {WINDOW}) > 0 AND '
        f'SUM(orders.price, {WINDOW}) > 20 OR '
        f'SUM(orders.price, {WINDOW}) = 17.5 {EACH}',
    ),
    # target_filter: a filter inside the target's aggregation. COUNT of a
    # single comparison (filter.count, filter.key) is the form a ranking's
    # questions take, which T2.8 lifts; every other form is T2.10's.
    'filter.count': _scenario(
        'target_filter',
        f'PREDICT COUNT(orders.* WHERE orders.price > 19, {WINDOW}) {EACH}',
    ),
    'filter.key': _scenario(
        'target_filter',
        f'PREDICT COUNT(buys.* WHERE buys.item_id = 1, {WINDOW}) > 0 {EACH}',
        bought(),
    ),
    'filter.sum': _scenario(
        'target_filter',
        f'PREDICT SUM(orders.price WHERE orders.price > 19, {WINDOW}) {EACH}',
    ),
    'filter.avg': _scenario(
        'target_filter',
        f'PREDICT AVG(orders.price WHERE orders.price > 19, {WINDOW}) {EACH}',
    ),
    'filter.max': _scenario(
        'target_filter',
        f'PREDICT MAX(orders.price WHERE orders.price < 21, {WINDOW}) {EACH}',
    ),
    'filter.and': _scenario(
        'target_filter',
        'PREDICT COUNT(orders.* WHERE (orders.price > 18) AND '
        f'(orders.price < 21), {WINDOW}) {EACH}',
    ),
    'filter.or': _scenario(
        'target_filter',
        'PREDICT COUNT(orders.* WHERE orders.price > 19 OR '
        f'orders.price < 18, {WINDOW}) > 0 {EACH}',
    ),
    'filter.not': _scenario(
        'target_filter',
        f'PREDICT COUNT(orders.* WHERE NOT orders.price > 19, {WINDOW}) {EACH}',
    ),
    'filter.in': _scenario(
        'target_filter',
        'PREDICT COUNT(orders.* WHERE orders.price IN (17.5, 22), '
        f'{WINDOW}) {EACH}',
    ),
    'filter.not-null': _scenario(
        'target_filter',
        f'PREDICT COUNT(orders.* WHERE orders.price != NULL, {WINDOW}) {EACH}',
    ),
    'filter.text': _scenario(
        'target_filter',
        "PREDICT COUNT(orders.* WHERE orders.channel = 'web', "
        f'{WINDOW}) {EACH}',
        [CHANNEL],
    ),
    'filter.text-in': _scenario(
        'target_filter',
        "PREDICT SUM(orders.price WHERE orders.channel IN ('web', 'app'), "
        f'{WINDOW}) {EACH}',
        [CHANNEL],
    ),
    'filter.decimal': _scenario(
        'target_filter',
        f'PREDICT COUNT(orders.* WHERE orders.amount > 19.5, {WINDOW}) {EACH}',
        [AMOUNT],
    ),
    # open_window and static_target: refused by the validator first, so no
    # row is read; asked at the shop's anchor, 2024-03-01, as
    # refuse.unknown-column, whose core calls their cases' are.
    'open-window': _scenario(
        'open_window',
        f'PREDICT SUM(orders.price, -INF, 30, days) {EACH}',
        materialize={'anchor_time': SHOP_ANCHOR},
    ),
    'static.other-table': _scenario(
        'static_target',
        'PREDICT users.age FOR EACH orders.order_id',
        materialize={'anchor_time': SHOP_ANCHOR},
    ),
    # Rankings: each candidate's question filters by the ranked key (T2.8's
    # single comparison), here a DECIMAL key and a key named with a dot.
    'rank.decimal-key': _scenario(
        'target_filter',
        f'PREDICT LIST_DISTINCT(buys.item_id, {WINDOW}) RANK TOP 2 {EACH}',
        bought(kind='DECIMAL(18,0)'),
        materialize={'candidates': [0, 1, 2]},
        probabilities=True,
        until='predict',
    ),
    'rank.odd-column': _scenario(
        'target_filter',
        f'PREDICT LIST_DISTINCT(buys.`i.d`, {WINDOW}) RANK TOP 2 {EACH}',
        declared('"i.d"'),
        materialize={'candidates': [0, 1, 2]},
        probabilities=True,
        until='predict',
        twin='rank.odd-column.twin',
    ),
    'rank.odd-column.twin': _scenario(
        'target_filter',
        f'PREDICT LIST_DISTINCT(buys.i_d, {WINDOW}) RANK TOP 2 {EACH}',
        declared('i_d'),
        materialize={'candidates': [0, 1, 2]},
        probabilities=True,
        until='predict',
    ),
}
#: The scenarios that are S1 cases, by case id.
CASE_OF = {
    'target.aggregation.list-distinct': 'aggregation.list-distinct',
    'target.condition.text': 'condition.text',
    'target.condition.in': 'condition.in',
    'target.logical.and': 'logical.and',
    'target.logical.not': 'logical.not',
    'target.filter.sum': 'filter.sum',
    'target.filter.and': 'filter.and',
    'target.filter.text-in': 'filter.text-in',
    'target.open-window': 'open-window',
    'target.static.other-table': 'static.other-table',
    'rank.decimal-key': 'rank.decimal-key',
    'rank.odd-column': 'rank.odd-column',
}
#: The twin's name for the column a scenario's name cannot say.
TWIN_NAME = 'i_d'
ODD_NAME = 'i.d'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('t210_run_python', ROOT / 'contract/runners/run_python.py')


def endpoint() -> ModuleType:
    return _module(
        'scripted_endpoint', ROOT / 'contract/runners/scripted_endpoint.py'
    )


def key(scenario_id: str, database: str) -> str:
    r"""A vector's key: ``<scenario>@<database>``."""
    return f'{scenario_id}@{database}'


def as_case(scenario_id: str) -> dict[str, Any]:
    r"""The scenario as the runner reads a case: its id, base, setup,
    query, arguments, endpoint and calls (every route of DuckDB's,
    ``base.regression``'s)."""
    spec = SCENARIOS[scenario_id]
    given: dict[str, Any] = {'base': 'shop'}
    if spec['setup']:
        given['setup'] = list(spec['setup'])
    given['pql'] = spec['pql']
    given['materialize'] = json.loads(json.dumps(spec['materialize']))
    if spec.get('probabilities'):
        given['probabilities'] = True
    routes = json.loads(ROUTES_FROM.read_text(encoding='utf-8'))['routes']
    return {
        'id': scenario_id,
        'input': given,
        'until': spec['until'],
        'routes': routes,
    }


def _cell(value: Any) -> Any:
    r"""A result cell as JSON writes it: a NumPy scalar as its Python
    value, a list as a list, a missing value as null."""
    if hasattr(value, 'tolist') and not isinstance(value, (str, bytes)):
        value = value.tolist()
    elif hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, list):
        return [_cell(item) for item in value]
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def projection(result: Any) -> dict[str, Any] | None:
    r"""A ``PredictionResult`` through ``to_pandas()``: its columns, row
    count and rows."""
    if result is None or not hasattr(result, 'to_pandas'):
        return None
    frame = result.to_pandas()
    rows = [
        [_cell(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    return {
        'columns': [str(name) for name in frame.columns],
        'count': len(rows),
        'rows': rows,
    }


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def labels(payload: Any) -> dict[str, Any]:
    r"""What one request says of its labels: the task whole, and the
    context's and predicted rows' instance tables (the related tables, the
    features, left out)."""
    payload = _plain(payload)
    kept: dict[str, Any] = {'task': payload.get('task')}
    for part in ('context', 'predict'):
        body = payload.get(part)
        kept[part] = (
            body.get('instance_table') if isinstance(body, dict) else None
        )
    return kept


def _rows_sent(body: bytes) -> int | None:
    try:
        rows = json.loads(body)['predict']['instance_table']['rows']
    except (ValueError, KeyError, TypeError):
        return None
    return len(rows) if isinstance(rows, list) else None


def observe(
    scenario_id: str, database: str, core: bool = False
) -> dict[str, Any]:
    r"""The scenario's public calls on ``database``: on the core where
    ``core`` (``SDM_CORE_ROUTES`` set to the scenario's routes there, as
    ``run_python.py`` sets them for ``python-core``), else on the routes
    the interpreter's client takes."""
    from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient

    run = runner()
    case = as_case(scenario_id)
    given = case['input']
    arguments = run._materialize_arguments(given['materialize'])
    previous = os.environ.get('SDM_CORE_ROUTES')
    if core:
        os.environ['SDM_CORE_ROUTES'] = ','.join(run.routes_on(case, database))
    observed: dict[str, Any] = {
        'outcome': None,
        'payloads': None,
        'calls': [],
        'result': None,
    }
    try:
        connection, cleanup, _ = run._open(case, run.load_bases(), database)
        stage = 'from_source'
        try:
            with (
                endpoint().serve(
                    probabilities=given.get('probabilities', False)
                ) as (url, received),
                warnings.catch_warnings(),
            ):
                warnings.simplefilter('ignore')
                try:
                    schema = Schema.from_source(
                        run._source(database, connection)
                    )
                    stage = 'materialize'
                    request = PredictiveQuery(given['pql'], schema).materialize(
                        **arguments
                    )
                    observed['payloads'] = [
                        labels(body) for body in request.payloads
                    ]
                    if case['until'] == 'predict':
                        stage = 'predict'
                        with SDMClient(url) as client:
                            result = client.predict(
                                request, model=KumoRelational()
                            )
                        observed['result'] = projection(result)
                    observed['outcome'] = run._outcome('success', stage)
                except Exception as error:  # noqa: BLE001 - the outcome
                    observed['outcome'] = run._outcome('refusal', stage, error)
                observed['calls'] = [
                    _rows_sent(body) for _, _, body in received.requests
                ]
        finally:
            cleanup()
    finally:
        if previous is None:
            os.environ.pop('SDM_CORE_ROUTES', None)
        else:
            os.environ['SDM_CORE_ROUTES'] = previous
    return observed


def load() -> dict[str, Any]:
    r"""The vector file, as committed."""
    return json.loads(OUT.read_text(encoding='utf-8'))


def _written(value: Any, depth: int = 0) -> str:
    r"""``value`` as JSON indented one space a level, a list of scalars on
    one line (a row), so the vectors stay small."""
    pad = ' ' * depth
    if isinstance(value, dict) and value:
        items = [
            f'{pad} {json.dumps(name, ensure_ascii=False)}: '
            f'{_written(item, depth + 1)}'
            for name, item in value.items()
        ]
        return '{\n' + ',\n'.join(items) + f'\n{pad}}}'
    if isinstance(value, list) and any(
        isinstance(item, (dict, list)) and item for item in value
    ):
        items = [f'{pad} {_written(item, depth + 1)}' for item in value]
        return '[\n' + ',\n'.join(items) + f'\n{pad}]'
    return json.dumps(value, ensure_ascii=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument(
        '--database', nargs='+', choices=DATABASES, default=list(DATABASES)
    )
    arguments = parser.parse_args(argv)
    run = runner()
    guard = run._Guard(arguments.reference)
    guard.check('before the first scenario')
    run._require_reference_client(arguments.reference)
    commit = run._git(arguments.reference)
    if not commit['clean']:
        print('the reference checkout is not clean', file=sys.stderr)
        return 1
    if OUT.exists():
        document = load()
        if document['reference'] != commit['sha']:
            print('the vectors are of another reference', file=sys.stderr)
            return 1
    else:
        document = {
            'format': FORMAT,
            'about': (
                "The reference's labels for T2.10's PQL targets "
                '(contract/tools/target_record.py SCENARIOS, one target '
                "under one of the core's rules aggregation, open_window, "
                'condition, logical_target, target_filter and '
                'static_target, over the shop base, on DuckDB, SQLite and '
                'Postgres), by <scenario>@<database>: the outcome, each '
                "request's task and its context's and predicted rows' "
                'instance tables (the labels in TARGET), the rows each call '
                'scores and the result through to_pandas(). '
                'contract/tests/test_t210_targets.py holds python-core to '
                'them, observed by the same code (observe).'
            ),
            'recorded_by': 'contract/tools/target_record.py',
            'reference': commit['sha'],
            'vectors': {},
        }
    for database in arguments.database:
        for scenario_id, spec in SCENARIOS.items():
            observed = observe(scenario_id, database)
            document['vectors'][key(scenario_id, database)] = {
                'rule': spec['rule'],
                **observed,
            }
            print(
                f'{key(scenario_id, database)}: {observed["outcome"]}',
                file=sys.stderr,
            )
    guard.check('after the last scenario')
    document['vectors'] = dict(sorted(document['vectors'].items()))
    OUT.write_text(_written(document) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
