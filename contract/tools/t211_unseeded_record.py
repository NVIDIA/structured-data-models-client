#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the properties of T2.11's unseeded cases into
``contract/t211-unseeded-vectors.json``.

Written by T2.11's test author, before the implementation. An unseeded
sample (``random_seed=None``) draws anew on every call, on every database
(DuckDB's sample without ``REPEATABLE``, SQLite's ``ORDER BY RANDOM()``,
Postgres's ``random()`` without ``setseed``), so no two runs send the same
request and no line of the reference can be compared row by row. The owner
judges it by properties (KO2-76): the right number of rows, every sampled
row in the source data, and the same rows when a seed is given
(``sample.seed-7``, a gate case). Run with the reference checkout's
interpreter, from this repository's root::

    ~/code/sdmc-reference/.venv/bin/python -I \
        contract/tools/t211_unseeded_record.py --reference ~/code/sdmc-reference

Postgres needs ``SDM_POSTGRES_TEST_DSN``, as ``run_python.py`` does.

The cases are ``contract/tests/t211-unseeded/*.json``, kept out of
``contract/cases/`` since the gate compares every request it runs row by
row. Each runs on each database twice through ``run_python.py --mode
python-ref``; per case key this records what does not change from draw to
draw, from the first run, and checks it against the second:

- ``outcome``: the result, stage, class and code;
- ``schema``: the request's ``schema`` section;
- ``predict``: the request's ``predict.instance_table`` (the entities
  asked about at the anchor), and the row count of each of its related
  tables;
- ``context_rows``: how many context examples were drawn;
- ``context_tables``: the related tables of the context;
- ``result``: the result's projection (its columns, rows and digest);
- ``draws_differ``: whether the two runs drew different context rows
  (each case must, or it does not exercise an unseeded draw).

And per case, from the case's DuckDB base: ``source``, each table's
feature columns' values, as the request writes them, which every sampled
row must be one of; ``entities``, the entity keys. The labels of each run's
context examples are checked here against the source (``label``), so the
tests' label rule is the reference's.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / 'contract/tests/t211-unseeded'
VECTORS = ROOT / 'contract/t211-unseeded-vectors.json'
RUNNER = ROOT / 'contract/runners/run_python.py'
DATABASES = ('duckdb', 'sqlite', 'postgres')
#: Each sampled table's feature columns: what a request carries of a row
#: beside its instance and node ids and the keys it is linked by.
FEATURES = {
    'users': ('user_id', 'age', 'country'),
    'orders': ('price', 'ts'),
    'lines': ('qty',),
}


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('t211_record_run_python', RUNNER)


def written(value: Any) -> Any:
    r"""A value as a request writes it, read back: a number as a float, a
    timestamp as its ISO text with microseconds and a Z, text as itself."""
    if isinstance(value, dict) and set(value) == {'number'}:
        return float(value['number'])
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, datetime.datetime):
        return value.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    return value


def source(
    case: dict[str, Any],
) -> tuple[dict[str, list[list[Any]]], list[Any]]:
    r"""The case's DuckDB database, its base and setup run: each sampled
    table's feature rows, as a request writes them, and the entity keys."""
    import duckdb

    connection = duckdb.connect()
    try:
        for statement in [
            *runner().load_bases()[case['input']['base']],
            *case['input'].get('setup', []),
        ]:
            connection.execute(statement)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'main'"
            ).fetchall()
        }
        rows = {
            table: sorted(
                [written(v) for v in row]
                for row in connection.execute(
                    f'SELECT {", ".join(columns)} FROM {table}'
                ).fetchall()
            )
            for table, columns in FEATURES.items()
            if table in tables
        }
        entities = [
            written(row[0])
            for row in connection.execute(
                'SELECT user_id FROM users ORDER BY user_id'
            ).fetchall()
        ]
    finally:
        connection.close()
    return rows, entities


def label(case: dict[str, Any], entity: Any, anchor: str) -> float:
    r"""The label of ``entity`` at ``anchor`` read from the source: the
    query's SUM(orders.price, 0, 30, days), the orders after the anchor up
    to 30 days later."""
    import duckdb

    assert case['input']['pql'].startswith('PREDICT SUM(orders.price, 0, 30')
    connection = duckdb.connect()
    try:
        for statement in [
            *runner().load_bases()[case['input']['base']],
            *case['input'].get('setup', []),
        ]:
            connection.execute(statement)
        found = connection.execute(
            'SELECT COALESCE(SUM(price), 0) FROM orders WHERE user_id = ? '
            'AND ts > CAST(? AS TIMESTAMP) '
            "AND ts <= CAST(? AS TIMESTAMP) + INTERVAL '30 days'",
            [int(entity), anchor[:26], anchor[:26]],
        ).fetchone()[0]
    finally:
        connection.close()
    return float(found)


def sent(line: dict[str, Any]) -> dict[str, Any]:
    r"""The one request a line's endpoint received, parsed."""
    payloads = line['public']['payloads']
    assert len(payloads) == 1, len(payloads)
    return payloads[0]['parsed']


def properties(line: dict[str, Any]) -> dict[str, Any]:
    r"""What an unseeded line holds that does not change from draw to
    draw."""
    outcome = line['outcome']
    found: dict[str, Any] = {
        'outcome': {
            'result': outcome['result'],
            'stage': outcome['stage'],
            'error': outcome.get('error'),
            'code': outcome.get('code'),
        }
    }
    if outcome['result'] != 'success':
        return found
    request = sent(line)
    predict = request['predict']
    found.update(
        {
            'schema': request['schema'],
            'predict': {
                'instance_table': predict['instance_table'],
                'rows': {
                    table: len(held['rows'])
                    for table, held in sorted(predict['related_tables'].items())
                },
            },
            'context_rows': len(request['context']['instance_table']['rows']),
            'context_tables': sorted(request['context']['related_tables']),
            'result': line['public']['result'],
        }
    )
    return found


def examples(line: dict[str, Any]) -> list[tuple[Any, str, Any]]:
    r"""Each context example of a line's request: its entity, anchor time
    and label."""
    table = sent(line)['context']['instance_table']
    at = {name: i for i, name in enumerate(table['columns'])}
    return [
        (
            written(row[at['__kumo_entity_ref_0']]),
            row[at['__kumo_anchor_time']],
            written(row[at['TARGET']]),
        )
        for row in table['rows']
    ]


def run(reference: Path, database: str, out: Path) -> dict[str, Any]:
    done = subprocess.run(
        [
            sys.executable,
            '-I',
            str(RUNNER),
            '--mode',
            'python-ref',
            '--reference',
            str(reference),
            '--database',
            database,
            '--out',
            str(out),
            str(CASES),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        check=False,
        timeout=1800,
    )
    if done.returncode != 0:
        raise SystemExit(done.stdout + done.stderr)
    return {
        json.loads(line)['case']: json.loads(line)
        for line in out.read_text(encoding='utf-8').splitlines()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--database', nargs='+', default=list(DATABASES))
    args = parser.parse_args()
    if 'postgres' in args.database and not os.environ.get(
        'SDM_POSTGRES_TEST_DSN'
    ):
        raise SystemExit('Postgres needs SDM_POSTGRES_TEST_DSN')
    loaded = {
        case['id']: case
        for case, _ in runner().load_cases(sorted(CASES.glob('*.json')))
    }
    document = (
        json.loads(VECTORS.read_text(encoding='utf-8'))
        if VECTORS.exists()
        else {}
    )
    document['about'] = __doc__.split('\n\n')[0].replace('\n', ' ')
    document['format'] = 't211-unseeded-vectors/1'
    document.setdefault('cases', {})
    document.setdefault('keys', {})
    for case_id, case in sorted(loaded.items()):
        rows, entities = source(case)
        document['cases'][case_id] = {'source': rows, 'entities': entities}
    with tempfile.TemporaryDirectory(prefix='t211-record-') as tmp:
        for database in args.database:
            first = run(args.reference, database, Path(tmp) / 'a.jsonl')
            second = run(args.reference, database, Path(tmp) / 'b.jsonl')
            for key, line in sorted(first.items()):
                case = loaded[key.split('@')[0]]
                held = properties(line)
                assert properties(second[key]) == held, key
                if held['outcome']['result'] == 'success':
                    drawn = examples(line)
                    for entity, anchor, target in drawn:
                        assert target == label(case, entity, anchor), (
                            key,
                            entity,
                            anchor,
                            target,
                        )
                    held['draws_differ'] = drawn != examples(second[key])
                    assert held['draws_differ'], f'{key}: one draw twice'
                document['keys'][key] = held
                print(key, held['outcome']['result'], held.get('context_rows'))
    VECTORS.write_text(json.dumps(document, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
