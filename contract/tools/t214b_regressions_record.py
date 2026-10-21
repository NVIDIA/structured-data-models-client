#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records what the reference gives for T2.14b's fix-round regressions into
``contract/t214b-regression-vectors.json``.

Written by T2.14b's test author after review round 2, from the reviewers'
probes. Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root, with a Postgres server for the Postgres scenario::

    SDM_POSTGRES_TEST_DSN=... <reference venv>/bin/python -I \
        contract/tools/t214b_regressions_record.py --reference <checkout>

Each scenario's public calls, on the routes the interpreter's client takes
(:func:`observe`, which ``contract/tests/test_t214b_regressions.py`` runs on
python-core with the core routes forced):

- ``candidates``: a ranking over date-like keys given repeated or equal
  temporal candidates (:data:`CANDIDATES`): equal dates, aware timestamps
  of one instant at two offsets (one key of ``dict.fromkeys``), and a naive
  timestamp beside an aware one (two keys): how many questions the request
  is asked as, each question asked, in order, and the rankings, or the
  refusal;
- ``stype``: a column's meaning set to ``5``: the refusal's class, code
  and message;
- ``where``: a filter inside an aggregation reading a double past
  float32's range: the outcome and each warning's class and text (pandas'
  ``RuntimeWarning`` among them, FX-1);
- ``json``: a Postgres ``json`` column whose documents escape a lone
  surrogate, through ``from_source``, ``materialize`` and ``predict``: the
  outcome, and each request the endpoint received (its request id left
  out), so a document is sent as the reference sends it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import re
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/t214b-regression-vectors.json'
FORMAT = 'sdm-t214b-regression-vectors/1'
UTC = dt.timezone.utc
PLUS_ONE = dt.timezone(dt.timedelta(hours=1))

#: How the ranking's keys are written, by scenario family: a date's text,
#: or an aware timestamp's text at UTC.
KEY_TEXT = {
    'date': '%Y-%m-%d',
    'stamp': '%Y-%m-%d 00:00:00+00:00',
}
#: The candidate lists, by name: the key text they are asked over, and the
#: candidates.
CANDIDATES: dict[str, tuple[str, Any]] = {
    'dates-repeated': (
        'date',
        lambda: [dt.date(2023, 1, 1), dt.date(2023, 1, 1), dt.date(2023, 1, 2)],
    ),
    'dates-distinct': (
        'date',
        lambda: [dt.date(2023, 1, 1), dt.date(2023, 1, 2)],
    ),
    'aware-one-instant': (
        'stamp',
        lambda: [
            dt.datetime(2023, 1, 1, tzinfo=UTC),
            dt.datetime(2023, 1, 1, 1, tzinfo=PLUS_ONE),
            dt.datetime(2023, 1, 2, tzinfo=UTC),
        ],
    ),
    'aware-distinct': (
        'stamp',
        lambda: [
            dt.datetime(2023, 1, 1, tzinfo=UTC),
            dt.datetime(2023, 1, 2, tzinfo=UTC),
        ],
    ),
    'naive-beside-aware': (
        'stamp',
        lambda: [
            dt.datetime(2023, 1, 1),
            dt.datetime(2023, 1, 1, tzinfo=UTC),
            dt.datetime(2023, 1, 2, tzinfo=UTC),
        ],
    ),
}
#: The Postgres scenario's documents, one per row in turn: a lone high
#: surrogate in a value, a lone low one in a key, one in a list beside a
#: pair, and none.
DOCUMENTS = (
    '{"s":"\\ud800"}',
    '{"\\udc00":"x"}',
    '["\\ud800", "\\ud83d\\ude00"]',
    '{"s":"ok"}',
)
JSON_TABLE = 't214b_json_lone_surrogate'
#: The code a core warning's text is prefixed with on python-core.
CODE_PREFIX = re.compile(r'^\[[A-Z][A-Z0-9_]*\] ')


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('t214b_run_python', ROOT / 'contract/runners/run_python.py')


def host() -> ModuleType:
    r"""T2.14b's host-facts recorder, whose shop, ranking and endpoint
    these scenarios reuse."""
    return _module(
        't214b_host_record', ROOT / 'contract/tools/t214b_host_record.py'
    )


def _refusal(error: BaseException) -> dict[str, Any]:
    return {
        'error': type(error).__name__,
        'code': getattr(error, 'code', None),
        'message': str(error),
    }


def _keyed(kind: str) -> Any:
    r"""The shop, the ranking's items keyed by text of ``kind``."""
    connection = host()._connection()
    text = KEY_TEXT[kind]
    for table in ('items', 'buys'):
        connection.execute(
            f'ALTER TABLE {table} ALTER item_id TYPE VARCHAR USING '
            f"strftime(DATE '2023-01-01' + item_id::INTEGER, '{text}')"
        )
    return connection


def candidates() -> dict[str, Any]:
    r"""Each candidate list's parts, questions and rankings."""
    out = {}
    for name, (kind, made) in CANDIDATES.items():
        connection = _keyed(kind)
        try:
            schema = host()._schema(connection)
            try:
                request = host()._materialize(
                    schema, host().RANKING, candidates=made()
                )
            except Exception as error:  # noqa: BLE001 - recorded
                out[name] = {'refusal': _refusal(error), 'parts': None}
                continue
            answer = host()._predicted(request, probabilities=True)
            out[name] = {
                'parts': len(request.parts or ()),
                'asked': answer['asked'],
                'rankings': answer.get('rankings'),
                'refusal': answer.get('refusal'),
            }
        finally:
            connection.close()
    return out


def stype() -> dict[str, Any]:
    r"""A column's meaning set to ``5``."""
    connection = host()._connection()
    try:
        schema = host()._schema(connection)
        try:
            schema['users']['age'].stype = 5
        except Exception as error:  # noqa: BLE001 - recorded
            return {'refusal': _refusal(error)}
        return {'refusal': None}
    finally:
        connection.close()


def where() -> dict[str, Any]:
    r"""An aggregation's filter reading a double past float32's range."""
    import pandas as pd
    from sdm_client import PredictiveQuery

    connection = host()._connection()
    connection.execute('ALTER TABLE orders ADD COLUMN large DOUBLE')
    connection.execute('UPDATE orders SET large = 1e39')
    out: dict[str, Any] = {'refusal': None}
    try:
        schema = host()._schema(connection)
        with warnings.catch_warnings(record=True) as shown:
            warnings.simplefilter('always')
            try:
                PredictiveQuery(
                    'PREDICT SUM(orders.price WHERE orders.large > 0, 0, 30, '
                    'days) FOR EACH users.user_id',
                    schema,
                ).materialize(
                    indices=[0, 1],
                    anchor_time=pd.Timestamp(host().ANCHOR),
                    random_seed=42,
                )
            except Exception as error:  # noqa: BLE001 - recorded
                out['refusal'] = _refusal(error)
        # Each by its words, a core code's prefix left out (E41(i), E59(d)),
        # as option_record.py holds a warning.
        out['runtime_warnings'] = [
            CODE_PREFIX.sub('', str(w.message))
            for w in shown
            if issubclass(w.category, RuntimeWarning)
        ]
    finally:
        connection.close()
    return out


def json_documents(dsn: str) -> dict[str, Any]:
    r"""The Postgres ``json`` column's documents through predict."""
    import psycopg
    from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient
    from sdm_connectors import PostgresSource

    out: dict[str, Any] = {'refusal': None, 'bodies': []}
    with psycopg.connect(dsn) as connection:
        connection.execute(f'DROP TABLE IF EXISTS {JSON_TABLE}')
        connection.execute(
            f'CREATE TABLE {JSON_TABLE} '
            '(id BIGINT PRIMARY KEY, age INTEGER, doc JSON)'
        )
        for i in range(8):
            connection.execute(
                f'INSERT INTO {JSON_TABLE} VALUES (%s, %s, %s::json)',
                (i, 20 + i, DOCUMENTS[i % len(DOCUMENTS)]),
            )
        connection.commit()
        try:
            with (
                warnings.catch_warnings(),
                host().endpoint().serve() as (url, received),
            ):
                warnings.simplefilter('ignore')
                try:
                    schema = Schema.from_source(
                        PostgresSource(connection),
                        tables={JSON_TABLE: JSON_TABLE},
                    )
                    request = PredictiveQuery(
                        f'PREDICT {JSON_TABLE}.age FOR EACH {JSON_TABLE}.id',
                        schema,
                    ).materialize(indices=[0, 1], random_seed=42)
                    with SDMClient(url) as client:
                        client.predict(request, model=KumoRelational())
                except Exception as error:  # noqa: BLE001 - recorded
                    out['refusal'] = _refusal(error)
                for _, _, body in received.requests:
                    parsed = json.loads(body)
                    parsed.get('metadata', {}).pop('request_id', None)
                    out['bodies'].append(parsed)
        finally:
            connection.rollback()
            connection.execute(f'DROP TABLE IF EXISTS {JSON_TABLE}')
            connection.commit()
    return out


def observe(core: bool, dsn: str | None) -> dict[str, Any]:
    r"""Every scenario, on the core where ``core`` (each public call's
    DuckDB or Postgres paths forced, as ``run_python.py`` forces a case's
    routes); the Postgres one only where ``dsn`` names a server."""
    if core:
        from sdm_client import _routing

        os.environ['SDM_CORE_ROUTES'] = ','.join(
            _routing.FROM_SOURCE_DUCKDB
            + _routing.MATERIALIZE_DUCKDB
            + _routing.PREDICT_DUCKDB
            + _routing.FROM_SOURCE_POSTGRES_TABLES
            + _routing.MATERIALIZE_POSTGRES
            + _routing.PREDICT_POSTGRES
        )
    try:
        return {
            'candidates': candidates(),
            'stype': stype(),
            'where': where(),
            'json': None if dsn is None else json_documents(dsn),
        }
    finally:
        os.environ.pop('SDM_CORE_ROUTES', None)


def load() -> dict[str, Any]:
    r"""The vector file, as committed."""
    return json.loads(OUT.read_text(encoding='utf-8'))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--reference', type=Path, required=True)
    arguments = parser.parse_args(argv)
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        print('SDM_POSTGRES_TEST_DSN names no server', file=sys.stderr)
        return 1
    run = runner()
    guard = run._Guard(arguments.reference)
    guard.check('before the first case')
    run._require_reference_client(arguments.reference)
    commit = run._git(arguments.reference)
    if not commit['clean']:
        print('the reference checkout is not clean', file=sys.stderr)
        return 1
    document = {
        'format': FORMAT,
        'about': (
            "What the reference's public calls give for T2.14b's fix-round "
            'regressions: repeated and equal temporal ranking candidates, a '
            "column's meaning set to a number, an aggregation's filter "
            "reading a double past float32's range, and Postgres json "
            'documents escaping lone surrogates through predict. '
            'contract/tests/test_t214b_regressions.py holds python-core to '
            'them, observed by the same code (observe).'
        ),
        'recorded_by': 'contract/tools/t214b_regressions_record.py',
        'reference': commit['sha'],
        **observe(core=False, dsn=dsn),
    }
    guard.check('after the last case')
    OUT.write_text(
        json.dumps(document, indent=1, ensure_ascii=True) + '\n',
        encoding='utf-8',
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
