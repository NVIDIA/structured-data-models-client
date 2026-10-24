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
    # Codex r3: a zone's repeated hour (fold 0 and fold 1 of New York's
    # 2023-11-05 01:30, which Python holds equal), the same wall time
    # against the UTC instant of its fold 0 (which Python holds unequal),
    # and aware times of day at two offsets (equal). Their keys are the
    # candidates' own texts (``literal``).
    'zone-fold': (
        'literal',
        lambda: [
            dt.datetime(2023, 11, 5, 1, 30, tzinfo=_zone(), fold=0),
            dt.datetime(2023, 11, 5, 1, 30, tzinfo=_zone(), fold=1),
            dt.datetime(2023, 11, 6, 1, 30, tzinfo=_zone()),
        ],
    ),
    'zone-against-utc': (
        'literal',
        lambda: [
            dt.datetime(2023, 11, 5, 1, 30, tzinfo=_zone(), fold=0),
            dt.datetime(2023, 11, 5, 5, 30, tzinfo=UTC),
            dt.datetime(2023, 11, 6, 1, 30, tzinfo=_zone()),
        ],
    ),
    # Codex r6: the same wall time in New York and in Toronto, whose texts
    # are the same and which Python holds unequal (two zones), and a third
    # day: three candidates.
    'same-text-two-zones': (
        'ends',
        lambda: [
            dt.datetime(2023, 11, 5, 1, 30, tzinfo=_zone()),
            dt.datetime(2023, 11, 5, 1, 30, tzinfo=_zone('America/Toronto')),
            dt.datetime(2023, 11, 6, 1, 30, tzinfo=_zone()),
        ],
    ),
    'times-one-instant': (
        'literal',
        lambda: [
            dt.time(0, tzinfo=UTC),
            dt.time(1, tzinfo=PLUS_ONE),
            dt.time(2, tzinfo=UTC),
        ],
    ),
    # Claude r4: a pandas Timestamp and NumPy's datetime64[ns] of the same
    # moment (one key of dict.fromkeys), beside a date.
    'timestamp-beside-datetime64': (
        'date',
        lambda: [
            _pandas_stamp('2023-01-01'),
            _numpy_stamp('2023-01-01T00:00:00.000000000'),
            dt.date(2023, 1, 2),
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


def _pandas_stamp(text: str) -> Any:
    import pandas as pd

    return pd.Timestamp(text)


def _numpy_stamp(text: str) -> Any:
    import numpy as np

    return np.datetime64(text)


def _zone(name: str = 'America/New_York') -> Any:
    from zoneinfo import ZoneInfo

    return ZoneInfo(name)


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


def _keyed(kind: str, given: list[Any]) -> Any:
    r"""The shop, the ranking's items keyed by text of ``kind``, or, for
    ``literal``, items 0 to 2 keyed by the ``str`` of the candidates
    given, in order (Codex r3's reproducer), the fourth ``other``; for
    ``ends``, items 0 and 1 by the first and the last given."""
    connection = host()._connection()
    for table in ('items', 'buys'):
        if kind == 'ends':
            # Two of the candidates share a text: items 0 and 1 are keyed
            # by the first and the last only (Codex r6's reproducer).
            using = (
                f"CASE item_id WHEN 0 THEN '{given[0]}' WHEN 1 THEN "
                f"'{given[-1]}' ELSE item_id::VARCHAR END"
            )
        elif kind == 'literal':
            cases = ' '.join(
                f"WHEN {at} THEN '{value}'" for at, value in enumerate(given)
            )
            using = f"CASE item_id {cases} ELSE 'other' END"
        else:
            using = (
                "strftime(DATE '2023-01-01' + item_id::INTEGER, "
                f"'{KEY_TEXT[kind]}')"
            )
        connection.execute(
            f'ALTER TABLE {table} ALTER item_id TYPE VARCHAR USING {using}'
        )
    return connection


def candidates() -> dict[str, Any]:
    r"""Each candidate list's parts, questions and rankings."""
    out = {}
    for name, (kind, made) in CANDIDATES.items():
        connection = _keyed(kind, made())
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
    else:
        os.environ['SDM_CORE_ROUTES'] = ''
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
