# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public scenarios for temporal candidates, semantic types and JSON values."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import re
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/t214b-regression-vectors.json'
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
    # A zone's repeated hour (fold 0 and fold 1 of New York's
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
    # The same wall time in New York and in Toronto, whose texts
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
    r"""Shared shop, ranking and endpoint helpers, whose
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


def observe(dsn: str | None) -> dict[str, Any]:
    """Run current-client regressions, including Postgres when configured."""
    return {
        'candidates': candidates(),
        'stype': stype(),
        'where': where(),
        'json': None if dsn is None else json_documents(dsn),
    }


def load() -> dict[str, Any]:
    r"""The vector file, as committed."""
    return json.loads(OUT.read_text(encoding='utf-8'))
