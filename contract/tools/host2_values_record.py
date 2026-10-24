#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records ``contract/t136-host2-vectors.json``: register HOST-2's values
the core refuses (T1.36), measured through DuckDB 1.5.6 and the public
calls on both routes.

Written by T1.36's independent test author. Run from this repository's
root::

    python -I contract/tools/host2_values_record.py \
        [--reference-python REF] [--candidate-python CAND] [--big]

``REF`` (default: the reference checkout's ``.venv``) runs each source on
the reference; ``CAND`` (default: this interpreter) runs it on the core
route, ``SDM_CORE_ROUTES`` set to every DuckDB path of ``from_source``,
``materialize`` and ``predict``. Each source is ``contract/bases.yaml``'s
``shop`` and the vector's ``setup``; the calls are ``Schema.from_source``,
then ``materialize`` of the vector's query (indices 0 to 7, anchor
2024-03-01), then ``SDMClient.predict`` against the scripted endpoint, as
far as each gets. What is recorded per route: the outcome (``success``, or
the stage it failed at, the exception's class, ``StructuredDataError.code``
and the first line of its message), the description (each table's columns
with dtype and stype, key and time column) and each stage's warnings (the
first line of each, DuckDB's missing-index advice left out).

``duckdb`` records what DuckDB itself does: the Arrow type and raw value it
hands over for each infinity, and whether it holds a finite ``TIMESTAMP_S``
or ``TIMESTAMP_MS`` past an int64 count of microseconds (it does not: its
casts and its Arrow import refuse one), and what a ``duration`` or
``date64`` becomes through it.

``--big`` also runs the sources holding one value just over 2 GiB (a
``VARCHAR`` and a ``BLOB`` of 2**31 + 16 bytes, with and without
``SET arrow_large_buffer_size = true``), each route in its own process,
one after the other: about 7 GB of memory at the peak and up to six
minutes each. Without it the ``big`` section is kept as recorded. The core's
answers are recorded beside the reference's as evidence of what the
candidate did when measured; no expectation of the core is derived from
them.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/t136-host2-vectors.json'
REFERENCE = Path.home() / 'code/sdmc-reference'
PQL = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
INDICES = list(range(8))
ANCHOR = '2024-03-01'
#: One value just over int32 offsets (2,147,483,647 bytes).
BIG = 2**31 + 16

#: The task kinds a query may resolve to; their paths are routed together.
KINDS = (
    'binary',
    'multiclass',
    'regression',
    'link-prediction',
    'ranking',
    'forecast',
)
#: Every DuckDB path of the three calls, forced onto the core.
CORE_ROUTES = [
    'schema.from-source/all-tables/duckdb',
    'schema.from-source/catalog-keys/duckdb',
    *(f'query.materialize/indices-{kind}/duckdb' for kind in KINDS),
    *(f'client.predict/{kind}/duckdb' for kind in KINDS),
]


def _sources() -> dict[str, dict[str, Any]]:
    r"""Each vector: its setup after ``shop``, the column it is about, and
    its role there."""
    out: dict[str, dict[str, Any]] = {}
    for kind in ('TIMESTAMP_S', 'TIMESTAMP_MS', 'TIMESTAMP', 'TIMESTAMP_NS'):
        for value in ('infinity', '-infinity'):
            out[f'time.{kind}.{value}'] = {
                'column': 'orders.ts',
                'role': 'time column',
                'setup': [
                    f'ALTER TABLE orders ALTER ts TYPE {kind}',
                    f"UPDATE orders SET ts = '{value}' WHERE order_id = 5",
                ],
            }
            out[f'feature.{kind}.{value}'] = {
                'column': 'orders.seen',
                'role': 'feature (orders.ts the time column)',
                'setup': [
                    f'ALTER TABLE orders ALTER ts TYPE {kind}',
                    f'ALTER TABLE orders ADD COLUMN seen {kind}',
                    'UPDATE orders SET seen = ts',
                    f"UPDATE orders SET seen = '{value}' WHERE order_id = 5",
                ],
            }
            out[f'users.{kind}.{value}'] = {
                'column': 'users.f',
                'role': "the entity table's only timestamp (inferred its time column)",
                'setup': [
                    f'ALTER TABLE users ADD COLUMN f {kind}',
                    "UPDATE users SET f = TIMESTAMP '2020-01-01'",
                    f"UPDATE users SET f = '{value}' WHERE user_id = 3",
                ],
            }
        out[f'time.{kind}.finite'] = {
            'column': 'orders.ts',
            'role': 'time column (control: no infinity)',
            'setup': [f'ALTER TABLE orders ALTER ts TYPE {kind}'],
        }
    extremes = {
        'time.TIMESTAMP_S.max': "'294247-01-10 04:00:54'::TIMESTAMP_S",
        'time.TIMESTAMP_MS.max': "'294247-01-10 04:00:54.775'::TIMESTAMP_MS",
        'time.TIMESTAMP_S.min': "'290309-12-22 (BC) 00:00:00'::TIMESTAMP_S",
        'time.TIMESTAMP_NS.max': (
            "'2262-04-11 23:47:16.854775806'::TIMESTAMP_NS"
        ),
    }
    for name, literal in extremes.items():
        kind = name.split('.')[1]
        out[name] = {
            'column': 'orders.ts',
            'role': 'time column (the largest finite value the type holds)',
            'setup': [
                f'ALTER TABLE orders ALTER ts TYPE {kind}',
                f'UPDATE orders SET ts = {literal} WHERE order_id = 5',
            ],
        }
    # Finite values past int64 microseconds: DuckDB's casts refuse them, but
    # an Arrow import keeps them (``duckdb``'s finite_past_int64_microseconds).
    far = {
        's.past': ('TIMESTAMP_S', 'timestamp[s]', 10**16),
        's.before': ('TIMESTAMP_S', 'timestamp[s]', -(10**16)),
        'ms.past': ('TIMESTAMP_MS', 'timestamp[ms]', 10**17),
        's.just-past': ('TIMESTAMP_S', 'timestamp[s]', 9_223_372_036_855),
    }
    for label, (kind, arrow_type, value) in far.items():
        imported = {'arrow': 'far', 'type': arrow_type, 'rows': [[5, value]]}
        out[f'time.far.{label}'] = {
            'column': 'orders.ts',
            'role': 'time column',
            'setup': [
                f'ALTER TABLE orders ALTER ts TYPE {kind}',
                imported,
                'UPDATE orders SET ts = far.f FROM far WHERE order_id = far.k',
            ],
        }
        if label == 's.just-past':
            continue
        out[f'feature.far.{label}'] = {
            'column': 'orders.seen',
            'role': 'feature (orders.ts the time column)',
            'setup': [
                f'ALTER TABLE orders ADD COLUMN seen {kind}',
                f'UPDATE orders SET seen = ts::{kind}',
                imported,
                'UPDATE orders SET seen = far.f FROM far WHERE order_id = far.k',
            ],
        }
        out[f'users.far.{label}'] = {
            'column': 'users.f',
            'role': "the entity table's only timestamp (inferred its time column)",
            'setup': [
                f'ALTER TABLE users ADD COLUMN f {kind}',
                f"UPDATE users SET f = TIMESTAMP '2020-01-01'::{kind}",
                {**imported, 'rows': [[3, value]]},
                'UPDATE users SET f = far.f FROM far WHERE user_id = far.k',
            ],
        }

    # The reference's from_source reads the first non-null value of each
    # time-column candidate (a timestamp not named as a load stamp or a birth
    # date) as a Python datetime (predictive_query/time_columns.py,
    # _refused_for_values: present.iloc[0]), which fails unnamed
    # (OverflowError) past the years 1 to 9999. So the same values in the
    # first row read: users' row 0, orders' row 0.
    def first(kind: str, value: Any, name: str = 'f') -> list[Any]:
        return [
            f'ALTER TABLE users ADD COLUMN {name} {kind}',
            f"UPDATE users SET {name} = TIMESTAMP '2020-01-01'::{kind}",
            *_put('users', 'user_id', name, kind, 0, value),
        ]

    def _put(
        table: str, key: str, column: str, kind: str, row: int, value: Any
    ) -> list[Any]:
        if isinstance(value, str):
            return [
                f'UPDATE {table} SET {column} = {value} WHERE {key} = {row}'
            ]
        unit = {'TIMESTAMP_S': 's', 'TIMESTAMP_MS': 'ms'}[kind]
        return [
            {
                'arrow': 'far',
                'type': f'timestamp[{unit}]',
                'rows': [[row, value]],
            },
            f'UPDATE {table} SET {column} = far.f FROM far WHERE {key} = far.k',
        ]

    entity = "the entity table's only timestamp, its first row"
    firsts: dict[str, tuple[str, list[Any]]] = {
        'first.users.TIMESTAMP_S.infinity': (
            entity,
            first('TIMESTAMP_S', "'infinity'"),
        ),
        'first.users.TIMESTAMP_S.-infinity': (
            entity,
            first('TIMESTAMP_S', "'-infinity'"),
        ),
        'first.users.TIMESTAMP_MS.infinity': (
            entity,
            first('TIMESTAMP_MS', "'infinity'"),
        ),
        'first.users.far.s.past': (entity, first('TIMESTAMP_S', 10**16)),
        'first.users.far.s.before': (entity, first('TIMESTAMP_S', -(10**16))),
        'first.users.far.s.just-past': (
            entity,
            first('TIMESTAMP_S', 9_223_372_036_855),
        ),
        'first.users.far.ms.past': (entity, first('TIMESTAMP_MS', 10**17)),
        'first.users.TIMESTAMP_S.year-10000': (
            entity,
            first('TIMESTAMP_S', "'10000-01-01'::TIMESTAMP_S"),
        ),
        'first.users.TIMESTAMP.infinity': (
            entity,
            first('TIMESTAMP', "'infinity'"),
        ),
        'first.users.TIMESTAMP.year-10000': (
            entity,
            first('TIMESTAMP', "'10000-01-01'::TIMESTAMP"),
        ),
        'first.users.TIMESTAMP_S.null-then-infinity': (
            "the entity table's only timestamp, null in its first row and "
            'past the years in its second',
            [
                *first('TIMESTAMP_S', 'NULL'),
                "UPDATE users SET f = 'infinity' WHERE user_id = 1",
            ],
        ),
        'first.feature.TIMESTAMP_S.infinity': (
            "a feature (orders.ts the time column), orders' first row",
            [
                'ALTER TABLE orders ALTER ts TYPE TIMESTAMP_S',
                'ALTER TABLE orders ADD COLUMN seen TIMESTAMP_S',
                'UPDATE orders SET seen = ts',
                "UPDATE orders SET seen = 'infinity' WHERE order_id = 0",
            ],
        ),
        'first.time.TIMESTAMP_S.infinity': (
            "the time column, orders' first row",
            [
                'ALTER TABLE orders ALTER ts TYPE TIMESTAMP_S',
                "UPDATE orders SET ts = 'infinity' WHERE order_id = 0",
            ],
        ),
        'first.time.TIMESTAMP_S.-infinity': (
            "the time column, orders' first row",
            [
                'ALTER TABLE orders ALTER ts TYPE TIMESTAMP_S',
                "UPDATE orders SET ts = '-infinity' WHERE order_id = 0",
            ],
        ),
        # Controls: the reference reads these first rows.
        'first.users.TIMESTAMP_S.year-9999': (
            entity + ' (control: within the years)',
            first('TIMESTAMP_S', "'9999-12-31'::TIMESTAMP_S"),
        ),
        'first.users.TIMESTAMP_NS.infinity': (
            entity + ' (control: a nanosecond infinity is within them)',
            first('TIMESTAMP_NS', "'infinity'"),
        ),
        'first.users.TIMESTAMP_S.infinity.loaded_at': (
            entity + ' (control: a load stamp is no candidate)',
            first('TIMESTAMP_S', "'infinity'", 'loaded_at'),
        ),
        'first.users.TIMESTAMP_S.infinity.birth_date': (
            entity + ' (control: a birth date is no candidate)',
            first('TIMESTAMP_S', "'infinity'", 'birth_date'),
        ),
    }
    for name, (role, setup) in firsts.items():
        column = name.split('.')[1]
        out[name] = {
            'column': {
                'users': 'users.'
                + (
                    name.split('.')[-1]
                    if name.endswith(('loaded_at', 'birth_date'))
                    else 'f'
                ),
                'feature': 'orders.seen',
                'time': 'orders.ts',
            }[column],
            'role': role,
            'setup': setup,
        }
    # The three sources of fix 3's differential (seed 3136) whose
    # reference from_source failed unnamed, as generated.
    out['generated.g24.users.TIMESTAMP_S'] = {
        'column': 'users.f',
        'role': entity + ' (-10**16), and two more past the years',
        'setup': [
            'ALTER TABLE users ADD COLUMN f TIMESTAMP_S',
            "UPDATE users SET f = TIMESTAMP '2020-01-01'",
            {'arrow': 'far0', 'type': 'timestamp[s]', 'rows': [[7, 10**16]]},
            'UPDATE users SET f = far0.f FROM far0 WHERE user_id = far0.k',
            {'arrow': 'far1', 'type': 'timestamp[s]', 'rows': [[0, -(10**16)]]},
            'UPDATE users SET f = far1.f FROM far1 WHERE user_id = far1.k',
            {
                'arrow': 'far2',
                'type': 'timestamp[s]',
                'rows': [[1, 9_223_372_036_855]],
            },
            'UPDATE users SET f = far2.f FROM far2 WHERE user_id = far2.k',
        ],
    }
    out['generated.g30.feature.TIMESTAMP_MS'] = {
        'column': 'orders.seen',
        'role': "a feature, 'infinity' in orders' first row",
        'setup': [
            'ALTER TABLE orders ALTER ts TYPE TIMESTAMP_MS',
            'ALTER TABLE orders ADD COLUMN seen TIMESTAMP_MS',
            'UPDATE orders SET seen = ts',
            "UPDATE orders SET seen = 'infinity'::TIMESTAMP_MS WHERE order_id = 0",
            "UPDATE orders SET seen = '-infinity'::TIMESTAMP_MS WHERE order_id = 5",
        ],
    }
    out['generated.g32.feature.TIMESTAMP_S'] = {
        'column': 'orders.seen',
        'role': "a feature, 9223372036855 s in orders' first row",
        'setup': [
            'ALTER TABLE orders ALTER ts TYPE TIMESTAMP_S',
            'ALTER TABLE orders ADD COLUMN seen TIMESTAMP_S',
            'UPDATE orders SET seen = ts',
            'UPDATE orders SET seen = NULL WHERE order_id = 40',
            "UPDATE orders SET seen = TIMESTAMP '1000-01-01'::TIMESTAMP_S "
            'WHERE order_id = 64',
            {
                'arrow': 'far2',
                'type': 'timestamp[s]',
                'rows': [[0, 9_223_372_036_855]],
            },
            'UPDATE orders SET seen = far2.f FROM far2 WHERE order_id = far2.k',
        ],
    }
    return out


SOURCES = _sources()

#: The sources holding one value past int32 offsets, by kind and mode.
BIG_KINDS = {
    'varchar': ('VARCHAR', f"repeat('x', {BIG})"),
    'blob': ('BLOB', f"repeat('x', {BIG})::BLOB"),
}
BIG_MODES = {
    'small': [],
    'large': ['SET arrow_large_buffer_size = true'],
}


def big_setup(kind: str, mode: str) -> list[str]:
    r"""The statements after ``shop``: ``users.f`` one short value per row,
    and user 3's just over 2 GiB."""
    sql_type, value = BIG_KINDS[kind]
    return [
        *BIG_MODES[mode],
        f'ALTER TABLE users ADD COLUMN f {sql_type}',
        "UPDATE users SET f = 'a'",
        f'UPDATE users SET f = {value} WHERE user_id = 3',
    ]


# --- the child: the public calls on one route --------------------------------


def _shop() -> list[str]:
    import yaml

    bases = yaml.safe_load((ROOT / 'contract/bases.yaml').read_text())
    shop: list[str] = bases['bases']['shop']
    return shop


def _described(schema: Any) -> dict[str, Any]:
    return {
        name: {
            'columns': [
                [c.name, str(c.dtype), str(c.stype)]
                for c in table.columns.values()
            ],
            'primary_key': table.primary_key,
            'time_column': table.time_column,
        }
        for name, table in sorted(schema.tables.items())
    }


def source(setup: list[Any]) -> Any:
    r"""A DuckDB connection holding ``shop`` and ``setup``: each a statement,
    or ``{"arrow": name, "type": "timestamp[s]", "rows": [[key, value],
    ...]}``, an Arrow table of an int64 ``k`` and ``f`` of that type,
    registered under ``name`` for the statements after it (DuckDB keeps
    such an import's values as they are)."""
    import duckdb
    import pyarrow as pa

    connection = duckdb.connect()
    # Setup's own settings first: a setting changes what DuckDB hands over.
    settings = [
        s for s in setup if isinstance(s, str) and s.upper().startswith('SET ')
    ]
    for statement in settings + _shop():
        connection.execute(statement)
    for step in setup:
        if isinstance(step, dict):
            unit = step['type'].removeprefix('timestamp[').removesuffix(']')
            keys, values = zip(*step['rows'], strict=True)
            connection.register(
                step['arrow'],
                pa.table(
                    {
                        'k': pa.array(keys, pa.int64()),
                        'f': pa.array(values, pa.timestamp(unit)),
                    }
                ),
            )
        elif step not in settings:
            connection.execute(step)
    return connection


def answer(setup: list[Any], route: str) -> dict[str, Any]:
    r"""The public calls over ``shop`` and ``setup`` on ``route``
    (``reference`` or ``core``), as far as they get."""
    if route == 'core':
        os.environ['SDM_CORE_ROUTES'] = ','.join(CORE_ROUTES)
    else:
        os.environ['SDM_CORE_ROUTES'] = ''
    import pandas as pd

    runners = str(ROOT / 'contract/runners')
    if runners not in sys.path:
        sys.path.insert(0, runners)
    from scripted_endpoint import serve
    from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient
    from sdm_connectors import DuckDBSource

    connection = source(setup)
    got: dict[str, Any] = {'described': None}
    stage = 'from_source'
    #: Where each stage's warnings start in ``shown``.
    starts = {'from_source': 0}
    with warnings.catch_warnings(record=True) as shown, serve() as (url, _):
        warnings.simplefilter('always')
        try:
            schema = Schema.from_source(DuckDBSource(connection))
            got['described'] = _described(schema)
            stage = 'materialize'
            starts[stage] = len(shown)
            request = PredictiveQuery(PQL, schema).materialize(
                indices=INDICES, anchor_time=pd.Timestamp(ANCHOR)
            )
            stage = 'predict'
            starts[stage] = len(shown)
            with SDMClient(url) as client:
                client.predict(request, model=KumoRelational())
            got['outcome'] = {'result': 'success', 'stage': 'predict'}
        except Exception as error:  # noqa: BLE001 - any is an answer
            code = getattr(error, 'code', None)
            got['outcome'] = {
                'result': 'refusal',
                'stage': stage,
                'error': type(error).__name__,
                'code': None if code is None else str(code),
                'message': str(error).splitlines()[0][:240],
            }
            # Which request of the run failed, where the error says.
            details = getattr(error, 'details', None)
            if isinstance(details, dict) and 'request' in details:
                got['outcome']['request'] = str(details['request'])
            if os.environ.get('HOST2_TRACE'):
                traceback.print_exception(error)
    ends = [*list(starts.values())[1:], len(shown)]
    got['warnings'] = {
        name: [
            [w.category.__name__, str(w.message).splitlines()[0][:160]]
            for w in shown[start:end]
            # DuckDB's missing-index advice says nothing about the values.
            if 'for optimal database querying' not in str(w.message)
        ]
        for (name, start), end in zip(starts.items(), ends, strict=True)
    }
    return got


def _child(argv: list[str]) -> int:
    if argv[0] == 'sources':
        route, names = argv[1], argv[2:]
        for name in names:
            got = answer(SOURCES[name]['setup'], route)
            print(json.dumps({'name': name, **got}), flush=True)
        return 0
    route, kind, mode = argv[1:4]
    start = time.monotonic()
    got = answer(big_setup(kind, mode), route)
    got['seconds'] = round(time.monotonic() - start, 1)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    got['peak_rss_bytes'] = peak * (
        1 if platform.system() == 'Darwin' else 1024
    )
    print(json.dumps(got), flush=True)
    return 0


# --- what DuckDB itself does -----------------------------------------------------


def duckdb_facts() -> dict[str, Any]:
    import duckdb
    import pyarrow as pa

    def arrow(sql: str, connection: Any = None) -> list[Any]:
        connection = connection or duckdb.connect()
        column = connection.execute(sql).to_arrow_table().column(0)
        values = column.cast(pa.int64()).to_pylist()
        return [str(column.type), values[0]]

    def refused(run: Any) -> str:
        try:
            run()
        except Exception as error:  # noqa: BLE001 - the refusal is the fact
            return f'{type(error).__name__}: {str(error).splitlines()[0]}'
        return 'accepted'

    def imported(unit: str, value: int) -> dict[str, Any]:
        r"""An Arrow value imported into a table: DuckDB's type for it, the
        Arrow it hands back, and its own cast of it to text."""
        connection = duckdb.connect()
        table = pa.table({'f': pa.array([value], pa.timestamp(unit))})
        connection.register('arrow_in', table)
        connection.execute('CREATE TABLE t AS SELECT f FROM arrow_in')
        [(stored,)] = connection.execute('SELECT typeof(f) FROM t').fetchall()
        return {
            'stored_as': stored,
            'handed_over': arrow('SELECT f FROM t', connection),
            'cast_to_text': refused(
                lambda: connection.execute('SELECT f::VARCHAR FROM t')
            ),
        }

    infinity = {
        f'{kind} {value}': arrow(f"SELECT '{value}'::{kind} AS f")
        for kind in ('TIMESTAMP_S', 'TIMESTAMP_MS', 'TIMESTAMP', 'TIMESTAMP_NS')
        for value in ('infinity', '-infinity')
    }
    casts = [
        "'294247-01-10 04:00:55'::TIMESTAMP_S",
        "'294247-01-10 04:00:54.776'::TIMESTAMP_MS",
        "'290309-12-21 (BC) 23:59:59'::TIMESTAMP_S",
    ]
    past: dict[str, Any] = {
        f'SELECT {cast}': refused(
            lambda cast=cast: duckdb.connect().execute(f'SELECT {cast}')
        )
        for cast in casts
    }
    for unit, value in (
        ('s', 10**16),
        ('s', -(10**16)),
        ('ms', 10**17),
        ('s', 9_223_372_036_855),
        ('s', 2**63 - 2),
    ):
        past[f'Arrow timestamp[{unit}] {value} imported'] = imported(
            unit, value
        )
    connection = duckdb.connect()
    connection.register(
        'arrow_in',
        pa.table(
            {
                'd': pa.array([1], pa.duration('s')),
                'm': pa.array([1], pa.duration('ms')),
                'e': pa.array([86_400_000], pa.date64()),
            }
        ),
    )
    connection.execute('CREATE TABLE t AS SELECT * FROM arrow_in')
    through = {
        f'{name} ({arrow_type}) stored then read': str(
            connection.execute(f'SELECT {name} FROM t')
            .to_arrow_table()
            .schema.field(0)
            .type
        )
        for name, arrow_type in (
            ('d', 'duration[s]'),
            ('m', 'duration[ms]'),
            ('e', 'date64'),
        )
    }
    return {
        'version': duckdb.__version__,
        'infinity': infinity,
        'finite_past_int64_microseconds': past,
        'duration_and_date64': through,
    }


# --- the recorder ------------------------------------------------------------------


def _run(python: str, argv: list[str]) -> tuple[list[str], int]:
    done = subprocess.run(
        [python, '-I', str(Path(__file__).resolve()), '--child', *argv],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    if done.returncode != 0:
        sys.stderr.write(done.stderr[-4000:])
    return done.stdout.strip().splitlines(), done.returncode


def _where(package: Path) -> str:
    r"""Where a route's ``sdm_client`` is, with no machine's path in it:
    relative to the checkout it is a source tree of (the one holding
    ``packages/``), or from the ``site-packages`` it is installed in."""
    parts = package.parts
    if 'site-packages' in parts:
        return '/'.join(parts[parts.index('site-packages') :])
    for parent in package.parents:
        if (parent / 'packages').is_dir():
            return package.relative_to(parent).as_posix()
    return package.name


def _environment(python: str) -> dict[str, str]:
    lines, _ = _run(python, ['environment'])
    loaded: dict[str, str] = json.loads(lines[-1])
    return loaded


def main() -> int:
    if sys.argv[1:2] == ['--child']:
        if sys.argv[2] == 'environment':
            import duckdb
            import pandas as pd
            import pyarrow as pa
            import sdm_client

            print(
                json.dumps(
                    {
                        'python': platform.python_version(),
                        'pandas': pd.__version__,
                        'pyarrow': pa.__version__,
                        'duckdb': duckdb.__version__,
                        'sdm_client': _where(Path(sdm_client.__file__).parent),
                    }
                )
            )
            return 0
        return _child(sys.argv[2:])
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--reference-python', default=str(REFERENCE / '.venv/bin/python')
    )
    parser.add_argument('--candidate-python', default=sys.executable)
    parser.add_argument('--big', action='store_true')
    args = parser.parse_args()
    reference_commit = subprocess.run(
        ['git', '-C', str(REFERENCE), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    candidate_commit = subprocess.run(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    previous = json.loads(OUT.read_text()) if OUT.exists() else {}
    pythons = {
        'reference': args.reference_python,
        'core': args.candidate_python,
    }
    answers: dict[str, dict[str, Any]] = {}
    for route, python in pythons.items():
        lines, code = _run(python, ['sources', route, *SOURCES])
        if code != 0:
            return code
        for line in lines:
            got = json.loads(line)
            answers.setdefault(got.pop('name'), {})[route] = got
    vectors = [
        {
            'name': name,
            **spec,
            'reference': answers[name]['reference'],
            'core_at_measurement': answers[name]['core'],
        }
        for name, spec in SOURCES.items()
    ]
    big = previous.get('big')
    if args.big:
        big = []
        for kind in BIG_KINDS:
            for mode in BIG_MODES:
                row: dict[str, Any] = {
                    'name': f'big.{kind}.{mode}',
                    'bytes': BIG,
                    'setup': big_setup(kind, mode),
                }
                for route, python in pythons.items():
                    lines, code = _run(python, ['big', route, kind, mode])
                    if code != 0:
                        return code
                    got = json.loads(lines[-1])
                    key = (
                        'reference'
                        if route == 'reference'
                        else ('core_at_measurement')
                    )
                    row[key] = got
                    print(row['name'], route, got['outcome'], file=sys.stderr)
                big.append(row)
    document = {
        'format': 'sdm-host2-vectors/1',
        'about': previous.get('about', ''),
        'recorded': {
            'by': 'contract/tools/host2_values_record.py',
            'reference': reference_commit,
            'candidate': candidate_commit,
            'environments': {
                route: _environment(python) for route, python in pythons.items()
            },
            'query': PQL,
            'indices': INDICES,
            'anchor_time': ANCHOR,
            'core_routes': CORE_ROUTES,
        },
        'duckdb': duckdb_facts(),
        'vectors': vectors,
        'big': big,
        'not_measured': previous.get('not_measured', []),
    }
    OUT.write_text(json.dumps(document, indent=1) + '\n', encoding='utf-8')
    print(f'{len(vectors)} vectors recorded to {OUT}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
