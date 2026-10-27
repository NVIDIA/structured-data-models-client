# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.19's exits on ``python-core`` (the package's test author's): two-path
sampling, ``main``'s eb6c68eb.

    SDM_POSTGRES_TEST_DSN=... python -m pytest \
        contract/tests/test_t219_two_paths.py -q

Written from the design's T2.19 row (each item an exit) and the reference
(python-ref at ``contract/reference-env.txt``'s commit, which holds
eb6c68eb), before the implementation. ``sdm-testkit``'s
``tests/t219_two_paths.rs`` holds ``rust-host`` to the same exits.

The cases, ``contract/cases/s1/shape.*.json``, each run on DuckDB, SQLite
and Postgres through ``predict``, so each payload is compared:

- ``diamond``, ``same-table-twice``, ``chain`` and ``star``: ``main``'s
  ``_SHAPES``, ``_KEYS`` and ``_shape_frames``
  (``test_sql_sampler_parity.py:304-376``), through ``Schema.from_source``
  with its tables, keys, time columns and links given
  (``shape.<shape>.<table>``), each sampled from every table of the shape
  that holds no foreign key. ``main`` samples each from its first three
  tables; the core refuses an entity table with a foreign key by
  ``parent_table`` (E92's class, which T2.11 lifts), a reason other than
  T2.19's, so those wait for T2.11;
- ``shape.diamond-from-source``: the diamond read by
  ``Schema.from_source(source)`` from the keys its catalog declares;
- ``shape.one-column-two-tables``: one column of a keyless event table
  linking two tables, as a role-playing join makes it, and
  ``shape.two-paths.keyless`` and ``.keyed``: the events reached from a
  customer directly and through its orders (``main``'s
  ``test_two_paths.py``).

Each test's one line says the exit it proves:

- "cases written for all three databases": each case is ``main``'s shape,
  runs on all three, and python-ref's line for it on each is recorded at
  its case file;
- "a table reached by two paths in one hop read only up to the slices it
  held when the hop began", "one column linking two tables" and "a
  ``from_source`` diamond compared at the payload": python-core answers
  each case on each database as python-ref does, its outcome and each
  request the scripted endpoint received (the ``payload`` rule: the parsed
  request but its request id), on the core route. ``chain`` and ``star``
  reach each table by one path: controls the core already holds.

The cases judge the fix (a probe, since the reference's code does not run
here): with eb6c68eb's change to ``sql_sampler.py`` reverted on the
reference, the diamond cases' and ``shape.two-paths.keyed``'s payloads
differ (rows read a hop too deep), ``shape.two-paths.keyless`` is refused
at materialize, and, with its column read once reverted too, so are the
``same-table-twice`` cases and ``shape.one-column-two-tables``; ``chain``
and ``star`` are unchanged.

Gating: ``t219-pending.json``'s ``pending`` names, as ``py:<case key>``,
the keys whose test failed when written. Each must still fail (strict);
once one passes it fails, naming itself, until it is taken out.
``pending`` stays within ``authored``. Where ``SDM_POSTGRES_TEST_DSN``
names no server and ``SDM_REQUIRE_POSTGRES`` is not set, the Postgres keys
have nothing to hold and return without asserting; with it set, a missing
server fails (H26).
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
CASES = ROOT / 'contract/cases/s1'
PENDING = Path(__file__).with_name('t219-pending.json')
DATABASES = ('duckdb', 'sqlite', 'postgres')

#: ``main``'s ``test_sql_sampler_parity.py:308-337`` (eb6c68eb, the
#: reference at 6298913c), as written there.
SHAPES = {
    'diamond': [
        ('orders', 'cid', 'customers'),
        ('customers', 'aid', 'addresses'),
        ('orders', 'aid', 'addresses'),
    ],
    'same-table-twice': [
        ('orders', 'cid', 'customers'),
        ('orders', 'cid', 'buyers'),
    ],
    'chain': [
        ('lines', 'oid', 'orders'),
        ('orders', 'cid', 'customers'),
        ('customers', 'aid', 'addresses'),
    ],
    'star': [
        ('orders', 'cid', 'customers'),
        ('orders', 'pid', 'products'),
        ('orders', 'sid', 'stores'),
    ],
}
KEYS = {
    'addresses': 'aid',
    'buyers': 'cid',
    'customers': 'cid',
    'lines': 'lid',
    'orders': 'oid',
    'products': 'pid',
    'stores': 'sid',
}
FROM_SOURCE = 'shape.diamond-from-source'
ONE_COLUMN = 'shape.one-column-two-tables'


def _tables(shape: str) -> list[str]:
    return sorted({t for edge in SHAPES[shape] for t in (edge[0], edge[2])})


#: Each shape from each of its tables that holds no foreign key.
SAMPLED = {
    f'shape.{shape}.{entity}': (shape, entity)
    for shape in SHAPES
    for entity in _tables(shape)
    if entity not in {edge[0] for edge in SHAPES[shape]}
}
#: ``main``'s ``test_two_paths.py``: the events reached from a customer
#: directly and through its orders, keyless as there and keyed.
TWO_PATHS = {
    'shape.two-paths.keyless': {'customers': 'cid', 'headers': 'hid'},
    'shape.two-paths.keyed': {
        'customers': 'cid',
        'headers': 'hid',
        'details': 'did',
    },
}
TWO_PATH_LINKS = [
    ('headers', 'cid', 'customers', 'cid'),
    ('details', 'hid', 'headers', 'hid'),
    ('details', 'cid', 'customers', 'cid'),
]
CASE_IDS = (*SAMPLED, FROM_SOURCE, ONE_COLUMN, *TWO_PATHS)


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('t219_run_python', RUNNER)


def key(case_id: str, database: str) -> str:
    return runner().case_key(case_id, database)


KEYS_ALL = tuple(key(c, d) for d in DATABASES for c in CASE_IDS)


def held(name: str, check: Any) -> None:
    r"""Runs ``check``: it must pass, or, while ``py:<name>`` is pending,
    fail."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    if f'py:{name}' not in gating['pending']:
        check()
        return
    try:
        check()
    except Exception:  # noqa: BLE001 - pending: failing, as when written
        return
    raise AssertionError(
        f'{name} passes now: take py:{name} out of '
        'contract/tests/t219-pending.json pending'
    )


def test_the_pending_list_only_shrinks() -> None:
    r"""Control: ``pending`` names only keys the test author wrote, and
    ``authored`` is every case's key on each database on python-core, and
    on DuckDB and SQLite on rust-host (its Postgres pass replays
    python-core's fetches, which ``cross-host (postgres)`` holds)."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    assert set(gating['pending']) <= set(gating['authored'])
    assert set(gating['authored']) == {f'py:{k}' for k in KEYS_ALL} | {
        f'rust:{k}' for k in KEYS_ALL if not k.endswith('@postgres')
    }


# --- the cases -------------------------------------------------------------------


@functools.cache
def cases() -> dict[str, tuple[dict[str, Any], str]]:
    r"""The T2.19 cases by id, each with its digest, as the runner loads
    them."""
    loaded = runner().load_cases([CASES / f'{c}.json' for c in CASE_IDS])
    return {case['id']: (case, digest) for case, digest in loaded}


def _links(case: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    return [
        (
            link['source_table'],
            link['source_column'],
            link['target_table'],
            link['target_column'],
        )
        for link in case['input']['from_source']['relationships']
    ]


def _entity(case: dict[str, Any]) -> tuple[str, str]:
    table, column = case['input']['pql'].rsplit(' FOR EACH ', 1)[1].split('.')
    return table, column


@pytest.mark.parametrize('case_id', list(SAMPLED))
def test_each_shape_is_main_s_sampled_from_main_s_tables(case_id: str) -> None:
    r"""Exit "``diamond``, ``same-table-twice``, ``chain``, ``star``": the
    case gives ``Schema.from_source`` the shape's tables, keys and links as
    ``main`` writes them, samples it from one of its tables, and runs
    through predict on all three databases."""
    shape, entity = SAMPLED[case_id]
    case, _ = cases()[case_id]
    given = case['input']['from_source']
    assert case['status'] == 'acceptance' and case['slice'] == 'S1'
    assert case['until'] == 'predict'
    assert runner().case_databases(case) == DATABASES
    assert sorted(given['tables']) == _tables(shape)
    assert given['primary_keys'] == {t: KEYS[t] for t in _tables(shape)}
    assert _links(case) == [(s, c, d, KEYS[d]) for s, c, d in SHAPES[shape]]
    assert _entity(case) == (entity, KEYS[entity])
    assert case['input']['materialize']['indices'] == [0, 1, 2]


def _built(case: dict[str, Any]) -> Any:
    r"""The case's DuckDB database, its base and setup run, as the runner
    builds it."""
    import duckdb

    connection = duckdb.connect()
    for statement in [
        *runner().load_bases()[case['input']['base']],
        *case['input'].get('setup', []),
    ]:
        connection.execute(statement)
    return connection


def test_the_from_source_diamond_declares_the_diamond_in_its_catalog() -> None:
    r"""Exit "a ``from_source`` diamond compared at the payload": the case
    calls ``Schema.from_source(source)`` with nothing given, over a catalog
    declaring the diamond's keys and links and no other table, sampled
    from addresses, the table both paths reach, through predict."""
    case, _ = cases()[FROM_SOURCE]
    assert 'from_source' not in case['input']
    assert case['until'] == 'predict'
    assert runner().case_databases(case) == DATABASES
    connection = _built(case)
    try:
        tables = sorted(
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'main'"
            ).fetchall()
        )
        rows = connection.execute(
            'SELECT table_name, constraint_type, constraint_column_names, '
            'referenced_table, referenced_column_names '
            "FROM duckdb_constraints() WHERE schema_name = 'main' "
            "AND constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')"
        ).fetchall()
    finally:
        connection.close()
    assert tables == _tables('diamond')
    keys = {t: c[0] for t, kind, c, _, _ in rows if kind == 'PRIMARY KEY'}
    links = sorted(
        (t, c[0], d, r[0]) for t, kind, c, d, r in rows if kind == 'FOREIGN KEY'
    )
    assert keys == {t: KEYS[t] for t in _tables('diamond')}
    assert links == sorted((s, c, d, KEYS[d]) for s, c, d in SHAPES['diamond'])
    assert _entity(case) == ('addresses', 'aid')


@pytest.mark.parametrize('case_id', list(TWO_PATHS))
def test_the_events_are_reached_by_two_paths(case_id: str) -> None:
    r"""Exit "a table reached by two paths in one hop": the events link to
    the customers directly and through the orders, as ``main``'s
    ``test_two_paths.py`` links them, and are counted for each customer;
    keyed, each line's customer is not its order's, so the orders reach
    events the customers do not."""
    case, _ = cases()[case_id]
    given = case['input']['from_source']
    assert case['until'] == 'predict'
    assert runner().case_databases(case) == DATABASES
    assert given['primary_keys'] == TWO_PATHS[case_id]
    assert _links(case) == TWO_PATH_LINKS
    assert _entity(case) == ('customers', 'cid')
    assert 'COUNT(details.*' in case['input']['pql']
    connection = _built(case)
    try:
        own = connection.execute(
            'SELECT count(*) FROM details d JOIN headers h USING (hid) '
            'WHERE d.cid = h.cid'
        ).fetchone()[0]
        total = connection.execute('SELECT count(*) FROM details').fetchone()[0]
    finally:
        connection.close()
    assert own == (total if case_id.endswith('keyless') else 0)


def test_one_column_links_two_tables() -> None:
    r"""Exit "one column linking two tables": one column of a keyless event
    table refers to two tables, each by its key."""
    case, _ = cases()[ONE_COLUMN]
    assert case['until'] == 'predict'
    assert runner().case_databases(case) == DATABASES
    links = _links(case)
    assert len(links) == 2
    assert links[0][:2] == links[1][:2] and links[0][2] != links[1][2]
    source = links[0][0]
    assert source not in case['input']['from_source']['primary_keys']
    assert case['input']['from_source']['time_columns'] == {source: 'ts'}
    assert _entity(case)[0] in {links[0][2], links[1][2]}


# --- the reference ---------------------------------------------------------------


@functools.cache
def reference_lines() -> dict[str, dict[str, Any]]:
    r"""python-ref's committed lines, by key."""
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    commit = next(
        line.split(': ', 1)[1].strip()
        for line in env.splitlines()
        if line.startswith('reference: ')
    )
    path = ROOT / f'contract/reference/{commit}.jsonl'
    return {
        json.loads(line)['case']: json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
    }


@pytest.mark.parametrize('case_key', KEYS_ALL)
def test_python_ref_s_line_is_at_its_case_file(case_key: str) -> None:
    r"""Exit "cases written for all three databases": python-ref's line for
    the case on the database, recorded at the case file, answered through
    predict with one request, so the payload is compared there."""
    case_id = case_key.split('@')[0]
    _, digest = cases()[case_id]
    line = reference_lines().get(case_key)
    assert line is not None, f'{case_key}: no python-ref line'
    assert line['identity']['runner'] == 'python-ref'
    assert line['identity']['inputs']['case'] == digest
    assert line['outcome']['result'] == 'success'
    assert line['outcome']['stage'] == 'predict'
    assert len(line['public']['payloads']) == 1


# --- python-core against the reference ---------------------------------------------


def _server() -> bool:
    r"""Whether a Postgres server is named; where none is, one must not be
    required (H26)."""
    if os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    return False


@functools.cache
def core_run(database: str) -> dict[str, Any]:
    r"""python-core over the T2.19 cases on ``database``, as that database's
    job runs them: each observation by its key, or under ``''`` why the run
    failed whole."""
    with tempfile.TemporaryDirectory(prefix='t219-') as tmp:
        given = Path(tmp) / 'cases'
        given.mkdir()
        for case_id in CASE_IDS:
            shutil.copy(CASES / f'{case_id}.json', given)
        out = Path(tmp) / 'python-core.jsonl'
        done = subprocess.run(
            [
                sys.executable,
                '-I',
                str(RUNNER),
                '--mode',
                'python-core',
                '--database',
                database,
                '--out',
                str(out),
                str(given),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            check=False,
            timeout=1800,
        )
        if done.returncode != 0:
            return {'': done.stdout + done.stderr}
        return {
            json.loads(line)['case']: json.loads(line)
            for line in out.read_text(encoding='utf-8').splitlines()
        }


def _outcome(line: dict[str, Any]) -> tuple[Any, ...]:
    outcome = line['outcome']
    return (
        outcome['result'],
        outcome['stage'],
        outcome.get('error'),
        outcome.get('code'),
    )


def payloads(line: dict[str, Any]) -> list[Any]:
    r"""Each request the scripted endpoint received, parsed, as the
    ``payload`` rule compares it: its request id, drawn at random for every
    request, left out (``compare-policy.yaml``)."""
    out = []
    for payload in line['public']['payloads']:
        parsed = copy.deepcopy(payload.get('parsed'))
        if isinstance(parsed, dict) and isinstance(
            parsed.get('metadata'), dict
        ):
            parsed['metadata'].pop('request_id', None)
        out.append(parsed)
    return out


def differences(want: Any, got: Any, at: str = '') -> list[str]:
    r"""Where ``got`` differs from ``want``, a few paths."""
    if isinstance(want, dict) and isinstance(got, dict):
        found = []
        for name in sorted(set(want) | set(got)):
            found += differences(want.get(name), got.get(name), f'{at}/{name}')
        return found[:8]
    if (
        isinstance(want, list)
        and isinstance(got, list)
        and len(want) == len(got)
    ):
        found = []
        for i, (w, g) in enumerate(zip(want, got, strict=True)):
            found += differences(w, g, f'{at}[{i}]')
        return found[:8]
    if want != got:
        if isinstance(want, list) and isinstance(got, list):
            return [
                f'{at}: {len(got)} items where the reference has {len(want)}'
            ]
        return [f'{at}: {got!r:.80} where the reference has {want!r:.80}']
    return []


@pytest.mark.parametrize('case_key', KEYS_ALL)
def test_python_core_answers_each_case_as_the_reference_does(
    case_key: str,
) -> None:
    r"""Exits "a table reached by two paths in one hop read only up to the
    slices it held when the hop began", "one column linking two tables" and
    "a ``from_source`` diamond compared at the payload": python-core runs
    the case on the database on the core route (the native module loaded,
    each of the case's core operations run), and its outcome and each
    request the scripted endpoint received are python-ref's."""
    database = case_key.partition('@')[2] or 'duckdb'
    if database == 'postgres' and not _server():
        return

    def check() -> None:
        run = core_run(database)
        assert '' not in run, run.get('')
        line = run.get(case_key)
        assert line is not None, f'{case_key}: not observed'
        case, _ = cases()[case_key.split('@')[0]]
        identity = line['identity']
        assert identity['native_loaded'] and line['core'] is not None
        missing = sorted(set(case['core']) - set(identity['operations']))
        assert not missing, f'{case_key}: core operations not run: {missing}'
        want = reference_lines()[case_key]
        assert _outcome(line) == _outcome(want), (
            f'{case_key}: {_outcome(line)} where python-ref has {_outcome(want)}'
        )
        # P3F-2 alone permits scalar FK fills and their numeric dtype effect.
        # The shared check retains all other fields and validates reached keys.
        approved = _module(
            'p3f2_payload', Path(__file__).with_name('_p3f2_payload.py')
        )
        found = approved.payload_problem(want, line, case_key)
        assert found is None, f'{case_key}: the payload differs: {found}'

    held(case_key, check)
