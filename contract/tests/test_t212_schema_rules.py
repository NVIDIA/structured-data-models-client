# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.12's exits on ``python-core`` (the package's test author's): the
schema rules.

    SDM_POSTGRES_TEST_DSN=... python -m pytest \
        contract/tests/test_t212_schema_rules.py -q

Written from the design's T2.12 row (each item an exit) and the reference
(python-ref at ``contract/reference-env.txt``'s commit), before the
implementation. ``sdm-testkit``'s ``tests/t212_schema_rules.rs`` holds
``rust-host`` to the same cases, and ``sdm-host``'s
``tests/t212_bool8_runs.rs`` the admission edge of FR P3-8.

The row's exits, and the test that proves each (one line per test says
which):

- "``float_nan`` first, ``float_signed_zero``, ``key_tie``,
  ``key_name_order``, ``composite_key``; cases per rule": the
  ``contract/cases/s1/rule.*`` cases (:data:`RULES`), each the rule's
  input on each database it can hold it, python-ref's line at each case
  file, and python-core answering each as python-ref does on the core
  route: its outcome, the schema ``Schema.from_source`` described, and each
  request the scripted endpoint received (the ``payload`` rule). A key
  whose outcome is not python-ref's passes only under an approved register
  entry naming it, citing no temporary approval (as T2.18's). The float
  cases leave SQLite out under an exclusion asked of the owner: SQLite
  stores a NaN as NULL (the contract loader refuses it by name) and -0.0 as
  0.0, so no SQLite database holds either input
  (:func:`test_sqlite_cannot_hold_the_float_cases`). ``key_name_order``'s
  refusal needs a column equal to one form of its table's name while
  another form also scores; no table name up to six characters over the
  letters of the forms' endings and the key suffixes gives one, so its
  case is the reachable side of the branch (a key equal to a form), which
  the core answers today;
- "the key-type pair matrix for composite keys on all three engines":
  ``contract/composite-key-vectors.json`` (``contract/tools/
  schema_rules_record.py composite``): a two-column key, each column of a
  type of :data:`TYPES`, every ordered pair, given to ``Schema.from_source``,
  declared by the catalog, and declared with a two-column foreign key
  referring to it, on each database; python-core is held to the
  reference's projection of each (outcome, keys and key columns as
  described, schema and requests by digest), a difference only under an
  approved register entry naming ``<vector>@<database>`` with its
  transition (``<from> -> <to>``) or the part that differs (``differs in
  schema`` or ``differs in requests``: a decimal entity key is sent
  exactly where the reference sends a double, KO2-35's P2B-1);
- "FR P3-8's key-cast vectors": ``contract/key-cast-p3-8-vectors.json``
  (``schema_rules_record.py key-cast``, in ``key_cast_record.py``'s form):
  ``Decimal('1E-7')`` into a text key, a ``timedelta`` and a ``time`` into
  an integer key, each held to the reference, or, where marked, to a
  transition an entry of register P1B-34 or P1B-35 names; the float of
  2**63 into int64 is ``key-cast-vectors.json``'s ``int64.float.2pow63``,
  held exactly by T1.32a's test; the ``arrow.bool8`` tag under run-end
  encoding is the Rust admission test's;
- "the nine ``edit.*`` cases acceptance": each is an acceptance case with
  its paths, python-ref's line at its case file on each database, and
  python-core's answer python-ref's (as for the rule cases). Their interim
  gate (``test_edit_cases.py``, ``edit_cases.rs``,
  ``contract/pinned/edits.core.json``) is retired by a later landing and is
  untouched here.

Gating: ``t212-pending.json``'s ``pending`` names, as ``py:<test id>``, the
tests that failed when written. Each must still fail (strict); once one
passes it fails, naming itself, until it is taken out. ``pending`` stays
within ``authored``. Where ``SDM_POSTGRES_TEST_DSN`` names no server and
``SDM_REQUIRE_POSTGRES`` is not set, the Postgres tests have nothing to
hold and return without asserting; with it set, a missing server fails
(H26).
"""

from __future__ import annotations

import fnmatch
import functools
import importlib.util
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
CASES = ROOT / 'contract/cases/s1'
PENDING = Path(__file__).with_name('t212-pending.json')
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'
COMPOSITE = ROOT / 'contract/composite-key-vectors.json'
KEY_CAST = ROOT / 'contract/key-cast-p3-8-vectors.json'
DATABASES = ('duckdb', 'sqlite', 'postgres')
UMBRELLAS = ('P1B-1', 'P1B-2', 'P1B-3', 'P1B-4')

#: The rule cases, by the rule each holds.
RULES = {
    'float_nan': (
        'rule.float-nan-feature',
        'rule.float-nan-aggregated',
        # T2.12 round 1: a NaN that changes what the preview infers (the
        # feature's meaning, the key the tie leaves), and a NaN key sampled
        # at materialize.
        'rule.float-nan-category-double',
        'rule.float-nan-category-float',
        'rule.float-nan-key-tie',
        'rule.float-nan-key-double',
        'rule.float-nan-key-float',
        # T2.12 round 2: a NaN of each sign, as DuckDB's preview keeps
        # them, and a static question over a NaN key.
        'rule.float-signed-nan-key-tie-double',
        'rule.float-signed-nan-key-tie-float',
        'rule.float-signed-nan-category-double',
        'rule.float-signed-nan-category-float',
        'rule.float-nan-key-static-double',
        'rule.float-nan-key-static-float',
        'rule.float-nan-key-static-linked-double',
        # T2.12 round 3: two encodings of a FLOAT NaN, which DuckDB alone
        # stores apart.
        'rule.float-nan-payload-key-tie',
        'rule.float-nan-payload-category',
    ),
    'float_signed_zero': (
        'rule.float-signed-zero-feature',
        'rule.float-signed-zero-key-tie',
    ),
    'key_tie': ('rule.key-tie-fact', 'rule.key-tie-entity'),
    'key_name_order': ('rule.key-name-form',),
    'composite_key': (
        'rule.composite-key-given',
        'rule.composite-key-declared',
        'rule.composite-foreign-key-declared',
    ),
}
#: The cases SQLite cannot hold, each under the exclusion asked of the
#: owner.
FLOATS = (*RULES['float_nan'], *RULES['float_signed_zero'])
EXCLUSION = 'KO2-65'
#: The nine edit cases the row admits as acceptance cases.
EDITS = (
    'edit.accepted-meaning-key-links',
    'edit.key-missing-column',
    'edit.key-time-column',
    'edit.leave-out-the-key',
    'edit.link-not-the-key',
    'edit.meaning-not-carried',
    'edit.meaning-unknown',
    'edit.time-column-not-a-time',
    'edit.time-column-text',
)
RULE_CASES = tuple(case for cases in RULES.values() for case in cases)
CASE_IDS = (*RULE_CASES, *EDITS)


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('t212_run_python', RUNNER)


def recorder() -> ModuleType:
    return _module(
        'schema_rules_record', ROOT / 'contract/tools/schema_rules_record.py'
    )


def key_cast() -> ModuleType:
    return _module(
        'key_cast_record', ROOT / 'contract/tools/key_cast_record.py'
    )


@functools.cache
def cases() -> dict[str, tuple[dict[str, Any], str]]:
    r"""The T2.12 cases by id, each with its file's digest, as the runner
    loads them."""
    loaded = runner().load_cases([CASES / f'{c}.json' for c in CASE_IDS])
    return {case['id']: (case, digest) for case, digest in loaded}


def keys_of(case_id: str) -> list[str]:
    case, _ = cases()[case_id]
    return [
        runner().case_key(case_id, database)
        for database in runner().case_databases(case)
    ]


#: The float cases DuckDB alone can hold: two encodings of a NaN, which
#: Postgres and SQLite do not keep apart.
DUCKDB_ONLY = (
    'rule.float-nan-payload-key-tie',
    'rule.float-nan-payload-category',
)
KEYS_ALL = tuple(
    runner().case_key(c, d)
    for d in DATABASES
    for c in CASE_IDS
    if not (d == 'sqlite' and c in FLOATS)
    and not (d == 'postgres' and c in DUCKDB_ONLY)
)


def _database(key: str) -> str:
    return key.partition('@')[2] or 'duckdb'


def held(name: str, check: Callable[[], None]) -> None:
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
        'contract/tests/t212-pending.json pending'
    )


def _server() -> bool:
    r"""Whether a Postgres server is named; where none is, one must not be
    required (H26)."""
    if os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    return False


def test_the_pending_list_only_shrinks() -> None:
    r"""Control: ``pending`` names only tests the test author wrote."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    assert set(gating['pending']) <= set(gating['authored'])
    assert len(set(gating['authored'])) == len(gating['authored'])


# --- the cases -------------------------------------------------------------------


def test_each_rule_has_its_cases_on_every_database_that_can_hold_it() -> None:
    r"""Exit "cases per rule": each of the five rules has its cases, S1
    acceptance cases on the shop base; every one runs on all three
    databases but the float cases, which leave SQLite out under the
    exclusion asked of the owner."""
    found = {
        case['id']
        for case, _ in runner().load_cases([CASES])
        if case['id'].startswith('rule.')
    }
    assert found == set(RULE_CASES)
    for case_id in RULE_CASES:
        case, _ = cases()[case_id]
        assert case['status'] == 'acceptance' and case['slice'] == 'S1'
        assert case['input']['base'] == 'shop'
        if case_id in DUCKDB_ONLY:
            assert runner().case_databases(case) == ('duckdb',)
            assert case['exclusion'] == EXCLUSION
        elif case_id in FLOATS:
            assert runner().case_databases(case) == ('duckdb', 'postgres')
            assert case['exclusion'] == EXCLUSION
        else:
            assert runner().case_databases(case) == DATABASES
            assert 'exclusion' not in case


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


def _column(case_id: str, table: str, column: str) -> list[Any]:
    case, _ = cases()[case_id]
    connection = _built(case)
    try:
        return [
            row[0]
            for row in connection.execute(
                f'SELECT "{column}" FROM "{table}"'
            ).fetchall()
        ]
    finally:
        connection.close()


def _constraints(case_id: str) -> list[tuple[str, str, list[str], list[str]]]:
    case, _ = cases()[case_id]
    connection = _built(case)
    try:
        return [
            (table, kind, list(columns), list(referenced or []))
            for table, kind, columns, referenced in connection.execute(
                'SELECT table_name, constraint_type, constraint_column_names, '
                'referenced_column_names FROM duckdb_constraints() '
                "WHERE constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')"
            ).fetchall()
        ]
    finally:
        connection.close()


def test_each_case_holds_its_rule_s_input() -> None:
    r"""Exit "cases per rule", not vacuous: each case's data hold the input
    its rule is about, as DuckDB builds it."""
    for case_id, table, column in (
        ('rule.float-nan-feature', 'users', 'score'),
        ('rule.float-nan-aggregated', 'orders', 'price'),
    ):
        values = _column(case_id, table, column)
        assert sum(math.isnan(v) for v in values) == 1, case_id
        assert len(values) > 1, case_id
    zeros = [
        v
        for v in _column('rule.float-signed-zero-feature', 'users', 'score')
        if v == 0.0
    ]
    assert {math.copysign(1, v) for v in zeros} == {-1.0, 1.0}
    # T2.12 round 1: a NaN or -0.0 that changes what is inferred.
    for case_id in (
        'rule.float-nan-category-double',
        'rule.float-nan-category-float',
    ):
        values = _column(case_id, 'users', 'feature')
        nans = sum(math.isnan(v) for v in values)
        assert (len(values), nans) == (1000, 970), case_id
        assert len({v for v in values if not math.isnan(v)}) == 30, case_id
    for case_id, held in (
        ('rule.float-nan-key-tie', 'nan'),
        ('rule.float-signed-zero-key-tie', 'minus-zero'),
    ):
        first = _column(case_id, 'orders', 'order_id')
        second = _column(case_id, 'orders', 'orderid')
        assert second == [float(i) for i in range(80)], case_id
        changed = [
            i
            for i, (a, b) in enumerate(zip(first, second, strict=True))
            if a != b or math.copysign(1, a) != math.copysign(1, b)
        ]
        assert changed == [3], case_id
        if held == 'nan':
            assert math.isnan(first[3])
        else:
            assert first[3] == 0.0 and math.copysign(1, first[3]) == -1.0
            assert math.copysign(1, first[0]) == 1.0 and first[0] == 0.0
    for case_id in ('rule.float-nan-key-double', 'rule.float-nan-key-float'):
        keys = _column(case_id, 'users', 'user_id')
        assert [i for i, v in enumerate(keys) if math.isnan(v)] == [3], case_id
        case, _ = cases()[case_id]
        connection = _built(case)
        try:
            kinds = dict(
                connection.execute(
                    'SELECT table_name, data_type FROM duckdb_columns() '
                    "WHERE column_name = 'user_id'"
                ).fetchall()
            )
        finally:
            connection.close()
        assert kinds['orders'] == 'BIGINT', case_id
        assert kinds['users'] == case_id.rsplit('-', 1)[1].upper(), case_id
    # T2.12 round 2: NaNs of both signs, and a static question over a NaN
    # key.
    for case_id in (
        'rule.float-signed-nan-key-tie-double',
        'rule.float-signed-nan-key-tie-float',
    ):
        first = _column(case_id, 'orders', 'order_id')
        second = _column(case_id, 'orders', 'orderid')
        assert second == [float(i) for i in range(80)], case_id
        nans = [i for i, v in enumerate(first) if math.isnan(v)]
        assert nans == [3, 4], case_id
        assert [math.copysign(1, first[i]) for i in nans] == [1.0, -1.0]
        assert [v for i, v in enumerate(first) if i not in nans] == [
            v for i, v in enumerate(second) if i not in nans
        ], case_id
    for case_id in (
        'rule.float-signed-nan-category-double',
        'rule.float-signed-nan-category-float',
    ):
        values = _column(case_id, 'users', 'feature')
        finite = {v for v in values if not math.isnan(v)}
        signs = {math.copysign(1, v) for v in values if math.isnan(v)}
        assert (len(values), len(finite), signs) == (
            1000,
            49,
            {1.0, -1.0},
        ), case_id
    for case_id, tables in (
        ('rule.float-nan-key-static-double', ['users']),
        ('rule.float-nan-key-static-float', ['users']),
        ('rule.float-nan-key-static-linked-double', ['orders', 'users']),
    ):
        case, _ = cases()[case_id]
        assert (
            case['input']['pql'] == 'PREDICT users.age FOR EACH users.user_id'
        )
        assert 'anchor_time' not in case['input']['materialize'], case_id
        keys = _column(case_id, 'users', 'user_id')
        assert [i for i, v in enumerate(keys) if math.isnan(v)] == [3], case_id
        connection = _built(case)
        try:
            found = sorted(
                row[0]
                for row in connection.execute(
                    'SELECT table_name FROM duckdb_tables() '
                    "WHERE schema_name = 'main'"
                ).fetchall()
            )
        finally:
            connection.close()
        assert found == tables, case_id
    # T2.12 round 3: the NaNs 0x7f800001 and 0x7fc00001, stored apart.
    payloads = {
        'rule.float-nan-payload-key-tie': ('orders', 'order_id', [3, 4]),
        'rule.float-nan-payload-category': (
            'users',
            'feature',
            list(range(49, 1000)),
        ),
    }
    for case_id, (table, column, at) in payloads.items():
        case, _ = cases()[case_id]
        connection = _built(case)
        try:
            array = (
                connection.execute(f'SELECT "{column}" FROM "{table}"')
                .to_arrow_table()
                .column(0)
                .combine_chunks()
            )
        finally:
            connection.close()
        assert str(array.type) == 'float', case_id
        import numpy as np

        bits = np.frombuffer(array.buffers()[1], dtype=np.uint32)[
            array.offset : array.offset + len(array)
        ]
        nans = [
            i
            for i in range(len(bits))
            if (bits[i] & 0x7F800000) == 0x7F800000 and bits[i] & 0x7FFFFF
        ]
        assert nans == at, case_id
        assert {hex(int(bits[i])) for i in nans} == {
            '0x7f800001',
            '0x7fc00001',
        }, case_id
        finite = {
            float(v) for i, v in enumerate(array.to_pylist()) if i not in at
        }
        assert len(finite) == len(bits) - len(at), case_id
    for case_id, table, pair in (
        ('rule.key-tie-fact', 'orders', ('order_id', 'orderid')),
        ('rule.key-tie-entity', 'users', ('user_id', 'userid')),
    ):
        first, second = (_column(case_id, table, c) for c in pair)
        assert first == second and len(set(first)) == len(first), case_id
        assert not _constraints(case_id), f'{case_id}: a declared key'
    assert _column('rule.key-name-form', 'users', 'user') == list(range(8))
    given, _ = cases()['rule.composite-key-given']
    assert given['input']['from_source']['primary_keys'] == {
        'orders': ['order_id', 'line']
    }
    assert given['core']['describe_source']['source']['primary_keys'] == {
        'orders': {'value': ['order_id', 'line']}
    }
    assert ('orders', 'PRIMARY KEY', ['order_no', 'line'], []) in _constraints(
        'rule.composite-key-declared'
    )
    declared = _constraints('rule.composite-foreign-key-declared')
    assert ('users', 'PRIMARY KEY', ['user_id', 'region'], []) in declared
    assert (
        'orders',
        'FOREIGN KEY',
        ['user_id', 'region'],
        ['user_id', 'region'],
    ) in declared


def test_sqlite_cannot_hold_the_float_cases(tmp_path: Path) -> None:
    r"""Exit "cases per rule", the float cases' exclusion: SQLite gives a
    NaN back as NULL and -0.0 as 0.0, so the contract loader refuses each
    by name, and neither rule's input can be put to a host there."""
    loader = _module('t212_databases', ROOT / 'contract/runners/databases.py')
    for case_id, held in (
        *((c, 'a NaN') for c in RULES['float_nan']),
        *((c, '-0.0') for c in RULES['float_signed_zero']),
    ):
        case, _ = cases()[case_id]
        connection = _built(case)
        try:
            with pytest.raises(loader.Unloadable, match=re.escape(held)):
                loader.load(connection, 'sqlite', tmp_path / f'{case_id}.db')
        finally:
            connection.close()
    connection = sqlite3.connect(':memory:')
    try:
        connection.execute('CREATE TABLE t (x REAL)')
        connection.execute('INSERT INTO t VALUES (?)', (-0.0,))
        (stored,) = connection.execute('SELECT x FROM t').fetchone()
    finally:
        connection.close()
    assert math.copysign(1, stored) == 1.0


def test_the_edit_cases_are_acceptance_cases() -> None:
    r"""Exit "the nine ``edit.*`` cases acceptance": each is an acceptance
    case with the paths its call covers, on all three databases."""
    found = {
        case['id']
        for case, _ in runner().load_cases([CASES])
        if case['id'].startswith('edit.')
    }
    assert found == set(EDITS)
    for case_id in EDITS:
        case, _ = cases()[case_id]
        assert case['status'] == 'acceptance', case_id
        assert runner().case_databases(case) == DATABASES
        assert 'schema.from-source/all-tables/duckdb' in case['paths']
        materialized = case['expect']['stage'] == 'materialize'
        assert (
            'query.materialize/indices-regression/duckdb' in case['paths']
        ) == materialized, case_id


# --- the reference -----------------------------------------------------------------


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
    r"""Exits "cases per rule" and "the nine ``edit.*`` cases acceptance":
    python-ref's line for the case on the database, recorded at the case
    file; on DuckDB, the case's expectation."""
    case_id = case_key.split('@')[0]
    case, digest = cases()[case_id]
    line = reference_lines().get(case_key)
    assert line is not None, f'{case_key}: no python-ref line'
    assert line['identity']['runner'] == 'python-ref'
    assert line['identity']['inputs']['case'] == digest
    if _database(case_key) == 'duckdb':
        assert line['outcome']['result'] == case['expect']['outcome']
        assert line['outcome']['stage'] == case['expect']['stage']


def test_no_reference_line_for_a_database_a_case_leaves_out() -> None:
    r"""Control: no python-ref line holds a float case on SQLite."""
    for case_id in FLOATS:
        assert f'{case_id}@sqlite' not in reference_lines()
        if case_id in DUCKDB_ONLY:
            assert f'{case_id}@postgres' not in reference_lines()
            assert keys_of(case_id) == [case_id]
            continue
        assert keys_of(case_id) == [case_id, f'{case_id}@postgres']


# --- python-core against the reference ---------------------------------------------


@functools.cache
def core_run(database: str) -> dict[str, Any]:
    r"""python-core over the T2.12 cases on ``database``, as that database's
    job runs them: each observation by its key, or under ``''`` why the run
    failed whole."""
    with tempfile.TemporaryDirectory(prefix='t212-') as tmp:
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


def _approved() -> list[dict[str, Any]]:
    entries = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))['entries']
    return [
        entry
        for entry in entries
        if entry.get('status') == 'approved' and entry['id'] not in UMBRELLAS
    ]


def _temporary() -> tuple[str, ...]:
    register = _module('t212_register', ROOT / 'scripts/rust_core/register.py')
    return tuple(register.TEMPORARY)


def _row(entry: dict[str, Any]) -> str | None:
    ref = entry.get('decision_ref')
    return ref.get('row') if isinstance(ref, dict) else None


def registered_key(key: str) -> str | None:
    r"""The approved entry, no umbrella and citing no temporary approval,
    naming the line key ``key`` in its ``cases`` or ``case_outcomes``."""
    temporary = _temporary()
    for entry in _approved():
        named = set(entry.get('cases') or []) | {
            item.get('case') for item in entry.get('case_outcomes') or []
        }
        if key in named and _row(entry) not in temporary:
            return str(entry['id'])
    return None


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


#: The public calls in order, and the call each core operation belongs to
#: (``compare.py``'s ``STAGES`` and ``STAGE_OF``).
STAGES = ('from_source', 'materialize', 'predict')
STAGE_OF = {
    'describe_source': 'from_source',
    'validate_query': 'materialize',
    'materialize': 'materialize',
    'predict': 'predict',
}


def held_operations(
    case: dict[str, Any], reference: dict[str, Any]
) -> set[str]:
    r"""The case's core operations a run on a database is held to having
    started: every one, where the reference answers there; where it refuses,
    each of a call before the one it refuses at (``compare.py``'s
    ``core_held``), so a case the reference refuses at materialize on one
    database and answers on another holds predict only where it answers."""
    outcome = reference['outcome']
    if outcome['result'] != 'refusal':
        return set(case['core'])
    stop = STAGES.index(outcome['stage'])
    return {op for op in case['core'] if STAGES.index(STAGE_OF[op]) < stop}


def test_the_operations_held_follow_the_reference_s_stop() -> None:
    r"""Control: an answer holds every operation; a refusal at
    materialize holds describe_source only."""
    case, _ = cases()['rule.float-nan-key-double']
    answered = {'outcome': {'result': 'success', 'stage': 'predict'}}
    refused = {'outcome': {'result': 'refusal', 'stage': 'materialize'}}
    assert held_operations(case, answered) == set(case['core'])
    assert held_operations(case, refused) == {'describe_source'}


@pytest.mark.parametrize('case_key', KEYS_ALL)
def test_python_core_answers_each_case_as_the_reference_does(
    case_key: str,
) -> None:
    r"""Exits "cases per rule" (each rule lifted) and "the nine ``edit.*``
    cases acceptance": python-core runs the case on the database on the
    core route (the native module loaded, each of the case's core
    operations run), and its outcome is python-ref's, or an approved,
    permanent register entry names the key; where both described the
    source, the schema is python-ref's; where both reached predict, each
    request the scripted endpoint received is python-ref's (its request id
    aside)."""
    database = _database(case_key)
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
        want = reference_lines()[case_key]
        missing = sorted(
            held_operations(case, want) - set(identity['operations'])
        )
        assert not missing, f'{case_key}: core operations not run: {missing}'
        if _outcome(line) != _outcome(want):
            entry = registered_key(case_key)
            assert entry is not None, (
                f'{case_key}: {_outcome(line)} where python-ref has '
                f'{_outcome(want)}, and no approved, permanent entry names it'
            )
        schemas = (want['public'].get('schema'), line['public'].get('schema'))
        if None not in schemas:
            found = differences(*schemas)
            assert not found, f'{case_key}: the schema differs: ' + '; '.join(
                found
            )
        if want['outcome']['stage'] == 'predict' == line['outcome']['stage']:
            found = differences(
                recorder().payloads(want), recorder().payloads(line)
            )
            assert not found, f'{case_key}: the payload differs: ' + (
                '; '.join(found)
            )

    held(
        f'test_python_core_answers_each_case_as_the_reference_does[{case_key}]',
        check,
    )


# --- the key-type pair matrix for composite keys ------------------------------------

RECORDED: dict[str, Any] = json.loads(COMPOSITE.read_text(encoding='utf-8'))
#: A ref naming composite vectors and the transition it covers.
TRANSITION = re.compile(
    re.escape('contract/composite-key-vectors.json#')
    + r'(?P<glob>\S+) (?P<old>success|refusal [A-Z][A-Z_]*)'
    + r' -> (?P<new>success|refusal [A-Z][A-Z_]*)'
)


def composite_transition(key: str, old: str, new: str) -> str | None:
    r"""The approved entry, citing no temporary approval, whose ``refs``
    name ``<vector>@<database>`` (a glob) with exactly this transition."""
    temporary = _temporary()
    for entry in _approved():
        if _row(entry) in temporary:
            continue
        for ref in entry.get('refs') or []:
            found = TRANSITION.fullmatch(ref) if isinstance(ref, str) else None
            if (
                found is not None
                and (found['old'], found['new']) == (old, new)
                and fnmatch.fnmatchcase(key, found['glob'])
            ):
                return str(entry['id'])
    return None


#: A ref naming composite vectors whose answer may differ from the
#: reference's in one part, the outcome aside: ``schema`` (the keys, the
#: key columns and the schema as described) or ``requests``.
DIFFERS = re.compile(
    re.escape('contract/composite-key-vectors.json#')
    + r'(?P<glob>\S+) differs in (?P<part>schema|requests)'
)


def composite_differs(key: str, part: str) -> str | None:
    r"""The approved entry, citing no temporary approval, whose ``refs``
    name ``<vector>@<database>`` (a glob) as differing in ``part``: a
    difference of a vector's answer beside its outcome, such as a decimal
    key sent exactly (KO2-35), is registered as any other."""
    temporary = _temporary()
    for entry in _approved():
        if _row(entry) in temporary:
            continue
        for ref in entry.get('refs') or []:
            found = DIFFERS.fullmatch(ref) if isinstance(ref, str) else None
            if (
                found is not None
                and found['part'] == part
                and fnmatch.fnmatchcase(key, found['glob'])
            ):
                return str(entry['id'])
    return None


def test_the_composite_refs_read_and_refuse() -> None:
    r"""Control: a ref reads as a transition or a part only in its own
    form."""
    ref = 'contract/composite-key-vectors.json#given.*@duckdb'
    assert TRANSITION.fullmatch(f'{ref} refusal INVALID_REQUEST -> success')
    assert not TRANSITION.fullmatch(f'{ref} refusal -> success')
    assert DIFFERS.fullmatch(f'{ref} differs in requests')
    assert not DIFFERS.fullmatch(f'{ref} differs in outcome')
    assert not DIFFERS.fullmatch(f'{ref} differs')


def test_the_composite_vectors_are_the_recorder_s() -> None:
    r"""Exit "the key-type pair matrix for composite keys on all three
    engines": the file is the recorder's, recorded at the reference of
    ``contract/reference-env.txt``, every ordered pair of the types in each
    of the three forms, the inputs the recorder's, an answer per
    database."""
    rec = recorder()
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {RECORDED["reference"]}' in env
    assert RECORDED['format'] == rec.COMPOSITE_FORMAT
    assert RECORDED['register_ref'] == rec.COMPOSITE_REF
    assert [tuple(t) for t in RECORDED['types']] == rec.TYPES
    assert len(rec.VECTORS) == len(rec.FORMS) * len(rec.TYPES) ** 2
    assert [
        {k: v for k, v in vector.items() if k != 'reference'}
        for vector in RECORDED['vectors']
    ] == json.loads(json.dumps(rec.VECTORS))
    for vector in RECORDED['vectors']:
        assert set(vector['reference']) == set(DATABASES), vector['id']


def test_the_pair_matrix_is_not_vacuous() -> None:
    r"""Exit "the key-type pair matrix", not vacuous: on DuckDB each
    vector's two key columns have the pair's types, and the catalog
    declares the two-column keys of the declared forms; the reference
    described the given keys as two columns on every database."""
    import duckdb

    rec = recorder()
    sql = dict(rec.TYPES)
    for vector in RECORDED['vectors']:
        connection = duckdb.connect()
        try:
            for statement in rec.key_cast().shop() + vector['setup']:
                connection.execute(statement)
            table, names = vector['key']
            types = dict(
                connection.execute(
                    'SELECT column_name, data_type FROM duckdb_columns() '
                    'WHERE table_name = ?',
                    [table],
                ).fetchall()
            )
            declared = connection.execute(
                'SELECT constraint_column_names FROM duckdb_constraints() '
                "WHERE table_name = ? AND constraint_type = 'PRIMARY KEY'",
                [table],
            ).fetchall()
        finally:
            connection.close()
        wanted = [sql[t].replace(' ', '') for t in vector['types']]
        assert [types[n].replace(' ', '') for n in names] == wanted, vector[
            'id'
        ]
        if vector['form'] == 'given':
            assert declared == []
            for database in DATABASES:
                described = vector['reference'][database]['keys']
                assert described['orders'] == names, (vector['id'], database)
        else:
            assert [list(d[0]) for d in declared] == [names], vector['id']


@functools.cache
def composite_core_run(database: str) -> dict[str, Any]:
    r"""python-core over the composite vectors on ``database``: each
    vector's projection by id, or under ``''`` why the run failed whole."""
    rec = recorder()
    with tempfile.TemporaryDirectory(prefix='t212-composite-') as tmp:
        try:
            observed = rec.run('python-core', rec.VECTORS, database, Path(tmp))
        except RuntimeError as error:
            return {'': str(error)}
    by_id = {vector['id']: vector for vector in rec.VECTORS}
    return {
        vid: rec.project(observation, by_id[vid])
        for vid, observation in observed.items()
    }


def composite_problems(database: str) -> list[str]:
    r"""Each composite vector on ``database`` python-core does not answer as
    the reference did: its outcome (or a registered transition), and, where
    both described the source, the keys, the key columns and the schema;
    where both reached predict, the requests."""
    rec = recorder()
    got_all = composite_core_run(database)
    if '' in got_all:
        return [got_all['']]
    problems = []
    for vector in RECORDED['vectors']:
        key = f'{vector["id"]}@{database}'
        want = vector['reference'][database]
        got = got_all.get(vector['id'])
        if got is None:
            problems.append(f'{key}: not observed')
            continue
        if got['outcome'] != want['outcome']:
            old, new = rec.outcome_class(want), rec.outcome_class(got)
            if old == new or composite_transition(key, old, new) is None:
                problems.append(
                    f'{key}: {got["outcome"]} where the reference has '
                    f'{want["outcome"]}, and no approved entry names '
                    f'{old} -> {new}'
                )
                continue
        if (
            want['schema'] is not None
            and got['schema'] is not None
            and composite_differs(key, 'schema') is None
        ):
            for field in ('keys', 'columns', 'schema'):
                if got[field] != want[field]:
                    problems.append(
                        f'{key}: {field} {got[field]} where the reference '
                        f'has {want[field]}'
                    )
        if (
            want['payloads'] is not None
            and got['payloads'] is not None
            and (got['requests'], got['payloads'])
            != (want['requests'], want['payloads'])
            and composite_differs(key, 'requests') is None
        ):
            problems.append(f"{key}: the requests are not the reference's")
    return problems


@pytest.mark.parametrize('database', DATABASES)
def test_python_core_holds_the_composite_key_pair_matrix(database: str) -> None:
    r"""Exit "the key-type pair matrix for composite keys on all three
    engines": python-core answers every pair of key column types in every
    form on ``database`` as the reference does, a different outcome only
    under an approved entry naming the vector and its transition."""
    if database == 'postgres' and not _server():
        return

    def check() -> None:
        problems = composite_problems(database)
        assert not problems, '\n'.join(problems[:20]) + (
            f'\n... {len(problems)} in all' if len(problems) > 20 else ''
        )

    held(
        f'test_python_core_holds_the_composite_key_pair_matrix[{database}]',
        check,
    )


# --- FR P3-8's key-cast vectors -----------------------------------------------------

KEY_CASTS: dict[str, Any] = json.loads(KEY_CAST.read_text(encoding='utf-8'))
#: A ref naming FR P3-8's vectors and the transition it covers.
KEY_CAST_TRANSITION = re.compile(
    re.escape('contract/key-cast-p3-8-vectors.json#')
    + r'(?P<glob>\S+) (?P<old>success|refusal [A-Z][A-Z_]*)'
    + r' -> (?P<new>success|refusal [A-Z][A-Z_]*)'
)
#: The register entries FR P3-8 names for its notes.
KEY_CAST_ENTRIES = ('P1B-34', 'P1B-35')


def key_cast_entry(vector_id: str, old: str, new: str) -> str | None:
    r"""P1B-34 or P1B-35, approved, if its ``refs`` name the vector with
    exactly this transition."""
    entries = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))['entries']
    for entry in entries:
        if entry['id'] not in KEY_CAST_ENTRIES:
            continue
        if entry.get('status') != 'approved':
            continue
        for ref in entry.get('refs') or []:
            found = (
                KEY_CAST_TRANSITION.fullmatch(ref)
                if isinstance(ref, str)
                else None
            )
            if (
                found is not None
                and (found['old'], found['new']) == (old, new)
                and fnmatch.fnmatchcase(vector_id, found['glob'])
            ):
                return str(entry['id'])
    return None


def test_fr_p3_8_has_a_vector_for_each_edge() -> None:
    r"""Exit "FR P3-8's key-cast vectors": each of the row's edges has its
    vector, recorded at the reference by the recorder, in
    ``key_cast_record.py``'s form, its register mark derived; the float of
    2**63 into int64 is T1.32a's ``int64.float.2pow63``, held exactly
    there."""
    rec = recorder()
    kc = key_cast()
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {KEY_CASTS["reference"]}' in env
    assert KEY_CASTS['format'] == rec.KEY_CAST_FORMAT
    assert [v['id'] for v in KEY_CASTS['vectors']] == [
        f'{key}.{name}' for key, name, _ in rec.KEY_CAST
    ]
    keys = {key['id']: key for key in kc.all_keys()}
    for vector, (key, _, values) in zip(
        KEY_CASTS['vectors'], rec.KEY_CAST, strict=True
    ):
        assert vector['key'] == key
        assert [kc.decode(t) for t in vector['input']] == values()
        assert vector['input'] == [kc.encode(v) for v in values()]
        assert vector['host_arrow'] == kc.host_arrow(values())
        assert {k['id']: k['setup'] for k in KEY_CASTS['keys']}[key] == keys[
            key
        ]['setup']
        marked = [dict(vector, register=None)]
        del marked[0]['register']
        kc.derive_marks(marked)
        assert marked[0]['register'] == vector['register'], vector['id']
    elsewhere = json.loads(
        (ROOT / 'contract/key-cast-vectors.json').read_text(encoding='utf-8')
    )
    two_pow_63 = next(
        v for v in elsewhere['vectors'] if v['id'] == rec.RECORDED_ELSEWHERE
    )
    assert kc.decode(two_pow_63['input'][0]) == float(2**63)
    assert not two_pow_63['register']['required']
    assert (ROOT / 'rust/crates/sdm-host/tests/t212_bool8_runs.rs').is_file()


@pytest.fixture
def core_route(monkeypatch: pytest.MonkeyPatch) -> None:
    from sdm_client import _routing

    monkeypatch.setenv(
        _routing.ENVIRONMENT,
        ','.join(_routing.FROM_SOURCE_DUCKDB + _routing.MATERIALIZE_DUCKDB),
    )


@pytest.mark.parametrize(
    'vector',
    [pytest.param(v, id=v['id']) for v in KEY_CASTS['vectors']],
)
def test_python_core_casts_fr_p3_8_s_entities(
    vector: dict[str, Any], core_route: None
) -> None:
    r"""Exit "FR P3-8's key-cast vectors": python-core casts the vector's
    entities to the key as the reference does, or, where the vector is
    marked, makes a transition register P1B-34 or P1B-35 names (the row's
    "note"); never a refusal by a rule the core does not take yet."""

    def check() -> None:
        kc = key_cast()
        key = {k['id']: k for k in KEY_CASTS['keys']}[vector['key']]
        values = [kc.decode(tagged) for tagged in vector['input']]
        got = kc.observe(key, values)
        assert got.get('code') != 'UNSUPPORTED_REQUEST', got
        public = vector['reference']['public']
        if kc.same(got, public):
            return
        old, new = kc.outcome_class(public), kc.outcome_class(got)
        assert vector['register']['required'], (
            f'{vector["id"]}: got {got}, the reference {public} (held exactly)'
        )
        assert key_cast_entry(vector['id'], old, new) is not None, (
            f'{vector["id"]}: got {got}, the reference {public}: neither '
            f'P1B-34 nor P1B-35 names it with {old} -> {new}'
        )

    held(f'test_python_core_casts_fr_p3_8_s_entities[{vector["id"]}]', check)
