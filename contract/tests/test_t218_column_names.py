# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.18's exits on ``python-core`` (the package's test author's): any
column name (KO2-36).

    SDM_POSTGRES_TEST_DSN=... python -m pytest \
        contract/tests/test_t218_column_names.py -q

Written from the design's T2.18 row (each item an exit) and the reference,
before the implementation. ``sdm-testkit``'s ``tests/t218_column_names.rs``
holds ``rust-host`` to the same cases. The vectors,
``contract/column-name-vectors.json``, are recorded from the reference by
``contract/tools/column_names_record.py``, whose ``run`` these tests call
on the core's routes, so both are observed by the same code. A vector is a
character (a dot, a backtick, a line feed, a carriage return) in a
column's name ``a<character>b``, in one role, on one database; its *twin*
is the same data and query with the column named ``a_b``, which the
reference can say. Column names should not matter, so the core's answer
for a vector is its twin's reference answer, the name ``a_b`` read as the
vector's (:func:`as_named`). Each test's one line says what it proves:

- control: the ``name.*`` S1 cases are the design row's (``a.b``, a
  backtick and a line break, each as a feature, the target, the entity, a
  filter and an aggregation's column, the entity users' primary key), each
  the recorder's scenario, each with python-ref's line on all three
  databases;
- control: the vectors are the reference's, and not vacuous: the reference
  leaves each named column out and refuses each query naming one, and
  answers each twin with its column described;
- step 1, "the core keeps every column as data": on each database, the
  public schema python-core describes for each vector is its twin's on the
  reference, the column kept under its own name, with no warning;
- step 2, "``(table, column)`` carried separately through the parsed
  query, a doubled backtick inside a quoted name, line breaks inside
  quotes": on each database, python-core's ``materialize`` of each vector
  answers as its twin does on the reference, the request's payloads the
  twin's with the name read as the vector's;
- "cases written for all three databases" and "the differences from
  ``main`` registered": python-core runs each ``name.*`` case on each
  database through ``predict`` on the core route, and an approved register
  entry citing no temporary approval names each ``<case>@<database>``
  outcome that is not python-ref's;
- "its temporary entries (KO2-45) retired (marked rejected or pointed at a
  permanent decision row; IDs are kept), and T2.1a's temporary-approval
  rule passes for them": P2B-2 is kept, rejected or citing a permanent
  row, and ``ratchet.check_temporary`` names no entry about column names.

The filter role (a column filtering the target's aggregation) needs the
core's ``target_filter`` rule lifted, which is T2.10's: its vectors and
cases are held on the core by ``contract/tests/test_t210_targets.py``,
and the core-run tests here leave them out (:data:`T210_ROLE`).

Gating: ``t218-pending.json``'s ``pending`` names, as ``py:<test>`` (with
``[<database>]`` for a test run per database), the tests that failed when
written. Each must still fail (strict); once one passes it fails, naming
itself, until it is taken out. ``pending`` stays within ``authored``.
Nothing is skipped: where ``SDM_POSTGRES_TEST_DSN`` names no server and
``SDM_REQUIRE_POSTGRES`` is not set, a Postgres leg has nothing to hold and
returns before it is judged, as T2.3's Postgres tests do; with it set, a
missing server fails (H26).
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
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
PENDING = Path(__file__).with_name('t218-pending.json')
VECTORS = ROOT / 'contract/column-name-vectors.json'
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'
DATABASES = ('duckdb', 'sqlite', 'postgres')
#: The register's umbrellas: approved as E92's Phase 2 class, they excuse
#: no case (T1.33).
UMBRELLAS = ('P1B-1', 'P1B-2', 'P1B-3', 'P1B-4')
#: The design row's names, by the case IDs' word for each.
CASE_NAMES = {'dot': 'a.b', 'backtick': 'a`b', 'line-break': 'a\nb'}
#: The character each case name holds, as the vectors label it.
#: The role T2.10's tests hold on the core (``test_t210_targets.py``).
T210_ROLE = 'filter'
CASE_CHARACTERS = {
    'dot': 'dot',
    'backtick': 'backtick',
    'line-break': 'line-feed',
}


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


R = _module(
    'column_names_record', ROOT / 'contract/tools/column_names_record.py'
)
RECORDED: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))
BY_ID = {vector['id']: vector for vector in RECORDED['vectors']}


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
        'contract/tests/t218-pending.json pending'
    )


def test_the_pending_list_only_shrinks() -> None:
    r"""Control: ``pending`` names only tests the test author wrote."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    assert set(gating['pending']) <= set(gating['authored'])


def reachable(database: str) -> bool:
    r"""Whether ``database`` can be run here: DuckDB and SQLite always;
    Postgres where ``SDM_POSTGRES_TEST_DSN`` names a server, and a failure
    where ``SDM_REQUIRE_POSTGRES`` asks for one that is not named (H26)."""
    if database != 'postgres' or os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    return False


# --- the reference: lines, vectors and the register ---------------------------


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


def _cases() -> dict[str, dict[str, Any]]:
    runner = _module('t218_run_python', RUNNER)
    return {
        case['id']: case
        for case, _ in runner.load_cases([ROOT / 'contract/cases/s1'])
        if case['id'].startswith('name.')
    }


def _approved() -> list[dict[str, Any]]:
    entries = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))['entries']
    return [
        entry
        for entry in entries
        if entry.get('status') == 'approved' and entry['id'] not in UMBRELLAS
    ]


def _temporary() -> tuple[str, ...]:
    register = _module('t218_register', ROOT / 'scripts/rust_core/register.py')
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


def as_named(value: Any, twin: dict[str, Any], vector: dict[str, Any]) -> Any:
    r"""The twin's answer read for ``vector``: the twin's query text as the
    vector's, and the twin's column name as the vector's, in every key and
    text; nothing else changes."""
    if isinstance(value, dict):
        return {
            as_named(key, twin, vector): as_named(item, twin, vector)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [as_named(item, twin, vector) for item in value]
    if value == twin['pql']:
        return vector['pql']
    if value == twin['name']:
        return vector['name']
    return value


def _texts(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [*value, *(t for item in value.values() for t in _texts(item))]
    if isinstance(value, list):
        return [t for item in value for t in _texts(item)]
    return [value] if isinstance(value, str) else []


def _vectors(database: str) -> list[dict[str, Any]]:
    r"""The vectors on ``database`` the core-run tests here hold: every
    character's but the filter role's, which T2.10's tests hold."""
    return [
        vector
        for vector in RECORDED['vectors']
        if vector['database'] == database
        and vector['character'] is not None
        and vector['role'] != T210_ROLE
    ]


def test_the_cases_are_the_design_row_s() -> None:
    r"""Control: the ``name.*`` cases are each of ``a.b``, a backtick and a
    line break as a feature, the target, the entity (users' declared
    primary key), a filter and an aggregation's column; each case's data
    and query are the recorder's scenario for its name and role, through
    ``predict``, run on all three databases, with python-ref's committed
    line on each, whose outcome is the case's ``expect``."""
    cases = _cases()
    assert set(cases) == {
        f'name.{word}-{role}' for word in CASE_NAMES for role in R.ROLES
    }
    reference = reference_lines()
    runner = _module('t218_run_python', RUNNER)
    for case_id, case in cases.items():
        word, role = case_id.removeprefix('name.').rsplit('-', 1)
        wanted = R.scenario(role, CASE_NAMES[word])
        assert case['input']['setup'] == wanted['setup'], case_id
        assert case['input']['pql'] == wanted['pql'], case_id
        assert case['input']['materialize'] == wanted['materialize'], case_id
        assert case['until'] == 'predict', case_id
        assert list(case['core']) == [
            'describe_source',
            'materialize',
            'predict',
        ], case_id
        assert runner.case_databases(case) == DATABASES, case_id
        for database in DATABASES:
            line = reference[runner.case_key(case_id, database)]
            assert (line['outcome']['result'], line['outcome']['stage']) == (
                case['expect']['outcome'],
                case['expect']['stage'],
            ), (case_id, database)
            assert line['outcome']['code'] == case['expect'].get('code')
        if role == 'entity':
            assert (
                f'"{CASE_NAMES[word]}" BIGINT PRIMARY KEY'
                in (case['input']['setup'][2])
            ), case_id


def test_the_vectors_are_the_reference_s_and_not_vacuous() -> None:
    r"""Control: recorded from the reference this tree names, every vector
    and twin of the recorder on all three databases. On the reference each
    named column is left out with an SDMWarning, a feature's request
    answered without it and every query naming one refused
    INVALID_REQUEST at ``materialize``; each twin answers, its column
    described, and holds its name only as the whole name or in its query,
    so :func:`as_named` reads every use of it."""
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {RECORDED["reference"]}' in env
    expected = R.vectors()
    assert [v['id'] for v in RECORDED['vectors']] == [v['id'] for v in expected]
    for vector, wanted in zip(RECORDED['vectors'], expected, strict=True):
        assert {k: vector[k] for k in wanted} == wanted, vector['id']
        answer = vector['reference']
        names = [
            column['name']
            for table in (answer['schema'] or {}).get('tables', [])
            for column in table['columns']
        ]
        if vector['character'] is None:
            assert answer['outcome'] == {
                'result': 'success',
                'stage': 'materialize',
            }, vector['id']
            assert R.TWIN in names, vector['id']
            assert answer['sdm_warnings']['from_source'] == 0, vector['id']
            assert answer['payloads'], vector['id']
            assert {
                text
                for text in _texts(answer)
                if R.TWIN in text and text not in (R.TWIN, vector['pql'])
            } == set(), vector['id']
            continue
        assert vector['name'] not in names, vector['id']
        assert answer['sdm_warnings']['from_source'] == 1, vector['id']
        if vector['role'] == 'feature':
            assert answer['outcome']['result'] == 'success', vector['id']
            assert answer['payloads'] != R.digest(
                BY_ID[vector['twin']]['reference']['payloads']
            ), vector['id']
        else:
            assert answer['outcome'] == {
                'result': 'refusal',
                'stage': 'materialize',
                'error': 'StructuredDataError',
                'code': 'INVALID_REQUEST',
            }, vector['id']


def test_the_name_reading_reads_only_the_name() -> None:
    r"""Control: :func:`as_named` reads the twin's name and query as the
    vector's, as keys and texts, and leaves every other value, a text
    holding the name among others included."""
    twin = {'name': 'a_b', 'pql': 'PREDICT users.a_b FOR EACH users.user_id'}
    vector = {
        'name': 'a.b',
        'pql': 'PREDICT users.`a.b` FOR EACH users.user_id',
    }
    given = {
        'a_b': ['a_b', 'xa_b', 1, None],
        'q': twin['pql'],
        'other': {'name': 'a_bc'},
    }
    assert as_named(given, twin, vector) == {
        'a.b': ['a.b', 'xa_b', 1, None],
        'q': vector['pql'],
        'other': {'name': 'a_bc'},
    }


# --- python-core over the vectors -----------------------------------------------


@functools.cache
def _core_answer(vector_id: str) -> dict[str, Any]:
    r"""The vector through the public calls with its database's
    ``from_source`` and ``materialize`` routes forced onto the core."""
    from sdm_client import _routing

    vector = BY_ID[vector_id]
    database = vector['database']
    routes = (
        *_routing.FROM_SOURCE[database],
        *_routing.MATERIALIZE[database],
        *_routing.S0,
    )
    # Not vacuous: the build routes each forced path on the core.
    assert set(routes) <= _routing.CORE_ROUTED
    previous = os.environ.get(_routing.ENVIRONMENT)
    os.environ[_routing.ENVIRONMENT] = ','.join(dict.fromkeys(routes))
    try:
        found: dict[str, Any] = R.run(vector)
        return found
    finally:
        if previous is None:
            os.environ.pop(_routing.ENVIRONMENT, None)
        else:
            os.environ[_routing.ENVIRONMENT] = previous


@pytest.mark.parametrize('database', DATABASES)
def test_the_core_keeps_every_column_whatever_its_name(database: str) -> None:
    r"""Step 1, "the core keeps every column as data": for each vector on
    ``database``, python-core's public schema is its twin's on the
    reference, the twin's name read as the vector's (the column described
    with the twin's type and meaning, a key kept as the key), and
    ``from_source`` warns as often as the twin's (never)."""
    if not reachable(database):
        return

    def check() -> None:
        problems = []
        for vector in _vectors(database):
            twin = BY_ID[vector['twin']]
            got = _core_answer(vector['id'])
            wanted = as_named(twin['reference']['schema'], twin, vector)
            if got['schema'] != wanted:
                problems.append(f'{vector["id"]}: {got["outcome"]}')
            elif (
                got['sdm_warnings'].get('from_source')
                != (twin['reference']['sdm_warnings']['from_source'])
            ):
                problems.append(f'{vector["id"]}: warned at from_source')
        assert not problems, '\n'.join(problems)

    held(
        f'test_the_core_keeps_every_column_whatever_its_name[{database}]', check
    )


@pytest.mark.parametrize('database', DATABASES)
def test_a_query_names_any_column_in_every_role(database: str) -> None:
    r"""Step 2, "``(table, column)`` carried separately through the parsed
    query, a doubled backtick inside a quoted name, line breaks inside
    quotes": for each vector on ``database`` (each character as the
    target, the entity and an aggregation's column, and as a
    feature no query names; the filter role is T2.10's), python-core's
    ``materialize`` answers as its
    twin does on the reference: the outcome, the warnings, and the
    request's payloads, the twin's name and query read as the vector's."""
    if not reachable(database):
        return

    def check() -> None:
        problems = []
        for vector in _vectors(database):
            twin = BY_ID[vector['twin']]
            got = _core_answer(vector['id'])
            if got['outcome'] != twin['reference']['outcome']:
                problems.append(f'{vector["id"]}: {got["outcome"]}')
                continue
            if got['payloads'] != as_named(
                twin['reference']['payloads'], twin, vector
            ):
                problems.append(f"{vector['id']}: not the twin's request")
            if got['sdm_warnings'] != twin['reference']['sdm_warnings']:
                problems.append(f'{vector["id"]}: {got["sdm_warnings"]}')
        assert not problems, '\n'.join(problems)

    held(f'test_a_query_names_any_column_in_every_role[{database}]', check)


# --- python-core over the cases ---------------------------------------------------


@functools.cache
def _cases_on(database: str) -> dict[str, dict[str, Any]]:
    r"""python-core over the ``name.*`` cases on ``database``, as that
    database's job runs it: each observation by its key."""
    paths = sorted(
        path
        for path in (ROOT / 'contract/cases/s1').glob('name.*.json')
        if not path.name.endswith(f'-{T210_ROLE}.json')
    )
    with tempfile.TemporaryDirectory(prefix='t218-cases-') as tmp:
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
                *map(str, paths),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        assert done.returncode == 0, done.stdout + done.stderr
        return {
            json.loads(line)['case']: json.loads(line)
            for line in out.read_text(encoding='utf-8').splitlines()
        }


@pytest.mark.parametrize('database', DATABASES)
def test_each_case_answers_through_predict_and_its_difference_is_registered(
    database: str,
) -> None:
    r""" "Cases written for all three databases" and "the differences from
    ``main`` registered": python-core runs each ``name.*`` case but the
    filter role's (T2.10's) on ``database`` on the core route through
    ``predict`` (as each twin
    answers on the reference), its core operations the case's, and an
    approved entry citing no temporary approval names each key whose
    outcome is not python-ref's committed line's."""
    if not reachable(database):
        return

    def check() -> None:
        runner = _module('t218_run_python', RUNNER)
        observed = _cases_on(database)
        reference = reference_lines()
        problems = []
        for case_id, case in sorted(_cases().items()):
            if case_id.endswith(f'-{T210_ROLE}'):
                continue
            key = runner.case_key(case_id, database)
            line = observed[key]
            outcome = line['outcome']
            if (outcome['result'], outcome['stage']) != ('success', 'predict'):
                problems.append(f'{key}: {outcome}')
                continue
            identity = line['identity']
            if not identity['native_loaded'] or not set(case['core']) <= set(
                identity['operations']
            ):
                problems.append(f'{key}: not on the core route')
            if outcome != reference[key]['outcome'] and not registered_key(key):
                problems.append(f'{key}: no approved, permanent entry names it')
        assert not problems, '\n'.join(problems)

    held(
        'test_each_case_answers_through_predict_and_its_difference_is_'
        f'registered[{database}]',
        check,
    )


# --- the temporary entry -----------------------------------------------------------


def test_the_temporary_entry_is_retired() -> None:
    r""" "Its temporary entries (KO2-45) retired (marked rejected or pointed
    at a permanent decision row; IDs are kept), and T2.1a's
    temporary-approval rule passes for them": P2B-2, the core's refusal of
    a column named with a dot, a backtick or a line break, is still in the
    register, rejected or citing a decision row that is not temporary, and
    ``ratchet.check_temporary`` names no entry whose title is about such a
    column."""

    def check() -> None:
        entries = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))[
            'entries'
        ]
        [entry] = [e for e in entries if e['id'] == 'P2B-2']
        assert entry['status'] == 'rejected' or (
            entry['status'] == 'approved' and _row(entry) not in _temporary()
        ), (entry['status'], _row(entry))
        ratchet = _module('t218_ratchet', ROOT / 'scripts/rust_core/ratchet.py')
        report = ratchet.Report()
        ratchet.check_temporary(report, entries)
        about = [
            e['id']
            for e in entries
            if e.get('status') == 'approved'
            and _row(e) in _temporary()
            and 'T2.18' in json.dumps(e)
        ]
        assert about == [], about
        assert not [p for p in report.problems if 'P2B-2' in p], report.problems

    held('test_the_temporary_entry_is_retired', check)
