# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.21's exits on ``python-core`` (the package's test author's): exact
numeric keys follow the effective keys.

    SDM_POSTGRES_TEST_DSN=... python -m pytest \
        contract/tests/test_t221_exact_keys.py -q

Written from the brief (the mid-phase design check's finding, its rule:
learnings lesson 19, design choice 5, KO2-35) and the reference
(python-ref at ``contract/reference-env.txt``'s commit), before the
implementation. ``sdm-testkit``'s ``tests/t221_exact_keys.rs`` holds
``rust-host`` to the same rule over a host provider that hands a key over
as Postgres's provider does.

The reproduction: a Postgres key column of unconstrained ``numeric``, which
the Postgres provider hands the core as exact text (``sdm:exact_number``),
holding ids from 2**53, where a double holds only every other integer.
Declared a key by the catalog, the core keeps it as its text and answers;
given to ``Schema.from_source``, or linked by a given or edited
relationship, the core reads it as doubles (decided from the catalog
alone, before the keys given are applied) and refuses 2**53 + 1 as
inexact. The contract loader cannot make that type (it declares a
``DECIMAL(p, s)`` as ``numeric(p,s)``), so the Postgres reproduction is
``contract/exact-key-vectors.json``, recorded from python-ref by
``contract/tools/exact_keys_record.py``, whose ``observe`` this test calls
on the core's Postgres routes, as T2.3's Postgres regressions are held.

Each test's one line says the exit it proves:

- "the reproduction and its supplied-foreign-key and edited-relationship
  variants work on Postgres as with a catalog key": python-core answers
  ``pk.given`` (the reproduction: the key given), ``fk.given`` (the keys
  and the link given) and ``fk.edited`` (the link set by an edit) exactly
  as python-ref answers the same tables with the key and link declared
  (``pk.catalog``, ``fk.catalog``): its outcome, public schema (each key
  column its text) and request (the payloads' digest). python-ref itself
  reads the key as doubles there and refuses (``INVALID_REQUEST``: the
  ids merged), so python-core's success is a difference from it, which an
  approved register entry must declare (``refusal INVALID_REQUEST ->
  success``, KO2-35's "never merges distinct keys"); where python-ref
  answers without merging (the same tables of ``bigint``), it answers a
  given, an edited and a declared key alike, which is what the vectors
  hold python-core to;
- "the controls are unchanged": python-core answers ``pk.catalog`` and
  ``fk.catalog`` as python-ref does;
- "cases in the corpus on all three databases": ``key.past-2-53.*``, a
  ``DECIMAL(38, 0)`` id past 2**53 (DuckDB's ``DECIMAL``, SQLite's text,
  Postgres's ``numeric(38,0)``), its key given (``given-key``), its keys
  and link given (``given-link``), its link set by an edit
  (``edited-link``), and the catalog's (``catalog``, the control), each
  run through predict on DuckDB, SQLite and Postgres, python-ref's line
  for it on each recorded at its case file, and python-core answering
  each as python-ref does (its outcome and each request the scripted
  endpoint received, the ``payload`` rule: but its request id).
  ``catalog`` on Postgres is the one that differs when written, and not
  for T2.21's reason: python-ref reads a ``numeric(38,0)`` as text where
  the catalog declares it a key and as ``int64`` otherwise (so its
  ``given-*`` lines send ``int64``), while python-core keeps its exact
  ``numeric(p,s)`` reading whatever the key's role and sends ``int64`` in
  all four; the request differs in the three key columns' ``dtype`` only.
  The orchestrator decided (2026-10-20) that this is KO2-35's exact-decimal
  class (register P2C-3), which the implementer registers: a payload
  difference is accepted only at the fields an approved register entry
  lists for the case and python-core under ``more_exact``, read and left
  out as the gate's comparator does (``compare.more_exact``,
  ``compare._left_out``), so the key leaves pending once the entry is in.
  DuckDB's and SQLite's lines, and the ``given-*`` lines on Postgres,
  python-core already answers as python-ref does.

Gating: ``t221-pending.json``'s ``pending`` names, as ``py:<key>``, the
keys whose test failed when written. Each must still fail (strict); once
one passes it fails, naming itself, until it is taken out. ``pending``
stays within ``authored``. Where ``SDM_POSTGRES_TEST_DSN`` names no server
and ``SDM_REQUIRE_POSTGRES`` is not set, the Postgres keys have nothing to
hold and return without asserting; with it set, a missing server fails
(H26).
"""

from __future__ import annotations

import copy
import fnmatch
import functools
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
CASES = ROOT / 'contract/cases/s1'
VECTORS = ROOT / 'contract/exact-key-vectors.json'
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'
PENDING = Path(__file__).with_name('t221-pending.json')
DATABASES = ('duckdb', 'sqlite', 'postgres')
FIRST = 2**53
GIVEN_KEY = 'key.past-2-53.given-key'
GIVEN_LINK = 'key.past-2-53.given-link'
EDITED_LINK = 'key.past-2-53.edited-link'
CATALOG = 'key.past-2-53.catalog'
CASE_IDS = (GIVEN_KEY, GIVEN_LINK, EDITED_LINK, CATALOG)
LINK = {
    'source_table': 'orders',
    'source_column': 'user_id',
    'target_table': 'users',
    'target_column': 'user_id',
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
    return _module('t221_run_python', RUNNER)


R = _module('exact_keys_record', ROOT / 'contract/tools/exact_keys_record.py')


def key(case_id: str, database: str) -> str:
    return runner().case_key(case_id, database)


CASE_KEYS = tuple(key(c, d) for d in DATABASES for c in CASE_IDS)
VECTOR_KEYS = tuple(f'vector:{v["id"]}' for v in R.VECTORS)
GIVEN_VECTORS = tuple(R.CONTROL)


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
        'contract/tests/t221-pending.json pending'
    )


def test_the_pending_list_only_shrinks() -> None:
    r"""Control: ``pending`` names only keys the test author wrote, and
    ``authored`` is every case's key on each database and every vector's,
    with the register's, on python-core, and each case's on rust-host's
    marked source (``t221_exact_keys.rs``)."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    assert set(gating['pending']) <= set(gating['authored'])
    assert set(gating['authored']) == (
        {f'py:{k}' for k in (*CASE_KEYS, *VECTOR_KEYS)}
        | {f'py:register:{v}' for v in GIVEN_VECTORS}
        | {f'rust:{c}@memory-marked' for c in CASE_IDS}
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


# --- the Postgres reproduction ----------------------------------------------------

RECORDED: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))
BY_ID = {vector['id']: vector for vector in RECORDED['vectors']}


def test_the_vectors_are_the_reference_s_on_postgres() -> None:
    r"""The vectors are python-ref's on Postgres 17, recorded from the
    recorder's own vectors, and not vacuous: each key column an
    unconstrained ``numeric``; python-ref answers each control, the key its
    text; and reads a given or edited key as doubles, refusing 2**53 + 1
    (the ids merged), the reproduction the brief names."""
    assert RECORDED['format'] == R.FORMAT
    assert RECORDED['recorded_by'] == 'contract/tools/exact_keys_record.py'
    assert RECORDED['register_ref'] == R.REGISTER_REF
    assert RECORDED['environment']['postgres'] == 17
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {RECORDED["reference"]}' in env
    assert [v['id'] for v in RECORDED['vectors']] == [
        v['id'] for v in R.VECTORS
    ]
    for kind in R.VECTORS:
        for name in kind:
            assert BY_ID[kind['id']][name] == kind[name], (kind['id'], name)
    # 2**53 + 1, the entity asked for, is a double no more.
    assert float(R.ASKED) == float(R.FIRST) and R.ASKED != R.FIRST
    for vector in RECORDED['vectors']:
        answer = vector['reference']
        assert answer['postgres'] == {
            'users.user_id': 'numeric',
            'orders.user_id': 'numeric',
        }, vector['id']
        declared = vector['id'].endswith('.catalog')
        assert any('PRIMARY KEY' in s for s in vector['setup']) == declared
        dtypes = {
            table['name']: column['dtype']
            for table in answer['schema']['tables']
            for column in table['columns']
            if column['name'] == 'user_id'
        }
        if declared:
            assert answer['outcome']['result'] == 'success', vector['id']
            assert answer['payloads'] is not None, vector['id']
            assert dtypes == {'users': 'string', 'orders': 'string'}
            continue
        assert answer['outcome'] == {
            'result': 'refusal',
            'stage': 'materialize',
            'error': 'StructuredDataError',
            'code': 'INVALID_REQUEST',
        }, vector['id']
        assert dtypes == {'users': 'float64', 'orders': 'float64'}
    assert BY_ID['pk.given']['from_source'] == {
        'primary_keys': {'users': 'user_id'}
    }
    assert BY_ID['fk.given']['from_source']['relationships'] == [LINK]
    assert BY_ID['fk.edited']['from_source']['relationships'] == []
    assert BY_ID['fk.edited']['edits'] == [
        {'op': 'set_relationships', 'relationships': [LINK]}
    ]


def _projection(answer: dict[str, Any]) -> dict[str, Any]:
    r"""An answer as a given key is held to its control's: its outcome,
    public schema and request."""
    return {k: answer[k] for k in ('outcome', 'schema', 'payloads')}


def _differs(got: dict[str, Any], want: dict[str, Any]) -> list[str]:
    return sorted(k for k in want if got.get(k) != want[k])


@pytest.mark.parametrize('vector_id', sorted(set(R.CONTROL.values())))
def test_python_core_answers_a_catalog_key_as_the_reference(
    vector_id: str,
) -> None:
    r"""Exit "as with a catalog key", the controls: python-core answers the
    tables with the key and link declared as python-ref does (outcome,
    public schema, request, each column's type), the key its text."""
    if not _server():
        return

    def check() -> None:
        want = BY_ID[vector_id]['reference']
        got = R.observe(
            BY_ID[vector_id], os.environ['SDM_POSTGRES_TEST_DSN'], True
        )
        found = _differs(got, want)
        assert not found, (
            f'{vector_id}: {found} differ: got {got["outcome"]}, the '
            f'reference {want["outcome"]}'
        )

    held(f'vector:{vector_id}', check)


@pytest.mark.parametrize('vector_id', GIVEN_VECTORS)
def test_python_core_answers_a_given_or_edited_key_as_a_catalog_key(
    vector_id: str,
) -> None:
    r"""Exit "the reproduction and its supplied-foreign-key and
    edited-relationship variants work on Postgres as with a catalog key":
    python-core answers the key given (``pk.given``), the keys and link
    given (``fk.given``) or the link set by an edit (``fk.edited``), nothing
    declared, exactly as python-ref answers its control with the key and
    link declared: the outcome, the public schema (each ``user_id`` its
    text) and the request."""
    if not _server():
        return

    def check() -> None:
        control = R.CONTROL[vector_id]
        want = _projection(BY_ID[control]['reference'])
        got = _projection(
            R.observe(
                BY_ID[vector_id], os.environ['SDM_POSTGRES_TEST_DSN'], True
            )
        )
        found = _differs(got, want)
        assert not found, (
            f'{vector_id}: {found} differ from python-ref on {control}: got '
            f'{got["outcome"]}, schema {got["schema"]}'
        )

    held(f'vector:{vector_id}', check)


TRANSITION = re.compile(
    re.escape(R.REGISTER_REF)
    + r'(?P<glob>\S+) (?P<old>success|refusal [A-Z][A-Z_]*)'
    + r' -> (?P<new>success|refusal [A-Z][A-Z_]*)'
)


def covered(vector_id: str, old: str, new: str) -> str | None:
    r"""The approved entry whose ``refs`` name the vector with exactly the
    transition ``old -> new``, if any."""
    document = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))
    for entry in document['entries']:
        if entry.get('status') != 'approved':
            continue
        for ref in entry.get('refs') or []:
            found = TRANSITION.fullmatch(ref) if isinstance(ref, str) else None
            if (
                found is not None
                and (found['old'], found['new']) == (old, new)
                and fnmatch.fnmatchcase(vector_id, found['glob'])
            ):
                return str(entry['id'])
    return None


@pytest.mark.parametrize('vector_id', GIVEN_VECTORS)
def test_the_reference_s_refusal_is_a_registered_difference(
    vector_id: str,
) -> None:
    r"""Exit "match its success; where it differs, KO2-35's registered
    differences apply": python-ref refuses each given or edited key (the
    ids merged as doubles) where python-core answers, so an approved entry
    names the vector with ``refusal INVALID_REQUEST -> success``."""

    def check() -> None:
        assert covered(vector_id, 'refusal INVALID_REQUEST', 'success'), (
            f'no approved entry names {R.REGISTER_REF}{vector_id} '
            'refusal INVALID_REQUEST -> success'
        )

    held(f'register:{vector_id}', check)


# --- the corpus's cases --------------------------------------------------------------


@functools.cache
def cases() -> dict[str, tuple[dict[str, Any], str]]:
    r"""The T2.21 cases by id, each with its digest, as the runner loads
    them."""
    loaded = runner().load_cases([CASES / f'{c}.json' for c in CASE_IDS])
    return {case['id']: (case, digest) for case, digest in loaded}


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


@pytest.mark.parametrize('case_id', CASE_IDS)
def test_each_case_holds_an_exact_key_past_2_53(case_id: str) -> None:
    r"""Exit "cases in the corpus on all three databases": each case runs
    through predict on DuckDB, SQLite and Postgres over ``user_id`` a
    ``DECIMAL(38, 0)`` in both tables, every id past 2**53, two of them
    one double (so a key read as doubles merges them), asked for 2**53 + 1
    and 2**53 + 2; the key and link declared only in ``catalog``, given in
    ``given-key`` (the key) and ``given-link`` (the keys and link), and the
    link set by an edit in ``edited-link`` (given no link)."""
    case, _ = cases()[case_id]
    given = case['input']
    assert case['status'] == 'acceptance' and case['slice'] == 'S1'
    assert case['until'] == 'predict'
    assert runner().case_databases(case) == DATABASES
    assert given['pql'] == (
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
    )
    assert given['materialize']['indices'] == [FIRST + 1, FIRST + 2]
    connection = _built(case)
    try:
        types = dict(
            connection.execute(
                'SELECT table_name, data_type FROM duckdb_columns() '
                "WHERE column_name = 'user_id'"
            ).fetchall()
        )
        ids = [
            Decimal(row[0])
            for row in connection.execute(
                'SELECT user_id FROM users UNION SELECT user_id FROM orders'
            ).fetchall()
        ]
        constraints = sorted(
            row[0]
            for row in connection.execute(
                'SELECT constraint_type FROM duckdb_constraints() '
                "WHERE constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')"
            ).fetchall()
        )
    finally:
        connection.close()
    assert types == {'users': 'DECIMAL(38,0)', 'orders': 'DECIMAL(38,0)'}
    assert min(ids) >= FIRST
    assert len({float(i) for i in ids}) < len(set(ids))
    if case_id == CATALOG:
        assert 'from_source' not in given and 'edits' not in given
        assert constraints == ['FOREIGN KEY', 'PRIMARY KEY', 'PRIMARY KEY']
        return
    assert constraints == []
    keys = {'users': 'user_id', 'orders': 'order_id'}
    expected = {
        GIVEN_KEY: ({'primary_keys': {'users': 'user_id'}}, None),
        GIVEN_LINK: ({'primary_keys': keys, 'relationships': [LINK]}, None),
        EDITED_LINK: (
            {'primary_keys': keys, 'relationships': []},
            [{'op': 'set_relationships', 'relationships': [LINK]}],
        ),
    }[case_id]
    assert (given['from_source'], given.get('edits')) == expected


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


@pytest.mark.parametrize('case_key', CASE_KEYS)
def test_python_ref_s_line_is_at_its_case_file(case_key: str) -> None:
    r"""Exit "cases in the corpus on all three databases": python-ref's
    line for the case on the database, recorded at the case file, answered
    through predict with one request, so the payload is compared there."""
    case_id = case_key.split('@')[0]
    _, digest = cases()[case_id]
    line = reference_lines().get(case_key)
    assert line is not None, f'{case_key}: no python-ref line'
    assert line['identity']['runner'] == 'python-ref'
    assert line['identity']['inputs']['case'] == digest
    assert line['outcome']['result'] == 'success'
    assert line['outcome']['stage'] == 'predict'
    assert len(line['public']['payloads']) == 1


@functools.cache
def core_run(database: str) -> dict[str, Any]:
    r"""python-core over the T2.21 cases on ``database``, as that database's
    job runs them: each observation by its key, or under ``''`` why the run
    failed whole."""
    with tempfile.TemporaryDirectory(prefix='t221-') as tmp:
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
    r"""Each request the scripted endpoint received, as the ``payload`` rule
    compares it (``scripts/rust_core/compare.py``): its method, path and
    parsed body, the body's request id, drawn at random for every request,
    left out (``compare-policy.yaml``)."""
    out = []
    for payload in line['public']['payloads']:
        parsed = copy.deepcopy(payload.get('parsed'))
        if isinstance(parsed, dict) and isinstance(
            parsed.get('metadata'), dict
        ):
            parsed['metadata'].pop('request_id', None)
        out.append([payload['method'], payload['path'], parsed])
    return out


def compare() -> ModuleType:
    return _module('t221_compare', ROOT / 'scripts/rust_core/compare.py')


def excused(case_key: str) -> tuple[str, ...]:
    r"""The payload fields an approved register entry lists for the case on
    python-core under ``more_exact``, read as the gate's comparator reads
    them (``compare.more_exact``: an entry the owner approved, a core
    runner, a path under a compared root); none where no entry names it."""
    return compare().more_exact(REGISTER).get((case_key, 'python-core'), ())


def payload_differs(
    line: dict[str, Any], want: dict[str, Any], fields: tuple[str, ...]
) -> bool:
    r"""Whether python-core's requests differ from python-ref's, each value
    at a field of ``fields`` (:func:`excused`) left out on both sides as the gate's
    comparator leaves it out (``compare._left_out``: only a value both
    sides hold there, so keys, lengths and every other field still
    count)."""
    mine, theirs = payloads(line), payloads(want)
    if fields:
        mine, theirs = compare()._left_out(
            mine, theirs, 'public.payloads', fields
        )
    return bool(mine != theirs)


def public_differs(
    line: dict[str, Any],
    want: dict[str, Any],
    part: str,
    fields: tuple[str, ...],
) -> str | None:
    r"""Where python-core's public ``part`` (``schema``, the schema the user
    sees, or ``result``, the prediction it returns) first differs from
    python-ref's, as a path, the value at each field of ``fields``
    (:func:`excused`) left out on both sides: the gate's comparator's own
    ``schema`` and ``result`` rules (``compare._path_of``); none where they
    agree."""
    return compare()._path_of(
        line['public'][part], want['public'][part], f'public.{part}', fields
    )


@pytest.mark.parametrize('case_key', CASE_KEYS)
def test_python_core_answers_each_case_as_the_reference_does(
    case_key: str,
) -> None:
    r"""Exit "cases in the corpus on all three databases": python-core runs
    the case on the database on the core route (the native module loaded,
    each of the case's core operations run), and its outcome and each
    request the scripted endpoint received are python-ref's, and so are the
    schema the user sees and the prediction returned (the comparator's
    ``schema``, ``result`` and ``payload`` rules), but at a field an
    approved register entry lists for the case and python-core under
    ``more_exact`` (as the gate's comparator excuses it)."""
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
        fields = excused(case_key)
        for part in ('schema', 'result'):
            found = public_differs(line, want, part, fields)
            assert found is None, (
                f'{case_key}: the public {part} differs at {found}, a field '
                'no approved register entry lists for python-core under '
                'more_exact'
            )
        assert not payload_differs(line, want, fields), (
            f'{case_key}: the payload differs, at a field no approved '
            'register entry lists for python-core under more_exact'
        )

    held(case_key, check)


def test_a_registered_field_excuses_that_field_alone() -> None:
    r"""Control: :func:`payload_differs` leaves out only the fields given,
    as the comparator does: the key's ``dtype`` named, a request differing
    there alone is the reference's; differing at another field too, or
    with nothing named, it differs."""
    field = (
        'public.payloads/*/2/schema/related_tables/users/columns/user_id/dtype'
    )

    def line(dtype: str, rows: int) -> dict[str, Any]:
        body = {
            'metadata': {'request_id': f'r{rows}{dtype}'},
            'schema': {
                'related_tables': {
                    'users': {'columns': {'user_id': {'dtype': dtype}}}
                }
            },
            'rows': rows,
        }
        return {
            'public': {
                'payloads': [{'method': 'POST', 'path': '/p', 'parsed': body}]
            }
        }

    want = line('string', 1)
    assert not payload_differs(line('string', 1), want, ())
    assert payload_differs(line('int64', 1), want, ())
    assert not payload_differs(line('int64', 1), want, (field,))
    assert payload_differs(line('int64', 2), want, (field,))
    assert payload_differs(
        line('int64', 1), want, (field.replace('users', 'orders'),)
    )


def test_a_registered_field_excuses_that_schema_or_result_field_alone() -> None:
    r"""Control: :func:`public_differs` leaves out only the fields given, as
    the comparator does, in the public schema and result alike."""
    schema = 'public.schema/tables/0/columns/0/dtype'
    result = 'public.result/columns/0/dtype'

    def line(dtype: str, other: str) -> dict[str, Any]:
        return {
            'public': {
                'schema': {
                    'tables': [
                        {
                            'name': 'users',
                            'columns': [{'name': 'user_id', 'dtype': dtype}],
                            'primary_key': other,
                        }
                    ]
                },
                'result': {
                    'columns': [{'name': 'ENTITY', 'dtype': dtype}],
                    'digest': other,
                },
            }
        }

    want = line('string', 'a')
    for part, field in (('schema', schema), ('result', result)):
        assert public_differs(line('string', 'a'), want, part, ()) is None
        assert public_differs(line('int64', 'a'), want, part, ()) is not None
        assert public_differs(line('int64', 'a'), want, part, (field,)) is None
        assert (
            public_differs(line('int64', 'b'), want, part, (field,)) is not None
        )
