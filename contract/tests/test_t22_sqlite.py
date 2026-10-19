# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.2's exits on ``python-core`` (the package's test author's).

    python -m pytest contract/tests/test_t22_sqlite.py -q

Written from the design's T2.2 row (each item an exit) and the reference,
before the implementation. ``sdm-testkit``'s ``tests/t22_sqlite.rs`` holds
``rust-host`` to the same exits. Each test's one line says the exit it
proves:

- the staging rule's marker: the core reports a SQLite provider, so the
  SQLite jobs and the gate run the core legs there (T2.1a's rule);
- "every S1 case on SQLite": python-core runs every S1 case on SQLite on
  the core route, each case's outcome python-ref's committed
  ``<case>@sqlite`` line's or an approved register entry naming that key;
  ``compare.py --database sqlite`` holds every other field there, as the
  SQLite job and the gate run it;
- "L4 goldens executed": every statement python-core ran on SQLite over the
  cases is a golden, pinned verbatim as a string literal in
  ``rust/crates/sdm-data/tests/sqlite*.rs``, so a SQL change without a
  golden update fails;
- "the key-type pair matrix on SQLite" and "the decimal-key pair matrix,
  equal keys across scales included": every vector of
  ``contract/sqlite-key-pair-vectors.json`` (recorded from the reference by
  ``contract/tools/sqlite_key_pairs_record.py``) is the reference's answer
  on python-core, and every request joins each row to its own entity.

Gating: ``t22-pending.json``'s ``pending`` names, as ``py:<test>``, the
tests that failed when written. Each must still fail (strict); once one
passes it fails, naming itself, until it is taken out. ``pending`` stays
within ``authored``. Nothing is skipped.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
PENDING = Path(__file__).with_name('t22-pending.json')
VECTORS_FILE = ROOT / 'contract/sqlite-key-pair-vectors.json'
GOLDENS = ROOT / 'rust/crates/sdm-data/tests'
DATABASE = 'sqlite'
#: The register's umbrellas: approved as E92's Phase 2 class, they excuse
#: no case and no vector (T1.33).
UMBRELLAS = ('P1B-1', 'P1B-2', 'P1B-3', 'P1B-4')


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


COMPARE = _module('t22_compare', ROOT / 'scripts/rust_core/compare.py')
M = _module(
    'sqlite_key_pairs_record',
    ROOT / 'contract/tools/sqlite_key_pairs_record.py',
)
R = M.R
E = R.E


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
        'contract/tests/t22-pending.json pending'
    )


def test_the_pending_list_only_shrinks() -> None:
    r"""Control: ``pending`` names only tests the test author wrote."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    assert set(gating['pending']) <= set(gating['authored'])


# --- the reference and the register --------------------------------------------


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


def _approved() -> list[dict[str, Any]]:
    path = ROOT / 'docs/design/rust-core/register.yaml'
    entries = yaml.safe_load(path.read_text(encoding='utf-8'))['entries']
    return [
        entry
        for entry in entries
        if entry.get('status') == 'approved' and entry['id'] not in UMBRELLAS
    ]


def registered_key(key: str) -> str | None:
    r"""The approved entry, no umbrella, naming the line key ``key`` (a
    ``<case>@sqlite``) in its ``cases`` or ``case_outcomes``."""
    for entry in _approved():
        named = set(entry.get('cases') or []) | {
            item.get('case') for item in entry.get('case_outcomes') or []
        }
        if key in named:
            return str(entry['id'])
    return None


def registered_vector(vector_id: str) -> str | None:
    r"""The approved entry, no umbrella, whose ``refs`` name the vector."""
    wanted = M.REGISTER_REF + vector_id
    for entry in _approved():
        if any(
            str(ref).split(' ')[0] == wanted for ref in entry.get('refs') or []
        ):
            return str(entry['id'])
    return None


# --- the staging marker ----------------------------------------------------------


def test_the_core_reports_a_sqlite_provider() -> None:
    r"""Exit (the staging rule's marker, T2.1a): the core under test reports
    a SQLite provider (``sdm_client._native.PROVIDERS``), so a SQLite job
    and the gate run its core legs there."""

    def check() -> None:
        from sdm_client import _native

        assert DATABASE in tuple(_native.PROVIDERS)

    held('test_the_core_reports_a_sqlite_provider', check)


# --- every S1 case on SQLite -------------------------------------------------------


def _s1() -> list[dict[str, Any]]:
    runner = _module('t22_run_python', RUNNER)
    return [
        case
        for case, _ in runner.load_cases([ROOT / 'contract/cases/s1'])
        if DATABASE in runner.case_databases(case)
    ]


@functools.cache
def _cases_on_sqlite() -> dict[str, dict[str, Any]]:
    r"""python-core over every case on SQLite, as the SQLite job runs it:
    each observation by its key."""
    with tempfile.TemporaryDirectory(prefix='t22-cases-') as tmp:
        out = Path(tmp) / 'python-core.jsonl'
        done = subprocess.run(
            [
                sys.executable,
                '-I',
                str(RUNNER),
                '--mode',
                'python-core',
                '--database',
                DATABASE,
                '--out',
                str(out),
                str(ROOT / 'contract/cases'),
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


def _outcome(observation: dict[str, Any]) -> tuple[Any, ...]:
    outcome = observation['outcome']
    return (
        outcome['result'],
        outcome['stage'],
        outcome.get('error'),
        outcome.get('code'),
    )


def test_python_core_runs_every_s1_case_on_sqlite_as_the_reference_does() -> (
    None
):
    r"""Exit "every S1 case on SQLite": python-core observes every S1 case
    on SQLite, keyed ``<case>@sqlite``, on the core route (the native module
    loaded, and the case's core operations run as ``compare.py`` holds a
    database line to them: ``line_stop`` and ``core_held``, those of the
    calls before the reference line's stop), and its outcome (result,
    stage, class, code) is python-ref's committed line's there, or an
    approved register entry names the key."""

    def check() -> None:
        observed = _cases_on_sqlite()
        reference = reference_lines()
        cases = _s1()
        assert len(cases) >= 52
        problems = []
        for case in cases:
            key = f'{case["id"]}@{DATABASE}'
            line = observed.get(key)
            if line is None:
                problems.append(f'{key}: not observed')
                continue
            identity = line['identity']
            # The core operations the line is held to, as the comparator
            # holds them (T2.1e): those of the calls before the reference
            # line's stop, and every one the runner ran.
            held_ops = COMPARE.core_held(
                case, line, COMPARE.line_stop(None, reference[key])
            )['core']
            missing = sorted(set(held_ops) - set(identity['operations']))
            if (
                not identity['native_loaded']
                or missing
                or (held_ops and not line['core'])
            ):
                problems.append(f'{key}: not on the core route ({missing})')
            if _outcome(line) != _outcome(reference[key]) and not (
                registered_key(key)
            ):
                problems.append(
                    f'{key}: {_outcome(line)} where python-ref has '
                    f'{_outcome(reference[key])}, no approved entry names it'
                )
        assert not problems, '\n'.join(problems)

    held(
        'test_python_core_runs_every_s1_case_on_sqlite_as_the_reference_does',
        check,
    )


# --- the L4 goldens ----------------------------------------------------------------

_RAW = re.compile(r'r(#*)"(.*?)"\1', re.S)
_ESCAPES = {'n': '\n', 't': '\t', 'r': '\r', '0': '\0'}


def literals(source: str) -> list[str]:
    r"""Every string literal of a Rust source, its escapes read: ``"..."``
    (``\n``, ``\t``, ``\r``, ``\0``, ``\\``, ``\"``, ``\'`` and a line
    continuation) and raw ``r"..."``, ``r#"..."#``; comments and character
    literals skipped (``sdm-testkit``'s ``t132a_keys_sql.rs`` reads them
    so)."""
    out: list[str] = []
    i = 0
    while i < len(source):
        c = source[i]
        if source.startswith('//', i):
            end = source.find('\n', i)
            i = len(source) if end < 0 else end
        elif c == "'":
            if source[i + 2 : i + 3] == "'":
                i += 3
            elif source[i + 1 : i + 2] == '\\':
                close = source.find("'", i + 2, i + 12)
                i = i + 1 if close < 0 else close + 1
            else:
                i += 1
        elif c == 'r' and (
            i == 0 or not (source[i - 1].isalnum() or source[i - 1] == '_')
        ):
            raw = _RAW.match(source, i)
            if raw is None:
                i += 1
                continue
            out.append(raw.group(2))
            i = raw.end()
        elif c == '"':
            text = []
            i += 1
            while i < len(source) and source[i] != '"':
                if source[i] == '\\':
                    i += 1
                    if source[i] == '\n':
                        while i + 1 < len(source) and source[i + 1].isspace():
                            i += 1
                    else:
                        text.append(_ESCAPES.get(source[i], source[i]))
                else:
                    text.append(source[i])
                i += 1
            out.append(''.join(text))
            i += 1
        else:
            i += 1
    return out


def goldens(directory: Path = GOLDENS) -> set[str]:
    r"""The SQLite goldens: every string literal of ``sdm-data``'s test
    files whose name holds ``sqlite``."""
    return {
        text
        for path in sorted(directory.glob('*sqlite*.rs'))
        for text in literals(path.read_text(encoding='utf-8'))
    }


def unpinned(
    observations: dict[str, dict[str, Any]], pinned: set[str]
) -> list[str]:
    r"""Each statement a run made that no golden pins, by key and public
    call; every one when none was made."""
    ran = [
        (key, statement['stage'], statement['sql'])
        for key, line in sorted(observations.items())
        for statement in ((line.get('driver') or {}).get('capture') or {}).get(
            'statements', []
        )
    ]
    if not ran:
        return ['no statement ran: no evidence']
    return [f'{k} {stage}: {sql}' for k, stage, sql in ran if sql not in pinned]


def test_the_golden_check_reads_rust_and_fails_closed() -> None:
    r"""Control: the literal reader reads Rust's strings, and the check
    fails on a statement no golden pins and on no statements at all."""
    source = (
        'let a = "x\\n\\"y\\""; // "not a pin"\n'
        'let b = r#"raw "q""#; let c = \'"\'; let d = "long \\\n    line";'
    )
    assert literals(source) == ['x\n"y"', 'raw "q"', 'long line']
    seen = {
        'k@sqlite': {
            'driver': {
                'capture': {
                    'statements': [
                        {'stage': 'from_source', 'sql': 'SELECT 1'},
                        {'stage': 'materialize', 'sql': 'SELECT 2'},
                    ]
                }
            }
        }
    }
    assert unpinned(seen, {'SELECT 1', 'SELECT 2'}) == []
    assert unpinned(seen, {'SELECT 1'}) == ['k@sqlite materialize: SELECT 2']
    assert unpinned({}, {'SELECT 1'}) == ['no statement ran: no evidence']


def test_every_statement_python_core_ran_on_sqlite_is_a_golden() -> None:
    r"""Exit "L4 goldens executed" (SQLite): every statement python-core ran
    on SQLite, over every case, is pinned verbatim in
    ``rust/crates/sdm-data/tests/sqlite*.rs``: each golden that matches was
    executed on SQLite, and a SQL change without a golden update fails."""

    def check() -> None:
        problems = unpinned(_cases_on_sqlite(), goldens())
        assert not problems, '\n'.join(problems[:40])

    held('test_every_statement_python_core_ran_on_sqlite_is_a_golden', check)


# --- the key-type and decimal-key pair matrices --------------------------------------


def recorded() -> dict[str, Any]:
    found: dict[str, Any] = json.loads(VECTORS_FILE.read_text('utf-8'))
    return found


def _by_id() -> dict[str, dict[str, Any]]:
    return {vector['id']: vector for vector in recorded()['vectors']}


def test_the_key_pair_vectors_are_the_reference_s_and_not_vacuous() -> None:
    r"""Control: the vectors are the recorder's inputs, recorded on SQLite
    from the reference this tree names; their keys reach SQLite in every
    pair of the forms the type map stores (INTEGER, REAL, TEXT); equal keys
    of different scales are one stored text; the wide keys are ones a
    double merges; the reference both joins and refuses, and each request
    it sent joins every row to its own entity."""
    document = recorded()
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {document["reference"]}' in env
    assert document['recorded_by'] == (
        'contract/tools/sqlite_key_pairs_record.py'
    )
    assert document['database'] == DATABASE
    by_id = _by_id()
    assert list(by_id) == [vector['id'] for vector in M.VECTORS]
    for vector in M.VECTORS:
        held_vector = by_id[vector['id']]
        assert held_vector['setup'] == vector['setup'], vector['id']
        assert held_vector['case'] == M.case_for(vector), vector['id']
    # Every pair of the stored forms, and every type of lesson 19's list as
    # primary and as foreign key against every other.
    forms = {
        (
            v['sqlite']['users.user_id']['declared'],
            v['sqlite']['orders.user_id']['declared'],
        )
        for v in by_id.values()
    }
    assert forms == {
        (p, f)
        for p in ('INTEGER', 'REAL', 'TEXT')
        for f in ('INTEGER', 'REAL', 'TEXT')
    }
    assert {i for i in by_id if i.startswith('pair.')} == {
        f'pair.{p}.{f}' for p in M.TYPES for f in M.TYPES
    }
    # Equal keys across scales, both ways, are one canonical text.
    for vector_id in (
        'decimal.key.pair.d10-2.d12-4',
        'decimal.key.pair.d12-4.d10-2',
        'decimal.key.pair-fraction.d10-2.d12-4',
        'decimal.key.pair-fraction.d12-4.d10-2',
    ):
        holds = by_id[vector_id]['sqlite']
        assert holds['users.user_id'] == holds['orders.user_id'], vector_id
    assert by_id['decimal.key.pair-fraction.d10-2.d12-4']['sqlite'][
        'orders.user_id'
    ]['values'][:2] == ["'0.25'", "'1.25'"]
    # The wide keys are distinct as stored, and a double merges them.
    for vector_id in (
        'wide.bigint.bigint',
        'decimal.key.pair-merging.d38-0.d38-2',
        'wide.ubigint.ubigint',
    ):
        values = by_id[vector_id]['sqlite']['users.user_id']['values']
        exact = [int(v.strip("'")) for v in values]
        assert len(set(exact)) == len(exact) == 8, vector_id
        assert len({float(v) for v in exact}) < len(exact), vector_id
    # The reference answers both ways, and every request it sent joins
    # right; the marks are derived from its answers.
    outcomes = [v['reference']['outcome']['result'] for v in by_id.values()]
    assert outcomes.count('success') >= 40
    assert outcomes.count('refusal') >= 30
    for vector_id, vector in by_id.items():
        success = vector['reference']['outcome']['result'] == 'success'
        assert vector['joined'] is (True if success else None), vector_id
        assert vector['register']['required'] == bool(
            M.register(vector['reference'], vector['joined'])
        )
    assert sorted(i for i, v in by_id.items() if v['register']['required']) == [
        'wide.ubigint.hugeint',
        'wide.ubigint.ubigint',
    ]


def test_the_join_check_fails_on_a_wrong_join() -> None:
    r"""Control: :func:`joins` passes the reference's own request on SQLite
    (``base.regression@sqlite``: the shop, its question and entities, as
    ``pair.bigint.bigint`` asks) and fails it with one order moved to
    another entity, and with one entity's age taken from another."""
    vector = _by_id()['pair.bigint.bigint']
    observation = reference_lines()['base.regression@sqlite']
    [body] = [p['parsed'] for p in observation['public']['payloads']]
    assert M.joins(body, M.offset(vector)) == []
    orders = body['predict']['related_tables']['orders']
    at = orders['columns'].index('user_id')
    moved = json.loads(json.dumps(body))
    moved['predict']['related_tables']['orders']['rows'][0][at] = {
        'number': '5'
    }
    assert M.joins(moved, M.offset(vector)) != []
    users = body['context']['related_tables']['users']
    at = users['columns'].index('age')
    aged = json.loads(json.dumps(body))
    aged['context']['related_tables']['users']['rows'][0][at] = {'number': '99'}
    assert M.joins(aged, M.offset(vector)) != []


@functools.cache
def _core_runs() -> dict[str, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix='t22-pairs-') as tmp:
        result: dict[str, dict[str, Any]] = M.run(
            'python-core', M.VECTORS, Path(tmp), routes=E.core_routes
        )
        return result


def _named_refusal(outcome: dict[str, Any]) -> bool:
    return outcome['result'] == 'refusal' and outcome['code'] not in (
        None,
        'INTERNAL_ERROR',
    )


def test_each_key_pair_on_sqlite_is_the_reference_s_on_python_core() -> None:
    r"""Exits "the key-type pair matrix on SQLite" and "the decimal-key pair
    matrix, equal keys across scales included": on SQLite every vector's
    answer on python-core is the reference's (``decimal_record.py``'s
    projection: outcome, schema, warnings, each request's digest as the
    ``payload`` rule compares it and its tables, the result, and the key
    columns as they reached the model), and every request joins each row
    to its own entity and asks about the entities given. Where the
    reference fails unnamed or with INTERNAL_ERROR (``register.required``),
    the core answers with a right join or refuses by name; any difference
    from the reference needs an approved register entry whose ``refs``
    name the vector."""

    def check() -> None:
        observed = _core_runs()
        problems = []
        for vector in recorded()['vectors']:
            vector_id = vector['id']
            observation = observed[vector_id]
            got = R.project(observation, vector)
            right = M.joined(observation, vector)
            if got['outcome']['result'] == 'success' and right is not True:
                problems.append(f'{vector_id}: a request joins a wrong row')
            if vector['register']['required']:
                if not (_named_refusal(got['outcome']) or right is True):
                    problems.append(f'{vector_id}: neither named nor right')
                if got != vector['reference'] and not registered_vector(
                    vector_id
                ):
                    problems.append(f'{vector_id}: differs, not registered')
            elif got != vector['reference'] and not registered_vector(
                vector_id
            ):
                differing = sorted(
                    k for k in got if got[k] != vector['reference'][k]
                )
                problems.append(f'{vector_id}: differs in {differing}')
        assert not problems, '\n'.join(problems)

    held(
        'test_each_key_pair_on_sqlite_is_the_reference_s_on_python_core', check
    )
