# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32b's exits for the decimal family on ``python-core``.

    python -m pytest contract/tests/test_t132b_decimal_family.py -q

Written by the package's independent test author from the brief, the
design's T1.32b row ("The ``decimal.*`` cases, as T1.32a", whose exits
apply to this family) and the reference, before the implementation. The
family is the ``decimal.*`` cases of ``contract/cases/s1``, T1.32b's three
and T2.0c's eleven exact-decimal ones (KO2-50); T1.32b's three:
``decimal.class-target-38-18`` (a static DECIMAL(38, 18) target, which the
reference fails INTERNAL_ERROR at ``materialize`` and the core refuses by
name, ``target_type``, register P1B-18), ``decimal.feature-38-0-wide`` and
``decimal.target-38-18``; each is run by
``contract/runners/run_python.py --mode python-core`` on the core route
(its own ``routes`` once admitted; until then every DuckDB route of
``Schema.from_source``, ``materialize`` and ``predict``, forced on a copy).

- Admission ("the cases admitted"): each case is an acceptance case with
  routes, carrying to the core every operation up to the stage the
  reference answers at (``describe_source`` for ``from_source``, then
  ``materialize``, then ``predict``), the entities the public call is
  handed.
- "Each with its reference observation" and the cases' expectations: each
  case's ``expect`` is what ``python-ref`` observed
  (``contract/reference/<commit>.jsonl``); and ``python-core``'s public
  answer is ``python-ref``'s (the outcome's result, stage, class and code;
  the public schema; the warning classes; each request as the ``payload``
  rule compares it; the result; and the case's decimal columns, each its
  public dtype and stype and how it reached the model), as
  ``contract/tools/decimal_record.py``'s ``project`` reads both, or an
  approved register entry other than the inventory's placeholder P1B-1
  lists the case in its ``cases`` (a proposed one excuses nothing: the
  static decimal target stays pending until P1B-18 is approved, as a
  success<->refusal change needs the owner, E66(b), E73(b)).
  ``sdm-testkit``'s ``tests/t132b_decimal_sql.rs`` holds ``rust-host`` to
  the same cases, and ``test_t132b_decimal_vectors.py`` both hosts over
  the family's vectors.
- "Statement counts equal the reference's or are registered,
  cross-checked against DuckDB's query log": per public call, as
  ``compare.py`` counts them, with the capture complete against the log;
  a difference passes where an approved entry lists the case.
- "Capture on and off give the same public projection on the Python
  runners": ``capture_check.py`` over the family's cases and the vectors
  (``contract/decimal-vectors.json``, as cases) run with and without the
  capture, every one required; and the check fails on this family's own
  evidence when one public projection is seeded to differ, or one case is
  missing.

Gating (``contract/tests/t132b_gate.py``): a test whose id is in
``t132b-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``) failed
when written. It still runs and must still fail; once it passes it fails,
naming itself, until it is taken out of the list, which may only shrink.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CASES = sorted((ROOT / 'contract/cases/s1').glob('decimal.*.json'))
IDS = [path.stem for path in CASES]
#: The register's placeholder for the inventory's cases.
PLACEHOLDER = 'P1B-1'
#: The umbrellas, the placeholder among them: approved by the owner as E92's
#: Phase 2 class, they excuse no case, even one they list (T1.33).
UMBRELLAS = ('P1B-1', 'P1B-2', 'P1B-3', 'P1B-4')
#: The core operations that answer up to each stage of a public call.
OPERATIONS = {
    'from_source': ('describe_source',),
    'materialize': ('describe_source', 'materialize'),
    'predict': ('describe_source', 'materialize', 'predict'),
}


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


G = _module('t132b_gate', ROOT / 'contract/tests/t132b_gate.py')
R = _module('decimal_record', ROOT / 'contract/tools/decimal_record.py')
#: Each case's decimal columns, projected as the vectors' are.
COLUMNS = R.CASE_COLUMNS


def gated(request: pytest.FixtureRequest, check: Any) -> None:
    r"""Runs ``check`` held to the pending list, as ``py:<test id>``."""
    G.held('exits', f'py:{request.node.name}', check)


def per_case() -> list[Any]:
    return [pytest.param(case, id=case) for case in IDS]


def case_file(case: str) -> dict[str, Any]:
    path = ROOT / f'contract/cases/s1/{case}.json'
    return json.loads(path.read_text(encoding='utf-8'))


def reference() -> dict[str, dict[str, Any]]:
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    commit = next(
        line.split(': ', 1)[1].strip()
        for line in env.splitlines()
        if line.startswith('reference: ')
    )
    path = ROOT / f'contract/reference/{commit}.jsonl'
    out = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        observation = json.loads(line)
        out[observation['case']] = observation
    return out


def registered_case(case: str) -> str | None:
    r"""The approved entry, other than the placeholder, that lists ``case``
    in its ``cases``."""
    path = ROOT / 'docs/design/rust-core/register.yaml'
    for entry in yaml.safe_load(path.read_text(encoding='utf-8'))['entries']:
        if entry['id'] in UMBRELLAS:
            continue
        if entry.get('status') == 'approved' and case in (
            entry.get('cases') or []
        ):
            return str(entry['id'])
    return None


# --- the family's runs ---------------------------------------------------------


def _routes() -> list[str]:
    from sdm_client import _routing

    return list(
        _routing.FROM_SOURCE_DUCKDB
        + _routing.MATERIALIZE_DUCKDB
        + _routing.PREDICT_DUCKDB
    )


def _forced(tmp: Path) -> Path:
    r"""Copies of the family's cases, each on its own routes, or on every
    DuckDB route of the three public calls while it names none; and the
    vectors as cases on the core's routes."""
    routes = _routes()
    out = tmp / 'cases'
    out.mkdir()
    for path in CASES:
        case = json.loads(path.read_text(encoding='utf-8'))
        case['routes'] = case['routes'] or routes
        (out / path.name).write_text(json.dumps(case), encoding='utf-8')
    for vector in R.VECTORS:
        case = R.case_for(vector, core=True)
        (out / f'{case["id"]}.json').write_text(
            json.dumps(case), encoding='utf-8'
        )
    return out


def _run(cases: Path, out: Path, *extra: str) -> list[dict[str, Any]]:
    done = subprocess.run(
        [
            sys.executable,
            '-I',
            str(ROOT / 'contract/runners/run_python.py'),
            '--mode',
            'python-core',
            *extra,
            '--out',
            str(out),
            str(cases),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return [
        json.loads(line)
        for line in out.read_text(encoding='utf-8').splitlines()
    ]


@pytest.fixture(scope='module')
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp('t132b')
    cases = _forced(tmp)
    on = _run(cases, tmp / 'on.jsonl')
    off = _run(cases, tmp / 'off.jsonl', '--no-capture')
    return {
        'on': {o['case']: o for o in on},
        'off': {o['case']: o for o in off},
        'on_path': tmp / 'on.jsonl',
        'off_path': tmp / 'off.jsonl',
        'tmp': tmp,
    }


# --- admission and the cases' expectations ------------------------------------


def test_the_family_is_its_fourteen_cases() -> None:
    # T1.32b's three and T2.0c's eleven exact-decimal cases (KO2-50).
    assert sorted(IDS) == [
        'decimal.class-target-38-18',
        'decimal.compare-finer-eq',
        'decimal.compare-finer-gt',
        'decimal.compare-finer-negative',
        'decimal.compare-past-precision',
        'decimal.compare-past-precision-gt',
        'decimal.compare-same-scale',
        'decimal.feature-38-0-wide',
        'decimal.feature-inexact-38-18',
        'decimal.key-wide',
        'decimal.sum-10-2',
        'decimal.sum-past-38',
        'decimal.target-38-18',
        'decimal.target-regression-collide',
    ]
    assert sorted(COLUMNS) == sorted(IDS)
    # The vectors' file holds the reference's observation of each case, as
    # the recorder projects it (rust-host's reading is held to it).
    recorded = json.loads(
        (ROOT / 'contract/decimal-vectors.json').read_text(encoding='utf-8')
    )
    assert recorded['cases'] == {
        case: R.project(reference()[case], {'columns': COLUMNS[case]})
        for case in IDS
    }


@pytest.mark.parametrize('case', per_case())
def test_each_case_is_admitted(
    case: str, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        loaded = case_file(case)
        assert loaded['status'] == 'acceptance', loaded['status']
        assert loaded['routes'], 'an admitted case names its routes'
        core = loaded.get('core') or {}
        stage = reference()[case]['outcome']['stage']
        for operation in OPERATIONS[stage]:
            assert core.get(operation), (
                f'the case carries {operation} to the core (the reference '
                f'answers at {stage})'
            )
        # The entities the core is handed are the public call's.
        assert len(core['materialize']['indices']) == len(
            loaded['input']['materialize']['indices']
        )

    gated(request, check)


@pytest.mark.parametrize('case', per_case())
def test_each_case_expects_what_python_ref_observed(
    case: str, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        expect = case_file(case)['expect']
        observed = reference()[case]
        assert observed['identity']['runner'] == 'python-ref'
        outcome = observed['outcome']
        assert expect['outcome'] == outcome['result']
        assert expect['stage'] == outcome['stage']
        if outcome['result'] == 'refusal':
            assert expect['error'] == outcome['error']
            assert expect['code'] == outcome['code']

    gated(request, check)


def answer_problem(
    case: str, ours: dict[str, Any], theirs: dict[str, Any]
) -> str | None:
    r"""Why ``python-core``'s observation of ``case`` (``ours``) fails
    ``python-ref``'s (``theirs``), both as the recorder projects them; None
    when they agree or an approved entry lists the case."""
    columns = {'columns': COLUMNS[case]}
    mine, want = R.project(ours, columns), R.project(theirs, columns)
    differs = sorted(k for k in want if mine.get(k) != want[k])
    if not differs or registered_case(case):
        return None
    return (
        f'{case}: {differs} differ from python-ref: {mine["outcome"]} '
        f'against {want["outcome"]}'
    )


@pytest.mark.parametrize('case', per_case())
def test_python_core_answers_each_case_as_python_ref(
    case: str, runs: dict[str, Any], request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        problem = answer_problem(case, runs['on'][case], reference()[case])
        assert problem is None, problem

    gated(request, check)


def test_the_answer_comparison_fails_closed() -> None:
    # On this family's own evidence: the reference's observation of a
    # success agrees with itself, and fails once any part of what the
    # public calls gave is changed: the outcome, the schema, a warning, the
    # request (the wide decimal's one cell), the result, and the decimal's
    # wire dtype.
    case = 'decimal.feature-38-0-wide'
    theirs = reference()[case]
    assert theirs['outcome']['result'] == 'success'
    assert answer_problem(case, theirs, theirs) is None

    def changed(edit: Any) -> dict[str, Any]:
        copy = json.loads(json.dumps(theirs))
        edit(copy)
        return copy

    def users(o: dict[str, Any]) -> dict[str, Any]:
        parsed = o['public']['payloads'][0]['parsed']
        return dict(parsed['context']['related_tables']['users'])

    def a_cell(o: dict[str, Any]) -> None:
        parsed = o['public']['payloads'][0]['parsed']
        table = parsed['context']['related_tables']['users']
        at = table['columns'].index('f')
        table['rows'][0][at] = '1.2345678901234568e+28'

    def wire(o: dict[str, Any]) -> None:
        parsed = o['public']['payloads'][0]['parsed']
        columns = parsed['schema']['related_tables']['users']['columns']
        columns['f']['dtype'] = 'float64'

    assert 'f' in users(theirs)['columns']
    edits = {
        'outcome': lambda o: o['outcome'].update(
            result='refusal', code='INVALID_REQUEST'
        ),
        'schema': lambda o: o['public']['schema']['tables'][1].update(
            primary_key=None
        ),
        'warnings': lambda o: o['host']['warnings'].append('SDMWarning'),
        'payload': a_cell,
        'result': lambda o: o['public']['result'].update(rows=0),
        'columns': wire,
    }
    for part, edit in edits.items():
        problem = answer_problem(case, changed(edit), theirs)
        assert problem is not None and part in problem, part
    # The placeholder lists cases but excuses none; an unlisted case has no
    # entry.
    assert registered_case('decimal.feature-38-0-wide') is None
    assert registered_case('decimal.target-38-18') is None


# --- statement counts and the capture -----------------------------------------


@pytest.mark.parametrize('case', per_case())
def test_statement_counts_equal_the_reference_s_or_are_registered(
    case: str, runs: dict[str, Any], request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        compare = _module('compare', ROOT / 'scripts/rust_core/compare.py')
        capture = _module(
            'capture_check', ROOT / 'contract/runners/capture_check.py'
        )
        ours = runs['on'][case]
        assert capture.complete(ours) == [], 'the capture is DuckDB query log'
        mine = compare.counts(ours)
        theirs = compare.counts(reference()[case])
        assert mine is not None and theirs is not None
        assert mine == theirs or registered_case(case), (mine, theirs)

    gated(request, check)


def _capture_check(on: Path, off: Path, required: list[str]) -> Any:
    return subprocess.run(
        [
            sys.executable,
            '-I',
            str(ROOT / 'contract/runners/capture_check.py'),
            '--on',
            str(on),
            '--off',
            str(off),
            *(arg for case in required for arg in ('--require', case)),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def _required() -> list[str]:
    return IDS + [R.CASE_PREFIX + vector['id'] for vector in R.VECTORS]


def test_capture_on_and_off_give_the_same_public_projection(
    runs: dict[str, Any],
) -> None:
    assert set(runs['on']) == set(_required())
    done = _capture_check(runs['on_path'], runs['off_path'], _required())
    assert done.returncode == 0, done.stdout + done.stderr
    assert 'complete and conserving' in done.stdout


@pytest.mark.parametrize(
    'seeded_case',
    [
        'decimal.feature-38-0-wide',
        'decimal.class-target-38-18',
        R.CASE_PREFIX + 'overflow.d4-0.sum-past',
        R.CASE_PREFIX + 'feature.orders.d38-38.max',
    ],
)
def test_the_capture_check_fails_on_a_seeded_difference_in_the_family(
    seeded_case: str, runs: dict[str, Any]
) -> None:
    # Not vacuous on this family's evidence: one case's public outcome
    # changed in the capture-off run fails the check.
    seeded = runs['tmp'] / f'seeded-off-{seeded_case}.jsonl'
    lines = []
    for line in runs['off_path'].read_text(encoding='utf-8').splitlines():
        observation = json.loads(line)
        if observation['case'] == seeded_case:
            observation['outcome'] = dict(
                observation['outcome'], result='refusal', code='SEEDED'
            )
        lines.append(json.dumps(observation))
    seeded.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    done = _capture_check(runs['on_path'], seeded, _required())
    assert done.returncode != 0, done.stdout
    assert seeded_case in done.stdout + done.stderr


def test_the_capture_check_fails_when_a_required_case_is_missing(
    runs: dict[str, Any],
) -> None:
    # Fail closed on missing evidence: a case of the family dropped from
    # the capture-off run fails the check, by name.
    dropped = 'decimal.target-38-18'
    thinned = runs['tmp'] / 'thinned-off.jsonl'
    kept = [
        line
        for line in runs['off_path'].read_text(encoding='utf-8').splitlines()
        if json.loads(line)['case'] != dropped
    ]
    thinned.write_text('\n'.join(kept) + '\n', encoding='utf-8')
    done = _capture_check(runs['on_path'], thinned, _required())
    assert done.returncode != 0, done.stdout
    assert dropped in done.stdout + done.stderr
