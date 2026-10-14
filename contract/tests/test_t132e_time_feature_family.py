# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32e part 2's exits for the time and feature family on ``python-core``.

    python -m pytest contract/tests/test_t132e_time_feature_family.py -q

Written by the package's independent test author from the brief, the
design's T1.32e row ("Part 2: the ``time.*`` and ``feature.*`` cases, as
T1.32a", whose exits apply to this family) and the reference, before the
implementation. The family is the twelve ``time.*`` and ``feature.*``
cases of ``contract/cases/s1``; each is run by
``contract/runners/run_python.py --mode python-core`` on the core route
(its own ``routes`` once admitted; until then every DuckDB route of
``Schema.from_source``, ``materialize`` and ``predict``, forced on a copy).

- Admission ("the cases admitted, each with its reference observation"):
  each case is an acceptance case with routes, carrying to the core the
  operations its call reaches there and no others (``REACHED``; ``compare.py``'s
  ``inputs`` rule holds a core runner to every operation a case names):
  the nine successes reach ``predict``; ``time.feature-interval`` stops at
  ``from_source`` (P1B-8, the host's admission), ``describe_source`` alone;
  and ``time.text-one-bad-value`` and ``time.text-us-date`` at
  ``materialize`` (DT-12, the source's cast, PROVIDER_ERROR),
  ``describe_source`` and ``materialize``. Admitted,
  the CI ``cross-host`` step's ``core`` rule gates ``python-core`` against
  ``rust-host`` on every ``core`` field of each (``contract/tests/
  test_cross_host.py`` holds that step).
- The cases' expectations: each case's ``expect`` is what ``python-ref``
  observed (``contract/reference/<commit>.jsonl``).
- "Both hosts agree on ``core`` fields": ``python-core``'s public answer is
  ``python-ref``'s (the outcome's result, stage, class and code; the public
  schema, each column's dtype and stype; the warning classes; each request
  the endpoint received, as the ``payload`` rule compares it; the result),
  as ``contract/tools/empty_null_record.py``'s ``project`` reads both, or an
  approved register entry other than the inventory's placeholder P1B-1
  lists the case in its ``cases`` (a proposed one excuses nothing).
  ``sdm-testkit``'s ``tests/t132e_time_feature_sql.rs`` holds ``rust-host``
  to the same answers (its description's stypes, its request's digest, its
  result's rows), so the two hosts agree through the reference on the
  description, the request and the outcome; and
  ``test_t132e_multicategorical.py`` and ``t132e_multicategorical.rs`` hold
  both to the multicategorical vectors.
- "Statement counts equal the reference's or are registered,
  cross-checked against DuckDB's query log": per public call, as
  ``compare.py`` counts them, with the capture complete against the log.
- "Capture on and off give the same public projection on the Python
  runners": ``capture_check.py`` over the family's cases and the
  multicategorical vectors (as cases) run with and without the capture,
  every one required; and the check fails on this family's own evidence
  when one public projection is seeded to differ, or a case is missing.

Gating (``contract/tests/t132e_gate.py``): a test whose id is in
``t132e-pending.json``'s ``pending["p2-exits"]`` (as ``py:<test id>``)
failed when written. It still runs and must still fail; once it passes it
fails, naming itself, until it is taken out of the list, which may only
shrink.
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
CASES = sorted(
    path
    for prefix in ('time', 'feature')
    for path in (ROOT / 'contract/cases/s1').glob(f'{prefix}.*.json')
)
IDS = [path.stem for path in CASES]
#: The register's placeholder for the inventory's cases (see above).
PLACEHOLDER = 'P1B-1'
#: The umbrellas, the placeholder among them: approved by the owner as E92's
#: Phase 2 class, they excuse no case, even one they list (T1.33).
UMBRELLAS = ('P1B-1', 'P1B-2', 'P1B-3', 'P1B-4')
#: The core operation that answers at each stage of a public call.
OPERATIONS = {
    'from_source': ('describe_source',),
    'materialize': ('describe_source', 'materialize'),
    'predict': ('describe_source', 'materialize', 'predict'),
}


#: The public call each case reaches on the core, by the register's
#: approved entries where the reference's differs (reviewed with the
#: refusal map, ``contract/refusal-maps/time.json``).
REACHED = {
    'feature.bool': 'predict',
    'time.date': 'predict',
    'time.feature-date-and-timestamp-with-null': 'predict',
    'time.feature-interval': 'from_source',  # P1B-8: the host's admission
    'time.feature-time-of-day': 'predict',
    'time.text-declared': 'predict',
    'time.text-iso-date': 'predict',
    'time.text-one-bad-value': 'materialize',  # DT-12: the source's cast
    'time.text-us-date': 'materialize',  # DT-12
    'time.timestamp-ns': 'predict',
    'time.timestamp-s': 'predict',
    'time.timestamptz': 'predict',
}


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


G = _module('t132e_gate', ROOT / 'contract/tests/t132e_gate.py')
R = _module('empty_null_record', ROOT / 'contract/tools/empty_null_record.py')
M = _module(
    'multicategorical_record',
    ROOT / 'contract/tools/multicategorical_record.py',
)


def gated(request: pytest.FixtureRequest, check: Any) -> None:
    r"""Runs ``check`` held to the pending list, as ``py:<test id>``."""
    G.held('p2-exits', f'py:{request.node.name}', check)


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
    multicategorical vectors as cases on the core's routes."""
    routes = _routes()
    out = tmp / 'cases'
    out.mkdir()
    for path in CASES:
        case = json.loads(path.read_text(encoding='utf-8'))
        case['routes'] = case['routes'] or routes
        (out / path.name).write_text(json.dumps(case), encoding='utf-8')
    for vector in M.VECTORS:
        case = M.case_for(vector, core=True)
        (out / f'{case["id"]}.json').write_text(
            json.dumps(case), encoding='utf-8'
        )
    return out


def _execute(cases: Path, out: Path, *extra: str) -> None:
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


def _read(out: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in out.read_text(encoding='utf-8').splitlines()
    ]


@pytest.fixture(scope='module')
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp('t132e2')
    cases = _forced(tmp)
    # The capture-off run is read from its file by the capture check alone,
    # never parsed here, and runs before the captured run is parsed: on
    # Linux a run parsed holds about 3.5 GB and the runner making one about
    # as much, past CI's runner when they overlap.
    _execute(cases, tmp / 'on.jsonl')
    _execute(cases, tmp / 'off.jsonl', '--no-capture')
    return {
        'on': {o['case']: o for o in _read(tmp / 'on.jsonl')},
        'on_path': tmp / 'on.jsonl',
        'off_path': tmp / 'off.jsonl',
        'tmp': tmp,
    }


# --- admission and the cases' expectations ------------------------------------


def test_the_family_is_the_twelve_cases() -> None:
    assert IDS == [
        'feature.bool',
        'time.date',
        'time.feature-date-and-timestamp-with-null',
        'time.feature-interval',
        'time.feature-time-of-day',
        'time.text-declared',
        'time.text-iso-date',
        'time.text-one-bad-value',
        'time.text-us-date',
        'time.timestamp-ns',
        'time.timestamp-s',
        'time.timestamptz',
    ]


@pytest.mark.parametrize('case', per_case())
def test_each_case_is_admitted(
    case: str, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        loaded = case_file(case)
        assert loaded['status'] == 'acceptance', loaded['status']
        assert loaded['routes'], 'an admitted case names its routes'
        core = loaded.get('core') or {}
        assert sorted(REACHED) == IDS, 'each case has the call it reaches'
        wanted = OPERATIONS[REACHED[case]]
        carried = tuple(op for op in OPERATIONS['predict'] if op in core)
        assert carried == wanted, (
            f'the case carries {carried} to the core; its call reaches '
            f'{REACHED[case]} there, so {wanted}'
        )
        assert set(core) == set(wanted), sorted(core)
        for operation in wanted:
            assert core[operation], f'{operation} carries no inputs'
        # Never short of the reference's stage.
        stage = reference()[case]['outcome']['stage']
        assert set(OPERATIONS[stage]) <= set(wanted), stage
        if 'materialize' in core:
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
    mine, want = R.project(ours, {}), R.project(theirs, {})
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
    # request (one cell) and the result.
    case = 'time.date'
    theirs = reference()[case]
    assert theirs['outcome']['result'] == 'success'
    assert answer_problem(case, theirs, theirs) is None

    def changed(edit: Any) -> dict[str, Any]:
        copy = json.loads(json.dumps(theirs))
        edit(copy)
        return copy

    def a_cell(o: dict[str, Any]) -> None:
        rows = o['public']['payloads'][0]['parsed']['context']['instance_table']
        rows['rows'][0][-1] = None

    def a_stype(o: dict[str, Any]) -> None:
        o['public']['schema']['tables'][0]['columns'][-1]['stype'] = 'text'

    edits = {
        'outcome': lambda o: o['outcome'].update(
            result='refusal', code='INVALID_REQUEST'
        ),
        'schema': a_stype,
        'warnings': lambda o: o['host']['warnings'].append('SDMWarning'),
        'payload': a_cell,
        'result': lambda o: o['public']['result'].update(rows=0),
    }
    for part, edit in edits.items():
        problem = answer_problem(case, changed(edit), theirs)
        assert problem is not None and part in problem, part
    # The placeholder lists cases but excuses none; an unlisted case has no
    # entry; DT-12, approved, lists the text dates the source cannot cast.
    assert registered_case('time.date') is None
    assert registered_case('time.text-us-date') == 'DT-12'


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
    return IDS + [M.CASE_PREFIX + vector['id'] for vector in M.VECTORS]


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
        'time.date',
        'time.feature-interval',
        M.CASE_PREFIX + 'sep.semicolon',
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
    dropped = 'time.timestamptz'
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
