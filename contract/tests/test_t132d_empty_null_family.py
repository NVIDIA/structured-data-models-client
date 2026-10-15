# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32d's exits for the empty, null and refuse family on ``python-core``.

    python -m pytest contract/tests/test_t132d_empty_null_family.py -q

Written by the package's independent test author from the brief, the
design's T1.32d row ("The ``empty.*``, ``null.*`` and ``refuse.*`` cases,
as T1.32a", whose exits apply to this family) and the reference, before the
implementation. The family is the eight ``empty.*``, ``null.*`` and
``refuse.*`` cases of ``contract/cases/s1`` (``empty.fact-table`` among
them, whose ``empty_table`` refusal T1.32d owns: register P1B-11; and
``empty.indices``, whose core input the case format cannot carry yet, a
gap ``contract/refusal-maps/empty-null.json`` gives T1.32d); each is run by
``contract/runners/run_python.py --mode python-core`` on the core route
(its own ``routes`` once admitted; until then every DuckDB route of
``Schema.from_source``, ``materialize`` and ``predict``, forced on a copy).

- Admission ("the cases admitted"): each case is an acceptance case with
  routes, carrying to the core every operation up to the stage the
  reference answers at (``describe_source`` for ``from_source``, then
  ``materialize``, then ``predict``): ``empty.indices``'s empty entity list
  included, so ``rust-host`` reaches the core's refusal too; a query the
  reference refused before reading any data (``refuse.unknown-column``:
  INVALID_REQUEST at ``materialize``, no statement of ``materialize`` in
  its observation) carries ``validate_query`` in place of ``materialize``,
  as python-core validates the query first.
- "Each with its reference observation" and the cases' expectations: each
  case's ``expect`` is what ``python-ref`` observed
  (``contract/reference/<commit>.jsonl``); and ``python-core``'s public
  answer is ``python-ref``'s (the outcome's result, stage, class and code;
  the public schema; the warning classes; each request the endpoint
  received, as the ``payload`` rule compares it; the result), as
  ``contract/tools/empty_null_record.py``'s ``project`` reads both, or an
  approved register entry other than the inventory's placeholder P1B-1
  lists the case in its ``cases`` (a proposed one excuses nothing, so its
  case stays pending until the owner approves it). The core's refusal of
  the empty table passes because P1B-11, which the owner approved, lists
  ``empty.fact-table``. Its code is pinned by the owner's switch of
  2026-10-13 (T1.32g): ``INVALID_REQUEST`` (class ``StructuredDataError``)
  at ``from_source``, in place of ``UNSUPPORTED_REQUEST``: the code alone
  changes, and with it the class the host raises it as. ``sdm-testkit``'s
  ``tests/t132d_empty_null_sql.rs`` holds ``rust-host`` to the same cases,
  and ``test_t132d_empty_null_vectors.py`` both hosts over the family's
  vectors.
- "Statement counts equal the reference's or are registered,
  cross-checked against DuckDB's query log": per public call, as
  ``compare.py`` counts them, with the capture complete against the log;
  a difference passes where an approved entry lists the case.
- "Capture on and off give the same public projection on the Python
  runners": ``capture_check.py`` over the family's cases and the vectors
  (``contract/empty-null-vectors.json``, as cases) run with and without
  the capture, every one required; and the check fails on this family's
  own evidence when one public projection is seeded to differ.

Gating (``contract/tests/t132d_gate.py``): a test whose id is in
``t132d-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``) failed
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
CASES = sorted(
    path
    for prefix in ('empty', 'null', 'refuse')
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
#: The cases the reference refuses at ``materialize`` by validating the
#: query, before reading any data: its observation holds no statement of
#: ``materialize``. Such a case may carry ``validate_query`` to the core in
#: place of ``materialize``, as python-core validates the query before the
#: core's ``materialize`` would run (``contract/refusal-maps/refusal.json``,
#: ``materialize/unknown_column``).
QUERY_REFUSALS = {'refuse.unknown-column'}


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


G = _module('t132d_gate', ROOT / 'contract/tests/t132d_gate.py')
R = _module('empty_null_record', ROOT / 'contract/tools/empty_null_record.py')


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
    tmp = tmp_path_factory.mktemp('t132d')
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


def test_the_family_is_the_eight_cases() -> None:
    assert IDS == [
        'empty.entity-without-facts',
        'empty.fact-table',
        'empty.indices',
        'null.feature-all-null',
        'null.some-foreign-keys',
        'null.some-times',
        'null.target-all-null',
        'refuse.unknown-column',
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
        observed = reference()[case]
        stage = observed['outcome']['stage']
        operations = OPERATIONS[stage]
        if case in QUERY_REFUSALS:
            # Only where the reference refused the query before reading:
            # INVALID_REQUEST at materialize, no statement of materialize.
            assert observed['outcome']['code'] == 'INVALID_REQUEST'
            assert stage == 'materialize'
            assert not [
                s
                for s in observed['driver']['capture']['statements']
                if s['stage'] == 'materialize'
            ]
            # validate_query in place of materialize (either is the
            # query's refusal in the core).
            if 'materialize' not in core:
                operations = ('describe_source', 'validate_query')
        for operation in operations:
            assert core.get(operation), (
                f'the case carries {operation} to the core (the reference '
                f'answers at {stage})'
            )
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


def test_the_empty_fact_table_is_refused_invalid_request(
    runs: dict[str, Any], request: pytest.FixtureRequest
) -> None:
    # P1B-11's switch (owner, 2026-10-13): the code alone changes, from
    # UNSUPPORTED_REQUEST; the stage stays from_source.
    def check() -> None:
        assert runs['on']['empty.fact-table']['outcome'] == {
            'result': 'refusal',
            'stage': 'from_source',
            'error': 'StructuredDataError',
            'code': 'INVALID_REQUEST',
        }

    gated(request, check)


def test_the_answer_comparison_fails_closed() -> None:
    # On this family's own evidence: the reference's observation of a
    # success agrees with itself, and fails once any part of what the
    # public calls gave is changed: the outcome, the schema, a warning, the
    # request (one cell) and the result.
    case = 'null.some-times'
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

    edits = {
        'outcome': lambda o: o['outcome'].update(
            result='refusal', code='INVALID_REQUEST'
        ),
        'schema': lambda o: o['public']['schema']['tables'][0].update(
            time_column=None
        ),
        'warnings': lambda o: o['host']['warnings'].append('SDMWarning'),
        'payload': a_cell,
        'result': lambda o: o['public']['result'].update(rows=0),
    }
    for part, edit in edits.items():
        problem = answer_problem(case, changed(edit), theirs)
        assert problem is not None and part in problem, part
    # The placeholder lists cases but excuses none; an unlisted case has no
    # entry; P1B-11, approved, lists the empty fact table.
    assert registered_case('null.some-times') is None
    assert registered_case('empty.fact-table') == 'P1B-11'


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
        'null.some-foreign-keys',
        'empty.fact-table',
        R.CASE_PREFIX + 'null.time-in-the-window',
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
    dropped = 'null.some-times'
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
