# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32g part 2a, fix: the old client's crash as python-old really
records it.

Written by the package's independent test author, after the subagent's
review of part 2a (P1.1, P1.2, P3.3, P3.5). ``tests/
test_t132g_reference_internal_error.py`` made its python-old observations
by relabelling the reference's lines, so it never met what the runner
writes: ``run_python.py`` records the warnings' origins (``host.origins``)
for python-old and never for python-ref, and the floor cell's pandas adds
``DeprecationWarning``s. ``tests/golden/t132g-python-old-ref-pandas3.jsonl``
and ``t132g-python-old-floor-pandas2.2.jsonl`` are python-old's own lines
for the seven cases, run at f504a9e9 in the ref-pandas3 and
floor-pandas2.2 cells (the reviewer's runs). Each claim runs the real
``ratchet.check`` and ``ratchet.accept`` on
``tests/test_contract_ratchet.py``'s temporary repository:

- P1.1: python-old's lines as run, in either cell, hold the seven at the
  reference's ``INTERNAL_ERROR``, and ``accept`` proposes them;
- P1.2 (the orchestrator's reading of "exactly"): equal in the semantic
  projection leaving out ``host.warnings`` and ``host.origins``, runtime
  noise; no warnings, or other ones, still match; another stage, error,
  statement or public schema still fails, as a panic;
- P3.3: the reference line matched is the checkpoint's: a line the
  candidate adds cannot make a crash match, in ``check`` or ``accept``;
- P3.5: an entry whose ``case`` is no string is a clean ``register`` or
  ``read`` failure of ``ratchet.py check`` and ``ratchet.py accept``,
  never a traceback.

Gating (``contract/tests/t132g_gate.py``): a test named in
``t132g-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``) failed
when written. It still runs and must still fail; once it passes it fails,
naming itself, until it is taken out of the list.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

# Gate tooling (T3.0): CI runs it in the tooling jobs, not in every matrix job.
pytestmark = pytest.mark.tooling

ROOT = Path(__file__).resolve().parents[1]
RATCHET = ROOT / 'scripts/rust_core/ratchet.py'
AS_RUN = {
    'ref-pandas3': ROOT / 'tests/golden/t132g-python-old-ref-pandas3.jsonl',
    'floor-pandas2.2': (
        ROOT / 'tests/golden/t132g-python-old-floor-pandas2.2.jsonl'
    ),
}


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


G = _module('t132g_gate', ROOT / 'contract/tests/t132g_gate.py')
helpers = _module('ratchet_tests', ROOT / 'tests/test_contract_ratchet.py')
crash = _module(
    't132g_internal_error_tests',
    ROOT / 'tests/test_t132g_reference_internal_error.py',
)
ratchet = helpers.ratchet

origin, repo = helpers.origin, crash.repo
git, commit, land = helpers.git, helpers.commit, helpers.land
check, observe = helpers.check, helpers.observe
append_entry, outcome = helpers.append_entry, helpers.outcome
SEVEN, CASE = crash.SEVEN, crash.CASE
admit, hold, lines_of = crash.admit, crash.hold, crash.lines_of
baseline_file, panics = crash.baseline_file, crash.panics
REFERENCE_PROJECTION = {
    'outcome.result': 'refusal',
    'outcome.stage': 'materialize',
    'outcome.error': 'StructuredDataError',
    'outcome.code': 'INTERNAL_ERROR',
}

Edit = Callable[[dict[str, Any]], None]


def as_run(cell: str) -> dict[str, dict[str, Any]]:
    r"""python-old's lines as run in ``cell``, each under its fixture
    case's id (``crash.OWN`` before the corpus id it was run as)."""
    found = {}
    for case, line in lines_of(AS_RUN[cell]).items():
        line['case'] = crash.OWN + case
        found[line['case']] = line
    return found


def observed(
    r: Path, tmp: Path, cell: str = 'ref-pandas3', edit: Edit | None = None
) -> Path:
    r"""python-old's observations: the reference's, relabelled, but for
    the seven, which are python-old's own lines as run in ``cell``,
    ``CASE``'s changed by ``edit``."""

    def real(lines: dict[str, dict[str, Any]]) -> None:
        for case, line in as_run(cell).items():
            lines[case] = copy.deepcopy(line)
        if edit is not None:
            edit(lines[CASE])

    return observe(r, tmp, 'python-old', real)


def holding(r: Path, files: Path, cases: tuple[str, ...] = SEVEN) -> str:
    r"""A landing (not judged) admitting ``cases`` with python-old's
    baselines of them held from ``files``."""

    def edit(tree: Path) -> None:
        for case in cases:
            admit(tree, case)
            hold(tree, 'python-old', case, files)

    return land(r, 'T1.32f', edit)


# --- the lines as run ------------------------------------------------------------


def test_the_lines_are_python_old_s_as_run() -> None:
    reference = lines_of(crash.FIXTURES / 'python-ref.jsonl')
    for cell in AS_RUN:
        lines = as_run(cell)
        assert sorted(lines) == list(SEVEN)
        for case, line in lines.items():
            assert line['identity']['runner'] == 'python-old'
            assert line['identity']['cell'] == cell
            assert line['identity']['native_loaded'] is False
            assert line['outcome'] == reference[case]['outcome'], case
            # What relabelled reference lines never had.
            assert line['host']['origins'], (cell, case)
            assert 'origins' not in (reference[case]['host'] or {})
    floor = as_run('floor-pandas2.2')[CASE]['host']['warnings']
    assert floor.count('DeprecationWarning') >= 2


# --- P1.1 and P1.2: held and proposed as run -------------------------------------


@pytest.mark.parametrize('cell', sorted(AS_RUN))
def test_python_old_as_run_holds_the_seven(
    repo: Path, tmp_path: Path, cell: str, request: pytest.FixtureRequest
) -> None:
    files = observed(repo, tmp_path, cell)
    landed = holding(repo, files)
    commit(repo)

    def held() -> None:
        found = check(repo, landed, (files,))
        assert found == [], found

    G.held('exits', f'py:{request.node.name}', held)


@pytest.mark.parametrize('cell', sorted(AS_RUN))
def test_accept_proposes_python_old_as_run_and_the_admission_passes(
    repo: Path, tmp_path: Path, cell: str, request: pytest.FixtureRequest
) -> None:
    before = helpers.checkpoint_of(repo)
    files = observed(repo, tmp_path, cell)

    def admission(r: Path) -> None:
        for case in SEVEN:
            admit(r, case)
        append_entry(
            r,
            {
                'id': 'SEED-4',
                'kind': 'keep',
                'cases': list(SEVEN),
                'case_outcomes': [
                    outcome(
                        case,
                        'python-core',
                        'none',
                        'refusal INVALID_REQUEST at materialize (ValueError)',
                    )
                    for case in SEVEN
                ],
            },
        )

    commit(repo, 'T1.32g: the admission', admission)

    def proposed() -> None:
        found = ratchet.accept(repo, [files], before)
        written = {
            x['case']: x['projection']
            for x in found['python-old']
            if x['case'] in SEVEN
        }
        assert written == dict.fromkeys(SEVEN, REFERENCE_PROJECTION), written
        commit(
            repo,
            'T1.32g: the baselines',
            lambda r: helpers.accepted(r, before, (files,)),
        )
        problems = check(repo, before, (files,))
        assert problems == [], problems

    G.held('exits', f'py:{request.node.name}', proposed)


def _no_warnings(line: dict[str, Any]) -> None:
    line['host'] = {'warnings': [], 'origins': []}


def _other_warnings(line: dict[str, Any]) -> None:
    line['host'] = {
        'warnings': ['DeprecationWarning', 'FutureWarning', 'UserWarning'],
        'origins': [
            {'class': 'DeprecationWarning', 'module': 'pandas'},
            {'class': 'FutureWarning', 'module': 'pyarrow'},
            {'class': 'UserWarning', 'module': 'sdm_relational_engine'},
        ],
    }


NOISE: dict[str, Edit] = {
    'no warnings': _no_warnings,
    'other warnings': _other_warnings,
}


@pytest.mark.parametrize('seed', sorted(NOISE))
def test_host_warnings_and_their_origins_are_runtime_noise(
    repo: Path, tmp_path: Path, seed: str, request: pytest.FixtureRequest
) -> None:
    files = observed(repo, tmp_path, edit=NOISE[seed])
    landed = holding(repo, files)
    commit(repo)

    def held() -> None:
        found = check(repo, landed, (files,))
        assert found == [], found

    G.held('exits', f'py:{request.node.name}', held)


def _another_statement(line: dict[str, Any]) -> None:
    statements = line['driver']['capture']['statements']
    [first, *_] = [s for s in statements if s['results']]
    first['results'][0]['rows'] += 1


def _another_schema(line: dict[str, Any]) -> None:
    line['public']['schema']['tables'][0]['duplicate_keys'] += 1


NOT_THE_CRASH: dict[str, Edit] = {
    'another stage': lambda line: line['outcome'].update(stage='predict'),
    'another error': lambda line: line['outcome'].update(error='ValueError'),
    'another statement': _another_statement,
    'another public schema': _another_schema,
}


@pytest.mark.parametrize('seed', sorted(NOT_THE_CRASH))
def test_python_old_as_run_unequal_to_the_reference_is_a_panic(
    repo: Path, tmp_path: Path, seed: str
) -> None:
    files = observed(repo, tmp_path, edit=NOT_THE_CRASH[seed])
    # Held at its own projection: the only problem is the panic.
    landed = holding(repo, files, (CASE,))
    commit(repo)
    found = check(repo, landed, (files,))
    assert panics(found, CASE, 'python-old'), found


# --- P3.3: the checkpoint's reference line -----------------------------------------


SHADOW = 'contract/reference/' + 'f' * 40 + '.jsonl'


def shadowed(files: Path) -> Callable[[Path], None]:
    r"""A second reference file, after the first, holding ``CASE``'s line
    as python-old observed it in ``files`` (its public schema changed by
    :func:`_another_schema`, its warnings and their origins too), relabelled
    python-ref: the candidate's reference would then match it exactly."""

    def edit(r: Path) -> None:
        line = copy.deepcopy(lines_of(files)[CASE])
        line['identity']['runner'] = 'python-ref'
        (r / SHADOW).write_text(json.dumps(line) + '\n')

    return edit


def test_a_candidate_s_reference_line_cannot_make_a_held_crash_match(
    repo: Path, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    files = observed(repo, tmp_path, edit=_another_schema)
    landed = holding(repo, files, (CASE,))
    commit(repo, 'T1.32g: a reference line', shadowed(files))

    def refused() -> None:
        found = check(repo, landed, (files,), protected=f'{SHADOW} (new)')
        assert panics(found, CASE, 'python-old'), found

    G.held('exits', f'py:{request.node.name}', refused)


def test_a_candidate_s_reference_line_cannot_make_accept_propose_a_crash(
    repo: Path, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    before = helpers.checkpoint_of(repo)
    files = observed(repo, tmp_path, edit=_another_schema)

    def edit(r: Path) -> None:
        admit(r, CASE)
        shadowed(files)(r)

    commit(repo, 'T1.32g: the admission and a reference line', edit)

    def refused() -> None:
        found = ratchet.accept(repo, [files], before)
        assert CASE not in {x['case'] for x in found['python-old']}, found

    G.held('exits', f'py:{request.node.name}', refused)


# --- P3.5: an entry's case of another type ---------------------------------------


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RATCHET), *args],
        capture_output=True,
        text=True,
        check=False,
    )


CASE_TYPES = {'a list': [CASE], 'a mapping': {'case': CASE}}


def _odd_entry(case: Any) -> Callable[[Path], None]:
    def edit(r: Path) -> None:
        admit(r, CASE)
        append_entry(
            r,
            {
                'id': 'SEED-5',
                'cases': [CASE],
                'case_outcomes': [
                    {
                        'case': case,
                        'cell': helpers.CELL,
                        'runner': 'python-core',
                        'old': 'none',
                        'new': 'success at predict',
                        'summary': 'seeded',
                    }
                ],
            },
        )

    return edit


def _clean(done: subprocess.CompletedProcess[str]) -> list[str]:
    r"""What is unclean about a run that must fail by name."""
    said = done.stdout + done.stderr
    return [
        problem
        for problem, bad in (
            (f'exit {done.returncode}', done.returncode != 1),
            ('a traceback', 'Traceback' in said),
            (
                'no register or read failure',
                not any(word in said for word in ('register', 'read', 'not a')),
            ),
        )
        if bad
    ]


@pytest.mark.parametrize('kind', sorted(CASE_TYPES))
def test_check_refuses_an_entry_s_case_of_another_type_by_name(
    repo: Path, tmp_path: Path, kind: str
) -> None:
    before = helpers.checkpoint_of(repo)
    files = observed(repo, tmp_path)
    commit(repo, 'T1.32g: an entry', _odd_entry(CASE_TYPES[kind]))
    done = _cli(
        'check',
        '--repo',
        str(repo),
        '--checkpoint',
        before,
        '--protected-paths',
        '',
        '--observations',
        str(files),
    )
    assert _clean(done) == [], done.stdout + done.stderr
    assert 'FAILS register' in done.stdout or 'FAILS read' in done.stdout


@pytest.mark.parametrize('kind', sorted(CASE_TYPES))
def test_accept_refuses_an_entry_s_case_of_another_type_by_name(
    repo: Path, tmp_path: Path, kind: str, request: pytest.FixtureRequest
) -> None:
    before = helpers.checkpoint_of(repo)
    files = observed(repo, tmp_path)
    commit(repo, 'T1.32g: an entry', _odd_entry(CASE_TYPES[kind]))
    proposed = baseline_file(repo, 'python-old').read_text()

    def refused() -> None:
        done = _cli(
            'accept',
            '--repo',
            str(repo),
            '--checkpoint',
            before,
            '--observations',
            str(files),
        )
        assert _clean(done) == [], done.stdout + done.stderr
        # Nothing written on a refusal.
        assert baseline_file(repo, 'python-old').read_text() == proposed

    G.held('exits', f'py:{request.node.name}', refused)
