# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32g part 2a: a first admission is held to the reference as the
approved entries change it.

Written by the package's independent test author, before the change. A
case admitted for the first time has no baseline at the checkpoint, so
``ratchet.py``'s ``effective`` (built from held baselines only) gives it no
expectation, ``compare.py`` holds every runner to the reference's outcome,
and ``accept`` writes a case only when every runner given meets the
reference. A case where the core answers otherwise under an approved
register entry can then never be admitted: empty.fact-table (P1B-11),
nested.fixed-array (P1B-9), nested.map and nested.struct (P1B-10),
nested.union (P1B-7) and refuse.unknown-column among the bundle's.

The rule: on a first admission, a runner's effective expectation is the
reference's outcome, changed by an approved entry whose ``case_outcomes``
names that case, cell and runner (``old: none``, the convention P1B-72
uses for a case no runner held), as a baselined case's already is; never
to ``INTERNAL_ERROR``. Each claim runs the real ``ratchet.check`` (and its
``--effective-out``), ``ratchet.accept`` and ``compare.compare`` on
``tests/test_contract_ratchet.py``'s temporary repository:

- passes: python-core and rust-host differ from the reference exactly as
  an approved entry says: the effective expectations say the entry's
  outcome, ``accept`` writes every runner's baseline, python-old's at the
  reference's own ``INTERNAL_ERROR`` included, and ``check`` passes; the
  comparator, handed those expectations, passes the outcome rule;
- still fails: the same difference under a proposed entry; a difference
  the entry does not name (another runner); a runner whose observation is
  not the entry's new outcome; an entry whose new outcome is
  ``INTERNAL_ERROR``;
- ``accept`` writes per runner: each runner meeting its own effective
  expectation gets its baseline, and one that does not gets none.

The comparator's seeds are python-old's (its observations are the
reference's, relabelled, and valid); a relabelled python-core or rust-host
line is not a valid observation, so for those runners the claim is on the
expectations ``compare.py --expect`` reads. The rule is runner-blind there
but for rust-host's unobserved fields.

Gating (``contract/tests/t132g_gate.py``): a test named in
``t132g-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``) failed
when written. It still runs and must still fail; once it passes it fails,
naming itself, until it is taken out of the list.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

# Gate tooling (T3.0): CI runs it in the tooling jobs, not in every matrix job.
pytestmark = pytest.mark.gate_tooling

ROOT = Path(__file__).resolve().parents[1]


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
compare = _module('compare_for_ratchet', ROOT / 'scripts/rust_core/compare.py')
run_python = _module('run_python', ROOT / 'contract/runners/run_python.py')
ratchet = helpers.ratchet

crash = _module(
    't132g_internal_error_tests',
    ROOT / 'tests/test_t132g_reference_internal_error.py',
)
#: The checkpoint holds the tests' own fixture cases (their own ids, as
#: diagnostic as they were at bc4b2d0c): no admission in the corpus
#: changes what is judged here.
origin, repo = helpers.origin, crash.repo
commit, check, observe = helpers.commit, helpers.check, helpers.observe
append_entry, outcome = helpers.append_entry, helpers.outcome
CELL = helpers.CELL

#: A diagnostic case the reference answers with its own INTERNAL_ERROR at
#: materialize, held by no runner.
CASE = crash.CASE
RUNNERS = ('python-core', 'python-old', 'rust-host')
#: The core's answer the seeded entry approves, as each runner records it.
CORE_NEW = 'refusal INVALID_REQUEST at materialize (ValueError)'
HOST_NEW = 'refusal INVALID_REQUEST'
CORE_OUTCOME = {
    'result': 'refusal',
    'stage': 'materialize',
    'error': 'ValueError',
    'code': 'INVALID_REQUEST',
}


def case_path(r: Path) -> Path:
    [path] = (r / 'contract/cases').glob(f'*/{CASE}.json')
    return path


def admit(r: Path) -> None:
    path = case_path(r)
    spec = json.loads(path.read_text())
    spec['status'] = 'acceptance'
    path.write_text(json.dumps(spec, indent=1))


def items(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [outcome(CASE, runner, 'none', new) for runner, new in pairs]


APPROVED = items(('python-core', CORE_NEW), ('rust-host', HOST_NEW))


def admission(
    case_outcomes: list[dict[str, str]], **entry: Any
) -> Callable[[Path], None]:
    r"""``CASE`` admitted under an entry with ``case_outcomes`` (approved by
    the reviewer unless ``entry`` says otherwise)."""

    def edit(r: Path) -> None:
        admit(r)
        append_entry(
            r,
            {
                'id': 'SEED-3',
                'kind': 'change',
                'cases': [CASE],
                'case_outcomes': case_outcomes,
                **entry,
            },
        )

    return edit


def core_answer(runner: str, **fields: Any) -> Callable[[Any], None]:
    r"""``runner``'s observation of ``CASE``: the core's refusal (rust-host
    observes no stage or error), changed by ``fields``."""

    def change(lines: dict[str, dict[str, Any]]) -> None:
        line = lines[CASE]['outcome']
        line.update(code='INVALID_REQUEST', error='ValueError')
        if runner == 'rust-host':
            line.update(stage=None, error=None)
        line.update(fields)

    return change


def observed(r: Path, tmp: Path, **changes: dict[str, Any]) -> dict[str, Path]:
    r"""python-core and rust-host answer as the core (each changed by
    ``changes[runner]``); python-old as the reference."""
    return {
        runner: observe(
            r,
            tmp,
            runner,
            None
            if runner == 'python-old'
            else core_answer(runner, **changes.get(runner, {})),
        )
        for runner in RUNNERS
    }


def mine(problems: list[str]) -> list[str]:
    r"""The problems naming the fixture case. The observations of every
    other case are the reference's, relabelled (``tests/
    test_contract_ratchet.py``'s ``observe``), which an admitted corpus
    case whose core answers otherwise does not meet; that is the corpus's,
    not what these tests judge."""
    return [p for p in problems if f' {CASE}:' in p or f'{CASE} ' in p]


def expectations(r: Path, checkpoint: str, tmp: Path) -> dict[str, Any]:
    r"""What ``check --effective-out`` writes for ``CASE``, by runner."""
    out = tmp / 'effective.json'
    ratchet.check(r, checkpoint, [], '', effective_out=out)
    found: dict[str, Any] = json.loads(out.read_text()).get(CASE, {})
    return found


def untransformed(want: dict[str, Any] | None) -> bool:
    r"""No expectation, or the reference's own refusal: never the core's."""
    return want is None or (want.get('result'), want.get('code')) == (
        'refusal',
        'INTERNAL_ERROR',
    )


def written(
    r: Path, checkpoint: str, files: dict[str, Path]
) -> dict[str, dict[str, Any] | None]:
    r"""``accept``'s proposed projection of ``CASE``, by runner."""
    found = ratchet.accept(r, list(files.values()), checkpoint)
    return {
        runner: next(
            (
                x['projection']
                for x in found.get(runner, [])
                if x['case'] == CASE
            ),
            None,
        )
        for runner in RUNNERS
    }


REFERENCE_PROJECTION = {
    'outcome.result': 'refusal',
    'outcome.stage': 'materialize',
    'outcome.error': 'StructuredDataError',
    'outcome.code': 'INTERNAL_ERROR',
}


# --- passes ------------------------------------------------------------------


def test_the_approved_difference_is_each_runner_s_effective_expectation(
    repo: Path, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    before = helpers.checkpoint_of(repo)
    commit(repo, 'T1.32g: the admission', admission(APPROVED))

    def expected() -> None:
        found = expectations(repo, before, tmp_path)
        assert found.get('python-core') == CORE_OUTCOME, found
        host = found.get('rust-host') or {}
        assert (host.get('result'), host.get('code')) == (
            'refusal',
            'INVALID_REQUEST',
        ), found
        # python-old, which no entry names, is held to the reference.
        assert untransformed(found.get('python-old')), found

    G.held('exits', f'py:{request.node.name}', expected)


def test_accept_writes_every_runner_s_baseline_and_the_admission_passes(
    repo: Path, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    before = helpers.checkpoint_of(repo)
    files = observed(repo, tmp_path)
    commit(repo, 'T1.32g: the admission', admission(APPROVED))

    def admitted() -> None:
        found = written(repo, before, files)
        assert found['python-core'] == {
            f'outcome.{k}': v for k, v in CORE_OUTCOME.items()
        }, found
        assert found['rust-host'] == {
            'outcome.result': 'refusal',
            'outcome.stage': None,
            'outcome.error': None,
            'outcome.code': 'INVALID_REQUEST',
        }, found
        # The earlier rule: python-old at the reference's own INTERNAL_ERROR.
        assert found['python-old'] == REFERENCE_PROJECTION, found
        commit(
            repo,
            'T1.32g: the baselines',
            lambda r: helpers.accepted(r, before, tuple(files.values())),
        )
        problems = mine(check(repo, before, tuple(files.values())))
        assert problems == [], problems

    G.held('exits', f'py:{request.node.name}', admitted)


# --- the comparator ------------------------------------------------------------


def compared(
    r: Path, tmp: Path, observation: dict[str, Any], expect: Path | None
) -> Any:
    r"""``compare.compare`` of python-old's ``observation`` of ``CASE``
    against the reference, under the candidate's case file (acceptance)."""
    cases = tmp / 'cases'
    cases.mkdir(exist_ok=True)
    written_case = cases / f'{CASE}.json'
    written_case.write_text(case_path(r).read_text())
    digest = run_python.digest(written_case.read_bytes())
    reference = copy.deepcopy(helpers.reference(r)[CASE])
    reference['identity']['inputs']['case'] = digest
    mine = copy.deepcopy(observation)
    mine['identity']['inputs']['case'] = digest
    paths = []
    for name, line in (('python-ref', reference), ('python-old', mine)):
        path = tmp / f'{name}-compared.jsonl'
        path.write_text(json.dumps(line) + '\n')
        paths.append(path)
    return compare.compare([cases], paths[0], paths[1:], expect_file=expect)


def outcome_failures(report: Any) -> list[str]:
    return [
        text
        for rule, text in report.failures
        if rule == 'outcome' and text.startswith(f'{CASE}: python-old')
    ]


def python_old_answering_as_the_core(r: Path, tmp: Path) -> dict[str, Any]:
    path = observe(r, tmp, 'python-old', core_answer('python-old'))
    line: dict[str, Any] = {
        json.loads(x)['case']: json.loads(x)
        for x in path.read_text().splitlines()
    }[CASE]
    return line


def effective_file(r: Path, checkpoint: str, tmp: Path) -> Path:
    out = tmp / 'expect.json'
    ratchet.check(r, checkpoint, [], '', effective_out=out)
    return out


def test_the_comparator_passes_an_approved_first_admission_difference(
    repo: Path, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    before = helpers.checkpoint_of(repo)
    commit(
        repo,
        'T1.32g: the admission',
        admission(items(('python-old', CORE_NEW))),
    )
    line = python_old_answering_as_the_core(repo, tmp_path)
    # Not vacuous: without the expectations, the outcome rule fails it.
    assert outcome_failures(compared(repo, tmp_path, line, None))

    def passes() -> None:
        expect = effective_file(repo, before, tmp_path)
        report = compared(repo, tmp_path, line, expect)
        assert outcome_failures(report) == [], report.failures

    G.held('exits', f'py:{request.node.name}', passes)


COMPARED_FAULTS: dict[str, tuple[list[dict[str, str]], dict[str, Any]]] = {
    'under a proposed entry': (
        items(('python-old', CORE_NEW)),
        {'status': 'proposed', 'approved_by': None},
    ),
    'a difference the entry does not name': (
        items(('python-core', CORE_NEW)),
        {},
    ),
    'not the entry s new outcome': (
        items(
            ('python-old', 'refusal INVALID_REQUEST at predict (ValueError)')
        ),
        {},
    ),
}


@pytest.mark.parametrize('fault', sorted(COMPARED_FAULTS))
def test_the_comparator_still_fails_an_unapproved_first_admission_difference(
    repo: Path, tmp_path: Path, fault: str
) -> None:
    case_outcomes, entry = COMPARED_FAULTS[fault]
    before = helpers.checkpoint_of(repo)
    commit(repo, 'T1.32g: the admission', admission(case_outcomes, **entry))
    line = python_old_answering_as_the_core(repo, tmp_path)
    expect = effective_file(repo, before, tmp_path)
    report = compared(repo, tmp_path, line, expect)
    assert outcome_failures(report), report.failures


# --- still fails: the expectations and accept ------------------------------------


FAULTS: dict[str, tuple[list[dict[str, str]], dict[str, Any], str]] = {
    # The same difference, the entry still proposed.
    'under a proposed entry': (
        APPROVED,
        {'status': 'proposed', 'approved_by': None},
        'python-core',
    ),
    # The entry names python-core only; rust-host differs as well.
    'a difference the entry does not name': (
        items(('python-core', CORE_NEW)),
        {},
        'rust-host',
    ),
}


@pytest.mark.parametrize('fault', sorted(FAULTS))
def test_an_unapproved_difference_is_no_expectation_and_gets_no_baseline(
    repo: Path, tmp_path: Path, fault: str
) -> None:
    case_outcomes, entry, runner = FAULTS[fault]
    before = helpers.checkpoint_of(repo)
    files = observed(repo, tmp_path)
    commit(repo, 'T1.32g: the admission', admission(case_outcomes, **entry))
    found = expectations(repo, before, tmp_path)
    assert untransformed(found.get(runner)), found
    assert written(repo, before, files)[runner] is None


def test_a_runner_answering_other_than_the_entry_gets_no_baseline(
    repo: Path, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    before = helpers.checkpoint_of(repo)
    # python-core refuses at predict; the entry says at materialize.
    files = observed(repo, tmp_path, **{'python-core': {'stage': 'predict'}})
    commit(repo, 'T1.32g: the admission', admission(APPROVED))
    found = written(repo, before, files)
    assert found['python-core'] is None, found

    def per_runner() -> None:
        # Per runner: the others meet theirs, and get their baselines.
        assert found['rust-host'] is not None, found
        assert found['python-old'] == REFERENCE_PROJECTION, found
        # And the admission fails: python-core holds no proposed baseline.
        commit(
            repo,
            'T1.32g: the baselines',
            lambda r: helpers.accepted(r, before, tuple(files.values())),
        )
        problems = mine(check(repo, before, tuple(files.values())))
        assert problems == [
            f'baselines: {CASE}: python-core: an acceptance case with no '
            'proposed baseline'
        ], problems

    G.held('exits', f'py:{request.node.name}', per_runner)


def test_an_entry_to_internal_error_is_no_expectation(
    repo: Path, tmp_path: Path
) -> None:
    before = helpers.checkpoint_of(repo)
    commit(
        repo,
        'T1.32g: the admission',
        admission(
            items(
                (
                    'python-core',
                    'refusal INTERNAL_ERROR at predict (StructuredDataError)',
                )
            ),
            approved_by='owner',
        ),
    )
    found = expectations(repo, before, tmp_path)
    want = found.get('python-core')
    assert want is None or want.get('stage') != 'predict', found
    assert any(
        'no entry accepts INTERNAL_ERROR' in p for p in check(repo, before, ())
    )
