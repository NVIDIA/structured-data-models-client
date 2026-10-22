# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32g part 2a: python-old may hold the reference's own INTERNAL_ERROR.

Written by the package's independent test author, before the change.
``ratchet.py``'s baselines rule takes every ``INTERNAL_ERROR`` for a panic
(the observations carry no panic flag) and fails any held baseline whose
runner observes one. python-old is the reference code on its default
routes, and the comparator refuses a python-old observation with
``_native`` loaded, so its ``INTERNAL_ERROR`` is the reference's own
``StructuredDataError``, never a core panic. Seven inventory cases have
the reference answer that way, so none of them can be admitted, and
``gate.sh --locked 1b`` cannot pass.

The rule (approved by the owner), for python-old only: where the
committed python-ref line for a case is ``INTERNAL_ERROR`` and python-old's
observation equals that line exactly (``ratchet.semantic``, the reference
rule's projection), that ``INTERNAL_ERROR`` is the old client's real crash,
recorded as such, with the reference result as python-old's baseline; the
gate's output says so (a ``baselines`` note, which ``ratchet.py check``
prints). It is still a crash, never called "not a panic". Each claim below
runs the real
``ratchet.check`` or ``ratchet.accept`` on
``tests/test_contract_ratchet.py``'s temporary repository:

- accepted: a checkpoint holding python-old's baseline at the reference's
  ``INTERNAL_ERROR``, each of the seven cases an acceptance case, passes
  when python-old answers as the reference does, and ``ratchet.py check``
  prints a note naming the case as the old client's crash; and an admission that
  proposes that baseline with ``accept`` lands, and the next change on top
  passes;
- still failing: python-old's ``INTERNAL_ERROR`` at another stage, with
  another error, or differing from the reference line outside the baseline
  fields; python-old's ``INTERNAL_ERROR`` where the reference succeeded;
  python-core's or rust-host's, even equal to the reference's; an
  acceptance case python-old observes with no proposed baseline, whatever
  the reference answered;
- still failing, at once: a proposed (not yet held) baseline at
  ``INTERNAL_ERROR`` on python-core or rust-host, or on python-old unequal
  to the reference line, fails the change that proposes it;
- ``accept`` writes python-old's baseline at the reference's
  ``INTERNAL_ERROR``;
- ``register.py`` still refuses an entry whose new outcome is
  ``INTERNAL_ERROR``, even an owner's, even the reference's own.

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
import yaml

# Gate tooling (T3.0): CI runs it in the tooling jobs, not in every matrix job.
pytestmark = pytest.mark.tooling

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
register = _module('sdm_register', ROOT / 'scripts/rust_core/register.py')
ratchet = helpers.ratchet

origin = helpers.origin
git, commit, land = helpers.git, helpers.commit, helpers.land
check, observe = helpers.check, helpers.observe
append_entry, outcome = helpers.append_entry, helpers.outcome
CELL = helpers.CELL
#: An acceptance case the reference answers with success at materialize.
SUCCEEDS = helpers.ACCEPTED

#: The tests' own cases (``tests/golden/t132g-fixture-cases``): frozen
#: copies, under ids of their own, of the inventory cases the reference
#: answers with its own INTERNAL_ERROR and of a diagnostic case it answers
#: with success, as they were at bc4b2d0c (diagnostic, held by no runner),
#: with the committed reference's lines for them. A landing adds them to
#: the temporary repository (:func:`repo`), so no admission in the corpus
#: changes what these tests judge.
FIXTURES = ROOT / 'tests/golden/t132g-fixture-cases'
OWN = 't132g.own.'
#: The inventory cases the reference answers with its own INTERNAL_ERROR.
REAL_SEVEN = (
    'decimal.class-target-38-18',
    'key.decimal-18-0',
    'key.decimal-38-10',
    'key.hugeint',
    'key.varchar-large-buffers',
    'time.text-one-bad-value',
    'time.text-us-date',
)
SEVEN = tuple(OWN + case for case in REAL_SEVEN)
CASE = OWN + 'key.hugeint'
#: A corpus case for the register's own check, which reads this checkout.
REAL_CASE = 'key.hugeint'


def with_fixtures(r: Path) -> None:
    r"""The fixture cases, each in its slice's directory, and their
    reference lines appended to the committed reference."""
    for path in sorted((FIXTURES / 'cases').glob('*.json')):
        spec = json.loads(path.read_text())
        target = r / 'contract/cases' / spec['slice'].lower() / path.name
        target.write_text(path.read_text())
    reference = next((r / 'contract/reference').glob('*.jsonl'))
    reference.write_text(
        reference.read_text() + (FIXTURES / 'python-ref.jsonl').read_text()
    )


@pytest.fixture
def repo(origin: tuple[Path, str], tmp_path: Path) -> Path:
    r"""A worktree at a landing (not judged) on the checkpoint adding the
    fixture cases: the checkpoint each test's candidate is judged against."""
    source, landed = origin
    tree = tmp_path / 'tree'
    git(source, 'worktree', 'add', '-q', '--detach', str(tree), landed)
    land(tree, 'T1.32e', with_fixtures)
    return tree


PANIC_TEXT = 'refusal INTERNAL_ERROR at materialize (StructuredDataError)'


def panics(problems: list[str], case: str, runner: str) -> bool:
    return any(
        p.startswith(f'baselines: {case}: {runner}: INTERNAL_ERROR')
        for p in problems
    )


def fields(r: Path) -> list[str]:
    policy = yaml.safe_load((r / 'contract/compare-policy.yaml').read_text())
    return [str(f) for f in policy['baseline']]


def lines_of(path: Path) -> dict[str, dict[str, Any]]:
    return {
        line['case']: line
        for line in map(json.loads, path.read_text().splitlines())
    }


def case_path(r: Path, case: str) -> Path:
    [path] = (r / 'contract/cases').glob(f'*/{case}.json')
    return path


def admit(r: Path, case: str) -> None:
    r"""``case``'s file, its status acceptance."""
    path = case_path(r, case)
    spec = json.loads(path.read_text())
    spec['status'] = 'acceptance'
    path.write_text(json.dumps(spec, indent=1))


def baseline_file(r: Path, runner: str) -> Path:
    return r / f'contract/baselines/{CELL}/{runner}.jsonl'


def write_baselines(r: Path, runner: str, by_case: dict[str, Any]) -> None:
    baseline_file(r, runner).write_text(
        ''.join(
            json.dumps(by_case[case], sort_keys=True) + '\n'
            for case in sorted(by_case)
        )
    )


def hold(
    r: Path, runner: str, case: str, observed: Path, drop: bool = False
) -> None:
    r"""``runner``'s baseline of ``case``: the projection of its line in
    ``observed``, or none where ``drop``."""
    by_case = lines_of(baseline_file(r, runner))
    by_case.pop(case, None)
    if not drop:
        projected = ratchet.projection(lines_of(observed)[case], fields(r))
        by_case[case] = {
            'case': case,
            'runner': runner,
            'cell': CELL,
            'projection': projected,
            'digest': ratchet.digest(projected),
        }
    write_baselines(r, runner, by_case)


def checkpoint_holding(
    r: Path,
    runner: str,
    case: str,
    observed: Path,
    admitted: bool = True,
    drop: bool = False,
) -> str:
    r"""A landing on top of the checkpoint (not judged here), with
    ``case`` an acceptance case where ``admitted`` and ``runner``'s
    baseline of it held from ``observed``: the checkpoint the candidate is
    judged against."""

    def edit(tree: Path) -> None:
        if admitted:
            admit(tree, case)
        hold(tree, runner, case, observed, drop)

    return land(r, 'T1.32f', edit)


def outcome_of(**values: Any) -> Callable[[dict[str, Any]], None]:
    return lambda line: line['outcome'].update(values)


def changed(
    case: str, *edits: Callable[[dict[str, Any]], None]
) -> Callable[[dict[str, dict[str, Any]]], None]:
    def change(lines: dict[str, dict[str, Any]]) -> None:
        for edit in edits:
            edit(lines[case])

    return change


def panic_outcome(line: dict[str, Any]) -> None:
    line['outcome'].update(
        result='refusal', code='INTERNAL_ERROR', error='StructuredDataError'
    )


# --- the seeds are what they say --------------------------------------------


def test_the_seven_are_the_reference_s_internal_errors_and_diagnostic(
    repo: Path,
) -> None:
    reference = helpers.reference(repo)
    assert sorted(
        case
        for case, line in reference.items()
        if case.startswith(OWN)
        and line['outcome'].get('code') == 'INTERNAL_ERROR'
    ) == list(SEVEN)
    for case in SEVEN:
        assert reference[case]['outcome'] == {
            'result': 'refusal',
            'stage': 'materialize',
            'error': 'StructuredDataError',
            'code': 'INTERNAL_ERROR',
        }
        assert json.loads(case_path(repo, case).read_text())['status'] == (
            'diagnostic'
        )
    assert reference[SUCCEEDS]['outcome']['result'] == 'success'


# --- accepted ----------------------------------------------------------------


@pytest.mark.parametrize('case', SEVEN)
def test_python_old_holds_the_reference_s_own_internal_error(
    repo: Path, tmp_path: Path, case: str, request: pytest.FixtureRequest
) -> None:
    observed = observe(repo, tmp_path, 'python-old')
    landed = checkpoint_holding(repo, 'python-old', case, observed)
    commit(repo)
    assert lines_of(baseline_file(repo, 'python-old'))[case]['projection'][
        'outcome.code'
    ] == ('INTERNAL_ERROR')

    def held() -> None:
        found = check(repo, landed, (observed,))
        assert found == [], found

    G.held('exits', f'py:{request.node.name}', held)


def test_the_gate_output_records_it_as_the_old_client_s_crash(
    repo: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    request: pytest.FixtureRequest,
) -> None:
    observed = observe(repo, tmp_path, 'python-old')
    landed = checkpoint_holding(repo, 'python-old', CASE, observed)
    commit(repo)

    def recorded() -> None:
        code = ratchet.main(
            [
                'check',
                '--repo',
                str(repo),
                '--checkpoint',
                landed,
                '--protected-paths',
                '',
                '--observations',
                str(observed),
            ]
        )
        printed = capsys.readouterr().out.splitlines()
        assert code == 0, printed
        # One line of the gate's output names the case as python-old's
        # recorded crash (the owner's words): the baselines rule's, the
        # case, the runner, the code, and "old client's crash".
        said = [
            line
            for line in printed
            if line.startswith(f'baselines: {CASE}: python-old: ')
        ]
        assert len(said) == 1, printed
        assert 'INTERNAL_ERROR' in said[0], said
        assert "old client's crash" in said[0], said
        assert 'not a panic' not in said[0], said
        # Nothing else is said of the six other cases nobody admitted here.
        assert not any(
            line.startswith(f'baselines: {other}: ')
            for other in SEVEN
            if other != CASE
            for line in printed
        ), printed

    G.held('exits', f'py:{request.node.name}', recorded)


def test_an_admission_proposing_it_lands_and_the_next_change_passes(
    repo: Path, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    before = helpers.checkpoint_of(repo)
    observed = observe(repo, tmp_path, 'python-old')
    files = (observed,)

    def admission(r: Path) -> None:
        admit(r, CASE)
        # python-old's own outcome no entry may state (INTERNAL_ERROR); the
        # case is named through another runner's, as an admission does.
        append_entry(
            r,
            {
                'id': 'SEED-1',
                'kind': 'keep',
                'cases': [CASE],
                'case_outcomes': [
                    outcome(
                        CASE,
                        'python-core',
                        'none',
                        'refusal INVALID_REQUEST at materialize (ValueError)',
                    )
                ],
            },
        )

    commit(repo, 'T1.32g: the admission', admission)
    commit(
        repo,
        'T1.32g: the baselines',
        lambda r: helpers.accepted(r, before, files),
    )
    proposed = lines_of(baseline_file(repo, 'python-old'))
    assert proposed[CASE]['projection']['outcome.code'] == 'INTERNAL_ERROR'
    # The admission itself: nothing is held yet, so nothing panics.
    assert check(repo, before, files) == []
    landed = commit(
        repo,
        'Land T1.32g: the admission',
        trailers={
            'Reviewed-by': 'independent reviewer and Codex, T1.32g',
            'Protected-paths': 'none',
            'Base-SHA': before,
            'Checkpoint-SHA': before,
        },
    )
    commit(repo, 'T1.33: the next change')

    def next_change() -> None:
        found = check(repo, landed, files)
        assert found == [], found

    G.held('exits', f'py:{request.node.name}', next_change)


# --- still failing -------------------------------------------------------------


def _differs_outside_the_baseline(line: dict[str, Any]) -> None:
    r"""The public schema of the source, outside the baseline fields and
    inside the semantic projection. (Host warnings and their origins were
    the seed first; E110 as the orchestrator read it leaves them out of
    "exactly", so they are runtime noise, not the client's behaviour.)"""
    line['public']['schema']['tables'][0]['duplicate_keys'] += 1


#: A committed reference success's public result (a reference line is never
#: changed or removed), for a crash whose line otherwise equals the
#: reference's but carries one.
A_PUBLIC_RESULT = helpers.reference(ROOT)['base.binary']['public']['result']


def _another_public_result(line: dict[str, Any]) -> None:
    r"""A public result where the reference's crash has none (T1.36 C17)."""
    assert line['public']['result'] is None
    assert A_PUBLIC_RESULT is not None
    line['public']['result'] = copy.deepcopy(A_PUBLIC_RESULT)


DIFFERENT: dict[str, Callable[[dict[str, Any]], None]] = {
    'another stage': outcome_of(stage='predict'),
    'another error': outcome_of(error='ValueError'),
    'outside the baseline fields': _differs_outside_the_baseline,
    # T1.36 C17 (T1.32g p2f review P3): same code, another outcome result.
    'another result': outcome_of(result='success'),
    # T1.36 C17 (T1.32g p2f review P3): same outcome, another public result.
    'another public result': _another_public_result,
}


@pytest.mark.parametrize('difference', sorted(DIFFERENT))
def test_python_old_s_internal_error_other_than_the_reference_s_fails(
    repo: Path, tmp_path: Path, difference: str
) -> None:
    observed = observe(
        repo, tmp_path, 'python-old', changed(CASE, DIFFERENT[difference])
    )
    # Its baseline is its own observation, so the only problem is the panic.
    landed = checkpoint_holding(repo, 'python-old', CASE, observed)
    commit(repo)
    ours = ratchet.semantic(lines_of(observed)[CASE])
    assert ours != ratchet.semantic(helpers.reference(repo)[CASE])
    found = check(repo, landed, (observed,))
    assert panics(found, CASE, 'python-old'), found


def test_python_old_s_internal_error_where_the_reference_succeeded_fails(
    repo: Path, tmp_path: Path
) -> None:
    observed = observe(
        repo, tmp_path, 'python-old', changed(SUCCEEDS, panic_outcome)
    )
    # Held at that same INTERNAL_ERROR: equal to its baseline, not to the
    # reference, which succeeded.
    landed = checkpoint_holding(
        repo, 'python-old', SUCCEEDS, observed, admitted=False
    )
    commit(repo)
    found = check(repo, landed, (observed,))
    assert panics(found, SUCCEEDS, 'python-old'), found


#: A diagnostic corpus case, held by no runner, the reference refuses with
#: INVALID_REQUEST at materialize (StructuredDataError): the reference's
#: crash but for its code.
REFUSES = 's0.boundary.escape-lone-surrogate'


def _the_reference_refuses_but_for_the_code(r: Path) -> dict[str, Any]:
    reference = helpers.reference(r)[REFUSES]
    assert reference['outcome'] == {
        'result': 'refusal',
        'stage': 'materialize',
        'error': 'StructuredDataError',
        'code': 'INVALID_REQUEST',
    }
    return reference


def test_python_old_s_internal_error_where_the_reference_refused_with_another_code_fails(
    repo: Path, tmp_path: Path
) -> None:
    # T1.36 C17 (T1.32g p2f review P3): python-old's INTERNAL_ERROR equal to
    # the reference line but for the code (INVALID_REQUEST) is no crash.
    reference = _the_reference_refuses_but_for_the_code(repo)
    observed = observe(
        repo, tmp_path, 'python-old', changed(REFUSES, panic_outcome)
    )
    ours = lines_of(observed)[REFUSES]
    assert ours['outcome'] == {**reference['outcome'], 'code': 'INTERNAL_ERROR'}
    # Held at that same INTERNAL_ERROR: equal to its baseline, not to the
    # reference, which refused with another code.
    landed = checkpoint_holding(repo, 'python-old', REFUSES, observed)
    commit(repo)
    found = check(repo, landed, (observed,))
    assert panics(found, REFUSES, 'python-old'), found


def test_python_old_s_other_code_where_the_reference_crashed_is_no_crash(
    repo: Path, tmp_path: Path
) -> None:
    # T1.36 C17 (T1.32g p2f review P3): python-old answering with another
    # code (INVALID_REQUEST) where the reference crashed, held at the
    # reference's INTERNAL_ERROR, fails as an unapproved change and is never
    # noted as the old client's crash.
    for name in ('crashed', 'other-code'):
        (tmp_path / name).mkdir()
    crashed = observe(repo, tmp_path / 'crashed', 'python-old')
    landed = checkpoint_holding(repo, 'python-old', CASE, crashed)
    commit(repo)
    observed = observe(
        repo,
        tmp_path / 'other-code',
        'python-old',
        changed(CASE, outcome_of(code='INVALID_REQUEST')),
    )
    assert lines_of(observed)[CASE]['outcome'] == {
        **helpers.reference(repo)[CASE]['outcome'],
        'code': 'INVALID_REQUEST',
    }
    report = ratchet.check(repo, landed, [observed], None)
    assert any(
        p.startswith(f'baselines: {CASE}: python-old: {PANIC_TEXT} became ')
        and 'INVALID_REQUEST' in p
        for p in report.problems
    ), report.problems
    # The corpus's own seven are noted; this case is not.
    assert not any(
        n.startswith(f'baselines: {CASE}: ') and "old client's crash" in n
        for n in report.notes
    ), report.notes


@pytest.mark.parametrize('runner', ['python-core', 'rust-host'])
def test_another_runner_s_internal_error_fails_even_as_the_reference_s(
    repo: Path, tmp_path: Path, runner: str
) -> None:
    observed = observe(repo, tmp_path, runner)
    landed = checkpoint_holding(repo, runner, CASE, observed)
    commit(repo)
    # The reference's own outcome, relabelled: no difference but the runner.
    assert lines_of(observed)[CASE]['outcome'] == (
        helpers.reference(repo)[CASE]['outcome']
        if runner == 'python-core'
        else {
            **helpers.reference(repo)[CASE]['outcome'],
            'stage': None,
            'error': None,
        }
    )
    found = check(repo, landed, (observed,))
    assert panics(found, CASE, runner), found


NO_BASELINE: dict[str, tuple[str, Callable[[dict[str, Any]], None] | None]] = {
    'the reference succeeded, python-old too': (SUCCEEDS, None),
    'the reference succeeded, python-old INTERNAL_ERROR': (
        SUCCEEDS,
        panic_outcome,
    ),
    'the reference INTERNAL_ERROR, python-old as it': (CASE, None),
}


@pytest.mark.parametrize('seed', sorted(NO_BASELINE))
def test_an_acceptance_case_python_old_observes_with_no_baseline_fails(
    repo: Path, tmp_path: Path, seed: str
) -> None:
    case, edit = NO_BASELINE[seed]
    observed = observe(
        repo,
        tmp_path,
        'python-old',
        changed(case, edit) if edit is not None else None,
    )
    landed = checkpoint_holding(
        repo, 'python-old', case, observed, admitted=True, drop=True
    )
    commit(repo)
    found = check(repo, landed, (observed,))
    assert (
        f'baselines: {case}: python-old: an acceptance case with no proposed '
        'baseline' in found
    ), found


# --- still failing, at once: a proposed baseline --------------------------------
#
# A proposed baseline at INTERNAL_ERROR that no landing may hold fails the
# change that proposes it, not only the next landing's. Each candidate
# admits a case nothing holds yet and proposes one runner's baseline of it,
# equal to that runner's observation, so the proposal rule as it stands
# passes it.

#: A diagnostic case the reference answers with success, held by no runner.
UNHELD_SUCCESS = OWN + 's0.boundary.date-long-field'


def admission(
    case: str, runner: str, observed: Path, core_new: str
) -> Callable[[Path], None]:
    r"""``case`` admitted, an approved entry naming it (on python-core,
    whose new outcome is ``core_new``), and ``runner``'s baseline of it
    proposed from ``observed``."""

    def edit(r: Path) -> None:
        admit(r, case)
        append_entry(
            r,
            {
                'id': 'SEED-2',
                'kind': 'keep',
                'cases': [case],
                'case_outcomes': [
                    outcome(case, 'python-core', 'none', core_new)
                ],
            },
        )
        hold(r, runner, case, observed)

    return edit


def refused_at_once(problems: list[str], case: str, runner: str) -> bool:
    return any(
        p.startswith(f'baselines: {case}: {runner}: ') and 'INTERNAL_ERROR' in p
        for p in problems
    )


@pytest.mark.parametrize('runner', ['python-core', 'rust-host'])
def test_a_proposed_internal_error_of_another_runner_fails_at_once(
    repo: Path, tmp_path: Path, runner: str, request: pytest.FixtureRequest
) -> None:
    before = helpers.checkpoint_of(repo)
    observed = observe(repo, tmp_path, runner)
    commit(
        repo,
        'T1.32g: the admission',
        admission(
            CASE,
            runner,
            observed,
            'refusal INVALID_REQUEST at materialize (ValueError)',
        ),
    )
    proposed = lines_of(baseline_file(repo, runner))[CASE]['projection']
    assert proposed['outcome.code'] == 'INTERNAL_ERROR'

    def refused() -> None:
        found = check(repo, before, (observed,))
        assert refused_at_once(found, CASE, runner), found

    G.held('exits', f'py:{request.node.name}', refused)


PROPOSED_UNEQUAL: dict[str, tuple[str, Callable[[dict[str, Any]], None]]] = {
    'another stage': (CASE, outcome_of(stage='predict')),
    'another error': (CASE, outcome_of(error='ValueError')),
    'outside the baseline fields': (CASE, _differs_outside_the_baseline),
    'the reference succeeded': (UNHELD_SUCCESS, panic_outcome),
    # T1.36 C17 (T1.32g p2f review P3): the three seeds below.
    'another result': (CASE, outcome_of(result='success')),
    'another public result': (CASE, _another_public_result),
    'the reference refused with another code': (REFUSES, panic_outcome),
}


@pytest.mark.parametrize('seed', sorted(PROPOSED_UNEQUAL))
def test_a_proposed_python_old_internal_error_unequal_to_the_reference_fails_at_once(
    repo: Path, tmp_path: Path, seed: str, request: pytest.FixtureRequest
) -> None:
    case, edit = PROPOSED_UNEQUAL[seed]
    before = helpers.checkpoint_of(repo)
    reference = helpers.reference(repo)[case]
    observed = observe(repo, tmp_path, 'python-old', changed(case, edit))
    ours = lines_of(observed)[case]
    assert ours['outcome']['code'] == 'INTERNAL_ERROR'
    assert ratchet.semantic(ours) != ratchet.semantic(reference)
    core_new = (
        'success at materialize'
        if reference['outcome']['result'] == 'success'
        else 'refusal INVALID_REQUEST at materialize (ValueError)'
    )
    commit(
        repo,
        'T1.32g: the admission',
        admission(case, 'python-old', observed, core_new),
    )

    def refused() -> None:
        found = check(repo, before, (observed,))
        assert refused_at_once(found, case, 'python-old'), found

    G.held('exits', f'py:{request.node.name}', refused)


# --- accept --------------------------------------------------------------------


def test_accept_writes_python_old_s_baseline_at_the_reference_s_internal_error(
    repo: Path, tmp_path: Path
) -> None:
    observed = observe(repo, tmp_path, 'python-old')
    want = {
        'outcome.code': 'INTERNAL_ERROR',
        'outcome.error': 'StructuredDataError',
        'outcome.result': 'refusal',
        'outcome.stage': 'materialize',
    }

    def written(checkpoint: str | None) -> Any:
        found = ratchet.accept(repo, [observed], checkpoint)
        return next(
            (x['projection'] for x in found['python-old'] if x['case'] == CASE),
            None,
        )

    # Admitted, nothing held yet: from the reference.
    commit(repo, 'T1.32g: admitted', lambda r: admit(r, CASE))
    assert written(helpers.checkpoint_of(repo)) == want
    # Held at the checkpoint: from the baseline.
    landed = checkpoint_holding(repo, 'python-old', CASE, observed)
    commit(repo)
    assert written(landed) == want


# --- the register ----------------------------------------------------------------


@pytest.mark.parametrize('runner', ['python-old', 'python-core', 'rust-host'])
def test_the_register_still_refuses_an_entry_accepting_internal_error(
    runner: str,
) -> None:
    document = yaml.safe_load(
        (ROOT / 'docs/design/rust-core/register.yaml').read_text()
    )
    checkpoint = copy.deepcopy(document)
    document['entries'].append(
        {
            'id': 'SEED-99',
            'title': 'The reference answers with INTERNAL_ERROR',
            'kind': 'keep',
            'old_behavior': 'Diagnostic.',
            'new_behavior': 'Held at the reference outcome.',
            'affected_inputs': 'One case.',
            'cases': [REAL_CASE],
            'status': 'approved',
            'approved_by': 'owner',
            # An owner's entry names an owner-approved row (register.py,
            # T3.2), so the one problem left is the one this test seeds.
            'decision_ref': {
                'file': 'docs/design/rust-core/phase-1/decisions.md',
                'row': 'E1',
            },
            'notes': 'Seeded: the reference outcome itself, owner-approved.',
            'case_outcomes': [
                {
                    'case': REAL_CASE,
                    'cell': CELL,
                    'runner': runner,
                    'old': 'none',
                    'new': PANIC_TEXT,
                    'summary': 'seeded',
                }
            ],
        }
    )
    found = register.problems(document, ROOT, checkpoint)
    assert found == [
        f'SEED-99: {REAL_CASE} on {runner}: no entry accepts INTERNAL_ERROR, a panic'
    ], found
