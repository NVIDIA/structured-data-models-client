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

origin, repo = helpers.origin, helpers.repo
git, commit, land = helpers.git, helpers.commit, helpers.land
check, observe = helpers.check, helpers.observe
append_entry, outcome = helpers.append_entry, helpers.outcome
CELL = helpers.CELL
#: An acceptance case the reference answers with success at materialize.
SUCCEEDS = helpers.ACCEPTED

#: The inventory cases the reference answers with its own INTERNAL_ERROR.
SEVEN = (
    'decimal.class-target-38-18',
    'key.decimal-18-0',
    'key.decimal-38-10',
    'key.hugeint',
    'key.varchar-large-buffers',
    'time.text-one-bad-value',
    'time.text-us-date',
)
CASE = 'key.hugeint'
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
        if line['outcome'].get('code') == 'INTERNAL_ERROR'
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
    line['host'] = {'warnings': ['UserWarning', 'FutureWarning']}


DIFFERENT: dict[str, Callable[[dict[str, Any]], None]] = {
    'another stage': outcome_of(stage='predict'),
    'another error': outcome_of(error='ValueError'),
    'outside the baseline fields': _differs_outside_the_baseline,
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
            'cases': [CASE],
            'status': 'approved',
            'approved_by': 'owner',
            'decision_ref': None,
            'notes': 'Seeded: the reference outcome itself, owner-approved.',
            'case_outcomes': [
                {
                    'case': CASE,
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
        f'SEED-99: {CASE} on {runner}: no entry accepts INTERNAL_ERROR, a panic'
    ], found
