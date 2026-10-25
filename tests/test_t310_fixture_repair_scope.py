# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An explicit fixture-repair scope requires approval and preserves real checks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/rust_core'))
pytestmark = pytest.mark.tooling


def _handoff() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        't310_fixture_scope_handoff', ROOT / 'scripts/rust_core/handoff.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HANDOFF = _handoff()
SCOPE = 't310-fixtures-7ab261a6d'
SOURCE = '7ab261a6d0139c87e6c1bae13d563fded79843e3'
DECISIONS = 'docs/design/rust-core/phase-3/decisions.md'
ROW = 'T310-OWNER-2026-10-25-FIXTURE-SCOPE-1'
NODES = (
    'tests/test_contract_ratchet.py::'
    'test_the_fixture_reuses_a_removed_worktree_without_pruning_failed_evidence',
    'tests/test_module_fixture_lifetime.py',
    'contract/tests/test_gate_controls.py::'
    'test_before_the_harness_lands_the_bootstrap_says_so',
    'contract/tests/test_gate_modes.py::'
    'test_only_runs_one_check_and_says_so[observations]',
)
OMITTED = (
    'tests/test_t20_bootstrap.py',
    'tests/test_t20_decisions.py',
    'tests/test_t20e_reference_change.py',
    'tests/test_t20f_case_keys.py',
    'tests/test_t21a_temporary_approval.py',
    'tests/test_t132g_reference_internal_error.py',
)
SUBSTANTIVE = (
    'tests/test_contract_coverage.py',
    'tests/test_coverage_rule6_evidence.py',
    'packages/structured-data-models-client/tests/test_shared_provider_rules.py',
    'packages/structured-data-models-client/tests/test_the_materialize_route.py',
    'packages/structured-data-models-client/tests/'
    'test_invalid_escape_warning_on_the_core.py',
    'rust/crates/sdm-contract/tests/data_vocabulary.rs',
    'scripts/rust_core/coverage.py',
)


def _policy(root: Path, status: str) -> None:
    path = root / DECISIONS
    path.parent.mkdir(parents=True, exist_ok=True)
    # Synthetic approval is confined to this test's temporary repository.
    path.write_text(
        '| # | Decision | Status |\n'
        '|---|---|---|\n'
        f'| {ROW} | {SCOPE}, source {SOURCE} | {status} |\n'
    )


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / 'repo'
    root.mkdir()
    for name in (*HANDOFF.FIXTURE_REPAIR_FILES, *SUBSTANTIVE):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    _policy(root, 'approved by owner 2026-10-25')
    return root


def test_default_selection_still_runs_complete_changed_files() -> None:
    touched = [*OMITTED, *(node.split('::')[0] for node in NODES)]
    before = HANDOFF.own_suites(ROOT, touched)
    assert before == [(*HANDOFF.affected.PYTEST, name) for name in touched]
    assert all('::' not in argument for suite in before for argument in suite)


def test_the_explicit_scope_keeps_every_substantive_target_and_all_factory_controls(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    touched = [*HANDOFF.FIXTURE_REPAIR_FILES, *SUBSTANTIVE]
    suites, proof = HANDOFF.fixture_repair_plan(root, touched, 'T3.10')
    assert {(*HANDOFF.affected.PYTEST, node) for node in NODES} <= set(suites)
    assert not {(*HANDOFF.affected.PYTEST, name) for name in OMITTED} & set(
        suites
    )
    assert set(HANDOFF.own_suites(root, list(SUBSTANTIVE))) <= set(suites)
    assert proof['id'] == SCOPE and proof['source'] == SOURCE
    assert proof['suites'] == [list(suite) for suite in suites]
    assert proof['files'] == {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in HANDOFF.FIXTURE_REPAIR_FILES
    }
    assert proof['decision_ref'] == {
        'file': DECISIONS,
        'row': ROW,
        'sha256': hashlib.sha256((root / DECISIONS).read_bytes()).hexdigest(),
    }
    # Future substantive tests cannot disappear behind this finite exception.
    extra = root / 'tests/test_new_substantive_regression.py'
    extra.write_text('def test_regression():\n    assert True\n')
    wider, _ = HANDOFF.fixture_repair_plan(
        root, [*touched, extra.relative_to(root).as_posix()], 'T3.10'
    )
    assert (
        *HANDOFF.affected.PYTEST,
        extra.relative_to(root).as_posix(),
    ) in wider


@pytest.mark.parametrize('package', ['T3.11', 'T4.3'])
def test_another_package_cannot_use_the_fixture_repair_scope(
    tmp_path: Path, package: str
) -> None:
    with pytest.raises(HANDOFF.Failure, match='only T3.10'):
        HANDOFF.fixture_repair_plan(_tree(tmp_path), [], package)


@pytest.mark.parametrize(
    'name',
    [
        *OMITTED,
        *(node.split('::')[0] for node in NODES),
        'conftest.py',
        '.github/workflows/ci.yml',
    ],
)
def test_every_pinned_file_refuses_unreviewed_changes_or_missing_inputs(
    tmp_path: Path, name: str
) -> None:
    root = _tree(tmp_path)
    path = root / name
    path.write_bytes(path.read_bytes() + b'\n# an unreviewed change\n')
    with pytest.raises(HANDOFF.Failure, match=name):
        HANDOFF.fixture_repair_plan(root, [], 'T3.10')
    path.unlink()
    with pytest.raises(HANDOFF.Failure, match=name):
        HANDOFF.fixture_repair_plan(root, [], 'T3.10')


@pytest.mark.parametrize(
    'status',
    [
        'PROPOSED, owner approval pending',
        'approved by reviewer 2026-10-25',
        'approved by owner 2026-10-25, pending',
    ],
)
def test_proposed_or_partial_policy_cannot_authorize_the_scope(
    tmp_path: Path, status: str
) -> None:
    root = _tree(tmp_path)
    _policy(root, status)
    with pytest.raises(HANDOFF.Failure, match='owner-approved'):
        HANDOFF.fixture_repair_plan(root, [], 'T3.10')
    (root / DECISIONS).unlink()
    with pytest.raises(HANDOFF.Failure, match='owner-approved'):
        HANDOFF.fixture_repair_plan(root, [], 'T3.10')


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            'GIT_AUTHOR_NAME': 'Fixture scope test',
            'GIT_AUTHOR_EMAIL': 'fixture@example.invalid',
            'GIT_COMMITTER_NAME': 'Fixture scope test',
            'GIT_COMMITTER_EMAIL': 'fixture@example.invalid',
            'GIT_CONFIG_GLOBAL': os.devnull,
            'GIT_CONFIG_NOSYSTEM': '1',
        },
    ).stdout.strip()


@pytest.mark.parametrize('scoped', [False, True])
@pytest.mark.parametrize('fails', [False, True])
@pytest.mark.parametrize('prior_python', [None, '/previous/python'])
def test_cli_stamps_only_completed_actual_selection_with_honest_scope_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    scoped: bool,
    fails: bool,
    prior_python: str | None,
) -> None:
    if prior_python is None:
        monkeypatch.delenv('SDM_PYTHON', raising=False)
    else:
        monkeypatch.setenv('SDM_PYTHON', prior_python)
    root = _tree(tmp_path)
    _git(root, 'init', '-q', '-b', 'topic')
    _git(root, 'commit', '-q', '--allow-empty', '-m', 'base')
    base = _git(root, 'rev-parse', 'HEAD')
    _git(root, 'add', '.')
    _git(root, 'commit', '-q', '-m', 'T3.10: synthetic fixture repair')
    sha = _git(root, 'rev-parse', 'HEAD')
    touched = [*HANDOFF.FIXTURE_REPAIR_FILES, *SUBSTANTIVE, DECISIONS]
    expected = (
        HANDOFF.fixture_repair_plan(root, touched, 'T3.10')[0]
        if scoped
        else HANDOFF.own_suites(root, touched)
    )
    received: list[list[tuple[str, ...]]] = []

    def run_suites(step, where, suites, python):
        assert where == root and python == '/lane/python'
        assert os.environ['SDM_PYTHON'] == '/lane/python'
        received.append(suites)
        if fails:
            step.problems.append('the selected control failed')
        return step

    monkeypatch.chdir(root)
    monkeypatch.setattr(HANDOFF, 'finish_on_signals', lambda: None)
    monkeypatch.setattr(
        HANDOFF, 'step_lint', lambda *args: HANDOFF.Step('lint')
    )
    monkeypatch.setattr(HANDOFF, 'run_suites', run_suites)
    argv = [
        'run',
        '--package',
        'T3.10',
        '--base',
        base,
        '--python',
        '/lane/python',
    ]
    if scoped:
        argv += ['--fixture-repair-scope', SCOPE]
    with monkeypatch.context() as environment:
        # Register the key even when absent, owning the CLI's direct write.
        environment.setenv('SDM_PYTHON', '/test-owned/python')
        assert HANDOFF.main(argv) == (1 if fails else 0)
    assert os.environ.get('SDM_PYTHON') == prior_python
    said = capsys.readouterr().out
    assert len(received) == 1 and set(received[0]) == set(expected)
    path = root / '.git/sdm-handoff' / f'{sha}.json'
    if fails:
        assert not path.exists() and 'NO STAMP' in said
    else:
        stamp = json.loads(path.read_text())
        assert stamp['ok'] is True and stamp['sha'] == sha
        if scoped:
            assert stamp['own_test_scope']['suites'] == [
                list(suite) for suite in received[0]
            ]
            assert (
                stamp['own_test_scope']['files'] == HANDOFF.FIXTURE_REPAIR_FILES
            )
            assert SCOPE in said and SOURCE in said
        else:
            assert 'own_test_scope' not in stamp and SCOPE not in said
