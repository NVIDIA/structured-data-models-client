# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where ``gate.sh --locked`` writes its evidence is judged where it really
resolves (T1.15 fix round 1, Codex P2): a directory reached through a
symbolic link into the candidate's worktree is refused and nothing is
written there, while a directory genuinely outside it, reached directly or
through a link, passes. Through the real ``gate.sh`` on
``contract/tests/test_gate_modes.py``'s temporary repository and stubs,
whose fixtures this reuses. From the subagent's review of the same round:
a relative directory given from the worktree reached through a link is
refused too; the runtime cell is recorded, and check 1 outside the
reference's cell is reported, not gated; check 6 lists ``l0`` once;
``--only ratchet`` with no observations says so."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
if 'gate_modes_tests' in sys.modules:
    modes = sys.modules['gate_modes_tests']
else:
    spec = importlib.util.spec_from_file_location(
        'gate_modes_tests', HERE / 'test_gate_modes.py'
    )
    assert spec is not None and spec.loader is not None
    modes = importlib.util.module_from_spec(spec)
    sys.modules['gate_modes_tests'] = modes
    spec.loader.exec_module(modes)

origin, observed, stubbed, stubs = (
    modes.origin,
    modes.observed,
    modes.stubbed,
    modes.stubs,
)


def _run(
    stubbed: Path,
    tmp_path: Path,
    observed: dict[str, list[dict[str, Any]]],
    stubs: dict[str, Any],
    evidence: Path,
) -> Any:
    modes.commit(stubbed, f'T1.16: the evidence path, {tmp_path.name}')
    return modes.locked(
        stubbed,
        stubs,
        '1a',
        *modes.files(observed, tmp_path, modes.CHECKED),
        evidence=evidence,
    )


@pytest.mark.parametrize('into', ['the worktree', 'a directory in it'])
def test_evidence_through_a_link_into_the_worktree_is_refused(
    stubbed: Path,
    tmp_path: Path,
    observed: dict[str, list[dict[str, Any]]],
    stubs: dict[str, Any],
    into: str,
) -> None:
    target = stubbed if into == 'the worktree' else stubbed / 'contract'
    link = tmp_path / 'output-link'
    link.symlink_to(target, target_is_directory=True)
    done = _run(stubbed, tmp_path, observed, stubs, link / 'evidence')
    assert done.returncode == 2, done.stdout + done.stderr
    assert 'inside the worktree' in done.stderr, done.stderr
    assert not (target / 'evidence').exists()


@pytest.mark.parametrize('through', ['directly', 'a link'])
def test_evidence_genuinely_outside_the_worktree_passes(
    stubbed: Path,
    tmp_path: Path,
    observed: dict[str, list[dict[str, Any]]],
    stubs: dict[str, Any],
    through: str,
) -> None:
    outside = tmp_path / 'outside'
    outside.mkdir()
    where = outside
    if through == 'a link':
        where = tmp_path / 'outside-link'
        where.symlink_to(outside, target_is_directory=True)
    done = _run(stubbed, tmp_path, observed, stubs, where / 'evidence')
    assert done.returncode == 0, done.stdout + done.stderr
    found = modes.evidence_of(stubbed, stubs, outside / 'evidence')
    assert found['halves']['1a']['passed'] is True


def test_a_relative_evidence_directory_given_through_a_linked_cwd_is_refused(
    stubbed: Path,
    tmp_path: Path,
    observed: dict[str, list[dict[str, Any]]],
    stubs: dict[str, Any],
) -> None:
    r"""The subagent's ``./ev2``: run from the worktree reached through a
    link, so the shell's spelling of the directory is not the worktree's."""
    modes.commit(stubbed, f'T1.16: the evidence path, {tmp_path.name}')
    link = tmp_path / 'tree-link'
    link.symlink_to(stubbed, target_is_directory=True)
    done = subprocess.run(
        [
            str(link / 'scripts/rust_core/gate.sh'),
            '--base',
            modes.git(stubbed, 'rev-parse', 'HEAD~1'),
            '--locked',
            '1a',
            '--evidence',
            './ev2',
            '--wheel',
            str(stubs['wheels']['shipped']),
            '--test-ops-wheel',
            str(stubs['wheels']['test-ops']),
            *modes.files(observed, tmp_path, modes.CHECKED),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=link,
        env={
            **os.environ,
            'PWD': str(link),
            'PYTHON': sys.executable,
            **stubs['env'],
        },
    )
    assert done.returncode == 2, done.stdout + done.stderr
    assert 'inside the worktree' in done.stderr, done.stderr
    assert not (stubbed / 'ev2').exists()


def _elsewhere(runner: str, lines: list[dict[str, Any]]) -> None:
    for line in lines:
        if line['identity'].get('cell') is not None:
            line['identity']['cell'] = 'dbr17-pandas2'


def test_check_1_outside_the_reference_cell_is_reported_not_gated(
    stubbed: Path,
    tmp_path: Path,
    observed: dict[str, list[dict[str, Any]]],
    stubs: dict[str, Any],
) -> None:
    r"""Subagent P3-2: the reference comparisons gate only in the
    reference's cell, so check 1 fails there, saying why, and the evidence
    records the cell; in the reference's cell (the control) it passes, l0
    listed once in check 6."""
    modes.commit(stubbed, f'T1.16: the cell, {tmp_path.name}')
    done = modes.locked(
        stubbed,
        stubs,
        '1a',
        *modes.files(observed, tmp_path, modes.CHECKED, _elsewhere),
    )
    assert done.returncode == 1, done.stdout + done.stderr
    assert modes.table(done, '1a')['1'] == 'FAIL', done.stdout
    assert 'reported, not gated: observed in dbr17-pandas2' in done.stdout
    half = modes.evidence_of(stubbed, stubs)['halves']['1a']
    assert half['cell'] == 'dbr17-pandas2'
    assert half['checks']['1']['result'] == 'reported, not gated'

    done = modes.locked(
        stubbed,
        stubs,
        '1a',
        *modes.files(observed, tmp_path, modes.CHECKED),
        evidence=tmp_path / 'control',
    )
    assert done.returncode == 0, done.stdout + done.stderr
    half = modes.evidence_of(stubbed, stubs, tmp_path / 'control')['halves']
    assert half['1a']['cell'] == 'ref-pandas3'
    assert half['1a']['checks']['6']['steps'].count('l0') == 1


def test_only_ratchet_without_observations_says_so(
    stubbed: Path, tmp_path: Path
) -> None:
    modes.commit(stubbed, f'T1.16: the ratchet alone, {tmp_path.name}')
    done = modes.gate(
        stubbed,
        '--base',
        modes.git(stubbed, 'rev-parse', 'HEAD~1'),
        '--only',
        'ratchet',
    )
    assert 'no observations: the ratchet judges the tree alone' in done.stdout
