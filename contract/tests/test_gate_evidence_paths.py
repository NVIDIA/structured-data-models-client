# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where ``gate.sh --locked`` writes its evidence is judged where it really
resolves (T1.15 fix round 1, Codex P2): a directory reached through a
symbolic link into the candidate's worktree is refused and nothing is
written there, while a directory genuinely outside it, reached directly or
through a link, passes. Through the real ``gate.sh`` on
``contract/tests/test_gate_modes.py``'s temporary repository and stubs,
whose fixtures this reuses."""

from __future__ import annotations

import importlib.util
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
