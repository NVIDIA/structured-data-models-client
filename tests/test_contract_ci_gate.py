# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whether CI's gate runs fails closed (T1.18; design, "CI").

Each gate step of ``.github/workflows/ci.yml``, its text as CI runs it
(``bash -e``), runs here against a stub ``python`` answering
``ratchet.py where`` and a stub ``gate.sh`` recording its call. On a push to
the phase branch or a branch under it the gate must run, decided from
GitHub's event and ref and not from the candidate's ``ratchet.py``: an empty
``where`` or a failing one fails the step and never reaches a skip; off
those branches an empty ``where`` skips the gate; with a base, the gate is
called with it.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / '.github/workflows/ci.yml'
LANE = 'rust-core-phase-1-lane-1a'


def _gate_steps() -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    r"""Each job's gate step, with the job's environment."""
    jobs = yaml.safe_load(CI.read_text(encoding='utf-8'))['jobs']
    found = []
    for name, job in jobs.items():
        for step in job.get('steps', []):
            if str(step.get('name', '')).startswith('gate ('):
                found.append((name, step, job.get('env') or {}))
    return found


STEPS = _gate_steps()


def test_every_job_that_gates_has_its_step_here() -> None:
    assert {name for name, _, _ in STEPS} == {
        'rust-test',
        'contract-python-ref-pandas3',
        'contract-python-dbr17-pandas2',
        'contract-python-floor-pandas2-2',
    }


def _executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run(
    tmp_path: Path,
    step: dict[str, Any],
    job_env: dict[str, Any],
    where: str | None,
    event: str,
    ref: str,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    r"""The step's text in a scratch directory: ``where`` is what the stub
    ``ratchet.py where`` prints, or None for a failing one."""
    work = tmp_path / 'work'
    bin_dir = tmp_path / 'bin'
    called = tmp_path / 'gate-called'
    answer = 'exit 1' if where is None else f'printf "%s\\n" {where!r}; exit 0'
    _executable(
        bin_dir / 'python',
        '#!/usr/bin/env bash\n'
        'if [ "$2" = where ]; then ' + answer + '; fi\n'
        'exit 0\n',
    )
    # The gate records its call and writes observations of the job's cell.
    _executable(
        work / 'scripts/rust_core/gate.sh',
        '#!/usr/bin/env bash\n'
        f'echo "$@" > {str(called)!r}\n'
        'mkdir -p gate-observations\n'
        'for r in python-core python-old; do\n'
        '  printf \'{"identity": {"cell": "%s"}}\\n\' "$CELL" '
        '> gate-observations/$r.jsonl\n'
        'done\n'
        'echo "gate: passed"\n',
    )
    env = {
        'PATH': f'{bin_dir}{os.pathsep}{os.environ["PATH"]}',
        'HOME': str(tmp_path),
        'GITHUB_EVENT_NAME': event,
        'GITHUB_REF_NAME': ref,
        **{k: str(v) for k, v in job_env.items()},
        **{k: str(v) for k, v in (step.get('env') or {}).items()},
    }
    done = subprocess.run(
        ['bash', '--noprofile', '--norc', '-e', '-c', step['run']],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return done, called


@pytest.mark.parametrize(
    ('job', 'step', 'job_env'), STEPS, ids=[name for name, _, _ in STEPS]
)
@pytest.mark.parametrize('ref', ['rust-core-phase-1', LANE])
@pytest.mark.parametrize('where', ['', None])
def test_on_the_phase_branch_and_under_it_the_gate_cannot_be_skipped(
    tmp_path: Path,
    job: str,
    step: dict[str, Any],
    job_env: dict[str, Any],
    ref: str,
    where: str | None,
) -> None:
    done, called = _run(tmp_path, step, job_env, where, 'push', ref)
    assert done.returncode != 0, done.stdout + done.stderr
    assert not called.exists()
    assert 'gate: ratchet.py where' in done.stdout


@pytest.mark.parametrize(
    ('job', 'step', 'job_env'), STEPS, ids=[name for name, _, _ in STEPS]
)
def test_with_a_base_the_gate_runs_with_it(
    tmp_path: Path, job: str, step: dict[str, Any], job_env: dict[str, Any]
) -> None:
    done, called = _run(
        tmp_path, step, job_env, '--base origin/rust-core-phase-1', 'push', LANE
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert called.read_text().startswith('--base origin/rust-core-phase-1 ')


def test_off_the_phase_branch_an_empty_where_skips_rust_host_s_gate(
    tmp_path: Path,
) -> None:
    [(_, step, job_env)] = [s for s in STEPS if s[0] == 'rust-test']
    done, called = _run(tmp_path, step, job_env, '', 'push', 'main')
    assert done.returncode == 0, done.stdout + done.stderr
    assert not called.exists()
    done, called = _run(tmp_path, step, job_env, '', 'pull_request', LANE)
    assert done.returncode == 0, done.stdout + done.stderr
    assert not called.exists()


def _git(work: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(work), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.mark.parametrize(
    'seed',
    [
        'comparator',
        'renamed',
        'ratchet-too',
        'not-warnings',
        'landed',
        'after-landing',
        'unreadable',
        'other-cell',
    ],
)
def test_the_floor_cell_reports_only_a_comparator_before_t1_18_landed(
    tmp_path: Path, seed: str
) -> None:
    r"""T1.18's own run (E65(c)): the floor job reports a gate whose one
    failing step is the comparator while the checkpoint's history holds no
    landing of T1.18, whatever its comparator's text, and says so; a
    checkpoint that is T1.18's landing or descends from it gates, its
    comparator's constant renamed or not, as do a failing ratchet beside
    the comparator, a difference other than ``warnings``, a checkpoint git
    cannot read, and another cell's job."""
    job = (
        'contract-python-dbr17-pandas2'
        if seed == 'other-cell'
        else 'contract-python-floor-pandas2-2'
    )
    [(_, step, job_env)] = [s for s in STEPS if s[0] == job]
    work = tmp_path / 'work'
    (work / 'scripts/rust_core').mkdir(parents=True)
    _git(work, 'init', '-q')
    identity = ('-c', 'user.name=t', '-c', 'user.email=t@t')

    def landing(subject: str, comparator: str) -> None:
        (work / 'scripts/rust_core/compare.py').write_text(comparator)
        _git(work, 'add', '.')
        _git(work, *identity, 'commit', '-qm', subject)

    landing('Land T1.26a: the one before', '# no cells\n')
    if seed == 'renamed':
        landing('Land T1.26b: a rename', 'REFERENCE_RUNTIME_CELL = 1\n')
    if seed in ('landed', 'after-landing'):
        landing('Land T1.18: L2 in CI', 'REFERENCE_RUNTIME_CELL = 1\n')
    if seed == 'after-landing':
        landing('Land T1.19: the next', 'REFERENCE_RUNTIME_CELL = 2\n')
    checkpoint = (
        'f' * 40 if seed == 'unreadable' else _git(work, 'rev-parse', 'HEAD')
    )
    failed = ['compare'] + (['ratchet'] if seed == 'ratchet-too' else [])
    differs = ['warnings'] + (['result'] if seed == 'not-warnings' else [])
    summary = ''.join(
        f'echo "    | DIFFERS {rule}: a case: differs"\n' for rule in differs
    ) + ''.join(f'echo "  FAIL {name}"\n' for name in failed)
    _executable(
        work / 'scripts/rust_core/gate.sh',
        '#!/usr/bin/env bash\n'
        f'echo "gate: base {checkpoint}, checkpoint {checkpoint}"\n'
        + summary
        + 'echo "gate: FAILED"\nexit 1\n',
    )
    bin_dir = tmp_path / 'bin'
    _executable(
        bin_dir / 'python',
        '#!/usr/bin/env bash\n'
        'if [ "$2" = where ]; then echo "--base x"; fi\nexit 0\n',
    )
    done = subprocess.run(
        ['bash', '--noprofile', '--norc', '-e', '-c', step['run']],
        cwd=work,
        env={
            'PATH': f'{bin_dir}{os.pathsep}{os.environ["PATH"]}',
            'HOME': str(tmp_path),
            'GITHUB_EVENT_NAME': 'push',
            'GITHUB_REF_NAME': LANE,
            **{k: str(v) for k, v in job_env.items()},
        },
        capture_output=True,
        text=True,
        check=False,
    )
    if seed in ('comparator', 'renamed'):
        assert done.returncode == 0, done.stdout + done.stderr
        assert 'gate: REPORT-ONLY in floor-pandas2.2' in done.stdout
    else:
        assert done.returncode == 1, done.stdout + done.stderr
        assert 'REPORT-ONLY' not in done.stdout
