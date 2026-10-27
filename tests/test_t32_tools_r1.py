# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T3.2 fix 5, Codex round 1, findings 7 to 9:

- (the warehouse census through the stand-in's forwarding provider went
  with the stand-in, B-lite, and its test with it);
- ``clean_tmp.sh`` keeps a directory a process holds a file open in, or
  names as its ``CARGO_TARGET_DIR`` or ``TMPDIR`` (an active lane's), and
  still lists a stale one;
- ``cells.sh`` installs the engine's test extras and Snowflake's driver in
  every cell, and the recipe is part of the environment's key.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

# It copies the checkout, so CI runs it in the tooling jobs (T3.0).
pytestmark = pytest.mark.tooling

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / 'scripts/rust_core/orchestration/clean_tmp.sh'
CELLS = ROOT / 'scripts/rust_core/cells.sh'


def _census() -> ModuleType:
    name = 't32_r1_census'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / 'contract/tools/refusal_census.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --- 8. clean_tmp.sh keeps what a process holds ---------------------------------------


def test_clean_tmp_keeps_a_held_file_s_and_an_active_lane_s_directories(
    tmp_path: Path,
) -> None:
    roots = {
        name: tmp_path / name for name in ('tmp', 'target', 'cells', 'state')
    }
    for path in roots.values():
        path.mkdir()
    held, lane, stale = (
        roots['tmp'] / 'held-open',
        roots['target'] / 'lane-x',
        roots['tmp'] / 'old',
    )
    for path in (held, lane, stale):
        path.mkdir()
        (path / 'file').write_text('x\n', encoding='utf-8')
        for each in (path / 'file', path):
            os.utime(each, (0, 0))
    # One process holds a file open in one directory; another names the
    # lane's target as its CARGO_TARGET_DIR, idle between builds.
    opener = subprocess.Popen(
        [sys.executable, '-c', 'import sys, time; f = open(sys.argv[1]); time.sleep(120)',
         str(held / 'file')],
    )  # fmt: skip
    builder = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(120)'],
        env={**os.environ, 'CARGO_TARGET_DIR': str(lane)},
        cwd=tmp_path,
    )
    try:
        # Both processes are up and holding before the dry run looks.
        for _ in range(100):
            if any(
                os.path.realpath(f'/proc/{opener.pid}/fd/{fd}')
                == str(held / 'file')
                for fd in os.listdir(f'/proc/{opener.pid}/fd')
            ):
                break
            subprocess.run(['sleep', '0.05'], check=True)
        done = subprocess.run(
            ['bash', str(CLEAN), '--dry-run'],
            capture_output=True,
            text=True,
            timeout=300,
            stdin=subprocess.DEVNULL,
            check=False,
            env={
                **os.environ,
                'SDM_TMP_ROOT': str(roots['tmp']),
                'SDM_TARGET_ROOT': str(roots['target']),
                'SDM_CELLS_ROOT': str(roots['cells']),
                'SDM_ORCH_STATE': str(roots['state']),
                'SDM_SCRATCH': '',
            },
        )
    finally:
        for process in (opener, builder):
            process.kill()
            process.wait(timeout=30)
    said = done.stdout + done.stderr
    assert f'skip in use {held}' in done.stdout, said
    assert f'skip in use {lane}' in done.stdout, said
    assert f'would remove {held}' not in done.stdout, said
    assert f'would remove {lane}' not in done.stdout, said
    assert f'would remove {stale}' in done.stdout, said


# --- 9. cells.sh's recipe ---------------------------------------------------------------


def _key(repo: Path) -> str:
    done = subprocess.run(
        ['bash', str(repo / 'scripts/rust_core/cells.sh'), '--print-key'],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
        check=True,
    )
    return done.stdout.strip()


def test_every_cell_installs_the_engine_s_tests_and_snowflake_and_keys_on_it(
    tmp_path: Path,
) -> None:
    text = CELLS.read_text(encoding='utf-8')
    assert 'ENGINE_EXTRAS=test\n' in text
    assert 'structured-data-models-relational-engine[$ENGINE_EXTRAS] @' in text
    assert '    "$SNOWFLAKE" \\\n' in text
    # A repository holding what the key reads, so cells.sh runs there.
    repo = tmp_path / 'repo'
    for name in (
        'scripts/rust_core/cells.sh',
        'scripts/rust_core/native_wheel.py',
        '.github/workflows/ci.yml',
        'uv.lock',
        'rust/Cargo.toml',
        'packages/structured-data-models-api/pyproject.toml',
        'packages/structured-data-models-connectors/pyproject.toml',
    ):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, repo / name)
    env = {
        **os.environ,
        'GIT_AUTHOR_NAME': 't',
        'GIT_AUTHOR_EMAIL': 't@example.invalid',
        'GIT_COMMITTER_NAME': 't',
        'GIT_COMMITTER_EMAIL': 't@example.invalid',
        'GIT_CONFIG_GLOBAL': os.devnull,
        'GIT_CONFIG_NOSYSTEM': '1',
    }
    for args in (['init', '-q'], ['add', '-A'], ['commit', '-q', '-m', 'x']):
        subprocess.run(['git', *args], cwd=repo, env=env, check=True)
    before = _key(repo)
    script = repo / 'scripts/rust_core/cells.sh'
    script.write_text(
        text.replace('ENGINE_EXTRAS=test\n', 'ENGINE_EXTRAS=test,explain\n'),
        encoding='utf-8',
    )
    # Another recipe, another environment: every cell is rebuilt.
    assert _key(repo) != before


# --- the census's control (Claude r1) --------------------------------------------------


@pytest.mark.parametrize(
    ('error', 'counted'),
    [
        ('StructuredDataError', True),
        ('ImportError', False),
        ('MissingExtraError', False),
    ],
)
def test_the_control_counts_only_a_refusal_of_the_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: str, counted: bool
) -> None:
    import json

    census = _census()
    case = json.loads((ROOT / 'contract/cases/s1/base.binary.json').read_text())
    cases = tmp_path / 'cases'
    cases.mkdir()
    (cases / 'base.binary.json').write_text(json.dumps(case))
    control = tmp_path / 'control.json'
    control.write_text(json.dumps({**case, 'id': 't32.control'}))
    refused = {'result': 'refusal', 'stage': 'from_source', 'error': error,
               'code': 'INVALID_REQUEST', 'rule': None}  # fmt: skip
    monkeypatch.setattr(
        census,
        '_real_outcome',
        lambda case, database: (
            refused
            if case['id'] == 't32.control'
            else {'result': 'success', 'stage': 'predict'}
        ),
    )
    if counted:
        made = census.real_data(cases, 'duckdb', control)
        assert made['control']['outcome'] == refused
    else:
        with pytest.raises(census.CensusError, match=error):
            census.real_data(cases, 'duckdb', control)
