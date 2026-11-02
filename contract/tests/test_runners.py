# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Current runner failures cannot leave apparently valid observations."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
CASE = ROOT / 'contract/cases/s1/base.regression.json'


def test_a_process_abort_fails_the_runner(tmp_path):
    out = tmp_path / 'observations.jsonl'
    out.write_text('stale observations\n')
    script = f"""
import os, resource, runpy, sys
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
runner = runpy.run_path({str(RUNNER)!r}, run_name='run_python')
import sdm_client
sdm_client.PredictiveQuery.materialize = lambda *a, **k: os.abort()
sys.exit(runner['main']([
    '--mode', 'python-core', '--out', {str(out)!r}, {str(CASE)!r}
]))
"""
    result = subprocess.run(
        [sys.executable, '-I', '-c', script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode < 0, result.stderr
    assert not out.exists()


def test_an_input_the_harness_cannot_build_fails_the_runner(tmp_path):
    case = json.loads(CASE.read_text())
    case['input']['materialize']['anchor_time'] = '2024-13-45'
    path = tmp_path / 'case.json'
    path.write_text(json.dumps(case))
    out = tmp_path / 'observations.jsonl'
    out.write_text('stale observations\n')
    result = subprocess.run(
        [
            sys.executable,
            '-I',
            str(RUNNER),
            '--mode',
            'python-core',
            '--out',
            str(out),
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert 'its materialize arguments could not be built' in result.stderr
    assert not out.exists()
