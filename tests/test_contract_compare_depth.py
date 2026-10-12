# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The comparator's reading of observations, outside JSON (E49's depth and
width probe, T1.15): a line nested 100,000 deep or 100,000 wide is refused
by name as a runner failure, never a traceback."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

LINES = {
    'deep': '[' * 100_000 + ']' * 100_000,
    'deep object': '{"a":' * 100_000 + '1' + '}' * 100_000,
    'wide': json.dumps({f'k{i}': i for i in range(100_000)}),
}


@pytest.mark.parametrize('shape', sorted(LINES))
def test_an_observation_past_the_parser_is_refused_by_name(
    tmp_path: Path, shape: str
) -> None:
    path = tmp_path / 'python-core.jsonl'
    path.write_text(LINES[shape] + '\n')
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'scripts/rust_core/compare.py'),
            '--cases',
            str(ROOT / 'contract/cases'),
            '--reference',
            *map(str, sorted((ROOT / 'contract/reference').glob('*.jsonl'))),
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 1, done.stdout + done.stderr
    assert 'Traceback' not in done.stderr, done.stderr
    assert 'python-core.jsonl:1: runner failure' in done.stdout, done.stdout
