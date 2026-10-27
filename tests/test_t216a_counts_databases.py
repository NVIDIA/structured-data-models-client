# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.16a fix 3 (the Claude review of round 1, P3): ``heldout.py counts``
learnt a case's legs from the observation files alone, so a database no
runner wrote, python-ref included, was never counted. With the run's
``--databases`` (``heldout_run.sh`` passes its own), a missing reference
observation fails closed before counts; a reference-observed leg no core
runner wrote is a panic, so a no-go.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELDOUT = ROOT / 'scripts/rust_core/heldout.py'


def _counts(run: Path, *extra: str) -> tuple[int, str]:
    counts = run / 'counts.md'
    counts.unlink(missing_ok=True)
    done = subprocess.run(
        [sys.executable, '-I', str(HELDOUT), 'counts', str(run),
         '--prefix', 'h9.', '--threshold', '4', *extra],
        capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    return done.returncode, '\n'.join(
        text
        for text in (
            counts.read_text() if counts.is_file() else '',
            done.stdout,
            done.stderr,
        )
        if text
    )


def _duckdb_run(tmp_path: Path) -> Path:
    run = tmp_path / 'run'
    (run / 'cases').mkdir(parents=True)
    (run / 'cases' / 'h9.one.json').write_text(json.dumps({'id': 'h9.one'}))
    ok = {'result': 'success', 'stage': 'predict'}
    for runner in ('python-ref', 'python-core', 'python-old', 'rust-host'):
        line = {'case': 'h9.one', 'identity': {'runner': runner}, 'outcome': ok}
        (run / f'{runner}.jsonl').write_text(json.dumps(line) + '\n')
    return run


def test_a_database_no_runner_wrote_is_counted_when_named(tmp_path: Path):
    run = _duckdb_run(tmp_path)
    code, text = _counts(run)
    assert code == 0, text
    code, text = _counts(run, '--databases', 'duckdb,sqlite')
    assert code == 1, text
    assert 'a required reference observation is missing' in text
    assert not (run / 'counts.md').exists()


def test_a_database_only_the_reference_wrote_is_counted_when_named(
    tmp_path: Path,
):
    run = _duckdb_run(tmp_path)
    reference = {
        'case': 'h9.one@sqlite',
        'identity': {'runner': 'python-ref'},
        'outcome': {'result': 'success', 'stage': 'predict'},
    }
    with (run / 'python-ref.jsonl').open('a') as handle:
        handle.write(json.dumps(reference) + '\n')
    code, text = _counts(run, '--databases', 'duckdb,sqlite')
    assert code == 1, text
    assert 'cases: 1; legs: 2 (duckdb 1, sqlite 1)' in text
    assert "panics: 1 ['h9.one@sqlite']" in text
    assert 'counts: NO-GO' in text
