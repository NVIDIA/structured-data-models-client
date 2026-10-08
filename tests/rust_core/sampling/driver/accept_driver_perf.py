# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P75.perf: query counts identical to today's for every scenario, and the
Rust timings recorded.

- With the core's runs standing in for today's calls, each call asks as many
  queries as today's call ran statements, and each scenario runs as many
  statements as its transcript holds (``_driver_run.py``).
- ``bench_sampling.py --out`` writes the performance record of
  ``CONTRACT.md`` ("Performance"), with those counts and a positive Rust
  time for every call of an in-memory scenario; so does the committed
  ``scripts/parity/sampling_baseline.json``.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from _driver import (
    BASELINE,
    BENCH,
    METHODS,
    REPO,
    TRANSCRIPTS,
    native,
    read_recording,
    recording_dirs,
    scenario_dir,
)
from accept_driver_replay import driven


def _counts() -> dict[str, dict[str, Any]]:
    r"""By scenario: its transcript's ``execute`` steps, its backend, and
    each recorded call's method and statements."""
    out = {}
    for directory in recording_dirs():
        recording = read_recording(directory)
        name = recording['scenario']
        transcript = json.loads(
            (scenario_dir(TRANSCRIPTS, name) / 'transcript.json').read_text(
                'utf-8'
            )
        )
        out[name] = {
            'backend': transcript['backend'],
            'queries': sum(
                1 for step in transcript['steps'] if step['op'] == 'execute'
            ),
            'calls': [
                (call['method'], len(call['steps']))
                for call in recording['calls']
            ],
        }
    assert len(out) >= 71, f'only {len(out)} recordings'
    return out


def test_the_core_asks_as_many_queries() -> None:
    counts = _counts()
    got = driven()
    problems = []
    for name, want in counts.items():
        entries = got.get(name, [])
        asked = [
            entry['queries'] if entry['sql'] is None else None
            for entry in entries
        ]
        recorded = [steps for _, steps in want['calls']]
        if asked != recorded:
            problems.append(
                f'{name}: the runs asked {asked} queries, today ran {recorded}'
            )
    assert not problems, '\n'.join(problems[:25])


def _check_record(record: Any, what: str) -> None:
    counts = _counts()
    assert isinstance(record, dict) and isinstance(
        record.get('scenarios'), dict
    ), f'{what}: not {{"scenarios": {{...}}}}'
    scenarios = record['scenarios']
    missing = sorted(set(counts) - set(scenarios))
    assert not missing, f'{what}: no entry for {missing[:10]}'
    problems = []
    timed = 0
    for name, want in counts.items():
        entry = scenarios[name]
        if entry.get('queries') != want['queries']:
            problems.append(
                f'{name}: {entry.get("queries")} queries, the transcript '
                f'runs {want["queries"]}'
            )
        calls = entry.get('calls')
        if not isinstance(calls, list) or len(calls) != len(want['calls']):
            problems.append(f'{name}: the calls are not the recorded ones')
            continue
        for i, (call, (method, steps)) in enumerate(
            zip(calls, want['calls'], strict=True)
        ):
            if call.get('method') != method or call.get('queries') != steps:
                problems.append(
                    f'{name} call {i}: {call.get("method")} asking '
                    f'{call.get("queries")}, recorded {method} running {steps}'
                )
            seconds = call.get('rust_seconds')
            if want['backend'] == 'memory':
                if not (
                    isinstance(seconds, (int, float))
                    and not isinstance(seconds, bool)
                    and math.isfinite(seconds)
                    and seconds > 0
                ):
                    problems.append(
                        f'{name} call {i}: rust_seconds {seconds!r}'
                    )
                else:
                    timed += 1
            elif seconds is not None and not isinstance(seconds, (int, float)):
                problems.append(f'{name} call {i}: rust_seconds {seconds!r}')
    assert not problems, f'{what}:\n' + '\n'.join(problems[:25])
    assert timed >= 20, f'{what}: only {timed} in-memory calls timed'


def test_the_committed_baseline() -> None:
    assert BASELINE.is_file(), f'{BASELINE} does not exist'
    _check_record(json.loads(BASELINE.read_text('utf-8')), BASELINE.name)


def test_the_bench_writes_the_record() -> None:
    assert BENCH.is_file(), f'{BENCH} does not exist'
    native(*METHODS, 'Outcome')
    with tempfile.TemporaryDirectory(prefix='sdm-p75-bench-') as tmp:
        out = Path(tmp) / 'bench.json'
        proc = subprocess.run(
            [sys.executable, str(BENCH), '--out', str(out)],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        assert proc.returncode == 0, (
            f'bench_sampling.py failed ({proc.returncode}):\n'
            f'{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}'
        )
        _check_record(json.loads(out.read_text('utf-8')), 'bench_sampling.py')
