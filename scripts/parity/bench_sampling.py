#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Time the Rust sampling runs over the boundary recordings.

    bench_sampling.py [--out PATH] [--repeat N] [--only NAME[,NAME...]]

Every call of every recording under ``tests/parity/sampling/`` is driven
through the installed ``sdm_core._native`` from its recording, as the
acceptance checks drive it: its run started from the recorded inputs and
tables, each query answered from the recorded statements. For each scenario
it writes, as ``tests/rust_core/sampling/driver/CONTRACT.md`` ("Performance")
describes:

- ``queries``: the statements the scenario runs with the core's runs
  standing in for today's calls, the transcript's ``execute`` steps (each
  call's run asks exactly its recorded statements, checked here);
- per call, its ``method``, the ``queries`` its run asked, and
  ``rust_seconds``, the best wall time of ``--repeat`` runs from start to
  ``done``.

Without ``--out`` it writes ``scripts/parity/sampling_baseline.json``, the
baseline later phases compare with. Timings are of this machine; the counts
are the same on every one. ``--only`` times just the named scenarios and
replaces their entries in the record already at ``--out``, so a re-recorded
scenario leaves every other timing as it was.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DRIVER = REPO / 'tests' / 'rust_core' / 'sampling' / 'driver'
sys.path.insert(0, str(DRIVER))

from _driver import (  # noqa: E402
    BASELINE,
    TRANSCRIPTS,
    drive,
    read_arrow,
    read_recording,
    recorded_tables,
    recording_dirs,
    scenario_dir,
)


def _time_call(module, directory: Path, call: dict) -> tuple[int, float]:
    r"""The queries one run of ``call`` asks, and its wall time."""
    steps = call['steps']

    def result_of(i: int) -> object:
        file = steps[i]['result']
        return None if file is None else read_arrow(directory / file)

    tables = recorded_tables(directory, call)
    inputs = json.dumps(call['inputs'], ensure_ascii=False)
    started = time.perf_counter()
    run = getattr(module, call['method'])(inputs, tables)
    try:
        _, asked = drive(module, run, steps, result_of)
    except Exception as error:  # noqa: BLE001 - a recorded failure is timed too
        if 'error' not in call['outputs']:
            raise
        del error
        asked = steps
    return len(asked), time.perf_counter() - started


def bench(repeat: int, only: frozenset[str] | None = None) -> dict:
    import sdm_core._native as module

    scenarios: dict[str, dict] = {}
    for directory in recording_dirs():
        recording = read_recording(directory)
        name = recording['scenario']
        if only is not None and name not in only:
            continue
        transcript = json.loads(
            (scenario_dir(TRANSCRIPTS, name) / 'transcript.json').read_text(
                'utf-8'
            )
        )
        calls = []
        for call in recording['calls']:
            best = None
            asked = 0
            for _ in range(max(repeat, 1)):
                asked, seconds = _time_call(module, directory, call)
                best = seconds if best is None else min(best, seconds)
            calls.append(
                {
                    'method': call['method'],
                    'queries': asked,
                    'rust_seconds': best,
                }
            )
        scenarios[name] = {
            'queries': sum(
                1 for step in transcript['steps'] if step['op'] == 'execute'
            ),
            'calls': calls,
        }
    return {'scenarios': scenarios}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--out', default=str(BASELINE))
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--only', help='comma-separated scenario names')
    args = parser.parse_args(argv)
    if args.only is None:
        record = bench(args.repeat)
    else:
        only = frozenset(name for name in args.only.split(',') if name)
        timed = bench(args.repeat, only)['scenarios']
        missing = sorted(only - set(timed))
        if missing:
            parser.error(f'no recording for {", ".join(missing)}')
        record = json.loads(Path(args.out).read_text(encoding='utf-8'))
        record['scenarios'].update(timed)
    Path(args.out).write_text(
        json.dumps(record, indent=1, sort_keys=True) + '\n', encoding='utf-8'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
