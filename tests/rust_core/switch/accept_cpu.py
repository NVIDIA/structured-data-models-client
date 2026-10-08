# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P9.perf-cpu: replaying the corpus offline takes no more CPU time on the
core than on the old path, by the median of seven interleaved rounds.

Each round replays every transcript once on each path, each path in a fresh
process, in alternating order; a process's CPU time (``time.process_time``)
counts from before the first transcript to after the last, so the imports
each path makes on the way are counted. Every replay must pass, and the core
processes must load neither ported package.

Both paths leave out the transcripts a brief task 7b fix re-recorded with the
core's answer (``FIXED``), which the old path cannot replay as stored by
design. They must be exactly the transcripts ``fix_changes.json`` lists, so
P9.shadow compares each between the paths, as a fix that must differ and
must replay as stored on the core.
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
from pathlib import Path

from _switch import (
    CORE,
    HERE,
    OLD,
    REPO,
    env_for,
    fix_changes,
    install_core,
    run,
    transcript_name,
    transcripts,
)

ROUNDS = 7
FIXED = (
    'duckdb/regression#decimal-entity-keys',
    'memory/link-prediction#multilabel',
)


def _once(impl: str) -> dict:
    with tempfile.TemporaryDirectory(prefix='sdm-p9-cpu-') as tmp:
        out = Path(tmp) / 'cpu.json'
        proc = run(
            [
                sys.executable,
                HERE / '_switch.py',
                'cpu',
                '--impl',
                impl,
                '--out',
                out,
                *(f'--skip={name}' for name in FIXED),
            ],
            cwd=REPO,
            env=env_for(impl, PYTHONHASHSEED='0'),
            timeout=1800,
        )
        assert proc.returncode == 0 and out.is_file(), (
            f'{impl} replay failed (exit {proc.returncode}):\n{proc.stderr[-4000:]}'
        )
        return json.loads(out.read_text())


def test_the_left_out_transcripts_are_the_listed_fixes():
    names = {transcript_name(path) for path in transcripts()}
    listed = {name for name in fix_changes() if name in names}
    assert set(FIXED) == listed, (sorted(FIXED), sorted(listed))


def test_the_core_replays_in_no_more_cpu_time_than_python():
    install_core()
    every = len(transcripts())
    spent = {OLD: [], CORE: []}
    for round_ in range(ROUNDS):
        order = (OLD, CORE) if round_ % 2 == 0 else (CORE, OLD)
        for impl in order:
            data = _once(impl)
            assert data['skipped'] == sorted(FIXED), (impl, data['skipped'])
            assert data['transcripts'] == every - len(FIXED), (
                impl,
                data['transcripts'],
                every,
            )
            assert not data['failed'], (impl, data['failed'][:10])
            if impl == CORE:
                assert data['native_loaded'] and not data['first_imports'], (
                    'the core replay did not run on the core: '
                    f'{data["first_imports"]}'
                )
            spent[impl].append(data['cpu'])
    old, core = statistics.median(spent[OLD]), statistics.median(spent[CORE])
    print(f'median CPU seconds: python {old:.3f}, core {core:.3f}; {spent}')
    assert core <= old, (
        f'the core took {core:.3f} s of CPU, the old path {old:.3f} s '
        f'(medians of {ROUNDS}): {spent}'
    )
