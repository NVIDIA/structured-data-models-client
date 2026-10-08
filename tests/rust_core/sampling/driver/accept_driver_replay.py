# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P75.replay and P75.boundary: the core's sampling runs, started from the
boundary recordings alone, ask for exactly the recorded statements and give
exactly the recorded outputs.

``_driver_run.py`` drives every recorded call in a process where pandas,
dateutil, pytz, torch, the engine, the client, ``predictive_query`` and the
connectors can't be imported, so the core answers by itself.

- ``test_statements`` (P75.replay): every call's run asks, in order, for
  the recorded statements, SQL byte for byte and parameters encoded alike,
  and for no other. Since the recorded statements are the transcript's
  (P75.capture), every corpus scenario issues its recorded SQL.
- ``test_outputs`` (P75.boundary): every call's ``done`` value equals the
  recorded outputs in canonical form: frames by Arrow value and type, the
  batch, row, col and count arrays and anchor times exactly, labels bit for
  bit as ``float32``; where today's call raised, the run raises alike.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from functools import cache
from pathlib import Path
from typing import Any

from _driver import (
    HERE,
    METHODS,
    RECORDINGS,
    REPO,
    difference,
    native,
    read_recording,
    recording_dirs,
)
from _sampling import INSTALLED


def run_core(hashseed: str | None = None) -> dict[str, list[dict[str, Any]]]:
    r"""Every recorded call driven through the core, in a process without
    today's Python (``_driver_run.py``)."""
    native(*METHODS, 'Outcome')
    env = dict(os.environ)
    env[INSTALLED] = '1'
    if hashseed is not None:
        env['PYTHONHASHSEED'] = hashseed
    with tempfile.TemporaryDirectory(prefix='sdm-p75-core-') as tmp:
        out = Path(tmp) / 'out.json'
        proc = subprocess.run(
            [
                sys.executable,
                str(HERE / '_driver_run.py'),
                str(RECORDINGS),
                out,
            ],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,
        )
        assert proc.returncode == 0, (
            f'_driver_run.py failed ({proc.returncode}):\n'
            f'{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}'
        )
        return json.loads(out.read_text('utf-8'))


@cache
def driven() -> dict[str, list[dict[str, Any]]]:
    return run_core()


def recordings() -> dict[str, dict[str, Any]]:
    out = {}
    for directory in recording_dirs():
        recording = read_recording(directory)
        out[recording['scenario']] = recording
    assert len(out) >= 71, f'only {len(out)} recordings'
    return out


def test_statements() -> None:
    recorded = recordings()
    got = driven()
    problems = []
    runs = statements = 0
    for name, recording in recorded.items():
        entries = got.get(name)
        if entries is None or len(entries) != len(recording['calls']):
            problems.append(f'{name}: the calls were not all driven')
            continue
        for i, (call, entry) in enumerate(
            zip(recording['calls'], entries, strict=True)
        ):
            outputs = entry['outputs'] or {}
            if entry['sql'] is not None:
                problems.append(
                    f'{name} call {i} ({call["method"]}): {entry["sql"]}'
                )
            elif 'unreadable' in outputs:
                problems.append(
                    f'{name} call {i} ({call["method"]}): {outputs["unreadable"]}'
                )
            elif 'error' in outputs and 'error' not in call['outputs']:
                problems.append(
                    f'{name} call {i} ({call["method"]}): the run raised '
                    f'{outputs["error"]} after {entry["queries"]} of '
                    f'{len(call["steps"])} statements'
                )
            elif entry['queries'] != len(call['steps']):
                problems.append(
                    f'{name} call {i}: {entry["queries"]} queries for '
                    f'{len(call["steps"])} statements'
                )
            else:
                runs += 1
                statements += entry['queries']
    assert not problems, (
        f'{len(problems)} calls did not ask for their recorded statements:\n'
        + '\n'.join(problems[:25])
    )
    assert runs >= 230 and statements >= 300, (
        f'only {runs} runs and {statements} statements compared'
    )


def test_outputs() -> None:
    recorded = recordings()
    got = driven()
    problems = []
    compared = 0
    for name, recording in recorded.items():
        entries = got.get(name) or []
        if len(entries) != len(recording['calls']):
            problems.append(f'{name}: the calls were not all driven')
            continue
        for i, (call, entry) in enumerate(
            zip(recording['calls'], entries, strict=True)
        ):
            where = f'{name} call {i} ({call["method"]})'
            if entry['outputs'] is None:
                problems.append(f'{where}: no output, {entry["sql"]}')
                continue
            found = difference(call['outputs'], entry['outputs'])
            if found:
                problems.append(f'{where}: {found}')
            elif 'error' not in call['outputs']:
                compared += 1
    assert not problems, (
        f"{len(problems)} calls gave other outputs than today's Python:\n"
        + '\n'.join(problems[:25])
    )
    assert compared >= 230, f'only {compared} outputs compared'
