# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expose existing sampling boundary recordings through the corpus tools."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from functools import cache
from pathlib import Path

from _parity import PARITY, REPO

RECORDINGS = PARITY / 'sampling'
DRIVER = REPO / 'tests/rust_core/sampling/driver'


@cache
def driver():
    sys.path.insert(0, str(DRIVER))
    spec = importlib.util.spec_from_file_location(
        '_parity_sampling_driver', DRIVER / '_driver.py'
    )
    if spec is None or spec.loader is None:
        raise ImportError('sampling corpus driver is unavailable')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stored(root: Path) -> dict[str, Path]:
    found = {}
    for directory in driver().recording_dirs(root):
        recording = driver().read_recording(directory)
        name = recording['scenario']
        if not isinstance(name, str) or not name or name in found:
            raise ValueError(
                'sampling recordings require unique scenario names'
            )
        found[name] = directory
    return found


def listing(root: Path) -> list[dict]:
    return [
        {
            'name': name,
            'status': 'recorded',
            'kind': 'sampling',
            'calls': len(driver().read_recording(directory)['calls']),
            'recording': (directory / 'boundary.json')
            .relative_to(root)
            .as_posix(),
        }
        for name, directory in sorted(stored(root).items())
    ]


def compare(recording: dict, entries: list | None) -> str | None:
    calls = recording['calls']
    if not isinstance(entries, list) or len(entries) != len(calls):
        return 'recorded calls were not all driven'
    for ordinal, (call, entry) in enumerate(zip(calls, entries, strict=True)):
        if entry.get('sql') is not None:
            return f'call {ordinal}: {entry["sql"]}'
        if entry.get('queries') != len(call['steps']):
            return f'call {ordinal}: recorded query count differs'
        if entry.get('outputs') is None:
            return f'call {ordinal}: no output'
        difference = driver().difference(call['outputs'], entry['outputs'])
        if difference is not None:
            return f'call {ordinal}: {difference}'
    return None


def replay(
    root: Path, selected: list[str] | None
) -> list[tuple[str, str | None]]:
    recordings = stored(root)
    names = selected if selected is not None else sorted(recordings)
    if not names:
        raise ValueError('no sampling boundary recordings')
    if len(names) != len(set(names)):
        raise ValueError('duplicate sampling scenario selection')
    missing = set(names) - set(recordings)
    if missing:
        raise ValueError('no sampling recording: ' + ', '.join(sorted(missing)))
    with tempfile.TemporaryDirectory(prefix='parity-sampling-') as temporary:
        scratch = Path(temporary)
        inputs = scratch / 'recordings'
        for ordinal, name in enumerate(names):
            shutil.copytree(recordings[name], inputs / str(ordinal))
        output = scratch / 'actual.json'
        try:
            process = subprocess.run(
                [
                    sys.executable,
                    str(DRIVER / '_driver_run.py'),
                    str(inputs),
                    str(output),
                ],
                cwd=REPO,
                env={**os.environ, 'SDM_P7_WHEEL_INSTALLED': '1'},
                capture_output=True,
                text=True,
                timeout=7200,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError('sampling driver timed out') from error
        if process.returncode:
            detail = process.stderr.strip() or process.stdout.strip()
            raise RuntimeError('sampling driver failed: ' + detail[-3000:])
        actual = json.loads(output.read_bytes())
        if set(actual) != set(names):
            raise ValueError('sampling driver returned different scenarios')
        return [
            (
                name,
                compare(
                    driver().read_recording(recordings[name]), actual[name]
                ),
            )
            for name in names
        ]
