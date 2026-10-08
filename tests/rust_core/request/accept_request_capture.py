# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P85.capture: ``tests/parity/requests/`` is what
``_request.py capture --out DIR`` writes, under ``PYTHONHASHSEED`` 1 and
4242: one recording per transcript, and nothing else (``CONTRACT.md``,
"Recordings")."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from _request import (
    CALLS_FILE,
    HERE,
    RECORDINGS,
    REPO,
    TRANSCRIPTS,
    load_json,
    native,
    recording_dirs,
    scenario_dir,
    transcripts,
)


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob('*'))
        if path.is_file()
    }


def test_one_recording_per_transcript() -> None:
    assert RECORDINGS.is_dir(), f'{RECORDINGS.relative_to(REPO)} does not exist'
    names = {load_json(p)['name']: p for p in transcripts()}
    found = {}
    for directory in recording_dirs():
        document = load_json(directory / CALLS_FILE)
        found[document['scenario']] = directory
        assert document.get('format') == 1, directory
    assert sorted(found) == sorted(names), (
        f'recordings without a transcript: {sorted(set(found) - set(names))}; '
        f'transcripts without a recording: {sorted(set(names) - set(found))}'
    )
    for name, directory in found.items():
        assert directory == scenario_dir(RECORDINGS, name), directory
        document = load_json(directory / CALLS_FILE)
        want = names[name].parent.relative_to(REPO).as_posix()
        assert document['transcript'] == want, (name, document['transcript'])
        assert (REPO / want).is_relative_to(TRANSCRIPTS)


def test_capture_writes_the_recordings(tmp_path: Path) -> None:
    native()
    stored = _files(RECORDINGS)
    assert stored, 'no recordings are stored'
    for seed in ('1', '4242'):
        out = tmp_path / f'seed-{seed}'
        env = dict(os.environ)
        env['PYTHONHASHSEED'] = seed
        env['SDM_P7_WHEEL_INSTALLED'] = '1'
        proc = subprocess.run(
            [
                sys.executable,
                str(HERE / '_request.py'),
                'capture',
                '--out',
                str(out),
            ],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, (
            f'capture under PYTHONHASHSEED={seed} failed:\n'
            f'{proc.stdout[-4000:]}\n{proc.stderr[-6000:]}'
        )
        written = _files(out)
        missing = sorted(set(stored) - set(written))
        extra = sorted(set(written) - set(stored))
        changed = sorted(
            name
            for name in set(stored) & set(written)
            if stored[name] != written[name]
        )
        assert not (missing or extra or changed), (
            f'PYTHONHASHSEED={seed}: {len(missing)} stored files not written '
            f'({missing[:5]}), {len(extra)} written files not stored '
            f'({extra[:5]}), {len(changed)} differ ({changed[:5]})'
        )
