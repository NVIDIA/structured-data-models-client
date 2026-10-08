# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P85.independent: every recording replays through the binding in a
process where pandas, dateutil, pytz and the packages cannot be imported,
so the core answers by itself (``CONTRACT.md``, "Recordings")."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from _request import HERE, RECORDINGS, REPO, native, recording_dirs, transcripts


def test_every_recording_replays_on_its_own() -> None:
    native()
    dirs = recording_dirs()
    assert len(dirs) == len(transcripts()), (
        f'{len(dirs)} recordings for {len(transcripts())} transcripts'
    )
    env = dict(os.environ)
    env['SDM_P7_WHEEL_INSTALLED'] = '1'
    proc = subprocess.run(
        [sys.executable, str(HERE / '_redrive.py'), '--root', str(RECORDINGS)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith('{')]
    assert lines, f'no summary:\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}'
    summary = json.loads(lines[-1])
    assert summary['recordings'] == len(dirs), summary['recordings']
    assert summary['calls'] >= len(dirs), summary['calls']
    assert not summary['problems'], (
        f'{len(summary["problems"])} problem(s):\n'
        + '\n'.join(summary['problems'][:40])
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
