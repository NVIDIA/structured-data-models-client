# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P9.replay: ``run_corpus.py replay`` passes every transcript on the
switched code.

The command runs as a user's process would, the selector unset, and must
pass every transcript stored. The same replay, made by the core-path capture
(``_switch.py``), shows the run was the core's: the binding was loaded and
neither ported package was.
"""

from __future__ import annotations

import re
import sys

from _switch import (
    CORE,
    REPO,
    SCRIPTS,
    capture,
    core_in_use,
    env_for,
    install_core,
    run,
    transcripts,
)


def test_run_corpus_replay_passes_every_transcript():
    install_core()
    proc = run(
        [sys.executable, SCRIPTS / 'run_corpus.py', 'replay'],
        cwd=REPO,
        env=env_for(CORE),
        timeout=3600,
    )
    failed = [x for x in proc.stdout.splitlines() if x.startswith('FAIL ')]
    passed = [x for x in proc.stdout.splitlines() if x.startswith('PASS ')]
    assert proc.returncode == 0 and not failed, (
        f'exit {proc.returncode}; {len(failed)} failed:\n'
        + '\n'.join(failed[:40])
        + f'\n{proc.stderr[-3000:]}'
    )
    stored = len(transcripts())
    assert len(passed) == stored >= 137, (len(passed), stored)
    assert re.search(rf'\b{stored} passed, 0 failed', proc.stderr), proc.stderr[
        -500:
    ]


def test_the_replay_ran_on_the_core():
    data = capture(CORE, 'transcripts')
    failing = [
        f'{r["name"]}: {r["check"]}' for r in data['records'] if r['check']
    ]
    assert not failing, '\n'.join(failing[:40])
    problems = core_in_use(data)
    assert not problems, '\n'.join(problems)
