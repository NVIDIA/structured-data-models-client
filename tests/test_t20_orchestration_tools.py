# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.0's acceptance tests for exits 8 and 9 (the independent test
author's): the watchdog fails loudly, and the Codex pane helper is in the
repository.

Exit 8: ``stuck_agents.py`` and ``rule_watch.py`` exit nonzero, with a line
on standard output, for a tasks directory that does not exist and for a
transcript they cannot read (a timestamp that is no time, bytes that are
not UTF-8). The proof is the stuck-agent loop itself: its text is read from
``docs/design/rust-core/orchestration/prompts.md`` ("Watchdog texts"),
whatever it greps, and one pass of it is run (the ``OUT=`` line, its
standard error dropped as the loop drops it) against each seeded
directory: it must fire. Controls: an empty directory and a sound
transcript with a fresh call leave the loop quiet and both tools exit 0.

Exit 9: ``codex_pane.py`` and ``codex_pane_selftest.py`` live in
``scripts/rust_core/orchestration/``, the self-test passes there (run from
that directory and from the repository root), the README's tools list
names the helper, and neither file holds an absolute home path, an
e-mail address but a placeholder's, an IP address or the running user's
name.

The interface assumed: none beyond the brief (the loop's text stays in
prompts.md's "Watchdog texts", in a ``bash`` block holding an ``OUT=$(``
line, with ``B`` and ``T=<TASKS>`` set before it).
"""

from __future__ import annotations

import getpass
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ORCH = ROOT / 'scripts/rust_core/orchestration'
PROMPTS = ROOT / 'docs/design/rust-core/orchestration/prompts.md'
TOOLS = ('stuck_agents.py', 'rule_watch.py')


# --- exit 8: the watchdog ----------------------------------------------------------------


def loop_pass() -> str:
    r"""One pass of the stuck-agent loop, from prompts.md: its settings and
    its ``OUT=`` line, then a line saying whether it fired."""
    text = PROMPTS.read_text(encoding='utf-8')
    section = text.split('### Watchdog texts', 1)[1]
    block = section.split('```bash', 1)[1].split('```', 1)[0]
    lines = block.strip().splitlines()
    settings = [x for x in lines if x.startswith('B=')]
    out = [x.strip() for x in lines if x.strip().startswith('OUT=$(')]
    assert len(settings) == 1 and len(out) == 1, block
    return '\n'.join(
        [settings[0], out[0], '[ -n "$OUT" ] && echo "FIRED: $OUT"', 'exit 0']
    )


def run_loop(tasks: Path) -> str:
    script = loop_pass().replace('<TASKS>', str(tasks))
    done = subprocess.run(
        ['bash', '--noprofile', '--norc', '-c', script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


def run_tool(tool: str, tasks: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ORCH / tool), str(tasks), '40'],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())


def call(
    stamp: str, command: str = 'git status', ident: str = 'toolu_1'
) -> str:
    return json.dumps(
        {
            'timestamp': stamp,
            'message': {
                'content': [
                    {
                        'type': 'tool_use',
                        'id': ident,
                        'name': 'Bash',
                        'input': {'command': command},
                    }
                ]
            },
        }
    )


def transcript(tasks: Path, body: bytes, name: str = 'a1.output') -> Path:
    tasks.mkdir(parents=True, exist_ok=True)
    path = tasks / name
    path.write_bytes(body)
    return path


SEEDS = {
    'a missing directory': lambda t: None,
    'a timestamp that is no time': lambda t: transcript(
        t, (call('not-a-time') + '\n').encode()
    ),
    'bytes that are not UTF-8': lambda t: transcript(
        t, (call(now()) + '\n').encode() + b'\xff\xfe\x80\n'
    ),
}


@pytest.mark.parametrize('seed', sorted(SEEDS))
def test_the_loop_fires_when_the_watchdog_cannot_watch(
    tmp_path: Path, seed: str
) -> None:
    r"""Exit 8: the loop as prompts.md has it fires on each seed."""
    tasks = tmp_path / 'tasks'
    SEEDS[seed](tasks)
    assert 'FIRED' in run_loop(tasks), seed


@pytest.mark.parametrize('tool', TOOLS)
@pytest.mark.parametrize('seed', sorted(SEEDS))
def test_each_tool_exits_nonzero_with_a_line_on_stdout(
    tmp_path: Path, seed: str, tool: str
) -> None:
    r"""Exit 8: nonzero, and a line on standard output (the loop drops
    standard error), naming what failed."""
    tasks = tmp_path / 'tasks'
    SEEDS[seed](tasks)
    done = run_tool(tool, tasks)
    assert done.returncode != 0, done.stdout + done.stderr
    assert done.stdout.strip(), done.stderr
    if seed == 'a missing directory':
        assert str(tasks) in done.stdout, done.stdout
    else:
        assert 'a1.output' in done.stdout, done.stdout


def test_the_loop_stays_quiet_on_an_empty_directory_and_a_sound_transcript(
    tmp_path: Path,
) -> None:
    r"""Controls: nothing to report, nothing fired, both tools exit 0
    (passes on the base)."""
    empty = tmp_path / 'empty'
    empty.mkdir()
    sound = tmp_path / 'sound'
    transcript(sound, (call(now()) + '\n').encode())
    for tasks in (empty, sound):
        assert 'FIRED' not in run_loop(tasks), tasks
        for tool in TOOLS:
            done = run_tool(tool, tasks)
            assert done.returncode == 0, done.stdout + done.stderr


def test_a_partial_last_line_is_not_a_failure(tmp_path: Path) -> None:
    r"""Control: a transcript being written ends in a partial JSON line,
    which both tools pass over, as they do now (passes on the base)."""
    tasks = tmp_path / 'tasks'
    transcript(tasks, (call(now()) + '\n{"timestamp": "20').encode())
    assert 'FIRED' not in run_loop(tasks)
    for tool in TOOLS:
        assert run_tool(tool, tasks).returncode == 0, tool


# --- exit 9: the Codex pane helper -------------------------------------------------------

PANE = ORCH / 'codex_pane.py'
PANE_SELFTEST = ORCH / 'codex_pane_selftest.py'


@pytest.mark.parametrize('cwd', [ORCH, ROOT], ids=['its directory', 'the root'])
def test_the_codex_pane_self_test_passes_in_the_repository(cwd: Path) -> None:
    r"""Exit 9: the helper and its self-test are here, and it passes."""
    assert PANE.is_file() and PANE_SELFTEST.is_file()
    done = subprocess.run(
        [sys.executable, str(PANE_SELFTEST)],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=600,
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]


def test_the_orchestration_readme_lists_the_codex_pane_helper() -> None:
    r"""Exit 9: the README's tools list names it."""
    readme = (ORCH / 'README.md').read_text(encoding='utf-8')
    tools = readme.split('## Tools', 1)[1]
    assert '`codex_pane.py`' in tools


#: What a private detail looks like in a file: a home directory, an e-mail
#: address other than a placeholder's, an IPv4 address.
PRIVATE = (
    re.compile(r'/home/[a-z]|/Users/[A-Za-z]'),
    re.compile(
        r'[\w.+-]+@(?!example\.(?:com|org|invalid)\b|noreply\b)[\w-]+(?:\.[\w-]+)+'
    ),
    re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
)


@pytest.mark.parametrize('path', [PANE, PANE_SELFTEST], ids=lambda p: p.name)
def test_the_codex_pane_files_hold_nothing_private(path: Path) -> None:
    r"""Exit 9: nothing private in the files moved into the public
    repository (terminal titles from herdr can carry user and host
    names; nothing herdr prints may be written here)."""
    text = path.read_text(encoding='utf-8')
    found = [m.group(0) for p in PRIVATE for m in p.finditer(text)]
    assert found == [], found
    user = getpass.getuser()
    if len(user) >= 4 and user not in ('root', 'runner'):
        assert user not in text, 'the running user is named'
