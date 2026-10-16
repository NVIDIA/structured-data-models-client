# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The watchdogs, ``scripts/rust_core/orchestration/stuck_agents.py`` and
``rule_watch.py``, fail loudly when they cannot see: a missing tasks
directory, an unreadable transcript or a line in one that is not JSON each
print a ``WATCHDOG-ERROR`` line and exit non-zero, and the loop text in
prompts.md ("Watchdog texts"), run as written, keeps that line. A partial
last line, which an agent is still writing, is not an error, and the
watchdogs still report what they exist for."""

from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'scripts/rust_core/orchestration'
PROMPTS = ROOT / 'docs/design/rust-core/orchestration/prompts.md'
WATCHDOGS = ('stuck_agents.py', 'rule_watch.py')


def _run(tool: str, tasks: Path, minutes: str, tmp_path: Path):
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith('SDM_')},
        'SDM_ORCH_STATE': str(tmp_path / 'state'),
    }
    return subprocess.run(
        [sys.executable, str(TOOLS / tool), str(tasks), minutes],
        capture_output=True,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )


def _stamp(minutes_ago: float) -> str:
    when = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        minutes=minutes_ago
    )
    return when.isoformat().replace('+00:00', 'Z')


def _call(command: str, minutes_ago: float, call_id: str = 't1') -> str:
    return json.dumps(
        {
            'timestamp': _stamp(minutes_ago),
            'message': {
                'content': [
                    {
                        'type': 'tool_use',
                        'id': call_id,
                        'name': 'Bash',
                        'input': {'command': command},
                    }
                ]
            },
        }
    )


def _loop_line() -> str:
    """The loop's OUT=... line, exactly as prompts.md gives it."""
    text = PROMPTS.read_text(encoding='utf-8')
    section = text.split('### Watchdog texts', 1)[1]
    found = re.search(r'^\s*(OUT=\$\(.*\))\s*$', section, re.MULTILINE)
    assert found, 'prompts.md has no loop OUT=... line'
    return found.group(1)


def _loop(tasks: Path, tmp_path: Path) -> str:
    """One pass of the loop's body: what it would act on."""
    script = f'B={TOOLS}; T={tasks}\n{_loop_line()}\nprintf %s "$OUT"\n'
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith('SDM_')},
        'SDM_ORCH_STATE': str(tmp_path / 'state'),
    }
    done = subprocess.run(
        ['bash', '-c', script],
        capture_output=True,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,
        timeout=120,
        check=False,
    )
    return done.stdout


@pytest.mark.parametrize('tool', WATCHDOGS)
def test_a_missing_tasks_directory_fails_loudly(tool, tmp_path):
    done = _run(tool, tmp_path / 'nowhere', '10', tmp_path)
    assert done.returncode != 0
    assert done.stdout.startswith(f'WATCHDOG-ERROR {tool}: '), done.stdout
    assert 'does not exist' in done.stdout


@pytest.mark.parametrize('tool', WATCHDOGS)
def test_a_file_in_place_of_the_tasks_directory_fails_loudly(tool, tmp_path):
    tasks = tmp_path / 'tasks'
    tasks.write_text('', encoding='utf-8')
    done = _run(tool, tasks, '10', tmp_path)
    assert done.returncode != 0
    assert 'WATCHDOG-ERROR' in done.stdout and 'not a directory' in done.stdout


@pytest.mark.parametrize('tool', WATCHDOGS)
def test_an_empty_tasks_directory_is_quiet(tool, tmp_path):
    done = _run(tool, tmp_path, '10', tmp_path)
    assert (done.returncode, done.stdout) == (0, '')


@pytest.mark.parametrize('tool', WATCHDOGS)
def test_a_bad_line_in_the_middle_fails_loudly(tool, tmp_path):
    (tmp_path / 'a1.output').write_text(
        _call('ls', 1) + '\n{not json\n' + _call('ls', 1, 't2') + '\n',
        encoding='utf-8',
    )
    done = _run(tool, tmp_path, '10', tmp_path)
    assert done.returncode != 0
    assert f'WATCHDOG-ERROR {tool}: a1.output: line 2 ' in done.stdout


@pytest.mark.parametrize('tool', WATCHDOGS)
def test_a_line_that_is_json_but_no_object_fails_loudly(tool, tmp_path):
    (tmp_path / 'a1.output').write_text('[1, 2]\n', encoding='utf-8')
    done = _run(tool, tmp_path, '10', tmp_path)
    assert done.returncode != 0 and 'WATCHDOG-ERROR' in done.stdout


@pytest.mark.parametrize('tool', WATCHDOGS)
def test_a_transcript_that_is_not_utf8_fails_loudly(tool, tmp_path):
    (tmp_path / 'a1.output').write_bytes(b'\xff\xfe{}\n')
    done = _run(tool, tmp_path, '10', tmp_path)
    assert done.returncode != 0
    assert 'WATCHDOG-ERROR' in done.stdout and 'cannot be read' in done.stdout


@pytest.mark.parametrize('tool', WATCHDOGS)
def test_a_record_of_another_shape_fails_loudly(tool, tmp_path):
    # A tool call with no timestamp: the age cannot be computed.
    record = json.loads(_call('ls', 1))
    record['timestamp'] = 12
    (tmp_path / 'a1.output').write_text(
        json.dumps(record) + '\n', encoding='utf-8'
    )
    done = _run(tool, tmp_path, '10', tmp_path)
    assert done.returncode != 0
    assert f'WATCHDOG-ERROR {tool}: a1.output: ' in done.stdout


@pytest.mark.parametrize('tool', WATCHDOGS)
def test_a_partial_last_line_is_not_an_error(tool, tmp_path):
    (tmp_path / 'a1.output').write_text(
        _call('ls', 1) + '\n' + _call('ls', 1, 't2')[:30], encoding='utf-8'
    )
    done = _run(tool, tmp_path, '10', tmp_path)
    assert (done.returncode, done.stdout) == (0, ''), done.stdout


def test_a_blocked_call_is_still_reported(tmp_path):
    (tmp_path / 'a1.output').write_text(
        _call('make thing', 50) + '\n' + _call('half', 1, 't2')[:20],
        encoding='utf-8',
    )
    done = _run('stuck_agents.py', tmp_path, '40', tmp_path)
    assert done.returncode == 0
    assert done.stdout.startswith('POSSIBLY-BLOCKED a1 50min Bash: ')


def test_a_violation_is_still_reported(tmp_path):
    (tmp_path / 'a1.output').write_text(
        _call('sudo ls', 1) + '\n', encoding='utf-8'
    )
    done = _run('rule_watch.py', tmp_path, '6', tmp_path)
    assert done.returncode == 0
    assert done.stdout.startswith('VIOLATION a1 no-sudo: sudo ls')


def test_the_loop_keeps_a_watchdog_error(tmp_path):
    out = _loop(tmp_path / 'nowhere', tmp_path)
    lines = out.splitlines()
    assert [line.split(':')[0] for line in lines] == [
        'WATCHDOG-ERROR stuck_agents.py',
        'WATCHDOG-ERROR rule_watch.py',
    ], out


def test_the_loop_keeps_an_unreadable_transcript(tmp_path):
    (tmp_path / 'a1.output').write_text('{oops\n{}\n', encoding='utf-8')
    assert _loop(tmp_path, tmp_path).count('WATCHDOG-ERROR') == 2


def test_the_loop_is_silent_when_all_is_well(tmp_path):
    (tmp_path / 'a1.output').write_text(_call('ls', 1) + '\n', encoding='utf-8')
    assert _loop(tmp_path, tmp_path) == ''
