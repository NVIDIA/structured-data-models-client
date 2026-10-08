# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P85.payloads, in Rust: ``cargo test -p sdm-core --test request_replay``
replays every recording without Python, and fails when a recording is
changed (``CONTRACT.md``, "Recordings", "In Rust").

The test is the implementer's. These checks prove it replays them all, and
that it compares: changing, in a copy of the recordings, a call's body, a
query's SQL, a model's response, a warning, a ``done`` value or an error's
message, each alone, makes it fail.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from _request import (
    CALLS_FILE,
    RECORDINGS,
    REPLAY_TEST,
    RUST,
    TABLES_DIR,
    load_json,
    rc_module,
    recording_dirs,
)

LINE = re.compile(r'request_replay: (\d+) recordings, (\d+) calls')
SEED = 20261024
MUTATIONS = ('body', 'sql', 'response', 'warning', 'done', 'raised')


def _core_name() -> str:
    rc = rc_module()
    return rc.find_crate('core', rc.cargo_metadata())['name']


def _replay(root: Path | None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop('SDM_REQUEST_RECORDINGS', None)
    if root is not None:
        env['SDM_REQUEST_RECORDINGS'] = str(root)
    return subprocess.run(
        [
            'cargo',
            'test',
            '-p',
            _core_name(),
            '--test',
            'request_replay',
            '--',
            '--nocapture',
        ],
        cwd=RUST,
        env=env,
        capture_output=True,
        text=True,
    )


def _counts(proc: subprocess.CompletedProcess[str]) -> tuple[int, int]:
    found = LINE.findall(proc.stdout + proc.stderr)
    assert found, (
        f'the test printed no "request_replay: N recordings, M calls" line:\n'
        f'{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}'
    )
    return int(found[-1][0]), int(found[-1][1])


def _expected() -> tuple[int, int]:
    dirs = recording_dirs()
    calls = sum(len(load_json(d / CALLS_FILE)['calls']) for d in dirs)
    return len(dirs), calls


def _copy(dest: Path) -> Path:
    r"""The recordings' ``calls.json`` files, copied, with the shared tables
    linked rather than copied."""
    for directory in recording_dirs():
        target = dest / directory.relative_to(RECORDINGS)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(directory / CALLS_FILE, target / CALLS_FILE)
    tables = RECORDINGS / TABLES_DIR
    if tables.is_dir():
        (dest / TABLES_DIR).symlink_to(tables, target_is_directory=True)
    return dest


@pytest.fixture(scope='module')
def baseline() -> subprocess.CompletedProcess[str]:
    assert REPLAY_TEST.is_file(), f'{REPLAY_TEST} does not exist'
    assert recording_dirs(), 'no recordings are stored'
    return _replay(None)


def test_the_replay_passes_on_every_recording(
    baseline: subprocess.CompletedProcess[str],
) -> None:
    assert baseline.returncode == 0, (
        f'{baseline.stdout[-4000:]}\n{baseline.stderr[-4000:]}'
    )
    assert _counts(baseline) == _expected()


def test_the_replay_reads_the_named_recordings(
    baseline: subprocess.CompletedProcess[str], tmp_path: Path
) -> None:
    assert baseline.returncode == 0
    proc = _replay(_copy(tmp_path / 'copy'))
    assert proc.returncode == 0, f'{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}'
    assert _counts(proc) == _expected()


# Mutations.


def _leaf(value: Any) -> tuple[Any, Any] | None:
    r"""The first number or string in a value, outside the tags that name
    files, as (container, key)."""
    if isinstance(value, dict):
        if len(value) == 1 and next(iter(value)) in (
            '$table',
            '$batch',
            '$array',
            '$chunked',
        ):
            return None
        for key, item in value.items():
            if isinstance(item, (int, float, str)) and not isinstance(
                item, bool
            ):
                return value, key
            found = _leaf(item)
            if found:
                return found
    if isinstance(value, list):
        for i, item in enumerate(value):
            if isinstance(item, (int, float, str)) and not isinstance(
                item, bool
            ):
                return value, i
            found = _leaf(item)
            if found:
                return found
    return None


def _bump(container: Any, key: Any) -> None:
    value = container[key]
    if isinstance(value, str):
        container[key] = value + 'x'
    elif isinstance(value, int):
        container[key] = value + 1
    else:
        container[key] = value + 0.5


def _candidates(kind: str, document: dict[str, Any]) -> list[Any]:
    r"""Each place ``kind`` can be changed in one recording, as a function
    that changes it."""
    out: list[Any] = []
    for call in document['calls']:
        if kind == 'raised' and 'raised' in call:
            out.append(
                lambda c=call: c['raised'].update(
                    message=c['raised']['message'] + ' x'
                )
            )
        for step in call.get('steps') or []:
            if kind == 'warning' and step.get('warnings'):
                out.append(
                    lambda s=step: s['warnings'][0].update(
                        message=s['warnings'][0]['message'] + ' x'
                    )
                )
            if kind == 'raised' and step['kind'] == 'raised':
                out.append(
                    lambda s=step: s['error'].update(
                        message=s['error']['message'] + ' x'
                    )
                )
            if kind == 'sql' and step['kind'] == 'run_queries':
                out.append(
                    lambda s=step: s['queries'][0].update(
                        sql=s['queries'][0]['sql'] + ' '
                    )
                )
            if kind == 'body' and step['kind'] == 'call_model':
                leaf = _leaf(step.get('body'))
                if leaf:
                    out.append(lambda leaf=leaf: _bump(*leaf))
            if kind == 'done' and step['kind'] == 'done':
                value = step['value']
                if isinstance(value, (int, float, str)) and not isinstance(
                    value, bool
                ):
                    leaf: Any = (step, 'value')
                else:
                    leaf = _leaf(value)
                if leaf:
                    out.append(lambda leaf=leaf: _bump(*leaf))
            if kind == 'response' and step['kind'] == 'call_model':
                outcome = step.get('outcome') or {}
                text = outcome.get('response')
                if step['endpoint'] not in ('predict', 'session_predict'):
                    continue
                if not isinstance(text, str):
                    continue
                body = json.loads(text)
                items = (
                    body.get('predictions') if isinstance(body, dict) else None
                )
                for item in items or []:
                    if isinstance(
                        item.get('prediction'), (int, float)
                    ) and not (isinstance(item.get('prediction'), bool)):

                        def change(o=outcome, b=body, i=item) -> None:
                            i['prediction'] = i['prediction'] + 1.5
                            o['response'] = json.dumps(b)

                        out.append(change)
                        break
    return out


@pytest.mark.parametrize('kind', MUTATIONS)
def test_the_replay_fails_on_a_changed_recording(
    baseline: subprocess.CompletedProcess[str], tmp_path: Path, kind: str
) -> None:
    assert baseline.returncode == 0, 'the replay does not pass as stored'
    root = _copy(tmp_path / 'copy')
    places = []
    for directory in sorted(p.parent for p in root.rglob(CALLS_FILE)):
        document = load_json(directory / CALLS_FILE)
        count = len(_candidates(kind, document))
        places += [(directory, i) for i in range(count)]
    assert places, f'no recording holds a {kind} to change'
    directory, at = random.Random(f'{SEED}-{kind}').choice(places)
    document = load_json(directory / CALLS_FILE)
    _candidates(kind, document)[at]()
    (directory / CALLS_FILE).write_text(json.dumps(document, indent=1))
    proc = _replay(root)
    assert proc.returncode != 0, (
        f'the replay passed with a {kind} changed in '
        f'{directory.relative_to(root)}'
    )
