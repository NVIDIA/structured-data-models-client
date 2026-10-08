# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P9.pickle: schemas, requests and results pickle and copy as they did at
BASE, and BASE's own pickles still read back and work.

``pickles/`` holds what BASE's code gave (``_pickles.py make``): each object
a few transcripts' public calls make, pickled, and ``expected.json``, what
it was, what it did, and what ``pickle``, ``copy.copy``, ``copy.deepcopy``
and ``snapshot()`` gave, with a schema copy's relationships then cleared.
Today's code, on the core, must give all of it again (``_pickles.py
fresh``), a raised error compared by class; and every BASE pickle must read
back as what it was, and do what it did (``_pickles.py old``). The reads a
DuckDB schema makes, and those of its first question after each copy, are
part of what is compared.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
from _switch import (
    CORE,
    HERE,
    REPO,
    env_for,
    first_difference,
    install_core,
    run,
)

EXPECTED = HERE / 'pickles' / 'expected.json'


def _by_class(value: Any) -> Any:
    r"""An outcome with any raised error reduced to its class."""
    if isinstance(value, dict):
        if 'raised' in value and set(value) <= {'raised', 'message'}:
            return {'raised': value['raised']}
        return {k: _by_class(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_by_class(v) for v in value]
    return value


def _run(command: str) -> dict[str, Any]:
    install_core()
    with tempfile.TemporaryDirectory(prefix='sdm-p9-pickle-') as tmp:
        out = Path(tmp) / 'out.json'
        proc = run(
            [sys.executable, HERE / '_pickles.py', command, '--out', out],
            cwd=REPO,
            env=env_for(CORE),
            timeout=1800,
        )
        assert proc.returncode == 0 and out.is_file(), (
            f'_pickles.py {command} failed (exit {proc.returncode}):\n'
            f'{proc.stdout[-3000:]}\n{proc.stderr[-5000:]}'
        )
        return json.loads(out.read_text())


@pytest.fixture(scope='module')
def expected() -> dict[str, Any]:
    return json.loads(EXPECTED.read_text())


def test_fresh_objects_round_trip_as_they_did_at_base(expected):
    got = _run('fresh')
    assert not any(got['modules'].values()), (
        f'not made on the core: {got["modules"]}'
    )
    problems = []
    assert set(got['objects']) == set(expected), sorted(
        set(got['objects']) ^ set(expected)
    )
    for name, want in sorted(expected.items()):
        want = {
            k: v
            for k, v in want.items()
            if k not in ('file', 'transcript', 'at')
        }
        found = first_difference(
            _by_class(want), _by_class(got['objects'][name])
        )
        if found:
            problems.append(f'{name}: {found}')
    assert not problems, '\n'.join(problems)


def test_base_pickles_read_back_and_work(expected):
    got = _run('old')['objects']
    problems = []
    pickled = {n: e for n, e in expected.items() if e['file']}
    assert len(pickled) >= 15, len(pickled)
    for name, want in sorted(pickled.items()):
        mine = got.get(name)
        if mine is None or 'raised' in mine:
            problems.append(f'{name}: does not read back: {mine}')
            continue
        for key, theirs in (
            ('described', want['original']),
            ('behavior', want['behavior']),
            (
                'copy',
                {'value': want['deepcopy']['described']}
                if 'described' in want['deepcopy']
                else want['deepcopy'],
            ),
        ):
            found = first_difference(_by_class(theirs), _by_class(mine[key]))
            if found:
                problems.append(f'{name} {key}: {found}')
    assert not problems, '\n'.join(problems)
