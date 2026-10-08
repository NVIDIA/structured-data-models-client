# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P3.replay: the Phase 2 recordings hold through this phase's code.

The recordings at ``BASE``, the commit phase 3 started from, are the
reference; the stored ones may since have been re-recorded.

* Every scenario recorded at ``BASE`` is still recorded, under the same name.
* PostgreSQL, in-memory, tabular and relational recordings are unchanged,
  file for file, so their replay (``tests/parity/acceptance/
  accept_replay.py``) is a byte-identical SQL step sequence with
  parsed-equal payloads. PostgreSQL scenarios recorded afresh against the
  database also reproduce them byte for byte.
* DuckDB and SQLite recordings may differ only in how key lists reach the
  database. Their calls, data, payloads (parsed, value and JSON type),
  answer and warnings are as at ``BASE``. So are their steps' statements,
  parameters and failures, once the key-passing steps are set aside on each
  side: at ``BASE``, the
  ``ingest``, ``register`` and ``commit`` steps and every statement naming a
  ``tmp_*`` table; now, every statement given a key list as a JSON
  parameter. A scenario passes keys now exactly when it did then.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pytest
from _seam import (
    BASE,
    REPO,
    RUN_CORPUS,
    TRANSCRIPTS,
    export_base_transcripts,
    git,
    load_transcript,
    output,
    passes_keys_as_json,
    read_arrow,
    run_python,
    scenario_dirs,
    strict,
)

UNCHANGED = ('postgres', 'memory', 'tabular', 'relational')
KEYED = ('duckdb', 'sqlite')
_TMP_TABLE = re.compile(r'\btmp_\w+', re.IGNORECASE)


@pytest.fixture(scope='module')
def base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return export_base_transcripts(tmp_path_factory.mktemp('base'))


def _group(name: str) -> str:
    return name.split('/', 1)[0]


def test_every_base_scenario_is_still_recorded(base: Path) -> None:
    before = sorted(scenario_dirs(base))
    now = sorted(scenario_dirs(TRANSCRIPTS))
    assert len(before) >= 33, f'only {len(before)} scenarios at BASE'
    assert now == before, (
        f'scenarios lost: {sorted(set(before) - set(now))}; '
        f'added: {sorted(set(now) - set(before))}'
    )


def test_unkeyed_recordings_are_unchanged() -> None:
    paths = [f'tests/parity/transcripts/{group}' for group in UNCHANGED]
    changed = git('diff', '--name-only', BASE, '--', *paths).split()
    untracked = git(
        'ls-files', '--others', '--exclude-standard', '--', *paths
    ).split()
    assert not changed and not untracked, (
        f'recordings that must not change did: {changed + untracked}'
    )


def test_postgres_records_byte_identically(base: Path, tmp_path: Path) -> None:
    if not os.getenv('SDM_POSTGRES_TEST_DSN'):
        pytest.fail('recording PostgreSQL needs SDM_POSTGRES_TEST_DSN')
    names = [n for n in scenario_dirs(base) if _group(n) == 'postgres']
    assert len(names) >= 6
    fresh = tmp_path / 'fresh'
    fresh.mkdir()
    proc = run_python(
        RUN_CORPUS,
        'record',
        '--corpus',
        fresh,
        '--only',
        ','.join(names),
        timeout=1800,
    )
    assert proc.returncode == 0, output(proc)
    for name in names:
        expected = base / name
        got = fresh / name
        assert sorted(p.name for p in got.iterdir()) == sorted(
            p.name for p in expected.iterdir()
        ), f'{name}: other files recorded'
        for path in sorted(expected.iterdir()):
            assert (got / path.name).read_bytes() == path.read_bytes(), (
                f'{name}/{path.name} records differently than at BASE'
            )


def _base_key_step(step: dict[str, Any]) -> bool:
    if step['op'] in ('ingest', 'register', 'commit'):
        return True
    return step['op'] == 'execute' and bool(_TMP_TABLE.search(step['sql']))


def _comparable_step(step: dict[str, Any]) -> dict[str, Any]:
    r"""What must not change: the statement, its parameters, and whether it
    failed. How a result is fetched is the connector's to choose."""
    out = {
        'op': step['op'],
        'sql': step.get('sql'),
        'params': step.get('params'),
    }
    if step.get('error'):
        out['error'] = step['error'].get('type')
    return out


def _keyed_names(base: Path) -> list[str]:
    return [n for n in scenario_dirs(base) if _group(n) in KEYED]


@pytest.mark.parametrize('group', KEYED)
def test_keyed_scenarios_exist(base: Path, group: str) -> None:
    names = [n for n in _keyed_names(base) if _group(n) == group]
    assert len(names) >= 6, f'only {names} recorded for {group}'


def test_keyed_recordings_differ_only_in_passing_keys(base: Path) -> None:
    problems: list[str] = []
    for name in _keyed_names(base):
        before_dir, now_dir = base / name, TRANSCRIPTS / name
        before = load_transcript(before_dir / 'transcript.json')
        now = load_transcript(now_dir / 'transcript.json')
        for key in ('backend', 'task_kind', 'data', 'calls', 'warnings'):
            if strict(before.get(key)) != strict(now.get(key)):
                problems.append(f'{name}: {key} differs')
        if strict(before['payloads']) != strict(now['payloads']):
            problems.append(f'{name}: the payloads differ')
        a = read_arrow(before_dir / before['result'])
        b = read_arrow(now_dir / now['result'])
        if not a.equals(b, check_metadata=True):
            problems.append(f'{name}: the decoded answer differs')

        base_keyed = [s for s in before['steps'] if _base_key_step(s)]
        now_keyed = [s for s in now['steps'] if passes_keys_as_json(s)]
        if bool(base_keyed) != bool(now_keyed):
            problems.append(
                f'{name}: {len(base_keyed)} key-passing steps at BASE, '
                f'{len(now_keyed)} JSON key-list statements now'
            )
        rest_before = [
            _comparable_step(s)
            for s in before['steps']
            if not _base_key_step(s)
        ]
        rest_now = [
            _comparable_step(s)
            for s in now['steps']
            if not passes_keys_as_json(s)
        ]
        if len(rest_before) != len(rest_now):
            problems.append(
                f'{name}: {len(rest_before)} other steps at BASE, '
                f'{len(rest_now)} now'
            )
            continue
        for index, (x, y) in enumerate(zip(rest_before, rest_now, strict=True)):
            if strict(x) != strict(y):
                keys = sorted(
                    k
                    for k in set(x) | set(y)
                    if strict(x.get(k)) != strict(y.get(k))
                )
                problems.append(
                    f'{name}: the {index}th step that passes no keys differs '
                    f'in {keys}: {str(x.get("sql"))[:120]!r}'
                )
                break
    assert not problems, '\n'.join(problems)


def test_the_base_reference_is_the_phase_2_corpus(base: Path) -> None:
    # The reference must be what phase 2 recorded, not something else.
    assert (base / 'sqlite' / 'count' / 'transcript.json').is_file()
    head = git('rev-parse', 'HEAD').strip()
    merge_base = git('merge-base', BASE, head).strip()
    assert merge_base == BASE, 'HEAD does not descend from BASE'
    assert REPO.is_dir()
