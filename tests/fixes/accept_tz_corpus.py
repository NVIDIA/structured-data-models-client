# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P55.tz-corpus: the session-zone scenarios, and what re-recording changed.

The corpus gains five scenarios recorded under a session zone other than UTC
(``DESIGN-timezone.md``, section 5), each naming its twin: the same data,
calls and anchor, with naive time columns under UTC.

* Each is recorded, replays, records its session and column types, and
  validates against the transcript schema; ``FORMAT.md`` and the schema
  describe the new keys.
* Each equals its twin: the same payloads and the same decoded answer.
* Admission: recorded with the code of ``BASE`` (this checkout's recorder,
  ``BASE``'s packages), each zoned scenario differs from its twin, so it
  catches the bug; each naive one equals it there too.
* The six PostgreSQL transcripts differ from ``BASE``'s only in that time
  bounds in their parameters gain ``+00:00``, and in one size probe before an
  inference sample; every result file is the same, renumbered. Every other
  transcript of ``BASE`` is unchanged, byte for byte.
* The promoted checker, ``scripts/parity/check_only_offset.py OLD NEW
  postgres``, accepts the re-recording and refuses a tampered one.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from _fixes import (
    BASE,
    PARITY_SCRIPTS,
    REPO,
    TRANSCRIPTS,
    assert_imports_base,
    base_env,
    output,
)

SCENARIOS = {
    'postgres/count#timestamptz-new-york': ('America/New_York', True),
    'postgres/inactivity#timestamptz-kolkata': ('Asia/Kolkata', True),
    'postgres/count#timestamp-kolkata': ('Asia/Kolkata', False),
    'duckdb/count#timestamptz-new-york': ('America/New_York', True),
    'duckdb/count#timestamp-new-york': ('America/New_York', False),
}
POSTGRES_KINDS = (
    'count',
    'classification',
    'forecast',
    'inactivity',
    'ranking',
    'regression',
)
RUN_CORPUS = PARITY_SCRIPTS / 'run_corpus.py'
CHECKER = PARITY_SCRIPTS / 'check_only_offset.py'
SCHEMA = REPO / 'tests' / 'parity' / 'schema' / 'transcript.schema.json'
FORMAT = REPO / 'tests' / 'parity' / 'FORMAT.md'


def _directory(name: str, root: Path = TRANSCRIPTS) -> Path:
    return root / name.replace('#', '--')


def _load(name: str, root: Path = TRANSCRIPTS) -> dict[str, Any]:
    path = _directory(name, root) / 'transcript.json'
    assert path.is_file(), f'{name} is not recorded at {path}'
    return json.loads(path.read_text())


def _twin_of(transcript: dict[str, Any]) -> str:
    twin = transcript.get('twin', transcript['data'].get('twin'))
    assert isinstance(twin, str) and twin, f'{transcript["name"]} names no twin'
    return twin


def _zoneless(data: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in data.items()
        if k not in ('session', 'column_types', 'twin')
    }


def _naive(types: Any) -> bool:
    r"""Whether recorded column types name no zoned type: ``timestamptz``,
    ``... with time zone``, ``TIMESTAMP_LTZ`` or ``TIMESTAMP_TZ`` are zoned;
    ``... without time zone`` and ``TIMESTAMP_NTZ`` are not."""
    text = json.dumps(types).lower()
    text = text.replace('without time zone', '').replace('ntz', '')
    return not re.search(r'tz|with time zone', text)


def _arrow(path: Path) -> Any:
    import pyarrow as pa

    data = path.read_bytes()
    try:
        return pa.ipc.open_file(pa.BufferReader(data)).read_all()
    except pa.ArrowInvalid:
        return pa.ipc.open_stream(pa.BufferReader(data)).read_all()


# The scenarios ###############################################################


@pytest.mark.parametrize('name', SCENARIOS)
def test_the_scenario_records_its_session_and_column_types(name: str) -> None:
    zone, zoned = SCENARIOS[name]
    transcript = _load(name)
    assert transcript['name'] == name
    data = transcript['data']
    assert zone in json.dumps(data.get('session')), data.get('session')
    types = data.get('column_types')
    assert types, f'{name} records no column types'
    assert _naive(types) != zoned, types


@pytest.mark.parametrize('name', SCENARIOS)
def test_the_twin_is_the_same_scenario_naive_under_utc(name: str) -> None:
    transcript = _load(name)
    twin = _load(_twin_of(transcript))
    assert twin['backend'] == transcript['backend']
    assert twin['task_kind'] == transcript['task_kind']
    assert twin['calls'] == transcript['calls']
    assert _zoneless(twin['data']) == _zoneless(transcript['data'])
    session = twin['data'].get('session')
    assert session is None or 'UTC' in json.dumps(session), session
    types = twin['data'].get('column_types')
    assert types is None or _naive(types), types


@pytest.mark.parametrize('name', SCENARIOS)
def test_the_scenario_equals_its_twin(name: str) -> None:
    transcript = _load(name)
    twin_name = _twin_of(transcript)
    twin = _load(twin_name)
    assert transcript['payloads'], name
    assert transcript['payloads'] == twin['payloads']
    got = _arrow(_directory(name) / transcript['result'])
    want = _arrow(_directory(twin_name) / twin['result'])
    assert got.equals(want, check_metadata=False)


def test_the_scenarios_and_their_twins_replay() -> None:
    names = set(SCENARIOS)
    names |= {_twin_of(_load(name)) for name in SCENARIOS}
    proc = subprocess.run(
        [sys.executable, str(RUN_CORPUS), 'replay', '--only', ','.join(names)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=900,
    )
    passed = {
        line.split()[1]
        for line in proc.stdout.splitlines()
        if line.startswith('PASS ')
    }
    assert proc.returncode == 0 and passed == names, output(proc)


def test_the_new_keys_are_described_and_validate() -> None:
    import jsonschema

    schema = json.loads(SCHEMA.read_text())
    text = json.dumps(schema)
    for key in ('session', 'column_types', 'twin'):
        assert f'"{key}"' in text, f'the schema does not describe {key}'
    described = FORMAT.read_text()
    for key in ('session', 'column_types', 'twin'):
        assert f'`{key}`' in described, f'FORMAT.md does not describe {key}'
    names = set(SCENARIOS) | {_twin_of(_load(n)) for n in SCENARIOS}
    for name in names:
        jsonschema.validate(_load(name), schema)


# Admission ###################################################################


@pytest.fixture(scope='module')
def base_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp('base')
    assert_imports_base(base_env(root), root)
    return root


@pytest.mark.parametrize('name', SCENARIOS)
def test_admission_base_code_tells_the_scenario_from_its_twin(
    base_root: Path, tmp_path: Path, name: str
) -> None:
    from _fixes import postgres_dsn

    if name.startswith('postgres/'):
        postgres_dsn()
    corpus = tmp_path / 'corpus'
    proc = subprocess.run(
        [
            sys.executable,
            str(RUN_CORPUS),
            'record',
            '--corpus',
            str(corpus),
            '--only',
            name,
        ],
        cwd=REPO,
        env=base_env(base_root),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, output(proc)
    at_base = _load(name, corpus)
    twin = _load(_twin_of(_load(name)))
    _, zoned = SCENARIOS[name]
    if zoned:
        assert at_base['payloads'] != twin['payloads'], (
            f'{name} gives its twin payloads under BASE code; it would not '
            'catch the bug'
        )
    else:
        assert at_base['payloads'] == twin['payloads']


# What re-recording changed ###################################################

_BOUND = re.compile(r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?$')


def _strip(value: Any, count: list[int]) -> Any:
    r"""``value`` with ``+00:00`` taken off every bound-shaped string, inside
    JSON text too."""
    if isinstance(value, str):
        if value.endswith('+00:00') and _BOUND.match(value[:-6]):
            count[0] += 1
            return value[:-6]
        try:
            inner = json.loads(value)
        except ValueError:
            return value
        if isinstance(inner, (list, dict)):
            stripped = _strip(inner, count)
            if stripped != inner:
                return json.dumps(stripped, separators=(',', ':'))
        return value
    if isinstance(value, list):
        return [_strip(v, count) for v in value]
    if isinstance(value, dict):
        return {k: _strip(v, count) for k, v in value.items()}
    return value


def _same_json_text(a: Any, b: Any) -> bool:
    r"""Parameters equal, JSON text compared by value."""
    if isinstance(a, str) and isinstance(b, str):
        if a == b:
            return True
        try:
            return json.loads(a) == json.loads(b)
        except ValueError:
            return False
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _same_json_text(x, y) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(
            _same_json_text(a[k], b[k]) for k in a
        )
    return a == b


def _step_matches(
    got: dict[str, Any], want: dict[str, Any], count: list[int]
) -> bool:
    r"""``got`` is ``want`` but for ``+00:00`` on its bounds; its result is
    compared apart, by content, since steps are renumbered."""
    keep = ('params', 'result')
    if {k: v for k, v in got.items() if k not in keep} != {
        k: v for k, v in want.items() if k not in keep
    }:
        return False
    stripped = [0]
    if not _same_json_text(
        _strip(got.get('params'), stripped), want.get('params')
    ):
        return False
    count[0] += stripped[0]
    return True


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_probe(step: dict[str, Any]) -> bool:
    sql = step.get('sql') or ''
    return (
        step.get('op') == 'execute'
        and step.get('error') is None
        and bool(re.search(r'pg_relation_size|relpages', sql))
        and '__kumo_order_' not in sql
    )


@pytest.fixture(scope='module')
def base_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp('base-corpus')
    archive = subprocess.run(
        ['git', 'archive', BASE, 'tests/parity/transcripts'],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout
    subprocess.run(['tar', '-x', '-C', str(root)], input=archive, check=True)
    return root / 'tests' / 'parity' / 'transcripts'


@pytest.mark.parametrize('kind', POSTGRES_KINDS)
def test_postgres_rerecording_differs_only_by_offsets_and_probes(
    base_corpus: Path, kind: str
) -> None:
    old_dir = base_corpus / 'postgres' / kind
    new_dir = TRANSCRIPTS / 'postgres' / kind
    old = json.loads((old_dir / 'transcript.json').read_text())
    new = json.loads((new_dir / 'transcript.json').read_text())
    assert {k: v for k, v in old.items() if k != 'steps'} == {
        k: v for k, v in new.items() if k != 'steps'
    }
    assert _digest(old_dir / old['result']) == _digest(new_dir / new['result'])

    count = [0]
    position = 0
    probes = 0
    steps = new['steps']
    for index, step in enumerate(steps):
        if position < len(old['steps']) and _step_matches(
            step, old['steps'][position], count
        ):
            want = old['steps'][position].get('result')
            got = step.get('result')
            if want is None:
                assert got is None, f'postgres/{kind} step {index}'
            else:
                assert got is not None, f'postgres/{kind} step {index}'
                assert _digest(new_dir / got) == _digest(old_dir / want), (
                    f'postgres/{kind} step {index}: its result changed'
                )
            position += 1
            continue
        assert _is_probe(step), (
            f'postgres/{kind} step {index} is neither BASE step {position} '
            f'with +00:00 bounds nor a size probe:\n{json.dumps(step)[:2000]}'
        )
        following = steps[index + 1] if index + 1 < len(steps) else {}
        assert '__kumo_order_' in (following.get('sql') or ''), (
            f'postgres/{kind} step {index}: a probe not right before an '
            'inference sample'
        )
        probes += 1
    assert position == len(old['steps']), (
        f'postgres/{kind}: BASE steps from {position} on are missing'
    )
    assert probes <= sum(
        '__kumo_order_' in (s.get('sql') or '') for s in old['steps']
    )
    assert count[0] > 0, f'postgres/{kind}: no bound gained +00:00'


def test_every_other_base_recording_is_unchanged(base_corpus: Path) -> None:
    changed = []
    for old in sorted(base_corpus.rglob('*')):
        if old.is_dir():
            continue
        relative = old.relative_to(base_corpus)
        if relative.parts[0] == 'postgres' and relative.parts[1] in (
            POSTGRES_KINDS
        ):
            continue
        new = TRANSCRIPTS / relative
        if not new.is_file() or _digest(new) != _digest(old):
            changed.append(str(relative))
    assert not changed, changed


# The promoted checker ########################################################


def _check(old: Path, new: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), str(old), str(new), 'postgres'],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_the_promoted_checker_accepts_the_rerecording(
    base_corpus: Path,
) -> None:
    assert CHECKER.is_file(), f'{CHECKER} is missing'
    proc = _check(base_corpus, TRANSCRIPTS)
    assert proc.returncode == 0, output(proc)


@pytest.mark.parametrize('tamper', ('sql', 'param', 'result', 'payload'))
def test_the_promoted_checker_refuses_a_tampered_rerecording(
    base_corpus: Path, tmp_path: Path, tamper: str
) -> None:
    assert CHECKER.is_file(), f'{CHECKER} is missing'
    copy = tmp_path / 'transcripts'
    shutil.copytree(TRANSCRIPTS, copy)
    directory = copy / 'postgres' / 'count'
    path = directory / 'transcript.json'
    transcript = json.loads(path.read_text())
    steps = transcript['steps']
    if tamper == 'sql':
        step = next(s for s in steps if '__kumo_order_' in (s.get('sql') or ''))
        step['sql'] = step['sql'].replace('LIMIT 1000', 'LIMIT 999')
    elif tamper == 'param':
        step = next(s for s in steps if s.get('params'))
        step['params'] = [*step['params'][:-1], 'tampered']
    elif tamper == 'result':
        step = next(s for s in steps if s.get('result'))
        target = directory / step['result']
        other = next(
            s['result']
            for s in steps
            if s.get('result')
            and s['result'] != step['result']
            and _digest(directory / s['result']) != _digest(target)
        )
        shutil.copy(directory / other, target)
    else:
        transcript['payloads'][0]['metadata']['source_query'] = 'tampered'
    path.write_text(json.dumps(transcript, indent=1))
    proc = _check(base_corpus, copy)
    assert proc.returncode != 0, output(proc)
    assert 'Traceback' not in proc.stderr, output(proc)
