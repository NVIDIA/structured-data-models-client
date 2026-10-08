# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P3.rerecord: the DuckDB and SQLite recordings are this phase's.

Each stored DuckDB and SQLite recording:

* writes nothing: no ``ingest``, ``register`` or ``commit`` step, and no
  statement that creates, drops, inserts into or reads a ``tmp_*`` table;
* passes its key lists as JSON parameters;
* is what recording the scenario again with this phase's code stores, file
  for file, byte for byte.

Adi approves the new recordings (``P3.corpus``); this check only shows they
are this code's.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from _seam import (
    RUN_CORPUS,
    TRANSCRIPTS,
    load_transcript,
    output,
    passes_keys_as_json,
    run_python,
    scenario_dirs,
)

KEYED = ('duckdb', 'sqlite')
_WRITES = re.compile(
    r'^\s*(CREATE|DROP|INSERT|ALTER|ATTACH|DETACH|DELETE|UPDATE|REPLACE|'
    r'COPY|CHECKPOINT|VACUUM|BEGIN|COMMIT)\b'
    r'|\bCREATE\s+(OR\s+REPLACE\s+)?(TEMP|TEMPORARY|TABLE|INDEX|VIEW)\b'
    r'|\bINSERT\s+INTO\b|\bDROP\s+(TABLE|INDEX|VIEW)\b'
    r'|\bPRAGMA\s+\w+\s*='
    r'|\btmp_\w+',
    re.IGNORECASE,
)


def _names() -> list[str]:
    return [
        name
        for name in scenario_dirs(TRANSCRIPTS)
        if name.split('/', 1)[0] in KEYED
    ]


def test_keyed_recordings_are_present() -> None:
    names = _names()
    for group in KEYED:
        assert sum(n.startswith(f'{group}/') for n in names) >= 6, names


@pytest.mark.parametrize('name', _names())
def test_the_recording_writes_nothing_and_passes_keys_as_json(
    name: str,
) -> None:
    transcript = load_transcript(TRANSCRIPTS / name / 'transcript.json')
    steps = transcript['steps']
    writes = [
        f'step {i}: {s["op"]} {s.get("sql", s.get("table", s.get("name")))!r:.120}'
        for i, s in enumerate(steps)
        if s['op'] in ('ingest', 'register', 'commit')
        or (s['op'] == 'execute' and _WRITES.search(s['sql']))
    ]
    assert not writes, f'{name} still writes to the database:\n' + '\n'.join(
        writes
    )
    assert any(passes_keys_as_json(s) for s in steps), (
        f'{name} passes no key list as a JSON parameter'
    )


def test_recording_again_stores_the_same_files(tmp_path: Path) -> None:
    names = _names()
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
    problems: list[str] = []
    for name in names:
        stored, got = TRANSCRIPTS / name, fresh / name
        stored_files = sorted(p.name for p in stored.iterdir() if p.is_file())
        got_files = sorted(p.name for p in got.iterdir() if p.is_file())
        if stored_files != got_files:
            problems.append(f'{name}: files {got_files} != {stored_files}')
            continue
        for file in stored_files:
            if (stored / file).read_bytes() != (got / file).read_bytes():
                problems.append(f'{name}/{file} records differently')
    assert not problems, '\n'.join(problems)
