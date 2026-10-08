# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P77.scenarios, the corpus side: offline scenarios take each of the three
branches the sampling gap list claimed unreachable, through the public API,
replay, and record again byte for byte (``CONTRACT.md``, "Scenarios").

``accept_gaps_core.py`` holds the core to them.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from _gaps import (
    INDICES_REFUSAL,
    PARITY_SCRIPTS,
    REPO,
    SELF_LINK_REFUSAL,
    SQLITE_DECLARED,
    TRANSCRIPTS,
    check_lines,
    describe_found,
    entity_of,
    error_of,
    found,
    mixed_indices,
    normalize_type,
    read,
)

ERROR_CLASS = 'sdm_client.errors.StructuredDataError'
OFFLINE = ('memory', 'duckdb', 'sqlite', 'postgres')


def _failed_candidates() -> list[str]:
    return [
        f'{path.parent.relative_to(TRANSCRIPTS)}: {replay.reason}'
        for path, replay in found().replays.items()
        if replay.reason is not None
    ]


def _context() -> str:
    failed = _failed_candidates()
    text = f'found: {describe_found()}'
    if failed:
        text += '\nreplays that failed:\n' + '\n'.join(failed[:10])
    return text


def test_the_branches_are_where_the_checks_say() -> None:
    check_lines()


def test_sqlite_entity_keys_declared_each_way() -> None:
    r"""On SQLite, a seeded entity draw takes the branch for a physical key
    column declared ``DECIMAL(10,0)``, one declared ``NUMERIC``, and one
    declared with no type."""
    taken = set()
    for types in found().sqlite_types.values():
        taken |= types
    missing = [t or '(no type)' for t in SQLITE_DECLARED if t not in taken]
    assert not missing, (
        'no SQLite transcript draws entities seeded over a key column '
        f'declared {missing}, taking sqlite/sampler.py 244->245\n' + _context()
    )


def test_the_sqlite_ones_complete() -> None:
    r"""Each sends at least one payload, answered, and ends without an
    error: the key reaches the request."""
    assert found().sqlite, 'no SQLite declared-key transcript\n' + _context()
    problems = []
    for name, path in found().sqlite.items():
        transcript = read(path)
        payloads = transcript.get('payloads') or []
        responses = transcript.get('responses') or []
        if error_of(transcript) is not None:
            problems.append(f'{name}: ended in {error_of(transcript)}')
        elif not payloads:
            problems.append(f'{name}: sent no payload')
        elif len(responses) != len(payloads) or not all(
            isinstance(r, dict) and r.get('status') == 200 for r in responses
        ):
            problems.append(f'{name}: not every payload was answered 200')
    assert not problems, '\n'.join(problems)


def _asked_before_last(transcript: dict[str, Any]) -> str | None:
    r"""The query the last ``materialize`` call materializes."""
    query = None
    for call in transcript.get('calls', []):
        if call.get('call') == 'PredictiveQuery':
            query = call.get('args', {}).get('query')
    return query if isinstance(query, str) else None


def _refused_at_last_materialize(transcript: dict[str, Any]) -> str | None:
    calls = transcript.get('calls', [])
    if not calls or calls[-1].get('call') != 'PredictiveQuery.materialize':
        return 'its last call is not PredictiveQuery.materialize'
    error = error_of(transcript)
    if error is None:
        return 'it ends without an error'
    if error.get('class') != ERROR_CLASS or error.get('code') != (
        'INVALID_REQUEST'
    ):
        return (
            f'it ends in {error.get("class")} {error.get("code")}, not '
            f'{ERROR_CLASS} INVALID_REQUEST'
        )
    return None


def test_duckdb_decimal_keys_with_mixed_indices() -> None:
    r"""On DuckDB, an entity key stored ``DECIMAL(10,0)``, and indices
    mixing ints and texts, each of which casts alone but not together: the
    refusal names no index (``rfm.py`` 262->267)."""
    assert found().indices, (
        'no DuckDB transcript refuses mixed indices over a decimal key '
        'without naming an index, taking rfm.py 262->267\n' + _context()
    )
    problems = []
    for name, path in found().indices.items():
        transcript = read(path)
        why = _refused_at_last_materialize(transcript)
        if why is not None:
            problems.append(f'{name}: {why}')
            continue
        last = transcript['calls'][-1]
        if not mixed_indices({'calls': [last]}):
            problems.append(
                f'{name}: the refused call asks indices '
                f'{last.get("args", {}).get("indices")}, not a mix of ints '
                'and texts'
            )
            continue
        query = _asked_before_last(transcript)
        entity = entity_of(query) if query else None
        if entity is None:
            problems.append(f'{name}: no FOR EACH in {query!r}')
            continue
        types = transcript.get('data', {}).get('column_types') or {}
        stored = normalize_type(types.get(f'{entity[0]}.{entity[1]}'))
        if stored != 'DECIMAL(10,0)':
            problems.append(
                f'{name}: data.column_types stores {entity[0]}.{entity[1]} '
                f'as {stored or "generated"}, not DECIMAL(10,0)'
            )
        message = str(error_of(transcript).get('message', ''))
        matched = INDICES_REFUSAL.search(message)
        if matched is None or matched['table'] != entity[0]:
            problems.append(f'{name}: the refusal is {message!r}')
        elif '; indices[' in message:
            problems.append(f'{name}: the refusal names an index: {message!r}')
    assert not problems, '\n'.join(problems)


def test_in_memory_self_link_with_a_count() -> None:
    r"""In memory, a table linked to itself and a ``COUNT`` over it for each
    of its rows: two edge types from the table to itself, and the engine's
    refusal (``base/sampler.py`` 428->429)."""
    assert found().self_link, (
        'no in-memory transcript is refused for a table linked to itself, '
        'taking base/sampler.py 428->429\n' + _context()
    )
    problems = []
    for name, path in found().self_link.items():
        transcript = read(path)
        why = _refused_at_last_materialize(transcript)
        if why is not None:
            problems.append(f'{name}: {why}')
            continue
        if transcript['calls'][0].get('call') != 'Schema.from_data':
            problems.append(f'{name}: its schema is not Schema.from_data')
        message = str(error_of(transcript).get('message', ''))
        table = SELF_LINK_REFUSAL.search(message)['table']
        query = _asked_before_last(transcript) or ''
        entity = entity_of(query)
        if entity is None or entity[0] != table:
            problems.append(f'{name}: {query!r} is not for each {table}')
        if not re.search(rf'\bCOUNT\s*\(\s*{table}\s*\.\s*\*', query, re.I):
            problems.append(f'{name}: {query!r} is not a COUNT over {table}')
    assert not problems, '\n'.join(problems)


def test_they_are_offline_corpus_scenarios() -> None:
    from _parity.scenarios import by_name

    every = found().every()
    assert every, 'none found\n' + _context()
    known = by_name()
    problems = []
    for name, path in every.items():
        transcript = read(path)
        scenario = known.get(name)
        if scenario is None:
            problems.append(f'{name}: not a scenario of scenarios.py')
        elif scenario.live or transcript['backend'] not in OFFLINE:
            problems.append(f'{name}: not an offline scenario')
        elif path.parent.relative_to(TRANSCRIPTS).as_posix() != name.replace(
            '#', '--'
        ):
            problems.append(f'{name}: stored at {path}')
    assert not problems, '\n'.join(problems)


def test_they_replay() -> None:
    every = found().every()
    assert every, 'none found\n' + _context()
    problems = [
        f'{name}: {found().replays[path].reason}'
        for name, path in every.items()
        if found().replays[path].reason is not None
    ]
    assert not problems, '\n'.join(problems[:10])


def test_recording_again_writes_the_committed_files() -> None:
    every = found().every()
    assert every, 'none found\n' + _context()
    names = sorted(every)
    with tempfile.TemporaryDirectory(prefix='sdm-p77-record-') as tmp:
        corpus = Path(tmp) / 'transcripts'
        proc = subprocess.run(
            [
                sys.executable,
                str(PARITY_SCRIPTS / 'run_corpus.py'),
                'record',
                '--corpus',
                str(corpus),
                '--only',
                ','.join(names),
            ],
            cwd=REPO,
            env=dict(os.environ),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        assert proc.returncode == 0, (
            f'run_corpus.py record failed ({proc.returncode}):\n'
            f'{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}'
        )
        problems = []
        for name in names:
            stored = every[name].parent
            fresh = corpus / stored.relative_to(TRANSCRIPTS)
            stored_files = sorted(
                p.relative_to(stored).as_posix()
                for p in stored.rglob('*')
                if p.is_file()
            )
            fresh_files = sorted(
                p.relative_to(fresh).as_posix()
                for p in fresh.rglob('*')
                if p.is_file()
            )
            if stored_files != fresh_files:
                problems.append(
                    f'{name}: files {fresh_files} != {stored_files}'
                )
                continue
            for file in stored_files:
                if (stored / file).read_bytes() != (fresh / file).read_bytes():
                    problems.append(
                        f'{name}: {file} differs from a fresh recording'
                    )
    assert not problems, '\n'.join(problems[:20])
