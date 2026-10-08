# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P65.schemas: for every corpus scenario, offline and live, the Rust core's
schema and warnings equal today's Python's.

Each transcript is replayed through today's Python, keeping every
``Schema.from_data`` and ``Schema.from_source`` call its public calls make.
The core makes the same call: ``schema_from_data`` over the describer's Arrow
tables and tags, or a ``schema_from_source`` run whose every query is answered
by the recorded ``execute`` step it matches, through ``Source._run`` on the
replaying connection. The outcomes must be equal (``CONTRACT.md``,
"Outcomes"), and the run must have asked for every step the Python call took.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from _schema import (
    TRANSCRIPTS,
    difference,
    native,
    python_over_matching_tape,
    recorded_executes,
    rust_scenario_from_data,
    rust_scenario_from_source,
    scenario_calls,
    transcript_paths,
)

PATHS = transcript_paths()


def test_the_corpus_is_there() -> None:
    backends = Counter(p.relative_to(TRANSCRIPTS).parts[0] for p in PATHS)
    for backend in (
        'databricks',
        'duckdb',
        'lakebase',
        'memory',
        'postgres',
        'snowflake',
        'sqlite',
        'tabular',
    ):
        assert backends[backend], f'no {backend} transcript to compare'


@pytest.mark.parametrize(
    'path',
    [p for p in PATHS if 'from_source' in p.read_text()],
    ids=lambda p: str(p.parent.relative_to(TRANSCRIPTS)),
)
def test_the_host_answers_python_as_its_recording_did(path: Path) -> None:
    r"""The host the core is driven by answers today's Python, call for call,
    as the replay did: the comparison stands on answers that are the
    recording's."""
    transcript, calls = scenario_calls(path)
    for call in calls:
        if call.kind != 'from_source':
            continue
        again, tape = python_over_matching_tape(transcript, path, call)
        assert call.steps is not None
        assert sorted(tape.used) == recorded_executes(transcript, call.steps)
        assert not tape.unmatched
        assert difference(call.python, again) is None


@pytest.mark.parametrize(
    'path', PATHS, ids=[str(p.parent.relative_to(TRANSCRIPTS)) for p in PATHS]
)
def test_the_core_infers_the_schema_python_infers(path: Path) -> None:
    transcript, calls = scenario_calls(path)
    assert calls, f'{transcript["name"]} makes no schema'
    native('schema_from_data', 'schema_from_source')
    failures = []
    for i, call in enumerate(calls):
        where = f'{transcript["name"]} schema #{i} ({call.kind})'
        if call.kind == 'from_data':
            got = rust_scenario_from_data(call)
        else:
            got, tape = rust_scenario_from_source(transcript, path, call)
            assert call.steps is not None
            wanted = recorded_executes(transcript, call.steps)
            unused = [index for index in wanted if index not in tape.used]
            if tape.unmatched:
                failures.append(
                    f'{where}: the run asked {len(tape.unmatched)} '
                    f'statement(s) Python did not run, the first '
                    f'{tape.unmatched[0][0][:200]!r}'
                )
            if unused:
                failures.append(
                    f'{where}: the run never asked for the recorded steps '
                    f'{unused}, which Python ran'
                )
        problem = difference(call.python, got)
        if problem:
            failures.append(f'{where}: {problem}')
    assert not failures, '\n'.join(failures)
