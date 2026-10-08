# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P76.multilabel: a scenario records today's multilabel refusal, a
``LIST_DISTINCT`` over a foreign key linked to two tables, and the core's
``task_table`` run raises what the engine raises (``CONTRACT.md``,
"Multilabel").
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from functools import cache
from pathlib import Path
from typing import Any

from _link import (
    MULTILABEL_START,
    OFFLINE,
    PARITY_SCRIPTS,
    REPO,
    TRANSCRIPTS,
    Boundary,
    check_rfm_lines,
    compare_runs,
    model_describer,
    multilabel_text,
    multilabel_transcripts,
    query_of,
    read,
    target_column,
    task_table_got,
    task_table_want,
)

#: How ``sdm_client`` reports what the engine raised while it built the
#: request.
CLIENT_ERROR = 'sdm_client.errors.StructuredDataError'
CLIENT_CODE = 'INTERNAL_ERROR'


@cache
def found() -> dict[str, Path]:
    transcripts = multilabel_transcripts()
    assert transcripts, (
        'no multilabel transcript: none under tests/parity/transcripts asks '
        "a LIST_DISTINCT query without RANK and ends in the engine's "
        f'"{MULTILABEL_START}"'
    )
    return transcripts


def test_a_scenario_records_the_refusal() -> None:
    from _parity.scenarios import by_name

    known = by_name()
    problems = []
    for name, path in found().items():
        transcript = read(path)
        scenario = known.get(name)
        if scenario is None:
            problems.append(f'{name}: not a scenario of scenarios.py')
            continue
        if scenario.live or transcript['backend'] not in OFFLINE:
            problems.append(f'{name}: not an offline scenario')
        if path.parent.relative_to(TRANSCRIPTS).as_posix() != name.replace(
            '#', '--'
        ):
            problems.append(f'{name}: stored at {path}')
        if transcript.get('payloads'):
            problems.append(f'{name}: sent a payload before it was refused')
        query = query_of(transcript)
        assert query is not None
        _, column = target_column(query)
        error = transcript['error']
        want = f'NotImplementedError: {multilabel_text(column)}'
        if error.get('class') != CLIENT_ERROR:
            problems.append(f'{name}: raised {error.get("class")}')
        if error.get('code') != CLIENT_CODE:
            problems.append(f'{name}: error code {error.get("code")!r}')
        if not str(error.get('message', '')).endswith(want):
            problems.append(
                f'{name}: the message {error.get("message")!r} does not end '
                f'with {want!r}'
            )
    assert not problems, '\n'.join(problems)


def test_they_replay() -> None:
    from _parity.corpus import check

    problems = []
    for name, path in found().items():
        reason = check(path)
        if reason is not None:
            problems.append(f'{name}: {reason}')
    assert not problems, '\n'.join(problems[:10])


def test_recording_again_writes_the_committed_files() -> None:
    names = sorted(found())
    with tempfile.TemporaryDirectory(prefix='sdm-p76-multilabel-') as tmp:
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
            stored = found()[name].parent
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


def test_the_core_raises_what_the_engine_raises() -> None:
    r"""Replaying each multilabel transcript, today's ``_get_task_table``
    raises the multilabel ``NotImplementedError`` (rfm.py 2029->2030) for a
    target column with edges to two tables or more; the core's
    ``task_table`` run, driven from the same call, raises the same class and
    text."""
    from _driver import error_of
    from _parity.corpus import check
    from sdm_relational_engine.rfm.rfm import KumoRelational

    check_rfm_lines()
    boundary = Boundary(
        KumoRelational, '_get_task_table', lambda model: model._sampler
    )
    describe = model_describer('_get_task_table')
    edges: list[list[tuple[str, str, str]]] = []

    def described(model: Any, arguments: dict[str, Any]) -> Any:
        edges.append([tuple(e) for e in model._sampler.edge_types])
        return describe(model, arguments)

    reasons = []
    with boundary.installed(described):
        for path in found().values():
            before = len(boundary.calls)
            reason = check(path)
            if reason is not None:
                reasons.append(f'{path}: {reason}')
            for call in boundary.calls[before:]:
                call['transcript'] = path
    assert not reasons, reasons[:3]
    assert len(edges) == len(boundary.calls)
    for call, edge_types in zip(boundary.calls, edges, strict=True):
        call['edge_types'] = edge_types

    calls, problems = compare_runs(
        'task_table', boundary, task_table_want, task_table_got
    )
    assert not problems, '\n'.join(problems[:20])
    compared = 0
    for name, path in found().items():
        query = query_of(read(path))
        assert query is not None
        table, column = target_column(query)
        expected = {
            'code': 'NotImplementedError',
            'message': multilabel_text(column),
        }
        mine = [call for call in calls if call['transcript'] == path]
        assert mine, f'{name}: no _get_task_table call'
        last = mine[-1]
        assert last['error'] is not None and error_of(last['error']) == (
            expected
        ), (
            f"{name}: today's _get_task_table gave "
            f'{None if last["error"] is None else error_of(last["error"])}, '
            f'not {expected}'
        )
        targets = {
            dst
            for src, fkey, dst in last['edge_types']
            if (src, fkey) == (table, column)
        }
        assert len(targets) >= 2, (
            f'{name}: {table}.{column} links to {sorted(targets)}, not to two '
            'tables or more'
        )
        assert last['core'] == 'raised' and last['core_error'] == expected, (
            f"{name}: the core's task_table run gave {last['core']} "
            f'{last["core_error"]}, not {expected}'
        )
        compared += 1
    assert compared >= 1
