# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P76.scenarios: the link-prediction scenarios exist on every offline
backend, were recorded by the recorder and answered by Phase 8's stand-in,
replay, and send ``TEMPORAL_LINK_PREDICTION`` requests; one is refused for
sampling no target entity (``CONTRACT.md``, "Scenarios").
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
    NO_TARGET_ENTITIES,
    OFFLINE,
    PARITY_SCRIPTS,
    REPO,
    TRANSCRIPTS,
    entity_table,
    is_refused,
    link_transcripts,
    query_of,
    read,
)


@cache
def found() -> dict[str, Path]:
    transcripts = link_transcripts()
    assert transcripts, (
        'no link-prediction transcript: none under tests/parity/transcripts '
        'asks a LIST_DISTINCT query without RANK'
    )
    return transcripts


def _completed() -> dict[str, dict[str, Any]]:
    out = {}
    for name, path in found().items():
        transcript = read(path)
        if not is_refused(transcript):
            out[name] = transcript
    return out


def _payload_problems(transcript: dict[str, Any]) -> list[str]:
    problems = []
    query = query_of(transcript)
    assert query is not None
    entity = entity_table(query)
    payloads = transcript.get('payloads') or []
    if not payloads:
        return ['sent no payload']
    for i, payload in enumerate(payloads):
        where = f'payload {i}'
        task = payload.get('task') or {}
        if task.get('kind') != 'temporal_link_prediction':
            problems.append(f'{where}: task kind {task.get("kind")!r}')
            continue
        names = task.get('entity_table_names')
        if not (
            isinstance(names, list)
            and len(names) == 2
            and names[0] == entity
            and names[1] != entity
        ):
            problems.append(f'{where}: entity_table_names {names!r}')
        fields = (payload.get('output') or {}).get('fields') or []
        if 'rankings' not in fields:
            problems.append(f'{where}: output fields {fields!r}')
        target = (task.get('target') or {}).get('column_name')
        context = (payload.get('context') or {}).get('instance_table') or {}
        columns = context.get('columns') or []
        if target not in columns:
            problems.append(f'{where}: the context has no {target!r} column')
        else:
            at = columns.index(target)
            labels = [row[at] for row in context.get('rows') or []]
            if not labels or not all(isinstance(v, list) for v in labels):
                problems.append(f'{where}: context labels are not all lists')
            elif not any(labels):
                problems.append(f'{where}: every context label is empty')
        predict = (payload.get('predict') or {}).get('instance_table') or {}
        if target in (predict.get('columns') or []):
            problems.append(f'{where}: the predicted rows carry {target!r}')
        if not predict.get('rows'):
            problems.append(f'{where}: no predicted rows')
    return problems


def _response_problems(transcript: dict[str, Any]) -> list[str]:
    payloads = transcript.get('payloads') or []
    responses = transcript.get('responses') or []
    if len(responses) != len(payloads):
        return [f'{len(responses)} responses for {len(payloads)} payloads']
    problems = []
    for i, response in enumerate(responses):
        if not isinstance(response, dict) or response.get('status') != 200:
            problems.append(f'response {i} is not a 200')
            continue
        items = (response.get('body') or {}).get('predictions') or []
        if not items:
            problems.append(f'response {i} predicts nothing')
        for item in items:
            if not item.get('rankings'):
                problems.append(f'response {i}: an item has no rankings')
                break
    return problems


def _result_problems(path: Path, transcript: dict[str, Any]) -> list[str]:
    import pyarrow.ipc as ipc

    result = path.parent / 'result.arrow'
    if not result.is_file():
        return ['no result.arrow']
    with result.open('rb') as handle:
        table = ipc.open_file(handle).read_all()
    if 'rankings' not in table.column_names:
        return [f'the result has no rankings column: {table.column_names}']
    predicted = sum(
        len(
            ((p.get('predict') or {}).get('instance_table') or {}).get('rows')
            or []
        )
        for p in transcript.get('payloads') or []
    )
    rankings = table.column('rankings').to_pylist()
    problems = []
    if len(rankings) != predicted:
        problems.append(
            f'{len(rankings)} result rows for {predicted} predicted'
        )
    if not all(rankings):
        problems.append('a result row has no ranking')
    return problems


def test_every_offline_backend_has_a_completed_link_prediction() -> None:
    completed = _completed()
    backends = {t['backend'] for t in completed.values()}
    missing = [b for b in OFFLINE if b not in backends]
    assert not missing, (
        f'no completed link-prediction transcript on {missing}; '
        f'found {sorted(completed)}'
    )


def test_the_payloads_are_link_prediction_requests() -> None:
    problems = []
    for name, transcript in _completed().items():
        problems += [f'{name}: {p}' for p in _payload_problems(transcript)]
    assert not problems, '\n'.join(problems[:30])


def test_the_stand_in_answered_with_rankings() -> None:
    problems = []
    for name, transcript in _completed().items():
        problems += [f'{name}: {p}' for p in _response_problems(transcript)]
        problems += [
            f'{name}: {p}' for p in _result_problems(found()[name], transcript)
        ]
    assert not problems, '\n'.join(problems[:30])


def test_one_is_refused_for_sampling_no_target_entity() -> None:
    refused = []
    for name, path in found().items():
        transcript = read(path)
        error = transcript.get('error')
        if error is None:
            continue
        message = str(error.get('message', ''))
        if message.endswith(NO_TARGET_ENTITIES):
            assert not transcript.get('payloads'), (
                f'{name} sent a payload before it was refused'
            )
            refused.append(name)
    assert refused, (
        'no link-prediction transcript ends in the "no historical target '
        f'entities" error; found {sorted(found())}'
    )


def test_they_are_offline_corpus_scenarios() -> None:
    from _parity.scenarios import by_name

    known = by_name()
    problems = []
    for name, path in found().items():
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
    from _parity.corpus import check

    problems = []
    for name, path in found().items():
        reason = check(path)
        if reason is not None:
            problems.append(f'{name}: {reason}')
    assert not problems, '\n'.join(problems[:10])


def test_recording_again_writes_the_committed_files() -> None:
    names = sorted(found())
    with tempfile.TemporaryDirectory(prefix='sdm-p76-record-') as tmp:
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
