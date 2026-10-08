# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P8.nim-v2: the stand-in model answers each ranking candidate and each
forecast step differently, every response is stored in its transcript, and
every transcript replays from what it stores.

- Every transcript that sent a payload stores ``responses``; each transcript
  of BASE, all of which only predict, stores one per payload. ``FORMAT.md``
  documents the key.
- In every ranking transcript of BASE, live ones included, each row ranks its
  candidates with pairwise different scores, the parts' responses differ, and
  in at least one row the order is not the order the candidates were named.
- In every forecast transcript of BASE, each row's steps predict pairwise
  different values.
- ``run_corpus.py replay`` passes every transcript on disk, and none of
  BASE's is gone.
- A replay answers from the stored responses rather than computing them: in a
  copy of a regression transcript, changing each predicted value in both the
  stored responses and ``result.arrow`` still replays, and changing it in the
  responses alone does not.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
from _net import (
    FORMAT,
    TRANSCRIPTS,
    base_transcripts,
    calls_named,
    passed,
    read_arrow,
    replay,
    result_rows,
    transcripts,
)

#: Regression transcripts of BASE the replay is checked to answer from.
TAMPERED = ('sqlite/regression', 'tabular/direct-regression')
SHIFT = 0.5


@pytest.fixture(scope='module')
def base() -> dict[str, Any]:
    return base_transcripts()


@pytest.fixture(scope='module')
def stored() -> dict[str, tuple[Path, Any]]:
    return transcripts()


def test_every_transcript_replays(
    base: dict[str, Any], stored: dict[str, tuple[Path, Any]]
) -> None:
    gone = sorted(set(base) - set(stored))
    assert not gone, f'transcripts of BASE are gone: {gone}'
    code, output = replay([])
    failed = [line for line in output.splitlines() if line.startswith('FAIL')]
    assert code == 0 and not failed, '\n'.join(failed) or output[-4000:]
    unreplayed = sorted(set(stored) - passed(output))
    assert not unreplayed, f'not replayed: {unreplayed}'


def test_every_response_is_stored(
    base: dict[str, Any], stored: dict[str, tuple[Path, Any]]
) -> None:
    problems = []
    for name, (_, data) in sorted(stored.items()):
        payloads = data.get('payloads') or []
        responses = data.get('responses')
        if not payloads:
            continue
        if not isinstance(responses, list) or not responses:
            problems.append(
                f'{name}: sent {len(payloads)} payloads, stores no responses'
            )
            continue
        if name in base and len(responses) != len(payloads):
            problems.append(
                f'{name}: {len(payloads)} payloads, {len(responses)} responses'
            )
    assert not problems, '\n'.join(problems)


def test_the_format_documents_responses() -> None:
    text = FORMAT.read_text(encoding='utf-8')
    assert re.search(r'^\|\s*`responses`\s*\|', text, re.M), (
        'FORMAT.md has no row for the transcript key `responses`'
    )


def _floats(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, float):
        return [value]
    if isinstance(value, dict):
        return [f for v in value.values() for f in _floats(v)]
    if isinstance(value, list):
        return [f for v in value for f in _floats(v)]
    if isinstance(value, str):
        try:
            return _floats(json.loads(value))
        except ValueError:
            return []
    return []


def test_ranking_candidates_get_distinct_answers(
    base: dict[str, Any], stored: dict[str, tuple[Path, Any]]
) -> None:
    names = sorted(
        n for n, d in base.items() if d.get('task_kind') == 'ranking'
    )
    assert len(names) >= 5, f'only {len(names)} ranking transcripts at BASE'
    problems = []
    reordered = 0
    for name in names:
        directory, data = stored[name]
        rows = result_rows(directory, data) or []
        if not rows:
            problems.append(f'{name}: no result rows')
            continue
        materialized = calls_named(data, 'PredictiveQuery.materialize')
        named = list(materialized[0]['args'].get('candidates') or [])
        for row in rows:
            scores = [entry['score'] for entry in row.get('rankings') or []]
            if len(scores) < 2 or len(set(scores)) != len(scores):
                problems.append(
                    f'{name}: row {row.get("row_index")} ranks with scores '
                    f'{scores}'
                )
            prediction = list(row.get('prediction') or [])
            if prediction and prediction != named[: len(prediction)]:
                reordered += 1
        answers = [
            tuple(sorted(_floats(r))) for r in data.get('responses') or []
        ]
        if len(answers) < 2 or len(set(answers)) != len(answers):
            problems.append(
                f"{name}: the {len(answers)} candidates' responses are not "
                f'pairwise different'
            )
    assert not problems, '\n'.join(problems[:40])
    assert reordered, (
        'every ranking keeps the order its candidates were named in, so the '
        'corpus cannot tell a sorted ranking from an unsorted one'
    )


def test_forecast_steps_get_distinct_answers(
    base: dict[str, Any], stored: dict[str, tuple[Path, Any]]
) -> None:
    names = sorted(
        n for n, d in base.items() if d.get('task_kind') == 'forecast'
    )
    assert len(names) >= 5, f'only {len(names)} forecast transcripts at BASE'
    problems = []
    for name in names:
        directory, data = stored[name]
        series: dict[Any, list[Any]] = defaultdict(list)
        for row in result_rows(directory, data) or []:
            series[(row.get('row_index'), row.get('id'))].append(
                row.get('prediction')
            )
        if not series:
            problems.append(f'{name}: no result rows')
        for key, values in series.items():
            if len(values) < 2 or len(set(values)) != len(values):
                problems.append(f'{name}: {key} predicts {values} by step')
    assert not problems, '\n'.join(problems[:40])


def _shift(value: Any, targets: set[float], hits: list[int]) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value in targets:
        hits.append(1)
        return value + SHIFT
    if isinstance(value, dict):
        return {k: _shift(v, targets, hits) for k, v in value.items()}
    if isinstance(value, list):
        return [_shift(v, targets, hits) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        if isinstance(parsed, (dict, list)):
            before = len(hits)
            shifted = _shift(parsed, targets, hits)
            return json.dumps(shifted) if len(hits) > before else value
    return value


def _shift_result(path: Path, targets: set[float]) -> int:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    table = read_arrow(path)
    hits = 0
    columns = []
    for column, field in zip(table.columns, table.schema, strict=True):
        if pa.types.is_floating(field.type):
            values = column.to_pylist()
            shifted = [
                v + SHIFT if isinstance(v, float) and v in targets else v
                for v in values
            ]
            hits += sum(a != b for a, b in zip(values, shifted, strict=True))
            column = pa.chunked_array([pa.array(shifted, type=field.type)])
        columns.append(column)
    out = pa.Table.from_arrays(columns, schema=table.schema)
    with (
        pa.OSFile(str(path), 'wb') as sink,
        ipc.new_file(sink, out.schema) as w,
    ):
        w.write_table(out)
    return hits


@pytest.mark.parametrize('name', TAMPERED)
def test_a_replay_answers_from_the_stored_responses(
    name: str,
    base: dict[str, Any],
    stored: dict[str, tuple[Path, Any]],
    tmp_path: Path,
) -> None:
    assert name in base
    directory, data = stored[name]
    relative = directory.relative_to(TRANSCRIPTS)

    def copy(where: str) -> Path:
        corpus = tmp_path / where
        shutil.copytree(directory, corpus / relative)
        return corpus

    corpus = copy('unchanged')
    code, output = replay([name], corpus)
    assert code == 0 and name in passed(output), (
        f'an unchanged copy of {name} does not replay:\n{output[-3000:]}'
    )

    rows = result_rows(directory, data) or []
    targets = {
        row['prediction']
        for row in rows
        if isinstance(row.get('prediction'), float)
    }
    assert targets, f'{name}: no float prediction in the result'

    for where, shift_result in (('both', True), ('responses-only', False)):
        corpus = copy(where)
        path = corpus / relative / 'transcript.json'
        transcript = json.loads(path.read_text(encoding='utf-8'))
        hits: list[int] = []
        transcript['responses'] = _shift(
            transcript.get('responses'), targets, hits
        )
        assert hits, (
            f'{name}: no stored response holds a predicted value of the '
            f'result, so the replay cannot be answering from them'
        )
        path.write_text(json.dumps(transcript, indent=1), encoding='utf-8')
        if shift_result:
            assert _shift_result(corpus / relative / data['result'], targets)
        code, output = replay([name], corpus)
        if shift_result:
            assert code == 0 and name in passed(output), (
                f'{name} with its responses and result changed alike does '
                f'not replay, so the replay does not answer from the stored '
                f'responses:\n{output[-3000:]}'
            )
        else:
            assert code != 0 and name not in passed(output), (
                f'{name} replays with its stored responses changed and its '
                f'result not, so the replay ignores the stored responses'
            )
