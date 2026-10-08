# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P76.boundary: the core's sampling driver gives what today's Python gives
on every link-prediction scenario (``CONTRACT.md``, "Boundary").

- The scenarios' boundary recordings exist and hold link prediction's
  shapes: list labels from ``sample_target``, and a subgraph that samples
  the target entities.
- Driven from those recordings alone, in Phase 7.5's independent process,
  the core asks for exactly the recorded statements and gives exactly the
  recorded outputs.
- The core's ``task_table`` and ``context`` runs give what today's
  ``_get_task_table`` and ``_get_context`` give on the same transcripts, and
  raise what they raise: the "no historical target entities" ``ValueError``
  where no target entity is sampled.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from _link import (
    NO_TARGET_ENTITIES,
    Boundary,
    compare_runs,
    context_got,
    context_want,
    drive_recordings,
    is_refused,
    link_transcripts,
    read,
    recording_dir_of,
    recording_problems,
    replay_with,
    task_table_got,
    task_table_want,
)


@cache
def found() -> dict[str, Path]:
    transcripts = link_transcripts()
    assert transcripts, 'no link-prediction transcript'
    return transcripts


def _recording(name: str) -> dict[str, Any]:
    from _driver import read_recording

    directory = recording_dir_of(found()[name])
    assert (directory / 'boundary.json').is_file(), (
        f'{name}: no boundary recording at {directory}'
    )
    recording = read_recording(directory)
    assert recording['scenario'] == name, (
        f'{directory} records {recording["scenario"]!r}, not {name!r}'
    )
    return recording


def test_the_recordings_hold_link_prediction() -> None:
    problems = []
    for name, path in found().items():
        try:
            recording = _recording(name)
        except AssertionError as missing:
            problems.append(str(missing))
            continue
        calls = recording['calls']
        methods = {call['method'] for call in calls}
        if not {'sample_target', 'sample_subgraph'} <= methods:
            problems.append(f'{name}: calls {sorted(methods)}')
            continue
        transcript = read(path)
        if is_refused(transcript):
            continue
        targets = [
            call['outputs'].get('train', {}).get('target', {})
            for call in calls
            if call['method'] == 'sample_target'
        ]
        if not any(
            str(t.get('type', '')).startswith('list<') and t.get('values')
            for t in targets
        ):
            problems.append(
                f'{name}: no sample_target gives list labels: '
                f'{[t.get("type") for t in targets]}'
            )
        wanted = set(transcript['payloads'][0]['task']['entity_table_names'])
        sampled = [
            {table['name'] for table in call['outputs'].get('tables', [])}
            for call in calls
            if call['method'] == 'sample_subgraph'
        ]
        if not any(wanted <= tables for tables in sampled):
            problems.append(
                f'{name}: no sample_subgraph samples {sorted(wanted)}: {sampled}'
            )
    assert not problems, '\n'.join(problems[:20])


@cache
def _driven() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    r"""The link recordings, and the core driven over them by
    ``_driver_run.py`` in a process without today's Python."""
    return drive_recordings(found())


def test_the_core_matches_the_recordings() -> None:
    recordings, got = _driven()
    problems, compared = recording_problems(recordings, got)
    assert not problems, (
        f"{len(problems)} link-prediction calls differ from today's Python:\n"
        + '\n'.join(problems[:25])
    )
    assert compared >= 2 * len(recordings), f'only {compared} calls compared'


def test_task_tables() -> None:
    from sdm_relational_engine.rfm.rfm import KumoRelational

    boundary = Boundary(
        KumoRelational, '_get_task_table', lambda model: model._sampler
    )
    reasons = replay_with(boundary, '_get_task_table', list(found().values()))
    assert not reasons, reasons[:3]
    calls, problems = compare_runs(
        'task_table', boundary, task_table_want, task_table_got
    )
    assert not problems, '\n'.join(problems[:20])
    by_transcript: dict[Path, list[dict[str, Any]]] = {}
    for call in calls:
        by_transcript.setdefault(call['transcript'], []).append(call)
    missing = [
        name for name, path in found().items() if path not in by_transcript
    ]
    assert not missing, f'no _get_task_table call in {missing}'
    wrong = [
        f'{call["transcript"]}: {call["result"].task_type}'
        for call in calls
        if call['error'] is None
        and str(call['result'].task_type.value) != 'temporal_link_prediction'
    ]
    assert not wrong, f'tasks that are not link prediction: {wrong}'


def test_contexts_and_the_refusal() -> None:
    from _driver import error_of
    from sdm_relational_engine.rfm.rfm import KumoRelational

    boundary = Boundary(
        KumoRelational, '_get_context', lambda model: model._sampler
    )
    reasons = replay_with(boundary, '_get_context', list(found().values()))
    assert not reasons, reasons[:3]
    calls, problems = compare_runs(
        'context', boundary, context_want, context_got
    )
    assert not problems, '\n'.join(problems[:20])
    expected = {'code': 'ValueError', 'message': NO_TARGET_ENTITIES}
    refusals = 0
    for name, path in found().items():
        mine = [call for call in calls if call['transcript'] == path]
        assert mine, f'{name}: no _get_context call'
        if not is_refused(read(path)):
            assert all(call['error'] is None for call in mine), name
            assert all(call['core'] == 'done' for call in mine), name
            continue
        raised = [call for call in mine if call['error'] is not None]
        assert raised and error_of(raised[-1]['error']) == expected, (
            f"{name}: today's _get_context raised "
            f'{[error_of(c["error"]) for c in raised]}, not {expected}'
        )
        assert (
            raised[-1]['core'] == 'raised'
            and raised[-1]['core_error'] == expected
        ), (
            f"{name}: the core's context run gave {raised[-1]['core']} "
            f'{raised[-1]["core_error"]}, not {expected}'
        )
        refusals += 1
    assert refusals >= 1, 'no refused link-prediction transcript was compared'
