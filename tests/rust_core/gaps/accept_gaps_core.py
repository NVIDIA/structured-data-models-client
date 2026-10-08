# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P77.scenarios, the core side: over the scenarios that take the three
branches, the core asks today's statements and gives today's sampling
outputs, task tables, contexts, payloads, results and errors
(``CONTRACT.md``, "The core").

``accept_gaps_scenarios.py`` finds the scenarios and holds them to the
corpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _gaps import (
    INDICES_REFUSAL,
    REQUEST_RECORDINGS,
    SAMPLING_RECORDINGS,
    SELF_LINK_REFUSAL,
    describe_found,
    found,
    read,
    recording_dir,
)


def _every() -> dict[str, Path]:
    every = found().every()
    f = found()
    assert f.sqlite and f.indices and f.self_link, (
        f'a kind of scenario is missing: {describe_found()}'
    )
    return every


def test_the_boundary_recordings() -> None:
    r"""Each has its sampling boundary recording, written by
    ``capture_sampling.py``; the self link's holds the engine's refusal as
    the ``sample_target`` call's outcome."""
    problems = []
    for name, path in _every().items():
        directory = recording_dir(SAMPLING_RECORDINGS, path)
        file = directory / 'boundary.json'
        if not file.is_file():
            problems.append(f'{name}: no {file}')
            continue
        recording = json.loads(file.read_text('utf-8'))
        if recording.get('scenario') != name:
            problems.append(f'{name}: {file} records {recording["scenario"]}')
        calls = recording.get('calls') or []
        if not calls:
            problems.append(f'{name}: {file} records no call')
            continue
        if name in found().sqlite and not any(
            c['method'] == 'sample_target' and c['steps'] for c in calls
        ):
            problems.append(f'{name}: no sample_target call ran a statement')
        if name in found().self_link and not any(
            c['method'] == 'sample_target'
            and c['outputs'].get('error', {}).get('code') == 'ValueError'
            and SELF_LINK_REFUSAL.search(
                str(c['outputs']['error'].get('message', ''))
            )
            for c in calls
        ):
            problems.append(
                f'{name}: no sample_target call records the refusal of a '
                'table linked to itself'
            )
    assert not problems, '\n'.join(problems)


def test_the_core_matches_the_recordings() -> None:
    r"""The core, driven from the recordings alone in Phase 7.5's
    independent process, asks the recorded statements and gives the
    recorded outputs or errors, call by call."""
    from _link import drive_recordings, recording_problems

    every = _every()
    recordings, got = drive_recordings(every)
    problems, compared = recording_problems(recordings, got)
    assert not problems, (
        f"{len(problems)} calls differ from today's Python:\n"
        + '\n'.join(problems[:25])
    )
    total = sum(len(r['calls']) for r in recordings.values())
    assert compared == total and total >= len(every), (
        f'{compared} of {total} calls compared'
    )


def _runs(method: str, run: str, paths: list[Path]) -> list[dict[str, Any]]:
    from _link import (
        Boundary,
        compare_runs,
        context_got,
        context_want,
        replay_with,
        task_table_got,
        task_table_want,
    )
    from sdm_relational_engine.rfm.rfm import KumoRelational

    want, got = {
        'task_table': (task_table_want, task_table_got),
        'context': (context_want, context_got),
    }[run]
    boundary = Boundary(KumoRelational, method, lambda m: m._sampler)
    reasons = replay_with(boundary, method, paths)
    assert not reasons, reasons[:3]
    calls, problems = compare_runs(run, boundary, want, got)
    assert not problems, '\n'.join(problems[:20])
    return calls


def _of(calls: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    return [call for call in calls if call.get('transcript') == path]


def test_task_tables_and_contexts() -> None:
    r"""The core's ``task_table`` and ``context`` runs, driven from today's
    calls as Phase 7.6's checks drive them, give what today's
    ``_get_task_table`` and ``_get_context`` give or raise what they raise:
    the SQLite ones finish; the mixed indices and the self link raise
    today's ``ValueError`` with today's text."""
    _every()
    f = found()
    paths = [*f.sqlite.values(), *f.indices.values(), *f.self_link.values()]
    tables = _runs('_get_task_table', 'task_table', paths)
    contexts = _runs('_get_context', 'context', list(f.sqlite.values()))
    problems = []
    for name, path in f.sqlite.items():
        for what, calls in (('task_table', tables), ('context', contexts)):
            mine = _of(calls, path)
            if not mine:
                problems.append(f'{name}: no {what} call')
            elif not all(call['core'] == 'done' for call in mine):
                problems.append(
                    f"{name}: the core's {what} run did not finish: "
                    f'{[call["core_error"] for call in mine]}'
                )
    for kind, pattern in (
        (f.indices, INDICES_REFUSAL),
        (f.self_link, SELF_LINK_REFUSAL),
    ):
        for name, path in kind.items():
            raised = [
                call
                for call in _of(tables, path)
                if call['error'] is not None
                and type(call['error']).__name__ == 'ValueError'
                and pattern.search(str(call['error']))
            ]
            if not raised:
                problems.append(
                    f"{name}: today's _get_task_table did not raise the refusal"
                )
            elif not all(call['core'] == 'raised' for call in raised):
                problems.append(f"{name}: the core's task_table run finished")
            elif any('; indices[' in str(call['error']) for call in raised):
                problems.append(f'{name}: the refusal names an index')
    assert not problems, '\n'.join(problems)


def test_payloads_from_the_core_outputs() -> None:
    r"""As Phase 7.5's ``accept_driver_payload.py``, over these: with every
    sampling call answered by the core, each replays as recorded."""
    from _driver import (
        METHODS,
        Interceptor,
        capture_module,
        native,
        replaying,
    )

    core = native(*METHODS, 'Outcome')
    capture = capture_module()
    interceptor = Interceptor(capture.describe, core, capture.to_python)
    problems = []
    with replaying(interceptor) as replay:
        for name, path in _every().items():
            reason, calls = replay(path)
            if reason is not None:
                problems.append(f'{name}: {reason}')
            elif not calls:
                problems.append(f'{name}: no call went to the core')
    assert not problems, '\n'.join(problems[:25])


def test_the_host_replays_them() -> None:
    r"""Each replays through Phase 8.5's host (``scripts/parity/
    request_host.py``), and so the core: today's statements, payloads,
    requests, result, warnings, and error with its class, code and
    message, each given by the core; and each has its request recording."""
    from _request import replay_core, today_frame

    problems = []
    refused = 0
    for name, path in _every().items():
        directory = recording_dir(REQUEST_RECORDINGS, path)
        if not (directory / 'calls.json').is_file():
            problems.append(f'{name}: no request recording at {directory}')
        report = replay_core(path, today_frame(path))
        problems += [f'{name}: [{a}] {m}' for a, m in report.problems]
        transcript = read(path)
        if transcript.get('error') is not None:
            if report.error is None:
                problems.append(f'{name}: the host gave no error')
            else:
                refused += 1
        elif not report.payloads:
            problems.append(f'{name}: the host sent no payload')
    assert not problems, '\n'.join(problems[:40])
    assert refused >= 2, f'only {refused} refusals compared'
