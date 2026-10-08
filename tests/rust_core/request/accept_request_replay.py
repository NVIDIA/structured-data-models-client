# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Every transcript replays through the host, and so through the core
(``CONTRACT.md``, "Replaying a transcript").

Each test reads the same replays, and holds them to one exit criterion:

- P85.payloads: every payload, every session body and every golden payload;
- P85.sql: every statement of every relational transcript, byte for byte;
- P85.results: the frame, the layout and what the result holds beyond it;
- P85.decompose: the rankings and forecasts, which the core decomposes;
- P85.sessions: the transcripts that fit and score a session;
- P85.diagnostics: every transcript's warnings and error;
- task 6: frames reach the core as Arrow, through the describer.

Every test also holds the host to the rules every replay keeps: no ported
Python ran beneath it, every run finished, and what crossed the binding was
plain data.
"""

from __future__ import annotations

from typing import Any

import pytest
from _request import (
    HOST_ASPECTS,
    IN_MEMORY,
    replay_core,
    today_frame,
    transcripts,
)

#: The fewest transcripts the corpus holds (Phase 8 recorded 123).
MIN_TRANSCRIPTS = 120


@pytest.fixture(scope='module')
def reports() -> list[Any]:
    paths = transcripts()
    assert len(paths) >= MIN_TRANSCRIPTS, f'only {len(paths)} transcripts'
    out = []
    for path in paths:
        today = today_frame(path)
        out.append(replay_core(path, today))
    return out


def _held(reports: list[Any], which: Any, *aspects: str) -> list[str]:
    chosen = [r for r in reports if which(r)]
    assert chosen, 'no transcript is of this kind'
    return [p for r in chosen for p in r.about(*HOST_ASPECTS, *aspects)]


def _fail(problems: list[str]) -> None:
    assert not problems, f'{len(problems)} problem(s):\n' + '\n'.join(
        problems[:60]
    )


def test_payloads(reports: list[Any]) -> None:
    r"""P85.payloads, through the binding: every payload sent, every session
    body, and the golden payloads, parsed equal; every request to the model
    is the stored one and is a call the core asked for."""
    _fail(
        _held(
            reports,
            lambda r: True,
            'payloads',
            'golden',
            'requests',
            'statements',
        )
    )
    from _request import golden_payloads

    names = {r.name for r in reports}
    missing = sorted(set(golden_payloads()) - names)
    assert not missing, f'golden scenarios without a transcript: {missing}'


def test_sql(reports: list[Any]) -> None:
    r"""P85.sql: each relational transcript's statements, byte for byte, in
    order, each one a query the core asked for, and every one taken."""
    _fail(
        _held(
            reports,
            lambda r: r.backend not in IN_MEMORY,
            'steps',
            'statements',
        )
    )


def test_results(reports: list[Any]) -> None:
    r"""P85.results: the layout equals ``result.arrow``, ``frame(layout)``
    equals today's frame, and the result's answer beyond the frame is
    today's."""
    _fail(
        _held(
            reports,
            lambda r: True,
            'result',
            'layout',
            'frame',
            'answer',
            'error',
        )
    )
    framed = [r for r in reports if r.layout is not None]
    assert len(framed) >= 100, f'only {len(framed)} transcripts made a frame'


def test_decompose(reports: list[Any]) -> None:
    r"""P85.decompose: every ranking and forecast transcript, with the v2
    stand-in: the order, top-k, unscored candidates, forecast steps and
    values, and every payload and request it took."""
    problems = _held(
        reports,
        lambda r: r.task_kind in ('ranking', 'forecast'),
        'payloads',
        'requests',
        'steps',
        'statements',
        'result',
        'layout',
        'frame',
        'answer',
        'error',
        'warnings',
    )
    _fail(problems)
    kinds = {r.task_kind for r in reports}
    assert {'ranking', 'forecast'} <= kinds, kinds


def test_sessions(reports: list[Any]) -> None:
    r"""P85.sessions: the transcripts that fit a session: the session's
    bodies, holdout and warnings, and everything else they replay."""
    sessions = [r for r in reports if 'SDMClient.fit' in r.calls]
    assert len(sessions) >= 8, f'only {len(sessions)} session transcripts'
    _fail(
        _held(
            sessions,
            lambda r: True,
            'payloads',
            'requests',
            'steps',
            'statements',
            'result',
            'layout',
            'frame',
            'answer',
            'error',
            'warnings',
            'warnings-source',
        )
    )


def test_warnings_and_errors(reports: list[Any]) -> None:
    r"""P85.diagnostics, over the transcripts: every warning, in order, with
    its category, and every error with its class, code, message and
    details, each given by the core."""
    _fail(
        _held(
            reports,
            lambda r: True,
            'warnings',
            'warnings-source',
            'error',
            'error-source',
        )
    )
    assert any(r.error for r in reports), 'no transcript ends in an error'


def test_inputs(reports: list[Any]) -> None:
    r"""Task 6: the frames of every in-memory transcript reach the core as
    ``describe_frames`` gives them, and no other table is passed to it."""
    in_memory = [r for r in reports if r.backend in IN_MEMORY]
    assert len(in_memory) >= 20, f'only {len(in_memory)} in-memory transcripts'
    _fail(_held(reports, lambda r: True, 'inputs'))
