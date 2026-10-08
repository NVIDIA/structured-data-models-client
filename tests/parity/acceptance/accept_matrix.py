# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P2.matrix: every offline cell recorded, every tabular case included,
every live cell defined and recorded.

Phase 2 defined the live cells and left them pending; Phase 5.6 recorded them
on the warehouses. So a live cell is now held to the offline cell's checks,
and its recordings must also be scrubbed and replay offline.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
from _support import (
    GOLDEN,
    LIVE_BACKENDS,
    OFFLINE_BACKENDS,
    REPO,
    RUN_CORPUS,
    SCRUB,
    TASK_KINDS,
    listing,
    load,
    offline_env,
    output,
    replay_lines,
    run,
)

# What a question of each kind looks like in PQL: the shop questions the the Databricks validation workspace
# notebooks ask.
_PQL = {
    'count': re.compile(r'\bCOUNT\s*\(', re.IGNORECASE),
    'forecast': re.compile(r'\bFORECAST\s+3\s+TIMEFRAMES?\b', re.IGNORECASE),
    'ranking': re.compile(r'\bRANK\s+TOP\s+\d+', re.IGNORECASE),
    'inactivity': re.compile(
        r'\bCOUNT\s*\([^)]*\)\s*(=|==)\s*0\b', re.IGNORECASE
    ),
}
_WIRE_KINDS = {
    'classification': {
        'classification',
        'binary_classification',
        'multiclass_classification',
    },
    'regression': {'regression'},
}


@pytest.fixture(scope='module')
def entries() -> list[dict[str, Any]]:
    return listing()


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _cell(
    entries: list[dict[str, Any]], backend: str, kind: str, status: str
) -> list[dict[str, Any]]:
    return [
        e
        for e in entries
        if e['backend'] == backend
        and e['task_kind'] == kind
        and e['status'] == status
    ]


@pytest.mark.parametrize('backend', OFFLINE_BACKENDS)
@pytest.mark.parametrize('kind', TASK_KINDS)
def test_offline_cell_is_recorded(
    entries: list[dict[str, Any]], backend: str, kind: str
) -> None:
    _check_cell(entries, backend, kind)


def _check_cell(
    entries: list[dict[str, Any]], backend: str, kind: str
) -> list[dict[str, Any]]:
    r"""Check a cell's recordings ask its question; return the cell."""
    cell = _cell(entries, backend, kind, 'recorded')
    assert cell, f'no recorded {kind} scenario on {backend}'
    posted = []
    for entry in cell:
        transcript = load(REPO / entry['transcript'])
        assert transcript['name'] == entry['name']
        assert transcript['backend'] == backend
        assert transcript['task_kind'] == kind
        # A session posts its bodies to the session routes, which the
        # transcript keeps under responses[].request. A run refused before
        # anything is sent records that error instead.
        sent = _sent(transcript)
        refused = not sent and _refused_before_sending(transcript)
        assert sent or refused, (
            f'{entry["name"]} sent nothing and records no refusal'
        )
        posted += transcript['payloads']
        if backend != 'memory':
            assert any(
                step['op'] == 'execute' for step in transcript['steps']
            ), f'{entry["name"]} ran no SQL'
        text = '\n'.join(_strings(transcript['calls']))
        if kind in _PQL:
            assert _PQL[kind].search(text), (
                f'{entry["name"]} does not ask a {kind} question'
            )
        if kind in _WIRE_KINDS:
            wire = {p['task']['kind'] for p in sent if 'task' in p}
            assert wire <= _WIRE_KINDS[kind], (entry['name'], wire)
    assert posted, f'no {kind} scenario on {backend} posts a prediction'
    return cell


def _sent(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    r"""The bodies the run posted: its payloads, or its session requests."""
    requests = [
        response['request']
        for response in transcript.get('responses', [])
        if isinstance(response.get('request'), dict)
    ]
    return transcript['payloads'] or requests


def _refused_before_sending(transcript: dict[str, Any]) -> bool:
    error = transcript.get('error')
    return (
        not transcript.get('responses')
        and transcript.get('result') is None
        and isinstance(error, dict)
        and isinstance(error.get('message'), str)
        and bool(error['message'])
    )


@pytest.fixture(scope='module')
def live_replay(
    entries: list[dict[str, Any]], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, tuple[str, str, str]]:
    r"""``{name: (PASS|FAIL, reason, log)}`` for every live recording, replayed
    with no credentials, no database and no network."""
    names = [
        e['name']
        for e in entries
        if e['backend'] in LIVE_BACKENDS and e['status'] == 'recorded'
    ]
    assert names, 'no live scenario is recorded'
    env = offline_env(tmp_path_factory.mktemp('live-replay') / 'home')
    proc = run(
        RUN_CORPUS, 'replay', '--only', ','.join(names), env=env, timeout=900
    )
    lines = replay_lines(proc.stdout)
    assert set(lines) == set(names), output(proc)
    return {name: (*lines[name], output(proc)) for name in names}


def _scan(paths: list[Path]) -> subprocess.CompletedProcess[str]:
    return run(SCRUB, 'scan', *paths, timeout=300)


@pytest.mark.parametrize('backend', LIVE_BACKENDS)
@pytest.mark.parametrize('kind', TASK_KINDS)
def test_live_cell_is_recorded_scrubbed_and_replays(
    entries: list[dict[str, Any]],
    live_replay: dict[str, tuple[str, str, str]],
    backend: str,
    kind: str,
) -> None:
    cell = _check_cell(entries, backend, kind)
    for entry in cell:
        verdict, reason, log = live_replay[entry['name']]
        assert verdict == 'PASS', f'{entry["name"]}: {reason}\n{log}'
    proc = _scan([(REPO / e['transcript']).parent for e in cell])
    assert proc.returncode == 0, output(proc)


def test_no_live_scenario_is_left_pending(
    entries: list[dict[str, Any]],
) -> None:
    r"""Phase 5.6 recorded every live scenario; one defined since must be
    recorded too, or this matrix has a cell that is only a definition."""
    live = [e for e in entries if e['backend'] in LIVE_BACKENDS]
    assert live, 'no live scenario is defined'
    pending = [e['name'] for e in live if e['status'] != 'recorded']
    assert not pending, pending


def _without_request_id(payloads: list[dict[str, Any]]) -> str:
    out = json.loads(json.dumps(payloads))
    for payload in out:
        payload.get('metadata', {}).pop('request_id', None)
    return json.dumps(out, sort_keys=True)


_TABULAR = sorted(
    n for n in json.loads(GOLDEN.read_text()) if n.startswith('tabular/')
)


@pytest.mark.parametrize('name', _TABULAR)
def test_every_tabular_case_is_recorded_with_its_golden_payloads(
    entries: list[dict[str, Any]], name: str
) -> None:
    match = [
        e for e in entries if e['name'] == name and e['status'] == 'recorded'
    ]
    assert match, f'tabular case {name} is not in the corpus'
    transcript = load(REPO / match[0]['transcript'])
    golden = json.loads(GOLDEN.read_text())[name]
    assert _without_request_id(transcript['payloads']) == _without_request_id(
        golden
    ), f'{name} payloads differ from tests/golden/wire_payloads.json'


def test_live_definitions_cover_a_metric_view_and_a_semantic_view(
    entries: list[dict[str, Any]],
) -> None:
    r"""The brief's live cells: Unity Catalog plus a metric view, and
    Snowflake plus a semantic view, defined in the runner and recorded."""
    text = '\n'.join(
        p.read_text()
        for p in sorted((REPO / 'scripts' / 'parity').rglob('*.py'))
    )
    assert re.search(r'metric[ _]view', text, re.IGNORECASE)
    assert re.search(r'semantic[ _]view', text, re.IGNORECASE)
    views = {
        (e['backend'], load(REPO / e['transcript'])['data'].get('view'))
        for e in entries
        if e['backend'] in LIVE_BACKENDS and e['status'] == 'recorded'
    }
    assert ('databricks', 'metric-view') in views, views
    assert ('snowflake', 'semantic-view') in views, views
