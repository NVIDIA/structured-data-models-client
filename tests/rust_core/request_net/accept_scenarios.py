# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P8.scenarios: the new scenarios of task 4 are recorded offline and
replay.

Each is found by what its transcript holds, not by its name (CONTRACT.md,
Scenarios), among the transcripts added since BASE of an offline backend
(in-memory, DuckDB, SQLite, PostgreSQL, and tabular where the scenario is not
relational):

- an unscored candidate: a ranking that warns that a candidate was left out
  unscored, sent fewer payloads than it named candidates, and records the
  result's ``unscored_candidates``;
- all candidates cold: a ranking that sent nothing and ended in
  ``UnsupportedRequestError`` saying there is nothing to rank;
- tied ranking scores: a ranking whose result ranks two candidates of one row
  with equal scores;
- more than 1,000 entities: a question that is neither ranking nor forecast,
  sent in at least two payloads, whose result has more than 1,000 rows;
- a missing positive-class probability: a run that ended in the refusal of a
  ranking part answered without saying how sure it was;
- an absent field: a run that warned that the model answered without a field
  the request asked for;
- explanations by step: a forecast that asked for ``explanation`` and records
  the result's explanation by forecast step, for at least two steps;
- NIM 4xx and 5xx bodies: a run that ended in an error after a stored
  response with a 4xx status, and one with a 5xx status;
- sessions, on each of in-memory, DuckDB, SQLite and PostgreSQL: a run that
  calls ``SDMClient.fit`` and ``FittedModel.predict`` and decodes a result,
  and one whose ``fit`` names a ``holdout`` and warns about the rows named as
  holdout.

Every scenario found replays, and a replay compares a recorded error: the
all-cold scenario with its recorded message changed does not replay.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from _net import (
    LIVE_BACKENDS,
    OFFLINE,
    OFFLINE_RELATIONAL,
    TRANSCRIPTS,
    base_transcripts,
    calls_named,
    error_class,
    passed,
    recorded_error,
    replay,
    result_rows,
    transcripts,
    walk,
    without,
)

Transcript = tuple[str, Path, dict[str, Any]]


@pytest.fixture(scope='module')
def offline() -> list[Transcript]:
    r"""The transcripts of an offline backend added since BASE."""
    old = set(base_transcripts())
    return [
        (name, directory, data)
        for name, (directory, data) in sorted(transcripts().items())
        if data.get('backend') in OFFLINE
        and data.get('backend') not in LIVE_BACKENDS
        and name not in old
    ]


def _warned(data: dict[str, Any], *parts: str, category: str = '') -> bool:
    return any(
        all(p in str(w.get('message', '')) for p in parts)
        and str(w.get('category', '')).endswith(category)
        for w in data.get('warnings') or []
    )


def _recorded(data: dict[str, Any], key: str) -> list[Any]:
    r"""Values under ``key`` anywhere in the transcript but its payloads and
    responses, which is where the run's own outcome is recorded."""
    return [
        v for k, v in walk(without(data, 'payloads', 'responses')) if k == key
    ]


def _statuses(data: dict[str, Any]) -> list[int]:
    out = []
    for k, v in walk(data.get('responses') or []):
        if k in ('status', 'status_code') and isinstance(v, int):
            out.append(v)
    return out


def _rows(directory: Path, data: dict[str, Any]) -> list[dict[str, Any]]:
    return result_rows(directory, data) or []


def _named_candidates(data: dict[str, Any]) -> list[Any]:
    for call in calls_named(data, 'PredictiveQuery.materialize'):
        found = call.get('args', {}).get('candidates')
        if found:
            return list(found)
    return []


def unscored_candidate(name: str, directory: Path, data: dict) -> bool:
    named = _named_candidates(data)
    return (
        data.get('task_kind') == 'ranking'
        and recorded_error(data) is None
        and _warned(
            data, 'left out of this ranking, unscored', category='SDMWarning'
        )
        and 1 <= len(data.get('payloads') or []) < len(named)
        and bool(_rows(directory, data))
        and any(
            isinstance(v, list) and v
            for v in _recorded(data, 'unscored_candidates')
        )
    )


def all_candidates_cold(name: str, directory: Path, data: dict) -> bool:
    error = recorded_error(data)
    return (
        data.get('task_kind') == 'ranking'
        and not data.get('payloads')
        and error is not None
        and error_class(error).endswith('UnsupportedRequestError')
        and 'nothing to rank and nothing was sent' in str(error.get('message'))
    )


def tied_scores(name: str, directory: Path, data: dict) -> bool:
    if data.get('task_kind') != 'ranking' or recorded_error(data) is not None:
        return False
    for row in _rows(directory, data):
        scores = [e['score'] for e in row.get('rankings') or []]
        if len(set(scores)) < len(scores):
            return True
    return False


def batches(name: str, directory: Path, data: dict) -> bool:
    return (
        data.get('task_kind') not in ('ranking', 'forecast')
        and recorded_error(data) is None
        and len(data.get('payloads') or []) >= 2
        and len(_rows(directory, data)) > 1000
    )


def missing_positive_probability(
    name: str, directory: Path, data: dict
) -> bool:
    error = recorded_error(data)
    return (
        data.get('task_kind') == 'ranking'
        and error is not None
        and 'without saying how sure it was' in str(error.get('message'))
    )


def absent_field(name: str, directory: Path, data: dict) -> bool:
    return recorded_error(data) is None and _warned(
        data, 'answered without', 'which this request asked for'
    )


def explanations_by_step(name: str, directory: Path, data: dict) -> bool:
    asked = any(
        'explanation' in (call.get('args', {}).get('outputs') or [])
        for call in calls_named(data, 'SDMClient.predict')
    )
    return (
        data.get('task_kind') == 'forecast'
        and recorded_error(data) is None
        and asked
        and any(
            isinstance(v, dict) and len(v) >= 2
            for v in _recorded(data, 'by_forecast_step')
        )
    )


def nim_4xx(name: str, directory: Path, data: dict) -> bool:
    return recorded_error(data) is not None and any(
        400 <= s < 500 for s in _statuses(data)
    )


def nim_5xx(name: str, directory: Path, data: dict) -> bool:
    return recorded_error(data) is not None and any(
        500 <= s < 600 for s in _statuses(data)
    )


RELATIONAL_ONLY = {
    'unscored-candidate',
    'all-candidates-cold',
    'tied-scores',
    'missing-positive-probability',
    'explanations-by-step',
}

SCENARIOS: dict[str, Callable[[str, Path, dict], bool]] = {
    'unscored-candidate': unscored_candidate,
    'all-candidates-cold': all_candidates_cold,
    'tied-scores': tied_scores,
    'batches': batches,
    'missing-positive-probability': missing_positive_probability,
    'absent-field': absent_field,
    'explanations-by-step': explanations_by_step,
    'nim-4xx': nim_4xx,
    'nim-5xx': nim_5xx,
}


def session(name: str, directory: Path, data: dict) -> bool:
    return (
        recorded_error(data) is None
        and bool(calls_named(data, 'SDMClient.fit'))
        and bool(calls_named(data, 'FittedModel.predict'))
        and bool(_rows(directory, data))
    )


def holdout(name: str, directory: Path, data: dict) -> bool:
    return (
        session(name, directory, data)
        and any(
            call.get('args', {}).get('holdout') is not None
            for call in calls_named(data, 'SDMClient.fit')
        )
        and _warned(data, 'named as holdout')
    )


def _found(
    offline: list[Transcript], test: Callable[[str, Path, dict], bool]
) -> list[str]:
    return [
        name for name, directory, data in offline if test(name, directory, data)
    ]


@pytest.mark.parametrize('scenario', sorted(SCENARIOS))
def test_the_scenario_is_recorded_and_replays(
    scenario: str, offline: list[Transcript]
) -> None:
    candidates = offline
    if scenario in RELATIONAL_ONLY:
        candidates = [t for t in offline if t[2].get('backend') != 'tabular']
    names = _found(candidates, SCENARIOS[scenario])
    assert names, f'no offline transcript records the {scenario} scenario'
    code, output = replay(names)
    assert code == 0 and set(names) <= passed(output), output[-4000:]


@pytest.mark.parametrize('backend', OFFLINE_RELATIONAL)
def test_sessions_are_recorded_and_replay(
    backend: str, offline: list[Transcript]
) -> None:
    mine = [t for t in offline if t[2].get('backend') == backend]
    fitted = _found(mine, session)
    held = _found(mine, holdout)
    assert fitted, (
        f'no {backend} transcript calls SDMClient.fit and FittedModel.predict'
    )
    assert held, (
        f'no {backend} transcript fits with a holdout that warns about the '
        f'rows named as holdout'
    )
    names = sorted(set(fitted) | set(held))
    code, output = replay(names)
    assert code == 0 and set(names) <= passed(output), output[-4000:]


def test_a_replay_compares_the_recorded_error(
    offline: list[Transcript], tmp_path: Path
) -> None:
    found = [t for t in offline if all_candidates_cold(*t)]
    assert found, (
        'no offline transcript records the all-candidates-cold scenario'
    )
    name, directory, data = found[0]
    relative = directory.relative_to(TRANSCRIPTS)
    target = tmp_path / relative
    shutil.copytree(directory, target)
    path = target / 'transcript.json'
    transcript = json.loads(path.read_text(encoding='utf-8'))
    transcript['error']['message'] = (
        transcript['error']['message'] + ' (changed)'
    )
    path.write_text(json.dumps(transcript, indent=1), encoding='utf-8')
    code, output = replay([name], tmp_path)
    assert code != 0 and name not in passed(output), (
        f'{name} replays with its recorded error message changed, so the '
        f'replay does not compare it'
    )
