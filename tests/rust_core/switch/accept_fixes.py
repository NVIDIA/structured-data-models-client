# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Brief tasks 7b and 7c, where a public call shows them: each bug answers
correctly or refuses clearly on the core, and unseeded runs do not repeat.

The scenarios are in ``_fixes.py`` and run on the core in a fresh process.
A clear refusal is a public ``sdm_client`` error, not ``INTERNAL_ERROR``,
not the engine's "failed while building" wrapper, and naming what it refuses.
Beside them, no transcript and no public fixture case ends, on the core, in
the engine's unforeseen-failure wrapper (the multilabel link prediction, the
complex column, the decimal entity keys). The bugs no public call here
shows are the reviewer's (``P9.fixes-review``).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
from _switch import (
    CORE,
    ENGINE_FAILED,
    HERE,
    REPO,
    capture,
    env_for,
    install_core,
    run,
)


@pytest.fixture(scope='module')
def outcomes() -> dict[str, Any]:
    install_core()
    with tempfile.TemporaryDirectory(prefix='sdm-p9-fixes-') as tmp:
        out = Path(tmp) / 'fixes.json'
        proc = run(
            [sys.executable, HERE / '_fixes.py', '--out', out],
            cwd=REPO,
            env=env_for(CORE),
            timeout=1800,
        )
        assert proc.returncode == 0 and out.is_file(), proc.stderr[-4000:]
        return json.loads(out.read_text())


def _clear_refusal(outcome: dict[str, Any], *names: str) -> str | None:
    if outcome.get('ok'):
        return None
    message = outcome.get('message', '')
    if not outcome.get('public') or not outcome['class'].startswith(
        'sdm_client.'
    ):
        return f'raised {outcome["class"]}, not a public sdm_client error: {message}'
    if outcome.get('code') == 'INTERNAL_ERROR' or ENGINE_FAILED in message:
        return f'an unforeseen failure, not a refusal: {message}'
    missing = [n for n in names if n not in message]
    if missing:
        return f'the refusal does not name {missing}: {message}'
    return None


@pytest.mark.parametrize(
    ('scenario', 'names'),
    [
        ('empty-table-from-source', ('orders',)),
        ('complex-column', ('signal',)),
        ('period-column', ('joined',)),
        ('interval-column', ('band',)),
        ('table-emptied-after-read', ('orders',)),
        ('target-dropped-after-schema', ('churn',)),
        ('entity-dropped-after-schema', ('user_id',)),
    ],
)
def test_answers_or_refuses_clearly(outcomes, scenario, names):
    found = _clear_refusal(outcomes[scenario], *names)
    assert found is None, f'{scenario}: {found}'


def test_an_index_past_uint64_is_refused_as_an_sdm_client_error(outcomes):
    r"""Passed exactly across the binding (Phase 7.6), such an index is
    refused as today's code refuses it: a public ``sdm_client`` error, not a
    float's error from the binding."""
    outcome = outcomes['index-past-uint64']
    assert not outcome.get('ok'), outcome
    assert outcome['public'] and outcome['class'].startswith('sdm_client.'), (
        outcome
    )


def test_the_emptied_table_is_not_reported_as_a_time_overflow(outcomes):
    outcome = outcomes['table-emptied-after-read']
    assert 'Cannot cast' not in outcome.get('message', ''), outcome


@pytest.mark.parametrize('name', ['instance_id', '__kumo_anchor_time'])
def test_a_reserved_name_added_after_the_schema_is_refused(outcomes, name):
    outcome = outcomes[f'{name}-added-after-schema']
    assert not outcome.get('ok'), f'{name} was sent: {outcome}'
    found = _clear_refusal(outcome, name)
    assert found is None, found
    assert outcome['class'].endswith('StructuredDataError'), outcome
    assert 'cannot insert' not in outcome['message'], outcome


def test_a_time_of_day_is_not_taken_as_the_time_column(outcomes):
    outcome = outcomes['time-of-day-column']
    assert outcome.get('ok'), outcome
    assert outcome['value'] != 'starts_at', outcome


def test_unseeded_runs_continue_the_stream(outcomes):
    outcome = outcomes['unseeded-runs']
    assert outcome.get('ok'), outcome
    first, second, third = outcome['value']
    assert len({first, second, third}) == 3, 'an unseeded run repeated another'


@pytest.mark.parametrize(
    ('what', 'key'), [('transcripts', 'name'), ('fixtures', 'id')]
)
def test_no_scenario_ends_in_an_unforeseen_engine_failure(what, key):
    data = capture(CORE, what)
    failed = [
        f'{r[key]}: {r["error"]["message"][:160]}'
        for r in data['records']
        if r.get('error')
        and ENGINE_FAILED in str(r['error'].get('message', ''))
    ]
    assert not failed, '\n'.join(failed)
