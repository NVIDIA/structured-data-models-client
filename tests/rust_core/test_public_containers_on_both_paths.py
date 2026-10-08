# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public refusals retain their validation order for non-JSON containers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CASES = (
    'list-cycle',
    'dict-cycle',
    'array-cycle',
    'array-list-cycle',
    'deep-list',
    'deep-dict',
    'deep-tuple',
    'deep-array',
    'shared-list',
    'shared-array',
    'ordinary-list',
)


@pytest.mark.parametrize('old', [True, False])
@pytest.mark.parametrize('entry', ['schema', 'indices'])
@pytest.mark.parametrize('case', CASES)
def test_public_container_refusals(monkeypatch, old, entry, case):
    from sdm_client import _impl

    monkeypatch.setattr(_impl, 'OLD', old)
    monkeypatch.syspath_prepend(str(REPO / 'scripts/parity'))
    from _container_events import capture

    outcome = capture(case, entry)
    assert outcome['class'] == 'sdm_client.errors.StructuredDataError'
    assert outcome['warnings'] == []
    if entry == 'schema':
        assert outcome['code'] == 'UNSUPPORTED_REQUEST'
        assert 'users.feature holds a' in outcome['message']
        assert 'give one value per row' in outcome['message']
    else:
        assert outcome['code'] == 'INVALID_REQUEST'
        assert (
            "gathering options apply only to relational materialization: ['random_seed']"
            in outcome['message']
        )
        assert 'model=KumoRelational()' in outcome['message']


@pytest.mark.parametrize('old', [True, False])
def test_container_recordings_match_both_selectors(monkeypatch, old):
    from sdm_client import _impl

    monkeypatch.setattr(_impl, 'OLD', old)
    monkeypatch.syspath_prepend(str(REPO / 'scripts/parity'))
    from _container_events import capture_container_events

    expected = json.loads(
        (REPO / 'tests/rust_core/shadow/host_events.json').read_text()
    )['containers']
    assert capture_container_events() == expected


@pytest.mark.parametrize('old', [True, False])
def test_supported_deep_result_metadata_is_not_bounded(monkeypatch, old):
    from sdm_client import PredictionResult, _impl

    monkeypatch.setattr(_impl, 'OLD', old)
    nested = 1
    for _ in range(105):
        nested = {'child': nested}
    result = PredictionResult(
        model='kumo-tabular',
        request_id='probe',
        items=[],
        metadata={'extra': nested},
    )
    assert result.absent(['prediction']) == ()


@pytest.mark.parametrize('old', [True, False])
def test_supported_deep_request_options_are_preserved(monkeypatch, old):
    import pandas as pd
    from sdm_client import Request, _impl

    monkeypatch.setattr(_impl, 'OLD', old)
    nested = 1
    for _ in range(105):
        nested = {'child': nested}
    options = {'extra': nested}
    request = Request(
        pd.DataFrame({'feature': [1, 2], 'target': [3.0, 4.0]}),
        pd.DataFrame({'feature': [3]}),
        target='target',
        task='regression',
        inference=options,
        task_options=options,
    )
    assert request.inference == options
    assert request.task.options == options


@pytest.mark.parametrize('old', [True, False])
@pytest.mark.parametrize('keys', [[1], (1,)])
def test_supported_entity_key_sequences_are_preserved(monkeypatch, old, keys):
    import pandas as pd
    from sdm_client import PredictiveQuery, _impl

    monkeypatch.setattr(_impl, 'OLD', old)
    query = PredictiveQuery(
        'PREDICT users.target FOR EACH users.user_id',
        {
            'users': pd.DataFrame(
                {
                    'user_id': [1, 2, 3],
                    'target': [3.1, 4.2, 5.3],
                }
            )
        },
    )
    request = query.materialize(indices=keys)
    assert request.predict.instance_table.to_dict('list') == {'user_id': [1]}
