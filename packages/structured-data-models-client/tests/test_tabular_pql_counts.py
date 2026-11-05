# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public count/task parity, independent of runtime-specific frozen receipts."""

import pytest
from tabular_pql_count_cases import CASES, observe_case

ROUTES = ','.join(
    (
        'query.materialize/indices-binary/frames',
        'query.materialize/indices-multiclass/frames',
        'query.materialize/indices-regression/frames',
        'query.materialize/predict-frame/frames',
        'request.tabular/classification/frames',
        'request.tabular/regression/frames',
        'client.predict/tabular/frames',
    )
)


@pytest.mark.parametrize('route', ['old', 'core'])
@pytest.mark.parametrize('name', CASES)
def test_public_count_task_journey(monkeypatch, route, name):
    monkeypatch.setenv('SDM_CORE_ROUTES', ROUTES if route == 'core' else '')
    actual = observe_case(name)
    outcome, events = actual['outcome'], actual['events']
    warnings = actual['warnings']
    if name.startswith('empty_'):
        assert outcome == {
            'error': {
                'type': 'StructuredDataError',
                'code': 'INVALID_REQUEST',
                'message': '[INVALID_REQUEST] context has no rows, so there is nothing for the model to learn the answer from',
            }
        }
        assert not events and not warnings
    elif name == 'warning_raise':
        assert outcome == {
            'error': {
                'type': 'StopWarning',
                'code': None,
                'message': 'warning handler stopped',
            }
        }
        assert len(events) == 1 and events[0][0] == 'warning'
        assert len(warnings) == 1 and 'were sampled' in warnings[0]['message']
    else:
        assert 'error' not in outcome
        expected_count = (
            12
            if name.startswith('task_')
            else 4
            if name == 'count_owner'
            else actual['cap']
        )
        assert len(outcome['context_ids']) == expected_count
        assert outcome['encoded_context_ids'] == outcome['context_ids']
        assert outcome['task'] == (
            'classification' if name.startswith('task_') else 'regression'
        )
        assert outcome['encoded_task'] == outcome['task']
        assert len(outcome['prediction']['rows']) == 1
        assert events[-3:] == [
            ['materialized'],
            ['endpoint', expected_count],
            ['predicted'],
        ]
        if name == 'count_owner':
            assert events[:-3] == [['count_gt', actual['cap']]]
        elif name.startswith('task_') or name in (
            'over_default',
            'warning_record',
        ):
            assert len(events) == 4 and events[0][0] == 'warning'
        else:
            assert len(events) == 3
        if name.startswith('task_'):
            assert (
                len(warnings) == 1
                and 'two whole numbers' in warnings[0]['message']
            )
        elif name in ('over_default', 'warning_record'):
            assert (
                len(warnings) == 1 and 'were sampled' in warnings[0]['message']
            )
        else:
            assert not warnings
