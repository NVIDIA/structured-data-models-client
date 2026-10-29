# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warehouse replay records starting arguments, never a later schema/state."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    't310_warehouse_run_python',
    Path(__file__).resolve().parents[1] / 'runners/run_python.py',
)
assert _SPEC is not None and _SPEC.loader is not None
run_python = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = run_python
_SPEC.loader.exec_module(run_python)


def legacy_capture(inputs):
    """Complete capture storage for explicitly authored historical starts."""
    capture = run_python._Capture()
    capture.inputs = inputs
    capture.operations = [started['op'] for started in inputs]
    return capture


def test_replay_keeps_provider_namespace_and_view_without_copying_answers() -> (
    None
):
    provider = {
        'provider': 'sql',
        'dialect': 'databricks',
        'namespace': ['test_catalog', 'test_schema'],
        'view': {'name': 'metric_view', 'trust_expressions': False},
    }
    source = {
        'tables': [{'name': 'people', 'source': {'source': ['users']}}],
        'primary_keys': {},
        'time_columns': {},
    }
    inputs = [
        {
            'op': 'describe_source',
            'provider': provider,
            'source': source,
            'check_keys': False,
        },
        {
            'op': 'materialize',
            'provider': provider,
            'indices': [{'t': 'i64', 'v': '1'}],
            'anchor_time': None,
            'seed': None,
            'schema': {'must_not_replay': 'held description'},
            'query': 'must not replace the case query',
        },
        {
            'op': 'predict',
            'model': 'kumo-relational',
            'state': {'must_not_replay': 'request state'},
        },
    ]
    core = run_python._replay_core(legacy_capture(inputs))
    assert core == {
        'describe_source': {
            'provider': provider,
            'source': source,
            'check_keys': False,
        },
        'materialize': {
            'provider': provider,
            'indices': [{'t': 'i64', 'v': '1'}],
            'anchor_time': None,
            'seed': None,
        },
        'predict': {'model': 'kumo-relational'},
    }


def test_missing_describe_start_cannot_become_an_empty_source() -> None:
    with pytest.raises(run_python.RunnerFailure, match='describe_source'):
        run_python._replay_core(legacy_capture([]))


def test_a_started_warehouse_case_with_no_fetch_is_still_recorded() -> None:
    capture = legacy_capture(
        [
            {
                'op': 'describe_source',
                'provider': {'provider': 'sql', 'dialect': 'databricks'},
                'source': {
                    'tables': [],
                    'primary_keys': {},
                    'time_columns': {},
                },
                'check_keys': True,
            }
        ]
    )
    line = run_python._replay_line(
        {'id': 'no-fetch'},
        'databricks-sql',
        capture,
        {'capture': {'statements': []}},
    )
    assert line['case'] == 'no-fetch@databricks-sql'
    assert line['fetches'] == []
    assert line['core']['describe_source'] == {
        key: value for key, value in capture.inputs[0].items() if key != 'op'
    }


def test_multiple_starts_are_not_silently_collapsed() -> None:
    start = {
        'op': 'describe_source',
        'provider': {'provider': 'sql', 'dialect': 'postgres'},
        'source': {'tables': [], 'primary_keys': {}, 'time_columns': {}},
        'check_keys': True,
    }
    with pytest.raises(run_python.RunnerFailure, match='more than one'):
        run_python._replay_core(legacy_capture([start, start]))


def observed_capture(core, operations):
    """An actual capture reader owning independently authored typed records."""
    capture = run_python._Capture()
    capture.operations = operations
    for name, value in core.items():
        setattr(capture, name, value)
    return capture


def test_actual_gathered_replay_projects_canonical_model_and_only_starting_arguments():
    from test_t44_prediction_observation import (
        ANSWER_MODEL,
        MODEL,
        actual_observation,
    )

    core, operations = actual_observation()
    captured = observed_capture(core, operations)
    replay = run_python._replay_core(captured)
    describe, materialize = core['inputs'][:2]
    assert replay == {
        'describe_source': {
            name: describe[name]
            for name in ('provider', 'source', 'check_keys')
        },
        'materialize': {
            name: materialize[name]
            for name in ('provider', 'indices', 'anchor_time', 'seed')
        },
        'predict': {'model': MODEL},
    }
    assert replay['predict']['model'] != ANSWER_MODEL
    assert 'schema' not in replay['materialize']
    assert 'query' not in replay['materialize']
    assert 'predicted' not in replay
    assert 'predictions' not in replay
    assert captured.operations == operations
    assert 'predict' not in captured.operations


def test_actual_gathered_replay_still_requires_the_original_description_start():
    from test_t44_prediction_observation import actual_observation

    core, operations = actual_observation()
    core['inputs'] = core['inputs'][1:]
    with pytest.raises(
        run_python.RunnerFailure, match='describe_source|description'
    ):
        run_python._replay_core(observed_capture(core, operations))


def test_actual_gathered_replay_refuses_two_complete_independent_predictions():
    from test_t44_prediction_observation import reader, two_observed_predictions

    core, operations = two_observed_predictions()
    assert len(reader().view(core, operations)['predictions']) == 2
    with pytest.raises(
        run_python.RunnerFailure, match='more than one|multiple|ambiguous'
    ):
        run_python._replay_core(observed_capture(core, operations))


def test_actual_failed_lookup_cannot_replay_the_raw_argument_as_a_resolved_model():
    from test_t44_prediction_observation import unresolved_observation

    core, operations = unresolved_observation()
    with pytest.raises(run_python.RunnerFailure, match='model|unresolved'):
        run_python._replay_core(observed_capture(core, operations))


def test_actual_unrelated_preparation_does_not_create_replay_prediction():
    from test_t44_prediction_observation import actual_observation

    core, operations = actual_observation()
    core['predictions'] = []
    core['predicted'] = None
    core['inputs'] = core['inputs'][:3]
    core['inputs'][2]['entry'] = None
    with pytest.raises(run_python.RunnerFailure):
        run_python._replay_core(observed_capture(core, operations[:3]))


def test_actual_broken_call_evidence_cannot_be_replayed_from_public_model():
    from test_t44_prediction_observation import actual_observation

    core, operations = actual_observation()
    core['predictions'][0]['calls'].clear()
    with pytest.raises(run_python.RunnerFailure):
        run_python._replay_core(observed_capture(core, operations))
