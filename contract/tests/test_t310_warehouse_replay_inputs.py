# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warehouse replay records starting arguments, never a later schema/state."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location(
    't310_warehouse_run_python',
    Path(__file__).resolve().parents[1] / 'runners/run_python.py',
)
assert _SPEC is not None and _SPEC.loader is not None
run_python = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = run_python
_SPEC.loader.exec_module(run_python)


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
    core = run_python._replay_core(SimpleNamespace(inputs=inputs))
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
        run_python._replay_core(SimpleNamespace(inputs=[]))


def test_a_started_warehouse_case_with_no_fetch_is_still_recorded() -> None:
    capture = SimpleNamespace(
        inputs=[
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
        ],
        answered=[],
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
        run_python._replay_core(SimpleNamespace(inputs=[start, start]))
