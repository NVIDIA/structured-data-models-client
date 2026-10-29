# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common comparisons consume checked materialization entry products.

These independently authored journal fixtures claim observation structure,
not an SDK execution. Actual source I/O and label consumers live separately.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from test_t44_materialization_raw_journal import _bytes
from test_t45_materialization_observations import actual_materialization
from test_t136_forecast_capture import _module

ROOT = Path(__file__).parents[2]
COMPARE = _module(
    't45_materialization_compare', ROOT / 'scripts/rust_core/compare.py'
)


def observed(core, operations):
    return {
        'identity': {'operations': operations, 'contract': 35},
        'core': core,
    }


def legacy():
    core = actual_materialization()
    for field in (
        'materialization_preparations',
        'materialization_entries',
        'materialization_guards',
        'materialization_capture_capabilities',
    ):
        core.pop(field)
    core['inputs'] = [
        {
            'op': 'materialize',
            'provider': {'provider': 'sql', 'dialect': 'duckdb'},
            'schema': {'tables': [], 'links': []},
            'query': 'PREDICT events.value FOR EACH events.id',
            'indices': [{'t': 'i64', 'v': '7'}],
            'anchor_time': None,
            'seed': None,
        }
    ]
    core['materialized'] = {
        'version': 2,
        'query': 'resolved query',
        'payload': {'context': {}, 'predict': {}},
    }
    return observed(core, ['materialize'])


def failures(left, right, stop=None):
    report = COMPARE.Report()
    COMPARE._core_agree(
        report, {'id': 'actual-materialization'}, left, right, stop=stop
    )
    return report.failures


def replace_payload(record, payload):
    record['output_reads'][0] = _bytes(11, json.dumps(payload).encode())


def two_entries():
    core = actual_materialization()
    second = copy.deepcopy(core['materialization_preparations'][0])
    second.update(run=1, entry=1)
    core['materialization_preparations'].append(second)
    entry = copy.deepcopy(core['materialization_entries'][0])
    entry.update(ordinal=1, runs=[1], terminal={'run': 1})
    core['materialization_entries'].append(entry)
    return core


def test_actual_nested_request_matches_legacy_common_product_without_global_publication():
    core = actual_materialization()
    original = copy.deepcopy(core)
    assert failures(observed(core, ['prepare_materialize']), legacy()) == []
    assert core == original
    assert core['inputs'] == [] and core['materialized'] is None


def test_changed_actual_materialized_payload_cannot_hide_behind_empty_global_summary():
    core = actual_materialization()
    replace_payload(
        core['materialization_preparations'][0],
        {'context': {'changed': 1}, 'predict': {}},
    )
    assert failures(observed(core, ['prepare_materialize']), legacy())


@pytest.mark.parametrize('changed', ['shared', 'longer-only'])
def test_shared_prefix_selects_its_actual_materialization_result(changed):
    longer = two_entries()
    replace_payload(
        longer['materialization_preparations'][0 if changed == 'shared' else 1],
        {'context': {'changed': 1}, 'predict': {}},
    )
    found = failures(
        observed(actual_materialization(), ['prepare_materialize']),
        observed(longer, ['prepare_materialize', 'prepare_materialize']),
        stop='materialize',
    )
    assert bool(found) == (changed == 'shared')
