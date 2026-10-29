# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Successful native scope and lineage remain common comparison evidence."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from test_t44_prediction_observation import (
    actual_observation,
    legacy_observation,
    reader,
    two_observed_predictions,
)
from test_t136_forecast_capture import _module

ROOT = Path(__file__).parents[2]
COMPARE = _module(
    't44_success_evidence_compare', ROOT / 'scripts/rust_core/compare.py'
)
CHANGES = ['source-row', 'answer-ordinal', 'response-rows', 'aggregation']


def observed(core, operations):
    return {
        'identity': {'operations': operations, 'contract': 35},
        'core': core,
    }


def failures(left, right, *, stop=None):
    report = COMPARE.Report(paths=(('metadata', 'request_id'),))
    COMPARE._core_agree(
        report, {'id': 'successful-native-evidence'}, left, right, stop=stop
    )
    return report.failures


def change_evidence(core, ordinal, changed):
    entry = core['predictions'][ordinal]
    [call] = entry['calls']
    [result] = entry['results']
    global_before = copy.deepcopy(core['predicted'])
    arrow_before = copy.deepcopy(result['result'])
    metadata_before = copy.deepcopy(result['metadata'])
    bodies_before = [copy.deepcopy(value['body']) for value in entry['calls']]
    if changed == 'source-row':
        result['data']['lineage'][0]['source_row'] = 1
    elif changed == 'answer-ordinal':
        result['data']['lineage'][0]['answer_ordinal'] = 1
        result['data']['lineage'][1]['answer_ordinal'] = 0
    elif changed == 'response-rows':
        call['scope']['response_rows'] = 3
        part = result['input']['input']['aggregation']['parts'][0]
        part['data']['scope'] = copy.deepcopy(call['scope'])
        part['attribution']['scope'] = copy.deepcopy(call['scope'])
        for record in core['inputs']:
            if record.get('run') == call['run']:
                record['result']['scope'] = copy.deepcopy(call['scope'])
    else:
        prepared = result['input']['input']
        calls = prepared['aggregation']
        # RelationalCallResult has a bound and no aggregation-kind tag.
        # This is the actual closed Steps shape, not an arbitrary marker.
        prepared['aggregation'] = {
            'kind': 'steps',
            'children': [
                {
                    'bound': copy.deepcopy(prepared['bound']),
                    'parts': calls['parts'],
                    'counted': calls['counted'],
                    'known_empty': calls['known_empty'],
                }
            ],
            'steps': [1],
        }
    for record in core['inputs']:
        if record.get('run') == result['run']:
            record['input'] = copy.deepcopy(result['input'])
            record['result'] = copy.deepcopy(result['data'])
    assert core['predicted'] == global_before
    assert result['result'] == arrow_before
    assert result['metadata'] == metadata_before
    assert [value['body'] for value in entry['calls']] == bodies_before


def test_identical_successful_native_products_pass_without_mutation():
    core, operations = actual_observation()
    untouched = copy.deepcopy(core)
    pair = observed(core, operations)
    assert failures(pair, copy.deepcopy(pair)) == []
    assert core == untouched


@pytest.mark.parametrize('changed', CHANGES)
def test_reader_valid_success_evidence_changes_fail_ordinary_comparison(
    changed,
):
    core, operations = actual_observation()
    other = copy.deepcopy(core)
    change_evidence(other, 0, changed)
    reader().view(core, operations)
    reader().view(other, operations)
    assert core['predicted'] == other['predicted']
    assert failures(observed(core, operations), observed(other, operations))


@pytest.mark.parametrize('changed', CHANGES)
def test_changed_common_success_evidence_is_not_hidden_by_final_global(changed):
    first, first_operations = actual_observation()
    longer, longer_operations = two_observed_predictions()
    final_before = copy.deepcopy(longer['predicted'])
    change_evidence(longer, 0, changed)
    reader().view(first, first_operations)
    reader().view(longer, longer_operations)
    assert longer['predicted'] == final_before
    assert failures(
        observed(first, first_operations),
        observed(longer, longer_operations),
        stop='predict',
    )


@pytest.mark.parametrize('changed', CHANGES)
def test_longer_only_native_detail_is_excluded_from_the_shared_prefix(changed):
    first, first_operations = actual_observation()
    longer, longer_operations = two_observed_predictions()
    first_before = copy.deepcopy(longer['predictions'][0])
    change_evidence(longer, 1, changed)
    reader().view(longer, longer_operations)
    assert longer['predictions'][0] == first_before
    assert (
        failures(
            observed(first, first_operations),
            observed(longer, longer_operations),
            stop='predict',
        )
        == []
    )


@pytest.mark.parametrize('changed', [None, *CHANGES])
def test_unobserved_historical_native_details_are_not_fabricated(changed):
    historical, historical_operations = legacy_observation()
    current, current_operations = actual_observation()
    if changed is not None:
        change_evidence(current, 0, changed)
    assert 'predictions' not in historical
    assert 'model_calls' not in historical
    assert 'legacy_predictions' not in historical
    assert 'outputs' not in historical['predicted']
    reader().view(historical, historical_operations)
    reader().view(current, current_operations)
    untouched = copy.deepcopy(historical)
    assert (
        failures(
            observed(current, current_operations),
            observed(historical, historical_operations),
        )
        == []
    )
    assert historical == untouched
