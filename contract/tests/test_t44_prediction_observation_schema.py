# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The finite prediction evidence wrappers are closed observation data."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError
from test_t44_prediction_observation import actual_observation


def validator(name):
    schema = json.loads(
        (Path(__file__).parents[1] / 'observation.schema.json').read_text()
    )
    return Draft202012Validator(
        {'$ref': '#/$defs/' + name, '$defs': schema['$defs']}
    )


def test_prediction_entry_schema_keeps_actual_typed_products_without_fake_operations():
    core, operations = actual_observation()
    entry = core['predictions'][0]
    validator('predictionEntry').validate(entry)
    for record in core['inputs'][2:]:
        validator('preparationObservation').validate(record)
    assert 'predict' not in operations
    assert (
        entry['calls'][0]['request_id']
        != entry['calls'][0]['scope']['request_id']
    )


@pytest.mark.parametrize(
    'field',
    [
        'ordinal',
        'parent',
        'argument',
        'completion',
        'runs',
        'calls',
        'results',
        'final',
    ],
)
def test_prediction_evidence_cannot_omit_its_observation_scope(field):
    core, _ = actual_observation()
    entry = copy.deepcopy(core['predictions'][0])
    del entry[field]
    with pytest.raises(ValidationError):
        validator('predictionEntry').validate(entry)


@pytest.mark.parametrize('nested', ['entry', 'argument', 'call', 'result'])
def test_prediction_observation_wrappers_reject_extra_control_fields(nested):
    core, _ = actual_observation()
    entry = copy.deepcopy(core['predictions'][0])
    target = {
        'entry': entry,
        'argument': entry['argument'],
        'call': entry['calls'][0],
        'result': entry['results'][0],
    }[nested]
    target['next_operation'] = 'invented-step'
    with pytest.raises(ValidationError):
        validator('predictionEntry').validate(entry)


def test_preparation_observation_requires_actual_run_and_owner_and_keeps_native_failure():
    core, _ = actual_observation()
    record = core['inputs'][2]
    for field in ['run', 'entry', 'input', 'result', 'error']:
        changed = copy.deepcopy(record)
        del changed[field]
        with pytest.raises(ValidationError):
            validator('preparationObservation').validate(changed)
    failed = copy.deepcopy(record)
    failed['result'] = None
    failed['error'] = {'kind': 'not_found', 'code': 'UNKNOWN_MODEL'}
    validator('preparationObservation').validate(failed)
    failed['error']['message'] = (
        'private diagnostic must not enter this wrapper'
    )
    with pytest.raises(ValidationError):
        validator('preparationObservation').validate(failed)


def test_started_native_input_bytes_are_required_even_for_serving_envelopes():
    core, _ = actual_observation()
    call = copy.deepcopy(core['predictions'][0]['calls'][0])
    validator('predictionCallObservation').validate(call)
    del call['input_body_base64']
    with pytest.raises(ValidationError):
        validator('predictionCallObservation').validate(call)
