# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed preparation references preserve the long-list validation proof."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from test_t136_forecast_capture import RUNNER


def schema():
    closed = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {'op': {'const': 'prepare_request'}},
        'required': ['op'],
    }
    branches = [
        {**copy.deepcopy(closed), 'properties': {'op': {'const': name}}}
        for name in ('describe', 'validate_query', 'materialize', 'predict')
    ]
    branches[2]['properties']['indices'] = {
        'type': ['array', 'null'],
        'items': {'type': 'object'},
    }
    branches.append({'$ref': '#/$defs/preparationObservation'})
    return {
        '$schema': 'https://json-schema.org/draft/2020-12/schema',
        '$id': 'urn:sdm:observation:1',
        '$defs': {'preparationObservation': closed},
        'properties': {
            'core': {
                'properties': {
                    'inputs': {'type': 'array', 'items': {'oneOf': branches}}
                }
            }
        },
    }


def branches(document):
    return document['properties']['core']['properties']['inputs']['items'][
        'oneOf'
    ]


def test_current_observation_schema_admits_its_closed_preparation_definition():
    actual = json.loads(
        (Path(__file__).parents[1] / 'observation.schema.json').read_text()
    )
    RUNNER._long_lists_held(actual)
    RUNNER.validator('observation')


@pytest.mark.parametrize('reference', [False, True])
def test_inline_and_local_closed_siblings_keep_the_same_tail_proof(reference):
    document = schema()
    if not reference:
        branches(document)[4] = document['$defs']['preparationObservation']
    RUNNER._long_lists_held(document)
    validator = Draft202012Validator(document)
    given = {'core': {'inputs': [{'op': 'materialize', 'indices': [{}] * 101}]}}
    original = copy.deepcopy(given)
    assert RUNNER._invalid(validator, given) is None
    given['core']['inputs'][0]['indices'][100] = 'not an object'
    assert RUNNER._invalid(validator, given) is not None
    assert list(validator.iter_errors(given))
    assert original['core']['inputs'][0]['indices'] == [{}] * 101
    # The referenced sibling cannot acquire the trimmed field or extra input.
    for extra in ({'indices': [{}] * 101}, {'unexpected': True}):
        given = {'core': {'inputs': [{'op': 'prepare_request', **extra}]}}
        assert RUNNER._invalid(validator, given) is not None


@pytest.mark.parametrize(
    'change',
    [
        'open',
        'indices',
        'pattern',
        'unresolved',
        'external',
        'pointer',
        'chain',
        'reference-sibling',
        'changed-items',
    ],
)
def test_unproved_siblings_and_changed_item_rules_still_refuse(change):
    document = schema()
    definition = document['$defs']['preparationObservation']
    if change == 'open':
        definition['additionalProperties'] = True
    elif change == 'indices':
        definition['properties']['indices'] = {'type': 'array'}
    elif change == 'pattern':
        definition['patternProperties'] = {'indices': {'type': 'array'}}
    elif change == 'unresolved':
        branches(document)[4] = {'$ref': '#/$defs/missing'}
    elif change == 'external':
        branches(document)[4] = {'$ref': 'urn:other:preparation'}
    elif change == 'pointer':
        branches(document)[4] = {'$ref': '#/$defs/preparationObservation/properties'}
    elif change == 'chain':
        document['$defs']['alias'] = copy.deepcopy(definition)
        definition['$ref'] = '#/$defs/alias'
    elif change == 'reference-sibling':
        branches(document)[4]['additionalProperties'] = False
    elif change == 'changed-items':
        branches(document)[2]['properties']['indices']['maxItems'] = 8
    with pytest.raises(RUNNER.RunnerFailure, match=r'T2\.1l'):
        RUNNER._long_lists_held(document)
