# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from types import MappingProxyType

import pytest
from sdm_core import _native
from sdm_core.replay import replay_snapshot


def typed(value):
    if isinstance(value, dict):
        return {
            'mapping': [
                [typed(key), typed(item)] for key, item in value.items()
            ]
        }
    if isinstance(value, list):
        return {'list': [typed(item) for item in value]}
    return {'scalar': value}


def recording(seed=0):
    return {
        'format': 'p97-generators-v1',
        'cases': [
            {
                'id': 'example',
                'depends_on': [],
                'expected': 'not an answer',
                'replay_input': {
                    'generator': {
                        'name': 'dialects.random_state',
                        'stream_seed': 0,
                        'ordinal': 0,
                    },
                    'generator_input': typed(
                        {
                            'calls': [
                                {
                                    'backend': 'databricks',
                                    'function': 'utils.random_state.randint',
                                    'arguments': {
                                        'random_seed': seed,
                                        'low': 0,
                                        'high': 2147483647,
                                        'size': 1,
                                    },
                                }
                            ]
                        }
                    ),
                },
            }
        ],
    }


def replay(document):
    files = MappingProxyType(
        {
            'tests/parity/frozen-generators/example.json': json.dumps(
                document
            ).encode(),
        }
    )
    before = dict(files)
    result = replay_snapshot(files, case_id='example')
    assert dict(files) == before
    return result


def test_snapshot_computes_input_not_expected_output():
    observed = replay(recording())
    assert observed['example']['family'] == 'original-generator-v1'
    assert observed['example']['observation']['result'] == typed(
        [{'result': [209652396]}]
    )
    corrupted = recording()
    corrupted['cases'][0]['expected'] = {'result': 'corrupted'}
    assert replay(corrupted) == observed
    assert replay(recording(1)) != observed


@pytest.mark.parametrize(
    'mutation', ['duplicate', 'dependency', 'typed', 'family']
)
def test_snapshot_rejects_invalid_inputs(mutation):
    document = recording()
    case = document['cases'][0]
    if mutation == 'duplicate':
        document['cases'].append(copy.deepcopy(case))
    elif mutation == 'dependency':
        case['depends_on'] = ['absent.arrow']
    elif mutation == 'typed':
        case['replay_input']['generator_input'] = {'tuple': []}
    else:
        case['replay_input']['generator']['name'] = 'unsupported'
    with pytest.raises(ValueError):
        replay(document)


def test_native_snapshot_rejects_duplicate_json_keys():
    with pytest.raises(ValueError, match='duplicate JSON'):
        _native.prepare_snapshot('{"files": {}, "files": {}}')


def test_preparation_cache_uses_complete_immutable_bytes():
    from sdm_core.replay import _prepare

    _prepare.cache_clear()
    document = recording()
    original = replay(document)
    assert _prepare.cache_info().misses == 1
    assert replay(copy.deepcopy(document)) == original
    assert _prepare.cache_info().hits == 1
    assert replay(recording(1)) != original
    assert _prepare.cache_info().misses == 2
    assert _prepare.cache_info().currsize == 1
    document['cases'][0]['expected'] = 'different expected bytes'
    assert replay(document) == original
    assert _prepare.cache_info().misses == 3
    with pytest.raises(TypeError, match='immutable bytes'):
        replay_snapshot({'example': bytearray(b'{}')}, case_id='example')


def test_replay_dispatches_natively_for_every_cached_case():
    import sys

    from sdm_core.replay import _prepare

    _prepare.cache_clear()
    calls = []

    def observe(_frame, event, argument):
        if event == 'c_call' and argument is _native.replay_snapshot:
            calls.append(argument)

    assert sys.getprofile() is None
    sys.setprofile(observe)
    try:
        assert replay(recording()) == replay(recording())
    finally:
        sys.setprofile(None)
    assert len(calls) == 2
    assert _prepare.cache_info().misses == 1
