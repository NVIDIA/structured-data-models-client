# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite date-only corrections from authenticated same-input original runs."""

import hashlib
import json

FAMILY = 'schema.date_columns'
SEED = 20261007
CONSUMER = 'tests/rust_core/schema/accept_dates_series.py'
OPERATIONS = {
    'dates.day_comes_first',
    'dates.looks_unambiguous',
    'dates.shapes_written',
    'dates.reads_as_a_date',
    'dates.conversion_that_works',
    'time_column.read',
    'infer.timestamp',
    'dates.read',
}


class Rejected(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise Rejected(reason)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(',', ':'), allow_nan=False
        ).encode()
    ).hexdigest()


def same(left, right):
    return digest(left) == digest(right)


def operations(outcome):
    require(
        outcome['status'] == 'success' and outcome['error'] is None,
        'date.consumer_refusal',
    )
    require(
        outcome['schema'] == outcome['payloads'] == outcome['sql'] == [],
        'date.unrelated_observation',
    )
    result = outcome['result']
    require(
        type(result) is dict and set(result) == {'list'}, 'date.result_shape'
    )
    found = {}
    for entry in result['list']:
        require(
            type(entry) is dict
            and set(entry) == {'tuple'}
            and len(entry['tuple']) == 3,
            'date.operation_shape',
        )
        name = entry['tuple'][0]
        require(
            type(name) is dict and set(name) == {'scalar'},
            'date.operation_name',
        )
        name = name['scalar']
        require(
            name in OPERATIONS and name not in found, 'date.operation_scope'
        )
        found[name] = entry
    require(
        set(found) == OPERATIONS or set(found) == OPERATIONS - {'dates.read'},
        'date.operation_set',
    )
    return found


def authenticate(receipt, major):
    require(
        receipt['pandas_version'] == ('3.0.6' if major == '3' else '2.3.3'),
        'date.runtime_version',
    )
    require(CONSUMER in receipt['sources'], 'date.consumer_source')
    require(
        set(receipt['samples']) == {str(ordinal) for ordinal in range(256)},
        'date.selected_domain',
    )
    for sample in receipt['samples'].values():
        require(
            sample['consumer_kind'] == 'original-python'
            and sample['public_calls'] == 0,
            'date.original_route',
        )
        events = sample['source_consumer_events']
        require(len(events) == 1, 'date.consumer_events')
        event = events[0]
        require(
            event['path'] == CONSUMER
            and event['function'] == '_with_read'
            and event['source_sha256'] == receipt['sources'][CONSUMER],
            'date.consumer_identity',
        )
        require(
            event['operation_calls'] > 0 and event['native_calls'] == 0,
            'date.consumer_dispatch',
        )
        operations(sample['outcome'])


def entry(ordinal, old, normative):
    require(same(old['input'], normative['input']), 'date.changed_input')
    old_operations = operations(old['outcome'])
    new_operations = operations(normative['outcome'])
    require(
        digest(old['outcome']) != digest(normative['outcome']),
        'date.unchanged_case',
    )
    changed = [
        name
        for name in sorted(OPERATIONS)
        if not same(old_operations.get(name), new_operations.get(name))
    ]
    warnings_changed = not same(
        old['outcome']['warnings'], normative['outcome']['warnings']
    )
    require(changed or warnings_changed, 'date.unrelated_change')
    return {
        'coordinate': {'name': FAMILY, 'stream_seed': SEED, 'ordinal': ordinal},
        'input_sha256': digest(old['input']),
        'actual_old_sha256': digest(old['outcome']),
        'corrected_sha256': digest(normative['outcome']),
        'changed_operations': changed,
        'warnings_changed': warnings_changed,
    }


def generate(old, normative):
    authenticate(old, '2')
    authenticate(normative, '3')
    require(
        same(old['sources'], normative['sources']), 'date.source_difference'
    )
    entries = {}
    for ordinal in range(256):
        prior, corrected = (
            old['samples'][str(ordinal)],
            normative['samples'][str(ordinal)],
        )
        require(same(prior['input'], corrected['input']), 'date.changed_input')
        if digest(prior['outcome']) != digest(corrected['outcome']):
            identity = f'generator/{FAMILY}/{SEED}/{ordinal}/pandas-2'
            entries[identity] = entry(ordinal, prior, corrected)
    return {
        'format': 'p97-finite-date-reference-v1',
        'normative_pandas': '3.0.6',
        'sources': old['sources'],
        'cases': entries,
    }


def verify(case, old, normative, policy):
    require(same(policy, generate(old, normative)), 'date.policy_provenance')
    require(case['id'] in policy['cases'], 'date.case_scope')
    bound = policy['cases'][case['id']]
    coordinate = case['replay_input']['generator']
    require(same(coordinate, bound['coordinate']), 'date.coordinate')
    ordinal = str(coordinate['ordinal'])
    prior, corrected = old['samples'][ordinal], normative['samples'][ordinal]
    require(
        same(case['replay_input']['generator_input'], prior['input'])
        and same(prior['input'], corrected['input']),
        'date.changed_input',
    )
    correction = case['corrected_reference']
    require(
        correction is not None and correction['dv'] == 'DV114', 'date.decision'
    )
    require(
        same(case['reference'], correction['actual_old'])
        and same(correction['actual_old'], prior['outcome']),
        'date.actual_old',
    )
    require(
        same(correction['corrected'], corrected['outcome']),
        'date.normative_reference',
    )
    require(
        same(case['expected'], correction['native'])
        and same(correction['native'], corrected['outcome']),
        'date.exact_native',
    )
    require(
        correction['source_sha256'] == normative['sources'][CONSUMER],
        'date.correction_source',
    )
