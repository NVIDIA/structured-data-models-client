# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The approved FK repair keeps complete requests strict and valid links local."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'p3f2_payload', Path(__file__).with_name('_p3f2_payload.py')
)
assert SPEC is not None and SPEC.loader is not None
checked = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checked)
CASE = 'shape.diamond-from-source'


def pair():
    def body(dtype, first, second):
        return {
            'metadata': {'request_id': dtype},
            'schema': {
                'related_tables': {
                    'orders': {'columns': {'cid': {'dtype': dtype}}},
                },
                'relationships': [
                    {
                        'source_table': 'orders',
                        'source_columns': ['cid'],
                        'target_table': 'customers',
                        'target_columns': ['__node_id'],
                    }
                ],
            },
            'context': {
                'related_tables': {
                    'orders': {
                        'columns': [
                            'instance_id',
                            'node',
                            'a',
                            'b',
                            'c',
                            'd',
                            'tag',
                            'cid',
                        ],
                        'rows': [
                            [0, 1, 2, 3, 4, 5, 6, first],
                            [1, 2, 2, 3, 4, 5, 7, second],
                        ],
                    },
                    'customers': {
                        'columns': ['instance_id', '__node_id', 'tag'],
                        'rows': [[0, 100, 6], [1, 200, 7]],
                    },
                },
            },
            'predict': {'related_tables': {}},
            'settings': {'seed': 42},
        }

    def line(value):
        return {
            'public': {
                'payloads': [
                    {'method': 'POST', 'path': '/predict', 'parsed': value}
                ]
            }
        }

    return line(body('float64', None, {'number': '200.0'})), line(
        body('int64', {'number': '100'}, {'number': '200'})
    )


def test_only_approved_valid_local_fill_and_dtype_are_accepted():
    old, new = pair()
    assert checked.payload_problem(old, new, CASE) is None
    assert (
        checked.payload_problem(old, new, 'shape.chain.addresses') is not None
    )
    assert checked.payload_problem(old, new, CASE + '@sqlite') is None


@pytest.mark.parametrize(
    'mutation',
    [
        'unknown',
        'other-instance',
        'missing-target',
        'old-key',
        'old-key-null',
        'unrelated',
        'order',
        'length',
        'method',
        'path',
    ],
)
def test_invalid_links_and_unapproved_payload_changes_are_refused(mutation):
    old, new = pair()
    body = new['public']['payloads'][0]['parsed']
    orders = body['context']['related_tables']['orders']['rows']
    customers = body['context']['related_tables']['customers']['rows']
    if mutation == 'unknown':
        orders[0][7] = 999
    elif mutation == 'other-instance':
        orders[0][7] = 200
    elif mutation == 'missing-target':
        customers.pop(0)
    elif mutation == 'old-key':
        orders[1][7] = 100
    elif mutation == 'old-key-null':
        orders[1][7] = None
    elif mutation == 'unrelated':
        body['settings']['seed'] = 43
    elif mutation == 'order':
        orders.reverse()
    elif mutation == 'length':
        orders.pop()
    else:
        new['public']['payloads'][0][mutation] = 'changed'
    assert checked.payload_problem(old, new, CASE) is not None


def test_a_real_fk_repair_witness_cannot_keep_every_old_missing_key():
    old, new = pair()
    new['public']['payloads'][0]['parsed']['context']['related_tables'][
        'orders'
    ]['rows'][0][7] = None
    assert checked.payload_problem(old, new, CASE) is not None


def test_unaffected_equal_payloads_stay_strict_and_inputs_are_not_mutated():
    old, _ = pair()
    frozen = copy.deepcopy(old)
    assert checked.payload_problem(old, old, 'shape.chain.addresses') is None
    assert old == frozen


def test_missing_owner_admission_cannot_excuse_the_difference(monkeypatch):
    old, new = pair()
    monkeypatch.setattr(checked.comparison(), 'more_exact', lambda path: {})
    checked.authority.cache_clear()
    try:
        assert checked.payload_problem(old, new, CASE) is not None
    finally:
        checked.authority.cache_clear()


def test_rust_body_fields_use_the_same_authority_and_reached_target_checks():
    old, new = pair()
    body = new['public']['payloads'][0]['parsed']
    document = {
        'observed': {CASE: {'core': {'predicted': {'body': body}}}},
        'reference': {CASE: old},
    }
    assert checked.rust_problems(document) == {CASE: None}
    body['context']['related_tables']['orders']['rows'][0][7] = 200
    assert checked.rust_problems(document)[CASE] is not None
    body['context']['related_tables']['orders']['rows'][0][7] = 100
    body['settings']['seed'] = 9
    assert checked.rust_problems(document)[CASE] is not None
    document['observed']['shape.chain.addresses'] = document['observed'][CASE]
    assert 'shape.chain.addresses' not in checked.rust_problems(document)
