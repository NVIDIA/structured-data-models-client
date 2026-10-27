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


def scalar_only_pair(case):
    """The actual Avito/Stack approved row positions have NO dtype exemption."""
    old, new = pair()
    for line, missing in ((old, True), (new, False)):
        body = line['public']['payloads'][0]['parsed']
        if case.endswith('avito'):
            table, column, position = 'AdsInfo', 'LocationID', 3
            columns = [
                'instance_id',
                'AdID',
                'Price',
                'LocationID',
                'CategoryID',
            ]
            rows = [
                [0, 10, 6, None if missing else 100, None],
                [1, 20, 7, 200.0, None],
            ]
            targets = {
                'Location': {
                    'columns': ['instance_id', '__node_id'],
                    'rows': [[0, 100], [1, 200]],
                },
                'Category': {
                    'columns': ['instance_id', '__node_id'],
                    'rows': [],
                },
            }
            relations = [
                {
                    'source_table': table,
                    'source_columns': [column],
                    'target_table': 'Location',
                    'target_columns': ['__node_id'],
                },
                {
                    'source_table': table,
                    'source_columns': ['CategoryID'],
                    'target_table': 'Category',
                    'target_columns': ['__node_id'],
                },
            ]
            schema_columns = {
                'LocationID': {'dtype': 'float64'},
                'CategoryID': {'dtype': 'float64'},
            }
        else:
            table, column, position = 'posts', 'OwnerUserId', 4
            columns = [
                'instance_id',
                'Id',
                'PostTypeId',
                'CreationDate',
                'OwnerUserId',
                'ParentId',
            ]
            rows = [
                [0, 10, 2, '2020-01-01', None if missing else 100, 10.0],
                [1, 20, 2, '2020-01-01', 200.0, 20.0],
            ]
            targets = {
                'users': {
                    'columns': ['instance_id', '__node_id'],
                    'rows': [[0, 100], [1, 200]],
                }
            }
            relations = [
                {
                    'source_table': table,
                    'source_columns': [column],
                    'target_table': 'users',
                    'target_columns': ['__node_id'],
                },
                {
                    'source_table': table,
                    'source_columns': ['ParentId'],
                    'target_table': table,
                    'target_columns': ['Id'],
                },
            ]
            schema_columns = {
                'OwnerUserId': {'dtype': 'float64'},
                'ParentId': {'dtype': 'float64'},
            }
        body['schema'] = {
            'related_tables': {table: {'columns': schema_columns}},
            'relationships': relations,
        }
        body['context']['related_tables'] = {
            table: {'columns': columns, 'rows': rows},
            **targets,
        }
    return old, new, table, column, position


@pytest.mark.parametrize(
    'case',
    [
        'sample.parent-table.relbench-avito',
        'sample.parent-table.relbench-stack',
    ],
)
def test_scalar_only_approval_checks_actual_row_columns_without_dtype_paths(
    case,
):
    old, new, _, _, _ = scalar_only_pair(case)
    assert checked.payload_problem(old, new, case) is None
    document = {
        'observed': {
            case: {
                'core': {
                    'predicted': {
                        'body': new['public']['payloads'][0]['parsed']
                    }
                }
            }
        },
        'reference': {case: old},
    }
    assert checked.rust_problems(document) == {case: None}


@pytest.mark.parametrize(
    'case',
    [
        'sample.parent-table.relbench-avito',
        'sample.parent-table.relbench-stack',
    ],
)
@pytest.mark.parametrize(
    'change', ['other-instance', 'unknown', 'old-key', 'dtype', 'unlisted-cell']
)
def test_scalar_only_approval_never_excuses_wrong_targets_or_other_fields(
    case, change
):
    old, new, table, column, position = scalar_only_pair(case)
    body = new['public']['payloads'][0]['parsed']
    rows = body['context']['related_tables'][table]['rows']
    if change == 'other-instance':
        rows[0][position] = 200
    elif change == 'unknown':
        rows[0][position] = 999
    elif change == 'old-key':
        rows[1][position] = 100
    elif change == 'dtype':
        body['schema']['related_tables'][table]['columns'][column]['dtype'] = (
            'int64'
        )
    else:
        rows[0][2] = 'changed'
    assert checked.payload_problem(old, new, case) is not None
    document = {
        'observed': {case: {'core': {'predicted': {'body': body}}}},
        'reference': {case: old},
    }
    assert checked.rust_problems(document)[case] is not None


@pytest.mark.parametrize(
    ('before', 'after'),
    [('1.0', '1'), ('0.0', '-0.0'), ('9007199254740992', '9007199254740993')],
)
def test_unlisted_captured_numbers_keep_kind_sign_and_exact_value(
    before, after
):
    old, new = pair()
    old['public']['payloads'][0]['parsed']['context']['related_tables'][
        'orders'
    ]['rows'][0][6] = {'number': before}
    body = new['public']['payloads'][0]['parsed']
    body['context']['related_tables']['orders']['rows'][0][6] = {
        'number': after
    }
    assert checked.payload_problem(old, new, CASE) is not None
    document = {
        'observed': {CASE: {'core': {'predicted': {'body': body}}}},
        'reference': {CASE: old},
    }
    assert checked.rust_problems(document)[CASE] is not None


def test_equal_float_spellings_follow_the_stock_lossless_number_rule():
    old, new = pair()
    old['public']['payloads'][0]['parsed']['context']['related_tables'][
        'orders'
    ]['rows'][0][6] = {'number': '1e-5'}
    body = new['public']['payloads'][0]['parsed']
    body['context']['related_tables']['orders']['rows'][0][6] = {
        'number': '0.00001'
    }
    assert checked.payload_problem(old, new, CASE) is None
    document = {
        'observed': {CASE: {'core': {'predicted': {'body': body}}}},
        'reference': {CASE: old},
    }
    assert checked.rust_problems(document) == {CASE: None}
