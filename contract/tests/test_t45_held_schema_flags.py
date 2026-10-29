# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The materialization schema retains the host's actual inference flags.

Complete DTOs below are independent fixtures, not frozen observations or a
claim of SDK execution. The current reader must validate their original bytes.
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
    't45_held_schema_compare', ROOT / 'scripts/rust_core/compare.py'
)
QUERY = 'PREDICT events.empty FOR EACH events.id'
RELATION = {'source': ['stored_events']}


def description():
    return {
        'tables': [
            {
                'name': 'events',
                'source': copy.deepcopy(RELATION),
                'columns': [
                    {
                        'name': 'id',
                        'dtype': 'int',
                        'received': {'type': 'int64'},
                        'nullable': False,
                        'unique': True,
                        'declared_type': 'BIGINT',
                        'exact_number': None,
                        'stype': 'ID',
                        'stype_before_key': 'numerical',
                    },
                    {
                        'name': 'empty',
                        'dtype': 'string',
                        'received': {'type': 'utf8'},
                        'nullable': True,
                        'unique': False,
                        'declared_type': None,
                        'exact_number': None,
                        'stype': 'categorical',
                        'stype_before_key': None,
                        'untyped': True,
                    },
                    {
                        'name': 'first',
                        'dtype': 'date',
                        'received': {
                            'type': 'timestamp',
                            'unit': 'us',
                            'zone': None,
                        },
                        'nullable': True,
                        'unique': False,
                        'declared_type': 'TIMESTAMP',
                        'stype': 'timestamp',
                        'stype_before_key': None,
                    },
                ],
                'primary_key': ['id'],
                'time_column': 'first',
                'time_column_inferred': None,
                'duplicate_keys': {'t': 'u64', 'v': '0'},
            }
        ],
        'links': [],
    }


def expected():
    # Explicit booleans are present even when false. The ID role is not an
    # stype assignment; the time-column setter is one, even if inferred.
    return {
        'tables': [
            {
                'name': 'events',
                'source': copy.deepcopy(RELATION),
                'columns': [
                    {
                        'name': 'id',
                        'dtype': 'int64',
                        'stype': 'ID',
                        'stype_before_key': 'numerical',
                        'untyped': False,
                        'stype_declared': False,
                        'source': {
                            'relation': copy.deepcopy(RELATION),
                            'dtype': 'int',
                            'received': {'type': 'int64'},
                            'nullable': False,
                            'unique': True,
                            'declared_type': 'BIGINT',
                        },
                    },
                    {
                        'name': 'empty',
                        'dtype': 'string',
                        'stype': 'categorical',
                        'stype_before_key': None,
                        'untyped': True,
                        'stype_declared': False,
                        'source': {
                            'relation': copy.deepcopy(RELATION),
                            'dtype': 'string',
                            'received': {'type': 'utf8'},
                            'nullable': True,
                            'unique': False,
                        },
                    },
                    {
                        'name': 'first',
                        'dtype': 'timestamp[us]',
                        'stype': 'timestamp',
                        'stype_before_key': None,
                        'untyped': False,
                        'stype_declared': True,
                        'source': {
                            'relation': copy.deepcopy(RELATION),
                            'dtype': 'date',
                            'received': {
                                'type': 'timestamp',
                                'unit': 'us',
                                'zone': None,
                            },
                            'nullable': True,
                            'unique': False,
                            'declared_type': 'TIMESTAMP',
                        },
                    },
                ],
                'primary_key': ['id'],
                'time_column': 'first',
                'duplicate_keys': {'t': 'u64', 'v': '0'},
            }
        ],
        'links': [],
    }


def test_complete_held_schema_keeps_flags_facts_and_original_order():
    described = description()
    before = copy.deepcopy(described)
    assert COMPARE.held_schema(described) == expected()
    assert described == before


@pytest.mark.parametrize(
    ('inferred', 'final', 'declared'),
    [
        (None, None, (False, False, False)),
        (None, 'first', (True, False, False)),
        ('first', None, (True, False, False)),
        ('first', 'first', (True, False, False)),
        ('first', 'last', (True, True, False)),
        (None, 'last', (False, True, False)),
    ],
)
def test_time_assignment_history_keeps_both_selected_columns_only(
    inferred, final, declared
):
    described = description()
    table = described['tables'][0]
    for name in ('last', 'unselected'):
        column = copy.deepcopy(table['columns'][-1])
        column['name'] = name
        table['columns'].append(column)
    table.update(time_column_inferred=inferred, time_column=final)
    columns = COMPARE.held_schema(described)['tables'][0]['columns']
    assert [column['name'] for column in columns] == [
        'id',
        'empty',
        'first',
        'last',
        'unselected',
    ]
    assert tuple(column['stype_declared'] for column in columns[2:]) == declared
    assert [column['stype_declared'] for column in columns[:2]] == [
        False,
        False,
    ]


@pytest.mark.parametrize('state', ['missing', 'false', 'true'])
def test_untyped_uses_the_described_fact_with_the_same_host_default(state):
    described = description()
    empty = described['tables'][0]['columns'][1]
    if state == 'missing':
        empty.pop('untyped')
    else:
        empty['untyped'] = state == 'true'
    written = COMPARE.held_schema(described)['tables'][0]['columns'][1]
    assert written['untyped'] is (state == 'true')
    assert written['stype_declared'] is False
    assert written['dtype'] == 'string' and written['stype'] == 'categorical'


def test_flags_are_table_local_and_source_and_relationship_fields_stay_exact():
    described = description()
    second = copy.deepcopy(described['tables'][0])
    second.update(
        name='other',
        source={'source': ['other']},
        time_column=None,
        time_column_inferred=None,
        duplicate_keys=None,
    )
    second['columns'][0].update(
        dtype='string', received={'type': 'utf8'}, exact_number='declared_key'
    )
    described['tables'].append(second)
    link = [
        {'table': 'events', 'column': 'id'},
        {'table': 'other', 'column': 'id'},
    ]
    described['links'] = [link]
    first, other = COMPARE.held_schema(described)['tables']
    assert first == expected()['tables'][0]
    assert other['name'] == 'other' and other['source'] == {'source': ['other']}
    assert 'duplicate_keys' not in other
    assert other['primary_key'] == ['id'] and other['time_column'] is None
    assert [column['stype_declared'] for column in other['columns']] == [
        False
    ] * 3
    assert other['columns'][0]['source'] == {
        'relation': {'source': ['other']},
        'dtype': 'string',
        'received': {'type': 'utf8'},
        'nullable': False,
        'unique': True,
        'declared_type': 'BIGINT',
        'exact_number': 'declared_key',
    }
    assert COMPARE.held_schema(described)['links'] == [link]


def test_actual_sdk_build_and_plain_serialization_match_the_independent_complete_schema():
    import duckdb
    from sdm_client import Schema
    from sdm_client._rust import schema as bridge
    from sdm_client._rust import spec
    from sdm_connectors import DuckDBSource

    described = description()
    before = copy.deepcopy(described)
    with duckdb.connect() as connection:
        schema = bridge.build(Schema, DuckDBSource(connection), described)
        written = spec.held_schema(schema)
    assert written == expected()
    assert COMPARE.held_schema(described) == written
    assert described == before


def materialization_observation(schema):
    core = actual_materialization()
    core['described'] = description()
    record = core['materialization_preparations'][0]
    record['input_reads'][0] = _bytes(1, json.dumps(schema).encode())
    record['input_reads'][1] = _bytes(2, json.dumps(QUERY).encode())
    return {
        'core': core,
        'identity': {'operations': ['prepare_materialize'], 'contract': 35},
    }


def case_input_failures(observation, *, edited=False):
    case = {
        'id': 'held-schema-flags',
        'core': {
            'materialize': {
                'provider': {'provider': 'sql', 'dialect': 'duckdb'},
                'indices': [{'t': 'i64', 'v': '7'}],
                'anchor_time': None,
                'seed': None,
            }
        },
        'input': {'pql': QUERY},
    }
    if edited:
        case['input']['edits'] = [
            {
                'op': 'set_stype',
                'table': 'events',
                'column': 'empty',
                'stype': 'text',
            }
        ]
    report = COMPARE.Report()
    COMPARE._case_inputs(report, case, 'python-core', observation)
    return report.failures


def test_current_actual_materialization_input_is_checked_against_the_complete_held_schema():
    observed = materialization_observation(expected())
    before = copy.deepcopy(observed)
    assert case_input_failures(observed) == []
    assert observed == before


@pytest.mark.parametrize(
    'changed',
    [
        'drop-untyped',
        'flip-untyped',
        'drop-declared',
        'flip-declared',
        'dtype',
        'order',
    ],
)
def test_schema_input_equality_still_refuses_every_unapproved_change(changed):
    schema = expected()
    columns = schema['tables'][0]['columns']
    if changed == 'drop-untyped':
        columns[1].pop('untyped')
    elif changed == 'flip-untyped':
        columns[1]['untyped'] = False
    elif changed == 'drop-declared':
        columns[2].pop('stype_declared')
    elif changed == 'flip-declared':
        columns[2]['stype_declared'] = False
    elif changed == 'dtype':
        columns[0]['dtype'] = 'float64'
    else:
        columns.reverse()
    failures = case_input_failures(materialization_observation(schema))
    assert [rule for rule, _ in failures] == ['inputs']


def test_caller_edited_schema_is_not_reconstructed_from_the_original_description():
    schema = expected()
    schema['tables'][0]['columns'][1].update(stype='text', stype_declared=True)
    observed = materialization_observation(schema)
    assert case_input_failures(observed)
    assert case_input_failures(observed, edited=True) == []


@pytest.mark.parametrize(
    'malformed', [None, [], {'tables': []}, {'tables': [None]}]
)
def test_non_description_passthrough_remains_unchanged(malformed):
    assert COMPARE.held_schema(malformed) is malformed
