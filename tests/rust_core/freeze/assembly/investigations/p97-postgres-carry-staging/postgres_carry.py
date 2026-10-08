# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public PostgreSQL carry dispatch with explicitly supplied fixture ownership."""

from __future__ import annotations

import contextlib
import copy
import sys
from pathlib import Path

from support import Rejected, require, same_json, sha

QUERY = 'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id'
OPTIONS = {'indices': [1, 2], 'num_context_samples': 5, 'random_seed': 7}
ORIGINAL = 'tests/rust_core/rfm/test_sampling_gaps.py'
CASES = {
    'carry/C14/postgres-nested-array': ('int[][]', 'ARRAY[[1,2],[3,4]]', False),
    'carry/C14/postgres-numeric-array-decimal128': (
        'numeric[]',
        'ARRAY[1.25,2.50]',
        True,
    ),
    'carry/C14/postgres-numeric-array-decimal256': (
        'numeric[]',
        'ARRAY[12345678901234567890.1234567890123456789,2]',
        True,
    ),
    'carry/C15/list-feature-null-first-postgres': (
        'text[]',
        "CASE WHEN user_id % 2 = 0 THEN ARRAY['a', 'b'] END",
        True,
    ),
    'carry/C24/postgres-nested-array-public-schema': (
        'text[][]',
        "ARRAY[['A','B'],['C','D']]",
        False,
    ),
}


def recipes() -> dict:
    result = {}
    for identity, (sql_type, fill, materialize) in CASES.items():
        result[identity] = {
            'operation': 'postgres.carry',
            'entrypoint': 'PredictiveQuery.materialize'
            if materialize
            else 'Schema.from_source',
            'input': {
                'carry_case_id': identity,
                'backend': 'postgres',
                'data': {'generator': 'shop', 'seed': 0},
                'sql_type': sql_type,
                'fill': fill,
                'restore_tags': materialize,
                'query': QUERY if materialize else None,
                'options': copy.deepcopy(OPTIONS) if materialize else {},
                'explicit_relational_model': identity.startswith('carry/C15/'),
            },
        }
    return result


def source_bindings(repo: Path) -> dict:
    names = (
        ORIGINAL,
        'scripts/parity/_parity/data.py',
        'scripts/parity/_parity/runner.py',
        'packages/sdm-client/src/sdm_client/schema.py',
        'packages/sdm-client/src/sdm_client/query.py',
        'packages/sdm-connectors/src/sdm_connectors/source.py',
    )
    current_paths = {
        'packages/sdm-client/src/sdm_client/schema.py': 'packages/structured-data-models-client/src/sdm_client/schema.py',
        'packages/sdm-client/src/sdm_client/query.py': 'packages/structured-data-models-client/src/sdm_client/query.py',
        'packages/sdm-connectors/src/sdm_connectors/source.py': 'packages/structured-data-models-connectors/src/sdm_connectors/source.py',
    }
    return {
        name: sha((repo / current_paths.get(name, name)).read_bytes())
        for name in names
    }


@contextlib.contextmanager
def owned_source(factory, frames, statements):
    try:
        manager = factory(frames, *statements)
        source = manager.__enter__()
    except Exception:
        raise Rejected('postgres_carry.source_setup_failed') from None
    try:
        yield source
    finally:
        exception = sys.exc_info()
        try:
            manager.__exit__(*exception)
        except Exception:
            raise Rejected('postgres_carry.source_cleanup_failed') from None


def operation(
    input: dict, stack: contextlib.ExitStack, *, postgres_source_factory=None
):
    require(
        callable(postgres_source_factory),
        'postgres_carry.owned_source_required',
    )
    require(type(input) is dict, 'postgres_carry.input')
    identity = input.get('carry_case_id')
    require(type(identity) is str and identity in CASES, 'postgres_carry.case')
    require(
        same_json(input, recipes()[identity]['input']),
        'postgres_carry.input_binding',
    )

    from _parity.data import generate
    from sdm_client import Column, KumoRelational, PredictiveQuery, Schema
    from sdm_connectors import PostgresSource

    frames = generate('shop', 0)
    statements = (
        f'ALTER TABLE {{schema}}.users ADD COLUMN tags {input["sql_type"]}',
        f'UPDATE {{schema}}.users SET tags = {input["fill"]}',
    )
    source = stack.enter_context(
        owned_source(postgres_source_factory, frames, statements)
    )
    require(isinstance(source, PostgresSource), 'postgres_carry.source_backend')
    schema = Schema.from_source(source)
    if not input['restore_tags']:
        return schema, None
    schema['users'].columns['tags'] = Column(
        'tags', 'string', 'multicategorical'
    )
    options = copy.deepcopy(input['options'])
    if input['explicit_relational_model']:
        options['model'] = KumoRelational()
    request = PredictiveQuery(input['query'], schema).materialize(**options)
    return schema, request
