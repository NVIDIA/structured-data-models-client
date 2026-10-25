# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constructor-only frame journeys; no model, sampler, or warehouse."""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from sdm_client import Schema, StructuredDataError, Table
from sdm_client._routing import ROUTES

PATHS = 'schema.from-data/frames/frames,schema.from-data/mapping/frames'


def describe(monkeypatch, frames, *, core, **options):
    monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        schema = Schema.from_data(frames, **options)
    return schema, [str(item.message) for item in caught]


def projection(schema):
    return (
        [
            (
                name,
                [(c.name, c.dtype, str(c.stype), c.untyped) for c in table.columns.values()],
                table.primary_key,
                table.time_column,
                table.duplicate_keys,
            )
            for name, table in schema.tables.items()
        ],
        [
            (r.source_table, r.source_column, r.target_table, r.target_column)
            for r in schema.relationships
        ],
    )


def users():
    return pd.DataFrame(
        {
            'user_id': pd.Series([1, 2, 3], dtype='int16'),
            'age': pd.Series([21, 31, 41], dtype='int32'),
            'active': pd.Series([True, False, True], dtype='boolean'),
            'score': pd.Series([1.5, None, 2.5], dtype='Float32'),
            'name': pd.Series(['a', 'b', 'a'], dtype='str'),
        }
    )


@pytest.mark.parametrize('variant', ['ordinary', 'nullable', 'uint64', 'empty', 'all_null'])
def test_public_primitive_description_matches_original(monkeypatch, variant):
    frame = users()
    if variant == 'nullable':
        frame['age'] = pd.Series([1, None, 3], dtype='Int64')
    elif variant == 'uint64':
        frame['user_id'] = pd.Series([2**63, 2**63 + 1, 2**63 + 2], dtype='uint64')
    elif variant == 'empty':
        frame = frame.iloc[:0]
    elif variant == 'all_null':
        frame['name'] = pd.Series([None] * 3, dtype='str')
    old, old_warnings = describe(monkeypatch, {'users': frame}, core=False)
    new, new_warnings = describe(monkeypatch, {'users': frame}, core=True)
    assert projection(new) == projection(old)
    assert new_warnings == old_warnings
    assert isinstance(new['users'], Table)
    assert new['users'].data is frame
    assert all(c._source is None for c in new['users'].columns.values())


def test_public_relationship_and_override_assembly_matches_original(monkeypatch):
    frames = {
        'users': users(),
        'orders': pd.DataFrame({'order_id': [10, 11, 12], 'user_id': [1, 1, 3]}),
    }
    for options in ({}, {'primary_keys': {'users': 'user_id'}, 'relationships': []}):
        old, old_warnings = describe(monkeypatch, frames, core=False, **options)
        new, new_warnings = describe(monkeypatch, frames, core=True, **options)
        assert projection(new) == projection(old)
        assert new_warnings == old_warnings


@pytest.mark.parametrize('options', [
    {'primary_keys': {'users': 'missing'}},
    {'time_columns': {'users': 'missing'}},
    {'primary_keys': {'absent': 'user_id'}},
])
def test_original_override_error_is_preserved(monkeypatch, options):
    errors = []
    for core in (False, True):
        with pytest.raises(StructuredDataError) as caught:
            describe(monkeypatch, {'users': users()}, core=core, **options)
        errors.append((caught.value.code, str(caught.value)))
    assert errors[0] == errors[1]


def test_first_table_warning_survives_later_column_failure(monkeypatch):
    first = pd.DataFrame({'user_id': pd.Series([*range(99), None], dtype='Int64')})
    second = pd.DataFrame([[1, 2]], columns=['same', 'same'])
    outcomes = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(StructuredDataError) as failed:
                Schema.from_data({'users': first, 'later': second})
        outcomes.append((str(failed.value), [str(item.message) for item in caught]))
    assert outcomes[0][1], 'the original constructor must warn before the later error'
    assert outcomes[1] == outcomes[0]


def test_first_failure_never_reads_later_custom_frame(monkeypatch):
    touched = []

    class Later(pd.DataFrame):
        def __arrow_c_stream__(self, requested_schema=None):
            touched.append('export')
            raise AssertionError('a later native export ran before the earlier refusal')

    monkeypatch.setenv('SDM_CORE_ROUTES', PATHS)
    first = pd.DataFrame([[1, 2]], columns=['same', 'same'])
    with pytest.raises(StructuredDataError, match='duplicate column'):
        Schema.from_data({'first': first, 'later': Later({'x': [1]})})
    assert touched == []


def test_snapshot_metadata_is_separate_but_rows_rebind_fresh(monkeypatch):
    frame = users()
    schema, _ = describe(monkeypatch, {'users': frame}, core=True)
    snapshot = schema.snapshot()
    schema['users']['age'].stype = 'categorical'
    assert str(snapshot['users']['age'].stype) == 'numerical'
    frame.loc[:, 'user_id'] = 1
    assert snapshot['users'].data is frame
    assert schema['users'].data is frame
    again, _ = describe(monkeypatch, {'users': frame}, core=True)
    old, _ = describe(monkeypatch, {'users': frame}, core=False)
    assert projection(again) == projection(old)
    assert again['users'].primary_key != snapshot['users'].primary_key
    restored = pickle.loads(pickle.dumps(snapshot))
    assert projection(restored) == projection(snapshot)
    assert all(c._source is None for c in restored['users'].columns.values())


def test_constructor_default_stays_old_and_custom_object_family_is_unchanged(monkeypatch):
    monkeypatch.delenv('SDM_CORE_ROUTES', raising=False)
    assert all(ROUTES[path] == 'old' for path in PATHS.split(','))
    frame = pd.DataFrame({'object_id': np.array([1, 2, 3], dtype=object)})
    schema = Schema.from_data({'objects': frame})
    assert schema['objects']['object_id'].dtype == 'int64'
