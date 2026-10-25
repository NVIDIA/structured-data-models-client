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
from sdm_client.core.warnings import SDMWarning

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
                [
                    (c.name, c.dtype, str(c.stype), c.untyped)
                    for c in table.columns.values()
                ],
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
            'active': pd.Series([True, False, True], dtype='bool'),
            'score': pd.Series([1.5, np.nan, 2.5], dtype='float32'),
        }
    )


@pytest.mark.parametrize(
    'variant', ['ordinary', 'nullable', 'uint64', 'empty', 'all_null']
)
def test_public_primitive_description_matches_original(monkeypatch, variant):
    frame = users()
    if variant == 'nullable':
        frame['age'] = pd.Series([1, np.nan, 3], dtype='float64')
    elif variant == 'uint64':
        frame['user_id'] = pd.Series(
            [2**63, 2**63 + 1, 2**63 + 2], dtype='uint64'
        )
    elif variant == 'empty':
        frame = frame.iloc[:0]
    elif variant == 'all_null':
        frame['score'] = pd.Series([np.nan] * 3, dtype='float32')
    old, old_warnings = describe(monkeypatch, {'users': frame}, core=False)
    new, new_warnings = describe(monkeypatch, {'users': frame}, core=True)
    assert projection(new) == projection(old)
    assert new_warnings == old_warnings
    assert isinstance(new['users'], Table)
    assert new['users'].data is frame
    assert all(c._source is None for c in new['users'].columns.values())


def test_public_relationship_and_override_assembly_matches_original(
    monkeypatch,
):
    frames = {
        'users': users(),
        'orders': pd.DataFrame(
            {'order_id': [10, 11, 12], 'user_id': [1, 1, 3]}
        ),
    }
    for options in (
        {},
        {'primary_keys': {'users': 'user_id'}, 'relationships': []},
    ):
        old, old_warnings = describe(monkeypatch, frames, core=False, **options)
        new, new_warnings = describe(monkeypatch, frames, core=True, **options)
        assert projection(new) == projection(old)
        assert new_warnings == old_warnings


@pytest.mark.parametrize(
    'options',
    [
        {'primary_keys': {'users': 'missing'}},
        {'time_columns': {'users': 'missing'}},
        {'primary_keys': {'absent': 'user_id'}},
    ],
)
def test_original_override_error_is_preserved(monkeypatch, options):
    errors = []
    for core in (False, True):
        with pytest.raises(StructuredDataError) as caught:
            describe(monkeypatch, {'users': users()}, core=core, **options)
        errors.append((caught.value.code, str(caught.value)))
    assert errors[0] == errors[1]


def test_first_table_warning_survives_later_column_failure(monkeypatch):
    first = pd.DataFrame(
        {'user_id': pd.Series([*range(99), np.nan], dtype='float64')}
    )
    second = pd.DataFrame([[1, 2]], columns=['same', 'same'])
    outcomes = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(StructuredDataError) as failed:
                Schema.from_data({'users': first, 'later': second})
        outcomes.append(
            (str(failed.value), [str(item.message) for item in caught])
        )
    assert outcomes[0][1], (
        'the original constructor must warn before the later error'
    )
    assert outcomes[1] == outcomes[0]


@pytest.mark.parametrize('later_invalid', [False, True])
def test_inferred_warning_precedes_whole_constructor_override(
    monkeypatch, later_invalid
):
    frame = pd.DataFrame(
        {
            'user_id': pd.Series([*range(99), np.nan], dtype='float64'),
            'alternate_id': range(100),
        }
    )
    frames = {'users': frame}
    options = {'primary_keys': {'users': 'alternate_id'}}
    if later_invalid:
        frames['later'] = pd.DataFrame([[1, 2]], columns=['same', 'same'])
        options = {'primary_keys': {'users': 'missing'}}
    found = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                schema = Schema.from_data(frames, **options)
            except StructuredDataError as error:
                outcome = str(error)
            else:
                outcome = projection(schema)
        found.append((outcome, [str(w.message) for w in caught]))
    assert found[0][1]
    assert found[1] == found[0]


def test_first_failure_never_reads_later_custom_frame(monkeypatch):
    touched = []

    class Later(pd.DataFrame):
        def __arrow_c_stream__(self, requested_schema=None):
            touched.append('export')
            raise AssertionError(
                'a later native export ran before the earlier refusal'
            )

    monkeypatch.setenv('SDM_CORE_ROUTES', PATHS)
    first = pd.DataFrame([[1, 2]], columns=['same', 'same'])
    with pytest.raises(StructuredDataError, match='duplicate column'):
        Schema.from_data({'first': first, 'later': Later({'x': [1]})})
    assert touched == []


def test_warning_as_error_prevents_later_read_and_override(monkeypatch):
    touched = []

    class Later(pd.DataFrame):
        @property
        def columns(self):
            touched.append('later columns')
            return super().columns

    later = Later({'x': [1]})
    touched.clear()
    first = pd.DataFrame(
        {'user_id': pd.Series([*range(99), np.nan], dtype='float64')}
    )
    monkeypatch.setenv('SDM_CORE_ROUTES', PATHS)
    with warnings.catch_warnings():
        warnings.simplefilter('error', SDMWarning)
        with pytest.raises(SDMWarning, match='have no value'):
            Schema.from_data(
                {'users': first, 'later': later},
                primary_keys={'users': 'missing'},
            )
    assert touched == []
    # No partial frame cache can survive and overwrite a fresh constructor.
    first.loc[99, 'user_id'] = 99
    schema, _ = describe(monkeypatch, {'users': first}, core=True)
    assert schema['users'].primary_key == 'user_id'


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


def test_constructor_default_stays_old_and_custom_object_family_is_unchanged(
    monkeypatch,
):
    monkeypatch.delenv('SDM_CORE_ROUTES', raising=False)
    assert all(ROUTES[path] == 'old' for path in PATHS.split(','))
    frame = pd.DataFrame({'object_id': np.array([1, 2, 3], dtype=object)})
    schema = Schema.from_data({'objects': frame})
    assert schema['objects']['object_id'].dtype == 'int64'


def test_typed_describe_result_preserves_arrow_role_without_changing_sql():
    from sdm_client._rust import results

    table = results.expect(
        {'op': 'describe_source', 'description': 1, 'frame_rows': 2},
        results.Described,
    )
    assert table.frame_rows == 2
    sql = results.expect(
        {'op': 'describe_source', 'description': 1}, results.Described
    )
    assert sql.frame_rows is None


@pytest.mark.parametrize(
    'axis',
    ['ambiguous_key', 'keyless_shared', 'duplicate_key'],
)
def test_primitive_final_notices_and_key_remedies_match_original(
    monkeypatch, axis
):
    if axis == 'ambiguous_key':
        frames = {
            'a': pd.DataFrame({'id': [1, 2, 3]}),
            'b': pd.DataFrame({'id': [4, 5, 6]}),
        }
    elif axis == 'keyless_shared':
        frames = {
            'items': pd.DataFrame({'SKU': [11, 12, 13]}),
            'sales': pd.DataFrame(
                {
                    'sale_id': [1, 2, 3],
                    'SKU': [11, 11, 12],
                }
            ),
        }
    else:
        frames = {
            'users': pd.DataFrame(
                {'user_id': [*range(99), 98], 'age': range(100)}
            )
        }
    old, old_warnings = describe(monkeypatch, frames, core=False)
    new, new_warnings = describe(monkeypatch, frames, core=True)
    assert old_warnings, (
        'the reference control must reach its meaningful notice'
    )
    assert projection(new) == projection(old)
    assert new_warnings == old_warnings


@pytest.mark.parametrize('option', ['primary_keys', 'time_columns'])
@pytest.mark.parametrize(
    'order', [('z_missing', 'a_missing'), ('a_missing', 'z_missing')]
)
def test_native_override_mapping_order_controls_first_error(
    monkeypatch, option, order
):
    found = []
    for core in (False, True):
        with pytest.raises(StructuredDataError) as caught:
            describe(
                monkeypatch,
                {'users': users()},
                core=core,
                **{option: {name: 'user_id' for name in order}},
            )
        found.append(str(caught.value))
    assert order[0] in found[0]
    assert found[1] == found[0]


def test_unproved_native_text_probe_stays_on_unchanged_default_route(
    monkeypatch,
):
    from sdm_client._routing import RouteError

    frame = pd.DataFrame(
        {
            'user_id': [1, 2, 3],
            'stamp': pd.Series(
                [
                    '2024-01-01',
                    '2024-01-02',
                    '2024-01-03',
                ],
                dtype='str',
            ),
        }
    )
    monkeypatch.delenv('SDM_CORE_ROUTES', raising=False)
    with pytest.warns(SDMWarning, match='holds dates but is stored as text'):
        legacy = Schema.from_data({'users': frame})
    assert legacy['users'].time_column is None
    monkeypatch.setenv('SDM_CORE_ROUTES', PATHS)
    # A development route gap is not a newly admitted SDK refusal or a
    # warning the new core is allowed to suppress. Default remains old.
    with pytest.raises(RouteError, match='numeric and Boolean'):
        Schema.from_data({'users': frame})


@pytest.mark.parametrize('key', [('user_id', 'user_id'), ('user_id', 'score')])
def test_constructor_key_sequences_and_native_composite_examples(
    monkeypatch, key
):
    frame = pd.DataFrame(
        {
            'user_id': [*range(99), 98],
            'score': [float(i) for i in range(99)] + [98.0],
        }
    )
    old, old_warnings = describe(
        monkeypatch, {'users': frame}, core=False, primary_keys={'users': key}
    )
    new, new_warnings = describe(
        monkeypatch, {'users': frame}, core=True, primary_keys={'users': key}
    )
    assert projection(new) == projection(old)
    assert new_warnings == old_warnings


@pytest.mark.parametrize(
    'family',
    ['text', 'date_text', 'object', 'nullable', 'datetime', 'categorical'],
)
def test_unfinished_native_families_are_not_forced_through_numeric_constructor(
    monkeypatch, family
):
    from sdm_client._routing import RouteError

    values = {
        'text': pd.Series(['a', 'b'], dtype='str'),
        'date_text': pd.Series(['2024-01-01', '2024-01-02'], dtype='str'),
        'object': pd.Series([1, 2], dtype=object),
        'nullable': pd.Series([1, None], dtype='Int64'),
        'datetime': pd.Series(pd.to_datetime(['2024-01-01', '2024-01-02'])),
        'categorical': pd.Series(['a', 'b'], dtype='category'),
    }
    monkeypatch.setenv('SDM_CORE_ROUTES', PATHS)
    with pytest.raises(RouteError, match='numeric and Boolean'):
        Schema.from_data({'users': pd.DataFrame({'value': values[family]})})


@pytest.mark.parametrize(
    'argument',
    ['tables', 'frame', 'keys', 'key_value', 'times', 'relationships'],
)
def test_development_constructor_does_not_traverse_custom_argument_hooks(
    monkeypatch, argument
):
    from sdm_client._routing import RouteError

    class CustomDict(dict):
        def items(self):
            raise AssertionError('unproved mapping hook ran')

    class CustomFrame(pd.DataFrame):
        @property
        def columns(self):
            raise AssertionError('unproved frame hook ran')

    class CustomKey:
        def __iter__(self):
            raise AssertionError('unproved key hook ran')

    frames = {'users': users()}
    options = {}
    if argument == 'tables':
        frames = CustomDict(frames)
    elif argument == 'frame':
        frames = {'users': CustomFrame({'user_id': [1, 2, 3]})}
    elif argument == 'keys':
        options['primary_keys'] = CustomDict(users='user_id')
    elif argument == 'key_value':
        options['primary_keys'] = {'users': CustomKey()}
    elif argument == 'times':
        options['time_columns'] = CustomDict(users=None)
    else:
        options['relationships'] = CustomKey()
    monkeypatch.setenv('SDM_CORE_ROUTES', PATHS)
    with pytest.raises(RouteError, match='ordinary built-in'):
        Schema.from_data(frames, **options)
