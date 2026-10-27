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


@pytest.mark.parametrize('dated', [False, True])
def test_builtin_text_preserves_schema_and_date_advice_on_both_routes(
    monkeypatch, dated
):
    frame = pd.DataFrame(
        {
            'user_id': [1, 2, 3],
            'stamp': pd.Series(
                ['2024-01-01', '2024-01-02', '2024-01-03']
                if dated
                else ['first', 'second', 'third'],
                dtype='str',
            ),
        }
    )
    monkeypatch.delenv('SDM_CORE_ROUTES', raising=False)
    with warnings.catch_warnings(record=True) as old_notices:
        warnings.simplefilter('always')
        legacy = Schema.from_data({'users': frame})
    assert legacy['users'].time_column is None
    monkeypatch.setenv('SDM_CORE_ROUTES', PATHS)
    with warnings.catch_warnings(record=True) as new_notices:
        warnings.simplefilter('always')
        core = Schema.from_data({'users': frame})
    assert projection(core) == projection(legacy)
    assert core['users'].data is frame
    assert [(type(n.message), str(n.message)) for n in new_notices] == [
        (type(n.message), str(n.message)) for n in old_notices
    ]
    assert len(old_notices) == int(dated)
    if dated:
        assert isinstance(old_notices[0].message, SDMWarning)
        assert 'holds dates but is stored as text' in str(
            old_notices[0].message
        )


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
    [
        'object',
        'categorical',
    ],
)
def test_unfinished_native_families_are_not_forced_through_numeric_constructor(
    monkeypatch, family
):
    from sdm_client._routing import RouteError

    values = {
        'object': pd.Series([1, 2], dtype=object),
        'categorical': pd.Series(['a', 'b'], dtype='category'),
    }
    monkeypatch.setenv('SDM_CORE_ROUTES', PATHS)
    with pytest.raises(RouteError, match='numeric and Boolean'):
        Schema.from_data({'users': pd.DataFrame({'value': values[family]})})


@pytest.mark.parametrize(
    'argument',
    [
        'tables',
        'frame',
        'keys',
        'key_value',
        'times',
        'relationships',
        'nested_relationship',
    ],
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

        def __repr__(self):
            raise AssertionError('unproved nested repr hook ran')

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
    elif argument == 'nested_relationship':
        options['relationships'] = [{'nested': CustomKey()}]
    else:
        options['relationships'] = CustomKey()
    monkeypatch.setenv('SDM_CORE_ROUTES', PATHS)
    with pytest.raises(RouteError, match='ordinary built-in'):
        Schema.from_data(frames, **options)


@pytest.mark.parametrize('key', ['user_id', ['user_id'], ('user_id',)])
def test_native_key_form_controls_graph_and_survives_snapshot_and_held_schema(
    monkeypatch, key
):
    from sdm_client._rust.spec import held_schema

    frames = {
        'users': pd.DataFrame({'user_id': [1, 2]}),
        'orders': pd.DataFrame({'order_id': [10, 11], 'user_id': [1, 1]}),
    }
    old, old_warnings = describe(
        monkeypatch, frames, core=False, primary_keys={'users': key}
    )
    new, new_warnings = describe(
        monkeypatch, frames, core=True, primary_keys={'users': key}
    )
    assert projection(new) == projection(old)
    assert new_warnings == old_warnings
    assert type(new['users'].primary_key) is type(old['users'].primary_key)
    expected = 'scalar' if type(key) is str else 'sequence'
    for schema in (new, new.snapshot(), pickle.loads(pickle.dumps(new))):
        assert schema['users'].primary_key == old['users'].primary_key
        written = held_schema(schema)
        assert written['tables'][0]['frame_key_form'] == expected


def test_frame_warning_text_category_order_and_message_filter_are_preserved(
    monkeypatch,
):
    frame = pd.DataFrame({'user_id': [*range(99), np.nan]})
    outcomes = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            Schema.from_data({'users': frame})
        outcomes.append([(str(w.message), w.category) for w in caught])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'error',
                message='users.user_id identifies a row',
                category=SDMWarning,
            )
            with pytest.raises(
                SDMWarning, match='^users.user_id identifies a row'
            ):
                Schema.from_data({'users': frame})
    assert outcomes[1] == outcomes[0]
    assert outcomes[0] and outcomes[0][0][1] is SDMWarning


def test_only_first_declined_key_per_table_is_presented(monkeypatch):
    values = [*range(98), 0, 1]
    frames = {
        name: pd.DataFrame({'x_id': values, 'y_id': values})
        for name in ('a', 'b')
    }
    old, old_warnings = describe(monkeypatch, frames, core=False)
    new, new_warnings = describe(monkeypatch, frames, core=True)
    assert projection(new) == projection(old)
    assert new_warnings == old_warnings
    assert len(old_warnings) == 2
    assert old_warnings[0].startswith('a.x_id')
    assert old_warnings[1].startswith('b.x_id')


@pytest.mark.parametrize(
    'axis', ['key', 'time', 'earlier_link', 'int', 'none', 'warning']
)
def test_relationship_slots_are_validated_after_overrides_and_in_order(
    monkeypatch, axis
):
    from sdm_client import Relationship

    frame = pd.DataFrame({'user_id': [*range(99), 98], 'other_id': range(100)})
    options = {'relationships': [1]}
    if axis == 'key':
        options['primary_keys'] = {'users': 'missing'}
    elif axis == 'time':
        options['time_columns'] = {'users': 'missing'}
    elif axis == 'earlier_link':
        options['relationships'] = [
            Relationship('absent', 'user_id', 'users', 'user_id'),
            1,
        ]
    elif axis == 'none':
        options['relationships'] = [None]
    elif axis == 'warning':
        options['primary_keys'] = {'users': 'user_id'}
    outcomes = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(StructuredDataError) as failed:
                Schema.from_data({'users': frame}, **options)
        outcomes.append(
            (
                failed.value.code,
                str(failed.value),
                [(str(w.message), w.category) for w in caught],
            )
        )
    assert outcomes[1] == outcomes[0]
    if axis == 'int':
        assert 'got int: 1.' in outcomes[0][1]
    elif axis == 'none':
        assert 'got NoneType: None.' in outcomes[0][1]


def test_retained_schema_error_traceback_does_not_own_frame_streams(
    monkeypatch,
):
    from sdm_client import _native

    monkeypatch.setenv('SDM_CORE_ROUTES', PATHS)
    with pytest.raises(StructuredDataError) as retained:
        Schema.from_data(
            {'users': pd.DataFrame({'user_id': range(100)})},
            primary_keys={'users': 'missing'},
        )

    # Inspect the genuine traceback while it is retained. Both a direct
    # loop-local alias and a stream nested in the call's dictionaries count.
    def owns(value, seen):
        if isinstance(value, (_native.ArrowStream, _native.Run)):
            return True
        if id(value) in seen:
            return False
        seen.add(id(value))
        if type(value) is dict:
            return any(owns(item, seen) for item in value.values())
        if type(value) in (tuple, list):
            return any(owns(item, seen) for item in value)
        return False

    traceback = retained.value.__traceback__
    visited = []
    while traceback is not None:
        module = traceback.tb_frame.f_globals.get('__name__', '')
        if module.startswith('sdm_client._rust'):
            visited.append(traceback.tb_frame.f_code.co_name)
            assert not owns(traceback.tb_frame.f_locals, set()), visited
        traceback = traceback.tb_next
    assert '_describe' in visited


@pytest.mark.parametrize('key', [['user_id'], ('user_id',)])
def test_wrong_existing_relationship_target_keeps_singleton_tuple_key_error(
    monkeypatch, key
):
    from sdm_client import Relationship

    frames = {
        'users': pd.DataFrame({'user_id': [1, 2], 'alternate_id': [3, 4]}),
        'orders': pd.DataFrame({'order_id': [10, 11], 'user_id': [1, 1]}),
    }
    outcomes = []
    for core in (False, True):
        with pytest.raises(StructuredDataError) as failed:
            describe(
                monkeypatch,
                frames,
                core=core,
                primary_keys={'users': key},
                relationships=[
                    Relationship('orders', 'user_id', 'users', 'alternate_id')
                ],
            )
        outcomes.append((failed.value.code, str(failed.value)))
    assert "table 'users' has ('user_id',)." in outcomes[0][1]
    assert outcomes[1] == outcomes[0]


@pytest.mark.parametrize('rows', ['omitted', None, 4])
def test_actual_describe_result_decode_accepts_sql_and_frame_attachment_shapes(
    rows,
):
    from sdm_client._rust import results

    # The ordinary SQL sample is the existing native messages.py result.
    payload = {'op': 'describe_source', 'description': 3}
    if rows != 'omitted':
        payload['frame_rows'] = rows
    answer = results.decode(payload)
    assert isinstance(answer, results.Described)
    assert answer.description == 3
    assert answer.frame_rows == (4 if rows == 4 else None)
    assert results.expect(payload, results.Described) == answer


@pytest.mark.parametrize('rows', [True, -1, 1.5, '4', [], {}])
def test_actual_describe_result_decode_refuses_malformed_frame_attachment_ids(
    rows,
):
    from sdm_client._rust import results

    with pytest.raises(
        StructuredDataError, match='frame_rows is not a count'
    ) as failed:
        results.expect(
            {'op': 'describe_source', 'description': 3, 'frame_rows': rows},
            results.Described,
        )
    assert failed.value.code == 'INTERNAL_ERROR'
