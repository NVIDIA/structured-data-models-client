# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Arrow float constructors consume separate reached native populations."""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from test_t42_frames import PATHS, projection

from sdm_client import Column, Schema, StructuredDataError, Table
from sdm_client._rust.frame_facts import native_declaration
from sdm_client.core.warnings import SDMWarning


def bit_cases(width):
    sign = 1 << (width - 1)
    quiet = 0x7FC00000 if width == 32 else 0x7FF8000000000000
    signaling = 0x7F800001 if width == 32 else 0x7FF0000000000001
    one = 0x3F800000 if width == 32 else 0x3FF0000000000000
    two = 0x40000000 if width == 32 else 0x4000000000000000
    infinity = 0x7F800000 if width == 32 else 0x7FF0000000000000
    return {
        'signed_zero': [0, sign],
        'nan_sign': [quiet, quiet | sign],
        'nan_payload': [quiet | 1, quiet | 2],
        'nan_signaling': [signaling, quiet | 1],
        'same_zero': [0, 0],
        'same_nan': [quiet, quiet],
        'ordinary': [one, two],
        'infinities': [infinity, infinity | sign],
        'subnormals': [1, sign | 1],
        'extrema': [infinity - 1, (infinity - 1) | sign],
    }


def raw_series(width, bits, validity=None, offset=0, chunks=False):
    raw = np.asarray(bits, dtype=f'uint{width}')
    array = pa.Array.from_buffers(
        pa.float32() if width == 32 else pa.float64(),
        len(raw),
        [
            None if validity is None else pa.py_buffer(validity),
            pa.py_buffer(raw),
        ],
    ).slice(offset)
    if chunks:
        array = pa.chunked_array(
            [
                array.slice(0, 0),
                array.slice(0, 1),
                array.slice(1),
                array.slice(0, 0),
            ]
        )
    return pd.Series(pd.arrays.ArrowExtensionArray(array))


def observe(make, core, monkeypatch, *, options=None, table=False):
    monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
    frame = make()
    before = (
        frame['user_id'].array.__arrow_array__() if 'user_id' in frame else None
    )
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            if table:
                made = Table(
                    'users',
                    frame,
                    {
                        'user_id': Column(
                            'user_id',
                            'float32'
                            if frame['user_id'].dtype.pyarrow_dtype.bit_width
                            == 32
                            else 'float64',
                            'numerical',
                        )
                    },
                    primary_key='user_id',
                )
                outcome = (
                    'table',
                    made.primary_key,
                    made.duplicate_keys,
                    made['user_id'].dtype,
                    str(made['user_id'].stype),
                )
            else:
                made = Schema.from_data({'users': frame}, **(options or {}))
                outcome = ('schema', projection(made))
                assert made['users'].data is frame
                assert all(
                    column._source is None
                    for column in made['users'].columns.values()
                )
                snapshot = made.snapshot()
                assert snapshot['users'].data is frame
                assert projection(
                    pickle.loads(pickle.dumps(snapshot))
                ) == projection(made)
        except StructuredDataError as error:
            outcome = (
                'error',
                type(error).__module__,
                type(error).__name__,
                error.code,
                str(error),
            )
    if before is not None:
        after = frame['user_id'].array.__arrow_array__()
        for first, second in zip(before.chunks, after.chunks):
            assert first.type == second.type
            assert first.offset == second.offset
            assert [
                None if value is None else value.to_pybytes()
                for value in first.buffers()
            ] == [
                None if value is None else value.to_pybytes()
                for value in second.buffers()
            ]
    return outcome, [
        (
            notice.category.__module__,
            notice.category.__name__,
            str(notice.message),
        )
        for notice in notices
    ]


@pytest.mark.parametrize('width', [32, 64])
@pytest.mark.parametrize('case', list(bit_cases(32)))
@pytest.mark.parametrize('form', ['inferred', 'scalar', 'table'])
def test_frozen_bit_boundary_forms_match_original(
    monkeypatch, width, case, form
):
    def make():
        return pd.DataFrame(
            {'user_id': raw_series(width, bit_cases(width)[case])}
        )

    options = {'primary_keys': {'users': 'user_id'}} if form == 'scalar' else {}
    old = observe(
        make, False, monkeypatch, options=options, table=form == 'table'
    )
    new = observe(
        make, True, monkeypatch, options=options, table=form == 'table'
    )
    assert new == old


@pytest.mark.parametrize('width', [32, 64])
@pytest.mark.parametrize(
    'form', [None, 'user_id', ['user_id'], ('user_id',), ('user_id', 'group')]
)
@pytest.mark.parametrize('rows', [2, 99, 100, 101])
@pytest.mark.parametrize('missing', [False, True])
def test_float_key_forms_thresholds_and_missing_precedence(
    monkeypatch, width, form, rows, missing
):
    def make():
        values = np.arange(rows, dtype=f'float{width}')
        values[-1] = values[-2]
        array = pa.array(
            values, mask=[missing] + [False] * (rows - 1), from_pandas=False
        )
        return pd.DataFrame(
            {
                'user_id': pd.Series(pd.arrays.ArrowExtensionArray(array)),
                'group': np.arange(rows),
            }
        )

    options = {} if form is None else {'primary_keys': {'users': form}}
    assert observe(make, True, monkeypatch, options=options) == observe(
        make, False, monkeypatch, options=options
    )


@pytest.mark.parametrize('width', [32, 64])
@pytest.mark.parametrize(
    'case',
    ['signed_zero', 'same_nan', 'nan_payload', 'nan_signaling', 'ordinary'],
)
@pytest.mark.parametrize(
    'key',
    [
        'measurement',
        ['measurement'],
        ('measurement',),
        ('measurement', 'group'),
    ],
)
def test_explicit_float_key_population_is_reached_after_keyless_inference(
    monkeypatch, width, case, key
):
    def make():
        return pd.DataFrame(
            {
                'measurement': raw_series(width, bit_cases(width)[case]),
                'group': [7, 7],
            }
        )

    options = {'primary_keys': {'users': key}}
    assert observe(make, True, monkeypatch, options=options) == observe(
        make, False, monkeypatch, options=options
    )


@pytest.mark.parametrize('width', [32, 64])
@pytest.mark.parametrize('offset', [0, 1, 3, 7])
@pytest.mark.parametrize('chunks', [False, True])
@pytest.mark.parametrize('masked', [False, True])
def test_float_raw_storage_slices_and_empty_chunks_keep_meaning(
    monkeypatch, width, offset, chunks, masked
):
    def make():
        bits = (
            bit_cases(width)['ordinary']
            + bit_cases(width)['nan_signaling']
            + bit_cases(width)['signed_zero']
            + bit_cases(width)['extrema']
            + [0]
        )
        values = raw_series(
            width,
            bits,
            bytearray([0b10101101, 1]) if masked else None,
            offset,
            chunks,
        )
        return pd.DataFrame({'value': values, 'row_id': np.arange(len(values))})

    assert observe(make, True, monkeypatch) == observe(make, False, monkeypatch)


@pytest.mark.parametrize('width', [32, 64])
@pytest.mark.parametrize('rows', [99, 100, 101])
def test_float_native_effect_order_first_candidate_stop_and_later_mutation(
    monkeypatch, width, rows
):
    outcomes = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        frame = pd.DataFrame(
            {
                'first_id': pd.Series(
                    pd.arrays.ArrowExtensionArray(
                        pa.array(
                            [*range(rows - 1), None],
                            type=pa.float32() if width == 32 else pa.float64(),
                        )
                    )
                ),
                'later_key': pd.Series(
                    pd.arrays.ArrowExtensionArray(
                        pa.array(np.arange(rows, dtype=f'float{width}'))
                    )
                ),
            }
        )
        trace = []
        with monkeypatch.context() as hooks:
            for name in ('nunique', 'duplicated', 'isna', 'notna'):
                original = getattr(pd.Series, name)

                def record(
                    series, *args, _name=name, _original=original, **kwargs
                ):
                    trace.append((_name, series.name, len(series)))
                    return _original(series, *args, **kwargs)

                hooks.setattr(pd.Series, name, record)
            with warnings.catch_warnings():
                warnings.simplefilter('always')

                def show(
                    message, category, filename, lineno, file=None, line=None
                ):
                    trace.append(('warning', category.__name__, str(message)))
                    if 'have no value' in str(message):
                        frame['later_key'] = pd.Series(
                            pd.arrays.ArrowExtensionArray(
                                pa.array(np.zeros(rows, dtype=f'float{width}'))
                            )
                        )

                hooks.setattr(warnings, 'showwarning', show)
                schema = Schema.from_data({'users': frame})
        outcomes.append((projection(schema), trace))
    assert outcomes[1] == outcomes[0]


@pytest.mark.parametrize('width', [32, 64])
def test_float_warning_error_stops_before_later_table_export(
    monkeypatch, width
):
    outcomes = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        first = pd.DataFrame(
            {
                'user_id': pd.Series(
                    pd.arrays.ArrowExtensionArray(
                        pa.array(
                            [*range(99), None],
                            type=pa.float32() if width == 32 else pa.float64(),
                        )
                    )
                )
            }
        )
        later = pd.DataFrame(
            {
                'value': pd.Series(
                    pd.arrays.ArrowExtensionArray(
                        pa.array(
                            [1.0, 2.0],
                            type=pa.float32() if width == 32 else pa.float64(),
                        )
                    )
                )
            }
        )
        touched = []
        original = pd.arrays.ArrowExtensionArray.__arrow_array__

        def export(array, *args, **kwargs):
            if array is later['value'].array:
                touched.append('later')
            return original(array, *args, **kwargs)

        with monkeypatch.context() as hooks:
            hooks.setattr(
                pd.arrays.ArrowExtensionArray, '__arrow_array__', export
            )
            with warnings.catch_warnings():
                warnings.simplefilter('error', SDMWarning)
                with pytest.raises(SDMWarning) as error:
                    Schema.from_data({'users': first, 'later': later})
        outcomes.append((str(error.value), touched))
    assert outcomes[1] == outcomes[0]
    assert outcomes[1][1] == []


@pytest.mark.parametrize('width', [32, 64])
def test_float_warning_mutation_is_seen_when_later_table_is_reached(
    monkeypatch, width
):
    results = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        first = pd.DataFrame(
            {
                'user_id': pd.Series(
                    pd.arrays.ArrowExtensionArray(
                        pa.array(
                            [*range(99), None],
                            type=pa.float32() if width == 32 else pa.float64(),
                        )
                    )
                )
            }
        )
        later = pd.DataFrame(
            {'item_id': raw_series(width, bit_cases(width)['ordinary'])}
        )
        notices = []
        with monkeypatch.context() as hooks, warnings.catch_warnings():
            warnings.simplefilter('always')

            def show(message, category, filename, lineno, file=None, line=None):
                notices.append((category.__name__, str(message)))
                if 'have no value' in str(message):
                    later['item_id'] = raw_series(
                        width, bit_cases(width)['signed_zero']
                    )

            hooks.setattr(warnings, 'showwarning', show)
            with pytest.raises(StructuredDataError) as error:
                Schema.from_data({'users': first, 'items': later})
        results.append(
            (
                type(error.value).__module__,
                error.value.code,
                str(error.value),
                notices,
            )
        )
    assert results[1] == results[0]
    assert 'items.item_id' in results[1][2]


@pytest.mark.parametrize('width', [32, 64])
def test_float_final_observation_is_fresh_after_native_warning_mutation(
    monkeypatch, width
):
    from sdm_client._rust import frames

    monkeypatch.setenv('SDM_CORE_ROUTES', PATHS)
    dtype = pa.float32() if width == 32 else pa.float64()
    data = pd.DataFrame(
        {
            'user_id': pd.Series(
                pd.arrays.ArrowExtensionArray(
                    pa.array([*range(99), None], type=dtype)
                )
            ),
            'measurement': pd.Series(
                pd.arrays.ArrowExtensionArray(pa.array([1.0] * 100, type=dtype))
            ),
        }
    )
    reached = []
    original = frames._describe

    def describe(stage, tables, *args, **kwargs):
        if stage == 'assemble_native':
            values = tables['users'].column('measurement')
            reached.append((values.type, values.to_pylist()))
        return original(stage, tables, *args, **kwargs)

    def show(message, category, filename, lineno, file=None, line=None):
        if 'have no value' in str(message):
            data['measurement'] = pd.Series(
                pd.arrays.ArrowExtensionArray(pa.array([2.0] * 100, type=dtype))
            )

    monkeypatch.setattr(frames, '_describe', describe)
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        monkeypatch.setattr(warnings, 'showwarning', show)
        schema = Schema.from_data({'users': data})
    assert reached == [(dtype, [2.0] * 100)]
    assert schema['users']['measurement'].dtype == f'float{width}'
    assert schema['users'].data is data


@pytest.mark.parametrize('width', [32, 64])
@pytest.mark.parametrize(
    'case', ['signed_zero', 'nan_payload', 'nan_signaling']
)
def test_arrow_and_masked_float_identical_buffers_keep_native_cardinality(
    monkeypatch, width, case
):
    outcomes = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        row = []
        for arrow in (False, True):
            bits = np.array(bit_cases(width)[case], dtype=f'uint{width}')
            if arrow:
                values = raw_series(width, bits)
            else:
                values = pd.Series(
                    pd.arrays.FloatingArray(
                        bits.view(f'float{width}'), np.zeros(2, dtype=bool)
                    )
                )
            try:
                schema = Schema.from_data(
                    {'users': pd.DataFrame({'user_id': values})}
                )
                row.append(('ok', projection(schema)))
            except StructuredDataError as error:
                row.append(('error', error.code, str(error)))
        outcomes.append(row)
    assert outcomes[1] == outcomes[0]
    assert outcomes[1][0][0] == 'ok'
    assert outcomes[1][1][0] == 'error'


@pytest.mark.parametrize('width', [32, 64])
def test_float_declaration_and_dtype_owners_remain_exact(monkeypatch, width):
    dtype = pd.ArrowDtype(pa.float32() if width == 32 else pa.float64())
    assert native_declaration(dtype) == {
        'declaration': 'physical',
        'physical': {'type': f'float{width}'},
    }
    monkeypatch.setenv('SDM_CORE_ROUTES', PATHS)
    frame = pd.DataFrame(
        {'value_id': raw_series(width, bit_cases(width)['ordinary'])}
    )
    owner = frame['value_id'].dtype
    schema = Schema.from_data({'users': frame})
    snapshot = schema.snapshot()
    initial = projection(snapshot)
    schema['users']['value_id'].stype = 'numerical'
    schema['users']['value_id'].stype = 'ID'
    assert frame['value_id'].dtype is owner
    frame['value_id'] = raw_series(width, bit_cases(width)['signed_zero'])
    assert projection(snapshot) == initial
    with pytest.raises(StructuredDataError, match='not unique enough'):
        Schema.from_data({'users': frame})


@pytest.mark.parametrize('width', [32, 64])
def test_many_float_chunks_and_nonkey_source_need_no_selected_value_gate(
    monkeypatch, width
):
    dtype = pa.float32() if width == 32 else pa.float64()

    def make():
        chunks = [
            pa.array(
                np.arange(start, start + 257, dtype=f'float{width}'), type=dtype
            )
            for start in range(0, 257 * 64, 257)
        ]
        values = pd.Series(
            pd.arrays.ArrowExtensionArray(pa.chunked_array(chunks))
        )
        return pd.DataFrame(
            {'row_id': np.arange(len(values)), 'measurement': values}
        )

    assert observe(make, True, monkeypatch) == observe(make, False, monkeypatch)


@pytest.mark.parametrize('core', [False, True], ids=['original', 'core'])
@pytest.mark.parametrize(
    'case',
    [
        'arrow-float64-python-True-False',
        'arrow-float64-pyarrow-True-False',
        'arrow-float64-python-True-True',
    ],
)
def test_float_numeric_text_reaches_frozen_graph_error_before_normalization(
    monkeypatch, core, case
):
    from sdm_relational_engine.rfm.base import sampler
    from t45_frame_routes import FRAME_ROUTES
    from test_t44_native_text_keys import (
        test_physical_numeric_text_keeps_actual_schema_and_native_graph_errors,
    )

    from sdm_client._rust import frame_facade
    from sdm_client.core.transport import Transport
    from sdm_client.materializers.relational import RelationalMaterializer

    reached = []

    def forbidden(*args, **kwargs):
        reached.append('unexpected normalization or transport')
        raise AssertionError(reached[-1])

    monkeypatch.setenv('SDM_CORE_ROUTES', FRAME_ROUTES if core else '')
    monkeypatch.setattr(sampler, '_normalize_text', forbidden)
    monkeypatch.setattr(frame_facade, 'convert_text', forbidden)
    monkeypatch.setattr(Transport, 'encode_prediction', forbidden)
    if core:
        monkeypatch.setattr(RelationalMaterializer, '_get_model', forbidden)
    test_physical_numeric_text_keeps_actual_schema_and_native_graph_errors(
        core, case
    )
    assert reached == []
