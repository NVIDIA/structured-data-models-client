# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen public ArrowDtype constructor rules, without materialization.

Reference: schema.py:1919, 2026, 3080; wire/dtypes.py:198;
predictive_query/time_columns.py:340, 465 at 43e55f4f.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from test_t42_frames import PATHS, describe, projection

from sdm_client import Relationship, Schema, StructuredDataError
from sdm_client._routing import RouteError
from sdm_client.core.warnings import SDMWarning

PRIMITIVES = [
    pa.int8(),
    pa.int16(),
    pa.int32(),
    pa.int64(),
    pa.uint8(),
    pa.uint16(),
    pa.uint32(),
    pa.uint64(),
    pa.bool_(),
    pa.string(),
    pa.large_string(),
]


def series(values, dtype):
    # Construct the Arrow array first: pandas' list constructor converts
    # a valid NaN to null, which would erase the distinction under test.
    return pd.Series(
        pd.arrays.ArrowExtensionArray(
            pa.array(values, type=dtype, from_pandas=False)
        )
    )


def paired(monkeypatch, frames, **options):
    outcomes = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                schema = Schema.from_data(frames, **options)
                result = ('ok', projection(schema))
            except StructuredDataError as error:
                schema = None
                result = ('error', type(error), error.code, str(error))
        outcomes.append(
            (result, [(str(n.message), n.category) for n in caught])
        )
    assert outcomes[1] == outcomes[0]
    return schema, [message for message, _ in outcomes[1][1]]


@pytest.mark.parametrize('dtype', PRIMITIVES)
@pytest.mark.parametrize('variant', ['ordinary', 'empty', 'null', 'complete'])
def test_arrow_primitive_metadata_snapshot_and_mutation(
    monkeypatch, dtype, variant
):
    values = (
        [True, False, True]
        if pa.types.is_boolean(dtype)
        else ['a', 'b', 'c']
        if pa.types.is_string(dtype) or pa.types.is_large_string(dtype)
        else [1, 2, 3]
    )
    if variant == 'ordinary':
        values[-1] = None
    elif variant == 'empty':
        values = []
    elif variant == 'null':
        values = [None] * 3
    frame = pd.DataFrame({'value_id': series(values, dtype)})
    schema, _ = paired(monkeypatch, {'rows': frame})
    assert schema['rows'].data is frame
    assert not schema['rows']['value_id'].untyped
    assert schema['rows']['value_id']._source is None
    snapshot = schema.snapshot()
    before = projection(snapshot)
    assert snapshot['rows'].data is frame
    assert projection(pickle.loads(pickle.dumps(snapshot))) == before
    if len(frame):
        frame['value_id'] = series([None] * len(frame), dtype)
        paired(monkeypatch, {'rows': frame})
        assert projection(snapshot) == before


@pytest.mark.parametrize('width', [32, 64])
@pytest.mark.parametrize(
    'variant', ['signed_zero', 'nan_sign', 'nan_payload', 'nan_signaling']
)
def test_arrow_float_native_nomination_boundary_remains_on_default(
    monkeypatch, width, variant
):
    # Frozen public witness: Arrow nunique nominates a two-row key from
    # different bits, then pandas duplicate equality refuses that key.
    # A 100-row case hides this boundary because both counts clear 99%.
    quiet = 0x7FC00000 if width == 32 else 0x7FF8000000000000
    signaling = 0x7F800001 if width == 32 else 0x7FF0000000000001
    sign = 1 << (width - 1)
    bits = {
        'signed_zero': [0, sign],
        'nan_sign': [quiet, quiet | sign],
        'nan_payload': [quiet | 1, quiet | 2],
        'nan_signaling': [signaling, quiet | 1],
    }[variant]
    raw = np.array(bits, dtype=f'uint{width}')
    dtype = pa.float32() if width == 32 else pa.float64()
    array = pa.Array.from_buffers(dtype, 2, [None, pa.py_buffer(raw)])
    frame = pd.DataFrame(
        {'user_id': pd.Series(pd.arrays.ArrowExtensionArray(array))}
    )
    monkeypatch.delenv('SDM_CORE_ROUTES', raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(StructuredDataError) as error:
            Schema.from_data({'users': frame})
    example = '-0.0' if variant == 'signed_zero' else 'nan'
    assert error.value.code == 'INVALID_REQUEST'
    assert str(error.value) == (
        '[INVALID_REQUEST] users.user_id is not unique enough to identify '
        'a row: 1 of 2 rows repeat a value already used, e.g. ' + example
    )
    assert not caught
    monkeypatch.setenv('SDM_CORE_ROUTES', PATHS)
    with pytest.raises(RouteError, match='partial frame constructor'):
        Schema.from_data({'users': frame})


@pytest.mark.parametrize('dtype', [pa.float32(), pa.float64()])
@pytest.mark.parametrize(
    'values', [[], [None, None], [1, 2], [np.nan, np.nan], [0.0, -0.0]]
)
@pytest.mark.parametrize('name', ['user_id', 'value'])
def test_arrow_float_whole_family_remains_unfinished(
    monkeypatch, dtype, values, name
):
    frame = pd.DataFrame({name: series(values, dtype)})
    monkeypatch.setenv('SDM_CORE_ROUTES', PATHS)
    with pytest.raises(RouteError, match='partial frame constructor'):
        Schema.from_data({'rows': frame})


@pytest.mark.parametrize('left', [pa.int32(), pa.uint64()])
@pytest.mark.parametrize('right', [pa.int64(), pa.uint32()])
@pytest.mark.parametrize(
    'key', ['user_id', ['user_id'], ('user_id',), ('user_id', 'group')]
)
def test_arrow_repeated_key_forms_and_native_composite_row(
    monkeypatch, left, right, key
):
    values = list(range(99)) + [98]
    frame = pd.DataFrame(
        {'user_id': series(values, left), 'group': series(values, right)}
    )
    _, notices = paired(
        monkeypatch, {'users': frame}, primary_keys={'users': key}
    )
    assert notices


@pytest.mark.parametrize('wide', [False, True])
def test_arrow_uint64_bounds_and_inferred_explicit_links(monkeypatch, wide):
    first = 2**63 if wide else 2**63 - 2
    frames = {
        'users': pd.DataFrame(
            {'user_id': series([first, first + 1], pa.uint64())}
        ),
        'orders': pd.DataFrame(
            {
                'order_id': series([10, 11, 12], pa.int8()),
                'user_id': series([first, None, first + 1], pa.uint64()),
            }
        ),
    }
    for options in (
        {},
        {
            'relationships': [
                Relationship('orders', 'user_id', 'users', 'user_id')
            ]
        },
        {'relationships': []},
    ):
        schema, _ = paired(monkeypatch, frames, **options)
        assert schema['users']['user_id'].dtype == (
            'string' if wide else 'int64'
        )


@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
@pytest.mark.parametrize(
    'zone', [None, 'UTC', 'America/Los_Angeles', 'Asia/Kathmandu']
)
@pytest.mark.parametrize('values', [[-1, 0, 1, None], [1, 1], [], [None, None]])
def test_arrow_timestamp_units_counts_zones_and_native_constant_notice(
    monkeypatch, unit, zone, values
):
    dtype = pa.timestamp(unit, zone)
    frame = pd.DataFrame({'observed_at': series(values, dtype)})
    original = frame['observed_at'].array.__arrow_array__()
    frames = {'events': frame, 'users': pd.DataFrame({'user_id': [1, 2]})}
    schema, _ = paired(monkeypatch, frames)
    assert schema['events']['observed_at'].dtype == 'timestamp[us]'
    assert schema['events'].primary_key is None
    assert schema['events'].data is frame
    assert frame['observed_at'].array.__arrow_array__().equals(original)
    snapshot = schema.snapshot()
    before = projection(snapshot)
    assert projection(pickle.loads(pickle.dumps(snapshot))) == before
    frame['observed_at'] = series([0] * len(frame), dtype)
    paired(monkeypatch, frames, time_columns={'events': 'observed_at'})
    assert projection(snapshot) == before


@pytest.mark.parametrize('dtype', [pa.string(), pa.large_string()])
@pytest.mark.parametrize(
    'values',
    [
        ['', 'a\x00b', 'é', '😺'],
        ['2024-01-01', '2024-01-02', None],
        ['2024-02-31', '2024-01-02'],
        [str(n) for n in range(65)],
    ],
)
def test_arrow_text_meaning_and_final_native_date_advice(
    monkeypatch, dtype, values
):
    frame = pd.DataFrame({'value': series(values, dtype)})
    paired(monkeypatch, {'rows': frame})
    schema, notices = paired(
        monkeypatch, {'rows': frame}, time_columns={'rows': 'value'}
    )
    assert schema['rows'].time_column == 'value'
    assert not notices


def test_arrow_table_warning_precedes_later_failure(monkeypatch):
    first = pd.DataFrame({'user_id': series([*range(99), None], pa.int64())})
    later = pd.DataFrame([[1, 2]], columns=['same', 'same'])
    schema, notices = paired(
        monkeypatch,
        {'users': first, 'later': later},
        primary_keys={'users': 'missing'},
    )
    assert schema is None
    assert len(notices) == 1
    assert 'have no value' in notices[0]


@pytest.mark.parametrize(
    'dtype',
    [
        pa.date32(),
        pa.date64(),
        pa.decimal128(10, 0),
        pa.duration('ns'),
        pa.time64('ns'),
        pa.binary(),
        pa.list_(pa.int64()),
        pa.dictionary(pa.int8(), pa.string()),
    ],
)
def test_other_arrow_families_remain_explicitly_unfinished(monkeypatch, dtype):
    monkeypatch.setenv('SDM_CORE_ROUTES', PATHS)
    frame = pd.DataFrame({'value': series([], dtype)})
    with pytest.raises(RouteError, match='partial frame constructor'):
        Schema.from_data({'rows': frame})


def test_arrow_custom_extension_array_export_is_not_invoked(monkeypatch):
    touched = []

    class Custom(pd.arrays.ArrowExtensionArray):
        def __arrow_array__(self, type=None):
            touched.append('export')
            raise AssertionError('custom Arrow conversion was invoked')

    frame = pd.DataFrame({'value': pd.Series(Custom(pa.array([1, 2])))})
    monkeypatch.setenv('SDM_CORE_ROUTES', PATHS)
    with pytest.raises(RouteError, match='partial frame constructor'):
        Schema.from_data({'rows': frame})
    assert not touched


def test_arrow_date_hint_warning_as_error_keeps_final_stage(monkeypatch):
    frame = pd.DataFrame(
        {'date': series(['2024-01-01', '2024-01-02'], pa.large_string())}
    )
    outcomes = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        with warnings.catch_warnings():
            warnings.simplefilter('error', SDMWarning)
            with pytest.raises(SDMWarning) as caught:
                Schema.from_data({'rows': frame})
        outcomes.append(str(caught.value))
    assert outcomes[1] == outcomes[0]


def test_arrow_schema_domain_rules_never_reenter_python_inference(monkeypatch):
    import sdm_client.schema as schema_module

    frames = {
        'users': pd.DataFrame({'user_id': series([1, 2], pa.int64())}),
        'events': pd.DataFrame(
            {
                'event_id': series([10, 11], pa.uint8()),
                'user_id': series([1, 2], pa.int64()),
                'at': series([1, 2], pa.timestamp('ns', 'UTC')),
            }
        ),
    }
    old, old_notices = describe(monkeypatch, frames, core=False)

    def forbidden(*args, **kwargs):
        raise AssertionError('Arrow conversion reentered Python inference')

    for name in (
        '_infer_table_and_time_column',
        '_infer_primary_key',
        '_infer_relationships',
    ):
        monkeypatch.setattr(schema_module, name, forbidden)
    new, new_notices = describe(monkeypatch, frames, core=True)
    assert projection(new) == projection(old)
    assert new_notices == old_notices
