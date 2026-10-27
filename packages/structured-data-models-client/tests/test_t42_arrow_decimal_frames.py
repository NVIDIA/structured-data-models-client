# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Decimal constructor journeys, before serialized composition.

Frozen source: schema.py:1919/2022 and wire/dtypes.py:198/307 at 43e55f4f.
The paired route exercises the public constructor; it does not call the new
decimal helper or synthesize a reference schema.
"""

from __future__ import annotations

import pickle
import warnings
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from test_t42_frames import PATHS, projection

from sdm_client import Relationship, Schema, StructuredDataError


def series(values, dtype):
    return pd.Series(
        pd.arrays.ArrowExtensionArray(pa.array(values, type=dtype))
    )


def paired(monkeypatch, build, **options):
    outcomes = []
    for core in (False, True):
        frames = build()
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        schema = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                schema = Schema.from_data(frames, **options)
            except StructuredDataError as error:
                outcome = ('error', type(error), error.code, str(error))
            else:
                outcome = ('ok', projection(schema))
        outcomes.append(
            (outcome, [(str(item.message), item.category) for item in caught])
        )
    assert outcomes[1] == outcomes[0]
    return schema, frames, outcomes[1]


@pytest.mark.parametrize('make', [pa.decimal128, pa.decimal256])
@pytest.mark.parametrize(
    'precision,scale,values',
    [
        (18, 0, [Decimal(1), Decimal(2), None]),
        (18, 2, [Decimal('1.25'), Decimal('2.50'), None]),
        (18, 2, [Decimal('1.00'), Decimal('2.00')]),
        (38, 0, [Decimal(2**63), Decimal(2**63 + 1)]),
        (
            38,
            18,
            [Decimal('1.250000000000000001'), Decimal('1.250000000000000002')],
        ),
        (18, -2, [Decimal(100), Decimal(200), None]),
        (
            18,
            -2,
            [Decimal('99999999999999999900'), Decimal('99999999999999999800')],
        ),
        (18, 2, [None, None]),
        (18, 2, []),
    ],
    ids=[
        'whole',
        'fraction',
        'scaled-whole',
        'past-int64',
        'past-float64',
        'negative-scale',
        'declared-precision-fast-path',
        'all-null',
        'empty',
    ],
)
@pytest.mark.parametrize('name', ['value', 'value_id'])
def test_public_decimal_metadata_and_owned_snapshots(
    monkeypatch, make, precision, scale, values, name
):
    dtype = make(precision, scale)

    def build():
        return {'rows': pd.DataFrame({name: series(values, dtype)})}

    schema, frames, _ = paired(monkeypatch, build)
    assert schema is not None
    assert schema['rows'].data is frames['rows']
    assert not schema['rows'][name].untyped
    snapshot = schema.snapshot()
    before = projection(snapshot)
    assert snapshot['rows'].data is frames['rows']
    assert projection(pickle.loads(pickle.dumps(snapshot))) == before
    if len(frames['rows']):
        frames['rows'][name] = series([None] * len(frames['rows']), dtype)
        assert projection(snapshot) == before


@pytest.mark.parametrize('make', [pa.decimal128, pa.decimal256])
@pytest.mark.parametrize('scale', [0, 18])
@pytest.mark.parametrize('key', ['value_id', ['value_id'], ('value_id', 'group')])
def test_decimal_duplicate_examples_preserve_native_values(
    monkeypatch, make, scale, key
):
    base = Decimal('1.250000000000000001') if scale else Decimal(2**53 + 1)
    other = Decimal('1.250000000000000002') if scale else Decimal(2**53 + 2)
    values = [base] + [other + Decimal(i) for i in range(98)] + [base]

    def build():
        return {
            'rows': pd.DataFrame(
                {
                    'value_id': series(values, make(38, scale)),
                    'group': np.zeros(100, dtype='int64'),
                    'measurement': np.arange(100, dtype='float64'),
                }
            )
        }

    schema, _, (_, notices) = paired(
        monkeypatch, build, primary_keys={'rows': key}
    )
    assert schema is not None
    assert schema['rows'].duplicate_keys == 1
    assert notices


@pytest.mark.parametrize('make', [pa.decimal128, pa.decimal256])
def test_adjacent_fractional_decimals_remain_distinct_keys(monkeypatch, make):
    def build():
        return {
            'rows': pd.DataFrame(
                {
                    'value_id': series(
                        [
                            Decimal('1.250000000000000001'),
                            Decimal('1.250000000000000002'),
                        ],
                        make(38, 18),
                    )
                }
            )
        }

    schema, _, _ = paired(monkeypatch, build)
    assert schema['rows'].primary_key == 'value_id'
    assert schema['rows'].duplicate_keys == 0


def test_decimal256_keys_keep_digits_past_i128(monkeypatch):
    values = [Decimal('1E+60'), Decimal('2E+60')]

    def build():
        return {
            'rows': pd.DataFrame(
                {'value_id': series(values, pa.decimal256(76, 0))}
            )
        }

    schema, _, _ = paired(monkeypatch, build)
    assert schema['rows']['value_id'].dtype == 'string'
    assert schema['rows'].primary_key == 'value_id'


@pytest.mark.parametrize('make', [pa.decimal128, pa.decimal256])
@pytest.mark.parametrize('explicit', [False, True])
def test_decimal_key_relationships_are_assembled_in_core(
    monkeypatch, make, explicit
):
    dtype = make(38, 0)

    def build():
        return {
            'users': pd.DataFrame(
                {'user_id': series([Decimal(2**63), Decimal(2**63 + 1)], dtype)}
            ),
            'orders': pd.DataFrame(
                {
                    'order_id': [10, 11, 12],
                    'user_id': series(
                        [Decimal(2**63), None, Decimal(2**63 + 1)], dtype
                    ),
                }
            ),
        }

    options = (
        {'relationships': [Relationship('orders', 'user_id', 'users', 'user_id')]}
        if explicit
        else {}
    )
    schema, _, _ = paired(monkeypatch, build, **options)
    assert len(schema.relationships) == 1


def raw_series(bits, precision, scale, values):
    make = pa.decimal128 if bits == 128 else pa.decimal256
    raw = b''.join(
        value.to_bytes(bits // 8, byteorder='little', signed=True)
        for value in values
    )
    array = pa.Array.from_buffers(
        make(precision, scale), len(values), [None, pa.py_buffer(raw)]
    )
    return pd.Series(pd.arrays.ArrowExtensionArray(array))


@pytest.mark.parametrize('bits', [128, 256])
@pytest.mark.parametrize('scale', [-129, -128, -1, 0, 1, 2, 129])
@pytest.mark.parametrize('key', [False, True])
def test_full_native_scale_descriptor_is_not_narrowed(monkeypatch, bits, scale, key):
    # Both frozen runtime cells accept these descriptors, including native
    # scale outside i8 and scale greater than precision.
    name = 'value_id' if key else 'value'

    def build():
        return {'rows': pd.DataFrame({name: raw_series(bits, 1, scale, [0, 1])})}

    schema, _, _ = paired(monkeypatch, build)
    assert schema['rows'][name].dtype == ('float64' if scale > 0 else 'int64')
    assert schema['rows'].primary_key == ('value_id' if key else None)


@pytest.mark.parametrize('bits', [128, 256])
@pytest.mark.parametrize(
    'scale,values,expected',
    [
        (0, [-(2**63), 2**63 - 1], 'int64'),
        (0, [-(2**63) - 1, 2**63], 'string'),
        (-1, [-922337203685477580, 922337203685477580], 'int64'),
        (-1, [-922337203685477581, 922337203685477581], 'string'),
    ],
)
def test_native_decimal_range_uses_logical_scale_after_precision_fast_path(
    monkeypatch, bits, scale, values, expected
):
    def build():
        return {
            'rows': pd.DataFrame(
                {'value_id': raw_series(bits, 19, scale, values)}
            )
        }

    schema, _, _ = paired(monkeypatch, build)
    assert schema['rows']['value_id'].dtype == expected
    assert schema['rows'].primary_key == 'value_id'


@pytest.mark.parametrize('bits', [128, 256])
@pytest.mark.parametrize('scale', [-129, 2, 129])
def test_repeated_key_error_uses_original_native_decimal_scalar(
    monkeypatch, bits, scale
):
    def build():
        return {
            'rows': pd.DataFrame(
                {'value_id': raw_series(bits, 1, scale, [1, 1, 0])}
            )
        }

    schema, _, (outcome, _) = paired(
        monkeypatch, build, primary_keys={'rows': 'value_id'}
    )
    assert schema is None
    assert outcome[0] == 'error'
    assert outcome[2] == 'INVALID_REQUEST'
    assert 'Decimal(' in outcome[3]


@pytest.mark.parametrize('bits', [128, 256])
def test_missing_key_advice_can_change_later_decimal_scale_and_example(monkeypatch, bits):
    outcomes = []
    make = pa.decimal128 if bits == 128 else pa.decimal256
    for core in (False, True):
        frame = pd.DataFrame({
            'value_id': series([None] + [Decimal(i) for i in range(1, 100)], make(3, 0)),
            'measurement': np.arange(100, dtype='float64'),
        })
        changed = False
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            show = warnings.showwarning

            def mutate(message, *args, **kwargs):
                nonlocal changed
                if not changed:
                    assert 'have no value for it' in str(message)
                    changed = True
                    frame['value_id'] = raw_series(bits, 3, 129, [1, 1] + list(range(2, 100)))
                show(message, *args, **kwargs)

            warnings.showwarning = mutate
            schema = Schema.from_data({'rows': frame})
        assert changed
        assert schema['rows']['value_id'].dtype == 'int64'
        assert schema['rows'].duplicate_keys == 1
        messages = [str(item.message) for item in caught]
        assert any("Decimal('1E-129')" in message for message in messages)
        outcomes.append((projection(schema), messages))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize('bits', [128, 256])
@pytest.mark.parametrize('scale', [-129, 129])
def test_decimal_public_inspection_edits_and_snapshot_ownership(monkeypatch, bits, scale):
    outcomes = []
    for core in (False, True):
        frame = pd.DataFrame({'value_id': raw_series(bits, 1, scale, [0, 1])})
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        schema = Schema.from_data({'rows': frame})
        snapshot = schema.snapshot()
        before = projection(snapshot)
        schema['rows'].primary_key = None
        schema['rows'].primary_key = ('value_id',)
        assert snapshot['rows'].data is frame
        assert projection(snapshot) == before
        assert projection(pickle.loads(pickle.dumps(snapshot))) == before
        assert schema['rows'].primary_key == ('value_id',)
        outcomes.append((projection(schema), before))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize('bits', [128, 256])
@pytest.mark.parametrize('scale', [-129, 129])
def test_decimal_constructor_rereads_mutated_native_buffers(monkeypatch, bits, scale):
    outcomes = []
    make = pa.decimal128 if bits == 128 else pa.decimal256
    for core in (False, True):
        width = bits // 8
        backing = bytearray(width) + bytearray((1).to_bytes(width, 'little', signed=True))
        array = pa.Array.from_buffers(make(1, scale), 2, [None, pa.py_buffer(backing)])
        frame = pd.DataFrame({'value_id': pd.Series(pd.arrays.ArrowExtensionArray(array))})
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        before = Schema.from_data({'rows': frame})
        snapshot = before.snapshot()
        backing[width:2 * width] = bytes(width)
        after = Schema.from_data({'rows': frame})
        assert snapshot['rows'].primary_key == 'value_id'
        assert after['rows'].primary_key is None
        assert snapshot['rows'].data is frame
        outcomes.append((projection(snapshot), projection(after)))
    assert outcomes[0] == outcomes[1]
