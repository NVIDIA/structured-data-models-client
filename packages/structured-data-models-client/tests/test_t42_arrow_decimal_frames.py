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
