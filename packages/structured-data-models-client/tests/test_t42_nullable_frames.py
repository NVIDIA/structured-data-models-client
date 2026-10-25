# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public constructor counterparts for the closed pandas masked family."""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from test_t42_frames import PATHS, describe, projection

from sdm_client import Relationship, Schema, StructuredDataError
from sdm_client.core.warnings import SDMWarning

DTYPES = (
    'Int8',
    'Int16',
    'Int32',
    'Int64',
    'UInt8',
    'UInt16',
    'UInt32',
    'UInt64',
    'Float32',
    'Float64',
    'boolean',
)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize(
    'variant', ['ordinary', 'empty', 'all_null', 'complete']
)
def test_masked_family_constructor_metadata_matches_native(
    monkeypatch, dtype, variant
):
    values = [True, False, None] if dtype == 'boolean' else [1, 2, None]
    if variant == 'empty':
        values = []
    elif variant == 'all_null':
        values = [None] * 3
    elif variant == 'complete':
        values = [True, False, True] if dtype == 'boolean' else [1, 2, 3]
    frame = pd.DataFrame({'value_id': pd.Series(values, dtype=dtype)})
    old, old_notices = describe(monkeypatch, {'rows': frame}, core=False)
    new, new_notices = describe(monkeypatch, {'rows': frame}, core=True)
    assert projection(new) == projection(old)
    assert new_notices == old_notices
    assert new['rows'].data is frame
    assert not new['rows']['value_id'].untyped
    assert new['rows']['value_id']._source is None
    snapshot = new.snapshot()
    before = projection(snapshot)
    original_stype = str(new['rows']['value_id'].stype)
    new['rows']['value_id'].stype = 'categorical'
    new['rows']['value_id'].stype = original_stype
    assert projection(new) == before
    assert projection(pickle.loads(pickle.dumps(snapshot))) == before
    assert snapshot['rows'].data is frame
    if len(frame):
        frame.loc[:, 'value_id'] = pd.Series(
            [True if dtype == 'boolean' else 1] * len(frame), dtype=dtype
        )
        fresh_old, fresh_old_notices = describe(
            monkeypatch, {'rows': frame}, core=False
        )
        fresh_new, fresh_new_notices = describe(
            monkeypatch, {'rows': frame}, core=True
        )
        assert projection(fresh_new) == projection(fresh_old)
        assert fresh_new_notices == fresh_old_notices
        assert projection(snapshot) == before


@pytest.mark.parametrize('dtype', DTYPES)
def test_masked_explicit_null_key_and_competing_override_error(
    monkeypatch, dtype
):
    values = [True, False, None] if dtype == 'boolean' else [1, 2, None]
    frame = pd.DataFrame({'value_id': pd.Series(values, dtype=dtype)})
    for options in (
        {'primary_keys': {'rows': 'value_id'}},
        {
            'primary_keys': {'rows': 'missing'},
            'time_columns': {'rows': 'absent'},
        },
    ):
        failures = []
        for core in (False, True):
            with pytest.raises(StructuredDataError) as caught:
                describe(monkeypatch, {'rows': frame}, core=core, **options)
            failures.append((caught.value.code, str(caught.value)))
        assert failures[1] == failures[0]


@pytest.mark.parametrize('wide', [False, True])
def test_masked_uint64_exact_width_and_relationships(monkeypatch, wide):
    first = 2**63 if wide else 2**63 - 2
    frames = {
        'users': pd.DataFrame(
            {'user_id': pd.Series([first, first + 1], dtype='UInt64')}
        ),
        'orders': pd.DataFrame(
            {
                'order_id': pd.Series([10, 11, 12], dtype='Int64'),
                'user_id': pd.Series([first, None, first + 1], dtype='UInt64'),
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
    ):
        old, old_notices = describe(monkeypatch, frames, core=False, **options)
        new, new_notices = describe(monkeypatch, frames, core=True, **options)
        assert projection(new) == projection(old)
        assert new_notices == old_notices
        assert new['users']['user_id'].dtype == ('string' if wide else 'int64')


@pytest.mark.parametrize(
    'key', ['user_id', ['user_id'], ('user_id',), ('user_id', 'group')]
)
def test_masked_native_key_forms_and_composite_repeated_example(
    monkeypatch, key
):
    frame = pd.DataFrame(
        {
            'user_id': pd.Series([*range(99), 98], dtype='Int64'),
            'group': pd.Series([*range(99), 98], dtype='Float32'),
        }
    )
    old, old_notices = describe(
        monkeypatch, {'users': frame}, core=False, primary_keys={'users': key}
    )
    new, new_notices = describe(
        monkeypatch, {'users': frame}, core=True, primary_keys={'users': key}
    )
    assert projection(new) == projection(old)
    assert old_notices
    assert new_notices == old_notices


@pytest.mark.parametrize('dtype', ['Float32', 'Float64'])
@pytest.mark.parametrize(
    'variant',
    ['nan', 'nan_duplicate', 'zero_duplicate', 'infinities', 'masked'],
)
def test_masked_float_native_presence_and_key_equality(
    monkeypatch, dtype, variant
):
    native_type = np.float32 if dtype == 'Float32' else np.float64
    values = np.arange(100, dtype=native_type)
    mask = np.zeros(100, dtype=bool)
    if variant == 'nan':
        values[-1] = np.nan
    elif variant == 'nan_duplicate':
        values[-2:] = np.nan
    elif variant == 'zero_duplicate':
        values[-1] = -0.0
    elif variant == 'infinities':
        values[-2:] = [-np.inf, np.inf]
    else:
        mask[-1] = True
    frame = pd.DataFrame(
        {'user_id': pd.Series(pd.arrays.FloatingArray(values, mask))}
    )
    old, old_notices = describe(monkeypatch, {'users': frame}, core=False)
    new, new_notices = describe(monkeypatch, {'users': frame}, core=True)
    assert projection(new) == projection(old)
    assert new_notices == old_notices
    if variant in ('nan', 'infinities'):
        assert new['users'].primary_key == 'user_id'
        assert not new_notices
    else:
        assert old_notices


def test_masked_key_warning_precedes_later_failure_and_releases_rows(
    monkeypatch,
):
    first = pd.DataFrame(
        {'user_id': pd.Series([*range(99), None], dtype='Int64')}
    )
    later = pd.DataFrame([[1, 2]], columns=['same', 'same'])
    outcomes = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            with pytest.raises(StructuredDataError) as caught:
                Schema.from_data({'users': first, 'later': later})
        outcomes.append(
            (str(caught.value), [(str(n.message), n.category) for n in notices])
        )
    assert outcomes[1] == outcomes[0]
    assert outcomes[0][1][0][1] is SDMWarning
    first.loc[99, 'user_id'] = 99
    new, notices = describe(monkeypatch, {'users': first}, core=True)
    assert new['users'].primary_key == 'user_id'
    assert not notices
