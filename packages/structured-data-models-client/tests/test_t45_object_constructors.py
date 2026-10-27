# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public object constructors retain exact native methods and schema products."""

from __future__ import annotations

import json
import warnings
from decimal import Context, Decimal, InvalidOperation, localcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sdm_client.schema as native_schema
import sdm_client.wire.dtypes as native_dtypes
from sdm_client import Schema, _routing

REFERENCE = json.loads(
    Path(__file__)
    .with_name('t45_object_constructors_reference.json')
    .read_text()
)
CELL = 'pandas2' if int(pd.__version__.split('.')[0]) == 2 else 'pandas3'
OBSERVED = REFERENCE['cells'][CELL]


@pytest.fixture(params=[False, True], ids=['original', 'core'])
def core(request, monkeypatch):
    selected = request.param
    monkeypatch.setenv(
        'SDM_CORE_ROUTES',
        ','.join(_routing.FRAME_CONSTRUCTORS) if selected else '',
    )
    if selected:

        def legacy_policy(*args, **kwargs):
            raise AssertionError(
                'core object constructor called the old Python inference policy'
            )

        for name in (
            '_infer_table_and_time_column',
            '_column',
            '_infer_primary_key',
            '_infer_time_column',
            '_stype_of',
            'infer_tfm_dtype',
        ):
            monkeypatch.setattr(native_schema, name, legacy_policy)
        monkeypatch.setattr(native_dtypes, '_decimal_dtype', legacy_policy)
    return selected


def table_value(table):
    return {
        'primary_key': table.primary_key,
        'time_column': table.time_column,
        'duplicate_keys': table.duplicate_keys,
        'columns': [
            {
                'name': name,
                'dtype': column.dtype,
                'stype': str(column.stype),
                'untyped': column.untyped,
            }
            for name, column in table.columns.items()
        ],
    }


def warning_values(notices):
    return [
        {'type': type(item.message).__name__, 'message': str(item.message)}
        for item in notices
    ]


class NativeFailure(RuntimeError):
    pass


ERROR_TYPES = {
    'InvalidOperation': InvalidOperation,
    'NativeFailure': NativeFailure,
    'AttributeError': AttributeError,
}


def observed_decimal(mode, trace, counts, frame, raised):
    class ObservedDecimal(Decimal):
        def _observe(self, method):
            counts[method] = counts.get(method, 0) + 1
            trace.append({'method': method, 'call': counts[method]})
            if (
                mode == 'raise-second-tuple'
                and method == 'as_tuple'
                and counts[method] == 2
            ):
                error = NativeFailure('second native Decimal tuple observation')
                raised.append(error)
                raise error
            if (
                mode == 'mutate-later'
                and method == 'is_finite'
                and counts[method] == 1
                and frame[0] is not None
            ):
                frame[0]['later'] = pd.Series(
                    [Decimal('2.25')] * len(frame[0]), dtype=object
                )
            return mode == 'malformed-tuple' and method == 'as_tuple'

        def is_finite(self):
            self._observe('is_finite')
            return super().is_finite()

        def as_tuple(self):
            return (
                ('invalid',)
                if self._observe('as_tuple')
                else super().as_tuple()
            )

        def to_integral_value(self, *args, **kwargs):
            self._observe('to_integral_value')
            return super().to_integral_value(*args, **kwargs)

        def adjusted(self):
            self._observe('adjusted')
            return super().adjusted()

        def __eq__(self, other):
            self._observe('eq')
            return super().__eq__(other)

        def __ne__(self, other):
            self._observe('ne')
            return super().__ne__(other)

        def __le__(self, other):
            self._observe('le')
            return super().__le__(other)

        def __ge__(self, other):
            self._observe('ge')
            return super().__ge__(other)

        def __hash__(self):
            self._observe('hash')
            return super().__hash__()

    return ObservedDecimal


@pytest.mark.parametrize(
    'case', OBSERVED['decimals'], ids=lambda value: value['case']
)
def test_public_object_decimal_constructor_matches_frozen_native_behavior(
    core, case
):
    trace, counts, raised = [], {}, []
    holder = [None]
    cls = (
        observed_decimal(case['mode'], trace, counts, holder, raised)
        if case['mode']
        else Decimal
    )
    with localcontext(Context()):
        values = [
            None if value is None else cls(value) for value in case['values']
        ]
        frame = pd.DataFrame(
            {
                'feature_id' if case['key_shaped'] else 'feature': pd.Series(
                    values, dtype=object
                ),
                'later': np.arange(len(values), dtype='int64'),
            }
        )
        holder[0] = frame
        trace.clear()
        counts.clear()
        raised.clear()
        expected = case['expected']
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            if 'error' in expected:
                with pytest.raises(
                    ERROR_TYPES[expected['error']['type']]
                ) as failure:
                    Schema.from_data({'t': frame})
                error = failure.value
                assert {
                    'type': type(error).__name__,
                    'code': getattr(error, 'code', None),
                    'message': str(error),
                } == expected['error']
                assert error.__cause__ is None
                assert error.__context__ is None
                if raised:
                    assert error is raised[0]
            else:
                schema = Schema.from_data({'t': frame})
                assert table_value(schema['t']) == expected['table']
        # These include the later key/advice native hash and equality reads.
        # No column position or post-inference suffix is excluded.
        assert trace == expected['trace']
        assert warning_values(notices) == expected['warnings']


def link_frames(kind):
    def keys(values):
        return pd.Series(
            ['item-' + str(value) for value in values],
            dtype=object if kind == 'object' else 'category',
        )

    return {
        'users': pd.DataFrame(
            {
                'user_id': np.arange(8, dtype='int64'),
                'age': np.arange(20, 28, dtype='int64'),
            }
        ),
        'items': pd.DataFrame(
            {'item_id': keys(range(4)), 'amount': [3.0, 4.0, 5.0, 6.0]}
        ),
        'buys': pd.DataFrame(
            {
                'buy_id': np.arange(120, dtype='int64'),
                'user_id': np.arange(120) % 8,
                'item_id': keys([(i * 3) % 4 for i in range(120)]),
                'ts': pd.date_range('2023-01-01', periods=120, freq='5D'),
            }
        ),
    }


@pytest.mark.parametrize(
    'case', OBSERVED['strings'], ids=lambda value: value['kind']
)
def test_public_object_strings_and_category_constructor_match_complete_frozen_schema(
    core, case
):
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        schema = Schema.from_data(
            link_frames(case['kind']),
            primary_keys={
                'users': 'user_id',
                'items': 'item_id',
                'buys': 'buy_id',
            },
            time_columns={'users': None, 'items': None, 'buys': 'ts'},
        )
        actual = [
            {'name': name, **table_value(table)}
            for name, table in schema.tables.items()
        ]
        assert actual == case['tables']
        assert [
            repr(relationship) for relationship in schema.relationships
        ] == case['relationships']
    assert warning_values(notices) == case['warnings']
