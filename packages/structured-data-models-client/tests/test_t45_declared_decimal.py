# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Decimal source edits, native coercion effects and graph reuse."""

from __future__ import annotations

import hashlib
import json
import warnings
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from t45_frame_routes import FRAME_ROUTES

from sdm_client import PredictiveQuery, Schema
from sdm_client.materializers.relational import RelationalMaterializer

PANDAS = int(pd.__version__.split('.')[0])
REFERENCE = json.loads(
    Path(__file__)
    .with_name(f't45_declared_decimal_pandas{PANDAS}_reference.json')
    .read_text()
)


def payload_digest(value):
    # Every payload field and array order participates, without policy masks.
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def frames():
    return {
        'users': pd.DataFrame(
            {
                'user_id': np.arange(8, dtype='int64'),
                'age': np.arange(20, 28, dtype='int64'),
            }
        ),
        'items': pd.DataFrame(
            {
                'item_id': np.arange(4, dtype='int64'),
                'amount': [3.0, 4.0, 5.0, 6.0],
            }
        ),
        'buys': pd.DataFrame(
            {
                'buy_id': np.arange(120, dtype='int64'),
                'user_id': np.arange(120) % 8,
                'item_id': [(i * 3) % 4 for i in range(120)],
                'ts': pd.date_range('2023-01-01', periods=120, freq='5D'),
            }
        ),
    }


def native_decimal(events, effect):
    class NativeDecimal(Decimal):
        def _called(self, name):
            events.append(name)
            effect(name)

        def __float__(self):
            self._called('float')
            return super().__float__()

        def __int__(self):
            self._called('int')
            return super().__int__()

        def __str__(self):
            self._called('str')
            return super().__str__()

        def __repr__(self):
            self._called('repr')
            return super().__repr__()

    return NativeDecimal


def values(storage, shape, indexes, callback):
    raw = [
        None
        if shape in ('missing', 'whole-missing') and i == 3
        else Decimal(2**63 + i)
        if shape == 'overflow'
        else Decimal(2**64 - 4 + i)
        if shape == 'uint64-max'
        else Decimal(2**64 + i)
        if shape == 'past-uint64'
        else Decimal(-(2**63) - i - 1)
        if shape == 'past-int64-min'
        else Decimal(i + 1).scaleb(129)
        if shape == 'scale-minus129'
        else Decimal(i + 1).scaleb(-129)
        if shape == 'scale-plus129'
        else Decimal(i)
        if shape in ('whole', 'whole-missing')
        else Decimal(i) + Decimal('0.70')
        for i in indexes
    ]
    if storage.startswith('object'):
        return pd.Series(
            [
                None
                if value is None
                else callback(value)
                if storage == 'object-callback'
                else value
                for value in raw
            ],
            dtype=object,
        )
    scale = (
        -129
        if shape == 'scale-minus129'
        else 129
        if shape == 'scale-plus129'
        else 0
        if shape
        in ('whole-missing', 'uint64-max', 'past-uint64', 'past-int64-min')
        else 2
    )
    if abs(scale) == 129:
        array = pa.Array.from_buffers(
            pa.decimal128(38, scale),
            len(indexes),
            [
                None,
                pa.py_buffer(
                    b''.join(
                        (i + 1).to_bytes(16, 'little', signed=True)
                        for i in indexes
                    )
                ),
            ],
        )
    else:
        array = pa.array(raw, type=pa.decimal128(38, scale))
    return pd.Series(pd.arrays.ArrowExtensionArray(array))


@pytest.mark.parametrize('core', [False, True], ids=['original', 'core'])
@pytest.mark.parametrize(
    'case',
    REFERENCE['cases'],
    ids=lambda case: '-'.join(
        str(case[name]) for name in ('storage', 'shape', 'declared', 'mutation')
    ),
)
def test_declared_decimal_materialize_matches_frozen_native_behavior(
    monkeypatch, core, case
):
    monkeypatch.setenv('SDM_CORE_ROUTES', FRAME_ROUTES if core else '')
    if core:

        def legacy_model(*args, **kwargs):
            raise AssertionError(
                'core Decimal materialization called the Python engine'
            )

        monkeypatch.setattr(RelationalMaterializer, '_get_model', legacy_model)
        monkeypatch.setattr(
            'sdm_client.materializers.relational._as_typed', legacy_model
        )
        monkeypatch.setattr(
            'sdm_client.materializers.relational._to_declared', legacy_model
        )
    observed = {'calls': []}
    events = []
    effect_reached = False
    effects_active = False
    schema = None

    def effect(name):
        nonlocal effect_reached
        if (
            effects_active
            and case['mutation']
            and not effect_reached
            and name in ('float', 'str')
        ):
            effect_reached = True
            schema['buys']['item_id'].dtype = 'string'
            schema['users'].data.loc[0, 'age'] = 999
            events.append('effect')

    callback = native_decimal(events, effect)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        schema = Schema.from_data(
            frames(),
            primary_keys={
                'users': 'user_id',
                'items': 'item_id',
                'buys': 'buy_id',
            },
            time_columns={'users': None, 'items': None, 'buys': 'ts'},
        )
        for table, indexes in [
            ('items', list(range(4))),
            ('buys', [(i * 3) % 4 for i in range(120)]),
        ]:
            schema[table].data['item_id'] = values(
                case['storage'], case['shape'], indexes, callback
            )
        for table in ('items', 'buys'):
            schema[table]['item_id'].dtype = case['declared']
        effects_active = True
        question = PredictiveQuery(
            'PREDICT LIST_DISTINCT(buys.item_id, 0, 30, days) FOR EACH users.user_id',
            schema,
        )
        for _ in range(2):
            start = len(events)
            try:
                request = question.materialize(
                    indices=[0, 1],
                    anchor_time=pd.Timestamp('2023-03-01'),
                    random_seed=7,
                )
                call = {
                    'outcome': 'success',
                    'payload_sha256': payload_digest(request.payloads),
                    'context_rows': request.context_rows,
                    'predict_rows': request.predict_rows,
                }
            except Exception as error:
                call = {
                    'outcome': 'error',
                    'error': {
                        'type': type(error).__name__,
                        'code': getattr(error, 'code', None),
                        'message': str(error),
                    },
                }
            call['native_calls'] = events[start:]
            observed['calls'].append(call)
        observed['warnings'] = [
            {'type': type(n.message).__name__, 'message': str(n.message)}
            for n in notices
        ]
    observed['effect_reached'] = effect_reached
    assert observed == case['expected']
