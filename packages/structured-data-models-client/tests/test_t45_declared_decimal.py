# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Decimal source edits, native coercion effects and graph reuse."""

from __future__ import annotations

import copy
import hashlib
import json
import warnings
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from t38_forced_routes_record import bare
from t45_frame_routes import FRAME_ROUTES

from sdm_client import PredictiveQuery, Schema
from sdm_client.materializers.relational import RelationalMaterializer

PANDAS = int(pd.__version__.split('.')[0])
REFERENCE = json.loads(
    Path(__file__)
    .with_name(f't45_declared_decimal_pandas{PANDAS}_reference.json')
    .read_text()
)


@pytest.mark.parametrize(
    'changed', ['message', 'class', 'order', 'call', 'effect', 'graph']
)
def test_bounded_decimal_pair_keeps_all_native_warnings_and_call_effects(
    changed,
):
    [case] = [
        case
        for case in REFERENCE['cases']
        if (case['storage'], case['shape'], case['declared'], case['mutation'])
        == ('object', 'scale-minus129', 'float64', False)
    ]
    original = copy.deepcopy(case['expected'])
    original['graphs'] = [
        copy.deepcopy(call['error']) for call in original['calls']
    ]
    _assert_decimal_profile(
        original, copy.deepcopy(original), case['expected'], core=False
    )
    actual = copy.deepcopy(original)
    if changed == 'message':
        actual['warnings'][0]['message'] += ' changed'
    elif changed == 'class':
        actual['warnings'][0]['type'] = 'OtherWarning'
    elif changed == 'order':
        actual['warnings'].reverse()
    elif changed == 'call':
        actual['calls'][0]['error']['message'] += ' changed'
    elif changed == 'effect':
        actual['effect_reached'] = True
    else:
        actual['graphs'][0]['message'] += ' changed'
    if changed in ('call', 'effect'):
        # Even matching twins cannot replace these unchanged pinned facts.
        original = copy.deepcopy(actual)
    with pytest.raises(AssertionError):
        _assert_decimal_profile(actual, original, case['expected'], core=False)


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


def _decimal_error_graph(error, seen=None):
    if error is None:
        return None
    if seen is None:
        seen = {}
    if id(error) in seen:
        return ('reference', seen[id(error)])
    seen[id(error)] = len(seen)
    return (
        type(error).__module__,
        type(error).__name__,
        str(error),
        error.__suppress_context__,
        _decimal_error_graph(error.__cause__, seen),
        _decimal_error_graph(error.__context__, seen),
    )


def _assert_decimal_profile(actual, original, expected, *, core):
    paired = copy.deepcopy(original)
    if core:
        for warning in paired['warnings']:
            warning['message'] = bare(warning['message'])
    assert actual == paired
    assert len(actual['graphs']) == len(expected['calls']) == 2
    # Only this bounded case's actual native warning population is supplied
    # by its fresh original twin. All pinned call and effect facts remain.
    assert actual['calls'] == expected['calls']
    assert actual['effect_reached'] == expected['effect_reached']


def _observe_decimal(case, *, capture_graphs=False):
    observed = {'calls': []}
    if capture_graphs:
        observed['graphs'] = []
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
                if capture_graphs:
                    observed['graphs'].append(None)
                call = {
                    'outcome': 'success',
                    'payload_sha256': payload_digest(request.payloads),
                    'context_rows': request.context_rows,
                    'predict_rows': request.predict_rows,
                }
            except Exception as error:
                if capture_graphs:
                    observed['graphs'].append(_decimal_error_graph(error))
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
    return observed


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
    bounded = (
        case['storage'],
        case['shape'],
        case['declared'],
        case['mutation'],
    ) == (
        'object',
        'scale-minus129',
        'float64',
        False,
    )

    def observed_route(core_route):
        with monkeypatch.context() as leg:
            leg.setenv('SDM_CORE_ROUTES', FRAME_ROUTES if core_route else '')
            if core_route:

                def legacy_model(*args, **kwargs):
                    raise AssertionError(
                        'core Decimal materialization called the Python engine'
                    )

                leg.setattr(RelationalMaterializer, '_get_model', legacy_model)
                leg.setattr(
                    'sdm_client.materializers.relational._as_typed',
                    legacy_model,
                )
                leg.setattr(
                    'sdm_client.materializers.relational._to_declared',
                    legacy_model,
                )
            return _observe_decimal(case, capture_graphs=bounded)

    original = observed_route(False) if bounded else None
    observed = observed_route(core) if core or not bounded else original
    if core:
        # P3E-2 permits only the core's leading notice code. Native effects,
        # warning classes/bodies/order and all complete payload hashes stay.
        for warning in observed['warnings']:
            warning['message'] = bare(warning['message'])
    if bounded:
        _assert_decimal_profile(observed, original, case['expected'], core=core)
    else:
        assert observed == case['expected']
