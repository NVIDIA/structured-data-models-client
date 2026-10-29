# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public materialize/predict twins for native PQL tabular task observations."""

import numbers
import warnings
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from predictive_query import Stype
from test_t42_frames import projection
from test_t46_direct_numeric import Scripted

from sdm_client import PredictiveQuery, SDMClient, _routing
from sdm_client.materializers import tabular as original


class Distinct:
    def __init__(self, events, values, size):
        self.events, self.values, self.size = events, values, size

    def __len__(self):
        self.events.append(('len', self.size))
        return self.size

    def __iter__(self):
        self.events.append('iter')
        for ordinal, value in enumerate(self.values):
            self.events.append(('next', ordinal))
            yield value
        self.events.append('exhausted')


def _journey(
    monkeypatch,
    *,
    core,
    factory,
    size=2,
    task=None,
    stype=Stype.numerical,
    path='indices',
):
    events = []
    frame = pd.DataFrame(
        {'entity_id': np.arange(12, dtype='int64'), 'answer': [0.0, 1.0] * 6}
    )
    query = PredictiveQuery(
        'PREDICT things.answer FOR EACH things.entity_id', {'things': frame}
    )
    query.schema['things']['answer'].stype = stype
    values = factory(events)
    target_owners = []
    dropped = []
    native_unique = pd.Series.unique
    native_dropna = pd.Series.dropna
    native_snapshot = query.schema.snapshot
    stated = task(events) if callable(task) else task

    def dropna(series, *args, **kwargs):
        if target_owners and series is target_owners[0]:
            events.append('dropna')
            value = native_dropna(series, *args, **kwargs)
            dropped.append(value)
            return value
        return native_dropna(series, *args, **kwargs)

    def unique(series, *args, **kwargs):
        if dropped and series is dropped[0]:
            events.append('unique')
            return Distinct(events, values, size)
        return native_unique(series, *args, **kwargs)

    def snapshot():
        events.append('snapshot')
        patch.setattr(pd.Series, 'dropna', native_dropna)
        patch.setattr(pd.Series, 'unique', native_unique)
        return native_snapshot()

    with monkeypatch.context() as patch:
        patch.setenv(
            _routing.ENVIRONMENT,
            ','.join(
                _routing.MATERIALIZE['frames']
                + _routing.TABULAR_PREDICT_FRAME
                + _routing.DIRECT_CONSUMERS
            )
            if core
            else '',
        )
        patch.setattr(pd.Series, 'dropna', dropna)
        patch.setattr(pd.Series, 'unique', unique)
        patch.setattr(query.schema, 'snapshot', snapshot)
        from sdm_client._rust import tabular_task

        actual_kind = (
            tabular_task.task_kind
            if core
            else original.TabularMaterializer._task_kind
        )

        def captured(asked, target, stated):
            target_owners.append(target)
            result = actual_kind(asked, target, stated)
            if stated is not None:
                events.append(('original-stated', result is stated))
            return result

        if core:
            patch.setattr(tabular_task, 'task_kind', captured)

            def forbidden(*args, **kwargs):
                raise AssertionError(
                    'core task inference called a legacy task helper'
                )

            patch.setattr(original.TabularMaterializer, '_task_kind', forbidden)
            patch.setattr(original, '_derive_task_kind', forbidden)
            patch.setattr(original, '_holds_two_whole_numbers', forbidden)
            patch.setattr(original, '_is_whole', forbidden)
        else:
            patch.setattr(
                original.TabularMaterializer,
                '_task_kind',
                staticmethod(captured),
            )
        scripted = Scripted()
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            try:
                arguments = (
                    {'indices': [11]}
                    if path == 'indices'
                    else {'predict': pd.DataFrame({'entity_id': [11]})}
                )
                request = query.materialize(**arguments, task=stated)
                # Constructor/predict distinct reads are separate from inference.
                patch.setattr(pd.Series, 'dropna', native_dropna)
                patch.setattr(pd.Series, 'unique', native_unique)
                with SDMClient('http://localhost:1', max_retries=0) as client:
                    client._transport.close()
                    client._transport = scripted
                    result = client.predict(request, model='kumo-tabular')
                    answer = result.to_pandas()
                outcome = (
                    'ok',
                    request.task.kind,
                    request.context.instance_table.to_dict('split'),
                    request.predict.instance_table.to_dict('split'),
                    projection(request.schema),
                    request.query,
                    answer.to_dict('split'),
                )
            except Exception as error:
                outcome = (
                    'error',
                    type(error),
                    getattr(error, 'code', None),
                    str(error),
                    getattr(error, 'details', None),
                )
    for body in scripted.calls:
        body['metadata']['request_id'] = '<drawn>'
    first_party = [
        (str(item.message), item.category)
        for item in notices
        if 'holds two whole numbers' in str(item.message)
    ]
    return outcome, scripted.calls, events, first_party


@pytest.mark.parametrize(
    'size,values,expected',
    [
        (0, (), 'regression'),
        (1, (1,), 'regression'),
        (3, (0, 1, 2), 'regression'),
        (2, (), 'classification'),
        (2, (0,), 'classification'),
        (2, (0, 1, 2), 'classification'),
        (2, (True, False), 'classification'),
        (2, (np.int64(0), 10**100), 'classification'),
        (2, (0.5, object()), 'regression'),
        (2, (float('nan'), 1), 'regression'),
        (2, (float('inf'), 1), 'regression'),
        (2, (Decimal('0'), Decimal('1')), 'classification'),
        (2, (Decimal('0.5'), object()), 'regression'),
        (2, (Decimal('NaN'), object()), 'regression'),
    ],
)
@pytest.mark.parametrize('path', ['indices', 'supplied'])
def test_task_inference_uses_actual_lazy_native_distinct_values(
    monkeypatch, size, values, expected, path
):
    # Both public forms retain the same final captured target observation.
    old = _journey(
        monkeypatch,
        core=False,
        factory=lambda events: values,
        size=size,
        path=path,
    )
    current = _journey(
        monkeypatch,
        core=True,
        factory=lambda events: values,
        size=size,
        path=path,
    )
    assert current == old
    assert current[0][0:2] == ('ok', expected)
    if size != 2:
        assert 'iter' not in current[2]
    if values and (
        values[0] == 0.5
        or isinstance(values[0], Decimal)
        and not values[0].is_finite()
    ):
        assert ('next', 1) not in current[2]
    assert current[2].index('snapshot') > current[2].index('unique')


@pytest.mark.parametrize(
    'stated,stype',
    [
        ('regression', Stype.numerical),
        ('classification', Stype.numerical),
        (None, Stype.categorical),
    ],
)
def test_explicit_and_categorical_shortcuts_do_not_read_distinct_values(
    monkeypatch, stated, stype
):
    old = _journey(
        monkeypatch,
        core=False,
        factory=lambda events: (),
        task=stated,
        stype=stype,
    )
    current = _journey(
        monkeypatch,
        core=True,
        factory=lambda events: (),
        task=stated,
        stype=stype,
    )
    assert current == old
    assert current[0][0] == 'ok'
    assert 'dropna' not in current[2] and 'unique' not in current[2]


@pytest.mark.parametrize(
    'fault', [None, OverflowError, TypeError, ValueError, RuntimeError]
)
def test_real_native_float_failures_keep_only_the_original_caught_ancestry(
    monkeypatch, fault
):
    def factory(events):
        class RealValue:
            def __float__(self):
                events.append('float')
                if fault is not None:
                    raise fault('native real conversion')
                return 1.0

        numbers.Real.register(RealValue)
        return [RealValue(), 0]

    old = _journey(monkeypatch, core=False, factory=factory)
    current = _journey(monkeypatch, core=True, factory=factory)
    assert current == old
    assert (
        ('next', 1) in current[2]
        if fault is None
        else ('next', 1) not in current[2]
    )
    if fault is RuntimeError:
        assert current[0][0:2] == ('error', RuntimeError)


@pytest.mark.parametrize('value', [[], 3, 'unknown'])
def test_invalid_stated_task_preserves_public_details_without_numeric_reads(
    monkeypatch, value
):
    old = _journey(
        monkeypatch, core=False, factory=lambda events: (), task=value
    )
    current = _journey(
        monkeypatch, core=True, factory=lambda events: (), task=value
    )
    assert current == old
    assert current[0][0] == 'error'
    assert current[0][-1] == {
        'task': value,
        'supported': ['classification', 'regression'],
    }
    assert 'dropna' not in current[2]
