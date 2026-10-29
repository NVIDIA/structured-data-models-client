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


def _error_graph(error):
    seen = {}

    def visit(value):
        if value is None:
            return None
        key = id(value)
        if key in seen:
            return ('same', seen[key])
        seen[key] = len(seen)
        return (
            type(value),
            str(value),
            value.__suppress_context__,
            visit(value.__cause__),
            visit(value.__context__),
        )

    return visit(error)


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
    failures=None,
    resolved_stype=None,
    on_notice=None,
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
    active = [False]
    stated = task(events, target_owners, active) if callable(task) else task

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
        if resolved_stype is not None:
            native_resolve = query._resolve

            def resolve(profile):
                asked = native_resolve(profile)
                asked.target_ast.stype = resolved_stype
                return asked

            patch.setattr(query, '_resolve', resolve)
        from sdm_client._rust import tabular_task

        actual_kind = (
            tabular_task.task_kind
            if core
            else original.TabularMaterializer._task_kind
        )

        def captured(asked, target, stated):
            target_owners.append(target)
            active[0] = True
            try:
                result = actual_kind(asked, target, stated)
            finally:
                active[0] = False
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
        graph = None
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            native_show = warnings.showwarning

            def show(message, category, filename, lineno, file=None, line=None):
                if 'holds two whole numbers' in str(message):
                    events.append('classification-notice')
                    if on_notice is not None:
                        on_notice(events, query, target_owners)
                return native_show(
                    message, category, filename, lineno, file=file, line=line
                )

            warnings.showwarning = show
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
                graph = _error_graph(error)
                if failures is not None:
                    failures.append(error)
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
    return outcome, scripted.calls, events, first_party, graph


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
        (2, (np.bool_(True), False), 'regression'),
        (2, (np.float32(0), np.float64(1)), 'classification'),
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
    if current[3]:
        assert (
            current[2].index('unique')
            < current[2].index('classification-notice')
            < current[2].index('snapshot')
        )


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
    'fault',
    [
        None,
        OverflowError,
        TypeError,
        ValueError,
        RuntimeError,
        MemoryError,
        StopIteration,
    ],
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


@pytest.mark.parametrize(
    'mode', ['accepted', 'hash-error', 'equality-error', 'repr-error']
)
def test_stated_subclass_native_protocols_and_original_object_survive(
    monkeypatch, mode
):
    def statement(events, owners, active):
        class Kind(str):
            def __hash__(self):
                if active[0]:
                    events.append(('hash', owners[0].name))
                    if mode == 'hash-error':
                        raise ValueError('native task hash')
                return str.__hash__(self)

            def __eq__(self, other):
                if active[0]:
                    events.append(('equal', other))
                    if mode == 'equality-error':
                        raise RuntimeError('native task equality')
                return str.__eq__(self, other)

            def __repr__(self):
                if active[0]:
                    events.append('repr')
                    if mode == 'repr-error':
                        raise ValueError('native task repr')
                return str.__repr__(self)

        return Kind('unknown' if mode == 'repr-error' else 'classification')

    old = _journey(
        monkeypatch, core=False, factory=lambda events: (), task=statement
    )
    current = _journey(
        monkeypatch, core=True, factory=lambda events: (), task=statement
    )
    assert current == old
    assert 'dropna' not in current[2]
    if mode == 'accepted':
        assert ('original-stated', True) in current[2]
    else:
        assert current[0][0] == 'error'


def test_target_stype_refusal_precedes_every_stated_protocol_and_distinct_read(
    monkeypatch,
):
    def statement(events, owners, active):
        class Kind(str):
            def __hash__(self):
                events.append('unexpected task hash')
                raise AssertionError('unsupported target observed stated task')

        return Kind('classification')

    arguments = dict(
        factory=lambda events: (), task=statement, resolved_stype=Stype.text
    )
    old = _journey(monkeypatch, core=False, **arguments)
    current = _journey(monkeypatch, core=True, **arguments)
    assert current == old
    assert current[0][0] == 'error'
    assert current[0][2] == 'UNSUPPORTED_REQUEST'
    assert current[0][-1] == {}
    assert 'unexpected task hash' not in current[2]
    assert 'dropna' not in current[2]


@pytest.mark.parametrize(
    'finite,truths',
    [(True, [True]), (False, [False, False]), (False, [False, True])],
)
def test_decimal_native_finite_round_equality_and_truth_order(
    monkeypatch, finite, truths
):
    def factory(events):
        class Truth:
            def __init__(self, values):
                self.values = iter(values)

            def __bool__(self):
                value = next(self.values)
                events.append(('truth', value))
                return value

        class Value(Decimal):
            def is_finite(self):
                events.append('finite')
                return True if finite else Truth(truths)

            def to_integral_value(self):
                events.append('round')
                return Decimal('1')

            def __eq__(self, other):
                events.append(('decimal-equal', str(other)))
                return Truth(truths)

        return [Value('1'), 0]

    old = _journey(monkeypatch, core=False, factory=factory)
    current = _journey(monkeypatch, core=True, factory=factory)
    assert current == old
    assert (
        current[2].index('finite') < current[2].index(('truth', truths[0]))
        if not finite
        else current[2].index('finite') < current[2].index('round')
    )
    if not finite:
        assert 'round' not in current[2]
        assert current[2].count(('truth', False)) == (
            2 if not truths[-1] else 1
        )


@pytest.mark.parametrize(
    'where', ['finite', 'round', 'equality', 'returned-truth']
)
@pytest.mark.parametrize('failure', [RuntimeError, StopIteration])
def test_decimal_native_failures_keep_generator_and_all_truth_boundaries(
    monkeypatch, where, failure
):
    def factory(events):
        class Truth:
            def __bool__(self):
                events.append('truth')
                raise failure('native decimal observation')

        class Value(Decimal):
            def is_finite(self):
                events.append('finite')
                if where == 'finite':
                    raise failure('native decimal observation')
                return True

            def to_integral_value(self):
                events.append('round')
                if where == 'round':
                    raise failure('native decimal observation')
                return Decimal('1')

            def __eq__(self, other):
                events.append('equal')
                if where == 'equality':
                    raise failure('native decimal observation')
                return Truth()

        return [Value('1'), 0]

    old = _journey(monkeypatch, core=False, factory=factory)
    current = _journey(monkeypatch, core=True, factory=factory)
    assert current == old
    assert current[0][0] == 'error'
    assert ('next', 1) not in current[2]


def test_final_sampled_target_not_full_source_selects_task_and_notice_order(
    monkeypatch,
):
    answers = []
    for core in (False, True):
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
            frame = pd.DataFrame(
                {
                    'entity_id': np.arange(12, dtype='int64'),
                    'answer': [0.0, 1.0, 2.0] * 4,
                }
            )
            query = PredictiveQuery(
                'PREDICT things.answer FOR EACH things.entity_id',
                {'things': frame},
            )
            query.schema['things']['answer'].stype = Stype.numerical
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                request = query.materialize(indices=[11], num_context_samples=2)
            assert len(frame.answer.unique()) == 3
            assert len(request.context.instance_table.answer.unique()) == 2
            assert request.task.kind == 'classification'
            assert any(
                'holds two whole numbers' in str(item.message)
                for item in caught
            )
            scripted = Scripted()
            with SDMClient('http://localhost:1', max_retries=0) as client:
                client._transport.close()
                client._transport = scripted
                result = client.predict(
                    request, model='kumo-tabular'
                ).to_pandas()
            for body in scripted.calls:
                body['metadata']['request_id'] = '<drawn>'
            answers.append(
                (
                    request.context.instance_table.to_dict('split'),
                    request.predict.instance_table.to_dict('split'),
                    request.task,
                    projection(request.schema),
                    request.query,
                    scripted.calls,
                    result.to_dict('split'),
                )
            )
    assert answers[0] == answers[1]


def test_uncaught_real_failure_keeps_the_actual_native_exception_owner(
    monkeypatch,
):
    for core in (False, True):
        native = RuntimeError('actual caller-owned real conversion')
        observed = []

        def factory(events):
            class Value:
                def __float__(self):
                    events.append('float')
                    raise native

            numbers.Real.register(Value)
            return [Value(), 1]

        answer = _journey(
            monkeypatch, core=core, factory=factory, failures=observed
        )
        assert observed == [native]
        assert observed[0] is native
        assert answer[0][0:2] == ('error', RuntimeError)
        assert ('next', 1) not in answer[2]


def test_virtual_integral_is_not_coerced_or_narrowed(monkeypatch):
    def factory(events):
        class Value:
            def __int__(self):
                events.append('unexpected integer conversion')
                raise AssertionError('Integral recognition must not coerce')

        numbers.Integral.register(Value)
        return [Value(), 10**1000]

    old = _journey(monkeypatch, core=False, factory=factory)
    current = _journey(monkeypatch, core=True, factory=factory)
    assert current == old
    assert current[0][0:2] == ('ok', 'classification')
    assert 'unexpected integer conversion' not in current[2]


@pytest.mark.parametrize('effect', ['schema-edit', 'native-failure'])
def test_first_party_task_notice_precedes_later_native_schema_and_request(
    monkeypatch, effect
):
    def notice(events, query, owners):
        events.append('notice effect')
        assert len(owners) == 1
        if effect == 'native-failure':
            raise ValueError('actual inference-notice handler')
        query.schema['things']['answer'].stype = Stype.categorical

    old = _journey(
        monkeypatch, core=False, factory=lambda events: [0, 1], on_notice=notice
    )
    current = _journey(
        monkeypatch, core=True, factory=lambda events: [0, 1], on_notice=notice
    )
    assert current == old
    if effect == 'native-failure':
        assert current[0][0:2] == ('error', ValueError)
        assert 'snapshot' not in current[2]
    else:
        assert current[0][0:2] == ('ok', 'classification')
        assert current[2].index('notice effect') < current[2].index('snapshot')


def test_decimal_true_finite_operand_is_released_before_rounding(monkeypatch):
    def factory(events):
        class Finite:
            def __bool__(self):
                events.append('finite truth')
                return True

            def __del__(self):
                events.append('finite released')

        class Value(Decimal):
            def is_finite(self):
                return Finite()

            def to_integral_value(self):
                events.append('round')
                return Decimal('1')

        return [Value('1'), 0]

    old = _journey(monkeypatch, core=False, factory=factory)
    current = _journey(monkeypatch, core=True, factory=factory)
    assert current == old
    assert current[2].index('finite released') < current[2].index('round')


def test_task_refusal_retains_the_ordinary_active_caller_context(monkeypatch):
    answers = []
    for core in (False, True):
        try:
            raise RuntimeError('active task caller')
        except RuntimeError:
            answer = _journey(
                monkeypatch,
                core=core,
                factory=lambda events: (),
                task='unknown',
            )
        answers.append(answer)
        assert answer[4][2] is False
        assert answer[4][4][0] is RuntimeError
    assert answers[0] == answers[1]
