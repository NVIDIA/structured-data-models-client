# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public materialize/predict twins for native PQL tabular task observations."""

import json
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
                if stated is not None:
                    events.append(
                        ('request-original-stated', request.task.kind is stated)
                    )
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
        assert ('request-original-stated', True) in current[2]
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


@pytest.mark.parametrize('text', ['classification', 'native-choice'])
def test_native_stated_object_keeps_constructor_and_per_call_protocol(
    monkeypatch, text
):
    def task(events, owners, active):
        class NativeTask(str):
            def __hash__(self):
                events.append('task hash')
                return hash('classification')

            def __eq__(self, other):
                events.append(
                    (
                        'task equal',
                        str.__str__(other)
                        if isinstance(other, str)
                        else type(other).__name__,
                    )
                )
                return other == 'classification'

            def __ne__(self, other):
                events.append(('task unequal', str.__str__(other)))
                return other != 'classification'

        return NativeTask(text)

    old = _journey(
        monkeypatch, core=False, factory=lambda events: (), task=task
    )
    current = _journey(
        monkeypatch, core=True, factory=lambda events: (), task=task
    )
    # JSON storage observes the original spelling without invoking equality or
    # formatting hooks, while the native operation trace is compared separately.
    assert json.dumps(current[0:2], default=str) == json.dumps(
        old[0:2], default=str
    )
    assert current[2:] == old[2:]
    assert current[0][0] == 'ok'
    assert str.__str__(current[0][1]) == text
    assert current[1][0]['task']['kind'] == text
    assert ('original-stated', True) in current[2]
    assert ('request-original-stated', True) in current[2]
    assert 'dropna' not in current[2]
    assert current[2].count('task hash') >= 4


def _envelope_task_journey(monkeypatch, *, core, effect):
    from dataclasses import replace

    from sdm_client import Request
    from sdm_client.request import Task

    events = []
    context = pd.DataFrame({'answer': [0, 1, 0], 'later': [2, 3, 4]})
    scoring = pd.DataFrame({'later': [5]})
    scripted = Scripted()
    native_classification = Task.is_classification.fget

    with monkeypatch.context() as patch:
        patch.setenv(
            _routing.ENVIRONMENT,
            ','.join(_routing.DIRECT_CONSUMERS) if core else '',
        )
        request = Request(
            context, scoring, target='answer', task='classification'
        )
        if effect == 'text':
            target = request.schema[request.entity_table]['answer']
            target.stype = 'categorical'
            target.dtype = 'string'
            target.stype = 'text'

        def classification(task):
            assert task is request.task
            events.append('actual classification')
            if effect == 'later':
                later = request.schema[request.entity_table]['later']
                later.stype = 'categorical'
                later.dtype = 'string'
                later.stype = 'text'
            elif effect == 'failure':
                raise LookupError('actual task meaning read')
            elif effect in ('target-replacement', 'target-failure'):
                if events.count('actual classification') == 1:
                    request.task = replace(
                        task,
                        target=replace(task.target, column_name='later'),
                    )
                elif effect == 'target-failure':
                    raise LookupError('actual later target meaning read')
            return native_classification(task)

        patch.setattr(Task, 'is_classification', property(classification))
        caught = None
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            try:
                with SDMClient('http://localhost:1', max_retries=0) as client:
                    client._transport.close()
                    client._transport = scripted
                    result = client.predict(request, model='kumo-tabular')
                    observed = ('ok', result.to_pandas().to_dict('split'))
            except Exception as error:
                caught = error
                observed = ('error', _error_graph(error))
        for body in scripted.calls:
            body['metadata']['request_id'] = '<drawn>'
        return (
            observed,
            scripted.calls,
            events,
            projection(request.schema),
            caught,
        )


@pytest.mark.parametrize(
    'effect',
    [
        'ordinary',
        'text',
        'later',
        'failure',
        'target-replacement',
        'target-failure',
    ],
)
def test_task_column_meaning_uses_reached_native_property_and_later_metadata(
    monkeypatch, effect
):
    old = _envelope_task_journey(monkeypatch, core=False, effect=effect)
    current = _envelope_task_journey(monkeypatch, core=True, effect=effect)
    assert current[:4] == old[:4]
    if effect == 'text':
        assert current[2] == []
        assert (
            current[1][0]['schema']['instance_table']['columns']['answer'][
                'stype'
            ]
            == 'categorical'
        )
    elif effect == 'failure':
        assert current[1] == []
        assert type(current[4]) is LookupError
        assert str(current[4]) == 'actual task meaning read'
    elif effect in ('target-replacement', 'target-failure'):
        assert current[2] == ['actual classification', 'actual classification']
        if effect == 'target-failure':
            assert current[1] == []
            assert type(current[4]) is LookupError
            assert str(current[4]) == 'actual later target meaning read'
        else:
            assert current[0][0] == 'ok'
            payload = current[1][0]
            assert payload['task']['target']['column_name'] == 'answer'
            assert (
                payload['schema']['instance_table']['columns']['later']['stype']
                == 'categorical'
            )
    else:
        assert current[2] == ['actual classification']
        assert current[0][0] == 'ok'
        if effect == 'later':
            assert current[1][0]['schema']['instance_table']['columns'][
                'later'
            ] == {'dtype': 'int64', 'stype': 'categorical'}


@pytest.mark.parametrize(
    'effect',
    ['truth-order', 'truth-failure', 'nonstring-mutation', 'repr-failure'],
)
def test_fresh_call_task_observations_preserve_truth_and_selected_refusal(
    monkeypatch, effect
):
    from dataclasses import replace

    from sdm_client import Request

    def observe(core):
        events = []
        scripted = Scripted()
        context = pd.DataFrame({'answer': [0, 1, 0], 'feature': [2, 3, 4]})
        scoring = pd.DataFrame({'feature': [5]})
        with monkeypatch.context() as patch:
            patch.setenv(
                _routing.ENVIRONMENT,
                ','.join(_routing.DIRECT_CONSUMERS) if core else '',
            )
            request = Request(
                context, scoring, target='answer', task='classification'
            )
            failure = ValueError('actual selected task repr')
            changed = [False]

            class Truth:
                def __init__(self, name):
                    self.name = name

                def __bool__(self):
                    events.append(('truth', self.name))
                    if effect == 'truth-failure' and self.name == 'link':
                        raise ArithmeticError(
                            'actual native task comparison truth'
                        )
                    return False

            class Choice(str):
                def __hash__(self):
                    events.append('native hash')
                    if effect == 'nonstring-mutation' and not changed[0]:
                        changed[0] = True
                        request.task = replace(request.task, kind=42)
                        events.append('task became integer')
                    return str.__hash__(self)

                def __eq__(self, other):
                    events.append(('native equal', other))
                    if other == 'temporal_link_prediction':
                        return Truth('link')
                    if other == 'forecasting':
                        return Truth('forecast')
                    return str.__eq__(self, other)

                def __repr__(self):
                    events.append('selected native repr')
                    raise failure

            request.task = replace(
                request.task,
                kind=Choice(
                    'absent' if effect == 'repr-failure' else 'classification'
                ),
            )
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                try:
                    with SDMClient(
                        'http://localhost:1', max_retries=0
                    ) as client:
                        client._transport.close()
                        client._transport = scripted
                        result = client.predict(request, model='kumo-tabular')
                        outcome = ('ok', result.to_pandas().to_dict('split'))
                except Exception as error:
                    outcome = (
                        'error',
                        _error_graph(error),
                        getattr(error, 'details', None),
                        error is failure,
                    )
            for body in scripted.calls:
                body['metadata']['request_id'] = '<drawn>'
            return outcome, scripted.calls, events

    old, current = observe(False), observe(True)
    assert current == old
    if effect == 'nonstring-mutation':
        assert current[1] == []
        assert current[0][2]['task'] == 42
        assert current[2].count('task became integer') == 1
    elif effect == 'repr-failure':
        assert current[0][3] is True
        assert current[2][-1] == 'selected native repr'
        assert current[1] == []
    elif effect == 'truth-failure':
        assert current[0][1][0] is ArithmeticError
        assert ('truth', 'forecast') not in current[2]
        assert current[1] == []
    else:
        link = current[2].index(('native equal', 'temporal_link_prediction'))
        forecast = current[2].index(('native equal', 'forecasting'))
        assert (
            link
            < forecast
            < current[2].index(('truth', 'link'))
            < current[2].index(('truth', 'forecast'))
        )
        assert current[0][0] == 'ok'


@pytest.mark.parametrize(
    'effect',
    [
        'changed-display',
        'second-call-failure',
        'invalid-response',
        'empty',
        'contains-false',
        'contains-failure',
        'contains-truth-failure',
    ],
)
def test_response_task_display_uses_captured_payload_owner_after_row_admission(
    monkeypatch, effect
):
    from dataclasses import replace

    from sdm_client import Request

    def observe(core):
        events = []
        context = pd.DataFrame({'answer': [0, 1, 0], 'feature': [2, 3, 4]})
        scoring = (
            pd.DataFrame({'feature': pd.Series([], dtype='int64')})
            if effect == 'empty'
            else pd.DataFrame({'feature': [5]})
        )
        calls = [0]
        native_failure = LookupError('actual second task display')

        class ContainmentTruth:
            def __bool__(self):
                events.append('containment truth')
                raise native_failure

        class Display(str):
            def __contains__(self, word):
                events.append(('contains', word))
                if effect == 'contains-failure':
                    raise native_failure
                if effect == 'contains-truth-failure':
                    return ContainmentTruth()
                return False

        class Choice(str):
            def __str__(self):
                calls[0] += 1
                events.append(('task display', calls[0]))
                if effect == 'second-call-failure' and calls[0] == 2:
                    raise native_failure
                if effect.startswith('contains-'):
                    return Display('classification')
                return (
                    'regression'
                    if effect == 'changed-display'
                    else 'classification'
                )

        owner = Choice('classification')
        results = []
        with monkeypatch.context() as patch:
            patch.setenv(
                _routing.ENVIRONMENT,
                ','.join(_routing.DIRECT_CONSUMERS) if core else '',
            )
            request = Request(context, scoring, target='answer', task=owner)
            assert request.task.kind is owner

            class Answer(Scripted):
                def predict(self, payload):
                    result = super().predict(payload)
                    events.append('transport')
                    if effect == 'changed-display':
                        request.task = replace(
                            request.task, kind='classification'
                        )
                        assert request.task.kind is not owner
                    if effect == 'invalid-response':
                        result['predictions'] = []
                    else:
                        for item in result['predictions']:
                            item['prediction'] = '3'
                    return result

            scripted = Answer()
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                with SDMClient('http://localhost:1', max_retries=0) as client:
                    client._transport.close()
                    client._transport = scripted
                    for _ in range(2 if effect == 'second-call-failure' else 1):
                        try:
                            result = client.predict(
                                request, model='kumo-tabular'
                            )
                            results.append(
                                ('ok', result.to_pandas().to_dict('split'))
                            )
                        except Exception as error:
                            results.append(
                                (
                                    'error',
                                    type(error),
                                    str(error),
                                    getattr(error, 'code', None),
                                    getattr(error, 'details', None),
                                    error is native_failure,
                                )
                            )
            for body in scripted.calls:
                body['metadata']['request_id'] = '<drawn>'
            return results, scripted.calls, events, calls[0]

    old, current = observe(False), observe(True)
    assert current == old
    if effect == 'changed-display':
        assert current[3] == 1
        frame = current[0][0][1]
        assert frame['data'][0][frame['columns'].index('prediction')] == '3'
        assert current[2] == ['transport', ('task display', 1)]
    elif effect == 'second-call-failure':
        assert current[3] == 2
        assert current[0][0][0] == 'ok'
        assert current[0][1][-1] is True
        assert len(current[1]) == 2
    elif effect.startswith('contains-'):
        assert current[3] == 1
        assert current[2][:3] == [
            'transport',
            ('task display', 1),
            ('contains', 'classification'),
        ]
        if effect == 'contains-false':
            frame = current[0][0][1]
            assert frame['data'][0][frame['columns'].index('prediction')] == '3'
        else:
            assert current[0][0][-1] is True
            if effect == 'contains-truth-failure':
                assert current[2][-1] == 'containment truth'
    else:
        assert current[3] == 0
        assert current[0][0][0] == (
            'error' if effect == 'invalid-response' else 'ok'
        )
        if effect == 'empty':
            assert current[1] == []


def test_runtime_task_membership_releases_its_replaced_owner_before_frame_reads(
    monkeypatch,
):
    from dataclasses import replace

    from sdm_client import Request

    def observe(core):
        events = []
        context = pd.DataFrame({'answer': [0, 1, 0], 'feature': [2, 3, 4]})
        scoring = pd.DataFrame({'feature': [5]})
        scripted = Scripted()
        hashes = [0]
        with monkeypatch.context() as patch:
            patch.setenv(
                _routing.ENVIRONMENT,
                ','.join(_routing.DIRECT_CONSUMERS) if core else '',
            )
            request = Request(
                context, scoring, target='answer', task='classification'
            )

            class Choice(str):
                def __hash__(self):
                    hashes[0] += 1
                    events.append(('hash', hashes[0]))
                    if hashes[0] == 3:
                        request.task = replace(request.task, kind='regression')
                    return str.__hash__(self)

                def __del__(self):
                    events.append('released task')
                    scoring.iloc[0, 0] = 91

            request.task = replace(request.task, kind=Choice('classification'))

            class Answer(Scripted):
                def predict(self, payload):
                    events.append('transport')
                    return super().predict(payload)

            scripted = Answer()
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                with SDMClient('http://localhost:1', max_retries=0) as client:
                    client._transport.close()
                    client._transport = scripted
                    result = client.predict(request, model='kumo-tabular')
            for body in scripted.calls:
                body['metadata']['request_id'] = '<drawn>'
            return result.to_pandas().to_dict('split'), scripted.calls, events

    old, current = observe(False), observe(True)
    assert current == old
    assert current[2].index('released task') < current[2].index('transport')
    assert current[1][0]['predict']['instance_table']['rows'][0] == [91]


def test_empty_prediction_admits_a_native_nonstring_task_without_wire_storage(
    monkeypatch,
):
    from sdm_client import Request

    def observe(core):
        events = []

        class Choice:
            def __hash__(self):
                events.append('hash')
                return hash('classification')

            def __eq__(self, other):
                events.append(('equal', other))
                return other == 'classification'

            def __str__(self):
                raise AssertionError('empty prediction displayed task')

            def __repr__(self):
                raise AssertionError('empty prediction represented task')

        owner = Choice()
        context = pd.DataFrame({'answer': [0, 1, 0], 'feature': [2, 3, 4]})
        scoring = pd.DataFrame({'feature': pd.Series([], dtype='int64')})
        scripted = Scripted()
        with monkeypatch.context() as patch:
            patch.setenv(
                _routing.ENVIRONMENT,
                ','.join(_routing.DIRECT_CONSUMERS) if core else '',
            )
            request = Request(context, scoring, target='answer', task=owner)
            with SDMClient('http://localhost:1', max_retries=0) as client:
                client._transport.close()
                client._transport = scripted
                result = client.predict(request, model='kumo-tabular')
            return (
                result.to_pandas().to_dict('split'),
                scripted.calls,
                events,
                request.task.kind is owner,
            )

    old, current = observe(False), observe(True)
    assert current == old
    assert current[0]['data'] == []
    assert current[1] == []
    assert current[3] is True


@pytest.mark.parametrize('rows', [10001, 9])
def test_required_capability_truth_owner_survives_capacity_then_releases(
    monkeypatch, rows
):
    from dataclasses import replace
    from sdm_client import Request

    def observe(core):
        events = []
        context = pd.DataFrame(
            {'answer': [i % 2 for i in range(rows)], 'feature': range(rows)}
        )
        scoring = pd.DataFrame({'feature': [rows + 1]})
        with monkeypatch.context() as patch:
            patch.setenv(
                _routing.ENVIRONMENT,
                ','.join(_routing.DIRECT_CONSUMERS) if core else '',
            )
            request = Request(
                context, scoring, target='answer', task='classification'
            )

            class Truth:
                def __bool__(self):
                    events.append('truth')
                    return False

                def __del__(self):
                    events.append('released capability')
                    context.drop(context.index[3:], inplace=True)

            class Choice(str):
                __hash__ = str.__hash__

                def __eq__(self, other):
                    if other == 'temporal_link_prediction':
                        return Truth()
                    return str.__eq__(self, other)

            request.task = replace(request.task, kind=Choice('classification'))

            class Answer(Scripted):
                def predict(self, payload):
                    events.append('transport')
                    return super().predict(payload)

            scripted = Answer()
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                with SDMClient('http://localhost:1', max_retries=0) as client:
                    client._transport.close()
                    client._transport = scripted
                    try:
                        result = client.predict(request, model='kumo-tabular')
                        outcome = ('ok', result.to_pandas().to_dict('split'))
                    except Exception as error:
                        # Record while the refusal traceback still owns preflight locals.
                        outcome = (
                            'error',
                            _error_graph(error),
                            getattr(error, 'details', None),
                            len(context),
                            list(events),
                        )
            for body in scripted.calls:
                body['metadata']['request_id'] = '<drawn>'
            return (
                outcome,
                scripted.calls,
                outcome[4] if rows > 10000 else events,
            )

    old, current = observe(False), observe(True)
    assert current == old
    if rows > 10000:
        assert current[0][0] == 'error'
        assert current[0][2]['context_rows'] == 10001
        assert current[0][3] == 10001
        assert 'released capability' not in current[0][4]
        assert current[1] == []
    else:
        assert current[0][0] == 'ok'
        assert current[2].index('released capability') < current[2].index(
            'transport'
        )
        assert len(current[1][0]['context']['instance_table']['rows']) == 3


@pytest.mark.parametrize(
    'effect',
    [
        'display',
        'contains',
        'display-release',
        'transport-kind',
        'nonclassification',
    ],
)
def test_response_restoration_reads_actual_payload_after_display_effects(
    monkeypatch, effect
):
    from sdm_client import Request

    def observe(core):
        events, holder = [], {}
        context = pd.DataFrame({'answer': [0, 1, 0], 'feature': [2, 3, 4]})
        scoring = pd.DataFrame({'feature': [5]})

        def mutate():
            holder['payload']['task']['target']['dtype'] = 'string'
            events.append('changed actual dtype')

        class Display(str):
            def __contains__(self, word):
                events.append(('contains', word))
                if effect == 'contains':
                    mutate()
                return str.__contains__(self, word)

            def __del__(self):
                if effect == 'display-release':
                    mutate()

        class Choice(str):
            def __str__(self):
                events.append('display')
                if effect in ('display', 'transport-kind'):
                    mutate()
                if effect in ('contains', 'display-release'):
                    return Display('classification')
                return (
                    'regression'
                    if effect == 'nonclassification'
                    else 'classification'
                )

        class NoTargetRead(dict):
            def get(self, key, *args):
                if key == 'target':
                    raise AssertionError('nonclassification read target dtype')
                return super().get(key, *args)

        class Answer(Scripted):
            def predict(self, payload):
                holder['payload'] = payload
                events.append('transport')
                answer = super().predict(payload)
                if effect == 'transport-kind':
                    payload['task']['kind'] = Choice('classification')
                elif effect == 'nonclassification':
                    payload['task'] = NoTargetRead(payload['task'])
                for item in answer['predictions']:
                    item['prediction'] = '3'
                return answer

        scripted = Answer()
        with monkeypatch.context() as patch:
            patch.setenv(
                _routing.ENVIRONMENT,
                ','.join(_routing.DIRECT_CONSUMERS) if core else '',
            )
            request = Request(
                context, scoring, target='answer', task=Choice('classification')
            )
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                with SDMClient('http://localhost:1', max_retries=0) as client:
                    client._transport.close()
                    client._transport = scripted
                    result = client.predict(request, model='kumo-tabular')
            for body in scripted.calls:
                body['metadata']['request_id'] = '<drawn>'
            return result.to_pandas().to_dict('split'), scripted.calls, events

    old, current = observe(False), observe(True)
    assert current == old
    frame = current[0]
    assert frame['data'][0][frame['columns'].index('prediction')] == '3'
    assert current[1][0]['task']['target']['dtype'] == 'int64'
    assert current[2][0] == 'transport'
    if effect != 'nonclassification':
        assert current[2][-1] == 'changed actual dtype'


def test_direct_byte_transport_receives_exact_core_bytes(monkeypatch):
    from sdm_client import Request

    received, offered = [], []
    original = SDMClient._send_body

    def send_body(self, adapter, body, **kwargs):
        offered.append(body)
        return original(self, adapter, body, **kwargs)

    class Answer(Scripted):
        def predict_body(self, body):
            received.append(body)
            return json.dumps(super().predict(json.loads(body))).encode()

    monkeypatch.setenv(
        _routing.ENVIRONMENT, ','.join(_routing.DIRECT_CONSUMERS)
    )
    monkeypatch.setattr(SDMClient, '_send_body', send_body)
    scripted = Answer()
    request = Request(
        pd.DataFrame({'answer': [0, 1, 0], 'feature': [2, 3, 4]}),
        pd.DataFrame({'feature': [5]}),
        target='answer',
        task='classification',
    )
    with SDMClient('http://localhost:1', max_retries=0) as client:
        client._transport.close()
        client._transport = scripted
        result = client.predict(request, model='kumo-tabular')
    assert len(result.to_pandas()) == 1
    assert len(received) == 1
    assert received == offered
    assert type(received[0]) is bytes


@pytest.mark.parametrize(
    'effect', ['false-truth', 'false-prefix', 'truth-error', 'prefix-error']
)
def test_late_native_dtype_shortcuts_and_failures_follow_actual_protocol(
    monkeypatch, effect
):
    from sdm_client import Request

    def observe(core):
        events = []
        held = LookupError('actual late dtype observation')

        class Dtype:
            def __bool__(self):
                events.append('dtype truth')
                if effect == 'truth-error':
                    raise held
                return effect != 'false-truth'

            def startswith(self, prefixes):
                events.append(('dtype prefix', prefixes))
                if effect == 'prefix-error':
                    raise held
                return False

        class Answer(Scripted):
            def predict(self, payload):
                answer = super().predict(payload)
                payload['task']['target']['dtype'] = Dtype()
                for item in answer['predictions']:
                    item['prediction'] = '3'
                return answer

        scripted = Answer()
        with monkeypatch.context() as patch:
            patch.setenv(
                _routing.ENVIRONMENT,
                ','.join(_routing.DIRECT_CONSUMERS) if core else '',
            )
            request = Request(
                pd.DataFrame({'answer': [0, 1, 0], 'feature': [2, 3, 4]}),
                pd.DataFrame({'feature': [5]}),
                target='answer',
                task='classification',
            )
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                with SDMClient('http://localhost:1', max_retries=0) as client:
                    client._transport.close()
                    client._transport = scripted
                    try:
                        result = client.predict(request, model='kumo-tabular')
                        outcome = ('ok', result.to_pandas().to_dict('split'))
                    except Exception as error:
                        outcome = ('error', _error_graph(error), error is held)
            for body in scripted.calls:
                body['metadata']['request_id'] = '<drawn>'
            return outcome, scripted.calls, events

    old, current = observe(False), observe(True)
    assert current == old
    if effect.endswith('error'):
        assert current[0][-1] is True
    else:
        frame = current[0][1]
        assert frame['data'][0][frame['columns'].index('prediction')] == '3'
    assert current[2][0] == 'dtype truth'
    if effect in ('false-truth', 'truth-error'):
        assert current[2] == ['dtype truth']
    else:
        assert current[2][-1] == ('dtype prefix', ('int', 'float', 'bool'))


@pytest.mark.parametrize(
    'effect',
    ['int', 'stateful', 'collision', 'prediction-error', 'probability-error'],
)
def test_positive_native_dtype_restores_each_prediction_and_key_in_order(
    monkeypatch, effect
):
    from sdm_client import Request

    def observe(core):
        events, holder = [], {}
        held = ArithmeticError('actual native dtype conversion')
        counters = {'bool': 0, 'int': 0, 'float': 0}

        class Dtype:
            def __bool__(self):
                events.append('dtype truth')
                return True

            def startswith(self, prefix):
                events.append(('prefix', prefix))
                if isinstance(prefix, tuple):
                    return True
                counters[prefix] += 1
                if (
                    effect == 'prediction-error'
                    and prefix == 'bool'
                    and counters[prefix] == 3
                ):
                    raise held
                if (
                    effect == 'probability-error'
                    and prefix == 'bool'
                    and counters[prefix] == 2
                ):
                    raise held
                if effect == 'stateful':
                    return prefix == (
                        'float' if counters['bool'] == 2 else 'int'
                    )
                return prefix == 'int'

        class Answer(Scripted):
            def predict(self, payload):
                answer = super().predict(payload)
                holder['payload'] = payload
                payload['task']['target']['dtype'] = Dtype()
                for ordinal, item in enumerate(answer['predictions']):
                    item['prediction'] = str(3 + ordinal)
                    item['probabilities'] = (
                        {'03': 0.4, '3': 0.6}
                        if effect == 'collision'
                        else {'03': 1.0}
                    )
                return answer

        scripted = Answer()
        with monkeypatch.context() as patch:
            patch.setenv(
                _routing.ENVIRONMENT,
                ','.join(_routing.DIRECT_CONSUMERS) if core else '',
            )
            request = Request(
                pd.DataFrame({'answer': [0, 1, 0], 'feature': [2, 3, 4]}),
                pd.DataFrame({'feature': [5, 6]}),
                target='answer',
                task='classification',
            )
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                with SDMClient('http://localhost:1', max_retries=0) as client:
                    client._transport.close()
                    client._transport = scripted
                    try:
                        result = client.predict(request, model='kumo-tabular')
                        frame = result.to_pandas().astype(object)
                        outcome = (
                            'ok',
                            frame.where(pd.notna(frame), None).to_dict('split'),
                        )
                    except Exception as error:
                        outcome = ('error', _error_graph(error), error is held)
            for body in scripted.calls:
                body['metadata']['request_id'] = '<drawn>'
            return outcome, scripted.calls, events

    old, current = observe(False), observe(True)
    assert current == old
    assert current[2][:2] == [
        'dtype truth',
        ('prefix', ('int', 'float', 'bool')),
    ]
    if effect.endswith('error'):
        assert current[0][-1] is True
    else:
        frame = current[0][1]
        assert frame['data'][0][frame['columns'].index('prediction')] == 3
        if effect == 'collision':
            assert 'probability_03' in frame['columns']
            assert 'probability_3' in frame['columns']
        elif effect == 'stateful':
            assert 'probability_3.0' in frame['columns']


def test_transport_replacement_releases_original_payload_kind_before_normalization(
    monkeypatch,
):
    from dataclasses import replace
    from sdm_client import Request

    def observe(core):
        events, holder = [], {}
        with monkeypatch.context() as patch:
            patch.setenv(
                _routing.ENVIRONMENT,
                ','.join(_routing.DIRECT_CONSUMERS) if core else '',
            )
            request = Request(
                pd.DataFrame({'answer': [0, 1, 0], 'feature': [2, 3, 4]}),
                pd.DataFrame({'feature': [5]}),
                target='answer',
                task='classification',
            )

            class Choice(str):
                def __deepcopy__(self, memo):
                    return 'classification'

                def __del__(self):
                    events.append('released original kind')
                    holder['payload']['task']['target']['dtype'] = 'string'

            request.task = replace(request.task, kind=Choice('classification'))

            class Answer(Scripted):
                def predict(self, payload):
                    answer = super().predict(payload)
                    holder['payload'] = payload
                    request.task = replace(request.task, kind='classification')
                    payload['task']['kind'] = 'classification'
                    events.append('after replacement')
                    for item in answer['predictions']:
                        item['prediction'] = '3'
                    return answer

            scripted = Answer()
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                with SDMClient('http://localhost:1', max_retries=0) as client:
                    client._transport.close()
                    client._transport = scripted
                    result = client.predict(request, model='kumo-tabular')
            for body in scripted.calls:
                body['metadata']['request_id'] = '<drawn>'
            return (
                result.to_pandas().to_dict('split'),
                scripted.calls,
                list(events),
            )

    old, current = observe(False), observe(True)
    assert current == old
    assert current[2] == ['released original kind', 'after replacement']
    frame = current[0]
    assert frame['data'][0][frame['columns'].index('prediction')] == '3'


def test_native_normalization_releases_captured_task_before_held_dtype(
    monkeypatch,
):
    from sdm_client import Request

    def observe(core):
        events, holder = [], {}

        class Dtype:
            def startswith(self, prefix):
                return isinstance(prefix, tuple) or prefix == 'int'

            def __del__(self):
                events.append('released dtype')

        class TaskDict(dict):
            def __del__(self):
                events.append('release task start')
                self['target'] = {}
                events.append('release task end')

        class Choice(str):
            def __str__(self):
                events.append('display')
                holder['payload']['task'] = {
                    'kind': 'classification',
                    'target': {'dtype': 'string'},
                }
                return 'classification'

        class Answer(Scripted):
            def predict(self, payload):
                answer = super().predict(payload)
                holder['payload'] = payload
                payload['task'] = TaskDict(payload['task'])
                payload['task']['target']['dtype'] = Dtype()
                for item in answer['predictions']:
                    item['prediction'] = '3'
                return answer

        scripted = Answer()
        with monkeypatch.context() as patch:
            patch.setenv(
                _routing.ENVIRONMENT,
                ','.join(_routing.DIRECT_CONSUMERS) if core else '',
            )
            request = Request(
                pd.DataFrame({'answer': [0, 1, 0], 'feature': [2, 3, 4]}),
                pd.DataFrame({'feature': [5]}),
                target='answer',
                task=Choice('classification'),
            )
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                with SDMClient('http://localhost:1', max_retries=0) as client:
                    client._transport.close()
                    client._transport = scripted
                    result = client.predict(request, model='kumo-tabular')
                    events.append('returned')
            for body in scripted.calls:
                body['metadata']['request_id'] = '<drawn>'
            return result.to_pandas().to_dict('split'), scripted.calls, events

    old, current = observe(False), observe(True)
    assert current == old
    assert current[2] == [
        'display',
        'release task start',
        'release task end',
        'released dtype',
        'returned',
    ]
    frame = current[0]
    assert frame['data'][0][frame['columns'].index('prediction')] == 3
