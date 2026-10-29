# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed native handover ordering through frame and provider domain ports.

Selections below are supplied facts from a scripted preparation boundary.
These tests never evaluate queries, comparisons or aggregation in Python.
Real Host and public SQL consumers cover the corresponding Rust arithmetic.
"""

from __future__ import annotations

import json
import weakref
from dataclasses import replace

import pandas as pd
import pyarrow as pa
import pytest

from sdm_client._rust import frame_facade
from sdm_client._rust.errors import raise_native
from sdm_client._rust.native_label_stage import (
    ColumnExport,
    complete_eligibility,
    complete_target,
    complete_what_if,
)
from sdm_client._rust.native_products import Bindings, Product
from sdm_client.errors import StructuredDataError

ROWS = [3, 1, 0, 2]
TARGET = {'source': {'source': ['events']}, 'column': 'target'}
OPERAND = {'source': {'source': ['events']}, 'column': 'flag'}


class InputOwner:
    def __init__(self, events):
        self.events = events

    def rows(self):
        self.events.append('input-rows')
        return pa.table(
            {
                'position': pa.array(range(4), type=pa.uint64()),
                'source_row': pa.array(ROWS, type=pa.uint64()),
                'target': pa.array([20, None, 2**53 + 1, 10], type=pa.int64()),
            }
        )


class SelectionOwner:
    def __init__(self, events, stage, selected):
        self.events, self.stage, self.selected = events, stage, selected

    def selection(self):
        self.events.append(('selection', self.stage))
        return pa.table({'selected': pa.array(self.selected, type=pa.bool_())})


class NativeFailure(ValueError):
    def __str__(self):
        raise AssertionError('native failure was formatted by the handover')


class Source:
    def __init__(self, events, *, failure=None, fail_at=None):
        self.events, self.failure, self.fail_at = events, failure, fail_at
        self.operands = 0

    def target_series(self, binding, rows):
        assert rows == ROWS
        name = binding['column']
        if name == 'flag':
            assert binding == OPERAND
            self.operands += 1
        else:
            assert binding == TARGET
        self.events.append(('read', name, self.operands))
        if (name == 'target' and self.fail_at == 'target') or (
            name == 'flag' and self.operands == 2 and self.fail_at == 'right'
        ):
            events, failure = self.events, self.failure

            class Refusing(pd.Series):
                @property
                def _constructor(self):
                    return Refusing

                def astype(self, dtype, *args, **kwargs):
                    events.append(('native-cast', dtype))
                    raise failure

                def notna(self):
                    pytest.fail('failed native cast observed missingness')

            return Refusing([1.0, 2.0, 3.0, 4.0]).iloc[rows]
        values = (
            [2**53 + 1, None, 10, 20]
            if name == 'target'
            else [1, 2, 3, 4]
            if self.operands == 1
            else [11, 12, 13, 14]
        )
        return pd.Series(values, dtype='Int64', name=name).iloc[rows]


def snapshot(converted):
    assert converted.rows.column_names == ['position', 'target']
    assert converted.rows.column('position').type == pa.uint64()
    return {
        'rows': converted.rows.to_pydict(),
        'presence': None
        if converted.presence is None
        else (converted.presence.column('present').to_pylist()),
        'encoding': converted.encoding,
    }


class Preparation:
    """Explicit domain methods; masks are supplied, never computed from values."""

    def __init__(self, inputs, events, *, false_left=False, core_failure=None):
        self.inputs, self.events = inputs, events
        self.false_left, self.core_failure = false_left, core_failure
        self.converted = []
        self.conditions = []
        self.failures = []
        self.owner_refs = []

    def product(self, stage, converted, selected):
        bindings = Bindings()
        input_id = bindings.add(self.inputs.owners[self.inputs.data['owner']])
        owner = SelectionOwner(self.events, stage, selected)
        self.owner_refs.append(weakref.ref(owner))
        owner_id = bindings.add(owner)
        conditions = [
            {'column': occurrence, 'converted': value.bind(bindings)}
            for occurrence, value in self.conditions
        ]
        return Product(
            'target_column',
            {
                'input': {**self.inputs.data, 'owner': input_id},
                'owner': owner_id,
                'rows': converted.rows.num_rows,
                'derivation': 'filtered'
                if stage == 'filtered'
                else ('windowed' if stage == 'windowed' else 'column'),
                'condition_values': conditions,
            },
            bindings.attachments,
        )

    def target_column(self, inputs, converted):
        self.events.append(('core', 'target'))
        self.converted.append(('target', snapshot(converted)))
        selected = (
            [True, False, True]
            if inputs.data['form'] == 'plain'
            else [False] * 4
        )
        return self.product('target', converted, selected)

    def condition_column(self, column, occurrence, converted):
        ordinal = len(self.conditions) + 1
        self.events.append(('core', 'condition', ordinal))
        self.converted.append((f'condition-{ordinal}', snapshot(converted)))
        if self.core_failure is not None:
            raise self.core_failure
        self.conditions.append((occurrence, converted))
        # Both right evaluation and the final condition are preparation facts.
        selected = [False] * 4 if self.false_left else [True, True, False, True]
        return self.product(f'condition-{ordinal}', converted, selected)

    def filtered_target(self, column, converted):
        self.events.append(('core', 'filtered'))
        self.converted.append(('filtered', snapshot(converted)))
        selected = [] if self.false_left else [False, True]
        return self.product('filtered', converted, selected)

    def windowed_target(self, column, converted):
        self.events.append(('core', 'windowed'))
        self.converted.append(('windowed', snapshot(converted)))
        return self.product('windowed', converted, [])

    def native_failure(self, failure, *args, **kwargs):
        self.events.append('native-failure')
        self.failures.append(failure)
        raise_native(failure.error)


def test_condition_input_keeps_all_native_values_without_a_filter_or_window_handover():
    events = []
    owner = InputOwner(events)
    inputs = Product(
        'label_inputs',
        {
            'population': 'population',
            'range': {'start': 0, 'stop': 4},
            'owner': 1,
            'target': TARGET,
            'form': 'condition',
            'condition_columns': [],
        },
        {1: owner},
    )
    preparation = Preparation(inputs, events)
    completed, native = complete_target(inputs, Source(events), preparation)
    assert native.values.dtype == pd.Int64Dtype()
    assert native.positions.tolist() == [0, 1, 2, 3]
    assert native.values.tolist() == [20, pd.NA, 2**53 + 1, 10]
    assert events == ['input-rows', ('read', 'target', 0), ('core', 'target')]
    assert preparation.converted == [
        (
            'target',
            {
                'rows': {
                    'position': [0, 1, 2, 3],
                    'target': [20, None, 2**53 + 1, 10],
                },
                'presence': [True, False, True, True],
                'encoding': None,
            },
        )
    ]
    assert completed.owners


def frame_port(monkeypatch, preparation):
    """Exercise the real frame methods, bindings, prepare cleanup and adoption."""

    class Finished:
        result = {'op': 'prepare_frames', 'value': 90_000}

        def __init__(self, product):
            self.product = product

        def take(self, attachment):
            if attachment == 90_000:
                return json.dumps(
                    {
                        'value': self.product.kind,
                        'data': self.product.data,
                    }
                ).encode()
            return self.product.owners.pop(attachment)

    def exported(value, attachments):
        return ColumnExport(
            attachments[value['rows']],
            None
            if value['presence'] is None
            else attachments[value['presence']],
            value['encoding'],
        )

    def run(operation, attachments, **kwargs):
        assert operation == {'op': 'prepare_frames', 'input': 1}
        value = json.loads(attachments[1])
        stage = value['stage']
        if stage == 'target_column':
            assert (
                attachments[value['inputs']['owner']]
                is preparation.inputs.owners[1]
            )
            product = preparation.target_column(
                preparation.inputs,
                exported(value['converted'], attachments),
            )
        else:
            assert (
                attachments[value['column']['owner']]
                is preparation.owner_refs[-1]()
            )
            previous = Product('target_column', value['column'], {})
            if stage == 'condition_column':
                product = preparation.condition_column(
                    previous,
                    value['operand']['column'],
                    exported(value['operand']['converted'], attachments),
                )
            elif stage == 'filtered_target':
                product = preparation.filtered_target(
                    previous,
                    exported(value['converted'], attachments),
                )
            else:
                assert stage == 'windowed_target'
                product = preparation.windowed_target(
                    previous,
                    exported(value['converted'], attachments),
                )
        return Finished(product)

    monkeypatch.setattr(frame_facade, 'run', run)
    monkeypatch.setattr(
        frame_facade, 'guard_failure', preparation.native_failure
    )
    return frame_facade._FrameTargetPreparation({}, {}, None)


def setup(
    monkeypatch,
    port,
    *,
    form='filtered',
    false_left=False,
    core_failure=None,
    fail_at=None,
):
    events = []
    inputs = Product(
        'label_inputs',
        {
            'owner': 1,
            'population': 'actual-input',
            'range': {'start': 0, 'stop': 4},
            'target': TARGET,
            'form': form,
            'condition_columns': [dict(OPERAND), dict(OPERAND)]
            if form == 'filtered'
            else [],
        },
        {1: InputOwner(events)},
    )
    failure = NativeFailure()
    source = Source(events, failure=failure, fail_at=fail_at)
    preparation = Preparation(
        inputs, events, false_left=false_left, core_failure=core_failure
    )
    selected = (
        preparation
        if port == 'provider'
        else frame_port(monkeypatch, preparation)
    )
    return inputs, source, preparation, selected, events, failure


@pytest.mark.parametrize('port', ['frame', 'provider'])
@pytest.mark.parametrize('false_left', [False, True])
def test_repeated_operands_follow_each_core_comparison_without_short_circuit(
    monkeypatch,
    port,
    false_left,
):
    inputs, source, preparation, selected, events, _ = setup(
        monkeypatch, port, false_left=false_left
    )
    final, native = complete_target(inputs, source, selected)
    assert events == [
        'input-rows',
        ('read', 'target', 0),
        ('core', 'target'),
        ('read', 'flag', 1),
        ('core', 'condition', 1),
        ('read', 'flag', 2),
        ('core', 'condition', 2),
        ('selection', 'condition-2'),
        ('core', 'filtered'),
        ('selection', 'filtered'),
        ('core', 'windowed'),
    ]
    assert preparation.converted[0][1] == {
        'rows': {'position': [0, 1, 2, 3], 'target': [20, None, 2**53 + 1, 10]},
        'presence': [True, False, True, True],
        'encoding': None,
    }
    for ordinal, values in ((1, [4, 2, 1, 3]), (2, [14, 12, 11, 13])):
        assert preparation.converted[ordinal][1] == {
            'rows': {'position': [0, 1, 2, 3], 'target': values},
            'presence': [True] * 4,
            'encoding': None,
        }
    assert preparation.converted[3][1] == {
        'rows': {
            'position': [] if false_left else [0, 3],
            'target': [] if false_left else [20, 10],
        },
        'presence': None,
        'encoding': None,
    }
    assert preparation.converted[4][1] == {
        'rows': {
            'position': [] if false_left else [3],
            'target': [] if false_left else [10],
        },
        'presence': [] if false_left else [True],
        'encoding': None,
    }
    assert native.positions.tolist() == ([] if false_left else [3])
    assert native.values.tolist() == ([] if false_left else [10])
    assert str(native.values.dtype) == 'Int64'
    assert source.operands == 2
    assert preparation.failures == []
    assert all(owner() is None for owner in preparation.owner_refs[:-1])
    assert preparation.owner_refs[-1]() is final.owners[final.data['owner']]
    assert inputs.owners


@pytest.mark.parametrize('port', ['frame', 'provider'])
def test_core_left_failure_stops_right_native_effects_and_keeps_its_actual_owner(
    monkeypatch, port
):
    failure = StructuredDataError(
        'actual left comparison failed', code='INVALID_REQUEST'
    )
    inputs, source, preparation, selected, events, _ = setup(
        monkeypatch, port, core_failure=failure
    )
    with pytest.raises(StructuredDataError) as caught:
        complete_target(inputs, source, selected)
    assert caught.value is failure
    assert events == [
        'input-rows',
        ('read', 'target', 0),
        ('core', 'target'),
        ('read', 'flag', 1),
        ('core', 'condition', 1),
    ]
    assert source.operands == 1
    assert preparation.failures == []
    assert all(owner() is None for owner in preparation.owner_refs)


@pytest.mark.parametrize('port', ['frame', 'provider'])
@pytest.mark.parametrize('fail_at', ['target', 'right'])
def test_native_cast_failure_is_presented_once_without_later_core_completion(
    monkeypatch, port, fail_at
):
    inputs, source, preparation, selected, events, failure = setup(
        monkeypatch, port, fail_at=fail_at
    )
    with pytest.raises(NativeFailure) as caught:
        complete_target(inputs, source, selected)
    assert caught.value is failure
    assert len(preparation.failures) == 1
    assert preparation.failures[0].error is failure
    expected = ['input-rows', ('read', 'target', 0)]
    if fail_at == 'right':
        expected += [
            ('core', 'target'),
            ('read', 'flag', 1),
            ('core', 'condition', 1),
            ('read', 'flag', 2),
        ]
    assert events == expected + [('native-cast', 'float32'), 'native-failure']
    assert all(owner() is None for owner in preparation.owner_refs)


@pytest.mark.parametrize('port', ['frame', 'provider'])
def test_plain_nullable_integer_uses_actual_native_cast_and_second_presence(
    monkeypatch, port
):
    inputs, source, preparation, selected, events, _ = setup(
        monkeypatch, port, form='plain'
    )
    _, native = complete_target(inputs, source, selected)
    assert events == [
        'input-rows',
        ('read', 'target', 0),
        ('core', 'target'),
        ('selection', 'target'),
        ('core', 'windowed'),
    ]
    assert preparation.converted[0][1] == {
        'rows': {'position': [0, 2, 3], 'target': [20, 2**53 + 1, 10]},
        'presence': [True, False, True, True],
        'encoding': None,
    }
    assert preparation.converted[1][1] == {
        'rows': {'position': [0, 3], 'target': [20, 10]},
        'presence': [True, True],
        'encoding': None,
    }
    assert native.values.tolist() == [20, 10]
    assert native.positions.tolist() == [0, 3]
    assert str(native.values.dtype) == 'int64'
    assert source.operands == 0


@pytest.mark.parametrize('role', ['eligibility', 'what_if'])
@pytest.mark.parametrize('fail_left', [False, True])
def test_condition_history_completes_each_occurrence_before_next_native_read(
    role, fail_left
):
    events = []
    failure = StructuredDataError(
        'actual first comparison', code='INVALID_REQUEST'
    )
    view = {
        'owner': 2,
        'population': 'actual-source',
        'range': {'start': 0, 'stop': 4},
        'target': OPERAND,
        'form': 'condition',
        'condition_columns': [],
    }
    second = {**view, 'owner': 3}
    initial = Product(
        'label_inputs' if role == 'eligibility' else 'label_groups',
        {
            'owner': 1,
            role: {'columns': [view, second], 'values': []},
            **(
                {'restored': {'rows': 4, 'encoding': None}}
                if role == 'what_if'
                else {}
            ),
        },
        {1: InputOwner(events), 2: InputOwner(events), 3: InputOwner(events)},
    )
    refs = []
    converted = []

    class Sources:
        source = Source(events)

        def columns(self, current):
            assert current.data in (view, second)
            events.append(('source', current.data['owner']))
            return self.source

    class Port:
        def target_column(self, inputs, values):
            converted.append(snapshot(values))
            owner = SelectionOwner(events, 'condition', [])
            refs.append(weakref.ref(owner))
            return Product('target_column', {'owner': 20}, {20: owner})

        def complete(self, previous, column):
            assert column.owners[20] is refs[-1]()
            ordinal = len(converted)
            events.append(('core', role, ordinal))
            if fail_left:
                raise failure
            assert previous.data[role]['values'] == list(range(ordinal - 1))
            owner = SelectionOwner(events, 'history', [])
            refs.append(weakref.ref(owner))
            return Product(
                initial.kind,
                {
                    **initial.data,
                    role: {
                        **initial.data[role],
                        'values': list(range(ordinal)),
                    },
                },
                {1: owner},
            )

        eligibility_column = complete
        what_if_column = complete

        def native_failure(self, error):
            pytest.fail(
                f'unexpected native failure type {type(error).__name__}'
            )

    complete = (
        complete_eligibility if role == 'eligibility' else complete_what_if
    )
    if fail_left:
        with pytest.raises(StructuredDataError) as caught:
            complete(initial, Sources(), Port())
        assert caught.value is failure
    else:
        final = complete(initial, Sources(), Port())
        assert final.data[role]['values'] == [0, 1]
        assert final.owners[1] is refs[-1]()
    expected = [
        ('source', 2),
        'input-rows',
        ('read', 'flag', 1),
        ('core', role, 1),
    ]
    if not fail_left:
        expected += [
            ('source', 3),
            'input-rows',
            ('read', 'flag', 2),
            ('core', role, 2),
        ]
    assert events == expected
    assert converted[0]['rows']['target'] == [4, 2, 1, 3]
    if not fail_left:
        assert converted[1]['rows']['target'] == [14, 12, 11, 13]
    assert all(ref() is None for ref in (refs if fail_left else refs[:-1]))
    assert initial.owners


def test_what_if_cannot_read_any_native_operand_before_restored_groups():
    class NoSources:
        def columns(self, _):
            pytest.fail('native operand reached before restoration')

    from sdm_client._rust.codec import CodecError

    with pytest.raises(CodecError, match='precede restored'):
        complete_what_if(Product('label_groups', {}, {}), NoSources(), object())


@pytest.mark.parametrize('form', ['plain', 'filtered'])
@pytest.mark.parametrize('effect', ['in-place', 'replacement'])
def test_window_time_values_come_from_the_captured_native_owner_after_conversion(
    monkeypatch, form, effect
):
    inputs, source, preparation, _, events, _ = setup(
        monkeypatch, 'provider', form=form
    )
    binding = {'source': TARGET['source'], 'column': 'ts'}
    inputs.data['time'] = binding
    captured = pd.Series(pd.date_range('2026-01-01', periods=4))
    current = [captured]
    actual_read = source.target_series

    def time_series(actual):
        assert actual == binding
        events.append('capture-time')
        return current[0]

    def values(actual, rows):
        original = actual_read(actual, rows)

        class Editing(pd.Series):
            @property
            def _constructor(self):
                return Editing

            def reset_index(self, *args, **kwargs):
                if effect == 'in-place':
                    captured.iloc[:] = pd.Timestamp('2030-01-01')
                else:
                    current[0] = pd.Series([pd.Timestamp('2030-01-01')] * 4)
                return super().reset_index(*args, **kwargs)

        return Editing(original)

    source.time_series = time_series
    source.target_series = values
    observed = []
    target_column, filtered_target = (
        preparation.target_column,
        preparation.filtered_target,
    )

    def target(inputs, converted):
        observed.append(converted.times)
        return target_column(inputs, replace(converted, times=None))

    def filtered(column, converted):
        observed.append(converted.times)
        return filtered_target(column, replace(converted, times=None))

    preparation.target_column = target
    preparation.filtered_target = filtered
    complete_target(inputs, source, preparation)
    assert events[:2] == ['input-rows', 'capture-time']
    expected = captured.iloc[ROWS].tolist()
    assert observed[-1].column(0).to_pylist() == expected
    if form == 'filtered':
        assert observed[0] is None
    if effect == 'replacement':
        assert expected != current[0].iloc[ROWS].tolist()
