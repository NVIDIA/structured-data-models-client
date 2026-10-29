# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-first native ownership and observation-boundary regressions."""

from __future__ import annotations

import pandas as pd
import pytest

from sdm_client._rust.frame_native import (
    NativeColumn,
    NativeFrameColumns,
    NativeFrameFailure,
    convert_text,
)
from sdm_client.errors import StructuredDataError


class DisplayFailure(Exception):
    def __init__(self):
        self.reads = 0

    def __str__(self):
        self.reads += 1
        return f'observation {self.reads}'


def test_capture_keeps_exception_identity_without_observing_native_text():
    error = DisplayFailure()
    context = ValueError('original context')

    def fail():
        raise error

    try:
        try:
            raise context
        except ValueError:
            fail()
    except DisplayFailure as caught:
        held = NativeFrameFailure.capture(caught)
        traceback = caught.__traceback__
    assert held.error is error
    assert held.traceback is traceback
    assert held.context is context
    assert held.cause is None
    assert held.facts.deepest_module == __name__
    assert error.reads == 0
    # A later core-selected presentation can observe the native message at
    # distinct positions. Caching it would change a stateful exception.
    assert held.read_message() == 'observation 1'
    assert held.read_message() == 'observation 2'
    assert held.read_type_name() == 'DisplayFailure'


def test_structured_failure_capture_never_calls_native_display():
    class Preserved(StructuredDataError):
        def __str__(self):
            raise AssertionError('StructuredDataError was displayed')

    error = Preserved('preserve me', code='INVALID_REQUEST')
    held = NativeFrameFailure.capture(error)
    assert held.error is error
    assert 'structured_data_error' in held.facts.ancestry
    assert held.facts.deepest_module is None


def test_key_error_is_not_lost_in_the_direct_conversion_ancestry():
    error = KeyError('missing')
    held = NativeFrameFailure.capture(error)
    assert held.facts.ancestry == ('key_error',)
    assert held.read_message() == "'missing'"


def test_native_display_failure_remains_the_actual_failure_at_that_boundary():
    failure = RuntimeError('display failed')

    class Unprintable(TypeError):
        def __str__(self):
            raise failure

    held = NativeFrameFailure.capture(Unprintable())
    assert held.facts.ancestry == ('type_error',)
    with pytest.raises(RuntimeError) as caught:
        held.read_message()
    assert caught.value is failure
    assert caught.value.__context__ is held.error
    assert held.error.__traceback__ is held.traceback


def test_native_message_observations_preserve_native_chain_mutations():
    changed = KeyError('changed by native display')

    class Mutating(ValueError):
        def __str__(self):
            self.__context__ = changed
            return 'native message'

    held = NativeFrameFailure.capture(Mutating())
    assert held.read_message() == 'native message'
    assert held.error.__context__ is changed
    assert held.read_message() == 'native message'
    assert held.error.__context__ is changed


@pytest.mark.parametrize('raised', [False, True])
@pytest.mark.parametrize('mutation', ['none', 'context', 'traceback'])
@pytest.mark.parametrize('observation', ['message', 'type_name'])
def test_failed_native_observation_keeps_actual_traceback_and_chain_edits(
    raised, mutation, observation
):
    failure = RuntimeError('actual native observation failure')
    changed = KeyError('changed by the native observation')
    try:
        raise KeyError('native traceback owner')
    except KeyError as native:
        changed_traceback = native.__traceback__

    def native_failure():
        if mutation == 'context':
            error.__context__ = changed
        elif mutation == 'traceback':
            error.__traceback__ = changed_traceback
        raise failure

    class Name(type):
        def __getattribute__(cls, name):
            if name == '__name__':
                native_failure()
            return super().__getattribute__(name)

    class Unprintable(TypeError, metaclass=Name):
        def __str__(self):
            native_failure()

    error = Unprintable()
    if raised:
        try:
            raise error
        except Unprintable:
            pass
    original_traceback = error.__traceback__
    original_context = error.__context__
    original_cause = error.__cause__
    original_suppression = error.__suppress_context__
    held = NativeFrameFailure.capture(error)
    with pytest.raises(RuntimeError) as caught:
        if observation == 'message':
            held.read_message()
        else:
            held.read_type_name()
    assert caught.value is failure
    assert caught.value.__context__ is error
    assert error.__traceback__ is (
        changed_traceback if mutation == 'traceback' else original_traceback
    )
    assert error.__context__ is (
        changed if mutation == 'context' else original_context
    )
    assert error.__cause__ is original_cause
    assert error.__suppress_context__ is original_suppression


def test_source_binding_uses_the_original_dtype_and_does_not_infer_aliases():
    dtype = pd.StringDtype(storage='python')
    columns = {'input_users': {'id': dtype}}
    held = NativeFrameColumns(columns)
    columns.clear()
    binding = {'source': {'source': ['input_users']}, 'column': 'id'}
    assert held.dtype(binding) is dtype
    native = held.prediction(binding, ['a', None])
    assert isinstance(native, NativeColumn)
    assert native.values.dtype == dtype
    assert native.values.iloc[0] == 'a'
    assert native.values.iloc[1] is pd.NA
    with pytest.raises(KeyError):
        held.dtype({'source': {'source': ['users']}, 'column': 'id'})


def test_whole_text_conversion_does_not_lower_split_or_cap():
    value = 'CAPITAL ' + 'x' * 301
    selected = pd.Series([value, None], dtype=pd.StringDtype(storage='python'))
    converted = convert_text(selected)
    assert isinstance(converted, NativeColumn)
    assert converted.values.iloc[0] == value
    assert converted.values.iloc[1] == ''
    assert selected.iloc[1] is pd.NA


def test_a_dotted_source_remains_one_relation_part():
    dtype = pd.StringDtype(storage='python')
    columns = NativeFrameColumns({'users.snapshot': {'id': dtype}})
    assert (
        columns.dtype(
            {'source': {'source': ['users.snapshot']}, 'column': 'id'}
        )
        is dtype
    )
    with pytest.raises(ValueError):
        columns.dtype(
            {'source': {'source': ['users', 'snapshot']}, 'column': 'id'}
        )


def test_actual_native_strings_export_losslessly_without_coercion_or_pair_merge():
    values = pd.Series(['ordinary', '\ud83d\ude3a', '😺', None], dtype=object)
    native = NativeColumn(values)
    rows, encoding = native.arrow()
    assert encoding == 'utf8_surrogatepass'
    assert rows.column(0).to_pylist() == [
        b'ordinary',
        b'\xed\xa0\xbd\xed\xb8\xba',
        '😺'.encode(),
        None,
    ]
    assert native.values is values


def test_prediction_retains_the_actual_whole_constructor_exception():
    failure = TypeError('native value cannot stringify')

    class BadString:
        def __str__(self):
            raise failure

    columns = NativeFrameColumns(
        {'input_users': {'id': pd.StringDtype('python')}}
    )
    converted = columns.prediction(
        {'source': {'source': ['input_users']}, 'column': 'id'}, [BadString()]
    )
    assert isinstance(converted, NativeFrameFailure)
    assert converted.error is failure
    assert converted.facts.ancestry == ('type_error',)
