# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native failure presentation retains the actual cause's current context.

The public later-Text conversion journey reproduces the extra internal
context node. These controls exercise both actual core-selected wrapper
branches while the facade's intermediate error handler is active.
"""

from __future__ import annotations

import sys

import pytest

from sdm_client._rust.frame_native import NativeFrameFailure, guard_failure
from sdm_client.errors import StructuredDataError


def capture(error):
    try:
        raise error
    except Exception as caught:
        return NativeFrameFailure.capture(caught)


def present(held, placeholder, *, part=None, origin='engine'):
    try:
        raise placeholder
    except StructuredDataError:
        guard_failure(held, origin=origin, part=part)


@pytest.mark.parametrize('branch', ['selected', 'carried'])
@pytest.mark.parametrize('ambient', [False, True])
@pytest.mark.parametrize('display_edit', [False, True])
@pytest.mark.parametrize('suppressed', [False, True])
def test_final_wrapper_keeps_current_native_context_and_original_cause(
    branch, ambient, display_edit, suppressed
):
    caller = LookupError('actual outer caller')
    changed = KeyError('context changed by native presentation')
    original_cause = RuntimeError('actual native cause')
    placeholder = StructuredDataError(
        'native frame text conversion failed', code='INVALID_REQUEST'
    )
    events = []

    class Observed(ValueError):
        def __str__(self):
            events.append('native.str')
            if display_edit:
                self.__context__ = changed
            return 'frame conversion refused'

    class NativeStructured(StructuredDataError):
        pass

    class ObservedText(str):
        def rstrip(self, *args, **kwargs):
            events.append('native.rstrip')
            if display_edit:
                native.__context__ = changed
            return super().rstrip(*args, **kwargs)

    if branch == 'selected':
        native = Observed()
        part = None
    else:
        native = NativeStructured(
            ObservedText('native declined'),
            code='INVALID_CONFIGURATION',
            details={'window_outran_the_data': False, 'retained': ['native']},
        )
        part = {'kind': 'forecast', 'position': 2, 'parts': 3}
    native.__cause__ = original_cause
    native.__suppress_context__ = suppressed

    def exercise():
        held = capture(native)
        assert held.error is native
        assert held.context is (caller if ambient else None)
        with pytest.raises(StructuredDataError) as raised:
            present(held, placeholder, part=part)
        return raised.value

    if ambient:
        try:
            raise caller
        except LookupError:
            selected = exercise()
    else:
        selected = exercise()

    assert selected is not native and selected is not placeholder
    assert selected.__cause__ is native
    assert selected.__context__ is native
    assert selected.__suppress_context__ is True
    assert native.__cause__ is original_cause
    assert native.__suppress_context__ is suppressed
    assert native.__context__ is (
        changed if display_edit else caller if ambient else None
    )
    deepest = native.__traceback__
    while deepest.tb_next is not None:
        deepest = deepest.tb_next
    assert deepest.tb_frame.f_code.co_name == 'capture'
    if branch == 'selected':
        assert type(selected) is StructuredDataError
        assert selected.code == 'INVALID_REQUEST'
        assert selected.message == 'frame conversion refused'
        assert events == ['native.str', 'native.str']
    else:
        assert type(selected) is NativeStructured
        assert selected.code == 'INVALID_CONFIGURATION'
        assert selected.message == (
            'native declined. This is forecast step 2 of 3.'
        )
        assert selected.details is native.details
        assert native.args == ('native declined',)
        assert events == ['native.rstrip']


@pytest.mark.parametrize('ambient', [False, True])
@pytest.mark.parametrize('display_edit', [False, True])
def test_native_display_failure_escapes_with_its_actual_context(
    ambient, display_edit
):
    caller = LookupError('actual outer caller')
    changed = KeyError('native display changed its own context')
    display = RuntimeError('actual native display failed')
    placeholder = StructuredDataError(
        'private intermediate', code='INVALID_REQUEST'
    )

    class Unprintable(ValueError):
        def __str__(self):
            if display_edit:
                self.__context__ = changed
            raise display

    native = Unprintable()

    def exercise():
        held = capture(native)
        with pytest.raises(RuntimeError) as raised:
            present(held, placeholder)
        assert raised.value is display

    if ambient:
        try:
            raise caller
        except LookupError:
            exercise()
    else:
        exercise()
    assert display.__context__ is native
    assert display.__cause__ is None
    assert native.__context__ is (
        changed if display_edit else caller if ambient else None
    )
    assert native.__cause__ is None
    assert native.__suppress_context__ is False


@pytest.mark.parametrize('ambient', [False, True])
@pytest.mark.parametrize('structured', [False, True])
def test_native_passthrough_keeps_identity_and_complete_original_chain(
    ambient, structured
):
    caller = LookupError('actual outer caller')
    cause = KeyError('actual native cause')
    native = (
        StructuredDataError('native refusal', code='INVALID_CONFIGURATION')
        if structured
        else RuntimeError('native unguarded failure')
    )
    native.__cause__ = cause
    native.__suppress_context__ = False
    placeholder = StructuredDataError(
        'private intermediate', code='INVALID_REQUEST'
    )

    def exercise():
        held = capture(native)
        with pytest.raises(type(native)) as raised:
            present(
                held,
                placeholder,
                origin='engine' if structured else 'unguarded',
            )
        assert raised.value is native

    if ambient:
        try:
            raise caller
        except LookupError:
            exercise()
    else:
        exercise()
    assert native.__context__ is (caller if ambient else None)
    assert native.__cause__ is cause
    assert native.__suppress_context__ is False


@pytest.mark.parametrize('getter_fails', [False, True])
def test_carried_cause_getter_keeps_edits_made_during_final_presentation(
    getter_fails,
):
    changed = KeyError('context edited by final native getter')
    changed_cause = LookupError('cause edited by final native getter')
    getter_failure = RuntimeError('actual final native getter failed')
    placeholder = StructuredDataError(
        'private intermediate', code='INVALID_REQUEST'
    )
    try:
        raise KeyError('traceback selected by final native getter')
    except KeyError as error:
        changed_traceback = error.__traceback__
    observed = []

    class NativeStructured(StructuredDataError):
        def __getattribute__(self, name):
            if name == '__cause__':
                observed.append((self, sys.exc_info()[1]))
                native.__context__ = changed
                native.__cause__ = changed_cause
                native.__suppress_context__ = True
                native.__traceback__ = changed_traceback
                if getter_fails:
                    raise getter_failure
            return BaseException.__getattribute__(self, name)

    native = NativeStructured('native declined', code='INVALID_CONFIGURATION')
    held = capture(native)
    assert held.context is None
    with pytest.raises(
        RuntimeError if getter_fails else NativeStructured
    ) as raised:
        present(
            held,
            placeholder,
            part={'kind': 'forecast', 'position': 2, 'parts': 3},
        )
    assert len(observed) == 1
    wrapper, handled = observed[0]
    assert wrapper is not native and wrapper is not placeholder
    assert handled is native
    # Inspect through the base descriptor so assertions do not invoke the
    # native getter a second time and conceal an overwritten context.
    assert BaseException.__getattribute__(wrapper, '__cause__') is native
    assert BaseException.__getattribute__(native, '__context__') is changed
    assert BaseException.__getattribute__(native, '__cause__') is changed_cause
    assert native.__suppress_context__ is True
    deepest = native.__traceback__
    while deepest.tb_next is not None:
        deepest = deepest.tb_next
    assert deepest is changed_traceback
    if getter_fails:
        assert raised.value is getter_failure
        assert getter_failure.__context__ is native
        assert getter_failure.__cause__ is None
    else:
        assert raised.value is wrapper
        assert (
            wrapper.message == 'native declined. This is forecast step 2 of 3.'
        )
        assert wrapper.__context__ is native
        assert wrapper.__suppress_context__ is True
