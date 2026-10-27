# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public guard twins with explicit native conversion fault injection.

These exercise actual public materialize calls. The injected native failures
isolate exception observation effects; they are not frozen builtin outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings

import pandas as pd
import pytest

from sdm_client import StructuredDataError

from test_t44_text_frames import core as core, question, shop


def inject(monkeypatch, error, events=None):
    original = pd.Series.astype
    reached = []

    def injected(values, dtype, *args, **kwargs):
        if (
            dtype is str
            and values.name == 'note'
            and sys._getframe(1).f_code.co_name
            in {'_normalize_text', 'convert_text'}
        ):
            reached.append(values.name)
            if events is not None:
                events.append('note.astype')
            raise error
        return original(values, dtype, *args, **kwargs)

    monkeypatch.setattr(pd.Series, 'astype', injected)
    return reached


@pytest.mark.parametrize(
    'case', ['actual-base', 'actual-subclass', 'forged-name']
)
def test_native_relational_error_uses_actual_class_identity(
    core, monkeypatch, case
):
    library = pytest.importorskip('sdm_relational_engine')
    canonical = library.KumoRelationalError
    query = question(shop('python'), prose=True)
    events = []

    def displayed(self):
        events.append('native.str')
        return 'actual native relational failure'

    if case == 'actual-base':
        kind = canonical
    elif case == 'actual-subclass':
        kind = type('ObservedRelational', (canonical,), {'__str__': displayed})
    else:
        kind = type(
            'KumoRelationalError',
            (Exception,),
            {'__module__': canonical.__module__, '__str__': displayed},
        )
    native = kind('actual native relational failure')
    reached = inject(monkeypatch, native, events)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        with pytest.raises(StructuredDataError) as raised:
            query.materialize(indices=[7, 8], rows_per_call=1, random_seed=7)
    error = raised.value
    observed = {
        'case': case,
        'outcome': 'error',
        'type': type(error).__name__,
        'module': type(error).__module__,
        'code': error.code,
        'message': str(error),
        'details': error.details,
        'cause_is_native': error.__cause__ is native,
        'cause_type': type(error.__cause__).__name__,
        'warnings': [
            {'type': type(w.message).__name__, 'message': str(w.message)}
            for w in notices
        ],
        'events': events,
    }
    expected = json.loads(
        Path(__file__)
        .with_name('t44_native_error_identity_reference.json')
        .read_text()
    )['records'][case]
    assert reached == ['note']
    assert observed == expected


@pytest.mark.parametrize(
    ('base', 'messages', 'code', 'final'),
    [
        (
            ValueError,
            ['aggregation time range', 'native second'],
            'INVALID_REQUEST',
            'native second',
        ),
        (
            TypeError,
            ['ordinary native value', 'aggregation time range', 'native third'],
            'INVALID_REQUEST',
            'native third',
        ),
        (
            TypeError,
            ['unexpected keyword argument', 'native second'],
            'INTERNAL_ERROR',
            'the relational engine was called incorrectly by this client: Observed: native second',
        ),
    ],
)
def test_public_guard_observes_signature_refusal_and_said_at_distinct_points(
    core, monkeypatch, base, messages, code, final
):
    query = question(shop('python'), prose=True)

    class Observed(base):
        reads = 0

        def __str__(self):
            value = messages[self.reads]
            self.reads += 1
            return value

    native = Observed()
    reached = inject(monkeypatch, native)
    with pytest.raises(StructuredDataError) as raised:
        query.materialize(indices=[7, 8], rows_per_call=1, random_seed=7)
    assert native.reads == len(messages)
    assert raised.value.code == code
    assert raised.value.message == final
    assert raised.value.__cause__ is native
    assert reached == ['note']
    deepest = native.__traceback__
    while deepest.tb_next is not None:
        deepest = deepest.tb_next
    assert deepest.tb_frame.f_code.co_name == 'injected'
    if code == 'INVALID_REQUEST':
        assert raised.value.details['window_outran_the_data'] is True


def test_public_guard_preserves_structured_error_without_native_display(
    core, monkeypatch
):
    query = question(shop('python'), prose=True)

    class Preserved(StructuredDataError):
        def __str__(self):
            raise AssertionError('structured native error was displayed')

    native = Preserved('native structured error', code='INVALID_REQUEST')
    reached = inject(monkeypatch, native)
    with pytest.raises(Preserved) as raised:
        query.materialize(indices=[7, 8], rows_per_call=1, random_seed=7)
    assert raised.value is native
    assert reached == ['note']


def test_public_native_message_failure_escapes_with_original_exception_context(
    core, monkeypatch
):
    query = question(shop('python'), prose=True)
    display = RuntimeError('native display failed')

    class Unprintable(TypeError):
        def __str__(self):
            raise display

    native = Unprintable()
    inject(monkeypatch, native)
    with pytest.raises(RuntimeError) as raised:
        query.materialize(indices=[7, 8], rows_per_call=1, random_seed=7)
    assert raised.value is display
    assert display.__context__ is native
    assert display.__cause__ is None
    deepest = native.__traceback__
    while deepest.tb_next is not None:
        deepest = deepest.tb_next
    assert deepest.tb_frame.f_code.co_name == 'injected'
