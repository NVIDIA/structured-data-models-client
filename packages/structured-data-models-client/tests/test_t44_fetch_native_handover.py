# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native fetch failures retain their owner across the actual run boundary.

The public schema journey uses the real core and source selection.
Scripted multi-request runs additionally expose cancellation, reply and memo
ordering that the public API intentionally hides.
"""

from __future__ import annotations

import json
import warnings
import weakref
from types import SimpleNamespace
from typing import Any

import duckdb
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import Schema, _routing
from sdm_client._rust import providers
from sdm_client._rust import run as run_module


class _UnreadError(Exception):
    """Any provider diagnostic read would replace the intended failure."""

    def __str__(self) -> str:
        raise AssertionError('native failure was formatted')

    def __getattribute__(self, name: str) -> Any:
        if name in {
            'sqlstate',
            'pgcode',
            'sql_state',
            'vendor_code',
            'errno',
            'retryable',
            'details',
            '_answered',
        }:
            raise AssertionError(f'native failure was inspected: {name}')
        return super().__getattribute__(name)


def _native_error() -> _UnreadError:
    try:
        raise LookupError('original conversion context')
    except LookupError:
        try:
            raise _UnreadError() from ValueError('original explicit cause')
        except _UnreadError as error:
            return error


def test_public_schema_propagates_the_actual_fetch_failure(monkeypatch):
    """A returned native failure never becomes a warehouse/CoreError reply."""
    error = _native_error()
    original_cause = error.__cause__
    original_context = error.__context__
    original_traceback = error.__traceback__
    calls = []

    class NativeProvider:
        def __init__(self, connection):
            pass

        def __call__(self, step, keys=None):
            calls.append(step)
            return run_module.FetchRefused(error)

        def error_fields(self, failure):
            raise AssertionError('native failure reached provider diagnostics')

    monkeypatch.setenv(
        _routing.ENVIRONMENT, ','.join(_routing.FROM_SOURCE_DUCKDB_TABLES)
    )
    monkeypatch.setattr(providers, 'DuckDBProvider', NativeProvider)
    with duckdb.connect() as connection:
        connection.execute('CREATE TABLE items (id BIGINT)')
        with pytest.raises(_UnreadError) as caught:
            Schema.from_source(DuckDBSource(connection), {'items': 'items'})

    assert caught.value is error
    assert len(calls) == 1
    assert error.__cause__ is original_cause
    assert error.__context__ is original_context
    assert error.__suppress_context__ is True
    trace = error.__traceback__
    while trace is not None and trace is not original_traceback:
        trace = trace.tb_next
    assert trace is original_traceback


class _CoreError(Exception):
    pass


class _Handle:
    def __init__(self, events, *, memo=False):
        self.events = events
        self.memo = memo
        self.started = False
        self.replies = []

    def next(self):
        if self.started:
            return json.dumps(
                {
                    'message': 'step',
                    'step': 'done',
                    'result': {'result': 'test'},
                }
            ).encode(), {}
        self.started = True
        return json.dumps(
            {
                'message': 'step',
                'step': 'fetch',
                'requests': [
                    {'id': i, 'request': {}, 'plan': {'statements': []}}
                    for i in (1, 2)
                ],
            }
        ).encode(), {}

    def take_warnings(self):
        return None

    def reply(self, control, attached):
        value = json.loads(control)
        self.events.append(('reply', value['id']))
        self.replies.append((value, attached))

    def cancel(self):
        self.events.append('cancel')

    def take_sampler_state(self):
        self.events.append('memo')
        if not self.memo:
            return None
        return json.dumps(
            {'message': 'sampler_state', 'state': {'version': 7, 'memo': 9}}
        ).encode(), {}

    def take_attachment(self, attachment):
        assert attachment == 9
        return b'current opaque memo'


def _steps(handle, provider, model=None):
    return run_module._steps(
        SimpleNamespace(CoreError=_CoreError),
        handle,
        provider,
        model,
        frozenset(),
        None,
    )


@pytest.mark.parametrize('warning_failure', [False, True])
@pytest.mark.parametrize('cancel_failure', [False, True])
def test_native_failure_cancels_before_reply_or_another_request(
    monkeypatch,
    warning_failure,
    cancel_failure,
):
    events = []
    error = _native_error()
    handle = _Handle(events)
    original_context = error.__context__

    if cancel_failure:

        def cancel():
            events.append('cancel')
            raise _CoreError('cancellation failed')

        monkeypatch.setattr(handle, 'cancel', cancel)

    def fail_warning(*args, **kwargs):
        raise error

    def model(body):
        raise AssertionError('native fetch failure reached a model call')

    class NativeProvider:
        def __call__(self, request, keys):
            events.append(('fetch', request['id']))
            if warning_failure:
                with warnings.catch_warnings():
                    warnings.simplefilter('always')
                    try:
                        warnings.warn(
                            'native conversion warning',
                            UserWarning,
                            stacklevel=2,
                        )
                    except Exception as actual:
                        return run_module.FetchRefused(actual)
                raise AssertionError('warning handler did not fail')
            return run_module.FetchRefused(error)

        def error_fields(self, failure):
            raise AssertionError('native failure reached provider diagnostics')

    monkeypatch.setattr(warnings, 'showwarning', fail_warning)
    with pytest.raises(_UnreadError) as caught:
        _steps(handle, NativeProvider(), model)
    assert caught.value is error
    assert error.__context__ is original_context
    assert events == [('fetch', 1), 'cancel']
    assert handle.replies == []
    # No answered/carry metadata is attached to the native exception.
    assert 'details' not in vars(error)
    assert '_answered' not in vars(error)


def test_actual_io_error_keeps_the_existing_provider_reply():
    events = []
    handle = _Handle(events)
    failure = RuntimeError('driver failed')
    failure.sqlstate = '08006'
    failure.retryable = True
    table = object()

    class IoProvider:
        def __call__(self, request, keys):
            if request['id'] == 1:
                raise failure
            return table

        def error_fields(self, error):
            assert error is failure
            events.append('driver fields')
            return {'vendor_code': 'driver-code'}

    finished = _steps(handle, IoProvider())
    assert finished.result == {'result': 'test'}
    assert events == ['driver fields', ('reply', 1), ('reply', 2)]
    assert handle.replies == [
        (
            {
                'message': 'fetch_reply',
                'id': 1,
                'outcome': {
                    'outcome': 'error',
                    'error': {
                        'kind': 'data_source',
                        'retryable': True,
                        'sqlstate': '08006',
                        'vendor_code': 'driver-code',
                        'driver_message': 'driver failed',
                    },
                },
            },
            {},
        ),
        (
            {
                'message': 'fetch_reply',
                'id': 2,
                'outcome': {'outcome': 'data', 'data': 1},
            },
            {1: table},
        ),
    ]


def test_ordinary_pages_still_attach_data_and_continuation():
    events = []
    handle = _Handle(events)
    first, last = object(), object()

    def provider(request, keys):
        if request['id'] == 1:
            return run_module.Page(first, 'next opaque page')
        return last

    _steps(handle, provider)
    assert events == [('reply', 1), ('reply', 2)]
    assert handle.replies == [
        (
            {
                'message': 'fetch_reply',
                'id': 1,
                'outcome': {
                    'outcome': 'data',
                    'data': 1,
                    'continuation': 'next opaque page',
                },
            },
            {1: first},
        ),
        (
            {
                'message': 'fetch_reply',
                'id': 2,
                'outcome': {'outcome': 'data', 'data': 1},
            },
            {1: last},
        ),
    ]


@pytest.mark.parametrize('failure', [KeyboardInterrupt(), SystemExit(3)])
def test_process_control_exceptions_are_not_native_or_io_outcomes(failure):
    events = []
    handle = _Handle(events)

    def provider(request, keys):
        raise failure

    with pytest.raises(type(failure)) as caught:
        _steps(handle, provider)
    assert caught.value is failure
    assert handle.replies == []


@pytest.mark.parametrize('failure', [KeyboardInterrupt(), SystemExit(3)])
def test_process_control_during_cancellation_is_not_masked(
    monkeypatch, failure
):
    events = []
    handle = _Handle(events)
    native = _native_error()
    original_context = native.__context__

    def provider(request, keys):
        return run_module.FetchRefused(native)

    def cancel():
        events.append('cancel')
        raise failure

    monkeypatch.setattr(handle, 'cancel', cancel)
    with pytest.raises(type(failure)) as caught:
        _steps(handle, provider)
    assert caught.value is failure
    assert events == ['cancel']
    assert handle.replies == []
    assert native.__context__ is original_context


@pytest.mark.parametrize('source_batch', [False, True])
def test_refusal_releases_the_owned_run_after_publishing_its_memo(
    monkeypatch, source_batch
):
    events = []
    handles = []
    error = _native_error()

    class Handle(_Handle):
        def next(self):
            control, attached = super().next()
            if source_batch:
                value = json.loads(control)
                for ordinal, request in enumerate(value.get('requests', [])):
                    request.update(
                        purpose='source_rows',
                        partition={'first': 1, 'ordinal': ordinal, 'parts': 2},
                    )
                control = json.dumps(value).encode()
            return control, attached

    def start(control, attached):
        handle = Handle(events, memo=True)
        handles.append(weakref.ref(handle))
        return handle

    core = SimpleNamespace(
        CONTRACT_VERSION=35,
        CoreError=_CoreError,
        reset_log_cache=lambda: None,
        Run=SimpleNamespace(start=start),
    )
    monkeypatch.setattr(run_module, 'native', lambda: core)
    monkeypatch.setattr(run_module.facts, 'host_facts', lambda core: {})

    def provider(request, keys):
        return run_module.FetchRefused(error)

    state = {}
    with pytest.raises(_UnreadError) as caught:
        run_module.run(
            {'op': 'materialize', 'provider': {'provider': 'frames'}},
            {},
            provider=provider,
            source_provider=(
                lambda requests, keys: run_module.FetchRefused(error)
            )
            if source_batch
            else None,
            state=state,
        )
    assert caught.value is error
    assert events == ['cancel', 'memo']
    assert state == {'held': {'version': 7, 'memo': b'current opaque memo'}}
    assert len(handles) == 1
    assert handles[0]() is None


@pytest.mark.parametrize('failure', [KeyboardInterrupt(), SystemExit(3), None])
def test_refusal_holds_only_an_actual_exception(failure):
    with pytest.raises(TypeError):
        run_module.FetchRefused(failure)
