# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The actual source batch I/O port preserves complete issued partitions."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pyarrow as pa
import pytest
from test_t44_fetch_native_handover import _CoreError, _Handle, _native_error

from sdm_client._rust import codec
from sdm_client._rust import run as run_module


def _requests():
    return [
        {
            'id': number,
            'request': {},
            'plan': {'statements': []},
            'purpose': 'source_rows',
            'partition': {'first': 2, 'ordinal': number - 2, 'parts': 2},
        }
        for number in (2, 3)
    ]


class Handle(_Handle):
    def __init__(self, events, requests):
        super().__init__(events)
        self.requests = requests

    def next(self):
        if self.started:
            return super().next()
        self.started = True
        return json.dumps(
            {'message': 'step', 'step': 'fetch', 'requests': self.requests}
        ).encode(), {}


def _steps(handle, provider, source):
    return run_module._steps(
        SimpleNamespace(CoreError=_CoreError),
        handle,
        provider,
        None,
        frozenset(),
        None,
        source_provider=source,
    )


def test_source_success_is_answered_only_after_the_complete_actual_partition():
    events = []
    requested = _requests()
    before = copy.deepcopy(requested)
    handle = Handle(
        events,
        requested + [{'id': 9, 'request': {}, 'plan': {'statements': []}}],
    )
    first, second, ordinary = [
        pa.table({'value': [value]}) for value in (1, 2, 9)
    ]

    def batch(requests, keys):
        events.append(('batch', [request['id'] for request in requests]))
        assert requests == before
        assert keys == {2: {}, 3: {}}
        return {2: first, 3: second}

    def provider(request, keys):
        events.append(('ordinary', request['id']))
        return ordinary

    _steps(handle, provider, batch)
    assert events == [
        ('batch', [2, 3]),
        ('reply', 2),
        ('reply', 3),
        ('ordinary', 9),
        ('reply', 9),
    ]
    assert [attached[1] for _, attached in handle.replies] == [
        first,
        second,
        ordinary,
    ]
    assert requested == before


def test_failed_source_partition_cancels_without_partial_replies_or_later_io():
    events = []
    handle = Handle(
        events,
        _requests() + [{'id': 9, 'request': {}, 'plan': {'statements': []}}],
    )
    error = _native_error()
    context = error.__context__

    def batch(requests, keys):
        events.append(('batch', [request['id'] for request in requests]))
        return run_module.FetchRefused(error)

    def provider(*args):
        pytest.fail('the selected source failure must prevent later I/O')

    with pytest.raises(type(error)) as caught:
        _steps(handle, provider, batch)
    assert caught.value is error
    assert error.__context__ is context
    assert events == [('batch', [2, 3]), 'cancel']
    assert handle.replies == []


@pytest.mark.parametrize(
    'change', ['short', 'ordinal', 'first', 'count', 'purpose']
)
def test_malformed_partition_never_reaches_native_io(change):
    events = []
    requests = _requests()
    if change == 'short':
        requests.pop()
    elif change == 'purpose':
        requests[1]['purpose'] = 'describe_result'
    else:
        requests[1]['partition'][
            {'ordinal': 'ordinal', 'first': 'first', 'count': 'parts'}[change]
        ] += 1
    handle = Handle(events, requests)

    def forbidden(*args):
        pytest.fail('an incomplete or foreign source partition reached I/O')

    with pytest.raises(codec.CodecError):
        _steps(handle, forbidden, forbidden)
    assert handle.replies == []
    assert events == ['cancel']


def test_representation_metadata_keeps_the_ordinary_provider_boundary():
    events = []
    requests = _requests()
    for request in requests:
        request['purpose'] = 'describe_result'
    handle = Handle(events, requests)
    returned = pa.table({'name': ['value']})

    def provider(request, keys):
        events.append(('metadata', request['id']))
        return returned

    def forbidden(*args):
        pytest.fail('metadata must not construct a native source column')

    _steps(handle, provider, forbidden)
    assert events == [
        ('metadata', 2),
        ('reply', 2),
        ('metadata', 3),
        ('reply', 3),
    ]
