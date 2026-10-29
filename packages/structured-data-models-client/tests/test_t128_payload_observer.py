# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Payload byte observers consume only actually returned typed attachments."""

import json

import pytest

from sdm_client._rust import results
from sdm_client._rust.run import Finished


class Run:
    def __init__(self, values):
        self.values = dict(values)
        self.takes = []

    def take_attachment(self, key):
        self.takes.append(key)
        return self.values.pop(key)


def materialized(payload=3):
    return {
        'state': {'version': 1, 'payload': payload, 'query': 4},
        'query': None,
    }


@pytest.mark.parametrize('prepared', [False, True])
def test_actual_payload_bytes_are_observed_once_in_their_producing_run(
    monkeypatch, prepared
):
    from payload_attachment_observer import record_payload_bytes

    payload = b'{ "number" : 1.00, "text" : "actual bytes" }'
    data = materialized()
    handle = Run(
        {
            1: json.dumps({'value': 'request', 'data': data}).encode(),
            3: payload,
            4: b'"query"',
            9: b'not a payload',
        }
    )
    terminal = Finished(
        results.MaterializePrepared(1)
        if prepared
        else results.Gathered.from_contract(data),
        handle,
    )
    taken = []
    record_payload_bytes(monkeypatch, taken)
    if prepared:
        actual = terminal.take(1)
        assert actual == json.dumps({'value': 'request', 'data': data}).encode()
    assert terminal.take(9) == b'not a payload'
    assert taken == []
    assert terminal.take(3) is payload
    assert terminal.take(4) == b'"query"'
    assert taken == [payload] and taken[0] is payload
    assert handle.takes == ([1] if prepared else []) + [9, 3, 4]


def test_attachment_numbers_from_another_run_do_not_supply_payload_authority(
    monkeypatch,
):
    from payload_attachment_observer import record_payload_bytes

    taken = []
    record_payload_bytes(monkeypatch, taken)
    data = materialized()
    first_run = Run(
        {
            1: json.dumps({'value': 'request', 'data': data}).encode(),
            3: b'first',
        }
    )
    second_run = Run({3: b'unrelated'})
    first = Finished(results.MaterializePrepared(1), first_run)
    second = Finished(results.MaterializePrepared(1), second_run)
    first.take(1)
    assert second.take(3) == b'unrelated'
    assert taken == []
    assert first.take(3) == b'first'
    assert taken == [b'first']


def test_nonrequest_preparation_values_never_name_payload_bytes(monkeypatch):
    from payload_attachment_observer import record_payload_bytes

    taken = []
    record_payload_bytes(monkeypatch, taken)
    raw = json.dumps({'value': 'source', 'data': {'payload': 3}}).encode()
    handle = Run({1: raw, 3: b'not a payload'})
    terminal = Finished(results.MaterializePrepared(1), handle)
    assert terminal.take(1) is raw
    assert terminal.take(3) == b'not a payload'
    assert taken == [] and handle.takes == [1, 3]
