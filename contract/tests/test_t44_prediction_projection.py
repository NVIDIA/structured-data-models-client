# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-only column projection never alters actual call evidence."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_t136_forecast_capture import RUNNER, _module
from test_t44_prediction_observation import (
    actual_observation,
    legacy_observation,
    reader,
    two_observed_predictions,
)

ROOT = Path(__file__).parents[2]
COMPARE = _module(
    't44_projection_compare', ROOT / 'scripts/rust_core/compare.py'
)
CASE = {'id': 'projection-owned-calls', 'until': 'predict'}
NAMED = {'items': frozenset({'i.d'})}


def related(raw, *, kept=True, other=False):
    columns, row = ['kind'], ['ordinary']
    if kept:
        columns.insert(0, 'i.d')
        row.insert(0, 7)
    if other:
        columns.append('not-approved')
        row.append(9)
    raw['context']['related_tables'] = {
        'items': {'columns': columns, 'rows': [row]}
    }
    return raw


def encoded_call(raw):
    encoded = json.dumps(raw).encode()
    return encoded, RUNNER._payload(encoded)['parsed']


def fixture(producer, count=1, *, other=False):
    if producer == 'gathered':
        core, operations = (
            actual_observation() if count == 1 else two_observed_predictions()
        )
        originals, reference = [], []
        for entry in core['predictions']:
            [call] = entry['calls']
            raw = json.loads(base64.b64decode(call['input_body_base64']))
            reference.append(
                encoded_call(related(copy.deepcopy(raw), kept=False))[1]
            )
            encoded, body = encoded_call(related(raw, other=other))
            originals.append(encoded)
            call.update(
                body=body,
                body_sha256=hashlib.sha256(encoded).hexdigest(),
                input_body_base64=base64.b64encode(encoded).decode('ascii'),
                input_body_sha256=hashlib.sha256(encoded).hexdigest(),
            )
        core['predicted']['body'] = copy.deepcopy(
            core['predictions'][-1]['calls'][-1]['body']
        )
    else:
        core, operations = legacy_observation()
        seed, _ = actual_observation()
        raw = json.loads(
            base64.b64decode(
                seed['predictions'][0]['calls'][0]['input_body_base64']
            )
        )
        originals, reference, calls = [], [], []
        for at in range(count):
            item = copy.deepcopy(raw)
            item['metadata']['request_id'] = f'legacy-call-{at}'
            reference.append(
                encoded_call(related(copy.deepcopy(item), kept=False))[1]
            )
            encoded, body = encoded_call(related(item, other=other))
            originals.append(encoded)
            calls.append(
                {
                    'run': 2,
                    'body': body,
                    'body_sha256': hashlib.sha256(encoded).hexdigest(),
                    'reply': {'outcome': 'response', 'status': 200},
                    'model_endpoint': None,
                }
            )
        core['model_calls'] = calls
        core['predicted']['body'] = copy.deepcopy(calls[-1]['body'])
        if count > 1:
            core['predicted']['calls'] = [
                copy.deepcopy(call['body']) for call in calls
            ]
        core['legacy_predictions'] = [
            {
                'run': 2,
                'completion': 'done',
                'result': copy.deepcopy(core['predicted']),
                'error': None,
                'answered': None,
            }
        ]
    checked = reader().view(core, operations)
    assert len(checked['calls']) == count
    received = [
        {
            'method': 'POST',
            'path': '/v1/predictions',
            'parsed': copy.deepcopy(body),
            'sha256': hashlib.sha256(encoded).hexdigest(),
        }
        for body, encoded in zip(checked['calls'], originals, strict=True)
    ]
    observed = {
        'identity': {'operations': operations, 'contract': 35},
        'core': core,
        'public': {'payloads': received},
    }
    return observed, {
        'public': {'payloads': [{'parsed': body} for body in reference]}
    }


def payload_failures(observation, reference, who, *, named=True):
    report = COMPARE.Report(paths=(('metadata', 'request_id'),))
    if named:
        report.core_only = {CASE['id']: NAMED}
    COMPARE._core_payload(report, CASE, who, observation, reference)
    return report.failures


@pytest.mark.parametrize('producer', ['gathered', 'legacy'])
@pytest.mark.parametrize('count', [1, 2])
def test_checked_original_calls_are_projected_only_against_reference(
    producer, count
):
    observed, reference = fixture(producer, count)
    untouched = copy.deepcopy((observed, reference))
    assert payload_failures(observed, reference, 'rust-host', named=False)
    assert payload_failures(observed, reference, 'rust-host') == []
    assert payload_failures(observed, reference, 'python-core') == []
    reader().view(observed['core'], observed['identity']['operations'])
    assert (observed, reference) == untouched


@pytest.mark.parametrize('producer', ['gathered', 'legacy'])
@pytest.mark.parametrize('count', [1, 2])
def test_unnamed_column_still_fails_in_each_checked_producer(producer, count):
    observed, reference = fixture(producer, count, other=True)
    assert payload_failures(observed, reference, 'rust-host')
    # The receipt is still its own actual attempted body, irrespective of
    # the narrower reference-side column exception.
    assert payload_failures(observed, reference, 'python-core') == []


@pytest.mark.parametrize('producer', ['gathered', 'legacy'])
@pytest.mark.parametrize('count', [1, 2])
@pytest.mark.parametrize('changed', ['body', 'hash', 'missing', 'order'])
def test_approved_projection_never_excuses_changed_actual_receipts(
    producer, count, changed
):
    observed, reference = fixture(producer, count)
    received = observed['public']['payloads']
    if changed == 'body':
        # Even an approved column's actual value must match this call's
        # independent endpoint receipt before reference projection.
        received[0]['parsed']['context']['related_tables']['items']['rows'][0][
            0
        ] = {'number': '700'}
    elif changed == 'hash':
        received[0]['sha256'] = '0' * 64
    elif changed == 'missing':
        received.pop(0)
    elif count == 1:
        received.append(copy.deepcopy(received[0]))
        received[1]['sha256'] = 'f' * 64
    else:
        received.reverse()
    assert payload_failures(observed, reference, 'python-core')


@pytest.mark.parametrize('producer', ['gathered', 'legacy'])
def test_raw_call_tamper_cannot_be_hidden_by_a_named_column(producer):
    observed, reference = fixture(producer)
    # A changed receipt is tested separately. Here the attempt itself no
    # longer matches the unmodified, independently recorded endpoint body.
    core = observed['core']
    call = (
        core['predictions'][0]['calls'][0]
        if producer == 'gathered'
        else core['model_calls'][0]
    )
    call['body_sha256'] = 'f' * 64
    assert payload_failures(observed, reference, 'python-core')
