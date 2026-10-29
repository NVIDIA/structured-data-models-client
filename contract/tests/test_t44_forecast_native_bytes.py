# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Current forecast byte authority is actual taken and received native bytes."""

import base64
import copy
import hashlib

import pytest
from test_t136_forecast import R
from test_t44_prediction_observation import actual_observation


def sample():
    core, operations = actual_observation()
    call = core['predictions'][0]['calls'][0]
    raw = base64.b64decode(call['input_body_base64'])
    return {
        'payloads': [
            {
                'method': 'POST',
                'path': '/predict',
                'masked_sha256': hashlib.sha256(
                    R.masked_bytes(raw)
                ).hexdigest(),
                'digest': R.EN.digest(call['body']),
                'tables': R.EN.tables_with_rows(call['body']),
            }
        ],
        'native_evidence': {
            'core': core,
            'operations': operations,
            'receipts': [
                {
                    'method': 'POST',
                    'path': '/predict',
                    'body_base64': call['input_body_base64'],
                }
            ],
        },
    }


def test_native_bytes_are_conserved_independently_of_historical_encoder_hashes():
    got = sample()
    before = copy.deepcopy(got)
    R.check_native_bytes(got)
    assert got == before


@pytest.mark.parametrize('fault', ['missing', 'foreign', 'hash', 'bytes'])
def test_native_forecast_bytes_cannot_be_assumed_from_the_public_digest(fault):
    got = sample()
    if fault == 'missing':
        del got['native_evidence']
    elif fault == 'foreign':
        got['native_evidence']['core']['predictions'][0]['calls'][0]['run'] = 2
    elif fault == 'hash':
        got['payloads'][0]['masked_sha256'] = '0' * 64
    else:
        got['native_evidence']['receipts'][0]['body_base64'] = base64.b64encode(
            b'{}'
        ).decode()
    with pytest.raises(ValueError):
        R.check_native_bytes(got)
