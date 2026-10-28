# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neutral reached casing observations keep complete public Text behavior."""

from __future__ import annotations

import unicodedata
import warnings
from unittest.mock import patch

import pandas as pd
import pytest
from test_t44_native_text_profiles import (
    expected_record,
    reference,
    send,
    warning_values,
)
from test_t44_text_frames import (
    Scripted,
    core as core,
    question,
    shop,
)

from sdm_client._rust.frame_native import NativeColumn, NativeFrameFailure


@pytest.mark.parametrize('storage', ['python', 'pyarrow'])
def test_observed_casing_facts_keep_the_frozen_public_text_journey(
    core, storage
):
    case = f'python-target-{storage}'
    observed = reference('profile')[case]
    expected = expected_record(observed, core)
    options = observed['requested_options']
    frames = shop('python')
    frames['users'].loc[0, 'note'] = 'ΟΣ ΟΣΑ AΣ́ AΣ1 AΣ-A'
    frames['users'].loc[6, 'note'] = 'ΟΣ İ AΣ́ ΟΣΑ'
    frames['users'].loc[7, 'note'] = 'é' * 298 + 'İΣx'
    scripted = Scripted('age')
    ids = [body['metadata']['request_id'] for body in expected['sent']]
    observations = []
    original = NativeColumn.text_facts

    def record(column, identity_input):
        fact = original(column, identity_input)
        observations.append(fact)
        return fact

    with (
        warnings.catch_warnings(record=True) as caught,
        pd.option_context(
            'mode.string_storage',
            options['storage'],
            'future.infer_string',
            options['infer_string'],
        ),
        patch('sdm_client.adapters.relational._request_id', side_effect=ids),
        patch.object(NativeColumn, 'text_facts', record),
    ):
        warnings.simplefilter('always')
        asked = question(frames, prose=True)
        for wanted in expected['requests']:
            request = asked.materialize(
                indices=[7, 8],
                num_context_samples=3,
                num_neighbors=[3],
                num_hops=1,
                random_seed=7,
            )
            send(request, scripted, wanted, core)
    assert scripted.calls == expected['sent']
    assert warning_values(caught) == expected['warnings']
    if not core:
        assert observations == []
        return
    assert observations
    if int(pd.__version__.split('.')[0]) == 2 or storage == 'python':
        assert all(
            fact
            == {
                'mapping': 'full_root',
                'unicode_version': unicodedata.unidata_version,
            }
            for fact in observations
        )
    else:
        supported = {'iσⰯᲉ꟎', 'iσⱟᲉ꟎', 'iσⱟᲊ꟎', 'iσⱟᲊ꟏'}
        assert all(
            set(fact) == {'mapping', 'identity'} for fact in observations
        )
        assert all(fact['mapping'] == 'simple' for fact in observations)
        assert all(fact['identity'] in supported for fact in observations)


def test_failure_capture_records_message_casing_without_displaying_native_error():
    class Undisplayed(ValueError):
        def __str__(self):
            raise AssertionError(
                'capturing failure facts displayed an exception'
            )

    error = Undisplayed()
    failure = NativeFrameFailure.capture(error)
    assert failure.error is error
    assert failure.facts.as_wire()['message_casing'] == {
        'mapping': 'full_root',
        'unicode_version': unicodedata.unidata_version,
    }
