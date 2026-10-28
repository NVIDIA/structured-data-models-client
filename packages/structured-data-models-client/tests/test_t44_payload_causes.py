# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reached frame serializer retains its native nested ValueError causes."""

import json

import pytest

from sdm_client._rust import errors


@pytest.mark.parametrize('marked', [False, True])
@pytest.mark.parametrize('part', [None, 'part_forecast_step'])
@pytest.mark.parametrize('ambient', [False, True])
def test_only_reached_frame_payload_range_has_native_nested_causes(
    marked, part, ambient
):
    fields = {
        'table': {'t': 'utf8', 'v': 'buys'},
        'column': {'t': 'utf8', 'v': 'part'},
        'row': {'t': 'u64', 'v': '0'},
        'dtype': {'t': 'utf8', 'v': 'int64'},
        'forecast_step': {'t': 'u32', 'v': '2'},
        'forecast_steps': {'t': 'u32', 'v': '3'},
    }
    if marked:
        fields['frame_payload_range'] = {'t': 'bool', 'v': True}
    sent = {
        'kind': 'invalid_request',
        'code': 'INVALID_REQUEST',
        'sentence': 'past_int64',
        'part': part,
        'fields': fields,
        'message': 'a value exceeds its request representation',
    }
    caller = LookupError('active caller context') if ambient else None
    bridge = RuntimeError('private core bridge')
    bridge.__context__ = caller
    made = errors.error_from_core(
        json.dumps({'message': 'error', 'error': sent}).encode(),
        {},
        internal_context=bridge,
    )
    assert 'frame_payload_range' not in made.details
    assert made.details['table'] == 'buys'
    assert made.details['column'] == 'part'
    assert made.details['row'] == 0
    if not marked:
        assert made.__cause__ is None
        return
    cause = made.__cause__
    assert type(cause) is ValueError
    bare = cause.__cause__
    assert type(bare) is ValueError
    assert cause.__context__ is bare
    assert bare.__context__ is caller
    assert bare.__cause__ is None
    assert not bare.__suppress_context__
    assert cause.__suppress_context__
    assert bare is not bridge and cause is not bridge
    assert str(bare) == (
        'a value is outside the range int64 represents '
        '(-9223372036854775808 to 9223372036854775807); send the column as strings'
    )
    assert str(cause) == f"Column 'buys.part' row 0: {bare}"
    assert 'forecast' not in str(cause)
