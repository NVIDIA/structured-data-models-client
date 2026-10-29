# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Only the reached native empty-context diagnostic keeps its ValueError cause."""

import json

import pytest

from sdm_client._rust import errors


@pytest.mark.parametrize('marked', [False, True])
@pytest.mark.parametrize('ambient', [False, True])
def test_native_context_value_error_uses_existing_scoped_validation_recipe(
    marked, ambient
):
    fields = {'condition': {'t': 'utf8', 'v': 'users.age > 22'}}
    if marked:
        fields['native_key_reason'] = {'t': 'utf8', 'v': 'task_validation'}
    diagnostic = {
        'kind': 'invalid_request',
        'code': 'INVALID_REQUEST',
        'sentence': 'context_too_restrictive',
        'fields': fields,
        'message': 'no context rows',
    }
    if marked:
        diagnostic['public_details'] = {}
    caller = LookupError('ambient caller') if ambient else None
    bridge = RuntimeError('private core bridge')
    bridge.__context__ = caller
    made = errors.error_from_core(
        json.dumps({'message': 'error', 'error': diagnostic}).encode(),
        {},
        internal_context=bridge,
    )
    expected = 'Failed to collect any context examples. Is your predictive query too restrictive?'
    assert str(made) == expected
    assert made.code == 'INVALID_REQUEST'
    if marked:
        assert made.details == {}
        assert type(made.__cause__) is ValueError
        assert str(made.__cause__) == expected
        assert made.__cause__.__cause__ is None
        assert not made.__cause__.__suppress_context__
    else:
        assert made.__cause__ is None
        assert made.details['condition'] == 'users.age > 22'
    with pytest.raises(type(made)) as caught:
        errors.raise_error(made, internal_context=bridge)
    assert caught.value is made
    if marked:
        assert made.__context__ is made.__cause__
        assert made.__cause__.__context__ is caller
