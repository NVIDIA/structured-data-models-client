# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual native entity casts precede the permanent gathered grammar."""

import pytest

from sdm_client._rust import relational_response


@pytest.mark.parametrize('text', ['\U00016ac5', '\U00011f55', '1\U00011f55'])
def test_completed_builtin_entity_keeps_unicode13_fallback_after_native_cast(
    monkeypatch, text
):
    calls = []

    def observed(value, dtype):
        calls.append((value, dtype))
        return 5

    monkeypatch.setattr(relational_response, 'as_declared', observed)
    assert relational_response._gathered_entity(text, 'int64') is text
    assert calls == [(text, 'int64')]


def test_native_cast_failure_and_custom_result_owner_are_untouched(monkeypatch):
    selected = ValueError('actual entity conversion')

    def refused(value, dtype):
        raise selected

    monkeypatch.setattr(relational_response, 'as_declared', refused)
    with pytest.raises(ValueError) as raised:
        relational_response._gathered_entity('\U00011f55', 'int64')
    assert raised.value is selected
    owner = object()
    monkeypatch.setattr(
        relational_response, 'as_declared', lambda value, dtype: owner
    )
    assert relational_response._gathered_entity('\U00011f55', 'int64') is owner


@pytest.mark.parametrize('text', ['5', '\u0665', ' +0_05 '])
def test_admitted_gathered_number_keeps_the_actual_native_result(
    monkeypatch, text
):
    converted = int(text)
    monkeypatch.setattr(
        relational_response, 'as_declared', lambda value, dtype: converted
    )
    assert relational_response._gathered_entity(text, 'int64') is converted
