# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native constructor semantic errors preserve their original exception graph."""

from __future__ import annotations

import traceback
import warnings

import pandas as pd
import pytest
from test_t42_arrow_decimal_frames import _decimal_error_graph
from test_t42_frames import PATHS

from sdm_client import Schema, StructuredDataError
from sdm_client._rust import errors


def _construct(case):
    frame = pd.DataFrame({'value_id': [1, 1, 2]})
    options = {
        'duplicate': {'primary_keys': {'rows': 'value_id'}},
        'edit': {'primary_keys': {'rows': 'missing'}},
        'time': {'time_columns': {'rows': 'missing'}},
        'relationship': {'relationships': [None]},
    }[case]
    return Schema.from_data({'rows': frame}, **options)


@pytest.mark.parametrize('case', ['duplicate', 'edit', 'time', 'relationship'])
@pytest.mark.parametrize('ambient', [False, True])
def test_constructor_semantic_errors_keep_complete_native_context(
    monkeypatch, case, ambient
):
    observations = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        caller = LookupError('active constructor caller')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(StructuredDataError) as failed:
                if ambient:
                    try:
                        raise caller
                    except LookupError:
                        _construct(case)
                else:
                    _construct(case)
        error = failed.value
        assert error.code == 'INVALID_REQUEST'
        assert error.__cause__ is None
        assert error.__context__ is (caller if ambient else None)
        assert error.__suppress_context__ is False
        observations.append(
            (
                _decimal_error_graph(error),
                [(str(item.message), item.category) for item in caught],
            )
        )
    assert observations[0] == observations[1]


def _raise_warning_error(sentinel, context):
    def show(*args, **kwargs):
        try:
            raise context
        except KeyError:
            raise sentinel from None

    return show


@pytest.mark.parametrize('ambient', [False, True])
def test_native_handler_can_reraise_an_earlier_selected_sdk_error_unchanged(
    monkeypatch, ambient
):
    observations = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS)
        with pytest.raises(StructuredDataError) as previous:
            _construct('duplicate')
        sentinel = previous.value
        assert sentinel._sentence == 'frame_key_not_unique'
        caller = LookupError('outer native caller')
        context = KeyError('actual native warning handler')
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        frame = pd.DataFrame(
            {'value_id': pd.Series([None, *range(1, 100)], dtype='float64')}
        )
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            warnings.showwarning = _raise_warning_error(sentinel, context)
            with pytest.raises(StructuredDataError) as failed:
                if ambient:
                    try:
                        raise caller
                    except LookupError:
                        Schema.from_data({'rows': frame})
                else:
                    Schema.from_data({'rows': frame})
        assert failed.value is sentinel
        assert sentinel.__context__ is context
        assert sentinel.__cause__ is None
        assert sentinel.__suppress_context__ is True
        observations.append(_decimal_error_graph(sentinel))
    assert observations[0] == observations[1]


@pytest.mark.parametrize('presentation', ['coded', 'frame_constructor'])
@pytest.mark.parametrize('code', ['INVALID_REQUEST', 'PROVIDER_ERROR'])
@pytest.mark.parametrize('causeful', [False, True])
def test_selected_constructor_recipe_preserves_coded_privacy_and_real_causes(
    presentation, code, causeful
):
    caller = LookupError('actual caller')
    bridge = RuntimeError('private bridge diagnostic')
    selected = StructuredDataError('public selected error', code=code)
    cause = ValueError('selected native cause') if causeful else None
    if cause is not None:
        selected.__cause__ = cause
    try:
        raise caller
    except LookupError:
        with pytest.raises(StructuredDataError) as failed:
            try:
                raise bridge
            except RuntimeError:
                errors.raise_error(
                    selected,
                    internal_context=bridge,
                    presentation=presentation,
                )
    assert failed.value is selected
    assert selected.__cause__ is cause
    assert 'private bridge diagnostic' not in ''.join(
        traceback.format_exception(selected)
    )
    if causeful:
        assert selected.__suppress_context__ is True
        assert cause.__context__ is caller
    elif presentation == 'frame_constructor' and code == 'INVALID_REQUEST':
        assert selected.__context__ is caller
        assert selected.__suppress_context__ is False
    else:
        assert selected.__context__ is bridge
        assert selected.__suppress_context__ is True
