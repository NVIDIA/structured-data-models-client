# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual public constructor consumers of the native-only census boundaries."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest
from test_t42_frames import PATHS

from sdm_client import Schema, StructuredDataError


def _unread_later(touched):
    class Later(pd.DataFrame):
        @property
        def columns(self):
            touched.append('later-columns')
            return super().columns

    frame = Later({'later': [1]})
    touched.clear()
    return frame


@pytest.mark.parametrize(
    'case,expected',
    [
        ('empty', 'a schema needs at least one table'),
        ('empty-name', "table names must be non-empty strings; got ''"),
        ('integer-name', 'table names must be non-empty strings; got 1'),
        ('none-name', 'table names must be non-empty strings; got None'),
        ('none-frame', "table 'rows' must be a pandas DataFrame, got NoneType"),
        ('list-frame', "table 'rows' must be a pandas DataFrame, got list"),
        ('duplicate', "table 'rows' has duplicate column name(s) ['x']"),
        (
            'duplicate-integer',
            "table 'rows' has duplicate column name(s) ['1']",
        ),
        (
            'integer-column',
            "table 'rows' column names must be strings; got [1]",
        ),
        (
            'none-column',
            "table 'rows' column names must be strings; got [None]",
        ),
    ],
)
def test_native_constructor_refusals_keep_complete_public_outcomes(
    monkeypatch, case, expected
):
    outcomes = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        touched = []
        later = _unread_later(touched)
        if case == 'empty':
            frames = {}
        elif case.endswith('-name'):
            name = {'empty-name': '', 'integer-name': 1, 'none-name': None}[
                case
            ]
            frames = {name: pd.DataFrame({'value': [1]})}
        elif case.endswith('-frame'):
            frames = {'rows': None if case == 'none-frame' else []}
        else:
            columns = {
                'duplicate': ['x', 'x'],
                'duplicate-integer': [1, 1],
                'integer-column': [1],
                'none-column': [None],
            }[case]
            frames = {
                'rows': pd.DataFrame([[1] * len(columns)], columns=columns)
            }
        if frames:
            frames['later'] = later
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(StructuredDataError) as refused:
                Schema.from_data(frames)
        error = refused.value
        assert error.code == 'INVALID_REQUEST'
        assert str(error) == f'[INVALID_REQUEST] {expected}'
        assert error.__cause__ is None
        assert error.__context__ is None
        assert touched == []
        outcomes.append(
            (
                type(error),
                error.code,
                str(error),
                error.details,
                error.__suppress_context__,
                [(str(item.message), item.category) for item in caught],
                touched,
            )
        )
    assert outcomes[0] == outcomes[1]
