# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings

import pytest

from sdm_client import _core
from sdm_client.core.warnings import ViewConversionWarning
from sdm_client.errors import StructuredDataError


@pytest.mark.parametrize(
    'internal', ['TableNotFoundError', 'TaskReferenceError']
)
def test_internal_cause_is_not_a_public_exception(internal):
    recorded = _core.refusal(
        {
            'class': 'sdm_client.errors.StructuredDataError',
            'code': 'INVALID_REQUEST',
            'message': 'public refusal',
            'details': {'table': 'users'},
            'chain': [
                {
                    'class': 'builtins.ValueError',
                    'message': 'internal cause',
                    '_sdm_internal_cause': internal,
                },
                {'class': 'builtins.TypeError', 'message': 'public cause'},
                {'from': 'query'},
            ],
        }
    )
    actual_host_error = OSError('host cause')
    public = _core.error_of(recorded, {'query': actual_host_error})
    assert type(public) is StructuredDataError
    assert public.code == 'INVALID_REQUEST'
    assert public.details == {'table': 'users'}
    assert type(public.__cause__) is TypeError
    assert str(public.__cause__) == 'public cause'
    assert public.__cause__.__cause__ is actual_host_error
    assert public.__suppress_context__
    assert public.__cause__.__suppress_context__
    assert '_sdm_internal_cause' not in vars(public)


def test_unmarked_value_error_remains_visible():
    public = _core._chain(
        [{'class': 'builtins.ValueError', 'message': 'ordinary cause'}], {}
    )
    assert type(public) is ValueError
    assert str(public) == 'ordinary cause'


def test_serialized_view_warning_names_its_defining_class():
    qualified = 'sdm_connectors._warnings.ViewConversionWarning'
    assert _core._category(qualified) is ViewConversionWarning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _core.give([{'category': qualified, 'message': 'converted view'}])
    category = caught[0].category
    assert f'{category.__module__}.{category.__qualname__}' == qualified


@pytest.mark.parametrize(
    ('category', 'expected'),
    [
        ('builtins.UserWarning', UserWarning),
        ('builtins.DeprecationWarning', DeprecationWarning),
        (
            'sdm_client.core.warnings.ViewConversionWarning',
            ViewConversionWarning,
        ),
    ],
)
def test_public_warning_category_and_filtering(category, expected):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _core.give([{'category': category, 'message': 'public warning'}])
    assert len(caught) == 1
    assert caught[0].category is expected
    assert str(caught[0].message) == 'public warning'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('ignore', expected)
        _core.give([{'category': category, 'message': 'public warning'}])
    assert not caught
