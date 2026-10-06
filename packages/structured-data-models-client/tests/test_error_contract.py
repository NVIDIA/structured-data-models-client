# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The error surface a caller can rely on.

Two things are being pinned: that ``except StructuredDataError`` catches everything the
SDK raises and each subclass is importable to branch on, and that a NIM's
per-field validation diagnosis reaches the message instead of being buried in
``details``.
"""

from __future__ import annotations

import json

import pytest

import sdm_client
from sdm_client import (
    MissingExtraError,
    NimRequestError,
    StructuredDataError,
    UnknownModelError,
)
from sdm_client.core.transport import _to_nim_error
from sdm_client.errors import MAX_INVALID_PARAMS, format_invalid_params


def _problem(**overrides: object) -> bytes:
    body = {
        'type': '/problems/validation-failed',
        'status': 422,
        'code': 'INVALID_SCHEMA',
        'detail': 'Request validation failed.',
    }
    body.update(overrides)
    return json.dumps(body).encode()


# --- the public surface -----------------------------------------------------


@pytest.mark.parametrize(
    'name',
    [
        'StructuredDataError',
        'NimRequestError',
        'MissingExtraError',
        'UnknownModelError',
    ],
)
def test_error_types_are_exported(name: str) -> None:
    r"""A caller cannot branch on an error they cannot import."""
    assert name in sdm_client.__all__
    assert hasattr(sdm_client, name)


@pytest.mark.parametrize(
    'error',
    [
        NimRequestError(503, code='X', message='m'),
        MissingExtraError('sdm_relational_engine', 'relational'),
        UnknownModelError('nope', ['kumo-tabular']),
    ],
)
def test_every_error_is_catchable_as_predict_error(
    error: StructuredDataError,
) -> None:
    with pytest.raises(StructuredDataError):
        raise error


@pytest.mark.parametrize(
    'cls',
    [
        StructuredDataError,
        NimRequestError,
        MissingExtraError,
        UnknownModelError,
    ],
)
def test_error_types_are_documented(cls: type) -> None:
    r"""These are the first thing a user reads when something breaks."""
    assert cls.__doc__ and len(cls.__doc__.strip()) > 40


def test_status_code_is_reachable_for_branching() -> None:
    error = NimRequestError(
        429, code='RATE_LIMIT_EXCEEDED', message='slow down'
    )
    assert error.status_code == 429
    assert str(error) == '[429 RATE_LIMIT_EXCEEDED] slow down'


# --- invalid_params rendering ----------------------------------------------


def test_invalid_params_reach_the_message() -> None:
    r"""The top-level detail is often only "Request validation failed."; the
    diagnosis that lets a caller fix the request is in ``invalid_params``.
    """
    error = _to_nim_error(
        422,
        _problem(
            invalid_params=[
                {
                    'name': 'context.instance_table.rows[0][age]',
                    'reason': 'expected int64, got string',
                }
            ]
        ),
    )

    assert 'context.instance_table.rows[0][age]' in error.message
    assert 'expected int64, got string' in error.message
    assert error.status_code == 422
    assert error.code == 'INVALID_SCHEMA'


def test_invalid_params_stay_in_details_too() -> None:
    r"""Rendering them is additive; structured access still works."""
    params = [{'name': 'a', 'reason': 'b'}]
    error = _to_nim_error(422, _problem(invalid_params=params))
    assert error.details['invalid_params'] == params


def test_invalid_params_are_capped_with_a_count() -> None:
    r"""A rejected batch can name thousands of rows."""
    params = [{'name': f'col{i}', 'reason': 'bad'} for i in range(12)]
    message = _to_nim_error(422, _problem(invalid_params=params)).message

    assert 'col0: bad' in message
    assert f'col{MAX_INVALID_PARAMS}' not in message
    assert f'(and {12 - MAX_INVALID_PARAMS} more)' in message


def test_error_without_invalid_params_is_unchanged() -> None:
    message = _to_nim_error(500, _problem(status=500, detail='boom')).message
    assert message == 'boom'


@pytest.mark.parametrize('params', [None, [], 'not-a-list', [1, 2], [{}]])
def test_malformed_invalid_params_render_as_nothing(params: object) -> None:
    r"""A proxy can return anything; rendering must not raise."""
    assert format_invalid_params(params) == ''  # type: ignore[arg-type]


def test_partial_entries_still_render() -> None:
    assert format_invalid_params([{'name': 'a'}]) == ' a'
    assert format_invalid_params([{'reason': 'b'}]) == ' b'


def test_rendered_shape_matches_the_relational_engine_path() -> None:
    r"""The two renderers cannot share code across the package boundary, so
    the shape is pinned on both sides instead.
    """
    engine_rfm = pytest.importorskip('sdm_relational_engine.rfm.rfm')
    params = [{'name': 'x', 'reason': 'y'}, {'name': 'p', 'reason': 'q'}]
    assert format_invalid_params(params) == engine_rfm._invalid_params_summary(
        {'invalid_params': params}
    )
