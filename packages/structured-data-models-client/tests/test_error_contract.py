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

import concurrent.futures
import copy
import functools
import json
import multiprocessing
import pickle
from collections.abc import Callable
from typing import Any

import pytest

import sdm_client
from sdm_client import (
    MissingExtraError,
    NimRequestError,
    StructuredDataError,
    UnknownModelError,
    UnsupportedRequestError,
)
from sdm_client._rust.errors import _instance
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


# --- errors that leave the process ------------------------------------------


def _every_error_class() -> set[type[StructuredDataError]]:
    found: set[type[StructuredDataError]] = set()
    pending = [StructuredDataError]
    while pending:
        cls = pending.pop()
        if cls.__module__.startswith('sdm_client') and cls not in found:
            found.add(cls)
            pending.extend(cls.__subclasses__())
    return found


def _error_with_notes() -> StructuredDataError:
    error = StructuredDataError(
        'the request was refused',
        code='INVALID_REQUEST',
        details={'rule': 'r', 'answered': [1, 2]},
    )
    error._sentence = 'some_sentence'
    error.add_note('a note added by the caller')
    return error


#: One error of each class, made as this client makes it: by its own
#: initializer, and by the core's route, which skips that initializer.
_SHIPPED = {
    'StructuredDataError': _error_with_notes(),
    'NimRequestError': _to_nim_error(
        422,
        _problem(
            code='INVALID_REQUEST',
            invalid_params=[{'name': 'predict.age', 'reason': 'not a number'}],
        ),
    ),
    'NimRequestError-without-code': NimRequestError(
        503, code=None, message='busy'
    ),
    'NimRequestError-from-core': _instance(
        NimRequestError,
        'the endpoint refused the call',
        'ENDPOINT_REFUSED',
        {'status': 503, 'refused_code': 'BUSY', 'kind': 'transport'},
    ),
    'UnknownModelError': UnknownModelError('nope', ['kumo-tabular']),
    'UnknownModelError-from-core': _instance(
        UnknownModelError,
        'no such model',
        'UNKNOWN_MODEL',
        {'model': 'nope', 'kind': 'invalid_request'},
    ),
    'UnsupportedRequestError': UnsupportedRequestError(
        'too many rows', details={'limit': 10}
    ),
    'MissingExtraError': MissingExtraError(
        'in-process', 'structured-data-models'
    ),
}


def _copies() -> list[tuple[str, Callable[[Any], Any]]]:
    ways: list[tuple[str, Callable[[Any], Any]]] = [
        (
            f'pickle-{protocol}',
            functools.partial(_pickled, protocol=protocol),
        )
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
    ]
    return [*ways, ('copy', copy.copy), ('deepcopy', copy.deepcopy)]


def _pickled(error: Any, protocol: int) -> Any:
    return pickle.loads(pickle.dumps(error, protocol=protocol))


def _assert_same_error(copied: Any, error: StructuredDataError) -> None:
    assert copied is not error
    assert type(copied) is type(error)
    assert str(copied) == str(error)
    assert copied.args == error.args
    assert copied.message == error.message
    assert copied.code == error.code
    assert copied.details == error.details
    assert vars(copied) == vars(error)


def test_every_error_class_is_shipped_below() -> None:
    r"""A new error class is checked for crossing a process too."""
    assert {type(error) for error in _SHIPPED.values()} == _every_error_class()


@pytest.mark.parametrize(('how', 'copier'), _copies())
@pytest.mark.parametrize('error', _SHIPPED.values(), ids=list(_SHIPPED))
def test_an_error_survives_pickle_and_copy(
    error: StructuredDataError, how: str, copier: Callable[[Any], Any]
) -> None:
    r"""Pickling, which multiprocessing, process pools, Spark, Ray and joblib
    use to hand an error back, and ``copy`` give the same error: its class,
    its text, its code, its details and every other attribute. A subclass
    whose initializer takes other arguments than its message is not called
    again to make it."""
    _assert_same_error(copier(error), error)


def _raise(error: BaseException) -> None:
    raise error


def test_an_error_raised_in_a_process_pool_reaches_the_caller() -> None:
    r"""The worker pickles the error it raised, and the caller unpickles it.
    Spawned, as on macOS and Windows, rather than forked from a process
    running threads."""
    error = _SHIPPED['NimRequestError']
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context('spawn')
    ) as pool:
        future = pool.submit(_raise, error)
        with pytest.raises(NimRequestError) as raised:
            future.result(timeout=60)
    _assert_same_error(raised.value, error)
    assert raised.value.status_code == 422
