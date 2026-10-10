# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Python host's run loop, driven through the core's test-only operations.

Every code reaches its class through a real run, and a kind outside the
code's set never reaches the caller as that code (decisions H6, E22);
warnings keep their order and class; a logging level change takes effect
at the next run, whose event is delivered once (H8); and a provider's
failure is raised with no chained driver exception and nothing of the
driver's text (H17).
"""

from __future__ import annotations

import logging
import traceback
import warnings
from typing import Any

import pytest

from sdm_client import _native
from sdm_client._rust import codec, errors
from sdm_client._rust.run import run
from sdm_client.core.warnings import SDMWarning
from sdm_client.errors import StructuredDataError

CANARY = 'CANARY-41d9'
KINDS = (
    'invalid_request',
    'unsupported',
    'not_found',
    'data_source',
    'model_endpoint',
    'cancelled',
    'protocol',
    'internal',
)


def _test(test: str, attached: Any = None, **fields: Any) -> Any:
    attachments = (
        {1: codec.attachment(attached)} if attached is not None else {}
    )
    return run({'op': 'test', 'test': test, **fields}, attachments)


def test_every_code_reaches_its_class_through_a_run() -> None:
    reached = {}
    for code in _native.ERROR_CODES:
        for kind in KINDS:
            error = {'kind': kind, 'code': code, 'message': CANARY}
            with pytest.raises(StructuredDataError) as raised:
                _test('fail', error, error=1)
            found = raised.value
            if found.details.get('attachment') == 1:
                # A kind the code does not carry: the message carrying it is
                # refused at the boundary, naming the attachment.
                assert found.code == 'INVALID_MESSAGE'
                continue
            assert found.code == code
            assert type(found) is errors.ERRORS[code][0]
            assert found.details['kind'] == kind
            assert CANARY not in str(found)
            reached.setdefault(code, []).append(kind)
    assert set(reached) == set(_native.ERROR_CODES)
    # A code's kind is fixed by sdm-contract's table: one kind for each,
    # two for a provider's error.
    assert {code: len(kinds) for code, kinds in reached.items()} == {
        code: 2 if code == 'PROVIDER_ERROR' else 1 for code in reached
    }


def test_retryability_is_carried_to_the_details() -> None:
    for retryable in (True, False):
        error = {
            'kind': 'data_source',
            'code': 'PROVIDER_ERROR',
            'message': 'm',
            'retryable': retryable,
        }
        with pytest.raises(StructuredDataError) as raised:
            _test('fail', error, error=1)
        assert raised.value.details['retryable'] is retryable


class _Synthetic(SDMWarning):
    pass


def test_warnings_keep_their_order_and_class_through_a_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(errors.WARNINGS, 'TEST_SYNTHETIC', _Synthetic)
    sent = [
        {'code': 'TEST_SYNTHETIC', 'message': 'first'},
        {'code': 'TEST_OTHER', 'message': 'second'},
        {'code': 'TEST_SYNTHETIC', 'message': 'third'},
    ]
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        _test('warn', sent, warnings=1)
    assert [(w.category, str(w.message)) for w in shown] == [
        (_Synthetic, '[TEST_SYNTHETIC] first'),
        (SDMWarning, '[TEST_OTHER] second'),
        (_Synthetic, '[TEST_SYNTHETIC] third'),
    ]


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _raw_log_run() -> None:
    """The test-only ``log`` operation, started without the host's reset."""
    start = codec.encode(
        {
            'message': 'start',
            'contract': _native.CONTRACT_VERSION,
            'facts': {
                'now': codec.timestamp(0, 'us', 'UTC'),
                'local_zone': None,
                'zone_roots': [],
                'platform': {'os': 'linux', 'strftime_pads_year': False},
            },
            'operation': {'op': 'test', 'test': 'log', 'hold_ms': 0},
        }
    )
    _native.Run.start(start, {}).next()


def test_a_logging_level_change_takes_effect_at_the_next_run() -> None:
    logger = logging.getLogger('sdm_core.test_ops')
    handler = _Collect()
    logger.addHandler(handler)
    event = (logging.INFO, 'sdm_core.test_ops', 'a test-only event')

    def delivered(start: Any) -> list[tuple[int, str, str]]:
        handler.records.clear()
        start()
        return [(r.levelno, r.name, r.getMessage()) for r in handler.records]

    try:
        host = lambda: _test('log', hold_ms=0)  # noqa: E731
        logger.setLevel(logging.WARNING)
        assert delivered(host) == []
        logger.setLevel(logging.INFO)
        # The bridge cached the level: without a reset the event is lost.
        assert delivered(_raw_log_run) == []
        # The host resets the cache at each run start: delivered, once.
        assert delivered(host) == [event]
        logger.setLevel(logging.WARNING)
        assert delivered(host) == []
        logger.setLevel(logging.INFO)
        assert delivered(host) == [event]
    finally:
        logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)


class DriverError(Exception):
    """A driver's exception: codes as attributes, values in its message."""

    sqlstate = '23505'
    errno = 1062


def test_a_provider_error_raises_with_no_driver_exception_chained() -> None:
    def provider(_request: dict[str, Any], _keys: Any) -> Any:
        raise DriverError(f'duplicate key value {CANARY}')

    fetch = {
        'request': {
            'request': 'scan',
            'table': 't',
            'columns': [],
            'filters': [],
            'limit': None,
        },
        'plan': {'plan': 'host', 'expect': {'expect': 'discover'}},
    }
    with pytest.raises(StructuredDataError) as raised:
        run(
            {'op': 'test', 'test': 'fetch', 'fetch': 1},
            {1: codec.attachment(fetch)},
            provider=provider,
        )
    error = raised.value
    assert error.code == 'PROVIDER_ERROR'
    assert error.details['sqlstate'] == '23505'
    assert error.details['vendor_code'] == '1062'
    assert error.details['driver_class'] == f'{__name__}.DriverError'
    assert error.__cause__ is None
    assert error.__suppress_context__
    formatted = ''.join(traceback.format_exception(error))
    assert CANARY not in formatted
    assert 'DriverError' not in formatted.split('StructuredDataError')[0]
