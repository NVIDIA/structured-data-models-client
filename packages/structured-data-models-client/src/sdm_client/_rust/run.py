# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The run loop: one core operation, from its ``start`` to its result.

The loop starts a run with this host's facts, asks it for steps, answers
each fetch through the caller's provider, shows the run's warnings in the
order given, and raises the run's error as this client's exception
(:mod:`.errors`). At each run start it resets the logging bridge's level
cache, so a change to the ``logging`` configuration takes effect
(decision H8).

A provider's failure reaches the core as the contract's ``ProviderError``:
kind, ``retryable``, SQLSTATE and vendor code, never the driver's text; the
exception the run then ends with names the driver's class and is raised
``from None`` (decision H17).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any

from sdm_client._routing import RouteError
from sdm_client._rust import codec, errors, facts
from sdm_client.errors import StructuredDataError

#: A provider: answers one fetch, the ``FetchStep`` as the contract writes
#: it, with an object that exports Arrow through ``__arrow_c_stream__``.
Provider = Callable[[dict[str, Any]], Any]


def native() -> ModuleType:
    r"""``sdm_client._native``, imported here and nowhere else.

    A wheel without it cannot take a core route, and never falls back to
    the old one: the call raises :class:`~sdm_client._routing.RouteError`.
    """
    try:
        from sdm_client import _native
    except ImportError as missing:
        raise RouteError(
            'a core route needs sdm_client._native, which this installation '
            'of the client cannot import'
        ) from missing
    return _native


def _driver(error: Exception) -> tuple[dict[str, Any], str]:
    r"""A driver's exception as the contract's ``ProviderError``, and its class.

    Only the SQLSTATE and the vendor code are kept, as text, from the
    attributes drivers put them in; the driver's message is not.
    """
    sqlstate = next(
        (
            value
            for name in ('sqlstate', 'pgcode', 'sql_state')
            if isinstance(value := getattr(error, name, None), str)
        ),
        None,
    )
    vendor = next(
        (
            str(value)
            for name in ('vendor_code', 'errno')
            if isinstance(value := getattr(error, name, None), (str, int))
            and not isinstance(value, bool)
        ),
        None,
    )
    kind = type(error)
    provided = {
        'kind': 'data_source',
        'retryable': False,
        'sqlstate': sqlstate,
        'vendor_code': vendor,
    }
    return provided, f'{kind.__module__}.{kind.__qualname__}'


class Finished:
    r"""A finished run: its result, and the attachments it still holds."""

    def __init__(self, result: dict[str, Any], run: Any) -> None:
        self.result = result
        self._run = run

    def take(self, attachment: int) -> Any:
        r"""Takes an attachment the result names: ``bytes`` or Arrow."""
        return self._run.take_attachment(attachment)


def run(
    operation: dict[str, Any],
    attachments: Mapping[int, Any],
    *,
    provider: Provider | None = None,
) -> Finished:
    r"""Runs ``operation`` to its result, with the attachments its fields name.

    Raises:
        StructuredDataError: The run's error, as :mod:`.errors` maps it.
        RouteError: ``sdm_client._native`` cannot be imported.
    """
    core = native()
    core.reset_log_cache()
    start = codec.encode(
        {
            'message': 'start',
            'contract': core.CONTRACT_VERSION,
            'facts': facts.host_facts(),
            'operation': operation,
        }
    )
    try:
        handle = core.Run.start(start, attachments)
    except core.CoreError as refused:
        errors.raise_from_core(refused.control, refused.attachments)
    driver_class: str | None = None
    while True:
        try:
            control, sent = handle.next()
        except core.CoreError as failed:
            _warn(handle)
            errors.raise_from_core(
                failed.control, failed.attachments, driver_class=driver_class
            )
        _warn(handle)
        step = codec.read(control, sent)
        if step['message'] != 'step':
            raise StructuredDataError(
                'the core gave a message that is not a step',
                code='INTERNAL_ERROR',
            )
        if step['step'] == 'done':
            return Finished(step['result'], handle)
        if step['step'] != 'fetch' or provider is None:
            handle.cancel()
            raise StructuredDataError(
                f'the core asked for a {step["step"]} step this call cannot '
                'answer',
                code='INTERNAL_ERROR',
            )
        for request in step['requests']:
            attached: dict[int, Any] = {}
            try:
                attached[1] = provider(request)
                outcome: dict[str, Any] = {'outcome': 'data', 'data': 1}
            except Exception as failure:  # noqa: BLE001 - the provider's
                provided, driver_class = _driver(failure)
                outcome = {'outcome': 'error', 'error': provided}
            reply = codec.encode(
                {
                    'message': 'fetch_reply',
                    'id': request['id'],
                    'outcome': outcome,
                }
            )
            try:
                handle.reply(reply, attached)
            except core.CoreError as refused:
                errors.raise_from_core(
                    refused.control,
                    refused.attachments,
                    driver_class=driver_class,
                )


def _warn(handle: Any) -> None:
    taken = handle.take_warnings()
    if taken is not None:
        errors.warn_from_core(*taken)
