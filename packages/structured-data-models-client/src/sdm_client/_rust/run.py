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

A model call (from T1.11d, ``predict``'s) is answered by the caller's
``model``: given the body's bytes, it returns the answer's bytes, which the
core reads as the endpoint's 200; it raises :class:`TransportFailed` when
the client's transport failed (the endpoint could not be reached, or its
answer could not be read), which the core is told as the contract's
transport error (kind and ``retryable`` only, never the transport's text);
anything else it raises (the endpoint's refusal, which the caller has made
safe to show) cancels the run and is raised as it is.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any

from sdm_client._routing import RouteError
from sdm_client._rust import codec, errors, facts
from sdm_client.errors import StructuredDataError

#: A provider: answers one fetch, the ``FetchStep`` as the contract writes
#: it, with an object that exports Arrow through ``__arrow_c_stream__``; the
#: key sets its plan names are handed over with it, by attachment id, each
#: an object exporting Arrow the same way.
Provider = Callable[[dict[str, Any], Mapping[int, Any]], Any]

#: A model: answers one call, given its body's bytes, with the bytes of the
#: endpoint's answer.
Model = Callable[[bytes], bytes]


class TransportFailed(Exception):
    r"""The client's transport failed: the endpoint could not be reached,
    or its answer could not be read. Raised by a :data:`Model`, and told to
    the core as a transport error. It carries no text."""


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
    model: Model | None = None,
    quiet: frozenset[str] = frozenset(),
    given: set[str] | None = None,
    kept: list[dict[str, Any]] | None = None,
    state: dict[str, Any] | None = None,
) -> Finished:
    r"""Runs ``operation`` to its result, with the attachments its fields name.

    A warning whose code ``quiet`` names is not shown: the caller has shown
    it already, or decides once the run ends whether to show it, from
    ``kept``, which then receives it (``errors.warn_held``). ``given``
    collects the code of every warning the run gave, shown or not.

    ``state``, for a ``materialize``, is where its materializer keeps the
    sampler state (decision E52): the state held there (``'held'``) is
    handed to the run as it was taken, and the state the run leaves, done
    or failed, is put there in its place. This passes the state through and
    reads nothing of it; what it holds is the core's.

    Raises:
        StructuredDataError: The run's error, as :mod:`.errors` maps it.
        RouteError: ``sdm_client._native`` cannot be imported.
    """
    core = native()
    core.reset_log_cache()
    held = None if state is None else state.get('held')
    if held is not None:
        memo = max(attachments, default=0) + 1
        operation = {
            **operation,
            'sampler_state': {'version': held['version'], 'memo': memo},
        }
        attachments = {**attachments, memo: held['memo']}
    start = codec.encode(
        {
            'message': 'start',
            'contract': core.CONTRACT_VERSION,
            'facts': facts.host_facts(core.is_posix_zone),
            'operation': operation,
        }
    )
    try:
        handle = core.Run.start(start, attachments)
    except core.CoreError as refused:
        if state is not None and held is not None and _names_the_state(refused):
            # A state the core refuses is no state to hand back again: the
            # next run starts without one.
            state.pop('held', None)
        errors.raise_from_core(refused.control, refused.attachments)
    try:
        return _steps(core, handle, provider, model, quiet, given, kept)
    finally:
        if state is not None:
            _keep(core, handle, state)


def _names_the_state(refused: Any) -> bool:
    r"""Whether the core's refusal of a start names the sampler state
    handed to it (``field`` ``sampler_state``, or a field of it)."""
    try:
        message = codec.read(refused.control, refused.attachments)
        field = message['error']['fields'].get('field') or {}
    except (codec.CodecError, KeyError, TypeError, AttributeError):
        return False
    value = field.get('v') if isinstance(field, dict) else None
    return isinstance(value, str) and value.split('.')[0] == 'sampler_state'


def _keep(core: Any, handle: Any, state: dict[str, Any]) -> None:
    r"""Puts the sampler state the run left in ``state``, as it was taken:
    its version and its memo's bytes (decision E52). A state that cannot be
    taken is lost, and the next run reads again what it would have
    skipped."""
    try:
        taken = handle.take_sampler_state()
        if taken is None:
            return
        message = codec.read(*taken)
        left = message['state']
        state['held'] = {
            'version': left['version'],
            'memo': handle.take_attachment(left['memo']),
        }
    except (core.CoreError, codec.CodecError):
        return


def _steps(
    core: Any,
    handle: Any,
    provider: Provider | None,
    model: Model | None,
    quiet: frozenset[str],
    given: set[str] | None,
    kept: list[dict[str, Any]] | None = None,
) -> Finished:
    r"""The run's steps, each answered, to its result."""
    driver_class: str | None = None
    while True:
        try:
            control, sent = handle.next()
        except core.CoreError as failed:
            _warn(handle, quiet, given, kept)
            errors.raise_from_core(
                failed.control, failed.attachments, driver_class=driver_class
            )
        _warn(handle, quiet, given, kept)
        step = codec.read(control, sent)
        if step['message'] != 'step':
            raise StructuredDataError(
                'the core gave a message that is not a step',
                code='INTERNAL_ERROR',
            )
        if step['step'] == 'done':
            return Finished(step['result'], handle)
        if step['step'] == 'call_model' and model is not None:
            _call(core, handle, step, model)
            continue
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
                attached[1] = provider(request, _key_sets(handle, request))
                outcome: dict[str, Any] = {'outcome': 'data', 'data': 1}
            except RouteError:
                # The route itself failed (an import of the old engine on
                # it, say), which is no provider's error to hand the core.
                handle.cancel()
                raise
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


def _call(core: Any, handle: Any, step: dict[str, Any], model: Model) -> None:
    r"""Answers one model call with ``model``: its answer as a 200, or, when
    the transport failed, the contract's transport error."""
    body = handle.take_attachment(step['body'])
    attached: dict[int, Any] = {}
    try:
        answer = model(body)
    except TransportFailed:
        outcome: dict[str, Any] = {
            'outcome': 'transport_error',
            'kind': 'model_endpoint',
            # The client's transport has already retried what it retries.
            'retryable': False,
        }
    except BaseException:
        # The endpoint's refusal, made safe to show by the caller, or an
        # interrupt: the run ends here, and the caller sees it as is.
        handle.cancel()
        raise
    else:
        attached[1] = answer
        outcome = {
            'outcome': 'response',
            'status': 200,
            'content_type': 'application/json',
            'body': 1,
        }
    reply = codec.encode(
        {'message': 'model_reply', 'id': step['id'], 'outcome': outcome}
    )
    try:
        handle.reply(reply, attached)
    except core.CoreError as refused:
        errors.raise_from_core(refused.control, refused.attachments)


def _key_sets(handle: Any, request: dict[str, Any]) -> dict[int, Any]:
    r"""The key sets ``request``'s plan names, taken from the run by id: the
    attachments the core sent with the step (decision Q1)."""
    plan = request['plan']
    taken: dict[int, Any] = {}
    for statement in plan.get('statements', ()):
        for param in statement['params']:
            if param['param'] == 'key_set':
                attachment = param['keys']['attachment']
                if attachment not in taken:
                    taken[attachment] = handle.take_attachment(attachment)
    return taken


def _warn(
    handle: Any,
    quiet: frozenset[str],
    given: set[str] | None,
    kept: list[dict[str, Any]] | None = None,
) -> None:
    taken = handle.take_warnings()
    if taken is not None:
        control, sent = taken
        errors.warn_from_core(
            control, sent, quiet=quiet, given=given, held=kept
        )
