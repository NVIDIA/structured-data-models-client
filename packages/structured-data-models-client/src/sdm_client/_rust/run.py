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
kind, ``retryable``, SQLSTATE, vendor code and an internal driver diagnostic; the
exception the run then ends with names the driver's class and is raised
``from None`` (decision H17). A failure is retryable when the provider's
exception says so, by a ``retryable`` attribute that is ``True``, and not
otherwise.

A provider may answer a fetch one page at a time (design choice 1,
paging): it returns a :class:`Page`, one page with more following, and the
core asks again with the same request and the page's ``continuation`` in
the step (``step['continuation']``), until the provider returns its last
page as an answer; the core reads the pages as one answer.

A model call (from T1.11d, ``predict``'s) is answered by the caller's
``model``: given the body's bytes, it returns the answer's bytes, which the
core reads as the endpoint's 200; it raises :class:`TransportFailed` when
the client's transport failed (the endpoint could not be reached, or its
answer could not be read), which the core is told as the contract's
transport error (kind and ``retryable`` only, never the transport's text);
anything else it raises (the endpoint's refusal, which the caller has made
safe to show) cancels the run and is raised as it is. A serving endpoint's
transport raises :class:`~sdm_client.core.transport.EndpointFailed` for a
failure with no answer, which the core is told as it is (the contract's
failed outcome: the status, the SDK's message as an attachment, its code
and class), and names (T3.13).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Named for annotations only: the warehouse package is an optional
    # dependency, and a core route over frames needs none of it.
    from sdm_connectors.providers import Provider

from sdm_client._routing import RouteError
from sdm_client._rust import codec, errors, facts
from sdm_client.core.transport import EndpointFailed
from sdm_client.errors import StructuredDataError

# A provider (``sdm_connectors.providers.Provider``) answers one fetch, the
# ``FetchStep`` as the contract writes it, with an object that exports Arrow
# through ``__arrow_c_stream__``, or with a :class:`Page` of it; the key
# sets its plan names are handed over with it, by attachment id, each an
# object exporting Arrow the same way.

#: A model: answers one call, given its body's bytes, with the bytes of the
#: endpoint's answer.
Model = Callable[[bytes], bytes]


class Page:
    r"""One page of a provider's answer to a fetch, more following:
    ``data`` exports Arrow through ``__arrow_c_stream__``, and
    ``continuation`` is the text the core hands back, in the step that asks
    for the next page."""

    __slots__ = ('continuation', 'data')

    def __init__(self, data: Any, continuation: str) -> None:
        self.data = data
        self.continuation = continuation


class TransportFailed(Exception):
    r"""The client's transport failed: the endpoint could not be reached,
    or its answer could not be read. Raised by a :data:`Model`, and told to
    the core as a transport error. It carries no text."""


class Refused(Exception):  # noqa: N818 - says what happened to the call
    r"""The call failed in a way the caller sees as ``raised``: the
    endpoint's refusal, as the client's transport reports it, another
    failure of the transport, or any other exception the transport raised
    (the Databricks SDK's ``ValueError`` for a failed OAuth refresh, which
    the reference's client carries the answers on too). Raised by a
    :data:`Model`. The core is told only that the call failed, its
    ``status`` where the endpoint answered with one, else as a transport
    error, so it ends the run carrying the answers of the calls before it
    (T2.8c); then ``raised`` is raised with them in its ``details``,
    keeping its own type, code and text."""

    def __init__(self, raised: Exception, status: int | None = None) -> None:
        super().__init__()
        self.raised = raised
        self.status = status


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


def _driver(
    error: Exception, provider: Any = None
) -> tuple[dict[str, Any], str]:
    r"""A driver's exception as the contract's ``ProviderError``, and its class.

    Only the SQLSTATE and the vendor code are kept, as text, from the
    attributes drivers put them in, and what ``provider`` reads of the
    failure where it reads one (its ``error_fields``, the
    :class:`~sdm_connectors.providers.Provider` protocol's hook: a
    Databricks failure's SQLSTATE its message writes and its error class);
    the driver's message is passed through unread as an internal diagnostic;
    only the core interprets it, and never echoes it into public errors.
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
        'retryable': getattr(error, 'retryable', False) is True,
        'sqlstate': sqlstate,
        'vendor_code': vendor,
        'driver_message': str(error),
    }
    # What the provider reads of its driver's failure beyond the attributes
    # above (a Databricks failure's SQLSTATE in its message, T3.4, and its
    # error class, T3.27), as the reference reads them.
    read = getattr(provider, 'error_fields', None)
    said = read(error) if callable(read) else None
    if said is not None:
        provided.update(said)
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
    # A frame call releases its local attachment bindings after Start.
    # Own that mutable map rather than clearing the caller's Mapping.
    attachments = dict(attachments)
    owned_frame = (
        operation.get('provider', {}).get('provider') == 'frames'
        or operation['op'] == 'prepare_request'
        or operation.get('direct') is not None
    )
    warning_presentation: errors.WarningPresentation = 'coded'
    if owned_frame and operation['op'] in (
        'describe_source',
        'prepare_request',
    ):
        warning_presentation = 'frame_schema'
    elif operation.get('direct') is not None:
        warning_presentation = 'frame_schema'
    elif owned_frame and operation['op'] == 'materialize':
        warning_presentation = 'frame_graph'
    held = None if state is None else state.get('held')
    if held is not None:
        memo = max(attachments, default=0) + 1
        operation = {
            **operation,
            'sampler_state': {'version': held['version'], 'memo': memo},
        }
        attachments = {**attachments, memo: held['memo']}
        if owned_frame:
            from sdm_client._rust import frame_materialize

            checkpoint = frame_materialize.rebind(held, attachments)
            if checkpoint is not None:
                operation['sampler_state']['frame_checkpoint'] = checkpoint
    start = codec.encode(
        {
            'message': 'start',
            'contract': core.CONTRACT_VERSION,
            'facts': facts.host_facts(core),
            'operation': operation,
        }
    )
    try:
        handle = core.Run.start(start, attachments)
    except core.CoreError as refused:
        if state is not None and held is not None and _drops_the_state(refused):
            # The core says the state it refused is to be dropped (decision
            # E74): the next run starts without one.
            state.pop('held', None)
        if owned_frame:
            attachments.clear()
            held = None
        errors.raise_from_core(refused.control, refused.attachments)
    if owned_frame:
        attachments.clear()
        held = None
    try:
        return _steps(
            core,
            handle,
            provider,
            model,
            quiet,
            given,
            kept,
            owned_frame,
            warning_presentation,
        )
    except BaseException:
        if owned_frame:
            if state is not None:
                _keep(core, handle, state)
                state = None
            # Drop the completed/current frame Run even when the caller
            # keeps this exception's traceback. Its snapshots are call-owned.
            handle = None
        raise
    finally:
        if state is not None:
            _keep(core, handle, state)


def _drops_the_state(refused: Any) -> bool:
    r"""Whether the core's refusal of a start tells the host to drop the
    sampler state handed to it (``drop_held``, decision E74): the core's
    word, never the field the error names."""
    try:
        message = codec.read(refused.control, refused.attachments)
        dropped = message['error'].get('drop_held') or []
    except (codec.CodecError, KeyError, TypeError, AttributeError):
        return False
    return isinstance(dropped, list) and 'sampler_state' in dropped


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
        if left.get('frame_checkpoint') is not None:
            from sdm_client._rust import frame_materialize

            state['held']['frame_checkpoint'] = frame_materialize.take(
                handle, left['frame_checkpoint']
            )
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
    owned_frame: bool = False,
    warning_presentation: errors.WarningPresentation = 'coded',
) -> Finished:
    r"""The run's steps, each answered, to its result."""
    try:
        driver_class: str | None = None
        refused: Refused | None = None
        while True:
            error: StructuredDataError | None = None
            try:
                control, sent = handle.next()
            except core.CoreError as failed:
                _warn(handle, quiet, given, kept, warning_presentation)
                error = errors.error_from_core(
                    failed.control,
                    failed.attachments,
                    driver_class=driver_class,
                )
                _answered(handle, failed, error)
            if error is not None:
                # Raised outside the handler, so nothing is its context.
                if refused is not None:
                    # The call's own failure, as the caller sees it, carrying
                    # what the core carried on its end of the run.
                    _carried(error, refused.raised)
                    _of_the_part(error, refused.raised)
                    raise refused.raised
                errors.raise_error(error)
            _warn(handle, quiet, given, kept, warning_presentation)
            step = codec.read(control, sent)
            if step['message'] != 'step':
                raise StructuredDataError(
                    'the core gave a message that is not a step',
                    code='INTERNAL_ERROR',
                )
            if step['step'] == 'done':
                return Finished(step['result'], handle)
            if step['step'] == 'warnings':
                continue
            if step['step'] == 'call_model' and model is not None:
                refused = _call(core, handle, step, model)
                continue
            if step['step'] != 'fetch' or provider is None:
                handle.cancel()
                raise StructuredDataError(
                    f'the core asked for a {step["step"]} step this call cannot '
                    'answer',
                    code='INTERNAL_ERROR',
                )
            # A provider's failure is named by its driver's class only for the
            # step that failed: the core may carry on past one (a whole-table
            # read it falls back from, T1.32e), and a later error is no other
            # step's.
            driver_class = None
            for request in step['requests']:
                attached: dict[int, Any] = {}
                try:
                    # A later page's step names the key sets the first took.
                    keys = (
                        {}
                        if request.get('continuation')
                        else _key_sets(handle, request)
                    )
                    answer = provider(request, keys)
                    outcome: dict[str, Any] = {'outcome': 'data', 'data': 1}
                    if isinstance(answer, Page):
                        attached[1] = answer.data
                        outcome['continuation'] = answer.continuation
                    else:
                        attached[1] = answer
                except RouteError:
                    # The route itself failed (an import of the old engine on
                    # it, say), which is no provider's error to hand the core.
                    handle.cancel()
                    raise
                except Exception as failure:  # noqa: BLE001 - the provider's
                    provided, driver_class = _driver(failure, provider)
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
    except BaseException:
        if owned_frame:
            handle = None
        raise


def _call(
    core: Any, handle: Any, step: dict[str, Any], model: Model
) -> Refused | None:
    r"""Answers one model call with ``model``: its answer as a 200, or, when
    the transport failed, the contract's transport error; a refused call
    (:class:`Refused`) as its status, or as a transport error where it has
    none, and returned, for the run's end to raise."""
    body = handle.take_attachment(step['body'])
    attached: dict[int, Any] = {}
    refused: Refused | None = None
    transport_error: dict[str, Any] = {
        'outcome': 'transport_error',
        'kind': 'model_endpoint',
        # The client's transport has already retried what it retries.
        'retryable': False,
    }
    try:
        answer = model(body)
    except TransportFailed:
        outcome: dict[str, Any] = transport_error
    except EndpointFailed as failed:
        said: dict[str, Any] = {'status': failed.status}
        if failed.detail:
            attached[1] = failed.detail.encode('utf-8', 'surrogatepass')
            said['detail'] = 1
        if failed.provider_code is not None:
            said['provider_code'] = failed.provider_code
        said['provider_error'] = failed.provider_error
        outcome = {'outcome': 'failed', 'answer': said}
    except Refused as failed:
        refused = failed
        status = failed.status
        if isinstance(status, int) and 300 <= status <= 599:
            attached[1] = b'{}'
            outcome = {
                'outcome': 'response',
                'status': status,
                'content_type': 'application/json',
                'body': 1,
            }
        else:
            outcome = transport_error
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
    except core.CoreError as failed:
        errors.raise_from_core(failed.control, failed.attachments)
    return refused


def _answered(handle: Any, failed: Any, error: StructuredDataError) -> None:
    r"""The answers the core carried on its error (``answered``, T2.8c),
    taken from the run and kept on ``error`` as ``_answered``: the Arrow
    result and the rows it scores, which the predict bridge reads."""
    try:
        message = codec.read(failed.control, failed.attachments)
        answered = message['error'].get('answered')
    except (codec.CodecError, KeyError, TypeError, AttributeError):
        return
    if not isinstance(answered, dict):
        return
    stream = handle.take_attachment(answered['result'])
    if stream is not None:
        error._answered = (stream, answered['rows'])  # type: ignore[attr-defined]


#: The core's fields that say what a failed request had answered (T2.8c).
_ANSWERED_FIELDS = ('calls_answered', 'parts_answered')
#: The core's fields that say which part of a question asked in parts a
#: failed call was (a forecast's step, a ranking's candidate by its position
#: among those given, and how many questions the ranking is asked as),
#: which the client says the refusal of (T2.14b).
_PART_FIELDS = ('forecast_step', 'ranking_candidate', 'ranking_parts')


def _carried(error: StructuredDataError, raised: Exception) -> None:
    r"""Puts on ``raised`` what the core carried on ``error``: how many calls
    or parts were answered before it, and their answers where the core
    carried them (a ranking's are counted, never kept, T2.8), beside any
    ``details`` it holds, whatever exception it is, as the reference's
    client does."""
    answered = getattr(error, '_answered', None)
    counts = {
        name: error.details[name]
        for name in _ANSWERED_FIELDS
        if name in (error.details or {})
    }
    if answered is None and not counts:
        return
    held = getattr(raised, 'details', None)
    details = held if isinstance(held, dict) else {}
    details.update(counts)
    raised.details = details  # type: ignore[attr-defined]
    if answered is not None:
        raised._answered = answered  # type: ignore[attr-defined]


def _of_the_part(error: StructuredDataError, raised: Exception) -> None:
    r"""Puts on ``raised``, a refusal the client raises, which part of a
    question asked in parts the core said ``error`` happened in
    (:data:`_PART_FIELDS`), which the client says it of (T2.14b)."""
    if not isinstance(raised, StructuredDataError):
        return
    named = {
        name: error.details[name]
        for name in _PART_FIELDS
        if name in (error.details or {})
    }
    if named:
        held = raised.details
        raised.details = {**(held if isinstance(held, dict) else {}), **named}


def _key_sets(handle: Any, request: dict[str, Any]) -> dict[int, Any]:
    r"""The key sets ``request``'s plan names, taken from the run by id: the
    attachments the core sent with the step (decision Q1), each a key set's
    Arrow or the one text its dialect wrote of it (T3.28)."""
    plan = request['plan']
    taken: dict[int, Any] = {}
    for statement in plan.get('statements', ()):
        for param in statement['params']:
            if param['param'] == 'key_set':
                attachment = param['keys']['attachment']
            elif param['param'] == 'text':
                attachment = param['attachment']
            else:
                continue
            if attachment not in taken:
                taken[attachment] = handle.take_attachment(attachment)
    return taken


def _warn(
    handle: Any,
    quiet: frozenset[str],
    given: set[str] | None,
    kept: list[dict[str, Any]] | None = None,
    warning_presentation: errors.WarningPresentation = 'coded',
) -> None:
    taken = handle.take_warnings()
    # Warning filters may raise and retain this frame in a traceback. The
    # caller owns the Run; presentation needs only the warning bytes now.
    handle = None
    if taken is not None:
        control, sent = taken
        errors.warn_from_core(
            control,
            sent,
            quiet=quiet,
            given=given,
            held=kept,
            presentation=warning_presentation,
        )
