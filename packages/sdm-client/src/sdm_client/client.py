# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""One connection to one NIM, serving every model it can reach.

The model is an argument:

>>> with SDMClient(url='http://localhost:8000') as client:
...     result = client.predict(request, model=KumoTabular())

Each client owns its own transport, so several can target different endpoints
or tenants at once, including from several threads: a prediction always goes to
the endpoint and credential of the client that started it.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from types import TracebackType
from typing import Any

from sdm_client.base import (
    ModelAdapter,
    ModelCapabilities,
    ModelSpec,
    check_request_against_model,
    describe,
)
from sdm_client.core.transport import Transport, redact_url
from sdm_client.core.warnings import SDMWarning, caller_stacklevel
from sdm_client.decompose import RANKING, Decomposition
from sdm_client.errors import StructuredDataError, UnknownModelError
from sdm_client.models import MODELS, Model, spec_for
from sdm_client.request import Request
from sdm_client.result import PredictionResult
from sdm_client.wire.keys import (
    _named_by,
    keys_in,
    scored_keys,
    without_the_keys,
)

# A NIM that does not serve the session routes at all, as opposed to one that
# has merely forgotten this session.
_SESSIONS_UNSUPPORTED = frozenset({404, 405, 501})


def _adapter_for(spec: ModelSpec) -> ModelAdapter:
    r"""The adapter that shapes requests for ``spec``.

    Imported here rather than at module scope so that importing the client does
    not pull in an adapter that reaches for an optional driver.
    """
    if spec.model_id == 'kumo-tabular':
        from sdm_client.adapters.tabular import TabularAdapter

        return TabularAdapter()
    if spec.model_id == 'kumo-relational':
        from sdm_client.adapters.relational import RelationalAdapter

        return RelationalAdapter()
    raise UnknownModelError(spec.model_id, list(MODELS))


class FittedModel:
    r"""A context pinned on the endpoint, ready to score against.

    Returned by :meth:`SDMClient.fit`. The contract's
    ``POST /v1/sessions`` uploads a context and runs no inference; scoring
    against it is a second call. That is what ``fit`` and ``predict`` are here.

    Close it when done, or use it as a context manager. A session holds memory
    on the endpoint until it is released or expires.

    ``pinned`` is what the session pinned, minus the context itself. Reading an
    answer needs the task and the schema -- the key an id stands for is named
    there -- but the rows have been uploaded and live on the endpoint, so
    keeping a second encoded copy for the lifetime of the handle would hold a
    whole dataset for nothing.
    """

    def __init__(
        self,
        client: SDMClient,
        adapter: ModelAdapter,
        session_id: str,
        outputs: tuple[str, ...],
        pinned: dict[str, Any] | None = None,
        named_outputs: Sequence[str] | None = None,
        pinned_keys: frozenset[Any] = frozenset(),
        pinned_over_a_window: bool = False,
        held_back: int = 0,
        question: str | None = None,
    ) -> None:
        self._client = client
        self._adapter = adapter
        self._session_id = session_id
        self._outputs = outputs
        self._named_outputs = (
            None if named_outputs is None else tuple(named_outputs)
        )
        self._pinned_keys = pinned_keys
        self._pinned_over_a_window = pinned_over_a_window
        self._held_back = held_back
        self._pinned = pinned or {}
        self._question = question
        self._closed = False

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def model(self) -> str:
        return self._adapter.spec.model_id

    def predict(self, request: Request) -> PredictionResult:
        r"""Score ``request``'s rows against the pinned context.

        Only the rows travel: the context, schema and task were pinned when
        this handle was created. Reading the answer needs both halves, though
        -- the key an id stands for is named in the pinned schema -- so the
        adapter is handed the whole request while the wire gets only the half
        the contract allows on a scoring call.
        """
        if not isinstance(request, Request):
            raise StructuredDataError(
                f'request must be a Request, got {type(request).__name__}',
                code='INVALID_REQUEST',
            )
        if self._closed:
            raise StructuredDataError(
                'this fitted model is closed; fit again to score more rows',
                code='INVALID_CONFIGURATION',
            )
        parts = getattr(request, 'parts', None)
        if parts is not None:
            raise StructuredDataError(
                f'this request is asked as {len(parts)} questions and a session '
                f'pins one context, so scoring it here would answer the pinned '
                f'question that many times. Ask it with predict()',
                code='UNSUPPORTED_REQUEST',
            )
        asked = getattr(request, 'query', None)
        if (
            self._question is not None
            and asked is not None
            and asked != self._question
        ):
            raise StructuredDataError(
                f'this session pinned {self._question!r} and was handed '
                f'{asked!r}. Only the rows travel on a scoring call, so '
                f'the pinned question is the one that would be answered. Fit '
                f'again to ask something else',
                code='INVALID_REQUEST',
            )
        if request.nothing_to_score:
            return _empty_answer(self.model, self._outputs)
        _warn_scoring_what_was_pinned(
            request, self._pinned_keys, self._pinned_over_a_window
        )
        parts = [
            self._adapter.normalize(
                self._client._transport.session_predict(
                    self._session_id, payload
                ),
                self._outputs,
                {**self._pinned, **payload},
            )
            for payload in self._adapter.session_prediction_plan(
                request, self._outputs
            )
        ]
        if not parts:
            raise StructuredDataError(
                'the adapter produced no scoring call for this request',
                code='INVALID_REQUEST',
            )
        return _report_absent(
            PredictionResult.merge(parts), self._named_outputs, self.model
        )

    def close(self) -> None:
        r"""Release the pinned context. Safe to call twice.

        The handle is marked closed only once the endpoint has released the
        session. Marking it first made a failed release permanent: the next
        call returned early and sent nothing, so the one thing the endpoint
        asks for when it answers `429 Session maintenance is already pending;
        retry the request later` was the one thing the handle could not do.
        """
        if self._closed:
            return
        self._client._transport.delete_session(self._session_id)
        self._closed = True

    def __enter__(self) -> FittedModel:
        return self

    def __exit__(self, *exc: object) -> None:
        r"""Release the context, raising as it did before -- unless the block
        is already failing.

        A failure to release is a real failure and is raised, which is what a
        caller wrapping the block in `try` relies on.

        The exception is a block that is already raising, where nothing is
        said at all. Anything leaving `__exit__` replaces the exception the
        block was raising, and under `-W error` that includes a warning, so
        the one place this must not speak is the one where speaking would
        lose the failure the caller actually needs to see. There the session
        is left to the endpoint's own expiry, and `close()` can be called
        again by a caller who wants to know.
        """
        try:
            self.close()
        except Exception:
            if not (exc and exc[0] is not None):
                raise

    def __repr__(self) -> str:
        state = 'closed' if self._closed else 'open'
        return f'FittedModel({self.model!r}, {state})'


class SDMClient:
    r"""A connection to one Universal TFM NIM.

    Use it as a context manager, or call :meth:`close`.
    """

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        *,
        verify_ssl: bool = True,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        r"""Open a client against one NIM.

        No request is made here: the endpoint is first contacted by
        :meth:`health_ready`, :meth:`models`, or a prediction.

        Args:
            url: Base URL of the NIM, e.g. ``'http://localhost:8000'``.
            api_key: Sent as ``X-API-Key`` when an authenticating gateway
                fronts the NIM. Refused over plaintext ``http://`` to a
                non-local host.
            verify_ssl: Whether to verify the server's TLS certificate.
            timeout: Seconds to wait for each individual attempt, not for the
                call as a whole.
            max_retries: Transport-level retries of a transient failure (429,
                500, 502, 503, 504, or a dropped connection) with exponential
                backoff. ``POST /v1/sessions`` is excluded: a re-sent create
                would orphan a pinned context on the endpoint.
        """
        self._transport = Transport(
            url,
            api_key,
            verify_ssl=verify_ssl,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._adapters: dict[str, ModelAdapter] = {}

    @property
    def url(self) -> str:
        r"""The NIM base URL this client was opened against."""
        return self._transport.url

    def models(self) -> list[str]:
        r"""The model ids **the endpoint** advertises.

        Read from ``/v1/models``, not from this client's own list. The previous
        generation reported its own registry here, which meant it claimed a
        model the endpoint did not serve.
        """
        return self._transport.list_models()

    def capabilities(self, model: str | Model) -> ModelCapabilities:
        r"""What ``model`` supports, as far as this client knows.

        Describes the client side and is tagged ``source='sdk'``. It needs no
        live connection, and it is not the endpoint's own answer -- use
        :meth:`models` for that.
        """
        return describe(self._spec(model))

    def health_ready(self) -> bool:
        r"""Whether the NIM answers ``GET /v1/health/ready`` with 200."""
        return self._transport.health_ready()

    def predict(
        self,
        request: Request,
        *,
        model: str | Model,
        outputs: Sequence[str] | None = None,
    ) -> PredictionResult:
        r"""Answer ``request`` with ``model``.

        Args:
            request: What to ask. Built directly, or by a predictive query.
            model: A model handle such as ``KumoTabular()``, or a model id.
            outputs: The output fields to ask for. Defaults to what the model
                produces for this task kind.

        Returns:
            A :class:`~sdm_client.result.PredictionResult`, always -- never a
            frame-or-explanation union.

        Raises:
            UnknownModelError: If ``model`` is not one the contract carries.
            UnsupportedRequestError: If the model cannot answer this request.
                Raised before anything is sent.
            NimRequestError: If the endpoint answered with an error.
        """
        if request.parts is not None:
            return self._answer_in_parts(request, model, outputs)
        adapter, fields = self._prepare(request, model, outputs)
        if request.nothing_to_score:
            return _empty_answer(adapter.spec.model_id, fields)
        # Every call's rows are kept. A request past the endpoint's per-call
        # cap is split by the adapter, and each call answers about its own
        # rows, so returning only the last would drop the earlier batches.
        parts = [
            adapter.normalize(self._send(adapter, payload), fields, payload)
            for payload in adapter.prediction_plan(request, fields)
        ]
        if not parts:
            raise StructuredDataError(
                'the adapter produced no call for this request',
                code='INVALID_REQUEST',
            )
        return _report_absent(
            PredictionResult.merge(parts), outputs, adapter.spec.model_id
        )

    def _answer_in_parts(
        self,
        request: Request,
        model: str | Model,
        outputs: Sequence[str] | None,
    ) -> PredictionResult:
        r"""Ask the questions this one is asked as, and give back one answer.

        A ranking and a forecast have no task on the endpoint. Each is asked
        as questions it does serve, and what comes back has to be put together
        as the answer that was actually asked for -- an order over candidates,
        or a value per step -- rather than handed back as the several answers
        it took.
        """
        rewritten = request.decomposition
        assert rewritten is not None
        ranking = rewritten.kind == RANKING
        if ranking:
            _refuse_outputs_a_ranking_cannot_give(outputs)
        # A ranking needs how sure each answer was, which is what orders the
        # candidates, and the fields it gives back are the merge's own. A step
        # of a forecast is an ordinary regression, so the caller's fields are
        # what each part is asked for.
        wanted = _RANKING_PARTS if ranking else outputs
        answers = []
        for part, (_, about) in zip(
            request.parts or (), rewritten.parts, strict=True
        ):
            try:
                answers.append(self.predict(part, model=model, outputs=wanted))
            except StructuredDataError as error:
                raise _which_part_failed(error, rewritten, about) from error
        if ranking:
            return _ranked(answers, rewritten, outputs)
        merged = _by_step(answers, rewritten)
        _warn_of_steps_that_carry_no_explanation(
            answers, rewritten, outputs, spec_for(model).model_id
        )
        return _report_absent(merged, outputs, spec_for(model).model_id)

    def _send(self, adapter: Any, payload: dict[str, Any]) -> dict[str, Any]:
        r"""Post one payload, letting the adapter explain a refusal."""
        try:
            return self._transport.predict(payload)
        except StructuredDataError as error:
            explained = adapter.explain_error(error, payload)
            if explained is error:
                # An adapter that had nothing to add returns the same object,
                # and `raise error from error` makes it its own cause, which
                # is a cycle for anything that walks the chain.
                raise
            raise explained from error

    def fit(
        self,
        request: Request,
        *,
        model: str | Model,
        outputs: Sequence[str] | None = None,
        holdout: Iterable[Any] | None = None,
    ) -> FittedModel:
        r"""Pin ``request``'s context on the endpoint and return a handle.

        The pinned half is model, task, schema and context; later calls send
        only the rows to score. Use this when asking several questions of the
        same context, which is what makes uploading it once worthwhile.

        `holdout` names every row the session will go on to score. The context
        is chosen once, here, and `request` only excludes the rows *it* names,
        so every later call scores rows that are sitting in the pinned context
        with their answers -- and unlike the one-shot path there is nothing the
        caller can do about it afterwards, because the context cannot be
        changed once it is pinned. Naming the population up front is what lets
        it be excluded while there is still a context to exclude it from.

        A row is held back at every moment it appears. For a target that is a
        window, the same entity at an *earlier* anchor is legitimate context
        rather than a leak, so this gives up examples it did not have to. It
        gives them up because a session cannot know which moments it will be
        asked about: holding the entity back everywhere is the only choice
        that cannot leak, and a leak here has no remedy afterwards. Score at
        known moments with `predict()` if those examples are worth more than
        the certainty.

        Args:
            request: The request whose context is pinned.
            model: Which model answers.
            outputs: The fields to ask for.
            holdout: Keys of every row this session may score. They are taken
                out of the pinned context before it is sent.

        Raises:
            UnsupportedRequestError: If the endpoint does not serve sessions.
        """
        _refuse_to_pin_a_rewritten_question(request)
        adapter, fields = self._prepare(request, model, outputs)
        payload = adapter.session_plan(request, fields)
        held = _held_back(payload, holdout)
        try:
            try:
                created = self._transport.create_session(payload)
            except StructuredDataError as error:
                explained = adapter.explain_error(error, payload)
                if explained is error:
                    raise
                raise explained from error
        except Exception as error:
            status = getattr(error, 'status_code', None)
            if status in _SESSIONS_UNSUPPORTED:
                raise StructuredDataError(
                    f'this endpoint does not serve sessions (HTTP {status}); '
                    f'use predict() instead, which sends the context with '
                    f'every call',
                    code='SESSIONS_UNSUPPORTED',
                ) from error
            raise
        session_id = created.get('session_id') or created.get('id')
        if not session_id:
            raise StructuredDataError(
                'the endpoint created a session but named no session id',
                code='INVALID_RESPONSE',
                details={'response_keys': sorted(created)},
            )
        return FittedModel(
            self,
            adapter,
            str(session_id),
            fields,
            {key: value for key, value in payload.items() if key != 'context'},
            named_outputs=outputs,
            pinned_keys=keys_in(
                payload, 'context', with_anchor=request.target_moves_with_time
            ),
            held_back=held,
            pinned_over_a_window=request.target_moves_with_time,
            question=getattr(request, 'query', None),
        )

    def _prepare(
        self,
        request: Request,
        model: str | Model,
        outputs: Sequence[str] | None,
    ) -> tuple[ModelAdapter, tuple[str, ...]]:
        r"""Resolve the adapter and output fields, and refuse early.

        Every check that can be made without the network is made here, so a
        request the model cannot answer fails with a message naming the model
        rather than as an opaque 422.
        """
        if not isinstance(request, Request):
            raise StructuredDataError(
                f'request must be a Request, got {type(request).__name__}',
                code='INVALID_REQUEST',
            )
        spec = self._spec(model)
        adapter = self._adapters.get(spec.model_id)
        if adapter is None:
            adapter = _adapter_for(spec)
            self._adapters[spec.model_id] = adapter

        fields = (
            tuple(outputs)
            if outputs is not None
            else adapter.default_output(request)
        )
        if not fields:
            raise StructuredDataError(
                'outputs must name at least one field',
                code='INVALID_REQUEST',
            )
        if isinstance(outputs, (str, bytes)):
            raise StructuredDataError(
                f'outputs must be a sequence of field names, not a single '
                f'string; pass [{outputs!r}]',
                code='INVALID_REQUEST',
            )
        check_request_against_model(request, spec, fields)
        return adapter, fields

    @staticmethod
    def _spec(model: str | Model) -> ModelSpec:
        return spec_for(model)

    def close(self) -> None:
        r"""Release pooled connections and retire this client.

        Adapters are asked to release what they opened first, then the
        transport is closed even if one of them fails -- the transport's pool
        is the one that cannot be rebuilt on demand.
        """
        try:
            error: BaseException | None = None
            for adapter in self._adapters.values():
                try:
                    adapter.close()
                except BaseException as adapter_error:  # noqa: BLE001
                    error = error or adapter_error
            if error is not None:
                raise error
        finally:
            self._transport.close()

    def __enter__(self) -> SDMClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f'SDMClient(url={redact_url(self._transport.url)!r})'


def _report_absent(
    result: PredictionResult, requested: Sequence[str] | None, model: str
) -> PredictionResult:
    r"""Say when the answer does not carry a field the caller named.

    Only for fields the caller named. The defaults are chosen per task kind and
    a model is entitled not to fill one of them; a caller who named a field is
    waiting for a column that is not going to appear.

    Not raised, because the answer is a real answer to the rest of the request
    and the endpoint accepted it. A caller who wants this to be fatal can turn
    warnings into errors; a caller who does not now at least knows.

    It reports what the answer did not contain and does not say why. The reason
    varies and is not knowable from here: `kumo-relational` returns
    `prediction` for a regression asked on its own and omits it when quantiles
    are asked for alongside, so "the model does not produce it for this task
    kind" would be false in exactly the case that prompted this.
    """
    if not requested:
        return result
    absent = result.absent(requested)
    if absent:
        warnings.warn(
            f'{model} answered without {list(absent)}, which this request '
            f'asked for, so no column of to_pandas() will carry it. The '
            f'answer to the rest of the request is unaffected.',
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )
    return result


def _empty_answer(model: str, outputs: Sequence[str]) -> PredictionResult:
    r"""The answer to a request with no rows in it, without asking.

    `Request` allows a zero-row `predict` on purpose: the contract sets no
    minimum and an empty batch inside a loop is a reasonable no-op. Sent, it
    was not a no-op -- the endpoint cannot encode zero rows and answered
    `500 Unexpected server error`, so the one shape documented as harmless was
    the one that failed, and failed opaquely.

    Nothing is asked, because there is nothing to ask about. The result carries
    the columns the request would have produced, which `to_pandas()` already
    knows how to build with no rows in them.
    """
    return PredictionResult(
        model=model, request_id='', items=(), outputs=tuple(outputs)
    )


def _held_back(payload: dict[str, Any], holdout: Iterable[Any] | None) -> int:
    r"""Take the rows a session will score out of the context it pins.

    Says how many went, because a caller who names a population expects it
    gone and a silent nothing would look the same as a silent everything.
    """
    if holdout is None:
        return 0
    keys = frozenset(
        _named_by(key if isinstance(key, (tuple, list)) else [key])
        for key in holdout
    )
    if not keys:
        return 0
    removed = without_the_keys(payload, keys)
    if removed:
        warnings.warn(
            f'{removed} of the rows named as holdout were in the context '
            f'about to be pinned, carrying the answers this session will be '
            f'asked for. They have been taken out of it, so the context is '
            f'that many rows smaller than the request gathered.',
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )
    return removed


def _warn_scoring_what_was_pinned(
    request: Request, pinned: frozenset[Any], over_a_window: bool = False
) -> None:
    r"""Say when a call scores rows the pinned context already answers.

    A session pins the context gathered by the `fit()` request, which excluded
    the rows *that* request named. Every later call names different rows, and
    nothing excluded those, so they sit in the pinned context with their target
    values and the model is asked to predict what it has been shown. Measured
    on a target that is pure noise, a session scored 0.595 where the same rows
    scored one-shot came back 0.455, against a chance rate of 0.5.

    It cannot be fixed here: the context lives on the endpoint and the rows
    cannot be taken out of it. Nor can the caller fix it after the fact -- the
    equivalent of the one-shot remedy is to blank the targets before `fit()`,
    which means knowing every row the session will ever score, and a caller who
    knows that has no reason to open a session. So it is reported, with the two
    things that do work.

    An entity asked about over a window is only given away at the moment it
    was pinned at, so there the moment is part of what is compared. Comparing
    the entity alone called every later call about the same entity a leak,
    which under warnings-as-errors is a working session that stops working.
    """
    if not pinned:
        return
    scored = scored_keys(request, with_anchor=over_a_window)
    shown = scored & pinned
    if not shown:
        return
    warnings.warn(
        f'{len(shown)} of the {len(scored)} rows being scored are in this '
        f"session's pinned context, with their target values, so the model "
        f'is being asked about rows it has already been shown the answers '
        f'for. The context was chosen when fit() ran and cannot be changed '
        f'now. Either clear the target on every row this session may score '
        f'before calling fit(), or use predict(), which excludes the rows it '
        f'scores from the context it sends.',
        category=SDMWarning,
        stacklevel=caller_stacklevel(),
    )


#: What a ranking's parts are asked for: how sure each answer was is what
#: orders the candidates, not which side of the line it fell.
_RANKING_PARTS = ('prediction', 'probabilities')
_RANKING_OUTPUTS = ('prediction', 'rankings')


def _ranked(
    answers: list[PredictionResult],
    rewritten: Decomposition,
    outputs: Sequence[str] | None,
) -> PredictionResult:
    r"""The candidates each entity reached, likeliest first.

    Ordered by how sure the answer to that candidate's question was, which is
    what a ranking is: the questions were asked separately and only their
    order matters between them.

    Each entity keeps the row it was asked about on, so the answer joins back
    onto the frame the caller scored. The candidates are the values the caller
    named, not what they looked like once written into a query.
    """

    from sdm_api.generated.tfm_api import PredictionItem

    # Keyed by the row, not by the entity on it: the same entity may be asked
    # about more than once, at different moments, and folding those together
    # would answer with one row where two were asked and could put the same
    # candidate in it twice.
    scores: dict[Any, list[tuple[float, Any]]] = {}
    asked_about: dict[Any, tuple[int | None, Any]] = {}
    for answer, (_, candidate) in zip(answers, rewritten.parts, strict=True):
        for item in answer.items:
            where = (item.row_index, item.id)
            scores.setdefault(where, []).append(
                (_how_sure(item, candidate), candidate)
            )
            asked_about.setdefault(where, (item.row_index, item.id))

    top = rewritten.top
    wanted = tuple(outputs or _RANKING_OUTPUTS)
    items = []
    for position, (where, scored) in enumerate(scores.items()):
        row, entity = asked_about[where]
        # Sort only on the score: candidates are of the caller's type and two
        # of them need not be comparable to each other.
        scored.sort(key=lambda pair: pair[0], reverse=True)
        chosen = scored[:top] if top else scored
        items.append(
            PredictionItem(
                row_index=position if row is None else row,
                id=entity,
                prediction=(
                    [candidate for _, candidate in chosen]
                    if 'prediction' in wanted
                    else None
                ),
                rankings=(
                    tuple(
                        {'candidate': candidate, 'score': score}
                        for score, candidate in chosen
                    )
                    if 'rankings' in wanted
                    else None
                ),
            )
        )
    return PredictionResult(
        model=answers[0].model if answers else '',
        request_id=answers[0].request_id if answers else '',
        items=items,
        outputs=wanted,
    )


def _how_sure(item: Any, candidate: Any) -> float:
    r"""How sure the endpoint was that this entity reaches this candidate.

    A ranking is nothing but the order of these, so a missing one cannot be
    read as zero: that would sort a candidate the endpoint was confident about
    to the bottom and give back an order that looks like an answer.
    """
    given = item.probabilities or {}
    for spelling in ('true', 'True', 'TRUE', '1'):
        if spelling in given:
            return float(given[spelling])
    raise StructuredDataError(
        f'the endpoint answered the question about candidate {candidate!r} '
        f'without saying how sure it was, and a ranking is only the order of '
        f'those. It gave back {sorted(given) or "no probabilities"}, and this '
        f'client knows a positive class spelled "true", "True", "TRUE" or '
        f'"1".',
        code='INTERNAL_ERROR',
    )


def _top_k_of(asked: str) -> int | None:
    r"""How many a ranking asked for, as the query says."""
    import re

    found = re.search(r'\bRANK\s+TOP\s+(\d+)', asked, flags=re.IGNORECASE)
    return int(found.group(1)) if found else None


def _which_part_failed(
    error: StructuredDataError, rewritten: Decomposition, about: Any
) -> StructuredDataError:
    """Name the candidate or step whose question failed.

    A ranking is asked as one yes-or-no question per candidate, so a candidate
    nothing reached makes its own question single-class and the endpoint
    refuses it for a task the caller never wrote. Which of them it was is
    known here and was not being said, and the answers for the candidates that
    did work are lost with it.
    """
    which = (
        f'candidate {about!r}'
        if rewritten.kind == RANKING
        else f'part {about} of {len(rewritten.parts)}'
    )
    return StructuredDataError(
        f'{error.args[0]} This is the question asked for {which}, one of '
        f'{len(rewritten.parts)} this {rewritten.kind} is asked as.',
        code=getattr(error, 'code', 'INVALID_REQUEST'),
    )


def _warn_of_steps_that_carry_no_explanation(
    answers: list[PredictionResult],
    rewritten: Decomposition,
    requested: Sequence[str] | None,
    model: str,
) -> None:
    """Say which steps of a forecast came back without an explanation.

    Each step is explained on its own, so some can carry one and others not.
    The merged answer holds whatever came back, and a check that asks only
    whether the whole result carries an explanation sees the ones that did and
    says nothing about the ones that did not.
    """
    if not requested or 'explanation' not in requested:
        return
    missing = [
        step
        for (_, step), answer in zip(rewritten.parts, answers, strict=True)
        if answer.explanation is None
    ]
    if not missing or len(missing) == len(answers):
        return
    warnings.warn(
        f'{model} explained {len(answers) - len(missing)} of '
        f'{len(answers)} forecast steps, and step '
        f'{", ".join(str(step) for step in missing)} carries none, so '
        f'the explanation this request asked for is not there for every step.',
        category=SDMWarning,
        stacklevel=caller_stacklevel(),
    )


def _by_step(
    answers: list[PredictionResult], rewritten: Decomposition
) -> PredictionResult:
    r"""What each step holds, in the order the steps come.

    `forecast_step` is the field the contract already carries for this, so a
    reader of the frame gets the series it asked for without knowing that it
    arrived as several answers. An entity keeps its own row across the steps:
    the pair of row and step is what names a point in the series, which is how
    the contract reads it.

    Each step is explained on its own, so the explanations are kept under the
    step they belong to. Dropping them left the result saying it carried an
    explanation while holding none.
    """
    import dataclasses

    items = [
        dataclasses.replace(item, forecast_step=step)
        for (_, step), answer in zip(rewritten.parts, answers, strict=True)
        for item in answer.items
    ]
    by_step = {
        step: answer.explanation
        for (_, step), answer in zip(rewritten.parts, answers, strict=True)
        if answer.explanation is not None
    }
    return PredictionResult(
        model=answers[0].model if answers else '',
        request_id=answers[0].request_id if answers else '',
        items=items,
        outputs=tuple(answers[0].outputs) if answers else (),
        explanation={'by_forecast_step': by_step} if by_step else None,
    )


def _refuse_to_pin_a_rewritten_question(request: Request) -> None:
    r"""A session pins one context, and a rewritten question has several.

    A ranking's parts each carry their own labels -- whether an entity reached
    *this* candidate -- and a forecast's parts each cover their own window. One
    pinned context can only be one of them, so pinning would answer the first
    part and give back something shaped like the whole question. Refusing says
    what to do instead.
    """
    rewritten = request.decomposition
    if rewritten is None:
        return
    raise StructuredDataError(
        f'a {rewritten.kind} is asked as {len(rewritten.parts)} questions, '
        f'each needing the context its own labels come from, and a session '
        f'pins one context. Ask it with predict(), which sends each and puts '
        f'the answers back together.',
        code='INVALID_REQUEST',
    )


def _refuse_outputs_a_ranking_cannot_give(
    outputs: Sequence[str] | None,
) -> None:
    r"""A ranking gives back an order, and the fields that describe one.

    The parts it was asked as are classifications, so their fields are the
    endpoint's; the merged answer's fields are this client's, and asking it
    for one it does not build would otherwise be answered with a result that
    names the field and does not carry it.

    Naming none is refused the way every other request refuses it. A ranking
    was the one shape that read an empty list as *give me the usual*, so the
    one caller who had said what they wanted was the one who did not get it.
    """
    if outputs is not None and not outputs:
        raise StructuredDataError(
            'outputs must name at least one field', code='INVALID_REQUEST'
        )
    unknown = [name for name in outputs or () if name not in _RANKING_OUTPUTS]
    if unknown:
        raise StructuredDataError(
            f'a ranking gives back {", ".join(_RANKING_OUTPUTS)}, and '
            f'{", ".join(repr(name) for name in unknown)} '
            f'{"are" if len(unknown) > 1 else "is"} not among them. The '
            f'per-candidate questions it is asked as are classifications; ask '
            f'one of those directly for the fields a classification gives.',
            code='INVALID_REQUEST',
        )
