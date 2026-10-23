# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Bridge code: on the core route, predict's host-held rules (design,
# "Host-held semantics"): the model and output checks, the empty answer,
# the merge of the parts and the notice of an absent output; their use on
# the core route is deleted in Phase 1 by T1.28b, and the old route's own
# goes in Phase 5.

r"""One connection to one NIM, serving every model it can reach.

The model is an argument:

>>> with SDMClient(url='http://localhost:8000') as client:
...     result = client.predict(request, model=KumoTabular())

Each client owns its own transport, so several can target different endpoints
or tenants at once, including from several threads: a prediction always goes to
the endpoint and credential of the client that started it.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import TracebackType
from typing import Any

from sdm_client.base import (
    ModelAdapter,
    ModelCapabilities,
    ModelSpec,
    check_request_against_model,
    describe,
)
from sdm_client.core.transport import Transport, WireTransport, redact_url
from sdm_client.core.warnings import SDMWarning, caller_stacklevel
from sdm_client.decompose import (
    RANKING,
    Decomposition,
    about_the_part,
)
from sdm_client.errors import (
    StructuredDataError,
    UnknownModelError,
    UnsupportedRequestError,
)
from sdm_client.models import MODELS, Model, spec_for
from sdm_client.request import (
    Question,
    Request,
    _envelope_rows,
    warn_of_rows_scored_from_their_own_answers,
)
from sdm_client.result import PredictionResult
from sdm_client.schema import _as_key_tuple
from sdm_client.wire.keys import (
    _named_by,
    keys_in,
    scored_rows,
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
    answer needs the task and the schema, where the key an id stands for is
    named. The rows have been uploaded and live on the endpoint, so
    keeping a second encoded copy for the lifetime of the handle would hold a
    whole dataset for nothing.

    It is not constructed directly. What it is built from is how the client
    talks to a model and what the session pinned, none of which is public.
    """

    _client: SDMClient
    _adapter: ModelAdapter
    _session_id: str
    _outputs: tuple[str, ...]
    _named_outputs: tuple[str, ...] | None
    _pinned_keys: frozenset[Any]
    _pinned_over_a_window: bool
    _held_back: int
    _pinned: dict[str, Any]
    _question: Question | None
    _closed: bool

    def __init__(self) -> None:
        raise TypeError(
            'a FittedModel is returned by SDMClient.fit(), not constructed '
            'directly'
        )

    @classmethod
    def _open(
        cls,
        client: SDMClient,
        adapter: ModelAdapter,
        session_id: str,
        outputs: tuple[str, ...],
        pinned: dict[str, Any] | None = None,
        named_outputs: Sequence[str] | None = None,
        pinned_keys: frozenset[Any] = frozenset(),
        pinned_over_a_window: bool = False,
        held_back: int = 0,
        question: Question | None = None,
    ) -> FittedModel:
        r"""The handle for a session ``SDMClient.fit`` has just created."""
        fitted = cls.__new__(cls)
        fitted._client = client
        fitted._adapter = adapter
        fitted._session_id = session_id
        fitted._outputs = outputs
        fitted._named_outputs = (
            None if named_outputs is None else tuple(named_outputs)
        )
        fitted._pinned_keys = pinned_keys
        fitted._pinned_over_a_window = pinned_over_a_window
        fitted._held_back = held_back
        fitted._pinned = pinned or {}
        fitted._question = question
        fitted._closed = False
        return fitted

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def model(self) -> str:
        return self._adapter.spec.model_id

    def predict(self, request: Request) -> PredictionResult:
        r"""Score ``request``'s rows against the pinned context.

        Only the rows travel: the context, schema and task were pinned when
        this handle was created. Reading the answer needs both halves because
        the key an id stands for is named in the pinned schema. The
        adapter is handed the whole request while the wire gets only the half
        the contract allows on a scoring call.
        """
        _require_a_request(request)
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
        asked = getattr(request, '_question', None)
        if self._question is not None and asked is not None:
            _refuse_a_question_the_session_did_not_pin(self._question, asked)
        if request.nothing_to_score:
            return _empty_answer(self.model, self._outputs, request)
        _warn_scoring_what_was_pinned(
            request,
            self._pinned_keys,
            self._pinned_over_a_window,
            _pinned_dtypes(self._pinned),
        )
        parts = _numbered_across_calls(
            self._adapter.session_prediction_plan(
                request, self._outputs, self._pinned
            ),
            lambda payload: self._adapter.normalize(
                self._client._transport.session_predict(
                    self._session_id, payload
                ),
                self._outputs,
                {**self._pinned, **payload},
            ),
        )
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
        pinned = getattr(self._client, '_pinned', None)
        if pinned is not None:
            pinned.discard(self)

    def __enter__(self) -> FittedModel:
        return self

    def __exit__(self, *exc: object) -> None:
        r"""Release the context without masking an active block exception.

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
                502, 503, 504, or a dropped connection) with exponential
                backoff. A 500 is not retried: the payload and the model are
                the same on every attempt. ``POST /v1/sessions`` is excluded:
                a re-sent create would orphan a pinned context on the
                endpoint.
        """
        self._start(
            Transport(
                url,
                api_key,
                verify_ssl=verify_ssl,
                timeout=timeout,
                max_retries=max_retries,
            )
        )

    def _start(self, transport: WireTransport) -> None:
        r"""Set everything a client holds, given what carries its requests.

        The one place a client's state is set. :meth:`for_databricks_serving`
        builds its client without ``__init__``, so state set only there was
        missing from a serving client until something reached for it. Add
        new state here, not to ``__init__``.
        """
        self._transport: WireTransport = transport
        self._adapters: dict[str, ModelAdapter] = {}
        self._pinned: set[FittedModel] = set()

    @classmethod
    def for_databricks_serving(
        cls,
        endpoint: str,
        *,
        workspace_client: Any | None = None,
        timeout: float | None = None,
    ) -> SDMClient:
        r"""Open a client against a Databricks Model Serving endpoint.

        The model is served by name inside a workspace, so there is no URL and
        no key: calls go through the Databricks SDK with the workspace's own
        identity, which in a notebook is the notebook's.

        >>> with SDMClient.for_databricks_serving('kumo-relational') as client:
        ...     result = client.predict(request, model=KumoRelational())

        A serving endpoint answers one request at a time, so :meth:`predict`
        works and :meth:`fit` is refused: it has no sessions to pin a context
        in.

        Args:
            endpoint: The serving endpoint's name, not its URL. A URL is
                refused without being repeated, since one can carry a token.
            workspace_client: A ``databricks.sdk.WorkspaceClient`` to call
                through, for a workspace or credential other than the ambient
                one. A ``WorkspaceClient`` itself is called through a copy
                that names ``structured-data-models-client`` in the
                User-Agent, so it is not changed; a subclass or test double
                is used as given, without the name.
            timeout: Seconds one attempt may take, and the longest the SDK
                keeps retrying. When omitted, the ambient workspace client
                gets 300, far above a NIM's default because a serving
                endpoint that has scaled to zero takes minutes to start, and
                a passed ``workspace_client`` keeps its own ``Config``
                timeouts. Given with a ``WorkspaceClient``, it is set on the
                copy, never on the caller's client; with a subclass or test
                double, which cannot be copied, it is refused.

        Raises:
            MissingExtraError: If ``databricks-sdk`` is not installed; install
                ``structured-data-models-client[databricks-serving]``.
            StructuredDataError: With code ``INVALID_CONFIGURATION``, if
                ``endpoint`` is not a name, ``timeout`` is not a positive
                number of seconds or is passed with a ``workspace_client``
                that cannot be copied, or the ambient configuration cannot
                authenticate.
        """
        from sdm_client.core.databricks_serving import (
            DatabricksServingTransport,
        )

        client = cls.__new__(cls)
        client._start(
            DatabricksServingTransport(
                endpoint, workspace_client, timeout=timeout
            )
        )
        return client

    @property
    def url(self) -> str:
        r"""Where this client sends requests.

        A NIM's base URL, or ``'databricks-serving:<name>'`` for a client
        opened with :meth:`for_databricks_serving`.
        """
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
        live connection, and it is not the endpoint's own answer. Use
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
            Always a :class:`~sdm_client.result.PredictionResult`, never a
            frame-or-explanation union.

        Raises:
            UnknownModelError: If ``model`` is not one the contract carries.
            UnsupportedRequestError: If the model cannot answer this request.
                Raised before anything is sent.
            NimRequestError: If the endpoint answered with an error.
        """
        _require_a_request(request)
        _require_output_names(outputs)
        if request.parts is not None:
            return self._answer_in_parts(request, model, outputs)
        # On the core route the model and output checks, the answer to a
        # request with no rows, the context's checks, the merge and the
        # notice of an absent output are the core's (T1.28b).
        on_the_core = _predicts_on_the_core(request)
        if on_the_core is not None:
            return on_the_core.predict(self, request, model, outputs)
        adapter, fields = self._prepare(request, model, outputs)
        if request.nothing_to_score:
            return _empty_answer(adapter.spec.model_id, fields, request)
        # Asked here, where both halves of the request travel together, which
        # is the only call at which the pairing is a leak.
        warn_of_rows_scored_from_their_own_answers(request)
        # Every call's rows are kept. A request past the endpoint's per-call
        # cap is split by the adapter, and each call answers about its own
        # rows, so returning only the last would drop the earlier batches.
        parts = _numbered_across_calls(
            adapter.prediction_plan(request, fields),
            lambda payload: adapter.normalize(
                self._send(adapter, payload), fields, payload
            ),
        )
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
        as the answer that was actually asked for: an order over candidates
        or a value per step, rather than handed back as the several answers
        it took.
        """
        rewritten = request._decomposition
        assert rewritten is not None
        ranking = rewritten.kind == RANKING
        on_the_core = (
            None
            if not request.parts
            or getattr(request.parts[0], '_core', None) is None
            else _predicts_on_the_core(request.parts[0])
        )
        if on_the_core is not None:
            # A forecast or a ranking the core gathered, which the core
            # predicts part by part and puts together (T1.36 fix 21, T2.8),
            # its refusals said of the part they name, as each part's is
            # below.
            try:
                merged = on_the_core.predict(self, request, model, outputs)
            except StructuredDataError as error:
                raise _said_of_the_part(error, request) from error
            if ranking:
                return merged
            return _report_absent(merged, outputs, spec_for(model).model_id)
        unscored: frozenset[int] = frozenset()
        if ranking:
            _refuse_outputs_a_ranking_cannot_give(outputs)
            unscored = self._candidates_no_context_teaches(
                request, model, rewritten
            )
        # A ranking needs how sure each answer was, which is what orders the
        # candidates, and the fields it gives back are the merge's own. A step
        # of a forecast is an ordinary regression, so the caller's fields are
        # what each part is asked for.
        wanted = _RANKING_PARTS if ranking else outputs
        answers: list[PredictionResult] = []
        answered: list[int] = []
        for position, (part, (_, about)) in enumerate(
            zip(request.parts or (), rewritten.parts, strict=True)
        ):
            if position in unscored:
                continue
            answered.append(position)
            try:
                answers.append(self.predict(part, model=model, outputs=wanted))
            except Exception as error:
                answered.pop()
                _carry_the_steps_already_paid_for(
                    error, answers, rewritten, answered, ranking
                )
                if not isinstance(error, StructuredDataError):
                    raise
                raise about_the_part(error, rewritten, about) from error
        if ranking:
            parts = rewritten.parts
            scored = [part for i, part in enumerate(parts) if i not in unscored]
            return _ranked(
                answers,
                dataclasses.replace(rewritten, parts=tuple(scored)),
                outputs,
                unscored=[parts[i][1] for i in sorted(unscored)],
            )
        merged = _by_step(answers, rewritten)
        _warn_of_steps_that_carry_no_explanation(
            answers, rewritten, outputs, spec_for(model).model_id
        )
        return _report_absent(merged, outputs, spec_for(model).model_id)

    def _candidates_no_context_teaches(
        self,
        request: Request,
        model: str | Model,
        rewritten: Decomposition,
    ) -> frozenset[int]:
        r"""Which of a ranking's parts ask about a one-class candidate.

        The endpoint refuses a classification whose context holds one class,
        so such a candidate cannot be scored. Checking every part before any
        call sends none for those, names them all at once, and refuses a
        ranking with nothing left before it spends a call.

        A candidate is scored or not for the whole ranking, because each
        candidate is one call: a part is never batched, and a ranking over
        more entities than one call scores is refused when it is gathered.
        Were a part ever split, the engine would sample a context per batch,
        and one cold batch would drop the candidate from every row rather
        than from its own, since ``unscored_candidates`` names a candidate
        for the whole result rather than per row.
        """
        positions = []
        for position, (part, (_, about)) in enumerate(
            zip(request.parts or (), rewritten.parts, strict=True)
        ):
            try:
                adapter, fields = self._prepare(part, model, _RANKING_PARTS)
                for _ in adapter.prediction_plan(part, fields):
                    pass
            except StructuredDataError as error:
                details = getattr(error, 'details', None) or {}
                if details.get('classes_in_context') != 1:
                    raise about_the_part(error, rewritten, about) from error
                positions.append(position)
        if not positions:
            return frozenset()
        cold = [rewritten.parts[position][1] for position in positions]
        many = len(cold) > 1
        named = ', '.join(repr(candidate) for candidate in cold[:-1])
        if len(cold) == len(rewritten.parts):
            which = (
                f'the contexts gathered for candidates {named} and '
                f'{cold[-1]!r} each hold'
                if many
                else f'the context gathered for candidate {cold[0]!r} holds'
            )
            raise UnsupportedRequestError(
                f'{which} one answer only, and a classification needs at '
                f'least two classes to learn from, so there is nothing to '
                f'rank and nothing was sent. {_SCORE_THE_UNSCORED}',
                details={
                    'model': spec_for(model).model_id,
                    'candidates': cold,
                    'classes_in_context': 1,
                },
            )
        which = (
            f'candidates {named} and {cold[-1]!r} are'
            if many
            else f'candidate {cold[0]!r} is'
        )
        them = 'them' if many else 'it'
        warnings.warn(
            f'{which} left out of this ranking, unscored: the context '
            f'sampled for {"each" if many else "it"} holds one answer only, '
            f'nearly always because no entity in it linked to the candidate '
            f'in the window, and a classification needs two to learn from. '
            f'No call was sent for {them}, so each row ranks only the rest, '
            f"and the result's unscored_candidates names {them}. "
            f'{_SCORE_THE_UNSCORED}',
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )
        return frozenset(positions)

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

    def _send_body(self, adapter: Any, body: bytes) -> bytes:
        r"""Post one call already written (the core's, on its route), its
        bytes unchanged where the transport sends bytes as they are, as the
        HTTP transport does, so the bytes the core's size limit counted are
        the bytes sent (register PD-2, T1.28a), and give back the answer as
        the bytes the endpoint sent, which the core reads (T2.14b); a
        transport that writes its own body (a Databricks serving endpoint's
        envelope) is handed the call decoded and gives back the answer's
        text. A transport with no ``predict_body`` answers decoded, which is
        written back as JSON text. A refusal the adapter explains as
        :meth:`_send` does."""
        send = getattr(self._transport, 'predict_body', None)
        try:
            if send is not None:
                return bytes(send(body))
            return json.dumps(
                self._transport.predict(json.loads(body))
            ).encode()
        except StructuredDataError as error:
            explained = adapter.explain_error(error, json.loads(body))
            if explained is error:
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
        with their answers. Unlike the one-shot path, there is nothing the
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
            StructuredDataError: If the endpoint does not serve sessions,
                with code ``'SESSIONS_UNSUPPORTED'``.
        """
        if not getattr(self._transport, 'serves_sessions', True):
            raise StructuredDataError(
                f'{self._transport.url} cannot pin a context: it answers one '
                f'request at a time. Use predict(), which sends the context '
                f'with every call',
                code='SESSIONS_UNSUPPORTED',
            )
        _require_a_request(request)
        _require_output_names(outputs)
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
        fitted = FittedModel._open(
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
            question=request._question,
        )
        self._pinned.add(fitted)
        return fitted

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
        _require_a_request(request)
        _require_output_names(outputs)
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

        Contexts this client pinned are released first, because the transport
        that would release them is about to close and `FittedModel.close()`
        is the only call that can. Adapters follow, then the transport itself
        even if one of them fails. Its pool is the one that cannot be
        rebuilt on demand.
        """
        try:
            error: BaseException | None = None
            for fitted in list(self._pinned):
                try:
                    fitted.close()
                except BaseException as session_error:  # noqa: BLE001
                    error = error or session_error
            self._pinned.clear()
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


def _predicts_on_the_core(request: Request) -> Any | None:
    r"""The core's ``predict`` bridge, when this call runs on the core.

    A request materialized over a DuckDB or SQLite source predicts on the
    core when the routing seam forces its paths there (T1.11d, T2.2); every
    other call, and every call by default, takes the old route. The bridge refuses, as
    the seam's own failure, a request the core did not materialize.
    """
    schema = getattr(request, 'schema', None)
    source = getattr(schema, 'source', None)
    if not request.payloads:
        return None
    from sdm_client import _routing

    if not _routing.on_the_core(source, _routing.PREDICT):
        return None
    from sdm_client._rust import predict as on_the_core

    return on_the_core


def _said_of_the_part(
    error: StructuredDataError, request: Request
) -> StructuredDataError:
    r"""A refusal of a question the core predicted in parts, said of the
    part it happened in, as :meth:`SDMClient._answer_in_parts` says each
    part's: a forecast's step (the core's ``forecast_step``), or a
    ranking's candidate (the core's ``ranking_candidate``), named as the
    caller gave it; a ranking's refusal naming candidates by their position
    among those given names them as given in its details. Its sentence
    names them as given already, from the values the predict route holds
    (T3.17)."""
    rewritten = request._decomposition
    assert rewritten is not None
    details = getattr(error, 'details', None)
    if not isinstance(details, dict):
        details = {}
    if rewritten.kind != RANKING:
        step = details.pop('forecast_step', None)
        return about_the_part(error, rewritten, step or 1)
    ranking = None if request._core is None else request._core.ranking
    positions = () if ranking is None else ranking.positions
    by_position = dict(
        zip(positions, (about for _, about in rewritten.parts), strict=True)
    )
    if isinstance(details.get('candidates'), list):
        details['candidates'] = [
            by_position.get(at, at) for at in details['candidates']
        ]
    if 'candidate' in details:
        details['candidate'] = by_position.get(
            details['candidate'], details['candidate']
        )
    position = details.pop('ranking_candidate', None)
    details.pop('ranking_parts', None)
    if position is not None:
        return about_the_part(error, rewritten, by_position[position])
    # The core's sentence already names the candidates as given, the values
    # the predict route holds for it (T3.17).
    return error


def _numbered_across_calls(
    plan: Iterable[dict[str, Any]],
    answer: Callable[[dict[str, Any]], PredictionResult],
) -> list[PredictionResult]:
    r"""Answer each call, numbering its rows from where the last call's ended.

    The endpoint numbers every call's rows from zero, so a request sent in
    several calls would otherwise give each ``row_index`` once per call, and
    ``to_pandas()``, which orders by it, would interleave the batches.

    Every call is counted before the first is sent, because a call whose rows
    cannot be counted leaves the ones after it unnumberable, and finding that
    out between two sends spends calls on an answer that cannot be returned.

    A call that fails ends the request, and the answers already paid for are
    carried on the error as ``details['answered']``, with the rows they cover
    as ``details['rows_answered']``: the caller can keep them and ask again
    for the rest rather than resend the whole population.
    """
    calls = list(plan)
    counted = [_envelope_rows(payload, 'predict') for payload in calls]
    if any(rows is None for rows in counted[:-1]):
        raise StructuredDataError(
            'a request sent in several calls has a call whose rows could not '
            'be counted, so the rows of the calls after it cannot be numbered',
            code='INTERNAL_ERROR',
        )
    answers: list[PredictionResult] = []
    start = 0
    for payload, rows in zip(calls, counted, strict=True):
        try:
            answers.append(_numbered_from(answer(payload), start))
        except Exception as error:
            _carry_the_answers_already_paid_for(error, answers, start)
            raise
        start += rows or 0
    return answers


def _carry_the_steps_already_paid_for(
    error: BaseException,
    answers: list[PredictionResult],
    rewritten: Decomposition,
    answered: Sequence[int],
    ranking: bool,
) -> None:
    r"""Record on ``error`` the parts answered before the one that failed.

    A forecast's steps are kept as the answer they would have been: numbered
    by step, since a step's rows are numbered from its own zero and only the
    step tells two of them apart.

    A ranking's are counted and not kept. Its parts are not an answer until
    they are all in: what a caller asked for is an order over the candidates,
    and an order over some of them is a different claim, not a partial one.
    """
    if not answers:
        return
    held = getattr(error, 'details', None)
    error.details = {  # type: ignore[attr-defined]
        **(held if isinstance(held, dict) else {}),
        'parts_answered': len(answers),
        **(
            {}
            if ranking
            else {
                'answered': _by_step(
                    answers,
                    dataclasses.replace(
                        rewritten,
                        parts=tuple(rewritten.parts[at] for at in answered),
                    ),
                )
            }
        ),
    }


def _carry_the_answers_already_paid_for(
    error: BaseException, answers: list[PredictionResult], rows: int
) -> None:
    r"""Record on ``error`` the calls that were answered before it.

    A request sent in fifty calls that fails on the thirty-seventh has
    thirty-six answers the endpoint was already paid for. Dropping them makes
    the caller buy them again; the error keeps its own type and code, so
    nothing branching on either has to know they are there.

    The same holds of a question asked in parts, where what was answered is
    whole parts rather than batches of one. ``about_the_part`` copies the
    error's ``__dict__``, so what is recorded here survives being said of the
    part it happened in.
    """
    if not answers:
        return
    # Not every failure is this client's own: the Databricks transport lets
    # its OAuth refresh raise, and those calls were paid for too.
    held = getattr(error, 'details', None)
    error.details = {  # type: ignore[attr-defined]
        **(held if isinstance(held, dict) else {}),
        'answered': PredictionResult.merge(answers),
        'calls_answered': len(answers),
        'rows_answered': rows,
    }


def _numbered_from(result: PredictionResult, start: int) -> PredictionResult:
    r"""``result`` with each row numbered ``start`` places further on."""
    if start:
        result.items = tuple(
            dataclasses.replace(item, row_index=item.row_index + start)
            if item.row_index is not None
            else item
            for item in result.items
        )
    return result


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


def _names_its_rows(request: Request) -> bool:
    r"""Whether answers to this request will say which row each one is about.

    An ``id`` is the entity a row names, so it is there when the rows being
    scored carry one. On the route that builds its own frames that needs both
    a key in the schema and every one of its columns in the half being sent:
    asking only the schema gave an empty answer an ``id`` where a full one had
    none. On the gathered route the entity is what was asked for, and the
    envelope spells it as a transport reference rather than as the schema's
    key. Asking the same question there answered no about requests whose
    every answer names its entity.
    """
    if request.payloads:
        # A gathered request is about entities by construction: the envelope
        # carries an entity reference for every row it scores, and the adapter
        # names each answer with the entity it asked about.
        return True
    table = request.entity_table
    if table is None:
        return False
    try:
        key = _as_key_tuple(request.schema[table].primary_key)
        if not key:
            return False
        scored = request.predict.instance_table.columns
        return all(name in scored for name in key)
    except Exception:  # noqa: BLE001 - a schema too partial to ask is a no
        return False


def _empty_answer(
    model: str, outputs: Sequence[str], request: Request | None = None
) -> PredictionResult:
    r"""The answer to a request with no rows in it, without asking.

    `Request` allows a zero-row `predict` on purpose: the contract sets no
    minimum and an empty batch inside a loop is a reasonable no-op. Sent, it
    was not a no-op. The endpoint cannot encode zero rows and answered
    `500 Unexpected server error`, so the one shape documented as harmless was
    the one that failed, and failed opaquely.

    Nothing is asked, because there is nothing to ask about. The result carries
    the columns the request would have produced, which `to_pandas()` already
    knows how to build with no rows in them.
    """
    built = PredictionResult(
        model=model, request_id='', items=(), outputs=tuple(outputs)
    )
    built._names_its_rows = request is None or _names_its_rows(request)
    return built


def _held_back(payload: dict[str, Any], holdout: Iterable[Any] | None) -> int:
    r"""Take the rows a session will score out of the context it pins.

    Says how many went, because a caller who names a population expects it
    gone and a silent nothing would look the same as a silent everything.
    """
    if holdout is None:
        return 0
    if isinstance(holdout, (str, bytes)):
        raise StructuredDataError(
            f'holdout names the rows this session may score, and {holdout!r} '
            f'is one string rather than a list of keys; pass [{holdout!r}] if '
            f'that is the key you mean',
            code='INVALID_REQUEST',
        )
    keys = frozenset(
        _named_by(key if isinstance(key, (tuple, list)) else [key])
        for key in holdout
    )
    if not keys:
        return 0
    removed = without_the_keys(payload, keys)
    if not removed:
        warnings.warn(
            f'none of the {len(keys)} keys named as holdout matched a row of '
            f'the context about to be pinned, so nothing was held back. The '
            f'keys are compared as the schema spells them, so a key given as '
            f'a different type matches nothing. Check they are the same '
            f'kind of value as the entity column holds.',
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )
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


def _pinned_dtypes(pinned: Mapping[str, Any]) -> dict[str, str] | None:
    r"""The dtypes a session pinned, read back off the envelope it sent.

    The pinned keys were written with these, so the scored ones have to be
    read with them or the same entity has two names.
    """
    columns = ((pinned.get('schema') or {}).get('instance_table') or {}).get(
        'columns'
    )
    if not isinstance(columns, Mapping):
        return None
    return {
        name: str(described['dtype'])
        for name, described in columns.items()
        if isinstance(described, Mapping) and described.get('dtype')
    }


def _warn_scoring_what_was_pinned(
    request: Request,
    pinned: frozenset[Any],
    over_a_window: bool = False,
    dtypes: Mapping[str, str] | None = None,
) -> None:
    r"""Say when a call scores rows the pinned context already answers.

    A session pins the context gathered by the `fit()` request, which excluded
    the rows *that* request named. Every later call names different rows, and
    nothing excluded those, so they sit in the pinned context with their target
    values and the model is asked to predict what it has been shown. Measured
    on a target that is pure noise, a session scored 0.595 where the same rows
    scored one-shot came back 0.455, against a chance rate of 0.5.

    It cannot be fixed here: the context lives on the endpoint and the rows
    cannot be taken out of it. Nor can the caller fix it after the fact. The
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
    # Rows, not the keys among them: two rows asking about one entity are two
    # rows being scored, and a set had already lost one of them before either
    # number was taken.
    scored = scored_rows(request, with_anchor=over_a_window, dtypes=dtypes)
    shown = sum(1 for named in scored if named in pinned)
    if not shown:
        return
    warnings.warn(
        f'{shown} of the {len(scored)} rows being scored are in this '
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
#: What can still rank a candidate whose sampled context holds one answer.
_SCORE_THE_UNSCORED = (
    'Gather a wider context with num_context_samples if other entities in '
    'the data answer differently, or ask over a table of (entity, '
    'candidate) pairs, which scores every candidate from its features.'
)


def _ranked(
    answers: list[PredictionResult],
    rewritten: Decomposition,
    outputs: Sequence[str] | None,
    unscored: Sequence[Any] = (),
) -> PredictionResult:
    r"""The candidates each entity reached, likeliest first.

    Ordered by how sure the answer to that candidate's question was, which is
    what a ranking is: the questions were asked separately and only their
    order matters between them.

    Each entity keeps the row it was asked about on, so the answer joins back
    onto the frame the caller scored. The candidates are the values the caller
    named, not what they looked like once written into a query.

    ``rewritten`` holds only the parts that were asked. The candidates that
    were not are carried beside the ranking, never in it, since any score
    given to them would be invented.
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
        unscored_candidates=unscored,
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
    arrived as several answers. It is a column of every such answer, so an
    empty one carries it too. An entity keeps its own row across the steps:
    the pair of row and step is what names a point in the series, which is how
    the contract reads it.

    Each step is explained on its own, so the explanations are kept under the
    step they belong to. Dropping them left the result saying it carried an
    explanation while holding none.
    """
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
    built = PredictionResult(
        model=answers[0].model if answers else '',
        request_id=answers[0].request_id if answers else '',
        items=items,
        outputs=tuple(answers[0].outputs) if answers else (),
        explanation={'by_forecast_step': by_step} if by_step else None,
    )
    # `forecast_step` is put on the items above, so with no items there is
    # nothing to read it from and an empty forecast lost the column that says
    # which step an answer is about, just as `id` loses the column that
    # says which row.
    built._names_its_step = True
    built._names_its_rows = all(part._names_its_rows for part in answers)
    return built


_WHAT_DIFFERS = (
    ('query', 'the query'),
    ('kind', 'the task kind'),
    ('target', 'the target column'),
    ('dtype', "the target column's dtype"),
    ('entity_table', 'the entity table'),
)


def _refuse_a_question_the_session_did_not_pin(
    pinned: Question, asked: Question
) -> None:
    r"""Require a scoring call to use the pinned question.

    Said field by field: the rest of the request is identical, so printing both
    whole does not show which one differs.
    """
    if pinned == asked:
        return
    for field_name, described in _WHAT_DIFFERS:
        mine = getattr(pinned, field_name)
        theirs = getattr(asked, field_name)
        if mine != theirs:
            said = f'{described} pinned is {mine!r} and this asks {theirs!r}'
            break
    else:
        said = (
            'the context pinned is not the one this request carries, and the '
            'context is what the answer is learnt from'
        )
    raise StructuredDataError(
        f'this session pinned a different question: {said}. Only the rows '
        f'travel on a scoring call, so the pinned question is the one that '
        f'would be answered. Fit again to ask something else',
        code='INVALID_REQUEST',
    )


def _require_a_request(request: Any) -> None:
    """Refuse anything that is not a request, before reading a field off it.

    `errors.py` promises that `except StructuredDataError` catches everything
    this client reports. `_prepare` held the right message and ran after the
    first attribute was dereferenced, so the wrong type escaped as an
    `AttributeError` naming a field the caller never heard of.

    Asked at the entry rather than in `_prepare`, because a rewritten question
    is answered in parts and never reaches it.
    """
    if not isinstance(request, Request):
        raise StructuredDataError(
            f'request must be a Request, got {type(request).__name__}; build '
            f'one with Request(context=..., predict=...) or '
            f'PredictiveQuery(...).materialize(...)',
            code='INVALID_REQUEST',
        )


def _require_output_names(outputs: Any) -> None:
    """Refuse an `outputs` that is not a sequence of field names."""
    if outputs is None:
        return
    if isinstance(outputs, str):
        raise StructuredDataError(
            f'outputs must be a sequence of field names, not a single '
            f'string; pass [{outputs!r}]',
            code='INVALID_REQUEST',
        )
    if not isinstance(outputs, Sequence) or not all(
        isinstance(name, str) for name in outputs
    ):
        raise StructuredDataError(
            f'outputs must be a sequence of field names, got '
            f'{type(outputs).__name__}',
            code='INVALID_REQUEST',
        )


def _refuse_to_pin_a_rewritten_question(request: Request) -> None:
    r"""A session pins one context, and a rewritten question has several.

    A ranking's parts each carry their own labels for whether an entity reached
    *this* candidate. A forecast's parts each cover their own window. One
    pinned context can only be one of them, so pinning would answer the first
    part and give back something shaped like the whole question. Refusing says
    what to do instead.
    """
    rewritten = request._decomposition
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
