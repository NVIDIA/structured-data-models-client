# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""One connection to one NIM, serving every model it can reach.

The model is an argument:

>>> with SDMClient(url='http://localhost:8000') as client:
... result = client.predict(request, model=KumoTabular)

Each client owns its own transport, so several can target different endpoints
or tenants at once, including from several threads: a prediction always goes to
the endpoint and credential of the client that started it.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Sequence
from types import TracebackType
from typing import Any

from sdm_client.base import (
    ModelAdapter,
    ModelCapabilities,
    ModelSpec,
)
from sdm_client.core.transport import Transport, WireTransport, redact_url
from sdm_client.core.warnings import SDMWarning, caller_stacklevel
from sdm_client.decompose import (
    RANKING,
    about_the_part,
)
from sdm_client.errors import (
    StructuredDataError,
    UnknownModelError,
)
from sdm_client.models import MODELS, Model, spec_for
from sdm_client.request import (
    Question,
    Request,
)
from sdm_client.result import PredictionResult
from sdm_client.sanitization import SanitizationReport
from sdm_client.schema import _as_key_tuple
from sdm_client.wire.keys import (
    _named_by,
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

    Returned by:meth:`SDMClient.fit`. The contract's
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
    _sanitization_report: SanitizationReport
    _closed: bool
    _tabular_session: dict[str, Any] | None
    _tabular_session_keys: bytes
    _relational_session: dict[str, Any] | None
    _relational_session_pinned: bytes
    _column_explanations: Any

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
        fitted._sanitization_report = SanitizationReport('not_applicable')
        fitted._closed = False
        fitted._tabular_session = None
        fitted._tabular_session_keys = b''
        fitted._relational_session = None
        fitted._relational_session_pinned = b''
        fitted._column_explanations = None
        return fitted

    @property
    def sanitization_report(self) -> SanitizationReport:
        """Preparation report retained after this fitted handle is closed."""
        return self._sanitization_report

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
        if getattr(self, '_relational_session', None) is not None:
            from sdm_client._rust import frame_sessions

            return frame_sessions.predict(self, request)
        if getattr(self, '_tabular_session', None) is not None:
            from sdm_client._rust import sessions

            return sessions.predict(self, request)
        raise StructuredDataError(
            'this fitted model has no native session state; fit again',
            code='INVALID_CONFIGURATION',
        )

    def close(self) -> None:
        r"""Release the pinned context. Safe to call twice.

        The handle is marked closed only once the endpoint has released the
        session. Marking it first made a failed release permanent: the next
        call returned early and sent nothing, so the one thing the endpoint
        asks for when it answers `429 Session maintenance is already pending;
        retry the request later` was the one thing the handle could not do.
        """
        if getattr(self, '_relational_session', None) is not None:
            from sdm_client._rust import frame_sessions

            frame_sessions.close(self)
            return
        if getattr(self, '_tabular_session', None) is not None:
            from sdm_client._rust import sessions

            sessions.close(self)
            return
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
        is left to the endpoint's own expiry, and `close` can be called
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

    Use it as a context manager, or call:meth:`close`.
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
        :meth:`health_ready`,:meth:`models`, or a prediction.

        Args:
            url: Base URL of the NIM, e.g. ``'http://localhost:8000'``.
            api_key: Sent as ``X-API-Key`` when an authenticating gateway
                fronts the NIM. Refused over plaintext ``http://`` to a
                non-local host.
            verify_ssl: Whether to verify the server's TLS certificate.
            timeout: Seconds to wait for each individual attempt, not for the
                call as a whole.
            max_retries: Transport-level retries of a transient failure (408, 429,
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

        The one place a client's state is set.:meth:`for_databricks_serving`
        builds its client without ``__init__``, so state set only there was
        missing from a serving client until something reached for it. Add
        new state here, not to ``__init__``.
        """
        self._transport: WireTransport = transport
        self._adapters: dict[str, ModelAdapter] = {}
        self._pinned: set[FittedModel] = set()

    @classmethod
    def in_process(cls, *models: Any, device: Any = None) -> SDMClient:
        """Run models locally using the same Rust preparation and result path.

        Pass loaded structured-data-models models, or omit them to load weights
        on first use. The optional ``in-process`` extra requires Python 3.11+.
        ``device`` controls models loaded by this client; supplied models keep
        their placement. Local refusals carry no HTTP status.
        """
        from sdm_client.in_process import InProcessBackend

        client = cls.__new__(cls)
        client._start(InProcessBackend(models, device))
        return client

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
        ... result = client.predict(request, model=KumoRelational)

        A serving endpoint answers one request at a time, so:meth:`predict`
        works and:meth:`fit` is refused: it has no sessions to pin a context
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
        opened with:meth:`for_databricks_serving`.
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
        from sdm_client._rust.capabilities import described

        return described(self._spec(model).model_id)

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
            model: A model handle such as ``KumoTabular``, or a model id.
            outputs: The output fields to ask for. Defaults to what the model
                produces for this task kind.

        Returns:
            Always a:class:`~sdm_client.result.PredictionResult`, never a
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
        return _prediction_handler(request).predict(
            self, request, model, outputs
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
        if not request.parts:
            raise StructuredDataError(
                'a decomposed request must contain at least one question',
                code='INVALID_REQUEST',
            )
        handler = _prediction_handler(request.parts[0])
        try:
            merged = handler.predict(self, request, model, outputs)
        except StructuredDataError as error:
            raise _said_of_the_part(error, request) from error
        if rewritten.kind == RANKING:
            return merged
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

    def _send_body(self, adapter: Any, body: bytes) -> bytes:
        r"""Post one call already written (the core's, on its route), its
        bytes unchanged where the transport sends bytes as they are, as the
        HTTP transport does, so the bytes the core's size limit counted are
        the bytes sent (register PD-2), and give back the answer as
        the bytes the endpoint sent, which the core reads; a
        Databricks serving endpoint's transport posts the envelope the core
        wrote and gives back the endpoint's reply as it came, which the
        core reads too. A transport with no ``predict_body`` answers decoded, which is
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
        known moments with `predict` if those examples are worth more than
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
        spec = self._spec(model)
        if spec.model_id == 'kumo-tabular':
            from sdm_client._rust import sessions

            return sessions.fit(self, request, model, outputs, holdout)
        from sdm_client._rust import frame_sessions

        return frame_sessions.fit(self, request, model, outputs, holdout)

    @staticmethod
    def _spec(model: str | Model) -> ModelSpec:
        return spec_for(model)

    def close(self) -> None:
        r"""Release pooled connections and retire this client.

        Contexts this client pinned are released first, because the transport
        that would release them is about to close and `FittedModel.close`
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


def _prediction_handler(request: Request) -> Any:
    """Choose the native adapter for direct arrays or materialized payloads."""
    if not request._envelopes:
        from sdm_client._rust import direct

        return direct
    from sdm_client._rust import predict

    return predict


def _said_of_the_part(
    error: StructuredDataError, request: Request
) -> StructuredDataError:
    r"""A refusal of a question the core predicted in parts, said of the
    part it happened in, as:meth:`SDMClient._answer_in_parts` says each
    part's: a forecast's step (the core's ``forecast_step``), or a
    ranking's candidate (the core's ``ranking_candidate``), named as the
    caller gave it; a ranking's refusal naming candidates by their position
    among those given names them as given in its details. A refusal of the
    core says its part in the core's own sentences already, naming the
    candidates as given from the values the predict route holds ( fix
    2), so only its details change; one this client raised itself, which
    the core never saw, is said of its part here."""
    rewritten = request._decomposition
    assert rewritten is not None
    details = getattr(error, 'details', None)
    if not isinstance(details, dict):
        details = {}
    of_the_core = error._sentence is not None
    if rewritten.kind != RANKING:
        step = details.pop('forecast_step', None)
        details.pop('forecast_steps', None)
        if of_the_core:
            return error
        return about_the_part(error, rewritten, step or 1)
    from sdm_client._rust.predict import positions

    by_position = dict(
        zip(
            positions(request),
            (about for _, about in rewritten.parts),
            strict=True,
        )
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
    if position is not None and not of_the_core:
        return about_the_part(error, rewritten, by_position[position])
    return error


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
    if request._envelopes:
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
    the columns the request would have produced, which `to_pandas` already
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
    _report_held_back(removed, len(keys))
    return removed


def _report_held_back(removed: int, named: int) -> None:
    """Present the selected holdout counts without choosing which rows to drop."""
    if not named:
        return
    if not removed:
        warnings.warn(
            f'none of the {named} keys named as holdout matched a row of '
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


def _report_pinned_overlap(shown: int, total: int) -> None:
    """Present overlap already computed by the active implementation."""
    if not shown:
        return
    warnings.warn(
        f'{shown} of the {total} rows being scored are in this '
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
#: What can still rank a candidate whose sampled context holds one answer.


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
