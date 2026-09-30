# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import math
import os
import re
import time
import uuid
import warnings
from collections import defaultdict
from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Literal, overload

import numpy as np
import pandas as pd
from predictive_query.ast import (
    Aggregation,
    Column,
    Condition,
    Constant,
    Join,
    LogicalOperation,
)
from predictive_query.query import (
    QueryType,
    ValidatedPredictiveQuery,
)
from predictive_query.schema.task import TaskType
from predictive_query.schema.typing import (
    AggregationType,
    ProblemType,
    Stype,
)
from requests.exceptions import RequestException, Timeout

from sdm_relational_engine._version import __version__
from sdm_relational_engine.client.client import NimClient
from sdm_relational_engine.client.rfm import RFMAPI
from sdm_relational_engine.exceptions import HTTPException, NimFailureError
from sdm_relational_engine.rfm import Graph, TaskTable
from sdm_relational_engine.rfm.base import DataBackend, Sampler, composite_key
from sdm_relational_engine.rfm.base.utils import Timestamp, to_naive_utc
from sdm_relational_engine.rfm.diagnostics import GraphSanitizationReport
from sdm_relational_engine.rfm.explain.result import (
    ExplainConfig,
    Explanation,
    extract_explanation,
)
from sdm_relational_engine.rfm.explain.summary import generate_summary
from sdm_relational_engine.rfm.graph_fingerprint import graph_fingerprint
from sdm_relational_engine.rfm.inference.config import (
    ClassificationInferenceConfig,
    InferenceConfig,
    RegressionInferenceConfig,
    RunMode,
)
from sdm_relational_engine.rfm.inference.context import (
    PredictionContext,
    SampledTable,
)
from sdm_relational_engine.rfm.inference.payload import (
    INSTANCE_ID,
    context_size_stats,
    high_cardinality_columns,
    payload_size_bytes,
    predict_request_to_json,
    session_create_payload,
    session_predict_payload,
    validate_payload_table_rows,
)
from sdm_relational_engine.rfm.inference.requests import RFMPredictRequest
from sdm_relational_engine.rfm.inference.telemetry import (
    PredictionRecord,
    describing,
    pending,
    recorded,
)
from sdm_relational_engine.rfm.query.parser import parse_query_locally
from sdm_relational_engine.utils.progress_logger import ProgressLogger

_RANDOM_SEED = 42

_MAX_PRED_SIZE: dict[TaskType, int] = defaultdict(lambda: 1_000)
_MAX_PRED_SIZE[TaskType.TEMPORAL_LINK_PREDICTION] = 200

_MAX_TEST_SIZE: dict[TaskType, int] = defaultdict(lambda: 2_000)
_MAX_TEST_SIZE[TaskType.TEMPORAL_LINK_PREDICTION] = 400

_MAX_CONTEXT_SIZE = {
    RunMode.DEBUG: 100,
    RunMode.FAST: 1_000,
    RunMode.NORMAL: 5_000,
    RunMode.BEST: 10_000,
}

_DEFAULT_NUM_NEIGHBORS = {
    RunMode.DEBUG: [16, 16, 4, 4, 1, 1],
    RunMode.FAST: [32, 32, 8, 8, 4, 4],
    RunMode.NORMAL: [64, 64, 8, 8, 4, 4],
    RunMode.BEST: [64, 64, 8, 8, 4, 4],
}

_MAX_SIZE = 30 * 1024 * 1024
_SIZE_LIMIT_MSG = (
    'Context size exceeds the 30MB limit. {stats}\nPlease '
    'reduce either the number of tables in the graph, their '
    'number of columns (e.g., large text columns), '
    'neighborhood configuration, or the run mode. If none of '
    'this is possible, please create a feature request at '
    "'https://github.com/NVIDIA/structured-data-models-client' if you must go "
    'beyond this for your use-case.'
)

_SESSION_UNSUPPORTED_STATUS = frozenset({404, 405, 501})


def _a_conversion_that_works(value: object) -> str:
    """A suggested conversion, only where it converts.

    `try pd.Timestamp('not a date')` was printed for any string, including the
    ones that make that call raise -- so the remedy was the input wrapped in a
    constructor that cannot take it.
    """
    if not isinstance(value, str):
        return ''
    try:
        converted = pd.Timestamp(value)
    except Exception:
        return ''
    if converted is pd.NaT:
        return ''
    return f'; try pd.Timestamp({value!r})'


def _sessions_unsupported(error: HTTPException) -> bool:
    return error.status_code in _SESSION_UNSUPPORTED_STATUS


def _sessions_disabled_by_env() -> bool:
    r"""Opt-out kill switch: set ``KUMO_RELATIONAL_DISABLE_SESSIONS`` to force the
    stateless per-batch path even for multi-batch jobs.
    """
    value = os.environ.get('KUMO_RELATIONAL_DISABLE_SESSIONS') or ''
    return value.strip().lower() in (
        '1',
        'true',
        'yes',
        'on',
    )


def _no_session_reason(random_seed: int | None) -> str:
    r"""Why a multi-batch run is re-uploading its context per batch.

    Both causes are invisible otherwise: the run still succeeds, and the only
    symptom is that every batch carries the full context instead of the
    prediction rows alone.
    """
    cause = (
        'KUMO_RELATIONAL_DISABLE_SESSIONS is set'
        if random_seed is not None
        else 'random_seed=None re-samples neighborhoods per batch'
    )
    return f'Sessions disabled ({cause}); each batch re-uploads the context'


@dataclass(frozen=True)
class MaterializedPredictionRequest:
    r"""A frozen request record with source-row provenance.

    The payload is the exact mapping passed to the live API and should be
    treated as read-only; nested payload containers are not copied or frozen.
    """

    payload: Mapping[str, Any]
    batch_index: int
    prediction_start: int
    prediction_stop: int
    request_size_bytes: int


@dataclass(frozen=True)
class _GeneratedPredictionRequest:
    materialized: MaterializedPredictionRequest
    entity_ids: tuple[Any, ...]
    instance_ids: tuple[Any, ...]
    entity_dtype: Any
    anchor_times: tuple[Any, ...] | None
    class_dtype: Any = None


@dataclass
class _SessionHandle:
    r"""Mutable session state for one batched prediction run.

    ``id`` is the server session id once created (mutated in place so it stays
    tracked for deletion even if a later call raises); ``active`` flips to
    ``False`` when the NIM turns out not to support sessions.
    """

    id: str | None = None
    active: bool = True


_NIM_UNAVAILABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
# Server-side ceilings, restated here so the request is refused locally with an
# actionable message instead of as an opaque failure from the NIM.
_MAX_HOPS = 6
_MAX_SUBGRAPH_TABLES = 15
# Seconds, raised to the power of the attempt number. Matches the transport's
# own `_RETRY_BACKOFF_FACTOR`; this is the application-level retry above it.
_RETRY_BACKOFF_BASE_SECONDS = 2
_CARDINALITY_RE = re.compile(r'categorical cardinality \d+ exceeds limit (\d+)')

_OPTIMIZABLE_BACKENDS = frozenset({DataBackend.SQLITE, DataBackend.DUCKDB})


def _fingerprint_of(graph: Any) -> str:
    r"""The graph's fingerprint, or an empty string if it cannot be taken.

    A prediction must not fail because its record could not be labelled.
    """
    try:
        return graph_fingerprint(graph)
    except Exception:
        try:
            logging.getLogger('sdm_relational_engine').warning(
                'Graph could not be fingerprinted, so predictions over it '
                'cannot be told apart from ones over a different graph.',
                exc_info=True,
            )
        except Exception:
            # Saying so must not become the failure either, for the same
            # reason the fingerprint itself does not fail the model.
            pass
        return ''


def _first_uncastable(
    values: Sequence[Any], dtype: Any
) -> tuple[int, Any] | None:
    r"""The first ``(position, value)`` that will not cast to ``dtype``.

    Lets a rejected ``indices`` name the element responsible. Passing the whole
    sequence to pandas reports only its own internals -- "int() argument must
    be a string, a bytes-like object or a real number, not 'NoneType'" -- which
    names neither the argument, the offending position, nor the key type the
    entity table actually uses.
    """
    for position, value in enumerate(values):
        try:
            pd.Series([value], dtype=dtype)
        except (TypeError, ValueError):
            return position, value
    return None


def _class_dtype(task: TaskTable, context: PredictionContext) -> Any:
    r"""The dtype the ``CLASS`` column of a prediction frame should carry.

    Class names and ranking ids are string-keyed on the wire, so ``CLASS``
    arrives as :class:`str`. Restoring the dtype of the column it names keeps
    it comparable and joinable, the way ``ENTITY`` already is.
    """
    if task.task_type.is_link_pred:
        table = context.sampled_subgraph.tables[context.entity_table_names[-1]]
        if table.primary_key is None:
            return None
        return table.df[table.primary_key].dtype
    if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return task._context_df[task.target_column.name].dtype
    return None


def _cast_class_column(values: pd.Series, dtype: Any) -> pd.Series:
    # A bool dtype is excluded: every non-empty class name would cast to
    # ``True``, which is worse than leaving the strings alone.
    if dtype is None or pd.api.types.is_bool_dtype(dtype):
        return values
    try:
        return values.astype(dtype)
    except (TypeError, ValueError, OverflowError):
        return values


_MAX_INVALID_PARAMS = 5


def _problem_document(error: Exception) -> dict[str, Any]:
    r"""The RFC-9457 problem document the NIM returned, if it returned one.

    ``HTTPException.detail`` carries the raw response body; a NIM error body is
    a problem document, but a proxy or a partially-deployed NIM may return
    anything at all.
    """
    detail = getattr(error, 'detail', None)
    if isinstance(detail, str):
        try:
            document = json.loads(detail)
        except Exception:
            return {}
        if isinstance(document, dict):
            return document
    return {}


def _invalid_params_summary(document: dict[str, Any]) -> str:
    r"""Render the NIM's per-field validation diagnosis.

    This is the only renderer: the client package passes ``invalid_params``
    through as error details rather than formatting them, so the shape a caller
    sees is decided here.

    The NIM names the exact table, row and column it rejected in
    ``invalid_params``; the top-level ``detail`` is often only "Request
    validation failed.", so dropping this leaves the caller with nothing to act
    on.
    """
    params = document.get('invalid_params')
    if not isinstance(params, list):
        return ''
    entries = []
    for param in params[:_MAX_INVALID_PARAMS]:
        if not isinstance(param, dict):
            continue
        name, reason = param.get('name'), param.get('reason')
        if name and reason:
            entries.append(f'{name}: {reason}')
        elif name or reason:
            entries.append(str(name or reason))
    if not entries:
        return ''
    omitted = len(params) - len(entries)
    more = f' (and {omitted} more)' if omitted > 0 else ''
    return ' ' + '; '.join(entries) + more


def _cardinality_guidance(
    detail: str,
    payload: dict[str, Any] | None,
) -> str:
    r"""Explain how to resolve a categorical-cardinality rejection.

    The NIM reports the offending count and limit but not the column, so the
    columns are recovered from the request that was just refused and the fix is
    spelled out against the worst offender.
    """
    match = _CARDINALITY_RE.search(detail)
    if match is None:
        return ''
    limit = int(match.group(1))
    columns = high_cardinality_columns(payload, limit) if payload else []
    if not columns:
        return (
            " Set the offending column's stype to Stype.ID and retry "
            "('text' does not lift the limit, its tokens are counted too)."
        )
    listed = ', '.join(
        f"'{table}.{column}' holds {count:,}"
        for table, column, count in columns[:3]
    )
    table, column, _ = columns[0]
    return (
        f" {listed}; set graph['{table}']['{column}'].stype = Stype.ID and "
        f"retry ('text' does not lift the limit, its tokens are counted "
        f'too).'
    )


def _nim_failure_error(
    error: Exception,
    explain: bool,
    payload: dict[str, Any] | None = None,
) -> NimFailureError:
    r"""Build a clear, actionable error for a failed NIM prediction call.

    A 5xx response or a dropped connection says the NIM did not complete the
    request, and nothing more: the cause is the server's to report, so its own
    detail is quoted rather than guessed at. Naming GPU memory pressure here
    was a guess, and a measured counter-example had the GPU at one percent
    while six identical retries failed and only a smaller neighbourhood sample
    succeeded -- so both remedies are named, in the order that distinguishes
    them. Explanations are the most GPU-intensive request (they hold the
    model's gradient graph on the device), so their message adds a
    one-at-a-time hint.

    A 4xx is by definition about the request the caller sent, so it is reported
    as such, quoting the NIM's per-field ``invalid_params`` diagnosis without
    inviting a bug report against the client. Only genuinely unclassifiable
    failures keep that invitation.

    A timeout is called out separately: it is the one failure the caller can
    fix from their side, by raising the timeout they configured.
    """
    subject = 'this explanation' if explain else 'this prediction'
    if isinstance(error, Timeout):
        return NimFailureError(
            f'The KumoRelational NIM did not answer {subject} within the configured '
            'timeout. Raise it with SDMClient(url, timeout=...), or retry '
            f'when the NIM is less busy. Original error: {error}',
            transient=True,
        )
    status = getattr(error, 'status_code', None)
    document = _problem_document(error)
    detail = document.get('detail', getattr(error, 'detail', None))
    invalid_params = document.get('invalid_params')
    invalid_params = invalid_params if isinstance(invalid_params, list) else []
    fields = _invalid_params_summary(document)

    if status in _NIM_UNAVAILABLE_STATUS or isinstance(error, RequestException):
        pacing = (
            (
                ' Explanations are the most GPU-intensive request, so send '
                'them one at a time.'
            )
            if explain
            else ''
        )
        said = str(detail).rstrip('.') if detail else ''
        server = f' The NIM said: {said}.' if said else ''
        how = (
            f' (HTTP {status})'
            if status is not None
            else ', and sent no response'
        )
        return NimFailureError(
            f'The KumoRelational NIM could not complete {subject}{how}.{server} '
            f'Retrying helps if the NIM was busy. If it fails the same way '
            f'again the request is too large for it, and what changes that is '
            f'a smaller one: fewer neighbours sampled (num_neighbors), fewer '
            f'rows scored, or a smaller context.{pacing}',
            status_code=status,
            detail=detail,
            invalid_params=invalid_params,
            transient=True,
        )

    if isinstance(status, int) and 400 <= status < 500:
        reason = str(detail or error).rstrip('.')
        return NimFailureError(
            f'The KumoRelational NIM rejected {subject} (HTTP {status}): '
            f'{reason}.{fields}{_cardinality_guidance(reason, payload)}',
            status_code=status,
            detail=detail,
            invalid_params=invalid_params,
        )

    return NimFailureError(
        f'An unexpected exception occurred. Please create an issue at '
        f"'https://github.com/NVIDIA/structured-data-models-client'. "
        f'{detail if detail else error}{fields}',
        status_code=status,
        detail=detail,
        invalid_params=invalid_params,
    )


def _check_anchor_time(value: Any, name: str) -> Any:
    r"""Reject an anchor time that is neither a ``Timestamp`` nor ``'entity'``,
    and return it converted to the timezone-naive UTC the graph is held in.

    A date *string* is what most pandas users reach for first, and it used to
    land on a bare ``assert`` with an empty message, and under ``python -O``
    on no check at all. A timezone-*aware* ``Timestamp`` is what the client's own
    ``predict`` output carries, so it is converted rather than refused.
    """
    if value is None or isinstance(value, pd.Timestamp):
        return to_naive_utc(value)
    if isinstance(value, str) and value == 'entity':
        return value
    hint = _a_conversion_that_works(value)
    raise TypeError(
        f"'{name}' must be a pandas.Timestamp or the literal 'entity' (got "
        f'{type(value).__name__} {value!r}){hint}'
    )


def _anchor_of(task: Any) -> str:
    r"""When the prediction was anchored, as the task table recorded it.

    Two runs of one query at different anchors are different answers, so a
    record that cannot say which anchor it used cannot explain either.

    A task anchored to each entity's own timestamp is named rather than dated,
    because there is no one time to report. It has to be named: `time_column`
    is `None` for that task exactly as it is for a task with no anchor at all,
    so reading it alone makes the two look the same.
    """
    if getattr(task, 'use_entity_time', False):
        return 'entity'
    held = getattr(task, 'time_column', None)
    column = getattr(held, 'name', held)
    frame = getattr(task, '_pred_df', None)
    if (
        not column
        or frame is None
        or column not in getattr(frame, 'columns', [])
    ):
        return ''
    try:
        return str(frame[column].max())
    except Exception:
        return ''


def _encode_composite_indices(
    indices: Sequence[Any],
    entity_key: tuple[str, ...],
) -> list[str]:
    r"""Folds tuple seeds into the identity the entity table is keyed on.

    A caller names an entity the way the data does -- ``('AB-10015',
    'Central')`` -- rather than by the derived value that carries it.
    """
    encoded: list[str] = []
    for index in indices:
        if isinstance(index, str):
            raise ValueError(
                f'Entity table is identified by {list(entity_key)}, so each '
                f'index has to name {len(entity_key)} value(s) as a tuple; '
                f'got the single value {index!r}. No row is identified by a '
                f'value on its own.'
            )
        if not isinstance(index, (tuple, list)):
            raise ValueError(
                f'Entity table is identified by {list(entity_key)}, so each '
                f'index has to name {len(entity_key)} value(s); got '
                f'{index!r}'
            )
        if len(index) != len(entity_key):
            raise ValueError(
                f'Entity table is identified by {list(entity_key)}, so each '
                f'index has to name {len(entity_key)} value(s); got '
                f'{list(index)}'
            )
        encoded.append(composite_key.encode_values(list(index)))
    return encoded


def _decode_composite_entities(
    result: Any,
    entity_key: tuple[str, ...],
) -> Any:
    r"""Reports a prediction against the columns the caller named.

    The derived identity is an implementation detail, so it is unfolded back
    into the key columns and ``ENTITY`` is dropped in their favour.
    """
    frame = result.prediction if hasattr(result, 'prediction') else result
    if not isinstance(frame, pd.DataFrame) or 'ENTITY' not in frame.columns:
        return result

    decoded = [
        composite_key.decode_value(str(value), len(entity_key))
        for value in frame['ENTITY']
    ]
    frame = frame.copy(deep=False)
    position = frame.columns.get_loc('ENTITY')
    frame = frame.drop(columns=['ENTITY'])
    occupied = [name for name in entity_key if name in frame.columns]
    if occupied:
        raise ValueError(
            f'Cannot report a prediction against key column(s) {occupied}: '
            f'the result already carries a column of that name. Rename the '
            f'key column(s) in the graph.'
        )
    for offset, name in enumerate(entity_key):
        frame.insert(
            position + offset, name, [parts[offset] for parts in decoded]
        )

    if hasattr(result, 'prediction'):
        result.prediction = frame
        return result
    return frame


class KumoRelational:
    r"""Run Kumo Relational predictions over a relational graph.

    :class:`KumoRelational` provides the prediction interface for a pre-trained
    relational foundation model. Build it from a :class:`Graph`, then issue PQL
    queries with :meth:`predict`.

    .. code-block:: python

        from sdm_relational_engine.rfm import Graph, KumoRelational

        df_users = pd.DataFrame(...)
        df_items = pd.DataFrame(...)
        df_orders = pd.DataFrame(...)

        graph = Graph.from_data({
            'users': df_users,
            'items': df_items,
            'orders': df_orders,
        })

        rfm = KumoRelational(graph)

        query = ("PREDICT COUNT(orders.*, 0, 30, days)>0 "
                 "FOR users.user_id=1")
        result = rfm.predict(query)

        print(result)  # user_id  COUNT(transactions.*, 0, 30, days) > 0
                       # 1        0.85

    Args:
        graph: The graph.
        verbose: Whether to print verbose output.
        optimize: If set to ``True``, will optimize the underlying data backend
            for optimal querying. For example, for transactional database
            backends, will create any missing indices. Requires write-access to
            the data backend. Only the :obj:`"sqlite"` and :obj:`"duckdb"`
            backends implement this; passing it on any other backend warns.
        _client: Internal. The already-resolved endpoint this instance predicts
            against, from :func:`sdm_relational_engine.rfm.init_client`. Binding it here
            pins the instance to one deployment for its whole lifetime;
            leaving it ``None`` falls back to resolving the process-wide
            engine configuration lazily, which is last-writer-wins across
            threads.
    """

    def __init__(
        self,
        graph: Graph,
        verbose: bool | ProgressLogger = True,
        optimize: bool = False,
        *,
        _client: NimClient | None = None,
    ) -> None:
        graph = graph.validate()
        self._graph = graph
        self._graph_def = graph._to_api_graph_definition()
        self._fingerprint = _fingerprint_of(graph)
        self._composite_key_dict: dict[str, tuple[str, ...]] = {
            name: table.primary_key_columns
            for name, table in graph.tables.items()
            if table.has_composite_primary_key
        }

        if optimize and graph.backend not in _OPTIMIZABLE_BACKENDS:
            warnings.warn(
                f"'optimize=True' has no effect on the "
                f"'{graph.backend.value}' backend; it is implemented "
                f'only for '
                f'{sorted(b.value for b in _OPTIMIZABLE_BACKENDS)}'
            )

        if graph.backend == DataBackend.LOCAL:
            from sdm_relational_engine.rfm.backend.local import LocalSampler

            self._sampler: Sampler = LocalSampler(graph, verbose)
        elif graph.backend == DataBackend.SQLITE:
            from sdm_relational_engine.rfm.backend.sqlite import SQLiteSampler

            self._sampler = SQLiteSampler(graph, verbose, optimize)
        elif graph.backend == DataBackend.DUCKDB:
            from sdm_relational_engine.rfm.backend.duckdb import DuckDBSampler

            self._sampler = DuckDBSampler(graph, verbose, optimize)
        elif graph.backend == DataBackend.SNOWFLAKE:
            from sdm_relational_engine.rfm.backend.snow import SnowSampler

            self._sampler = SnowSampler(graph, verbose)
        elif graph.backend == DataBackend.DATABRICKS:
            from sdm_relational_engine.rfm.backend.databricks import (
                DatabricksSampler,
            )

            self._sampler = DatabricksSampler(graph, verbose)
        else:
            raise NotImplementedError

        self._client: RFMAPI | None = (
            RFMAPI(_client) if _client is not None else None
        )

        self._batch_size: ContextVar[int | Literal['max'] | None] = ContextVar(
            f'{type(self).__name__}._batch_size', default=None
        )
        self._num_retries: ContextVar[int] = ContextVar(
            f'{type(self).__name__}._num_retries', default=0
        )

    @property
    def _api_client(self) -> RFMAPI:
        r"""The endpoint this instance predicts against.

        Bound at construction when the caller resolved one; otherwise read
        from the process-wide engine configuration on first use, which is the
        legacy path and reflects whichever configuration was applied last.
        """
        if self._client is not None:
            return self._client

        from sdm_relational_engine.client.engine_access import global_state

        self._client = RFMAPI(global_state.client)
        return self._client

    @property
    def sanitization_report(self) -> GraphSanitizationReport:
        r"""Structured graph-sanitization diagnostics for this backend."""
        return self._sampler.sanitization_report

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    @contextmanager
    def retry(
        self,
        num_retries: int = 1,
    ) -> Generator[None, None, None]:
        r"""Context manager to retry failed queries due to unexpected server
        issues.

        .. code-block:: python

            with model.retry(num_retries=1):
                df = model.predict(query, indices=...)

        Args:
            num_retries: The maximum number of retries.
        """
        if num_retries < 0:
            raise ValueError(
                f"'num_retries' must be greater than or equal to "
                f'zero (got {num_retries})'
            )

        token = self._num_retries.set(num_retries)
        try:
            yield
        finally:
            self._num_retries.reset(token)

    @contextmanager
    def batch_mode(
        self,
        batch_size: int | Literal['max'] = 'max',
        num_retries: int = 1,
    ) -> Generator[None, None, None]:
        r"""Context manager to predict in batches.

        .. code-block:: python

            with model.batch_mode(batch_size='max', num_retries=1):
                df = model.predict(query, indices=...)

        Args:
            batch_size: The batch size. If set to ``"max"``, will use the
                maximum applicable batch size for the given task.
            num_retries: The maximum number of retries for failed queries due
                to unexpected server issues.

        Note:
            The setting is held per execution context, so one model can be
            shared by concurrent threads and by concurrent asyncio tasks
            without their blocks overwriting each other. That covers the ways a
            service dispatches work, but not a process pool: a subprocess
            starts with a fresh context, so a prediction dispatched into one
            runs at the default rather than the size set around the call.

        Note:
            A multi-batch prediction uploads its context once, into a session
            the batches share, unless ``random_seed=None`` or
            ``KUMO_RELATIONAL_DISABLE_SESSIONS`` rules that out -- in which case every
            batch re-uploads the context and says so in the progress output.
        """
        if batch_size != 'max' and (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"'batch_size' must be a positive int or the "
                f"literal 'max' (got {batch_size!r})"
            )

        token = self._batch_size.set(batch_size)
        try:
            with self.retry(num_retries):
                yield
        finally:
            self._batch_size.reset(token)

    @overload
    def predict(
        self,
        query: str | ValidatedPredictiveQuery,
        indices: Sequence[str] | Sequence[float] | Sequence[int] | None = None,
        *,
        explain: Literal[False] = False,
        return_embeddings: bool = False,
        anchor_time: pd.Timestamp | Literal['entity'] | None = None,
        context_anchor_time: pd.Timestamp | None = None,
        run_mode: RunMode | str = RunMode.FAST,
        num_neighbors: list[int] | None = None,
        use_prediction_time: bool = False,
        lag_timesteps: int = 0,
        inference_config: InferenceConfig | dict[str, Any] | None = None,
        num_hops: int = 2,
        max_pq_iterations: int = 10,
        random_seed: int | None = _RANDOM_SEED,
        verbose: bool | ProgressLogger = True,
    ) -> pd.DataFrame:
        pass

    @overload
    def predict(
        self,
        query: str | ValidatedPredictiveQuery,
        indices: Sequence[str] | Sequence[float] | Sequence[int] | None = None,
        *,
        explain: Literal[True] | ExplainConfig | dict[str, Any],
        return_embeddings: bool = False,
        anchor_time: pd.Timestamp | Literal['entity'] | None = None,
        context_anchor_time: pd.Timestamp | None = None,
        run_mode: RunMode | str = RunMode.FAST,
        num_neighbors: list[int] | None = None,
        use_prediction_time: bool = False,
        lag_timesteps: int = 0,
        inference_config: InferenceConfig | dict[str, Any] | None = None,
        num_hops: int = 2,
        max_pq_iterations: int = 10,
        random_seed: int | None = _RANDOM_SEED,
        verbose: bool | ProgressLogger = True,
    ) -> Explanation:
        pass

    @overload
    def predict(
        self,
        query: str | ValidatedPredictiveQuery,
        indices: Sequence[str] | Sequence[float] | Sequence[int] | None = None,
        *,
        explain: bool | ExplainConfig | dict[str, Any] = False,
        return_embeddings: bool = False,
        anchor_time: pd.Timestamp | Literal['entity'] | None = None,
        context_anchor_time: pd.Timestamp | None = None,
        run_mode: RunMode | str = RunMode.FAST,
        num_neighbors: list[int] | None = None,
        use_prediction_time: bool = False,
        lag_timesteps: int = 0,
        inference_config: InferenceConfig | dict[str, Any] | None = None,
        num_hops: int = 2,
        max_pq_iterations: int = 10,
        random_seed: int | None = _RANDOM_SEED,
        verbose: bool | ProgressLogger = True,
    ) -> pd.DataFrame | Explanation:
        pass

    def predict(
        self,
        query: str | ValidatedPredictiveQuery,
        indices: Sequence[str] | Sequence[float] | Sequence[int] | None = None,
        *,
        explain: bool | ExplainConfig | dict[str, Any] = False,
        return_embeddings: bool = False,
        anchor_time: pd.Timestamp | Literal['entity'] | None = None,
        context_anchor_time: pd.Timestamp | None = None,
        run_mode: RunMode | str = RunMode.FAST,
        num_neighbors: list[int] | None = None,
        use_prediction_time: bool = False,
        lag_timesteps: int = 0,
        inference_config: InferenceConfig | dict[str, Any] | None = None,
        num_hops: int = 2,
        max_pq_iterations: int = 10,
        random_seed: int | None = _RANDOM_SEED,
        verbose: bool | ProgressLogger = True,
    ) -> pd.DataFrame | Explanation:
        r"""Returns predictions for a predictive query.

        Inference Configuration
        -----------------------
        The ``inference_config`` argument controls inference-time model
        behavior, including ensembling. Pass either a dictionary or a config
        object from ``sdm_relational_engine.rfm.inference.config``. Dictionary inputs
        are cast based on the task type: classification tasks use
        ``ClassificationInferenceConfig`` and regression or forecasting tasks
        use ``RegressionInferenceConfig``. If omitted, defaults are selected
        automatically based on the task type.

        Common options:

        * ``num_estimators``: Number of estimators to ensemble. Defaults to
          ``1`` and must be between ``1`` and ``4``.
        * ``column_shuffle``: Whether to shuffle column order across
          estimators.
        * ``category_shuffle``: Whether to shuffle categories within
          categorical columns across estimators.
        * ``hop_shuffle``: Whether to shuffle subgraph depth across
          estimators.

        Classification option:

        * ``class_shuffle``: Whether to shuffle class order across estimators.

        Regression and forecasting options:

        * ``target_transforms``: Target preprocessing transforms to vary
          across estimators. Supported values are ``"clip"``, ``"power"``,
          ``"quantile"``, and ``None``. Defaults to ``["quantile"]``.
        * ``output_type``: How to summarize the output distribution. Supported
          values are ``"median"``, ``"mean"`` and ``"quantiles"``. Defaults to
          ``"median"``. ``"quantiles"`` returns the full distribution as
          additional ``Q_<level>`` columns alongside the point prediction.
          ``"mean"`` is unreliable on **zero-inflated regression targets**:
          once roughly 40% of the predicted quantile grid sits at zero, the
          NIM returns exactly ``0.0`` for every entity, with no error --
          which is precisely the data shape a mean is chosen for. Prefer
          ``"median"`` or ``"quantiles"`` there.

        A key other than those listed above is rejected, naming the offending
        key: nothing forwards an unknown key to the NIM, so accepting one could
        only ever mean silently substituting the default.

        .. code-block:: python

            result = rfm.predict(
                query,
                inference_config=dict(
                    num_estimators=4,
                    column_shuffle=True,
                    hop_shuffle=True,
                    class_shuffle=True,
                ),
            )

            result = rfm.predict(
                regression_query,
                inference_config=dict(
                    num_estimators=4,
                    column_shuffle=True,
                    target_transforms=['quantile', 'clip', 'power', None],
                    output_type='median',
                ),
            )

        Args:
            query: The predictive query.
            indices: The entity primary keys to predict for. Will override the
                indices given as part of the predictive query. Predictions will
                be generated for all indices, independent of whether they
                fulfill entity filter constraints.
            explain: Configuration for explainability.
                If set to ``True``, will additionally explain the prediction.
                Passing in an :class:`ExplainConfig` instance provides control
                over which parts of explanation are generated.
                Explainability is currently only supported for single entity
                predictions with ``run_mode="FAST"``.
                Note that the natural-language summary is built by an external
                LLM endpoint (OpenAI by default) and sends the query,
                predictions and raw subgraph cell values off the machine --
                see :class:`ExplainConfig` for what is transmitted and pass
                ``explain=dict(skip_summary=True)`` to disable it.
            return_embeddings: Whether to also return the embeddings for each
                prediction example.
            anchor_time: The anchor timestamp for the prediction, as a
                :class:`pandas.Timestamp`; a date string is not coerced. If set
                to ``None``, will use the maximum timestamp in the data.
                If set to ``"entity"``, will use the timestamp of the entity.
            context_anchor_time: The maximum anchor timestamp for context
                examples. If set to ``None``, ``anchor_time`` will
                determine the anchor time for context examples.
            run_mode: The :class:`RunMode` for the query.
            num_neighbors: The number of neighbors to sample for each hop.
                If specified, the ``num_hops`` option will be ignored.
            use_prediction_time: Whether to use the anchor timestamp as an
                additional feature during prediction.
            lag_timesteps: The number of past timesteps included as lagged
                features.
            inference_config: Optional inference-time model configuration. See
                the inference configuration section above for supported
                dictionary keys.
            num_hops: The number of hops to sample when generating the
                context, between 1 and 6. Deprecated in favor of
                ``num_neighbors``.
            max_pq_iterations: The maximum number of iterations to perform to
                collect valid labels, at least 1. It is advised to increase the
                number of
                iterations in case the predictive query has strict entity
                filters, in which case, :class:`KumoRelational` needs to sample more
                entities to find valid labels.
            random_seed: A manual seed for generating pseudo-random numbers.
                The :obj:`"sqlite"` and :obj:`"snowflake"` backends cannot
                seed their random row sampling and warn once when a seed is
                given. Passing :obj:`None` re-samples neighborhoods for every
                batch, which rules out the shared-context session a
                multi-batch prediction otherwise opens, so each batch
                re-uploads the full context.
            verbose: Whether to print verbose output.

        Returns:
            The predictions as a :class:`pandas.DataFrame` when
            ``explain`` is ``False``, which it is by default. Anything
            else -- including an empty mapping -- asks for an
            :class:`Explanation`, which carries that frame together with
            a summary and its details.
        """
        query_def = self._parse_query(query)

        if indices is None:
            if query_def.rfm_entity_ids is None:
                raise ValueError(
                    'Cannot find entities to predict for. Please '
                    'pass them via `predict(query, indices=...)`'
                )
            indices = query_def.get_rfm_entity_id_list()
        entity_key = self._composite_entity_key(query_def)
        if entity_key is not None:
            indices = _encode_composite_indices(indices, entity_key)
        query_def = replace(
            query_def,
            for_each='FOR EACH',
            rfm_entity_ids=None,
        )

        if not isinstance(verbose, ProgressLogger):
            query_repr = query_def.to_string(rich=True, exclude_predict=True)
            if explain is not False:
                msg = f'[bold]EXPLAIN[/bold] {query_repr}'
            else:
                msg = f'[bold]PREDICT[/bold] {query_repr}'
            verbose = ProgressLogger.default(msg=msg, verbose=verbose)

        described = describing(
            query_type=str(getattr(query_def, 'query_type', '')),
            entities=len(indices),
        )
        with described, verbose as logger:
            task_table = self._get_task_table(
                query=query_def,
                indices=indices,
                anchor_time=anchor_time,
                context_anchor_time=context_anchor_time,
                run_mode=RunMode(run_mode),
                lag_timesteps=lag_timesteps,
                max_pq_iterations=max_pq_iterations,
                random_seed=random_seed,
                logger=logger,
            )
            task_table._query = query_def.to_string()

            result = self.predict_task(
                task_table,
                explain=explain,
                return_embeddings=return_embeddings,
                run_mode=RunMode(run_mode),
                num_neighbors=num_neighbors,
                inference_config=inference_config,
                num_hops=num_hops,
                verbose=verbose,
                exclude_cols_dict=query_def.get_exclude_cols_dict(),
                use_prediction_time=use_prediction_time,
                top_k=query_def.top_k,
                random_seed=random_seed,
            )
            if entity_key is not None:
                result = _decode_composite_entities(result, entity_key)
            return result

    def _graph_fingerprint(self) -> str:
        r"""Name the graph this prediction ran against.

        Taken once, when the model was built, because that is when the graph
        stops mattering: the query is parsed against `_graph_def` and the rows
        come from `_sampler`, both fixed here. Re-reading the caller's graph
        per call would report a link they removed afterwards as though the
        prediction had used it.
        """
        return self._fingerprint

    def _composite_entity_key(self, query: Any) -> tuple[str, ...] | None:
        r"""The columns the query's entity table spreads its identity across.

        Returns ``None`` unless that identity is composite, so a single-column
        entity takes exactly the path it did before.
        """
        return self._composite_key_dict.get(query.entity_table)

    def materialize_task(
        self,
        task: TaskTable,
        *,
        explain: bool | ExplainConfig | dict[str, Any] = False,
        return_embeddings: bool = False,
        run_mode: RunMode | str = RunMode.FAST,
        num_neighbors: list[int] | None = None,
        inference_config: InferenceConfig | dict[str, Any] | None = None,
        num_hops: int = 2,
        verbose: bool | ProgressLogger = True,
        exclude_cols_dict: dict[str, list[str]] | None = None,
        use_prediction_time: bool = False,
        top_k: int | None = None,
        random_seed: int | None = _RANDOM_SEED,
    ) -> tuple[MaterializedPredictionRequest, ...]:
        r"""Materialize final prediction requests without contacting a NIM.

        The returned payloads are the exact mappings used by
        :meth:`predict_task`, paired with half-open positional ranges into
        the original prediction rows.
        """
        run_mode, num_neighbors, inference_config = (
            self._resolve_task_request_options(
                task,
                run_mode=run_mode,
                num_neighbors=num_neighbors,
                inference_config=inference_config,
                num_hops=num_hops,
            )
        )
        explain_config, run_mode = self._apply_explain_guard(
            explain, run_mode, task
        )

        if not isinstance(verbose, ProgressLogger):
            verbose = ProgressLogger.default(
                msg=f'Materializing {task.task_type} task',
                verbose=verbose,
            )
        with verbose as logger:
            requests = self._iter_task_requests(
                task,
                explain=explain_config is not None,
                return_embeddings=return_embeddings,
                run_mode=run_mode,
                num_neighbors=num_neighbors,
                inference_config=inference_config,
                logger=logger,
                exclude_cols_dict=exclude_cols_dict,
                use_prediction_time=use_prediction_time,
                top_k=top_k,
                random_seed=random_seed,
            )
            return tuple(request.materialized for request in requests)

    def _apply_explain_guard(
        self,
        explain: bool | ExplainConfig | dict[str, Any],
        run_mode: RunMode,
        task: TaskTable,
    ) -> tuple[ExplainConfig | None, RunMode]:
        r"""Resolves ``explain`` and enforces what explainability requires.

        Explanations are only produced for a single entity at ``RunMode.FAST``;
        a higher run mode is lowered with a warning, more than one prediction
        example is an error.
        """
        explain_config = self._resolve_explain_config(explain)
        if explain_config is None:
            return None, run_mode
        if run_mode in {RunMode.NORMAL, RunMode.BEST}:
            warnings.warn(
                f'Explainability is currently only supported for run mode '
                f"'FAST' (got '{run_mode}'). Provided run mode has been reset. "
                f'Please lower the run mode to suppress this warning.',
                stacklevel=3,
            )
            run_mode = RunMode.FAST
        if task.num_prediction_examples > 1:
            raise ValueError(
                f'Cannot explain predictions for more than a single entity '
                f'(got {task.num_prediction_examples:,})'
            )
        return explain_config, run_mode

    def _resolve_task_request_options(
        self,
        task: TaskTable,
        *,
        run_mode: RunMode | str,
        num_neighbors: list[int] | None,
        inference_config: InferenceConfig | dict[str, Any] | None,
        num_hops: int,
    ) -> tuple[RunMode, list[int], InferenceConfig]:
        run_mode = RunMode(run_mode)
        if num_hops != 2 and num_neighbors is not None:
            warnings.warn(
                f"Received custom 'num_neighbors' option; ignoring custom "
                f"'num_hops={num_hops}' option",
                stacklevel=3,
            )
        if num_neighbors is None:
            key = RunMode.FAST if task.task_type.is_link_pred else run_mode
            # 'num_hops' is a slice bound, so an out-of-range value would be
            # silently reinterpreted: -1 asks for fewer hops and yields the
            # second-deepest sample, and anything above the maximum clamps.
            max_hops = len(_DEFAULT_NUM_NEIGHBORS[key])
            if not 1 <= num_hops <= max_hops:
                raise ValueError(
                    f"'num_hops' must be between 1 and "
                    f'{max_hops} (got {num_hops})'
                )
            num_neighbors = _DEFAULT_NUM_NEIGHBORS[key][:num_hops]

        if inference_config is None:
            inference_config = InferenceConfig.from_task_type(task.task_type)
        elif isinstance(inference_config, dict):
            # Holds a class, not an instance, so it is named like one.
            Config = InferenceConfig  # noqa: N806
            if task.task_type.is_classification:
                Config = ClassificationInferenceConfig  # noqa: N806
            if task.task_type in {TaskType.REGRESSION, TaskType.FORECASTING}:
                Config = RegressionInferenceConfig  # noqa: N806
            inference_config = Config(**inference_config)  # type: ignore
        return run_mode, num_neighbors, inference_config

    @staticmethod
    def _resolve_explain_config(
        explain: bool | ExplainConfig | dict[str, Any],
    ) -> ExplainConfig | None:
        if explain is True:
            return ExplainConfig()
        if explain is not False:
            return ExplainConfig._cast(explain)
        return None

    def _validate_task_references(self, task: TaskTable) -> None:
        entity_pkey = pd.concat(
            [
                task._context_df[task._entity_column],
                task._pred_df[task._entity_column],
            ],
            axis=0,
            ignore_index=True,
        )
        self._sampler.validate_entity_references(
            task.entity_table_name,
            entity_pkey,
        )

    def _resolve_batch_size(self, task: TaskTable) -> int:
        batch_size = self._batch_size.get()
        if batch_size is None:
            return task.num_prediction_examples
        if batch_size == 'max':
            return _MAX_PRED_SIZE[task.task_type]
        return batch_size

    def _resolve_num_batches(self, task: TaskTable) -> int:
        batch_size = self._resolve_batch_size(task)
        if batch_size <= 0:
            return 1
        return math.ceil(task.num_prediction_examples / batch_size)

    def _session_predict_batch(
        self,
        generated: '_GeneratedPredictionRequest',
        session: '_SessionHandle',
    ) -> Any:
        r"""Send one batch through a session, uploading the pinned context only
        once. Mutates ``session`` in place so a session created here is always
        tracked for deletion even if a later call raises.

        Robust to the NIM's session constraints: an expired or evicted session
        (HTTP 404, TTL runs from creation) is transparently recreated and
        retried, and a NIM without session support (404/405/501 on create)
        disables sessions and serves the batch via the stateless path, so a
        batched job never regresses below plain prediction.
        """
        payload = generated.materialized.payload
        correlate = dict(
            entity_ids=generated.entity_ids,
            instance_ids=generated.instance_ids,
            anchor_times=generated.anchor_times,
        )
        if session.id is None:
            try:
                session.id = self._api_client.create_session(
                    session_create_payload(payload)
                )
            except HTTPException as error:
                if _sessions_unsupported(error):
                    session.active = False
                    return self._api_client.predict(payload, **correlate)
                raise
        predict_payload = session_predict_payload(payload)
        try:
            return self._api_client.session_predict(
                session.id, predict_payload, **correlate
            )
        except HTTPException as error:
            if error.status_code != 404:
                raise
            session.id = self._api_client.create_session(
                session_create_payload(payload)
            )
            return self._api_client.session_predict(
                session.id, predict_payload, **correlate
            )

    def _delete_session_quietly(self, session_id: str) -> None:
        r"""Best-effort session deletion; the server also reaps it on TTL."""
        try:
            self._api_client.delete_session(session_id)
        except (HTTPException, RequestException):
            pass

    def _predict_batches(
        self,
        requests: Iterator['_GeneratedPredictionRequest'],
        *,
        task: TaskTable,
        explain_config: 'ExplainConfig | None',
        session: '_SessionHandle | None',
        verbose: ProgressLogger,
    ) -> tuple[list[pd.DataFrame], str | None, Any, str | None]:
        r"""Run every batch, sending through ``session`` when active and via
        stateless prediction otherwise, with bounded retries per batch.
        """
        predictions: list[pd.DataFrame] = []
        summary: str | None = None
        details: Any | None = None
        warning: str | None = None
        for generated in requests:
            materialized = generated.materialized
            request_payload = materialized.payload
            num_retries = self._num_retries.get()
            for attempt in range(num_retries + 1):
                try:
                    if session is not None and session.active:
                        resp = self._session_predict_batch(generated, session)
                    else:
                        resp = self._api_client.predict(
                            request_payload,
                            entity_ids=generated.entity_ids,
                            instance_ids=generated.instance_ids,
                            anchor_times=generated.anchor_times,
                        )
                    df = pd.DataFrame(**resp.prediction)
                    if explain_config is not None:
                        df, summary, details, warning = extract_explanation(
                            df, explain_config
                        )

                    if 'ENTITY' in df:
                        df['ENTITY'] = df['ENTITY'].astype(
                            generated.entity_dtype
                        )

                    if 'CLASS' in df:
                        df['CLASS'] = _cast_class_column(
                            df['CLASS'], generated.class_dtype
                        )

                    if 'ANCHOR_TIMESTAMP' in df:
                        ser = df['ANCHOR_TIMESTAMP']
                        if not pd.api.types.is_datetime64_any_dtype(ser):
                            if isinstance(ser.iloc[0], str):
                                unit = None
                            else:
                                unit = 'ms'
                            df['ANCHOR_TIMESTAMP'] = pd.to_datetime(
                                ser, errors='coerce', unit=unit
                            )

                    if 'TIME' in df:
                        ser = df['TIME']
                        if not pd.api.types.is_datetime64_any_dtype(ser):
                            if isinstance(ser.iloc[0], str):
                                unit = None
                            else:
                                unit = 'ms'
                            df['TIME'] = pd.to_datetime(
                                ser, errors='coerce', unit=unit
                            )

                    predictions.append(df.reset_index(drop=True))

                    if (
                        materialized.prediction_stop
                        < task.num_prediction_examples
                        or materialized.batch_index > 0
                    ):
                        verbose.step()

                    break
                except (HTTPException, RequestException) as e:
                    if getattr(e, 'status_code', None) == 413:
                        num_entities = (
                            materialized.prediction_stop
                            - materialized.prediction_start
                        )
                        detail = getattr(e, 'detail', e)
                        raise ValueError(
                            f'The request is too large for the endpoint '
                            f'({detail}). It carries {num_entities:,} '
                            f"entities; retry with a smaller 'batch_size' "
                            f'(e.g. {max(1, num_entities // 2):,}), or '
                            f'sample fewer related rows per entity with '
                            f"'num_neighbors'."
                        ) from None
                    failure = _nim_failure_error(
                        e, explain_config is not None, request_payload
                    )
                    if attempt == num_retries or not failure.transient:
                        raise failure from None
                    time.sleep(_RETRY_BACKOFF_BASE_SECONDS**attempt)
        return predictions, summary, details, warning

    def _iter_task_requests(
        self,
        task: TaskTable,
        *,
        explain: bool,
        return_embeddings: bool,
        run_mode: RunMode,
        num_neighbors: list[int],
        inference_config: InferenceConfig,
        logger: ProgressLogger,
        exclude_cols_dict: dict[str, list[str]] | None,
        use_prediction_time: bool,
        top_k: int | None,
        random_seed: int | None,
        progress_message: str | None = None,
    ) -> Iterator[_GeneratedPredictionRequest]:
        # Validate the complete task before generating or sending any batch.
        self._validate_task_references(task)

        max_ctx = _MAX_CONTEXT_SIZE[run_mode]
        if task.num_context_examples > max_ctx:
            logger.log(
                f'Sub-sampled {max_ctx:,} out of '
                f'{task.num_context_examples:,} in-context examples'
            )
            task = task.narrow_context(0, max_ctx)

        if (
            task.task_type == TaskType.FORECASTING
            and task.num_forecasts > task.num_context_examples
        ):
            raise ValueError(
                f'The number of forecast steps ({task.num_forecasts:,}) '
                f'exceeds the number of available in-context examples '
                f'({task.num_context_examples:,}). Please provide more '
                f'historical data or reduce the number of forecast steps.'
            )

        batch_size = self._resolve_batch_size(task)

        max_batch_size = _MAX_PRED_SIZE[task.task_type]
        if batch_size > max_batch_size:
            # The example has to stay under the cap it just quoted: temporal
            # link prediction caps at 200, so a fixed `batch_size=500` told the
            # user to retry with a value that reproduces this same error.
            raise ValueError(
                f'Cannot predict for more than {max_batch_size:,} entities at '
                f'once (got {batch_size:,}). Pass `batch_size=` to '
                f'`predict(...)` (for example '
                f'`batch_size={min(500, max_batch_size)}`, or '
                f"`batch_size='max'`) to process entities in batches."
            )

        num_batches = math.ceil(task.num_prediction_examples / batch_size)
        if num_batches > 1:
            logger.log(
                f'Splitting {task.num_prediction_examples:,} entities into '
                f'{num_batches:,} batches of size {batch_size:,}'
            )
            if progress_message is not None:
                logger.init_progress(msg=progress_message, total=num_batches)

        for batch_index, start in enumerate(
            range(0, task.num_prediction_examples, batch_size)
        ):
            stop = min(start + batch_size, task.num_prediction_examples)
            batch_task = task.narrow_prediction(start, length=batch_size)
            context = self._get_context(
                task=batch_task,
                run_mode=run_mode,
                num_neighbors=num_neighbors,
                exclude_cols_dict=exclude_cols_dict,
                top_k=top_k,
                random_seed=random_seed,
                _validate_references=False,
            )
            context.y_test = None
            request = RFMPredictRequest(
                context=context,
                run_mode=run_mode,
                query=task._query,
                use_prediction_time=use_prediction_time,
                inference_config=inference_config,
                return_embeddings=return_embeddings,
            )
            payload = predict_request_to_json(request, explain=explain)
            prediction_row_count = len(
                payload['predict']['instance_table']['rows']
            )
            expected_prediction_rows = stop - start
            if prediction_row_count != expected_prediction_rows:
                raise RuntimeError(
                    f'Request batch {batch_index} serialized '
                    f'{prediction_row_count:,} prediction rows; expected '
                    f'{expected_prediction_rows:,} from its source range.'
                )
            request_size = payload_size_bytes(payload)
            if batch_index == 0:
                logger.log(
                    f'Generated context of size '
                    f'{request_size / (1024 * 1024):.2f}MB'
                )
            if request_size > _MAX_SIZE:
                stats = context_size_stats(context)
                raise ValueError(_SIZE_LIMIT_MSG.format(stats=stats))
            validate_payload_table_rows(payload, batch_index=batch_index)
            materialized = MaterializedPredictionRequest(
                payload=payload,
                batch_index=batch_index,
                prediction_start=start,
                prediction_stop=stop,
                request_size_bytes=request_size,
            )
            entity_table = context.sampled_subgraph.tables[
                context.entity_table_names[0]
            ]
            predict_table = payload['predict']['instance_table']
            instance_id_index = predict_table['columns'].index(INSTANCE_ID)
            anchor_column = payload['task'].get('anchor_time_column')
            anchor_index = (
                predict_table['columns'].index(anchor_column)
                if anchor_column in predict_table['columns']
                else None
            )
            yield _GeneratedPredictionRequest(
                materialized=materialized,
                entity_ids=tuple(
                    batch_task._pred_df[batch_task.entity_column.name].tolist()
                ),
                instance_ids=tuple(
                    row[instance_id_index] for row in predict_table['rows']
                ),
                entity_dtype=entity_table.df[entity_table.primary_key].dtype,
                anchor_times=tuple(
                    row[anchor_index] for row in predict_table['rows']
                )
                if anchor_index is not None
                else None,
                class_dtype=_class_dtype(task, context),
            )

    @overload
    def predict_task(
        self,
        task: TaskTable,
        *,
        explain: Literal[False] = False,
        return_embeddings: bool = False,
        run_mode: RunMode | str = RunMode.FAST,
        num_neighbors: list[int] | None = None,
        inference_config: InferenceConfig | dict[str, Any] | None = None,
        num_hops: int = 2,
        verbose: bool | ProgressLogger = True,
        exclude_cols_dict: dict[str, list[str]] | None = None,
        use_prediction_time: bool = False,
        top_k: int | None = None,
        random_seed: int | None = _RANDOM_SEED,
    ) -> pd.DataFrame:
        pass

    @overload
    def predict_task(
        self,
        task: TaskTable,
        *,
        explain: Literal[True] | ExplainConfig | dict[str, Any],
        return_embeddings: bool = False,
        run_mode: RunMode | str = RunMode.FAST,
        num_neighbors: list[int] | None = None,
        inference_config: InferenceConfig | dict[str, Any] | None = None,
        num_hops: int = 2,
        verbose: bool | ProgressLogger = True,
        exclude_cols_dict: dict[str, list[str]] | None = None,
        use_prediction_time: bool = False,
        top_k: int | None = None,
        random_seed: int | None = _RANDOM_SEED,
    ) -> Explanation:
        pass

    @overload
    def predict_task(
        self,
        task: TaskTable,
        *,
        explain: bool | ExplainConfig | dict[str, Any] = False,
        return_embeddings: bool = False,
        run_mode: RunMode | str = RunMode.FAST,
        num_neighbors: list[int] | None = None,
        inference_config: InferenceConfig | dict[str, Any] | None = None,
        num_hops: int = 2,
        verbose: bool | ProgressLogger = True,
        exclude_cols_dict: dict[str, list[str]] | None = None,
        use_prediction_time: bool = False,
        top_k: int | None = None,
        random_seed: int | None = _RANDOM_SEED,
    ) -> pd.DataFrame | Explanation:
        pass

    def predict_task(
        self,
        task: TaskTable,
        *,
        explain: bool | ExplainConfig | dict[str, Any] = False,
        return_embeddings: bool = False,
        run_mode: RunMode | str = RunMode.FAST,
        num_neighbors: list[int] | None = None,
        inference_config: InferenceConfig | dict[str, Any] | None = None,
        num_hops: int = 2,
        verbose: bool | ProgressLogger = True,
        exclude_cols_dict: dict[str, list[str]] | None = None,
        use_prediction_time: bool = False,
        top_k: int | None = None,
        random_seed: int | None = _RANDOM_SEED,
    ) -> pd.DataFrame | Explanation:
        r"""Returns predictions for a custom task specification.

        Args:
            task: The custom :class:`TaskTable`.
            explain: Configuration for explainability.
                If set to ``True``, will additionally explain the prediction.
                Passing in an :class:`ExplainConfig` instance provides control
                over which parts of explanation are generated.
                Explainability is currently only supported for single entity
                predictions with ``run_mode="FAST"``.
                Note that the natural-language summary is built by an external
                LLM endpoint (OpenAI by default) and sends the query,
                predictions and raw subgraph cell values off the machine --
                see :class:`ExplainConfig` for what is transmitted and pass
                ``explain=dict(skip_summary=True)`` to disable it.
            return_embeddings: Whether to also return the embeddings for each
                prediction example.
            run_mode: The :class:`RunMode` for the query.
            num_neighbors: The number of neighbors to sample for each hop.
                If specified, the ``num_hops`` option will be ignored.
            inference_config: Optional inference-time model configuration. See
                :meth:`predict` for supported dictionary keys.
            num_hops: The number of hops to sample when generating the
                context, between 1 and 6.
            verbose: Whether to print verbose output.
            exclude_cols_dict: Any column in any table to exclude from the
                model input.
            use_prediction_time: Whether to use the anchor timestamp as an
                additional feature during prediction.
            top_k: The number of predictions to return per entity.
            random_seed: A manual seed for neighborhood sampling. Reusing a
                seed produces the same sampled local neighborhoods for the
                same graph, ordered task rows, batching, and options. The
                :obj:`"sqlite"` and :obj:`"snowflake"` backends cannot seed
                their random row sampling and warn once when a seed is given.
                Passing :obj:`None` re-samples neighborhoods for every batch,
                which rules out the shared-context session a multi-batch
                prediction otherwise opens, so each batch re-uploads the full
                context.

        Returns:
            The predictions as a :class:`pandas.DataFrame` when
            ``explain`` is ``False``, which it is by default. Anything
            else -- including an empty mapping -- asks for an
            :class:`Explanation`, which carries that frame together with
            a summary and its details.
        """
        run_mode, num_neighbors, inference_config = (
            self._resolve_task_request_options(
                task,
                run_mode=run_mode,
                num_neighbors=num_neighbors,
                inference_config=inference_config,
                num_hops=num_hops,
            )
        )

        explain_config, run_mode = self._apply_explain_guard(
            explain, run_mode, task
        )

        if not isinstance(verbose, ProgressLogger):
            if task.task_type == TaskType.BINARY_CLASSIFICATION:
                task_type_repr = 'binary classification'
            elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                task_type_repr = 'multi-class classification'
            elif task.task_type == TaskType.REGRESSION:
                task_type_repr = 'regression'
            elif task.task_type == TaskType.FORECASTING:
                task_type_repr = 'forecasting'
            elif task.task_type == TaskType.TEMPORAL_LINK_PREDICTION:
                task_type_repr = 'link prediction'
            else:
                task_type_repr = str(task.task_type)

            if explain_config is not None:
                msg = f'Explaining {task_type_repr} task'
            else:
                msg = f'Predicting {task_type_repr} task'
            verbose = ProgressLogger.default(msg=msg, verbose=verbose)

        record = PredictionRecord(
            graph_fingerprint=self._graph_fingerprint(),
            engine_version=__version__,
            run_mode=str(RunMode(run_mode)),
            batch_size=str(self._batch_size.get()),
            # The resolved neighbour list, not the argument: `num_hops` is
            # deprecated in favour of `num_neighbors`, and passing the latter
            # sets the hop count while leaving the former at its default. The
            # list is what the sampler walks, so its length is what ran.
            num_hops=len(num_neighbors),
            explain=explain_config is not None,
            prediction_id=uuid.uuid4().hex,
            task_type=str(getattr(task, 'task_type', '')),
            entity_table=','.join(
                getattr(task, 'entity_table_names', ()) or ()
            ),
            anchor_time=_anchor_of(task),
            **pending(),
        )
        with recorded(record), verbose as logger:
            requests = self._iter_task_requests(
                task,
                explain=explain_config is not None,
                return_embeddings=return_embeddings,
                run_mode=run_mode,
                num_neighbors=num_neighbors,
                inference_config=inference_config,
                logger=logger,
                exclude_cols_dict=exclude_cols_dict,
                use_prediction_time=use_prediction_time,
                top_k=top_k,
                random_seed=random_seed,
                progress_message='Predicting',
            )
            use_sessions = (
                explain_config is None
                and random_seed is not None
                and not _sessions_disabled_by_env()
                and self._resolve_num_batches(task) > 1
            )
            if (
                not use_sessions
                and explain_config is None
                and self._resolve_num_batches(task) > 1
            ):
                logger.log(_no_session_reason(random_seed))
            session = _SessionHandle() if use_sessions else None
            try:
                predictions, summary, details, warning = self._predict_batches(
                    requests,
                    task=task,
                    explain_config=explain_config,
                    session=session,
                    verbose=verbose,
                )
            finally:
                if session is not None and session.id is not None:
                    self._delete_session_quietly(session.id)

        if len(predictions) == 1:
            prediction = predictions[0]
        else:
            prediction = pd.concat(predictions, ignore_index=True)

        if explain_config is not None:
            assert len(predictions) == 1
            assert details is not None
            summary = summary or ''
            if (
                not explain_config.skip_summary
                and not summary
                and isinstance(details, dict)
                and details.get('format') == 'kumo_rfm_v2_1'
            ):
                inner = details.get('details')
                if not isinstance(inner, dict):
                    inner = {}
                cohorts = inner.get('cohorts')
                subgraphs = inner.get('subgraphs')
                if not isinstance(cohorts, list):
                    cohorts = []
                if not isinstance(subgraphs, list):
                    subgraphs = []
                if cohorts or subgraphs:
                    summary = generate_summary(
                        query=getattr(task, '_query', '') or '',
                        prediction=prediction.to_dict('records'),
                        cohorts=cohorts,
                        subgraphs=subgraphs,
                    )
            return Explanation(
                prediction=prediction,
                summary=summary,
                details=details,
                warning=warning,
            )

        return prediction

    def get_train_table(
        self,
        query: str | ValidatedPredictiveQuery,
        size: int,
        *,
        anchor_time: pd.Timestamp | Literal['entity'] | None = None,
        random_seed: int | None = _RANDOM_SEED,
        max_iterations: int = 10,
    ) -> pd.DataFrame:
        r"""Returns the labels of a predictive query for a specified anchor
        time.

        Args:
            query: The predictive query.
            size: The maximum number of entities to generate labels for.
            anchor_time: The anchor timestamp for the query. If set to
                :obj:`None`, will use the maximum timestamp in the data.
                If set to :`"entity"`, will use the timestamp of the entity.
            random_seed: A manual seed for generating pseudo-random numbers.
            max_iterations: The number of steps to run before aborting.

        Returns:
            The labels as a :class:`pandas.DataFrame`.
        """
        anchor_time = _check_anchor_time(anchor_time, 'anchor_time')
        query_def = self._parse_query(query)

        if anchor_time is None:
            anchor_time = self._get_default_anchor_time(query_def)
            if query_def.target_ast.date_offset_range is not None:
                offset = query_def.target_ast.date_offset_range.end_date_offset
                offset *= query_def.num_forecasts
                anchor_time -= offset

        assert anchor_time is not None
        if isinstance(anchor_time, pd.Timestamp):
            self._validate_time(query_def, anchor_time, None, evaluate=True)
        else:
            assert anchor_time == 'entity'
            if query_def.entity_table not in self._sampler.time_column_dict:
                raise ValueError(
                    f"Anchor time 'entity' requires the entity "
                    f"table '{query_def.entity_table}' "
                    f'to have a time column'
                )

        try:
            _train, test = self._sampler.sample_target(
                query=query_def,
                num_train_examples=0,
                train_anchor_time=anchor_time,
                num_train_trials=0,
                num_test_examples=size,
                test_anchor_time=anchor_time,
                num_test_trials=max_iterations * size,
                random_seed=random_seed,
            )
        except RuntimeError as e:
            if 'Failed to collect any' in str(e):
                return pd.DataFrame(
                    {
                        'ENTITY': [],
                        'ANCHOR_TIMESTAMP': [],
                        'TARGET': [],
                    }
                )
            raise

        return pd.DataFrame(
            {
                'ENTITY': test.entity_pkey,
                'ANCHOR_TIMESTAMP': test.anchor_time,
                'TARGET': test.target,
            }
        )

    def add_lagged_target(
        self,
        task: TaskTable,
        query: str | ValidatedPredictiveQuery,
        lag_timesteps: int,
    ) -> TaskTable:
        r"""Adds lagged targets as input features to the task.

        Args:
            task: The task table.
            query: The predictive query to compute lagged target features from.
            lag_timesteps: Number of previous timesteps to use as lagged target
                features.
        """
        if not task.has_time_column():
            raise ValueError(
                'Task requires to have a time columns in order '
                'to add lagged target features'
            )
        assert task.time_column is not None

        if lag_timesteps <= 0:
            raise ValueError(
                f"'lag_timesteps' needs to be positive (got '{lag_timesteps}')"
            )

        query_def = self._parse_query(query)

        if query_def.query_type != QueryType.TEMPORAL:
            raise ValueError(
                'Lagged target features can only be added for '
                'temporal predictive queries'
            )

        lagged_df = self._sampler.sample_lagged_target(
            query=query_def,
            entity_pkey=pd.concat(
                [
                    task._context_df[task.entity_column.name],
                    task._pred_df[task.entity_column.name],
                ],
                ignore_index=True,
            ),
            anchor_time=pd.concat(
                [
                    task._context_df[task.time_column.name],
                    task._pred_df[task.time_column.name],
                ],
                ignore_index=True,
            ),
            lag_timesteps=lag_timesteps,
        )

        task._context_df = pd.concat(
            [
                task._context_df,
                lagged_df.iloc[: task.num_context_examples].reset_index(
                    drop=True
                ),
            ],
            axis=1,
        )
        task._pred_df = pd.concat(
            [
                task._pred_df,
                lagged_df.iloc[task.num_context_examples :].reset_index(
                    drop=True
                ),
            ],
            axis=1,
        )
        task.add_columns(lagged_df.columns.tolist())

        return task

    def update_connection(self, connection: Any) -> None:
        r"""Updates the connection to a database."""
        if self._sampler.backend == DataBackend.SQLITE:
            from adbc_driver_sqlite.dbapi import AdbcSqliteConnection

            from sdm_relational_engine.rfm.backend.sqlite import SQLiteSampler

            assert isinstance(self._sampler, SQLiteSampler)
            assert isinstance(connection, AdbcSqliteConnection)
            self._sampler._connection = connection
        if self._sampler.backend == DataBackend.DUCKDB:
            from sdm_relational_engine.rfm.backend.duckdb import (
                Connection,
                DuckDBSampler,
            )

            assert isinstance(self._sampler, DuckDBSampler)
            assert isinstance(connection, Connection)
            self._sampler._connection = connection
        if self._sampler.backend == DataBackend.SNOWFLAKE:
            from snowflake.connector import SnowflakeConnection

            from sdm_relational_engine.rfm.backend.snow import SnowSampler

            assert isinstance(self._sampler, SnowSampler)
            assert isinstance(connection, SnowflakeConnection)
            self._sampler._connection = connection
        if self._sampler.backend == DataBackend.DATABRICKS:
            from sdm_relational_engine.rfm.backend.databricks import (
                Connection,
                DatabricksSampler,
            )

            assert isinstance(self._sampler, DatabricksSampler)
            assert isinstance(connection, Connection)
            self._sampler._connection = connection

    # Helpers #################################################################

    def _parse_query(
        self,
        query: str | ValidatedPredictiveQuery,
    ) -> ValidatedPredictiveQuery:
        if isinstance(query, ValidatedPredictiveQuery):
            return query

        return parse_query_locally(query, self._graph_def)

    @staticmethod
    def _get_task_type(
        query: ValidatedPredictiveQuery,
        edge_types: list[tuple[str, str, str]],
    ) -> TaskType:
        if query.problem_type == ProblemType.FORECAST:
            return TaskType.FORECASTING

        if isinstance(query.target_ast, (Condition, LogicalOperation)):
            return TaskType.BINARY_CLASSIFICATION

        target = query.target_ast
        if isinstance(target, Join):
            target = target.rhs_target
        if isinstance(target, Aggregation):
            if target.aggr == AggregationType.LIST_DISTINCT:
                table_name, col_name = target.get_target_column_name().split(
                    '.'
                )
                target_edge_types = [
                    edge_type
                    for edge_type in edge_types
                    if edge_type[0] == table_name and edge_type[1] == col_name
                ]
                if len(target_edge_types) != 1:
                    raise NotImplementedError(
                        f'Multilabel-classification queries based on '
                        f"'LIST_DISTINCT' are not supported yet. If you "
                        f'planned to write a link prediction query instead, '
                        f"make sure to register '{col_name}' as a "
                        f'foreign key.'
                    )
                return TaskType.TEMPORAL_LINK_PREDICTION

            return TaskType.REGRESSION

        assert isinstance(target, Column)

        if target.stype in {Stype.ID, Stype.categorical}:
            return TaskType.MULTICLASS_CLASSIFICATION

        if target.stype in {Stype.numerical}:
            return TaskType.REGRESSION

        raise NotImplementedError('Task type not yet supported')

    def _get_default_anchor_time(
        self,
        query: ValidatedPredictiveQuery | None = None,
    ) -> pd.Timestamp:
        if query is not None and query.query_type == QueryType.TEMPORAL:
            aggr_table_names = [
                aggr.get_target_column_name().split('.')[0]
                for aggr in query.get_all_target_aggregations()
            ]
            return self._sampler.get_max_time(aggr_table_names)

        return self._sampler.get_max_time()

    def _validate_time(
        self,
        query: ValidatedPredictiveQuery,
        anchor_time: pd.Timestamp,
        context_anchor_time: pd.Timestamp | None,
        evaluate: bool,
    ) -> None:

        if len(self._sampler.time_column_dict) == 0:
            return  # Graph without timestamps

        if query.query_type == QueryType.TEMPORAL:
            aggr_table_names = [
                aggr.get_target_column_name().split('.')[0]
                for aggr in query.get_all_target_aggregations()
            ]
            min_time = self._sampler.get_min_time(aggr_table_names)
            max_time = self._sampler.get_max_time(aggr_table_names)
        else:
            min_time = self._sampler.get_min_time()
            max_time = self._sampler.get_max_time()

        if anchor_time < min_time:
            raise ValueError(
                f"Anchor timestamp '{anchor_time}' is before "
                f"the earliest timestamp '{min_time}' in the "
                f'data.'
            )

        if context_anchor_time is not None and context_anchor_time < min_time:
            raise ValueError(
                f'Context anchor timestamp is too early or '
                f'aggregation time range is too large. To make '
                f'this prediction, we would need data back to '
                f"'{context_anchor_time}', however, your data "
                f"only contains data back to '{min_time}'."
            )

        if query.target_ast.date_offset_range is not None:
            end_offset = query.target_ast.date_offset_range.end_date_offset
        else:
            end_offset = pd.DateOffset(0)

        if (
            context_anchor_time is not None
            and context_anchor_time > anchor_time
        ):
            warnings.warn(
                f'Context anchor timestamp '
                f"(got '{context_anchor_time}') is set to a later "
                f'date than the prediction anchor timestamp '
                f"(got '{anchor_time}'). Please make sure this is "
                f'intended.'
            )
        elif (
            query.query_type == QueryType.TEMPORAL
            and context_anchor_time is not None
            and context_anchor_time + end_offset > anchor_time
        ):
            warnings.warn(
                f'Aggregation for context examples at timestamp '
                f"'{context_anchor_time}' will leak information "
                f'from the prediction anchor timestamp '
                f"'{anchor_time}'. Please make sure this is "
                f'intended.'
            )

        elif (
            context_anchor_time is not None
            and context_anchor_time - end_offset * query.num_forecasts
            < min_time
        ):
            _time = context_anchor_time - end_offset * query.num_forecasts
            warnings.warn(
                f'Context anchor timestamp is too early or '
                f'aggregation time range is too large. To form '
                f'proper input data, we would need data back to '
                f"'{_time}', however, your data only contains "
                f"data back to '{min_time}'."
            )

        if not evaluate and anchor_time > max_time + pd.DateOffset(days=1):
            warnings.warn(
                f"Anchor timestamp '{anchor_time}' is after the "
                f"latest timestamp '{max_time}' in the data. Please "
                f'make sure this is intended.'
            )

        if (
            evaluate
            and anchor_time > max_time - end_offset * query.num_forecasts
        ):
            raise ValueError(
                f'Anchor timestamp for evaluation is after the latest '
                f"supported timestamp '{max_time - end_offset}'."
            )

    def _get_task_table(
        self,
        query: ValidatedPredictiveQuery,
        indices: Sequence[str] | Sequence[float] | Sequence[int] | None,
        anchor_time: pd.Timestamp | Literal['entity'] | None = None,
        context_anchor_time: pd.Timestamp | Literal['entity'] | None = None,
        run_mode: RunMode = RunMode.FAST,
        lag_timesteps: int = 0,
        max_pq_iterations: int = 10,
        random_seed: int | None = _RANDOM_SEED,
        logger: ProgressLogger | None = None,
    ) -> TaskTable:

        anchor_time = _check_anchor_time(anchor_time, 'anchor_time')
        context_anchor_time = _check_anchor_time(
            context_anchor_time, 'context_anchor_time'
        )
        if max_pq_iterations < 1:
            raise ValueError(
                f"'max_pq_iterations' must be greater than zero "
                f'(got {max_pq_iterations})'
            )

        task_type = self._get_task_type(
            query=query,
            edge_types=self._sampler.edge_types,
        )

        num_train_examples = _MAX_CONTEXT_SIZE[run_mode]
        num_test_examples = _MAX_TEST_SIZE[task_type] if indices is None else 0

        if (
            task_type == TaskType.FORECASTING
            and indices is not None
            and len(set(indices)) > 1
        ):
            raise ValueError(
                'Forecasting requires a single entity ID, but got '
                f"{len(set(indices))} unique IDs in 'indices'."
            )

        if logger is not None:
            if task_type == TaskType.BINARY_CLASSIFICATION:
                task_type_repr = 'binary classification'
            elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
                task_type_repr = 'multi-class classification'
            elif task_type == TaskType.REGRESSION:
                task_type_repr = 'regression'
            elif task_type == TaskType.FORECASTING:
                task_type_repr = 'forecasting'
            elif task_type == TaskType.TEMPORAL_LINK_PREDICTION:
                task_type_repr = 'link prediction'
            else:
                task_type_repr = str(task_type)
            logger.log(f'Identified {query.query_type} {task_type_repr} task')

        if query.target_ast.date_offset_range is None:
            step_offset = pd.DateOffset(0)
        else:
            step_offset = query.target_ast.date_offset_range.end_date_offset

        if anchor_time is None:
            anchor_time = self._get_default_anchor_time(query)
            if num_test_examples > 0:
                anchor_time = anchor_time - step_offset * query.num_forecasts

            if logger is not None:
                assert isinstance(anchor_time, pd.Timestamp)
                if anchor_time == pd.Timestamp.min:
                    pass  # Static graph
                elif (
                    anchor_time.hour == 0
                    and anchor_time.minute == 0
                    and anchor_time.second == 0
                    and anchor_time.microsecond == 0
                ):
                    logger.log(f'Derived anchor time {anchor_time.date()}')
                else:
                    logger.log(f'Derived anchor time {anchor_time}')

        if isinstance(anchor_time, pd.Timestamp):
            if context_anchor_time == 'entity':
                raise ValueError(
                    "Anchor time 'entity' needs to be shared "
                    'for context and prediction examples'
                )
            if context_anchor_time is None:
                context_anchor_time = anchor_time - step_offset
            self._validate_time(
                query,
                anchor_time,
                context_anchor_time,
                evaluate=num_test_examples > 0,
            )
        else:
            assert anchor_time == 'entity'
            if query.query_type != QueryType.STATIC:
                raise ValueError(
                    "Anchor time 'entity' is only valid for "
                    'static predictive queries'
                )
            if query.entity_table not in self._sampler.time_column_dict:
                raise ValueError(
                    f"Anchor time 'entity' requires the entity "
                    f"table '{query.entity_table}' to "
                    f'have a time column'
                )
            if isinstance(context_anchor_time, pd.Timestamp):
                raise ValueError(
                    "Anchor time 'entity' needs to be shared "
                    'for context and prediction examples'
                )
            context_anchor_time = 'entity'
        assert context_anchor_time is not None

        # For forecasting with an explicit indices list, pin context sampling
        # to that entity so graph features AND target labels are consistent.
        if task_type == TaskType.FORECASTING and indices is not None:
            query = replace(
                query,
                rfm_entity_ids=Condition(
                    target=Column(fqn=query.entity_column),
                    op='=',
                    value=Constant.from_value(indices[0]),
                ),
            )

        train, test = self._sampler.sample_target(
            query=query,
            num_train_examples=num_train_examples,
            train_anchor_time=context_anchor_time,
            num_train_trials=max_pq_iterations * num_train_examples,
            num_test_examples=num_test_examples,
            test_anchor_time=anchor_time,
            num_test_trials=max_pq_iterations * num_test_examples,
            random_seed=random_seed,
        )
        train_pkey, train_time, train_y = train
        test_pkey, test_time, test_y = test

        if num_test_examples > 0 and logger is not None:
            if task_type == TaskType.BINARY_CLASSIFICATION:
                pos = 100 * int((test_y > 0).sum()) / len(test_y)
                msg = (
                    f'Collected {len(test_y):,} test examples with '
                    f'{pos:.2f}% positive cases'
                )
            elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
                msg = (
                    f'Collected {len(test_y):,} test examples holding '
                    f'{test_y.nunique()} classes'
                )
            elif task_type in {TaskType.REGRESSION, TaskType.FORECASTING}:
                _min, _max = float(test_y.min()), float(test_y.max())
                msg = (
                    f'Collected {len(test_y):,} test examples with targets '
                    f'between {format_value(_min)} and '
                    f'{format_value(_max)}'
                )
            elif task_type == TaskType.TEMPORAL_LINK_PREDICTION:
                num_rhs = test_y.explode().nunique()
                msg = (
                    f'Collected {len(test_y):,} test examples with '
                    f'{num_rhs:,} unique items'
                )
            else:
                raise NotImplementedError
            logger.log(msg)

        if num_test_examples == 0:
            assert indices is not None
            try:
                test_pkey = pd.Series(indices, dtype=train_pkey.dtype)
            except (TypeError, ValueError) as error:
                bad = _first_uncastable(indices, train_pkey.dtype)
                where = (
                    f'; indices[{bad[0]}] is {bad[1]!r}'
                    if bad is not None
                    else ''
                )
                raise ValueError(
                    f"'indices' must hold values matching the primary key of "
                    f"'{query.entity_table}', which is {train_pkey.dtype}"
                    f'{where}'
                ) from error
            if isinstance(anchor_time, pd.Timestamp):
                test_time = (
                    pd.Series([anchor_time])
                    .repeat(len(indices))
                    .reset_index(drop=True)
                )
            else:
                train_time = test_time = 'entity'

        if logger is not None:
            if task_type == TaskType.BINARY_CLASSIFICATION:
                pos = 100 * int((train_y > 0).sum()) / len(train_y)
                msg = (
                    f'Collected {len(train_y):,} in-context examples with '
                    f'{pos:.2f}% positive cases'
                )
            elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
                msg = (
                    f'Collected {len(train_y):,} in-context examples '
                    f'holding {train_y.nunique()} classes'
                )
            elif task_type in {TaskType.REGRESSION, TaskType.FORECASTING}:
                _min, _max = float(train_y.min()), float(train_y.max())
                msg = (
                    f'Collected {len(train_y):,} in-context examples with '
                    f'targets between {format_value(_min)} and '
                    f'{format_value(_max)}'
                )
            elif task_type == TaskType.TEMPORAL_LINK_PREDICTION:
                num_rhs = train_y.explode().nunique()
                msg = (
                    f'Collected {len(train_y):,} in-context examples with '
                    f'{num_rhs:,} unique items'
                )
            else:
                raise NotImplementedError
            logger.log(msg)

        entity_table_names: tuple[str] | tuple[str, str]
        if task_type.is_link_pred:
            final_aggr = query.get_final_target_aggregation()
            assert final_aggr is not None
            edge_fkey = final_aggr.get_target_column_name()
            for edge_type in self._sampler.edge_types:
                if edge_fkey == f'{edge_type[0]}.{edge_type[1]}':
                    entity_table_names = (
                        query.entity_table,
                        edge_type[2],
                    )
        else:
            entity_table_names = (query.entity_table,)

        context_df = pd.DataFrame({'ENTITY': train_pkey, 'TARGET': train_y})
        if isinstance(train_time, pd.Series):
            context_df['ANCHOR_TIMESTAMP'] = train_time
        pred_df = pd.DataFrame({'ENTITY': test_pkey})
        if num_test_examples > 0:
            pred_df['TARGET'] = test_y
        if isinstance(test_time, pd.Series):
            pred_df['ANCHOR_TIMESTAMP'] = test_time

        task = TaskTable(
            task_type=task_type,
            context_df=context_df,
            pred_df=pred_df,
            entity_table_name=entity_table_names,
            entity_column='ENTITY',
            target_column='TARGET',
            time_column='ANCHOR_TIMESTAMP'
            if isinstance(train_time, pd.Series)
            else TaskTable.ENTITY_TIME,
            num_forecasts=query.num_forecasts,
            step_size=_date_offset_to_ns(step_offset)
            if task_type == TaskType.FORECASTING
            else None,
        )

        if lag_timesteps < 0:
            raise ValueError(
                f"'lag_timesteps' cannot be negative (got {lag_timesteps})"
            )

        if (
            query.query_type == QueryType.TEMPORAL
            and task_type
            in {
                TaskType.BINARY_CLASSIFICATION,
                TaskType.FORECASTING,
                TaskType.MULTICLASS_CLASSIFICATION,
                TaskType.REGRESSION,
            }
            and lag_timesteps > 0
        ):
            task = self.add_lagged_target(task, query, lag_timesteps)

        return task

    def _get_context(
        self,
        task: TaskTable,
        run_mode: RunMode | str = RunMode.FAST,
        num_neighbors: list[int] | None = None,
        exclude_cols_dict: dict[str, list[str]] | None = None,
        top_k: int | None = None,
        random_seed: int | None = _RANDOM_SEED,
        _validate_references: bool = True,
    ) -> PredictionContext:

        if num_neighbors is None:
            key = (
                RunMode.FAST
                if task.task_type.is_link_pred
                else RunMode(run_mode)
            )
            num_neighbors = _DEFAULT_NUM_NEIGHBORS[key][:2]

        if len(num_neighbors) > _MAX_HOPS:
            raise ValueError(
                f'Cannot predict on subgraphs with more than {_MAX_HOPS} '
                f'hops (got {len(num_neighbors)}). Reduce the '
                f'number of hops and try again. Please create a '
                f'feature request at '
                f"'https://github.com/NVIDIA/structured-data-models-client' if you "
                f'must go beyond this for your use-case.'
            )

        if _validate_references:
            self._validate_task_references(task)

        entity_pkey = pd.concat(
            [
                task._context_df[task._entity_column],
                task._pred_df[task._entity_column],
            ],
            axis=0,
            ignore_index=True,
        )

        if task.use_entity_time:
            if task.entity_table_name not in self._sampler.time_column_dict:
                raise ValueError(
                    f'The given anchor time requires the entity '
                    f"table '{task.entity_table_name}' to have a "
                    f'time column'
                )
            anchor_time = 'entity'
        elif task._time_column is not None:
            anchor_time = pd.concat(
                [
                    task._context_df[task._time_column],
                    task._pred_df[task._time_column],
                ],
                axis=0,
                ignore_index=True,
            )
        else:
            anchor_time = (
                pd.Series(self._get_default_anchor_time())
                .repeat(len(entity_pkey))
                .reset_index(drop=True)
            )

        subgraph = self._sampler.sample_subgraph(
            entity_table_names=task.entity_table_names,
            entity_pkey=entity_pkey,
            anchor_time=anchor_time,
            num_neighbors=num_neighbors,
            exclude_cols_dict=exclude_cols_dict,
            random_seed=random_seed,
        )

        if len(subgraph.tables) > _MAX_SUBGRAPH_TABLES:
            raise ValueError(
                f'Cannot query from a graph with more than '
                f'{_MAX_SUBGRAPH_TABLES} '
                f'tables (got {len(subgraph.tables)}). '
                f'Please create a feature request at '
                f"'https://github.com/NVIDIA/structured-data-models-client' if you "
                f'must go beyond this for your use-case.'
            )

        if (
            task.task_type.is_link_pred
            and task.entity_table_names[-1] not in subgraph.tables
        ):
            raise ValueError(
                'Cannot perform link prediction on subgraphs '
                'without any historical target entities. Please '
                'increase the number of hops and try again.'
            )

        return PredictionContext(
            task_type=task.task_type,
            entity_table_names=task.entity_table_names,
            sampled_subgraph=subgraph,
            y_train=task._context_df[task.target_column.name],
            y_test=task._pred_df[task.target_column.name]
            if task.has_prediction_targets
            else None,
            task_table=SampledTable(
                df=pd.concat(
                    [
                        task._context_df[
                            [c.name for c in task.feature_columns]
                        ],
                        task._pred_df[[c.name for c in task.feature_columns]],
                    ],
                    axis=0,
                    ignore_index=True,
                ),
                row=None,
                batch=np.arange(task._num_rows),
                num_sampled_nodes=[],
                stype_dict={
                    column.name: column.stype for column in task.feature_columns
                },
                primary_key=None,
            )
            if len(task.feature_columns) > 0
            else None,
            top_k=top_k,
            step_size=task.step_size,
            num_forecasts=task.num_forecasts,
        )


def _date_offset_to_ns(offset: pd.DateOffset) -> int | None:
    r"""Convert a pandas DateOffset to an integer number of nanoseconds."""
    ref = Timestamp('2020-01-01')
    delta = (ref + offset) - ref
    return int(delta.total_seconds()) * 1_000_000_000


def format_value(value: int | float) -> str:
    if value == int(value):
        return f'{int(value):,}'
    if abs(value) >= 1000:
        return f'{value:,.0f}'
    if abs(value) >= 10:
        return f'{value:.1f}'
    return f'{value:.2f}'
