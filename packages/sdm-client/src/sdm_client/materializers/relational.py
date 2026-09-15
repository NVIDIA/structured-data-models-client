# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Gather and encode relational requests with the relational engine."""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from numbers import Integral
from typing import TYPE_CHECKING, Any

import pandas as pd
from predictive_query.ast import Filter
from predictive_query.validator.model_validator import (
    MAX_TOP_K as _MAX_TOP_K,
)
from predictive_query.validator.utils import table_name

from sdm_client.core import dates
from sdm_client.core.warnings import caller_stacklevel
from sdm_client.errors import MissingExtraError, StructuredDataError
from sdm_client.models import RELATIONAL_LIMITS
from sdm_client.request import Request
from sdm_client.schema import Schema, Table, _as_key_tuple

if TYPE_CHECKING:
    from predictive_query import TaskType, ValidatedPredictiveQuery
    from sdm_relational_engine.rfm import (
        Graph,
        KumoRelational,
        MaterializedPredictionRequest,
        TaskTable,
    )


class RelationalMaterializer:
    r"""Materialize resolved PQL against a relational schema."""

    _MATERIALIZE_OPTIONS = frozenset(
        {
            'anchor_time',
            'context_anchor_time',
            'lag_timesteps',
            'max_pq_iterations',
            'num_hops',
            'num_neighbors',
            'random_seed',
            'use_prediction_time',
            'inference_config',
            'exclude_cols_dict',
            'return_embeddings',
            'top_k',
        }
    )

    def __init__(self, schema: Schema) -> None:
        self.schema = schema
        self._model: KumoRelational | None = None

    def materialize(
        self,
        *,
        query: ValidatedPredictiveQuery,
        indices: Sequence[Any],
        num_context_samples: int | None,
        options: dict[str, Any],
    ) -> Request:
        r"""Gather the selected entities and return their encoded request."""
        unknown = sorted(set(options) - self._MATERIALIZE_OPTIONS)
        if unknown:
            raise StructuredDataError(
                f'unknown materialize option(s) {unknown}; accepted: '
                f'{sorted(self._MATERIALIZE_OPTIONS)}',
                code='INVALID_REQUEST',
            )
        if 'run_mode' in options:  # pragma: no cover - caught by the above
            raise StructuredDataError(
                'run_mode is not a knob here; set num_context_samples and '
                'the client derives what the endpoint needs',
                code='INVALID_REQUEST',
            )

        _warn_the_filter_chose_only_the_examples(query, indices)
        payloads = self._gather(query, indices, num_context_samples, options)
        _drop_context_that_answers_the_question(
            payloads,
            windowed=getattr(query.target_ast, 'date_offset_range', None)
            is not None,
        )
        return Request.materialized(
            payloads=payloads,
            schema=self.schema.snapshot(),
            query=query.to_string(),
            indices=list(indices),
            entity_table=table_name(query.entity_column),
            requested_inference={
                'inference_config': options.get('inference_config') or {}
            },
        )

    def _gather(
        self,
        query: ValidatedPredictiveQuery,
        indices: Sequence[Any],
        num_context_samples: int | None,
        options: dict[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        r"""Sample and encode using the engine's own materializer."""
        run_mode = self._run_mode_for(num_context_samples)
        self._check_entity_count(indices)
        self._check_shape(options, RELATIONAL_LIMITS)
        with self._engine_errors():
            model = self._get_model()
        _warn_rows_the_graph_dropped(model)

        with self._engine_errors(model):
            task = model._get_task_table(
                query=query,
                indices=list(indices),
                run_mode=run_mode,
                **{
                    key: options[key]
                    for key in (
                        'anchor_time',
                        'context_anchor_time',
                        'lag_timesteps',
                        'max_pq_iterations',
                        'random_seed',
                    )
                    if key in options
                },
            )
        task._query = query.to_string()
        if not hasattr(task, 'task_type'):
            raise StructuredDataError(
                'the resolved task does not say what kind it is, so the '
                'entity limit for it cannot be checked',
                code='INTERNAL_ERROR',
            )
        self._check_entity_count(indices, task.task_type)
        task = self._narrow_context(task, num_context_samples)

        with self._engine_errors(model):
            parts = model.materialize_task(
                task,
                run_mode=run_mode,
                verbose=False,
                top_k=_candidates_to_rank(query, options),
                exclude_cols_dict=_columns_to_leave_out(query, options),
                **{
                    key: options[key]
                    for key in (
                        'num_neighbors',
                        'num_hops',
                        'inference_config',
                        'use_prediction_time',
                        'return_embeddings',
                        'random_seed',
                    )
                    if key in options
                },
            )
        if not parts:
            raise StructuredDataError(
                'the engine gathered nothing for this query',
                code='INVALID_REQUEST',
            )
        for part in parts:
            self._check_limits(part, RELATIONAL_LIMITS)
        return tuple(part.payload for part in parts)

    def _get_model(self) -> KumoRelational:
        if self._model is None:
            try:
                from sdm_relational_engine.rfm import KumoRelational
            except ImportError as error:  # pragma: no cover - optional extra
                raise MissingExtraError(
                    'relational', 'sdm-relational-engine'
                ) from error

            self._model = KumoRelational(self._build_graph(), verbose=False)
        return self._model

    def _build_graph(self) -> Graph:
        r"""Build exactly the graph described by the client schema."""
        from sdm_relational_engine.rfm import Graph

        self.schema.check_consistent()

        frames = {
            name: _as_typed(table) for name, table in self.schema.tables.items()
        }
        graph = Graph.from_data(frames, edges=[], verbose=False)

        for name, table in self.schema.tables.items():
            engine_table = graph[name]
            if table.primary_key is not None:
                engine_table.primary_key = table.primary_key
            else:
                # Said, not merely left unsaid. The engine infers a key from a
                # column *name* without checking uniqueness, so a child's
                # foreign key called `ID` -- holding [1, 1, 2, 2, 3] -- becomes
                # that table's primary key, and linking it is then refused for
                # being one. The client let the caller set the key to None and
                # then never passed that on, so a correct schema over correct
                # data could not be materialized and the error named a key the
                # caller had explicitly removed.
                engine_table.primary_key = None
            if table.time_column is not None:
                engine_table.time_column = table.time_column
            for column_name, column in table.columns.items():
                if column_name in engine_table:
                    engine_table[column_name].stype = column.stype

        linked: set[tuple[str, str, str]] = set()
        for link in self.schema.relationships:
            edge = (link.source_table, link.source_column, link.target_table)
            if edge in linked:
                continue
            linked.add(edge)
            graph.link(
                src_table=link.source_table,
                fkey=link.source_column,
                dst_table=link.target_table,
            )

        sent = {
            (edge.src_table, edge.fkey, edge.dst_table) for edge in graph.edges
        }
        described = {
            (link.source_table, link.source_column, link.target_table)
            for link in self.schema.relationships
        }
        misdirected = [
            link
            for link in self.schema.relationships
            if link.target_column
            not in _as_key_tuple(self.schema[link.target_table].primary_key)
        ]
        if misdirected:
            shown = ', '.join(
                f'{link.source_table}.{link.source_column} -> '
                f'{link.target_table}.{link.target_column}'
                for link in misdirected
            )
            raise StructuredDataError(
                f'{shown} name a target column that is not the primary key '
                f'of the table they point at, and an edge can only join to that',
                code='INVALID_REQUEST',
            )
        if sent != described:  # pragma: no cover - defensive assertion
            raise StructuredDataError(
                f'the graph built for this request does not match the schema: '
                f'{sorted(sent ^ described)}',
                code='INTERNAL_ERROR',
            )
        return graph

    @contextmanager
    def _engine_errors(
        self, model: KumoRelational | None = None
    ) -> Iterator[None]:
        r"""Re-raise engine failures through this client's error surface.

        ``model`` is optional because the phase that most needs this is the one
        that builds it. `Graph.from_data`, the key and stype assignments, the
        links and `graph.validate()` all raise bare `ValueError` out of a
        package the caller never imported, and all of them used to run one line
        above this guard -- so a caller who wrote the `except
        StructuredDataError` the reference tells them to write did not catch
        the whole construction phase.
        """
        try:
            from sdm_relational_engine import KumoRelationalError
        except ImportError:  # pragma: no cover - the extra is optional
            caller_errors = (ValueError, KeyError, TypeError)
        else:
            caller_errors = (
                ValueError,
                KeyError,
                TypeError,
                KumoRelationalError,
            )

        try:
            yield
        except StructuredDataError:
            raise
        except Exception as error:
            if not isinstance(error, caller_errors):
                raise StructuredDataError(
                    f'the relational engine failed while building this '
                    f'request: {type(error).__name__}: {error}',
                    code='INTERNAL_ERROR',
                ) from error
            if isinstance(error, TypeError) and self._is_call_signature_error(
                error
            ):
                raise StructuredDataError(
                    f'the relational engine was called incorrectly by this '
                    f'client: {type(error).__name__}: {error}',
                    code='INTERNAL_ERROR',
                ) from error
            raise StructuredDataError(
                f'{error}{self._sanitization_detail(model)}',
                code='INVALID_REQUEST',
            ) from error

    @staticmethod
    def _is_call_signature_error(error: TypeError) -> bool:
        r"""Whether a ``TypeError`` is a bad call rather than a bad value."""
        traceback = error.__traceback__
        if traceback is not None:
            deepest = traceback
            while deepest.tb_next is not None:
                deepest = deepest.tb_next
            if deepest.tb_frame.f_globals.get('__name__') == __name__:
                return True
        message = str(error)
        return any(
            phrase in message
            for phrase in (
                'unexpected keyword argument',
                'positional argument',
                'required argument',
                'takes no arguments',
            )
        )

    @staticmethod
    def _sanitization_detail(model: KumoRelational) -> str:
        r"""Describe rows the engine dropped while constructing its graph."""
        report = getattr(model, 'sanitization_report', None)
        tables = getattr(report, 'tables', None)
        if not tables:
            return ''
        reasons = []
        for name, table in tables.items():
            dropped = getattr(table, 'input_rows', 0) - getattr(
                table, 'output_rows', 0
            )
            if dropped <= 0:
                continue
            causes = [
                f'{count} with {label}'
                for label, count in (
                    (
                        'no primary key',
                        getattr(table, 'null_primary_key_rows', 0),
                    ),
                    (
                        'a repeated primary key',
                        getattr(table, 'duplicate_primary_key_rows', 0),
                    ),
                    ('no time', getattr(table, 'null_time_rows', 0)),
                )
                if count
            ]
            reasons.append(
                f'{name}: {dropped} of {table.input_rows} rows dropped'
                + (f' ({", ".join(causes)})' if causes else '')
            )
        return f' [{"; ".join(reasons)}]' if reasons else ''

    @staticmethod
    def _check_entity_count(
        indices: Sequence[object], task_kind: TaskType | None = None
    ) -> None:
        r"""Refuse more entities than the endpoint will score."""
        ceiling = RELATIONAL_LIMITS['predict_rows_max']
        described = 'the endpoint scores at most'
        if task_kind is not None and 'link_prediction' in str(task_kind):
            ceiling = RELATIONAL_LIMITS['predict_rows_temporal_link_prediction']
            described = 'a temporal link prediction scores at most'
        if len(indices) > ceiling:
            raise StructuredDataError(
                f'{len(indices):,} entities were asked for, and {described} '
                f'{ceiling:,} at a time; split them across calls',
                code='UNSUPPORTED_REQUEST',
            )

    @staticmethod
    def _narrow_context(
        task: TaskTable, num_context_samples: int | None
    ) -> TaskTable:
        r"""Narrow gathered context to the requested number of examples."""
        if num_context_samples is None:
            return task
        gathered = task.num_context_examples
        if gathered <= num_context_samples:
            return task
        return task.narrow_context(0, num_context_samples)

    @staticmethod
    def _run_mode_for(num_context_samples: int | None) -> str:
        r"""Return the endpoint run mode admitting the requested context."""
        if num_context_samples is None:
            return 'fast'
        if num_context_samples <= RELATIONAL_LIMITS['context_rows_fast']:
            return 'fast'
        if num_context_samples <= RELATIONAL_LIMITS['context_rows_max']:
            return 'best'
        raise StructuredDataError(
            f'num_context_samples is {num_context_samples:,}, and the endpoint '
            f'accepts at most {RELATIONAL_LIMITS["context_rows_max"]:,} context '
            f'rows',
            code='UNSUPPORTED_REQUEST',
        )

    @staticmethod
    def _check_limits(
        part: MaterializedPredictionRequest, limits: Mapping[str, int]
    ) -> None:
        r"""Refuse a gathered part the endpoint would reject before sending it."""
        payload = getattr(part, 'payload', None)
        if not isinstance(payload, dict):
            return
        predict = payload.get('predict', {}).get('instance_table', {})
        rows = predict.get('rows')
        if rows is None:
            return
        kind = payload.get('task', {}).get('kind')
        cap = (
            limits['predict_rows_temporal_link_prediction']
            if kind == 'temporal_link_prediction'
            else limits['predict_rows_max']
        )
        if len(rows) > cap:
            raise StructuredDataError(
                f'this batch scores {len(rows)} rows and the endpoint accepts at '
                f'most {cap} for a {kind} task',
                code='UNSUPPORTED_REQUEST',
            )

    def _check_shape(
        self, options: Mapping[str, Any], limits: Mapping[str, int]
    ) -> None:
        r"""Refuse a graph wider or deeper than the endpoint will read."""
        if len(self.schema) > limits['tables_max']:
            raise StructuredDataError(
                f'this schema has {len(self.schema)} tables and the endpoint '
                f'reads at most {limits["tables_max"]}',
                code='UNSUPPORTED_REQUEST',
            )
        neighbors = options.get('num_neighbors')
        if neighbors is not None and (
            not isinstance(neighbors, Sequence)
            or isinstance(neighbors, (str, bytes))
        ):
            raise StructuredDataError(
                'num_neighbors must be a sequence of hop sizes',
                code='INVALID_REQUEST',
            )

        num_hops = options.get('num_hops')
        if num_hops is not None and (
            isinstance(num_hops, bool) or not isinstance(num_hops, Integral)
        ):
            raise StructuredDataError(
                f'num_hops must be a whole number, got '
                f'{type(num_hops).__name__} {num_hops!r}',
                code='INVALID_REQUEST',
            )

        hops = len(neighbors) if neighbors else num_hops
        if hops is not None and hops > limits['hops_max']:
            raise StructuredDataError(
                f'this request samples {hops} hops and the endpoint reads at most '
                f'{limits["hops_max"]}',
                code='UNSUPPORTED_REQUEST',
            )


def _as_typed(table: Table) -> pd.DataFrame:
    r"""The table's rows, with any decimal column cast to what it was typed.

    A warehouse `DECIMAL` arrives from pandas as an `object` column of
    :class:`~decimal.Decimal`. The schema layer reads the values and types it
    -- `int64` for whole numbers, `float64` otherwise -- and the tabular path
    sends it happily. The relational path handed the engine the raw frame
    instead, which re-infers from the object column, reaches a different answer
    and eventually meets a `Decimal` where a dtype name belongs:

        dtype '<class 'decimal.Decimal'>' not understood

    naming no table, no column and no remedy. Since every numeric column of a
    Databricks or Snowflake table is `DECIMAL`, that is the first question
    anyone asks of a warehouse through this SDK's own connector.

    The two halves also disagreed about what such a column is. The engine maps
    decimal content to float unconditionally, so an exact identifier became a
    float and composite keys over warehouse data were refused outright. Handing
    over the type the schema already computed settles both: there is one answer
    now, and it is the one the schema reports.
    """
    frame = table.data
    if frame is None or frame.empty:
        return frame
    cast = {}
    for name, column in table.columns.items():
        if name not in frame.columns:
            continue
        where = f'{table.name}.{name}'
        if name == table.time_column and _holds_text(frame[name]):
            cast[name] = _to_time(frame[name], where)
        elif _holds_decimal(frame[name]):
            cast[name] = _to_declared(frame[name], column.dtype, where)
    return frame.assign(**cast) if cast else frame


def _holds_text(values: pd.Series) -> bool:
    r"""Whether this column holds strings, rather than merely being `object`.

    An `object` column can hold `datetime.date` or `pd.Timestamp` as readily as
    text, and `infer_tfm_dtype` already calls those `timestamp[us]`. Reading
    the dtype alone would send a column of real dates down the text path and
    refuse it for being ambiguous, which it is not.
    """
    if not (
        pd.api.types.is_object_dtype(values.dtype)
        or pd.api.types.is_string_dtype(values.dtype)
    ):
        return False
    return pd.api.types.infer_dtype(values, skipna=True) == 'string'


def _to_time(values: pd.Series, where: str) -> pd.Series:
    r"""A string time column read as dates, or refused for saying two things.

    `supports_dtype('timestamp', 'string')` is true, so a column of date
    strings can be named as a table's time column and is accepted. The strings
    then have to become timestamps before the model sees them, and that was
    left to `pd.to_datetime`'s defaults, which read an ambiguous slash date
    month-first. A day-first column was silently re-dated -- twelve monthly
    events across a year became twelve daily events in one week of February --
    and nothing in the request, the schema or the response recorded that a
    conversion had happened at all. It surfaced, when it surfaced, as an
    unrelated complaint about anchor times.

    So only a year-first column is read, because only a year-first column says
    one thing. `02/01/2020` is the 2nd of January to half the world and the 1st
    of February to the other half, and the column does not say which; the
    caller does, by converting it themselves.

    The shape check reads a sample and is only a pre-filter; the conversion
    below reads every value, so a column that changes shape in its last row is
    caught there rather than here.
    """
    present = values.dropna()
    if present.empty:
        return values
    if dates.looks_unambiguous(present):
        how = dates.conversion_that_works(present)
        if how is not None:
            return dates.read(values, how)
    example = next(
        (value for value in present if isinstance(value, str)), present.iloc[0]
    )
    column = where.split('.')[-1]
    raise StructuredDataError(
        f'{where} is the time column and holds text such as {example!r}, '
        f'which does not say unambiguously what date it is: read day-first it '
        f'is one date and month-first another, and nothing in the column says '
        f'which. Convert it yourself so the reading is yours -- '
        f'frame[{column!r}] = pd.to_datetime(frame[{column!r}], '
        f'dayfirst=True) -- or store it year-first',
        code='INVALID_REQUEST',
    )


def _holds_decimal(values: pd.Series) -> bool:
    if isinstance(getattr(values.dtype, 'pyarrow_dtype', None), object):
        if 'decimal' in str(getattr(values.dtype, 'pyarrow_dtype', '')):
            return True
    if not pd.api.types.is_object_dtype(values.dtype):
        return False
    return pd.api.types.infer_dtype(values, skipna=True) == 'decimal'


def _to_declared(values: pd.Series, dtype: str, where: str) -> pd.Series:
    r"""``values`` as ``dtype``, exactly.

    Only what the schema asked for. A whole decimal too wide for an `int64` is
    typed `string` precisely so its digits survive, and reading "not an int" as
    "a float" would put 9223372036854775808 back as 9.223372036854776e+18 --
    the corruption this exists to prevent, reintroduced one branch over.

    A column with a missing value cannot be a plain `int64`, so it becomes
    pandas' nullable `Int64` rather than widening to float and rounding an
    identifier.
    """
    if dtype.startswith('float'):
        return pd.to_numeric(values.astype(object), errors='coerce').astype(
            'float64'
        )
    if not dtype.startswith('int'):
        # `string`, for a whole number wider than an int64 holds. Only the
        # digits matter and only a string keeps all of them.
        return values.map(lambda value: None if pd.isna(value) else str(value))
    # int() on the Decimal, never pd.to_numeric: that converts through a
    # float64 and 9007199254740993 comes back 9007199254740992. These are
    # identifiers, and the low bit is the whole of their job.
    fractional = values.map(
        lambda value: not pd.isna(value) and value != int(value)
    )
    if fractional.any():
        example = values[fractional].iloc[0]
        raise StructuredDataError(
            f'column {where} is declared {dtype!r} and holds {example!r}, '
            f'which is not a whole number. Sending it would round the value '
            f'without saying so; declare the column float64, or round it '
            f'yourself if that is what you meant',
            code='INVALID_REQUEST',
        )
    whole = values.map(lambda value: pd.NA if pd.isna(value) else int(value))
    if whole.isna().any():
        return whole.astype('Int64')
    return whole.astype('int64')


def _columns_to_leave_out(
    query: ValidatedPredictiveQuery, options: Mapping[str, Any]
) -> dict[str, list[str]]:
    r"""Every column either the query or the caller asked to keep back.

    `exclude_cols_dict` was accepted by `materialize()`, checked against the
    options this route takes, and then not passed on: the query's own answer
    was sent in its place, so the one option someone reaches for to keep a
    column away from the model did nothing.

    Merged rather than replaced. The query excludes columns for reasons of its
    own -- the target among them -- and letting a caller's list stand in for
    that would hand the model back something the query had deliberately
    removed.
    """
    merged = {
        table: list(columns)
        for table, columns in (query.get_exclude_cols_dict() or {}).items()
    }
    # `.get(name)` and not `.get(name) or {}`: an empty list is not a mapping
    # and saying nothing about it is the same silence this is fixing, so only
    # an absent option and an explicit `None` mean "nothing to exclude".
    given = options.get('exclude_cols_dict')
    if given is None:
        given = {}
    if not isinstance(given, Mapping):
        raise StructuredDataError(
            f'exclude_cols_dict must map a table name to the columns to leave '
            f'out, got {type(given).__name__}',
            code='INVALID_REQUEST',
        )
    for table, columns in given.items():
        if isinstance(columns, str) or not isinstance(columns, Sequence):
            raise StructuredDataError(
                f'exclude_cols_dict[{table!r}] must be a sequence of column '
                f'names, got {type(columns).__name__}',
                code='INVALID_REQUEST',
            )
        kept = merged.setdefault(table, [])
        kept.extend(column for column in columns if column not in kept)
    return merged


def _candidates_to_rank(
    query: ValidatedPredictiveQuery, options: Mapping[str, Any]
) -> int | None:
    r"""How many candidates to rank, held to the limits the language holds.

    A `TOP K` written in the query is checked -- at least one, and no more than
    the endpoint ranks. Passing it as an option went straight through to the
    task, so an override could put a number in a request that the same number
    written in the query would have been refused for.
    """
    if 'top_k' not in options:
        return query.top_k
    given = options['top_k']
    if isinstance(given, bool) or not isinstance(given, Integral):
        raise StructuredDataError(
            f'top_k must be a whole number of candidates, got '
            f'{type(given).__name__}',
            code='INVALID_REQUEST',
        )
    if not 1 <= int(given) <= _MAX_TOP_K:
        raise StructuredDataError(
            f'top_k is {given}, and this model ranks between 1 and '
            f'{_MAX_TOP_K} candidates',
            code='INVALID_REQUEST',
        )
    return int(given)


def _warn_rows_the_graph_dropped(model: KumoRelational) -> None:
    r"""Say what the engine left out while building the graph.

    A fact row with no time, no key, or a key another row already used cannot
    be placed in the graph and is dropped. The count and the reason are worked
    out either way -- `_sanitization_detail` builds the sentence -- and it was
    only ever shown when something else had already failed. So a request that
    succeeded was answered from a graph smaller than the one the caller
    passed, the result looked ordinary, and the number was sitting there
    unread.
    """
    detail = RelationalMaterializer._sanitization_detail(model)
    if not detail:
        return
    warnings.warn(
        f'the graph was built without some of the rows given to it '
        f'{detail.strip()}. Those rows are not in the context the model sees; '
        f'fill the missing values or drop the rows yourself if that is not '
        f'what you meant.',
        stacklevel=caller_stacklevel(),
    )


#: What the wire calls the column carrying the entity a row is about.
_ENTITY_REF = '__kumo_entity_ref'

#: And the moment it is asked about.
_ANCHOR = '__kumo_anchor_time'


def _drop_context_that_answers_the_question(
    payloads: tuple[dict[str, Any], ...], *, windowed: bool = True
) -> None:
    r"""Take out of the context any row that is a scored row at its own anchor.

    On a target with no time window the entity being scored is sampled into the
    context as well, at the same anchor, carrying the value being asked for --
    so the model is shown the answer to the question it is being asked.
    Measured against the live endpoint on a label that is pure noise, a hundred
    scored entities came back 0.660 correct where chance is 0.500.

    **What counts as the same question depends on the target.** For an
    aggregation, an entity in the context at an *earlier* anchor is not a leak:
    it is an earlier example, with its features cut at that anchor and its
    label window closed before the prediction window opens, and it is what
    temporal in-context learning runs on. Dropping those would delete the
    examples an aggregation depends on, to fix a leak that is not there. So
    there, only the same entity at the same moment goes.

    A static column is not a window over anything. Its value does not move with
    the anchor, so the same entity at *any* anchor carries the answer being
    asked for -- which a caller reaches by setting `context_anchor_time`, and
    which matching on the anchor alone would have let straight through.

    The two halves carry their own `instance_id` space, so removing a context
    row means removing the related rows that hang off it and nothing else.
    """
    found: list[tuple[Mapping[str, Any], set[Any]]] = []
    for payload in payloads:
        context = payload.get('context')
        predict = payload.get('predict')
        if not isinstance(context, Mapping) or not isinstance(predict, Mapping):
            continue
        anchor_column = (payload.get('task') or {}).get(
            'anchor_time_column'
        ) or _ANCHOR
        asked = _entities_at_anchors(
            predict.get('instance_table'), anchor_column, windowed
        )
        if not asked:
            continue
        answered = _instances_matching(
            context.get('instance_table'), asked, anchor_column, windowed
        )
        if answered:
            found.append((context, answered))
    if not found:
        return

    # Warning before removing anything, because a caller running with
    # `simplefilter('error')` turns this into a raise, and a raise partway
    # through leaves the payloads it had already reached stripped of rows the
    # caller never learned were gone.
    warnings.warn(
        f'{sum(len(a) for _, a in found)} of the rows sampled as context are '
        f'the rows being scored, at the same moment, carrying the value being '
        f'asked for. They have been left out, so the model is not shown the '
        f'answers to its own question; the context is that many rows smaller '
        f'than it would otherwise have been.',
        stacklevel=caller_stacklevel(),
    )
    for context, answered in found:
        _remove_instances(context.get('instance_table'), answered)
        for related in (context.get('related_tables') or {}).values():
            _remove_instances(related, answered)


def _positions(
    table: Any, anchor_column: str
) -> tuple[list[Any], list[Any], int | None, int | None]:
    r"""Where the entity and the anchor sit in a wire table.

    The anchor column is read from the task rather than assumed: the engine
    suffixes the name when it would collide with a real column, and looking
    only for the bare name went blind on exactly those graphs.
    """
    columns = (table or {}).get('columns')
    rows = (table or {}).get('rows')
    if not isinstance(columns, list) or not isinstance(rows, list):
        return [], [], None, None
    reference = next(
        (
            index
            for index, name in enumerate(columns)
            if str(name).startswith(_ENTITY_REF)
        ),
        None,
    )
    anchor = columns.index(anchor_column) if anchor_column in columns else None
    return columns, rows, reference, anchor


def _entities_at_anchors(
    table: Any, anchor_column: str = _ANCHOR, windowed: bool = True
) -> set[tuple[Any, Any]]:
    columns, rows, reference, anchor = _positions(table, anchor_column)
    if reference is None or (windowed and anchor is None):
        return set()
    needed = max(reference, anchor) if windowed else reference
    return {
        (row[reference], row[anchor] if windowed else None)
        for row in rows
        if isinstance(row, list) and len(row) > needed
    }


def _instances_matching(
    table: Any,
    asked: set[tuple[Any, Any]],
    anchor_column: str = _ANCHOR,
    windowed: bool = True,
) -> set[Any]:
    columns, rows, reference, anchor = _positions(table, anchor_column)
    if reference is None or 'instance_id' not in columns:
        return set()
    if windowed and anchor is None:
        return set()
    at = columns.index('instance_id')
    needed = max(reference, at, anchor) if windowed else max(reference, at)
    return {
        row[at]
        for row in rows
        if isinstance(row, list)
        and len(row) > needed
        and (row[reference], row[anchor] if windowed else None) in asked
    }


def _remove_instances(table: Any, instances: set[Any]) -> None:
    columns = (table or {}).get('columns')
    rows = (table or {}).get('rows')
    if not isinstance(columns, list) or not isinstance(rows, list):
        return
    if 'instance_id' not in columns:
        return
    at = columns.index('instance_id')
    table['rows'] = [
        row
        for row in rows
        if not (
            isinstance(row, list) and len(row) > at and row[at] in instances
        )
    ]


def _warn_the_filter_chose_only_the_examples(
    query: ValidatedPredictiveQuery, indices: Sequence[Any]
) -> None:
    r"""Say that an entity filter shapes the context and not the scored rows.

    `FOR EACH sellers.seller_id WHERE sellers.seller_state = 'RJ'` reads as one
    question and is answered as two. The filter selects the labelled examples
    the model learns from; the rows actually scored are the caller's indices,
    untouched by it. Measured on synthetic data, a filter matching 12 of 50
    named sellers still scored all 50 -- so the model was shown a thousand
    northern sellers and then asked about thirty-eight southern ones, with
    nothing said and predictions that look entirely ordinary.

    Reported rather than repaired. Applying the filter here would mean
    evaluating the condition a second time, against the sampled neighbourhood
    the engine builds to evaluate it the first time, and two implementations of
    one condition disagreeing is a worse failure than the one being fixed.
    Naming the condition is what the caller needs to act, either by narrowing
    the indices or by dropping the filter.
    """
    if not indices or not isinstance(query.entity_ast, Filter):
        return
    warnings.warn(
        f'the entity filter {str(query.entity_ast.condition)!r} selects the '
        f'labelled examples the model learns from, and does not narrow the '
        f'{len(indices)} rows named for scoring -- those are scored as given, '
        f'whether or not they satisfy it. Pass only indices that satisfy the '
        f'condition if that is what you meant.',
        stacklevel=caller_stacklevel(),
    )
