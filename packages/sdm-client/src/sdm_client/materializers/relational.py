# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Gather and encode relational requests with the relational engine."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from numbers import Integral
from typing import TYPE_CHECKING, Any

from sdm_client.errors import MissingExtraError, StructuredDataError
from sdm_client.models import RELATIONAL_LIMITS
from sdm_client.request import Request
from sdm_client.schema import Schema, _as_key_tuple

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

        payloads = self._gather(query, indices, num_context_samples, options)
        return Request.materialized(
            payloads=payloads,
            schema=self.schema,
            query=query.to_string(),
            indices=list(indices),
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
        model = self._get_model()

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
                top_k=query.top_k,
                exclude_cols_dict=query.get_exclude_cols_dict(),
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
            name: table.data for name, table in self.schema.tables.items()
        }
        graph = Graph.from_data(frames, edges=[], verbose=False)

        for name, table in self.schema.tables.items():
            engine_table = graph[name]
            if table.primary_key is not None:
                engine_table.primary_key = table.primary_key
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
    def _engine_errors(self, model: KumoRelational) -> Iterator[None]:
        r"""Re-raise engine failures through this client's error surface."""
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
