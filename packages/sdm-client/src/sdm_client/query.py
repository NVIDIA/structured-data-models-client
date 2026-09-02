# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Asking a question in PQL, and gathering the data that answers it.

A predictive query says *what* to predict. It never says *which rows* to
predict for -- that is always an explicit argument, because a query can only
enumerate rows that are already in the tables:

>>> query = PredictiveQuery(
...     'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
...     {'users': users_df, 'orders': orders_df},
... )
>>> request = query.materialize(indices=[1, 2, 3])

:meth:`~PredictiveQuery.materialize` is the gathering step, and it is explicit
because it is the expensive one: it parses the query, finds labelled examples
in history, samples the neighbourhood of each entity, and encodes the result.
Nothing is sent -- what comes back is a :class:`~sdm_client.Request` that any
model whose profile accepts it can answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import TYPE_CHECKING, Any

import pandas as pd

from sdm_client.errors import MissingExtraError, StructuredDataError
from sdm_client.models import RELATIONAL_LIMITS
from sdm_client.request import Request
from sdm_client.schema import Schema

if TYPE_CHECKING:  # pragma: no cover - imported for typing only.
    from predictive_query import Profile

# Gathering knobs a caller may pass through to the sampler. `run_mode` is
# deliberately absent: it is a preset over these, and the client derives the
# wire value from them rather than asking a caller to know the presets.
# A resolved query reports a coarse problem type; the wire wants a task kind.
# Only the coarse classification is emitted, because a tabular model narrows it
# itself and guessing from cardinality is what the source design got wrong.
_TASK_FOR_PROBLEM = {
    'ProblemType.CLASSIFY': 'classification',
    'CLASSIFY': 'classification',
    'ProblemType.FORECAST': 'forecasting',
    'FORECAST': 'forecasting',
    'ProblemType.RANK': 'ranking',
    'RANK': 'ranking',
}

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


def _engine() -> Any:
    r"""The relational engine's ``rfm`` module, or the install that supplies it."""
    try:
        from sdm_relational_engine import rfm
    except ImportError as error:  # pragma: no cover - exercised by extras
        raise MissingExtraError(
            'relational', 'sdm-relational-engine'
        ) from error
    return rfm


class PredictiveQuery:
    r"""A question in PQL, bound to the tables that can answer it.

    Constructing one binds the text to a schema; the query is resolved against
    that schema when it is materialized, which is also where a query naming a
    column that does not exist is refused.
    """

    def __init__(
        self,
        query: str,
        data: Schema | Mapping[str, pd.DataFrame],
        *,
        profile: Profile | None = None,
    ) -> None:
        r"""Bind ``query`` to ``data``.

        Args:
            query: The predictive query.
            data: A :class:`~sdm_client.Schema`, or the frames to infer one
                from. Passing frames directly is the common case: a caller
                asking one question should not have to meet ``Schema`` first.
            profile: What the query is for. Defaults to the relational
                profile, which accepts the most. A capability may only reject,
                so a narrower profile refuses more queries, never fewer.

        Raises:
            StructuredDataError: If the query does not parse or does not
                validate against the schema.
            MissingExtraError: If the relational engine is not installed.
        """
        if not isinstance(query, str) or not query.strip():
            raise StructuredDataError(
                'query must be a non-empty string', code='INVALID_REQUEST'
            )
        self.text = query
        self.schema = (
            data if isinstance(data, Schema) else Schema.from_data(dict(data))
        )
        self._profile = profile
        self._model: Any = None

    def materialize(
        self,
        *,
        indices: Sequence[Any] | None = None,
        predict: pd.DataFrame | None = None,
        num_context_samples: int | None = None,
        **options: Any,
    ) -> Request:
        r"""Gather the data that answers this query.

        Args:
            indices: The entity keys to predict for. These are rows that exist
                in the tables.
            predict: Rows to score that are **not** in the tables. This is the
                single-table path: a query enumerates rows the tables already
                hold, so rows it has never seen have to be supplied rather
                than derived. Mutually exclusive with ``indices``.
            num_context_samples: How many labelled examples to gather, at
                least 1. The client derives the endpoint's ``run_mode`` from it
                rather than asking you to know the presets, and narrows the
                gathered examples to exactly this many. Omitted, you get the
                smaller budget of 1,000, which is the endpoint's own default.
            **options: Sampler options, e.g. ``num_neighbors``, ``anchor_time``.

        Returns:
            A :class:`~sdm_client.Request` carrying the gathered data.

        Raises:
            StructuredDataError: If neither ``indices`` nor ``predict`` is
                given, if both are, or if an option is not one this accepts.
        """
        # Before the branch, so a bad value is refused the same way whichever
        # path it was heading for.
        num_context_samples = _valid_context_samples(num_context_samples)

        if predict is not None and indices is not None:
            raise StructuredDataError(
                'pass either indices=[...] to score rows the tables already '
                'contain, or predict=<frame> to score rows they do not -- not '
                'both',
                code='INVALID_REQUEST',
            )
        if predict is not None:
            if num_context_samples is not None:
                raise StructuredDataError(
                    'num_context_samples has nothing to narrow when the rows '
                    'to score are supplied: the context is the table you gave, '
                    'in full. Pass a smaller table instead',
                    code='INVALID_REQUEST',
                )
            if options:
                raise StructuredDataError(
                    f'{sorted(options)} cannot apply when the rows to score '
                    f'are supplied: nothing is gathered, so there is nothing '
                    f'to sample',
                    code='INVALID_REQUEST',
                )
            return self._materialize_rows(predict)
        if indices is None:
            raise StructuredDataError(
                'materialize() needs the rows to score: pass indices=[...] '
                'naming the entities. There is no all-rows default, because '
                'it would silently score an entire table',
                code='INVALID_REQUEST',
            )
        if not isinstance(indices, Sequence) or isinstance(
            indices, (str, bytes)
        ):
            raise StructuredDataError(
                f'indices must be a sequence of entity keys, got '
                f'{type(indices).__name__}',
                code='INVALID_REQUEST',
            )
        if not indices:
            raise StructuredDataError(
                'indices is empty, so there is nothing to score',
                code='INVALID_REQUEST',
            )

        unknown = sorted(set(options) - _MATERIALIZE_OPTIONS)
        if unknown:
            raise StructuredDataError(
                f'unknown materialize option(s) {unknown}; accepted: '
                f'{sorted(_MATERIALIZE_OPTIONS)}',
                code='INVALID_REQUEST',
            )
        if 'run_mode' in options:  # pragma: no cover - caught by the above
            raise StructuredDataError(
                'run_mode is not a knob here; set num_context_samples and the '
                'client derives what the endpoint needs',
                code='INVALID_REQUEST',
            )

        payloads = self._gather(indices, num_context_samples, options)
        return Request.materialized(
            payloads=payloads,
            schema=self.schema,
            query=self.text,
            indices=list(indices),
            requested_inference={
                'inference_config': options.get('inference_config') or {}
            },
        )

    def _materialize_rows(self, predict: pd.DataFrame) -> Request:
        r"""Build a single-table request for rows the tables do not contain.

        This is the answer to the question the source design left open. A
        predictive query resolves against a schema, so it can only ever name
        rows that are already there; scoring a new row means supplying it. The
        query still earns its place -- it says which column is the target and
        what kind of question this is -- so the two halves come from different
        places on purpose.

        The task kind is deliberately **not** narrowed here. A tabular model
        accepts only the coarse ``classification`` and decides binary against
        multiclass itself, and narrowing from the target's cardinality is the
        guess that turns a two-class target into a multiclass one.
        """
        if len(self.schema) != 1:
            raise StructuredDataError(
                f'scoring supplied rows is the single-table path, and this '
                f'schema has {len(self.schema)} tables. Use indices=[...] to '
                f'score entities the tables already contain',
                code='INVALID_REQUEST',
            )

        resolved = self._resolve()
        table_name = next(iter(self.schema))
        # The language names a column by its table; the wire names it alone.
        target = resolved.target_column
        if isinstance(target, str) and target.startswith(f'{table_name}.'):
            target = target[len(table_name) + 1 :]
        if target is None or target not in self.schema[table_name]:
            raise StructuredDataError(
                f'{self.text!r} does not predict a column of '
                f'{table_name!r}, so there is nothing to score rows against',
                code='INVALID_REQUEST',
            )

        return Request(
            context=self.schema[table_name].data,
            predict=predict,
            target=target,
            task=_TASK_FOR_PROBLEM.get(
                str(resolved.problem_type), 'classification'
            ),
            schema=self.schema,
            entity_table=table_name,
        )

    def _resolve(self) -> Any:
        r"""Parse and validate this query, under this query's profile."""
        from predictive_query import RELATIONAL, validate

        from sdm_client.schema import to_graph_definition

        profile = self._profile or RELATIONAL
        resolved, response = validate(
            self.text, to_graph_definition(self.schema), profile
        )
        if resolved is None or not response.ok:
            reasons = (
                '; '.join(error.message for error in response.errors)
                or 'the query did not resolve'
            )
            raise StructuredDataError(
                f'{self.text!r} is not a query {profile.capabilities.model} '
                f'can answer: {reasons}',
                code='INVALID_REQUEST',
            )
        return resolved

    def _gather(
        self,
        indices: Sequence[Any],
        num_context_samples: int | None,
        options: dict[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        r"""Parse, sample and encode, using the engine's own materializer.

        The engine already splits building a request from sending one, and
        each part it returns is a wire payload. Reimplementing that here would
        duplicate PQL resolution, point-in-time correctness and neighbourhood
        sampling that already live -- and are tested -- there.
        """
        engine = _engine()
        run_mode = _run_mode_for(num_context_samples)

        model = self._engine_model(engine)
        # Resolve here rather than letting the engine default to its own
        # profile: the profile is a public argument, so both paths have to
        # honour it or it means different things depending on which was used.
        query_def = self._resolve()
        task = model._get_task_table(
            query=query_def,
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
        task._query = query_def.to_string()
        task = _narrow_context(task, num_context_samples)

        parts = model.materialize_task(
            task,
            run_mode=run_mode,
            top_k=query_def.top_k,
            exclude_cols_dict=query_def.get_exclude_cols_dict(),
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
            _check_limits(part, RELATIONAL_LIMITS)
        _check_shape(self.schema, options, RELATIONAL_LIMITS)
        return tuple(part.payload for part in parts)

    def _engine_model(self, engine: Any) -> Any:
        if self._model is None:
            graph = _to_engine_graph(engine, self.schema)
            self._model = engine.KumoRelational(graph, verbose=False)
        return self._model

    def __repr__(self) -> str:
        return f'PredictiveQuery({self.text!r}, {self.schema!r})'


def _valid_context_samples(value: Any) -> int | None:
    r"""Reject a count that is not a count.

    This number reaches a frame slice, and pandas is happy to take a negative
    one -- ``narrow_context(0, -1)`` quietly drops the last example rather than
    failing. Zero is worse: it gathers nothing and the model is asked to
    predict from an empty context. A bool is an int in Python, and
    ``num_context_samples=True`` meaning one example helps nobody.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise StructuredDataError(
            f'num_context_samples must be a whole number of labelled '
            f'examples, got {type(value).__name__} {value!r}',
            code='INVALID_REQUEST',
        )
    if value < 1:
        raise StructuredDataError(
            f'num_context_samples must be at least 1; a model predicts from '
            f'labelled examples, and {value} leaves it none',
            code='INVALID_REQUEST',
        )
    ceiling = RELATIONAL_LIMITS['context_rows_max']
    if value > ceiling:
        raise StructuredDataError(
            f'num_context_samples is {value:,}, and the endpoint accepts at '
            f'most {ceiling:,} context rows',
            code='UNSUPPORTED_REQUEST',
        )
    return int(value)


def _narrow_context(task: Any, num_context_samples: int | None) -> Any:
    r"""Gather exactly as many labelled examples as were asked for.

    The endpoint enforces a ceiling, and the engine trims to that ceiling, but
    a ceiling is not a request: asking for 500 and being handed 4,000 makes the
    number decorative, which is the whole complaint that retired ``run_mode``.
    So the count is applied here, where the examples exist and before anything
    is encoded.

    Asking for more than the data holds is not an error -- there simply are not
    that many labelled examples -- so this only ever narrows.

    Which examples survive is the engine's existing rule, inherited rather than
    chosen: the first ``n``. For most tasks the candidates are shuffled before
    this point, so that is a sample. For forecasting the engine orders context
    chronologically first, so narrowing keeps the **oldest** ``n``. That is what
    the endpoint does when it trims to its own ceiling, and matching it means a
    narrowed request and a capped one behave the same way.
    """
    if num_context_samples is None:
        return task
    gathered = task.num_context_examples
    if gathered <= num_context_samples:
        return task
    return task.narrow_context(0, num_context_samples)


def _run_mode_for(num_context_samples: int | None) -> str:
    r"""The wire value that admits ``num_context_samples`` labelled examples.

    The endpoint distinguishes only ``fast`` from everything else -- 1,000
    context rows against 10,000 -- so this is a two-way choice dressed as a
    four-value enum. Deriving it means a caller sets the number they want
    rather than learning which preset contains it.
    """
    if num_context_samples is None:
        return 'fast'
    if num_context_samples <= RELATIONAL_LIMITS['context_rows_fast']:
        return 'fast'
    if num_context_samples <= RELATIONAL_LIMITS['context_rows_max']:
        return 'best'
    # Unreachable through `materialize`, which validates first. Kept because
    # this function is callable on its own, and a guard that only holds when
    # its caller happens to check first is not a guard.
    raise StructuredDataError(
        f'num_context_samples is {num_context_samples:,}, and the endpoint '
        f'accepts at most {RELATIONAL_LIMITS["context_rows_max"]:,} context '
        f'rows',
        code='UNSUPPORTED_REQUEST',
    )


def _check_limits(part: Any, limits: Mapping[str, int]) -> None:
    r"""Refuse a gathered part the endpoint would reject, before sending it."""
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
    schema: Schema, options: Mapping[str, Any], limits: Mapping[str, int]
) -> None:
    r"""Refuse a graph wider or deeper than the endpoint will read."""
    if len(schema) > limits['tables_max']:
        raise StructuredDataError(
            f'this schema has {len(schema)} tables and the endpoint reads at '
            f'most {limits["tables_max"]}',
            code='UNSUPPORTED_REQUEST',
        )
    neighbors = options.get('num_neighbors')
    hops = len(neighbors) if neighbors else options.get('num_hops')
    if hops is not None and hops > limits['hops_max']:
        raise StructuredDataError(
            f'this request samples {hops} hops and the endpoint reads at most '
            f'{limits["hops_max"]}',
            code='UNSUPPORTED_REQUEST',
        )


def _to_engine_graph(engine: Any, schema: Schema) -> Any:
    r"""Build the engine's graph from this client's schema.

    The engine infers a graph from frames in the same way, so the frames are
    handed over and the schema's corrections are applied on top. Those
    corrections are the whole reason a caller edited the schema, and inferring
    twice would discard them.
    """
    frames = {name: table.data for name, table in schema.tables.items()}
    graph = engine.Graph.from_data(frames, verbose=False)

    for name, table in schema.tables.items():
        engine_table = graph[name]
        if table.primary_key is not None:
            engine_table.primary_key = table.primary_key
        if table.time_column is not None:
            engine_table.time_column = table.time_column
        for column_name, column in table.columns.items():
            if column_name in engine_table:
                engine_table[column_name].stype = column.stype

    # The engine infers edges from the same frames, so most of these are
    # already there. Linking one twice is an error, and the schema is the
    # authority only for the ones inference missed.
    existing = {
        (edge.src_table, edge.fkey, edge.dst_table) for edge in graph.edges
    }
    for link in schema.relationships:
        edge = (link.source_table, link.source_column, link.target_table)
        if edge not in existing:
            graph.link(
                src_table=link.source_table,
                fkey=link.source_column,
                dst_table=link.target_table,
            )
    return graph
