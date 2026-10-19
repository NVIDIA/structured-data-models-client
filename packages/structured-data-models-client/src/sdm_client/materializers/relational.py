# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Bridge code: the core route's branch of materialize, deleted in Phase 5
# with the old engine; the old route's rules here, which the core route no
# longer runs (T1.28a moved them into the core: design, "Host-held
# semantics"), go with it.

r"""Gather and encode relational requests with the relational engine."""

from __future__ import annotations

import copy
import math
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
from sdm_client.core.warnings import SDMWarning, caller_stacklevel
from sdm_client.errors import MissingExtraError, StructuredDataError
from sdm_client.materializers.sources import EngineTables
from sdm_client.materializers.tabular import _holds_two_whole_numbers
from sdm_client.models import (
    KUMO_RELATIONAL,
    PREDICTION_STATISTICS,
    RELATIONAL_LIMITS,
    SUMMARISED_TASK_KINDS,
)
from sdm_client.request import Request
from sdm_client.schema import (
    Relationship,
    Schema,
    SourceTable,
    Table,
    _and_list,
    _as_key_tuple,
)
from sdm_client.wire.dtypes import arrow_decimal_type

if TYPE_CHECKING:
    from predictive_query import ValidatedPredictiveQuery
    from sdm_relational_engine.rfm import (
        Graph,
        KumoRelational,
        MaterializedPredictionRequest,
        TaskTable,
    )


def _warn_of_a_context_the_default_mode_capped(
    task: TaskTable, num_context_samples: int | None
) -> None:
    r"""Say when the default gathered as many examples as it may, not as many
    as there are.

    How much of the data the model sees is a choice, and the tabular route
    already says so when it makes one. Here the choice is made by the run mode
    a missing ``num_context_samples`` selects, which is quieter but no less a
    choice.
    """
    if num_context_samples is not None:
        return
    cap = RELATIONAL_LIMITS['context_rows_fast']
    if getattr(task, 'num_context_examples', 0) < cap:
        return
    warnings.warn(
        f'{cap:,} context examples were gathered, which is all this route '
        f'takes by default; pass num_context_samples to ask for up to '
        f'{RELATIONAL_LIMITS["context_rows_max"]:,}.',
        category=SDMWarning,
        stacklevel=caller_stacklevel(),
    )


#: More entities than this in one `materialize()` and the request is gathered
#: in pieces that no longer fit in memory together. Every batch is built
#: before any is sent, each carrying its own context.
MOST_ENTITIES = 100_000

#: And more calls than this, which is the other way to hold too much: a
#: context is gathered for every call, so a `rows_per_call` of 1 would hold
#: one per entity. Loose enough that lowering the batch size for a wide graph
#: is unaffected, and only a caller who set it very low meets it.
MOST_CALLS = 100


def _rows_per_call(
    kind: object, limits: Mapping[str, int] = RELATIONAL_LIMITS
) -> int:
    r"""How many entities the endpoint scores in one call of a task ``kind``."""
    if str(kind) == 'temporal_link_prediction':
        return limits['predict_rows_temporal_link_prediction']
    return limits['predict_rows_max']


def rows_one_call_scores(
    asked: Any = None,
    kind: object = None,
    limits: Mapping[str, int] = RELATIONAL_LIMITS,
) -> int:
    r"""How many entities one call will score, as the caller asked or as the
    endpoint allows.

    ``rows_per_call`` only ever goes down. The endpoint refuses a call past
    its own cap, so a larger number would buy a 422 rather than a bigger call;
    a smaller one is how a graph too wide for a full batch, or a deployment
    holding to less than this client knows of, is asked in pieces that fit.
    """
    cap = _rows_per_call(kind, limits)
    if asked is None:
        return cap
    return min(int(asked), cap)


def rows_per_call_asked_for(options: Mapping[str, Any]) -> int | None:
    r"""The batch size the caller chose, refused if it is not a size.

    ``None`` is not choosing one. Every gathering option travels as a keyword
    and a caller passing one through reads an unset value as absent, so a
    question asked in parts, which drops them before passing them on, and one
    asked whole have to agree about that.
    """
    given = options.get('rows_per_call')
    if given is None:
        return None
    if (
        not isinstance(given, Integral)
        or isinstance(given, bool)
        or int(given) < 1
    ):
        raise StructuredDataError(
            f'rows_per_call is {given!r}, and a call scores a whole number of '
            f'rows, at least one',
            code='INVALID_REQUEST',
        )
    return int(given)


def _the_largest_batch(task: TaskTable, per_call: int) -> int:
    r"""How many rows the largest batch of this request actually carries.

    Not the size asked for: a request for one entity is one call of one row
    whatever the batch size allows, and telling its caller to ask for smaller
    calls names a remedy that has nothing left to divide.
    """
    asked = getattr(task, 'num_prediction_examples', None)
    if not isinstance(asked, Integral) or int(asked) < 1:
        return per_call
    return min(per_call, int(asked))


def refuse_more_entities_than_fit_in_memory(asked: int, held: int = 1) -> None:
    r"""Refuse a population too large to hold, from the count alone.

    Checked before the engine is reached, because reaching it means building
    the graph and computing a label for every index, which for a population
    far past this ceiling is minutes of work and the memory that goes with it
    before anything says no.

    ``held`` is how many times over the population is gathered: one for an
    ordinary question, and one per part for a question asked in parts, each
    of which gathers the whole population with a context of its own.
    """
    total = asked * held
    if total <= MOST_ENTITIES:
        return
    gathered = (
        f'{asked:,} entities'
        if held == 1
        else f'{asked:,} entities gathered {held:,} times over, {total:,} in all,'
    )
    raise StructuredDataError(
        f'{gathered} is past the {MOST_ENTITIES:,} one materialize() will '
        f'hold at once, since every batch is built, with its own context, '
        f'before any is sent; ask about fewer entities at a time',
        code='UNSUPPORTED_REQUEST',
        details={'entities': asked, 'gathered_times': held, 'total': total},
    )


def _refuse_more_entities_than_fit_in_memory(
    task: TaskTable, per_call: int
) -> None:
    r"""Refuse a population too large to hold as gathered batches.

    The engine builds every batch before the first is sent, so the cost of a
    population is paid in memory up front. This is the ceiling that the old
    per-call entity check used to supply as a side effect of refusing to batch
    at all.
    """
    asked = getattr(task, 'num_prediction_examples', None)
    if not isinstance(asked, Integral) or per_call <= 0:
        return
    asked = int(asked)
    calls = math.ceil(asked / per_call)
    met = [
        said
        for said, over in (
            (f'{MOST_ENTITIES:,} entities', asked > MOST_ENTITIES),
            (f'{MOST_CALLS:,} calls', calls > MOST_CALLS),
        )
        if over
    ]
    if not met:
        return
    raise StructuredDataError(
        f'{asked:,} entities are gathered as {calls:,} calls of '
        f'{per_call:,}, and every one is built, with its own context, before '
        f'any is sent. That is past the {" and ".join(met)} this will hold at '
        f'once; ask about fewer entities at a time',
        code='UNSUPPORTED_REQUEST',
        details={
            'entities': asked,
            'calls': calls,
            'rows_per_call': per_call,
        },
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
            'prediction_statistic',
            'rows_per_call',
        }
    )

    def __init__(self, schema: Schema) -> None:
        self.schema = schema
        self._model: KumoRelational | None = None
        # On the core route (T1.26b, E52): the sampler state the core's last
        # run left, which its next run is handed back.
        self._core_sampler_state: dict[str, Any] = {}

    _RESERVED = ('instance_id', '__kumo_anchor_time')

    def materialize(
        self,
        *,
        query: ValidatedPredictiveQuery,
        indices: Sequence[Any],
        num_context_samples: int | None,
        options: dict[str, Any],
        candidates: Sequence[Any] | None = None,
    ) -> Request:
        r"""Gather the selected entities and return their encoded request.

        ``candidates``, a ranking's, are the core's to ask after (T2.8), on
        the core route only; the old route asks a ranking in parts before it
        comes here.
        """
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
        # Over a DuckDB, SQLite or Postgres source on the core route (T1.11c, T2.2, T2.3),
        # the core gathers and lays out the request, the rules around the
        # gathering the core's (T1.28a); the old route holds its own.
        from sdm_client import _routing

        core = _routing.on_the_core(self.schema.source, _routing.MATERIALIZE)
        if core:
            from sdm_client._rust import materialize as on_the_core

            payloads, core_state = on_the_core.gather(
                self, query, indices, num_context_samples, options, candidates
            )
            windowed, entity_table = on_the_core.asked(query, core_state)
        else:
            windowed = (
                getattr(query.target_ast, 'date_offset_range', None) is not None
            )
            entity_table = table_name(query.entity_column)
            if len(indices) == 0:
                raise StructuredDataError(
                    'indices names no entity, and gathering a relational '
                    'request means sampling a neighbourhood around each one, '
                    'so there is nothing here to gather. An empty batch is '
                    'still answered: build it as Request(predict=<no rows>) '
                    'and predict() returns the empty result without sending',
                    code='INVALID_REQUEST',
                )

            statistic = _statistic_asked_for(options)
            rows_per_call_asked_for(options)
            self._refuse_a_name_the_wire_reserves()
            self._warn_two_whole_numbers_are_being_asked_as_quantities(query)
            _warn_the_filter_chose_only_the_examples(query, indices)
            _warn_of_relationships_that_join_nothing(self.schema)
            payloads, core_state = self._gather(
                query, indices, num_context_samples, options
            )
            if statistic is not None:
                _record_the_statistic(payloads, statistic)
            sent = _tables_sent(payloads, self.schema)
            _warn_of_tables_left_out(self.schema, sent, entity_table)
            _warn_of_columns_the_model_does_not_read(
                self.schema, _columns_to_leave_out(query, options), sent
            )
            _drop_context_that_answers_the_question(payloads, windowed=windowed)
        request = Request.materialized(
            payloads=payloads,
            windowed=windowed,
            schema=self.schema.snapshot(),
            query=query.to_string(),
            indices=list(indices),
            entity_table=entity_table,
            # A copy: the payloads were gathered from the config as it is
            # now, and a caller's later edit to it must not change what the
            # adapter believes was asked for.
            requested_inference={
                'inference_config': copy.deepcopy(
                    options.get('inference_config') or {}
                )
            },
        )
        if core_state is not None:
            # Gathered on the core (T1.11d): the request state's version,
            # which predict on the core hands back with the payload.
            request._core_state = core_state
        return request

    def _gather(
        self,
        query: ValidatedPredictiveQuery,
        indices: Sequence[Any],
        num_context_samples: int | None,
        options: dict[str, Any],
    ) -> tuple[tuple[dict[str, Any], ...], dict[str, Any] | None]:
        r"""Sample and encode using the engine's own materializer: the
        payloads, and no request state (the core route's, from T1.11d, is
        its own, :func:`sdm_client._rust.materialize.materialize`).

        More entities than one call scores are gathered as several payloads
        of at most that many, which `predict()` sends one at a time and
        answers as one result. One call scores what the endpoint takes, or
        the smaller ``rows_per_call`` the caller asked for.
        """
        run_mode = self._run_mode_for(num_context_samples)
        refuse_more_entities_than_fit_in_memory(len(indices))
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
                'entities it scores per call cannot be known',
                code='INTERNAL_ERROR',
            )
        _warn_of_a_context_the_default_mode_capped(task, num_context_samples)
        task = self._narrow_context(task, num_context_samples)

        per_call = rows_one_call_scores(
            rows_per_call_asked_for(options), task.task_type
        )
        _refuse_more_entities_than_fit_in_memory(task, per_call)
        largest = _the_largest_batch(task, per_call)
        with (
            self._engine_errors(model, largest),
            model.batch_mode(batch_size=per_call),
        ):
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
        return tuple(part.payload for part in parts), None

    def _get_model(self) -> KumoRelational:
        if self._model is None:
            try:
                from sdm_relational_engine.rfm import KumoRelational
            except ImportError as error:  # pragma: no cover - optional extra
                raise MissingExtraError(
                    'relational', 'structured-data-models-relational-engine'
                ) from error

            self._model = KumoRelational(self._build_graph(), verbose=False)
        return self._model

    def _refuse_a_name_the_wire_reserves(self) -> None:
        """Refuse a column whose name the request itself needs.

        The wire adds its own columns to every table it sends, and a caller
        column of the same name collided with them inside pandas, which said
        `cannot insert instance_id, already exists` and named no table, no
        remedy, and nothing about the name being spoken for.

        The suggestion is made per column, and by suffixing rather than by
        substituting a word only one of the two names contains. That left
        `__kumo_anchor_time` told to rename itself to `__kumo_anchor_time`.
        """
        taken = sorted(
            (name, column)
            for name, table in self.schema.tables.items()
            for column in table.columns
            if column in self._RESERVED
        )
        if not taken:
            return
        named = _and_list([f'{name}.{column}' for name, column in taken])
        if len(taken) == 1:
            column = taken[0][1]
            renamed = f'to {self._unreserved(column)!r}'
        else:
            renamed = ', '.join(
                f'{column!r} to {self._unreserved(column)!r}'
                for _, column in taken
            )
        many = len(taken) > 1
        raise StructuredDataError(
            f'{named} {"use names" if many else "uses a name"} this client '
            f'needs for the request it builds, so '
            f'{"they cannot also be columns" if many else "it cannot also be a column"} '
            f'of the data. Rename {"them" if many else "it"}, for instance '
            f'{renamed}.',
            code='INVALID_SCHEMA',
        )

    def _unreserved(self, column: str) -> str:
        """A name like ``column`` that this client does not need for itself."""
        suggestion = f'{column.lstrip("_")}_from_the_data'
        return suggestion if suggestion not in self._RESERVED else column + '_'

    def _warn_two_whole_numbers_are_being_asked_as_quantities(
        self, query: ValidatedPredictiveQuery
    ) -> None:
        """Say when a 0/1 column is being asked as a regression.

        The other route recognises two whole numbers as a label and says so.
        Here the task follows the column's stype, so the same question over
        the same column is asked as a regression and nothing is said. Also,
        `task=` is refused on this route, so the warning's own remedy does not
        apply either.
        """
        named = str(getattr(query, 'target_column', '') or '')
        table, _, column = named.partition('.')
        if table not in self.schema or column not in self.schema[table]:
            return
        if self.schema[table][column].stype != 'numerical':
            return
        # A table in its source has no rows here to look at; the check reads
        # the whole column, which against a warehouse is a scan per request.
        frame = getattr(self.schema[table], 'data', None)
        if frame is None or not _holds_two_whole_numbers(frame[column]):
            return
        warnings.warn(
            f'{named} holds two whole numbers, which is how a binary label is '
            f'usually written, and this is being asked as a regression '
            f'because that is what the column says it holds. If they are '
            f'labels, say so on the column: '
            f"schema[{table!r}][{column!r}].stype = 'categorical'.",
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )

    def _build_graph(self) -> Graph:
        r"""Build exactly the graph described by the client schema.

        The engine is not asked to guess. Its inference reads column *names*
        without checking what the column holds, so a child's foreign key called
        `ID` holding `[1, 1, 2, 2, 3]` became that table's key and linking it
        was then refused for being one, and a string column called `release`
        became a time column whose values it could not parse. Both are guesses
        about tables the schema already describes, and the second could fail
        before the caller had any chance to undo it.

        Every key, time column and semantic type is therefore written here from
        the schema, which is the only thing that decides them. A table the
        caller gave no key or no time column is given none.
        """
        from sdm_relational_engine.rfm import Graph

        self.schema.check_consistent()

        if self.schema.source is not None:
            # Read in place: the engine picks the SQL sampler for this source,
            # and only the columns the schema describes are read, so a column
            # it left out as unsendable is never sampled either.
            if self.schema._engine_tables is None:
                self.schema._engine_tables = EngineTables()
            graph = self.schema._engine_tables.graph(
                self.schema.source,
                {
                    name: (table.source_name, list(table.columns))
                    for name, table in self.schema.tables.items()
                    if isinstance(table, SourceTable)
                },
            )
        else:
            frames = {
                name: _as_typed(table)
                for name, table in self.schema.tables.items()
            }
            graph = Graph.from_data(
                frames, edges=[], infer_metadata=False, verbose=False
            )

        for name, table in self.schema.tables.items():
            engine_table = graph[name]
            if table.primary_key is not None:
                engine_table.primary_key = table.primary_key
            else:
                engine_table.primary_key = None
            if table.time_column is not None:
                engine_table.time_column = table.time_column
            else:
                engine_table.time_column = None
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
        self,
        model: KumoRelational | None = None,
        per_call: int | None = None,
    ) -> Iterator[None]:
        r"""Re-raise engine failures through this client's error surface.

        ``model`` is optional because the phase that most needs this is the one
        that builds it. `Graph.from_data`, the key and stype assignments, the
        links and `graph.validate()` all raise bare `ValueError` out of a
        package the caller never imported, and all of them used to run one line
        above this guard. A caller who wrote the `except
        StructuredDataError` the reference tells them to write did not catch
        the whole construction phase.

        What the graph dropped is appended to a complaint about the data and
        not to one about an argument. A `TypeError` is raised over the type of
        a value the caller passed, and `anchor_time='2020-10-01'` came back
        with a postscript counting rows with no time, which reads as a second
        fault and sends the caller to look at rows that were never the
        problem.
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
            detail = (
                ''
                if isinstance(error, TypeError)
                else self._sanitization_detail(model)
            )
            refused = _what_the_engine_refused(str(error))
            said = str(error)
            if refused.get('batch_too_large_to_send'):
                said = _said_in_this_client_s_terms(said, per_call)
            raise StructuredDataError(
                f'{said}{detail}',
                code='INVALID_REQUEST',
                details=refused,
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
        r"""Refuse a gathered part the endpoint would reject before sending."""
        payload = getattr(part, 'payload', None)
        if not isinstance(payload, dict):
            return
        predict = payload.get('predict', {}).get('instance_table', {})
        rows = predict.get('rows')
        if rows is None:
            return
        kind = payload.get('task', {}).get('kind')
        cap = _rows_per_call(kind, limits)
        if len(rows) > cap:
            raise StructuredDataError(
                f'this batch scores {len(rows)} rows and the endpoint accepts at '
                f'most {cap} for a {kind} task',
                code='UNSUPPORTED_REQUEST',
            )
        carried = payload.get('schema', {}).get('related_tables')
        if isinstance(carried, dict) and len(carried) > limits['tables_max']:
            raise StructuredDataError(
                f'this request carries {len(carried)} tables and the endpoint '
                f'reads at most {limits["tables_max"]}; sample fewer hops, or '
                f'keep fewer relationships in the schema',
                code='UNSUPPORTED_REQUEST',
            )

    def _check_shape(
        self, options: Mapping[str, Any], limits: Mapping[str, int]
    ) -> None:
        r"""Refuse sampling options the endpoint will not read.

        How wide the graph is, is not among them. The cap is on the sampled
        subgraph and the payload carries nothing else, so the declared table
        count is the wrong quantity: a warehouse schema of 21 tables whose
        query reaches 6 of them was refused here while the endpoint answered
        it. Reachability from the entity table is the wrong quantity too, and
        for the same reason: it is an upper bound on what the sampler
        gathers, and the 21-table graph above reaches all 21. Only the
        gathered subgraph knows, which is why the check is in
        `_check_limits`, against the payload that was actually built.
        """
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

        seed = options.get('random_seed')
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, Integral)
        ):
            raise StructuredDataError(
                f'random_seed must be a whole number, got '
                f'{type(seed).__name__} {seed!r}',
                code='INVALID_REQUEST',
            )
        if seed is not None and int(seed) < 0:
            raise StructuredDataError(
                f'random_seed is {seed}, and a seed is not negative; pass a '
                f'number from 0 upwards, or leave it out to let the sampler '
                f'choose',
                code='INVALID_REQUEST',
            )

        hops = len(neighbors) if neighbors is not None else num_hops
        if hops is not None and hops < 1:
            spelled = (
                'num_neighbors is empty'
                if neighbors is not None
                else f'num_hops is {hops}'
            )
            raise StructuredDataError(
                f'{spelled}, which asks for no hop at all, and the endpoint '
                f'reads between 1 and {limits["hops_max"]}. A request that '
                f'crosses no relationship is a single-table one: give a hop, '
                f'such as num_neighbors=[8, 8], or ask it of kumo-tabular',
                code='INVALID_REQUEST',
            )
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
    (`int64` for whole numbers and `float64` otherwise), and the tabular path
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
        elif _holds_timestamps_loosely(frame[name], column.dtype):
            cast[name] = pd.to_datetime(frame[name], utc=True)
    return frame.assign(**cast) if cast else frame


def _holds_timestamps_loosely(values: pd.Series, declared: str | None) -> bool:
    """Whether schema timestamps are still loose in pandas.

    An `object` column of `Timestamp` is typed `timestamp[us]` by the schema,
    correctly, and handed to the engine unchanged it re-infers `string` and
    then refuses mixed offsets as an internal error. Only mixed offsets fail,
    which is what `from_records` and a warehouse read of a column spanning a
    daylight-saving change both produce.
    """
    if declared is None or not declared.startswith('timestamp'):
        return False
    if not pd.api.types.is_object_dtype(values.dtype):
        return False
    return pd.api.types.infer_dtype(values, skipna=True) in {
        'datetime',
        'datetime64',
        'date',
    }


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


#: How the engine says a gathered batch is too large for the endpoint's body.
_A_BATCH_TOO_LARGE_TO_SEND = ('size exceeds the 30mb limit',)

#: How the engine offers a smaller batch, which it does only where the batch
#: that failed has more than one row in it. Read rather than worked out again
#: here, because which batch failed is the engine's to know: a request may be
#: gathered as several of different sizes, and the last of them is smaller
#: than the size the rest were built to.
_THE_ENGINE_S_ENTITIES = 'the number of entities predicted at once'

#: Where the engine's own remedy for that begins. It ends by offering the run
#: mode, which this client derives rather than takes, so a caller following it
#: lands on a refusal. Replaced rather than appended to, because advice that
#: cannot be taken is worse than none.
_THE_ENGINE_S_REMEDY = 'Please reduce either'

#: What to do about it here. Both halves are real and which one binds was
#: measured: a wide graph at the default context is gathered by asking in
#: smaller calls, and the same graph at 10,000 context rows is over the body
#: on the context alone, where only a narrower context helps.
_ASK_IN_SMALLER_CALLS = (
    'What comes down is one of three. For an ordinary question it is the '
    'call: materialize(rows_per_call=N) below the {per_call} this one holds '
    'asks for the same entities in more and smaller calls. Where the context '
    'is itself over the limit, no call is small enough, and it is '
    'num_context_samples, num_neighbors or the columns the graph carries. '
    'For a ranking or a forecast, which is one call per candidate or step '
    'and is never batched, it is the entities.'
)


#: What to say when a call already scores one row. There is no smaller call
#: to ask for, so offering one would be asking for a number that does not
#: exist.
_NO_CALL_IS_SMALLER = (
    'A call already scores one row, so there is no smaller one to ask for: '
    'what comes down is num_context_samples, num_neighbors or the columns '
    'the graph carries.'
)


_A_WINDOW_THAT_OUTRAN_THE_DATA = (
    'aggregation time range',
    'we would need data back to',
    'anchor timestamp is too early',
)


def _said_in_this_client_s_terms(said: str, per_call: int | None) -> str:
    r"""The engine's size complaint, keeping its figures and not its remedy.

    What it measured is the useful half and is kept whole. What to do about it
    is written for a caller who can set a run mode and a batch size directly,
    and through this client one of those is derived and the other is named
    differently, so that half is replaced.

    The size named is the one this question holds to, which is not always the
    10,000 of an ordinary task: a temporal link prediction holds 200, and
    advice to come down to 500 would be followed into the same failure.

    Whether a smaller call is worth offering at all is the engine's answer,
    not one worked out from the size asked for. Only the engine knows which
    batch failed, and a request gathered as `2, 2, 1` can fail on the one
    that no batch size can divide.
    """
    at = said.find(_THE_ENGINE_S_REMEDY)
    measured = said if at < 0 else said[:at].rstrip()
    if _THE_ENGINE_S_ENTITIES not in said:
        return f'{measured} {_NO_CALL_IS_SMALLER}'
    holds = f'{per_call:,}' if per_call else 'number'
    return f'{measured} ' + _ASK_IN_SMALLER_CALLS.format(per_call=holds)


def _what_the_engine_refused(said: str) -> dict[str, Any]:
    """What the engine's complaint is about, as something to branch on.

    A question asked as several parts appends an account of which part failed,
    and that account can only name a remedy where it knows the failure is one
    the remedy addresses. The engine says so in words; this records it so the
    seam reads a value rather than the words.
    """
    lowered = said.lower()
    if any(mark in lowered for mark in _A_WINDOW_THAT_OUTRAN_THE_DATA):
        return {'window_outran_the_data': True}
    if any(mark in lowered for mark in _A_BATCH_TOO_LARGE_TO_SEND):
        return {'batch_too_large_to_send': True}
    return {}


def _only_junk_is_unread(present: pd.Series) -> bool:
    """Whether what the column cannot read is junk rather than another format.

    A value the column reads past can be either. `02/01/2020` among year-first
    dates is a date written the other way, and that is the ambiguity the
    message below is for. A value that is not a date on its own terms is not,
    and saying it is ambiguous sends the caller to a remedy that cannot work.

    Asked of each value, by the one function that answers that question. It
    was asked here with the default reading, which leaves most of a day-first
    column unread, so the single value that is genuinely not a date was
    reported as ambiguity, while the same column written year-first was
    reported correctly.
    """
    read = dates.reads_as_a_date(present)
    if read.all():
        return False
    if not read.any():
        return dates.looks_unambiguous(present)
    return bool(dates.looks_unambiguous(present[read]))


def _refuse_values_that_do_not_read(present: pd.Series, where: str) -> None:
    """Refuse a column that says what its dates are and still will not read.

    The ambiguity message below is true only of a column that says two things.
    A column that says one thing and carries a value that is not a date is a
    different mistake, and telling that caller their dates are ambiguous sends
    them to a remedy that cannot work: `dayfirst=True` on a year-first column
    either raises or reads every date it can as a different one.

    One unreadable value is enough to make the whole column fail the shape
    test, so what the readable values say is asked separately.

    Coercing is offered only where something would survive it: where no value
    reads it leaves a column of NaT, which this client refuses in turn.

    What does not read is counted by the one function that answers that, or
    the count names as unread the dates the column reads perfectly well.
    """
    column = where.split('.')[-1]
    unread = present[~dates.reads_as_a_date(present)]
    example = unread.iloc[0] if len(unread) else present.iloc[0]
    if len(unread) == len(present):
        raise StructuredDataError(
            f'{where} is the time column and none of its {len(present)} '
            f'values read as a date, such as {example!r}. A table is windowed '
            f'by its time column, so this one has no time axis: give it a '
            f'column that holds dates, or leave it without one and ask a '
            f'question that does not window it',
            code='INVALID_REQUEST',
        )
    raise StructuredDataError(
        f'{where} is the time column and {len(unread)} of {len(present)} of its '
        f'values do not read as a date, such as {example!r}. Fix or drop those '
        f'values, or read the column yourself and let them become NaT: '
        f'frame[{column!r}] = pd.to_datetime(frame[{column!r}], '
        f"errors='coerce'). This client accepts it and says how many rows "
        f'the model will not see',
        code='INVALID_REQUEST',
    )


def _refuse_a_column_written_two_ways(
    shapes: dict[str, str], where: str, unread: int = 0
) -> None:
    """Refuse a column whose rows are written in more than one shape.

    Every row says exactly what it is, so it is not ambiguous. What it has no
    answer to is which single call reads it: the one that reads `13/02/2021`
    reads `2021-02-04` as the second of April.

    Said before the values that do not read, where a column has both, and
    saying so. Coercing is what the other message offers, and coercing a
    column written two ways throws away every row of the shape it does not
    settle on. The caller therefore has to know about the shapes first.
    """
    year_first = shapes['year-first']
    other = shapes['day-or-month-first']
    column = where.split('.')[-1]
    also = (
        f' {unread} of its values do not read as a date either, and are a '
        f'separate thing to fix.'
        if unread
        else ''
    )
    raise StructuredDataError(
        f'{where} is the time column and is written two ways: {year_first!r} '
        f'leads with its year and {other!r} leads with its day or its month. '
        f'No one reading covers both. The call that reads {other!r} reads '
        f'{year_first!r} as a different date, so read the halves apart and '
        f'put them back as one column, or store frame[{column!r}] in one '
        f'shape before naming it as the time column.{also}',
        code='INVALID_REQUEST',
    )


def _to_time(values: pd.Series, where: str) -> pd.Series:
    r"""A string time column read as dates, or refused for saying two things.

    `supports_dtype('timestamp', 'string')` is true, so a column of date
    strings can be named as a table's time column and is accepted. The strings
    then have to become timestamps before the model sees them, and that was
    left to `pd.to_datetime`'s defaults, which read an ambiguous slash date
    month-first. A day-first column was silently re-dated: twelve monthly
    events across a year became twelve daily events in one week of February,
    and nothing in the request, the schema or the response recorded that a
    conversion had happened at all. It surfaced, when it surfaced, as an
    unrelated complaint about anchor times.

    So a column is read only where it says what its dates are. A year-first one
    says it outright. A slash-separated one says it when a value of its own
    rules a reading out. No month is 15, so `15/03/2020` is day-first and
    every other row reads the same way. This is the ordinary case for any
    column spanning more than twelve days. Where nothing decides, `02/01/2020`
    is the 2nd of January to half the world and the 1st of February to the
    other half, and there the caller says, by converting it themselves.

    The shape check reads a sample and is only a pre-filter; the conversion
    below reads every value, so a column that changes shape in its last row is
    caught there rather than here.

    Ambiguity is considered last. A column written in no date shape, and one
    written in two, are each a different mistake with a different remedy.

    Reading each value on its own terms is the last reading tried, and only
    once the column is known to be one shape: it will also read a column
    written two ways, by re-dating the half it does not settle on.
    """
    present = values.dropna()
    if present.empty:
        return values
    reads = dates.looks_unambiguous(present)
    if reads:
        how = dates.conversion_that_works(present, only_plausible_years=False)
        if how is not None:
            return dates.read(values, how)
    shapes = dates.shapes_written(present)
    if len(shapes) > 1:
        unread = int((~dates.reads_as_a_date(present)).sum())
        return _refuse_a_column_written_two_ways(shapes, where, unread)
    if reads:
        how = dates.conversion_that_works(
            present, only_plausible_years=False, per_row=True
        )
        if how is not None:
            return dates.read(values, how)
    if not shapes or _only_junk_is_unread(present):
        return _refuse_values_that_do_not_read(present, where)
    example = next(
        (value for value in present if isinstance(value, str)), present.iloc[0]
    )
    column = where.split('.')[-1]
    raise StructuredDataError(
        f'{where} is the time column and holds text such as {example!r}, '
        f'which does not say unambiguously what date it is: read day-first it '
        f'is one date and month-first another, and nothing in the column says '
        f'which. Convert it yourself so the reading is yours: '
        f'frame[{column!r}] = pd.to_datetime(frame[{column!r}], '
        f'dayfirst=True), or store it year-first',
        code='INVALID_REQUEST',
    )


def _holds_decimal(values: pd.Series) -> bool:
    r"""Whether a column holds decimals, however it was read.

    Through Arrow the dtype says so; through pandas it is an `object` column
    and only the values do.
    """
    if arrow_decimal_type(values.dtype) is not None:
        return True
    if not pd.api.types.is_object_dtype(values.dtype):
        return False
    return pd.api.types.infer_dtype(values, skipna=True) == 'decimal'


def _to_declared(values: pd.Series, dtype: str, where: str) -> pd.Series:
    r"""``values`` as ``dtype``, exactly.

    Only what the schema asked for. A whole decimal too wide for an `int64` is
    typed `string` precisely so its digits survive, and reading "not an int" as
    "a float" would put 9223372036854775808 back as 9.223372036854776e+18,
    the corruption this exists to prevent, reintroduced one branch over.

    A column with a missing value cannot be a plain `int64`, so it becomes
    pandas' nullable `Int64` rather than widening to float and rounding an
    identifier.

    The remaining declaration is `string`, which is how a whole number wider
    than an int64 travels: only the digits matter and only a string keeps all
    of them. An integer is read with `int()` on the value and never through
    `pd.to_numeric`, which converts by way of a float64 and brings
    9007199254740993 back as 9007199254740992. These are identifiers, and the
    low bit is the whole of their job.
    """
    if dtype.startswith('float'):
        return pd.to_numeric(values.astype(object), errors='coerce').astype(
            'float64'
        )
    if not dtype.startswith('int'):
        return values.map(lambda value: None if pd.isna(value) else str(value))
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
    own, including the target. Letting a caller's list stand in for
    that would hand the model back something the query had deliberately
    removed.

    Only an absent option and an explicit `None` mean there is nothing to
    exclude. Anything else is read and refused if it is not a mapping, since
    passing over a value that cannot be one is the same silence this exists to
    remove.
    """
    merged = {
        table: list(columns)
        for table, columns in (query.get_exclude_cols_dict() or {}).items()
    }
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

    A `TOP K` written in the query is checked: at least one, and no more than
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


def _statistic_asked_for(options: Mapping[str, Any]) -> str | None:
    r"""The regression statistic the caller chose, refused if it is not one
    the endpoint reports."""
    if 'prediction_statistic' not in options:
        return None
    given = options['prediction_statistic']
    if not isinstance(given, str) or given not in PREDICTION_STATISTICS:
        raise StructuredDataError(
            f'prediction_statistic is {given!r}, and a regression reports '
            f'{" or ".join(repr(name) for name in sorted(PREDICTION_STATISTICS))} '
            f'of its predicted distribution',
            code='INVALID_REQUEST',
        )
    return str(given)


def _record_the_statistic(
    payloads: tuple[dict[str, Any], ...], statistic: str
) -> None:
    r"""Carry the chosen statistic on each call, or refuse a question that
    has no distribution to summarise."""
    for payload in payloads:
        kind = str(payload.get('task', {}).get('kind', ''))
        if kind not in SUMMARISED_TASK_KINDS:
            raise StructuredDataError(
                f'prediction_statistic applies only to a regression, and '
                f'this question is a {kind}; ask it without '
                f'prediction_statistic',
                code='INVALID_REQUEST',
            )
        output = payload.setdefault('output', {})
        output['prediction_statistic'] = statistic


def _warn_rows_the_graph_dropped(model: KumoRelational) -> None:
    r"""Say what the engine left out while building the graph.

    A fact row with no time, no key, or a key another row already used cannot
    be placed in the graph and is dropped. The count and the reason are worked
    out either way, and `_sanitization_detail` builds the sentence. It was
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
        category=SDMWarning,
        stacklevel=caller_stacklevel(),
    )


#: What the wire calls the column carrying the entity a row is about.
_ENTITY_REF = '__kumo_entity_ref'

#: And the moment it is asked about.
_ANCHOR = '__kumo_anchor_time'

#: What the envelope calls the answer a context row carries.
_TARGET = 'TARGET'

#: What ties a related row to the instance it hangs off.
INSTANCE_ID = 'instance_id'


def _drop_context_that_answers_the_question(
    payloads: tuple[dict[str, Any], ...], *, windowed: bool = True
) -> None:
    r"""Take out of the context any row that is a scored row at its own anchor.

    On a target with no time window the entity being scored is sampled into the
    context as well, at the same anchor, carrying the value being asked for,
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
    asked for. A caller reaches this by setting `context_anchor_time`, and
    which matching on the anchor alone would have let straight through.

    The two halves carry their own `instance_id` space, so removing a context
    row means removing the related rows that hang off it and nothing else.

    What is found is collected before any of it is taken, keyed by the identity
    of the context it belongs to and unioned: two payloads can share one
    context object, where counting the entries would report a row once per
    entry and keeping only the first entry's rows would leave the rest behind.

    The notice comes before the removal. A caller running with
    `simplefilter('error')` turns it into a raise, and a raise partway through
    would leave the payloads it had already reached short of rows nobody was
    told were gone. Which is why it states the rule rather than the edit:
    raised as an exception, nothing has been removed and no context has been
    sent, and a message claiming either would be false in the one case the
    ordering exists for.
    """
    found: dict[int, tuple[Mapping[str, Any], set[Any]]] = {}
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
        if not answered:
            continue
        carried = found.setdefault(id(context), (context, set()))
        carried[1].update(answered)
    if not found:
        return

    total = sum(len(answered) for _, answered in found.values())
    # The two targets lose different rows, so they are told different things.
    # A window closes, so only the same moment carries the answer; a static
    # column does not move, so every moment does.
    when = 'at the same moment' if windowed else 'at one moment or another'
    warnings.warn(
        f'{total} of the rows sampled as context are the rows being scored, '
        f'{when}, carrying the value being asked for. Rows that carry the '
        f'answer to the question being asked are not sent as context, so '
        f'these do not go into it.',
        category=SDMWarning,
        stacklevel=caller_stacklevel(),
    )
    for context, answered in found.values():
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


def _tables_sent(
    payloads: Sequence[Mapping[str, Any]], schema: Schema
) -> frozenset[str]:
    r"""Which of the schema's tables the built payloads carry.

    Read off the payload rather than worked out from the schema, because only
    the sampler knows: a table the schema links and whose foreign key matches
    no row is not in the request, and an answer derived from the schema would
    say it was. A payload that names no related tables at all is a shape this
    does not recognise, and there the schema is assumed sent, so nothing is
    reported as lost on a guess.
    """
    carried: set[str] = set()
    described = False
    for payload in payloads:
        related = (payload.get('schema') or {}).get('related_tables')
        if isinstance(related, Mapping):
            described = True
            carried.update(related)
    return frozenset(carried) if described else frozenset(schema.tables)


def _tables_linked_to(schema: Schema, entity_table: str) -> frozenset[str]:
    r"""The tables a relationship path connects to ``entity_table``.

    Read in both directions, because the sampler traverses an edge either way:
    a parent is reached from its child as readily as a child from its parent.
    Used only to say why a table is missing, never to refuse one. What is
    reachable is an upper bound on what is gathered, and refusing on it denies
    graphs the endpoint answers.
    """
    neighbours: dict[str, set[str]] = {}
    for link in schema.relationships:
        neighbours.setdefault(link.source_table, set()).add(link.target_table)
        neighbours.setdefault(link.target_table, set()).add(link.source_table)
    seen = {entity_table}
    queue = [entity_table]
    while queue:
        for nearby in neighbours.get(queue.pop(), ()):
            if nearby not in seen:
                seen.add(nearby)
                queue.append(nearby)
    return frozenset(seen)


def _warn_of_tables_left_out(
    schema: Schema, sent: frozenset[str], entity_table: str
) -> None:
    r"""Say which of the schema's tables this request does not carry.

    A table the sampler does not reach is removed from the payload, and the
    removal was silent. Measured on a parent whose value is the mean of its
    children's, leaving the one edge between them undeclared moved the error
    from 0.008 to 0.134, sixteen times worse, with nothing said and a result
    that looks entirely ordinary.

    The two causes are said separately because they take different remedies.
    Nothing links the first kind, and only a relationship puts it back. The
    second is linked and was still not reached, which is either a hop budget
    that stops short of it or a foreign key that matches no row.
    """
    missing = sorted(set(schema.tables) - sent)
    if not missing:
        return
    linked = _tables_linked_to(schema, entity_table)
    unlinked = [name for name in missing if name not in linked]
    unreached = [name for name in missing if name in linked]
    if unlinked:
        many = len(unlinked) > 1
        warnings.warn(
            f'{", ".join(repr(name) for name in unlinked)} '
            f'{"are" if many else "is"} in the schema and not in this '
            f'request: no relationship connects {"them" if many else "it"} '
            f'to {entity_table!r}, so the model answers without '
            f'{"them" if many else "it"}. Say how '
            f'{"they join" if many else "it joins"}: '
            f'schema.relationships.append(Relationship({unlinked[0]!r}, '
            f'<foreign key>, <parent table>, <its primary key>)), or leave '
            f'{"them" if many else "it"} out of the schema to say '
            f'{"they are" if many else "it is"} not wanted.',
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )
    if unreached:
        many = len(unreached) > 1
        warnings.warn(
            f'{", ".join(repr(name) for name in unreached)} '
            f'{"are" if many else "is"} linked into the schema and not in '
            f'this request: the sampler reached no row of '
            f'{"them" if many else "it"} from the entities being scored, so '
            f'the model answers without {"them" if many else "it"}. Raise '
            f'num_hops if {"they lie" if many else "it lies"} further out '
            f'than the hops sampled, and check the foreign key holds values '
            f'the parent key actually has if '
            f'{"they do" if many else "it does"} not.',
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )


def _warn_of_relationships_that_join_nothing(schema: Schema) -> None:
    r"""Say when a declared relationship matches no row on either side.

    Nothing looks at what a foreign key holds. `_require_joinable` checks the
    dtype and the shape of the parent key, and the sanitization report counts
    null and repeated keys, so an edge between two disjoint namespaces of the
    same dtype passes both. Declaring `flights.AIRLINE -> airports.IATA_CODE`
    where `flights.ORIGIN_AIRPORT` was meant is one word, and the same query
    on the same frames answered 159.7 with the right column and -0.006 with
    the wrong one, in byte-identical silence.

    Said rather than refused: a child with no rows yet is legal and joins
    nothing for a reason that is not a mistake, which is why an empty frame on
    either side is passed over rather than reported.
    """
    empty = [
        link for link in schema.relationships if _joins_nothing(schema, link)
    ]
    if not empty:
        return
    first = empty[0]
    many = len(empty) > 1
    warnings.warn(
        f'{", ".join(str(link) for link in empty)} '
        f'{"join" if many else "joins"} none of the rows given: no value of '
        f'the foreign key appears in the key it points at, so the tables are '
        f'related in the schema and in no row of the data, and the question '
        f'is answered as though {"they" if many else "it"} were not declared. '
        f'Check that the foreign key is the column naming the parent row; one '
        f'spelled differently from the key it points at has to be said '
        f'outright, as relationships=[Relationship({first.source_table!r}, '
        f'<that column>, {first.target_table!r}, {first.target_column!r})].',
        category=SDMWarning,
        stacklevel=caller_stacklevel(),
    )


def _joins_nothing(schema: Schema, link: Relationship) -> bool:
    r"""Whether no row of the child's key is a row of the parent's.

    Every shape this cannot read is not one: a link naming a table or a column
    the schema has not got is `check_consistent`'s to refuse, and answering it
    here would report a missing table as a join that found nothing.
    """
    if (
        link.source_table not in schema.tables
        or link.target_table not in schema.tables
    ):
        return False
    child = getattr(schema[link.source_table], 'data', None)
    parent = getattr(schema[link.target_table], 'data', None)
    if child is None or parent is None:
        return False
    if (
        link.source_column not in child.columns
        or link.target_column not in parent.columns
    ):
        return False
    values = child[link.source_column].dropna().unique()
    keys = parent[link.target_column].dropna().unique()
    if not len(values) or not len(keys):
        return False
    return not bool(pd.Series(values).isin(keys).any())


def _warn_of_columns_the_model_does_not_read(
    schema: Schema, left_out: Mapping[str, Sequence[str]], sent: frozenset[str]
) -> None:
    r"""Say which feature columns reach the model and are dropped there.

    The model's recipe carries no text processor, so a column typed `text` is
    ignored wherever it reaches it. Refusing the request was considered and
    rejected: a column the answer never depended on should not be a reason not
    to answer. That leaves the caller believing a column contributed when it
    did not, which is worth one line. Measured on one determining token, the
    same column scores 1.00 typed `categorical` and chance typed `text`.

    Read against the columns the request actually carries. A column the query
    or the caller already left out is not sent, so naming it would report a
    loss that did not happen, the target among them.

    `sent` is the same question asked of the table. A table nothing links, or
    one further out than the hops sampled, is not in the payload either, and
    naming its columns reported a loss that did not happen. Its remedy, to
    type it categorical, changes nothing under that hop budget. Why
    that table is missing is said by `_warn_of_tables_left_out`, which is the
    line a caller can act on.
    """
    unread = KUMO_RELATIONAL.runtime.unread_feature_stypes
    if not unread:
        return
    ignored = sorted(
        f'{table_name}.{column_name}'
        for table_name, table in schema.tables.items()
        if table_name in sent
        for column_name, column in table.columns.items()
        if column.stype in unread
        and column_name not in left_out.get(table_name, ())
    )
    if not ignored:
        return
    many = len(ignored) > 1
    warnings.warn(
        f'{", ".join(ignored)} {"are" if many else "is"} typed '
        f'{" or ".join(sorted(unread))}, which this model does not read, so '
        f'the request is answered without {"them" if many else "it"} and '
        f'{"they contribute" if many else "it contributes"} nothing to the '
        f'prediction. Type a column categorical if its values are labels '
        f'rather than prose, or leave it out to say so.',
        category=SDMWarning,
        stacklevel=caller_stacklevel(),
    )


def _warn_the_filter_chose_only_the_examples(
    query: ValidatedPredictiveQuery, indices: Sequence[Any]
) -> None:
    r"""Say that an entity filter shapes the context and not the scored rows.

    `FOR EACH sellers.seller_id WHERE sellers.seller_state = 'RJ'` reads as one
    question and is answered as two. The filter selects the labelled examples
    the model learns from; the rows actually scored are the caller's indices,
    untouched by it. Measured on synthetic data, a filter matching 12 of 50
    named sellers still scored all 50. The model was shown a thousand
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
        f'{len(indices)} rows named for scoring. Those are scored as given, '
        f'whether or not they satisfy it. Pass only indices that satisfy the '
        f'condition if that is what you meant.',
        category=SDMWarning,
        stacklevel=caller_stacklevel(),
    )
