# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Bridge code: on the core route, the num_context_samples bound in
# _validate_materialize_args (design, "Host-held semantics") is deleted in
# Phase 1 by T1.28a; the decomposition and the model and route choice move
# in a later phase (the ledger), and the S0 hook in _resolve goes in Phase 5.

r"""Asking a question in PQL, and gathering the data that answers it.

A predictive query says *what* to predict. It never says *which rows* to
predict for. That is always an explicit argument because a query can only
enumerate rows that are already in the tables:

>>> query = PredictiveQuery(
...     'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
...     {'users': users_df, 'orders': orders_df},
... )
>>> request = query.materialize(indices=[1, 2, 3])

:meth:`~PredictiveQuery.materialize` is the gathering step, and it is explicit
because it is the expensive one: it parses the query, finds labelled examples
in history, samples the neighbourhood of each entity, and encodes the result.
Nothing is sent. What comes back is a :class:`~sdm_client.Request` gathered
for one model. Which one follows the schema's shape unless ``model`` says
otherwise, and the model asked to answer it has to be that same one.
"""

from __future__ import annotations

import dataclasses
import re
import warnings
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import TYPE_CHECKING, Any, NoReturn

import pandas as pd

from sdm_client.base import PROFILE_MODELS
from sdm_client.core.warnings import SDMWarning
from sdm_client.decompose import (
    FORECAST,
    RANKING,
    Decomposition,
    about_the_part,
    parts_of,
    refuse_a_part_that_would_take_several_calls,
    said_as_parts_are,
)
from sdm_client.errors import MissingExtraError, StructuredDataError
from sdm_client.materializers import RelationalMaterializer, TabularMaterializer
from sdm_client.materializers.relational import (
    refuse_more_entities_than_fit_in_memory,
    rows_one_call_scores,
    rows_per_call_asked_for,
)
from sdm_client.models import RELATIONAL_LIMITS, Model, spec_for
from sdm_client.request import Request
from sdm_client.schema import _MAX_CATEGORICAL_CARDINALITY as _MAX_CATEGORICAL
from sdm_client.schema import Schema

#: The language's profiles a query is resolved for, by name: the core
#: validates for a name, and only the old route's validator reads the old
#: PQL package's object of it (T3.16, rule 11).
RELATIONAL = 'RELATIONAL'
TABULAR = 'TABULAR'

if TYPE_CHECKING:
    from predictive_query import ValidatedPredictiveQuery

    from sdm_client._rust.results import CoreParts, CoreQuery


def _refuse_the_task(task: str, why: str) -> NoReturn:
    r"""The refusal of ``task=`` on a call whose query settles its kind."""
    raise StructuredDataError(
        f'task={task!r} settles what a single-table question is '
        f'asking, which is the only route that has to decide. '
        f'{why.capitalize()} resolves to its own kind, so state it in '
        f'the query rather than beside it',
        code='INVALID_REQUEST',
        details={'task': repr(task)},
    )


def _alone(asked: Any) -> bool:
    r"""Whether ``asked`` is a query only the core resolves (KO2-36), read
    on the core route only, where the bridge is imported."""
    from sdm_client._rust.results import CoreQuery

    return isinstance(asked, CoreQuery)


class PredictiveQuery:
    r"""A question expressed in PQL and answered from the tables in its schema.

    The query is resolved and validated when :meth:`materialize` builds a
    request.

    Args:
        query: PQL query text.
        data: A schema or mapping of table names to DataFrames.

    Raises:
        StructuredDataError: If the query is empty or the data is invalid.
    """

    def __init__(
        self,
        query: str,
        data: Schema | Mapping[str, pd.DataFrame],
    ) -> None:
        if not isinstance(query, str) or not query.strip():
            raise StructuredDataError(
                'query must be a non-empty string', code='INVALID_REQUEST'
            )
        self.text = query
        self.schema = (
            data if isinstance(data, Schema) else Schema.from_data(dict(data))
        )
        self._tabular_materializer = TabularMaterializer(self.schema)
        self._relational_materializer = RelationalMaterializer(self.schema)

    def materialize(
        self,
        *,
        indices: Sequence[Any] | None = None,
        predict: pd.DataFrame | None = None,
        num_context_samples: int | None = None,
        model: str | Model | None = None,
        candidates: Sequence[Any] | None = None,
        task: str | None = None,
        **options: Any,
    ) -> Request:
        r"""Build a request with the data needed to answer this query.

        The request takes one of three paths: predict=<frame> uses the single
        table as context, the tabular route builds the request directly in
        pandas, and the relational route gathers the tables as a graph. Either
        single-table path samples the context down to what the endpoint
        accepts, and says so when it does.

        ``model`` chooses between the last two. Left unsaid, the schema's shape
        decides, and the shape that counts is the one
        :attr:`~sdm_client.Request.is_relational` reports: a relationship sends
        a schema relationally as surely as a second table does, so a lone table
        that points at itself is gathered as the graph it is. Name a model when
        that default is not what you want. A single table with no
        relationships is a graph of one node, and the relational model answers
        it.

        Args:
            indices: Keys of rows already present in the tables. Mutually
                exclusive with ``predict``. On the relational route, more than
                the endpoint scores in one call are gathered as several calls,
                and :meth:`~sdm_client.SDMClient.predict` answers them as one.
                A question asked in parts, a ranking or a forecast, is one
                call per part and is refused rather than batched.
            predict: Rows supplied directly for scoring. Supported for
                single-table schemas and mutually exclusive with ``indices``.
            num_context_samples: For ``indices`` requests, the maximum number
                of labelled context rows to use. Capped at what the endpoint
                accepts. When omitted, a single-table request uses every
                eligible row up to that cap, and a relational one uses up to
                1,000.
            model: The model this request is being gathered for, as a handle
                such as ``KumoRelational()`` or a model id. Defaults to the
                schema's shape.
            candidates: The rows a ranking is over. A ranking is answered as
                one question per candidate, so the set has to be named rather
                than assumed, and one call is made for each.
            task: The kind of question this is, when the target does not
                settle it. Authoritative, and single-table only: passing it
                with a relational or decomposed query is refused.
            **options: Relational sampling options, such as ``num_neighbors``
                and ``anchor_time``; ``prediction_statistic``: what a
                regression's prediction reports of its predicted distribution,
                ``'median'`` (the default) or ``'mean'``; and
                ``rows_per_call``: how many entities one call scores, which
                defaults to what the endpoint takes, 10,000 or 200 for a
                temporal link prediction, and may only be lowered. Lower it
                for a graph wide enough that a full batch outruns the 30 MiB
                body, or for a deployment holding to less than this client
                knows of.

        Returns:
            A :class:`~sdm_client.Request` carrying the data.

        Raises:
            UnknownModelError: If ``model`` is an id the contract does not
                carry.
            StructuredDataError: If neither ``indices`` nor ``predict`` is
                given, if both are, if an option is not one this accepts, or if
                ``model`` cannot answer a schema of this shape.
        """
        spec = spec_for(model) if model is not None else None
        in_place = self.schema.source is not None
        relational = (
            spec.profile == RELATIONAL
            if spec is not None
            else in_place
            or len(self.schema) > 1
            or bool(self.schema.relationships)
        )
        if in_place and predict is not None and relational:
            raise StructuredDataError(
                'predict= carries rows, and a schema over tables in their '
                'source is sampled there instead; name the entities to score '
                'with indices=',
                code='INVALID_REQUEST',
            )
        if in_place and not relational:
            raise StructuredDataError(
                'a schema over tables in their source is sampled with SQL, '
                'which only the relational model reads; the tabular model '
                'takes rows in memory. Use model=KumoRelational() with '
                'indices=, or build the schema with Schema.from_data from the '
                'rows you want',
                code='INVALID_REQUEST',
            )
        from sdm_client import _routing

        self._validate_materialize_args(
            relational=relational,
            model_id=spec.model_id if spec is not None else None,
            len_schema=len(self.schema),
            relationships=len(self.schema.relationships),
            indices=indices,
            predict=predict,
            num_context_samples=num_context_samples,
            options=options,
            # The core bounds the context size on its route (T1.28a): a
            # schema over a DuckDB, SQLite or Postgres source is relational (refused above if
            # not), and goes to the core where the seam forces it there.
            bounded_by_the_core=_routing.on_the_core(
                self.schema.source, _routing.MATERIALIZE
            ),
        )

        # On the core route a question asked in parts is the core's to split
        # (lesson 17): a forecast in its steps (T1.36 fix 21) and a ranking
        # in its candidates (T2.8), each refusal of either the core's, said
        # here as this client says it; as is a query only the core resolves
        # (KO2-36).
        on_the_core = relational and _routing.on_the_core(
            self.schema.source, _routing.MATERIALIZE
        )
        # Resolved once, and against the profile this request is for, because
        # whether it is asked as several questions is a property of the query
        # and the profile answers what the query is allowed to be.
        asked = self._resolve(RELATIONAL if relational else TABULAR)
        if on_the_core:
            # The call's own checks first, as for any relational query.
            if task is not None:
                _refuse_the_task(task, 'a relational query')
            assert indices is not None
            return self._on_the_core(
                asked,
                indices=indices,
                num_context_samples=num_context_samples,
                candidates=candidates,
                options=options,
            )
        rewritten = parts_of(
            asked,
            self.text,
            list(candidates) if candidates is not None else None,
            options,
            self.schema,
        )
        if task is not None and (relational or rewritten is not None):
            _refuse_the_task(
                task,
                'a decomposed question'
                if rewritten is not None
                else 'a relational query',
            )

        if rewritten is not None:
            return self._in_parts(
                rewritten,
                indices=indices,
                predict=predict,
                num_context_samples=num_context_samples,
                model=model,
                **options,
            )

        if predict is not None:
            return self._tabular_materializer.table(
                query=asked,
                predict=predict,
                task=task,
            )

        assert indices is not None
        if not relational:
            return self._tabular_materializer.indices(
                query=asked,
                indices=indices,
                num_context_samples=num_context_samples,
                task=task,
            )

        return self._relational_materializer.materialize(
            query=asked,
            indices=indices,
            num_context_samples=num_context_samples,
            options=options,
        )

    @staticmethod
    def _validate_materialize_args(
        *,
        relational: bool,
        model_id: str | None,
        len_schema: int,
        relationships: int,
        indices: Sequence[Any] | None,
        predict: pd.DataFrame | None,
        num_context_samples: int | None,
        options: dict[str, Any] | None,
        bounded_by_the_core: bool = False,
    ) -> None:
        r"""Refuse a call that cannot be gathered, in an order that advises.

        Which rows to score is a property of the call rather than of the route,
        and naming a model resolves neither, so both are settled before
        anything below can suggest one. The route comes next, because a model
        that cannot read this schema at all explains every refusal after it.

        Each message names a way out that works, which sometimes depends on the
        schema rather than only on the call. Gathering options are honoured by
        the relational model, which scores rows the tables already hold, so
        naming that model alone would land the caller on the next refusal and
        the message names ``indices`` too. A supplied frame can be gathered
        tabularly only where the schema has no relationship to lose, so where
        it has one the message says to drop it rather than to name a model that
        would be refused for it.
        """
        if indices is not None and predict is not None:
            raise StructuredDataError(
                'pass either indices=[...] to score rows the tables already '
                'contain, or predict=<frame> to score rows they do not, not both',
                code='INVALID_REQUEST',
            )

        if indices is None and predict is None:
            raise StructuredDataError(
                'materialize() needs the rows to score: pass indices=[...] '
                'naming the entities. There is no all-rows default, because '
                'it would silently score an entire table',
                code='INVALID_REQUEST',
            )

        if not relational and len_schema != 1:
            raise StructuredDataError(
                f'{model_id} answers from a single table, and this schema has '
                f'{len_schema}. Pass model=KumoRelational() to gather them as '
                f'a graph',
                code='INVALID_REQUEST',
            )

        if not relational and relationships:
            raise StructuredDataError(
                f'{model_id} reads no relationships, and this schema declares '
                f'{relationships}. Pass model=KumoRelational() to gather it as '
                f'a graph, or drop the relationship to ask about the table on '
                f'its own',
                code='INVALID_REQUEST',
            )

        if not relational and options:
            raise StructuredDataError(
                f'gathering options apply only to relational materialization: '
                f'{sorted(options)}. Pass indices=[...] with '
                f'model=KumoRelational() to gather this schema as a graph',
                code='INVALID_REQUEST',
            )

        if predict is not None:
            if len_schema != 1:
                raise StructuredDataError(
                    f'predict=<frame> requires a single-table schema; '
                    f'got {len_schema} tables. For a relational schema, use '
                    f'indices=[...] to select entities already in the tables',
                    code='INVALID_REQUEST',
                )
            if relational:
                instead = (
                    'drop the relationship to gather this table for the '
                    'tabular model'
                    if relationships
                    else 'gather this request for the tabular model'
                )
                raise StructuredDataError(
                    'predict=<frame> is the single-table form, and '
                    f'{model_id or "a relational gather"} scores rows the '
                    f'tables already contain. Pass indices=[...] naming them, '
                    f'or {instead}',
                    code='INVALID_REQUEST',
                )
            if num_context_samples is not None:
                raise StructuredDataError(
                    'num_context_samples cannot be used with '
                    'predict=<frame>: the schema table is the context. To '
                    'choose the context, build the schema from the rows you '
                    'want',
                    code='INVALID_REQUEST',
                )

        if indices is not None:
            if not isinstance(indices, Sequence) or isinstance(
                indices, (str, bytes)
            ):
                raise StructuredDataError(
                    f'indices must be a sequence of entity keys, got '
                    f'{type(indices).__name__}',
                    code='INVALID_REQUEST',
                )
            unusable = [value for value in indices if not _is_hashable(value)]
            if unusable:
                kinds = sorted({type(v).__name__ for v in unusable})
                raise StructuredDataError(
                    f'indices must be entity keys, and {len(unusable)} of the '
                    f'{len(indices)} given cannot be one: {kinds}. A key is '
                    f'looked up by value, so it has to be hashable. This is '
                    f'what a row of json.loads(...) or '
                    f'frame.to_dict("records") gives you; pass the key field '
                    f'of each instead, e.g. [row[{"<key>"!r}] for row in '
                    f'rows].',
                    code='INVALID_REQUEST',
                )

        if num_context_samples is not None:
            if isinstance(num_context_samples, bool) or not isinstance(
                num_context_samples, Integral
            ):
                raise StructuredDataError(
                    'num_context_samples must be a whole number of labelled '
                    f'examples, got {type(num_context_samples).__name__} '
                    f'{num_context_samples!r}',
                    code='INVALID_REQUEST',
                )
            if num_context_samples < 1 and not bounded_by_the_core:
                raise StructuredDataError(
                    'num_context_samples must be at least 1; a model predicts '
                    f'from labelled examples, and {num_context_samples} '
                    f'leaves it none',
                    code='INVALID_REQUEST',
                )

    def _in_parts(self, rewritten: Decomposition, **how: Any) -> Request:
        r"""Gather the questions this one is asked as, and keep them together.

        Each is gathered like any other query, because each needs the context
        its own labels come from: what was true in the second window is not
        what was true in the first, and whether an entity reached one
        candidate is not whether it reached another.

        A part is one call, which is what makes the fan-out countable, so a
        question over more entities than one call scores is refused here
        rather than gathered as parts that each batch.
        """
        import copy

        how = {name: value for name, value in how.items() if value is not None}
        asked = 0 if how.get('indices') is None else len(how['indices'])
        refuse_a_part_that_would_take_several_calls(
            rewritten, asked, rows_one_call_scores(rows_per_call_asked_for(how))
        )
        refuse_more_entities_than_fit_in_memory(asked, len(rewritten.parts))
        # How many candidates to keep belongs to the whole ranking. Each part
        # is one yes-or-no question, and a task that ranks nothing is refused
        # by the endpoint for carrying an option it cannot act on.
        how.pop('top_k', None)
        gathered = []
        for text, step in rewritten.parts:
            try:
                gathered.append(
                    PredictiveQuery(text, self.schema).materialize(**how)
                )
            except MissingExtraError:
                raise
            except StructuredDataError as error:
                raise about_the_part(error, rewritten, step) from error
        # A copy, because the carrier cannot be one of the things it carries:
        # asking it would ask the first part, which would ask the first part.
        whole = copy.copy(gathered[0])
        whole._parts = tuple(gathered)
        whole._decomposition = rewritten
        whole._asked = self.text
        return whole

    def _on_the_core(
        self,
        asked: ValidatedPredictiveQuery | CoreQuery,
        *,
        indices: Sequence[Any],
        num_context_samples: int | None,
        candidates: Sequence[Any] | None,
        options: dict[str, Any],
    ) -> Request:
        r"""A query gathered on the core, which asks a forecast as one
        regression per step (T1.36 fix 21) and a ranking as one binary
        question per candidate (T2.8), kept together as :meth:`_in_parts`
        keeps the parts it gathers. The core's refusals are said as this
        client says them of any query (:func:`decompose.said_as_parts_are`):
        of the step or the candidate they happened in, which the core names,
        and, for a query only the core resolves (KO2-36) that the core does
        not resolve either, with the old validator's diagnostics."""
        given = None if candidates is None else list(candidates)
        try:
            whole = self._relational_materializer.materialize(
                query=asked,
                indices=indices,
                num_context_samples=num_context_samples,
                options=options,
                candidates=given,
            )
        except MissingExtraError:
            raise
        except StructuredDataError as error:
            details = getattr(error, 'details', None) or {}
            if (
                _alone(asked)
                and asked.refusal is not None
                and error.code == 'INVALID_REQUEST'
                and details.get('rule') is None
                and details.get('field') == 'query'
            ):
                # The core does not resolve it either: the old validator's
                # diagnostics say why, a mistyped column's among them.
                raise asked.refusal from error
            said = said_as_parts_are(
                error,
                self.text,
                given,
                steps=asked.num_forecasts,
                options=options,
            )
            if said is error:
                # The core's own words: raised as the core raised them, with
                # no cause chained.
                raise
            raise said from error
        parts = None if whole._core is None else whole._core.parts
        if parts is None:
            return whole
        if parts.kind == 'ranking':
            assert given is not None
            return self._kept_as_parts(whole, parts, given, indices)
        return self._kept_as_steps(
            whole, parts, Decomposition(FORECAST, ()), indices
        )

    def _kept_as_parts(
        self,
        whole: Request,
        parts: CoreParts,
        given: list[Any],
        indices: Sequence[Any],
    ) -> Request:
        r"""A ranking the core gathered as one question per candidate, kept
        together as :meth:`_in_parts` keeps the parts it gathers: one
        request per candidate, with its question's payload and query, each
        about the candidate as the caller gave it."""
        import copy

        from sdm_client._rust.results import CoreRanking, CoreRequest

        version = whole._core.version
        gathered = []
        for position, named, payload, query in [
            (parts.position, parts.name, whole.payloads[0], parts.query),
            *(
                (part.position, part.name, part.payload, part.query)
                for part in parts.later
            ),
        ]:
            part = Request.materialized(
                payloads=(payload,),
                windowed=whole._windowed,
                schema=self.schema.snapshot(),
                query=query,
                indices=list(indices),
                entity_table=whole.entity_table,
                requested_inference=whole.requested_inference,
            )
            part._core = CoreRequest(version=version)
            gathered.append((part, query, position, named))
        carrier = copy.copy(gathered[0][0])
        carrier._parts = tuple(part for part, *_ in gathered)
        carrier._decomposition = Decomposition(
            RANKING,
            tuple(
                (query, given[position]) for _, query, position, _ in gathered
            ),
            parts.top_k,
        )
        # What predict on the core reads of the ranking: how many candidates
        # a row keeps, and each question's candidate by its position among
        # those given and by its name in the result.
        carrier._core = CoreRequest(
            version=version,
            ranking=CoreRanking(
                top_k=parts.top_k,
                positions=tuple(position for _, _, position, _ in gathered),
                values=tuple(str(named) for *_, named in gathered),
            ),
        )
        carrier._asked = self.text
        return carrier

    def _kept_as_steps(
        self,
        whole: Request,
        parts: CoreParts,
        rewritten: Decomposition,
        indices: Sequence[Any],
    ) -> Request:
        r"""A forecast the core gathered in steps, kept together as
        :meth:`_in_parts` keeps the parts it gathers: one request per step,
        with the step's payload and query."""
        import copy

        from sdm_client._rust.results import CoreRequest

        version = whole._core.version
        gathered = []
        for step, payload, query in [
            (parts.position, whole.payloads[0], parts.query),
            *(
                (part.position, part.payload, part.query)
                for part in parts.later
            ),
        ]:
            part = Request.materialized(
                payloads=(payload,),
                windowed=whole._windowed,
                schema=self.schema.snapshot(),
                query=query,
                indices=list(indices),
                entity_table=whole.entity_table,
                requested_inference=whole.requested_inference,
            )
            part._core = CoreRequest(version=version)
            gathered.append((part, query, step))
        carrier = copy.copy(gathered[0][0])
        carrier._parts = tuple(part for part, _, _ in gathered)
        carrier._decomposition = dataclasses.replace(
            rewritten,
            parts=tuple((query, step) for _, query, step in gathered),
        )
        carrier._asked = self.text
        return carrier

    def _resolve(self, profile: str) -> ValidatedPredictiveQuery | CoreQuery:
        r"""Parse and validate the query.

        On the core route (S0, the private routing seam) the core decides
        validity (P2-MID2-1, design choice 9). A query it accepts that a
        relational materialize gathers on the core is the core's to resolve
        as it gathers it: :class:`~sdm_client._rust.results.CoreQuery`,
        the text alone, so the old validator is never reached on the core
        route. Only where the old materializer gathers it is the accepted
        query resolved by the old validator, for that materializer, which
        also checks that the two agree.

        On the old route the old validator decides. Where a relational
        materialize runs on the core over a schema holding a column the
        old validator's language cannot name (a dot, a backtick or a line
        break in its name), a query the old validator refuses is the core's
        to resolve as it gathers it (KO2-36, T2.18): ``CoreQuery``, holding
        the old validator's refusal for a call only the old route answers.
        """
        from sdm_client import _routing

        core = _routing.route(_routing.S0) == 'core'
        gathered_on_the_core = profile == RELATIONAL and _routing.on_the_core(
            self.schema.source, _routing.MATERIALIZE
        )
        from sdm_client.schema import (
            names_a_query_cannot_say,
            to_schema_definition,
        )

        if core:
            from sdm_client._rust import s0
            from sdm_client._rust.spec import query_schema

            # The schema as the core reads it, written with no object of the
            # old PQL package (T3.16, rule 11).
            accepted, messages, held, resolved = s0.validate(
                self.text, query_schema(self.schema), profile
            )
            if not accepted:
                raise self._refusal(messages, profile)
            if gathered_on_the_core:
                from sdm_client._rust import errors
                from sdm_client._rust.results import CoreQuery

                # No shadow check reads it to say its day-first literal, so
                # the core's warning is shown.
                errors.warn_held(held)
                return CoreQuery(
                    text=str(
                        (None if resolved is None else resolved.text)
                        or self.text
                    ),
                    num_forecasts=(
                        None if resolved is None else resolved.num_forecasts
                    ),
                    refusal=None,
                )
            with warnings.catch_warnings():
                # Said once, as the core's query schema was written.
                warnings.simplefilter('ignore', SDMWarning)
                definition = to_schema_definition(self.schema)
            return s0.shadow(self.text, definition, profile)
        from predictive_query import validate

        from sdm_client.base import language_profile

        definition = to_schema_definition(self.schema)

        old, response = validate(
            self.text, definition, language_profile(profile)
        )
        if old is not None and response.ok:
            return old
        reasons = [error.message for error in response.errors]
        if gathered_on_the_core and names_a_query_cannot_say(self.schema):
            from sdm_client._rust.results import CoreQuery

            return CoreQuery(
                text=self.text,
                num_forecasts=None,
                refusal=self._refusal(reasons, profile),
            )
        raise self._refusal(reasons, profile)

    def _refusal(
        self, messages: list[str], profile: str
    ) -> StructuredDataError:
        r"""The refusal of a query the language does not accept.

        The hints below recognise the refusal by the old validator's wording.
        On the core route the messages are the core's diagnostics, whose
        wording is not promised, so a hint may not appear there; S0 parity is
        the validity and the code, never the text.
        """
        reasons = (
            '; '.join(
                _without_the_validator_trace(message) for message in messages
            )
            or 'the query did not resolve'
        )
        return StructuredDataError(
            f'{self.text!r} is not a query {PROFILE_MODELS[profile]} '
            f'can answer: {_without_a_remedy_this_caller_cannot_run(reasons)}'
            f'{self._why_the_target_is_text(reasons, profile)}'
            f'{self._how_to_resolve_an_ambiguous_link(reasons)}',
            code='INVALID_REQUEST',
        )

    def _how_to_resolve_an_ambiguous_link(self, reasons: str) -> str:
        r"""Say how to resolve an ambiguous link, in the words of this object.

        The language advises in its own vocabulary: `schema.unlink(...)`,
        which is a call on its `SchemaDefinition` and not on the `Schema` a
        caller holds. So the one instruction that would fix their query names
        something they cannot do.

        The reasons are read only to recognise the failure. What the competing
        columns are is taken from the schema, which knows, rather than parsed
        back out of a sentence that was written to be read.
        """
        if 'is not unique' not in reasons:
            return ''
        ways: dict[tuple[str, str], list[str]] = {}
        for link in self.schema.relationships:
            pair = (link.source_table, link.target_table)
            ways.setdefault(pair, []).append(link.source_column)
        ambiguous = {
            pair: sorted(columns)
            for pair, columns in ways.items()
            if len(columns) > 1
        }
        if not ambiguous:
            return ''
        # Which pair the query crossed is in the refusal and nowhere else, so
        # the tables are read from it, but only to choose among pairs the
        # schema already knows about. Advising the first pair found sent a
        # caller to edit a relationship their query never touched.
        crossed = [
            pair
            for pair in ambiguous
            if f'table {pair[0]} ' in reasons and f'table {pair[1]} ' in reasons
        ]
        if len(ambiguous) > 1 and not crossed:
            return ''
        (source, target), columns = sorted(
            (pair, ambiguous[pair]) for pair in (crossed or ambiguous)
        )[0]
        kept, *rest = columns
        return (
            f' In this schema {source!r} reaches {target!r} through '
            f'{", ".join(repr(name) for name in columns)}; to keep {kept!r}, '
            f'schema.relationships = [link for link in schema.relationships '
            f'if not (link.source_table == {source!r} and link.target_table '
            f'== {target!r} and link.source_column in {rest!r})].'
        )

    def _why_the_target_is_text(self, reasons: str, profile: str) -> str:
        r"""Say why a target is `text`, when that is what refused the query.

        Inference types a string column by how many distinct values it holds,
        and past sixty-four it is `text` rather than `categorical`. That is a
        reasonable rule for a feature and a poor one for a target: an ordinary
        multiclass label with seventy classes is refused as unpredictable, and
        the refusal names a semantic type the caller never chose. Worse, it
        depends on how many rows were handed to `from_data`, so the same query
        answers on a sample and stops answering on the full table.

        Setting the meaning by hand fixes it, verified at seventy classes and
        at two hundred, so the refusal now says so.

        The column named is the target rather than whichever text column comes
        first. A schema with a free-text feature and a many-class label has
        two, and naming the wrong one offers a remedy that changes nothing
        about the refusal.
        """
        if 'semantic type text' not in reasons:
            return ''
        named = self._target_named_by_the_query()
        if named is None:
            return ''
        table, name = named
        column = self.schema[table].columns.get(name)
        if column is None or column.stype != 'text' or column.stype_declared:
            return ''
        frame = getattr(self.schema[table], 'data', None)
        distinct = frame[name].nunique() if frame is not None else None
        holds = (
            f'holds {distinct:,} distinct values'
            if distinct is not None
            else 'was read as prose'
        )
        said = (
            f'. {table}.{name} {holds}, and a '
            f'string column with more than {_MAX_CATEGORICAL} is typed text '
            f'rather than categorical. That is fair for a feature and a poor '
            f'one for a label'
        )
        ceiling = (
            RELATIONAL_LIMITS['classes_max'] if profile == RELATIONAL else None
        )
        if ceiling is not None and distinct is not None and distinct > ceiling:
            return (
                f'{said}. Typing it categorical would not help here, though: '
                f'{PROFILE_MODELS[profile]} classifies at most {ceiling} '
                f'classes and this column holds {distinct:,}, so the send '
                f'would be refused. Predict a column with fewer classes, or '
                f'group these into that many first'
            )
        return (
            f'{said}. If it is a label, say so with '
            f"schema[{table!r}][{name!r}].stype = 'categorical'"
        )

    def _target_named_by_the_query(self) -> tuple[str, str] | None:
        r"""The table and column the query predicts, read from its text.

        Read from the text rather than from a resolved query, because this runs
        precisely when resolution failed.
        """
        match = _PREDICTED_COLUMN.search(self.text)
        if match is None:
            return None
        table, name = match.group(1), match.group(2)
        if table not in self.schema:
            return None
        return table, name

    def __repr__(self) -> str:
        return f'PredictiveQuery({self.text!r}, {self.schema!r})'


#: `PREDICT <table>.<column>` is the plain column target, which is the only
#: shape a semantic type can be refused for.
_PREDICTED_COLUMN = re.compile(
    r'PREDICT\s+([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s', re.IGNORECASE
)

#: The English sentence inside a pydantic validation error, without the
#: `[type=..., input_value=...]` tail the library appends to it.
_VALIDATOR_SENTENCE = re.compile(r'Value error, (.+?)(?= \[type=)', re.DOTALL)


_A_REMEDY_IN_THE_LANGUAGE_S_OWN_WORDS = re.compile(
    r'\s*(?:Keep the one this query means and drop the rest, for example:'
    r'|You can drop the extra foreign keys? with:)'
    r'[^.]*?(?:schema|graph)\.unlink\([^)]*\)'
    r'(?:\s*;\s*(?:schema|graph)\.unlink\([^)]*\))*\.?'
)


def _without_a_remedy_this_caller_cannot_run(reasons: str) -> str:
    r"""The refusal without the repair written in the language's vocabulary.

    The query language advises `schema.unlink(...)`, a call on its own
    `SchemaDefinition` and not on the `Schema` a caller holds, and it chooses
    which link to keep by position. This object restates both in terms the
    caller can run, and two remedies that disagree about which link to keep
    are worse than one: followed in order they keep the wrong one. So the one
    that cannot be run is removed rather than printed beside it.
    """
    return _A_REMEDY_IN_THE_LANGUAGE_S_OWN_WORDS.sub('', reasons)


def _without_the_validator_trace(message: str) -> str:
    r"""The sentence a refusal is making, without the trace around it.

    A few window errors, including zero-width and inverted windows, are raised
    by
    the grammar's own models rather than by its parser, so they arrive as a
    pydantic trace: an error count, the internal type that raised, the repr of
    the arguments it was given, and a link to pydantic's documentation. The
    English inside is as actionable as the parser's own row-and-column
    messages; nothing around it is, and none of it is this package's to show.

    A message written any other way is left alone, so a validator that stops
    wrapping its errors this way loses the scaffolding rather than the text.
    """
    said = _VALIDATOR_SENTENCE.findall(message)
    return '; '.join(part.strip() for part in said) if said else message


def _is_hashable(value: Any) -> bool:
    r"""Whether this value can be a dict key, asked by trying.

    `isinstance(value, Hashable)` is structural: a class that defines
    `__hash__` and raises from it passes, and the `TypeError` surfaces later
    from inside pandas, which is the shape of failure this check exists to
    prevent.
    """
    try:
        hash(value)
    except TypeError:
        return False
    return True
