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
Nothing is sent -- what comes back is a :class:`~sdm_client.Request` gathered
for one model. Which one follows the schema's shape unless ``model`` says
otherwise, and the model asked to answer it has to be that same one.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import TYPE_CHECKING, Any

import pandas as pd
from predictive_query import RELATIONAL, TABULAR

from sdm_client.errors import StructuredDataError
from sdm_client.materializers import RelationalMaterializer, TabularMaterializer
from sdm_client.models import Model, spec_for
from sdm_client.request import Request
from sdm_client.schema import _MAX_CATEGORICAL_CARDINALITY as _MAX_CATEGORICAL
from sdm_client.schema import Schema

if TYPE_CHECKING:
    from predictive_query import Profile, ValidatedPredictiveQuery


class PredictiveQuery:
    r"""A question expressed in PQL and answered from the tables in its schema.

    The query is resolved and validated when :meth:`materialize` builds a request.

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
        that default is not what you want -- a single table with no
        relationships is a graph of one node, and the relational model answers
        it.

        Args:
            indices: Keys of rows already present in the tables. Mutually
                exclusive with ``predict``.
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
            **options: Relational sampling options, such as ``num_neighbors``
                and ``anchor_time``.

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
        relational = (
            spec.profile == RELATIONAL
            if spec is not None
            else len(self.schema) > 1 or bool(self.schema.relationships)
        )
        self._validate_materialize_args(
            relational=relational,
            model_id=spec.model_id if spec is not None else None,
            len_schema=len(self.schema),
            relationships=len(self.schema.relationships),
            indices=indices,
            predict=predict,
            num_context_samples=num_context_samples,
            options=options,
        )

        if predict is not None:
            return self._tabular_materializer.table(
                query=self._resolve(TABULAR),
                predict=predict,
            )

        assert indices is not None
        if not relational:
            return self._tabular_materializer.indices(
                query=self._resolve(TABULAR),
                indices=indices,
                num_context_samples=num_context_samples,
            )

        return self._relational_materializer.materialize(
            query=self._resolve(RELATIONAL),
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
                'contain, or predict=<frame> to score rows they do not—not both',
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
            if len(indices) == 0:
                raise StructuredDataError(
                    'indices is empty, so there is nothing to score',
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
            if num_context_samples < 1:
                raise StructuredDataError(
                    'num_context_samples must be at least 1; a model predicts '
                    f'from labelled examples, and {num_context_samples} '
                    f'leaves it none',
                    code='INVALID_REQUEST',
                )

    def _resolve(self, profile: Profile) -> ValidatedPredictiveQuery:
        r"""Parse and validate the query."""
        from predictive_query import validate

        from sdm_client.schema import to_graph_definition

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
                f'can answer: {reasons}{self._why_the_target_is_text(reasons)}',
                code='INVALID_REQUEST',
            )
        return resolved

    def _why_the_target_is_text(self, reasons: str) -> str:
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
        distinct = self.schema[table].data[name].nunique()
        return (
            f'. {table}.{name} holds {distinct:,} distinct values, and a '
            f'string column with more than {_MAX_CATEGORICAL} is typed text '
            f'rather than categorical -- a fair rule for a feature and a poor '
            f'one for a label. If it is a label, say so with '
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


#: `PREDICT <table>.<column>` -- the plain column target, which is the only
#: shape a semantic type can be refused for.
_PREDICTED_COLUMN = re.compile(
    r'PREDICT\s+([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s', re.IGNORECASE
)


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
