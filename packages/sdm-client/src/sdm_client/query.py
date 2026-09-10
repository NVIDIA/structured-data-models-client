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
from predictive_query import RELATIONAL, TABULAR

from sdm_client.errors import StructuredDataError
from sdm_client.materializers import RelationalMaterializer, TabularMaterializer
from sdm_client.models import Model, spec_for
from sdm_client.request import Request
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

        The request takes one of three paths: predict=<frame> uses the full
        single table as context, the tabular route builds the request directly
        in pandas, and the relational route gathers the tables as a graph.

        ``model`` chooses between the last two. Left unsaid, the schema's shape
        decides: one table is gathered tabularly and several relationally. Name
        a model when that default is not what you want -- a single table is a
        graph of one node, and the relational model answers it.

        Args:
            indices: Keys of rows already present in the tables. Mutually
                exclusive with ``predict``.
            predict: Rows supplied directly for scoring. Supported for
                single-table schemas and mutually exclusive with ``indices``.
            num_context_samples: For ``indices`` requests, the maximum number
                of labelled context rows to use. When omitted, single-table
                requests use all eligible rows; relational requests use up to
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
            else len(self.schema) > 1
        )
        self._validate_materialize_args(
            relational=relational,
            model_id=spec.model_id if spec is not None else None,
            len_schema=len(self.schema),
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
        indices: Sequence[Any] | None,
        predict: pd.DataFrame | None,
        num_context_samples: int | None,
        options: dict[str, Any] | None,
    ) -> None:
        # The tabular route reads one table. Reachable only when the caller
        # named the tabular model, since the shape-based default sends a wider
        # schema the other way.
        if not relational and len_schema != 1:
            raise StructuredDataError(
                f'{model_id} answers from a single table, and this schema has '
                f'{len_schema}. Pass model=KumoRelational() to gather them as '
                f'a graph',
                code='INVALID_REQUEST',
            )

        if not relational and options:
            raise StructuredDataError(
                f'a single-table request takes no gathering options: '
                f'{sorted(options)}. They apply only to relational '
                f'materialization, so pass model=KumoRelational() to gather '
                f'this schema as a graph',
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
                raise StructuredDataError(
                    'predict=<frame> is the single-table form, and '
                    f'{model_id} scores rows the tables already contain. '
                    f'Pass indices=[...] naming them, or gather this request '
                    f'for the tabular model',
                    code='INVALID_REQUEST',
                )
            if indices is not None:
                raise StructuredDataError(
                    'pass either indices=[...] to score rows the tables already '
                    'contain, or predict=<frame> to score rows they do not—not both',
                    code='INVALID_REQUEST',
                )
            if num_context_samples is not None:
                raise StructuredDataError(
                    'num_context_samples cannot be used with predict=<frame>: '
                    'the schema table is used as context in full. To use less '
                    'context, construct the query with a smaller table',
                    code='INVALID_REQUEST',
                )

        if indices is None and predict is None:
            raise StructuredDataError(
                'materialize() needs the rows to score: pass indices=[...] '
                'naming the entities. There is no all-rows default, because '
                'it would silently score an entire table',
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
                f'can answer: {reasons}',
                code='INVALID_REQUEST',
            )
        return resolved

    def __repr__(self) -> str:
        return f'PredictiveQuery({self.text!r}, {self.schema!r})'
