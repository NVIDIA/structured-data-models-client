# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Whether the model on the other end can answer a query the language accepts.

A separate pass, deliberately. It reads a query that has already resolved and
says only whether this model will take it. It cannot change how a name, a type,
a window or a join resolved, and it cannot accept anything resolution rejected,
because it never runs unless resolution succeeded.

That is what keeps "add a model" from meaning "edit the analyzer". See
``docs/spec.md`` §8.1, and ``predictive_query.profiles`` for the capabilities
themselves.
"""

from predictive_query.ast import Aggregation, ASTNode, Column, Filter, Join
from predictive_query.profiles import Profile
from predictive_query.query import ValidatedPredictiveQuery
from predictive_query.schema import codes
from predictive_query.schema.common import (
    Span,
    ValidationError,
    ValidationResponse,
)
from predictive_query.schema.graph import ColumnKey, GraphDefinition
from predictive_query.schema.typing import AggregationType
from predictive_query.validator.utils import col_name, table_name


def check_capabilities(
    query: ValidatedPredictiveQuery,
    graph: GraphDefinition,
    profile: Profile,
) -> ValidationResponse:
    r"""Diagnostics for what *profile* cannot answer about *query*.

    Args:
        query: A query that has already resolved against *graph*.
        graph: The schema it resolved against.
        profile: What the model on the other end can do.

    Returns:
        A response carrying one ``PQL_MODEL_*`` error per capability the query
        needs and the model lacks. Empty means this model will take it.
    """
    can = profile.capabilities
    response = ValidationResponse()
    span = Span.at(query.target_ast.get_location())

    tables = sorted({table_name(fqn) for fqn in query.all_query_columns})
    if not can.multiple_tables and len(tables) > 1:
        entity_table = table_name(query.entity_column)
        others = [name for name in tables if name != entity_table]
        response.errors.append(
            ValidationError(
                title='More than one table',
                message=(
                    f'This query reads {_and_list(others)} as well as '
                    f"'{entity_table}', and {can.model} predicts from one "
                    f'table. Move the columns into '
                    f"'{entity_table}', or use a model that reads "
                    f'relationships.'
                ),
                code=codes.MODEL_SINGLE_TABLE_ONLY,
                span=span,
            )
        )
    elif not can.relationships and _joins(query):
        response.errors.append(
            ValidationError(
                title='Relationship not supported',
                message=(
                    f'This query follows a foreign key, and {can.model} does '
                    f'not read relationships.'
                ),
                code=codes.MODEL_NO_RELATIONSHIPS,
                span=span,
            )
        )

    temporal = query.target_ast.date_offset_range is not None
    if temporal and not can.temporal:
        response.errors.append(
            ValidationError(
                title='Time range not supported',
                message=(
                    f'The target carries a time range, and {can.model} '
                    f'predicts a value as it stands.'
                ),
                code=codes.MODEL_NO_TEMPORAL_TARGET,
                span=span,
            )
        )
    if not temporal and not can.static:
        response.errors.append(
            ValidationError(
                title='Time range required',
                message=(
                    f'{can.model} predicts over a time range, and this '
                    f'target carries none. Add one, for example '
                    f'(0, 30, days).'
                ),
                code=codes.MODEL_NO_STATIC_TARGET,
                span=span,
            )
        )

    target = query.target_ast
    if isinstance(target, Join):
        target = target.rhs_target
    if not can.derived_targets and not isinstance(target, Column):
        response.errors.append(
            ValidationError(
                title='Derived target not supported',
                message=(
                    f'The target is computed from the table rather than '
                    f'stored in it, and {can.model} predicts a column as it '
                    f'stands. Add the derived value as a column of '
                    f"'{table_name(query.entity_column)}' and predict that "
                    f'column.'
                ),
                code=codes.MODEL_NO_DERIVED_TARGET,
                span=span,
            )
        )

    if not can.entity_filters and isinstance(query.entity_ast, Filter):
        response.errors.append(
            ValidationError(
                title='Filter not supported',
                message=(
                    f'This query filters the entities it predicts for, and '
                    f'{can.model} scores the rows it is given. Filter the '
                    f'rows before constructing the query.'
                ),
                code=codes.MODEL_NO_ENTITY_FILTER,
                span=span,
            )
        )

    if not can.counterfactuals and query.whatif_ast is not None:
        response.errors.append(
            ValidationError(
                title='ASSUMING not supported',
                message=(
                    f'This query asks what would happen under an ASSUMING '
                    f'clause, and {can.model} scores the rows it is given as '
                    f'they stand. Change the values in those rows instead.'
                ),
                code=codes.MODEL_NO_COUNTERFACTUAL,
                span=span,
            )
        )

    if not can.link_prediction and _ranks_a_foreign_key(query, graph):
        response.errors.append(
            ValidationError(
                title='Link prediction not supported',
                message=(
                    f'This query ranks a foreign key, and {can.model} does '
                    f'not do link prediction.'
                ),
                code=codes.MODEL_CANNOT_RANK_FOREIGN_KEY,
                span=span,
            )
        )

    if not can.forecasting and query.num_forecasts > 1:
        response.errors.append(
            ValidationError(
                title='Forecasting not supported',
                message=(
                    f'This query asks for {query.num_forecasts} timeframes, '
                    f'and {can.model} does not forecast.'
                ),
                code=codes.MODEL_NO_FORECASTING,
                span=span,
            )
        )

    return response


def _and_list(names: list[str]) -> str:
    quoted = [f"'{name}'" for name in names]
    if len(quoted) == 1:
        return quoted[0]
    return ', '.join(quoted[:-1]) + f' and {quoted[-1]}'


def _joins(query: ValidatedPredictiveQuery) -> bool:
    r"""Whether resolution had to follow a foreign key to reach anything."""

    def walk(node: object) -> bool:
        if isinstance(node, Join):
            return True
        if isinstance(node, ASTNode):
            return any(walk(child) for child in node.children)
        return False

    return any(
        walk(node)
        for node in (query.target_ast, query.entity_ast, query.whatif_ast)
        if node is not None
    )


def _ranks_a_foreign_key(
    query: ValidatedPredictiveQuery, graph: GraphDefinition
) -> bool:
    r"""Whether the target is a LIST_DISTINCT over a foreign key column."""
    target = query.target_ast
    if isinstance(target, Join):
        target = target.rhs_target
    if not isinstance(target, Aggregation):
        return False
    if target.aggr != AggregationType.LIST_DISTINCT:
        return False
    if not isinstance(target.target, Column):
        fqn = target.get_target_column_name()
    else:
        fqn = target.target.fqn
    table, column = table_name(fqn), col_name(fqn)
    if table not in graph.tables:
        return False
    if graph.tables[table].pkey == column:
        return False
    return any(
        ColumnKey(table, column) in group.columns for group in graph.col_groups
    )
