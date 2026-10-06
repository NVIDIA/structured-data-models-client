# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from dataclasses import field

from pydantic.dataclasses import dataclass
from typing_extensions import Self

from predictive_query.ast import (
    Aggregation,
    ASTNode,
    Column,
    Condition,
    Constant,
    Filter,
    Join,
    LogicalOperation,
)
from predictive_query.ast.ast_node import ArrayDtype
from predictive_query.ast.date_offset_range import DateOffsetRange
from predictive_query.ast.location_interval import (
    ASTQueryLocationInterval,
)
from predictive_query.schema.common import ValidationResponse
from predictive_query.schema.typing import (
    BoolOp,
    Dtype,
    ProblemType,
    RelOp,
    Stype,
)
from predictive_query.schema.utils import QueryType, maybe_bold


@dataclass
class ParsedPredictiveQuery:
    r"""A predictive query as the parser produced it.

    Carries no schema-dependent data, including types or join keys.
    :class:`ValidatedPredictiveQuery` is the same query once those are known.

    Args:
        entity_ast: The abstract syntax tree corresponding to the entity
            definition, following `FOR`.
        target_ast: The abstract syntax tree corresponding to the target
            definition, following `PREDICT`.
        whatif_ast: The abstract syntax tree corresponding to the `ASSUMING`
            clause, if present in the query.
        top_k: Integer K corresponding to the `TOP K`
            clause, if present in the query.
        problem_type: `RANK` or `CLASSIFY`, if present in the query.
        rfm_query: :obj:`True` if this query is written against a
            pretrained model rather than one trained on it.
        for_each: `FOR` or `FOR EACH` keyword.
        rfm_entity_ids: Abstract syntax tree defining the RFM entity IDs.
        num_forecasts: The number of forecasts. Internal use only.
        evaluate: Whether the caller asked to evaluate rather than
            predict, via the `EVALUATE` prefix.
        explain: Whether the caller asked for an explanation, via the
            `EXPLAIN` prefix.

    """

    entity_ast: Column | Filter
    target_ast: LogicalOperation | Join | Condition | Column | Aggregation
    whatif_ast: Condition | LogicalOperation | None = None
    top_k: int | None = None
    problem_type: ProblemType | str | None = None
    rfm_query: bool = False
    for_each: str = 'FOR EACH'
    rfm_entity_ids: Condition | None = None
    num_forecasts: int = 1
    evaluate: bool = False  # TODO: deprecate
    explain: bool = False  # TODO: deprecate

    def __post_init__(self) -> None:
        if self.problem_type is not None and not isinstance(
            self.problem_type, ProblemType
        ):
            self.problem_type = ProblemType(self.problem_type.upper())

    @property
    def entity_column(self) -> str:
        r"""The name of the entity column of the query in the
        `table.column` format.
        """
        if isinstance(self.entity_ast, Column):
            return self.entity_ast.fqn
        if isinstance(self.entity_ast, Filter):
            assert self.entity_ast.target is not None
            return self.entity_ast.target.fqn
        raise ValueError(
            f'`{self.entity_ast}` has an invalid type '
            f'`{type(self.entity_ast)}`, expected `Column` or `Filter`.'
        )

    @property
    def entity_table(self) -> str:
        r"""Table corresponding to the entity column."""
        return self.entity_column.split('.')[0]

    @property
    def entity_column_obj(self) -> Column:
        r"""The entity column object of the query in the
        `table.column` format.
        """
        if isinstance(self.entity_ast, Column):
            return self.entity_ast
        if isinstance(self.entity_ast, Filter):
            assert self.entity_ast.target is not None
            return self.entity_ast.target
        raise AssertionError(
            f'unsupported entity AST {type(self.entity_ast).__name__}'
        )

    @property
    def target_column(self) -> str:
        r"""The name of the target column of the query in the `table.column`
        format.
        """
        raise NotImplementedError

    @property
    def entity_timeframe(self) -> DateOffsetRange | None:
        r"""Timeframe of the entity AST. :class:`DateOffsetRange` or
        :obj:`None` if there are no time intervals in the entity definition.
        """
        return self.entity_ast.date_offset_range

    @property
    def target_timeframe(self) -> DateOffsetRange | None:
        r"""Timeframe of the target AST. :class:`DateOffsetRange` or
        :obj:`None` if there are no time intervals in the target definition.
        """
        return self.target_ast.date_offset_range

    @property
    def whatif_timeframe(self) -> DateOffsetRange | None:
        r"""Timeframe of the whatif AST. :class:`DateOffsetRange` or
        :obj:`None` if there is no `whatif` condition or if there are no time
        intervals in the whatif definition.
        """
        if self.whatif_ast is None:
            return None
        return self.whatif_ast.date_offset_range

    @property
    def all_query_columns_with_locations(
        self,
    ) -> list[tuple[str, ASTQueryLocationInterval]]:
        r"""Returns the list of all columns that appear in the query in the
        `table.column` format.
        """
        all_columns: set[tuple[str, ASTQueryLocationInterval]] = set()
        all_columns |= set(self.entity_ast.all_query_columns_with_locations)
        all_columns |= set(self.target_ast.all_query_columns_with_locations)
        if self.whatif_ast is not None:
            all_columns |= set(self.whatif_ast.all_query_columns_with_locations)
        return list(all_columns)

    @property
    def all_query_columns(self) -> list[str]:
        r"""Returns the list of all columns that appear in the query in the
        `table.column` format.
        """
        all_columns: set[str] = set()
        all_columns |= set(self.entity_ast.all_query_columns)
        all_columns |= set(self.target_ast.all_query_columns)
        if self.whatif_ast is not None:
            all_columns |= set(self.whatif_ast.all_query_columns)
        return list(all_columns)

    def to_string(
        self, rich: bool = False, exclude_predict: bool = False
    ) -> str:
        r"""String representation of the predictive query.

        Args:
            rich: If :obj:`True`, add bold tags for prettier display.
            exclude_predict: If :obj:`True`, omit `PREDICT` prefix.

        Returns:
            String, corresponding to the input query.
        """
        predict_repr = (
            '' if exclude_predict else maybe_bold('PREDICT', rich) + ' '
        )
        query_str = f'{predict_repr}{self.target_ast.to_string(rich=rich)}'
        if self.evaluate and not exclude_predict:
            query_str = 'EVALUATE ' + query_str
        if self.explain and not exclude_predict:
            query_str = 'EXPLAIN ' + query_str
        if self.problem_type is not None:
            assert isinstance(self.problem_type, ProblemType)
            if self.problem_type == ProblemType.FORECAST:
                query_str += (
                    f' {maybe_bold("FORECAST", rich)}'
                    f' {self.num_forecasts}'
                    f' {maybe_bold("TIMEFRAMES", rich)}'
                )
            else:
                query_str += f' {self.problem_type.value}'
        if self.top_k is not None:
            query_str += ' ' + maybe_bold(f'TOP {self.top_k}', rich)
        query_str += ' ' + maybe_bold(self.for_each, rich)
        # We "split" the entity IDs and entity def into 2 separate ASTs for
        # convenience. We put them back together.
        if self.rfm_entity_ids is None:
            query_str += f' {self.entity_ast.to_string(rich=rich)}'
        elif isinstance(self.entity_ast, Column):
            query_str += f' {self.rfm_entity_ids.to_string(rich=rich)}'
        else:
            assert isinstance(self.entity_ast, Filter)
            assert self.entity_ast.condition is not None
            query_str += f' {self.rfm_entity_ids.to_string(rich=rich)}'
            query_str += (
                f' {maybe_bold("WHERE", rich)} '
                f'{self.entity_ast.condition.to_string(rich=rich)}'
            )
        if self.whatif_ast is not None:
            query_str += (
                f' {maybe_bold("ASSUMING", rich)} '
                f'{self.whatif_ast.to_string(rich=rich)}'
            )
        return query_str

    def to_autoregressive_query(self) -> Self:
        r"""Modifies the query for usage as an autoregressive label generator.

        In particular,
        * removes what-if clauses
        * removes any entity filters
        * removes target condition for single, numerical aggregations
        """
        out = copy.copy(self)

        out.whatif_ast = None

        if isinstance(out.entity_ast, Filter):
            assert out.entity_ast is not None
            out.entity_ast = out.entity_ast.target  # type: ignore

        if (
            isinstance(out.target_ast, LogicalOperation)
            and out.target_ast.bool_op == BoolOp.NOT
        ):
            out.target_ast = out.target_ast.left  # type: ignore

        if (
            isinstance(out.target_ast, Condition)
            and isinstance(out.target_ast.target, Aggregation)
            and isinstance(out.target_ast.op, RelOp)
        ):
            out.target_ast = out.target_ast.target

        return out


@dataclass
class ValidatedPredictiveQuery(ParsedPredictiveQuery):
    r"""A parsed query the validator has accepted, against a schema.

    Adds the types, join keys and task metadata that
    :class:`ParsedPredictiveQuery` deliberately lacks. It also keeps the
    :class:`ValidationResponse` that produced them. The response may still
    carry warnings because only errors stop a query.
    """

    validation_response: ValidationResponse = field(
        default_factory=ValidationResponse
    )

    @property
    def entity_dtype(self) -> Dtype:
        dtype = self.entity_ast.dtype
        assert dtype is not None
        assert isinstance(dtype, Dtype)
        return dtype

    @property
    def target_dtype(self) -> Dtype:
        dtype = self.target_ast.dtype
        assert dtype is not None
        if isinstance(dtype, ArrayDtype):
            dtype = dtype.to_dtype()
        return dtype

    @property
    def target_column(self) -> str:
        column = self.get_final_target_column()
        if column is not None:
            return column.fqn
        return self.target_ast.to_string()

    @property
    def target_stype(self) -> Stype:
        assert self.target_ast.stype is not None
        return self.target_ast.stype

    @property
    def query_type(self) -> QueryType:
        if self.target_timeframe is not None:
            return QueryType.TEMPORAL
        return QueryType.STATIC

    def get_single_target_fkey(self, lhs: bool = False) -> str | None:
        r"""Returns the fkey to the entity table from one target table.
        If there are multiple tables, it returns the alphabetically smallest
        one.

        Args:
            lhs: When the hop in question is a fkey->pkey
                hop, the fkey is going to be in the entity table, pointing
                to the target table. If this argument is set as `True`, the
                method will return the left hand side (lhs) key in the join,
                that is, the foreign key pointing to the target table pkey.

        Returns:
            Fully qualified fkey name in table.column format or
            :obj:`None` if the target and the entity table are the same.
        """
        return self._get_single_target_fkey(self.target_ast, lhs)

    def _get_single_target_fkey(self, node: ASTNode, lhs: bool) -> str | None:
        if isinstance(node, Join):
            if lhs:
                return node.lhs_key
            return node.rhs_key
        if isinstance(node, Filter):
            assert node.target is not None
            return self._get_single_target_fkey(node.target, lhs)
        keys = [
            self._get_single_target_fkey(child, lhs) for child in node.children
        ]
        str_keys = [x for x in keys if x is not None]
        if len(str_keys) == 0:
            return None
        return min(str_keys)

    def get_final_target_aggregation(self) -> Aggregation | None:
        r"""Returns the final aggregation node performed on
        the target data if the last operation performed on the data was an
        aggregation. If the last performed operation was a condition,
        logical operation, or column, it returns :obj:`None`.
        """
        return self._get_final_aggregation(self.target_ast)

    def _get_final_aggregation(self, node: ASTNode) -> Aggregation | None:
        if isinstance(node, Aggregation):
            return node
        if isinstance(node, Join):
            assert node.rhs_target is not None
            return self._get_final_aggregation(node.rhs_target)
        return None

    def get_final_target_column(self) -> Column | None:
        r"""Returns the column node if the target labels are taken directly
        from a column. If the last performed operation was a condition,
        logical operation, or an aggregation, it returns :obj:`None`.
        """
        return self._get_final_column(self.target_ast)

    def _get_final_column(self, node: ASTNode) -> Column | None:
        if isinstance(node, Column):
            return node
        if isinstance(node, Join):
            assert node.rhs_target is not None
            return self._get_final_column(node.rhs_target)
        return None

    def get_all_target_aggregations(self) -> list[Aggregation]:
        r"""Returns the list of aggregation nodes used in computing the
        label, excluding the ones that appear in filters. Unlike
        `get_final_target_aggregation`, it also includes aggregations within
        conditions. Returns an empty list if no aggregations are involved in
        label computation, e.g. in static queries.
        """
        return self._get_all_target_aggregations(self.target_ast)

    def _get_all_target_aggregations(self, node: ASTNode) -> list[Aggregation]:
        if isinstance(node, Aggregation):
            return [node]
        if isinstance(node, Filter):
            assert node.target is not None
            return self._get_all_target_aggregations(node.target)
        result: list[Aggregation] = []
        for child in node.children:
            result.extend(self._get_all_target_aggregations(child))
        return result

    def get_combined_date_offset_range(self) -> DateOffsetRange:
        r"""Returns the :class:`DateOffsetRange` that corresponds to combined
        target aggregation time ranges. This is exactly the timespan that
        should not overlap between splits lest we risk data leakage.
        """
        target_aggrs = self.get_all_target_aggregations()
        if len(target_aggrs) == 0:
            raise ValueError(
                'Cannot compute combined time range because '
                'target does not have any aggregations.'
            )
        cumulative_date_offset = None
        for aggr in target_aggrs:
            assert aggr.aggr_time_range is not None
            if cumulative_date_offset is None:
                cumulative_date_offset = aggr.aggr_time_range
            else:
                cumulative_date_offset = DateOffsetRange.merge_ranges(
                    cumulative_date_offset, aggr.aggr_time_range
                )
        assert cumulative_date_offset is not None
        return cumulative_date_offset

    def get_rfm_entity_id_list(self) -> list:
        r"""Returns a list of entity specified by the user. If there are no
        entities specified by the user, the returned list is empty. If the
        user specified only one entity with 'FOR T.C = <ID>', the list will
        contain only one entity.
        """
        if self.rfm_entity_ids is None:
            return []
        assert isinstance(self.rfm_entity_ids.value, Constant)
        ids = self.rfm_entity_ids.value.typed_value()
        if isinstance(ids, list):
            return ids
        return [ids]

    def get_exclude_cols_dict(self) -> dict[str, list[str]]:
        r"""The columns of tables to exclude during model execution.
        Applies to static node prediction query targets, i.e.
        target columns that appear outside of aggregations and filter
        conditions.
        """

        def _get_exclude_cols_dict(
            target: ASTNode,
        ) -> dict[str, list[str]]:

            if isinstance(target, Column):
                return {target.fqn.split('.')[0]: [target.fqn.split('.')[1]]}
            if isinstance(target, Aggregation):
                return {}
            if isinstance(target, Filter):
                assert target.target is not None
                return _get_exclude_cols_dict(target.target)

            dicts = [_get_exclude_cols_dict(child) for child in target.children]

            out_dict = {}
            for cols_dict in dicts:
                for table, cols in cols_dict.items():
                    if table not in out_dict:
                        out_dict[table] = cols
                    else:
                        out_dict[table] = list(set(out_dict[table]) | set(cols))
            return out_dict

        return _get_exclude_cols_dict(self.target_ast)
