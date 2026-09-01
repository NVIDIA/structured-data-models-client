# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
from predictive_query.ast import (
    Aggregation,
    Column,
    Condition,
    Constant,
    Filter,
    Join,
    LogicalOperation,
)
from predictive_query.query import ValidatedPredictiveQuery
from predictive_query.schema.typing import (
    AggregationType,
    BoolOp,
    MemberOp,
    RelOp,
)

from sdm_relational_engine.rfm.pquery import PQueryExecutor


class PQueryPandasExecutor(PQueryExecutor[pd.DataFrame, pd.Series, np.ndarray]):
    def execute_column(
        self,
        column: Column,
        feat_dict: dict[str, pd.DataFrame],
        filter_na: bool = True,
    ) -> tuple[pd.Series, np.ndarray]:
        table_name, column_name = column.fqn.split('.')
        if column_name == '*':
            out = pd.Series(np.ones(len(feat_dict[table_name]), dtype='int64'))
        else:
            out = feat_dict[table_name][column_name]
            out = out.reset_index(drop=True)

        if pd.api.types.is_float_dtype(out):
            out = out.astype('float32')

        out.name = None
        out.index.name = None

        mask = out.notna().to_numpy()

        if not filter_na:
            return out, mask

        out = out[mask].reset_index(drop=True)

        # Cast to primitive dtype:
        if pd.api.types.is_integer_dtype(out):
            out = out.astype('int64')
        elif pd.api.types.is_bool_dtype(out):
            out = out.astype('bool')

        return out, mask

    def execute_aggregation_type(
        self,
        op: AggregationType,
        feat: pd.Series,
        batch: np.ndarray,
        batch_size: int,
        filter_na: bool = True,
    ) -> tuple[pd.Series, np.ndarray]:

        mask = feat.notna()
        feat, batch = feat[mask], batch[mask]

        if op == AggregationType.LIST_DISTINCT:
            df = pd.DataFrame(dict(feat=feat, batch=batch))
            df = df.drop_duplicates()
            out = df.groupby('batch')['feat'].agg(list)

        else:
            df = pd.DataFrame(dict(feat=feat, batch=batch))
            if op == AggregationType.AVG:
                agg = 'mean'
            elif op == AggregationType.COUNT:
                agg = 'size'
            else:
                agg = op.lower()
            out = df.groupby('batch')['feat'].agg(agg)

            if not pd.api.types.is_datetime64_any_dtype(out):
                out = out.astype('float32')

        out.name = None
        out.index.name = None

        if op in {AggregationType.SUM, AggregationType.COUNT}:
            out = out.reindex(range(batch_size), fill_value=0)
            mask = np.ones(batch_size, dtype=bool)
            return out, mask

        mask = np.zeros(batch_size, dtype=bool)
        mask[batch] = True

        if filter_na:
            return out.reset_index(drop=True), mask

        out = out.reindex(range(batch_size), fill_value=pd.NA)

        return out, mask

    def execute_aggregation(
        self,
        aggr: Aggregation,
        feat_dict: dict[str, pd.DataFrame],
        time_dict: dict[str, pd.Series],
        batch_dict: dict[str, np.ndarray],
        anchor_time: pd.Series,
        filter_na: bool = True,
    ) -> tuple[pd.Series, np.ndarray]:
        target_table = aggr.get_target_column_name().split('.')[0]
        target_batch = batch_dict[target_table]
        target_time = time_dict[target_table]
        if isinstance(aggr.target, Column):
            target_feat, target_mask = self.execute_column(
                column=aggr.target,
                feat_dict=feat_dict,
                filter_na=True,
            )
        else:
            assert isinstance(aggr.target, Filter)
            target_feat, target_mask = self.execute_filter(
                filter=aggr.target,
                feat_dict=feat_dict,
                time_dict=time_dict,
                batch_dict=batch_dict,
                anchor_time=anchor_time,
                filter_na=True,
            )

        assert aggr.aggr_time_range is not None
        anchor_target_time = anchor_time.iloc[target_batch]
        anchor_target_time = anchor_target_time.reset_index(drop=True)

        time_filter_mask = (
            target_time
            <= anchor_target_time + aggr.aggr_time_range.end_date_offset
        )
        if aggr.aggr_time_range.start is not None:
            start_offset = aggr.aggr_time_range.start_date_offset
            time_filter_mask = time_filter_mask & (
                target_time > anchor_target_time + start_offset
            )
        curr_target_mask = target_mask & time_filter_mask

        return self.execute_aggregation_type(
            AggregationType(aggr.aggr),
            feat=target_feat[
                time_filter_mask[target_mask].reset_index(drop=True)
            ],
            batch=target_batch[curr_target_mask],
            batch_size=len(anchor_time),
            filter_na=filter_na,
        )

    def execute_rel_op(
        self,
        left: pd.Series,
        op: RelOp,
        right: Constant,
    ) -> pd.Series:

        if right.typed_value() is None:
            if op == RelOp.EQ:
                return left.isna()
            assert op == RelOp.NEQ
            return left.notna()

        # Promote left to float if right is a float to avoid lossy coercion.
        right_value = right.typed_value()
        if pd.api.types.is_integer_dtype(left) and isinstance(
            right_value, float
        ):
            left = left.astype('float64')
        value = pd.Series([right_value], dtype=left.dtype).iloc[0]

        if op == RelOp.EQ:
            return (left == value).fillna(False).astype(bool)
        if op == RelOp.NEQ:
            out = (left != value).fillna(False).astype(bool)
            out[left.isna()] = False  # N/A != right should always be `False`.
            return out
        if op == RelOp.LEQ:
            return (left <= value).fillna(False).astype(bool)
        if op == RelOp.GEQ:
            return (left >= value).fillna(False).astype(bool)
        if op == RelOp.LT:
            return (left < value).fillna(False).astype(bool)
        if op == RelOp.GT:
            return (left > value).fillna(False).astype(bool)

        raise NotImplementedError(f"Operator '{op}' not implemented")

    def execute_member_op(
        self,
        left: pd.Series,
        op: MemberOp,
        right: Constant,
    ) -> pd.Series:

        if op == MemberOp.IN:
            ser = pd.Series(right.typed_value(), dtype=left.dtype)
            return left.isin(ser).astype(bool)

        raise NotImplementedError(f"Operator '{op}' not implemented")

    def execute_condition(
        self,
        condition: Condition,
        feat_dict: dict[str, pd.DataFrame],
        time_dict: dict[str, pd.Series],
        batch_dict: dict[str, np.ndarray],
        anchor_time: pd.Series,
        filter_na: bool = True,
    ) -> tuple[pd.Series, np.ndarray]:
        assert isinstance(condition.value, Constant)
        value_is_na = condition.value.typed_value() is None
        if isinstance(condition.target, Column):
            left, mask = self.execute_column(
                column=condition.target,
                feat_dict=feat_dict,
                filter_na=filter_na if not value_is_na else False,
            )
        elif isinstance(condition.target, Join):
            left, mask = self.execute_join(
                join=condition.target,
                feat_dict=feat_dict,
                time_dict=time_dict,
                batch_dict=batch_dict,
                anchor_time=anchor_time,
                filter_na=filter_na if not value_is_na else False,
            )
        else:
            assert isinstance(condition.target, Aggregation)
            left, mask = self.execute_aggregation(
                aggr=condition.target,
                feat_dict=feat_dict,
                time_dict=time_dict,
                batch_dict=batch_dict,
                anchor_time=anchor_time,
                filter_na=filter_na if not value_is_na else False,
            )

        if filter_na and value_is_na:
            mask = np.ones(len(left), dtype=bool)

        if isinstance(condition.op, RelOp):
            out = self.execute_rel_op(
                left=left,
                op=condition.op,
                right=condition.value,
            )
        else:
            assert isinstance(condition.op, MemberOp)
            out = self.execute_member_op(
                left=left,
                op=condition.op,
                right=condition.value,
            )

        return out, mask

    def execute_bool_op(
        self,
        left: pd.Series,
        op: BoolOp,
        right: pd.Series | None,
    ) -> pd.Series:

        # TODO: implement Kleene-Priest three-value logic.
        if op == BoolOp.AND:
            assert right is not None
            return left & right
        if op == BoolOp.OR:
            assert right is not None
            return left | right
        if op == BoolOp.NOT:
            return ~left

        raise NotImplementedError(f"Operator '{op}' not implemented")

    def execute_logical_operation(
        self,
        logical_operation: LogicalOperation,
        feat_dict: dict[str, pd.DataFrame],
        time_dict: dict[str, pd.Series],
        batch_dict: dict[str, np.ndarray],
        anchor_time: pd.Series,
        filter_na: bool = True,
    ) -> tuple[pd.Series, np.ndarray]:
        if isinstance(logical_operation.left, Condition):
            left, mask = self.execute_condition(
                condition=logical_operation.left,
                feat_dict=feat_dict,
                time_dict=time_dict,
                batch_dict=batch_dict,
                anchor_time=anchor_time,
                filter_na=False,
            )
        else:
            assert isinstance(logical_operation.left, LogicalOperation)
            left, mask = self.execute_logical_operation(
                logical_operation=logical_operation.left,
                feat_dict=feat_dict,
                time_dict=time_dict,
                batch_dict=batch_dict,
                anchor_time=anchor_time,
                filter_na=False,
            )

        right = right_mask = None
        if isinstance(logical_operation.right, Condition):
            right, right_mask = self.execute_condition(
                condition=logical_operation.right,
                feat_dict=feat_dict,
                time_dict=time_dict,
                batch_dict=batch_dict,
                anchor_time=anchor_time,
                filter_na=False,
            )
        elif isinstance(logical_operation.right, LogicalOperation):
            right, right_mask = self.execute_logical_operation(
                logical_operation=logical_operation.right,
                feat_dict=feat_dict,
                time_dict=time_dict,
                batch_dict=batch_dict,
                anchor_time=anchor_time,
                filter_na=False,
            )

        out = self.execute_bool_op(left, logical_operation.bool_op, right)

        if right_mask is not None:
            mask = mask & right_mask

        if filter_na:
            out = out[mask].reset_index(drop=True)

        return out, mask

    def execute_join(
        self,
        join: Join,
        feat_dict: dict[str, pd.DataFrame],
        time_dict: dict[str, pd.Series],
        batch_dict: dict[str, np.ndarray],
        anchor_time: pd.Series,
        filter_na: bool = True,
    ) -> tuple[pd.Series, np.ndarray]:
        if isinstance(join.rhs_target, Aggregation):
            return self.execute_aggregation(
                aggr=join.rhs_target,
                feat_dict=feat_dict,
                time_dict=time_dict,
                batch_dict=batch_dict,
                anchor_time=anchor_time,
                filter_na=True,
            )
        raise NotImplementedError(
            f'Unexpected {type(join.rhs_target)} nested in Join'
        )

    def execute_filter(
        self,
        filter: Filter,
        feat_dict: dict[str, pd.DataFrame],
        time_dict: dict[str, pd.Series],
        batch_dict: dict[str, np.ndarray],
        anchor_time: pd.Series,
        filter_na: bool = True,
    ) -> tuple[pd.Series, np.ndarray]:
        assert filter.target is not None
        out, mask = self.execute_column(
            column=filter.target,
            feat_dict=feat_dict,
            filter_na=False,
        )

        if isinstance(filter.condition, Condition):
            _mask = self.execute_condition(
                condition=filter.condition,
                feat_dict=feat_dict,
                time_dict=time_dict,
                batch_dict=batch_dict,
                anchor_time=anchor_time,
                filter_na=False,
            )[0].to_numpy()
        else:
            assert isinstance(filter.condition, LogicalOperation)
            _mask = self.execute_logical_operation(
                logical_operation=filter.condition,
                feat_dict=feat_dict,
                time_dict=time_dict,
                batch_dict=batch_dict,
                anchor_time=anchor_time,
                filter_na=False,
            )[0].to_numpy()

        if filter_na:
            return out[_mask & mask].reset_index(drop=True), _mask & mask
        return out[_mask].reset_index(drop=True), mask & _mask

    def execute(
        self,
        query: ValidatedPredictiveQuery,
        feat_dict: dict[str, pd.DataFrame],
        time_dict: dict[str, pd.Series],
        batch_dict: dict[str, np.ndarray],
        anchor_time: pd.Series,
        filter_na: bool = True,
    ) -> tuple[pd.Series, np.ndarray]:
        if isinstance(query.entity_ast, Column):
            mask = self.execute_column(
                column=query.entity_ast,
                feat_dict=feat_dict,
                filter_na=False,
            )[1]
        else:
            assert isinstance(query.entity_ast, Filter)
            mask = self.execute_filter(
                filter=query.entity_ast,
                feat_dict=feat_dict,
                time_dict=time_dict,
                batch_dict=batch_dict,
                anchor_time=anchor_time,
                filter_na=False,
            )[1]

        if isinstance(query.target_ast, Column):
            out, _mask = self.execute_column(
                column=query.target_ast,
                feat_dict=feat_dict,
                filter_na=filter_na,
            )
        elif isinstance(query.target_ast, Condition):
            out, _mask = self.execute_condition(
                condition=query.target_ast,
                feat_dict=feat_dict,
                time_dict=time_dict,
                batch_dict=batch_dict,
                anchor_time=anchor_time,
                filter_na=filter_na,
            )
        elif isinstance(query.target_ast, Aggregation):
            out, _mask = self.execute_aggregation(
                aggr=query.target_ast,
                feat_dict=feat_dict,
                time_dict=time_dict,
                batch_dict=batch_dict,
                anchor_time=anchor_time,
                filter_na=filter_na,
            )
        elif isinstance(query.target_ast, Join):
            out, _mask = self.execute_join(
                join=query.target_ast,
                feat_dict=feat_dict,
                time_dict=time_dict,
                batch_dict=batch_dict,
                anchor_time=anchor_time,
                filter_na=filter_na,
            )
        else:
            assert isinstance(query.target_ast, LogicalOperation)
            out, _mask = self.execute_logical_operation(
                logical_operation=query.target_ast,
                feat_dict=feat_dict,
                time_dict=time_dict,
                batch_dict=batch_dict,
                anchor_time=anchor_time,
                filter_na=filter_na,
            )

        if query.whatif_ast is not None:
            if isinstance(query.whatif_ast, Condition):
                mask = (
                    mask
                    & self.execute_condition(
                        condition=query.whatif_ast,
                        feat_dict=feat_dict,
                        time_dict=time_dict,
                        batch_dict=batch_dict,
                        anchor_time=anchor_time,
                        filter_na=False,
                    )[0].to_numpy()
                )
            else:
                assert isinstance(query.whatif_ast, LogicalOperation)
                mask = (
                    mask
                    & self.execute_logical_operation(
                        logical_operation=query.whatif_ast,
                        feat_dict=feat_dict,
                        time_dict=time_dict,
                        batch_dict=batch_dict,
                        anchor_time=anchor_time,
                        filter_na=False,
                    )[0].to_numpy()
                )

        if filter_na:
            out = out[mask[_mask]].reset_index(drop=True)

        return out, mask & _mask
