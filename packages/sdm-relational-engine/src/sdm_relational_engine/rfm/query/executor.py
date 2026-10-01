# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from predictive_query.ast import (
    Aggregation,
    Column,
    Condition,
    Filter,
    Join,
    LogicalOperation,
)
from predictive_query.query import ValidatedPredictiveQuery

TableData = TypeVar('TableData')
ColumnData = TypeVar('ColumnData')
IndexData = TypeVar('IndexData')


class PQueryExecutor(Generic[TableData, ColumnData, IndexData], ABC):
    @abstractmethod
    def execute_column(
        self,
        column: Column,
        feat_dict: dict[str, TableData],
        filter_na: bool = True,
    ) -> tuple[ColumnData, IndexData]:
        pass

    @abstractmethod
    def execute_aggregation(
        self,
        aggr: Aggregation,
        feat_dict: dict[str, TableData],
        time_dict: dict[str, ColumnData],
        batch_dict: dict[str, IndexData],
        anchor_time: ColumnData,
        filter_na: bool = True,
    ) -> tuple[ColumnData, IndexData]:
        pass

    @abstractmethod
    def execute_condition(
        self,
        condition: Condition,
        feat_dict: dict[str, TableData],
        time_dict: dict[str, ColumnData],
        batch_dict: dict[str, IndexData],
        anchor_time: ColumnData,
        filter_na: bool = True,
    ) -> tuple[ColumnData, IndexData]:
        pass

    @abstractmethod
    def execute_logical_operation(
        self,
        logical_operation: LogicalOperation,
        feat_dict: dict[str, TableData],
        time_dict: dict[str, ColumnData],
        batch_dict: dict[str, IndexData],
        anchor_time: ColumnData,
        filter_na: bool = True,
    ) -> tuple[ColumnData, IndexData]:
        pass

    @abstractmethod
    def execute_join(
        self,
        join: Join,
        feat_dict: dict[str, TableData],
        time_dict: dict[str, ColumnData],
        batch_dict: dict[str, IndexData],
        anchor_time: ColumnData,
        filter_na: bool = True,
    ) -> tuple[ColumnData, IndexData]:
        pass

    @abstractmethod
    def execute_filter(
        self,
        filter: Filter,
        feat_dict: dict[str, TableData],
        time_dict: dict[str, ColumnData],
        batch_dict: dict[str, IndexData],
        anchor_time: ColumnData,
    ) -> tuple[ColumnData, IndexData]:
        pass

    @abstractmethod
    def execute(
        self,
        query: ValidatedPredictiveQuery,
        feat_dict: dict[str, TableData],
        time_dict: dict[str, ColumnData],
        batch_dict: dict[str, IndexData],
        anchor_time: ColumnData,
    ) -> tuple[ColumnData, IndexData]:
        pass
