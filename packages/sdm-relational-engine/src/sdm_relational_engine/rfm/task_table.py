# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from collections.abc import Sequence

import pandas as pd
from predictive_query.schema.task import TaskType
from predictive_query.schema.typing import Stype
from typing_extensions import Self

from sdm_relational_engine.rfm.base import Column
from sdm_relational_engine.rfm.base.utils import to_naive_utc
from sdm_relational_engine.rfm.infer import infer_dtype, infer_stype


class TaskTable:
    r"""A :class:`TaskTable` fully specifies the task, *i.e.* its context and
    prediction examples with entity IDs, targets and timestamps.

    Args:
        task_type: The task type.
        context_df: The data frame holding context examples.
        pred_df: The data frame holding prediction examples.
        entity_table_name: The entity table to predict for. For link prediction
            tasks, needs to hold both entity and target table names.
        entity_column: The name of the entity column.
        target_column: The name of the target column.
        time_column: The name of the time column to use as anchor time. If
            ``TaskTable.ENTITY_TIME``, use the timestamp of the entity table
            as anchor time.
    """

    ENTITY_TIME = '__entity_time__'

    def __init__(
        self,
        task_type: TaskType,
        context_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        entity_table_name: str | Sequence[str],
        entity_column: str,
        target_column: str,
        time_column: str | None = None,
        step_size: int | None = None,
        num_forecasts: int = 1,
    ) -> None:

        task_type = TaskType(task_type)
        supported_task_types = {  # Currently supported task types:
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.REGRESSION,
            TaskType.FORECASTING,
            TaskType.TEMPORAL_LINK_PREDICTION,
        }
        if task_type not in supported_task_types:
            supported = sorted(t.value for t in supported_task_types)
            raise ValueError(
                f"Task type '{task_type.value}' is not supported "
                f"by 'TaskTable' (got one of {supported})"
            )
        self._task_type = task_type

        # TODO: unify this frame validation with LocalTable.
        if context_df.empty:
            raise ValueError('No context examples given')
        self._context_df = context_df.copy(deep=False)

        if pred_df.empty:
            raise ValueError('Provide at least one entity to predict for')
        self._pred_df = pred_df.copy(deep=False)

        self._column_dict: dict[str, Column] = {}
        self.add_columns(context_df.columns.tolist())

        self._entity_table_names: tuple[str] | tuple[str, str]
        if isinstance(entity_table_name, str):
            self._entity_table_names = (entity_table_name,)
        elif len(entity_table_name) == 1:
            self._entity_table_names = (entity_table_name[0],)
        elif len(entity_table_name) == 2:
            self._entity_table_names = (
                entity_table_name[0],
                entity_table_name[1],
            )
        else:
            raise ValueError(
                f"'entity_table_name' must hold one entity table "
                f'name, or two for link prediction tasks (got '
                f'{len(entity_table_name)}: '
                f'{list(entity_table_name)})'
            )

        self._entity_column: str = ''
        self._target_column: str = ''
        self._time_column: str | None = None

        self.entity_column = entity_column
        self.target_column = target_column
        if time_column is not None:
            self.time_column = time_column

        self.step_size = step_size
        self.num_forecasts = num_forecasts

        self._query: str = ''  # A description of the task, e.g., for XAI.

    @property
    def num_context_examples(self) -> int:
        return len(self._context_df)

    @property
    def num_prediction_examples(self) -> int:
        return len(self._pred_df)

    @property
    def task_type(self) -> TaskType:
        r"""The task type."""
        return self._task_type

    def narrow_context(self, start: int, length: int) -> Self:
        r"""Returns a new :class:`TaskTable` that holds a narrowed version of
        context examples.

        Args:
            start: Index of the prediction examples to start narrowing.
            length: Length of the prediction examples.
        """
        out = copy.copy(self)
        df = out._context_df.iloc[start : start + length].reset_index(drop=True)
        out._context_df = df
        return out

    def narrow_prediction(self, start: int, length: int) -> Self:
        r"""Returns a new :class:`TaskTable` that holds a narrowed version of
        prediction examples.

        Args:
            start: Index of the prediction examples to start narrowing.
            length: Length of the prediction examples.
        """
        out = copy.copy(self)
        df = out._pred_df.iloc[start : start + length].reset_index(drop=True)
        out._pred_df = df
        return out

    # Entity column ###########################################################

    @property
    def entity_table_name(self) -> str:
        return self._entity_table_names[0]

    @property
    def entity_table_names(self) -> tuple[str] | tuple[str, str]:
        return self._entity_table_names

    @property
    def entity_column(self) -> Column:
        return self._column_dict[self._entity_column]

    @entity_column.setter
    def entity_column(self, name: str) -> None:
        self._column_dict[name].stype = Stype.ID
        self._entity_column = name

    # Target column ###########################################################

    @property
    def has_prediction_targets(self) -> bool:
        r"""Returns ``True`` if prediction rows include target values."""
        return self._target_column in self._pred_df

    @property
    def target_column(self) -> Column:
        return self._column_dict[self._target_column]

    @target_column.setter
    def target_column(self, name: str) -> None:
        self._column_dict[name].stype = _get_target_stype(self.task_type)
        self._target_column = name

    # Time column #############################################################

    def has_time_column(self) -> bool:
        r"""Returns ``True`` if this task has a time column; ``False``
        otherwise.
        """
        return self._time_column not in {None, self.ENTITY_TIME}

    @property
    def use_entity_time(self) -> bool:
        r"""Whether to use the timestamp of the entity table as anchor time."""
        return self._time_column == self.ENTITY_TIME

    @property
    def time_column(self) -> Column | None:
        r"""The time column of this task.

        The getter returns the time column of this task, or ``None`` if no
        such time column is present.

        The setter sets a column as a time column for this task, and raises a
        :class:`ValueError` if the time column has a non-timestamp compatible
        data type or if the column name does not match a column in the data
        frame.
        """
        if not self.has_time_column():
            return None
        assert self._time_column is not None
        return self._column_dict[self._time_column]

    @time_column.setter
    def time_column(self, name: str | None) -> None:
        if name is None or name == self.ENTITY_TIME:
            self._time_column = name
            return

        self._column_dict[name].stype = Stype.timestamp
        # Normalized per frame, before the two are concatenated into one anchor
        # series: a timezone-aware column and a timezone-naive one cannot be
        # concatenated, and `predict` hands back a timezone-aware
        # `ANCHOR_TIMESTAMP` that callers feed straight back in here.
        for df in (self._context_df, self._pred_df):
            if name in df.columns:
                df[name] = to_naive_utc(df[name])
        self._time_column = name

    # Metadata ################################################################

    def has_column(self, name: str) -> bool:
        r"""Returns ``True`` if this table holds a column with name ``name``;
        ``False`` otherwise.
        """
        return name in self._column_dict

    def column(self, name: str) -> Column:
        r"""Returns the data column named with name ``name`` in this table.

        Args:
            name: The name of the column.

        Raises:
            KeyError: If ``name`` is not present in this table.
        """
        if not self.has_column(name):
            raise KeyError(f"Column '{name}' not found in task table")
        return self._column_dict[name]

    @property
    def columns(self) -> list[Column]:
        r"""Returns a list of :class:`Column` objects that represent the
        columns in this table.
        """
        return list(self._column_dict.values())

    @property
    def feature_columns(self) -> list[Column]:
        r"""Returns a list of the feature columns in this task table."""
        feature_columns: list[Column] = []
        for column in self.columns:
            if column.name not in {
                self._entity_column,
                self._target_column,
                self._time_column,
            }:
                feature_columns.append(column)
        return feature_columns

    def add_columns(self, columns: Sequence[str]) -> None:
        r"""Adds a set of columns to this task.

        Args:
            columns: The columns to add.
        """
        for column_name in columns:
            if column_name in self._column_dict:
                raise KeyError(
                    f"Column '{column_name}' already exists in task table"
                )
            if column_name not in self._context_df.columns:
                raise KeyError(
                    f"Column '{column_name}' not found in task table"
                )

            ser = self._context_df[column_name]

            try:
                dtype = infer_dtype(ser)
            except Exception as e:
                raise RuntimeError(
                    f"Encountered unsupported data type '{ser.dtype}' for "
                    f"column '{column_name}' in task table. Please either "
                    f"manually override the columns's data type or remove the "
                    f'column from this table.'
                ) from e

            try:
                stype = infer_stype(ser, column_name, dtype)
            except Exception as e:
                raise RuntimeError(
                    f'Could not determine semantic type for column '
                    f"'{column_name}' with data type '{dtype}' in task "
                    f"table. Please either change the column's data type "
                    f'or remove the column from this table.'
                ) from e

            self._column_dict[column_name] = Column(
                name=column_name,
                expr=None,
                dtype=dtype,
                stype=stype,
            )

    @property
    def metadata(self) -> pd.DataFrame:
        raise NotImplementedError

    def print_metadata(self) -> None:
        raise NotImplementedError

    @property
    def _num_rows(self) -> int:
        return len(self._context_df) + len(self._pred_df)

    # Python builtins #########################################################

    def __contains__(self, name: str) -> bool:
        return self.has_column(name)

    def __getitem__(self, name: str) -> Column:
        return self.column(name)

    def __hash__(self) -> int:
        metadata = [
            self.task_type,
            self.entity_table_names,
            self._entity_column,
            self._target_column,
            self._time_column,
        ]
        return hash(tuple(self.columns + metadata))

    def __repr__(self) -> str:
        if self.task_type.is_link_pred:
            entity_table_repr = f'entity_table_names={self.entity_table_names}'
        else:
            entity_table_repr = f'entity_table_name={self.entity_table_name}'

        if self.use_entity_time:
            time_repr = 'use_entity_time=True'
        else:
            time_repr = f'time_column={self._time_column}'

        return (
            f'{self.__class__.__name__}(\n'
            f'  task_type={self.task_type},\n'
            f'  num_context_examples={self.num_context_examples},\n'
            f'  num_prediction_examples={self.num_prediction_examples},\n'
            f'  num_columns={len(self.columns)},\n'
            f'  {entity_table_repr},\n'
            f'  entity_column={self._entity_column},\n'
            f'  target_column={self._target_column},\n'
            f'  {time_repr},\n'
            f')'
        )


def _get_target_stype(task_type: TaskType) -> Stype:
    if task_type in {
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
    }:
        return Stype.categorical
    if task_type in {TaskType.REGRESSION, TaskType.FORECASTING}:
        return Stype.numerical
    if task_type.is_link_pred:
        return Stype.multicategorical
    raise ValueError(
        f'Cannot determine the semantic type of the target '
        f"column for task type '{task_type.value}'"
    )
