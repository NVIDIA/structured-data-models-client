# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import cast

import pandas as pd

from sdm_relational_engine.rfm.base import (
    Column,
    ColumnSpec,
    DataBackend,
    SourceColumn,
    SourceForeignKey,
    Table,
    composite_key,
)
from sdm_relational_engine.runmode import MissingType


class LocalTable(Table):
    r"""A table backed by a :class:`pandas.DataFrame`.

    A :class:`LocalTable` fully specifies the relevant metadata, *i.e.*
    selected columns, column semantic types, primary keys and time columns.
    :class:`LocalTable` is used to create a :class:`Graph`.

    .. code-block:: python

        import pandas as pd
        import sdm_relational_engine.rfm as rfm

        # Load data from a CSV file:
        df = pd.read_csv("data.csv")

        # Create a table from a `pandas.DataFrame` and infer its metadata ...
        table = rfm.LocalTable(df, name="my_table").infer_metadata()

        # ... or create a table explicitly:
        table = rfm.LocalTable(
            df=df,
            name="my_table",
            primary_key="id",
            time_column="time",
            end_time_column=None,
        )

        # Verify metadata:
        table.print_metadata()

        # Change the semantic type of a column:
        table[column].stype = "text"

    Args:
        df: The data frame to create this table from.
        name: The name of this table.
        primary_key: The name of the primary key of this table, if it exists.
        time_column: The name of the time column of this table, if it exists.
        end_time_column: The name of the end time column of this table, if it
            exists.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        name: str,
        primary_key: MissingType | str | None = MissingType.VALUE,
        time_column: str | None = None,
        end_time_column: str | None = None,
    ) -> None:

        if df.empty:
            raise ValueError('Data frame is empty')
        if isinstance(df.columns, pd.MultiIndex):
            raise ValueError('Data frame must not have a multi-index')
        if not df.columns.is_unique:
            raise ValueError('Data frame must have unique column names')
        if any(col == '' for col in df.columns):
            raise ValueError('Data frame must have non-empty column names')

        self._data = df.copy(deep=False)

        super().__init__(
            name=name,
            primary_key=primary_key,
            time_column=time_column,
            end_time_column=end_time_column,
        )

    @property
    def backend(self) -> DataBackend:
        return cast(DataBackend, DataBackend.LOCAL)

    def _local_row_count(self) -> int:
        return len(self._data)

    def _get_source_columns(self) -> list[SourceColumn]:
        return [
            SourceColumn(
                name=column_name,
                dtype=None,
                is_primary_key=False,
                is_unique_key=False,
                is_nullable=True,
            )
            for column_name in self._data.columns
        ]

    def _get_source_foreign_keys(self) -> list[SourceForeignKey]:
        return []

    def _get_source_sample_df(self) -> pd.DataFrame:
        return self._data

    def _materialize_derived_key(
        self,
        derived: str,
        names: Sequence[str],
    ) -> None:
        self._data[derived] = composite_key.encode_frame(
            self._data, list(names)
        )
        self.__dict__.pop('_source_column_dict', None)
        self.__dict__.pop('_source_sample_df', None)
        if not self.has_column(derived):
            self.add_column(derived)

    def _forget_derived_key(self, derived: str) -> None:
        if derived in self._data.columns:
            self._data = self._data.drop(columns=[derived])
        self.__dict__.pop('_source_column_dict', None)
        self.__dict__.pop('_source_sample_df', None)

    def _get_expr_sample_df(
        self,
        columns: Sequence[ColumnSpec | Column],
    ) -> pd.DataFrame:
        raise RuntimeError(
            f'Column expressions are not supported in '
            f"'{self.__class__.__name__}'. Please apply your "
            f'expressions on the `pd.DataFrame` directly.'
        )

    def _get_num_rows(self) -> int | None:
        return len(self._data)
