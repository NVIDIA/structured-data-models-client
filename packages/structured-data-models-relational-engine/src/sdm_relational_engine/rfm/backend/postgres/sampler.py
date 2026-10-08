# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from predictive_query.query import ValidatedPredictiveQuery
from predictive_query.schema.typing import Dtype

from sdm_relational_engine.rfm.backend.postgres import PostgresTable, dialect
from sdm_relational_engine.rfm.backend.postgres.table import (
    _frame,
    _rows,
    _sanitize_postgres_frame,
)
from sdm_relational_engine.rfm.base import DataBackend, SQLSampler
from sdm_relational_engine.rfm.base.source import Statement
from sdm_relational_engine.rfm.base.utils import Timestamp
from sdm_relational_engine.rfm.query.pandas_executor import PQueryPandasExecutor
from sdm_relational_engine.utils.progress_logger import ProgressLogger

if TYPE_CHECKING:
    from sdm_relational_engine.rfm import Graph


class PostgresSampler(SQLSampler):
    r"""Relational sampler using PostgreSQL-native SQL.

    Filtering, temporal predicates, ordering, and neighbor limits execute in
    PostgreSQL. Pandas holds only rows returned by those queries. ``_by_time``
    intentionally returns every row in the requested target window because
    truncating that result would change the predictive-query label.
    """

    def __init__(
        self,
        graph: 'Graph',
        verbose: bool | ProgressLogger = True,
    ) -> None:
        super().__init__(graph=graph, verbose=verbose)
        tables: list[PostgresTable] = []
        for table in graph.tables.values():
            assert isinstance(table, PostgresTable)
            tables.append(table)
        self._postgres_type_dict: dict[str, dict[str, str]] = {
            table.name: {
                column.name: table._postgres_type(column.name)
                or dialect.element_type(column.dtype)
                for column in table.columns
            }
            for table in tables
        }
        self._num_rows_dict: dict[str, int] = {
            table.name: cast(int, table._num_rows)
            for table in graph.tables.values()
        }

    @property
    def num_rows_dict(self) -> dict[str, int]:
        return self._num_rows_dict

    @property
    def backend(self) -> DataBackend:
        return DataBackend.POSTGRES

    def _projections(self, table_name: str, columns: Any) -> list[str]:
        proj_dict = self.table_column_proj_dict[table_name]
        wanted = set(columns)
        missing = wanted.difference(proj_dict)
        if missing:
            raise KeyError(
                f'Unknown columns requested for table {table_name!r}: '
                f'{", ".join(sorted(missing))}'
            )
        return [proj for column, proj in proj_dict.items() if column in wanted]

    def _rows(self, statement: Statement) -> list[tuple]:
        return _rows(self._run(statement))

    def _flatten(
        self, statement: Callable[[list], Statement], rows: list
    ) -> pd.DataFrame:
        r"""The rows the ``statement`` for each chunk of ``rows`` reads,
        with ``__KUMO_BATCH__`` counted over all of ``rows``.
        """
        frames: list[pd.DataFrame] = []
        offset = 0
        for chunk in dialect.chunk_rows(rows):
            frame = _frame(self._run(statement(chunk)))
            if '__KUMO_BATCH__' in frame:
                frame['__KUMO_BATCH__'] += offset
            frames.append(frame)
            offset += len(chunk)
        return (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )

    def _postgres_type(self, table_name: str, column: str) -> str:
        return self._postgres_type_dict[table_name][column]

    def _utc_bounds(self, table_name: str, time_column: str) -> bool:
        r"""Whether bounds on ``time_column`` are written as UTC instants:
        unless it holds text, which compares with them as text.
        """
        dtype = self.table_dtype_dict[table_name].get(time_column)
        return dtype != Dtype.string

    def _tie_break_term(self, table_name: str, column: str, ref: str) -> str:
        is_string = (
            self.table_dtype_dict[table_name].get(column) == Dtype.string
        )
        return dialect.tie_break_term(ref, is_string)

    def _sanitize_output(
        self, table_name: str, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray]:
        batch = frame.pop('__KUMO_BATCH__').to_numpy(dtype=np.int64)
        for helper in ('__KUMO_ROW__',):
            if helper in frame:
                frame.pop(helper)
        return _sanitize_postgres_frame(
            df=frame,
            dtype_dict=self.table_dtype_dict[table_name],
            stype_dict=self.table_stype_dict[table_name],
        ), batch

    def _get_min_max_time_dict(
        self,
        table_names: list[str],
    ) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        tables: list[tuple[str, str]] = []
        for table_name in table_names:
            column = self.time_column_dict[table_name]
            column_ref = self.table_column_ref_dict[table_name][column]
            tables.append((self.source_name_dict[table_name], column_ref))
        out: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        for index, minimum, maximum in self._rows(dialect.time_ranges(tables)):
            out[table_names[int(index)]] = (
                pd.Timestamp.max if minimum is None else Timestamp(minimum),
                pd.Timestamp.min if maximum is None else Timestamp(maximum),
            )
        return out

    def _sample_entity_table(
        self,
        table_name: str,
        columns: set[str],
        num_rows: int,
        random_seed: int | None = None,
        entity_ids: list | None = None,
    ) -> pd.DataFrame:
        source_table = self.source_table_dict[table_name]
        filters: list[str] = []
        key = self.primary_key_dict[table_name]
        if key not in source_table or source_table[key].is_nullable:
            filters.append(
                f'{self.table_column_ref_dict[table_name][key]} IS NOT NULL'
            )
        time_column = self.time_column_dict.get(table_name)
        if time_column is not None and (
            time_column not in source_table
            or source_table[time_column].is_nullable
        ):
            filters.append(
                f'{self.table_column_ref_dict[table_name][time_column]} '
                'IS NOT NULL'
            )

        projections = self._projections(table_name, columns)
        source = self.source_name_dict[table_name]
        if entity_ids is None:
            if random_seed is not None:
                self._run(dialect.set_seed(random_seed))
            frame = _frame(
                self._run(
                    dialect.random_rows(source, projections, filters, num_rows)
                )
            )
        else:
            dtype = self.table_dtype_dict[table_name][key]
            postgres_type = self._postgres_type(table_name, key)
            key_ref = self.table_column_ref_dict[table_name][key]
            order_by = self._rows_order_by(table_name, columns)
            rows = [
                dialect.coerce_id(value, dtype, postgres_type)
                for value in entity_ids
            ]
            frame = self._flatten(
                lambda chunk: dialect.rows_by_ids(
                    source,
                    projections,
                    filters,
                    key=key_ref,
                    postgres_type=postgres_type,
                    ids=chunk,
                    order_by=order_by,
                ),
                rows,
            )

        return _sanitize_postgres_frame(
            df=frame,
            dtype_dict=self.table_dtype_dict[table_name],
            stype_dict=self.table_stype_dict[table_name],
        )

    def _sample_target(
        self,
        query: ValidatedPredictiveQuery,
        entity_df: pd.DataFrame,
        train_index: np.ndarray,
        train_time: pd.Series,
        num_train_examples: int,
        test_index: np.ndarray,
        test_time: pd.Series,
        num_test_examples: int,
        columns_dict: dict[str, set[str]],
        time_offset_dict: dict[
            tuple[str, str, str],
            tuple[pd.DateOffset | None, pd.DateOffset],
        ],
    ) -> tuple[pd.Series, np.ndarray, pd.Series, np.ndarray]:
        del num_train_examples, num_test_examples
        index = np.concatenate([train_index, test_index])
        entity_df = entity_df.iloc[index].reset_index(drop=True)
        time = pd.concat([train_time, test_time], axis=0, ignore_index=True)
        feat_dict, time_dict, batch_dict = self._sample_query_data(
            entity_table_name=query.entity_table,
            entity_pkey=entity_df[self.primary_key_dict[query.entity_table]],
            anchor_time=time,
            columns_dict={
                key: columns
                for key, columns in columns_dict.items()
                if key != query.entity_table
            },
            time_offset_dict=time_offset_dict,
        )
        feat_dict[query.entity_table] = entity_df
        time_column = self.time_column_dict.get(query.entity_table)
        if time_column in columns_dict[query.entity_table]:
            time_dict[query.entity_table] = entity_df[time_column]
        y, mask = PQueryPandasExecutor().execute_query(
            query=query,
            feat_dict=feat_dict,
            time_dict=time_dict,
            batch_dict=batch_dict,
            anchor_time=time,
        )
        train_mask = mask[: len(train_index)]
        test_mask = mask[len(train_index) :]
        boundary = int(train_mask.sum())
        train_y = y.iloc[:boundary]
        test_y = y.iloc[boundary:].reset_index(drop=True)
        return train_y, train_mask, test_y, test_mask

    def _by_pkey(
        self,
        table_name: str,
        index: pd.Series,
        columns: set[str],
        scored: np.ndarray | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        if len(index) == 0:
            return pd.DataFrame(), np.empty(0, dtype=int)
        key = self.primary_key_dict[table_name]
        dtype = self.table_dtype_dict[table_name][key]
        postgres_type = self._postgres_type(table_name, key)
        key_ref = self.table_column_ref_dict[table_name][key]
        projections = self._projections(table_name, columns)
        # Of the rows holding an id, the first in a total order of the values
        # read, so the same row is kept however they are stored.
        order_by = ', '.join(self._tie_break_terms(table_name, columns))
        order_by = order_by or key_ref
        rows = [
            dialect.coerce_id(value, dtype, postgres_type) for value in index
        ]
        frame = self._flatten(
            lambda chunk: dialect.rows_by_key(
                self.source_name_dict[table_name],
                projections,
                key=key_ref,
                postgres_type=postgres_type,
                ids=chunk,
                order_by=order_by,
            ),
            rows,
        )
        if '__KUMO_MATCHES__' in frame:
            matches = frame.pop('__KUMO_MATCHES__').to_numpy(dtype=np.int64)
            self._refuse_an_id_scored_twice(
                table_name,
                index,
                frame['__KUMO_BATCH__'].to_numpy(dtype=np.int64),
                matches,
                scored,
            )
        return self._sanitize_output(table_name, frame)

    def _by_fkey(
        self,
        table_name: str,
        foreign_key: str,
        index: pd.Series,
        num_neighbors: int,
        anchor_time: pd.Series | None,
        columns: set[str],
    ) -> tuple[pd.DataFrame, np.ndarray]:
        time_column = self.time_column_dict.get(table_name)
        dtype = self.table_dtype_dict[table_name][foreign_key]
        postgres_type = self._postgres_type(table_name, foreign_key)
        fields: tuple[str, ...] = ()
        if time_column is not None and anchor_time is not None:
            utc = self._utc_bounds(table_name, time_column)
            end_time = [
                dialect.serialize_time(value, utc) for value in anchor_time
            ]
            rows: list = [
                {
                    'id': dialect.coerce_id(i, dtype, postgres_type),
                    'e': e,
                }
                for i, e in zip(index, end_time)
            ]
            fields = ('e',)
        else:
            rows = [dialect.coerce_id(i, dtype, postgres_type) for i in index]

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        projections = self._projections(table_name, columns)
        time_ref = (
            None
            if time_column is None
            else self.table_column_ref_dict[table_name][time_column]
        )
        order_by = self._neighbor_order_by(
            table_name, time_ref, key_ref, columns
        )
        time_type = (
            'timestamp'
            if time_column is None
            else self._postgres_type(table_name, time_column)
        )
        frame = self._flatten(
            lambda chunk: dialect.neighbors(
                self.source_name_dict[table_name],
                projections,
                key=key_ref,
                postgres_type=postgres_type,
                rows=chunk,
                fields=fields,
                time=time_ref,
                time_type=time_type,
                order_by=order_by,
                num_neighbors=num_neighbors,
            ),
            rows,
        )
        return self._sanitize_output(table_name, frame)

    def _by_time(
        self,
        table_name: str,
        foreign_key: str,
        index: pd.Series,
        anchor_time: pd.Series,
        min_offset: pd.DateOffset | None,
        max_offset: pd.DateOffset,
        columns: set[str],
    ) -> tuple[pd.DataFrame, np.ndarray]:
        time_column = self.time_column_dict[table_name]
        dtype = self.table_dtype_dict[table_name][foreign_key]
        postgres_type = self._postgres_type(table_name, foreign_key)
        time_type = self._postgres_type(table_name, time_column)
        fields: tuple[str, ...]
        utc = self._utc_bounds(table_name, time_column)
        end_time = [
            dialect.serialize_time(value, utc)
            for value in anchor_time + max_offset
        ]
        if min_offset is None:
            fields = ('e',)
            rows: list = [
                {
                    'id': dialect.coerce_id(i, dtype, postgres_type),
                    'e': e,
                }
                for i, e in zip(index, end_time)
            ]
        else:
            fields = ('e', 's')
            start_time = [
                dialect.serialize_time(value, utc)
                for value in anchor_time + min_offset
            ]
            rows = [
                {
                    'id': dialect.coerce_id(i, dtype, postgres_type),
                    'e': e,
                    's': s,
                }
                for i, e, s in zip(index, end_time, start_time)
            ]

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        time_ref = self.table_column_ref_dict[table_name][time_column]
        projections = self._projections(table_name, columns)
        # Aggregated in this order; the time alone left equal times in any
        # order. See `_window_order_by`.
        order_by = self._window_order_by(table_name, time_ref, key_ref, columns)
        frame = self._flatten(
            lambda chunk: dialect.window(
                self.source_name_dict[table_name],
                projections,
                key=key_ref,
                postgres_type=postgres_type,
                rows=chunk,
                fields=fields,
                time=time_ref,
                time_type=time_type,
                order_by=order_by,
            ),
            rows,
        )
        return self._sanitize_output(table_name, frame)
