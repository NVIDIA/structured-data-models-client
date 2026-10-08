# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections import defaultdict
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from predictive_query.query import ValidatedPredictiveQuery

from sdm_relational_engine.rfm.backend.duckdb import DuckDBTable, dialect
from sdm_relational_engine.rfm.base import DataBackend, SQLSampler, Table
from sdm_relational_engine.rfm.base.utils import Timestamp
from sdm_relational_engine.rfm.query.pandas_executor import PQueryPandasExecutor
from sdm_relational_engine.utils.progress_logger import ProgressLogger

if TYPE_CHECKING:
    from sdm_relational_engine.rfm import Graph


class DuckDBSampler(SQLSampler):
    def __init__(
        self,
        graph: 'Graph',
        verbose: bool | ProgressLogger = True,
        optimize: bool = False,
    ) -> None:
        super().__init__(graph=graph, verbose=verbose)

        self._graph_tables: dict[str, DuckDBTable] = {}
        for table in graph.tables.values():
            assert isinstance(table, DuckDBTable)
            self._graph_tables[table.name] = table

        # The engine only reads the database, so `optimize` no longer creates
        # the indices it used to; it only stops the missing ones being
        # reported.
        if not optimize:
            index_dict = self._get_missing_index_dict(graph=graph)
            if len(index_dict) > 0:
                self._warn_missing_indices(index_dict)

    @property
    def backend(self) -> DataBackend:
        return cast(DataBackend, DataBackend.DUCKDB)

    def _time_ref(self, table_name: str, column: str) -> str:
        r"""The time ``column`` of ``table_name`` as it is compared: its UTC
        wall-clock time, whatever the connection's ``TimeZone``.
        """
        table = self._graph_tables[table_name]
        return dialect.time_column(
            self.table_column_ref_dict[table_name][column],
            zoned=table._is_zoned_time(column),
        )

    def _get_missing_index_dict(self, graph: 'Graph') -> dict[str, set[str]]:
        # DuckDB ART indexes are only eligible for index scans when they are
        # single-column indexes. Keep this intentionally narrower than SQLite.
        index_dict: dict[str, set[str]] = defaultdict(set)
        for table_name, primary_key in self.primary_key_dict.items():
            source_table = self.source_table_dict[table_name]
            if primary_key not in source_table:
                continue  # No physical column.
            if source_table[primary_key].is_unique_key:
                continue
            index_dict[table_name].add(primary_key)

        for src_table_name, foreign_key, _ in graph.edges:
            source_table = self.source_table_dict[src_table_name]
            if foreign_key not in source_table:
                continue  # No physical column.
            if source_table[foreign_key].is_unique_key:
                continue
            index_dict[src_table_name].add(foreign_key)

        if len(index_dict) > 0:
            for source_name, expressions in self._rows(dialect.indices()):
                for table_name, table in graph.tables.items():
                    if source_name != table.source_name:
                        continue
                    column = self._indexed_column(expressions)
                    if column is not None:
                        index_dict[table_name].discard(column)
                        if len(index_dict[table_name]) == 0:
                            del index_dict[table_name]
                    break

        return index_dict

    @staticmethod
    def _indexed_column(expressions: str) -> str | None:
        expression = expressions.strip()
        if not expression.startswith('[') or not expression.endswith(']'):
            return None
        expression = expression[1:-1].strip()
        if ',' in expression:
            return None
        expression = expression.strip("'")
        if expression.startswith('"') and expression.endswith('"'):
            expression = expression[1:-1].replace('""', '"')
        return expression

    def _by_pkey(
        self,
        table_name: str,
        index: pd.Series,
        columns: set[str],
        scored: np.ndarray | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        key = self.primary_key_dict[table_name]
        key_ref = self.table_column_ref_dict[table_name][key]
        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]

        # Of the rows holding an id, the first in a total order of the values
        # read, so the same row is kept however they are stored.
        order_by = ', '.join(self._tie_break_terms(table_name, columns))
        order_by = order_by or key_ref
        table = self._run(
            dialect.rows_by_key(
                self.source_name_dict[table_name],
                key_ref,
                pa.array(index),
                projections,
                order_by=order_by,
            )
        )

        batch = table['__kumo_batch__'].to_numpy()
        self._refuse_an_id_scored_twice(
            table_name,
            index,
            batch,
            table['__kumo_matches__'].to_numpy(),
            scored,
        )
        for helper in ('__kumo_batch__', '__kumo_matches__'):
            table = table.remove_column(table.schema.get_field_index(helper))

        return Table._sanitize(
            df=table.to_pandas(types_mapper=pd.ArrowDtype),
            dtype_dict=self.table_dtype_dict[table_name],
            stype_dict=self.table_stype_dict[table_name],
        ), batch

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

        end_times: pa.Array | None = None
        if time_column is not None and anchor_time is not None:
            end_times = pa.array(anchor_time)

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]
        time_ref = (
            None
            if time_column is None
            else self._time_ref(table_name, time_column)
        )
        order_by = self._neighbor_order_by(
            table_name, time_ref, key_ref, columns
        )
        table = self._run(
            dialect.neighbors(
                self.source_name_dict[table_name],
                key_ref,
                pa.array(index),
                projections,
                num_neighbors=num_neighbors,
                order_by=order_by,
                time=time_ref,
                end_times=end_times,
            )
        )

        batch = table['__kumo_batch__'].to_numpy()
        batch_index = table.schema.get_field_index('__kumo_batch__')
        table = table.remove_column(batch_index)

        return Table._sanitize(
            df=table.to_pandas(types_mapper=pd.ArrowDtype),
            dtype_dict=self.table_dtype_dict[table_name],
            stype_dict=self.table_stype_dict[table_name],
        ), batch

    # Helper Methods ##########################################################

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

        end_times = pa.array(anchor_time + max_offset)
        start_times: pa.Array | None = None
        if min_offset is not None:
            start_times = pa.array(anchor_time + min_offset)

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        time_ref = self._time_ref(table_name, time_column)
        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]
        # Aggregated in this order; see `_window_order_by`.
        order_by = self._window_order_by(table_name, time_ref, key_ref, columns)
        table = self._run(
            dialect.window(
                self.source_name_dict[table_name],
                key_ref,
                pa.array(index),
                projections,
                time=time_ref,
                end_times=end_times,
                start_times=start_times,
                order_by=order_by,
            )
        )

        batch = table['__kumo_batch__'].to_numpy()
        batch_index = table.schema.get_field_index('__kumo_batch__')
        table = table.remove_column(batch_index)

        return Table._sanitize(
            df=table.to_pandas(types_mapper=pd.ArrowDtype),
            dtype_dict=self.table_dtype_dict[table_name],
            stype_dict=self.table_stype_dict[table_name],
        ), batch

    def _warn_missing_indices(
        self,
        index_dict: Mapping[str, Collection[object]],
    ) -> None:
        num = sum(len(indices) for indices in index_dict.values())
        index_repr = '1 index' if num == 1 else f'{num} indices'
        num = len(index_dict)
        table_repr = '1 table' if num == 1 else f'{num} tables'
        warnings.warn(
            f'Missing {index_repr} in {table_repr} for optimal '
            f'database querying. For improving runtime, we '
            f'strongly suggest to create indices for primary '
            f'and foreign keys, e.g., automatically by '
            f'instantiating Kumo Relational via '
            f'`KumoRelational(graph, optimize=True)`.'
        )

    def _get_min_max_time_dict(
        self,
        table_names: list[str],
    ) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        tables: list[tuple[str, str]] = []
        for table_name in table_names:
            column = self.time_column_dict[table_name]
            time_ref = self._time_ref(table_name, column)
            tables.append((self.source_name_dict[table_name], time_ref))

        out_dict: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        for index, _min, _max in self._rows(dialect.time_ranges(tables)):
            out_dict[table_names[int(index)]] = (
                pd.Timestamp.max if _min is None else Timestamp(_min),
                pd.Timestamp.min if _max is None else Timestamp(_max),
            )

        return out_dict

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
            key_ref = self.table_column_ref_dict[table_name][key]
            filters.append(f' {key_ref} IS NOT NULL')

        column = self.time_column_dict.get(table_name)
        if column is None:
            pass
        elif column not in source_table or source_table[column].is_nullable:
            column_ref = self.table_column_ref_dict[table_name][column]
            filters.append(f' {column_ref} IS NOT NULL')

        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]
        order_by = self._rows_order_by(table_name, columns)
        if entity_ids is None:
            statement = dialect.random_rows(
                self.source_name_dict[table_name],
                projections,
                filters,
                num_rows=num_rows,
                random_seed=random_seed,
                order_by=order_by,
            )
        else:
            # A specific set of entities: bind the ids as parameters rather
            # than interpolating them into the SQL, so that string keys
            # containing quotes are handled safely.
            statement = dialect.rows_by_ids(
                self.source_name_dict[table_name],
                projections,
                filters,
                key=self.table_column_ref_dict[table_name][key],
                ids=list(entity_ids),
                order_by=order_by,
            )
        table = self._run(statement)

        return Table._sanitize(
            df=table.to_pandas(types_mapper=pd.ArrowDtype),
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
        train_y, train_mask = self._sample_target_set(
            query=query,
            entity_df=entity_df,
            index=train_index,
            anchor_time=train_time,
            num_examples=num_train_examples,
            columns_dict=columns_dict,
            time_offset_dict=time_offset_dict,
        )

        test_y, test_mask = self._sample_target_set(
            query=query,
            entity_df=entity_df,
            index=test_index,
            anchor_time=test_time,
            num_examples=num_test_examples,
            columns_dict=columns_dict,
            time_offset_dict=time_offset_dict,
        )

        return train_y, train_mask, test_y, test_mask

    def _sample_target_set(
        self,
        query: ValidatedPredictiveQuery,
        entity_df: pd.DataFrame,
        index: np.ndarray,
        anchor_time: pd.Series,
        num_examples: int,
        columns_dict: dict[str, set[str]],
        time_offset_dict: dict[
            tuple[str, str, str],
            tuple[pd.DateOffset | None, pd.DateOffset],
        ],
        batch_size: int = 10_000,
    ) -> tuple[pd.Series, np.ndarray]:

        count = 0
        ys: list[pd.Series] = []
        mask = np.full(len(index), False, dtype=bool)
        for start in range(0, len(index), batch_size):
            df = entity_df.iloc[index[start : start + batch_size]]
            entity_pkey = df[self.primary_key_dict[query.entity_table]]
            feat_dict, time_dict, batch_dict = self._sample_query_data(
                entity_table_name=query.entity_table,
                entity_pkey=entity_pkey,
                anchor_time=anchor_time.iloc[start : start + batch_size],
                columns_dict={
                    key: columns
                    for key, columns in columns_dict.items()
                    if key != query.entity_table
                },
                time_offset_dict=time_offset_dict,
            )
            feat_dict[query.entity_table] = df
            time_column = self.time_column_dict.get(query.entity_table)
            if time_column in columns_dict[query.entity_table]:
                time_dict[query.entity_table] = df[time_column]

            y, _mask = PQueryPandasExecutor().execute_query(
                query=query,
                feat_dict=feat_dict,
                time_dict=time_dict,
                batch_dict=batch_dict,
                anchor_time=anchor_time.iloc[start : start + batch_size],
            )
            ys.append(y)
            mask[start : start + batch_size] = _mask

            count += len(y)
            if count >= num_examples:
                break

        if len(ys) == 0:
            y = pd.Series([], dtype=float)
        elif len(ys) == 1:
            y = ys[0]
        else:
            y = pd.concat(ys, axis=0, ignore_index=True)

        return y, mask
