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
from sdm_connectors import quote_ident

from sdm_relational_engine.rfm.backend.duckdb import DuckDBTable
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

        for table in graph.tables.values():
            assert isinstance(table, DuckDBTable)
            self._connection = table._connection

        index_dict = self._get_missing_index_dict(graph=graph)
        if optimize:
            self._optimize(index_dict=index_dict)
        elif len(index_dict) > 0:
            self._warn_missing_indices(index_dict)

    @property
    def backend(self) -> DataBackend:
        return cast(DataBackend, DataBackend.DUCKDB)

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
            with self._connection.cursor() as cursor:
                cursor.execute(
                    'SELECT table_name, expressions FROM duckdb_indexes()'
                )
                for source_name, expressions in cursor.fetchall():
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

    def _optimize(self, index_dict: dict[str, set[str]]) -> None:
        if len(index_dict) == 0:
            return

        with self._connection.cursor() as cursor:
            for table_name, columns in index_dict.items():
                source_name = self.source_name_dict[table_name]
                for column in columns:
                    name = f'kumo_index_{table_name}_{column}'
                    sql = (
                        f'CREATE INDEX IF NOT EXISTS {quote_ident(name)}\n'
                        f'ON {source_name}({quote_ident(column)})'
                    )
                    cursor.execute(sql)
            self._connection.commit()

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

    def _random_sample_sql(
        self,
        table_name: str,
        projections: list[str],
        filters: list[str],
        num_rows: int,
        random_seed: int | None = None,
    ) -> str:
        # NOTE DuckDB's sampling avoids SQLite's ORDER BY RANDOM() full sort.
        sql = f'SELECT {", ".join(projections)}\n'
        sql += f'FROM (SELECT * FROM {self.source_name_dict[table_name]}'
        if len(filters) > 0:
            sql += f'\nWHERE{" AND".join(filters)}'
        sql += f')\nUSING SAMPLE reservoir({num_rows} ROWS)'
        if random_seed is not None:
            sql += f' REPEATABLE ({random_seed})'
        return sql

    def _by_pkey(
        self,
        table_name: str,
        index: pd.Series,
        columns: set[str],
    ) -> tuple[pd.DataFrame, np.ndarray]:
        key = self.primary_key_dict[table_name]
        key_ref = self.table_column_ref_dict[table_name][key]
        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]

        tmp = self._key_table(index)
        tmp_name = f'tmp_{table_name}_{key}_{id(tmp)}'

        sql = (
            f'SELECT '
            f'tmp.__kumo_batch__, '
            f'{", ".join(projections)}\n'
            f'FROM {quote_ident(tmp_name)} tmp\n'
            f'JOIN {self.source_name_dict[table_name]}\n'
            f'  ON {key_ref} = tmp.__kumo_id__\n'
            f'QUALIFY ROW_NUMBER() OVER (\n'
            f'  PARTITION BY tmp.__kumo_batch__\n'
            f') = 1'
        )

        with self._connection.cursor() as cursor:
            cursor.register(tmp_name, tmp)
            cursor.execute(sql)
            table = cursor.to_arrow_table()

        batch = table['__kumo_batch__'].to_numpy()
        batch_index = table.schema.get_field_index('__kumo_batch__')
        table = table.remove_column(batch_index)

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

        tmp = self._key_table(index)
        if time_column is not None and anchor_time is not None:
            tmp = tmp.append_column('__kumo_time__', pa.array(anchor_time))
        tmp_name = f'tmp_{table_name}_{foreign_key}_{id(tmp)}'

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]
        sql = (
            f'SELECT '
            f'tmp.__kumo_batch__, '
            f'{", ".join(projections)}\n'
            f'FROM {quote_ident(tmp_name)} tmp\n'
            f'JOIN {self.source_name_dict[table_name]}\n'
            f'  ON {key_ref} = tmp.__kumo_id__\n'
        )
        if time_column is not None and anchor_time is not None:
            time_ref = self._time_ref(table_name, time_column)
            sql += f' AND {time_ref} <= tmp.__kumo_time__\n'
        sql += (
            'QUALIFY ROW_NUMBER() OVER (\n  PARTITION BY tmp.__kumo_batch__\n'
        )
        time_ref = (
            None
            if time_column is None
            else self._time_ref(table_name, time_column)
        )
        order_by = self._neighbor_order_by(table_name, time_ref, key_ref)
        sql += f'  ORDER BY {order_by}\n'
        sql += f') <= {num_neighbors}'

        with self._connection.cursor() as cursor:
            cursor.register(tmp_name, tmp)
            cursor.execute(sql)
            table = cursor.to_arrow_table()

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

        tmp = self._key_table(index)
        end_time = anchor_time + max_offset
        tmp = tmp.append_column('__kumo_end__', pa.array(end_time))
        if min_offset is not None:
            start_time = anchor_time + min_offset
            tmp = tmp.append_column('__kumo_start__', pa.array(start_time))
        tmp_name = f'tmp_{table_name}_{foreign_key}_{id(tmp)}'

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        time_ref = self._time_ref(table_name, time_column)
        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]
        sql = (
            f'SELECT '
            f'tmp.__kumo_batch__, '
            f'{", ".join(projections)}\n'
            f'FROM {quote_ident(tmp_name)} tmp\n'
            f'JOIN {self.source_name_dict[table_name]}\n'
            f'  ON {key_ref} = tmp.__kumo_id__\n'
            f' AND {time_ref} <= tmp.__kumo_end__'
        )
        if min_offset is not None:
            sql += f'\n AND {time_ref} > tmp.__kumo_start__'

        with self._connection.cursor() as cursor:
            cursor.register(tmp_name, tmp)
            cursor.execute(sql)
            table = cursor.to_arrow_table()

        batch = table['__kumo_batch__'].to_numpy()
        batch_index = table.schema.get_field_index('__kumo_batch__')
        table = table.remove_column(batch_index)

        return Table._sanitize(
            df=table.to_pandas(types_mapper=pd.ArrowDtype),
            dtype_dict=self.table_dtype_dict[table_name],
            stype_dict=self.table_stype_dict[table_name],
        ), batch

    def _key_table(self, index: pd.Series) -> pa.Table:
        return pa.table(
            {
                '__kumo_batch__': pa.array(range(len(index)), type=pa.int64()),
                '__kumo_id__': pa.array(index),
            }
        )

    def _time_ref(self, table_name: str, column: str) -> str:
        column_ref = self.table_column_ref_dict[table_name][column]
        return f'CAST({column_ref} AS TIMESTAMP)'

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
        selects: list[str] = []
        for index, table_name in enumerate(table_names):
            column = self.time_column_dict[table_name]
            time_ref = self._time_ref(table_name, column)
            select = (
                f'SELECT\n'
                f'  {index} as table_index,\n'
                f'  MIN({time_ref}) as min_date,\n'
                f'  MAX({time_ref}) as max_date\n'
                f'FROM {self.source_name_dict[table_name]}'
            )
            selects.append(select)
        sql = '\nUNION ALL\n'.join(selects)

        out_dict: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        with self._connection.cursor() as cursor:
            cursor.execute(sql)
            for index, _min, _max in cursor.fetchall():
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

        # A specific set of entities: bind the ids as parameters rather than
        # interpolating them into the SQL, so that string keys containing
        # quotes are handled safely (same pattern as `_by_pkey`).
        parameters: list | None = None
        if entity_ids is not None:
            key_ref = self.table_column_ref_dict[table_name][key]
            placeholders = ', '.join(['?'] * len(entity_ids))
            filters.append(f' {key_ref} IN ({placeholders})')
            parameters = list(entity_ids)

        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]
        if entity_ids is None:
            sql = self._random_sample_sql(
                table_name=table_name,
                projections=projections,
                filters=filters,
                num_rows=num_rows,
                random_seed=random_seed,
            )
        else:
            sql = self._entity_select_sql(
                table_name=table_name,
                projections=projections,
                filters=filters,
            )

        with self._connection.cursor() as cursor:
            cursor.execute(sql, parameters)
            table = cursor.to_arrow_table()

        return Table._sanitize(
            df=table.to_pandas(types_mapper=pd.ArrowDtype),
            dtype_dict=self.table_dtype_dict[table_name],
            stype_dict=self.table_stype_dict[table_name],
        )

    def _entity_select_sql(
        self,
        table_name: str,
        projections: list[str],
        filters: list[str],
    ) -> str:
        sql = (
            f'SELECT {", ".join(projections)}\n'
            f'FROM {self.source_name_dict[table_name]}'
        )
        if len(filters) > 0:
            sql += f'\nWHERE{" AND".join(filters)}'
        return sql

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

            y, _mask = PQueryPandasExecutor().execute(
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
