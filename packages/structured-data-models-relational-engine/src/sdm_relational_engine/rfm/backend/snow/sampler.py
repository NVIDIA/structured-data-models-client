# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from predictive_query.query import ValidatedPredictiveQuery

from sdm_relational_engine.rfm.backend.snow import SnowTable
from sdm_relational_engine.rfm.backend.snow.binding import paramstyle
from sdm_relational_engine.rfm.base import DataBackend, SQLSampler, Table
from sdm_relational_engine.rfm.base.utils import Timestamp
from sdm_relational_engine.rfm.query.pandas_executor import PQueryPandasExecutor
from sdm_relational_engine.utils.progress_logger import ProgressLogger

if TYPE_CHECKING:
    from sdm_relational_engine.rfm import Graph


class SnowSampler(SQLSampler):
    def __init__(
        self,
        graph: 'Graph',
        verbose: bool | ProgressLogger = True,
    ) -> None:
        super().__init__(graph=graph, verbose=verbose)

        for table in graph.tables.values():
            assert isinstance(table, SnowTable)
            self._connection = table._connection

        self._num_rows_dict: dict[str, int] = {
            table.name: cast(int, table._num_rows)
            for table in graph.tables.values()
        }

    @property
    def num_rows_dict(self) -> dict[str, int]:
        return self._num_rows_dict

    @property
    def backend(self) -> DataBackend:
        return cast(DataBackend, DataBackend.SNOWFLAKE)

    def _get_min_max_time_dict(
        self,
        table_names: list[str],
    ) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        selects: list[str] = []
        for index, table_name in enumerate(table_names):
            column = self.time_column_dict[table_name]
            column_ref = self.table_column_ref_dict[table_name][column]
            select = (
                f'SELECT\n'
                f'  {index} as table_index,\n'
                f'  MIN({column_ref}) as min_date,\n'
                f'  MAX({column_ref}) as max_date\n'
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
        # NOTE Snowflake does support `SEED` only as part of `SYSTEM` sampling,
        # which samples blocks rather than a fixed number of rows.
        if entity_ids is None:
            self._warn_random_seed_unsupported(random_seed)

        num_rows = min(num_rows, 1_000_000)  # Snowflake's upper limit.

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
        parameters: tuple | None = None
        if entity_ids is not None:
            key_ref = self.table_column_ref_dict[table_name][key]
            placeholders = ', '.join(['?'] * len(entity_ids))
            filters.append(f' {key_ref} IN ({placeholders})')
            parameters = tuple(entity_ids)

        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]
        if entity_ids is not None:
            sql = (
                f'SELECT {", ".join(projections)}\n'
                f'FROM {self.source_name_dict[table_name]}'
            )
        else:
            sql = (
                f'SELECT {", ".join(projections)}\n'
                f'FROM {self.source_name_dict[table_name]}\n'
                f'SAMPLE ROW ({num_rows} ROWS)'
            )
        if len(filters) > 0:
            sql += f'\nWHERE{" AND".join(filters)}'

        with paramstyle(self._connection), self._connection.cursor() as cursor:
            # NOTE This may return duplicate primary keys. This is okay.
            cursor.execute(sql, parameters)
            table = cursor.fetch_arrow_all(force_return_table=True)

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

        # NOTE For Snowflake, we execute everything at once to pay minimal
        # query initialization costs.
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

        y, mask = PQueryPandasExecutor().execute(
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
        key_ref = self.table_column_ref_dict[table_name][key]
        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]

        payload = json.dumps(list(index))

        sql = 'WITH TMP as (\n  SELECT\n    f.index as __KUMO_BATCH__,\n'
        if self.table_dtype_dict[table_name][key].is_int():
            sql += '    f.value::NUMBER as __KUMO_ID__\n'
        elif self.table_dtype_dict[table_name][key].is_float():
            sql += '    f.value::FLOAT as __KUMO_ID__\n'
        else:
            sql += '    f.value::VARCHAR as __KUMO_ID__\n'
        sql += (
            f'  FROM TABLE(FLATTEN(INPUT => PARSE_JSON(?))) f\n'
            f')\n'
            f'SELECT '
            f'TMP.__KUMO_BATCH__ as __KUMO_BATCH__, '
            f'{", ".join(projections)}\n'
            f'FROM TMP\n'
            f'JOIN {self.source_name_dict[table_name]}\n'
            f'  ON {key_ref} = TMP.__KUMO_ID__'
        )

        with paramstyle(self._connection), self._connection.cursor() as cursor:
            cursor.execute(sql, (payload,))
            table = cursor.fetch_arrow_all(force_return_table=True)

        # Remove any duplicated primary keys in post-processing:
        tmp = table.append_column('__KUMO_ID__', pa.array(range(len(table))))
        gb = tmp.group_by('__KUMO_BATCH__').aggregate(
            [('__KUMO_ID__', 'min'), ('__KUMO_ID__', 'count')]
        )
        table = table.take(gb['__KUMO_ID___min'])

        batch = table['__KUMO_BATCH__'].cast(pa.int64()).to_numpy()
        self._refuse_an_id_scored_twice(
            table_name,
            index,
            batch,
            gb['__KUMO_ID___count'].to_numpy(),
            scored,
        )
        batch_index = table.schema.get_field_index('__KUMO_BATCH__')
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

        end_time: pd.Series | None = None
        start_time: pd.Series | None = None
        if time_column is not None and anchor_time is not None:
            # In order to avoid a full table scan, we limit foreign key
            # sampling to a certain time range, approximated by the number of
            # rows, timestamp ranges and `num_neighbors` value.
            # Downstream, this helps Snowflake to apply partition pruning:
            dst_table_name = [
                dst_table
                for key, dst_table in self.foreign_key_dict[table_name]
                if key == foreign_key
            ][0]
            num_facts = self.num_rows_dict[table_name]
            num_entities = self.num_rows_dict[dst_table_name]
            min_time = self.get_min_time([table_name])
            max_time = self.get_max_time([table_name])
            freq = num_facts / num_entities
            freq = freq / max((max_time - min_time).total_seconds(), 1)
            # Look up at most 5 years of history (and prevent out-of-bounds):
            seconds = 5 * 365 * 24 * 60 * 60
            seconds = min(math.ceil(5 * num_neighbors / freq), seconds)
            offset = pd.Timedelta(seconds=seconds)

            end_time = anchor_time.dt.strftime('%Y-%m-%d %H:%M:%S')
            start_time = anchor_time - offset
            start_time = start_time.dt.strftime('%Y-%m-%d %H:%M:%S')
            payload = json.dumps(list(zip(index, end_time, start_time)))
        else:
            payload = json.dumps(list(zip(index)))

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]

        sql = 'WITH TMP as (\n  SELECT\n    f.index as __KUMO_BATCH__,\n'
        if self.table_dtype_dict[table_name][foreign_key].is_int():
            sql += '    f.value[0]::NUMBER as __KUMO_ID__'
        elif self.table_dtype_dict[table_name][foreign_key].is_float():
            sql += '    f.value[0]::FLOAT as __KUMO_ID__'
        else:
            sql += '    f.value[0]::VARCHAR as __KUMO_ID__'
        if end_time is not None and start_time is not None:
            sql += (
                ',\n'
                '    f.value[1]::TIMESTAMP_NTZ as __KUMO_END_TIME__,\n'
                '    f.value[2]::TIMESTAMP_NTZ as __KUMO_START_TIME__'
            )
        sql += (
            f'\n'
            f'  FROM TABLE(FLATTEN(INPUT => PARSE_JSON(?))) f\n'
            f')\n'
            f'SELECT '
            f'TMP.__KUMO_BATCH__ as __KUMO_BATCH__, '
            f'{", ".join(projections)}\n'
            f'FROM TMP\n'
            f'JOIN {self.source_name_dict[table_name]}\n'
            f'  ON {key_ref} = TMP.__KUMO_ID__\n'
        )
        bounds: list[str] = []
        if end_time is not None and start_time is not None:
            assert time_column is not None
            time_ref = self.table_column_ref_dict[table_name][time_column]
            sql += (
                f' AND {time_ref} <= TMP.__KUMO_END_TIME__\n'
                f' AND {time_ref} > TMP.__KUMO_START_TIME__\n'
                f'WHERE {time_ref} <= ?\n'
                f'  AND {time_ref} > ?\n'
            )
            bounds = [end_time.max(), start_time.min()]
        sql += (
            'QUALIFY ROW_NUMBER() OVER (\n  PARTITION BY TMP.__KUMO_BATCH__\n'
        )
        time_ref = (
            None
            if time_column is None
            else self.table_column_ref_dict[table_name][time_column]
        )
        order_by = self._neighbor_order_by(table_name, time_ref, key_ref)
        sql += f'  ORDER BY {order_by}\n'
        sql += f') <= {num_neighbors}\n'
        # Returned in that order too, so the rows reach the request the same
        # way however the warehouse reads them.
        sql += f'ORDER BY TMP.__KUMO_BATCH__, {order_by}'

        with paramstyle(self._connection), self._connection.cursor() as cursor:
            cursor.execute(sql, (payload, *bounds))
            table = cursor.fetch_arrow_all(force_return_table=True)

        batch = table['__KUMO_BATCH__'].cast(pa.int64()).to_numpy()
        batch_index = table.schema.get_field_index('__KUMO_BATCH__')
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
        if len(index) == 0:
            # The window's outer bounds are the latest and earliest of the
            # entities' times, which for none is NaN, and a warehouse refuses.
            return self._no_rows(table_name, columns)

        time_column = self.time_column_dict[table_name]

        end_time = anchor_time + max_offset
        end_time = end_time.dt.strftime('%Y-%m-%d %H:%M:%S')
        start_time: pd.Series | None = None
        if min_offset is not None:
            start_time = anchor_time + min_offset
            start_time = start_time.dt.strftime('%Y-%m-%d %H:%M:%S')
            payload = json.dumps(list(zip(index, end_time, start_time)))
        else:
            payload = json.dumps(list(zip(index, end_time)))

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        time_ref = self.table_column_ref_dict[table_name][time_column]
        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]
        sql = 'WITH TMP as (\n  SELECT\n    f.index as __KUMO_BATCH__,\n'
        if self.table_dtype_dict[table_name][foreign_key].is_int():
            sql += '    f.value[0]::NUMBER as __KUMO_ID__,\n'
        elif self.table_dtype_dict[table_name][foreign_key].is_float():
            sql += '    f.value[0]::FLOAT as __KUMO_ID__,\n'
        else:
            sql += '    f.value[0]::VARCHAR as __KUMO_ID__,\n'
        sql += '    f.value[1]::TIMESTAMP_NTZ as __KUMO_END_TIME__'
        if min_offset is not None:
            sql += ',\n    f.value[2]::TIMESTAMP_NTZ as __KUMO_START_TIME__'
        sql += (
            f'\n'
            f'  FROM TABLE(FLATTEN(INPUT => PARSE_JSON(?))) f\n'
            f')\n'
            f'SELECT '
            f'TMP.__KUMO_BATCH__ as __KUMO_BATCH__, '
            f'{", ".join(projections)}\n'
            f'FROM TMP\n'
            f'JOIN {self.source_name_dict[table_name]}\n'
            f'  ON {key_ref} = TMP.__KUMO_ID__\n'
            f' AND {time_ref} <= TMP.__KUMO_END_TIME__\n'
        )
        if start_time is not None:
            sql += f'AND {time_ref} > TMP.__KUMO_START_TIME__\n'
        # Add global time bounds to enable partition pruning, bound like
        # every other value rather than written into the SQL as a literal:
        sql += f'WHERE {time_ref} <= ?'
        bounds = [end_time.max()]
        if start_time is not None:
            sql += f'\nAND {time_ref} > ?'
            bounds.append(start_time.min())

        with paramstyle(self._connection), self._connection.cursor() as cursor:
            cursor.execute(sql, (payload, *bounds))
            table = cursor.fetch_arrow_all(force_return_table=True)

        batch = table['__KUMO_BATCH__'].cast(pa.int64()).to_numpy()
        batch_index = table.schema.get_field_index('__KUMO_BATCH__')
        table = table.remove_column(batch_index)

        return Table._sanitize(
            df=table.to_pandas(types_mapper=pd.ArrowDtype),
            dtype_dict=self.table_dtype_dict[table_name],
            stype_dict=self.table_stype_dict[table_name],
        ), batch
