# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from predictive_query.query import ValidatedPredictiveQuery
from predictive_query.schema.typing import Dtype

from sdm_relational_engine.rfm.backend.databricks import DatabricksTable
from sdm_relational_engine.rfm.backend.databricks.table import BACKTICK
from sdm_relational_engine.rfm.base import DataBackend, SQLSampler, Table
from sdm_relational_engine.rfm.base.utils import Timestamp
from sdm_relational_engine.rfm.query.pandas_executor import PQueryPandasExecutor
from sdm_relational_engine.utils.progress_logger import ProgressLogger

if TYPE_CHECKING:
    from sdm_relational_engine.rfm import Graph

# We over-sample by this factor when drawing a random row sample, since
# `TABLESAMPLE (... PERCENT)` is a Bernoulli sample (the row count fluctuates)
# and subsequent `NOT NULL` filters may further reduce the number of rows:
_SAMPLE_OVERSAMPLE = 3.0


def _json_default(obj: Any) -> Any:
    # `numpy`/`pandas` scalars are not JSON-serializable out-of-the-box:
    if hasattr(obj, 'item'):
        return obj.item()
    return str(obj)


def _dumps(obj: Any) -> str:
    # Serialize with no whitespace so `_chunk_rows`' per-row size accounting
    # (which adds one byte for the single element separator) matches the
    # payload bytes produced here exactly; the default `json.dumps` inserts a
    # ``', '`` separator, undercounting the payload by one byte per row.
    return json.dumps(obj, separators=(',', ':'), default=_json_default)


class DatabricksSampler(SQLSampler):
    # Databricks quotes SQL identifiers with backticks rather than double
    # quotes (which it interprets as string literals):
    _QUOTE_CHAR: str = BACKTICK

    def __init__(
        self,
        graph: 'Graph',
        verbose: bool | ProgressLogger = True,
    ) -> None:
        super().__init__(graph=graph, verbose=verbose)

        for table in graph.tables.values():
            assert isinstance(table, DatabricksTable)
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
        return cast(DataBackend, DataBackend.DATABRICKS)

    # Helper Methods ##########################################################

    def _projections(self, table_name: str, columns: Any) -> list[str]:
        r"""The SELECT list for ``columns``, in the table's declared order.

        Driven by the insertion-ordered projection dict, not by ``columns``,
        which is a ``set``: iterating the set put column order at the mercy of
        ``PYTHONHASHSEED``, and the model is order-sensitive, so one graph
        predicted anywhere between 117 and 237 for a fixed ``random_seed``.
        """
        proj_dict = self.table_column_proj_dict[table_name]
        wanted = set(columns)
        missing = wanted.difference(proj_dict)
        if missing:
            raise KeyError(
                f'Unknown columns requested for table {table_name!r}: '
                f'{", ".join(sorted(missing))}'
            )
        return [proj for column, proj in proj_dict.items() if column in wanted]

    @staticmethod
    def _elem_type(dtype: Dtype) -> str:
        r"""The Databricks SQL type used to deserialize a key from JSON."""
        if dtype.is_int():
            return 'bigint'
        if dtype.is_float():
            return 'double'
        return 'string'

    @staticmethod
    def _coerce_id(value: Any, dtype: Dtype) -> Any:
        r"""Coerces a key to a native Python type matching ``dtype`` so that it
        round-trips through JSON into the declared Databricks SQL type.
        """
        if dtype.is_int():
            return int(value)
        if dtype.is_float():
            return float(value)
        return str(value)

    def _chunk_rows(
        self,
        rows: list,
        max_bytes: int = 900_000,
    ) -> list[list]:
        r"""Splits ``rows`` into chunks whose JSON payload stays below the
        Databricks native parameter size limit (1 MB).
        """
        chunks: list[list] = []
        current: list = []
        current_bytes = 2  # Account for the enclosing '[]'.
        for row in rows:
            size = len(_dumps(row)) + 1
            if current and current_bytes + size > max_bytes:
                chunks.append(current)
                current, current_bytes = [], 2
            current.append(row)
            current_bytes += size
        if current:
            chunks.append(current)
        return chunks or [[]]

    def _flatten(self, sql: str, rows: list) -> pa.Table:
        r"""Executes ``sql`` (which exposes the ``rows`` payload via a single
        ``?`` parameter) once per chunk and stitches the results back into a
        single batch space via the ``__KUMO_BATCH__`` column.
        """
        chunks = self._chunk_rows(rows)

        if len(chunks) == 1:
            payload = _dumps(chunks[0])
            with self._connection.cursor() as cursor:
                cursor.execute(sql, parameters=[payload])
                return cursor.fetchall_arrow()

        tables: list[pa.Table] = []
        offset = 0
        for chunk in chunks:
            payload = _dumps(chunk)
            with self._connection.cursor() as cursor:
                cursor.execute(sql, parameters=[payload])
                table = cursor.fetchall_arrow()
            # Shift the per-chunk batch index into the global batch space:
            batch = table['__KUMO_BATCH__'].cast(pa.int64())
            batch = batch.to_numpy(zero_copy_only=False) + offset
            index = table.schema.get_field_index('__KUMO_BATCH__')
            table = table.set_column(index, '__KUMO_BATCH__', pa.array(batch))
            tables.append(table)
            offset += len(chunk)
        return pa.concat_tables(tables)

    # Sampling ################################################################

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

        projections = self._projections(table_name, columns)

        # A specific set of entities: filter by a bound JSON parameter joined
        # via `posexplode` rather than interpolating the ids into the SQL, so
        # that string keys containing quotes are handled safely (same pattern
        # as `_by_pkey`).
        parameters: list | None = None
        join = ''
        if entity_ids is not None:
            key_ref = self.table_column_ref_dict[table_name][key]
            dtype = self.table_dtype_dict[table_name][key]
            join = (
                f'\nJOIN posexplode(from_json(?, '
                f"'array<{self._elem_type(dtype)}>')) AS __ids__"
                f'(__pos__, __KUMO_ID__)\n'
                f'  ON {key_ref} = __ids__.__KUMO_ID__'
            )
            parameters = [
                _dumps([self._coerce_id(value, dtype) for value in entity_ids])
            ]

        sql = (
            f'SELECT {", ".join(projections)}\n'
            f'FROM {self.source_name_dict[table_name]}'
        )

        if entity_ids is None:
            # NOTE `TABLESAMPLE (n ROWS)` is implemented as `LIMIT` in
            # Databricks (*i.e.*, not random), so we draw a random percentage:
            total = self.num_rows_dict.get(table_name)
            if total is not None and total > num_rows:
                percent = min(
                    100.0, 100.0 * num_rows / total * _SAMPLE_OVERSAMPLE
                )
            else:
                percent = 100.0
            sql += f'\nTABLESAMPLE ({percent} PERCENT)'
            if random_seed is not None:
                sql += f' REPEATABLE ({random_seed})'

        sql += join

        if len(filters) > 0:
            sql += f'\nWHERE{" AND".join(filters)}'

        if entity_ids is None:
            sql += f'\nLIMIT {num_rows}'

        with self._connection.cursor() as cursor:
            # NOTE This may return duplicate primary keys. This is okay.
            cursor.execute(sql, parameters=parameters)
            table = cursor.fetchall_arrow()

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

        # NOTE As for Snowflake, we execute everything at once to pay minimal
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
    ) -> tuple[pd.DataFrame, np.ndarray]:

        if len(index) == 0:
            return pd.DataFrame(), np.empty(0, dtype=int)

        key = self.primary_key_dict[table_name]
        key_ref = self.table_column_ref_dict[table_name][key]
        dtype = self.table_dtype_dict[table_name][key]
        projections = self._projections(table_name, columns)

        sql = (
            f'WITH TMP AS (\n'
            f'  SELECT pos AS __KUMO_BATCH__, col AS __KUMO_ID__\n'
            f"  FROM posexplode(from_json(?, 'array<{self._elem_type(dtype)}"
            f">'))\n"
            f')\n'
            f'SELECT '
            f'TMP.__KUMO_BATCH__ as __KUMO_BATCH__, '
            f'{", ".join(projections)}\n'
            f'FROM TMP\n'
            f'JOIN {self.source_name_dict[table_name]}\n'
            f'  ON {key_ref} = TMP.__KUMO_ID__'
        )

        rows = [self._coerce_id(value, dtype) for value in index]
        table = self._flatten(sql, rows)

        # Remove any duplicated primary keys in post-processing:
        tmp = table.append_column('__KUMO_ROW__', pa.array(range(len(table))))
        gb = tmp.group_by('__KUMO_BATCH__').aggregate([('__KUMO_ROW__', 'min')])
        table = table.take(gb['__KUMO_ROW___min'])

        batch = table['__KUMO_BATCH__'].cast(pa.int64()).to_numpy()
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
        dtype = self.table_dtype_dict[table_name][foreign_key]
        elem = self._elem_type(dtype)

        end_time: pd.Series | None = None
        start_time: pd.Series | None = None
        if time_column is not None and anchor_time is not None:
            # In order to avoid a full table scan, we limit foreign key
            # sampling to a certain time range, approximated by the number of
            # rows, timestamp ranges and `num_neighbors` value.
            # Downstream, this helps Databricks to apply data skipping:
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
            rows: list = [
                {
                    'id': self._coerce_id(i, dtype),
                    'e': e,
                    's': s,
                }
                for i, e, s in zip(index, end_time, start_time)
            ]
            schema = f'array<struct<id:{elem},e:string,s:string>>'
        else:
            rows = [self._coerce_id(i, dtype) for i in index]
            schema = f'array<{elem}>'

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        projections = self._projections(table_name, columns)

        select = ['pos AS __KUMO_BATCH__']
        if end_time is not None and start_time is not None:
            select.append('col.id AS __KUMO_ID__')
            select.append('col.e AS __KUMO_END_TIME__')
            select.append('col.s AS __KUMO_START_TIME__')
        else:
            select.append('col AS __KUMO_ID__')
        sql = (
            f'WITH TMP AS (\n'
            f'  SELECT {", ".join(select)}\n'
            f"  FROM posexplode(from_json(?, '{schema}'))\n"
            f')\n'
            f'SELECT '
            f'TMP.__KUMO_BATCH__ as __KUMO_BATCH__, '
            f'{", ".join(projections)}\n'
            f'FROM TMP\n'
            f'JOIN {self.source_name_dict[table_name]}\n'
            f'  ON {key_ref} = TMP.__KUMO_ID__\n'
        )
        if end_time is not None and start_time is not None:
            assert time_column is not None
            time_ref = self.table_column_ref_dict[table_name][time_column]
            sql += (
                f' AND {time_ref} <= TMP.__KUMO_END_TIME__\n'
                f' AND {time_ref} > TMP.__KUMO_START_TIME__\n'
                f"WHERE {time_ref} <= '{end_time.max()}'\n"
                f"  AND {time_ref} > '{start_time.min()}'\n"
            )
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
        # Returned in that order too. Without it the rows came back in the
        # order each engine read them, which serverless Spark does differently
        # from a SQL warehouse, and the request -- and the answer -- differed.
        # Chunks are stitched in batch order, so ordering each keeps the whole.
        sql += f'ORDER BY TMP.__KUMO_BATCH__, {order_by}'

        table = self._flatten(sql, rows)

        batch = table['__KUMO_BATCH__'].cast(pa.int64()).to_numpy()
        batch_index = table.schema.get_field_index('__KUMO_BATCH__')
        table = table.remove_column(batch_index)

        return Table._sanitize(
            df=table.to_pandas(types_mapper=pd.ArrowDtype),
            dtype_dict=self.table_dtype_dict[table_name],
            stype_dict=self.table_stype_dict[table_name],
        ), batch

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
        elem = self._elem_type(dtype)

        end_time = anchor_time + max_offset
        end_time = end_time.dt.strftime('%Y-%m-%d %H:%M:%S')
        start_time: pd.Series | None = None
        if min_offset is not None:
            start_time = anchor_time + min_offset
            start_time = start_time.dt.strftime('%Y-%m-%d %H:%M:%S')
            rows: list = [
                {
                    'id': self._coerce_id(i, dtype),
                    'e': e,
                    's': s,
                }
                for i, e, s in zip(index, end_time, start_time)
            ]
            schema = f'array<struct<id:{elem},e:string,s:string>>'
        else:
            rows = [
                {
                    'id': self._coerce_id(i, dtype),
                    'e': e,
                }
                for i, e in zip(index, end_time)
            ]
            schema = f'array<struct<id:{elem},e:string>>'

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        time_ref = self.table_column_ref_dict[table_name][time_column]
        projections = self._projections(table_name, columns)

        select = [
            'pos AS __KUMO_BATCH__',
            'col.id AS __KUMO_ID__',
            'col.e AS __KUMO_END_TIME__',
        ]
        if min_offset is not None:
            select.append('col.s AS __KUMO_START_TIME__')
        sql = (
            f'WITH TMP AS (\n'
            f'  SELECT {", ".join(select)}\n'
            f"  FROM posexplode(from_json(?, '{schema}'))\n"
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
        # Add global time bounds to enable data skipping:
        sql += f"WHERE {time_ref} <= '{end_time.max()}'"
        if start_time is not None:
            sql += f"\nAND {time_ref} > '{start_time.min()}'"

        table = self._flatten(sql, rows)

        batch = table['__KUMO_BATCH__'].cast(pa.int64()).to_numpy()
        batch_index = table.schema.get_field_index('__KUMO_BATCH__')
        table = table.remove_column(batch_index)

        return Table._sanitize(
            df=table.to_pandas(types_mapper=pd.ArrowDtype),
            dtype_dict=self.table_dtype_dict[table_name],
            stype_dict=self.table_stype_dict[table_name],
        ), batch
