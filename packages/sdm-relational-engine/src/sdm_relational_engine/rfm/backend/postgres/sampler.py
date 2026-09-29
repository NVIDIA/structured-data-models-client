# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from predictive_query.query import ValidatedPredictiveQuery
from predictive_query.schema.typing import Dtype

from sdm_relational_engine.rfm.backend.postgres import PostgresTable
from sdm_relational_engine.rfm.backend.postgres.table import (
    _frame,
    _reading,
    _sanitize_postgres_frame,
)
from sdm_relational_engine.rfm.base import DataBackend, SQLSampler
from sdm_relational_engine.rfm.base.utils import Timestamp
from sdm_relational_engine.rfm.query.pandas_executor import PQueryPandasExecutor
from sdm_relational_engine.utils.progress_logger import ProgressLogger

if TYPE_CHECKING:
    from sdm_relational_engine.rfm import Graph


def _json_default(obj: Any) -> Any:
    if hasattr(obj, 'item'):
        return obj.item()
    return str(obj)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(',', ':'), default=_json_default)


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
            self._connection = table._connection
        self._postgres_type_dict: dict[str, dict[str, str]] = {
            table.name: {
                column.name: table._postgres_type(column.name)
                or self._elem_type(column.dtype)
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

    @staticmethod
    def _elem_type(dtype: Dtype) -> str:
        if dtype.is_int():
            return 'bigint'
        if dtype.is_float():
            return 'double precision'
        return 'text'

    @staticmethod
    def _coerce_id(value: Any, dtype: Dtype, postgres_type: str) -> Any:
        if hasattr(value, 'item'):
            value = value.item()
        if dtype.is_int():
            return int(value)
        if dtype.is_float():
            if postgres_type.lower().startswith(('numeric', 'decimal')):
                return str(value)
            return float(value)
        return str(value)

    @staticmethod
    def _postgres_seed(random_seed: int) -> float:
        r"""Map an integer seed stably into PostgreSQL's ``[-1, 1]`` range.

        Hashing avoids the guaranteed collisions every 2,000,001 values that
        a direct modulo mapping would introduce. PostgreSQL accepts a double,
        so the final mapping necessarily has finite precision.
        """
        digest = hashlib.blake2b(
            str(random_seed).encode('ascii'), digest_size=8
        ).digest()
        value = int.from_bytes(digest, byteorder='big', signed=False)
        return 2.0 * value / ((1 << 64) - 1) - 1.0

    def _chunk_rows(
        self,
        rows: list,
        max_bytes: int = 900_000,
    ) -> list[list]:
        chunks: list[list] = []
        current: list = []
        current_bytes = 2
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

    def _flatten(self, sql: str, rows: list) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        offset = 0
        for chunk in self._chunk_rows(rows):
            with _reading(self._connection) as cursor:
                cursor.execute(sql, (_dumps(chunk),))
                frame = _frame(cursor)
            if '__KUMO_BATCH__' in frame:
                frame['__KUMO_BATCH__'] += offset
            frames.append(frame)
            offset += len(chunk)
        return (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )

    @staticmethod
    def _json_cte(
        postgres_type: str,
        fields: tuple[str, ...] = (),
        time_type: str = 'timestamp',
    ) -> str:
        select = ['(ordinality - 1)::bigint AS __KUMO_BATCH__']
        if fields:
            select.append(f"(value->>'id')::{postgres_type} AS __KUMO_ID__")
            if 'e' in fields:
                select.append(
                    f"(value->>'e')::{time_type} AS __KUMO_END_TIME__"
                )
            if 's' in fields:
                select.append(
                    f"(value->>'s')::{time_type} AS __KUMO_START_TIME__"
                )
        else:
            select.append(f"(value #>> '{{}}')::{postgres_type} AS __KUMO_ID__")
        return (
            'WITH TMP AS (\n'
            f'  SELECT {", ".join(select)}\n'
            '  FROM jsonb_array_elements(%s::jsonb) WITH ORDINALITY\n'
            '       AS input(value, ordinality)\n'
            ')\n'
        )

    def _postgres_type(self, table_name: str, column: str) -> str:
        return self._postgres_type_dict[table_name][column]

    def _tie_break_term(self, table_name: str, column: str, ref: str) -> str:
        # json and xml read as strings but cannot be ordered, and a domain can
        # hide either, so every string column is ordered by its text.
        if self.table_dtype_dict[table_name].get(column) == Dtype.string:
            return f'CAST({ref} AS text)'
        return ref

    @staticmethod
    def _serialize_time(value: Any) -> str | None:
        if pd.isna(value):
            return None
        return pd.Timestamp(value).isoformat()

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
        selects: list[str] = []
        for index, table_name in enumerate(table_names):
            column = self.time_column_dict[table_name]
            column_ref = self.table_column_ref_dict[table_name][column]
            selects.append(
                f'SELECT {index} AS table_index, MIN({column_ref}) AS '
                f'min_date, MAX({column_ref}) AS max_date\n'
                f'FROM {self.source_name_dict[table_name]}'
            )
        sql = '\nUNION ALL\n'.join(selects)
        out: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        with _reading(self._connection) as cursor:
            cursor.execute(sql)
            for index, minimum, maximum in cursor.fetchall():
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
        if entity_ids is None:
            sql = (
                f'SELECT {", ".join(projections)}\n'
                f'FROM {self.source_name_dict[table_name]}'
            )
            if filters:
                sql += f'\nWHERE {" AND ".join(filters)}'
            sql += f'\nORDER BY random()\nLIMIT {num_rows}'
            with _reading(self._connection) as cursor:
                if random_seed is not None:
                    cursor.execute(
                        'SELECT setseed(%s)',
                        (self._postgres_seed(random_seed),),
                    )
                cursor.execute(sql)
                frame = _frame(cursor)
        else:
            dtype = self.table_dtype_dict[table_name][key]
            postgres_type = self._postgres_type(table_name, key)
            key_ref = self.table_column_ref_dict[table_name][key]
            sql = self._json_cte(postgres_type)
            sql += (
                f'SELECT {", ".join(projections)}\n'
                'FROM TMP\n'
                f'JOIN {self.source_name_dict[table_name]}\n'
                f'  ON {key_ref} = TMP.__KUMO_ID__'
            )
            if filters:
                sql += f'\nWHERE {" AND ".join(filters)}'
            rows = [
                self._coerce_id(value, dtype, postgres_type)
                for value in entity_ids
            ]
            frame = self._flatten(sql, rows)

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
        dtype = self.table_dtype_dict[table_name][key]
        postgres_type = self._postgres_type(table_name, key)
        key_ref = self.table_column_ref_dict[table_name][key]
        projections = self._projections(table_name, columns)
        sql = self._json_cte(postgres_type)
        sql += (
            'SELECT * FROM (\n'
            '  SELECT TMP.__KUMO_BATCH__, '
            f'{", ".join(projections)},\n'
            '         ROW_NUMBER() OVER (PARTITION BY '
            'TMP.__KUMO_BATCH__\n'
            f'           ORDER BY {key_ref}) AS __KUMO_ROW__\n'
            '  FROM TMP\n'
            f'  JOIN {self.source_name_dict[table_name]}\n'
            f'    ON {key_ref} = TMP.__KUMO_ID__\n'
            ') AS sampled\n'
            'WHERE __KUMO_ROW__ = 1\n'
            'ORDER BY __KUMO_BATCH__'
        )
        rows = [self._coerce_id(value, dtype, postgres_type) for value in index]
        return self._sanitize_output(table_name, self._flatten(sql, rows))

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
            end_time = [self._serialize_time(value) for value in anchor_time]
            rows: list = [
                {
                    'id': self._coerce_id(i, dtype, postgres_type),
                    'e': e,
                }
                for i, e in zip(index, end_time)
            ]
            fields = ('e',)
        else:
            rows = [self._coerce_id(i, dtype, postgres_type) for i in index]

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        projections = self._projections(table_name, columns)
        time_ref = (
            None
            if time_column is None
            else self.table_column_ref_dict[table_name][time_column]
        )
        order_by = self._neighbor_order_by(table_name, time_ref, key_ref)
        time_type = (
            'timestamp'
            if time_column is None
            else self._postgres_type(table_name, time_column)
        )
        sql = self._json_cte(postgres_type, fields, time_type)
        sql += (
            'SELECT TMP.__KUMO_BATCH__, sampled.*\n'
            'FROM TMP\n'
            'JOIN LATERAL (\n'
            f'  SELECT {", ".join(projections)},\n'
            f'         ROW_NUMBER() OVER (ORDER BY {order_by}) '
            'AS __KUMO_ROW__\n'
            f'  FROM {self.source_name_dict[table_name]}\n'
            f'  WHERE {key_ref} = TMP.__KUMO_ID__\n'
        )
        if fields:
            assert time_ref is not None
            sql += f'    AND {time_ref} <= TMP.__KUMO_END_TIME__\n'
        sql += (
            f'  ORDER BY {order_by}\n'
            f'  LIMIT {num_neighbors}\n'
            ') AS sampled ON TRUE\n'
            'ORDER BY __KUMO_BATCH__, __KUMO_ROW__'
        )
        return self._sanitize_output(table_name, self._flatten(sql, rows))

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
        end_time = [
            self._serialize_time(value) for value in anchor_time + max_offset
        ]
        if min_offset is None:
            fields = ('e',)
            rows: list = [
                {
                    'id': self._coerce_id(i, dtype, postgres_type),
                    'e': e,
                }
                for i, e in zip(index, end_time)
            ]
        else:
            fields = ('e', 's')
            start_time = [
                self._serialize_time(value)
                for value in anchor_time + min_offset
            ]
            rows = [
                {
                    'id': self._coerce_id(i, dtype, postgres_type),
                    'e': e,
                    's': s,
                }
                for i, e, s in zip(index, end_time, start_time)
            ]

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        time_ref = self.table_column_ref_dict[table_name][time_column]
        projections = self._projections(table_name, columns)
        sql = self._json_cte(postgres_type, fields, time_type)
        sql += (
            'SELECT TMP.__KUMO_BATCH__, '
            f'{", ".join(projections)}\n'
            'FROM TMP\n'
            f'JOIN {self.source_name_dict[table_name]}\n'
            f'  ON {key_ref} = TMP.__KUMO_ID__\n'
            f' AND {time_ref} <= TMP.__KUMO_END_TIME__\n'
        )
        if min_offset is not None:
            sql += f' AND {time_ref} > TMP.__KUMO_START_TIME__\n'
        sql += f'ORDER BY TMP.__KUMO_BATCH__, {time_ref}'
        return self._sanitize_output(table_name, self._flatten(sql, rows))
