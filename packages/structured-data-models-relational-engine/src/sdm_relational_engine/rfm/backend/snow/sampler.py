# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from predictive_query.query import ValidatedPredictiveQuery

from sdm_relational_engine.rfm.backend.snow import SnowTable, dialect
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

        self._num_rows_dict: dict[str, int] = {
            table.name: cast(int, table._num_rows)
            for table in graph.tables.values()
        }

        self._snowflake_type_dict: dict[str, dict[str, str]] = {
            table.name: cast(SnowTable, table)._source_snowflake_type_dict
            for table in graph.tables.values()
        }
        self._table_dict: dict[str, SnowTable] = {
            table.name: cast(SnowTable, table)
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
        tables: list[tuple[str, str]] = []
        for table_name in table_names:
            column = self.time_column_dict[table_name]
            column_ref = self.table_column_ref_dict[table_name][column]
            tables.append((self.source_name_dict[table_name], column_ref))

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
        num_rows = min(num_rows, dialect.MAX_SAMPLE_ROWS)

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
        seeded = None
        if entity_ids is None and random_seed is not None:
            seeded = self._seeded_key_hash(table_name, key, random_seed)
        # A seeded draw from a large table hashes the rows of a seeded sample
        # of its micro-partitions rather than every row; if those hold fewer
        # rows than the draw needs, it draws from every row instead.
        prefilter = None
        if seeded is not None:
            assert random_seed is not None
            prefilter = self._table_dict[table_name]._prefilter(
                max(SnowTable._SAMPLE_PREFILTER_ROWS, 10 * num_rows),
                seed=random_seed,
            )

        def draw(prefilter: str | None) -> pa.Table:
            # A specific set of entities: bind the ids as parameters rather
            # than interpolating them into the SQL, so that string keys
            # containing quotes are handled safely (same pattern as
            # `_by_pkey`).
            # NOTE This may return duplicate primary keys. This is okay.
            return self._run(
                dialect.entity_rows(
                    self.source_name_dict[table_name],
                    projections,
                    filters,
                    order_by=order_by,
                    num_rows=num_rows,
                    seeded=seeded,
                    key=self.table_column_ref_dict[table_name][key],
                    ids=None if entity_ids is None else list(entity_ids),
                    prefilter=prefilter,
                )
            )

        table = draw(prefilter)
        if prefilter is not None and table.num_rows < num_rows:
            table = draw(None)

        return Table._sanitize(
            df=table.to_pandas(types_mapper=pd.ArrowDtype),
            dtype_dict=self.table_dtype_dict[table_name],
            stype_dict=self.table_stype_dict[table_name],
        )

    def _seeded_key_hash(
        self,
        table_name: str,
        key: str,
        random_seed: int,
    ) -> str:
        r"""A hash of the seed and a row's key, the same in every session.

        ``TO_VARCHAR`` renders dates, times and binary values as the session's
        ``*_OUTPUT_FORMAT`` and ``TIMEZONE`` say, so hashing that text would
        draw different rows from one seed under different settings. Each key
        is rendered in a form no session parameter changes: strings as they
        are, numbers and booleans by ``TO_VARCHAR``, which no parameter
        formats, dates and times with an explicit format, timestamps as
        nanoseconds since the epoch and binary values as hex. A ``DATE`` is
        told from a timestamp by its source type, as ``DATE_PART`` gives no
        ``EPOCH_NANOSECOND`` of a ``DATE``; a key computed by an expression
        has none, and is taken to be a timestamp.

        Args:
            table_name: The table being sampled.
            key: Its primary key.
            random_seed: The seed of the draw.
        """
        key_ref = self.table_column_ref_dict[table_name][key]
        source = self.source_table_dict[table_name].get(key)
        dtype = source.dtype if source is not None else None
        if dtype is None:
            dtype = self.table_dtype_dict[table_name].get(key)
        return dialect.seeded_key_hash(
            key_ref, dtype, self._is_date(table_name, key), random_seed
        )

    def _zone(self, table_name: str, column: str) -> str | None:
        r"""``LTZ`` or ``TZ`` if ``column`` of ``table_name`` holds
        ``TIMESTAMP_LTZ`` or ``TIMESTAMP_TZ`` values, else ``None``.

        A source column says so in its type. A column computed by an
        expression has no declared type, so the type its inference sample was
        read as says it is zoned, though not which zone type it is; its bounds
        are read as ``TIMESTAMP_TZ``, which either compares with as the
        instant it names. One not yet sampled is taken to be naive, as
        before.
        """
        if column not in self.source_table_dict[table_name]:
            if column in self._table_dict[table_name]._zoned_expr_columns:
                return 'TZ'
            return None
        source_types = self._snowflake_type_dict.get(table_name, {})
        return dialect.zone_of(source_types.get(column))

    def _bound_format(self, table_name: str, column: str) -> str:
        r"""How a window bound on ``column`` is written: with ``+00:00`` for
        a zoned column, as :func:`dialect.window` reads it, and bare for any
        other, as before.
        """
        if self._zone(table_name, column) is None:
            return dialect.NAIVE_BOUND_FORMAT
        return dialect.ZONED_BOUND_FORMAT

    def _is_date(self, table_name: str, column: str) -> bool:
        r"""Whether ``column`` of ``table_name`` is a source ``DATE`` column.

        Args:
            table_name: The table.
            column: The column.
        """
        if column not in self.source_table_dict[table_name]:
            return False
        source_types = self._snowflake_type_dict.get(table_name, {})
        return source_types.get(column) == 'DATE'

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
        key_ref = self.table_column_ref_dict[table_name][key]
        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]

        # Of the rows holding an id, the first in a total order of the values
        # read is kept below, so the same row is kept however they are read.
        table = self._run(
            dialect.rows_by_key(
                self.source_name_dict[table_name],
                projections,
                key=key_ref,
                dtype=self.table_dtype_dict[table_name][key],
                ids=list(index),
                order_by=self._tie_break_terms(table_name, columns),
            )
        )

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

            spec = self._bound_format(table_name, time_column)
            end_time = anchor_time.dt.strftime(spec)
            start_time = anchor_time - offset
            start_time = start_time.dt.strftime(spec)
            rows = list(zip(index, end_time, start_time))
        else:
            rows = list(zip(index))

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]

        windowed = end_time is not None and start_time is not None
        bounds: list[str] = []
        if windowed:
            assert end_time is not None and start_time is not None
            bounds = [end_time.max(), start_time.min()]
        time_ref = (
            None
            if time_column is None
            else self.table_column_ref_dict[table_name][time_column]
        )
        order_by = self._neighbor_order_by(
            table_name, time_ref, key_ref, columns
        )
        table = self._run(
            dialect.neighbors(
                self.source_name_dict[table_name],
                projections,
                key=key_ref,
                dtype=self.table_dtype_dict[table_name][foreign_key],
                rows=rows,
                windowed=windowed,
                time=time_ref,
                bounds=bounds,
                order_by=order_by,
                num_neighbors=num_neighbors,
                zone=(
                    None
                    if time_column is None
                    else self._zone(table_name, time_column)
                ),
            )
        )

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

        spec = self._bound_format(table_name, time_column)
        end_time = anchor_time + max_offset
        end_time = end_time.dt.strftime(spec)
        start_time: pd.Series | None = None
        if min_offset is not None:
            start_time = anchor_time + min_offset
            start_time = start_time.dt.strftime(spec)
            rows = list(zip(index, end_time, start_time))
        else:
            rows = list(zip(index, end_time))

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        time_ref = self.table_column_ref_dict[table_name][time_column]
        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]
        # Global time bounds enable partition pruning, bound like every other
        # value rather than written into the SQL as a literal:
        bounds = [end_time.max()]
        if start_time is not None:
            bounds.append(start_time.min())
        # Aggregated in this order; see `_window_order_by`.
        order_by = self._window_order_by(table_name, time_ref, key_ref, columns)
        table = self._run(
            dialect.window(
                self.source_name_dict[table_name],
                projections,
                key=key_ref,
                dtype=self.table_dtype_dict[table_name][foreign_key],
                rows=rows,
                with_start=start_time is not None,
                time=time_ref,
                bounds=bounds,
                order_by=order_by,
                zone=self._zone(table_name, time_column),
            )
        )

        batch = table['__KUMO_BATCH__'].cast(pa.int64()).to_numpy()
        batch_index = table.schema.get_field_index('__KUMO_BATCH__')
        table = table.remove_column(batch_index)

        return Table._sanitize(
            df=table.to_pandas(types_mapper=pd.ArrowDtype),
            dtype_dict=self.table_dtype_dict[table_name],
            stype_dict=self.table_stype_dict[table_name],
        ), batch
