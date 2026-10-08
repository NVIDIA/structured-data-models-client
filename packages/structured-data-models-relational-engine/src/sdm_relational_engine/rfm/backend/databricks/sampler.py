# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from predictive_query.query import ValidatedPredictiveQuery
from predictive_query.schema.typing import Dtype

from sdm_relational_engine.rfm.backend.databricks import (
    DatabricksTable,
    dialect,
)
from sdm_relational_engine.rfm.backend.databricks.table import BACKTICK
from sdm_relational_engine.rfm.base import DataBackend, SQLSampler, Table
from sdm_relational_engine.rfm.base.source import Statement
from sdm_relational_engine.rfm.base.utils import Timestamp
from sdm_relational_engine.rfm.query.pandas_executor import PQueryPandasExecutor
from sdm_relational_engine.utils.progress_logger import ProgressLogger

if TYPE_CHECKING:
    from sdm_relational_engine.rfm import Graph

# We over-sample by this factor when drawing a random row sample, since
# `TABLESAMPLE (... PERCENT)` is a Bernoulli sample (the row count fluctuates)
# and subsequent `NOT NULL` filters may further reduce the number of rows:
_SAMPLE_OVERSAMPLE = 3.0


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

    def _bound_format(self, table_name: str, time_column: str) -> str:
        r"""How a window bound on ``time_column`` is written.

        The engine means a bound as UTC, and Databricks reads a bare one in
        the session's ``spark.sql.session.timeZone`` when it compares it with
        a ``TIMESTAMP`` (LTZ) column, so it is written with ``+00:00``, which
        ``TIMESTAMP_NTZ`` and ``DATE`` columns ignore. A ``STRING`` column
        compares with it as text, so its bound stays bare.
        """
        if self.table_dtype_dict[table_name].get(time_column) == Dtype.string:
            return '%Y-%m-%d %H:%M:%S'
        return '%Y-%m-%d %H:%M:%S+00:00'

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

    def _flatten(
        self, statement: Callable[[list], Statement], rows: list
    ) -> pa.Table:
        r"""Runs the ``statement`` for each chunk of ``rows``, which exposes
        the chunk through its first parameter, and stitches the results back
        into a single batch space via the ``__KUMO_BATCH__`` column.
        """
        chunks = dialect.chunk_rows(rows)

        if len(chunks) == 1:
            return self._run(statement(chunks[0]))

        tables: list[pa.Table] = []
        offset = 0
        for chunk in chunks:
            table = self._run(statement(chunk))
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

        # NOTE `TABLESAMPLE (n ROWS)` is implemented as `LIMIT` in Databricks
        # (*i.e.*, not random), so we draw a random percentage:
        percent: float | None = None
        if entity_ids is None:
            total = self.num_rows_dict.get(table_name)
            if total is not None and total > num_rows:
                percent = min(
                    100.0, 100.0 * num_rows / total * _SAMPLE_OVERSAMPLE
                )
            else:
                percent = 100.0

        # A specific set of entities: filter by a bound JSON parameter joined
        # via `posexplode` rather than interpolating the ids into the SQL, so
        # that string keys containing quotes are handled safely (same pattern
        # as `_by_pkey`).
        # NOTE This may return duplicate primary keys. This is okay.
        table = self._run(
            dialect.entity_rows(
                self.source_name_dict[table_name],
                projections,
                filters,
                key=self.table_column_ref_dict[table_name][key],
                order_by=self._rows_order_by(table_name, columns),
                percent=percent,
                random_seed=random_seed,
                num_rows=num_rows,
                ids=None if entity_ids is None else list(entity_ids),
                dtype=self.table_dtype_dict[table_name][key],
            )
        )

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
        dtype = self.table_dtype_dict[table_name][key]
        projections = self._projections(table_name, columns)

        # Of the rows holding an id, the first in a total order of the values
        # read is kept below, so the same row is kept however they are read.
        tie_break = self._tie_break_terms(table_name, columns)
        rows = [dialect.coerce_id(value, dtype) for value in index]
        table = self._flatten(
            lambda chunk: dialect.rows_by_key(
                self.source_name_dict[table_name],
                projections,
                key=key_ref,
                dtype=dtype,
                ids=chunk,
                order_by=tie_break,
            ),
            rows,
        )

        # Remove any duplicated primary keys in post-processing:
        tmp = table.append_column('__KUMO_ROW__', pa.array(range(len(table))))
        gb = tmp.group_by('__KUMO_BATCH__').aggregate(
            [('__KUMO_ROW__', 'min'), ('__KUMO_ROW__', 'count')]
        )
        table = table.take(gb['__KUMO_ROW___min'])

        batch = table['__KUMO_BATCH__'].cast(pa.int64()).to_numpy()
        self._refuse_an_id_scored_twice(
            table_name,
            index,
            batch,
            gb['__KUMO_ROW___count'].to_numpy(),
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
        dtype = self.table_dtype_dict[table_name][foreign_key]

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

            spec = self._bound_format(table_name, time_column)
            end_time = anchor_time.dt.strftime(spec)
            start_time = anchor_time - offset
            start_time = start_time.dt.strftime(spec)
            rows: list = [
                {
                    'id': dialect.coerce_id(i, dtype),
                    'e': e,
                    's': s,
                }
                for i, e, s in zip(index, end_time, start_time)
            ]
        else:
            rows = [dialect.coerce_id(i, dtype) for i in index]

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        projections = self._projections(table_name, columns)

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
        table = self._flatten(
            lambda chunk: dialect.neighbors(
                self.source_name_dict[table_name],
                projections,
                key=key_ref,
                dtype=dtype,
                rows=chunk,
                windowed=windowed,
                time=time_ref,
                bounds=bounds,
                order_by=order_by,
                num_neighbors=num_neighbors,
            ),
            rows,
        )

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
        if len(index) == 0:
            # The window's outer bounds are the latest and earliest of the
            # entities' times, which for none is NaN, and a warehouse refuses.
            return self._no_rows(table_name, columns)

        time_column = self.time_column_dict[table_name]
        dtype = self.table_dtype_dict[table_name][foreign_key]

        spec = self._bound_format(table_name, time_column)
        end_time = anchor_time + max_offset
        end_time = end_time.dt.strftime(spec)
        start_time: pd.Series | None = None
        if min_offset is not None:
            start_time = anchor_time + min_offset
            start_time = start_time.dt.strftime(spec)
            rows: list = [
                {
                    'id': dialect.coerce_id(i, dtype),
                    'e': e,
                    's': s,
                }
                for i, e, s in zip(index, end_time, start_time)
            ]
        else:
            rows = [
                {
                    'id': dialect.coerce_id(i, dtype),
                    'e': e,
                }
                for i, e in zip(index, end_time)
            ]

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        time_ref = self.table_column_ref_dict[table_name][time_column]
        projections = self._projections(table_name, columns)

        # Global time bounds enable data skipping, bound like every other
        # value rather than written into the SQL as a literal:
        bounds = [end_time.max()]
        if start_time is not None:
            bounds.append(start_time.min())
        # Aggregated in this order; see `_window_order_by`. Chunks are
        # stitched in batch order, so ordering each keeps the whole.
        order_by = self._window_order_by(table_name, time_ref, key_ref, columns)
        table = self._flatten(
            lambda chunk: dialect.window(
                self.source_name_dict[table_name],
                projections,
                key=key_ref,
                dtype=dtype,
                rows=chunk,
                with_start=start_time is not None,
                time=time_ref,
                bounds=bounds,
                order_by=order_by,
            ),
            rows,
        )

        batch = table['__KUMO_BATCH__'].cast(pa.int64()).to_numpy()
        batch_index = table.schema.get_field_index('__KUMO_BATCH__')
        table = table.remove_column(batch_index)

        return Table._sanitize(
            df=table.to_pandas(types_mapper=pd.ArrowDtype),
            dtype_dict=self.table_dtype_dict[table_name],
            stype_dict=self.table_stype_dict[table_name],
        ), batch
