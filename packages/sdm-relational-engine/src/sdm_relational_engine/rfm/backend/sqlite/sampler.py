# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from predictive_query.query import ValidatedPredictiveQuery

from sdm_relational_engine.rfm.backend.sqlite import SQLiteTable
from sdm_relational_engine.rfm.base import DataBackend, SQLSampler, Table
from sdm_relational_engine.rfm.base.utils import Timestamp
from sdm_relational_engine.rfm.pquery import PQueryPandasExecutor
from sdm_relational_engine.utils import ProgressLogger, quote_ident

if TYPE_CHECKING:
    from sdm_relational_engine.rfm import Graph


class SQLiteSampler(SQLSampler):
    def __init__(
        self,
        graph: 'Graph',
        verbose: bool | ProgressLogger = True,
        optimize: bool = False,
    ) -> None:
        super().__init__(graph=graph, verbose=verbose)

        for table in graph.tables.values():
            assert isinstance(table, SQLiteTable)
            self._connection = table._connection

        # Per table, decided once on first use by `_time_separator`. Held on
        # the instance rather than in a `functools.cache`, which would key on
        # `self` and so keep every sampler -- and its graph and its open
        # connection -- alive for the life of the process.
        self._time_separator_dict: dict[str, str] = {}

        if optimize:
            with self._connection.cursor() as cursor:
                cursor.execute('PRAGMA temp_store = MEMORY')
                cursor.execute('PRAGMA cache_size = -2000000')  # 2 GB

        # Collect database indices for speeding sampling:
        index_dict: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        for table_name, primary_key in self.primary_key_dict.items():
            source_table = self.source_table_dict[table_name]
            if primary_key not in source_table:
                continue  # No physical column.
            if source_table[primary_key].is_unique_key:
                continue
            index_dict[table_name].add((primary_key,))
        for src_table_name, foreign_key, _ in graph.edges:
            source_table = self.source_table_dict[src_table_name]
            if foreign_key not in source_table:
                continue  # No physical column.
            if source_table[foreign_key].is_unique_key:
                continue
            time_column = self.time_column_dict.get(src_table_name)
            if time_column is not None and time_column in source_table:
                index_dict[src_table_name].add((foreign_key, time_column))
            else:
                index_dict[src_table_name].add((foreign_key,))

        # Only maintain missing indices:
        with self._connection.cursor() as cursor:
            for table_name in list(index_dict.keys()):
                indices = index_dict[table_name]
                source_name = self.source_name_dict[table_name]
                sql = f'PRAGMA index_list({source_name})'
                cursor.execute(sql)
                for _, index_name, *_ in cursor.fetchall():
                    sql = f'PRAGMA index_info({quote_ident(index_name)})'
                    cursor.execute(sql)
                    # Fetch index information and sort by `seqno`:
                    index_info = tuple(
                        info[2]
                        for info in sorted(
                            cursor.fetchall(), key=lambda x: x[0]
                        )
                    )
                    # Remove all indices in case primary index already exists:
                    for index in list(indices):
                        if index_info[0] == index[0]:
                            indices.discard(index)
                if len(indices) == 0:
                    del index_dict[table_name]

        if optimize and len(index_dict) > 0:
            if not isinstance(verbose, ProgressLogger):
                verbose = ProgressLogger.default(
                    msg='Optimizing SQLite database',
                    verbose=verbose,
                )

            with verbose as logger, self._connection.cursor() as cursor:
                for table_name, indices in index_dict.items():
                    for index in indices:
                        name = f'kumo_index_{table_name}_{"_".join(index)}'
                        name = quote_ident(name)
                        columns = ', '.join(quote_ident(v) for v in index)
                        columns += ' DESC' if len(index) > 1 else ''
                        source_name = self.source_name_dict[table_name]
                        sql = (
                            f'CREATE INDEX IF NOT EXISTS {name}\n'
                            f'ON {source_name}({columns})'
                        )
                        cursor.execute(sql)
                        self._connection.commit()
                        if len(index) > 1:
                            logger.log(
                                f'Created index on {index} in table '
                                f"'{table_name}'"
                            )
                        else:
                            logger.log(
                                f"Created index on '{index[0]}' in "
                                f"table '{table_name}'"
                            )

        elif len(index_dict) > 0:
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

    @property
    def backend(self) -> DataBackend:
        return cast(DataBackend, DataBackend.SQLITE)

    def _time_separator(self, table_name: str) -> str:
        r"""The separator the stored timestamps of ``table_name`` actually use.

        SQLite has no datetime type, so a window bound is compared as text and
        has to be spelled the way the column is. ``datetime()`` and
        ``pandas.to_sql`` write ``YYYY-MM-DD HH:MM:SS``, but an ISO-8601 writer
        emits a ``T``, and ``'...T00:00:00' <= '... 00:00:00'`` is false for
        the same instant because ``T`` sorts after a space -- silently dropping
        every row that lands on a window edge.

        One stored value decides it, and the answer is remembered per table.
        Anything unreadable falls back to the space form, which is what this
        code assumed unconditionally before, so a column that is not written
        the ISO way behaves exactly as it did.

        A column that mixes both spellings is the one case this cannot get
        right: whichever spelling the probe reads, rows written the other way
        are misplaced by one boundary instant. Widening the window to admit
        both -- a space lower bound and a ``T`` upper bound -- is worse, since
        it also admits every space-written row on the whole of the end date.
        Store one spelling per column.
        """
        separator = self._time_separator_dict.get(table_name)
        if separator is None:
            separator = self._probe_time_separator(table_name)
            self._time_separator_dict[table_name] = separator
        return separator

    def _probe_time_separator(self, table_name: str) -> str:
        r"""Read one stored timestamp and report which separator it uses."""
        column = self.time_column_dict.get(table_name)
        if column is None:
            return ' '
        column_ref = self.table_column_ref_dict[table_name][column]
        sql = (
            f'SELECT {column_ref} FROM {self.source_name_dict[table_name]} '
            f'WHERE {column_ref} IS NOT NULL LIMIT 1'
        )
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(sql)
                row = cursor.fetchone()
        except Exception:
            return ' '
        if not row or not isinstance(row[0], str) or len(row[0]) < 11:
            return ' '
        return 'T' if row[0][10] == 'T' else ' '

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
        # NOTE SQLite does not natively support passing a `random_seed`.
        if entity_ids is None:
            self._warn_random_seed_unsupported(random_seed)

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

        # TODO: avoid the full table scan here.
        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]
        sql = (
            f'SELECT {", ".join(projections)}\n'
            f'FROM {self.source_name_dict[table_name]}'
        )
        if len(filters) > 0:
            sql += f'\nWHERE{" AND".join(filters)}'
        if entity_ids is None:
            sql += f'\nORDER BY RANDOM() LIMIT {num_rows}'

        with self._connection.cursor() as cursor:
            # NOTE This may return duplicate primary keys. This is okay.
            cursor.execute(sql, parameters)
            table = cursor.fetch_arrow_table()

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

    def _by_pkey(
        self,
        table_name: str,
        index: pd.Series,
        columns: set[str],
    ) -> tuple[pd.DataFrame, np.ndarray]:
        source_table = self.source_table_dict[table_name]
        key = self.primary_key_dict[table_name]
        key_ref = self.table_column_ref_dict[table_name][key]
        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]

        tmp = pa.table([pa.array(index)], names=['__kumo_id__'])
        tmp_name = f'tmp_{table_name}_{key}_{id(tmp)}'

        sql = (
            f'SELECT '
            f'tmp.rowid - 1 as __kumo_batch__, '
            f'{", ".join(projections)}\n'
            f'FROM {quote_ident(tmp_name)} tmp\n'
            f'JOIN {self.source_name_dict[table_name]} ent\n'
        )
        if key in source_table and source_table[key].is_unique_key:
            sql += f'  ON {key_ref} = tmp.__kumo_id__'
        else:
            sql += (
                f'  ON ent.rowid = (\n'
                f'    SELECT rowid\n'
                f'    FROM {self.source_name_dict[table_name]}\n'
                f'    WHERE {key_ref} == tmp.__kumo_id__\n'
                f'    LIMIT 1\n'
                f')'
            )

        with self._connection.cursor() as cursor:
            cursor.adbc_ingest(tmp_name, tmp, mode='replace')
            cursor.execute(sql)
            table = cursor.fetch_arrow_table()

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

        # NOTE SQLite has no native datetime type. The bounds below are
        # compared as text, so they have to be spelled the way the column is:
        # `TEXT` in SQLite's own `YYYY-MM-DD HH:MM:SS` UTC form, which is what
        # `datetime()` returns and what `pandas.to_sql` writes. A column stored
        # with the ISO-8601 `T` separator instead does NOT compare correctly on
        # the boundary -- `'...T00:00:00' <= '... 00:00:00'` is false for the
        # same instant, because `T` sorts after a space -- so rows landing
        # exactly on a window edge are dropped.
        tmp = pa.table([pa.array(index)], names=['__kumo_id__'])
        if time_column is not None and anchor_time is not None:
            _sep = self._time_separator(table_name)
            anchor_time = anchor_time.dt.strftime(f'%Y-%m-%d{_sep}%H:%M:%S')
            tmp = tmp.append_column('__kumo_time__', pa.array(anchor_time))
        tmp_name = f'tmp_{table_name}_{foreign_key}_{id(tmp)}'

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]
        sql = (
            f'SELECT '
            f'tmp.rowid - 1 as __kumo_batch__, '
            f'{", ".join(projections)}\n'
            f'FROM {quote_ident(tmp_name)} tmp\n'
            f'JOIN {self.source_name_dict[table_name]} fact\n'
            f'ON fact.rowid IN (\n'
            f'  SELECT rowid\n'
            f'  FROM {self.source_name_dict[table_name]}\n'
            f'  WHERE {key_ref} = tmp.__kumo_id__\n'
        )
        if time_column is not None and anchor_time is not None:
            time_ref = self.table_column_ref_dict[table_name][time_column]
            sql += f'  AND {time_ref} <= tmp.__kumo_time__\n'
        time_ref = (
            None
            if time_column is None
            else self.table_column_ref_dict[table_name][time_column]
        )
        order_by = self._neighbor_order_by(table_name, time_ref, key_ref)
        sql += f'  ORDER BY {order_by}\n'
        sql += f'  LIMIT {num_neighbors}\n)'

        with self._connection.cursor() as cursor:
            cursor.adbc_ingest(tmp_name, tmp, mode='replace')
            cursor.execute(sql)
            table = cursor.fetch_arrow_table()

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

        # NOTE SQLite has no native datetime type. The bounds below are
        # compared as text, so they have to be spelled the way the column is:
        # `TEXT` in SQLite's own `YYYY-MM-DD HH:MM:SS` UTC form, which is what
        # `datetime()` returns and what `pandas.to_sql` writes. A column stored
        # with the ISO-8601 `T` separator instead does NOT compare correctly on
        # the boundary -- `'...T00:00:00' <= '... 00:00:00'` is false for the
        # same instant, because `T` sorts after a space -- so rows landing
        # exactly on a window edge are dropped.
        tmp = pa.table([pa.array(index)], names=['__kumo_id__'])
        _sep = self._time_separator(table_name)
        end_time = anchor_time + max_offset
        end_time = end_time.dt.strftime(f'%Y-%m-%d{_sep}%H:%M:%S')
        tmp = tmp.append_column('__kumo_end__', pa.array(end_time))
        if min_offset is not None:
            start_time = anchor_time + min_offset
            start_time = start_time.dt.strftime(f'%Y-%m-%d{_sep}%H:%M:%S')
            tmp = tmp.append_column('__kumo_start__', pa.array(start_time))
        tmp_name = f'tmp_{table_name}_{foreign_key}_{id(tmp)}'

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        time_ref = self.table_column_ref_dict[table_name][time_column]
        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]
        sql = (
            f'SELECT '
            f'tmp.rowid - 1 as __kumo_batch__, '
            f'{", ".join(projections)}\n'
            f'FROM {quote_ident(tmp_name)} tmp\n'
            f'JOIN {self.source_name_dict[table_name]}\n'
            f'  ON {key_ref} = tmp.__kumo_id__\n'
            f' AND {time_ref} <= tmp.__kumo_end__'
        )
        if min_offset is not None:
            sql += f'\n AND {time_ref} > tmp.__kumo_start__'

        with self._connection.cursor() as cursor:
            cursor.adbc_ingest(tmp_name, tmp, mode='replace')
            cursor.execute(sql)
            table = cursor.fetch_arrow_table()

        batch = table['__kumo_batch__'].to_numpy()
        batch_index = table.schema.get_field_index('__kumo_batch__')
        table = table.remove_column(batch_index)

        return Table._sanitize(
            df=table.to_pandas(types_mapper=pd.ArrowDtype),
            dtype_dict=self.table_dtype_dict[table_name],
            stype_dict=self.table_stype_dict[table_name],
        ), batch

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
