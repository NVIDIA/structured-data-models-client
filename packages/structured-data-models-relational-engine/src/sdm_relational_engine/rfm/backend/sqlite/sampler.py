# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from predictive_query.query import ValidatedPredictiveQuery

from sdm_relational_engine.rfm.backend.sqlite import SQLiteTable, dialect
from sdm_relational_engine.rfm.base import DataBackend, SQLSampler, Table
from sdm_relational_engine.rfm.base.utils import Timestamp
from sdm_relational_engine.rfm.query.pandas_executor import PQueryPandasExecutor
from sdm_relational_engine.utils.progress_logger import ProgressLogger

if TYPE_CHECKING:
    from sdm_relational_engine.rfm import Graph


class _TimeSpelling(NamedTuple):
    r"""How one table writes its timestamps as text.

    SQLite has no datetime type, so a window bound is compared with the stored
    values as text, and has to be written exactly the way they are: the same
    separator, the same number of fractional digits, the same zone suffix. A
    shorter string with the same prefix sorts first, so a bound missing the
    stored ``.000000`` is *before* a value at the same instant.
    """

    separator: str = ' '
    digits: int = 0
    suffix: str = ''

    def spell(self, times: pd.Series) -> pd.Series:
        spelled = times.dt.strftime(f'%Y-%m-%d{self.separator}%H:%M:%S')
        if self.digits:
            # `%f` stops at microseconds, and a stored nanosecond sorts after
            # a bound padded with zeros, so the last three digits are added.
            # A NaT turns the column to floats, which would print as `0.0`.
            nanos = times.dt.nanosecond.fillna(0).astype('int64')
            nanos = nanos.astype(str).str.zfill(3)
            fraction = times.dt.strftime('%f') + nanos
            fraction = fraction.str.ljust(self.digits, '0')
            spelled = spelled + '.' + fraction.str[: self.digits]
        return spelled + self.suffix


# A timestamp as SQLite text: date, separator, time, optional fraction, and
# whatever follows. Only a UTC suffix is kept, because the bounds are UTC; any
# other offset spells a different instant and is left as it was.
_STORED_TIME = re.compile(
    r'^\d{4}-\d{2}-\d{2}([T ])\d{2}:\d{2}:\d{2}(?:\.(\d+))?(.*)$'
)
_UTC_SUFFIXES = frozenset({'', 'Z', '+00:00', '+0000'})


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

        # Per table, decided once on first use by `_time_spelling`. Held on
        # the instance rather than in a `functools.cache`, which would key on
        # `self` and so keep every sampler, its graph, and its open connection
        # alive for the life of the process.
        self._time_spelling_dict: dict[str, _TimeSpelling] = {}

        # The engine only reads the database, so `optimize` no longer creates
        # the indices it used to; it only stops the missing ones being
        # reported.
        if not optimize:
            self._warn_missing_indices(graph)

    def _warn_missing_indices(self, graph: 'Graph') -> None:
        r"""Warn about the keys no index covers, which make lookups slow."""
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
        for table_name in list(index_dict.keys()):
            indices = index_dict[table_name]
            source_name = self.source_name_dict[table_name]
            for _, index_name, *_ in self._rows(
                dialect.index_list(source_name)
            ):
                # Fetch index information and sort by `seqno`:
                index_info = tuple(
                    info[2]
                    for info in sorted(
                        self._rows(dialect.index_info(index_name)),
                        key=lambda x: x[0],
                    )
                )
                # Remove all indices in case primary index already exists:
                for index in list(indices):
                    if index_info[0] == index[0]:
                        indices.discard(index)
            if len(indices) == 0:
                del index_dict[table_name]

        if len(index_dict) > 0:
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

    def _time_spelling(self, table_name: str) -> _TimeSpelling:
        r"""How the stored timestamps of ``table_name`` are written.

        SQLite has no datetime type, so a window bound is compared as text and
        has to be spelled the way the column is. ``datetime()`` and
        ``pandas.to_sql`` write ``YYYY-MM-DD HH:MM:SS``, but an ISO-8601 writer
        emits a ``T``, and ``'...T00:00:00' <= '... 00:00:00'`` is false for
        the same instant because ``T`` sorts after a space. A writer that keeps
        fractional seconds, as pandas through ADBC does with ``.000000``,
        breaks it the same way: ``'...T14:00:00.000000' <= '...T14:00:00'`` is
        false, so a row on the end of a window was dropped and a row on the
        anchor was counted. Either way, silently: no error, just a different
        answer.

        One stored value decides it, and the answer is remembered per table.
        Anything unreadable falls back to ``YYYY-MM-DD HH:MM:SS``, which is
        what this code assumed unconditionally before, so a column that is not
        written some other way behaves exactly as it did.

        A column that mixes spellings is the one case this cannot get right:
        whichever the probe reads, rows written the other way are misplaced by
        one boundary instant. Store one spelling per column.
        """
        spelling = self._time_spelling_dict.get(table_name)
        if spelling is None:
            spelling = self._probe_time_spelling(table_name)
            self._time_spelling_dict[table_name] = spelling
        return spelling

    def _probe_time_spelling(self, table_name: str) -> _TimeSpelling:
        r"""Read one stored timestamp and report how it is written."""
        column = self.time_column_dict.get(table_name)
        if column is None:
            return _TimeSpelling()
        column_ref = self.table_column_ref_dict[table_name][column]
        statement = dialect.first_time(
            self.source_name_dict[table_name], column_ref
        )
        try:
            found = self._rows(statement)
        except Exception:
            return _TimeSpelling()
        row = found[0] if found else None
        if not row or not isinstance(row[0], str):
            return _TimeSpelling()
        match = _STORED_TIME.match(row[0])
        if match is None:
            return _TimeSpelling()
        separator, fraction, suffix = match.groups()
        return _TimeSpelling(
            separator=separator,
            digits=len(fraction or ''),
            suffix=suffix if suffix in _UTC_SUFFIXES else '',
        )

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

    def _seeded_order_by(
        self,
        table_name: str,
        columns: set[str],
        random_seed: int,
    ) -> str:
        r"""The ``ORDER BY`` of a seeded entity draw.

        An integer primary key is hashed, so the draw follows the data alone.
        SQLite has no hash of a string, so a table with another kind of key
        hashes its ``rowid`` instead: the draw then follows the physical row
        ids, which ``VACUUM`` may renumber unless the key is the table's
        ``INTEGER PRIMARY KEY``, and so is the rowid itself.

        Args:
            table_name: The table being sampled.
            columns: The columns being read.
            random_seed: The seed of the draw.
        """
        key = self.primary_key_dict[table_name]
        source = self.source_table_dict[table_name].get(key)
        dtype = source.dtype if source is not None else None
        if dtype is None:
            dtype = self.table_dtype_dict[table_name].get(key)
        if dtype is not None and dtype.is_int():
            key_ref = self.table_column_ref_dict[table_name][key]
            hashed = dialect.seeded_hash(random_seed, key_ref)
            return f'{hashed}, {self._rows_order_by(table_name, columns)}'
        return f'{dialect.seeded_hash(random_seed, "rowid")}, rowid'

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
        # TODO: avoid the full table scan here.
        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]
        if entity_ids is None and random_seed is None:
            order_by = None
        elif entity_ids is None:
            order_by = self._seeded_order_by(table_name, columns, random_seed)
        else:
            order_by = self._rows_order_by(table_name, columns)

        # NOTE This may return duplicate primary keys. This is okay.
        table = self._run(
            dialect.entity_rows(
                self.source_name_dict[table_name],
                projections,
                filters,
                order_by=order_by,
                num_rows=num_rows,
                key=self.table_column_ref_dict[table_name][key],
                ids=None if entity_ids is None else list(entity_ids),
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
        scored: np.ndarray | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        source_table = self.source_table_dict[table_name]
        key = self.primary_key_dict[table_name]
        key_ref = self.table_column_ref_dict[table_name][key]
        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]

        unique = key in source_table and source_table[key].is_unique_key
        # Counted only where it is asked about: each count is a lookup of its
        # own, and an id the source keeps unique matches one row anyway.
        counted = scored is not None and len(scored) > 0 and not unique
        # Of the rows holding the id, the first in a total order of the
        # values read, so the same row is kept however they are stored.
        table = self._run(
            dialect.rows_by_key(
                self.source_name_dict[table_name],
                key_ref,
                pa.array(index),
                projections,
                unique=unique,
                tie_break=self._tie_break_terms(table_name, columns),
                count_matches=counted,
            )
        )

        batch = table['__kumo_batch__'].to_numpy()
        batch_index = table.schema.get_field_index('__kumo_batch__')
        table = table.remove_column(batch_index)
        if counted:
            self._refuse_an_id_scored_twice(
                table_name,
                index,
                batch,
                table['__kumo_matches__'].to_numpy(),
                scored,
            )
            matches_index = table.schema.get_field_index('__kumo_matches__')
            table = table.remove_column(matches_index)

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

        # SQLite compares these bounds with the stored text, so they are
        # spelled exactly as the column is; see `_time_spelling`.
        end_times: pa.Array | None = None
        if time_column is not None and anchor_time is not None:
            anchor_time = self._time_spelling(table_name).spell(anchor_time)
            end_times = pa.array(anchor_time)

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        projections = [
            self.table_column_proj_dict[table_name][column]
            for column in self._ordered_columns(table_name, columns)
        ]
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

        # SQLite compares these bounds with the stored text, so they are
        # spelled exactly as the column is; see `_time_spelling`.
        spelling = self._time_spelling(table_name)
        end_times = pa.array(spelling.spell(anchor_time + max_offset))
        start_times: pa.Array | None = None
        if min_offset is not None:
            start_times = pa.array(spelling.spell(anchor_time + min_offset))

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        time_ref = self.table_column_ref_dict[table_name][time_column]
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
