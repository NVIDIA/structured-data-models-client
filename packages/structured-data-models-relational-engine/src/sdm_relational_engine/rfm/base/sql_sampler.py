# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from predictive_query.schema.typing import Dtype
from sdm_connectors import quote_ident

from sdm_relational_engine.rfm.base import (
    LocalExpression,
    Sampler,
    SamplerOutput,
    SourceColumn,
)
from sdm_relational_engine.rfm.base.mapper import Mapper
from sdm_relational_engine.rfm.inference.context import SampledSubgraph
from sdm_relational_engine.utils.progress_logger import ProgressLogger

if TYPE_CHECKING:
    from sdm_relational_engine.rfm import Graph

EdgeType = tuple[str, str, str]


# Every SQL backend joins against a temporary key table whose columns carry
# this prefix. A source column sharing one of those names makes the generated
# join ambiguous, so the prefix is reserved.
RESERVED_COLUMN_PREFIX = '__kumo_'


def _reject_reserved_columns(graph: 'Graph') -> None:
    r"""Refuse a source column that would collide with an internal one.

    The generated joins select the temporary key table's ``__kumo_*`` columns
    beside the caller's own projections. A source column of the same name makes
    the reference ambiguous, which the warehouse reports as a binder error
    naming only the column, with nothing in that message pointing back here.
    Saying so up front, with the table and column, costs one pass over the
    schema.
    """
    collisions = [
        f'{table.name}.{column.name}'
        for table in graph.tables.values()
        for column in table.columns
        if column.name.startswith(RESERVED_COLUMN_PREFIX)
    ]
    if collisions:
        raise ValueError(
            f'{", ".join(sorted(collisions))} '
            f'{"use names" if len(collisions) > 1 else "uses a name"} '
            f'reserved by the SQL backends: a column may not start with '
            f'{RESERVED_COLUMN_PREFIX!r}, because the generated joins already '
            f'select columns of that name and the reference would be '
            f'ambiguous. Rename the column, or project it under another name.'
        )


class SQLSampler(Sampler):
    # The character used to quote SQL identifiers. Backends whose dialect does
    # not use double quotes (*e.g.*, Databricks uses backticks) override this:
    _QUOTE_CHAR: str = '"'

    def __init__(
        self,
        graph: 'Graph',
        verbose: bool | ProgressLogger = True,
    ) -> None:
        super().__init__(graph=graph, verbose=verbose)

        _reject_reserved_columns(graph)

        self._warned_random_seed = False

        self._source_name_dict: dict[str, str] = {
            table.name: table._quoted_source_name
            for table in graph.tables.values()
        }

        self._source_table_dict: dict[str, dict[str, SourceColumn]] = {}
        for table in graph.tables.values():
            self._source_table_dict[table.name] = {}
            for column in table.columns:
                if not column.is_source:
                    continue
                src_column = table._source_column_dict[column.name]
                self._source_table_dict[table.name][column.name] = src_column

        self._table_dtype_dict: dict[str, dict[str, Dtype]] = {}
        for table in graph.tables.values():
            self._table_dtype_dict[table.name] = {}
            for column in table.columns:
                self._table_dtype_dict[table.name][column.name] = column.dtype

        self._table_column_ref_dict: dict[str, dict[str, str]] = {}
        self._table_column_proj_dict: dict[str, dict[str, str]] = {}
        for table in graph.tables.values():
            column_ref_dict: dict[str, str] = {}
            column_proj_dict: dict[str, str] = {}
            for column in table.columns:
                if column.expr is not None:
                    assert isinstance(column.expr, LocalExpression)
                    column_ref_dict[column.name] = column.expr.value
                    column_proj_dict[column.name] = (
                        f'{column.expr} AS '
                        f'{quote_ident(column.name, self._QUOTE_CHAR)}'
                    )
                else:
                    ident = quote_ident(column.name, self._QUOTE_CHAR)
                    column_ref_dict[column.name] = ident
                    column_proj_dict[column.name] = ident
            self._table_column_ref_dict[table.name] = column_ref_dict
            self._table_column_proj_dict[table.name] = column_proj_dict

    def _ordered_columns(
        self,
        table_name: str,
        columns: Iterable[str],
    ) -> list[str]:
        r"""Order column names by their position in the table schema.

        Callers pass ``columns`` as a :class:`set`, whose iteration order
        depends on the per-process string hash seed. Projecting a ``SELECT``
        straight from that set makes every sampled frame's column order differ
        between interpreters. The serialised request and derived prediction
        also differ, defeating ``random_seed``.
        Ordering by schema position instead is stable across processes and
        matches the column order the in-memory backend produces from a
        dataframe. Names absent from the schema sort last, by name, so an
        unexpected column can never reintroduce the instability.

        Args:
            table_name: The table whose schema defines the order.
            columns: The column names to order.
        """
        order = {
            column: position
            for position, column in enumerate(
                self.table_column_proj_dict[table_name]
            )
        }
        return sorted(
            columns,
            key=lambda column: (order.get(column, len(order)), column),
        )

    def _no_rows(
        self,
        table_name: str,
        columns: Iterable[str],
    ) -> tuple[pd.DataFrame, np.ndarray]:
        r"""What a lookup for no entities returns, without asking for it.

        The frame has the columns a query would have returned, with the time
        column typed, so a target computed over it compares times as usual.
        """
        time_column = self.time_column_dict.get(table_name)
        df = pd.DataFrame(
            {
                column: pd.Series(
                    [],
                    dtype='datetime64[ns]' if column == time_column else object,
                )
                for column in self._ordered_columns(table_name, columns)
            }
        )
        return df, np.empty(0, dtype=np.int64)

    def _neighbor_order_by(
        self,
        table_name: str,
        time_ref: str | None,
        key_ref: str,
    ) -> str:
        r"""Builds the ``ORDER BY`` that picks which neighbours to keep.

        ``_by_fkey`` keeps the first ``num_neighbors`` rows of each partition,
        so the ordering decides which rows reach the model. Ordering on the
        time column alone lets the engine arbitrarily break repeated
        timestamps, which are common in fact tables recorded at day
        granularity. This changes the sampled rows between otherwise identical
        requests. Appending the primary key makes the choice total and stable.

        Ordering on ``key_ref`` alone cannot serve as that tie-break: it is the
        foreign key the partition is keyed by, so it holds one value per
        partition. A table with no primary key is ordered by every one of its
        scalar columns instead, in name order. Two rows that agree on all of
        them are the same row as far as the model can tell, so the order
        between them cannot change what is sent. Without this, rows sharing a
        timestamp came back in whatever order the engine read them, and the
        same question built a different request on Databricks serverless Spark
        than on a SQL warehouse.

        Args:
            table_name: The table being sampled from.
            time_ref: The reference to its time column, if it has one.
            key_ref: The reference to the foreign key being traversed.
        """
        terms = [] if time_ref is None else [f'{time_ref} DESC']

        primary_key = self.primary_key_dict.get(table_name)
        if primary_key is not None:
            terms.append(self.table_column_ref_dict[table_name][primary_key])
            return ', '.join(terms)

        dtypes = self.table_dtype_dict.get(table_name, {})
        for column, ref in sorted(
            self.table_column_ref_dict[table_name].items()
        ):
            dtype = dtypes.get(column)
            if ref == time_ref or (dtype is not None and dtype.is_list()):
                continue
            terms.append(self._tie_break_term(table_name, column, ref))
        if not terms:
            terms.append(key_ref)

        return ', '.join(terms)

    def _tie_break_term(self, table_name: str, column: str, ref: str) -> str:
        r"""The ``ORDER BY`` term for one column of the keyless tie-break.

        A backend that reads some unorderable types as scalars overrides this
        to order a comparable form of them, since one fails the whole query.
        """
        return ref

    def _warn_random_seed_unsupported(self, random_seed: int | None) -> None:
        r"""Warns once that this backend cannot draw a reproducible random
        sample of rows, rather than dropping ``random_seed`` silently.

        Args:
            random_seed: The seed requested by the caller.
        """
        if random_seed is None or self._warned_random_seed:
            return

        self._warned_random_seed = True
        warnings.warn(
            f"The '{self.backend}' backend does not support seeded "
            f"random sampling, so 'random_seed' is ignored when "
            f'drawing in-context examples. Repeated calls with the '
            f'same seed may return different predictions.'
        )

    @property
    def source_name_dict(self) -> dict[str, str]:
        r"""The source table names for all tables in the graph."""
        return self._source_name_dict

    @property
    def source_table_dict(self) -> dict[str, dict[str, SourceColumn]]:
        r"""The source column information for all tables in the graph."""
        return self._source_table_dict

    @property
    def table_dtype_dict(self) -> dict[str, dict[str, Dtype]]:
        r"""The data types for all columns in all tables in the graph."""
        return self._table_dtype_dict

    @property
    def table_column_ref_dict(self) -> dict[str, dict[str, str]]:
        r"""The SQL reference expression for all columns in all tables in the
        graph.
        """
        return self._table_column_ref_dict

    @property
    def table_column_proj_dict(self) -> dict[str, dict[str, str]]:
        r"""The SQL projection expressions for all columns in all tables in the
        graph.
        """
        return self._table_column_proj_dict

    def _sample_subgraph(
        self,
        entity_table_name: str,
        entity_pkey: pd.Series,
        anchor_time: pd.Series | Literal['entity'],
        columns_dict: dict[str, set[str]],
        num_neighbors: list[int],
        random_seed: int | None = None,
        scored: np.ndarray | None = None,
    ) -> SamplerOutput:

        # SQL backends do not use the local pseudo-random neighborhood
        # sampler, so this option does not affect their traversal. Random row
        # sampling is seeded in `_sample_entity_table` instead.
        del random_seed

        # Make sure to always include primary key, foreign key and time columns
        # during data fetching since these are needed for graph traversal:
        sample_columns_dict: dict[str, set[str]] = {}
        for table, columns in columns_dict.items():
            sample_columns = columns | {
                foreign_key for foreign_key, _ in self.foreign_key_dict[table]
            }
            if primary_key := self.primary_key_dict.get(table):
                sample_columns |= {primary_key}
            sample_columns_dict[table] = sample_columns
        if not isinstance(anchor_time, pd.Series):
            sample_columns_dict[entity_table_name] |= {
                self.time_column_dict[entity_table_name]
            }

        # Sample Entity Table #################################################

        df, batch = self._by_pkey(
            table_name=entity_table_name,
            index=entity_pkey,
            columns=sample_columns_dict[entity_table_name],
            scored=scored,
        )
        if len(batch) != len(entity_pkey):
            mask = np.ones(len(entity_pkey), dtype=bool)
            mask[batch] = False
            raise KeyError(
                f'The primary keys '
                f'{entity_pkey.iloc[mask].tolist()} do not exist '
                f"in the '{entity_table_name}' table"
            )

        # Make sure that entities are returned in expected order:
        perm = batch.argsort()
        batch = batch[perm]
        df = df.iloc[perm].reset_index(drop=True)

        # Fill 'entity' anchor times with actual values:
        if not isinstance(anchor_time, pd.Series):
            time_column = self.time_column_dict[entity_table_name]
            anchor_time = df[time_column]
        assert isinstance(anchor_time, pd.Series)

        # Recursive Neighbor Sampling #########################################

        mapper_dict: dict[str, Mapper] = defaultdict(
            lambda: Mapper(num_examples=len(entity_pkey))
        )
        mapper_dict[entity_table_name].add(
            pkey=df[self.primary_key_dict[entity_table_name]],
            batch=batch,
        )

        dfs_dict: dict[str, list[pd.DataFrame]] = defaultdict(list)
        dfs_dict[entity_table_name].append(df)
        batches_dict: dict[str, list[np.ndarray]] = defaultdict(list)
        batches_dict[entity_table_name].append(batch)
        num_sampled_nodes_dict: dict[str, list[int]] = defaultdict(
            lambda: [0] * (len(num_neighbors) + 1)
        )
        num_sampled_nodes_dict[entity_table_name][0] = len(entity_pkey)

        rows_dict: dict[EdgeType, list[np.ndarray]] = defaultdict(list)
        cols_dict: dict[EdgeType, list[np.ndarray]] = defaultdict(list)
        num_sampled_edges_dict: dict[EdgeType, list[int]] = defaultdict(
            lambda: [0] * len(num_neighbors)
        )

        # The start index of data frame slices of the previous hop:
        offset_dict: dict[str, int] = defaultdict(int)

        visited_hops = 0
        for hop, neighbors in enumerate(num_neighbors):
            if neighbors == 0:
                break  # Abort early.

            for table in list(num_sampled_nodes_dict.keys()):
                # Only sample from tables that have been visited in the
                # previous hop:
                if num_sampled_nodes_dict[table][hop] == 0:
                    continue

                # Collect the slices of data sampled in the previous hop
                # (but maintain only required key information):
                cols = [fkey for fkey, _ in self.foreign_key_dict[table]]
                if table in self.primary_key_dict:
                    cols.append(self.primary_key_dict[table])
                dfs = [df[cols] for df in dfs_dict[table][offset_dict[table] :]]
                df = (
                    pd.concat(
                        dfs,
                        axis=0,
                        ignore_index=True,
                    )
                    if len(dfs) > 1
                    else dfs[0]
                )
                batches = batches_dict[table][offset_dict[table] :]
                batch = (
                    np.concatenate(batches) if len(batches) > 1 else batches[0]
                )
                offset_dict[table] = len(batches_dict[table])  # Increase.

                pkey: pd.Series | None = None
                index: np.ndarray | None = None
                if table in self.primary_key_dict:
                    pkey = df[self.primary_key_dict[table]]
                    index = mapper_dict[table].get(pkey, batch)

                # Iterate over foreign keys in the current table:
                for fkey, dst_table in self.foreign_key_dict[table]:
                    row = mapper_dict[dst_table].get(df[fkey], batch)
                    mask = row == -1
                    if mask.any():
                        key_df = pd.DataFrame(
                            {
                                'fkey': df[fkey],
                                'batch': batch,
                            }
                        ).iloc[mask]
                        # Only maintain unique keys per example:
                        unique_key_df = key_df.drop_duplicates()
                        # Fully de-duplicate keys across examples:
                        code, fkey_index = pd.factorize(unique_key_df['fkey'])

                        _df, _batch = self._by_pkey(
                            table_name=dst_table,
                            index=pd.Series(fkey_index),
                            columns=sample_columns_dict[dst_table],
                        )

                        if len(_df) > 0:
                            # Ensure result is sorted according to input order:
                            _df = _df.iloc[_batch.argsort()]

                            # Compute valid entries in `unique_fkey_df`
                            # (without dangling foreign keys):
                            _mask = np.full(len(fkey_index), fill_value=False)
                            _mask[_batch] = True
                            _mask = _mask[code]

                            # Recontruct unique (key, batch) pairs:
                            code = unique_key_df['fkey'][_mask]
                            code, _ = pd.factorize(code)
                            _df = _df.iloc[code].reset_index(drop=True)
                            _batch = unique_key_df['batch'].to_numpy()[_mask]

                            # Register node IDs:
                            mapper_dict[dst_table].add(
                                pkey=_df[self.primary_key_dict[dst_table]],
                                batch=_batch,
                            )
                            # NOTE `row` may still hold `-1` for dangling keys.
                            row[mask] = mapper_dict[dst_table].get(
                                pkey=key_df['fkey'],
                                batch=key_df['batch'].to_numpy(),
                            )

                            dfs_dict[dst_table].append(_df)
                            batches_dict[dst_table].append(_batch)
                            num_sampled_nodes_dict[dst_table][hop + 1] += (  #
                                len(_batch)
                            )

                    mask = row != -1

                    col = index
                    if col is None:
                        start = sum(num_sampled_nodes_dict[table][:hop])
                        end = sum(num_sampled_nodes_dict[table][: hop + 1])
                        col = np.arange(start, end)

                    row = row[mask]
                    col = col[mask]

                    if len(col) > 0:
                        edge_type = (table, fkey, dst_table)
                        edge_type = SampledSubgraph.rev_edge_type(edge_type)
                        rows_dict[edge_type].append(row)
                        cols_dict[edge_type].append(col)
                        num_sampled_edges_dict[edge_type][hop] = len(col)
                        visited_hops = hop + 1

                # Iterate over foreign keys that reference the current table:
                for src_table, fkey in self.rev_foreign_key_dict[table]:
                    assert pkey is not None and index is not None
                    _df, _batch = self._by_fkey(
                        table_name=src_table,
                        foreign_key=fkey,
                        index=pkey,
                        num_neighbors=neighbors,
                        anchor_time=anchor_time.iloc[batch],
                        columns=sample_columns_dict[src_table],
                    )

                    if len(_df) == 0:
                        continue

                    edge_type = (src_table, fkey, table)
                    cols_dict[edge_type].append(index[_batch])
                    num_sampled_edges_dict[edge_type][hop] = len(_batch)
                    visited_hops = hop + 1

                    _batch = batch[_batch]
                    num_nodes = sum(num_sampled_nodes_dict[src_table])
                    if src_table in self.primary_key_dict:
                        _pkey = _df[self.primary_key_dict[src_table]]
                        mapper_dict[src_table].add(_pkey, _batch)
                        row = mapper_dict[src_table].get(_pkey, _batch)

                        # Only preserve unknown rows:
                        mask = row >= num_nodes
                        # Drop duplicated (pkey, batch) pairs:
                        mask[pd.Series(row).duplicated().to_numpy()] = False

                        _df = _df.iloc[mask]
                        _batch = _batch[mask]
                    else:
                        row = np.arange(num_nodes, num_nodes + len(_batch))

                    rows_dict[edge_type].append(row)
                    num_sampled_nodes_dict[src_table][hop + 1] += len(_batch)

                    dfs_dict[src_table].append(_df)
                    batches_dict[src_table].append(_batch)

        # Post-Processing #####################################################

        df_dict = {
            table: pd.concat(dfs, axis=0, ignore_index=True)
            if len(dfs) > 1
            else dfs[0]
            for table, dfs in dfs_dict.items()
        }

        # Only store unique rows in `df` above a certain threshold:
        inverse_dict: dict[str, np.ndarray] = {}
        for table, df in df_dict.items():
            if table not in self.primary_key_dict:
                continue
            unique, index, inverse = np.unique(
                df_dict[table][self.primary_key_dict[table]],
                return_index=True,
                return_inverse=True,
            )
            if len(df) > 1.05 * len(unique):
                df_dict[table] = df.iloc[index].reset_index(drop=True)
                inverse_dict[table] = inverse

        df_dict = {  # Post-filter column set:
            table: df[sorted(columns_dict[table], key=df.columns.get_loc)]
            for table, df in df_dict.items()
        }
        batch_dict = {
            table: np.concatenate(batches) if len(batches) > 1 else batches[0]
            for table, batches in batches_dict.items()
        }
        row_dict = {
            edge_type: np.concatenate(rows)
            for edge_type, rows in rows_dict.items()
        }
        col_dict = {
            edge_type: np.concatenate(cols)
            for edge_type, cols in cols_dict.items()
        }

        if visited_hops != len(num_neighbors):
            num_sampled_nodes_dict = {
                key: value[: visited_hops + 1]
                for key, value in num_sampled_nodes_dict.items()
            }
            num_sampled_edges_dict = {
                key: value[:visited_hops]
                for key, value in num_sampled_edges_dict.items()
            }

        return SamplerOutput(
            anchor_time=anchor_time.astype(int).to_numpy(),
            df_dict=df_dict,
            inverse_dict=inverse_dict,
            batch_dict=batch_dict,
            num_sampled_nodes_dict=num_sampled_nodes_dict,
            row_dict=row_dict,
            col_dict=col_dict,
            num_sampled_edges_dict=num_sampled_edges_dict,
        )

    def _sample_query_data(
        self,
        entity_table_name: str,
        entity_pkey: pd.Series,
        anchor_time: pd.Series,
        columns_dict: dict[str, set[str]],
        time_offset_dict: dict[
            tuple[str, str, str],
            tuple[pd.DateOffset | None, pd.DateOffset],
        ],
    ) -> tuple[
        dict[str, pd.DataFrame],
        dict[str, pd.Series],
        dict[str, np.ndarray],
    ]:
        feat_dict: dict[str, pd.DataFrame] = {}
        time_dict: dict[str, pd.Series] = {}
        batch_dict: dict[str, np.ndarray] = {}

        if len(columns_dict.get(entity_table_name, set())) > 1:
            entity_df, batch = self._by_pkey(
                table_name=entity_table_name,
                index=entity_pkey,
                columns=columns_dict[entity_table_name],
            )
            perm = batch.argsort()
            entity_df = entity_df.iloc[perm].reset_index(drop=True)
            feat_dict[entity_table_name] = entity_df
        else:
            feat_dict[entity_table_name] = pd.DataFrame(
                {
                    self.primary_key_dict[entity_table_name]: entity_pkey,
                }
            )
        batch_dict[entity_table_name] = np.arange(
            len(feat_dict[entity_table_name])
        )

        for edge_type, (_min, _max) in time_offset_dict.items():
            table_name, foreign_key, _ = edge_type
            feat_dict[table_name], batch_dict[table_name] = self._by_time(
                table_name=table_name,
                foreign_key=foreign_key,
                index=entity_pkey,
                anchor_time=anchor_time,
                min_offset=_min,
                max_offset=_max,
                columns=columns_dict[table_name],
            )
            time_column = self.time_column_dict.get(table_name)
            if time_column in columns_dict[table_name]:
                time_dict[table_name] = feat_dict[table_name][time_column]

        return feat_dict, time_dict, batch_dict

    # Abstract Methods ########################################################

    @abstractmethod
    def _by_pkey(
        self,
        table_name: str,
        index: pd.Series,
        columns: set[str],
        scored: np.ndarray | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        r"""One row for each id in ``index``, and the position each came for.

        An id matching several rows gets one of them, except at a position in
        ``scored``, which is refused by :meth:`_refuse_an_id_scored_twice`.
        """

    def _refuse_an_id_scored_twice(
        self,
        table_name: str,
        index: pd.Series,
        batch: np.ndarray,
        matches: np.ndarray,
        scored: np.ndarray | None,
    ) -> None:
        r"""Refuse to score an id that more than one row holds.

        Args:
            table_name: The table the ids were looked up in.
            index: The ids looked up.
            batch: The position in ``index`` of each row kept.
            matches: How many rows matched the id of each row kept.
            scored: The positions in ``index`` being scored.
        """
        if scored is None or len(scored) == 0:
            return
        many = np.zeros(len(index), dtype=bool)
        many[np.asarray(batch, dtype=np.int64)] = np.asarray(matches) > 1
        repeated = [int(position) for position in scored if many[position]]
        if not repeated:
            return
        key = self.primary_key_dict[table_name]
        ids = list(dict.fromkeys(index.iloc[repeated].tolist()))
        shown = ', '.join(repr(value) for value in ids[:5])
        more = f' and {len(ids) - 5:,} more' if len(ids) > 5 else ''
        which = 'each of these values' if len(ids) > 1 else 'this value'
        raise ValueError(
            f'Cannot score {shown}{more}: more than one row of table '
            f"'{table_name}' holds {which} of '{key}', so which row is meant "
            f'would be a guess. The source does not enforce the key; keep '
            f'one row per value, or key the table on columns that identify '
            f'a row.'
        )

    @abstractmethod
    def _by_fkey(
        self,
        table_name: str,
        foreign_key: str,
        index: pd.Series,
        num_neighbors: int,
        anchor_time: pd.Series | None,
        columns: set[str],
    ) -> tuple[pd.DataFrame, np.ndarray]:
        pass

    @abstractmethod
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
        pass
