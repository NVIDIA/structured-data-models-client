# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import re
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
import pandas as pd
from predictive_query.ast import Aggregation, ASTNode, Filter
from predictive_query.query import (
    QueryType,
    ValidatedPredictiveQuery,
)
from predictive_query.schema.typing import ProblemType, Stype

from sdm_relational_engine.rfm.base import DataBackend
from sdm_relational_engine.rfm.base.utils import (
    TimeBound,
    date_offset_to_seconds,
    to_naive_utc,
)
from sdm_relational_engine.rfm.diagnostics import GraphSanitizationReport
from sdm_relational_engine.rfm.inference.context import (
    SampledEdgeLayout,
    SampledLink,
    SampledSubgraph,
    SampledTable,
)
from sdm_relational_engine.rfm.query.pandas_executor import PQueryPandasExecutor
from sdm_relational_engine.utils.progress_logger import ProgressLogger

if TYPE_CHECKING:
    from sdm_relational_engine.rfm import Graph

_coverage_warned = False


@dataclass
class SamplerOutput:
    anchor_time: np.ndarray
    df_dict: dict[str, pd.DataFrame]
    inverse_dict: dict[str, np.ndarray]
    batch_dict: dict[str, np.ndarray]
    num_sampled_nodes_dict: dict[str, list[int]]
    row_dict: dict[tuple[str, str, str], np.ndarray]
    col_dict: dict[tuple[str, str, str], np.ndarray]
    num_sampled_edges_dict: dict[tuple[str, str, str], list[int]]


class TargetOutput(NamedTuple):
    entity_pkey: pd.Series
    anchor_time: pd.Series
    target: pd.Series


class Sampler(ABC):
    r"""A base class to sample relational data (*i.e.*, subgraphs and
    ground-truth targets) from a custom backend.

    Args:
        graph: The graph.
        verbose: Whether to print verbose output.
    """

    def __init__(
        self,
        graph: 'Graph',
        verbose: bool | ProgressLogger = True,
    ) -> None:

        self._edge_types: list[tuple[str, str, str]] = []
        for edge in graph.edges:
            edge_type = (edge.src_table, edge.fkey, edge.dst_table)
            self._edge_types.append(edge_type)
            self._edge_types.append(SampledSubgraph.rev_edge_type(edge_type))

        # Source Table -> [(Foreign Key, Destination Table)]
        self._foreign_key_dict: dict[str, list[tuple[str, str]]] = {}
        # Destination Table -> [(Source Table, Foreign Key)]
        self._rev_foreign_key_dict: dict[str, list[tuple[str, str]]] = {}
        for table in graph.tables.values():
            self._foreign_key_dict[table.name] = []
            self._rev_foreign_key_dict[table.name] = []
        for src_table, fkey, dst_table in graph.edges:
            self._foreign_key_dict[src_table].append((fkey, dst_table))
            self._rev_foreign_key_dict[dst_table].append((src_table, fkey))

        self._primary_key_dict: dict[str, str] = {
            table.name: table._primary_key
            for table in graph.tables.values()
            if table._primary_key is not None
        }

        self._time_column_dict: dict[str, str] = {
            table.name: table._time_column
            for table in graph.tables.values()
            if table._time_column is not None
        }

        self._end_time_column_dict: dict[str, str] = {
            table.name: table._end_time_column
            for table in graph.tables.values()
            if table._end_time_column is not None
        }

        foreign_keys = {(edge.src_table, edge.fkey) for edge in graph.edges}
        self._table_stype_dict: dict[str, dict[str, Stype]] = {}
        for table in graph.tables.values():
            self._table_stype_dict[table.name] = {}
            for column in table.columns:
                if column == table.primary_key:
                    continue
                if (table.name, column.name) in foreign_keys:
                    continue
                self._table_stype_dict[table.name][column.name] = column.stype

        self._min_time_dict: dict[str, pd.Timestamp] = {}
        self._max_time_dict: dict[str, pd.Timestamp] = {}

    # Properties ##############################################################

    @property
    def edge_types(self) -> list[tuple[str, str, str]]:
        r"""All available edge types in the graph."""
        return self._edge_types

    @property
    def foreign_key_dict(self) -> dict[str, list[tuple[str, str]]]:
        r"""The foreign keys for all tables in the graph."""
        return self._foreign_key_dict

    @property
    def rev_foreign_key_dict(self) -> dict[str, list[tuple[str, str]]]:
        r"""The foreign key back references for all tables in the graph."""
        return self._rev_foreign_key_dict

    @property
    def primary_key_dict(self) -> dict[str, str]:
        r"""All available primary keys in the graph."""
        return self._primary_key_dict

    @property
    def time_column_dict(self) -> dict[str, str]:
        r"""All available time columns in the graph."""
        return self._time_column_dict

    @property
    def end_time_column_dict(self) -> dict[str, str]:
        r"""All available end time columns in the graph."""
        return self._end_time_column_dict

    @property
    def table_stype_dict(self) -> dict[str, dict[str, Stype]]:
        r"""The registered semantic types for all feature columns in all tables
        in the graph.
        """
        return self._table_stype_dict

    @property
    def sanitization_report(self) -> GraphSanitizationReport:
        r"""Graph sanitization diagnostics, when supported by the backend."""
        return GraphSanitizationReport.not_available()

    def validate_entity_references(
        self,
        table_name: str,
        entity_pkey: pd.Series,
    ) -> None:
        r"""Validate task references when supported by the backend."""
        return

    def get_min_time(
        self,
        table_names: list[str] | None = None,
    ) -> pd.Timestamp:
        r"""Returns the minimal timestamp in the union of a set of tables.

        Args:
            table_names: The set of tables.
        """
        if table_names is None or len(table_names) == 0:
            table_names = list(self.time_column_dict.keys())
        unknown = list(set(table_names) - set(self._min_time_dict.keys()))
        if len(unknown) > 0:
            min_max_time_dict = self._get_min_max_time_dict(unknown)
            for table_name, (min_time, max_time) in min_max_time_dict.items():
                self._min_time_dict[table_name] = min_time
                self._max_time_dict[table_name] = max_time
        return min(
            [self._min_time_dict[table] for table in table_names]
            + [pd.Timestamp.max]
        )

    def get_max_time(
        self,
        table_names: list[str] | None = None,
    ) -> pd.Timestamp:
        r"""Returns the maximum timestamp in the union of a set of tables.

        Args:
            table_names: The set of tables.
        """
        if table_names is None or len(table_names) == 0:
            table_names = list(self.time_column_dict.keys())
        unknown = list(set(table_names) - set(self._max_time_dict.keys()))
        if len(unknown) > 0:
            min_max_time_dict = self._get_min_max_time_dict(unknown)
            for table_name, (min_time, max_time) in min_max_time_dict.items():
                self._min_time_dict[table_name] = min_time
                self._max_time_dict[table_name] = max_time
        return max(
            [self._max_time_dict[table] for table in table_names]
            + [pd.Timestamp.min]
        )

    # Subgraph Sampling #######################################################

    def sample_subgraph(
        self,
        entity_table_names: tuple[str, ...],
        entity_pkey: pd.Series,
        anchor_time: pd.Series | Literal['entity'],
        num_neighbors: list[int],
        exclude_cols_dict: dict[str, list[str]] | None = None,
        random_seed: int | None = None,
        scored: np.ndarray | None = None,
    ) -> SampledSubgraph:
        r"""Samples distinct subgraphs for each entity primary key.

        Args:
            entity_table_names: The entity table names.
            entity_pkey: The primary keys to use as seed nodes.
            anchor_time: The anchor time of the subgraphs.
            num_neighbors: The number of neighbors to sample for each hop.
            exclude_cols_dict: The columns to exclude from the subgraph.
            random_seed: A manual seed for neighborhood sampling.
            scored: The positions in ``entity_pkey`` of the entities being
                scored, rather than given as context. A source that does not
                enforce its key could hold two rows for one of them, and
                scoring either would be a guess, so that is refused.

        Raises:
            ValueError: If ``exclude_cols_dict`` names a table or a feature
                column that does not exist. ``exclude_cols_dict`` is a public
                ``predict_task`` argument, so a typo in it is a caller mistake
                and must read as one. Deleting directly from the stype map
                reported it as a bare ``KeyError``, which the client could only
                classify as an internal failure.
        """
        # Exclude all columns that leak target information:
        table_stype_dict: dict[str, dict[str, Stype]] = self.table_stype_dict
        if exclude_cols_dict is not None:
            table_stype_dict = copy.deepcopy(table_stype_dict)
            for table_name, exclude_cols in exclude_cols_dict.items():
                if table_name not in table_stype_dict:
                    raise ValueError(
                        f"'exclude_cols_dict' names table '{table_name}', "
                        f'which is not in the graph. Available tables: '
                        f'{sorted(table_stype_dict)}'
                    )
                stype_dict = table_stype_dict[table_name]
                for column_name in exclude_cols:
                    if column_name not in stype_dict:
                        raise ValueError(
                            f"'exclude_cols_dict' names column "
                            f"'{column_name}' of table '{table_name}', which "
                            f'is not one of its feature columns (primary and '
                            f'foreign keys are never sent as features). '
                            f'Excludable columns: {sorted(stype_dict)}'
                        )
                    del stype_dict[column_name]

        # Collect all columns being used as features:
        columns_dict: dict[str, set[str]] = {
            table_name: set(stype_dict.keys())
            for table_name, stype_dict in table_stype_dict.items()
        }
        # Make sure to store primary key information for entity tables:
        for table_name in entity_table_names:
            columns_dict[table_name].add(self.primary_key_dict[table_name])

        if isinstance(anchor_time, pd.Series):
            anchor_time = to_naive_utc(anchor_time)
            if anchor_time.dtype != 'datetime64[ns]':
                anchor_time = anchor_time.astype('datetime64[ns]')

        out = self._sample_subgraph(
            entity_table_name=entity_table_names[0],
            entity_pkey=entity_pkey,
            anchor_time=anchor_time,
            columns_dict=columns_dict,
            num_neighbors=num_neighbors,
            random_seed=random_seed,
            scored=scored,
        )

        # Parse `SamplerOutput` into a sampled subgraph:
        subgraph = SampledSubgraph(
            anchor_time=out.anchor_time,
            tables={},
            links={},
        )

        for table_name, batch in out.batch_dict.items():
            if len(batch) == 0:
                continue

            primary_key: str | None = None
            if table_name in entity_table_names:
                primary_key = self.primary_key_dict[table_name]

            df = out.df_dict[table_name].reset_index(drop=True)
            if end_time_column := self.end_time_column_dict.get(table_name):
                # Set end time to NaT for all values greater than anchor time:
                assert table_name not in out.inverse_dict
                ser = df[end_time_column]
                mask = ser.astype(int).to_numpy() > out.anchor_time[batch]
                df.loc[mask, end_time_column] = pd.NaT

            stype_dict = table_stype_dict[table_name]
            for column_name, stype in stype_dict.items():
                if stype == Stype.text:
                    df[column_name] = _normalize_text(df[column_name])

            subgraph.tables[table_name] = SampledTable(
                df=df,
                row=out.inverse_dict.get(table_name),
                batch=batch,
                num_sampled_nodes=out.num_sampled_nodes_dict[table_name],
                stype_dict=stype_dict,
                primary_key=primary_key,
            )

        for edge_type in out.row_dict.keys():
            row: np.ndarray | None = out.row_dict[edge_type]
            col: np.ndarray | None = out.col_dict[edge_type]

            if row is None or col is None or len(row) == 0:
                continue

            # Do not store reverse edge type if it is an exact replica:
            rev_edge_type = SampledSubgraph.rev_edge_type(edge_type)
            if (
                rev_edge_type in subgraph.links
                and np.array_equal(row, out.col_dict[rev_edge_type])
                and np.array_equal(col, out.row_dict[rev_edge_type])
            ):
                subgraph.links[edge_type] = SampledLink(
                    layout=SampledEdgeLayout.REV,
                    row=None,
                    col=None,
                    num_sampled_edges=out.num_sampled_edges_dict[edge_type],
                )
                continue

            # Do not store non-informative edges:
            layout = SampledEdgeLayout.COO
            if np.array_equal(row, np.arange(len(row))):
                row = None
            if np.array_equal(col, np.arange(len(col))):
                col = None

            # Store in compressed representation if more efficient:
            num_cols = subgraph.tables[edge_type[2]].num_rows
            if (
                col is not None
                and len(col) > num_cols + 1
                and ((col[1:] - col[:-1]) >= 0).all()
            ):
                layout = SampledEdgeLayout.CSC
                colcount = np.bincount(col, minlength=num_cols)
                col = np.empty(num_cols + 1, dtype=col.dtype)
                col[0] = 0
                np.cumsum(colcount, out=col[1:])

            subgraph.links[edge_type] = SampledLink(
                layout=layout,
                row=row,
                col=col,
                num_sampled_edges=out.num_sampled_edges_dict[edge_type],
            )

        return subgraph

    # Predictive Query ########################################################

    def _get_query_columns_dict(
        self,
        query: ValidatedPredictiveQuery,
    ) -> dict[str, set[str]]:
        columns_dict: dict[str, set[str]] = defaultdict(set)
        for fqn in query.all_query_columns:
            table_name, column_name = fqn.split('.')
            if column_name == '*':
                continue
            columns_dict[table_name].add(column_name)
        return columns_dict

    def _get_query_time_offset_dict(
        self,
        query: ValidatedPredictiveQuery,
    ) -> dict[
        tuple[str, str, str],
        tuple[pd.DateOffset | None, pd.DateOffset],
    ]:
        time_offset_dict: dict[
            tuple[str, str, str],
            tuple[pd.DateOffset | None, pd.DateOffset],
        ] = {}

        def _add_time_offset(node: ASTNode) -> None:
            if isinstance(node, Aggregation):
                table_name = node.get_target_column_name().split('.')[0]

                edge_types = [
                    edge_type
                    for edge_type in self.edge_types
                    if edge_type[0] == table_name
                    and edge_type[2] == query.entity_table
                ]
                if len(edge_types) != 1:
                    raise ValueError(
                        f'Could not find a unique foreign key '
                        f"from table '{table_name}' to "
                        f"'{query.entity_table}'"
                    )
                assert node.aggr_time_range is not None
                if edge_types[0] not in time_offset_dict:
                    start = node.aggr_time_range.start_date_offset
                    end = node.aggr_time_range.end_date_offset
                else:
                    start, end = time_offset_dict[edge_types[0]]
                    start = min_date_offset(
                        start,
                        node.aggr_time_range.start_date_offset,
                    )
                    end = max_date_offset(
                        end,
                        node.aggr_time_range.end_date_offset,
                    )
                time_offset_dict[edge_types[0]] = (start, end)

            for child in node.children:
                _add_time_offset(child)

        _add_time_offset(query.target_ast)
        _add_time_offset(query.entity_ast)
        if query.whatif_ast is not None:
            _add_time_offset(query.whatif_ast)

        return time_offset_dict

    def _empty_sample_reason(
        self,
        query: ValidatedPredictiveQuery,
        anchor_time: pd.Timestamp | Literal['entity'],
        examples: Literal['context', 'test'] = 'context',
    ) -> str:
        r"""Why a sample came back empty, in terms of where the anchor fell.

        Each example needs the query's whole window of data after its anchor:
        context examples are drawn from before the anchor, test examples sit
        at it. An anchor at the end of the data leaves nowhere to draw from.
        The default anchor is the latest timestamp in the tables the query
        aggregates, which puts every default prediction at exactly that edge.

        Asking whether the query is too restrictive sends the caller to rewrite
        something that was never wrong, so state the anchor and the range it
        sits against and let them judge. An anchor with room after it for the
        whole window cannot be the explanation, though, and a caller sent to
        move it has been handed a remedy that cannot work: there the entity
        filter is what selected nothing.

        Before either, an entity table whose time column starts after the
        anchor has no entity to ask about at all. A column that records an
        edit, not a creation, does that. A later anchor is offered only when
        the data still holds a whole window after the first entity starts.
        """
        generic = ' Is your predictive query too restrictive?'
        if not isinstance(anchor_time, pd.Timestamp):
            return generic
        kind = 'Context' if examples == 'context' else 'Test'

        try:
            aggregations = query.get_all_target_aggregations()
            table_names = sorted(
                {
                    aggr.get_target_column_name().split('.')[0]
                    for aggr in aggregations
                }
            )
            max_time = self.get_max_time(table_names) if table_names else None
            min_time = self.get_min_time(table_names) if table_names else None
        except Exception:
            aggregations, max_time, min_time = [], None, None

        try:
            entity_table = query.entity_table
            column = self.time_column_dict.get(entity_table)
            earliest = None
            if column is not None:
                earliest = self.get_min_time([entity_table])
            # An all-null column reports pd.Timestamp.max: not a start time.
            starts_after = (
                earliest is not None
                and earliest != pd.Timestamp.max
                and earliest > anchor_time
            )
            later_fits = starts_after and (
                max_time is not None
                and _window_fits_after(aggregations, earliest, max_time)
            )
        except Exception:
            starts_after = later_fits = False
        if starts_after:
            later = f', or pass an anchor_time after {earliest}'
            later = later if later_fits else ''
            return (
                f' No row of {entity_table!r} existed at the anchor: it is '
                f'{anchor_time}, and the earliest {column!r} is {earliest}. '
                f'{kind} examples only ask about an entity that existed by '
                f'its anchor. If {column!r} does not record when each entity '
                f'began, set the time column of {entity_table!r} to None'
                f'{later}.'
            )

        if max_time is None:
            return generic

        listed = ', '.join(repr(name) for name in table_names)
        holds = 'holds' if len(table_names) == 1 else 'hold'
        facts = (
            f'it is {anchor_time}, and {listed} {holds} data from '
            f'{min_time} to {max_time}'
        )

        if _window_fits_after(aggregations, anchor_time, max_time):
            condition = _entity_filter_condition(query)
            if condition is None:
                return generic
            return (
                f' The anchor time is not the reason: {facts}, which leaves '
                f"room after it for the query's whole window. No entity "
                f'passed {condition!r}, so there was nothing to sample: check '
                f'that the values it compares against occur in the data, '
                f'including their case.'
            )

        where = (
            'come from before the anchor'
            if examples == 'context'
            else 'are asked at the anchor'
        )
        return (
            f' The anchor time is most likely the reason: {facts}. {kind} '
            f'examples {where} and each needs the '
            f"query's full window of data after it, so an anchor at or near "
            f'the end of the range leaves none to draw from. Pass an earlier '
            f'anchor_time, or shorten the window the query asks about.'
        )

    def sample_target(
        self,
        query: ValidatedPredictiveQuery,
        num_train_examples: int,
        train_anchor_time: pd.Timestamp | Literal['entity'],
        num_train_trials: int,
        num_test_examples: int,
        test_anchor_time: pd.Timestamp | Literal['entity'],
        num_test_trials: int,
        random_seed: int | None = None,
    ) -> tuple[TargetOutput, TargetOutput]:
        r"""Samples ground-truth targets given a predictive query, split into
        training and test set.

        Args:
            query: The predictive query.
            num_train_examples: How many training examples to produce.
            train_anchor_time: The anchor timestamp for the training set.
                If set to ``"entity"``, will use the timestamp of the entity.
            num_train_trials: The number of training examples to try before
                aborting.
            num_test_examples: How many test examples to produce.
            test_anchor_time: The anchor timestamp for the test set.
                If set to ``"entity"``, will use the timestamp of the entity.
            num_test_trials: The number of test examples to try before
                aborting.
            random_seed: A manual seed for generating pseudo-random numbers.
        """
        rng = np.random.default_rng(random_seed)

        is_forecasting = query.problem_type == ProblemType.FORECAST
        forecast_num = query.num_forecasts

        if num_train_examples == 0 or num_train_trials == 0:
            num_train_examples = num_train_trials = 0
        if num_test_examples == 0 or num_test_trials == 0:
            num_test_examples = num_test_trials = 0

        # 1. Collect information on what to query #############################
        columns_dict = self._get_query_columns_dict(query)
        time_offset_dict = self._get_query_time_offset_dict(query)
        for table_name, _, _ in time_offset_dict.keys():
            columns_dict[table_name].add(self.time_column_dict[table_name])

        # 2. Sample random rows from entity table #############################
        shared_train_test = (query.query_type == QueryType.STATIC) & (
            train_anchor_time == test_anchor_time
        )
        if shared_train_test:
            num_entity_rows = num_train_trials + num_test_trials
        else:
            num_entity_rows = max(num_train_trials, num_test_trials)
        assert num_entity_rows > 0

        entity_ids: list | None = None
        if is_forecasting and query.rfm_entity_ids is not None:
            entity_ids = query.get_rfm_entity_id_list()

        entity_columns = columns_dict[query.entity_table]
        if column_name := self.time_column_dict.get(query.entity_table):
            entity_columns = entity_columns | {column_name}
        if column_name := self.end_time_column_dict.get(query.entity_table):
            entity_columns = entity_columns | {column_name}

        entity_df = self._sample_entity_table(
            table_name=query.entity_table,
            columns=entity_columns,
            num_rows=num_entity_rows,
            random_seed=random_seed,
            entity_ids=entity_ids,
        )

        if len(entity_df) == 0:
            raise ValueError(
                f'Failed to find any rows in the entity table '
                f"'{query.entity_table}'."
            )
        # A key the source does not enforce can repeat. Two rows of one entity
        # would be two examples at one anchor time, whose labels can disagree.
        entity_df = entity_df.drop_duplicates(
            subset=[self.primary_key_dict[query.entity_table]],
            ignore_index=True,
        )

        entity_pkey = entity_df[self.primary_key_dict[query.entity_table]]
        entity_time: pd.Series | None = None
        if column_name := self.time_column_dict.get(query.entity_table):
            entity_time = entity_df[column_name]
        entity_end_time: pd.Series | None = None
        if column_name := self.end_time_column_dict.get(query.entity_table):
            entity_end_time = entity_df[column_name]

        def get_valid_entity_index(
            time: pd.Timestamp | Literal['entity'],
            max_size: int | None = None,
        ) -> np.ndarray:

            if time == 'entity':
                index: np.ndarray = np.arange(len(entity_pkey))
            elif entity_time is None and entity_end_time is None:
                index = np.arange(len(entity_pkey))
            else:
                mask: np.ndarray | None = None
                if entity_time is not None:
                    mask = (entity_time <= time).to_numpy()
                if entity_end_time is not None:
                    _mask = (entity_end_time > time).to_numpy()
                    _mask = _mask | entity_end_time.isna().to_numpy()
                    mask = _mask if mask is None else mask & _mask
                assert mask is not None
                index = mask.nonzero()[0]

            rng.shuffle(index)

            if max_size is not None:
                index = index[:max_size]

            return index

        # 3. Build training and test candidates ###############################
        # A temporal query walks `train_anchor_time` backwards below, so the
        # anchor a diagnosis quotes has to be kept aside here: the caller can
        # only act on the one they passed.
        requested_train_anchor_time = train_anchor_time
        train_index = test_index = np.array([], dtype=np.int64)
        train_time = test_time = pd.Series([], dtype='datetime64[ns]')

        # Pre-compute temporal setup:
        is_temporal = query.query_type == QueryType.TEMPORAL
        aggr_table_names: list[str] | None = None
        step_offset: pd.Timedelta | None = None
        if is_temporal:
            aggr_table_names = [
                aggr.get_target_column_name().split('.')[0]
                for aggr in query.get_all_target_aggregations()
            ]
            assert query.target_timeframe is not None
            assert query.target_timeframe.timeframe is not None
            step_offset = query.target_timeframe.timeframe

        # For forecasting, pick a single entity to use across all time steps:
        forecast_entity_idx: int | None = None
        if is_forecasting:
            # If the entity is user-provided, we use it.
            if query.rfm_entity_ids is not None:
                entity_ids = query.get_rfm_entity_id_list()
                matches = np.where(entity_pkey.to_numpy() == entity_ids[0])[0]
                if len(matches) == 0:
                    raise RuntimeError(
                        f"Entity '{entity_ids[0]}' not found in entity table "
                        f"'{query.entity_table}'"
                    )
                forecast_entity_idx = int(matches[0])
            else:
                valid_index = get_valid_entity_index(train_anchor_time)
                if len(valid_index) == 0:
                    raise RuntimeError('No valid entities for forecasting')
                forecast_entity_idx = int(valid_index[0])

        if shared_train_test:
            train_index = get_valid_entity_index(train_anchor_time)
            if train_anchor_time == 'entity':  # Sort by timestamp:
                assert entity_time is not None
                train_time = entity_time.iloc[train_index]
                train_time = train_time.reset_index(drop=True)
                train_time = train_time.sort_values(ascending=False)
                perm = train_time.index.to_numpy()
                train_index = train_index[perm]
                train_time = train_time.reset_index(drop=True)
            else:
                train_time = to_ser(train_anchor_time, size=len(train_index))
        else:
            if num_test_examples > 0:
                if is_forecasting:
                    # Generate forecast_num timestamps going backward:
                    assert test_anchor_time != 'entity'
                    assert step_offset is not None
                    assert forecast_entity_idx is not None
                    test_indices_list: list[int] = []
                    test_times_list: list[pd.Timestamp] = []
                    current_time = test_anchor_time
                    for _ in range(forecast_num):
                        test_indices_list.append(forecast_entity_idx)
                        test_times_list.append(current_time)
                        current_time += step_offset
                    test_index = np.array(test_indices_list, dtype=np.int64)
                    test_time = pd.Series(
                        test_times_list, dtype='datetime64[ns]'
                    )
                else:
                    test_index = get_valid_entity_index(  #
                        test_anchor_time, max_size=num_test_trials
                    )
                    assert test_anchor_time != 'entity'
                    test_time = to_ser(test_anchor_time, len(test_index))

            if query.query_type == QueryType.STATIC and num_train_examples > 0:
                train_index = get_valid_entity_index(  #
                    train_anchor_time, max_size=num_train_trials
                )
                assert train_anchor_time != 'entity'
                train_time = to_ser(train_anchor_time, len(train_index))
            elif is_temporal and num_train_examples > 0:
                assert aggr_table_names is not None
                assert step_offset is not None

                train_indices: list[np.ndarray] = []
                train_times: list[pd.Series] = []
                while True:
                    if is_forecasting:
                        assert forecast_entity_idx is not None
                        train_index = np.array(
                            [forecast_entity_idx], dtype=np.int64
                        )
                    else:
                        train_index = get_valid_entity_index(  #
                            train_anchor_time, max_size=num_train_trials
                        )
                    assert train_anchor_time != 'entity'
                    train_time = to_ser(train_anchor_time, len(train_index))
                    train_indices.append(train_index)
                    train_times.append(train_time)
                    if sum(len(x) for x in train_indices) >= num_train_trials:
                        break
                    train_anchor_time -= step_offset
                    if train_anchor_time < self.get_min_time(aggr_table_names):
                        break
                if is_forecasting:
                    # Reverse so they're in chronological order:
                    train_times.reverse()
                    train_indices.reverse()
                train_index = np.concatenate(train_indices, axis=0)
                train_index = train_index[:num_train_trials]
                train_time = pd.concat(train_times, axis=0, ignore_index=True)
                train_time = train_time.iloc[:num_train_trials]

        # 4. Sample training and test labels ##################################
        if len(train_index) == 0 and len(test_index) == 0:
            # Nothing to label, and a warehouse sampler bounds its window by
            # the latest end time of the entities asked about: NaN for none.
            train_y = test_y = pd.Series([], dtype=float)
            train_mask = test_mask = np.zeros(0, dtype=bool)
        else:
            train_y, train_mask, test_y, test_mask = self._sample_target(
                query=query,
                entity_df=entity_df,
                train_index=train_index,
                train_time=train_time,
                num_train_examples=(
                    num_train_examples + num_test_examples
                    if shared_train_test
                    else num_train_examples
                ),
                test_index=test_index,
                test_time=test_time,
                num_test_examples=(
                    0 if shared_train_test else num_test_examples
                ),
                columns_dict=columns_dict,
                time_offset_dict=time_offset_dict,
            )

        # 5. Post-processing ##################################################
        if shared_train_test:
            num_examples = num_train_examples + num_test_examples
            train_index = train_index[train_mask][:num_examples]
            train_time = train_time.iloc[train_mask].iloc[:num_examples]
            train_y = train_y.iloc[:num_examples]

            _num_test = num_test_examples
            _num_train = min(num_train_examples, 1000)
            if (
                num_test_examples > 0
                and num_train_examples > 0
                and len(train_y) < num_examples
                and len(train_y) < _num_test + _num_train
            ):
                # Not enough labels to satisfy requested split without losing
                # large number of training examples:
                _num_test = len(train_y) - _num_train
                if _num_test < _num_train:  # Fallback to 50/50 split:
                    _num_test = len(train_y) // 2

            test_index = train_index[:_num_test]
            test_pkey = entity_pkey.iloc[test_index]
            test_time = train_time.iloc[:_num_test]
            test_y = train_y.iloc[:_num_test]

            train_index = train_index[_num_test:]
            train_pkey = entity_pkey.iloc[train_index]
            train_time = train_time.iloc[_num_test:]
            train_y = train_y.iloc[_num_test:]
        else:
            train_index = train_index[train_mask][:num_train_examples]
            train_pkey = entity_pkey.iloc[train_index]
            train_time = train_time.iloc[train_mask].iloc[:num_train_examples]
            train_y = train_y.iloc[:num_train_examples]

            test_index = test_index[test_mask][:num_test_examples]
            test_pkey = entity_pkey.iloc[test_index]
            test_time = test_time.iloc[test_mask].iloc[:num_test_examples]
            test_y = test_y.iloc[:num_test_examples]

        train_pkey = train_pkey.reset_index(drop=True)
        train_time = train_time.reset_index(drop=True)
        train_y = train_y.reset_index(drop=True)
        test_pkey = test_pkey.reset_index(drop=True)
        test_time = test_time.reset_index(drop=True)
        test_y = test_y.reset_index(drop=True)

        # ValueError, not RuntimeError: the caller can fix this by moving the
        # anchor or widening the window, and the client maps ValueError to
        # INVALID_REQUEST while an unrecognised error becomes INTERNAL_ERROR,
        # which reads as a fault in the service rather than in the request.
        if num_train_examples > 0 and len(train_y) == 0:
            raise ValueError(
                'Failed to collect any context examples.'
                + self._empty_sample_reason(query, requested_train_anchor_time)
            )

        if num_test_examples > 0 and len(test_y) == 0:
            raise ValueError(
                'Failed to collect any test examples for evaluation.'
                + self._empty_sample_reason(query, test_anchor_time, 'test')
            )

        global _coverage_warned
        if (
            num_train_examples > 0
            and not _coverage_warned
            and len(entity_df) >= num_entity_rows
            and len(train_y) < num_train_examples // 2
        ):
            _coverage_warned = True
            warnings.warn(
                f'Failed to collect {num_train_examples:,} context '
                f'examples within {num_train_trials:,} candidates. '
                f'To improve coverage, consider increasing the '
                f'number of PQ iterations using the '
                f"'max_pq_iterations' option. This warning will not "
                f'be shown again in this run.'
            )

        if (
            num_test_examples > 0
            and not _coverage_warned
            and len(entity_df) >= num_entity_rows
            and len(test_y) < num_test_examples // 2
        ):
            _coverage_warned = True
            warnings.warn(
                f'Failed to collect {num_test_examples:,} test '
                f'examples within {num_test_trials:,} candidates. '
                f'To improve coverage, consider increasing the '
                f'number of PQ iterations using the '
                f"'max_pq_iterations' option. This warning will not "
                f'be shown again in this run.'
            )

        return (
            TargetOutput(train_pkey, train_time, train_y),
            TargetOutput(test_pkey, test_time, test_y),
        )

    def sample_lagged_target(
        self,
        query: ValidatedPredictiveQuery,
        entity_pkey: pd.Series,
        anchor_time: pd.Series,
        lag_timesteps: int,
    ) -> pd.DataFrame:
        r"""Samples lag targets given a predictive query.

        Args:
            query: The predictive query.
            entity_pkey: The primary keys.
            anchor_time: The anchor time.
            lag_timesteps: The number of lagged timesteps.
        """
        query = query.to_autoregressive_query()

        time_offset_dict = {
            edge_type: (start, lag_timesteps * end)
            for edge_type, (start, end) in self._get_query_time_offset_dict(  #
                query
            ).items()
        }
        columns_dict = self._get_query_columns_dict(query)
        for table_name, _, _ in time_offset_dict:
            columns_dict[table_name].add(self.time_column_dict[table_name])
        assert query.target_timeframe is not None
        offset = query.target_timeframe.end_date_offset
        anchor_time = anchor_time - lag_timesteps * offset

        feat_dict, time_dict, batch_dict = self._sample_query_data(
            entity_table_name=query.entity_table,
            entity_pkey=entity_pkey,
            anchor_time=anchor_time,
            columns_dict=columns_dict,
            time_offset_dict=time_offset_dict,
        )

        y_lag_dict: dict[str, pd.Series] = {}
        for i in range(lag_timesteps)[::-1]:
            y_lag = PQueryPandasExecutor().execute(
                query=query,
                feat_dict=feat_dict,
                time_dict=time_dict,
                batch_dict=batch_dict,
                anchor_time=anchor_time + i * offset,
                filter_na=False,
            )[0]
            y_lag_dict[f'__kumo_arl{lag_timesteps - 1 - i}__'] = y_lag

        return pd.DataFrame(y_lag_dict)

    # Abstract Methods ########################################################

    @property
    @abstractmethod
    def backend(self) -> DataBackend:
        r"""The data backend of this sampler."""

    @abstractmethod
    def _get_min_max_time_dict(
        self,
        table_names: list[str],
    ) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        r"""Returns the minimum and maximum timestamps for a set of tables.

        Args:
            table_names: The tables.
        """

    @abstractmethod
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
        r"""Samples distinct subgraphs for each entity primary key.

        Args:
            entity_table_name: The entity table name.
            entity_pkey: The primary keys to use as seed nodes.
            anchor_time: The anchor time of the subgraphs.
            columns_dict: The columns to return for each table.
            num_neighbors: The number of neighbors to sample for each hop.
            random_seed: A manual seed for neighborhood sampling.
            scored: The positions in ``entity_pkey`` of the entities being
                scored. Each has to match exactly one row.
        """

    @abstractmethod
    def _sample_entity_table(
        self,
        table_name: str,
        columns: set[str],
        num_rows: int,
        random_seed: int | None = None,
        entity_ids: list | None = None,
    ) -> pd.DataFrame:
        r"""Returns a random sample of rows from the entity table.

        Args:
            table_name: The table.
            columns: The columns to return.
            num_rows: Maximum number of rows to return. Can be smaller in case
                the entity table contains less rows.
            random_seed: A manual seed for generating pseudo-random numbers.
            entity_ids: If provided, only return rows matching these primary
                key values (ignores ``num_rows``).
        """

    @abstractmethod
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
        r"""Sample feature rows needed to compute query labels on the fly.

        Returns per-table sampled frames, per-table anchor times, and
        per-table batch assignments for the tables involved in the query.
        """

    @abstractmethod
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
        r"""Sample ground-truth labels from train and test candidate rows.

        Returns labels and row selections for the requested train and test
        examples, using ``columns_dict`` and ``time_offset_dict`` to fetch any
        related rows needed by the predictive query.
        """


# Helper Functions ############################################################

PUNCTUATION = re.compile(r'[\'\"\.,\(\)\!\?\;\:]')
MULTISPACE = re.compile(r'\s+')


def _normalize_text(
    ser: pd.Series,
    max_words: int | None = 50,
) -> pd.Series:
    r"""Normalizes text into a list of lower-case words.

    Args:
        ser: The :class:`pandas.Series` to normalize.
        max_words: The maximum number of words to return.
            This will auto-shrink any large text column to avoid blowing up
            context size.
    """
    if len(ser) == 0 or pd.api.types.is_list_like(ser.iloc[0]):
        return ser

    def normalize_fn(line: str) -> list[str]:
        line = PUNCTUATION.sub(' ', line)
        line = re.sub(r'<br\s*/?>', ' ', line)  # Handle <br /> or <br>
        line = MULTISPACE.sub(' ', line)
        words = line.split()
        if max_words is not None:
            words = words[:max_words]
        return words

    ser = ser.fillna('').astype(str)

    if max_words is not None:
        # We estimate the number of words as 5 characters + 1 space in an
        # English text on average. We need this pre-filter here, as word
        # splitting on a giant text can be very expensive:
        ser = ser.str[: 6 * max_words]

    ser = ser.str.lower()
    ser = ser.map(normalize_fn)

    return ser


def _window_fits_after(
    aggregations: list[Aggregation],
    anchor_time: pd.Timestamp,
    max_time: pd.Timestamp,
) -> bool:
    r"""Whether the data still reaches past every aggregation's window.

    An aggregation whose window is unknown counts as not fitting, so a query
    the diagnosis cannot read keeps the anchor explanation rather than ruling
    it out on no evidence.
    """
    ends = []
    for aggr in aggregations:
        window = getattr(aggr, 'aggr_time_range', None)
        if window is None:
            return False
        ends.append(window.end_date_offset)
    if not ends:
        return False
    return anchor_time + max_date_offset(*ends) <= max_time


def _entity_filter_condition(query: ValidatedPredictiveQuery) -> str | None:
    if not isinstance(query.entity_ast, Filter):
        return None
    return query.entity_ast.condition.to_string()


def _pooled_offset(
    offsets: Sequence[pd.DateOffset], *, earliest: bool
) -> pd.DateOffset:
    r"""An offset that bounds every one of ``offsets`` at every anchor.

    One of them is returned where one of them bounds the rest: adding it to any
    timestamp lands on the right side of adding any other, which is true when
    the most it can span still falls short of the least any other can. An
    offset equal to the candidate is one of them and is skipped, or two
    written the same way would be found to bound each other in neither
    direction and a calendar window would pool with itself into a rigid one.

    Otherwise none of them does, and the bound is built. A month is shorter
    than thirty days from some anchors and longer from others, so neither is a
    bound for the pair; what bounds both is the extreme of what either can
    span. That rigid span works both for the caller that converts it to seconds
    and the caller that adds it to a timestamp.
    """
    assert len(offsets) > 0
    bound: TimeBound = 'earliest' if earliest else 'latest'
    opposite: TimeBound = 'latest' if earliest else 'earliest'
    for candidate in offsets:
        mine = date_offset_to_seconds(candidate, opposite)
        if all(
            mine <= date_offset_to_seconds(other, bound)
            if earliest
            else mine >= date_offset_to_seconds(other, bound)
            for other in offsets
            if other != candidate
        ):
            return candidate
    spans = [date_offset_to_seconds(offset, bound) for offset in offsets]
    return pd.DateOffset(seconds=min(spans) if earliest else max(spans))


def min_date_offset(*args: pd.DateOffset | None) -> pd.DateOffset | None:
    r"""Return the offset that reaches furthest back for any anchor."""
    offsets = [arg for arg in args if arg is not None]

    if len(offsets) != len(args):
        return None

    return _pooled_offset(offsets, earliest=True)


def max_date_offset(*args: pd.DateOffset) -> pd.DateOffset:
    r"""Return the offset that reaches furthest forward for any anchor."""
    return _pooled_offset(list(args), earliest=False)


def to_ser(value: Any, size: int) -> pd.Series:
    dtype = 'datetime64[ns]' if isinstance(value, pd.Timestamp) else None
    return pd.Series([value], dtype=dtype).repeat(size).reset_index(drop=True)
