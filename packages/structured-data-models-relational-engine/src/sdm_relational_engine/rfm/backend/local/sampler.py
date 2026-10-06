# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
from predictive_query.query import ValidatedPredictiveQuery

from sdm_relational_engine.rfm.backend.local import LocalGraphStore
from sdm_relational_engine.rfm.base import DataBackend, Sampler, SamplerOutput
from sdm_relational_engine.rfm.base.utils import (
    date_offset_to_seconds,
)
from sdm_relational_engine.rfm.diagnostics import GraphSanitizationReport
from sdm_relational_engine.rfm.query.pandas_executor import PQueryPandasExecutor
from sdm_relational_engine.utils.progress_logger import ProgressLogger

if TYPE_CHECKING:
    from sdm_relational_engine.rfm import Graph


class LocalSampler(Sampler):
    def __init__(
        self,
        graph: 'Graph',
        verbose: bool | ProgressLogger = True,
    ) -> None:
        super().__init__(graph=graph, verbose=verbose)

        import sdm_relational_engine.relationallib as relationallib

        self._graph_store = LocalGraphStore(graph, verbose)
        self._graph_sampler = relationallib.NeighborSampler(
            list(self.table_stype_dict.keys()),
            self.edge_types,
            {
                '__'.join(edge_type): colptr
                for edge_type, colptr in self._graph_store.colptr_dict.items()
            },
            {
                '__'.join(edge_type): row
                for edge_type, row in self._graph_store.row_dict.items()
            },
            self._graph_store.time_dict,
        )

    @property
    def backend(self) -> DataBackend:
        return cast(DataBackend, DataBackend.LOCAL)

    @property
    def sanitization_report(self) -> GraphSanitizationReport:
        return self._graph_store.sanitization_report

    def validate_entity_references(
        self,
        table_name: str,
        entity_pkey: pd.Series,
    ) -> None:
        self._graph_store.validate_entity_references(
            table_name,
            entity_pkey,
        )

    def _get_min_max_time_dict(
        self,
        table_names: list[str],
    ) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        return {
            key: value
            for key, value in self._graph_store.min_max_time_dict.items()
            if key in table_names
        }

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
        # The graph store keeps one row per key, so no id can match two.
        del scored

        index = self._graph_store.get_node_id(entity_table_name, entity_pkey)

        if isinstance(anchor_time, pd.Series):
            time = anchor_time.astype(int).to_numpy() // 1000**3  # to seconds
        else:
            assert anchor_time == 'entity'
            time = self._graph_store.time_dict[entity_table_name][index]

        if random_seed is not None:
            self._graph_sampler.seed(random_seed)

        (
            row_dict,
            col_dict,
            node_dict,
            batch_dict,
            num_sampled_nodes_dict,
            num_sampled_edges_dict,
        ) = self._graph_sampler.sample(
            {
                '__'.join(edge_type): num_neighbors
                for edge_type in self.edge_types
            },
            {},
            entity_table_name,
            index,
            time,
        )

        df_dict: dict[str, pd.DataFrame] = {}
        inverse_dict: dict[str, np.ndarray] = {}
        for table_name, node in node_dict.items():
            df = self._graph_store.df_dict[table_name]
            columns = columns_dict[table_name]
            if (
                self.end_time_column_dict.get(table_name, None) in columns
                or len(columns) == 0
            ):
                df = df.iloc[node]
            else:
                # Only store unique rows in `df` above a certain threshold:
                unique_node, inverse = np.unique(node, return_inverse=True)
                if len(node) > 1.05 * len(unique_node):
                    df = df.iloc[unique_node]
                    inverse_dict[table_name] = inverse
                else:
                    df = df.iloc[node]
            df = df.reset_index(drop=True)
            df = df[sorted(columns, key=df.columns.get_loc)]
            df_dict[table_name] = df

        num_sampled_nodes_dict = {
            table_name: num_sampled_nodes.tolist()
            for table_name, num_sampled_nodes in num_sampled_nodes_dict.items()
        }

        row_dict = {
            edge_type: row_dict['__'.join(edge_type)]
            for edge_type in self.edge_types
        }
        col_dict = {
            edge_type: col_dict['__'.join(edge_type)]
            for edge_type in self.edge_types
        }
        num_sampled_edges_dict = {
            edge_type: num_sampled_edges_dict['__'.join(edge_type)].tolist()
            for edge_type in self.edge_types
        }

        return SamplerOutput(
            anchor_time=time * 1000**3,  # to nanoseconds
            df_dict=df_dict,
            inverse_dict=inverse_dict,
            batch_dict=batch_dict,
            num_sampled_nodes_dict=num_sampled_nodes_dict,
            row_dict=row_dict,
            col_dict=col_dict,
            num_sampled_edges_dict=num_sampled_edges_dict,
        )

    def _sample_entity_table(
        self,
        table_name: str,
        columns: set[str],
        num_rows: int,
        random_seed: int | None = None,
        entity_ids: list | None = None,
    ) -> pd.DataFrame:
        pkey_map = self._graph_store.pkey_map_dict[table_name]
        if entity_ids is not None:
            entity_set = set(entity_ids)
            mask = np.fromiter(
                (v in entity_set for v in pkey_map.index),
                dtype=bool,
                count=len(pkey_map),
            )
            pkey_map = pkey_map[mask]
        elif len(pkey_map) > num_rows:
            pkey_map = pkey_map.sample(
                n=num_rows,
                random_state=random_seed,
                ignore_index=True,
            )
        df = self._graph_store.df_dict[table_name]
        df = df.iloc[pkey_map['arange']][
            sorted(columns, key=df.columns.get_loc)
        ]
        return df

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
        num_hops = 1 if len(time_offset_dict) > 0 else 0
        num_neighbors_dict: dict[str, list[int]] = {}
        unix_time_offset_dict: dict[str, list[list[int | None]]] = {}
        for edge_type, (start, end) in time_offset_dict.items():
            unix_time_offset_dict['__'.join(edge_type)] = [
                [
                    date_offset_to_seconds(start, 'earliest')
                    if start is not None
                    else None,
                    date_offset_to_seconds(end, 'latest'),
                ]
            ]
        for edge_type in set(self.edge_types) - set(time_offset_dict.keys()):
            num_neighbors_dict['__'.join(edge_type)] = [0] * num_hops

        _, _, node_dict, batch_dict, _, _ = self._graph_sampler.sample(
            num_neighbors_dict,
            unix_time_offset_dict,
            entity_table_name,
            self._graph_store.get_node_id(entity_table_name, entity_pkey),
            anchor_time.astype(int).to_numpy() // 1000**3,  # to seconds
        )

        feat_dict: dict[str, pd.DataFrame] = {}
        time_dict: dict[str, pd.Series] = {}
        for table_name, columns in columns_dict.items():
            df = self._graph_store.df_dict[table_name]
            df = df.iloc[node_dict[table_name]].reset_index(drop=True)
            df = df[sorted(columns, key=df.columns.get_loc)]
            feat_dict[table_name] = df

            time_column = self.time_column_dict.get(table_name)
            if time_column in columns:
                time_dict[table_name] = df[time_column]

        return feat_dict, time_dict, batch_dict

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
            entity_pkey=entity_df[self.primary_key_dict[query.entity_table]],
            index=train_index,
            anchor_time=train_time,
            num_examples=num_train_examples,
            columns_dict=columns_dict,
            time_offset_dict=time_offset_dict,
        )

        test_y, test_mask = self._sample_target_set(
            query=query,
            entity_pkey=entity_df[self.primary_key_dict[query.entity_table]],
            index=test_index,
            anchor_time=test_time,
            num_examples=num_test_examples,
            columns_dict=columns_dict,
            time_offset_dict=time_offset_dict,
        )

        return train_y, train_mask, test_y, test_mask

    # Helper Methods ##########################################################

    def _sample_target_set(
        self,
        query: ValidatedPredictiveQuery,
        entity_pkey: pd.Series,
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
            feat_dict, time_dict, batch_dict = self._sample_query_data(
                entity_table_name=query.entity_table,
                entity_pkey=entity_pkey.iloc[index[start : start + batch_size]],
                anchor_time=anchor_time.iloc[start : start + batch_size],
                columns_dict=columns_dict,
                time_offset_dict=time_offset_dict,
            )

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


# Helper Functions ############################################################
