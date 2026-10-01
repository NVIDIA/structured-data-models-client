# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1


import numpy as np
import pandas as pd
from predictive_query.schema.task import TaskType
from predictive_query.schema.typing import StrEnum, Stype
from pydantic.dataclasses import dataclass

REVERSE_RELATION_PREFIX = '__###REV###__'


class SampledEdgeLayout(StrEnum):
    COO = 'COO'
    CSC = 'CSC'
    REV = 'REV'


@dataclass(config={'arbitrary_types_allowed': True})
class SampledTable:
    df: pd.DataFrame
    row: np.ndarray | None
    batch: np.ndarray
    num_sampled_nodes: list[int]
    stype_dict: dict[str, Stype]
    primary_key: str | None

    @property
    def num_rows(self) -> int:
        return sum(self.num_sampled_nodes)


@dataclass(config={'arbitrary_types_allowed': True})
class SampledLink:
    layout: SampledEdgeLayout
    row: np.ndarray | None
    col: np.ndarray | None
    num_sampled_edges: list[int]

    def __post_init__(self) -> None:
        if self.layout == SampledEdgeLayout.REV:
            assert self.row is None and self.col is None

    @property
    def num_edges(self) -> int:
        return sum(self.num_sampled_edges)


@dataclass(config={'arbitrary_types_allowed': True})
class SampledSubgraph:
    anchor_time: np.ndarray
    tables: dict[str, SampledTable]
    links: dict[tuple[str, str, str], SampledLink]

    @property
    def batch_size(self) -> int:
        return len(self.anchor_time)

    @property
    def num_hops(self) -> int:
        return max(
            [len(link.num_sampled_edges) for link in self.links.values()] + [0]
        )

    @staticmethod
    def rev_edge_type(edge_type: tuple[str, str, str]) -> tuple[str, str, str]:
        src, rel, dst = edge_type
        if rel.startswith(REVERSE_RELATION_PREFIX):
            return (dst, rel[len(REVERSE_RELATION_PREFIX) :], src)
        return (dst, f'{REVERSE_RELATION_PREFIX}{rel}', src)


@dataclass(config={'arbitrary_types_allowed': True})
class PredictionContext:
    task_type: TaskType
    entity_table_names: tuple[str, ...]
    sampled_subgraph: SampledSubgraph
    y_train: pd.Series
    y_test: pd.Series | None
    task_table: SampledTable | None = None
    top_k: int | None = None
    step_size: int | None = None
    num_forecasts: int = 1

    def __post_init__(self) -> None:
        if len(self.entity_table_names) == 0:
            raise ValueError(
                "'entity_table_names' needs to at least contain "
                'one entity table name'
            )

        if self.task_table is not None:
            assert self.task_table.row is None
            assert self.task_table.num_sampled_nodes == []
            assert self.task_table.primary_key is None

    @property
    def num_train(self) -> int:
        return len(self.y_train)

    @property
    def num_test(self) -> int:
        return self.sampled_subgraph.batch_size - self.num_train
