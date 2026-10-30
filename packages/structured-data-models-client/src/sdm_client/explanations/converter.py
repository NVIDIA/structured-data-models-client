# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""sdm's gradient tensors as the ``sdm_gradient_v1`` record a NIM sends per row."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from sdm_client.errors import StructuredDataError


class GradientConverter:
    r"""sdm's gradient tensors as the ``sdm_gradient_v1`` record a NIM sends per row."""

    FORMAT = 'sdm_gradient_v1'

    @staticmethod
    def class_outputs(labels: Sequence[Any]) -> list[dict[str, Any]]:
        r"""One probability output per class, named by the caller's label text."""
        return [
            {'kind': 'probability', 'class': str(label)} for label in labels
        ]

    @staticmethod
    def regression_outputs(
        statistic: str, levels: Sequence[float]
    ) -> list[dict[str, Any]]:
        r"""The point statistic, then one output per requested quantile level."""
        return [
            {'kind': 'prediction', 'statistic': statistic},
            *({'kind': 'quantile', 'level': level} for level in levels),
        ]

    @staticmethod
    def entity_rows(x: Any, related_tables: Any) -> dict[str, list[int]]:
        r"""Each related row's predict row per sdm's task graph; ``-1`` if none."""
        from sdm.models.kumo.relational.task import TaskGraph

        graph = TaskGraph.from_input(x=x, related_tables=related_tables)
        return {
            name: assignment.tolist()
            for name, assignment in graph.task_row_by_table.items()
        }

    @classmethod
    def convert(
        cls,
        outputs: Sequence[Mapping[str, Any]],
        instance: Any,
        related: Any = None,
        *,
        rows: int,
        related_rows: Mapping[str, int] | None = None,
        owners: Mapping[str, Sequence[int]] | None = None,
    ) -> list[dict[str, Any]]:
        r"""One record per predict row from ``GradientExplainer`` attributions."""
        count = len(outputs)
        sizes = dict(related_rows or {})
        tables = related.tables if related is not None else {}
        if set(tables) != set(sizes):
            raise StructuredDataError(
                'incompatible explanation related tables', code='INTERNAL_ERROR'
            )
        instance_block = cls._gradient_block(instance, rows, count)
        blocks = {
            name: cls._gradient_block(table, sizes[name], count)
            for name, table in tables.items()
        }
        grouped: dict[str, list[list[int]]] = {}
        for name in blocks:
            grouped[name] = [[] for _ in range(rows)]
            for position, owner in enumerate((owners or {})[name]):
                if owner >= 0:
                    grouped[name][owner].append(position)
        return [
            {
                'format': cls.FORMAT,
                'details': {
                    'input_space': 'preprocessed_numerical',
                    'outputs': [dict(output) for output in outputs],
                    'instance_table': cls._rows_block(*instance_block, [row]),
                    'related_tables': {
                        name: cls._rows_block(
                            columns, values, grouped[name][row]
                        )
                        for name, (columns, values) in blocks.items()
                    },
                },
            }
            for row in range(rows)
        ]

    @staticmethod
    def _gradient_block(
        table: Any, rows: int, count: int
    ) -> tuple[list[str], np.ndarray]:
        r"""A table's numerical columns and gradients ``[output, row, column]``."""
        # By stype, as the NIM reads it: `numerical_columns` is newer than the
        # released sdm 0.2.1 this must run against (see CHANGELOG).
        columns: list[str] = []
        for stype, names in table.columns.items():
            if getattr(stype, 'value', stype) == 'numerical':
                columns = list(names)
                break
        values = table.numerical.detach().float().cpu().numpy()
        if values.ndim < 3 or any(size != 1 for size in values.shape[1:-2]):
            raise StructuredDataError(
                'invalid explanation batch dimensions', code='INTERNAL_ERROR'
            )
        values = values.reshape(
            values.shape[0], values.shape[-2], values.shape[-1]
        )
        if (
            values.shape != (count, rows, len(columns))
            or not np.isfinite(values).all()
        ):
            raise StructuredDataError(
                'invalid explanation shape or gradients', code='INTERNAL_ERROR'
            )
        return columns, values

    @staticmethod
    def _rows_block(
        columns: list[str], values: np.ndarray, rows: list[int]
    ) -> dict[str, Any]:
        return {
            'row_indices': rows,
            'columns': columns,
            'gradients': values[:, rows, :].tolist(),
        }
