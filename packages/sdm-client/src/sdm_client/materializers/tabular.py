# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Build single-table requests from rows supplied by the caller."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd
from predictive_query import Stype
from predictive_query.validator.utils import col_name, table_name

from sdm_client.errors import StructuredDataError
from sdm_client.request import Request
from sdm_client.schema import Schema

if TYPE_CHECKING:
    from predictive_query import ValidatedPredictiveQuery

_CONTEXT_SEED = 42


class TabularMaterializer:
    r"""Materialize resolved PQL against a single-table schema.

    The ``TABULAR`` profile has already refused anything this cannot honour --
    a filtered entity, an ``ASSUMING`` clause, a windowed or derived target --
    so the target here is a real column of the one table.
    """

    def __init__(self, schema: Schema) -> None:
        self.schema = schema

    def table(
        self,
        *,
        query: ValidatedPredictiveQuery,
        predict: pd.DataFrame,
    ) -> Request:
        r"""Build a request from rows supplied separately by the caller."""
        entity_table_name = table_name(query.entity_column)
        target_col = col_name(query.target_column)

        return Request(
            context=self.schema[entity_table_name].data,
            predict=predict,
            target=target_col,
            task=self._infer_task_type(query),
            schema=self.schema,
            entity_table=entity_table_name,
        )

    def indices(
        self,
        query: ValidatedPredictiveQuery,
        indices: Sequence[Any],
        num_context_samples: int | None,
    ) -> Request:
        r"""Build a single-table request for rows the one table does contain."""
        entity_table_name = table_name(query.entity_column)
        entity_table = self.schema[entity_table_name].data
        entity_col = col_name(query.entity_column)
        target_col = col_name(query.target_column)

        indices = list(indices)
        positions: dict[Hashable, list[int]] = {}
        for position, value in enumerate(entity_table[entity_col]):
            positions.setdefault(value, []).append(position)

        missing = [value for value in indices if value not in positions]
        if missing:
            shown = missing[:10]
            len_missing = len(missing)
            tail = '' if len_missing == len(shown) else ', ...'
            raise StructuredDataError(
                f'{len_missing} of the {len(indices)} indices are not in '
                f'{query.entity_column}: {shown}{tail}. Pass predict=<frame> '
                f'to score rows not present in the table',
                code='INVALID_REQUEST',
            )

        scored = [
            position for value in indices for position in positions[value]
        ]
        predict = entity_table.iloc[scored].drop(columns=[target_col])
        context_mask = entity_table[target_col].notna()
        context_mask.iloc[scored] = False
        candidates = entity_table[context_mask]

        if num_context_samples is None:
            context = candidates
        else:
            context = candidates.sample(
                n=min(num_context_samples, len(candidates)),
                random_state=_CONTEXT_SEED,
            ).sort_index()

        return Request(
            context=context,
            predict=predict,
            target=target_col,
            task=self._infer_task_type(query),
            schema=self.schema,
            entity_table=entity_table_name,
        )

    @staticmethod
    def _infer_task_type(query: ValidatedPredictiveQuery) -> str:
        if query.target_stype not in {Stype.categorical, Stype.numerical}:
            raise StructuredDataError(
                f'tabular materialization requires a categorical or '
                f'numerical target; {query.target_column!r} has semantic '
                f'type {query.target_stype.value!r}',
                code='UNSUPPORTED_REQUEST',
            )
        if query.target_stype == Stype.categorical:
            return 'classification'
        return 'regression'
