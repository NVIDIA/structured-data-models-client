# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Build single-table requests from rows supplied by the caller."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from sdm_client.errors import StructuredDataError
from sdm_client.request import Request
from sdm_client.schema import Schema

if TYPE_CHECKING:
    from predictive_query import ValidatedPredictiveQuery

_TASK_FOR_PROBLEM = {
    'ProblemType.CLASSIFY': 'classification',
    'CLASSIFY': 'classification',
    'ProblemType.FORECAST': 'forecasting',
    'FORECAST': 'forecasting',
    'ProblemType.RANK': 'ranking',
    'RANK': 'ranking',
}


class TabularMaterializer:
    r"""Materialize resolved PQL against a single-table schema."""

    def __init__(self, schema: Schema) -> None:
        self.schema = schema

    def table(
        self,
        *,
        query: ValidatedPredictiveQuery,
        predict: pd.DataFrame,
        num_context_samples: int | None,
        options: dict[str, Any],
    ) -> Request:
        r"""Build a request from rows supplied separately by the caller."""
        if num_context_samples is not None:
            raise StructuredDataError(
                'num_context_samples cannot be used with predict=<frame>: '
                'the schema table is used as context in full. To use less '
                'context, construct the query with a smaller table',
                code='INVALID_REQUEST',
            )
        if options:
            raise StructuredDataError(
                f'predict=<frame> does not accept options: {sorted(options)}; '
                f'the schema table is used as context without sampling',
                code='INVALID_REQUEST',
            )
        if len(self.schema) != 1:
            raise StructuredDataError(
                f'predict=<frame> requires a single-table schema; '
                f'got {len(self.schema)} tables. For a relational schema, use '
                f'indices=[...] to select entities already in the tables',
                code='INVALID_REQUEST',
            )

        table_name = next(iter(self.schema))
        target = query.target_column
        if isinstance(target, str) and target.startswith(f'{table_name}.'):
            target = target[len(table_name) + 1 :]
        if target is None or target not in self.schema[table_name]:
            raise StructuredDataError(
                f'query target {target!r} is not a column in table '
                f'{table_name!r}',
                code='INVALID_REQUEST',
            )

        return Request(
            context=self.schema[table_name].data,
            predict=predict,
            target=target,
            # TODO: Derive the runtime task from the validated query.
            # ProblemType has no REGRESSION member, so this fallback cannot
            # materialize a tabular regression request.
            task=_TASK_FOR_PROBLEM.get(
                str(query.problem_type), 'classification'
            ),
            schema=self.schema,
            entity_table=table_name,
        )
