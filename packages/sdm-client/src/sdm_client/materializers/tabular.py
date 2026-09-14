# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Build single-table requests from rows supplied by the caller."""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd
from predictive_query import Stype
from predictive_query.validator.utils import col_name, table_name

from sdm_client.core.warnings import caller_stacklevel
from sdm_client.errors import StructuredDataError
from sdm_client.models import TABULAR_LIMITS
from sdm_client.request import Request
from sdm_client.schema import Schema

if TYPE_CHECKING:
    from predictive_query import ValidatedPredictiveQuery

_CONTEXT_SEED = 42
_CONTEXT_ROWS_MAX = TABULAR_LIMITS['context_rows_max']


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
        entity_col = col_name(query.entity_column)
        target_col = col_name(query.target_column)

        return Request(
            context=_context_within_limit(
                _without_the_rows_being_scored(
                    self.schema[entity_table_name].data, predict, entity_col
                ),
                None,
                entity_table_name,
                'This route takes no num_context_samples, so choose the '
                'context by building the schema from the rows you want.',
            ),
            predict=predict,
            target=target_col,
            task=self._infer_task_type(query),
            schema=self.schema.snapshot(),
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

        context = _context_within_limit(
            candidates,
            num_context_samples,
            entity_table_name,
            'Pass num_context_samples to choose how many, or select the rows '
            'yourself.',
        )

        return Request(
            context=context,
            predict=predict,
            target=target_col,
            task=self._infer_task_type(query),
            schema=self.schema.snapshot(),
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


def _context_within_limit(
    candidates: pd.DataFrame,
    num_context_samples: int | None,
    table: str,
    remedy: str,
) -> pd.DataFrame:
    r"""The context rows, never more than the endpoint will take.

    The endpoint refuses a context above its limit, and refuses it *after* the
    whole thing has been encoded into JSON and transmitted -- a table of
    hundreds of thousands of rows sent in full to be told the limit is ten
    thousand. The relational side has declared its limits and refused locally
    all along; this is the same thing on the other model.

    A number the caller named above the limit is an error, because they asked
    for something that cannot happen and quietly giving them less would answer
    a different question. Saying nothing is a choice about how much data the
    model sees, so it is made here and said out loud.
    """
    if num_context_samples is not None:
        if num_context_samples > _CONTEXT_ROWS_MAX:
            raise StructuredDataError(
                f'num_context_samples is {num_context_samples:,}, and the '
                f'endpoint accepts at most {_CONTEXT_ROWS_MAX:,} context '
                f'rows',
                code='UNSUPPORTED_REQUEST',
            )
        wanted = num_context_samples
    else:
        wanted = _CONTEXT_ROWS_MAX
    if len(candidates) <= wanted:
        return candidates
    if num_context_samples is None:
        warnings.warn(
            f'{table} has {len(candidates):,} rows that could be context and '
            f'the endpoint accepts {_CONTEXT_ROWS_MAX:,}, so {wanted:,} were '
            f'sampled. {remedy}',
            stacklevel=caller_stacklevel(),
        )
    return candidates.sample(n=wanted, random_state=_CONTEXT_SEED).sort_index()


def _without_the_rows_being_scored(
    table: pd.DataFrame, predict: pd.DataFrame, entity_col: str
) -> pd.DataFrame:
    r"""The table, minus any row the caller is about to ask about.

    The `indices=` route does this and this one did not, so the two
    single-table routes disagreed about whether a row may be both scored and
    shown. Rows supplied separately are often rows the table already holds --
    that is how anyone checks a model against answers they have -- and leaving
    them in the context hands the model those answers, labelled, and returns
    what reads like accuracy.

    Rows the table does not hold are what the route is for, and they simply
    match nothing here.
    """
    if entity_col not in table.columns or entity_col not in predict.columns:
        return table
    scored = set(predict[entity_col])
    if not scored:
        return table
    return table[~table[entity_col].isin(scored)]
