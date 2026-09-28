# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Build single-table requests from rows supplied by the caller."""

from __future__ import annotations

import numbers
import warnings
from collections.abc import Hashable, Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd
from predictive_query import Stype
from predictive_query.validator.utils import col_name, table_name

from sdm_client.core.warnings import SDMWarning, caller_stacklevel
from sdm_client.errors import StructuredDataError
from sdm_client.models import TABULAR_LIMITS
from sdm_client.request import Request
from sdm_client.schema import Schema, SourceTable, Table

if TYPE_CHECKING:
    from predictive_query import ValidatedPredictiveQuery

_CONTEXT_SEED = 42
_CONTEXT_ROWS_MAX = TABULAR_LIMITS['context_rows_max']


TABULAR_TASK_KINDS = frozenset({'classification', 'regression'})


def _is_whole(value: object) -> bool:
    r"""Whether ``value`` is a number with nothing after the decimal point."""
    if isinstance(value, bool):
        return True
    if isinstance(value, numbers.Integral):
        return True
    if isinstance(value, Decimal):
        return value.is_finite() and value == value.to_integral_value()
    if isinstance(value, numbers.Real):
        try:
            return float(value).is_integer()
        except (OverflowError, TypeError, ValueError):
            return False
    return False


def _derive_task_kind(
    query: ValidatedPredictiveQuery, target: pd.Series
) -> str:
    r"""The task the target implies, when the caller named none."""
    if query.target_stype == Stype.categorical:
        return 'classification'
    if _holds_two_whole_numbers(target):
        warnings.warn(
            f'{query.target_column} holds two whole numbers, which is how '
            f'a binary label is usually written, so this is being asked as '
            f'a classification. If they are quantities rather than labels, '
            f'pass task="regression" to ask for one; pass '
            f'task="classification" to say this was meant and silence '
            f'this.',
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )
        return 'classification'
    return 'regression'


def _judge_a_context_that_teaches_nothing(
    column: str, target: pd.Series, kind: str
) -> None:
    r"""Refuse, or remark on, a context whose every row says the same thing.

    A classification over one class is refused by the endpoint, so it is
    refused here. A regression over a constant is accepted there, so it is
    only warned about. A context with no answered rows is left to ``Request``.
    """
    answered = target.dropna()
    if len(answered) == 0 or answered.nunique() >= 2:
        return

    if kind == 'classification':
        raise StructuredDataError(
            f'every context row has the same {column}, and a classification '
            f'needs at least two classes to learn from. This usually means '
            f'the rows being scored are all of one class and the rest were '
            f'left behind. Score fewer of them, or build the schema from rows '
            f'that keep both.',
            code='INVALID_REQUEST',
            details={'target': column, 'classes_in_context': 1},
        )

    warnings.warn(
        f'every context row has the same {column}, so nothing in the context '
        f'distinguishes one answer from another. The endpoint will answer, '
        f'but the answer cannot say much. Build the schema from rows that '
        f'vary, or score fewer of them.',
        category=SDMWarning,
        stacklevel=caller_stacklevel(),
    )


def _holds_two_whole_numbers(target: pd.Series) -> bool:
    r"""Whether ``target`` is two distinct whole numbers and nothing else.

    The values decide, not their storage: a 0/1 column with a null arrives as
    ``float64`` and is still a label. Anything wider or fractional is left to
    ``task=``.
    """
    values = target.dropna().unique()
    if len(values) != 2:
        return False
    return all(_is_whole(value) for value in values)


def _asking(request: Request, query: Any) -> Request:
    """Record the question a request was built from.

    A session pins one question and only the rows travel on a scoring call, so
    the question is what tells one request from another. The relational route
    records it when it gathers; this route builds the request directly and did
    not, which left a tabular session unable to tell that it had been handed
    something else.
    """
    request.query = query.to_string()
    return request


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
        task: str | None = None,
    ) -> Request:
        r"""Build a request from rows supplied separately by the caller.

        The context is the table minus the rows being scored and minus the
        rows with nothing to teach. The other route already left out both;
        this one left out neither, so the same table asked the same question
        two ways built two different contexts.
        """
        entity_table_name = table_name(query.entity_column)
        entity_col = col_name(query.entity_column)
        target_col = col_name(query.target_column)

        context = _context_within_limit(
            _rows_that_carry_an_answer(
                _without_the_rows_being_scored(
                    _rows(self.schema[entity_table_name]), predict, entity_col
                ),
                target_col,
            ),
            None,
            entity_table_name,
            'This route takes no num_context_samples, so choose the '
            'context by building the schema from the rows you want.',
        )

        return _asking(
            Request(
                context=context,
                predict=predict,
                target=target_col,
                task=self._task_kind(query, context[target_col], task),
                schema=self.schema.snapshot(),
                entity_table=entity_table_name,
            ),
            query,
        )

    def indices(
        self,
        query: ValidatedPredictiveQuery,
        indices: Sequence[Any],
        num_context_samples: int | None,
        task: str | None = None,
    ) -> Request:
        r"""Build a single-table request for rows the one table does contain."""
        entity_table_name = table_name(query.entity_column)
        entity_table = _rows(self.schema[entity_table_name])
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

        return _asking(
            Request(
                context=context,
                predict=predict,
                target=target_col,
                task=self._task_kind(query, context[target_col], task),
                schema=self.schema.snapshot(),
                entity_table=entity_table_name,
            ),
            query,
        )

    @staticmethod
    def _task_kind(
        query: ValidatedPredictiveQuery,
        target: pd.Series,
        stated: str | None,
    ) -> str:
        r"""What is being asked of the target, and who decided it.

        ``target`` is the context that will be sent, not the column it came
        from; the two differ, and the narrower one is what the model learns
        from.
        """
        if query.target_stype not in {Stype.categorical, Stype.numerical}:
            raise StructuredDataError(
                f'tabular materialization requires a categorical or '
                f'numerical target; {query.target_column!r} has semantic '
                f'type {query.target_stype.value!r}',
                code='UNSUPPORTED_REQUEST',
            )
        if stated is not None and (
            not isinstance(stated, str) or stated not in TABULAR_TASK_KINDS
        ):
            raise StructuredDataError(
                f'the tabular model does not serve the task kind '
                f'{stated!r}; it serves {sorted(TABULAR_TASK_KINDS)}',
                code='UNSUPPORTED_REQUEST',
                details={
                    'task': stated,
                    'supported': sorted(TABULAR_TASK_KINDS),
                },
            )

        kind = (
            stated if stated is not None else _derive_task_kind(query, target)
        )
        _judge_a_context_that_teaches_nothing(query.target_column, target, kind)
        return kind


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
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )
    return candidates.sample(n=wanted, random_state=_CONTEXT_SEED).sort_index()


def _rows_that_carry_an_answer(
    frame: pd.DataFrame, target: str
) -> pd.DataFrame:
    r"""The rows that can teach the model something.

    A row whose target is null is not an example of anything. It takes up
    room in a context the endpoint caps, and says nothing in exchange.
    """
    if target not in frame:
        return frame
    return frame[frame[target].notna()]


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


def _rows(table: Table | SourceTable) -> pd.DataFrame:
    r"""The rows of a table the tabular model reads, which it holds in memory.

    `PredictiveQuery.materialize` sends a schema over tables in their source to
    the relational model, so only a hand-built route reaches here with one.
    """
    if isinstance(table, SourceTable):
        raise StructuredDataError(
            f'{table.name!r} stays in its source, and the tabular model reads '
            f'rows in memory; build the schema with Schema.from_data',
            code='INVALID_REQUEST',
        )
    return table.data
