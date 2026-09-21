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
from sdm_client.schema import Schema

if TYPE_CHECKING:
    from predictive_query import ValidatedPredictiveQuery

_CONTEXT_SEED = 42
_CONTEXT_ROWS_MAX = TABULAR_LIMITS['context_rows_max']


#: What the tabular deployment answers. Stated here rather than read from a
#: ``ModelSpec`` because a materializer builds a request before a model handle
#: is necessarily in hand; ``models.py`` is the source of truth and a test in
#: tests/ holds the two together.
TABULAR_TASK_KINDS = frozenset({'classification', 'regression'})


def _is_whole(value: object) -> bool:
    r"""Whether ``value`` is a number with nothing after the decimal point.

    Deliberately not ``float(value).is_integer()``. That silently answers yes
    for ``Decimal('2.0000000000000000001')``, which rounds to 2.0 on the way
    through a float, and raises ``OverflowError`` for an integer too large to
    be one at all -- so a warehouse column of exact decimals or big integers
    got the wrong answer or no answer.
    """
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


def _warn_if_the_context_teaches_nothing(
    column: str, target: pd.Series
) -> None:
    r"""Say when every context row carries the same answer.

    This is reachable without doing anything obviously wrong: score all the
    rows of one class and the context left behind holds only the other. The
    endpoint refuses that as a classification -- ``TASK_SCHEMA_MISMATCH``, two
    classes needed and one supplied -- and would take it as a regression over
    a constant, which is worse, because it answers.

    Said rather than refused: a caller who means it can be scoring a holdout
    they know is one-sided, and the endpoint is entitled to the last word on
    what it will accept.
    """
    distinct = target.dropna().nunique()
    if distinct >= 2:
        return
    warnings.warn(
        f'every context row has the same {column} '
        f'({"no rows carry an answer at all" if distinct == 0 else "value"}), '
        f'so nothing in the context distinguishes one answer from another. '
        f'This usually means the rows being scored are all of one class and '
        f'the rest were left behind. Score fewer of them, or build the schema '
        f'from rows that keep both.',
        category=SDMWarning,
        stacklevel=caller_stacklevel(),
    )


def _holds_two_whole_numbers(target: pd.Series) -> bool:
    r"""Whether ``target`` is two distinct whole numbers and nothing else.

    Two is the whole rule, and it is about the values rather than how they are
    stored: a 0/1 column with one missing value arrives from pandas as
    ``float64``, and it is still a label. So ``2.0`` and ``3.0`` read the same
    way as ``2`` and ``3`` -- treating those differently would be an accident
    of storage, not a distinction anyone meant.

    What is excluded is anything wider or fractional. Three whole numbers may
    be a count -- 0, 1 and 2 items bought is a regression and reads
    identically -- and 0.5 and 1.5 are two values of which neither is a label.
    Both are left alone, and ``task=`` settles them.
    """
    values = target.dropna().unique()
    if len(values) != 2:
        return False
    return all(_is_whole(value) for value in values)


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
                    self.schema[entity_table_name].data, predict, entity_col
                ),
                target_col,
            ),
            None,
            entity_table_name,
            'This route takes no num_context_samples, so choose the '
            'context by building the schema from the rows you want.',
        )

        return Request(
            context=context,
            predict=predict,
            target=target_col,
            task=self._task_kind(query, context[target_col], task),
            schema=self.schema.snapshot(),
            entity_table=entity_table_name,
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
            task=self._task_kind(query, context[target_col], task),
            schema=self.schema.snapshot(),
            entity_table=entity_table_name,
        )

    @staticmethod
    def _task_kind(
        query: ValidatedPredictiveQuery,
        target: pd.Series,
        stated: str | None,
    ) -> str:
        r"""What is being asked of the target, and who decided it.

        A stated kind is authoritative and nothing below runs. Otherwise the
        semantic type decides, except for the one case where it decided wrong
        often enough to matter: a target holding two whole numbers.

        ``target`` is the context that will actually be sent, not the column it
        was drawn from. Those differ -- the context leaves out the rows being
        scored, the rows with no answer, and anything above the endpoint's cap
        -- and judging the wider one declares a classification over a context
        that may hold a single class, which the endpoint refuses with
        ``TASK_SCHEMA_MISMATCH``. The rows that teach the model are the rows
        that decide what it is being taught.
        """
        if query.target_stype not in {Stype.categorical, Stype.numerical}:
            raise StructuredDataError(
                f'tabular materialization requires a categorical or '
                f'numerical target; {query.target_column!r} has semantic '
                f'type {query.target_stype.value!r}',
                code='UNSUPPORTED_REQUEST',
            )
        # Above the `stated` branch on purpose: a context that teaches nothing
        # is a fact about the rows, not about which task was asked of them, and
        # naming the task was silencing it.
        _warn_if_the_context_teaches_nothing(query.target_column, target)

        if stated is not None:
            # Checked before the membership test: an unhashable value raises
            # TypeError out of `in frozenset`, which is not an error about
            # anything the caller did wrong.
            if not isinstance(stated, str) or stated not in TABULAR_TASK_KINDS:
                raise StructuredDataError(
                    f'the tabular model does not serve the task kind '
                    f'{stated!r}; it serves '
                    f'{sorted(TABULAR_TASK_KINDS)}',
                    code='UNSUPPORTED_REQUEST',
                    details={
                        'task': stated,
                        'supported': sorted(TABULAR_TASK_KINDS),
                    },
                )
            return stated
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
