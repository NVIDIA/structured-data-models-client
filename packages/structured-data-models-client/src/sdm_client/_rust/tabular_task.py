# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual native task observations for the shared tabular meaning products."""

from __future__ import annotations

import json
import numbers
import sys
import warnings
from collections.abc import Callable, Iterator
from decimal import Decimal
from typing import TYPE_CHECKING, Any, TypeVar

from sdm_client._rust import errors
from sdm_client.core.warnings import SDMWarning, caller_stacklevel

if TYPE_CHECKING:
    import pandas as pd
    from predictive_query import ValidatedPredictiveQuery

    from sdm_client import _native

_T = TypeVar('_T')


def _checked(function: Callable[..., _T], *args: Any) -> _T:
    from sdm_client import _native

    try:
        return function(*args)
    except _native.CoreError as caught:
        selected = caught
    # Native diagnostic repr must execute outside the internal error handler.
    errors.raise_from_core(
        selected.control,
        selected.attachments,
        internal_context=selected,
        presentation='tabular_task',
    )


def _observed_number(
    prefix: _native.TabularTaskNumbers, ordinal: int, value: Any
) -> Iterator[tuple[str, Any, Any]]:
    # The original all(generator-expression) promotes StopIteration raised
    # inside _is_whole, but not a failure of the returned object's truth test.
    if isinstance(value, bool):
        yield ('boolean', None, None)
    elif isinstance(value, numbers.Integral):
        yield ('integral', None, None)
    elif isinstance(value, Decimal):
        finite: Any = value.is_finite()
        if finite:
            # The original and-expression discards its true left operand
            # before calling the rounding/equality operations.
            finite = None
            yield ('decimal', True, value == value.to_integral_value())
        else:
            yield ('decimal', False, finite)
    elif isinstance(value, numbers.Real):
        try:
            converted = float(value)
        except Exception as native:
            ancestry = type.__getattribute__(type(native), '__mro__')
            completed = _checked(
                prefix.real_failure,
                ordinal,
                any(kind is OverflowError for kind in ancestry),
                any(kind is TypeError for kind in ancestry),
                any(kind is ValueError for kind in ancestry),
            )
            if completed.propagates():
                raise
            yield ('completed', completed, None)
        else:
            yield ('real', converted, None)
    else:
        yield ('other', None, None)


def _completed_number(
    prefix: _native.TabularTaskNumbers, ordinal: int, observation: Any
) -> _native.TabularTaskNumbers:
    kind, value, returned = observation
    if kind == 'boolean':
        return _checked(prefix.boolean, ordinal)
    if kind == 'integral':
        return _checked(prefix.integral, ordinal)
    if kind == 'decimal':
        truth = bool(returned)
        return _checked(
            prefix.decimal,
            ordinal,
            value,
            truth if value else None,
            None if value else truth,
        )
    if kind == 'real':
        return _checked(prefix.real, ordinal, value)
    if kind == 'completed':
        return value
    return _checked(prefix.other, ordinal)


def task_kind(
    query: ValidatedPredictiveQuery, target: pd.Series, stated: Any
) -> Any:
    from predictive_query import Stype

    from sdm_client import _native

    stypes, tasks, categorical = _native.tabular_task_vocabulary()
    tasks = [sys.intern(value) for value in tasks]

    def target_display(observed: dict[str, Any]) -> tuple[str, str]:
        return repr(query.target_column), repr(query.target_stype.value)

    with errors.holding(
        tabular_task_target=target_display,
        relational_model_task=lambda observed: (repr(stated), stated),
    ):
        meaning = _checked(
            _native.TabularTaskMeaning,
            query.target_stype in {Stype(value) for value in stypes},
        )
        if stated is None:
            meaning = _checked(meaning.stated, False, None, None)
        else:
            string = isinstance(stated, str)
            member = stated in frozenset(tasks) if string else None
            meaning = _checked(meaning.stated, True, string, member)
        if meaning.original():
            return stated
        meaning = _checked(
            meaning.inferred, bool(query.target_stype == Stype(categorical))
        )
        kind = meaning.kind()
        if kind is not None:
            return kind
        values: Any = target.dropna().unique()
        prefix = _checked(meaning.numbers, len(values))
        ordinal = 0
        if prefix.kind() is None:
            iterator: Any = iter(values)
            value: Any = None
            while prefix.kind() is None:
                exhausted = False
                try:
                    value = next(iterator)
                except StopIteration:
                    exhausted = True
                if exhausted:
                    # Release native owners outside the StopIteration handler,
                    # before the later inference notice and schema reads.
                    iterator = None
                    value = None
                    prefix = _checked(prefix.exhausted, ordinal)
                    break
                observation: Any = next(
                    _observed_number(prefix, ordinal, value)
                )
                prefix = _completed_number(prefix, ordinal, observation)
                observation = None
                ordinal += 1
            iterator = None
            value = None
        values = None
        for notice in json.loads(prefix.notices()):
            # Original formatting is reached only after completed inference.
            shown = format(query.target_column)
            with errors.holding(tabular_target=lambda observed: shown):
                warnings.warn(
                    errors.said(notice),
                    category=SDMWarning,
                    stacklevel=caller_stacklevel(),
                )
        result = prefix.kind()
        assert result is not None
        return result
