# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable facts about how a request's context examples were drawn."""

from dataclasses import dataclass
from typing import Literal

ContextStop = Literal['full', 'source_exhausted', 'row_budget', 'time_limit']


@dataclass(frozen=True)
class ContextDraw:
    """How one part's context examples were drawn from the source.

    A request has one part; a forecast one per step, a ranking one per
    candidate question.
    """

    #: The draws of the entity table: more than one where the first kept
    #: too few examples.
    draws: int
    #: The rows those draws read, together.
    rows_drawn: int
    #: The examples of the last draw labelled.
    candidates: int
    #: The candidates with an answer the query's conditions keep, before
    #: the context is cut to the size asked for.
    kept: int
    #: The context examples asked for.
    asked: int
    #: Why drawing stopped: enough were kept, the source had no more rows,
    #: another draw would have read more rows than one request may, or the
    #: draws had taken the time one request's context may
    #: (``context_time_limit``).
    stopped: ContextStop
    #: Whether any of the query's conditions went to the source with the
    #: draw.
    filtered_at_source: bool

    def __post_init__(self) -> None:
        counts = (
            self.draws,
            self.rows_drawn,
            self.candidates,
            self.kept,
            self.asked,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError('invalid context draw count')
        if self.stopped not in (
            'full',
            'source_exhausted',
            'row_budget',
            'time_limit',
        ):
            raise ValueError('invalid context draw stop')
        if type(self.filtered_at_source) is not bool:
            raise ValueError('invalid context draw filter flag')


@dataclass(frozen=True)
class ContextReport:
    """How the context was drawn; unavailable where the request does not
    say (context the caller gave, or a request saved before reports)."""

    status: Literal['available', 'unavailable']
    parts: tuple[ContextDraw, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'parts', tuple(self.parts))
        if any(not isinstance(part, ContextDraw) for part in self.parts):
            raise TypeError('parts must contain ContextDraw values')
        if self.status not in ('available', 'unavailable'):
            raise ValueError('invalid context report status')
        if self.status != 'available' and self.parts:
            raise ValueError('only available reports carry parts')
