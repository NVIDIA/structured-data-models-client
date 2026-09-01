# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# This is not needed in python 3.11
class StrEnum(str, Enum):
    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Span:
    r"""Where in the query text a diagnostic points.

    Rows and columns are 1-based, matching what the message says in prose. The
    interval is closed at both ends: ``(start_row, start_col)`` is the first
    character and ``(end_row, end_col)`` the last.
    """

    start_row: int
    start_col: int
    end_row: int
    end_col: int

    @classmethod
    def at(cls, location: Any) -> 'Span | None':
        r"""The span of an AST location, or :obj:`None` if it has none.

        Takes anything carrying the five attributes rather than importing the
        AST, which is a layer above this one. A location that is absent, or
        that says it has no data, gives :obj:`None`. A location whose shape has
        changed raises :class:`AttributeError` rather than quietly giving
        :obj:`None`, so a regression in the AST surfaces here instead of
        turning every span off.
        """
        if location is None:
            return None
        if not location.data_available:
            return None
        return cls(
            start_row=location.start_row,
            start_col=location.start_col,
            end_row=location.end_row,
            end_col=location.end_col,
        )


@dataclass(frozen=True)
class ValidationInfo:
    r"""Represents non-critical information about automatic actions or decisions
    taken during validation that the user should be aware of, but do not
    require
    immediate attention or action.
    """

    # Short (< 1 line) title
    title: str
    # Precise message, customer-understandable
    message: str
    # Id to match position of info in the frontend
    id: str


@dataclass(frozen=True)
class ValidationWarning:
    # Short (< 1 line) description of what is wrong
    title: str

    # A multi-line, customer-understandable description of the problem, which
    # also contain 1 or more ways for the customer to fix it. It should not
    # contain any internal terminology (eg. only use terms that are consistent
    # with our public docs).
    message: str

    # Stable identifier for this warning; see `predictive_query.schema.codes`.
    # Branch on this, not on the message, which is prose and may be reworded.
    code: str = ''

    # Where in the query text this points. `None` means this check had no
    # node in scope to point at, not that it covers the whole query.
    span: Span | None = None


@dataclass(frozen=True)
class ValidationError:
    title: str
    message: str

    # Stable identifier for this error; see `predictive_query.schema.codes`.
    # Branch on this, not on the message, which is prose and may be reworded.
    code: str = ''

    # Where in the query text this points. `None` means this check had no
    # node in scope to point at, not that it covers the whole query.
    span: Span | None = None


@dataclass
class ValidationResponse:
    warnings: list[ValidationWarning] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)
    info_items: list[ValidationInfo] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def empty(self) -> bool:
        return (
            len(self.errors) == 0
            and len(self.warnings) == 0
            and len(self.info_items) == 0
        )

    def message(self) -> str:
        r"""Returns a simple string message enumerating errors, warnings, and
        info items in this validation response.
        """
        message = self.error_message()
        if len(self.warnings) > 0:
            if len(self.errors) > 0:
                message += '\n'
            message += 'Warnings:\n'
            message += '\n'.join(
                [
                    f'{i}. {warn.message}'
                    for i, warn in enumerate(self.warnings, 1)
                ]
            )

        if self.info_items:
            if message:
                message += '\n'
            message += 'Info:\n'
            message += '\n'.join(
                [
                    f'{i}. {info.title}. {info.message}'
                    for i, info in enumerate(self.info_items, 1)
                ]
            )

        return message

    def error_message(self) -> str:
        r"""Serialized, pretty-printed error message from all the errors
        contained in this ValidationResponse.
        """
        message = ''
        if len(self.errors) > 0:
            if len(self.errors) == 1:
                message += f'{self.errors[0].title}. {self.errors[0].message}'
            else:
                message += '\n'.join(
                    [
                        f'{i}. {err.title}. {err.message}'
                        for i, err in enumerate(self.errors, 1)
                    ]
                )
        return message
