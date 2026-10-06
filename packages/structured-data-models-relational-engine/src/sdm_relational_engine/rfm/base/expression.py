# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass


class Expression(ABC):
    r"""A base expression to define logical columns."""

    @classmethod
    def coerce(
        cls,
        spec: Expression | Mapping[str, str] | str | None,
    ) -> Expression | None:
        r"""Coerces an expression specification into an :class:`Expression`, if
        possible.
        """
        if spec is None:
            return None
        if isinstance(spec, Expression):
            return spec
        if isinstance(spec, str):
            return LocalExpression(spec)
        if isinstance(spec, Mapping):
            for sub_cls in (LocalExpression,):
                try:
                    return sub_cls(**dict(spec))
                except TypeError:
                    pass
        raise TypeError(f"Unable to coerce 'Expression' from '{spec}'")


@dataclass(frozen=True, repr=False)
class LocalExpression(Expression):
    r"""A local expression to define a row-level logical attribute based on
    physical columns of the data source in the same row.

    Args:
        value: The value of the expression.
    """

    value: str

    def __repr__(self) -> str:
        return self.value
