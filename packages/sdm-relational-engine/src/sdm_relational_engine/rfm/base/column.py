# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from predictive_query.schema.typing import Dtype, Stype
from typing_extensions import Self

from sdm_relational_engine.rfm.base import Expression
from sdm_relational_engine.rfm.base.casting import CastMixin


@dataclass(init=False, repr=False, eq=False)
class ColumnSpec(CastMixin):
    r"""A column specification for adding a column to a table.

    A column specification can either refer to a physical column present in
    the data source, or be defined logically via an expression.

    Args:
        name: The name of the column.
        expr: A column expression to define logical columns.
        dtype: The data type of the column.
    """

    def __init__(
        self,
        name: str,
        expr: Expression | Mapping[str, str] | str | None = None,
        dtype: Dtype | str | None = None,
        stype: Stype | str | None = None,
    ) -> None:

        self.name = name
        self.expr = Expression.coerce(expr)
        self.dtype = Dtype(dtype) if dtype is not None else None
        self.stype = Stype(stype) if stype is not None else None

    @classmethod
    def coerce(cls, spec: ColumnSpec | Mapping[str, Any] | str) -> Self:
        r"""Coerces a column specification into a :class:`ColumnSpec`."""
        if isinstance(spec, cls):
            return spec
        if isinstance(spec, str):
            return cls(name=spec)
        if isinstance(spec, Mapping):
            try:
                return cls(**dict(spec))
            except TypeError:
                pass
        raise TypeError(f"Unable to coerce 'ColumnSpec' from '{spec}'")

    @property
    def is_source(self) -> bool:
        r"""Whether the column specification refers to a physical column
        present in the data source.
        """
        return self.expr is None

    def __repr__(self) -> str:
        parts = [f'name={self.name}']
        if self.expr is not None:
            parts.append(f'expr={self.expr}')
        if self.dtype is not None:
            parts.append(f'dtype={self.dtype}')
        if self.stype is not None:
            parts.append(f'stype={self.stype}')
        return f'{self.__class__.__name__}({", ".join(parts)})'


ColumnSpecType: TypeAlias = ColumnSpec | Mapping[str, Any] | str


@dataclass(init=False, repr=False, eq=False)
class Column:
    r"""Column-level metadata information.

    A column can either refer to a physical column present in the data source,
    or be defined logically via an expression.

    Args:
        name: The name of the column.
        expr: A column expression to define logical columns.
        dtype: The data type of the column.
        stype: The semantic type of the column.
    """

    stype: Stype

    def __init__(
        self,
        name: str,
        expr: Expression | None,
        dtype: Dtype,
        stype: Stype,
    ) -> None:
        self._name = name
        self._expr = expr
        self._dtype = Dtype(dtype)

        self._is_primary_key = False
        self._is_time_column = False
        self._is_end_time_column = False

        self.stype = Stype(stype)

    @property
    def name(self) -> str:
        r"""The name of the column."""
        return self._name

    @property
    def expr(self) -> Expression | None:
        r"""The expression of column (if logically)."""
        return self._expr

    @property
    def dtype(self) -> Dtype:
        r"""The data type of the column."""
        return self._dtype

    @property
    def is_source(self) -> bool:
        r"""Whether the column refers to a phyiscal column present in the data
        source.
        """
        return self.expr is None

    def __setattr__(self, key: str, val: Any) -> None:
        if key == 'stype':
            if isinstance(val, str):
                val = Stype(val)
            assert isinstance(val, Stype)
            if not val.supports_dtype(self.dtype):
                raise ValueError(
                    f"Column '{self.name}' received an "
                    f'incompatible semantic type (got '
                    f"dtype='{self.dtype}' and stype='{val}')"
                )
            if self._is_primary_key and val != Stype.ID:
                raise ValueError(
                    f"Primary key '{self.name}' must have 'ID' "
                    f"semantic type (got '{val}')"
                )
            if self._is_time_column and val != Stype.timestamp:
                raise ValueError(
                    f"Time column '{self.name}' must have "
                    f"'timestamp' semantic type (got '{val}')"
                )
            if self._is_end_time_column and val != Stype.timestamp:
                raise ValueError(
                    f"End time column '{self.name}' must have "
                    f"'timestamp' semantic type (got '{val}')"
                )

        super().__setattr__(key, val)

    def __hash__(self) -> int:
        return hash((self.name, self.expr, self.dtype, self.stype))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Column):
            return False
        return hash(self) == hash(other)

    def __repr__(self) -> str:
        parts = [f'name={self.name}']
        if self.expr is not None:
            parts.append(f'expr={self.expr}')
        parts.append(f'dtype={self.dtype}')
        parts.append(f'stype={self.stype}')
        return f'{self.__class__.__name__}({", ".join(parts)})'
