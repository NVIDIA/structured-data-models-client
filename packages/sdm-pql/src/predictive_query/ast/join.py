# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

from typing import Union

import pydantic
from pydantic.dataclasses import dataclass

from predictive_query.ast.ast_node import ASTNode
from predictive_query.ast.column import Column


@dataclass(repr=False)
class Join(ASTNode):
    r"""Creates an atomic description of a hop from `lhs_key` to
    `rhs_key`. Inferred automatically and used internally.

    Args:
        rhs_target: :class:`ASTNode` defining the rhs df to join.
        lhs_key: Join key of the left table in `table.col` format.
        rhs_key: Join key of the right table in `table.col` format.
    """

    rhs_target: Union['Aggregation', Column] = None  # type: ignore[assignment]
    lhs_key: str = ''
    rhs_key: str = ''

    def __post_init__(self) -> None:
        if self.rhs_target is None:
            raise ValueError(
                f"Class '{self.__class__.__name__}' is missing a rhs_target."
            )
        super().__post_init__()
        self.dtype_maybe = self.rhs_target.dtype_maybe
        self.stype_maybe = self.rhs_target.stype_maybe
        self.location = self.rhs_target.location

    @property
    def children(self) -> list['ASTNode']:
        assert self.rhs_target is not None
        return [self.rhs_target]

    def to_string(self, rich: bool = False) -> str:
        r"""Creates a predictive query statement from the filter."""
        assert self.rhs_target is not None
        return self.rhs_target.to_string(rich)

    @property
    def all_join_columns(self) -> list[str]:
        r"""List of all columns that are needed for joins in the query, given
        with in a fully-qualified name format: `table.column`.
        """
        assert self.rhs_target is not None
        targets = self.rhs_target.all_join_columns
        targets.append(self.lhs_key)
        targets.append(self.rhs_key)
        return list(set(targets))


from predictive_query.ast.aggregation import (  # noqa: E402
    Aggregation,
)

if pydantic.__version__.startswith('1.'):
    Join.__pydantic_model__.update_forward_refs()  # type: ignore[attr-defined]
