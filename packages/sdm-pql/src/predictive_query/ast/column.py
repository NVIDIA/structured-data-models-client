# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1


from pydantic.dataclasses import dataclass

from predictive_query.ast.ast_node import ASTNode
from predictive_query.ast.location_interval import (
    ASTQueryLocationInterval,
)
from predictive_query.names import fqn as quote_fqn


@dataclass(repr=False)
class Column(ASTNode):
    r"""Creates an atomic description of a column :obj:`fqn`.

    Args:
        fqn: A fully-qualified name in the "table.column" format.
    """

    fqn: str = ''

    def __post_init__(self) -> None:
        if len(self.fqn.split('.')) != 2:
            raise ValueError(
                f'The column name {self.fqn} was not given in its '
                f'fully-qualified name form. Format it as "table.column".'
            )

        # internal helper var for checking if wildcard columns correctly
        # appear only inside COUNT aggregations
        self._count_col = False

        super().__post_init__()

    @property
    def all_query_columns_with_locations(
        self,
    ) -> list[tuple[str, ASTQueryLocationInterval]]:
        assert self.location is not None
        return [(self.fqn, self.location)]

    @property
    def all_query_columns(self) -> list[str]:
        return [self.fqn]

    def to_string(self, rich: bool = False) -> str:
        # ``fqn`` holds the name as the data spells it, so a name the bare
        # identifier cannot express is quoted back up here; the rendered query
        # is the one the user would have to write to reproduce this node.
        table, _, column = self.fqn.partition('.')
        return quote_fqn(table, column)
