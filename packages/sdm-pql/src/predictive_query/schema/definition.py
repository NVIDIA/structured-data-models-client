# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic.dataclasses import dataclass

from predictive_query.schema.table import TableDefinition

TableName = str


@dataclass(frozen=True)
class ColumnKey:
    r"""Reference to a column within a PQuery/Schema definition."""

    table_name: TableName
    col_name: str


@dataclass(frozen=True)
class ColumnKeyGroup:
    r"""Group of column keys to be linked together in a schema."""

    columns: tuple[ColumnKey, ...]  # Always sorted and deduped, immutable


@dataclass(frozen=True, eq=True)
class Relationship:
    r"""A representation of an edge between two tables."""

    src_table: str
    fkey: str
    dst_table: str


@dataclass
class SchemaDefinition:
    r"""A schema: its tables, and the column groups that link them."""

    tables: dict[TableName, TableDefinition]
    col_groups: list[ColumnKeyGroup]
