# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

from predictive_query.schema.typing import Dtype
from sdm_connectors.source import Source, adopt

if TYPE_CHECKING:
    import pyarrow as pa

#: One statement and the values its placeholders take, in order, as a
#: backend's ``dialect`` module writes it and the internal
#: ``sdm_connectors.Source._run`` runs it.
Statement: TypeAlias = tuple[str, Sequence[Any] | None]


class KeyCountPart(NamedTuple):
    r"""One table's part of a key count, as a dialect writes it.

    Attributes:
        label: The table's position in the count.
        source: What its rows are read from.
        columns: Its key's columns, quoted.
        unique: Whether the source keeps the key unique.
    """

    label: int
    source: str
    columns: tuple[str, ...]
    unique: bool


@dataclass
class SourceColumn:
    name: str
    dtype: Dtype | None
    is_primary_key: bool
    is_unique_key: bool
    is_nullable: bool


@dataclass
class SourceForeignKey:
    name: str
    dst_table: str
    primary_key: str


def as_source(connection: Any, backend: str) -> Any:
    r"""What the engine runs ``backend``'s SQL through.

    The engine reaches a database only by handing a statement to a source's
    internal ``_run``. A :class:`sdm_connectors.Source` of the backend is used as it
    is; anything else is the backend's connection, as the engine has checked
    it, and is held in a source of its own.

    Args:
        connection: A source, or a connection of ``backend``.
        backend: The backend, as :class:`DataBackend` names it.
    """
    if isinstance(connection, Source):
        if connection.backend != backend:
            raise ValueError(
                f'A {backend} table cannot be read through a '
                f'{connection.backend} source'
            )
        return connection
    return adopt(connection, str(backend))


def rows(table: 'pa.Table') -> list[tuple[Any, ...]]:
    r"""The rows of ``table``, each a tuple of Python values in column
    order.
    """
    columns = [column.to_pylist() for column in table.columns]
    return list(zip(*columns, strict=True))
