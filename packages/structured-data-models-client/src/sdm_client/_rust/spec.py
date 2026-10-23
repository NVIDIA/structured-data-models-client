# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The schema as the core reads it: plain data, written from the schema as
it stands (T3.16, the structure design's fix A).

:func:`held_schema` writes the contract's ``HeldSchema`` that
``edit_schema`` and ``materialize`` take: each table by the name the schema
holds it under, its source's name of the table, its columns' data types,
meanings and what each meant before the schema joined on it, with what the
source said of each column it was asked of (``Column._source``), its key,
time column and repeated keys, and the links. Nothing the core answered
earlier is merged in: a schema built by hand, unpickled, copied or
described on the old route is written as it stands, and the core reads
from the source the facts of a column that carries none.

:func:`query_schema` writes the contract's ``QuerySchema`` that
``validate_query`` takes, from the same schema, with no object of the old
PQL package (the structure design's rule 11).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sdm_client._rust import codec

if TYPE_CHECKING:
    from sdm_client.schema import Schema


def held_schema(schema: Schema) -> dict[str, Any]:
    r"""``schema`` as the contract's ``HeldSchema``."""
    from sdm_client.schema import _as_key_tuple

    tables = []
    for name, table in schema.tables.items():
        key = table.primary_key
        columns = []
        for column_name, column in table.columns.items():
            written: dict[str, Any] = {
                'name': column_name,
                'dtype': column.dtype,
                'stype': str(column.stype),
                'stype_before_key': (
                    None
                    if column._stype_before_key is None
                    else str(column._stype_before_key)
                ),
            }
            if column._source is not None:
                written['source'] = column._source.to_contract()
            columns.append(written)
        held: dict[str, Any] = {
            'name': name,
            'source': codec.relation(getattr(table, 'source_name', name)),
            'columns': columns,
            'primary_key': [] if key is None else list(_as_key_tuple(key)),
            'time_column': table.time_column,
        }
        repeated = getattr(table, 'duplicate_keys', None)
        if repeated is not None:
            held['duplicate_keys'] = codec.integer('u64', int(repeated))
        tables.append(held)
    held_schema: dict[str, Any] = {'tables': tables, 'links': _links(schema)}
    # Where the schema read each table's relation it read (T3.4), for the
    # tables it holds now, whichever schema each table came from.
    read_under = getattr(schema, '_read_under', None) or {}
    named = {
        getattr(table, 'source_name', name)
        for name, table in schema.tables.items()
    }
    under = [
        {'relation': codec.relation(source_name), 'namespace': list(parts)}
        for source_name, parts in read_under.items()
        if source_name in named
    ]
    if under:
        held_schema['read_under'] = under
    return held_schema


def _links(schema: Schema) -> list[list[dict[str, str]]]:
    from sdm_client.schema import links_of

    return links_of(schema)


def query_schema(schema: Schema) -> dict[str, Any]:
    r"""``schema`` as the contract's ``QuerySchema``
    (:func:`sdm_client.schema.query_schema`, the one translation of a schema
    into the query language's, which the old route's definition is built
    from too)."""
    from sdm_client.schema import query_schema as written

    return written(schema)
