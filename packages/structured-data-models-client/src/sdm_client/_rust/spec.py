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

#: The data types and meanings the contract's ``QuerySchema`` names, as it
#: spells them (``sdm_contract::ops::Dtype`` and ``Stype``, the language's
#: own): a column of another is not one a query can name, and is left out
#: of the query's schema, as the language's own schema leaves it out.
DTYPES = frozenset(
    {
        'bool',
        'int',
        'byte',
        'int16',
        'int32',
        'int64',
        'float',
        'float32',
        'float64',
        'string',
        'binary',
        'date',
        'time',
        'timedelta',
        'floatlist',
        'intlist',
        'stringlist',
        'unsupported',
    }
)
STYPES = frozenset(
    {
        'numerical',
        'categorical',
        'multicategorical',
        'ID',
        'text',
        'timestamp',
        'sequence',
        'image',
        'unsupported',
    }
)


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
    return {'tables': tables, 'links': _links(schema)}


def _links(schema: Schema) -> list[list[dict[str, str]]]:
    return [
        [
            {'table': link.source_table, 'column': link.source_column},
            {'table': link.target_table, 'column': link.target_column},
        ]
        for link in schema.relationships
    ]


def query_schema(schema: Schema) -> dict[str, Any]:
    r"""``schema`` as the contract's ``QuerySchema``, as the language's own
    schema reads it (``schema.to_schema_definition``): each column of a
    data type and meaning a query can name, by the language's names; a
    single-column key, a composite one said and dropped
    (:func:`sdm_client.schema.say_a_composite_key_is_dropped`); each link
    a group of its two ends, in order of table and column."""
    from sdm_client.schema import (
        WIRE_TO_PQL_DTYPE,
        say_a_composite_key_is_dropped,
    )

    tables = []
    for name, table in schema.tables.items():
        columns = []
        for column in table.columns.values():
            dtype = WIRE_TO_PQL_DTYPE.get(column.dtype, column.dtype)
            stype = str(column.stype)
            if dtype not in DTYPES or stype not in STYPES:
                continue
            columns.append(
                {'name': column.name, 'dtype': dtype, 'stype': stype}
            )
        key = table.primary_key
        if key is not None and not isinstance(key, str):
            say_a_composite_key_is_dropped(name, key)
        tables.append(
            {
                'name': name,
                'columns': columns,
                # An empty name is none, as the language's table reads it.
                'primary_key': key if isinstance(key, str) and key else None,
                'time_column': table.time_column or None,
                'end_time_column': None,
            }
        )
    links = [
        sorted(link, key=lambda end: (end['table'], end['column']))
        for link in _links(schema)
    ]
    return {'tables': tables, 'links': links}
