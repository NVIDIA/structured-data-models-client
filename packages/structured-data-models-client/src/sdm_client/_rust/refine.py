# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host orchestration for explicit schema refinement; Rust owns all decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sdm_client._rust import edits, spec
from sdm_client.errors import StructuredDataError

if TYPE_CHECKING:
    from sdm_client.schema import Relationship, Schema


def changed(schema: Schema, edit: dict) -> Schema:
    from sdm_client.schema import Relationship, _Links

    result = schema.snapshot()
    answer = edits._made(result, edit)
    links = [
        Relationship(
            foreign['table'],
            foreign['column'],
            target['table'],
            target['column'],
        )
        for foreign, target in answer['links']
    ]
    object.__setattr__(result, 'relationships', _Links(links, result))
    return result


def infer(schema: Schema, links: bool) -> Schema:
    from sdm_client.schema import Schema

    if not schema.tables:
        return schema.snapshot()
    if schema.source is None:
        candidate = Schema.from_data(
            {
                name: table.data.loc[:, list(table.columns)]
                for name, table in schema.tables.items()
            }
        )
    else:
        candidate = Schema.from_source(
            schema.source,
            tables={
                name: table.source_name for name, table in schema.tables.items()
            },
        )
    return changed(
        schema,
        {
            'edit': 'infer',
            'candidate': spec.held_schema(candidate),
            'links': links,
        },
    )


def link_edit(
    schema: Schema, relationship: Relationship, remove: bool
) -> Schema:
    from sdm_client.schema import Relationship

    if not isinstance(relationship, Relationship) or any(
        not isinstance(name, str)
        for name in (
            relationship.source_table,
            relationship.source_column,
            relationship.target_table,
            relationship.target_column,
        )
    ):
        raise StructuredDataError(
            'relationship must contain four names', code='INVALID_REQUEST'
        )
    return changed(
        schema,
        {
            'edit': 'remove_link' if remove else 'add_link',
            'link': [
                {
                    'table': relationship.source_table,
                    'column': relationship.source_column,
                },
                {
                    'table': relationship.target_table,
                    'column': relationship.target_column,
                },
            ],
        },
    )
