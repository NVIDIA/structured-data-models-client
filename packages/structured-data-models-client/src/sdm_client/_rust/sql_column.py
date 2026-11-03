# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt a named provider SQL projection to the engine's schema operation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sdm_client._rust import codec, providers
from sdm_client._rust.run import run
from sdm_client._rust.schema import build
from sdm_client._rust.spec import held_schema
from sdm_client.errors import StructuredDataError

if TYPE_CHECKING:
    from sdm_client.schema import Schema


def add(schema: Schema, table: str, name: str, expression: str) -> Schema:
    if not all(isinstance(value, str) for value in (table, name, expression)):
        raise StructuredDataError(
            'table, name and expression must be strings', code='INVALID_REQUEST'
        )
    if schema.source is None:
        raise StructuredDataError(
            'SQL columns require a source-backed schema; calculate in-memory columns before admission',
            code='UNSUPPORTED_REQUEST',
        )
    choice, provider = providers.for_source(schema.source)
    finished = run(
        {'op': 'add_sql_column', 'provider': choice, 'schema': 1, 'input': 2},
        {
            1: codec.attachment(held_schema(schema)),
            2: codec.attachment(
                {'table': table, 'name': name, 'expression': expression}
            ),
        },
        provider=provider,
    )
    answer = json.loads(finished.take(finished.result['description']))
    result = build(type(schema), schema.source, answer)
    result._conversion_notes = schema._conversion_notes
    return result
