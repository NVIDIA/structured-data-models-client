# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind a replacement host connection to Rust-revalidated declarations."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from sdm_client._rust import codec, host_providers
from sdm_client._rust.run import run
from sdm_client._rust.schema import build
from sdm_client._rust.spec import held_schema
from sdm_client.errors import StructuredDataError

if TYPE_CHECKING:
    from sdm_client.provider import Source
    from sdm_client.schema import Schema


def rebind(schema: Schema, source: Source) -> Schema:
    from sdm_client.schema import _require_a_source

    if schema.source is None:
        raise StructuredDataError(
            'rebinding requires a source-backed schema',
            code='UNSUPPORTED_REQUEST',
        )
    _require_a_source(source)
    previous, _ = host_providers.chosen(schema.source)
    choice, provider = host_providers.chosen(source)
    finished = run(
        {
            'op': 'rebind_source',
            'provider': choice,
            'previous_provider': previous,
            'source_kind': _kind(source, choice),
            'previous_source_kind': _kind(schema.source, previous),
            'schema': 1,
        },
        {1: codec.attachment(held_schema(schema))},
        provider=provider,
    )
    answer = json.loads(finished.take(finished.result['description']))
    result = build(type(schema), source, answer)
    result._conversion_notes = schema._conversion_notes
    return result


def _kind(source: Source, choice: dict[str, Any]) -> str:
    r"""What the core compares a source and its replacement by, beyond
    their choices: the source's class, and whether it reads a semantic
    model (its ``choice`` declares one), as the Java host names it."""
    model = 'view' in choice['capabilities']
    return type(source).__name__ + ('.semantic_model' if model else '')
