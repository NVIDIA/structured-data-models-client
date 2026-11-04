# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind a replacement host connection to Rust-revalidated declarations."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sdm_client._rust import codec, providers
from sdm_client._rust.run import run
from sdm_client._rust.schema import build
from sdm_client._rust.spec import held_schema
from sdm_client.errors import StructuredDataError

if TYPE_CHECKING:
    from sdm_connectors import Source

    from sdm_client.schema import Schema


def rebind(schema: Schema, source: Source) -> Schema:
    from sdm_client.schema import _require_a_source

    if schema.source is None:
        raise StructuredDataError(
            'rebinding requires a source-backed schema',
            code='UNSUPPORTED_REQUEST',
        )
    _require_a_source(source)
    previous, _ = providers.for_source(schema.source)
    choice, provider = providers.for_source(source)
    finished = run(
        {
            'op': 'rebind_source',
            'provider': choice,
            'previous_provider': previous,
            'source_kind': _kind(source),
            'previous_source_kind': _kind(schema.source),
            'schema': 1,
        },
        {1: codec.attachment(held_schema(schema))},
        provider=provider,
    )
    answer = json.loads(finished.take(finished.result['description']))
    result = build(type(schema), source, answer)
    result._conversion_notes = schema._conversion_notes
    return result


def _kind(source: Source) -> str:
    model = providers._model_of(source)
    return (
        source.kind if model is None else f'{model.source.kind}/{source.kind}'
    )
