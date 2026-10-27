# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Bridge code: hands native frame values and opaque checkpoint owners to
# the core; this routing seam is deleted in Phase 5.

"""Native input and opaque ownership for the first frame materializer."""

from __future__ import annotations

import secrets
from typing import Any

import pandas as pd

from sdm_client._routing import RouteError
from sdm_client._rust import codec
from sdm_client._rust.frames import _convert


def inputs(schema: Any, state: dict[str, Any]) -> dict[int, Any]:
    """Convert live inputs once; resume from opaque checkpoint owners."""
    import pyarrow as pa

    tables = []
    attached: dict[int, Any] = {}
    checkpoint = state.get('held', {}).get('frame_checkpoint')
    if checkpoint is None:
        for number, (name, table) in enumerate(schema.tables.items(), 8):
            frame = table.data
            if type(frame) is not pd.DataFrame:
                raise RouteError(
                    'the partial frame materializer requires built-in DataFrames'
                )
            rows, native_types = _convert(name, frame)
            # Exact Arrow LargeUTF8 has a frozen native-engine failure;
            # StringDtype exports can use the same physical offset width.
            # Keep this native family unfinished after generic conversion
            # validation, and never inspect caller frames on checkpoint reuse.
            if any(
                type(dtype) is pd.ArrowDtype
                and pa.types.is_large_string(dtype.pyarrow_dtype)
                for dtype in native_types.values()
            ):
                raise RouteError(
                    'exact Arrow large_string columns remain unfinished in the partial frame materializer'
                )
            attached[number] = rows
            tables.append({'source': codec.relation(name), 'rows': number})
    attached[7] = codec.attachment(
        {'tables': tables, 'entropy': secrets.randbits(64)}
    )
    return attached


def rebind(
    held: dict[str, Any], attached: dict[int, Any]
) -> dict[str, Any] | None:
    """Bind retained owners under this call's ids, without reading metadata."""
    checkpoint = held.get('frame_checkpoint')
    if checkpoint is None:
        return None
    next_id = max(attached, default=0) + 1
    metadata = next_id
    attached[metadata] = checkpoint['metadata']
    tables = []
    for table in checkpoint['tables']:
        next_id += 1
        attached[next_id] = table['rows']
        tables.append({'source': table['source'], 'rows': next_id})
    next_id += 1
    attached[next_id] = codec.attachment(
        {'metadata': metadata, 'tables': tables}
    )
    return {'version': checkpoint['version'], 'manifest': next_id}


def take(handle: Any, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Take every exported owner before dropping the completed native run."""
    import json

    import pyarrow as pa

    manifest = json.loads(handle.take_attachment(checkpoint['manifest']))
    metadata = handle.take_attachment(manifest['metadata'])
    tables = []
    for table in manifest['tables']:
        owner = handle.take_attachment(table['rows'])
        tables.append({'source': table['source'], 'rows': pa.table(owner)})
    return {
        'version': checkpoint['version'],
        'metadata': metadata,
        'tables': tables,
    }
