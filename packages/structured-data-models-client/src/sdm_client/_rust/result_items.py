# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn projected Arrow fields into host values, without result policy."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from sdm_client._rust import codec, results
from sdm_client._rust.run import run


def primitive(value: Any) -> Any:
    """Decode the contract's existing lossless PrimitiveOption representation."""
    if not isinstance(value, dict) or len(value) != 1:
        raise codec.CodecError('invalid projected primitive')
    tag, body = next(iter(value.items()))
    if tag == 'value':
        return codec.decode_scalar(body)
    if tag == 'integer' and isinstance(body, str):
        return int(body)
    if tag == 'array' and isinstance(body, list):
        return [primitive(item) for item in body]
    if tag == 'object' and isinstance(body, list):
        return {key: primitive(item) for key, item in body}
    raise codec.CodecError('unknown projected primitive')


def read(
    table: Any,
    *,
    model: str,
    request_id: str,
    metadata: Mapping[str, Any],
    outputs: Sequence[str],
    candidates: Mapping[int, Any] | None = None,
) -> list[dict[str, Any]]:
    """Consume a shared projection; positions resolve to original caller values."""
    import pyarrow as pa

    descriptor = {
        'model': model,
        'request_id': request_id,
        'metadata': json.dumps(metadata),
        'outputs': list(outputs),
        'unscored_candidates': [],
    }
    finished = run(
        {'op': 'read_result_items', 'input': 1, 'table': 2},
        {1: codec.attachment(descriptor), 2: table},
    )
    answer = results.expect(finished.result, results.ResultItems)

    def candidate(position: Any) -> Any:
        if (
            type(position) is not int
            or candidates is None
            or position not in candidates
        ):
            raise codec.CodecError('projected candidate position has no owner')
        return candidates[position]

    rows = []
    for segment in answer.segments:
        stream = finished.take(segment.table)
        with pa.RecordBatchReader._import_from_c_capsule(
            stream.__arrow_c_stream__()
        ) as reader:
            projected = reader.read_all()
        if projected.column_names != [name for name, _ in segment.fields]:
            raise codec.CodecError(
                'projected fields do not match Arrow columns'
            )
        maps = {
            field.name
            for field in projected.schema
            if pa.types.is_map(field.type)
        }

        segment_rows = projected.to_pylist()
        for row in segment_rows:
            for name, encoding in segment.fields:
                value = row[name]
                if value is None:
                    continue
                if encoding in ('value', 'mapping', 'mappings'):
                    value = primitive(json.loads(value))
                    if encoding == 'mapping' and value is not None:
                        value = dict(value)
                    elif encoding == 'mappings' and value is not None:
                        value = [dict(entry) for entry in value]
                elif encoding == 'candidate_list':
                    value = [candidate(position) for position in value]
                elif encoding == 'candidate_rankings':
                    value = [
                        {
                            'candidate': candidate(entry['position']),
                            'score': entry['score'],
                        }
                        for entry in value
                    ]
                elif encoding == 'native' and name in maps:
                    value = dict(value)
                if (
                    name in ('scores', 'embeddings', 'rankings')
                    and value is not None
                ):
                    value = tuple(value)
                row[name] = value
        rows.extend(segment_rows)
    return rows
