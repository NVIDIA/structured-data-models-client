# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-less materialization: typed frames in, existing request bytes out.

The native graph is retained by one RelationalMaterializer, just as its old
engine graph was. Rust owns graph construction, sampling and model policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sdm_client._rust import codec, errors, results
from sdm_client._rust.materialize import (
    TAKEN,
    VALUED,
    _anchor,
    _candidates,
    _context_anchor,
    _indices,
    unpack,
)
from sdm_client._rust.run import run
from sdm_client._rust.spec import held_schema


def _graph(materializer: Any) -> Any:
    if materializer._core_frame_graph is not None:
        return materializer._core_frame_graph
    import pyarrow as pa

    schema = materializer.schema
    names = list(schema.tables)
    attachments: dict[int, Any] = {
        1: codec.attachment(held_schema(schema)),
        2: codec.attachment(names),
    }
    ids = []
    for number, name in enumerate(names, start=3):
        table = schema[name]
        arrays = []
        for column in table.columns:
            array = pa.array(table.data[column])
            if pa.types.is_dictionary(array.type):
                array = array.dictionary_decode()
            arrays.append(array)
        attachments[number] = pa.Table.from_arrays(
            arrays, names=list(table.columns)
        )
        ids.append(number)
    finished = run(
        {'op': 'prepare_frame_graph', 'schema': 1, 'names': 2, 'tables': ids},
        attachments,
    )
    result = results.expect(finished.result, results.PreparedFrameGraph)
    materializer._core_frame_graph = finished.take(result.graph)
    return materializer._core_frame_graph


def gather(
    materializer: Any,
    query: Any,
    indices: Sequence[Any],
    num_context_samples: int | None,
    options: dict[str, Any],
    candidates: Sequence[Any] | None = None,
) -> tuple[tuple[dict[str, Any], ...], results.QueryFacts, results.CoreRequest]:
    from sdm_client import decompose
    from sdm_client.materializers import relational

    anchor = options.get('anchor_time')
    call = {
        'query': query.to_string(),
        'candidates': None if candidates is None else _candidates(candidates),
        'anchor_time': _anchor(anchor),
        'anchor_entity': isinstance(anchor, str) and anchor == 'entity',
        'seed': codec.seed(options.get('random_seed')),
        'unseeded': 'random_seed' in options and options['random_seed'] is None,
        'options': sorted(set(options) - TAKEN),
        'most_parts': max(0, min(int(decompose.MOST_PARTS), 2**32 - 1)),
        'most_entities': max(0, min(int(relational.MOST_ENTITIES), 2**64 - 1)),
        'most_calls': max(0, min(int(relational.MOST_CALLS), 2**32 - 1)),
    }
    attachments = {
        1: _graph(materializer),
        2: codec.attachment(call),
        3: _indices(indices),
        4: codec.attachment(
            named=True,
            option_values=True,
            value={
                'num_context_samples': num_context_samples,
                **{
                    name: options[name]
                    for name in sorted(set(options) & VALUED)
                },
                **_context_anchor(options),
            },
        ),
    }
    with errors.holding(
        inference_option=lambda fields: options['inference_config'][
            fields['key']
        ]
    ):
        finished = run(
            {
                'op': 'materialize_frames',
                'graph': 1,
                'input': 2,
                'indices': 3,
                'values': 4,
            },
            attachments,
        )
    return unpack(finished)
