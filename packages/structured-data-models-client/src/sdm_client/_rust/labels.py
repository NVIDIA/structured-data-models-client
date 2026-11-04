# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host adaptation for Rust ground-truth extraction; no model transport."""

from __future__ import annotations

from typing import Any

import pandas as pd

from sdm_client._rust import codec, providers
from sdm_client._rust.frame_materialize import _graph
from sdm_client._rust.materialize import _anchor
from sdm_client._rust.run import run
from sdm_client._rust.spec import held_schema
from sdm_client.errors import StructuredDataError


def extract(
    materializer: Any,
    query: str,
    size: int,
    anchor_time: Any,
    random_seed: int | None,
    max_iterations: int,
) -> pd.DataFrame:
    for name, value, minimum in (
        ('size', size, 0),
        ('max_iterations', max_iterations, 1),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= 2**64 - 1
        ):
            raise StructuredDataError(
                f'{name} must be an integer from {minimum} to {2**64 - 1}',
                code='INVALID_REQUEST',
            )
    if random_seed is not None and codec.seed(random_seed) is None:
        raise StructuredDataError(
            'random_seed must be an integer or None', code='INVALID_REQUEST'
        )
    call = {
        'query': query,
        'size': size,
        'max_iterations': max_iterations,
        'anchor_time': _anchor(anchor_time),
        'anchor_entity': isinstance(anchor_time, str)
        and anchor_time == 'entity',
        'seed': codec.seed(random_seed),
        'unseeded': random_seed is None,
    }
    schema = materializer.schema
    if schema.source is None:
        with materializer._core_frame_lock:
            finished = run(
                {'op': 'labels_frames', 'graph': 1, 'input': 2},
                {1: _graph(materializer), 2: codec.attachment(call)},
            )
    else:
        choice, provider = providers.for_source(schema.source)
        finished = run(
            {
                'op': 'labels_source',
                'provider': choice,
                'schema': 1,
                'input': 2,
            },
            {
                1: codec.attachment(held_schema(schema)),
                2: codec.attachment(call),
            },
            provider=provider,
        )
    if finished.result.get('op') != 'labels':
        raise StructuredDataError(
            'unexpected labels result', code='INTERNAL_ERROR'
        )
    import pyarrow as pa

    # Arrow 14 supports capsule import, but not the public from_stream helper.
    with pa.RecordBatchReader._import_from_c_capsule(
        finished.take(finished.result['data']).__arrow_c_stream__()
    ) as reader:
        return reader.read_all().to_pandas()
