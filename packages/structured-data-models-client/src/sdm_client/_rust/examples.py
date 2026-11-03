# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt supplied example tables; all task and graph decisions stay in Rust."""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from sdm_client._rust import codec, providers, results
from sdm_client._rust.frame_materialize import _graph
from sdm_client._rust.materialize import unpack_state
from sdm_client._rust.run import run
from sdm_client._rust.spec import held_schema
from sdm_client.errors import StructuredDataError
from sdm_client.relational_task import RelationalTask

if TYPE_CHECKING:
    from sdm_client.request import Request
    from sdm_client.schema import Schema


def materialize(
    schema: Schema,
    task: RelationalTask,
    context: Any,
    predict: Any,
    *,
    num_neighbors: Any,
    random_seed: int,
    inference: Any,
) -> Request:
    import pandas as pd
    import pyarrow as pa

    from sdm_client.materializers.relational import RelationalMaterializer
    from sdm_client.request import Request

    if not isinstance(task, RelationalTask):
        raise StructuredDataError(
            'task must be a RelationalTask', code='INVALID_REQUEST'
        )

    def arrow(value: Any) -> Any:
        if isinstance(value, pa.Table):
            return value
        if isinstance(value, pa.RecordBatch):
            return pa.Table.from_batches([value])
        if isinstance(value, pd.DataFrame):
            return pa.Table.from_pandas(value, preserve_index=False)
        raise StructuredDataError(
            'examples must be DataFrames or Arrow tables',
            code='INVALID_REQUEST',
        )

    from sdm_client.inference import InferenceConfig

    if isinstance(inference, InferenceConfig):
        inference = inference.to_dict()
    context, predict = arrow(context), arrow(predict)
    input = {
        'task': asdict(task),
        'num_neighbors': None if num_neighbors is None else list(num_neighbors),
        'random_seed': codec.seed(random_seed),
    }
    attachments: dict[int, Any] = {
        2: codec.attachment(input),
        3: context,
        4: predict,
        5: codec.attachment(inference or {}, named=True, option_values=True),
    }
    owner = None
    provider = None
    if schema.source is None:
        owner = RelationalMaterializer(schema)
        attachments[1] = _graph(owner)
        source = {'source': 'frames', 'graph': 1}
    else:
        choice, provider = providers.for_source(schema.source)
        attachments[1] = codec.attachment(held_schema(schema))
        source = {'source': 'sql', 'provider': choice, 'schema': 1}
    finished = run(
        {
            'op': 'materialize_examples',
            'source': source,
            'input': 2,
            'context': 3,
            'predict': 4,
            'inference': 5,
        },
        attachments,
        provider=provider,
    )
    gathered = results.expect(finished.result, results.Gathered)
    facts = gathered.examples
    if facts is None:
        raise StructuredDataError(
            'the core omitted supplied task facts', code='INTERNAL_ERROR'
        )
    payloads, retained = unpack_state(finished, gathered.state)
    request = Request.materialized(
        payloads=payloads,
        windowed=facts.windowed,
        schema=schema.snapshot(),
        query='',
        indices=predict['ENTITY'].to_pylist(),
        entity_table=facts.entity_table,
        requested_inference={
            'inference_config': copy.deepcopy(inference or {})
        },
    )
    request._core = retained
    if owner is not None:
        request._sanitization_report = owner._sanitization_report
    return request
