# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host ownership for Rust relational materialization and continuing sampler state."""

from __future__ import annotations

import copy
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from sdm_client.errors import StructuredDataError
from sdm_client.request import Request
from sdm_client.schema import Schema

if TYPE_CHECKING:
    from sdm_client._rust.results import CoreQuery
    from sdm_client.sanitization import SanitizationReport


class RelationalMaterializer:
    """Retain host resources while Rust owns sampling and request construction."""

    _sanitization_report: SanitizationReport

    _MATERIALIZE_OPTIONS = frozenset(
        {
            'anchor_time',
            'context_anchor_time',
            'lag_timesteps',
            'max_pq_iterations',
            'num_hops',
            'num_neighbors',
            'random_seed',
            'use_prediction_time',
            'inference_config',
            'exclude_cols_dict',
            'return_embeddings',
            'top_k',
            'prediction_statistic',
            'rows_per_call',
        }
    )

    def __init__(self, schema: Schema) -> None:
        self.schema = schema
        self._core_sampler_state: dict[str, Any] = {}
        self._core_frame_graph: Any = None
        self._core_frame_lock = threading.Lock()

    def materialize(
        self,
        *,
        query: CoreQuery,
        indices: Sequence[Any],
        num_context_samples: int | None,
        options: dict[str, Any],
        candidates: Sequence[Any] | None = None,
    ) -> Request:
        from sdm_client.inference import InferenceConfig

        if isinstance(options.get('inference_config'), InferenceConfig):
            options = {
                **options,
                'inference_config': options['inference_config'].to_dict(),
            }
        unknown = sorted(set(options) - self._MATERIALIZE_OPTIONS)
        if unknown:
            raise StructuredDataError(
                f'unknown materialize option(s) {unknown}; accepted: '
                f'{sorted(self._MATERIALIZE_OPTIONS)}',
                code='INVALID_REQUEST',
            )
        if self.schema.source is None:
            from sdm_client._rust.frame_materialize import gather
        else:
            from sdm_client._rust.materialize import gather
        payloads, facts, retained = gather(
            self, query, indices, num_context_samples, options, candidates
        )
        request = Request.materialized(
            payloads=payloads,
            windowed=facts.windowed,
            schema=self.schema.snapshot(),
            query=query.to_string(),
            indices=list(indices),
            entity_table=facts.entity_table,
            requested_inference={
                'inference_config': copy.deepcopy(
                    options.get('inference_config') or {}
                )
            },
        )
        request._core = retained
        if self.schema.source is None:
            request._sanitization_report = self._sanitization_report
        return request
