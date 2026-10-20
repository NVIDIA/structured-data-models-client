# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``kumo-tabular`` in process: one table, each row scored from its columns."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from sdm_client.errors import StructuredDataError
from sdm_client.in_process.columns import (
    Context,
    Shape,
    key_columns,
    labelled,
    stype_of,
)
from sdm_client.in_process.tables import values

_READS = frozenset({'numerical', 'categorical'})


class Tabular:
    r"""Reads a ``kumo-tabular`` request into the model."""

    model_id = 'kumo-tabular'
    outputs = {
        'classification': ('prediction', 'probabilities'),
        'regression': ('prediction', 'quantiles'),
    }

    @staticmethod
    def load(device: Any) -> Any:
        from sdm.models import KumoTabular

        return KumoTabular(device=device)

    @staticmethod
    def accepts(model: Any) -> bool:
        from sdm.models import KumoTabular

        return isinstance(model, KumoTabular)

    def context(self, payload: Mapping[str, Any]) -> Context:
        task = payload['task']
        self.check_inference(payload.get('inference') or {}, task)
        spec = payload['schema']['instance_table']
        table = payload['context']['instance_table']
        target = task['target']['column_name']
        keys = key_columns(spec)
        stypes = {
            name: stype_of(name, spec['columns'][name])
            for name in table['columns']
            if name != target and name not in keys
        }
        unread = sorted(
            name for name, stype in stypes.items() if stype not in _READS
        )
        if unread:
            raise StructuredDataError(
                f'{self.model_id} reads numerical and categorical columns, '
                f'not {unread}',
                code='UNSUPPORTED_DTYPE',
            )
        instance = Shape.of(spec, stypes)
        y, encoded, labels = labelled(table, task, task['kind'] != 'regression')
        return Context(
            x=instance.tensor(table, 'the context'),
            y=y,
            instance=instance,
            encoded=encoded,
            labels=labels,
        )

    def check_inference(
        self,
        options: Mapping[str, Any],
        task: Mapping[str, Any],
        settled: bool = False,
    ) -> None:
        r"""Refuse an inference option this model does not act on."""
        remaining = dict(options)
        if remaining.pop('inference_config', {}):
            raise StructuredDataError(
                'inference_config is reserved for API compatibility and must '
                'be omitted or empty',
                code='INVALID_REQUEST',
            )
        if remaining.pop('use_prediction_time', False):
            raise StructuredDataError(
                f"inference option 'use_prediction_time' is not supported by "
                f'{self.model_id}; only the neutral value False is accepted',
                code='INVALID_REQUEST',
            )
        unknown = sorted(set(remaining) - {'run_mode'})
        if unknown:
            raise StructuredDataError(
                f'unsupported inference options: {unknown}',
                code='INVALID_REQUEST',
            )

    @staticmethod
    def query(
        context: Context,
        payload: Mapping[str, Any],
        schema: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        r"""The rows to score, read as the context was, each carrying the
        key its answer is given back under."""
        table = payload['predict']['instance_table']
        absent = sorted(
            key_columns(schema['instance_table']) - set(table['columns'])
        )
        if absent:
            raise StructuredDataError(
                f'the rows to score leave out key column(s) {absent}, which '
                f'name the row each answer is about',
                code='INVALID_SCHEMA',
            )
        return context.instance.tensor(table, 'predict instance table'), None

    @staticmethod
    def row_ids(
        table: Mapping[str, Any], spec: Mapping[str, Any]
    ) -> list[str | None]:
        r"""A key of several columns answers under the JSON of its values as
        they were sent, which the client hands back as the row's id.

        Otherwise there is no id: each answer is matched to its row by
        position, and the client puts the row's own key on it.
        """
        key = spec.get('primary_key')
        if not isinstance(key, list) or len(key) < 2:
            return [None] * len(table['rows'])
        cells = [values(table, name) for name in key]
        return [
            json.dumps(list(values_of), separators=(',', ':'), default=str)
            for values_of in zip(*cells, strict=True)
        ]
