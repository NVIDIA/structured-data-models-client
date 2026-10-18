# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``kumo-tabular`` in process: one table, each row scored from its columns."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sdm_client.in_process.columns import (
    Context,
    Shape,
    key_columns,
    labelled,
    stype_of,
)
from sdm_client.in_process.refusal import refusal

_READS = frozenset({'numerical', 'categorical'})


class Tabular:
    r"""Reads a ``kumo-tabular`` request into the model, as its NIM does."""

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
        schema = payload['schema']
        related = schema.get('related_tables') or any(
            (payload.get(split) or {}).get('related_tables')
            for split in ('context', 'predict')
        )
        for present, what in (
            (schema.get('relationships'), 'relationships'),
            (related, 'related tables'),
        ):
            if present:
                raise refusal(
                    'UNSUPPORTED_FEATURE',
                    f'{what} are not supported by model {self.model_id!r}.',
                )
        spec = schema['instance_table']
        table = payload['context']['instance_table']
        task = payload['task']
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
            raise refusal(
                'UNSUPPORTED_DTYPE',
                f'Kumo Tabular does not support these feature columns: {unread}',
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
        r"""Refuse an inference option this model's NIM refuses."""
        remaining = dict(options)
        if remaining.pop('inference_config', {}):
            raise refusal(
                'INVALID_REQUEST',
                'inference_config is reserved for API compatibility and must '
                'be omitted or empty',
            )
        if remaining.pop('use_prediction_time', False):
            raise refusal(
                'INVALID_REQUEST',
                f"inference option 'use_prediction_time' is not supported by "
                f'{self.model_id}; only the neutral value False is accepted',
            )
        unknown = sorted(set(remaining) - {'run_mode'})
        if unknown:
            raise refusal(
                'INVALID_REQUEST', f'unsupported inference options: {unknown}'
            )

    @staticmethod
    def query(
        context: Context,
        payload: Mapping[str, Any],
        schema: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        r"""The rows to score, read as the context was."""
        table = payload['predict']['instance_table']
        return context.instance.tensor(table, 'predict instance table'), None
