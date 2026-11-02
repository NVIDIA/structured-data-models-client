# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``kumo-anomaly`` in process: one table of readings, scored as a series.

This model is unlike the other two in the one way that matters here. It is
unsupervised, so there is no context to fit and no target column to name: it
reconstructs the rows it is given and scores each one by how badly the
reconstruction missed. That is why :class:`Anomaly` has no ``context`` or
``query``, and why a session over it is refused rather than pinned.

The answer is three numbers per row, and the contract's ``PredictionItem`` has
nowhere to put three. ``prediction`` carries the score, which is the answer to
the question that was asked. The two labels ride in ``probabilities``, whose
name is wrong for them and whose shape is right: a mapping that reaches the
frame as one flat column each. The contract is generated from a spec that does
not describe this model, so widening it is not this client's to do.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from sdm_api.generated.tfm_api import PredictionItem

from sdm_client.errors import StructuredDataError
from sdm_client.in_process.columns import (
    Shape,
    check_columns,
    key_columns,
    stype_of,
)
from sdm_client.in_process.tables import values

#: The thresholders the model applies, in the order it returns them.
THRESHOLDERS = ('scs', 'macs')

#: How the published deployment samples. Reconstructions are combined by their
#: median, so fewer samples is a noisier score rather than a different one.
_SAMPLES = 30
_STEPS = 20


class Anomaly:
    r"""Reads a ``kumo-anomaly`` request into the model."""

    model_id = 'kumo-anomaly'
    outputs = {'anomaly_detection': ('prediction', 'probabilities')}

    @staticmethod
    def load(device: Any) -> Any:
        from sdm.models import KumoAnomaly

        return KumoAnomaly(device=device)

    @staticmethod
    def accepts(model: Any) -> bool:
        from sdm.models import KumoAnomaly

        return isinstance(model, KumoAnomaly)

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
        unknown = sorted(set(remaining) - {'num_samples', 'num_steps'})
        if unknown:
            raise StructuredDataError(
                f'unsupported inference options: {unknown}',
                code='INVALID_REQUEST',
            )
        for name in ('num_samples', 'num_steps'):
            value = remaining.get(name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise StructuredDataError(
                    f'{name} is {value!r}; it is a whole number of at least 1',
                    code='INVALID_REQUEST',
                )
            if value < 1:
                raise StructuredDataError(
                    f'{name} is {value}; it is a whole number of at least 1',
                    code='INVALID_REQUEST',
                )

    def rows(self, payload: Mapping[str, Any]) -> Any:
        r"""The readings to score, as the model reads them.

        Every column is a channel of one multivariate series, so the whole
        table is read rather than all but a target. A key column names the
        row instead of measuring anything, so reconstructing it would score
        rows by how surprising their identifiers are.
        """
        spec = payload['schema']['instance_table']
        table = payload['predict']['instance_table']
        check_columns(table, spec, 'the rows to score')
        keys = key_columns(spec)
        stypes = {
            name: stype_of(name, spec['columns'][name])
            for name in table['columns']
            if name not in keys
        }
        unread = sorted(
            name for name, stype in stypes.items() if stype != 'numerical'
        )
        if unread:
            raise StructuredDataError(
                f'{self.model_id} reconstructs numerical columns, and '
                f'{unread} is not one. A reading it cannot reconstruct has no '
                f'error to score, so leave it out of the rows you send',
                code='UNSUPPORTED_DTYPE',
            )
        if not stypes:
            raise StructuredDataError(
                f'{self.model_id} scores a row by reconstructing it, and '
                f'these rows carry no numerical column to reconstruct',
                code='INVALID_REQUEST',
            )
        return Shape.of(spec, stypes).tensor(table, 'the rows to score')

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
        if not set(key) <= set(table['columns']):
            # The readings need not carry the key: it names the row and is
            # not a channel, so a caller may leave it out. Then there is no
            # value to answer under and each answer is matched by position.
            return [None] * len(table['rows'])
        cells = [values(table, name) for name in key]
        return [
            json.dumps(list(values_of), separators=(',', ':'), default=str)
            for values_of in zip(*cells, strict=True)
        ]

    def answer(
        self,
        model: Any,
        payload: Mapping[str, Any],
        ids: Sequence[Any],
        fields: Sequence[str],
        generator: Any = None,
    ) -> list[PredictionItem]:
        r"""Score each row, under the id the request gave it.

        ``generator`` is what makes the same readings score the same twice.
        Reconstruction is sampled, so without one an unchanged request comes
        back with different numbers every call.
        """
        options = payload.get('inference') or {}
        # The rows are read into CPU tensors, and the model may be anywhere.
        # The fitted path moves its context the same way; this one has no
        # context, so the rows are all there is to move.
        rows = self.rows(payload)
        if generator is not None:
            rows = rows.to(generator.device)
        out = model(
            rows,
            num_samples=int(options.get('num_samples') or _SAMPLES),
            num_steps=int(options.get('num_steps') or _STEPS),
            generator=generator,
        )
        scored = out['mae'].numerical[..., 0]
        if not bool(scored.isfinite().all()):
            # The same refusal the other models make. A score that is not a
            # number is not a row that is unremarkable, and returning one
            # hands a caller a frame they cannot threshold or plot.
            raise StructuredDataError(
                'the model answered with a malformed output',
                code='INTERNAL_ERROR',
            )
        scores = scored.tolist()
        labelled = 'probabilities' in fields
        labels = (
            {
                name: out[name].categorical.decode()[..., 0].tolist()
                for name in THRESHOLDERS
            }
            if labelled
            else {}
        )
        return [
            PredictionItem(
                row_index=index,
                id=ids[index] if index < len(ids) else None,
                prediction=score,
                probabilities={
                    name: float(labels[name][index]) for name in THRESHOLDERS
                }
                if labelled
                else None,
            )
            for index, score in enumerate(scores)
        ]
