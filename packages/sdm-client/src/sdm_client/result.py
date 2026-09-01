# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What a prediction gives back.

One type, always. The older design returned a DataFrame, or an explanation
object if you had asked to explain, which made the return an undocumented
union. Here an explanation is an output field like any other:

>>> result = client.predict(request, model=KumoRelational(),
...                         outputs=['prediction', 'explanation'])
>>> result.to_pandas()       # always a frame
>>> result.explanation       # None unless you asked for it

The frame's shape depends on the **task**, not the model: classification gives
one row per entity, multiclass one row per entity per class, ranking ``k`` rows
per entity, forecasting one row per entity per step.

Every row carries ``row_index``. It also carries ``id`` when the model has an
entity key -- a single-table request may have none. **Join on the key when it
is there, otherwise on** ``row_index``, **and never on row order.**
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import pandas as pd
from sdm_api.generated.tfm_api import PredictionItem

# Emitted as columns when present on an item, in this order. ``row_index``
# leads because it is the one field the contract always requires.
_SCALAR_FIELDS = ('row_index', 'id', 'forecast_step', 'prediction')
# Spread into one column each, under this prefix. Named explicitly rather than
# singularised by stripping an 's', which turns 'probabilities' into
# 'probabilitie'.
_MAPPING_FIELDS = {'probabilities': 'probability', 'quantiles': 'quantile'}
_SEQUENCE_FIELDS = ('scores', 'embeddings')


class PredictionResult:
    r"""The answer to one request.

    Attributes:
        model: The model id that answered.
        request_id: The endpoint's id for the call, useful when reading its
            logs.
        items: The response items, in the order the endpoint returned them.
        metadata: Whatever the endpoint reported alongside the predictions.
    """

    def __init__(
        self,
        *,
        model: str,
        request_id: str,
        items: Sequence[PredictionItem],
        metadata: dict[str, Any] | None = None,
        explanation: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.request_id = request_id
        self.items = tuple(items)
        self.metadata = dict(metadata or {})
        self._explanation = explanation

    @property
    def explanation(self) -> dict[str, Any] | None:
        r"""The explanation, when asked for and the model produced one.

        ``None`` otherwise, rather than a different return type.
        """
        return self._explanation

    def to_pandas(self) -> pd.DataFrame:
        r"""The predictions as a frame.

        A column appears when at least one item carries that field, so a
        classification result has ``probabilities`` and a regression one has
        ``quantiles`` without either inventing the other. A probability or
        quantile mapping is spread into one column per key, prefixed, so the
        frame stays flat and joinable.

        Rows come back ordered by ``row_index``. The endpoint is not required
        to preserve request order, and a caller who reads the first few rows
        would otherwise be reading whichever rows finished first.
        """
        rows: list[dict[str, Any]] = []
        for item in self.items:
            row: dict[str, Any] = {}
            for name in _SCALAR_FIELDS:
                value = getattr(item, name, None)
                if value is not None:
                    row[name] = value
            for name, prefix in _MAPPING_FIELDS.items():
                mapping = getattr(item, name, None)
                if mapping:
                    for key, value in mapping.items():
                        row[f'{prefix}_{key}'] = value
            for name in _SEQUENCE_FIELDS:
                values = getattr(item, name, None)
                if values is not None:
                    row[name] = list(values)
            if item.rankings:
                row['rankings'] = [dict(entry) for entry in item.rankings]
            rows.append(row)

        frame = pd.DataFrame(rows)
        if 'row_index' in frame.columns:
            frame = frame.sort_values('row_index', kind='stable').reset_index(
                drop=True
            )
        return frame

    @classmethod
    def merge(cls, parts: Sequence[PredictionResult]) -> PredictionResult:
        r"""Join the results of one request that was sent in several calls.

        A request larger than the endpoint's per-call cap is split by the
        adapter, and each call answers about its own rows. Keeping only the
        last answer would silently drop every earlier batch, so the parts are
        concatenated here.

        Raises:
            ValueError: If ``parts`` is empty, which would mean a request was
                sent with nothing to score.
        """
        if not parts:
            raise ValueError('no parts to merge')
        if len(parts) == 1:
            return parts[0]
        first = parts[0]
        models = {part.model for part in parts}
        if len(models) > 1:
            raise ValueError(
                f'cannot merge results from different models: {sorted(models)}'
            )
        items = [item for part in parts for item in part.items]
        metadata = dict(first.metadata)
        metadata['batches'] = len(parts)
        # Each call has its own id in the endpoint's logs, and only the first
        # would otherwise survive -- which is the one you do not have when a
        # later batch is the one that misbehaved.
        metadata['batch_request_ids'] = [part.request_id for part in parts]
        explanation = next(
            (part.explanation for part in parts if part.explanation), None
        )
        return cls(
            model=first.model,
            request_id=first.request_id,
            items=items,
            metadata=metadata,
            explanation=explanation,
        )

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[PredictionItem]:
        return iter(self.items)

    def __repr__(self) -> str:
        explained = ', explained' if self._explanation is not None else ''
        return (
            f'PredictionResult({self.model!r}, {len(self.items)} rows'
            f'{explained})'
        )
