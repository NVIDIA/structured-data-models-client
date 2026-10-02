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

The frame's shape depends on the **task**, not the model. Classification and
multiclass both give one row per entity. Multiclass carries a probability per
class as columns rather than a row per class, while forecasting gives one row
per step. Ranking is the exception to the row-per-answer rule: the ``k`` ranked
candidates arrive as a **list in one row**, under ``rankings``, because
expanding them would make the frame's key no longer identify a row. A
candidate a ranking could not score is in no row, and is named on
``unscored_candidates`` instead.

Every row carries ``row_index``. It also carries ``id`` when the model has an
entity key, though a single-table request may have none. **Join on the key when
it is there, otherwise on** ``row_index``, **and never on row order.**
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import pandas as pd
from sdm_api.generated.tfm_api import PredictionItem

from sdm_client.errors import StructuredDataError

# Emitted as columns when present on an item, in this order. ``row_index``
# leads because it is the one field the contract always requires.
_SCALAR_FIELDS = ('row_index', 'id', 'forecast_step', 'prediction')
# Spread into one column each, under this prefix. Named explicitly rather than
# singularised by stripping an 's', which turns 'probabilities' into
# 'probabilitie'.
_MAPPING_FIELDS = {'probabilities': 'probability', 'quantiles': 'quantile'}
_SEQUENCE_FIELDS = ('scores', 'embeddings')

_ITEM_FIELDS = (
    'prediction',
    'probabilities',
    'quantiles',
    'scores',
    'rankings',
    'embeddings',
)


class PredictionResult:
    r"""The answer to one request.

    Attributes:
        model: The model id that answered.
        request_id: The endpoint's id for the call, useful when reading its
            logs.
        items: The response items, in the order the endpoint returned them.
            Each item's ``id`` is the key that was asked about, in the type
            the schema gave it, not the string the wire carried. The wire
            spells every id as text, and a relational request spells it as an
            opaque transport reference, so an item handed back unchanged would
            answer a question about ``uid 3`` with something that is neither
            ``3`` nor recognisably about it.
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
        outputs: Sequence[str] = (),
        unscored_candidates: Sequence[Any] = (),
    ) -> None:
        self.model = model
        self.request_id = request_id
        self.items = tuple(items)
        self.outputs = tuple(outputs)
        # Whether answers to this request name the row each one is about.
        # Read only when there are no rows to read it from; `_empty_answer`
        # sets it, because only the request knows.
        self._names_its_rows = True
        # Whether answers to this request say which forecast step they are
        # about. Set where a forecast is assembled, for the same reason.
        self._names_its_step = False
        self.metadata = dict(metadata or {})
        self._explanation = explanation
        self._unscored_candidates = tuple(unscored_candidates)

    def absent(self, requested: Sequence[str]) -> tuple[str, ...]:
        r"""Which of ``requested`` the answer does not carry.

        A field a model produces for one task kind and not another is accepted
        by the endpoint, answered `200`, and simply left out. A caller who
        names it reads a frame that has no such column and nothing anywhere
        said why. Rather than deciding in advance which model produces what for
        which kind, which is a moving target across deployments, this reports
        what the answer turned out not to contain.

        An empty batch is a supported request with a real empty answer, so
        nothing can be looked for and nothing is reported. Calling every field
        missing there would make the one case that is always legitimate the
        noisiest.
        """
        if not self.items:
            return ()
        return tuple(
            name for name in dict.fromkeys(requested) if not self._carries(name)
        )

    def _carries(self, name: str) -> bool:
        r"""Whether the answer holds anything under ``name``.

        A name this type does not know how to look for is treated as carried:
        reporting it absent would be a guess, and a field no model produces
        was refused before the request was sent.
        """
        if name == 'explanation':
            return self._explanation is not None
        if name not in _ITEM_FIELDS:
            return True
        return any(
            getattr(item, name, None) not in (None, {}, [], ())
            for item in self.items
        )

    @property
    def explanation(self) -> dict[str, Any] | None:
        r"""The explanation, when asked for and the model produced one.

        ``None`` otherwise, rather than a different return type.
        """
        return self._explanation

    @property
    def unscored_candidates(self) -> tuple[Any, ...]:
        r"""The candidates a ranking left out, in the order they were named.

        A candidate is left out when the context sampled for it holds one
        answer only, which nearly always means no entity in it linked to the
        candidate in the window. The endpoint refuses that question, so no
        call is made for it and no score is invented for it: a row asked about
        in that ranking does not hold it in ``prediction`` or ``rankings``,
        and may therefore rank fewer than ``k`` candidates.

        On a merge of rankings asked separately, a candidate is named here when
        it was left out of at least one part's rows, and may be scored in the
        rows of another part.

        Empty when every candidate was scored, and on any result that is not
        a ranking.
        """
        return self._unscored_candidates

    def to_pandas(self) -> pd.DataFrame:
        r"""The predictions as a frame.

        A column appears when at least one item carries that field, and
        not otherwise: this reads the answer, not the task or the fields
        that were asked for, so nothing is invented to fill a shape and
        nothing carried is dropped. A probability or
        quantile mapping is spread into one column per key, prefixed, so the
        frame stays flat and joinable.

        Rows come back ordered by ``row_index``. The endpoint is not required
        to preserve request order, and a caller who reads the first few rows
        would otherwise be reading whichever rows finished first.

        A result with no rows still has columns. Scoring an empty frame is
        legal because the contract puts no minimum on ``rows``. It is what a
        filter that matched nothing produces, so it happens inside ordinary
        loops. Returning a frame with no columns at all made that case fail
        differently from every other: ``frame['prediction']`` became a
        ``KeyError`` and a join on ``row_index`` could not be written.

        The per-class columns cannot be known with no rows to read them
        from, so an empty classification result carries the fields that
        were asked for but never a ``probability_*``.
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

        frame = pd.DataFrame(rows) if rows else self._empty_frame()
        if 'row_index' in frame.columns:
            frame = frame.sort_values('row_index', kind='stable').reset_index(
                drop=True
            )
        return frame

    def _empty_frame(self) -> pd.DataFrame:
        r"""The columns this request would have produced, with no rows.

        ``row_index`` and ``id`` say which row an answer belongs to rather
        than what the answer is, so they are here whatever was asked for.
        The join a caller writes must not depend on the batch being empty.
        ``id`` only when this request names its rows at all, which is the
        same condition a non-empty answer carries it under.
        """
        columns = ['row_index']
        if self._names_its_rows:
            columns.append('id')
        if self._names_its_step:
            columns.append('forecast_step')
        columns += [
            name
            for name in (*_SCALAR_FIELDS, *_SEQUENCE_FIELDS, 'rankings')
            if name not in ('row_index', 'id', 'forecast_step')
            and name in self.outputs
        ]
        return pd.DataFrame({name: pd.Series(dtype=object) for name in columns})

    @classmethod
    def merge(cls, parts: Sequence[PredictionResult]) -> PredictionResult:
        r"""Join the results of one request that was sent in several calls.

        A request larger than the endpoint's per-call cap is split by the
        adapter, and each call answers about its own rows. Keeping only the
        last answer would silently drop every earlier batch, so the parts are
        concatenated here.

        Raises:
            StructuredDataError: If ``parts`` is empty, which would mean a
                request was sent with nothing to score, or if the parts do
                not describe one request.
        """
        if not parts:
            raise StructuredDataError(
                'no parts to merge', code='INTERNAL_ERROR'
            )
        if len(parts) == 1:
            return parts[0]
        first = parts[0]
        models = {part.model for part in parts}
        if len(models) > 1:
            raise StructuredDataError(
                f'cannot merge results from different models: {sorted(models)}',
                code='INTERNAL_ERROR',
            )
        items = [item for part in parts for item in part.items]
        metadata = dict(first.metadata)
        metadata['batches'] = len(parts)
        # Each call has its own id in the endpoint's logs, and only the first
        # would otherwise survive. That is the one you do not have when a
        # later batch is the one that misbehaved.
        metadata['batch_request_ids'] = [part.request_id for part in parts]
        explanation = next(
            (part.explanation for part in parts if part.explanation), None
        )
        # Every part answers one slice of the same request, so they were asked
        # for the same fields. If they were not, the merged result cannot
        # honestly claim either set, and an empty one would take its shape from
        # whichever part happened to be first.
        shapes = {part.outputs for part in parts}
        if len(shapes) > 1:
            raise StructuredDataError(
                f'these parts were asked for different outputs '
                f'({sorted(shapes)}), so they are not slices of one request',
                code='INTERNAL_ERROR',
            )
        # Named when any part left it out. Which rows ranked without it is
        # not recorded, so the union is the most that can be said.
        unscored: list[Any] = []
        for part in parts:
            unscored += [
                candidate
                for candidate in part.unscored_candidates
                if candidate not in unscored
            ]
        merged = cls(
            model=first.model,
            request_id=first.request_id,
            items=items,
            metadata=metadata,
            explanation=explanation,
            outputs=first.outputs,
            unscored_candidates=unscored,
        )
        merged._names_its_rows = all(part._names_its_rows for part in parts)
        merged._names_its_step = any(part._names_its_step for part in parts)
        return merged

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[PredictionItem]:
        return iter(self.items)

    def __repr__(self) -> str:
        explained = ', explained' if self._explanation is not None else ''
        left_out = len(self._unscored_candidates)
        unscored = (
            f', {left_out} candidate{"s" if left_out > 1 else ""} unscored'
            if left_out
            else ''
        )
        return (
            f'PredictionResult({self.model!r}, {len(self.items)} rows'
            f'{explained}{unscored})'
        )
