# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Sending a single-table request to ``kumo-tabular``.

This model does no heavy client-side work: there is no graph to build and
nothing to sample, so the adapter shapes the envelope, sends it, and reads the
answer back. That is why the tabular path ships in the base install and runs
anywhere, while the relational one needs a native extra.

Two rules this adapter holds to, both of which the endpoint enforces and would
otherwise report as an opaque 422.

The **target's declared dtype must equal the schema's dtype for that column**.
They are taken from one place here so they cannot drift.

The **task kind is never narrowed**. The endpoint accepts only the coarse
``classification`` and decides binary versus multiclass itself from the
context, so a client that narrowed it would both be guessing and be rejected.
"""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pandas as pd
from sdm_api.generated.tfm_api import (
    PredictionItem,
    PredictionResponse,
    ResponseContractError,
)

from sdm_client.adapters.single_table import coherent_stype, named_entities
from sdm_client.base import ModelSpec
from sdm_client.core.warnings import SDMWarning, caller_stacklevel
from sdm_client.errors import StructuredDataError, UnsupportedRequestError
from sdm_client.explanations import Explanations, GradientDecoder
from sdm_client.models import KUMO_RELATIONAL, KUMO_TABULAR
from sdm_client.request import Request
from sdm_client.result import PredictionResult
from sdm_client.wire import (
    UNTYPED,
    encode_table,
    infer_tfm_dtype,
    json_ready,
    widen_tfm_dtype,
)
from sdm_client.wire.answers import as_declared_target
from sdm_client.wire.inference import typed_inference

# Output fields to ask for when the caller named none, by task kind. Asking for
# everything would make a regression request carry probabilities it cannot
# have, which the endpoint rejects.
_DEFAULT_OUTPUTS = {
    'classification': ('prediction', 'probabilities'),
    'regression': ('prediction',),
}

# A session pins model, task, schema and context; only the rows being scored
# travel with each later call. ``metadata`` rides along on both halves because
# its request id is what correlates the call with the endpoint's logs.
_SESSION_PINNED = ('model', 'task', 'schema', 'context')
_SESSION_SCORING = ('predict', 'output', 'inference', 'metadata')


def _request_id() -> str:
    return f'sdm-{uuid.uuid4().hex[:12]}'


class TabularAdapter:
    r"""Shapes and reads ``kumo-tabular`` requests."""

    spec: ModelSpec = KUMO_TABULAR

    def default_output(self, request: Request) -> tuple[str, ...]:
        return _DEFAULT_OUTPUTS.get(request.task.kind, ('prediction',))

    def prediction_plan(
        self, request: Request, outputs: tuple[str, ...]
    ) -> Iterator[dict[str, Any]]:
        r"""One payload.

        The endpoint caps the context at 10,000 rows, which is enforced where
        the context is chosen rather than by splitting here: every call would
        have to carry the same context, so splitting would multiply the upload
        without raising the cap.
        """
        yield self._envelope(request, outputs)

    def session_plan(
        self, request: Request, outputs: tuple[str, ...]
    ) -> dict[str, Any]:
        envelope = self._envelope(request, outputs)
        pinned = {
            key: envelope[key] for key in _SESSION_PINNED if key in envelope
        }
        pinned['metadata'] = envelope['metadata']
        return pinned

    def session_prediction_plan(
        self,
        request: Request,
        outputs: tuple[str, ...],
        pinned: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        declared = _declared_by(pinned)
        try:
            envelope = self._envelope(
                request, outputs, declared, include_context=False
            )
        except Exception as error:
            # Writing the rows to the pinned schema is what makes the call
            # legible to the endpoint, and it is also the only thing that can
            # fail for a reason the caller cannot see: the failure names a
            # dtype they never wrote. So the session is named instead.
            raise _a_row_the_pinned_schema_cannot_hold(
                request, declared, error
            ) from error
        yield {
            key: envelope[key] for key in _SESSION_SCORING if key in envelope
        }

    def normalize(
        self,
        response: dict[str, Any],
        outputs: tuple[str, ...],
        payload: dict[str, Any] | None = None,
    ) -> PredictionResult:
        try:
            parsed = PredictionResponse.from_dict(response)
        except (ResponseContractError, TypeError, ValueError) as error:
            # A 200 whose body does not match the contract is still a failure,
            # and it has to arrive as one of this client's errors rather than
            # as a binding-layer exception nobody is catching.
            raise StructuredDataError(
                f'{self.spec.model_id} returned a response that does not '
                f'match the contract: {error}',
                code='INVALID_RESPONSE',
            ) from error
        return self.answered(
            parsed.predictions,
            outputs,
            payload,
            model=parsed.model,
            request_id=parsed.id,
            metadata=parsed.metadata,
        )

    def answered(
        self,
        items: Sequence[PredictionItem],
        outputs: tuple[str, ...],
        payload: dict[str, Any] | None,
        *,
        model: str,
        request_id: str = '',
        metadata: Mapping[str, Any] | None = None,
    ) -> PredictionResult:
        named = as_declared_target(named_entities(items, payload), payload)
        return PredictionResult(
            model=model,
            request_id=request_id,
            items=named,
            metadata=dict(metadata or {}),
            outputs=outputs,
            explanations=Explanations(
                # One table, so an entry has no related rows to name.
                GradientDecoder(labelled_by=payload).decode(named),
            ),
        )

    def explain_error(
        self, error: Exception, payload: dict[str, Any]
    ) -> Exception:
        r"""Nothing to add: this endpoint's refusals already name the column
        they are about."""
        return error

    def close(self) -> None:
        r"""Nothing to release: this adapter sends through the client's own
        transport and opens nothing of its own."""

    def _envelope(
        self,
        request: Request,
        outputs: tuple[str, ...],
        pinned_dtypes: Mapping[str, str] | None = None,
        *,
        include_context: bool = True,
    ) -> dict[str, Any]:
        if request.payloads is not None:
            raise UnsupportedRequestError(
                f'this request was gathered for '
                f'{KUMO_RELATIONAL.model_id} and carries the sampled '
                f'payloads that model reads. Gather it with '
                f'materialize(..., model={self.spec.model_id!r}) to ask '
                f'this one'
            )

        # The same question the relational path asks, at the same moment: a
        # column re-declared one assignment at a time can be left saying it
        # holds one thing and means another, and neither assignment is the
        # place to refuse it.
        request.schema.check_consistent()

        table = request.schema[request.entity_table]
        context = request.context.instance_table
        predict = request.predict.instance_table

        # A column's dtype has to describe both halves. Taking it from the
        # context alone calls an all-null context column 'string' and then
        # sends the predict half's integers under it as text, which changes
        # what the model is asked to predict on. `infer_tfm_dtype` reports
        # UNTYPED for a column that carries no type precisely so it cannot
        # re-type the other frame, so the schema's answer is set aside for
        # those and the frames decide.
        dtypes = {
            name: (UNTYPED if column.untyped else column.dtype)
            for name, column in table.columns.items()
        }
        for name in predict.columns:
            if name in dtypes:
                dtypes[name] = widen_tfm_dtype(
                    (dtypes[name], _raw_dtype_of(predict[name], dtypes[name]))
                )
        # Untyped on both sides means there is genuinely no information, and
        # 'string' round-trips whatever arrives without claiming a width.
        dtypes = {name: dtype or 'string' for name, dtype in dtypes.items()}
        # On a scoring call the schema is not sent: it was pinned when the
        # session opened, and the rows have to be written to describe *that*.
        # Typing them from the half that travels sent `1.0` under a column the
        # session had pinned as `int64`, which the endpoint refused as
        # `SESSION_PREDICT_SCHEMA_MISMATCH`, a failure about a value the
        # caller never wrote and cannot see.
        if pinned_dtypes is not None:
            dtypes = {
                **dtypes,
                **{name: d for name, d in pinned_dtypes.items() if d},
            }

        target = request.task.target.column_name
        missing = [name for name in context.columns if name not in dtypes]
        if missing:
            raise StructuredDataError(
                f'schema does not describe context column(s) {missing}',
                code='INVALID_REQUEST',
            )

        schema_columns = {}
        as_text = []
        for name in context.columns:
            declared = table[name].stype
            if declared == 'text':
                as_text.append(name)
            schema_columns[name] = {
                'dtype': dtypes[name],
                'stype': coherent_stype(
                    self._stype_for(request, declared, name),
                    dtypes[name],
                    table[name],
                ),
            }
        if as_text:
            warnings.warn(
                f'{self.spec.model_id} cannot read a text column, so '
                f'{", ".join(as_text)} travel(s) as categorical instead. ',
                category=SDMWarning,
                stacklevel=caller_stacklevel(),
            )
        instance_schema: dict[str, Any] = {'columns': schema_columns}
        key = table.primary_key
        if key:
            instance_schema['primary_key'] = (
                [key] if isinstance(key, str) else list(key)
            )

        envelope: dict[str, Any] = {
            'model': self.spec.model_id,
            'task': {
                'kind': request.task.kind,
                'target': {
                    'column_name': target,
                    # The one place this dtype is decided, so it cannot differ
                    # from the schema's answer for the same column.
                    'dtype': dtypes[target],
                },
            },
            'schema': {'instance_table': instance_schema},
            'predict': {'instance_table': encode_table(predict, dtypes)},
            'output': {'fields': list(outputs)},
            'metadata': {'request_id': _request_id()},
        }
        # Cached scoring keeps the schema checks above, but the context is
        # already on the server. Do not encode rows that will not be sent.
        if include_context:
            envelope['context'] = {
                'instance_table': encode_table(context, dtypes)
            }
        if request.inference:
            envelope['inference'] = typed_inference(
                json_ready(request.inference)
            )
        return envelope

    @staticmethod
    def _stype_for(request: Request, inferred: str, column: str) -> str:
        r"""The semantic type to declare for one column.

        Schema inference is task-agnostic, so an integer label column comes
        back ``numerical``, which is right for a regression target and wrong
        for a classification one. The task is what settles it, and only the
        adapter knows the task, so the correction happens here rather than
        being baked into the schema a caller may reuse for another question.

        The same argument carries a second correction: inference is also
        model-agnostic, and ``'text'`` is a type this model cannot read at
        all. Only the adapter knows which model is being asked.
        """
        if inferred == 'text':
            return 'categorical'
        if column != request.task.target.column_name:
            return inferred
        if request.task.is_classification:
            return 'categorical'
        return 'numerical' if inferred == 'categorical' else inferred


def _dtype_of(series: pd.Series) -> str:
    dtype = infer_tfm_dtype(series)
    return dtype if dtype != UNTYPED else 'string'


def _raw_dtype_of(series: pd.Series, declared: str | None = None) -> str:
    r"""Like :func:`_dtype_of`, but keeping UNTYPED as the answer.

    A column with no values contributes nothing to a widening. Turning it into
    ``'string'`` first is what made it decide the outcome.

    A `float64` that is float only because pandas has no integer that holds a
    null is reported as the integer it was declared to be. Widening on it sent
    an id through a float: `9007199254740993` left as `9007199254740992.0`,
    which is the corruption the integer rules exist to prevent, arriving by the
    one route that looks like an ordinary type difference. The values decide,
    not the dtype. A genuine float column of whole numbers is still a float,
    because nothing declared it otherwise.
    """
    if (
        declared
        and declared.startswith('int')
        and pd.api.types.is_float_dtype(series.dtype)
        and series.isna().any()
    ):
        present = series.dropna()
        if present.empty or bool((present == present.astype('int64')).all()):
            return declared
    return infer_tfm_dtype(series)


def _a_row_the_pinned_schema_cannot_hold(
    request: Request,
    declared: Mapping[str, str] | None,
    error: Exception,
) -> Exception:
    r"""Name the session, for a failure about a dtype the caller never wrote.

    The scoring rows are written to the schema the session pinned, so a value
    this package cannot write that way fails naming a type the caller did not
    choose, or where pandas raises first, as a bare `ValueError` from
    inside. Which column and which two types is what makes it actionable.
    """
    if declared is None:
        if isinstance(error, (StructuredDataError, UnsupportedRequestError)):
            return error
        return StructuredDataError(str(error), code='INVALID_REQUEST')
    try:
        predict = request.predict.instance_table
    except Exception:  # noqa: BLE001 - already reporting a failure
        return StructuredDataError(str(error), code='INVALID_REQUEST')
    differing = [
        f'{name} was pinned as {declared[name]} and these rows hold '
        f'{predict[name].dtype}'
        for name in predict.columns
        if name in declared and str(predict[name].dtype) != declared[name]
    ]
    if not differing:
        return error
    return StructuredDataError(
        f'these rows cannot be written to the schema this session pinned: '
        f'{"; ".join(differing)}. A scoring call does not send a schema. It '
        f'was settled by fit(), so the rows have to be writable as it '
        f'describes them. Give the column the type it was pinned with, or '
        f'fit() again on a context that has the type you want. ({error})',
        code='INVALID_REQUEST',
    )


def _declared_by(pinned: Mapping[str, Any] | None) -> dict[str, str] | None:
    r"""The dtypes a session pinned, read back off the envelope it sent."""
    if not pinned:
        return None
    columns = ((pinned.get('schema') or {}).get('instance_table') or {}).get(
        'columns'
    )
    if not isinstance(columns, Mapping):
        return None
    return {
        name: str(described.get('dtype'))
        for name, described in columns.items()
        if isinstance(described, Mapping) and described.get('dtype')
    }
