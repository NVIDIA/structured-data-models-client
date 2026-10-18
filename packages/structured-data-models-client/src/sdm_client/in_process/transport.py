# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A client's requests, answered by structured-data-models in this process.

The client sends what it would send a NIM and reads back what a NIM answers,
so everything that works against a NIM works here, sessions included. The
model is fitted the way the hosted deployments fit it, so the same request is
answered the same way wherever it runs.
"""

from __future__ import annotations

import contextlib
import copy
import json
import sys
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sdm_api import TFM_MODEL_VALUES

from sdm_client.core.transport import _check_encodable
from sdm_client.errors import MissingExtraError, StructuredDataError
from sdm_client.in_process.answers import (
    QUANTILE_LEVELS,
    classified,
    regressed,
    row_ids,
)
from sdm_client.in_process.columns import Context
from sdm_client.in_process.refusal import in_session, refusal
from sdm_client.in_process.relational import Relational
from sdm_client.in_process.tabular import Tabular

#: How the hosted deployments fit a context.
_SEED = 42
_ESTIMATORS = 1

_SERVED = {served.model_id: served for served in (Tabular(), Relational())}
_INFERENCE_OPTIONS = frozenset(
    {'run_mode', 'use_prediction_time', 'inference_config'}
)


@dataclass
class _Session:
    served: Tabular | Relational
    task: Mapping[str, Any]
    schema: Mapping[str, Any]
    context: Context
    fitted: Any


class InProcessTransport:
    r"""Answers the contract's routes with structured-data-models models.

    Not part of the public API surface; :meth:`SDMClient.in_process` builds
    one.

    Args:
        models: Loaded ``sdm.models.KumoTabular`` or
            ``sdm.models.KumoRelational`` models, at most one of each. With
            none, each is loaded from its published weights on first use.
        device: Where to load models this transport loads itself. Defaults to
            CUDA when available, else the CPU.
    """

    serves_sessions = True

    def __init__(self, models: Sequence[Any] = (), device: Any = None) -> None:
        torch = _require_sdm()
        if models and device is not None:
            raise StructuredDataError(
                'device places the models this client loads itself; the '
                'models passed in stay where they are, so pass one or the '
                'other',
                code='INVALID_CONFIGURATION',
            )
        self._models: dict[str, Any] = {}
        for model in models:
            model_id = _served_by(model).model_id
            if model_id in self._models:
                raise StructuredDataError(
                    f'two {model_id} models were passed; pass one of each',
                    code='INVALID_CONFIGURATION',
                )
            self._models[model_id] = model
        self._loads = not models
        self._device = _device(device, torch) if self._loads else None
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()
        self._closed = False

    @property
    def url(self) -> str:
        return 'in-process'

    def close(self) -> None:
        r"""Release every session and the models this transport loaded."""
        with self._lock:
            self._sessions.clear()
            self._models.clear()
            self._closed = True

    def health_ready(self) -> bool:
        self._require_open()
        return True

    def list_models(self) -> list[str]:
        r"""The models this transport answers for."""
        self._require_open()
        return [
            model_id
            for model_id in _SERVED
            if self._loads or model_id in self._models
        ]

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = _received(payload)
        with self._answering():
            served, model = self._serving(payload.get('model'))
            task = payload['task']
            fields = _fields(served, task['kind'], payload.get('output'))
            _check_trained_for(served, model, task['kind'])
            _check_inference(payload.get('inference'), task)
            context = served.context(payload)
            output = _run(_fit(model, context), served.query(context, payload))
            return _response(
                served,
                task,
                payload['schema'],
                context,
                payload,
                output,
                fields,
            )

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        r"""Fit the context once, to be scored against until released."""
        payload = _received(payload)
        with self._answering():
            served, model = self._serving(payload.get('model'))
            _outputs(served, payload['task']['kind'])
            _check_trained_for(served, model, payload['task']['kind'])
            with in_session('SESSION_CREATE_VALIDATION_FAILED'):
                context = served.context(payload)
            session_id = f'sess_{uuid.uuid4().hex}'
            fitted = _fit(model, context)
            self._sessions[session_id] = _Session(
                served=served,
                task=payload['task'],
                schema=payload['schema'],
                context=context.fitted(),
                fitted=fitted,
            )
            return {'session_id': session_id}

    def session_predict(
        self, session_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        payload = _received(payload)
        with self._answering():
            session = self._sessions.get(session_id)
            if session is None:
                raise refusal(
                    'SESSION_NOT_FOUND',
                    f'session {session_id!r} does not exist or was released',
                )
            task = session.task
            fields = _fields(
                session.served, task['kind'], payload.get('output')
            )
            with in_session('SESSION_PREDICT_SCHEMA_MISMATCH'):
                _check_inference(payload.get('inference'), task, settled=True)
                query = session.served.query(session.context, payload)
            output = _run(session.fitted, query)
            return _response(
                session.served,
                task,
                session.schema,
                session.context,
                payload,
                output,
                fields,
            )

    def delete_session(self, session_id: str) -> None:
        r"""Release a session. Releasing one that is gone is not an error."""
        with self._lock:
            self._require_open()
            self._sessions.pop(session_id, None)

    def _serving(self, model_id: Any) -> tuple[Tabular | Relational, Any]:
        r"""What reads requests for ``model_id``, and the model it runs.

        Called holding the lock, so a model is loaded once.
        """
        if model_id not in TFM_MODEL_VALUES:
            named = ' or '.join(repr(value) for value in TFM_MODEL_VALUES)
            raise refusal(
                'VALIDATION_FAILED',
                f'Request validation failed. model: Input should be {named}',
            )
        served = _SERVED[model_id]
        model = self._models.get(model_id)
        if model is None and self._loads:
            model = self._models[model_id] = _loaded(served, self._device)
        if model is None:
            raise refusal(
                'INVALID_SCHEMA',
                f'model {model_id!r} is not served here; available: '
                f'{self.list_models()}',
            )
        if model.training:
            raise StructuredDataError(
                f'the {served.model_id} model is in training mode, and scores '
                f'only in evaluation mode; call .eval() on it',
                code='INVALID_CONFIGURATION',
            )
        return served, model

    @contextlib.contextmanager
    def _answering(self) -> Iterator[None]:
        r"""Answer one call at a time, reporting a failure the way a NIM
        does: an internal error, here with the original as its cause."""
        with self._lock:
            self._require_open()
            try:
                yield
            except StructuredDataError:
                raise
            except Exception as error:
                raise refusal(
                    'INTERNAL_ERROR', f'{type(error).__name__}: {error}'
                ) from error

    def _require_open(self) -> None:
        if self._closed:
            raise StructuredDataError(
                'this client is closed; construct a new SDMClient',
                code='INVALID_CONFIGURATION',
            )


def _require_sdm() -> Any:
    try:
        import sdm  # noqa: F401
        import torch
    except ImportError as error:
        if sys.version_info < (3, 11):
            raise StructuredDataError(
                f'running a model in process needs structured-data-models, '
                f'which needs Python 3.11 or later; this is Python '
                f'{sys.version_info.major}.{sys.version_info.minor}',
                code='MISSING_EXTRA',
                details={
                    'extra': 'in-process',
                    'package': 'structured-data-models',
                },
            ) from error
        raise MissingExtraError(
            'in-process', 'structured-data-models'
        ) from error
    return torch


def _served_by(model: Any) -> Tabular | Relational:
    for served in _SERVED.values():
        if served.accepts(model):
            return served
    raise StructuredDataError(
        f'in_process() takes the KumoTabular and KumoRelational models of '
        f'structured-data-models, and got a {type(model).__name__}; the '
        f'contract names no other model',
        code='INVALID_CONFIGURATION',
    )


def _device(device: Any, torch: Any) -> Any:
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        return torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise StructuredDataError(
            f'device {device!r} is not a device torch recognises',
            code='INVALID_CONFIGURATION',
        ) from error


def _loaded(served: Tabular | Relational, device: Any) -> Any:
    try:
        return served.load(device)
    except Exception as error:
        raise refusal(
            'MODEL_NOT_READY',
            f'{served.model_id} could not be loaded: '
            f'{type(error).__name__}: {error}',
        ) from error


def _received(payload: dict[str, Any]) -> dict[str, Any]:
    r"""The payload as a NIM would read it: through JSON.

    Encoding it refuses what an HTTP call refuses, and decoding it hands the
    model a copy, so nothing done here reaches the request it came from.
    """
    _check_encodable(payload)
    try:
        return json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise StructuredDataError(
            f'this request carries a value that cannot be sent as JSON: '
            f'{error}',
            code='INVALID_REQUEST',
        ) from error


def _outputs(served: Tabular | Relational, kind: str) -> tuple[str, ...]:
    r"""What this model produces for this task, which it must serve."""
    available = served.outputs.get(kind)
    if available is None:
        raise refusal(
            'UNSUPPORTED_TASK',
            f'task {kind!r} not supported by model {served.model_id!r}; '
            f'supported: {list(served.outputs)}',
        )
    return available


def _check_trained_for(
    served: Tabular | Relational, model: Any, kind: str
) -> None:
    r"""Refuse a task the model was loaded without.

    A model can be loaded for classification or regression alone, which a
    NIM's never is, so this is asked of the model rather than of its kind.
    """
    needed = 'regression' if kind == 'regression' else 'classification'
    loaded = sorted(str(task) for task in model.tasks)
    if needed not in loaded:
        raise refusal(
            'UNSUPPORTED_TASK',
            f'this {served.model_id} model was loaded for {loaded} only, so '
            f'it cannot answer {kind!r}; load it for both, which is the '
            f'default',
        )


def _fields(
    served: Tabular | Relational,
    kind: str,
    output: Mapping[str, Any] | None,
) -> list[str]:
    r"""The fields asked for, refused where this model does not produce
    them for this task."""
    available = _outputs(served, kind)
    fields = list((output or {}).get('fields') or ['prediction'])
    unsupported = [field for field in fields if field not in available]
    if unsupported:
        raise refusal(
            'UNSUPPORTED_OUTPUT',
            f'output(s) {unsupported} not supported by model '
            f'{served.model_id!r} for task {kind!r}; supported: '
            f'{list(available)}',
        )
    return fields


def _check_inference(
    inference: Mapping[str, Any] | None,
    task: Mapping[str, Any],
    settled: bool = False,
) -> None:
    r"""Refuse an inference option the hosted deployments refuse.

    A session's context is fitted without prediction time, since creating one
    carries no inference options, so a prediction cannot ask for it after.
    """
    options = dict(inference or {})
    unknown = sorted(set(options) - _INFERENCE_OPTIONS)
    if unknown:
        raise refusal(
            'INVALID_REQUEST', f'unsupported inference options: {unknown}'
        )
    if options.get('inference_config'):
        raise refusal(
            'INVALID_REQUEST',
            'inference_config is reserved for API compatibility and must be '
            'omitted or empty',
        )
    if not options.get('use_prediction_time'):
        return
    if not task.get('anchor_time_column'):
        raise refusal(
            'TASK_SCHEMA_MISMATCH',
            'use_prediction_time needs task.anchor_time_column, which this '
            'request does not name',
        )
    if settled:
        raise refusal(
            'INVALID_REQUEST',
            'use_prediction_time is settled when a session is created and '
            'cannot be changed by a prediction',
        )


def _fit(model: Any, context: Context) -> Any:
    r"""A copy of ``model`` fitted on ``context``.

    The copy shares the model's weights and holds its own fitted context, so
    a session never disturbs another, nor anything fitted on the model itself.
    """
    import torch

    device = _placement(model)
    fitted = copy.copy(model)
    related = None if context.related is None else context.related.to(device)
    with _autocast(device):
        fitted.fit(
            context.x.to(device),
            context.y.to(device),
            related,
            num_estimators=_ESTIMATORS,
            generator=torch.Generator(device=device).manual_seed(_SEED),
        )
    return fitted


def _run(fitted: Any, query: tuple[Any, Any]) -> Any:
    x, related = query
    device = _placement(fitted)
    with _autocast(device):
        return fitted.predict(
            x.to(device), None if related is None else related.to(device)
        )


def _placement(model: Any) -> Any:
    return next(model.parameters()).device


def _autocast(device: Any) -> Any:
    r"""The precision the hosted deployments run at on a GPU."""
    import torch

    if device.type != 'cuda':
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type='cuda', dtype=dtype)


def _response(
    served: Tabular | Relational,
    task: Mapping[str, Any],
    schema: Mapping[str, Any],
    context: Context,
    payload: Mapping[str, Any],
    output: Any,
    fields: list[str],
) -> dict[str, Any]:
    instance = schema['instance_table']
    dtypes = {
        name: str(column.get('dtype'))
        for name, column in instance['columns'].items()
    }
    ids = row_ids(
        payload['predict']['instance_table'],
        instance.get('primary_key'),
        dtypes,
    )
    kind = task['kind']
    if kind == 'regression':
        spec = payload.get('output') or {}
        statistic = spec.get('prediction_statistic') or 'mean'
        if statistic not in ('mean', 'median'):
            raise refusal(
                'UNSUPPORTED_OUTPUT',
                'prediction_statistic must be mean or median',
            )
        levels = spec.get('quantile_levels') or QUANTILE_LEVELS
        answers = regressed(output, ids, fields, statistic, levels)
        output_type = statistic
    else:
        answers = classified(
            output, context.encoded, context.labels, ids, fields
        )
        output_type = (
            'probabilities' if 'probabilities' in fields else 'prediction'
        )
    return {
        'id': f'pred_{uuid.uuid4().hex[:12]}',
        'model': served.model_id,
        'predictions': answers,
        'metadata': {'task_kind': kind, 'output_type': output_type},
    }
