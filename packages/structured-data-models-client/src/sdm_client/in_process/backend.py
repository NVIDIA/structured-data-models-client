# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A client's requests, answered by structured-data-models in this process.

The client prepares a request here exactly as it does for a NIM, and this
backend runs the model on it directly, with no HTTP in between. The context is
fitted the way the hosted deployments fit it, so a request is answered alike
wherever it runs. A request the model cannot answer as asked is refused with
a :class:`~sdm_client.errors.StructuredDataError` whose code says why, from
the codes a NIM uses, and no HTTP status, since nothing was sent.
"""

from __future__ import annotations

import contextlib
import copy
import sys
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sdm_client.errors import MissingExtraError, StructuredDataError
from sdm_client.in_process.answers import QUANTILE_LEVELS, classified, regressed
from sdm_client.in_process.columns import Context
from sdm_client.in_process.relational import Relational
from sdm_client.in_process.tabular import Tabular

if TYPE_CHECKING:  # pragma: no cover - imported for typing only.
    from sdm_client.base import ModelAdapter
    from sdm_client.result import PredictionResult

#: How the hosted deployments fit a context.
_SEED = 42
_ESTIMATORS = 1

_SERVED = {served.model_id: served for served in (Tabular(), Relational())}


@dataclass
class _Session:
    served: Tabular | Relational
    task: Mapping[str, Any]
    schema: Mapping[str, Any]
    context: Context
    fitted: Any


class InProcessBackend:
    r"""Runs ``kumo-tabular`` and ``kumo-relational`` with
    structured-data-models.

    Not part of the public API surface; :meth:`SDMClient.in_process` builds
    one.

    Args:
        models: Loaded ``sdm.models.KumoTabular`` or
            ``sdm.models.KumoRelational`` models, at most one of each. With
            none, each is loaded from its published weights on first use.
        device: Where to load models this backend loads itself. Defaults to
            CUDA when available, else the CPU.

    One model runs at a time: a second thread asking for a prediction waits
    for the first to finish. Releasing a session, closing the client and
    asking whether it is ready do not wait, so a caller can still let go of
    what it holds while a prediction is running.
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
        self._state = threading.RLock()
        self._runs = threading.Lock()
        self._closed = False

    @property
    def url(self) -> str:
        return 'in-process'

    def close(self) -> None:
        r"""Release every session and the models this backend loaded."""
        with self._state:
            self._sessions.clear()
            self._models.clear()
            self._closed = True

    def health_ready(self) -> bool:
        r"""Whether this backend can answer. A closed one says no, rather
        than raising at a caller polling it as a boolean."""
        return not self._closed

    def list_models(self) -> list[str]:
        r"""The models this backend answers for."""
        with self._state:
            self._require_open()
            return [
                model_id
                for model_id in _SERVED
                if self._loads or model_id in self._models
            ]

    def predict(
        self,
        adapter: ModelAdapter,
        payload: dict[str, Any],
        outputs: tuple[str, ...],
    ) -> PredictionResult:
        with self._running():
            served, model = self._serving(payload['model'])
            task = payload['task']
            fields = _fields(served, model, task['kind'], payload)
            context = served.context(payload)
            query = served.query(context, payload, payload['schema'])
            answer = _run(_fit(model, context), query)
            items = _items(
                served,
                task,
                payload['schema'],
                context,
                payload,
                answer,
                fields,
            )
            return adapter.answered(
                items,
                outputs,
                payload,
                model=served.model_id,
                request_id=_request_id(payload),
            )

    def create_session(
        self, adapter: ModelAdapter, payload: dict[str, Any]
    ) -> str:
        r"""Fit the context once, to be scored against until released."""
        with self._running():
            served, model = self._serving(payload['model'])
            _fields(served, model, payload['task']['kind'], payload)
            context = served.context(payload)
            fitted = _fit(model, context)
            session_id = uuid.uuid4().hex
            with self._state:
                self._require_open()
                self._sessions[session_id] = _Session(
                    served=served,
                    task=copy.deepcopy(payload['task']),
                    schema=copy.deepcopy(payload['schema']),
                    context=context.fitted(),
                    fitted=fitted,
                )
            return session_id

    def session_predict(
        self,
        adapter: ModelAdapter,
        session_id: str,
        payload: dict[str, Any],
        outputs: tuple[str, ...],
        pinned: Mapping[str, Any],
    ) -> PredictionResult:
        with self._running():
            with self._state:
                session = self._sessions.get(session_id)
            if session is None:
                raise StructuredDataError(
                    f'no session {session_id!r} is open in this process',
                    code='SESSION_NOT_FOUND',
                )
            served, task = session.served, session.task
            fields = _fields(served, session.fitted, task['kind'], payload)
            served.check_inference(
                payload.get('inference') or {}, task, settled=True
            )
            query = served.query(session.context, payload, session.schema)
            answer = _run(session.fitted, query)
            items = _items(
                served,
                task,
                session.schema,
                session.context,
                payload,
                answer,
                fields,
            )
            return adapter.answered(
                items,
                outputs,
                {**pinned, **payload},
                model=served.model_id,
                request_id=_request_id(payload),
            )

    def delete_session(self, session_id: str) -> None:
        r"""Release a session. Releasing one that is gone is not an error."""
        with self._state:
            self._require_open()
            self._sessions.pop(session_id, None)

    def _serving(self, model_id: str) -> tuple[Tabular | Relational, Any]:
        r"""What reads requests for ``model_id``, and the model it runs.

        Runs are serialized, so a model is loaded once. The load is not held
        under the state lock, because it fetches published weights and
        releasing a session should not wait for that, and a client closed
        while it ran does not keep what it loaded.
        """
        served = _SERVED.get(model_id)
        with self._state:
            model = self._models.get(model_id)
        if model is None and self._loads and served is not None:
            model = _loaded(served, self._device)
            with self._state:
                self._require_open()
                self._models[model_id] = model
        if model is None or served is None:
            raise StructuredDataError(
                f'{model_id} is not served by this client, which was opened '
                f'with {self.list_models()} only',
                code='INVALID_CONFIGURATION',
            )
        if model.training:
            raise StructuredDataError(
                f'the {served.model_id} model is in training mode, and scores '
                f'only in evaluation mode; call .eval() on it',
                code='INVALID_CONFIGURATION',
            )
        return served, model

    @contextlib.contextmanager
    def _running(self) -> Iterator[None]:
        r"""Run one call at a time, reporting a failure the client did not
        foresee as one of its own errors, with the original as its cause.

        Holds the run lock alone, so releasing a session or closing the
        client does not wait for a model to finish.
        """
        with self._runs:
            with self._state:
                self._require_open()
            try:
                yield
            except StructuredDataError:
                raise
            except Exception as error:
                raise StructuredDataError(
                    f'the model failed: {type(error).__name__}: {error}',
                    code='INTERNAL_ERROR',
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
        f'client asks no other model',
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
        raise StructuredDataError(
            f'{served.model_id} could not be loaded: '
            f'{type(error).__name__}: {error}',
            code='MODEL_NOT_READY',
        ) from error


def _request_id(payload: Mapping[str, Any]) -> str:
    r"""The id the client put on this request, which a NIM echoes back."""
    return str((payload.get('metadata') or {}).get('request_id') or '')


def _check_task(served: Tabular | Relational, model: Any, kind: str) -> None:
    r"""Refuse a task this model does not answer, or was loaded without.

    A model can be loaded for classification or regression alone, which a
    NIM's never is, so that is asked of the model rather than of its kind.
    """
    if kind not in served.outputs:
        raise StructuredDataError(
            f'{served.model_id} does not answer {kind!r}; it answers '
            f'{sorted(served.outputs)}',
            code='UNSUPPORTED_TASK',
        )
    needed = 'regression' if kind == 'regression' else 'classification'
    loaded = sorted(str(task) for task in model.tasks)
    if needed not in loaded:
        raise StructuredDataError(
            f'this {served.model_id} model was loaded for {loaded} only, so '
            f'it cannot answer {kind!r}; load it for both, which is the '
            f'default',
            code='UNSUPPORTED_TASK',
        )


def _fields(
    served: Tabular | Relational,
    model: Any,
    kind: str,
    payload: Mapping[str, Any],
) -> list[str]:
    r"""The fields asked for, refused where this model does not produce
    them for this task."""
    _check_task(served, model, kind)
    output = payload.get('output') or {}
    fields = list(output.get('fields') or ['prediction'])
    unsupported = [name for name in fields if name not in served.outputs[kind]]
    if unsupported:
        raise StructuredDataError(
            f'{served.model_id} does not produce {unsupported} for {kind!r}; '
            f'it produces {list(served.outputs[kind])}',
            code='UNSUPPORTED_OUTPUT',
        )
    if kind == 'regression':
        _check_regression_output(payload, fields)
    return fields


def _statistic(payload: Mapping[str, Any]) -> str:
    return (payload.get('output') or {}).get('prediction_statistic') or 'mean'


def _levels(payload: Mapping[str, Any]) -> Sequence[float]:
    levels = (payload.get('output') or {}).get('quantile_levels')
    return levels or QUANTILE_LEVELS


def _check_regression_output(
    payload: Mapping[str, Any], fields: Sequence[str]
) -> None:
    r"""Refuse an output option the answer cannot be built from.

    Checked here rather than where the answer is built, because a fit and a
    run stand between the two and can take minutes on a GPU.
    """
    statistic = _statistic(payload)
    if statistic not in ('mean', 'median'):
        raise StructuredDataError(
            f'prediction_statistic is {statistic!r}; it is mean or median',
            code='UNSUPPORTED_OUTPUT',
        )
    if 'quantiles' in fields and not all(
        isinstance(level, (int, float)) and 0.001 <= level <= 0.999
        for level in _levels(payload)
    ):
        raise StructuredDataError(
            'quantile levels must be between 0.001 and 0.999',
            code='UNSUPPORTED_OUTPUT',
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


def _items(
    served: Tabular | Relational,
    task: Mapping[str, Any],
    schema: Mapping[str, Any],
    context: Context,
    payload: Mapping[str, Any],
    answer: Any,
    fields: list[str],
) -> list[Any]:
    r"""One answer per scored row, under the id the request gave that row."""
    ids = served.row_ids(
        payload['predict']['instance_table'], schema['instance_table']
    )
    if task['kind'] != 'regression':
        return classified(answer, context.encoded, context.labels, ids, fields)
    return regressed(answer, ids, fields, _statistic(payload), _levels(payload))
