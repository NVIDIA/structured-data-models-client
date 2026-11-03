# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``kumo-forecast`` in process: a series continued for a number of steps.

It reads a context and fits nothing. The history, the series to continue and
the number of steps all travel on every call, and the model answers from them
in one pass, so there is no fitted context to hold and no session over it.

There is one row to score, carrying the series being asked about and nothing
in it, and `task.num_forecasts` says how many steps of it are wanted. That is
the contract's own shape for a forecast: one answer per scored row per
horizon, each naming its step from one. The model discards a future covariate
with a warning nobody is obliged to see, so `Request.forecast` gives no way
to supply one; a column the caller cannot pass is one they cannot believe
was used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from sdm_api.generated.tfm_api import PredictionItem

from sdm_client.errors import StructuredDataError
from sdm_client.in_process.columns import (
    Shape,
    check_columns,
    key_columns,
    stype_of,
)
from sdm_client.request import MAX_FORECAST_STEPS

#: Retrieval blends historical windows into the forecast. One is the direct
#: forecast alone, which is what the model does when it is not asked.
_ALPHA = 1.0
_NEIGHBOURS = 64
_TEMPERATURE = 0.05

_KNOBS = ('alpha', 'k', 'temperature', 'feature_axis')


class Forecast:
    r"""Reads a ``kumo-forecast`` request into the model."""

    model_id = 'kumo-forecast'
    outputs = {'forecasting': ('prediction', 'explanation')}

    @staticmethod
    def installed() -> bool:
        r"""Whether the structured-data-models here carries this model."""
        try:
            _model_type()
        except StructuredDataError:
            return False
        return True

    @staticmethod
    def load(device: Any) -> Any:
        return _model_type()(device=device)

    @staticmethod
    def accepts(model: Any) -> bool:
        r"""Whether ``model`` is this one.

        Asked of every model handed to the client, including ones this does
        not serve, so an installed structured-data-models that predates
        ``KumoForecast`` has to answer no rather than raise.
        """
        try:
            return isinstance(model, _model_type())
        except StructuredDataError:
            return False

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
        unknown = sorted(set(remaining) - set(_KNOBS))
        if unknown:
            raise StructuredDataError(
                f'unsupported inference options: {unknown}',
                code='INVALID_REQUEST',
            )
        alpha = remaining.get('alpha')
        if alpha is not None and not 0.0 <= float(alpha) <= 1.0:
            raise StructuredDataError(
                f'alpha is {alpha}; it is between zero and one, where one is '
                f'the direct forecast alone and zero is retrieval alone',
                code='INVALID_REQUEST',
            )
        neighbours = remaining.get('k')
        if neighbours is not None and (
            not isinstance(neighbours, int)
            or isinstance(neighbours, bool)
            or neighbours < 1
        ):
            raise StructuredDataError(
                f'k is {neighbours!r}; it is a whole number of at least 1',
                code='INVALID_REQUEST',
            )

    def series(
        self, payload: Mapping[str, Any], target: str
    ) -> tuple[Any, Any]:
        r"""The history, split into the series to continue and what else was
        measured alongside it.

        A key names the step rather than measuring anything, so it is not
        history. The series being forecast is the exception: one whose values
        happen to be unique is inferred as a key, and dropping it would drop
        the very column being asked about.

        Nothing measured alongside it is the ordinary case. An empty shape
        has no column to read a length from, so the history's own length is
        what says how long it is.
        """
        spec = payload['schema']['instance_table']
        table = payload['context']['instance_table']
        check_columns(table, spec, 'the history')
        if target not in table['columns']:
            raise StructuredDataError(
                f'the history carries no column {target!r}, which is the '
                f'series being forecast',
                code='INVALID_REQUEST',
            )
        keys = key_columns(spec) - {target}
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
                f'{self.model_id} forecasts numbers from numbers, and '
                f'{unread} is not numerical. Leave it out of the history',
                code='UNSUPPORTED_DTYPE',
            )
        measured = {
            name: stype for name, stype in stypes.items() if name != target
        }
        history = Shape.of(spec, {target: stypes[target]}).tensor(
            table, 'the history'
        )
        alongside = (
            Shape.of(spec, measured).tensor(table, 'the history')
            if measured
            else history[..., :0]
        )
        return history, alongside

    @staticmethod
    def row_ids(
        table: Mapping[str, Any], spec: Mapping[str, Any]
    ) -> list[str | None]:
        r"""A forecast answers about a step rather than a row of the request,
        so each answer names its step and carries no id."""
        return [None] * len(table['rows'])

    def answer(
        self,
        model: Any,
        payload: Mapping[str, Any],
        ids: Sequence[Any],
        fields: Sequence[str],
        generator: Any = None,
    ) -> list[PredictionItem]:
        r"""Continue the series for as many steps as were asked for.

        How many rows the query carries is the horizon, and that does not
        depend on how long the history is: slicing the history to length
        capped a long forecast at the number of steps already seen.

        One row is scored, and one answer per step of it, with
        ``forecast_step`` counting from one as the contract declares it.
        """
        import torch

        target = payload['task']['target']['column_name']
        history, alongside = self.series(payload, target)
        horizon = int(payload['task'].get('num_forecasts') or 0)
        if not 1 <= horizon <= MAX_FORECAST_STEPS:
            raise StructuredDataError(
                f'a forecast asks for between 1 and {MAX_FORECAST_STEPS} '
                f'steps as num_forecasts, and this names {horizon or "none"}',
                code='INVALID_REQUEST',
            )
        options = payload.get('inference') or {}
        alpha = float(options.get('alpha', _ALPHA))
        self._warn_where_retrieval_does_nothing(
            model,
            alpha,
            len(payload['context']['instance_table']['rows']),
            horizon,
        )
        device = history.device
        if generator is not None:
            device = generator.device
            history = history.to(device)
            alongside = alongside.to(device)
        steps = torch.empty(horizon, 0, device=device)
        out = model(
            alongside,
            history,
            steps,
            alpha=alpha,
            k=int(options.get('k', _NEIGHBOURS)),
            temperature=options.get('temperature', _TEMPERATURE),
        )
        forecast = out[target].numerical[..., 0]
        if not bool(forecast.isfinite().all()):
            raise StructuredDataError(
                'the model answered with a malformed output',
                code='INTERNAL_ERROR',
            )
        items = [
            PredictionItem(row_index=0, forecast_step=step, prediction=value)
            for step, value in enumerate(forecast.tolist(), start=1)
        ]
        if 'explanation' in fields:
            self._refuse_where_retrieval_answered(
                model,
                alpha,
                len(payload['context']['instance_table']['rows']),
                horizon,
            )
        if 'explanation' in fields and items:
            items[0] = replace(
                items[0],
                explanation=self._explained(
                    model, payload, target, horizon, forecast, alongside
                ),
            )
        return items

    def _explained(
        self,
        model: Any,
        payload: Mapping[str, Any],
        target: str,
        horizon: int,
        forecast: Any,
        alongside: Any,
    ) -> dict[str, Any]:
        r"""Why that forecast, attributed to the history it was made from.

        The attribution reads the model's gradients and walks its latent
        trajectory, so it reads the model itself rather than its answer, and
        exists only where the model runs here.
        """
        from sdm_client.explain import explanation_of

        table = payload['context']['instance_table']
        spec = payload['schema']['instance_table']
        keys = key_columns(spec) - {target}
        channels = [name for name in table['columns'] if name not in keys]
        history = Shape.of(spec, dict.fromkeys(channels, 'numerical')).tensor(
            table, 'the history'
        )
        options = payload.get('inference') or {}
        return explanation_of(
            model,
            history.numerical.detach().cpu().numpy().T,
            channels=channels,
            target=target,
            horizon=horizon,
            forecast=forecast.detach().cpu().numpy(),
            feature_axis=bool(options.get('feature_axis', False)),
        )

    @staticmethod
    def _refuse_where_retrieval_answered(
        model: Any, alpha: float, history: int, horizon: int
    ) -> None:
        r"""Refuse to explain a forecast retrieval had a hand in.

        The attribution describes the model's own forward pass. Where
        retrieval contributes, the forecast returned is a blend of that pass
        and windows pulled from history, and attributing the pass would
        describe a number nobody was given.
        """
        if alpha >= 1.0:
            return
        needed = model.model.context_size + max(
            model.model.prediction_size, horizon
        )
        if history < needed:
            return
        raise StructuredDataError(
            f'alpha={alpha} blends retrieved history into this forecast, and '
            f"an explanation describes the model's own forward pass only. "
            f'Explaining it would describe a forecast that was not returned. '
            f'Ask for the explanation with alpha=1, or ask for the forecast '
            f'without one',
            code='UNSUPPORTED_REQUEST',
        )

    @staticmethod
    def _warn_where_retrieval_does_nothing(
        model: Any, alpha: float, history: int, horizon: int
    ) -> None:
        r"""Retrieval needs a history longer than one window to retrieve from,
        and below that the model ignores ``alpha`` without saying so."""
        import warnings

        from sdm_client.core.warnings import SDMWarning, caller_stacklevel

        if alpha >= 1.0:
            return
        needed = model.model.context_size + max(
            model.model.prediction_size, horizon
        )
        if history >= needed:
            return
        warnings.warn(
            f'alpha={alpha} asks for retrieval, which needs at least '
            f'{needed} rows of history to retrieve from and this has '
            f'{history}. The forecast is the direct one, as though alpha '
            f'were 1.',
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )


def _model_type() -> Any:
    r"""``sdm.models.KumoForecast``, or why it could not be reached."""
    try:
        from sdm.models import KumoForecast
    except ImportError as error:
        raise StructuredDataError(
            'this structured-data-models carries no KumoForecast. The model '
            'is on its main and in no release, so until one ships it is '
            'reached by installing that package from source',
            code='MODEL_NOT_READY',
        ) from error
    return KumoForecast
