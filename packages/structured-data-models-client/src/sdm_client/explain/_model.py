# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""structured-data-models' forecaster, as the attribution code expects one.

The ported attribution is written against a protocol of two methods rather
than against a class: a forward pass that returns a forecast, and an embedding
pass that returns one vector per window. structured-data-models' forecaster
has both, under its own names, so what stands between them is this.

The embedding is the model's own, the one it builds retrieval keys from. An
attribution is a statement about what the model did, so reducing its encoder
output some other way here would attribute something the model never computed.

The windows it is handed are padded, and which steps are padding arrives as a
mask. structured-data-models' forecaster takes no mask: it reads an
unobserved step as ``NaN``, and its normalisation ignores those when it takes
a window's statistics. So the mask is applied as ``NaN`` here. Dropping it
instead made padding into observed zeros, and a flat series then showed the
latent trajectory moving when it had not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class _Forecast:
    forecast: torch.Tensor


@dataclass(frozen=True)
class _Embedding:
    embeddings: torch.Tensor


class Forecaster:
    r"""Wraps ``sdm.models.KumoForecast`` for the attribution code.

    Args:
        model: The loaded ``KumoForecast``.
    """

    def __init__(self, model: Any) -> None:
        self._inner = model.model

    @property
    def context_size(self) -> int:
        r"""How many steps the model reads."""
        return int(self._inner.context_size)

    @property
    def mixes_channels(self) -> bool:
        r"""Whether a channel can reach the forecast of another one.

        Without it, a column measured alongside the series contributes
        nothing to the forecast, though the embedding still pools it.
        """
        return self._inner.cross_channel is not None

    @property
    def prediction_size(self) -> int:
        r"""How many steps it forecasts in one pass."""
        return int(self._inner.prediction_size)

    def __call__(
        self, *, x_enc: torch.Tensor, input_mask: torch.Tensor | None = None
    ) -> _Forecast:
        r"""The forecast for ``x_enc`` of shape ``[B, C, R]``.

        Deliberately not differentiable. The attribution reaches for autograd
        where the forecast carries gradients and falls back to finite
        differences where it does not, and autograd is the wrong answer for
        this forecaster: it normalises each window by statistics it takes
        without gradients, so a gradient through it measures sensitivity with
        that normalisation held still. It came out some hundreds of times
        larger than the difference the same perturbation actually makes.

        The upstream package patches its own normaliser to carry those
        gradients, by a method name this one does not have, so the patch
        finds nothing here and passes in silence.
        """
        with torch.no_grad():
            return _Forecast(forecast=self._inner(_observed(x_enc, input_mask)))

    def embed(
        self, *, x_enc: torch.Tensor, input_mask: torch.Tensor | None = None
    ) -> _Embedding:
        r"""One embedding per window of ``x_enc``, shaped ``[B, D]``."""
        return _Embedding(embeddings=self.embed_tensor(x_enc, input_mask))

    def embed_tensor(
        self, x_enc: torch.Tensor, input_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""The embedding as a plain tensor, differentiably.

        Kept apart from :meth:`embed` because integrated gradients needs the
        path from the input to it to carry gradients, and the protocol's own
        method is called where they are not wanted.
        """
        from sdm.models.kumo.timeseries.forecast.darr import embed

        encoded, _, _ = self._inner.encode(_observed(x_enc, input_mask))
        return embed(encoded)


def _observed(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    r"""``x`` with the steps the mask calls padding marked unobserved.

    The forecaster has no mask argument and reads ``NaN`` as unobserved: its
    normalisation takes a window's statistics over what is there, and zeroes
    what is not once they are taken.
    """
    if mask is None:
        return x
    keep = torch.as_tensor(mask, device=x.device).bool()
    if keep.all():
        return x
    # The mask's last axis is steps and anything before it is the window it
    # belongs to; the readings have a channel axis between the two. It is one
    # mask for every channel of a window, so the channel axis is where the
    # singleton goes.
    keep = keep.unsqueeze(-2)
    while keep.ndim < x.ndim:
        keep = keep.unsqueeze(0)
    return x.masked_fill(~keep.expand_as(x), float('nan'))
