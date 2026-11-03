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
    def prediction_size(self) -> int:
        r"""How many steps it forecasts in one pass."""
        return int(self._inner.prediction_size)

    def __call__(
        self, *, x_enc: torch.Tensor, input_mask: torch.Tensor | None = None
    ) -> _Forecast:
        r"""The forecast for ``x_enc`` of shape ``[B, C, R]``."""
        return _Forecast(forecast=self._inner(x_enc))

    def embed(
        self, *, x_enc: torch.Tensor, input_mask: torch.Tensor | None = None
    ) -> _Embedding:
        r"""One embedding per window of ``x_enc``, shaped ``[B, D]``."""
        return _Embedding(embeddings=self.embed_tensor(x_enc))

    def embed_tensor(self, x_enc: torch.Tensor) -> torch.Tensor:
        r"""The embedding as a plain tensor, differentiably.

        Kept apart from :meth:`embed` because integrated gradients needs the
        path from the input to it to carry gradients, and the protocol's own
        method is called where they are not wanted.
        """
        from sdm.models.kumo.timeseries.forecast.darr import embed

        encoded, _, _ = self._inner.encode(x_enc)
        return embed(encoded)
