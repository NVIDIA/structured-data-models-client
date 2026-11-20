# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Why a forecast is the forecast it is.

Asked for as an output field, like any other, and produced only where the
model runs in this process: the attribution reads the model's gradients and
its latent trajectory, and neither of those crosses a wire.

It is expensive. The latent trajectory walks a window per step of the
context, so an explanation costs tens of seconds on a CPU where a forecast
costs a fraction of one, and feature-axis attribution costs minutes. That is
why it is asked for rather than always computed.

Not public API itself. :func:`sdm_client.reporting.forecast_report` draws
what this produces.
"""

from sdm_client.explain._explain import explanation_of

__all__ = ['explanation_of']
