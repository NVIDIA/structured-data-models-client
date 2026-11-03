# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-model adapters. One module per model."""

from sdm_client.adapters.anomaly import AnomalyAdapter
from sdm_client.adapters.forecast import ForecastAdapter
from sdm_client.adapters.tabular import TabularAdapter

__all__ = ['AnomalyAdapter', 'ForecastAdapter', 'TabularAdapter']
