# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Drawing an answer, for the models whose answer is worth looking at.

Needs ``pip install structured-data-models-client[report]``, which brings
matplotlib. Nothing else in this client draws anything, so a caller who does
not want a plotting stack does not get one.
"""

from sdm_client.reporting.anomaly import anomaly_report
from sdm_client.reporting.forecast import forecast_report

__all__ = ['anomaly_report', 'forecast_report']
