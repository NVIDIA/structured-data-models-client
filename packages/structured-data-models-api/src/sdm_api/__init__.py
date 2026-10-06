# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Universal TFM API as this client speaks it.

Wire types only. Nothing here parses a query, samples a graph or opens a
socket: those belong to ``predictive_query``, ``sdm_relational_engine`` and
``sdm_client`` respectively.

Everything under :mod:`sdm_api.generated` is produced from the canonical
OpenAPI contract by ``scripts/generate_sdm_api.py`` and must not be edited by
hand. ``generated/PROVENANCE.json`` records the contract revision and both
digests, and a test asserts they still agree.
"""

from sdm_api.endpoints import Endpoint, HTTPMethod
from sdm_api.generated.tfm_api import (
    TFM_ENDPOINTS_BY_OPERATION_ID,
    TFM_MODEL_KUMO_RELATIONAL,
    TFM_MODEL_KUMO_TABULAR,
    TFM_MODEL_VALUES,
    TFM_OUTPUT_FIELD_VALUES,
    TFM_TASK_KIND_VALUES,
    TFMOperation,
    TFMOperations,
)

__all__ = [
    'TFM_ENDPOINTS_BY_OPERATION_ID',
    'TFM_MODEL_KUMO_RELATIONAL',
    'TFM_MODEL_KUMO_TABULAR',
    'TFM_MODEL_VALUES',
    'TFM_OUTPUT_FIELD_VALUES',
    'TFM_TASK_KIND_VALUES',
    'Endpoint',
    'HTTPMethod',
    'TFMOperation',
    'TFMOperations',
]
