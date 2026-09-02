# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The Universal TFM wire format, in one place.

Every model posts the same envelope to ``POST /v1/predictions``; the server
tells them apart by the ``model`` field alone. These modules own the parts of
that envelope that are the same whatever the model is: how a column's dtype is
named, how a cell is written, and how a frame becomes an ``arrays`` table.

The boundary is DataFrames plus plain metadata. Nothing here knows what a
graph, a subgraph or a predictive query is, which is what lets the relational
path reach the same code: its adapter samples first and arrives with frames
like any other.
"""

from sdm_client.wire.dtypes import (
    JSON_SAFE_INT_MAX,
    JSON_SAFE_INT_MIN,
    UNTYPED,
    infer_tfm_dtype,
    json_ready,
    require_scalar_cells,
    serialize_cell,
    serialize_column,
    widen_tfm_dtype,
)
from sdm_client.wire.tables import encode_table

__all__ = [
    'JSON_SAFE_INT_MAX',
    'JSON_SAFE_INT_MIN',
    'UNTYPED',
    'encode_table',
    'infer_tfm_dtype',
    'json_ready',
    'require_scalar_cells',
    'serialize_cell',
    'serialize_column',
    'widen_tfm_dtype',
]
