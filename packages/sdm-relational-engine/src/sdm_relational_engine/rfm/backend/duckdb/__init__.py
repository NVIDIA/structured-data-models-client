# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sdm_connectors.backends.duckdb import Connection, connect


from .table import DuckDBTable
from .sampler import DuckDBSampler

__all__ = [
    'Connection',
    'DuckDBSampler',
    'DuckDBTable',
    'connect',
]
