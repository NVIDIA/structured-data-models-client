# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TypeAlias

from sdm_connectors.backends.databricks import (
    Connection,
    connect,
)
from sdm_connectors.backends.spark import SparkConnection

#: Every connection the Databricks backend samples through: a SQL warehouse,
#: or a notebook's own Spark session. Checked with ``isinstance``, so the
#: session route needs no pyspark import to be recognised.
CONNECTION_TYPES = (Connection, SparkConnection)
AnyConnection: TypeAlias = Connection | SparkConnection

from .table import DatabricksTable  # noqa: E402
from .sampler import DatabricksSampler  # noqa: E402

__all__ = [
    'CONNECTION_TYPES',
    'AnyConnection',
    'Connection',
    'DatabricksSampler',
    'DatabricksTable',
    'SparkConnection',
    'connect',
]
