# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TypeAlias

from sdm_connectors.backends.spark import SparkConnection
from sdm_connectors.sql import MissingBackendError

#: Every connection the Databricks backend samples through: a SQL warehouse,
#: or a notebook's own Spark session. Checked with ``isinstance``, so the
#: session route needs no pyspark import to be recognised.
CONNECTION_TYPES: tuple[type, ...]
try:
    from sdm_connectors.backends.databricks import Connection
except MissingBackendError:
    # A notebook's own Spark session needs no SQL connector, so its absence
    # leaves that route open.
    CONNECTION_TYPES = (SparkConnection,)
else:
    CONNECTION_TYPES = (Connection, SparkConnection)

AnyConnection: TypeAlias = 'Connection | SparkConnection'

from .table import DatabricksTable  # noqa: E402
from .sampler import DatabricksSampler  # noqa: E402

__all__ = [
    'CONNECTION_TYPES',
    'AnyConnection',
    'DatabricksSampler',
    'DatabricksTable',
    'SparkConnection',
]
