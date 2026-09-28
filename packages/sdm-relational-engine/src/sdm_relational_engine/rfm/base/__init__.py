# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from predictive_query.schema.common import StrEnum


class DataBackend(StrEnum):
    LOCAL = 'local'
    SQLITE = 'sqlite'
    DUCKDB = 'duckdb'
    SNOWFLAKE = 'snowflake'
    DATABRICKS = 'databricks'
    POSTGRES = 'postgres'


from .source import SourceColumn, SourceForeignKey  # noqa: E402
from .expression import Expression, LocalExpression  # noqa: E402
from .column import ColumnSpec, ColumnSpecType, Column  # noqa: E402
from .table import Table  # noqa: E402
from .sampler import SamplerOutput, Sampler  # noqa: E402
from .sql_sampler import SQLSampler  # noqa: E402

__all__ = [
    'Column',
    'ColumnSpec',
    'ColumnSpecType',
    'DataBackend',
    'Expression',
    'LocalExpression',
    'SQLSampler',
    'Sampler',
    'SamplerOutput',
    'SourceColumn',
    'SourceForeignKey',
    'Table',
]
