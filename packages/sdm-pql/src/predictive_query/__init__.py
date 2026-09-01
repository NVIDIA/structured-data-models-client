# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Predictive Query Language (PQL).

A predictive query states a prediction as text, the way SQL states a lookup::

    PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id

This package turns that text into a validated syntax tree, checked against the
schema it is written for. It runs no model and reads no data.

The three stages::

    text  --PQLParser-->  ParsedPredictiveQuery
                          --PredictiveQueryValidator-->  ValidatedPredictiveQuery

and, separately, :mod:`predictive_query.rewriting` turns one validated query
into another.
"""

from predictive_query.api import validate
from predictive_query.parser import Delegate, PQLParser
from predictive_query.profiles import (
    ANOMALY,
    RELATIONAL,
    TABULAR,
    TIMESERIES,
    FINE_TUNING,
    Capabilities,
    Mode,
    Profile,
)
from predictive_query.schema import codes
from predictive_query.query import (
    ParsedPredictiveQuery,
    ValidatedPredictiveQuery,
)
from predictive_query.schema.common import (
    Span,
    ValidationError,
    ValidationResponse,
    ValidationWarning,
)
from predictive_query.schema.graph import (
    ColumnKey,
    ColumnKeyGroup,
    Edge,
    GraphDefinition,
)
from predictive_query.schema.table import Column, TableDefinition, TimestampUnit
from predictive_query.schema.task import TaskType
from predictive_query.schema.typing import (
    AggregationType,
    Dtype,
    ProblemType,
    Stype,
    TimeUnit,
)
from predictive_query.validator import PredictiveQueryValidator

__version__ = '0.1.0'

#: The version of the *language*, as distinct from this package. MINOR goes up
#: when the language accepts something it did not before, so every query valid
#: under an earlier MINOR is still valid. MAJOR goes up only when a query stops
#: being valid or changes meaning. Neither moves for a new capability or
#: profile. See ``docs/stability.md``.
LANGUAGE_VERSION = '1.1'

__all__ = [
    'ANOMALY',
    'FINE_TUNING',
    'LANGUAGE_VERSION',
    'RELATIONAL',
    'TABULAR',
    'TIMESERIES',
    'AggregationType',
    'Capabilities',
    'Column',
    'ColumnKey',
    'ColumnKeyGroup',
    'Delegate',
    'Dtype',
    'Edge',
    'GraphDefinition',
    'Mode',
    'PQLParser',
    'ParsedPredictiveQuery',
    'PredictiveQueryValidator',
    'ProblemType',
    'Profile',
    'Span',
    'Stype',
    'TableDefinition',
    'TaskType',
    'TimeUnit',
    'TimestampUnit',
    'ValidatedPredictiveQuery',
    'ValidationError',
    'ValidationResponse',
    'ValidationWarning',
    '__version__',
    'codes',
    'validate',
]
