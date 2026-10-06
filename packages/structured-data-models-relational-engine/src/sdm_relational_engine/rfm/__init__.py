# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sdm_relational_engine.client.engine_access import (
    _CLIENT_TOKEN as _CLIENT_TOKEN,
)
from sdm_relational_engine.client.engine_access import (
    close_client,
    global_state as global_state,
    init,
    init_client,
    init_snowflake_serving,
)

from .base import Table
from .diagnostics import (
    GraphSanitizationReport,
    SanitizationStatus,
    TableSanitizationReport,
    TaskReferenceError,
)
from .explain.result import ExplainConfig, Explanation
from .graph import Edge as Edge
from .base.table import TimeColumnWarning
from .graph import Graph, ViewConversionWarning
from .inference.task import TaskTable
from .rfm import KumoRelational, MaterializedPredictionRequest


LocalGraph = Graph  # NOTE Backward compatibility - do not use anymore.


__all__ = [
    'ExplainConfig',
    'Explanation',
    'Graph',
    'GraphSanitizationReport',
    'KumoRelational',
    'LocalTable',
    'MaterializedPredictionRequest',
    'SanitizationStatus',
    'Table',
    'TableSanitizationReport',
    'TaskReferenceError',
    'TaskTable',
    'TimeColumnWarning',
    'ViewConversionWarning',
    'close_client',
    'init',
    'init_client',
    'init_snowflake_serving',
]


def __getattr__(name: str) -> object:
    r"""Resolve ``LocalTable`` only when something asks for it.

    Importing it here would pull in ``backend.local``, which requires the
    compiled neighbour sampler at import time. That made every path through
    this package depend on the extension, including the Snowflake serving
    path, which runs inference on the service and never samples locally.
    Deferring the import keeps the requirement where the requirement actually
    is.
    """
    if name == 'LocalTable':
        from .backend.local import LocalTable

        return LocalTable
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
