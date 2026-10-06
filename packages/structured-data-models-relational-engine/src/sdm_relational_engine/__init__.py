# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from predictive_query.schema.typing import Dtype, Stype

from sdm_relational_engine._version import __version__
from sdm_relational_engine.client.client import NimClient
from sdm_relational_engine.client.initialization import (
    init,
    init_snowflake_serving as init_snowflake_serving,
    set_log_level,
)
from sdm_relational_engine.client.state import global_state
from sdm_relational_engine.exceptions import (
    AuthenticationError,
    ClientInitializationError,
    EmptyTableError,
    GraphConstructionError,
    HTTPException,
    InvalidResponseError,
    KumoRelationalError,
    MetricViewAsTableError,
    NimFailureError,
    NimTimeoutError,
    NimUnreachableError,
    NoEligibleEntityError,
    NoTablesFoundError,
    TableNotFoundError,
    UnknownDatasetError,
    ViewConversionError,
)
from sdm_relational_engine.utils.environment import (
    in_jupyter_notebook as in_jupyter_notebook,
    in_notebook as in_notebook,
    in_streamlit_notebook as in_streamlit_notebook,
    in_tmux as in_tmux,
    in_vnext_notebook as in_vnext_notebook,
)
from sdm_relational_engine.utils.logging import initialize_logging

initialize_logging()

__all__ = [
    'AuthenticationError',
    'ClientInitializationError',
    'Dtype',
    'EmptyTableError',
    'GraphConstructionError',
    'HTTPException',
    'InvalidResponseError',
    'KumoRelationalError',
    'MetricViewAsTableError',
    'NimClient',
    'NimFailureError',
    'NimTimeoutError',
    'NimUnreachableError',
    'NoEligibleEntityError',
    'NoTablesFoundError',
    'Stype',
    'TableNotFoundError',
    'UnknownDatasetError',
    'ViewConversionError',
    '__version__',
    'global_state',
    'init',
    'set_log_level',
]
