# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

from sdm_relational_engine.rfm.backend.databricks import Connection
from sdm_relational_engine.rfm.backend.databricks import connect as _connect


def connect(
    server_hostname: str | None = None,
    http_path: str | None = None,
    access_token: str | None = None,
    catalog: str | None = None,
    schema: str | None = None,
    **kwargs: Any,
) -> Connection:
    r"""Opens a connection to a Databricks SQL warehouse for testing, reading
    credentials from the ``DATABRICKS_*`` environment variables.
    """
    return _connect(
        server_hostname=(
            server_hostname or os.getenv('DATABRICKS_SERVER_HOSTNAME')
        ),
        http_path=http_path or os.getenv('DATABRICKS_HTTP_PATH'),
        access_token=access_token or os.getenv('DATABRICKS_TOKEN'),
        catalog=catalog or os.getenv('DATABRICKS_CATALOG'),
        schema=schema or os.getenv('DATABRICKS_SCHEMA'),
        **kwargs,
    )
