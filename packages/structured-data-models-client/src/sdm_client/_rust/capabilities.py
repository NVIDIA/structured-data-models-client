# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Each model's capabilities on the core route: the core's table
(``sdm_host::capabilities``), the one table both hosts describe a model
from, which the Java host's ``SdmClient.capabilities`` reads too (T3.12)."""

from __future__ import annotations

import json

from sdm_client._rust.run import native
from sdm_client.base import ModelCapabilities
from sdm_client.errors import StructuredDataError


def described(model: str) -> ModelCapabilities:
    r"""What the core's table says ``model`` supports, as
    :class:`~sdm_client.base.ModelCapabilities`.

    Raises:
        StructuredDataError: The core's table names no such model, which
            the client's registry named (``INTERNAL_ERROR``).
    """
    for each in json.loads(native().capabilities())['models']:
        if each['model'] == model:
            return ModelCapabilities(
                model=each['model'],
                tasks=tuple(each['tasks']),
                outputs=tuple(each['outputs']),
                single_table=each['single_table'],
                multiple_tables=each['multiple_tables'],
                relationships=each['relationships'],
                explanation_formats=tuple(each['explanation_formats']),
                sessions=each['sessions'],
                source=each['source'],
            )
    raise StructuredDataError(
        f"the core's table describes no model {model!r}",
        code='INTERNAL_ERROR',
    )
