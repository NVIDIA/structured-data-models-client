# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The existing checked reader supplies consumer roles, never fake starts."""

import functools
import importlib.util
from pathlib import Path
from typing import Any


@functools.cache
def _reader():
    path = Path(__file__).parents[1] / 'runners/prediction_observation.py'
    spec = importlib.util.spec_from_file_location(
        'contract_actual_core_reader', path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_view(observation: dict[str, Any]) -> dict[str, Any]:
    return _reader().view(
        observation['core'], observation['identity']['operations']
    )


def checked_operations(observation: dict[str, Any]) -> list[str]:
    return checked_view(observation)['operations']
