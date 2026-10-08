# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared staged adapters; the original acceptance predicates are unchanged."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'p97-execution-coupling'))
from coupling import (  # noqa: E402
    Rejected,
    join_observation,
    observe_public_call,
    require,
    require_path,
)

__all__ = [
    'Rejected',
    'join_observation',
    'observe_public_call',
    'require',
    'require_path',
]


def load(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    require(
        specification is not None and specification.loader is not None,
        'adapter.source',
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def helpers(repo: Path) -> tuple[Any, Any, Any]:
    for relative in (
        'tests/parity',
        'scripts/parity',
        'tests/rust_core/sampling/driver',
    ):
        sys.path.insert(0, str(repo / relative))
    switch = load('_p97_switch', repo / 'tests/rust_core/switch/_switch.py')
    driver = load(
        '_driver', repo / 'tests/rust_core/sampling/driver/_driver.py'
    )
    request = load('_p97_request', repo / 'tests/rust_core/request/_request.py')
    return switch, driver, request


def outcome(
    result: Any = None,
    *,
    sql: list | None = None,
    error: dict | None = None,
    warnings: list | None = None,
    schema: list | None = None,
    payloads: list | None = None,
) -> dict:
    return {
        'status': 'refused' if error else 'success',
        'schema': schema or [],
        'payloads': payloads or [],
        'result': result,
        'sql': sql or [],
        'error': error,
        'warnings': warnings or [],
    }


def diagnostic(error: BaseException, encode: Any) -> dict:
    return {
        'class': f'{type(error).__module__}.{type(error).__qualname__}',
        'code': encode(getattr(error, 'code', None)),
        'message': str(getattr(error, 'message', str(error))),
        'details': encode(getattr(error, 'details', None)),
    }


def strict_json(raw: bytes | str) -> Any:
    def unique(items: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in items:
            require(key not in result, 'adapter.duplicate_key')
            result[key] = value
        return result

    def constant(_value: str) -> None:
        raise Rejected('adapter.nonfinite_json')

    return json.loads(raw, object_pairs_hook=unique, parse_constant=constant)
