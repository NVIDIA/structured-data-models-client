# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write boundaries and environment failures for live contract runners."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

GUARD = Path(__file__).resolve().with_name('write_guard.py')


def load_guard(path: Path | str | None = GUARD) -> ModuleType:
    r"""The live write guard's module (``write_guard.py``, its
    ``WriteGuard``, ``install`` and ``from_environment``); a run without it refuses to
    start."""
    if not path or not Path(path).is_file():
        raise SystemExit(
            f'no live write guard at {path}; every live write goes through it'
        )
    spec = importlib.util.spec_from_file_location('sdm_live_write_guard', path)
    if spec is None or spec.loader is None:
        raise SystemExit(f'the live write guard {path} cannot be read')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ('WriteGuard', 'install', 'from_environment'):
        if not hasattr(module, name):
            raise SystemExit(f'the live write guard {path} has no {name}')
    return module


DIALECT = {
    'databricks-sql': 'databricks',
    'databricks-spark': 'databricks',
    'snowflake': 'snowflake',
    'lakebase': 'postgres',
}


def install_guard(leg: str, guard: ModuleType, loader: Any) -> Any:
    r"""Every live statement of the run through the one write guard:
    every driver's entry points (``guard.install``, each dialect's test
    schema from ``SDM_LIVE_TEST_*``, a dialect with none refusing every
    statement) and the loader's own statements (``databases.WRITE_GUARD``).
    The leg's test schema must be set, and be the one its cases resolve
    ``{catalog}`` and ``{schema}`` to (``SDM_CAPTURE_*``), or the run does
    not start. Returns what undoes it."""
    guards = guard.from_environment()
    own = guards.get(DIALECT[leg])
    if own is None:
        raise SystemExit(
            f'the write guard names no test schema for {leg} (SDM_LIVE_TEST_*)'
        )
    for variable, held in (
        ('SDM_CAPTURE_CATALOG', own.database),
        ('SDM_CAPTURE_SCHEMA', own.schema),
    ):
        value = os.environ.get(variable)
        if held is None or value is None:
            continue
        same = (
            value.lower() == held.lower()
            if own.dialect == 'databricks'
            else value == held
        )
        if not same:
            raise SystemExit(
                f"{variable} is not the write guard's test schema "
                '(SDM_LIVE_TEST_*): the loads would be refused'
            )
    undo = guard.install(guards)
    loader.WRITE_GUARD = guard._keeping(own.check)
    loader.WRITE_GUARD_SETUP = getattr(guard, 'setup', None)

    def uninstall() -> None:
        loader.WRITE_GUARD = None
        loader.WRITE_GUARD_SETUP = None
        undo()

    return uninstall


ENVIRONMENT_CODES = frozenset({'TRANSPORT_ERROR'})

ENVIRONMENT_ERRORS = frozenset(
    {'ConnectionError', 'ConnectionRefusedError', 'ConnectTimeout'}
)


def environment_failure(lines: list[Any]) -> str | None:
    r"""Why the observed ``lines`` hold an environment's failure rather
    than a product outcome, or ``None``: an outcome of
    :data:`ENVIRONMENT_CODES` or :data:`ENVIRONMENT_ERRORS`."""
    for line in lines:
        outcome = (line or {}).get('outcome') or {}
        if (
            outcome.get('code') in ENVIRONMENT_CODES
            or outcome.get('error') in ENVIRONMENT_ERRORS
        ):
            return (
                f'{outcome.get("error")} {outcome.get("code")} at '
                f'{outcome.get("stage")}: the run could not reach what it '
                'serves or reads, an environment failure, so no capture'
            )
    return None
