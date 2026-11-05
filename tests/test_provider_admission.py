# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Native provider admission and advertised capabilities."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _module(
    ROOT / 'contract/runners/run_python.py', 'provider_admission_runner'
)


def test_the_lakebase_core_run_reaches_its_case_without_opening_a_warehouse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sdm_client._native as native
    from sdm_client._rust import run as host_run

    assert 'postgres' in native.PROVIDERS
    # The runner wraps this function to record native calls; restore its
    # original value when the test ends without changing capabilities.
    monkeypatch.setattr(host_run, 'native', host_run.native)
    monkeypatch.delenv('SDM_LIVE_CONFIG', raising=False)

    class CaseReached(Exception):
        pass

    def stop_before_connections(*args: Any) -> Any:
        assert args[0]['id'] == 'lakebase.binary'
        raise CaseReached

    monkeypatch.setattr(RUNNER, '_call', stop_before_connections)
    cases = RUNNER.load_cases([ROOT / 'contract/cases/s2/lakebase.binary.json'])
    with pytest.raises(CaseReached):
        RUNNER.run('python-core', cases, database='lakebase')


def test_the_built_core_reports_its_providers_as_plain_data() -> None:
    r"""Exit: python-core's marker is a private list of database names in
    the native module, DuckDB among them; never public API."""
    import sdm_client
    import sdm_client._native as native

    providers = native.PROVIDERS
    assert isinstance(providers, tuple)
    assert all(isinstance(name, str) for name in providers)
    assert 'duckdb' in providers
    # Every advertised provider is supported by the case format.
    assert set(providers) <= {
        'duckdb',
        'sqlite',
        'postgres',
        'databricks-sql',
        'databricks-spark',
        'snowflake',
    }
    assert not hasattr(sdm_client, 'PROVIDERS')


@pytest.mark.parametrize('database', ['databricks-sql', 'databricks-spark'])
def test_databricks_admission_keeps_separate_provider_ids(
    database: str,
) -> None:
    assert RUNNER.provider_for_leg(database) == database


def test_lakebase_uses_the_postgres_provider() -> None:
    assert RUNNER.provider_for_leg('lakebase') == 'postgres'
