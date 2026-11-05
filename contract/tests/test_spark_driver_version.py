# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Managed Spark's consumed driver may have no distribution metadata.

The declared versions below belong to synthetic driver fixtures, not live
runtime evidence. Exercise the version producer directly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
NAME = 't310_spark_recording_fixtures'
spec = importlib.util.spec_from_file_location(
    NAME, HERE / 'test_driver_capture.py'
)
assert spec is not None and spec.loader is not None
OWN = importlib.util.module_from_spec(spec)
sys.modules[NAME] = OWN
spec.loader.exec_module(OWN)
run = OWN.run

MODULE_VERSION = '4.0.0.test-fixture'
DISTRIBUTION_VERSION = '4.0.1.test-fixture'


class SparkSession:
    version = 'server-version-is-not-the-client-version'


class OtherSession:
    pass


SparkSession.__module__ = 'pyspark.sql.session'


def _metadata(monkeypatch: pytest.MonkeyPatch, **given: str) -> None:
    original = run.importlib.metadata.version

    def version(name: str) -> str:
        if name in {'pyspark', 'databricks-connect'}:
            if name in given:
                return given[name]
            raise run.importlib.metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(run.importlib.metadata, 'version', version)
    monkeypatch.setenv('DATABRICKS_RUNTIME_VERSION', 'runtime-release-fixture')


def test_a_managed_module_reports_its_client_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _metadata(monkeypatch)
    module = ModuleType('pyspark')
    module.__version__ = MODULE_VERSION
    monkeypatch.setitem(sys.modules, 'pyspark', module)
    version = run._driver_version(
        SimpleNamespace(leg='databricks-spark', connection=SparkSession()),
        'pyspark',
    )
    assert version == MODULE_VERSION


def test_distribution_metadata_keeps_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _metadata(monkeypatch, pyspark=DISTRIBUTION_VERSION)
    module = ModuleType('pyspark')
    module.__version__ = MODULE_VERSION
    monkeypatch.setitem(sys.modules, 'pyspark', module)
    version = run._driver_version(
        SimpleNamespace(leg='databricks-spark', connection=SparkSession()),
        'pyspark',
    )
    assert version == DISTRIBUTION_VERSION


@pytest.mark.parametrize('value', [None, '', '   ', 7])
def test_an_unusable_module_version_is_not_reported(
    monkeypatch: pytest.MonkeyPatch, value: Any
) -> None:
    _metadata(monkeypatch)
    module = ModuleType('pyspark')
    if value is not None:
        module.__version__ = value
    monkeypatch.setitem(sys.modules, 'pyspark', module)
    version = run._driver_version(
        SimpleNamespace(leg='databricks-spark', connection=SparkSession()),
        'pyspark',
    )
    assert version is None


def test_an_unloaded_module_is_not_imported_to_manufacture_a_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _metadata(monkeypatch)
    monkeypatch.delitem(sys.modules, 'pyspark', raising=False)
    version = run._driver_version(
        SimpleNamespace(leg='databricks-spark', connection=SparkSession()),
        'pyspark',
    )
    assert 'pyspark' not in sys.modules
    assert version is None


def test_a_different_consumed_driver_cannot_borrow_the_module_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _metadata(monkeypatch)
    module = ModuleType('pyspark')
    module.__version__ = MODULE_VERSION
    monkeypatch.setitem(sys.modules, 'pyspark', module)
    version = run._driver_version(
        SimpleNamespace(leg='databricks-spark', connection=OtherSession()),
        'pyspark',
    )
    assert version is None


def test_another_backend_keeps_its_existing_version_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _metadata(monkeypatch)
    module = ModuleType('pyspark')
    module.__version__ = MODULE_VERSION
    monkeypatch.setitem(sys.modules, 'pyspark', module)
    assert (
        run._driver_version(
            SimpleNamespace(leg='databricks-sql', connection=SparkSession()),
            'pyspark',
        )
        is None
    )
