# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Managed Spark's consumed driver may have no distribution metadata.

The declared versions below belong to synthetic driver fixtures, not live
runtime evidence. Exercise the producer, serialized record and locked checker.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow as pa
import pytest

HERE = Path(__file__).resolve().parent
NAME = 't310_spark_recording_fixtures'
spec = importlib.util.spec_from_file_location(
    NAME, HERE / 'test_t32_recording_own.py'
)
assert spec is not None and spec.loader is not None
OWN = importlib.util.module_from_spec(spec)
sys.modules[NAME] = OWN
spec.loader.exec_module(OWN)
run, check = OWN.run, OWN.check

MODULE_VERSION = '4.0.0.test-fixture'
DISTRIBUTION_VERSION = '4.0.1.test-fixture'


class Frame:
    def __init__(self, table: Any) -> None:
        self.table = table
        self.columns = table.column_names

    def toArrow(self) -> Any:  # noqa: N802 - Spark's public method
        return self.table


class SparkSession:
    version = 'server-version-is-not-the-client-version'

    def sql(self, query: str, args: Any = None) -> Frame:
        return Frame(OWN._answers()[query.split('\nLIMIT')[0]])


SparkSession.__module__ = 'pyspark.sql.session'


class OtherSession:
    sql = SparkSession.sql


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


def _record(connection: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    case = OWN._case()
    bases = run.load_bases()
    tables = check._duck_tables(case, bases)
    log = run.DriverLog()
    log.recorder = run.Recorder()
    captured = run._CapturedSpark(connection, log)
    log.stage = 'from_source'
    captured.sql('SELECT current_catalog()').toArrow()
    log.stage = 'materialize'
    captured.sql(OWN._sql(OWN.SAMPLE)).toArrow()
    captured.sql(OWN._sql(OWN.VIEW_READ) + '\nLIMIT 1000').toArrow()
    observation = OWN._reference_line('base.binary@postgres')
    observation['case'] = 'base.binary@databricks-spark'
    observation['driver']['capture'] = log.record()
    live = OWN.FakeLive(tables)
    live.leg = 'databricks-spark'
    live.connection = connection
    relations = [
        {
            'statement': 2,
            'sql': OWN._sql(OWN.VIEW_READ),
            'table': pa.table({'price': [9.5, 12.0]}),
        }
    ]
    line = run.capture_line(live, case, bases, log, observation, relations)
    return case, json.loads(
        run.written(line, run.Placeheld(live.names, live.private))
    )


def _locked(tmp_path: Path, case: dict[str, Any], line: dict[str, Any]) -> int:
    case_path, capture_path = tmp_path / 'case.json', tmp_path / 'record.jsonl'
    case_path.write_text(json.dumps(case), encoding='utf-8')
    capture_path.write_text(json.dumps(line) + '\n', encoding='utf-8')
    return check.recording_main(
        [
            '--cases',
            str(case_path),
            '--capture',
            str(capture_path),
            '--leg',
            'databricks-spark',
            '--locked',
        ]
    )


def test_a_managed_module_version_produces_a_complete_locked_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _metadata(monkeypatch)
    module = ModuleType('pyspark')
    module.__version__ = MODULE_VERSION
    monkeypatch.setitem(sys.modules, 'pyspark', module)
    case, line = _record(SparkSession())
    assert line['versions']['pyspark'] == MODULE_VERSION
    assert line['versions']['runtime'] == 'runtime-release-fixture'
    assert _locked(tmp_path, case, line) == 0


def test_distribution_metadata_keeps_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _metadata(monkeypatch, pyspark=DISTRIBUTION_VERSION)
    module = ModuleType('pyspark')
    module.__version__ = MODULE_VERSION
    monkeypatch.setitem(sys.modules, 'pyspark', module)
    case, line = _record(SparkSession())
    assert line['versions']['pyspark'] == DISTRIBUTION_VERSION
    assert _locked(tmp_path, case, line) == 0


@pytest.mark.parametrize('value', [None, '', '   ', 7])
def test_an_unusable_module_version_still_fails_the_locked_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: Any
) -> None:
    _metadata(monkeypatch)
    module = ModuleType('pyspark')
    if value is not None:
        module.__version__ = value
    monkeypatch.setitem(sys.modules, 'pyspark', module)
    case, line = _record(SparkSession())
    assert 'pyspark' not in line['versions']
    assert _locked(tmp_path, case, line) == 1


def test_an_unloaded_module_is_not_imported_to_manufacture_a_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _metadata(monkeypatch)
    monkeypatch.delitem(sys.modules, 'pyspark', raising=False)
    case, line = _record(SparkSession())
    assert 'pyspark' not in sys.modules
    assert 'pyspark' not in line['versions']
    assert _locked(tmp_path, case, line) == 1


def test_a_different_consumed_driver_cannot_borrow_the_module_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _metadata(monkeypatch)
    module = ModuleType('pyspark')
    module.__version__ = MODULE_VERSION
    monkeypatch.setitem(sys.modules, 'pyspark', module)
    case, line = _record(OtherSession())
    assert 'pyspark' not in line['versions']
    assert _locked(tmp_path, case, line) == 1


def test_another_backend_keeps_its_existing_version_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _metadata(monkeypatch)
    module = ModuleType('pyspark')
    module.__version__ = MODULE_VERSION
    monkeypatch.setitem(sys.modules, 'pyspark', module)
    case, line, tables = OWN._capture()
    assert line['leg'] == 'databricks-sql'
    assert 'pyspark' not in line['versions']
    assert check.capture_problems(line, case, tables) == []
