# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The existing Java runner admits and prepares its actual S2 SQL legs."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'contract/runners'))
import run_java  # noqa: E402


def case(name: str, slice_: str = 's2') -> dict[str, Any]:
    return json.loads(
        (ROOT / f'contract/cases/{slice_}/{name}.json').read_text()
    )


@pytest.mark.parametrize(
    'name',
    [
        'view.metric.from-source',
        'view.metric.multiclass',
        'view.metric.regression',
        'databricks.key.bigint-string',
    ],
)
def test_sql_warehouse_and_metric_view_cases_have_java_legs(name: str) -> None:
    given = case(name)
    assert run_java.on_java(given, 'databricks-sql')
    assert not run_java.on_java(given, 'databricks-spark')
    assert not run_java.on_java(given, 'snowflake')
    assert not run_java.on_java(given, 'lakebase')


def test_warehouse_capable_s1_binary_case_is_shared_with_java_sql() -> None:
    assert run_java.on_java(case('base.binary', 's1'), 'databricks-sql')


@pytest.mark.parametrize(
    'name', ['view.metric.ranking', 'view.metric.forecast']
)
def test_java_s2_never_admits_ranking_or_forecast(name: str) -> None:
    assert not run_java.on_java(case(name), 'databricks-sql')


def test_an_empty_path_list_cannot_establish_java_coverage() -> None:
    given = case('view.metric.from-source')
    given['paths'] = []
    assert not run_java.on_java(given, 'databricks-sql')


def test_an_empty_java_warehouse_selection_fails_before_starting_a_jvm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(run_java.run_python, 'load_cases', lambda paths: [])
    with pytest.raises(run_java.RunnerFailure, match='no Java cases'):
        run_java.run([], jar=tmp_path / 'absent.jar', database='databricks-sql')


def test_private_jdbc_configuration_requires_the_resolved_test_namespace() -> (
    None
):
    spec = {
        'jdbc': {
            'url': 'jdbc:databricks://host:443',
            'properties': {'token': 'secret'},
        }
    }
    for names in [{}, {'catalog': 'catalog'}, {'schema': 'schema'}]:
        with pytest.raises(run_java.RunnerFailure, match='catalog and schema'):
            run_java.jdbc_configuration(spec, names)
    with pytest.raises(run_java.RunnerFailure, match='JDBC'):
        run_java.jdbc_configuration(
            {}, {'catalog': 'catalog', 'schema': 'schema'}
        )
    made = run_java.jdbc_configuration(
        spec, {'catalog': 'catalog', 'schema': 'schema'}
    )
    assert made['url'] == spec['jdbc']['url']
    assert made['properties'] == {
        'token': 'secret',
        'ConnCatalog': 'catalog',
        'ConnSchema': 'schema',
    }
    assert spec['jdbc']['properties'] == {'token': 'secret'}


def test_a_metric_case_must_name_the_view_it_claims_to_exercise() -> None:
    given = case('view.metric.from-source')
    names = {'catalog': 'catalog', 'schema': 'schema'}
    assert (
        run_java.source_configuration(given, names)['view']
        == given['input']['view']
    )
    given['input'].pop('view')
    with pytest.raises(run_java.RunnerFailure, match='metric view'):
        run_java.source_configuration(given, names)


def test_source_configuration_preserves_partial_namespace_and_view_trust() -> (
    None
):
    given = case('view.metric.from-source')
    given['input']['named_part'] = 'schema'
    given['input']['view']['trust_expressions'] = True
    made = run_java.source_configuration(
        given, {'catalog': 'catalog', 'schema': 'schema'}
    )
    assert made['catalog'] is None
    assert made['schema'] == 'schema'
    assert made['view']['trust_expressions'] is True


def test_s2_loading_reuses_the_guarded_python_live_loader_and_always_closes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    given = case('view.metric.from-source')
    names = {'catalog': 'catalog', 'schema': 'schema'}
    monkeypatch.setattr(
        run_java.run_python,
        'live_configuration',
        lambda leg: (
            {'jdbc': {'url': 'jdbc:databricks://host:443', 'properties': {}}},
            names,
            [],
        ),
    )
    monkeypatch.setattr(
        run_java.run_python, 'resolve_placeholders', copy.deepcopy
    )

    class Live:
        def __init__(self, selected: object, bases: object, leg: str) -> None:
            events.append(('load', selected, bases, leg))

        def close(self) -> None:
            events.append('close')

    monkeypatch.setattr(run_java.run_python, 'Live', Live)
    monkeypatch.setattr(run_java.run_python, '_sibling', lambda name: 'loader')
    capture = SimpleNamespace(
        load_guard=lambda: 'guard',
        install_guard=lambda leg, guard, loader: (
            events.append(('guard', leg, guard, loader))
            or (lambda: events.append('unguard'))
        ),
    )
    monkeypatch.setattr(run_java, '_capture_run', lambda: capture)
    with pytest.raises(RuntimeError, match='caller failure'):
        with run_java.loaded(given, {}, 'databricks-sql') as how:
            assert how['properties']['ConnSchema'] == 'schema'
            assert (
                how['source']['view']['name'] == given['input']['view']['name']
            )
            raise RuntimeError('caller failure')
    assert events[0] == ('guard', 'databricks-sql', 'guard', 'loader')
    assert events[-2:] == ['close', 'unguard']


def test_java_sql_guard_uses_the_existing_validator_and_never_admits_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('SDM_LIVE_TEST_DATABASE', 'catalog')
    monkeypatch.setenv('SDM_LIVE_TEST_SCHEMA', 'schema')
    for sql in [
        'SELECT 1',
        'DESCRIBE QUERY SELECT CAST(NULL AS TIMESTAMP_NTZ)',
    ]:
        run_java.check_sql(sql)
    for sql in [
        'DROP TABLE catalog.schema.users',
        'CREATE TABLE catalog.schema.users (id INT)',
        'INSERT INTO catalog.schema.users VALUES (1)',
        'SELECT 1; DELETE FROM catalog.schema.users',
    ]:
        with pytest.raises(run_java.RunnerFailure):
            run_java.check_sql(sql)


def test_a_java_guard_refusal_prevents_jdbc_execution_and_is_not_a_product_answer(
    tmp_path: Path,
) -> None:
    # The same JAR/JDK gate as the existing Java runner acceptance tests.
    given = os.environ.get('SDM_JAVA_JAR')
    if not given:
        assert not os.environ.get('SDM_REQUIRE_JAVA')
        return
    jar = Path(given).resolve()
    home = os.environ.get('SDM_JAVA_HOME') or os.environ.get('JAVA_HOME')
    java = run_java.java_binary(str(Path(home) / 'bin/java') if home else None)
    javac = str(Path(java).with_name('javac'))
    guard = tmp_path / 'refuse.py'
    guard.write_text('import sys\nsys.stdin.read()\nraise SystemExit(1)\n')
    probe = tmp_path / 'GuardProbe.java'
    probe.write_text("""
import java.lang.reflect.Proxy;
import java.sql.Connection;
import java.sql.SQLException;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public class GuardProbe {
  public static void main(String[] args) throws Exception {
    AtomicInteger reached = new AtomicInteger();
    Connection raw = (Connection) Proxy.newProxyInstance(
        Connection.class.getClassLoader(), new Class<?>[] {Connection.class},
        (self, method, values) -> {
          reached.incrementAndGet();
          throw new AssertionError("unguarded JDBC reached");
        });
    ContractRunner.SqlGuard guard = new ContractRunner.SqlGuard(
        Map.of("guard_python", args[0], "guard_program", args[1]));
    try {
      guard.wrap(raw).prepareStatement("DROP TABLE private_test.users");
      throw new AssertionError("a refused statement was admitted");
    } catch (SQLException expected) {
      if (reached.get() != 0) throw new AssertionError("the driver was reached");
    }
    try {
      guard.checkPassed();
      throw new AssertionError("a refusal could become a product observation");
    } catch (IllegalStateException expected) {
      System.out.println("GUARD_REFUSAL_IS_RUNNER_FAILURE");
    }
  }
}
""")
    classpath = run_java.classpath(jar, None)
    compiled = subprocess.run(
        [
            javac,
            '-cp',
            classpath,
            '-d',
            str(tmp_path),
            str(run_java.PROGRAM),
            str(probe),
        ],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    done = subprocess.run(
        [
            java,
            *run_java.FLAGS,
            '-cp',
            str(tmp_path) + os.pathsep + classpath,
            'GuardProbe',
            sys.executable,
            str(guard),
        ],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip() == 'GUARD_REFUSAL_IS_RUNNER_FAILURE'
