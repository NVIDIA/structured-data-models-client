# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T3.12's CI jobs, read from the workflow: Java's S1 legs, ``java (<jdk>,
<platform>, duckdb|sqlite|postgres)``, on JDK 17 and 21 on Linux x86_64 and
JDK 21 on aarch64, each judged on its own against the reference with the
effective expectations; L5 for Java beside the Postgres leg; the Java tests
with the pinned Postgres service, required; and the JDBC drivers in the
JAR's artifact. The macOS runners hold no client wheel for the runner's
capture to read with, so the macOS Java jobs stay S0's (the brief's open
question 3, read so: Postgres and the S1 legs where the runner has them).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / '.github/workflows/ci.yml'
#: Java's S1 legs: JDK, platform, database.
S1_LEGS = {
    (jdk, platform, database)
    for jdk, platform in (
        ('17', 'linux-x86_64'),
        ('21', 'linux-x86_64'),
        ('21', 'linux-aarch64'),
    )
    for database in ('duckdb', 'sqlite', 'postgres')
}


def _t311() -> ModuleType:
    path = ROOT / 'tests/test_t311_java_ci.py'
    spec = importlib.util.spec_from_file_location('t312_t311_ci', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['t312_t311_ci'] = module
    spec.loader.exec_module(module)
    return module


def _jobs() -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    return {name: (job, values) for name, job, values in _t311()._expanded()}


def _text(job: dict[str, Any]) -> str:
    return '\n'.join(str(step.get('run', '')) for step in job.get('steps', []))


def test_t312_each_s1_leg_runs_on_its_jdk_and_platform() -> None:
    jobs = _jobs()
    found = set()
    for name in jobs:
        parts = name.removeprefix('java (').removesuffix(')').split(', ')
        if name.startswith('java (') and len(parts) == 3 and parts[2] != 's0':
            found.add(tuple(parts))
    assert found == S1_LEGS


def test_t312_each_s1_leg_runs_the_cases_and_judges_them_alone() -> None:
    jobs = _jobs()
    for jdk, platform, database in sorted(S1_LEGS):
        job, values = jobs[f'java ({jdk}, {platform}, {database})']
        text = _text(job)
        assert 'contract/runners/run_java.py --database' in text, database
        assert '--drivers jar/jdbc' in text
        assert 'scripts/rust_core/ratchet.py effective' in text
        assert 'scripts/rust_core/compare.py --cases java-cases' in text
        assert 'postgres' in (job.get('services') or {}), database
        assert job['env']['SDM_POSTGRES_TEST_DSN'].startswith('postgresql://')


def test_t312_l5_for_java_runs_beside_the_postgres_leg() -> None:
    job, _ = _jobs()['java (21, linux-x86_64, postgres)']
    steps = [
        step
        for step in job['steps']
        if 'tests_test_ops/test_l5_java.py' in str(step.get('run', ''))
    ]
    assert len(steps) == 1
    assert steps[0]['env']['SDM_REQUIRE_POSTGRES'] == '1'
    assert steps[0]['env']['SDM_REQUIRE_JAVA'] == '1'
    assert 'native-test-ops' in steps[0]['run']
    # The driver-by-driver parity with Python's providers runs beside it.
    assert 'tests_test_ops/test_t312_driver_parity.py' in steps[0]['run']
    # Codex T3.12 r1 P2-7: the step changes directory before pytest reads
    # the JAR, so its path is the checkout's, absolute.
    assert steps[0]['env']['SDM_JAVA_JAR'].startswith(
        '${{ github.workspace }}/'
    ), steps[0]['env']['SDM_JAVA_JAR']


def test_t312_the_java_tests_run_with_postgres_required() -> None:
    jobs = _jobs()
    for jdk in ('17', '21'):
        job, _ = jobs[f'java-test ({jdk})']
        assert 'postgres' in job['services']
        assert job['env']['SDM_REQUIRE_POSTGRES'] == '1'
        assert job['env']['SDM_POSTGRES_TEST_DSN'].startswith('postgresql://')


def test_t312_the_jar_s_artifact_holds_the_jdbc_drivers() -> None:
    job, _ = _jobs()['java-jar']
    assert 'cp -R java/target/jdbc java-jar/jdbc' in _text(job)
    pom = (ROOT / 'java/pom.xml').read_text(encoding='utf-8')
    for driver in ('duckdb_jdbc', 'sqlite-jdbc', 'postgresql'):
        assert driver in pom, driver
