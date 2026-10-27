# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T3.12: a Java path's JUnit test counts only where the Java test run the
coverage check itself starts (``coverage.py --junit-run``) ran it and did
not skip it. The run starts from an empty Surefire reports directory, so no
report an earlier build left, copied in, or dated by a skewed clock is that
run's (Codex T3.12 r6 P1-1, r7 P1-1).

The two-build case runs real Maven (``SDM_MAVEN``), offline against the
local repository, as CI's java-test job sets it; without it the test returns
without asserting unless ``SDM_REQUIRE_JAVA`` is set. The other cases need
no Maven: a stand-in that runs no test.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / 'scripts/rust_core/coverage.py'
CLAIM = 'java/src/test/java/com/nvidia/sdm/CoveredTest.java#covered'
REPORT = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<testsuite name="com.nvidia.sdm.CoveredTest">'
    '<testcase name="covered" classname="com.nvidia.sdm.CoveredTest" time="0.1"/>'
    '</testsuite>\n'
)


def _project(root: Path) -> Path:
    r"""A repository holding a Java module with ``CoveredTest.covered`` and a
    control test, and a Java manifest claiming the first."""
    java = root / 'java'
    sources = java / 'src/test/java/com/nvidia/sdm'
    sources.mkdir(parents=True)
    plugins = ''.join(
        '<plugin><groupId>org.apache.maven.plugins</groupId>'
        f'<artifactId>{name}</artifactId><version>{version}</version></plugin>'
        for name, version in (
            ('maven-compiler-plugin', '3.14.0'),
            ('maven-resources-plugin', '3.3.1'),
            ('maven-jar-plugin', '3.4.2'),
            ('maven-surefire-plugin', '3.5.3'),
        )
    )
    (java / 'pom.xml').write_text(
        '<project xmlns="http://maven.apache.org/POM/4.0.0"><modelVersion>4.0.0'
        '</modelVersion><groupId>t312</groupId><artifactId>stale</artifactId>'
        '<version>1</version><properties><maven.compiler.release>17'
        '</maven.compiler.release></properties><dependencies><dependency>'
        '<groupId>org.junit.jupiter</groupId><artifactId>junit-jupiter</artifactId>'
        '<version>5.13.4</version><scope>test</scope></dependency></dependencies>'
        f'<build><plugins>{plugins}</plugins></build></project>',
        encoding='utf-8',
    )
    (sources / 'CoveredTest.java').write_text(
        'package com.nvidia.sdm; import org.junit.jupiter.api.Test; '
        'class CoveredTest { @Test void covered() {} }',
        encoding='utf-8',
    )
    (sources / 'ControlTest.java').write_text(
        'package com.nvidia.sdm; import org.junit.jupiter.api.Test; '
        'class ControlTest { @Test void control() {} }',
        encoding='utf-8',
    )
    manifest = root / 'docs/design/rust-core/phase-0/scope-java.yaml'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        f'paths:\n- id: covered/default/sqlite\n  tests:\n  - {CLAIM}\n',
        encoding='utf-8',
    )
    return root / 'java/target/surefire-reports'


def _checked(
    root: Path, maven: str, *more: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(COVERAGE),
            '--repo',
            str(root),
            '--junit-run',
            '--maven',
            maven,
            *more,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )


def _no_test_run(tmp: Path) -> str:
    r"""A stand-in for Maven that succeeds and runs no test."""
    script = tmp / 'maven'
    script.write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
    script.chmod(0o755)
    return str(script)


def test_a_report_dated_ahead_by_a_skewed_clock_claims_nothing(
    tmp_path: Path,
) -> None:
    reports = _project(tmp_path)
    reports.mkdir(parents=True)
    stale = reports / 'TEST-com.nvidia.sdm.CoveredTest.xml'
    stale.write_text(REPORT, encoding='utf-8')
    ahead = stale.stat().st_mtime + 3600
    os.utime(stale, (ahead, ahead))
    ran = _checked(tmp_path, _no_test_run(tmp_path))
    assert ran.returncode == 1, ran.stdout + ran.stderr
    assert f'{CLAIM}: not a test JUnit ran' in ran.stdout


def test_a_report_copied_in_before_the_run_claims_nothing(
    tmp_path: Path,
) -> None:
    reports = _project(tmp_path)
    reports.mkdir(parents=True)
    copied = tmp_path / 'kept.xml'
    copied.write_text(REPORT, encoding='utf-8')
    shutil.copyfile(copied, reports / 'TEST-com.nvidia.sdm.CoveredTest.xml')
    ran = _checked(tmp_path, _no_test_run(tmp_path))
    assert ran.returncode == 1, ran.stdout + ran.stderr


def test_reports_are_never_read_but_from_the_check_s_own_run(
    tmp_path: Path,
) -> None:
    # A marker and a report an earlier run left, read on their own: there
    # is no such mode (Claude T3.12 r7 P2-1).
    reports = _project(tmp_path)
    reports.mkdir(parents=True)
    (reports / '.coverage-junit-run').write_text('', encoding='utf-8')
    (reports / 'TEST-com.nvidia.sdm.CoveredTest.xml').write_text(
        REPORT, encoding='utf-8'
    )
    read = subprocess.run(
        [
            sys.executable,
            str(COVERAGE),
            '--repo',
            str(tmp_path),
            '--junit-reports',
            str(reports),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert read.returncode != 0, read.stdout
    ran = _checked(tmp_path, _no_test_run(tmp_path))
    assert ran.returncode == 1, ran.stdout + ran.stderr


def test_the_run_leaves_the_rest_of_the_build_directory(tmp_path: Path) -> None:
    # Only the reports directory is emptied: the natives a CI job puts in
    # java/target stay.
    reports = _project(tmp_path)
    natives = tmp_path / 'java/target/natives/linux-x86_64'
    natives.mkdir(parents=True)
    (natives / 'libsdm_jni.so').write_bytes(b'native')
    reports.mkdir(parents=True)
    _checked(tmp_path, _no_test_run(tmp_path))
    assert (natives / 'libsdm_jni.so').read_bytes() == b'native'


def test_a_test_removed_between_two_builds_fails_coverage(
    tmp_path: Path,
) -> None:
    # Codex T3.12 r6 P1-1, end to end: two successful Maven builds of one
    # project, the claimed test's @Test removed between them.
    maven = os.environ.get('SDM_MAVEN')
    if not maven:
        assert not os.environ.get('SDM_REQUIRE_JAVA'), (
            'SDM_REQUIRE_JAVA is set but SDM_MAVEN is not'
        )
        return
    _project(tmp_path)
    first = _checked(tmp_path, maven, '--maven-arg=-o', '--maven-arg=-q')
    assert first.returncode == 0, first.stdout + first.stderr
    (
        tmp_path / 'java/src/test/java/com/nvidia/sdm/CoveredTest.java'
    ).write_text(
        'package com.nvidia.sdm; class CoveredTest { void covered() {} }',
        encoding='utf-8',
    )
    second = _checked(tmp_path, maven, '--maven-arg=-o', '--maven-arg=-q')
    assert second.returncode == 1, second.stdout + second.stderr
    assert f'{CLAIM}: not a test JUnit ran' in second.stdout
