# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The nightly's Java probes use the current API and survive a changed cwd."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_a_relative_jar_is_resolved_before_the_probe_changes_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        't313_java_acceptance', ROOT / 'contract/tests/test_t311_java.py'
    )
    assert spec is not None and spec.loader is not None
    acceptance = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = acceptance
    spec.loader.exec_module(acceptance)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('SDM_JAVA_JAR', 'jar/client.jar')
    commands = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, '', '')

    monkeypatch.setattr(acceptance.subprocess, 'run', run)
    acceptance._probe(Path('/jdk'), acceptance._jar(), tmp_path / 'scratch')
    command, options = commands[0]
    assert command[command.index('-cp') + 1] == str(tmp_path / 'jar/client.jar')
    assert options['cwd'] == tmp_path / 'scratch'


def test_the_ci_jar_load_source_compiles_against_the_shipped_api(tmp_path: Path) -> None:
    given = os.environ.get('SDM_JAVA_JAR')
    home = os.environ.get('SDM_JAVA_HOME') or os.environ.get('JAVA_HOME')
    if not given or not home:
        pytest.skip('needs the package JAR and JDK, as the nightly load job does')
    jar = Path(given).resolve()
    done = subprocess.run(
        [str(Path(home) / 'bin/javac'), '-cp', str(jar), '-d', str(tmp_path),
         str(ROOT / 'java/jar-load/JarLoad.java')],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr
