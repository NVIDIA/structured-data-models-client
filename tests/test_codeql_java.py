# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A green analysis must include the actual shipped Java sources."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _check(archive: Path, root: Path) -> int:
    spec = importlib.util.spec_from_file_location(
        'check_codeql_java', ROOT / 'scripts/check_codeql_java.py'
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check(archive, root)


@pytest.mark.parametrize('failure', [None, 'missing', 'stale', 'duplicate'])
def test_extraction_requires_every_current_source(
    tmp_path: Path, failure: str | None
) -> None:
    paths = [
        'java/src/main/java/Client.java',
        'java/src/main/java/SparkProvider.java',
    ]
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('class Example {}')
    archive = tmp_path / 'src.zip'
    with zipfile.ZipFile(archive, 'w') as out:
        for path in paths:
            if failure == 'missing' and 'Spark' in path:
                continue
            out.writestr(
                '/home/runner/work/repo/' + path,
                'old bytes' if failure == 'stale' else 'class Example {}',
            )
        if failure == 'duplicate':
            out.writestr('/another/checkout/' + paths[0], 'class Example {}')
    if failure:
        with pytest.raises(ValueError):
            _check(archive, tmp_path)
    else:
        assert _check(archive, tmp_path) == 2


def test_empty_source_tree_is_not_coverage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='no shipped Java'):
        _check(tmp_path / 'not-needed.zip', tmp_path)


def test_java_build_is_traced_before_analysis_and_checked_after() -> None:
    workflow = yaml.safe_load(
        (ROOT / '.github/workflows/codeql.yml').read_text()
    )
    job = workflow['jobs']['analyze']
    modes = {
        item['language']: item['build-mode']
        for item in job['strategy']['matrix']['include']
    }
    assert modes == {'python': 'none', 'rust': 'none', 'java-kotlin': 'manual'}
    steps = job['steps']
    init = next(
        i
        for i, step in enumerate(steps)
        if 'codeql-action/init@' in step.get('uses', '')
    )
    build = next(
        i for i, step in enumerate(steps) if 'mvn ' in step.get('run', '')
    )
    analyze = next(
        i
        for i, step in enumerate(steps)
        if 'codeql-action/analyze@' in step.get('uses', '')
    )
    verify = next(
        i
        for i, step in enumerate(steps)
        if 'check_codeql_java.py' in step.get('run', '')
    )
    assert init < build < analyze < verify
    assert steps[build]['run'].endswith('clean compile')
    assert (
        steps[build]['if']
        == steps[verify]['if']
        == "matrix.language == 'java-kotlin'"
    )
    assert job['permissions']['security-events'] == 'write'
