# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Java CI exercises supported JDKs, platforms and clean JAR loading."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / '.github/workflows/ci.yml'
#: The brief's Java S0 jobs: JDK, platform.
S0_JOBS = (
    ('17', 'linux-x86_64'),
    ('21', 'linux-x86_64'),
    ('21', 'linux-aarch64'),
    ('21', 'macos-arm64'),
    ('21', 'macos-x86_64'),
)
#: Each platform's runner label, as the workflow's other jobs name them.
RUNNERS = {
    'linux-x86_64': re.compile(r'^ubuntu-(?!.*arm)'),
    'linux-aarch64': re.compile(r'^ubuntu-.*arm'),
    'macos-arm64': re.compile(r'^macos-15$'),
    'macos-x86_64': re.compile(r'^macos-15-intel$'),
}
#: A step that sets Rust up.
RUST = re.compile(r'rust-toolchain|rustup|\bcargo\b|maturin')


def _combinations(
    matrix: dict[str, Any], axes: dict[str, list[Any]]
) -> list[dict[str, Any]]:
    r"""A matrix's combinations as GitHub forms them: the axes' product, less
    each ``exclude`` it matches; then each ``include`` added to every
    combination it overwrites no axis value of, or, added to none, a
    combination of its own. No matrix: one empty combination."""
    if not matrix:
        return [{}]
    combos = [
        dict(zip(axes, values, strict=True)) for values in itertools.product(*axes.values())
    ] if axes else []  # fmt: skip
    for excluded in matrix.get('exclude') or []:
        combos = [
            c
            for c in combos
            if not all(c.get(k) == v for k, v in excluded.items())
        ]
    original = [dict(c) for c in combos]
    for included in matrix.get('include') or []:
        added = False
        for combo, before in zip(
            combos[: len(original)], original, strict=True
        ):
            if all(before[k] == v for k, v in included.items() if k in before):
                combo.update(included)
                added = True
        if not added:
            combos.append(dict(included))
    return combos or [{}]


def _expanded() -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    r"""Every Java job: its expanded name, job, and matrix values.

    Other job families may use runtime matrices; Java coverage must not
    depend on statically evaluating an unrelated job's GitHub expression.
    """
    workflow = yaml.safe_load(CI.read_text(encoding='utf-8'))
    found = []
    for key, job in workflow['jobs'].items():
        if not str(job.get('name', key)).startswith('java'):
            continue
        matrix = (job.get('strategy') or {}).get('matrix') or {}
        if isinstance(matrix, str):
            matrix = {}
        axes = {
            name: values
            for name, values in matrix.items()
            if name not in ('include', 'exclude') and isinstance(values, list)
        }
        for values in _combinations(matrix, axes):
            name = str(job.get('name', key))
            for axis, value in values.items():
                name = re.sub(
                    r'\$\{\{\s*matrix\.' + re.escape(axis) + r'\s*\}\}',
                    str(value),
                    name,
                )
            found.append((name, job, values))
    return found


def _runs_on(job: dict[str, Any], values: dict[str, Any]) -> str:
    label = job.get('runs-on', '')
    if isinstance(label, list):
        label = ' '.join(map(str, label))
    for axis, value in values.items():
        label = re.sub(
            r'\$\{\{\s*matrix\.' + re.escape(axis) + r'\s*\}\}',
            str(value),
            str(label),
        )
    return str(label)


def test_t311_the_java_s0_jobs_run_on_each_jdk_and_platform() -> None:
    # Exit 1's CI line and exit 4's platforms: one java (<jdk>, <platform>,
    # s0) job each, on its platform's runner.
    jobs = {name: (job, values) for name, job, values in _expanded()}
    for jdk, platform in S0_JOBS:
        name = f'java ({jdk}, {platform}, s0)'
        assert name in jobs, (name, sorted(jobs))
        job, values = jobs[name]
        assert RUNNERS[platform].search(_runs_on(job, values)), (
            name,
            _runs_on(job, values),
        )
    # No Java S2 leg yet: a java job names S0, or from T3.12 one of the S1
    # databases (tests/test_t312_java_ci.py), never a warehouse.
    legs = {
        match.group(1)
        for name in jobs
        for match in [re.match(r'java \(\d+, [a-z0-9_-]+, ([^)]+)\)$', name)]
        if match
    }
    assert legs <= {'s0', 'duckdb', 'sqlite', 'postgres'}, legs


def test_t311_the_jar_load_jobs_run_on_each_jdk_and_platform_without_rust() -> (
    None
):
    # Exit 4: the JAR-load test on JDK 17 and 21 on Linux x86_64 and JDK 21
    # on Linux aarch64 and both macOS, where no Rust toolchain is: a
    # container on Linux, no Rust step anywhere.
    loads = [
        (name, job, values)
        for name, job, values in _expanded()
        if 'load' in name.lower()
    ]
    for jdk, platform in S0_JOBS:
        found = [
            (name, job, values)
            for name, job, values in loads
            if f'({jdk}, {platform}' in name
        ]
        assert len(found) == 1, (jdk, platform, [n for n, _, _ in loads])
        name, job, values = found[0]
        assert RUNNERS[platform].search(_runs_on(job, values)), name
        if platform.startswith('linux'):
            assert job.get('container'), f'{name} runs in no container'
        steps = job.get('steps') or []
        rust = [
            step
            for step in steps
            if RUST.search(str(step.get('uses', '')) + str(step.get('run', '')))
        ]
        assert rust == [], (name, rust)


def test_each_parser_platform_runs_behavioral_assertions():
    jobs = [
        job
        for name, job, _ in _expanded()
        if name.startswith('java (') and name.endswith(', s0)')
    ]
    assert len(jobs) == len(S0_JOBS)
    for job in jobs:
        commands = '\n'.join(str(step.get('run', '')) for step in job['steps'])
        assert '-Dtest=PredictiveQueryTest test' in commands
        assert '-Dsdm.natives=' in commands


def test_windows_java_builds_are_packaged_and_loaded():
    workflow = yaml.safe_load(CI.read_text(encoding='utf-8'))
    jobs = workflow['jobs']
    builds = jobs['java-native']['strategy']['matrix']['include']
    loads = jobs['java-load']['strategy']['matrix']['include']
    pom = (ROOT / 'java/pom.xml').read_text()
    assert '<include>*/sdm_jni.dll</include>' in pom
    for arch, runner in [
        ('x86_64', 'windows-latest'),
    ]:
        platform = f'windows-{arch}'
        assert {'platform': platform, 'runner': runner} in builds
        for jdk in ['17', '21']:
            assert any(
                row['platform'] == platform
                and row['jdk'] == jdk
                and row['runner'] == runner
                for row in loads
            )
        assert f'${{sdm.natives}}/{platform}/sdm_jni.dll' in pom
    commands = '\n'.join(
        step.get('run', '') for step in jobs['java-load']['steps']
    )
    assert "separator=';'" in commands


@pytest.mark.parametrize(
    'corruption', [None, 'commit', 'digest', 'missing', 'version']
)
def test_java_release_rejects_mismatched_artifacts(
    tmp_path, monkeypatch, corruption
):
    document = yaml.safe_load(
        (ROOT / '.github/workflows/publish.yml').read_text()
    )
    step = next(
        step
        for step in document['jobs']['build']['steps']
        if step.get('name') == 'Verify and assemble the tested Java artifacts'
    )
    code = step['run'].split("python - <<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GITHUB_SHA', 'candidate')
    monkeypatch.setenv('GITHUB_REF_NAME', 'v2.3.4')
    root = tmp_path / 'java-release'
    root.mkdir()
    (tmp_path / 'dist').mkdir()
    stem = 'structured-data-models-client-2.3.4'
    jar = root / (stem + '.jar')
    jar.write_bytes(b'tested artifact fixture')
    pom = root / (stem + '.pom')
    version = '2.3.5' if corruption == 'version' else '2.3.4'
    pom.write_text(
        f'<project xmlns="http://maven.apache.org/POM/4.0.0"><version>{version}</version></project>'
    )
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [jar, pom]
    }
    if corruption == 'digest':
        jar.write_bytes(b'changed after testing')
    if corruption == 'missing':
        del files[pom.name]
    record = {
        'commit': 'other' if corruption == 'commit' else 'candidate',
        'files': files,
    }
    (root / 'java-artifacts.json').write_text(json.dumps(record))
    if corruption:
        with pytest.raises(AssertionError):
            exec(compile(code, '<release artifact check>', 'exec'), {})
    else:
        exec(compile(code, '<release artifact check>', 'exec'), {})
        assert (tmp_path / 'dist' / jar.name).read_bytes() == jar.read_bytes()
        assert (tmp_path / 'dist' / pom.name).read_bytes() == pom.read_bytes()


def test_java_release_paths_follow_the_pom_version():
    ci = CI.read_text()
    live = (ROOT / '.github/workflows/parity-live.yml').read_text()
    assert 'structured-data-models-client-1.0.0.jar' not in ci + live
    assert "ET.parse('java/pom.xml')" in ci
    assert "ET.parse('java/pom.xml')" in live
    assert 'SDM_JAVA_VERSION' in live
