# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T3.11's CI jobs, read from the workflow (exits 1 and 4, and locked check
14's CI line, ``java (<jdk>, <platform>, <database or leg>)``).

The brief names the S0 jobs ``java (17, linux-x86_64, s0)``, ``java (21,
linux-x86_64, s0)``, ``java (21, linux-aarch64, s0)``, ``java (21,
macos-arm64, s0)`` and ``java (21, macos-x86_64, s0)`` (its "Open
questions", item 3: ``s0`` for the leg), each on its platform's runner
(macOS on ``macos-15`` and ``macos-15-intel``, as the engine wheels), and
JAR-load jobs on the same JDKs and platforms, run where no Rust toolchain
is: in a container on Linux, and with no Rust set up on macOS.
A job's name is read with its matrix expanded.

ASSUMED until the implementer relays it: a load job's name holds ``load``
and its ``(<jdk>, <platform>`` as the S0 jobs spell them.

Gating (``contract/tests/t311_gate.py``): each test named in
``contract/tests/t311-pending.json``'s ``pending`` failed when written and
must still fail until the implementer takes it out; once it passes it
fails, naming itself.
"""

from __future__ import annotations

import importlib.util
import itertools
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

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


def _module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G = _module('t311_gate', ROOT / 'contract/tests/t311_gate.py')
#: A test held by the gate.
held = G.gated()


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
    r"""Every job the workflow runs: its name with the matrix expanded, the
    job, and the matrix values it runs with."""
    workflow = yaml.safe_load(CI.read_text(encoding='utf-8'))
    found = []
    for key, job in workflow['jobs'].items():
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


@held
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
    # No Java S1 or S2 leg yet: no java job names a database or warehouse.
    legs = {
        match.group(1)
        for name in jobs
        for match in [re.match(r'java \(\d+, [a-z0-9_-]+, ([^)]+)\)$', name)]
        if match
    }
    assert legs == {'s0'}, legs


@held
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


def test_t311_a_changed_java_test_is_one_of_the_package_s_own_tests(
    tmp_path: Path,
) -> None:
    # The handoff's own tests (owner, 2026-10-23: lint and the package's own
    # tests) run the Java library's tests when a Java test source changes:
    # Maven runs them all, once.
    sys.path.insert(0, str(ROOT / 'scripts/rust_core'))
    handoff = _module('t311_handoff', ROOT / 'scripts/rust_core/handoff.py')
    tests = [
        'java/src/test/java/com/nvidia/sdm/internal/WordingTest.java',
        'java/src/test/java/com/nvidia/sdm/internal/RunTest.java',
    ]
    for path in tests:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text('class T {}\n')
    assert handoff.own_suites(tmp_path, tests) == [handoff.affected.JAVA]
