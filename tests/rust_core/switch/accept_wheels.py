# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P9.wheels: ``sdm-core`` wheels build for the four platforms, and a clean
venv on each installs ``sdm-client`` with them and passes the smoke tests.

- Linux x86_64 and aarch64: cibuildwheel in Docker (Phase 4's
  ``_rc.linux_wheels``), one abi3 ``manylinux_2_28`` wheel each. Each wheel
  passes the binding's tests in its build container, except that where the
  x86_64 container runs under Rosetta (Docker Desktop on Apple silicon) the
  one test asking for a 4 PB allocation is skipped (``_rosetta.py``): Rosetta
  ends the process instead of failing the allocation. The aarch64 container
  runs every test, and the full x86_64 suite runs on real x86 in Phase 9.5's
  the Databricks validation workspace jobs.
- macOS arm64 and x86_64: maturin, ``--target aarch64-apple-darwin`` and
  ``x86_64-apple-darwin``, one abi3 ``macosx`` wheel each.
- The first-party distributions are built as wheels beside them, and
  ``sdm-client``'s requires ``sdm-core`` at its own version, and ``pyarrow``,
  neither as an extra (DV29, DV30).
- On each platform a fresh Python 3.12 environment, with nothing of this
  checkout on its path, installs ``sdm-client`` from those wheels (third
  parties from the index) and runs ``tests/rust_core/smoke.py`` and
  ``client_smoke.py``: on this Mac natively and under Rosetta, and in
  ``python:3.12-slim`` containers for each Linux architecture.

The wheels are built once per clean commit and kept in the system's
temporary directory.
"""

from __future__ import annotations

import contextlib
import email
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest
from _switch import (
    HERE,
    REPO,
    RUST_CORE,
    add_paths,
    head_sha,
    run,
    worktree_clean,
)

DISTRIBUTIONS = (
    'structured-data-models-api',
    'structured-data-models-pql',
    'structured-data-models-connectors',
    'structured-data-models-client',
)
MAC_TARGETS = {'arm64': 'aarch64-apple-darwin', 'x86_64': 'x86_64-apple-darwin'}
LINUX = {'x86_64': 'linux/amd64', 'aarch64': 'linux/arm64'}
BINDING = REPO / 'rust' / 'bindings' / 'python'


def _tags(name: str) -> tuple[str, str, str, list[str]]:
    parts = name[: -len('.whl')].split('-')
    return parts[0], parts[-3], parts[-2], parts[-1].split('.')


#: The binding's test runs in cibuildwheel's Linux containers, where the
#: checkout is ``/project``, load ``_rosetta.py``, and list skips with reasons.
ROSETTA_SKIP = 'P9.wheels: Rosetta ends the process on a 4 PB allocation'
TEST_ENVIRONMENT = (
    'PYTHONPATH=/project/tests/rust_core/switch '
    'PYTEST_ADDOPTS="-p _rosetta -rs"'
)


@contextlib.contextmanager
def _rosetta_plugin():
    name = 'CIBW_TEST_ENVIRONMENT_LINUX'
    before = os.environ.get(name)
    os.environ[name] = TEST_ENVIRONMENT
    try:
        yield
    finally:
        if before is None:
            del os.environ[name]
        else:
            os.environ[name] = before


def _linux_wheels() -> tuple[list[Path], str]:
    add_paths()
    from _rc import linux_wheels

    with _rosetta_plugin():
        return linux_wheels()


def _build(dest: Path) -> None:
    wheels, _ = _linux_wheels()
    for wheel in wheels:
        shutil.copy2(wheel, dest)
    maturin = shutil.which('maturin', path=os.path.dirname(sys.executable))
    assert maturin, 'maturin is not installed in this environment'
    for target in MAC_TARGETS.values():
        proc = run(
            [
                maturin,
                'build',
                '--release',
                '--target',
                target,
                '--interpreter',
                sys.executable,
                '--out',
                dest,
            ],
            cwd=BINDING,
        )
        assert proc.returncode == 0, (
            f'maturin could not build for {target}:\n{proc.stderr[-4000:]}'
        )
    uv = shutil.which('uv')
    assert uv, 'uv is needed to build the distributions'
    for name in DISTRIBUTIONS:
        proc = run(
            [
                uv,
                'build',
                '--wheel',
                '--out-dir',
                dest,
                REPO / 'packages' / name,
            ],
            cwd=REPO,
        )
        assert proc.returncode == 0, f'{name}:\n{proc.stderr[-4000:]}'


@pytest.fixture(scope='module')
def wheelhouse() -> Path:
    if worktree_clean():
        root = (
            Path(tempfile.gettempdir()) / 'sdm-rust-core-p9-wheels' / head_sha()
        )
        if (root / 'DONE').exists():
            return root
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        _build(root)
        (root / 'DONE').write_text('built\n')
        return root
    root = Path(tempfile.mkdtemp(prefix='sdm-p9-wheels-dirty-'))
    _build(root)
    return root


def _core_wheels(house: Path) -> list[Path]:
    return sorted(house.glob('sdm_core-*.whl'))


def test_linux_wheels_for_both_architectures(wheelhouse):
    found = {}
    for wheel in _core_wheels(wheelhouse):
        _, py, abi, plats = _tags(wheel.name)
        for arch in LINUX:
            if any(
                p.startswith('manylinux') and p.endswith('_' + arch)
                for p in plats
            ):
                assert (py, abi) == ('cp310', 'abi3'), wheel.name
                for plat in plats:
                    match = re.fullmatch(rf'manylinux_2_(\d+)_{arch}', plat)
                    assert match is None or int(match.group(1)) <= 28, (
                        wheel.name
                    )
                found.setdefault(arch, []).append(wheel.name)
    assert sorted(found) == sorted(LINUX), found
    assert all(len(v) == 1 for v in found.values()), found


def test_the_aarch64_wheel_ran_every_binding_test(wheelhouse):
    r"""The Rosetta skip stays in the x86_64 container: the aarch64 one ran
    the binding's tests twice (newest pyarrow, then pyarrow 8), all passing
    and the hostile allocation not skipped."""
    _, log = _linux_wheels()
    sections = re.split(
        r'^Building (cp\d+-manylinux_\w+) wheel$', log, flags=re.M
    )
    by_build = dict(zip(sections[1::2], sections[2::2], strict=True))
    aarch64 = [text for build, text in by_build.items() if 'aarch64' in build]
    assert len(aarch64) == 1, sorted(by_build)
    assert ROSETTA_SKIP not in aarch64[0]
    runs = re.findall(r'^=*\s*(\d+ passed[^\n]*)$', aarch64[0], flags=re.M)
    assert len(runs) == 2 and not any('failed' in r for r in runs), runs


def test_macos_wheels_for_both_architectures(wheelhouse):
    found = {}
    for wheel in _core_wheels(wheelhouse):
        _, py, abi, plats = _tags(wheel.name)
        for arch in MAC_TARGETS:
            if any(
                p.startswith('macosx_') and p.endswith('_' + arch)
                for p in plats
            ):
                assert (py, abi) == ('cp310', 'abi3'), wheel.name
                found.setdefault(arch, []).append(wheel.name)
    assert sorted(found) == sorted(MAC_TARGETS), found
    assert all(len(v) == 1 for v in found.values()), found


def _metadata(wheel: Path) -> email.message.Message:
    with zipfile.ZipFile(wheel) as archive:
        name = next(
            n for n in archive.namelist() if n.endswith('.dist-info/METADATA')
        )
        return email.message_from_bytes(archive.read(name))


def test_sdm_client_requires_the_core_at_its_version_and_pyarrow(wheelhouse):
    clients = sorted(wheelhouse.glob('structured_data_models_client-*.whl'))
    assert len(clients) == 1, clients
    meta = _metadata(clients[0])
    version = meta['Version']
    base = [
        r for r in meta.get_all('Requires-Dist') or [] if 'extra ==' not in r
    ]
    core = [r for r in base if re.match(r'sdm[-_]core\b', r)]
    assert core and re.fullmatch(
        rf'sdm[-_]core\s*\(?==\s*{re.escape(version)}\)?',
        core[0].split(';')[0].strip(),
    ), f'sdm-client {version} does not pin sdm-core=={version}: {base}'
    assert any(re.match(r'pyarrow\b', r) for r in base), base
    for wheel in _core_wheels(wheelhouse):
        assert _metadata(wheel)['Version'] == version, (wheel.name, version)


def _smoke_commands(python: str) -> list[list[str]]:
    return [
        [python, '-I', str(RUST_CORE / 'smoke.py')],
        [python, '-I', str(HERE / 'client_smoke.py')],
    ]


@pytest.mark.parametrize('arch', sorted(MAC_TARGETS))
def test_a_clean_mac_venv_installs_sdm_client_and_passes_the_smoke_tests(
    wheelhouse, arch
):
    uv = shutil.which('uv')
    with tempfile.TemporaryDirectory(prefix=f'sdm-p9-clean-{arch}-') as tmp:
        venv = Path(tmp) / 'venv'
        made = run(
            [
                uv,
                'venv',
                '--python',
                f'cpython-3.12-macos-{arch.replace("arm64", "aarch64")}-none',
                venv,
            ]
        )
        assert made.returncode == 0, made.stderr
        python = str(venv / 'bin' / 'python')
        machine = run(
            [python, '-c', 'import platform; print(platform.machine())']
        )
        assert machine.stdout.strip() == arch, machine.stdout
        installed = run(
            [
                uv,
                'pip',
                'install',
                '--python',
                python,
                '--find-links',
                wheelhouse,
                'structured-data-models-client',
            ],
            cwd=tmp,
        )
        assert installed.returncode == 0, installed.stderr[-4000:]
        assert 'sdm-core' in installed.stderr, installed.stderr[-2000:]
        for command in _smoke_commands(python):
            proc = run(
                command, cwd=tmp, env={'PATH': os.environ['PATH'], 'HOME': tmp}
            )
            assert proc.returncode == 0 and '_SMOKE_OK' in proc.stdout, (
                f'{command[-1]} on macOS {arch}:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}'
            )


@pytest.mark.parametrize('arch', sorted(LINUX))
def test_a_clean_linux_container_installs_sdm_client_and_passes_the_smoke_tests(
    wheelhouse, arch
):
    script = (
        'set -e; python -m pip install -q --find-links /wheels structured-data-models-client; '
        'python -m pip show sdm-core >/dev/null; cd /tmp; '
        'test "$(python -c \'import platform; print(platform.machine())\')" = '
        f'{arch}; python -I /rc/smoke.py; python -I /switch/client_smoke.py'
    )
    proc = run(
        [
            'docker',
            'run',
            '--rm',
            '--platform',
            LINUX[arch],
            '-v',
            f'{wheelhouse}:/wheels:ro',
            '-v',
            f'{RUST_CORE}:/rc:ro',
            '-v',
            f'{HERE}:/switch:ro',
            'python:3.12-slim',
            'sh',
            '-c',
            script,
        ],
        timeout=3600,
    )
    assert proc.returncode == 0, f'{proc.stdout[-3000:]}\n{proc.stderr[-4000:]}'
    assert 'SDM_CORE_SMOKE_OK' in proc.stdout, proc.stdout[-2000:]
    assert 'SDM_CLIENT_SMOKE_OK' in proc.stdout, proc.stdout[-2000:]
