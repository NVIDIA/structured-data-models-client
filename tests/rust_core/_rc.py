# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What the Phase 4 acceptance checks share: finding the Rust workspace's
crates, copying the repository, building and installing the wheels, and
comparing JSON by value and type.

The interface these rely on is ``CONTRACT.md``, beside this file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RUST = REPO / 'rust'
BINDING = RUST / 'bindings' / 'python'
SMOKE = HERE / 'smoke.py'
SDM_API_PROVENANCE = (
    REPO
    / 'packages/structured-data-models-api/src/sdm_api/generated/PROVENANCE.json'
)
GOLDEN_PAYLOADS = REPO / 'tests/golden/wire_payloads.json'
TRANSCRIPTS = REPO / 'tests/parity/transcripts'

CRATES = (
    'wire',
    'pql',
    'schema',
    'dialects',
    'sampling',
    'labels',
    'request',
    'core',
)
SMOKE_MARKER = 'SDM_CORE_SMOKE_OK'

# Linux wheels take a long time to build, so P4.linux and P4.pyarrow8 share
# them. They are kept per commit, and only when the worktree is clean, so a
# cached wheel is always the build of exactly the commit it is filed under.
# The cache is in the system's temporary directory, never in the worktree,
# which the gate needs clean.
CACHE = Path(
    os.environ.get(
        'SDM_RC_P4_CACHE',
        str(Path(tempfile.gettempdir()) / 'sdm-rust-core-p4'),
    )
).expanduser()


def run(
    cmd: list[str],
    *,
    cwd: Path = REPO,
    env: dict[str, str] | None = None,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f'{" ".join(map(str, cmd))} (in {cwd}) exited '
            f'{proc.returncode}\n--- stdout\n{proc.stdout[-8000:]}'
            f'\n--- stderr\n{proc.stderr[-8000:]}'
        )
    return proc


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- The workspace -----------------------------------------------------------


def cargo_metadata(root: Path = RUST) -> dict[str, Any]:
    assert (root / 'Cargo.toml').is_file(), f'no Cargo workspace at {root}'
    out = run(
        ['cargo', 'metadata', '--no-deps', '--format-version', '1'], cwd=root
    )
    return json.loads(out.stdout)


def _short(name: str) -> str:
    for prefix in ('sdm-', 'sdm_'):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def find_crate(name: str, meta: dict[str, Any]) -> dict[str, Any]:
    r"""The workspace package for core crate ``name`` (``CONTRACT.md``)."""
    members = set(meta['workspace_members'])
    found = [
        p
        for p in meta['packages']
        if p['id'] in members
        and _short(p['name']) == name
        and any('lib' in t['kind'] for t in p['targets'])
    ]
    assert len(found) == 1, (
        f'expected one workspace library named {name}, sdm-{name} or '
        f'sdm_{name}; found {[p["name"] for p in found]}'
    )
    return found[0]


def lib_target(package: dict[str, Any]) -> dict[str, Any]:
    return next(t for t in package['targets'] if 'lib' in t['kind'])


def binding_package(meta: dict[str, Any]) -> dict[str, Any]:
    manifest = (BINDING / 'Cargo.toml').resolve()
    members = set(meta['workspace_members'])
    found = [
        p
        for p in meta['packages']
        if p['id'] in members and Path(p['manifest_path']).resolve() == manifest
    ]
    assert found, f'{manifest} is not a member of the rust/ workspace'
    return found[0]


# --- Copies of the repository -------------------------------------------------


def repo_files(root: Path = REPO) -> list[str]:
    r"""Every file git would commit: tracked, or new and not ignored."""
    out = run(
        [
            'git',
            'ls-files',
            '--cached',
            '--others',
            '--exclude-standard',
            '-z',
        ],
        cwd=root,
    ).stdout
    return [f for f in out.split('\0') if f and (root / f).is_file()]


def copy_repo(dest: Path) -> Path:
    r"""Copy the repository as git sees it, without build output."""
    for name in repo_files():
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
    return dest


def worktree_clean() -> bool:
    return not run(['git', 'status', '--porcelain'], cwd=REPO).stdout.strip()


def head_sha() -> str:
    return run(['git', 'rev-parse', 'HEAD'], cwd=REPO).stdout.strip()


# --- The contract --------------------------------------------------------------


def sdm_api_provenance() -> dict[str, Any]:
    return json.loads(SDM_API_PROVENANCE.read_text())


def contract_clone(dest: Path) -> Path:
    r"""A local clone of the contract repository at the pinned revision.

    Returns the spec's path. The clone is the generator's input, so it can
    read the revision the way ``generate_sdm_api.py`` does.
    """
    record = sdm_api_provenance()
    source = Path(
        os.environ.get('SDM_CONTRACT_REPO', '~/code/structured-data-api')
    ).expanduser()
    assert (source / '.git').exists(), (
        f'the contract repository is not at {source}; set SDM_CONTRACT_REPO'
    )
    clone = dest / 'structured-data-api'
    run(['git', 'clone', '-q', '--shared', '--no-checkout', source, clone])
    run(
        ['git', 'checkout', '-q', record['source_revision']],
        cwd=clone,
    )
    spec = clone / record['source_path']
    assert sha256(spec) == record['source_sha256'], (
        'the contract at the pinned revision is not the bytes sdm-api records'
    )
    return spec


def load_spec(spec: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(spec.read_text())


# --- Payloads ------------------------------------------------------------------


def golden_payloads() -> list[tuple[str, Any]]:
    data = json.loads(GOLDEN_PAYLOADS.read_text())
    return [
        (f'golden:{name}[{i}]', payload)
        for name, payloads in data.items()
        for i, payload in enumerate(payloads)
    ]


def transcript_files(root: Path = TRANSCRIPTS) -> list[Path]:
    return sorted(root.glob('**/transcript.json'))


def corpus_payloads() -> list[tuple[str, Any]]:
    out = []
    for path in transcript_files():
        transcript = json.loads(path.read_text())
        rel = path.parent.relative_to(TRANSCRIPTS)
        for i, payload in enumerate(transcript['payloads']):
            out.append((f'corpus:{rel}[{i}]', payload))
    return out


def json_diff(a: Any, b: Any, path: str = '$') -> str | None:
    r"""Where two parsed JSON values differ, by value or by JSON type."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return (
                f'{path}: keys differ, missing {sorted(set(a) - set(b))}, '
                f'extra {sorted(set(b) - set(a))}'
            )
        for key in a:
            found = json_diff(a[key], b[key], f'{path}.{key}')
            if found:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f'{path}: length {len(a)} != {len(b)}'
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            found = json_diff(x, y, f'{path}[{i}]')
            if found:
                return found
        return None
    if type(a) is not type(b):
        return (
            f'{path}: {a!r} ({type(a).__name__}) != {b!r} ({type(b).__name__})'
        )
    if a != b:
        return f'{path}: {a!r} != {b!r}'
    return None


# --- Wheels --------------------------------------------------------------------


def wheel_tags(name: str) -> tuple[str, str, str, list[str]]:
    r"""Distribution, python tag, abi tag and platform tags of a wheel."""
    match = re.fullmatch(
        r'([^-]+)-[^-]+(?:-\d[^-]*)?-([^-]+)-([^-]+)-([^-]+)\.whl', name
    )
    assert match, f'not a wheel name: {name}'
    dist, py, abi, plat = match.groups()
    return dist, py, abi, plat.split('.')


def installer() -> list[str]:
    uv = shutil.which('uv')
    if uv:
        return [uv, 'pip', 'install', '--python', sys.executable]
    return [sys.executable, '-m', 'pip', 'install']


def build_local_wheel(out: Path) -> Path:
    r"""Build the binding's wheel for this machine with maturin."""
    assert (BINDING / 'pyproject.toml').is_file(), (
        f'no maturin project at {BINDING}'
    )
    maturin = shutil.which('maturin', path=os.path.dirname(sys.executable))
    maturin = maturin or shutil.which('maturin')
    assert maturin, 'maturin is not installed in this environment'
    run(
        [
            maturin,
            'build',
            '--release',
            '--interpreter',
            sys.executable,
            '--out',
            out,
        ],
        cwd=BINDING,
    )
    wheels = sorted(out.glob('*.whl'))
    assert len(wheels) == 1, f'expected one wheel, built {wheels}'
    return wheels[0]


def install_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        expected = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if name.startswith('sdm_core/') and not name.endswith('/')
        }
    assert expected, 'the binding wheel contains no sdm_core artifacts'
    loaded = {
        name: module
        for name, module in sys.modules.copy().items()
        if name == 'sdm_core' or name.startswith('sdm_core.')
    }
    roots = {path / 'sdm_core' for path in site_packages()}
    for name, module in loaded.items():
        filename = getattr(module, '__file__', None)
        origin = getattr(getattr(module, '__spec__', None), 'origin', None)
        reason = (
            f'already-loaded {name} is not equivalent to the binding wheel; '
            'use a fresh interpreter'
        )
        assert (
            filename
            and origin
            and Path(filename).resolve() == Path(origin).resolve()
        ), reason
        path = Path(filename).resolve()
        matches = [root for root in roots if path.is_relative_to(root)]
        assert len(matches) == 1, reason
        member = 'sdm_core/' + path.relative_to(matches[0]).as_posix()
        assert (
            member in expected
            and path.is_file()
            and sha256(path) == expected[member]
        ), reason
    run([*installer(), '--reinstall', '--no-deps', wheel])
    assert all(
        sys.modules.get(name) is module for name, module in loaded.items()
    ), (
        'binding module identity changed during installation; use a fresh interpreter'
    )
    for member, checksum in expected.items():
        relative = Path(member).relative_to('sdm_core')
        matches = [
            root / relative for root in roots if (root / relative).is_file()
        ]
        assert len(matches) == 1 and sha256(matches[0]) == checksum, (
            f'installed binding artifact {member} does not match the wheel; '
            'use a fresh interpreter'
        )


def install_local_wheel() -> Path:
    r"""Build this checkout's wheel and install it into this environment."""
    out = Path(tempfile.mkdtemp(prefix='sdm-core-wheel-'))
    wheel = build_local_wheel(out)
    install_wheel(wheel)
    return wheel


def site_packages() -> list[Path]:
    paths = sysconfig.get_paths()
    return [Path(paths['purelib']).resolve(), Path(paths['platlib']).resolve()]


def smoke_lines(output: str) -> list[dict[str, str]]:
    r"""The success lines ``smoke.py`` printed, as key-value pairs."""
    found = []
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith(SMOKE_MARKER + ' '):
            continue
        fields = dict(
            item.split('=', 1)
            for item in line[len(SMOKE_MARKER) + 1 :].split()
            if '=' in item
        )
        found.append(fields)
    return found


def _linux_build(dest: Path) -> tuple[list[Path], str]:
    with tempfile.TemporaryDirectory(prefix='sdm-core-linux-') as tmp:
        project = copy_repo(Path(tmp) / 'repo')
        out = dest / 'wheels'
        out.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env['CIBW_ARCHS_LINUX'] = 'x86_64 aarch64'
        cibw = shutil.which(
            'cibuildwheel', path=os.path.dirname(sys.executable)
        )
        cibw = cibw or shutil.which('cibuildwheel')
        assert cibw, 'cibuildwheel is not installed in this environment'
        proc = run(
            [
                cibw,
                '--platform',
                'linux',
                'rust/bindings/python',
                '--output-dir',
                out,
            ],
            cwd=project,
            env=env,
            check=False,
        )
        log = f'{proc.stdout}\n{proc.stderr}\nexit={proc.returncode}\n'
        (dest / 'cibuildwheel.log').write_text(log)
        assert proc.returncode == 0, (
            f'cibuildwheel failed; log in {dest / "cibuildwheel.log"}\n'
            f'{log[-6000:]}'
        )
        return sorted(out.glob('*.whl')), log


def linux_wheels() -> tuple[list[Path], str]:
    r"""The manylinux wheels of this checkout, and cibuildwheel's output.

    Built once per clean commit and kept under ``CACHE``; a dirty worktree
    is built afresh every time, into a directory nothing else reads.
    """
    if worktree_clean():
        dest = CACHE / head_sha()
        done = dest / 'DONE'
        if done.exists():
            return (
                sorted((dest / 'wheels').glob('*.whl')),
                (dest / 'cibuildwheel.log').read_text(),
            )
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        wheels, log = _linux_build(dest)
        done.write_text('built\n')
        return wheels, log
    dest = Path(tempfile.mkdtemp(prefix='sdm-core-linux-dirty-'))
    return _linux_build(dest)


def wheel_by_arch(wheels: list[Path]) -> dict[str, Path]:
    r"""The one manylinux ``sdm_core`` wheel of each architecture."""
    found: dict[str, list[Path]] = {}
    for wheel in wheels:
        dist, _, _, plats = wheel_tags(wheel.name)
        if dist != 'sdm_core' or not all(
            p.startswith('manylinux') for p in plats
        ):
            continue
        for arch in ('x86_64', 'aarch64'):
            if any(p.endswith('_' + arch) for p in plats):
                found.setdefault(arch, []).append(wheel)
    for arch, items in found.items():
        assert len(items) == 1, f'{arch}: expected one wheel, got {items}'
    return {arch: items[0] for arch, items in found.items()}
