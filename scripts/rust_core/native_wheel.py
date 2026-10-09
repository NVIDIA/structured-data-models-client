#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builds the client wheel, which carries `sdm_client._native`, with maturin.

CI's `native-wheel` job runs this inside a manylinux2014 container with
`--manylinux`, for the wheel that ships; the reference machine runs it
without, for the local build the reference comparisons use, which is never
shipped (design, "Packaging"). Either way it:

- uses the pinned maturin, and the production features unless `--test-ops`
  asks for the test-only operations;
- records the build's effective Rust target flags, from the environment, any
  Cargo configuration and every rustc invocation, and fails if one sets
  `target-cpu`, which would tie the wheel to the build machine's processor
  (decision F7);
- with `--manylinux`, reads the wheel with `auditwheel show` and fails unless
  it is manylinux2014 (`manylinux_2_17`) or older;
- writes beside the wheel `<wheel>.json`, its identity: the SHA-256 of the
  wheel and of the extension module in it, commit, features, toolchain,
  maturin version and the target flags, which a runner records as the
  identity of the build it ran.

    scripts/rust_core/native_wheel.py --out DIR [--test-ops] [--manylinux]

maturin must be on PATH at the pinned version, for instance through
`uv run --no-project --with maturin==1.15.0`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
import zipfile
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[2]
CLIENT = REPO / 'packages/structured-data-models-client'
#: The maturin every build of the wheel uses; CI installs this version.
MATURIN = '1.15.0'
#: Environment variables that carry flags to rustc.
FLAG_VARIABLES = (
    'RUSTFLAGS',
    'CARGO_ENCODED_RUSTFLAGS',
    'CARGO_BUILD_RUSTFLAGS',
    'CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS',
    'CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS',
    'CARGO_TARGET_X86_64_APPLE_DARWIN_RUSTFLAGS',
)
#: The newest glibc a manylinux2014 wheel may need.
MANYLINUX2014_GLIBC = (2, 17)


def _run(*command: str, **kwargs: object) -> str:
    return subprocess.run(
        command, check=True, capture_output=True, text=True, **kwargs
    ).stdout


def _toolchain() -> str:
    """The toolchain rust/rust-toolchain.toml pins."""
    text = (REPO / 'rust/rust-toolchain.toml').read_text()
    found = re.search(r'^channel = "([^"]+)"', text, re.MULTILINE)
    if not found:
        sys.exit('rust/rust-toolchain.toml names no channel')
    return found.group(1)


def _cargo_configs() -> dict[str, str]:
    """Every Cargo configuration file a build in rust/ reads."""
    found = {}
    homes = [
        pathlib.Path(os.environ.get('CARGO_HOME', '~/.cargo')).expanduser()
    ]
    directory = REPO / 'rust'
    while True:
        homes.append(directory / '.cargo')
        if directory == directory.parent:
            break
        directory = directory.parent
    for home in homes:
        for name in ('config', 'config.toml'):
            path = home / name
            if path.is_file():
                found[str(path)] = path.read_text()
    return found


def _rustc_flags(log: str) -> list[str]:
    """The codegen and cfg flags of every rustc invocation in a verbose
    cargo log, each once, in the order first seen."""
    flags: list[str] = []
    for line in log.splitlines():
        if 'Running `' not in line or 'rustc ' not in line:
            continue
        command = line.split('Running `', 1)[1].rsplit('`', 1)[0]
        try:
            words = shlex.split(command)
        except ValueError:
            words = command.split()
        for at, word in enumerate(words):
            if word in ('-C', '--codegen') and at + 1 < len(words):
                flag = f'-C {words[at + 1]}'
            elif word.startswith('-C') and len(word) > 2:
                flag = f'-C {word[2:]}'
            elif word.startswith('--target'):
                flag = word
            else:
                continue
            # Paths and per-crate hashes say nothing about the target.
            if flag.startswith(('-C metadata=', '-C extra-filename=')):
                continue
            if flag not in flags:
                flags.append(flag)
    return flags


def target_cpu_refusal(
    environment: dict[str, str], configs: dict[str, str], flags: list[str]
) -> str | None:
    """Why the build's target flags are refused, or `None`: a build log
    with no rustc invocation read from it, or `target-cpu` set anywhere,
    which ties the wheel to the build machine's processor (decision F7)."""
    if not any(flag.startswith('-C opt-level') for flag in flags):
        return 'no rustc invocation was read from the build log'
    sources = [*environment.values(), *configs.values(), *flags]
    if any('target-cpu' in text for text in sources):
        return (
            'a target-cpu flag is set, which ties the wheel to the build '
            f"machine's processor (decision F7): {environment} {flags}"
        )
    return None


def _native_sha256(wheel: pathlib.Path) -> str:
    """The SHA-256 of the extension module inside the wheel: unlike the
    wheel's, it is the same for every build of one commit, since maturin's
    SBOM in the wheel carries a serial number and a timestamp."""
    with zipfile.ZipFile(wheel) as archive:
        (name,) = [
            n
            for n in archive.namelist()
            if n.startswith('sdm_client/_native')
            and n.endswith(('.so', '.pyd'))
        ]
        return hashlib.sha256(archive.read(name)).hexdigest()


def _manylinux(wheel: pathlib.Path) -> str:
    """`auditwheel show`'s reading of the wheel; fails past manylinux2014."""
    shown = _run('auditwheel', 'show', str(wheel))
    # auditwheel wraps its lines; the tag is quoted after "platform tag:".
    found = re.search(r'platform tag:\s*"([^"]+)"', ' '.join(shown.split()))
    tag = found.group(1) if found else ''
    legacy = {
        'manylinux1_x86_64': (2, 5),
        'manylinux2010_x86_64': (2, 12),
        'manylinux2014_x86_64': (2, 17),
    }
    numbered = re.fullmatch(r'manylinux_(\d+)_(\d+)_x86_64', tag)
    glibc = (
        (int(numbered.group(1)), int(numbered.group(2)))
        if numbered
        else legacy.get(tag)
    )
    if glibc is None or glibc > MANYLINUX2014_GLIBC:
        sys.exit(
            f'auditwheel does not read the wheel as manylinux2014:\n{shown}'
        )
    return shown


def _build(
    arguments: argparse.Namespace,
    staged: pathlib.Path,
    environment: dict[str, str],
    configs: dict[str, str],
    version: str,
) -> dict[str, Any]:
    """Builds one wheel into `staged` and writes its identity beside it."""
    command = ['maturin', 'build', '--release', '--locked', '--verbose']
    command += ['--out', str(staged)]
    if arguments.test_ops:
        command += ['--features', 'test-ops']
    if arguments.manylinux:
        command += ['--compatibility', 'manylinux2014']
    # maturin runs cargo from the client's directory, where rustup sees no
    # toolchain file, so the pin is passed on.
    pinned = {**os.environ, 'RUSTUP_TOOLCHAIN': _toolchain()}
    build = subprocess.run(
        command,
        cwd=CLIENT,
        env=pinned,
        capture_output=True,
        text=True,
        check=False,
    )
    log = build.stdout + build.stderr
    if build.returncode != 0:
        sys.stderr.write(log)
        sys.exit(build.returncode)
    built = sorted(staged.glob('*.whl'))
    if len(built) != 1:
        sys.exit(f'expected one wheel from the build, found {built}')
    wheel = built[0]

    flags = _rustc_flags(log)
    refusal = target_cpu_refusal(environment, configs, flags)
    if refusal:
        sys.exit(refusal)

    identity = {
        'wheel': wheel.name,
        'sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
        'native_sha256': _native_sha256(wheel),
        'commit': _run('git', 'rev-parse', 'HEAD', cwd=REPO).strip(),
        # A build from a tree with changes is not the commit's build.
        'tree_clean': not _run(
            'git', 'status', '--porcelain', cwd=REPO
        ).strip(),
        'features': ['test-ops'] if arguments.test_ops else [],
        'maturin': version,
        'rustc': _run('rustc', '--version', env=pinned).strip(),
        'target_flags': {
            'environment': environment,
            'cargo_config': sorted(configs),
            'rustc': flags,
        },
    }
    if arguments.manylinux:
        identity['auditwheel'] = _manylinux(wheel)
    record = staged / f'{wheel.name}.json'
    record.write_text(json.dumps(identity, indent=2) + '\n')
    return identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True, type=pathlib.Path)
    parser.add_argument('--test-ops', action='store_true')
    parser.add_argument('--manylinux', action='store_true')
    arguments = parser.parse_args()

    version = _run('maturin', '--version').split()[-1]
    if version != MATURIN:
        sys.exit(f'maturin {version} is on PATH; the build pins {MATURIN}')

    environment = {
        name: os.environ[name] for name in FLAG_VARIABLES if name in os.environ
    }
    configs = _cargo_configs()

    out = arguments.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # maturin writes over a wheel of the same name, so it builds into a
    # fresh directory beside the destination, and the wheel and its
    # identity move in together once both are known good.
    with tempfile.TemporaryDirectory(dir=out, prefix='.staging-') as staged:
        identity = _build(
            arguments, pathlib.Path(staged), environment, configs, version
        )
        name = identity['wheel']
        os.replace(pathlib.Path(staged) / name, out / name)
        os.replace(pathlib.Path(staged) / f'{name}.json', out / f'{name}.json')
    print(json.dumps(identity, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
