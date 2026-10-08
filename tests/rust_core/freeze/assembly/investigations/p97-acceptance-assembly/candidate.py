# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Root-owned per-run candidate/build binding, separate from frozen BASE policy."""

from __future__ import annotations

import email
import subprocess
from pathlib import Path

from support import checked_path, require, sha, unique_json


def context(repo: Path, receipt: dict) -> dict:
    head = (
        subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo)
        .decode()
        .strip()
    )
    require(head == receipt['candidate'], 'candidate.head')
    build = receipt['native_build']
    raw = Path(build['receipt_path']).expanduser().read_bytes()
    require(sha(raw) == build['receipt_sha256'], 'candidate.build_receipt')
    retained = unique_json(raw)
    require(
        retained['build']['native_sha256'] == build['native_sha256'],
        'candidate.native_receipt',
    )
    changed = (
        subprocess.check_output(
            [
                'git',
                'diff',
                '--name-only',
                build['reviewed_source_equivalent_commit'],
                head,
                '--',
                'rust/Cargo.lock',
                'rust/Cargo.toml',
                ':(glob)rust/crates/**/src/**',
                ':(glob)rust/crates/**/Cargo.toml',
                ':(glob)rust/crates/**/build.rs',
                'rust/bindings/python/src',
                'rust/bindings/python/Cargo.toml',
                'rust/bindings/python/build.rs',
            ],
            cwd=repo,
        )
        .decode()
        .splitlines()
    )
    require(not changed, 'candidate.native_source_changed_since_build')
    return receipt


def host_routes(policy: dict, receipt: dict) -> dict:
    routes = {identity: dict(value) for identity, value in policy.items()}
    hosts = {
        identity
        for identity, value in routes.items()
        if value['kind'] == 'public_host_refusal'
    }
    require(
        set(receipt.get('host_bindings', {})) == hosts,
        'candidate.host_case_map',
    )
    for identity in hosts:
        require(
            'host_binding' not in routes[identity],
            'candidate.host_identity_frozen_as_policy',
        )
        routes[identity]['host_binding'] = receipt['host_bindings'][identity]
        require(
            routes[identity]['host_binding']['native_sha256']
            == receipt['native_build']['native_sha256'],
            'candidate.host_native',
        )
    return routes


def lane(repo: Path, major: str, receipt: dict):
    binding = receipt['lanes'][major]
    runtime = Path(binding['runtime']).expanduser().resolve()
    require(
        sha((runtime / 'bin/python').resolve().read_bytes())
        == binding['interpreter_sha256'],
        'candidate.interpreter',
    )
    site = (
        runtime
        / 'lib'
        / ('python' + '.'.join(binding['python_version'].split('.')[:2]))
        / 'site-packages'
    )
    version = binding['pandas_version']
    require(version.split('.')[0] == major, 'candidate.pandas_major')
    metadata = email.message_from_bytes(
        (site / f'pandas-{version}.dist-info/METADATA').read_bytes()
    )
    require(metadata['Version'] == version, 'candidate.pandas_metadata')
    natives = list((site / 'sdm_core').glob('_native*.so'))
    require(
        len(natives) == 1
        and sha(natives[0].read_bytes())
        == receipt['native_build']['native_sha256'],
        'candidate.installed_native',
    )
    helpers = {}
    for relative in ('sdm_core/answers.py', 'sdm_core/describe.py'):
        raw = checked_path(site, relative).read_bytes()
        require(
            raw
            == checked_path(
                repo / 'rust/bindings/python/python', relative
            ).read_bytes(),
            'candidate.installed_helper',
        )
        helpers[relative] = sha(raw)
    return runtime, {
        'python_version': binding['python_version'],
        'pandas_version': version,
        'native_sha256': receipt['native_build']['native_sha256'],
        'installed_helper_bindings': helpers,
        'native_source_relation': 'reviewed source-equivalent build, not candidate rebuild',
        'build_receipt_sha256': receipt['native_build']['receipt_sha256'],
    }
