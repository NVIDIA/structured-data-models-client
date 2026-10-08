# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P4.workspace: the Rust workspace is laid out as the brief says.

A workspace under ``rust/`` with the eight core crates, the six not ported
yet still empty, a PyO3 ``cdylib`` binding built by maturin as ``sdm-core``,
a toolchain pinned to a release, and a committed lockfile.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tomllib
from _rc import BINDING, CRATES, REPO, RUST, cargo_metadata, find_crate, run

EMPTY = ('pql', 'schema', 'dialects', 'sampling', 'labels', 'request')


@pytest.fixture(scope='module')
def meta():
    return cargo_metadata()


@pytest.mark.parametrize('name', CRATES)
def test_each_core_crate_is_a_workspace_library(meta, name):
    package = find_crate(name, meta)
    assert (
        Path(package['manifest_path']).resolve().is_relative_to(RUST.resolve())
    )


@pytest.mark.parametrize('name', EMPTY)
def test_crates_not_ported_yet_are_empty(meta, name):
    package = find_crate(name, meta)
    assert not package['dependencies'], (
        f'{package["name"]} is meant to be empty, but depends on '
        f'{[d["name"] for d in package["dependencies"]]}'
    )
    root = Path(package['manifest_path']).parent
    sources = sorted(
        p
        for p in root.rglob('*.rs')
        if 'target' not in p.relative_to(root).parts
    )
    lib = Path(
        next(t for t in package['targets'] if 'lib' in t['kind'])['src_path']
    )
    assert sources == [lib], f'{package["name"]} holds {sources}'
    docs = 0
    for line in lib.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith('//!'):
            docs += 1
            assert stripped[3:].strip(), 'the module doc line is empty'
        elif stripped and not stripped.startswith('//'):
            pytest.fail(f'{lib} holds code: {line!r}')
    assert docs == 1, f'{lib} has {docs} module doc lines, not one'


def test_the_binding_is_a_pyo3_cdylib_built_by_maturin(meta):
    from _rc import binding_package

    package = binding_package(meta)
    lib = next(
        (
            t
            for t in package['targets']
            if 'cdylib' in t.get('crate_types', t['kind'])
        ),
        None,
    )
    assert lib is not None, 'the binding builds no cdylib'
    assert any(d['name'] == 'pyo3' for d in package['dependencies']), (
        'the binding does not depend on pyo3'
    )
    project = tomllib.loads((BINDING / 'pyproject.toml').read_text())
    assert project['build-system']['build-backend'] == 'maturin'
    assert project['project']['name'] == 'sdm-core'


def test_the_toolchain_is_pinned_to_a_stable_release():
    config = tomllib.loads((RUST / 'rust-toolchain.toml').read_text())
    channel = str(config['toolchain']['channel'])
    assert re.fullmatch(r'1\.\d+(\.\d+)?', channel), (
        f'channel {channel!r} is not a pinned stable release'
    )


def test_the_lockfile_is_committed():
    tracked = run(
        ['git', 'ls-files', '--error-unmatch', 'rust/Cargo.lock'], check=False
    )
    assert tracked.returncode == 0, 'rust/Cargo.lock is not committed'


def test_build_output_is_ignored():
    probe = run(
        ['git', 'check-ignore', '-q', 'rust/target/probe'],
        cwd=REPO,
        check=False,
    )
    assert probe.returncode == 0, 'rust/target/ is not ignored by git'
