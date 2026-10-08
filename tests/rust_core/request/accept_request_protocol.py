# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P85.protocol: the run loop's types live in a small crate of their own,
so ``sdm-core`` can drive the request, sampling and schema crates without a
cycle, and ``Diagnostic`` has the brief's second shape (tasks 1 and 2;
``CONTRACT.md``, "The protocol crate").

Phase 4's checks of the run loop's shape and its hosts
(``tests/rust_core/accept_loop.py``) run beside these.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from _request import RUST, rc_module

#: The crates that report problems, which all need ``Diagnostic``.
USERS = ('core', 'request', 'sampling', 'schema', 'dialects')
#: What ``sdm-core`` drives.
DRIVEN = ('request', 'sampling', 'schema')

ALONE = r"""
#![allow(dead_code, unused_imports, unused_variables, clippy::all)]

use proto::{Diagnostic, Outcome, Query, Run, Step};

fn diagnostic(d: &Diagnostic) {
    let _: &Option<String> = &d.code;
    let _: &String = &d.message;
    let _ = &d.class;
    let _ = &d.category;
    let _ = &d.details;
}

fn run<T>(_: &mut dyn Run<T>) {}

fn step<T>(step: Step<T>) {
    match step {
        Step::RunQueries(queries) => {
            let _: Vec<Query> = queries;
        }
        Step::Done(_) => {}
        _ => {}
    }
}

fn outcome(outcome: Outcome) {
    match outcome {
        Outcome::Error { message, .. } => {
            let _: String = message;
        }
        _ => {}
    }
}
"""

SAME = r"""
#![allow(dead_code, unused_imports, clippy::all)]

fn diagnostic(d: sdm_core::Diagnostic) -> proto::Diagnostic {
    d
}

fn outcome(o: sdm_core::Outcome) -> proto::Outcome {
    o
}

fn query(q: sdm_core::Query) -> proto::Query {
    q
}

fn step<T>(s: sdm_core::Step<T>) -> proto::Step<T> {
    s
}

fn run<T>(r: &mut dyn sdm_core::Run<T>) -> &mut dyn proto::Run<T> {
    r
}
"""


@pytest.fixture(scope='module')
def meta() -> dict[str, Any]:
    return rc_module().cargo_metadata()


def _members(meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ids = set(meta['workspace_members'])
    return {p['name']: p for p in meta['packages'] if p['id'] in ids}


def _uses(meta: dict[str, Any]) -> dict[str, set[str]]:
    r"""Each workspace package's normal dependencies on other members."""
    members = _members(meta)
    return {
        name: {
            d['name']
            for d in package['dependencies']
            if d.get('kind') in (None, 'normal') and d['name'] in members
        }
        for name, package in members.items()
    }


def _closure(uses: dict[str, set[str]], name: str) -> set[str]:
    seen: set[str] = set()
    todo = [name]
    while todo:
        for dep in uses.get(todo.pop(), set()):
            if dep not in seen:
                seen.add(dep)
                todo.append(dep)
    return seen


def _crate(meta: dict[str, Any], short: str) -> str:
    return rc_module().find_crate(short, meta)['name']


def _probe(
    tmp: Path, name: str, deps: dict[str, tuple[str, Path]], source: str
) -> tuple[bool, str]:
    rc = rc_module()
    probe = tmp / name
    (probe / 'src').mkdir(parents=True)
    lines = '\n'.join(
        f'{alias} = {{ package = "{package}", path = "{path}" }}'
        for alias, (package, path) in deps.items()
    )
    (probe / 'Cargo.toml').write_text(
        f'[package]\nname = "{name}"\nversion = "0.0.0"\n'
        f'edition = "2024"\npublish = false\n\n[workspace]\n\n'
        f'[dependencies]\n{lines}\n'
    )
    (probe / 'src' / 'lib.rs').write_text(source)
    for file in ('Cargo.lock', 'rust-toolchain.toml'):
        if (RUST / file).exists():
            shutil.copy(RUST / file, probe / file)
    env = dict(os.environ)
    env['CARGO_TARGET_DIR'] = str(rc.CACHE / 'target-p85-protocol')
    proc = rc.run(
        ['cargo', 'check', '--quiet'], cwd=probe, env=env, check=False
    )
    return proc.returncode == 0, proc.stderr[-3000:]


def _protocol(meta: dict[str, Any], tmp: Path) -> tuple[str, dict[str, str]]:
    r"""The protocol crate: a member other than the run loop's, with no
    member dependency, that ``sdm-core`` uses and that the probe compiles
    against. Its name, and why each other candidate failed."""
    members = _members(meta)
    uses = _uses(meta)
    core = _crate(meta, 'core')
    failed: dict[str, str] = {}
    for name in sorted(uses.get(core, set())):
        package = members[name]
        if uses[name] or not any(
            'lib' in t['kind'] for t in package['targets']
        ):
            continue
        path = Path(package['manifest_path']).parent
        ok, why = _probe(
            tmp, f'probe-alone-{name}', {'proto': (name, path)}, ALONE
        )
        if ok:
            return name, failed
        failed[name] = why
    return '', failed


def test_the_protocol_types_are_a_crate_of_their_own(
    meta: dict[str, Any], tmp_path: Path
) -> None:
    name, failed = _protocol(meta, tmp_path)
    assert name, (
        'no member of the workspace without member dependencies, used by '
        'sdm-core, exports Run, Step, Query, Outcome and a Diagnostic with '
        'class, category, code: Option<String>, message: String and details:\n'
        + '\n'.join(f'--- {k}\n{v}' for k, v in failed.items())
    )
    uses = _uses(meta)
    lacking = [
        short
        for short in USERS
        if name not in _closure(uses, _crate(meta, short))
    ]
    assert not lacking, f'{lacking} do not depend on {name}'


def test_core_drives_the_crates_and_nothing_drives_core(
    meta: dict[str, Any],
) -> None:
    uses = _uses(meta)
    core = _crate(meta, 'core')
    missing = [s for s in DRIVEN if _crate(meta, s) not in uses[core]]
    assert not missing, f'{core} does not depend on {missing}'
    binding = rc_module().binding_package(meta)['name']
    above = sorted(
        name
        for name in uses
        if name not in (core, binding) and core in _closure(uses, name)
    )
    assert not above, f'{above} depend on {core}, which drives them'


def test_core_reexports_the_protocol_types(
    meta: dict[str, Any], tmp_path: Path
) -> None:
    name, _ = _protocol(meta, tmp_path / 'find')
    assert name, 'no protocol crate'
    members = _members(meta)
    core = _crate(meta, 'core')
    ok, why = _probe(
        tmp_path,
        'probe-same',
        {
            'proto': (name, Path(members[name]['manifest_path']).parent),
            'sdm_core': (core, Path(members[core]['manifest_path']).parent),
        },
        SAME,
    )
    assert ok, f"sdm_core does not re-export {name}'s own types:\n{why}"
