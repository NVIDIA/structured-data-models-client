# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P4.ci: ``.gitlab-ci.yml`` runs the Rust workspace's five jobs, with the
commands the Phase 4 brief names.

- ``rust-lint``: ``cargo fmt --check`` and
  ``cargo clippy --workspace --all-targets -- -D warnings``;
- ``rust-test``: ``cargo test --workspace``;
- ``rust-deny``: ``cargo deny check licenses bans``;
- ``rust-wire``: the wire generator's check mode, against the contract
  fetched at the revision the provenance record pins;
- ``rust-wheels``: cibuildwheel for the binding, on x86_64 and aarch64.

The file is parsed, ``extends`` resolved, and each job's commands read;
nothing is run.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
JOBS = ('rust-lint', 'rust-test', 'rust-deny', 'rust-wire', 'rust-wheels')
BLOCKING = ('rust-lint', 'rust-test', 'rust-deny', 'rust-wire')


@pytest.fixture(scope='module')
def config() -> dict[str, Any]:
    return yaml.safe_load((REPO / '.gitlab-ci.yml').read_text())


def resolve(config: dict[str, Any], name: str) -> dict[str, Any]:
    r"""A job with its ``extends`` applied, as GitLab merges them."""
    job = config[name]
    parents = job.get('extends', [])
    if isinstance(parents, str):
        parents = [parents]
    merged: dict[str, Any] = {}
    for parent in parents:
        merged = _merge(merged, resolve(config, parent))
    return _merge(merged, {k: v for k, v in job.items() if k != 'extends'})


def _merge(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _flatten(items: Any) -> list[str]:
    if items is None:
        return []
    if isinstance(items, str):
        return [items]
    out = []
    for item in items:
        out.extend(_flatten(item))
    return out


def commands(job: dict[str, Any]) -> list[str]:
    return _flatten(job.get('before_script')) + _flatten(job.get('script'))


def words(command: str) -> list[str]:
    try:
        return shlex.split(command, comments=True)
    except ValueError:
        return command.split()


def runs(job: dict[str, Any], program: list[str]) -> list[list[str]]:
    r"""Each command in the job that runs ``program``, as its words.

    A command can be one of several joined by ``&&`` or ``;``, or a line of a
    multi-line block.
    """
    found = []
    for command in commands(job):
        for line in command.splitlines():
            for part in re.split(r'&&|\|\||;', line):
                tokens = words(part)
                for i in range(len(tokens) - len(program) + 1):
                    if tokens[i : i + len(program)] == program:
                        found.append(tokens[i:])
                        break
    return found


def test_every_rust_job_is_defined(config):
    stages = config.get('stages', ['build', 'test', 'deploy'])
    for name in JOBS:
        assert name in config, f'.gitlab-ci.yml has no {name} job'
        job = resolve(config, name)
        assert job.get('stage', 'test') in stages, (
            f'{name} is in an undeclared stage'
        )
        assert commands(job), f'{name} runs nothing'


@pytest.mark.parametrize('name', BLOCKING)
def test_the_checking_jobs_block(config, name):
    job = resolve(config, name)
    assert job.get('allow_failure') is not True, f'{name} may fail'


def test_rust_lint_checks_formatting_and_clippy_warnings(config):
    job = resolve(config, 'rust-lint')
    fmt = runs(job, ['cargo', 'fmt'])
    assert any('--check' in cmd for cmd in fmt), (
        'rust-lint does not run cargo fmt --check'
    )
    clippy = runs(job, ['cargo', 'clippy'])
    good = []
    for cmd in clippy:
        if '--' not in cmd:
            continue
        cargo_args = cmd[: cmd.index('--')]
        lint_args = ' '.join(cmd[cmd.index('--') + 1 :])
        if (
            '--workspace' in cargo_args
            and '--all-targets' in cargo_args
            and re.search(r'-D\s*warnings', lint_args)
        ):
            good.append(cmd)
    assert good, (
        'rust-lint does not run cargo clippy --workspace --all-targets '
        f'-- -D warnings; it runs {clippy}'
    )


def test_rust_test_runs_the_whole_workspace(config):
    job = resolve(config, 'rust-test')
    tests = runs(job, ['cargo', 'test'])
    assert any('--workspace' in cmd for cmd in tests), (
        f'rust-test does not run cargo test --workspace; it runs {tests}'
    )


def test_rust_deny_audits_licences_and_bans(config):
    job = resolve(config, 'rust-deny')
    deny = runs(job, ['cargo', 'deny'])
    assert any(
        'check' in cmd and 'licenses' in cmd and 'bans' in cmd for cmd in deny
    ), f'rust-deny does not run cargo deny check licenses bans; it runs {deny}'


def test_rust_wire_requires_a_configured_contract_and_stays_blocking(config):
    job = resolve(config, 'rust-wire')
    assert (
        job['rules']
        == resolve(config, 'contract')['rules']
        == [{'if': '$CONTRACT_PROJECT'}]
    )
    assert job.get('allow_failure', False) is False


def test_rust_wire_checks_the_generator_against_the_pinned_contract(config):
    job = resolve(config, 'rust-wire')
    text = '\n'.join(commands(job))
    assert 'CONTRACT_PROJECT' in text, 'rust-wire does not fetch the contract'
    assert re.search(r'PROVENANCE\.json', text), (
        'rust-wire does not read the pinned revision from a provenance record'
    )
    assert runs(job, ['git', 'clone']) and (
        runs(job, ['git', 'checkout']) or re.search(r'checkout', text)
    ), 'rust-wire does not check the contract out at a revision'
    generators = [
        cmd
        for cmd in runs(job, ['python']) + runs(job, ['python3'])
        if any(
            Path(word).name in ('generate_wire_rs.py', 'generate_sdm_api.py')
            for word in cmd
        )
    ]
    assert any('--check' in cmd and '--spec' in cmd for cmd in generators), (
        'rust-wire does not run the wire generator in check mode on a spec'
    )


def test_rust_wheels_builds_both_architectures_with_cibuildwheel(config):
    job = resolve(config, 'rust-wheels')
    builds = runs(job, ['cibuildwheel'])
    assert any(
        any(w.rstrip('/') == 'rust/bindings/python' for w in cmd)
        for cmd in builds
    ), f'rust-wheels does not build rust/bindings/python: {builds}'
    assert any(
        '--platform' in cmd and 'linux' in cmd for cmd in builds
    ) or 'linux' in str(job.get('variables', {})), (
        'rust-wheels does not build for Linux'
    )
    everything = yaml.safe_dump(job)
    for arch in ('x86_64', 'aarch64'):
        assert arch in everything, f'rust-wheels does not build {arch}'
