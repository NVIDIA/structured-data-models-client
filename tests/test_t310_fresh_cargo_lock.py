# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh dependency testing must leave locked source acceptance honest."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
STEP = 'Preserve fresh Cargo evidence and restore the committed lock'
pytestmark = pytest.mark.tooling


def command(where: Path, env: dict[str, str], *argv: str) -> str:
    result = subprocess.run(
        argv, cwd=where, env=env, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def steps() -> list[dict]:
    return yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())[
        'jobs'
    ]['rust-test']['steps']


@pytest.fixture
def fresh_checkout(tmp_path: Path) -> tuple[Path, dict[str, str], bytes, bytes]:
    """Build both revisions of a real tiny local Git dependency with Cargo."""
    env = {
        **os.environ,
        'GIT_CONFIG_GLOBAL': os.devnull,
        'GIT_CONFIG_NOSYSTEM': '1',
        'GIT_AUTHOR_NAME': 'Fixture',
        'GIT_AUTHOR_EMAIL': 'fixture@example.invalid',
        'GIT_COMMITTER_NAME': 'Fixture',
        'GIT_COMMITTER_EMAIL': 'fixture@example.invalid',
        'GIT_ALLOW_PROTOCOL': 'file',
        'CARGO_NET_GIT_FETCH_WITH_CLI': 'true',
        'CARGO_HOME': str(tmp_path / 'cargo-home'),
        'CARGO_TARGET_DIR': str(tmp_path / 'target'),
        'CARGO_BUILD_JOBS': '2',
        'RUSTUP_TOOLCHAIN': '1.97.0',
        'RUNNER_TEMP': str(tmp_path / 'runner-temp'),
    }
    dep = tmp_path / 'dependency'
    (dep / 'src').mkdir(parents=True)
    (dep / 'Cargo.toml').write_text(
        '[package]\nname="witness_dep"\nversion="0.1.0"\nedition="2024"\n'
    )
    (dep / 'src/lib.rs').write_text(
        'pub fn value() -> &\'static str { "locked" }\n'
    )
    command(dep, env, 'git', 'init', '-q')
    command(dep, env, 'git', 'add', '-A')
    command(dep, env, 'git', 'commit', '-qm', 'locked dependency')
    repo = tmp_path / 'repo'
    rust = repo / 'rust'
    (rust / 'src').mkdir(parents=True)
    (rust / 'Cargo.toml').write_text(
        '[package]\nname="source_acceptance"\nversion="0.1.0"\n'
        'edition="2024"\n[dependencies]\n'
        f'witness_dep={{git="{dep.as_uri()}"}}\n'
    )
    (rust / 'src/main.rs').write_text(
        'fn main() { println!("{}", witness_dep::value()); }\n'
    )
    manifest = str(rust / 'Cargo.toml')
    command(
        repo, env, 'cargo', 'generate-lockfile', '--manifest-path', manifest
    )
    assert (
        command(
            repo,
            env,
            'cargo',
            'run',
            '-q',
            '--manifest-path',
            manifest,
            '--locked',
        )
        == 'locked'
    )
    command(repo, env, 'git', 'init', '-q')
    command(repo, env, 'git', 'add', '-A')
    command(repo, env, 'git', 'commit', '-qm', 'committed source and lock')
    committed = (rust / 'Cargo.lock').read_bytes()
    (dep / 'src/lib.rs').write_text(
        'pub fn value() -> &\'static str { "fresh" }\n'
    )
    command(dep, env, 'git', 'commit', '-qam', 'fresh dependency')
    command(repo, env, 'cargo', 'update', '--manifest-path', manifest)
    assert (
        command(
            repo,
            env,
            'cargo',
            'run',
            '-q',
            '--manifest-path',
            manifest,
            '--locked',
        )
        == 'fresh'
    )
    updated = (rust / 'Cargo.lock').read_bytes()
    assert updated != committed
    return repo, env, committed, updated


def restore_step(
    repo: Path, env: dict[str, str]
) -> subprocess.CompletedProcess:
    step = next(step for step in steps() if step.get('name') == STEP)
    assert step['working-directory'] == '.'
    return subprocess.run(
        ['bash', '-e', '-o', 'pipefail', '-c', step['run']],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )


def test_fresh_cargo_evidence_survives_and_locked_cargo_rebuilds_the_old_dependency(
    fresh_checkout: tuple[Path, dict[str, str], bytes, bytes],
) -> None:
    repo, env, committed, updated = fresh_checkout
    before = command(repo, env, 'git', 'rev-parse', 'HEAD')
    result = restore_step(repo, env)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = Path(env['RUNNER_TEMP']) / 'fresh-cargo-lock'
    assert (evidence / 'fresh-Cargo.lock').read_bytes() == updated
    assert (evidence / 'committed-Cargo.lock').read_bytes() == committed
    assert before in (evidence / 'source.txt').read_text()
    assert 'rust/Cargo.lock' in (evidence / 'status-before.txt').read_text()
    assert 'diff --git' in (evidence / 'diff-before.patch').read_text()
    assert (repo / 'rust/Cargo.lock').read_bytes() == committed
    assert command(repo, env, 'git', 'status', '--porcelain') == ''
    assert command(repo, env, 'git', 'rev-parse', 'HEAD') == before
    assert command(
        repo, env, 'cargo', 'run', '-q', '--manifest-path',
        str(repo / 'rust/Cargo.toml'), '--locked',
    ) == 'locked'  # fmt: skip


@pytest.mark.parametrize(
    'mutation', ['tracked source', 'staged lock', 'lock mode']
)
def test_unexpected_tracked_changes_stop_before_any_restore(
    fresh_checkout: tuple[Path, dict[str, str], bytes, bytes], mutation: str
) -> None:
    repo, env, _, updated = fresh_checkout
    source = repo / 'rust/src/main.rs'
    if mutation == 'tracked source':
        source.write_text(source.read_text() + '// unexpected tracked edit\n')
    elif mutation == 'staged lock':
        command(repo, env, 'git', 'add', 'rust/Cargo.lock')
    else:
        command(repo, env, 'git', 'config', 'core.filemode', 'true')
        (repo / 'rust/Cargo.lock').chmod(0o755)
    before = command(repo, env, 'git', 'status', '--porcelain')
    source_bytes = source.read_bytes()
    result = restore_step(repo, env)
    assert result.returncode != 0
    assert 'unexpected tracked' in result.stdout + result.stderr
    assert command(repo, env, 'git', 'status', '--porcelain') == before
    assert (repo / 'rust/Cargo.lock').read_bytes() == updated
    assert source.read_bytes() == source_bytes
    evidence = Path(env['RUNNER_TEMP']) / 'fresh-cargo-lock'
    assert before in (evidence / 'status-before.txt').read_text().strip()
    assert (evidence / 'diff-before.patch').read_text()


def test_ci_keeps_fresh_workspace_tests_before_locked_source_observations() -> (
    None
):
    workflow = steps()
    restore = next(
        i for i, step in enumerate(workflow) if step.get('name') == STEP
    )
    workspace = next(
        i
        for i, step in enumerate(workflow)
        if step.get('run') == 'cargo test --workspace --locked'
    )
    observe = next(
        i
        for i, step in enumerate(workflow)
        if str(step.get('run', '')).startswith('cargo run -p sdm-testkit')
    )
    assert workspace < restore < observe
    assert '--locked' in workflow[observe]['run']
    assert (
        workflow[restore]['if'] == "${{ !cancelled() && env.FRESH == 'true' }}"
    )
    artifact = next(
        step
        for step in workflow
        if step.get('with', {}).get('name') == 'fresh-cargo-lock-rust-test'
    )
    assert artifact['if'] == "${{ !cancelled() && env.FRESH == 'true' }}"
    assert artifact['with']['path'] == '${{ runner.temp }}/fresh-cargo-lock'
