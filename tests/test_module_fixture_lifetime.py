# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owned module fixtures release successes and preserve usable failed evidence."""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.tooling


@pytest.mark.parametrize('failure_phase', ['none', 'setup', 'call', 'teardown'])
def test_owned_module_directories_follow_the_complete_outcome(
    tmp_path: Path, failure_phase: str
) -> None:
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'objects').mkdir()
    (outside / 'keep').write_text('unrelated data')
    source_marker = tmp_path / 'source-path'
    tree_marker = tmp_path / 'tree-path'
    script = tmp_path / 'test_lifetime.py'
    # Exercise the real lifecycle hook, not a replacement test implementation.
    shutil.copy2(ROOT / 'conftest.py', tmp_path / 'conftest.py')
    script.write_text(
        textwrap.dedent(
            f"""
            import importlib.util
            import pathlib
            import subprocess
            import sys
            import pytest

            spec = importlib.util.spec_from_file_location(
                'lifetime_owner', {str(ROOT / 'tests/test_contract_coverage.py')!r}
            )
            owner = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = owner
            spec.loader.exec_module(owner)

            def git(path, *args):
                return subprocess.run(
                    ['git', '-C', str(path), '-c', 'user.name=Fixture Test',
                     '-c', 'user.email=fixture@example.com', *args],
                    capture_output=True, text=True, check=True,
                ).stdout.strip()

            def build(path):
                pathlib.Path({str(source_marker)!r}).write_text(str(path))
                git(path, 'init', '-q')
                (path / 'payload').write_text('a small real repository')
                git(path, 'add', 'payload')
                git(path, 'commit', '-q', '-m', 'fixture')
                info = path / '.git/objects/info'
                info.mkdir(parents=True, exist_ok=True)
                (info / 'alternates').write_text({str(outside / 'objects')!r} + '\\n')
                (path / 'external-link').symlink_to({str(outside)!r}, target_is_directory=True)
                if {failure_phase!r} == 'setup':
                    raise RuntimeError('owned setup failure')
                return path, git(path, 'rev-parse', 'HEAD')

            owner.build = build
            origin = owner.origin

            @pytest.fixture(scope='module')
            def consumer(origin, owned_module_dir):
                path, sha = origin
                tree = owned_module_dir('candidate') / 'tree'
                git(path, 'worktree', 'add', '-q', '--detach', str(tree), sha)
                pathlib.Path({str(tree_marker)!r}).write_text(str(tree))
                yield tree
                assert (path / 'payload').is_file()
                assert git(tree, 'rev-parse', 'HEAD') == sha
                if {failure_phase!r} == 'teardown':
                    raise RuntimeError('owned teardown failure')

            def test_consume(consumer):
                assert (consumer / 'payload').read_text() == 'a small real repository'
                if {failure_phase!r} == 'call':
                    raise RuntimeError('owned call failure')
            """
        )
    )
    done = subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            '-q',
            '-p',
            'no:cacheprovider',
            '--basetemp',
            str(tmp_path / 'module-temps'),
            str(script),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == (0 if failure_phase == 'none' else 1), (
        done.stdout + done.stderr
    )
    assert source_marker.is_file(), done.stdout + done.stderr
    source = Path(source_marker.read_text())
    if failure_phase == 'none':
        assert not source.exists()
        assert not Path(tree_marker.read_text()).parent.exists()
    else:
        assert f'owned {failure_phase} failure' in done.stdout
        assert (source / 'payload').read_text() == 'a small real repository'
        where = Path(tree_marker.read_text()) if tree_marker.exists() else source
        history = subprocess.run(
            ['git', '-C', str(where), 'log', '-1', '--format=%s'],
            capture_output=True,
            text=True,
            check=True,
        )
        assert history.stdout.strip() == 'fixture'
    assert (outside / 'keep').read_text() == 'unrelated data'
    assert (outside / 'objects').is_dir()
