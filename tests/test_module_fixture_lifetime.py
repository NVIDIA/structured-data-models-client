# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A real module fixture releases only its own directory after its consumers."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.tooling


@pytest.mark.parametrize('fail_setup', [False, True])
def test_owned_module_directory_is_released_after_use(tmp_path, fail_setup):
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'keep').write_text('unrelated data')
    marker = tmp_path / 'owned-path'
    script = tmp_path / 'test_lifetime.py'
    script.write_text(
        textwrap.dedent(
            f"""
            import importlib.util
            import pathlib
            import sys
            import pytest

            spec = importlib.util.spec_from_file_location(
                'lifetime_owner', {str(ROOT / 'tests/test_contract_coverage.py')!r}
            )
            owner = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = owner
            spec.loader.exec_module(owner)

            def build(path):
                pathlib.Path({str(marker)!r}).write_text(str(path))
                (path / 'payload').write_text('a small stand-in repository')
                info = path / '.git/objects/info'
                info.mkdir(parents=True)
                (info / 'alternates').write_text({str(outside)!r} + '\\n')
                (path / 'external-link').symlink_to({str(outside)!r}, target_is_directory=True)
                if {fail_setup!r}:
                    raise RuntimeError('owned setup failure')
                return path, 'a' * 40

            owner.build = build
            origin = owner.origin

            @pytest.fixture(scope='module')
            def consumer(origin):
                path, _ = origin
                yield path
                assert (path / 'payload').is_file()

            def test_consume(consumer):
                assert (consumer / 'payload').is_file()
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
    assert done.returncode == (1 if fail_setup else 0), (
        done.stdout + done.stderr
    )
    if fail_setup:
        assert 'owned setup failure' in done.stdout
    assert marker.is_file(), done.stdout + done.stderr
    assert not Path(marker.read_text()).exists()
    assert (outside / 'keep').read_text() == 'unrelated data'
