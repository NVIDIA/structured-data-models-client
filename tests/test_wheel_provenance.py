# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate installation checks through the same CLI used by CI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/check_wheel_provenance.py'
PACKAGES = {
    'structured-data-models-api': 'sdm_api',
    'structured-data-models-client': 'sdm_client',
    'structured-data-models-connectors': 'sdm_connectors',
}


@pytest.fixture
def candidate(tmp_path: Path) -> tuple[list[Path], Path]:
    artifacts = tmp_path / 'candidate # artifacts'
    installed = tmp_path / 'site-packages'
    artifacts.mkdir()
    wheels = []
    for name, module in PACKAGES.items():
        normalized = name.replace('-', '_')
        wheel = artifacts / f'{normalized}-2.0.0rc1-py3-none-any.whl'
        wheel.touch()
        wheels.append(wheel)
        package = installed / module
        package.mkdir(parents=True)
        (package / '__init__.py').write_text('')
        metadata = installed / f'{normalized}-2.0.0rc1.dist-info'
        metadata.mkdir()
        (metadata / 'METADATA').write_text(
            f'Metadata-Version: 2.1\nName: {name}\nVersion: 2.0.0rc1\n'
        )
        (metadata / 'direct_url.json').write_text(
            json.dumps({'url': wheel.as_uri(), 'archive_info': {}})
        )
    return wheels, installed


def run(
    wheels: list[Path], installed: Path, *options: str
) -> subprocess.CompletedProcess[str]:
    # -S excludes the developer's installed packages; only this fixture is on
    # the import path. The script itself requires no third-party packages.
    return subprocess.run(
        [sys.executable, '-S', str(SCRIPT), *options, *map(str, wheels)],
        env={**os.environ, 'PYTHONPATH': str(installed)},
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_exact_candidates_and_escaped_local_paths_pass(candidate, tmp_path):
    wheels, installed = candidate
    version = tmp_path / 'VERSION'
    result = run(
        wheels, installed, '--installed', '--write-version', str(version)
    )
    assert result.returncode == 0, result.stderr
    assert version.read_text() == '2.0.0rc1'


def test_a_missing_distribution_fails(candidate):
    wheels, installed = candidate
    result = run(wheels[:-1], installed)
    assert result.returncode == 1
    assert 'missing distributions' in result.stderr


def test_three_files_do_not_imply_three_distributions(candidate):
    wheels, installed = candidate
    duplicate = wheels[0].with_name(wheels[0].name.replace('py3', 'py310'))
    duplicate.touch()
    result = run([wheels[0], duplicate, wheels[1]], installed)
    assert result.returncode == 1
    assert 'duplicate distribution' in result.stderr


def test_mixed_versions_fail(candidate):
    wheels, installed = candidate
    changed = wheels[0].with_name(wheels[0].name.replace('rc1', 'rc2'))
    changed.touch()
    result = run([changed, *wheels[1:]], installed)
    assert result.returncode == 1
    assert 'disagree on version' in result.stderr


@pytest.mark.parametrize(
    'url_kind', ['elsewhere', 'remote', 'missing', 'relative']
)
def test_untrusted_install_provenance_fails(candidate, tmp_path, url_kind):
    wheels, installed = candidate
    record = next(installed.glob('*.dist-info/direct_url.json'))
    same_name = tmp_path / 'wrong artifacts' / wheels[0].name
    urls = {
        'elsewhere': same_name.as_uri(),
        'remote': 'https://example.invalid/' + wheels[0].name,
        'missing': '',
        'relative': 'file:' + wheels[0].name,
    }
    record.write_text(json.dumps({'url': urls[url_kind]}))
    result = run(wheels, installed, '--installed')
    assert result.returncode == 1
    assert 'installed from' in result.stderr


def test_checkout_shadowing_the_install_fails(candidate, tmp_path):
    wheels, installed = candidate
    checkout = tmp_path / 'checkout'
    package = checkout / 'sdm_client'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    result = run(
        wheels, Path(str(checkout) + os.pathsep + str(installed)), '--installed'
    )
    assert result.returncode == 1
    assert 'import does not resolve' in result.stderr


def test_localhost_file_url_passes(candidate):
    wheels, installed = candidate
    for record in installed.glob('*.dist-info/direct_url.json'):
        data = json.loads(record.read_text())
        data['url'] = data['url'].replace('file:///', 'file://localhost/')
        record.write_text(json.dumps(data))
    result = run(wheels, installed, '--installed')
    assert result.returncode == 0, result.stderr


def test_installed_version_must_match(candidate):
    wheels, installed = candidate
    record = next(installed.glob('*.dist-info/METADATA'))
    record.write_text(
        record.read_text().replace('Version: 2.0.0rc1', 'Version: 1.0.0')
    )
    result = run(wheels, installed, '--installed')
    assert result.returncode == 1
    assert 'installed version' in result.stderr


def test_missing_installed_distribution_fails(candidate):
    wheels, installed = candidate
    metadata = next(installed.glob('*.dist-info/METADATA'))
    metadata.parent.rename(metadata.parent.with_suffix('.not-metadata'))
    result = run(wheels, installed, '--installed')
    assert result.returncode == 1
    assert 'No package metadata' in result.stderr


def test_workflows_check_after_the_final_dependency_install():
    import yaml

    ci = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())
    publish = yaml.safe_load(
        (ROOT / '.github/workflows/publish.yml').read_text()
    )
    jobs = [
        ci['jobs'][name]
        for name in ('installed-wheel', 'runtime-install', 'sbom')
    ]
    jobs.append(
        next(
            job
            for job in publish['jobs'].values()
            if any(
                step.get('name') == 'Software bill of materials'
                for step in job['steps']
            )
        )
    )
    for job in jobs:
        commands = '\n'.join(step.get('run', '') for step in job['steps'])
        assert commands.rindex('check_wheel_provenance.py') > commands.rindex(
            'pip install'
        )
        assert '--installed "$GITHUB_WORKSPACE"/dist/' in commands
    build = '\n'.join(
        step.get('run', '') for step in ci['jobs']['build']['steps']
    )
    assert (
        'check_wheel_provenance.py --write-version dist/VERSION dist/*.whl'
        in build
    )


@pytest.mark.parametrize(
    ('tag', 'accepted'),
    [
        ('v2.0.0', True),
        ('v2.0.0rc1', True),
        ('v2.0.0a1', True),
        ('v2.0.0b12', True),
        ('v2.0.0-rc.1', False),
        ('v2.0.0rc', False),
        ('vv2.0.0', False),
    ],
)
def test_the_release_accepts_final_and_pre_release_tags(
    tmp_path, monkeypatch, tag, accepted
):
    import re

    import yaml

    publish = yaml.safe_load(
        (ROOT / '.github/workflows/publish.yml').read_text()
    )
    filters = publish[True]['push']['tags']
    pushed = any(re.fullmatch(f.replace('.', r'\.'), tag) for f in filters)
    step = next(
        step
        for step in publish['jobs']['assemble']['steps']
        if step.get('name')
        == 'The tag must match what the distributions declare'
    )
    code = step['run'].split("python - <<'EOF'\n", 1)[1].rsplit('\nEOF', 1)[0]
    version = tag.removeprefix('v')
    for package in PACKAGES:
        (tmp_path / 'packages' / package).mkdir(parents=True)
        (tmp_path / 'packages' / package / 'pyproject.toml').touch()
    (tmp_path / 'dist').mkdir()
    for package in PACKAGES:
        name = package.replace('-', '_')
        (tmp_path / 'dist' / f'{name}-{version}-py3-none-any.whl').touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GITHUB_REF_NAME', tag)
    try:
        exec(compile(code, '<release tag check>', 'exec'), {})
    except SystemExit:
        checked = False
    else:
        checked = True
    assert (pushed, checked) == (accepted, accepted)
