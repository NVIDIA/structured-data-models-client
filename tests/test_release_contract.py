# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-package checks that no single distribution's suite can make.

Each package tests its own behaviour. Nothing tested that the five agree with
each other, that what they export stays what they exported, or that the claims
the configuration files make about themselves are true. Those are the failures
a release notices and a unit test does not.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import tomllib

REPO = pathlib.Path(__file__).resolve().parent.parent
PACKAGES = REPO / 'packages'
GOLDEN = REPO / 'tests' / 'golden'


def _pyproject(name: str) -> dict:
    return tomllib.loads((PACKAGES / name / 'pyproject.toml').read_text())


def _distributions() -> list[str]:
    return sorted(
        p.name for p in PACKAGES.iterdir() if (p / 'pyproject.toml').is_file()
    )


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO / 'scripts' / script)],
        capture_output=True,
        text=True,
    )


class TestPublicSurface:
    """What a caller can name, and what the endpoint receives."""

    def test_the_exported_api_matches_the_recorded_snapshot(self):
        result = _run('snapshot_public_api.py')
        assert result.returncode == 0, (
            'The public API changed. If that was intended, re-record it with '
            '`python scripts/snapshot_public_api.py --write` and say so in the '
            f'changelog.\n{result.stdout}{result.stderr}'
        )

    def test_the_wire_payloads_match_the_recorded_snapshot(self):
        result = _run('snapshot_wire_payloads.py')
        assert result.returncode == 0, (
            'A canonical request now serializes differently. If that was '
            'intended, re-record with `python scripts/snapshot_wire_payloads.py '
            f'--write`.\n{result.stdout}{result.stderr}'
        )

    def test_every_recorded_export_is_still_importable(self):
        recorded = json.loads((GOLDEN / 'public_api.json').read_text())
        import importlib

        for package, surface in recorded.items():
            exported = surface.get('__all__')
            if exported is None:
                continue
            module = importlib.import_module(package)
            missing = [name for name in exported if not hasattr(module, name)]
            assert not missing, f'{package} no longer exports {missing}'


class TestErrorContract:
    """Exception identity is what callers branch on, so it is pinned."""

    def test_every_error_descends_from_one_base(self):
        import sdm_client
        from sdm_client import StructuredDataError

        errors = [
            getattr(sdm_client, name)
            for name in sdm_client.__all__
            if isinstance(getattr(sdm_client, name), type)
            and issubclass(getattr(sdm_client, name), Exception)
            and not issubclass(getattr(sdm_client, name), Warning)
        ]
        assert errors, 'no exception types are exported'
        for error in errors:
            assert issubclass(error, StructuredDataError), (
                f'{error.__name__} is raised at users but does not descend from '
                f'StructuredDataError, so `except StructuredDataError` misses it'
            )

    def test_the_warning_category_is_the_library_s_own(self):
        from sdm_client import SDMWarning

        assert issubclass(SDMWarning, Warning)
        assert not issubclass(SDMWarning, DeprecationWarning), (
            'SDMWarning must stay filterable on its own terms'
        )


class TestDistributionCoherence:
    """The five are released together, so they must agree."""

    def test_every_distribution_declares_the_same_version(self):
        versions = {}
        for name in _distributions():
            project = _pyproject(name)['project']
            if 'version' in project:
                versions[name] = project['version']
                continue
            module = name.replace('-', '_')
            if module == 'sdm_pql':
                module = 'predictive_query'
            source = PACKAGES / name / 'src' / module / '_version.py'
            text = source.read_text()
            versions[name] = text.split('=')[-1].strip().strip('\'"')
        assert len(set(versions.values())) == 1, (
            f'distributions disagree on the version: {versions}'
        )

    def test_every_distribution_declares_the_same_licence(self):
        licences = {
            name: _pyproject(name)['project'].get('license')
            for name in _distributions()
        }
        assert set(licences.values()) == {'Apache-2.0'}, licences

    def test_every_distribution_ships_its_licence_files(self):
        for name in _distributions():
            declared = _pyproject(name)['project'].get('license-files')
            assert declared, f'{name} declares no license-files'
            for filename in declared:
                assert (PACKAGES / name / filename).is_file(), (
                    f'{name} declares {filename} but does not carry it'
                )

    def test_every_distribution_ships_a_typed_marker(self):
        for name in _distributions():
            markers = list((PACKAGES / name / 'src').glob('*/py.typed'))
            assert markers, f'{name} claims to be typed but ships no py.typed'

    def test_the_declared_python_floor_is_the_same_everywhere(self):
        floors = {}
        for name in _distributions():
            requires = _pyproject(name)['project']['requires-python']
            floors[name] = requires.split(',')[0].strip()
        assert len(set(floors.values())) == 1, (
            f'distributions disagree on the minimum Python: {floors}'
        )

    def test_every_advertised_extra_of_the_client_resolves_to_a_real_target(
        self,
    ):
        client = _pyproject('sdm-client')['project']
        known = set(_distributions())
        for extra, requirements in client['optional-dependencies'].items():
            for requirement in requirements:
                name = (
                    requirement.split('[')[0]
                    .split('==')[0]
                    .split('>=')[0]
                    .strip()
                )
                assert name in known or name in {'pytest', 'requests-mock'}, (
                    f'extra {extra!r} requires {name!r}, which is not one of '
                    f"this repository's distributions"
                )


class TestConfigurationTellsTheTruth:
    """A comment that states a checkable fact is checked.

    The relational engine's lint configuration carried a note saying its ignored
    rules were enabled in two sibling packages. One of them was not: the rule
    belongs to a rule family the shared configuration never selected. Nothing
    caught it because nothing reads a comment. This does.
    """

    def test_the_shared_lint_baseline_is_not_narrowed_by_any_package(self):
        root = tomllib.loads((REPO / 'pyproject.toml').read_text())
        baseline = set(root['tool']['ruff']['lint']['select'])
        for name in _distributions():
            config = _pyproject(name).get('tool', {}).get('ruff', {})
            assert config.get('extend', '').endswith('pyproject.toml'), (
                f'{name} does not extend the shared ruff configuration, so it '
                f'can drift from the baseline silently'
            )
            narrowed = set(config.get('lint', {}).get('select', baseline))
            assert baseline <= narrowed or 'select' not in config.get(
                'lint', {}
            ), f'{name} narrows the shared lint baseline to {narrowed}'

    def test_the_recorded_style_backlog_is_clean_where_the_note_says_it_is(
        self,
    ):
        """The engine's ignore list records a backlog and names where it is clean.

        That claim used to be wrong in both halves -- the rules were not
        selected anywhere, and the two packages named did not pass clean under
        them. A comment cannot be wrong for long if a test reads it.
        """
        engine = _pyproject('sdm-relational-engine')['tool']['ruff']['lint']
        root = tomllib.loads((REPO / 'pyproject.toml').read_text())
        selected = tuple(root['tool']['ruff']['lint']['select'])
        backlog = [
            rule
            for rule in engine.get('ignore', [])
            if not rule.startswith('D') and not rule.startswith(selected)
        ]
        assert backlog, 'the recorded style backlog vanished; update this test'

        ruff = pathlib.Path(sys.executable).parent / 'ruff'
        if not ruff.exists():  # pragma: no cover - ruff not installed
            import pytest

            pytest.skip('ruff is not installed in this environment')

        result = subprocess.run(
            [
                str(ruff),
                'check',
                '--no-cache',
                '--quiet',
                '--select',
                ','.join(backlog),
                str(PACKAGES / 'sdm-client' / 'src'),
                str(PACKAGES / 'sdm-connectors' / 'src'),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            'sdm-client and sdm-connectors no longer pass clean under the '
            'style backlog the engine records, so the note beside that ignore '
            f'list is now wrong:\n{result.stdout}{result.stderr}'
        )
