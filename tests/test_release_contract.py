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


class TestDocumentedExamples:
    """Examples are the most-read code in the repository and nothing ran them.

    Executing them needs a live NIM, so that belongs in the live suite. What can
    be checked with nothing running is still most of what goes wrong in a
    published example: it stopped being valid Python, or it names something the
    library no longer exports.
    """

    @staticmethod
    def _python_blocks() -> list[tuple[str, str]]:
        import re

        blocks = []
        sources = [REPO / 'README.md'] + sorted((REPO / 'docs').rglob('*.md'))
        for path in sources:
            for match in re.finditer(
                r'^```python\n(.*?)^```', path.read_text(), re.S | re.M
            ):
                blocks.append((str(path.relative_to(REPO)), match.group(1)))
        return blocks

    def test_every_documented_python_block_is_valid_python(self):
        import ast

        broken = []
        for where, source in self._python_blocks():
            try:
                ast.parse(source)
            except SyntaxError as error:
                broken.append(f'{where}: {error}')
        assert not broken, (
            'a fenced ```python block no longer parses. Make it runnable, or '
            'fence it as ```text if it is deliberately a fragment:\n'
            + '\n'.join(broken)
        )

    def test_every_symbol_the_documentation_imports_still_exists(self):
        import ast
        import importlib

        missing = []
        for where, source in self._python_blocks():
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if not node.module or not node.module.startswith(
                    ('sdm_', 'predictive_query')
                ):
                    continue
                try:
                    module = importlib.import_module(node.module)
                except ImportError:  # an extra this environment lacks
                    continue
                missing += [
                    f'{where}: {node.module}.{alias.name}'
                    for alias in node.names
                    if not hasattr(module, alias.name)
                ]
        assert not missing, (
            'the documentation imports names the library no longer exports:\n'
            + '\n'.join(missing)
        )

    def test_the_readme_quickstart_imports_only_public_names(self):
        import ast
        import importlib

        readme = (REPO / 'README.md').read_text()
        import re

        for match in re.finditer(r'^```python\n(.*?)^```', readme, re.S | re.M):
            for node in ast.walk(ast.parse(match.group(1))):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == 'sdm_client'
                ):
                    exported = importlib.import_module('sdm_client').__all__
                    outside = [
                        alias.name
                        for alias in node.names
                        if alias.name not in exported
                    ]
                    assert not outside, (
                        f'the README imports {outside} from sdm_client, which '
                        f'is not in __all__ -- so the first thing a new user '
                        f'copies is an unsupported import'
                    )


class TestInterpreterSupportIsStated:
    """`requires-python` and the classifiers are two claims about one thing.

    They disagreed: every distribution admitted 3.14 through `requires-python`
    and none of them listed it, so an index would show a package as unsupported
    on an interpreter it runs on perfectly well.
    """

    @staticmethod
    def _admitted(requires: str) -> list[str]:
        floor = requires.split(',')[0].strip().lstrip('>=').strip()
        ceiling = None
        for clause in requires.split(','):
            clause = clause.strip()
            if clause.startswith('<'):
                ceiling = clause.lstrip('<=').strip()
        known = ['3.10', '3.11', '3.12', '3.13', '3.14']
        admitted = [v for v in known if v >= floor]
        if ceiling:
            admitted = [v for v in admitted if v < ceiling]
        return admitted

    def test_every_admitted_interpreter_is_also_a_classifier(self):
        for name in _distributions():
            project = _pyproject(name)['project']
            admitted = self._admitted(project['requires-python'])
            classified = {
                line.rsplit('::', 1)[-1].strip()
                for line in project.get('classifiers', [])
                if line.startswith('Programming Language :: Python :: 3.')
            }
            missing = [v for v in admitted if v not in classified]
            assert not missing, (
                f'{name} admits Python {missing} through requires-python '
                f'{project["requires-python"]!r} but does not classify them, so '
                f'an index reports it unsupported on an interpreter it runs on'
            )

    def test_an_extra_narrower_than_its_parent_says_so(self):
        """The relational extra does not resolve on every Python the client does.

        That is correct -- the engine is a native build with a shorter wheel
        matrix -- but pip's error names a distribution, not a reason. Anything
        that narrow has to be explained where someone reads it.
        """
        client = _pyproject('sdm-client')
        engine = _pyproject('sdm-relational-engine')
        if (
            client['project']['requires-python']
            == engine['project']['requires-python']
        ):
            return
        source = (PACKAGES / 'sdm-client' / 'pyproject.toml').read_text()
        assert '3.14' in source and 'native build' in source, (
            'sdm-client admits interpreters its [relational] extra cannot '
            'install on, and its pyproject does not explain that anywhere'
        )
        install_doc = (
            REPO / 'docs' / 'get-started' / 'installation.md'
        ).read_text()
        assert 'relational' in install_doc, (
            'the installation guide does not mention the relational extra'
        )
