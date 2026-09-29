# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-package checks that no single distribution's suite can make.

Each package tests its own behaviour. Nothing tested that the five agree with
each other, that what they export stays what they exported, or that the claims
the configuration files make about themselves are true. Those are the failures
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
from collections.abc import Sequence

import tomllib

REPO = pathlib.Path(__file__).resolve().parent.parent
PACKAGES = REPO / 'packages'
GOLDEN = REPO / 'tests' / 'golden'


def _split_rule(code: str) -> tuple[str, str]:
    """A ruff code as its linter prefix and its number, e.g. ``('FIX', '002')``."""
    match = re.match(r'([A-Z]+)(\d*)', code)
    assert match, f'not a ruff rule code: {code!r}'
    return match.group(1), match.group(2)


def _linter_of(code: str) -> str:
    return _split_rule(code)[0]


def _covered_by(rule: str, selectors: Sequence[str]) -> bool:
    """Whether any selector actually enables ``rule``.

    Plain string prefixes are wrong here: `'FIX002'.startswith('F')` is True,
    but ruff's `F` is Pyflakes and `FIX` is flake8-fixme -- selecting one does
    not select the other, which is verifiable with
    """
    if 'ALL' in selectors:
        return True
    rule_linter, rule_number = _split_rule(rule)
    return any(
        selector_linter == rule_linter
        and rule_number.startswith(selector_number)
        for selector_linter, selector_number in map(_split_rule, selectors)
    )


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

    @staticmethod
    def _declared_version() -> str:
        versions = TestDistributionCoherence._declared_versions()
        assert len(set(versions.values())) == 1, versions
        return next(iter(versions.values()))

    @staticmethod
    def _declared_versions() -> dict[str, str]:
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
        return versions

    def test_every_distribution_declares_the_same_version(self):
        versions = self._declared_versions()
        assert len(set(versions.values())) == 1, (
            f'distributions disagree on the version: {versions}'
        )

    def test_every_first_party_pin_names_the_version_being_released(self):
        """The five pin each other by exact version, and those pins are literals.

        `test_every_distribution_declares_the_same_version` proves the five
        agree on what they *are*. It says nothing about what they *require*,
        and those are thirteen separate hard-coded strings across two files.
        """
        declared = self._declared_version()
        distributions = {name for name in _distributions()}

        pattern = re.compile(
            r'(?P<name>sdm-[a-z-]+)(?:\[[^\]]*\])?==(?P<version>[0-9][^\'"\s,]*)'
        )
        wrong = []
        for source in sorted(PACKAGES.glob('*/pyproject.toml')) + sorted(
            PACKAGES.glob('*/setup.py')
        ):
            for match in pattern.finditer(source.read_text()):
                if match.group('name') not in distributions:
                    continue
                if match.group('version') != declared:
                    wrong.append(
                        f'{source.relative_to(REPO)}: '
                        f'{match.group(0)} but the release is {declared}'
                    )
        assert not wrong, (
            'a first-party pin names a version other than the one being '
            'released:\n' + '\n'.join(wrong)
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
                # databricks-sdk is imported by the client's own serving
                # transport, so the client is the distribution that declares it.
                assert name in known or name in {
                    'pytest',
                    'requests-mock',
                    'databricks-sdk',
                }, (
                    f'extra {extra!r} requires {name!r}, which is not one of '
                    f"this repository's distributions"
                )

    def test_the_databricks_extra_brings_what_reading_a_metric_view_needs(
        self,
    ):
        """`Schema.from_source(..., metric_view=...)` parses the view's YAML.

        The client's `databricks` extra reaches Databricks through the
        connectors' own, and Databricks Runtime 14.3 LTS has no YAML parser, so
        a metric view failed there with `No module named 'yaml'` until the
        connectors' extra carried one.
        """
        client = _pyproject('sdm-client')['project']['optional-dependencies']
        assert 'sdm-connectors[databricks]==1.0.0' in client['databricks']
        connectors = _pyproject('sdm-connectors')['project']
        names = [
            requirement.split('>=')[0].split('==')[0].strip().lower()
            for requirement in connectors['optional-dependencies']['databricks']
        ]
        assert 'pyyaml' in names


class TestConfigurationTellsTheTruth:
    """A comment that states a checkable fact is checked.

    The relational engine's lint configuration carried a note saying its ignored
    rules were enabled in two sibling packages. One of them was not: the rule
    belongs to a rule family the shared configuration never selected. Nothing
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

    def test_the_ignore_list_sorts_itself_the_way_its_headings_say(self):
        """The engine's ignore list is written in two labelled blocks.

        One is rules the shared baseline really does select, so listing them
        here silences something; the other is a record of rules nothing selects
        anywhere. An earlier note called the whole list inert, which was wrong
        """
        source = (
            PACKAGES / 'sdm-relational-engine' / 'pyproject.toml'
        ).read_text()
        root = tomllib.loads((REPO / 'pyproject.toml').read_text())
        selected = root['tool']['ruff']['lint']['select']
        engine = _pyproject('sdm-relational-engine')['tool']['ruff']['lint']
        effective = list(selected) + list(engine.get('extend-select', []))

        active_block, _, backlog_block = source.partition(
            '--- Recorded backlog'
        )
        _, _, active_block = active_block.partition('--- Active exemptions')
        assert active_block and backlog_block, (
            'the ignore list no longer carries both headings this test reads'
        )

        def listed(block: str) -> set[str]:
            return {
                rule
                for rule in engine.get('ignore', [])
                if f'"{rule}"' in block
            }

        for rule in listed(active_block):
            assert _covered_by(rule, effective), (
                f'{rule} sits under "Active exemptions" but nothing selects '
                f'it, so ignoring it silences nothing'
            )
        for rule in listed(backlog_block):
            assert not _covered_by(rule, selected), (
                f'{rule} sits under "Recorded backlog", which says the shared '
                f'baseline does not select it -- but {selected} does'
            )

    def test_the_tabular_task_kinds_match_what_the_model_declares(self):
        """`TABULAR_TASK_KINDS` restates what `models.py` says, and says so.

        The materializer refuses a stated `task=` the tabular deployment does
        not serve, and it has to do that before a model handle is necessarily
        in hand -- so the list is written twice. A comment claiming two things
        """
        from sdm_client.materializers.tabular import TABULAR_TASK_KINDS
        from sdm_client.models import spec_for

        assert TABULAR_TASK_KINDS == spec_for('kumo-tabular').runtime.tasks, (
            'the materializer and the model spec disagree about what the '
            'tabular deployment serves'
        )

    @staticmethod
    def _module_is_on_disk(dotted: str) -> bool | None:
        """Whether ``dotted`` exists as a file, without importing anything.

        Three times now this check has confused "absent" with something else,
        because it asked by importing. ``find_spec`` executes every parent
        ``__init__`` on the way down, so it reports a module as missing when a
        """
        import importlib.machinery
        import importlib.util

        top, *rest = dotted.split('.')
        try:
            spec = importlib.util.find_spec(top)
        except (ImportError, ValueError):  # pragma: no cover - not installed
            return None
        if spec is None:
            return None
        if not rest:
            return True
        roots = spec.submodule_search_locations or []
        for root in roots:
            base = pathlib.Path(root, *rest[:-1])
            leaf = rest[-1]
            if (base / leaf / '__init__.py').is_file():
                return True
            if any(
                (base / f'{leaf}{suffix}').is_file()
                for suffix in importlib.machinery.all_suffixes()
            ):
                return True
            # A directory with no __init__.py is a namespace package and is
            # importable all the same. This tree has thirteen of them, so
            # requiring the file would call an existing package missing.
            if (base / leaf).is_dir():
                return True
        return False

    def test_every_module_the_type_baseline_names_still_exists(self):
        """A baseline entry for a module that is gone silences nothing.

        The engine was restructured on main and three entries kept naming
        modules that no longer exist, while the errors they covered moved to
        new ones. Nothing complains about an override that matches nothing, so
        """
        root = tomllib.loads((REPO / 'pyproject.toml').read_text())
        named = [
            module
            for override in root['tool']['mypy']['overrides']
            for module in (
                override['module']
                if isinstance(override['module'], list)
                else [override['module']]
            )
        ]
        assert named, 'the type-checking baseline vanished; update this test'

        checked, missing = 0, []
        for module in named:
            present = self._module_is_on_disk(module)
            if present is None:
                continue  # the distribution is absent from this job
            checked += 1
            if not present:
                missing.append(module)

        assert checked, (
            'no distribution the baseline names is installed here, so this '
            'checked nothing'
        )
        assert not missing, (
            f'the mypy baseline names modules that do not exist: {missing}. '
            f'Delete the entry, or point it at whatever the code became.'
        )

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
        selected = root['tool']['ruff']['lint']['select']
        backlog = [
            rule
            for rule in engine.get('ignore', [])
            if _linter_of(rule) != 'D' and not _covered_by(rule, selected)
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

    KNOWN = ('3.10', '3.11', '3.12', '3.13', '3.14')

    @staticmethod
    def _parts(version: str) -> tuple[int, ...]:
        """Compare versions numerically.

        `'3.10' >= '3.9'` is False as strings, because `'1' < '9'`. Nothing
        here has a 3.9 floor today, so a lexical comparison happened to agree
        -- and would have started disagreeing silently at 3.20, in a helper
        """
        return tuple(int(part) for part in version.split('.'))

    @classmethod
    def _admitted(cls, requires: str) -> list[str]:
        floor = ceiling = None
        inclusive = False
        for clause in requires.split(','):
            clause = clause.strip()
            if clause.startswith('>='):
                floor = cls._parts(clause[2:].strip())
            elif clause.startswith('<='):
                ceiling = cls._parts(clause[2:].strip())
                inclusive = True
            elif clause.startswith('<'):
                ceiling = cls._parts(clause[1:].strip())
        assert floor is not None, f'no lower bound in {requires!r}'
        admitted = [v for v in cls.KNOWN if cls._parts(v) >= floor]
        if ceiling is not None:
            admitted = [
                v
                for v in admitted
                if (
                    cls._parts(v) <= ceiling
                    if inclusive
                    else cls._parts(v) < ceiling
                )
            ]
        return admitted

    def test_the_version_comparison_this_class_relies_on_is_numeric(self):
        """The helper above is the only thing standing between a metadata
        mistake and a release, so it is tested rather than trusted.
        """
        assert self._parts('3.10') > self._parts('3.9')
        assert self._parts('3.9') < self._parts('3.14')
        assert self._admitted('>=3.10') == list(self.KNOWN)
        assert self._admitted('>=3.10,<3.14') == [
            '3.10',
            '3.11',
            '3.12',
            '3.13',
        ]
        assert self._admitted('>=3.12') == ['3.12', '3.13', '3.14']
        assert self._admitted('>=3.10,<=3.13') == [
            '3.10',
            '3.11',
            '3.12',
            '3.13',
        ]

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
