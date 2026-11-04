# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Architecture guards: JNI dependency boundaries and SQL ownership."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUST_CORE = ROOT / 'scripts/rust_core'
#: The L0 scan's test copies the crates into a temporary tree: the judging
#: tools' CI job runs this file once, outside the Python version matrix.
pytestmark = pytest.mark.tooling


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _tool(name: str) -> ModuleType:
    r"""A module of ``scripts/rust_core``, its siblings importable."""
    if str(RUST_CORE) not in sys.path:
        sys.path.insert(0, str(RUST_CORE))
    return _module(f't32_tools_{name}', RUST_CORE / f'{name}.py')


# --- l0_deps.py and l0_scan.py ---------------------------------------------------


@pytest.mark.parametrize('missing_first', [False, True])
def test_sql_cli_discovers_error_members_once_and_refreshes_next_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    missing_first: bool,
) -> None:
    scan = _tool('l0_scan')
    declarations = tmp_path / 'rust/crates/sdm-contract/src/error.rs'
    declarations.parent.mkdir(parents=True)
    members = (
        'impl CoreError {\n'
        '    pub fn new(k: Kind, c: Code, m: &str) -> Self { todo!() }\n'
        '    pub fn with_field(self, k: &str, v: Detail) -> Self { todo!() }\n'
        '}\n'
        'impl Detail {\n'
        '    pub fn text(v: &str) -> Self { todo!() }\n'
        '}\n'
    )
    declarations.write_text(members, encoding='utf-8')
    present = (
        scan.SQL_FREE_CRATES[1:] if missing_first else scan.SQL_FREE_CRATES
    )
    for crate in present:
        source = tmp_path / f'rust/crates/{crate}/src/lib.rs'
        source.parent.mkdir(parents=True)
        # The existing nested-diagnostic control, with actual declarations
        # in a non-SQL-free crate so only workspace discovery reads them.
        source.write_text(
            'pub fn f() -> CoreError { CoreError::new(K, C, "m")'
            '.with_field("rule", Detail::text("select a from t where x = 1")) }\n',
            encoding='utf-8',
        )
    real_read = Path.read_text
    reads: list[Path] = []

    def read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == declarations:
            reads.append(path)
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read_text)
    missing = (
        ['sql: rust/crates/sdm-schema/src: missing'] if missing_first else []
    )
    first_status = scan.main(['sql', '--root', str(tmp_path)])
    first_output = capsys.readouterr()
    first_reads = len(reads)
    # Only Detail changes between complete CLI invocations, on disk.
    # A process-global cache would wrongly retain its diagnostic exemption.
    declarations.write_text(
        members.replace(
            'fn text(v: &str) -> Self',
            'fn text(v: &str) -> Result<Self, CoreError>',
        ),
        encoding='utf-8',
    )
    second_status = scan.main(['sql', '--root', str(tmp_path)])
    second_output = capsys.readouterr()

    assert first_status == int(missing_first)
    assert first_output.err == ''
    assert first_output.out.splitlines() == [
        *missing,
        'sql: FAIL' if missing_first else 'sql: ok',
    ]
    assert second_status == 1
    assert second_output.err == ''
    assert second_output.out.splitlines() == [
        *missing,
        *[
            f'sql: rust/crates/{crate}/src/lib.rs:1: the strings '
            'of a statement hold `select a from t where`, SQL or a '
            "provider's name; SQL and providers are sdm-data's alone"
            for crate in present
        ],
        'sql: FAIL',
    ]
    assert (first_reads, len(reads)) == (1, 2)


def test_sql_cli_without_present_crates_does_not_discover_error_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scan = _tool('l0_scan')
    real_read = Path.read_text
    reads: list[Path] = []

    def read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        reads.append(path)
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read_text)
    assert scan.main(['sql', '--root', str(tmp_path)]) == 1
    output = capsys.readouterr()
    assert output.err == ''
    assert output.out.splitlines() == [
        *[
            f'sql: rust/crates/{crate}/src: missing'
            for crate in scan.SQL_FREE_CRATES
        ],
        'sql: FAIL',
    ]
    assert reads == []


def _crate_manifest(name: str) -> Path:
    for parent in ('rust/crates', 'rust/bindings'):
        manifest = ROOT / parent / name / 'Cargo.toml'
        if manifest.is_file():
            return manifest
    raise AssertionError(f'no crate {name}')


class Workspace:
    r"""``cargo metadata``'s shape for the workspace as the design draws it,
    built without cargo: every first-party crate ``l0_deps.py`` classifies
    and finds in the tree, at its real manifest, on the first-party edges
    its ``ALLOWED`` gives it; ``without`` leaves a crate out, as if it
    did not exist yet, so ``sdm-jni`` can be seeded whether or not the tree
    holds it."""

    def __init__(self, l0: ModuleType, without: tuple[str, ...] = ()) -> None:
        crates = [
            name
            for name in sorted(l0.CLASSES)
            if name not in without
            and any(
                (ROOT / parent / name / 'Cargo.toml').is_file()
                for parent in ('rust/crates', 'rust/bindings')
            )
        ]
        self.data: dict[str, Any] = {
            'packages': [],
            'workspace_members': [],
            'workspace_root': str(ROOT / 'rust'),
            'resolve': {'nodes': []},
        }
        for name in crates:
            self.package(name, _crate_manifest(name), member=True)
        for name in crates:
            for dep in sorted(l0.ALLOWED.get(name, frozenset())):
                if dep in crates:
                    self.edge(name, dep)

    def _id(self, name: str) -> str:
        return f't32+{name}@0.0.0'

    def package(self, name: str, manifest: Path, member: bool = False) -> None:
        if any(p['name'] == name for p in self.data['packages']):
            return
        self.data['packages'].append(
            {'id': self._id(name), 'name': name, 'manifest_path': str(manifest)}
        )
        self.data['resolve']['nodes'].append({'id': self._id(name), 'deps': []})
        if member:
            self.data['workspace_members'].append(self._id(name))

    def edge(self, crate: str, dep: str) -> None:
        self.package(dep, Path('/nonexistent/Cargo.toml'))
        for node in self.data['resolve']['nodes']:
            if node['id'] == self._id(crate):
                node['deps'].append(
                    {
                        'name': dep.replace('-', '_'),
                        'pkg': self._id(dep),
                        'dep_kinds': [{'kind': None, 'target': None}],
                    }
                )


def _binding(tmp: Path, name: str) -> Path:
    manifest = tmp / name / 'Cargo.toml'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        f'[package]\nname = "{name}"\n\n[lints.rust]\nunsafe_code = "deny"\n',
        encoding='utf-8',
    )
    return manifest


def _l0_problems(l0: ModuleType, workspace: Workspace) -> list[str]:
    return [
        problem
        for problems in l0.run_checks(workspace.data, ROOT).values()
        for problem in problems
    ]


def test_seeded_workspace_matches_allowed_dependencies() -> None:
    r"""Not vacuous: the workspace built without cargo passes every L0
    check as the tree is, and an unclassified binding in it fails (the
    brief's first ``l0_deps.py`` example, which passes today too)."""
    l0 = _tool('l0_deps')
    assert _l0_problems(l0, Workspace(l0)) == []


def test_jni_depends_only_on_host_and_contract(
    tmp_path: Path,
) -> None:
    r"""``sdm-jni`` is a binding of ``l0_deps.py``'s own lists, allowed only
    ``sdm-host`` and ``sdm-contract``, and passes absent: a workspace
    holding it on those and ``jni`` passes; with a ``pyo3`` edge, or an
    edge to ``sdm-core``, it fails naming the edge; an unclassified binding
    fails as unclassified."""
    l0 = _tool('l0_deps')
    assert l0.CLASSES.get('sdm-jni') == l0.BINDING
    assert l0.ALLOWED.get('sdm-jni') == frozenset({'sdm-host', 'sdm-contract'})
    absent = Workspace(l0, without=('sdm-jni',))
    assert 'sdm-jni' not in [p['name'] for p in absent.data['packages']]
    assert _l0_problems(l0, absent) == []

    def with_jni(*extra: str) -> list[str]:
        workspace = Workspace(l0, without=('sdm-jni',))
        manifest = _binding(tmp_path / '-'.join(('jni', *extra)), 'sdm-jni')
        workspace.package('sdm-jni', manifest, True)
        for dep in ('sdm-host', 'sdm-contract', 'jni', *extra):
            workspace.edge('sdm-jni', dep)
        return _l0_problems(l0, workspace)

    assert with_jni() == []
    pyo3 = with_jni('pyo3')
    assert any('sdm-jni reaches pyo3' in p for p in pyo3), pyo3
    core = with_jni('sdm-core')
    assert any('sdm-jni -> sdm-core' in p for p in core), core
    other = Workspace(l0, without=('sdm-jni',))
    other.package('sdm-rb', _binding(tmp_path / 'b', 'sdm-rb'), True)
    other.edge('sdm-rb', 'sdm-host')
    problems = _l0_problems(l0, other)
    assert any('sdm-rb: a workspace crate with no class' in p for p in problems)


def test_jni_cannot_contain_provider_specific_words(
    tmp_path: Path,
) -> None:
    r"""``l0_scan.py``'s dialects check holds ``sdm-jni`` as it holds
    ``sdm-py``, once the crate exists, and asks nothing of it before."""
    scan = _tool('l0_scan')
    assert not [p for p in scan.check_dialects(ROOT) if 'sdm-jni' in p]
    for parent in ('rust/crates', 'rust/bindings'):
        shutil.copytree(
            ROOT / parent,
            tmp_path / parent,
            ignore=shutil.ignore_patterns('target', '__pycache__', '.venv'),
        )
    # The copy without sdm-jni, whether or not the tree holds it yet: the
    # check asks nothing of a crate that does not exist.
    for parent in ('rust/crates', 'rust/bindings'):
        if (tmp_path / parent / 'sdm-jni').exists():
            shutil.rmtree(tmp_path / parent / 'sdm-jni')
    assert scan.check_dialects(tmp_path) == []
    source = tmp_path / 'rust/bindings/sdm-jni/src/lib.rs'
    source.parent.mkdir(parents=True)
    source.write_text(
        'pub const DEFAULT_BACKEND: &str = "snowflake";\n', encoding='utf-8'
    )
    (source.parent.parent / 'Cargo.toml').write_text(
        '[package]\nname = "sdm-jni"\n', encoding='utf-8'
    )
    found = scan.check_dialects(tmp_path)
    assert any('sdm-jni/src/lib.rs' in p and 'snowflake' in p for p in found), (
        found
    )


@pytest.mark.parametrize(
    'relative, declaration',
    [
        (
            'dispatch.rs',
            'Operation::AddSqlColumn(request); crate::sql_column::start();',
        ),
        ('describe.rs', 'crate::sql_column::project();'),
        ('lib.rs', 'mod sql_column;'),
        ('materialize/examples.rs', 'ExampleSource::Sql { schema };'),
        ('rebind.rs', 'sdm_contract::data::ProviderChoice::Sql { provider };'),
        (
            'sql_column.rs',
            'use sdm_contract::sql_column::{SqlColumn, SqlColumnCall}; fn f(request: SqlColumn, call: SqlColumnCall) {}',
        ),
        ('view.rs', 'crate::sql_column::project();'),
    ],
)
def test_core_orchestration_references_keep_sql_and_provider_bans(
    tmp_path, monkeypatch, relative, declaration
):
    scan = _tool('l0_scan')
    monkeypatch.setattr(scan, 'SQL_FREE_CRATES', ('sdm-core',))
    path = tmp_path / 'rust/crates/sdm-core/src' / relative
    path.parent.mkdir(parents=True)
    path.write_text(declaration)
    assert scan.check_sql(tmp_path) == []
    for fault in (
        'const Q: &str = "SELECT x FROM y WHERE x = 1";',
        'let provider = Provider::Duckdb;',
        'fn render_sql() {}',
        'OtherSource::Sql { schema };',
        'OtherOperation::AddSqlColumn(request);',
        'use unrelated::{SqlColumn, SqlColumnCall};',
    ):
        path.write_text(declaration + '\n' + fault)
        assert scan.check_sql(tmp_path), fault
    # The same symbols in an unreviewed module must still be rejected.
    path.unlink()
    (path.parent / 'unreviewed.rs').write_text(declaration)
    assert scan.check_sql(tmp_path)
