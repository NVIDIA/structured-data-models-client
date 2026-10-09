#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The L0 self-tests: each guard passes on the tree and fails on a seed.

For every rule of ``l0_deps.py`` and every check of ``l0_scan.py``, this
runs the guard on the tree as it is, which must pass, and on a fixture
seeded with one fault, which must fail with the message that names that
fault; some fixtures are controls, seeded with something the guard must let
through (``skip_while``, a dev-only edge, emulation inside an allowlisted
module). The fixtures are the tree's own files, copied into a temporary
directory and changed there, or ``cargo metadata``'s graph with one edge
added or removed. The clippy bans are also run for real: a fixture crate
under ``rust/clippy.toml`` uses every ban the file holds, one to one (the
allow-invalid rand entries aside, inert while rand is no dependency), and
must fail clippy on each; the reasoned ``#[allow]`` that ``sdm-host``
carries must silence its one call and pass the bans check (decision E24);
and each suppression a review found (``cfg_attr``, ``warnings``, a spaced
path, an ``expect``, an allow passed through a macro, the allow widened
to a function or its statement widened) must silence clippy there and
fail the bans check. That run needs the pinned toolchain and the
workspace's dependencies already fetched; ``--no-clippy`` leaves it out.

Phase 1 design, task T1.10c (formerly T1.16); the rest of ``selftest.py``,
the seeded faults of the runners and the gate, arrives with T1.19.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l0_deps  # noqa: E402
import l0_scan  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
IGNORED = shutil.ignore_patterns('target', '__pycache__', '.venv', '*.pyc')

# The tokens the skip ban refuses, in pieces: this file is in its scope.
IGNORE_ATTR = '#[' + 'ignore]'
PYTEST_SKIP = 'pytest.' + 'skip'
XFAIL = 'x' + 'fail'
SKIP = 'sk' + 'ip'
SKIPIF = 'skip' + 'if'
ALLOWED_TO_FAIL = 'continue-on-' + 'error'


@dataclass
class Result:
    name: str
    ok: bool
    detail: str


# --- fixtures -----------------------------------------------------------------


@contextmanager
def tree(*paths: str) -> Iterator[Path]:
    """A temporary copy of these paths of the repository."""
    with tempfile.TemporaryDirectory(prefix='sdm-l0-') as tmp:
        root = Path(tmp)
        for rel in paths:
            source = REPO_ROOT / rel
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target, ignore=IGNORED)
            elif source.is_file():
                shutil.copy2(source, target)
        yield root


RUST_TREE = (
    'rust/Cargo.toml',
    'rust/clippy.toml',
    'rust/crates',
    'rust/bindings',
    '.github/workflows/ci.yml',
)


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if old not in text:
        raise AssertionError(f'{path}: the fixture expects {old!r}')
    path.write_text(text.replace(old, new, 1))


def append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(text)


class Metadata:
    """``cargo metadata`` for the workspace, copied per fixture."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = copy.deepcopy(data)
        self.ids = {p['name']: p['id'] for p in self.data['packages']}

    def node(self, name: str) -> dict[str, Any]:
        for node in self.data['resolve']['nodes']:
            if node['id'] == self.ids[name]:
                result: dict[str, Any] = node
                return result
        raise KeyError(name)

    def package(self, name: str, manifest: Path | None = None) -> None:
        """Add a third-party package, or move a package's manifest."""
        if name in self.ids:
            for package in self.data['packages']:
                if package['name'] == name and manifest is not None:
                    package['manifest_path'] = str(manifest)
            return
        pid = f'fixture+{name}@0.0.0'
        self.ids[name] = pid
        self.data['packages'].append(
            {
                'id': pid,
                'name': name,
                'manifest_path': str(manifest or '/nonexistent/Cargo.toml'),
            }
        )
        self.data['resolve']['nodes'].append({'id': pid, 'deps': []})

    def member(self, name: str, manifest: Path) -> None:
        self.package(name, manifest)
        self.data['workspace_members'].append(self.ids[name])

    def edge(
        self,
        crate: str,
        dep: str,
        kind: str | None = None,
        name: str | None = None,
    ) -> None:
        self.package(dep)
        self.node(crate)['deps'].append(
            {
                'name': name or dep.replace('-', '_'),
                'pkg': self.ids[dep],
                'dep_kinds': [{'kind': kind, 'target': None}],
            }
        )

    def drop_edge(self, crate: str, dep: str) -> None:
        node = self.node(crate)
        node['deps'] = [d for d in node['deps'] if d['pkg'] != self.ids[dep]]

    def manifest(self, name: str) -> Path:
        for package in self.data['packages']:
            if package['name'] == name:
                return Path(package['manifest_path'])
        raise KeyError(name)


# --- running a guard ----------------------------------------------------------


def expect(
    results: list[Result],
    name: str,
    problems: list[str],
    *,
    fails_with: str | None,
) -> None:
    """Record whether ``problems`` is what the case expects.

    ``fails_with`` is a fragment the problems must hold (the seeded fault
    named); ``None`` means the guard must pass.
    """
    if fails_with is None:
        ok = not problems
        detail = 'passes' if ok else f'should pass, but: {problems[:3]}'
    else:
        ok = any(fails_with in p for p in problems)
        detail = (
            f'fails: {next(p for p in problems if fails_with in p)}'
            if ok
            else f'should fail naming {fails_with!r}, but gave {problems[:3]}'
        )
    results.append(Result(name, ok, detail))


def deps_problems(metadata: Metadata, root: Path = REPO_ROOT) -> list[str]:
    return [
        p for ps in l0_deps.run_checks(metadata.data, root).values() for p in ps
    ]


# --- l0_deps.py ---------------------------------------------------------------


def deps_cases(results: list[Result], base: dict[str, Any]) -> None:
    def case(
        name: str, seed: Callable[[Metadata], object], fails_with: str | None
    ) -> None:
        metadata = Metadata(base)
        seed(metadata)
        expect(
            results,
            f'l0_deps: {name}',
            deps_problems(metadata),
            fails_with=fails_with,
        )

    case('the tree', lambda m: None, None)

    def design_graph(m: Metadata) -> None:
        for crate, deps in l0_deps.ALLOWED.items():
            for dep in deps:
                if not any(
                    d['pkg'] == m.ids[dep] for d in m.node(crate)['deps']
                ):
                    m.edge(crate, dep)

    case('control: every edge of the design graph', design_graph, None)

    with tempfile.TemporaryDirectory(prefix='sdm-l0-') as tmp:
        manifest = Path(tmp) / 'sdm-extra/Cargo.toml'
        manifest.parent.mkdir()
        manifest.write_text(
            '[package]\nname = "sdm-extra"\n\n[lints]\nworkspace = true\n'
        )
        case(
            'an unclassified crate',
            lambda m: m.member('sdm-extra', manifest),
            'sdm-extra: a workspace crate with no class',
        )
    case(
        'a crate below sdm-core on it',
        lambda m: m.edge('sdm-schema', 'sdm-core'),
        'sdm-schema -> sdm-core',
    )
    case(
        'a sideways edge',
        lambda m: m.edge('sdm-engine', 'sdm-data'),
        'sdm-engine -> sdm-data',
    )
    case(
        'sdm-host on a domain crate below sdm-core',
        lambda m: m.edge('sdm-host', 'sdm-engine'),
        'sdm-host -> sdm-engine',
    )
    case(
        'control: sdm-host on sdm-data, which it builds for a run (E30)',
        lambda m: m.edge('sdm-host', 'sdm-data'),
        None,
    )
    case(
        'sdm-py on sdm-core (E21)',
        lambda m: m.edge('sdm-py', 'sdm-core'),
        'sdm-py -> sdm-core',
    )
    case(
        'sdm-py without sdm-contract (E21)',
        lambda m: m.drop_edge('sdm-py', 'sdm-contract'),
        'sdm-py: does not depend on sdm-contract',
    )
    case(
        'the parser from sdm-host',
        lambda m: m.edge('sdm-host', l0_deps.PARSER),
        "only sdm-core's adapter",
    )
    case(
        'a dev edge on the parser',
        lambda m: m.edge('sdm-request', l0_deps.PARSER, 'dev'),
        'sdm-request -> sdm-pql-interim (dev)',
    )
    case(
        'sdm-host on sdm-testkit',
        lambda m: m.edge('sdm-host', 'sdm-testkit'),
        'sdm-host reaches sdm-testkit',
    )

    def through_third_party(m: Metadata) -> None:
        m.package('some-helper')
        m.edge('some-helper', 'sdm-testkit')
        m.edge('sdm-wire', 'some-helper')

    case(
        'a shipped crate reaching sdm-testkit through another crate',
        through_third_party,
        'sdm-wire reaches sdm-testkit (sdm-wire -> some-helper -> sdm-testkit)',
    )
    for crate, dep in (
        ('sdm-engine', 'pyo3'),
        ('sdm-schema', 'jni'),
        ('sdm-request', 'arrow-pyarrow'),
        ('sdm-data', 'numpy'),
        ('sdm-core', 'duckdb'),
        ('sdm-host', 'pyo3'),
    ):

        def seed(m: Metadata, crate: str = crate, dep: str = dep) -> None:
            m.edge(crate, dep)

        case(f'{crate} on {dep}', seed, f'{crate} reaches {dep}')

    def transitive_pyo3(m: Metadata) -> None:
        m.package('helper')
        m.edge('helper', 'pyo3', 'build')
        m.edge('sdm-contract', 'helper')

    case(
        'a domain crate reaching pyo3 over a build edge',
        transitive_pyo3,
        'sdm-contract reaches pyo3',
    )
    case(
        'control: a dev-only edge on duckdb',
        lambda m: m.edge('sdm-wire', 'duckdb', 'dev'),
        None,
    )
    case(
        'sdm-py on duckdb',
        lambda m: m.edge('sdm-py', 'duckdb'),
        'sdm-py reaches duckdb',
    )

    # A future Java binding (sdm-jni) links jni, and no Python.
    with tempfile.TemporaryDirectory(prefix='sdm-l0-') as tmp:
        manifest = Path(tmp) / 'sdm-jni/Cargo.toml'
        manifest.parent.mkdir()
        manifest.write_text(
            '[package]\nname = "sdm-jni"\n\n[lints.rust]\nunsafe_code = "deny"\n'
        )
        classes, allowed = dict(l0_deps.CLASSES), dict(l0_deps.ALLOWED)
        l0_deps.CLASSES['sdm-jni'] = l0_deps.BINDING
        l0_deps.ALLOWED['sdm-jni'] = frozenset({'sdm-host', 'sdm-contract'})
        try:

            def jni(m: Metadata, extra: str | None = None) -> None:
                m.member('sdm-jni', manifest)
                for dep in ('sdm-host', 'sdm-contract', 'jni'):
                    m.edge('sdm-jni', dep)
                if extra:
                    m.edge('sdm-jni', extra)

            case('control: a Java binding, sdm-jni, on jni', jni, None)
            case(
                'a Java binding on pyo3',
                lambda m: jni(m, 'pyo3'),
                'sdm-jni reaches pyo3',
            )
            case(
                'a Java binding on duckdb',
                lambda m: jni(m, 'duckdb'),
                'sdm-jni reaches duckdb',
            )
        finally:
            l0_deps.CLASSES.clear()
            l0_deps.CLASSES.update(classes)
            l0_deps.ALLOWED.clear()
            l0_deps.ALLOWED.update(allowed)

    # The parser rule's source half, and the unsafe check: crates copied.
    def with_crate(
        name: str,
        crate: str,
        change: Callable[[Path], object],
        fails_with: str,
        seed: Callable[[Metadata], object] = lambda m: None,
    ) -> None:
        with tree(f'rust/crates/{crate}', 'rust/bindings/sdm-py') as root:
            metadata = Metadata(base)
            seed(metadata)
            directory = root / 'rust/crates' / crate
            if not directory.is_dir():
                directory = root / 'rust/bindings' / crate
            metadata.package(crate, directory / 'Cargo.toml')
            change(directory)
            expect(
                results,
                f'l0_deps: {name}',
                deps_problems(metadata),
                fails_with=fails_with,
            )

    with_crate(
        'the parser named outside the adapter',
        'sdm-core',
        lambda d: append(
            d / 'src/dispatch.rs', '\nuse sdm_pql_interim::parser;\n'
        ),
        'sdm-core/src/dispatch.rs: names sdm_pql_interim',
    )

    def renamed(m: Metadata) -> None:
        m.drop_edge('sdm-core', l0_deps.PARSER)
        m.edge('sdm-core', l0_deps.PARSER, name='pqlp')

    with_crate(
        'the parser renamed (pqlp = { package = "sdm-pql-interim" })',
        'sdm-core',
        lambda d: append(d / 'src/dispatch.rs', '\nuse pqlp::parse;\n'),
        'sdm-core/src/dispatch.rs: names pqlp, the parser',
        seed=renamed,
    )
    with_crate(
        "the parser named in sdm-core's tests",
        'sdm-core',
        lambda d: (d / 'tests/parser.rs').write_text(
            'use sdm_pql_interim::parse;\n'
        ),
        'sdm-core/tests/parser.rs: names sdm_pql_interim',
    )
    with_crate(
        'test_ops compiled into the shipped build',
        'sdm-core',
        lambda d: edit(
            d / 'src/lib.rs',
            '#[cfg(feature = "test-ops")]\nmod test_ops;',
            'mod test_ops;',
        ),
        'is not compiled only with the test-ops feature',
    )
    with_crate(
        'unsafe: a domain root without forbid',
        'sdm-engine',
        lambda d: edit(d / 'src/lib.rs', '#![forbid(unsafe_code)]', ''),
        'sdm-engine: its root does not forbid unsafe_code',
    )
    with_crate(
        'unsafe: a domain crate off the workspace lints',
        'sdm-schema',
        lambda d: edit(
            d / 'Cargo.toml',
            '[lints]\nworkspace = true',
            '[lints.rust]\nunsafe_code = "allow"',
        ),
        'sdm-schema: does not take exactly the workspace lints',
    )
    with_crate(
        'unsafe: sdm-host without forbid',
        'sdm-host',
        lambda d: edit(
            d / 'src/lib.rs', '#![forbid(unsafe_code)]', '#![deny(unsafe_code)]'
        ),
        'sdm-host: its root does not forbid unsafe_code',
    )
    with_crate(
        'unsafe: the binding allowing it',
        'sdm-py',
        lambda d: edit(
            d / 'Cargo.toml', 'unsafe_code = "deny"', 'unsafe_code = "allow"'
        ),
        'sdm-py: unsafe_code is neither denied nor forbidden',
    )
    with tree('rust/Cargo.toml') as root:
        metadata = Metadata(base)
        metadata.data['workspace_root'] = str(root / 'rust')
        edit(
            root / 'rust/Cargo.toml',
            'unsafe_code = "forbid"',
            'unsafe_code = "warn"',
        )
        expect(
            results,
            'l0_deps: unsafe: the workspace lints not forbidding it',
            deps_problems(metadata),
            fails_with='the workspace lints do not forbid unsafe_code',
        )

    # Bridge headers.
    client = 'packages/structured-data-models-client/src'

    def bridges(
        name: str, change: Callable[[Path], object], fails_with: str | None
    ) -> None:
        with tree(client, 'rust/crates', 'rust/bindings') as root:
            change(root)
            problems = l0_deps.check_bridges(
                root, root / 'rust/crates' / l0_deps.PARSER
            )
            expect(
                results,
                f'l0_deps: bridges: {name}',
                problems,
                fails_with=fails_with,
            )

    bridges('the tree, copied', lambda r: None, None)
    routing = l0_deps.BRIDGE_FILES[0]
    bridges(
        'a bridge file without its header',
        lambda r: edit(
            r / routing, '# Bridge code: deleted in Phase 5', '# The seam'
        ),
        f'{routing}: no header naming the phase',
    )
    bridges(
        'a bridge header without its phase',
        lambda r: edit(
            r / l0_deps.BRIDGE_FILES[1], 'is deleted in Phase 5', 'goes later'
        ),
        f'{l0_deps.BRIDGE_FILES[1]}: no header naming the phase',
    )
    bridges(
        'SDM_CORE_ROUTES read outside the bridge files',
        lambda r: append(
            r / client / 'sdm_client/_rust/extra.py',
            "import os\n\nROUTES = os.environ.get('SDM_CORE_ROUTES')\n",
        ),
        'sdm_client/_rust/extra.py: reads SDM_CORE_ROUTES',
    )
    bridges(
        'a bridge header on the parser (E18)',
        lambda r: edit(
            r / 'rust/crates' / l0_deps.PARSER / 'src/lib.rs',
            '// SPDX-License-Identifier: Apache-2.0\n',
            '// SPDX-License-Identifier: Apache-2.0\n// Bridge code: deleted in Phase 5.\n',
        ),
        'the parser is permanent, not bridge code (E18)',
    )


# --- l0_scan.py ---------------------------------------------------------------


def scan_case(
    results: list[Result],
    check: str,
    name: str,
    paths: tuple[str, ...],
    change: Callable[[Path], object],
    fails_with: str | None,
) -> None:
    with tree(*paths) as root:
        change(root)
        problems = l0_scan.CHECKS[check](root)
        expect(
            results, f'l0_scan {check}: {name}', problems, fails_with=fails_with
        )


def scan_cases(results: list[Result]) -> None:
    for check in l0_scan.CHECKS:
        expect(
            results,
            f'l0_scan {check}: the tree',
            l0_scan.CHECKS[check](REPO_ROOT),
            fails_with=None,
        )

    vocab = ('rust/crates', 'scripts/rust_core/emulation-allowlist.txt')
    crates = 'rust/crates'

    def vocabulary(
        name: str, change: Callable[[Path], object], fails_with: str | None
    ) -> None:
        scan_case(results, 'vocabulary', name, vocab, change, fails_with)

    vocabulary(
        'a repr helper in sdm-core',
        lambda r: append(
            r / crates / 'sdm-core/src/run.rs',
            '\nfn py_repr(x: f64) -> String { x.to_string() }\n',
        ),
        '`py_repr` is Python-emulation vocabulary',
    )
    vocabulary(
        'a pandas dtype name in sdm-host',
        lambda r: append(
            r / crates / 'sdm-host/src/admission.rs',
            '\nconst NAME: &str = "pandas object";\n',
        ),
        'a string names `pandas`',
    )
    vocabulary(
        'a dunder method in sdm-engine',
        lambda r: append(
            r / crates / 'sdm-engine/src/lib.rs',
            '\npub fn __bool__() -> bool { true }\n',
        ),
        '`__bool__` is Python-emulation vocabulary',
    )
    vocabulary(
        'a truthiness rule in sdm-schema',
        lambda r: append(
            r / crates / 'sdm-schema/src/lib.rs',
            '\npub fn is_truthy(x: i64) -> bool { x != 0 }\n',
        ),
        '`is_truthy` is Python-emulation vocabulary',
    )
    vocabulary(
        'an allowlisted name outside the parser crate',
        lambda r: append(
            r / crates / 'sdm-core/src/run.rs',
            '\nfn f(_: sdm_pql_interim::pyint::PyInt) {}\n',
        ),
        '`PyInt` is Python-emulation vocabulary',
    )
    vocabulary(
        'a new emulation module in the parser',
        lambda r: (r / crates / l0_deps.PARSER / 'src/pystr.rs').write_text(
            'pub fn py_str() {}\n'
        ),
        '`py_str` is Python-emulation vocabulary',
    )
    vocabulary(
        'an allowlist entry outside the parser crate',
        lambda r: append(
            r / l0_scan.ALLOWLIST, 'rust/crates/sdm-core/src/run.rs\n'
        ),
        'is outside rust/crates/sdm-pql-interim/src',
    )
    vocabulary(
        'an allowlist entry that does not exist',
        lambda r: append(
            r / l0_scan.ALLOWLIST, 'rust/crates/sdm-pql-interim/src/gone.rs\n'
        ),
        'gone.rs does not exist',
    )

    def emulation_where_allowed(root: Path) -> None:
        append(
            root / crates / l0_deps.PARSER / 'src/float_repr.rs',
            '\nfn py_repr_helper() {}\n',
        )
        append(
            root / crates / 'sdm-core/src/run.rs',
            '\n// py_repr, as pandas does\n#[cfg(test)]\nmod more_tests {\n'
            '    fn as_python_does() {}\n}\n',
        )

    vocabulary(
        'control: emulation inside an allowlisted module, comments and tests',
        emulation_where_allowed,
        None,
    )

    def sql(
        name: str, change: Callable[[Path], object], fails_with: str | None
    ) -> None:
        scan_case(results, 'sql', name, (crates,), change, fails_with)

    sql(
        'a SELECT in sdm-engine',
        lambda r: append(
            r / crates / 'sdm-engine/src/lib.rs',
            '\nconst Q: &str = "SELECT a FROM t";\n',
        ),
        'hold `SELECT a FROM t`',
    )
    sql(
        'lowercase SQL in sdm-schema',
        lambda r: append(
            r / crates / 'sdm-schema/src/lib.rs',
            '\nconst Q: &str = "select a from t";\n',
        ),
        'hold `select a from t`',
    )
    sql(
        'a provider match in sdm-request',
        lambda r: append(
            r / crates / 'sdm-request/src/lib.rs',
            '\npub fn f(d: sdm_contract::data::Dialect) -> bool { matches!(d, sdm_contract::data::Dialect::Duckdb) }\n',
        ),
        '`Duckdb` names a provider or SQL',
    )
    sql(
        'control: SQL in sdm-data',
        lambda r: append(
            r / crates / 'sdm-data/src/lib.rs',
            '\nconst Q: &str = "SELECT a FROM t";\n',
        ),
        None,
    )

    def test_module_file(root: Path) -> None:
        append(
            root / crates / 'sdm-engine/src/lib.rs',
            '\n#[cfg(test)]\nmod query_tests;\n',
        )
        (root / crates / 'sdm-engine/src/query_tests.rs').write_text(
            'const Q: &str = "SELECT a FROM t WHERE b = 1";\n'
        )

    sql('control: SQL in a test module of its own file', test_module_file, None)

    # SQL's unambiguous shapes outside diagnostic text (decision E28):
    # `bare` is a message built outside any diagnostic, `error` the same
    # text as a CoreError's message.
    def bare(text: str) -> str:
        return f'pub fn m(c: &str) -> String {{ format!("{text}") }}'

    def error(text: str) -> str:
        return (
            'pub fn e(c: &str) -> sdm_contract::CoreError {\n'
            '    sdm_contract::CoreError::new(\n'
            '        sdm_contract::ErrorCode::InvalidRequest,\n'
            f'        format!("{text}"),\n'
            '    )\n}'
        )

    engine = 'sdm-engine/src/lib.rs'
    held = 'the strings of a statement hold'
    # Verified helpers: free functions of the crate that build a CoreError.
    helpers = (
        'fn invalid(m: impl Into<String>) -> sdm_contract::CoreError { todo!() }\n'
        'fn internal(m: &str) -> CoreError { todo!() }\n'
        'fn protocol(c: u8, m: String) -> CoreError { todo!() }\n'
    )
    failing = [
        ('const Q: &str = "cast(t as timestamp)";', 'cast(t as'),
        ('const Q: &str = "count(*) from";', 'count(*'),
        (
            'pub fn f(t: &str) -> String { ["select", "*", "from", t].join(" ") }',
            'select * from',
        ),
        (
            'const Q: &str = concat!("select ", "x", " from ", "t");',
            'select x from',
        ),
        (
            'pub fn f() -> String { format!("{} {}", "select x", "from t") }',
            'select x from',
        ),
        (
            'pub fn f(t: &str) -> String { "select a ".to_owned() + "from " + t }',
            'select a from',
        ),
        ('const B: &str = "duckdb";', '`duckdb`'),
        ('pub const S: &str = "Spark";', 'hold `Spark`'),
        ('const Q: &str = include_str!("q.sql");', 'include_str'),
        (bare('drop table {t}'), 'drop table {t}'),
        (bare('query failed: SELECT {c} FROM {t}'), 'SELECT {c} FROM {t}'),
        (
            bare('select max(a) as m from {t} where b'),
            'select max(a) as m from {t} where',
        ),
        (bare('insert into {t} (a, b) values'), 'insert into {t} ('),
        (bare('delete from {t} where a = 1'), 'delete from {t} where'),
        (bare('create table {t} (a int)'), 'create table {t} ('),
        (bare('cast({c} as timestamp)'), 'cast({c} as timestamp)'),
        (bare('with q as (select 1)'), 'with q as (select'),
        (bare('coalesce({c}, 0)'), 'coalesce({c},'),
        (bare('where exists (select 1)'), 'exists (select'),
        (bare('values (?, ?)'), 'values (?,'),
        ('pub const Q: &str = "SELECT 1";', 'SELECT 1'),
        ('pub const Q: &str = r#"SELECT\n  a\nFROM t"#;', 'SELECT a FROM t'),
        *(
            (bare(f'select a from {{t}}{tail}'), 'select a from {t}')
            for tail in (
                ' natural join u',
                ', u',
                ' qualify x',
                ' having x',
                ' union select b',
                ' limit 3',
                ' right join u',
                ' full join u',
                ' cross join u',
                ' using (k)',
            )
        ),
        (bare('select a from ({t})'), 'select a from'),
        (bare('select * from {t} t'), 'select * from {t} t'),
        (
            bare('select {c} from {t} t where t.k = 1'),
            'select {c} from {t} t where',
        ),
        (
            bare('select a, b from {t} t join u on x'),
            'select a, b from {t} t join',
        ),
        # Every shape decision E28 lists.
        (bare('truncate table {t}'), 'truncate table {t}'),
        (bare('update {t} set a = 1'), 'update {t} set a ='),
        (bare('{a} union all {b}'), 'union all'),
        (bare('from information_schema.columns'), 'information_schema'),
        (bare('from {t} tablesample (10 percent)'), 'tablesample'),
        (bare('from {t} using sample 10 rows'), 'using sample'),
        (bare('row_number() over (order by a)'), 'row_number() over'),
        (bare('rank over (partition by {c})'), 'over (partition by'),
        (bare('insert into {t} select * from u'), 'insert into {t} select'),
        (bare('alter table {t} add column a int'), 'alter table {t} add'),
        (bare('alter table {t} rename to u'), 'alter table {t} rename'),
        # Unverified helpers exempt nothing (fix round 1 of T1.10d): a local
        # function of a helper's name returning text, a path elsewhere, a
        # builder's with_field, an operation returning Result, and a call
        # nested in a verified helper's message.
        *(
            (
                f'fn {name}(m: &str) -> String {{ m.into() }}\n'
                f'pub fn q() -> String {{ {name}("select a from t where x = 1") }}',
                held,
            )
            for name in ('internal', 'protocol', 'refused')
        ),
        (
            f'{helpers}pub fn q() -> CoreError {{ text::invalid("select a from t where x = 1") }}',
            held,
        ),
        (
            'pub fn q(b: Builder) -> Builder { b.with_field("k", "select a from t where x = 1") }',
            held,
        ),
        (
            'pub fn q() -> Builder { Builder::new().with_field("k", "select a from t where x = 1") }',
            held,
        ),
        (
            'fn read(q: &str) -> Result<u8, CoreError> { todo!() }\n'
            'pub fn q() -> Result<u8, CoreError> { read("select a from t where x = 1") }',
            held,
        ),
        (
            f'{helpers}pub fn q() -> CoreError {{ internal(&run("select a from t where x = 1")) }}',
            held,
        ),
        (
            'pub fn q() -> CoreError { CoreError::new(K, C, "m").with_field("k", run("select a from t where x = 1")) }',
            held,
        ),
        # Escapes decoded and continuations joined (fix round 4).
        (bare('select {c}\\nfrom {t}\\nwhere {c} is not null'), held),
        (bare('select {c}\\tfrom {t}'), held),
        (bare('select {c}\\r\\nfrom {t} where x'), held),
        (bare('select {c} \\\n    from {t}'), held),
        (bare('select\\x20{c}\\x20from {t}'), held),
        (bare('select\\u{20}*\\u{20}from {t}'), held),
        ('pub const Q: &[u8] = b"select a\\nfrom t";', held),
        # An assertion's condition and operands are code, past a turbofish
        # whose commas separate nothing; a bare `<` makes every argument
        # code (fix round 4).
        (
            'pub fn f(t: u8, y: String) { assert!(same::<String, String>(&format!("SELECT * FROM {t} WHERE x"), y)); }',
            'SELECT * FROM {t} WHERE',
        ),
        (
            'pub fn f(t: u8, y: u8) { assert!(same::<Vec<(A, B)>, C>(&format!("SELECT * FROM {t} WHERE x"), y)); }',
            'SELECT * FROM {t} WHERE',
        ),
        (
            'pub fn f(t: u8, x: u8) { assert_eq!(f::<A, B>(format!("SELECT * FROM {t} WHERE x")), x, "msg"); }',
            'SELECT * FROM {t} WHERE',
        ),
        (
            'pub fn f(t: u8) { assert!(same :: < A, B > (format!("SELECT * FROM {t} WHERE x")), "ok"); }',
            'SELECT * FROM {t} WHERE',
        ),
        (
            'pub fn f(t: u8) { assert!(g::<fn(A) -> B, C>(format!("SELECT * FROM {t} WHERE x")), "ok"); }',
            'SELECT * FROM {t} WHERE',
        ),
        (
            'pub fn f(t: u8) { assert!(check(|m: HashMap<String, u8>| format!("select * from {t} where x")), "ok"); }',
            'select * from {t} where',
        ),
        (
            'pub fn f(a: u8, t: u8) { assert_eq!(format!("SELECT {a} FROM {t}"), "x", "the select"); }',
            'SELECT {a} FROM {t}',
        ),
    ]
    english = (
        'required field is null',
        'the seed time in column {c} is null',
        'could not select a column from {c}',
        'order by time is required',
        'came from a Spark job',
        '{column} >= {lower}',
        '{c} > 0 and {c} is not null',
        'select {c} from {t} first',
    )
    passing = [
        *(code for text in english for code in (bare(text), error(text))),
        # Diagnostic text is exempt entirely, through verified helpers too
        # (decision E28): the crate's own functions returning CoreError or
        # Detail, the error types' own, and chains starting at them.
        error('select * from {t} where x = 1'),
        'pub fn p(t: &str) { panic!("select * from {t} where x = 1") }',
        'pub fn f(v: Option<u8>) { v.expect("SELECT 1"); }',
        'pub fn f(a: u8) { assert!(a > 0, "select a from t where a = {a}"); }',
        f'{helpers}pub fn f() -> CoreError {{ invalid("expected a window where start <= end") }}',
        f'{helpers}pub fn f() -> CoreError {{ invalid("select a from t where x = 1") }}',
        f'{helpers}pub fn f(lo: u8, hi: u8) -> CoreError {{ let m = format!("window start {{lo}} > end {{hi}}"); invalid(m) }}',
        'pub fn f(e: CoreError) -> CoreError { e.with_field("rule", Detail::text("by time desc, then key, nulls last")) }',
        'pub fn f() -> CoreError { CoreError::new(K, C, "m").with_field("rule", Detail::text("select a from t where x = 1")) }',
        f'{helpers}pub fn f() -> CoreError {{ internal("m").caused_by(invalid("x")).with_field("q", Detail::text("select a from t where x = 1")) }}',
        'pub fn f(c: &str) -> CoreError { CoreError::new(K, C, format!("{c} > 0 and {c} is not null in the window")) }',
        f'{helpers}pub fn f(c: &str) -> CoreError {{ protocol(C, format!("select {{c}} from {{t}} where it is set")) }}',
        '#[cfg(test)]\nmod more_tests {\n    const Q: &str = "SELECT a FROM t WHERE b = 1";\n}',
    ]
    for number, code in enumerate(passing):

        def seed_pass(root: Path, code: str = code) -> None:
            append(root / crates / engine, f'\n{code}\n')

        sql(f'control {number}: {" ".join(code.split())[:70]}', seed_pass, None)
    for code, fragment in failing:

        def seed_fail(root: Path, code: str = code) -> None:
            append(root / crates / engine, f'\n{code}\n')

        sql(f'SQL: {" ".join(code.split())[:70]}', seed_fail, fragment)

    skips_tree = (
        'rust/crates/sdm-contract',
        'rust/bindings/sdm-py/tests',
        'scripts/rust_core',
        'contract/tests',
        'packages/structured-data-models-client/tests',
        'packages/structured-data-models-client/tests_test_ops',
        '.github/workflows/ci.yml',
    )
    client_tests = 'packages/structured-data-models-client/tests'

    def skips(
        name: str, change: Callable[[Path], object], fails_with: str | None
    ) -> None:
        scan_case(results, 'skips', name, skips_tree, change, fails_with)

    skips('the tree, copied', lambda r: None, None)
    skips(
        'an ignored Rust test',
        lambda r: append(
            r / 'rust/crates/sdm-contract/tests/types.rs',
            f'\n{IGNORE_ATTR}\n#[test]\nfn later() {{}}\n',
        ),
        f'`{IGNORE_ATTR}`',
    )
    skips(
        'an ignored Rust test with a reason',
        lambda r: append(
            r / 'rust/crates/sdm-contract/tests/types.rs',
            '\n#[' + 'ignore = "slow"]\n#[test]\nfn later() {}\n',
        ),
        f'`{IGNORE_ATTR}`',
    )
    skips(
        'an ignore by cfg_attr',
        lambda r: append(
            r / 'rust/crates/sdm-contract/tests/types.rs',
            '\n#[cfg_attr(miri, ' + 'ign' + 'ore)]\n#[test]\nfn later() {}\n',
        ),
        'cfg_attr',
    )
    skips(
        "a skip in the binding's tests",
        lambda r: append(
            r / 'rust/bindings/sdm-py/tests/test_binding.py',
            f'\n{PYTEST_SKIP}("later")\n',
        ),
        f'`{PYTEST_SKIP}`',
    )
    skips(
        'an expected failure in a client test of _native',
        lambda r: append(
            r / client_tests / 'test_the_routing_seam.py',
            f'\n@pytest.mark.{XFAIL}\ndef test_later():\n    pass\n',
        ),
        f'`{XFAIL}`',
    )
    skips(
        'a skip marker with a condition in a client test of _rust',
        lambda r: (r / client_tests / 'test_fixture_core.py').write_text(
            'import pytest\nfrom sdm_client import _rust\n\n'
            f'@pytest.mark.{SKIPIF}(True, reason="x")\ndef test_x():\n    pass\n'
        ),
        f'`{SKIPIF}`',
    )
    skips(
        "a skip in the conftest over the client's tests",
        lambda r: append(
            r / client_tests / 'conftest.py',
            f'\nimport pytest\n{PYTEST_SKIP}("x", allow_module_level=True)\n',
        ),
        f'{client_tests}/conftest.py',
    )
    skips(
        'control: a skip in a client test of neither',
        lambda r: (r / client_tests / 'test_fixture_other.py').write_text(
            f'import pytest\n\n\ndef test_x():\n    {PYTEST_SKIP}("x")\n'
        ),
        None,
    )
    skips(
        'an expected failure in the scripts',
        lambda r: append(
            r / 'scripts/rust_core/compare.py', f'\n# @pytest.mark.{XFAIL}\n'
        ),
        f'`{XFAIL}`',
    )
    skips(
        'an import-or-skip in the contract tests',
        lambda r: append(
            r / 'contract/tests/test_runners.py',
            f'\nduckdb = pytest.importor{SKIP}("duckdb")\n',
        ),
        'importor',
    )
    skips(
        'control: skip_while and schemars(skip)',
        lambda r: append(
            r / 'rust/crates/sdm-contract/src/lib.rs',
            '\n#[derive(schemars::JsonSchema)]\nstruct S { #[schemars(skip)] a: u8 }\n'
            'fn f(v: &[u8]) -> usize { v.iter().skip_while(|x| **x == 0).count() }\n',
        ),
        None,
    )
    for name, path, text, token in (
        (
            'a test deselected in a Rust job',
            'contract/tests/test_runners.py',
            '\n# run as: pytest --de'
            + 'select contract/tests/test_runners.py::test_x\n',
            '--de' + 'select',
        ),
        (
            'a test file ignored by path',
            'rust/bindings/sdm-py/tests/test_binding.py',
            '\n# pytest --ig'
            + 'nore=rust/bindings/sdm-py/tests/test_binding.py\n',
            '--ig' + 'nore',
        ),
        (
            'a conftest dropping a test file',
            f'{client_tests}/conftest.py',
            '\ncollect_ig' + 'nore = ["test_the_routing_seam.py"]\n',
            'collect_ig' + 'nore',
        ),
    ):

        def seed_deselect(
            root: Path, path: str = path, text: str = text
        ) -> None:
            append(root / path, text)

        skips(name, seed_deselect, f'`{token}`')
    skips(
        'a skip marker through an imported mark',
        lambda r: append(
            r / client_tests / 'test_the_routing_seam.py',
            f'\nfrom pytest import mark\n\n\n@mark.{SKIP}\ndef test_later():\n    pass\n',
        ),
        f'`mark.{SKIP}`',
    )
    skips(
        'an expected failure by unittest',
        lambda r: append(
            r / client_tests / 'test_the_routing_seam.py',
            '\nimport unittest\n\n\n@unittest.expected'
            + 'Failure\ndef test_later():\n    pass\n',
        ),
        '`expected' + 'Failure`',
    )
    skips(
        'a skip through an alias of pytest',
        lambda r: append(
            r / 'rust/bindings/sdm-py/tests/test_binding.py',
            f'\nimport pytest as pt\n\n\ndef test_later():\n    pt.{SKIP}("x")\n',
        ),
        f'`{SKIP}(`',
    )
    skips(
        "control: Rust's iterator skip",
        lambda r: append(
            r / 'rust/crates/sdm-contract/src/lib.rs',
            f'\nfn g(v: &[u8]) -> usize {{ v.iter().{SKIP}(1).count() }}\n',
        ),
        None,
    )

    def ci(job: str, after: str) -> Callable[[Path], object]:
        def change(root: Path) -> None:
            edit(
                root / l0_scan.CI_FILE,
                after,
                after + f'\n        {ALLOWED_TO_FAIL}: true',
            )

        return change

    skips(
        'a Rust job step allowed to fail',
        ci('rust-test', '      - run: cargo test --workspace --locked'),
        f'`{ALLOWED_TO_FAIL}` in job rust-test',
    )
    skips(
        "the client's test job allowed to fail, at job level",
        lambda r: edit(
            r / l0_scan.CI_FILE,
            '  test:\n    runs-on: ubuntu-latest\n',
            f'  test:\n    runs-on: ubuntu-latest\n    {ALLOWED_TO_FAIL}: true\n',
        ),
        f'`{ALLOWED_TO_FAIL}` in job test',
    )
    skips(
        'a step of installed-wheel allowed to fail',
        lambda r: edit(
            r / l0_scan.CI_FILE,
            '  installed-wheel:\n    runs-on: ubuntu-latest\n    needs: build\n'
            '    steps:\n      - uses: actions/checkout@v7\n',
            '  installed-wheel:\n    runs-on: ubuntu-latest\n    needs: build\n'
            f'    steps:\n      - uses: actions/checkout@v7\n        {ALLOWED_TO_FAIL}: true\n',
        ),
        f'`{ALLOWED_TO_FAIL}` in job installed-wheel',
    )
    skips(
        "the contract cases' step allowed to fail",
        ci('engine', '          python -m pytest contract/tests -q'),
        f'`{ALLOWED_TO_FAIL}` in job engine',
    )
    skips(
        'control: a step allowed to fail outside the Rust and contract jobs',
        ci('lint', '      - run: ruff check .'),
        None,
    )

    def bans(
        name: str, change: Callable[[Path], object], fails_with: str | None
    ) -> None:
        scan_case(results, 'bans', name, RUST_TREE, change, fails_with)

    bans('the tree, copied', lambda r: None, None)
    bans(
        'the HashMap ban removed',
        lambda r: edit(
            r / 'rust/clippy.toml',
            '{ path = "std::collections::HashMap"',
            '{ path = "std::collections::BTreeMap"',
        ),
        'the type ban on std::collections::HashMap is missing',
    )
    bans(
        'the zone database ban removed',
        lambda r: edit(
            r / 'rust/clippy.toml',
            '{ path = "jiff::tz::db"',
            '{ path = "jiff::tz::nothing"',
        ),
        'the method ban on jiff::tz::db is missing',
    )
    bans(
        'sdm-py not repeating a ban',
        lambda r: edit(
            r / 'rust/bindings/sdm-py/clippy.toml',
            '{ path = "std::env::var", ',
            '{ path = "std::env::vars_fixture", ',
        ),
        "does not repeat the workspace's method ban on std::env::var",
    )
    bans(
        'a domain crate with a clippy.toml of its own',
        lambda r: (r / 'rust/crates/sdm-engine/clippy.toml').write_text(
            'msrv = "1.97"\n'
        ),
        'rust/crates/sdm-engine/clippy.toml: does not repeat',
    )
    bans(
        'a second allow in sdm-host',
        lambda r: append(
            r / 'rust/crates/sdm-host/src/zones.rs',
            '\n#[allow(clippy::disallowed_methods, reason = "x")]\nfn g() {}\n',
        ),
        'sdm-host/src/zones.rs:',
    )
    bans(
        'an allow in a domain crate',
        lambda r: append(
            r / 'rust/crates/sdm-core/src/run.rs',
            '\n#[allow(clippy::disallowed_types, reason = "x")]\nfn g() {}\n',
        ),
        'sdm-core/src/run.rs',
    )
    bans(
        'a crate-wide allow in sdm-host',
        lambda r: edit(
            r / 'rust/crates/sdm-host/src/lib.rs',
            '#![forbid(unsafe_code)]',
            '#![forbid(unsafe_code)]\n#![allow(clippy::disallowed_methods, reason = "x")]',
        ),
        'sdm-host/src/lib.rs',
    )
    bans(
        'the style group allowed in the binding',
        lambda r: append(
            r / 'rust/bindings/sdm-py/src/lib.rs',
            '\n#[allow(clippy::style)]\nfn g() {}\n',
        ),
        'shipped code allows clippy::style',
    )
    bans(
        'a test file allowing a ban without a reason',
        lambda r: append(
            r / 'rust/crates/sdm-core/tests/fixture.rs',
            '#![allow(clippy::disallowed_methods)]\n',
        ),
        'allows a clippy ban without a reason',
    )
    bans(
        'a lint table turning a ban off',
        lambda r: edit(
            r / 'rust/Cargo.toml',
            'all = { level = "warn", priority = -1 }',
            'all = { level = "warn", priority = -1 }\ndisallowed_methods = "allow"',
        ),
        'clippy::disallowed_methods set to allow',
    )
    run = 'rust/crates/sdm-core/src/run.rs'
    for name, code in (
        (
            'an inner allow inside cfg_attr',
            '#![cfg_attr(all(), allow(clippy::disallowed_types))]',
        ),
        (
            'an allow inside cfg_attr',
            '#[cfg_attr(all(), allow(clippy::disallowed_methods))]\nfn g() {}',
        ),
        ('allow(warnings)', '#[allow(warnings)]\nfn g() {}'),
        ('a spaced path', '#[allow(clippy :: disallowed_methods)]\nfn g() {}'),
        (
            'an expect inside cfg_attr',
            '#[cfg_attr(all(), expect(clippy::disallowed_methods))]\nfn g() {}',
        ),
    ):

        def seed_ban(root: Path, code: str = code) -> None:
            append(root / run, f'\n{code}\n')

        bans(f'shipped code: {name}', seed_ban, f'{run}:')
    bans(
        'an allow passed through a macro',
        lambda r: append(
            r / run,
            '\nmacro_rules! wa { ($a:meta, $i:item) => { #[$a] $i } }\n'
            'wa!(allow(clippy::disallowed_methods), fn e() { let _ = std::env::var("X"); });\n',
        ),
        'outside an attribute',
    )
    for name, code in (
        (
            'a lint passed to an allow as a macro metavariable',
            'macro_rules! permit { ($lint:path, $item:item) => { #[allow($lint)] $item } }\n'
            'permit!(clippy::disallowed_methods, pub fn environment() '
            '{ let _ = std::env::var("SYNTHETIC"); });',
        ),
        (
            "a lint's name passed as a macro metavariable",
            'macro_rules! permit { ($l:ident, $item:item) => { #[allow(clippy::$l)] $item } }\n'
            'permit!(disallowed_methods, pub fn environment() '
            '{ let _ = std::env::var("SYNTHETIC"); });',
        ),
        (
            'a whole attribute as a macro metavariable',
            'macro_rules! wa { ($a:meta, $i:item) => { #[$a] $i } }',
        ),
        (
            'an attribute built by cfg_attr from a metavariable',
            'macro_rules! wa { ($a:meta, $i:item) => { #[cfg_attr(all(), $a)] $i } }',
        ),
        (
            'a lint named in capitals',
            '#[allow(clippy::DISALLOWED_METHODS)]\nfn g() {}',
        ),
    ):

        def seed_macro(root: Path, code: str = code) -> None:
            append(root / run, f'\n{code}\n')

        bans(name, seed_macro, f'{run}:')

    for name, code in (
        (
            'a module read from outside src',
            '#[path = "../tests/helper.rs"]\nmod helper;',
        ),
        ('code included from outside src', 'include!("../tests/helper.rs");'),
    ):

        def seed_outside(root: Path, code: str = code) -> None:
            append(root / run, f'\n{code}\n')

        bans(name, seed_outside, 'brings in code from outside')

    def cargo_config(root: Path) -> None:
        append(
            root / 'rust/.cargo/config.toml',
            '[build]\nrustflags = ["-A", "clippy::disallowed_methods"]\n',
        )

    bans('rustflags allowing a ban', cargo_config, 'rust/.cargo/config.toml:2')
    for name, setting in (
        (
            'RUSTFLAGS allowing a ban in CI',
            'RUSTFLAGS: -Aclippy::disallowed_methods',
        ),
        (
            'clippy pointed at another configuration in CI',
            'CLIPPY_CONF_DIR: /tmp',
        ),
        ('lints capped in CI', "RUSTFLAGS: '--cap-lints warn'"),
    ):

        def seed_ci(root: Path, setting: str = setting) -> None:
            edit(
                root / l0_scan.CI_FILE,
                'permissions:\n  contents: read\n',
                f'permissions:\n  contents: read\n\nenv:\n  {setting}\n',
            )

        bans(name, seed_ci, f'{l0_scan.CI_FILE}:')
    zones = 'rust/crates/sdm-host/src/zones.rs'
    statement = 'let db = TimeZoneDatabase::from_dir(&root.path).ok();'
    bans(
        "an environment read in sdm-host's allowed initializer",
        lambda r: edit(
            r / zones,
            statement,
            'let db = TimeZoneDatabase::from_dir(std::env::var("TZDIR").unwrap_or_default()).ok();',
        ),
        f'{zones}:',
    )

    def widened(root: Path) -> None:
        path = root / zones
        text = path.read_text()
        start = text.index('                #[allow(')
        end = text.index(')]', start) + 3
        allow = text[start:end]
        text = text[:start] + text[end:]
        text = text.replace(
            '    pub fn new(',
            allow.replace('                ', '    ') + '    pub fn new(',
            1,
        )
        path.write_text(text)

    bans("sdm-host's allow moved up to the function", widened, f'{zones}:')
    bans(
        'a Zoned parsed by type annotation',
        lambda r: append(
            r / run,
            '\nfn z() { let _z: jiff::Zoned = "2024-01-01T00:00[UTC]".parse().unwrap(); }\n',
        ),
        'parses a Zoned',
    )
    bans(
        'a Zoned parsed by turbofish',
        lambda r: append(
            r / run, '\nfn z() { let _ = "x".parse::<Zoned>(); }\n'
        ),
        'parses a Zoned',
    )
    bans(
        'every warning allowed in a lint table',
        lambda r: edit(
            r / 'rust/Cargo.toml',
            'unsafe_code = "forbid"',
            'unsafe_code = "forbid"\nwarnings = "allow"',
        ),
        'warnings set to allow',
    )

    contract = 'rust/crates/sdm-contract'

    def steps(
        name: str, change: Callable[[Path], object], fails_with: str | None
    ) -> None:
        scan_case(
            results,
            'steps',
            name,
            (f'{contract}/src', f'{contract}/schema'),
            change,
            fails_with,
        )

    steps(
        'a fourth step variant in the source',
        lambda r: edit(
            r / contract / 'src/run.rs',
            '    CallModel(ModelCall),\n',
            '    CallModel(ModelCall),\n    SortRows { column: String },\n',
        ),
        "Step has ['fetch', 'call_model', 'sort_rows', 'done']",
    )

    def schema_variant(root: Path) -> None:
        path = next((root / contract / 'schema').glob('sdm-contract-*.json'))
        schema = json.loads(path.read_text())
        for branch in schema['oneOf']:
            if (
                branch.get('properties', {}).get('message', {}).get('const')
                == 'step'
            ):
                extra = copy.deepcopy(branch['oneOf'][0])
                extra['properties']['step']['const'] = 'truthiness'
                branch['oneOf'].append(extra)
        path.write_text(json.dumps(schema, indent=2) + '\n')

    steps('a fourth step variant in the schema', schema_variant, "'truthiness'")


# --- the clippy bans, for real ------------------------------------------------

# One use of every ban rust/clippy.toml holds that resolves today; the
# rand entries are allow-invalid, inert while rand is no dependency.
BANNED_USES = {
    # Types.
    'std::collections::HashMap': 'let _ = std::collections::HashMap::<u8, u8>::new();',
    'std::collections::HashSet': 'let _ = std::collections::HashSet::<u8>::new();',
    'std::hash::RandomState': 'let _ = std::hash::RandomState::new();',
    'std::time::SystemTime': 'let _: Option<std::time::SystemTime> = None;',
    'std::time::Instant': 'let _: Option<std::time::Instant> = None;',
    'std::fs::File': 'let _: Option<std::fs::File> = None;',
    'std::fs::OpenOptions': 'let _: Option<std::fs::OpenOptions> = None;',
    'std::fs::DirEntry': 'let _: Option<std::fs::DirEntry> = None;',
    # Clocks.
    'std::time::SystemTime::now': 'let _ = std::time::SystemTime::now();',
    'std::time::SystemTime::elapsed': 'let _ = std::time::UNIX_EPOCH.elapsed();',
    'std::time::Instant::now': 'let _ = std::time::Instant::now();',
    'jiff::Timestamp::now': 'let _ = jiff::Timestamp::now();',
    'jiff::Zoned::now': 'let _ = jiff::Zoned::now();',
    # The environment.
    'std::env::var': 'let _ = std::env::var("X");',
    'std::env::var_os': 'let _ = std::env::var_os("X");',
    'std::env::vars': 'let _ = std::env::vars();',
    'std::env::vars_os': 'let _ = std::env::vars_os();',
    'std::env::args': 'let _ = std::env::args();',
    'std::env::args_os': 'let _ = std::env::args_os();',
    'std::env::current_dir': 'let _ = std::env::current_dir();',
    'std::env::set_current_dir': 'let _ = std::env::set_current_dir("x");',
    'std::env::current_exe': 'let _ = std::env::current_exe();',
    'std::env::temp_dir': 'let _ = std::env::temp_dir();',
    'std::env::home_dir': 'let _ = std::env::home_dir();',
    'std::env::set_var': 'unsafe { std::env::set_var("X", "Y") };',
    'std::env::remove_var': 'unsafe { std::env::remove_var("X") };',
    # The file system.
    'std::fs::read': 'let _ = std::fs::read("x");',
    'std::fs::read_to_string': 'let _ = std::fs::read_to_string("x");',
    'std::fs::write': 'let _ = std::fs::write("x", b"");',
    'std::fs::read_dir': 'let _ = std::fs::read_dir("x");',
    'std::fs::metadata': 'let _ = std::fs::metadata("x");',
    'std::fs::symlink_metadata': 'let _ = std::fs::symlink_metadata("x");',
    'std::fs::canonicalize': 'let _ = std::fs::canonicalize("x");',
    'std::fs::read_link': 'let _ = std::fs::read_link("x");',
    'std::fs::exists': 'let _ = std::fs::exists("x");',
    'std::fs::copy': 'let _ = std::fs::copy("x", "y");',
    'std::fs::rename': 'let _ = std::fs::rename("x", "y");',
    'std::fs::hard_link': 'let _ = std::fs::hard_link("x", "y");',
    'std::fs::create_dir': 'let _ = std::fs::create_dir("x");',
    'std::fs::create_dir_all': 'let _ = std::fs::create_dir_all("x");',
    'std::fs::remove_file': 'let _ = std::fs::remove_file("x");',
    'std::fs::remove_dir': 'let _ = std::fs::remove_dir("x");',
    'std::fs::remove_dir_all': 'let _ = std::fs::remove_dir_all("x");',
    'std::fs::set_permissions': (
        'if let Ok(m) = std::fs::metadata("x") '
        '{ let _ = std::fs::set_permissions("x", m.permissions()); }'
    ),
    'std::path::Path::exists': 'let _ = std::path::Path::new("x").exists();',
    'std::path::Path::try_exists': 'let _ = std::path::Path::new("x").try_exists();',
    'std::path::Path::is_file': 'let _ = std::path::Path::new("x").is_file();',
    'std::path::Path::is_dir': 'let _ = std::path::Path::new("x").is_dir();',
    'std::path::Path::is_symlink': 'let _ = std::path::Path::new("x").is_symlink();',
    'std::path::Path::metadata': 'let _ = std::path::Path::new("x").metadata();',
    'std::path::Path::symlink_metadata': 'let _ = std::path::Path::new("x").symlink_metadata();',
    'std::path::Path::read_dir': 'let _ = std::path::Path::new("x").read_dir();',
    'std::path::Path::read_link': 'let _ = std::path::Path::new("x").read_link();',
    'std::path::Path::canonicalize': 'let _ = std::path::Path::new("x").canonicalize();',
    # Zone databases.
    'jiff::tz::db': 'let _ = jiff::tz::db();',
    'jiff::tz::TimeZoneDatabase::from_env': 'let _ = TimeZoneDatabase::from_env();',
    'jiff::tz::TimeZoneDatabase::from_dir': 'let _ = TimeZoneDatabase::from_dir("x");',
    'jiff::tz::TimeZoneDatabase::from_concatenated_path': (
        'let _ = TimeZoneDatabase::from_concatenated_path("x");'
    ),
    'jiff::tz::TimeZone::get': 'let _ = jiff::tz::TimeZone::get("UTC");',
    'jiff::tz::TimeZone::system': 'let _ = jiff::tz::TimeZone::system();',
    'jiff::tz::TimeZone::try_system': 'let _ = jiff::tz::TimeZone::try_system();',
    'jiff::Timestamp::in_tz': 'let _ = jiff::Timestamp::UNIX_EPOCH.in_tz("UTC");',
    'jiff::Zoned::in_tz': (
        'let _ = jiff::Timestamp::UNIX_EPOCH'
        '.to_zoned(jiff::tz::TimeZone::UTC).in_tz("UTC");'
    ),
    'jiff::civil::DateTime::in_tz': (
        'let _ = jiff::civil::date(2000, 1, 1).at(0, 0, 0, 0).in_tz("UTC");'
    ),
    'jiff::civil::Date::in_tz': 'let _ = jiff::civil::date(2000, 1, 1).in_tz("UTC");',
    'jiff::fmt::temporal::DateTimeParser::parse_zoned': (
        'let _ = jiff::fmt::temporal::DateTimeParser::new().parse_zoned("x");'
    ),
    'jiff::Zoned::strptime': 'let _ = jiff::Zoned::strptime("%Y", "2000");',
    'jiff::fmt::strtime::BrokenDownTime::to_zoned': (
        'let _ = jiff::fmt::strtime::parse("%Y", "2000").map(|t| t.to_zoned());'
    ),
}
_REASON = 'reason = "reads the zone lookup roots the host names (E24)"'
# sdm-host's exemption, as zones.rs has it: clippy must not report it, and
# l0_scan.py must accept it.
EXEMPT_USE = f"""pub fn exempt(root: &Root) {{
    #[allow(clippy::disallowed_methods, {_REASON})]
    let db = TimeZoneDatabase::from_dir(&root.path).ok();
    let _ = db;
}}
"""
# The reviewers' bypasses: each silences clippy (shown by the run) and each
# must fail l0_scan.py's bans check, read as if it were sdm-host's zones.rs.
BYPASSES = {
    'an inner allow inside cfg_attr': (
        'mod bypass_inner {\n'
        '    #![cfg_attr(all(), allow(clippy::disallowed_types))]\n'
        '    pub fn f() { let _ = std::collections::HashMap::<u8, u8>::new(); }\n'
        '}\n'
    ),
    'an allow inside cfg_attr': (
        '#[cfg_attr(all(), allow(clippy::disallowed_methods))]\n'
        'pub fn bypass_cfg_attr() { let _ = std::env::var("X"); }\n'
    ),
    'allow(warnings)': (
        '#[allow(warnings)]\npub fn bypass_warnings() { let _ = std::env::var("X"); }\n'
    ),
    'a spaced path': (
        '#[allow(clippy :: disallowed_methods)]\n'
        'pub fn bypass_spaced() { let _ = std::env::var("X"); }\n'
    ),
    'an expect inside cfg_attr': (
        '#[cfg_attr(all(), expect(clippy::disallowed_methods))]\n'
        'pub fn bypass_expect() { let _ = std::env::var("X"); }\n'
    ),
    "sdm-host's allow moved up to the function": (
        f'#[allow(clippy::disallowed_methods, {_REASON})]\n'
        'pub fn bypass_wide(root: &Root) {\n'
        '    let _ = TimeZoneDatabase::from_dir(&root.path).ok();\n'
        '    let _ = std::env::var("TZDIR");\n'
        '}\n'
    ),
    "an environment read in sdm-host's allowed initializer": (
        'pub fn bypass_initializer(root: &Root) {\n'
        f'    #[allow(clippy::disallowed_methods, {_REASON})]\n'
        '    let db = TimeZoneDatabase::from_dir(std::env::var("TZDIR").unwrap_or_default()).ok();\n'
        '    let _ = (db, &root.path);\n'
        '}\n'
    ),
    'an allow passed through a macro': (
        'macro_rules! wa { ($a:meta, $i:item) => { #[$a] $i } }\n'
        'wa!(allow(clippy::disallowed_methods), pub fn bypass_macro() '
        '{ let _ = std::env::var("X"); });\n'
    ),
    'a lint passed to an allow as a macro metavariable': (
        'macro_rules! permit { ($lint:path, $item:item) => { #[allow($lint)] $item } }\n'
        'permit!(clippy::disallowed_methods, pub fn bypass_metavariable() '
        '{ let _ = std::env::var("SYNTHETIC"); });\n'
    ),
    "a lint's name passed as a macro metavariable": (
        'macro_rules! permit_named { ($l:ident, $item:item) => '
        '{ #[allow(clippy::$l)] $item } }\n'
        'permit_named!(disallowed_methods, pub fn bypass_named() '
        '{ let _ = std::env::var("SYNTHETIC"); });\n'
    ),
    "a block in sdm-host's allowed initializer": (
        'pub fn bypass_block(root: &Root) {\n'
        f'    #[allow(clippy::disallowed_methods, {_REASON})]\n'
        '    let db = { let _ = std::env::var("X"); '
        'TimeZoneDatabase::from_dir(&root.path).ok() };\n'
        '    let _ = db;\n'
        '}\n'
    ),
}


def clippy_bans(config: dict[str, Any]) -> set[str]:
    """Every ban the configuration holds, but the allow-invalid ones."""
    return {
        entry['path']
        for key in ('disallowed-types', 'disallowed-methods')
        for entry in config.get(key, [])
        if isinstance(entry, dict) and not entry.get('allow-invalid')
    }


def clippy_case(results: list[Result]) -> None:
    rust = REPO_ROOT / 'rust'
    config = tomllib.loads((rust / 'clippy.toml').read_text())
    banned = clippy_bans(config)
    missing, extra = banned - BANNED_USES.keys(), BANNED_USES.keys() - banned
    results.append(
        Result(
            'clippy: the fixture uses every ban, one to one',
            not missing and not extra,
            'passes'
            if not missing and not extra
            else f'bans without a use: {sorted(missing)}; uses without a ban: '
            f'{sorted(extra)}',
        )
    )
    jiff = re.search(r'^jiff = (.*)$', (rust / 'Cargo.toml').read_text(), re.M)
    if not jiff:
        results.append(
            Result(
                'clippy: the bans, for real',
                False,
                'no jiff in rust/Cargo.toml',
            )
        )
        return
    with tempfile.TemporaryDirectory(prefix='sdm-l0-clippy-') as tmp:
        crate = Path(tmp) / 'fixture'
        (crate / 'src').mkdir(parents=True)
        shutil.copy2(rust / 'clippy.toml', crate / 'clippy.toml')
        shutil.copy2(
            rust / 'rust-toolchain.toml', crate / 'rust-toolchain.toml'
        )
        shutil.copy2(rust / 'Cargo.lock', crate / 'Cargo.lock')
        (crate / 'Cargo.toml').write_text(
            '[package]\nname = "l0-fixture"\nversion = "0.0.0"\nedition = "2024"\n'
            'publish = false\n\n[workspace]\n\n[dependencies]\n'
            f'jiff = {jiff.group(1)}\n'
        )
        body = '\n'.join(f'    {use}' for use in BANNED_USES.values())
        text = (
            '#![allow(deprecated, reason = "home_dir is deprecated")]\n'
            'use jiff::tz::TimeZoneDatabase;\n\n'
            'pub struct Root {\n    pub path: std::path::PathBuf,\n}\n\n'
            f'pub fn uses() {{\n{body}\n}}\n\n{EXEMPT_USE}\n'
        )
        spans: dict[str, tuple[int, int]] = {}
        for name, snippet in BYPASSES.items():
            first = text.count('\n') + 1
            text += snippet + '\n'
            spans[name] = (first, text.count('\n'))
        (crate / 'src/lib.rs').write_text(text)
        # A target directory of its own under rust/target, so the fixture's
        # build is kept between runs and never mixed with the workspace's.
        target = rust / 'target' / 'l0-selftest'
        env = {**os.environ, 'CARGO_TARGET_DIR': str(target)}
        env.pop('RUSTUP_TOOLCHAIN', None)
        run = subprocess.run(
            ['cargo', 'clippy', '--offline', '--quiet', '--', '-D', 'warnings'],
            cwd=crate,
            capture_output=True,
            text=True,
            env=env,
        )
        output = run.stderr
        if run.returncode == 0 or 'error[E' in output:
            results.append(
                Result(
                    'clippy: the bans, for real',
                    False,
                    'clippy passed the fixture'
                    if run.returncode == 0
                    else f'the fixture did not build: {output[-2000:]}',
                )
            )
            return
        reports = [
            (path, int(line))
            for path, line in re.findall(
                r'use of a disallowed (?:method|type) `([^`]+)`'
                r'(?:[^\n]*\n){0,3}?\s*--> src/lib\.rs:(\d+):',
                output,
            )
        ]
        reported = {path for path, _ in reports}
        lines = {line for _, line in reports}
        for path in BANNED_USES:
            ok = path in reported
            results.append(
                Result(
                    f'clippy: {path} refused',
                    ok,
                    'fails' if ok else f'not reported: {output[-1500:]}',
                )
            )
        source = text.splitlines()
        exempt_line = next(
            i
            for i, line in enumerate(source, 1)
            if 'let db = TimeZoneDatabase::from_dir(&root.path)' in line
        )
        exempt_scan = l0_scan.ban_problems(
            l0_scan.BAN_EXEMPTION, l0_scan.read_rust(EXEMPT_USE), True
        )
        results.append(
            Result(
                "clippy: control: sdm-host's reasoned allow",
                exempt_line not in lines and not exempt_scan,
                'passes'
                if exempt_line not in lines and not exempt_scan
                else f'reported by clippy: {exempt_line in lines}; '
                f'by the scan: {exempt_scan}',
            )
        )
        for name, (first, last) in spans.items():
            silenced = not any(first <= line <= last for line in lines)
            scan = l0_scan.ban_problems(
                l0_scan.BAN_EXEMPTION, l0_scan.read_rust(BYPASSES[name]), True
            )
            results.append(
                Result(
                    f'clippy: bypass refused: {name}',
                    silenced and bool(scan),
                    f'clippy silenced, the scan fails: {scan[0]}'
                    if silenced and scan
                    else f'clippy silenced: {silenced}; the scan: {scan}',
                )
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument(
        '--no-clippy', action='store_true', help='leave out the real clippy run'
    )
    args = parser.parse_args(argv)
    results: list[Result] = []
    deps_cases(results, l0_deps.cargo_metadata(REPO_ROOT))
    scan_cases(results)
    if not args.no_clippy:
        clippy_case(results)
    for result in results:
        print(
            f'{"ok  " if result.ok else "FAIL"} {result.name}: {result.detail}'
        )
    failed = [r for r in results if not r.ok]
    print(
        f'{len(results) - len(failed)} of {len(results)} self-tests as expected'
    )
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
