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
under ``rust/clippy.toml`` that uses each banned item must fail clippy on
each, and the reasoned ``#[allow]`` that ``sdm-host`` carries must silence
its one (decision E24). That run needs the pinned toolchain and the
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l0_deps  # noqa: E402
import l0_scan  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
IGNORED = shutil.ignore_patterns('target', '__pycache__', '.venv', '*.pyc')

# The tokens the skip ban refuses, in pieces: this file is in its scope.
IGNORE_ATTR = '#[' + 'ignore]'
PYTEST_SKIP = 'pytest.' + 'skip'
XFAIL = 'x' + 'fail'
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

    def edge(self, crate: str, dep: str, kind: str | None = None) -> None:
        self.package(dep)
        self.node(crate)['deps'].append(
            {
                'name': dep.replace('-', '_'),
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

    # The parser rule's source half, and the unsafe check: crates copied.
    def with_crate(
        name: str, crate: str, change: Callable[[Path], object], fails_with: str
    ) -> None:
        with tree(f'rust/crates/{crate}', 'rust/bindings/sdm-py') as root:
            metadata = Metadata(base)
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
        'holds the SQL `SELECT`',
    )
    sql(
        'lowercase SQL in sdm-schema',
        lambda r: append(
            r / crates / 'sdm-schema/src/lib.rs',
            '\nconst Q: &str = "select a from t";\n',
        ),
        'holds the SQL `select a from`',
    )
    sql(
        'an ORDER BY clause in sdm-request',
        lambda r: append(
            r / crates / 'sdm-request/src/lib.rs',
            '\nconst Q: &str = "x ORDER BY y NULLS LAST";\n',
        ),
        'holds the SQL `ORDER BY`',
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
            '\nduckdb = pytest.importor' + 'skip("duckdb")\n',
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
        'allows clippy::disallowed_methods 2 times',
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

BANNED_USES = {
    'std::collections::HashMap': 'let _ = std::collections::HashMap::<u8, u8>::new();',
    'std::collections::HashSet': 'let _ = std::collections::HashSet::<u8>::new();',
    'std::time::SystemTime::now': 'let _ = std::time::SystemTime::now();',
    'std::time::Instant::now': 'let _ = std::time::Instant::now();',
    'std::env::var': 'let _ = std::env::var("X");',
    'std::env::var_os': 'let _ = std::env::var_os("X");',
    'std::fs::read_to_string': 'let _ = std::fs::read_to_string("x");',
    'std::path::Path::exists': 'let _ = std::path::Path::new("x").exists();',
    'jiff::Timestamp::now': 'let _ = jiff::Timestamp::now();',
    'jiff::tz::db': 'let _ = jiff::tz::db();',
    'jiff::tz::TimeZone::get': 'let _ = jiff::tz::TimeZone::get("UTC");',
    'jiff::tz::TimeZone::system': 'let _ = jiff::tz::TimeZone::system();',
    'jiff::tz::TimeZoneDatabase::from_dir': 'let _ = jiff::tz::TimeZoneDatabase::from_dir("x");',
    'jiff::Timestamp::in_tz': 'let _ = jiff::Timestamp::UNIX_EPOCH.in_tz("UTC");',
}
# sdm-host's exemption, reproduced: this one must not be reported.
EXEMPT_USE = """
    #[allow(
        clippy::disallowed_methods,
        reason = "reads the zone lookup roots the host names (E24)"
    )]
    let _exempt = jiff::tz::TimeZoneDatabase::from_dir("roots");
"""


def clippy_case(results: list[Result]) -> None:
    rust = REPO_ROOT / 'rust'
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
        (crate / 'src/lib.rs').write_text(
            f'pub fn uses() {{\n{body}\n{EXEMPT_USE}}}\n'
        )
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
        if 'could not compile' not in output and run.returncode == 0:
            results.append(
                Result(
                    'clippy: the bans, for real',
                    False,
                    'clippy passed the fixture',
                )
            )
            return
        if 'error[E' in output:
            results.append(
                Result(
                    'clippy: the bans, for real',
                    False,
                    f'the fixture did not build: {output[-2000:]}',
                )
            )
            return
        reported = re.findall(
            r'use of a disallowed (?:method|type) `([^`]+)`', output
        )
        for path in BANNED_USES:
            ok = path in reported
            results.append(
                Result(
                    f'clippy: {path} refused',
                    ok,
                    'fails' if ok else f'not reported: {output[-1500:]}',
                )
            )
        lines = [int(m) for m in re.findall(r'--> src/lib\.rs:(\d+):', output)]
        source = (crate / 'src/lib.rs').read_text().splitlines()
        exempt_line = next(
            i for i, line in enumerate(source, 1) if '_exempt' in line
        )
        results.append(
            Result(
                "clippy: control: sdm-host's reasoned allow",
                exempt_line not in lines
                and reported.count('jiff::tz::TimeZoneDatabase::from_dir') == 1,
                'passes'
                if exempt_line not in lines
                else 'the allowed call was reported',
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
