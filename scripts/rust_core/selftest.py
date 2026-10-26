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

The bridge freeze (``bridge_freeze.py``, decision E50) is held the same
way, against its committed baseline and declarations taken as the merge
base's: every evasion the T1.10e reviews found (decorators, dispatch
tables, bound methods, getattr, aliases, a zero-count helper, swallowing,
early returns, module code, ledgered bodies, a class attribute moved to
module scope, new dunders, the bridges' package rebinding a method, a
file leaving the freeze) fails as an undeclared change; a source-encoding
cookie hiding code (one behind a lone CR included) and a package shadowing
a frozen module are refused; a declared refactor, a rename declared, a
rule moved out declared, a widened coverage declared, a frozen file added
or removed, and a spent declaration of an unchanged entry pass, as does
the tree as landed judged against itself, as does a ``main`` reason its
judge accepts (T2.0e); a stale or spent declaration, a retired
``allowance`` reason, a ``main`` reason with no judge or one its judge
refuses, and missing, empty, retyped, too deep or
too wide evidence fail; both baseline formats are read (format 2's
counts ignored, format 3 written); and the merge base is read through git
on a temporary repository. From T1.28b (E56(d)) the counts, allowances
and netting are retired: a declared change that adds a rule passes, a
control here, and review judges it.

The parser crate's name is the checked tree's own copy of each L0 guard's
(T1.28w): a tree whose crates and guard data rename the parser passes, as
does the tree unrenamed; crates renamed under data that are not, data
renamed over crates that are not, and a guard copy missing, not a literal,
not shaped as a crate's name (a trailing space, a path) or naming another
crate fail, as does ``l0_scan.py``'s name naming no crate directory or
differing from ``l0_deps.py``'s.

Phase 1 design, task T1.10c (formerly T1.16); the rest of ``selftest.py``,
the seeded faults of the runners and the gate, arrives with T1.19.
"""

from __future__ import annotations

import argparse
import ast
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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge_freeze  # noqa: E402
import l0_deps  # noqa: E402
import l0_scan  # noqa: E402
import phases  # noqa: E402

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
            if crate in l0_deps.PLANNED:
                continue
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
        'sdm-request -> sdm-pql (dev)',
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
        # The drivers of the dialects after DuckDB (T1.21c).
        ('sdm-data', 'rusqlite'),
        ('sdm-host', 'tokio-postgres'),
        ('sdm-contract', 'odbc-api'),
        # SQL toolkits and Arrow's database connectors (T1.21d).
        ('sdm-core', 'sqlx'),
        ('sdm-data', 'sqlx-postgres'),
        ('sdm-schema', 'sqlx-sqlite'),
        ('sdm-engine', 'adbc_core'),
        ('sdm-host', 'adbc_driver_manager'),
        ('sdm-request', 'arrow-odbc'),
        ('sdm-wire', 'pq-sys'),
        ('sdm-contract', 'connectorx'),
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
    case(
        "sdm-py on a later dialect's driver",
        lambda m: m.edge('sdm-py', 'snowflake-connector-rs'),
        'sdm-py reaches snowflake-connector-rs',
    )

    # Java's binding (sdm-jni, T3.2): classified before it exists, so a
    # workspace without it passes, and once it exists it links jni, no
    # Python and no driver, and reaches the core through sdm-host alone.
    with tempfile.TemporaryDirectory(prefix='sdm-l0-') as tmp:
        manifest = Path(tmp) / 'sdm-jni/Cargo.toml'
        manifest.parent.mkdir()
        manifest.write_text(
            '[package]\nname = "sdm-jni"\n\n[lints.rust]\nunsafe_code = "deny"\n'
        )
        assert l0_deps.CLASSES['sdm-jni'] == l0_deps.BINDING

        def jni(m: Metadata, extra: str | None = None, drop: str = '') -> None:
            m.member('sdm-jni', manifest)
            # The binding exists from T3.11: its edges are the fixture's.
            m.node('sdm-jni')['deps'] = []
            for dep in ('sdm-host', 'sdm-contract', 'jni'):
                if dep != drop:
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
        case(
            'a Java binding on sdm-core',
            lambda m: jni(m, 'sdm-core'),
            'sdm-jni -> sdm-core',
        )
        case(
            'a Java binding without sdm-contract (E21)',
            lambda m: jni(m, drop='sdm-contract'),
            'sdm-jni: does not depend on sdm-contract',
        )
        bound = Path(tmp) / 'sdm-other/Cargo.toml'
        bound.parent.mkdir()
        bound.write_text(
            '[package]\nname = "sdm-other"\n\n[lints.rust]\nunsafe_code = "deny"\n'
        )
        case(
            'an unclassified binding',
            lambda m: (
                m.member('sdm-other', bound),
                m.edge('sdm-other', 'sdm-host'),
            ),
            'sdm-other: a workspace crate with no class',
        )

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
        lambda d: append(d / 'src/dispatch.rs', '\nuse sdm_pql::parser;\n'),
        'sdm-core/src/dispatch.rs: names sdm_pql',
    )

    def renamed(m: Metadata) -> None:
        m.drop_edge('sdm-core', l0_deps.PARSER)
        m.edge('sdm-core', l0_deps.PARSER, name='pqlp')

    with_crate(
        'the parser renamed (pqlp = { package = "sdm-pql" })',
        'sdm-core',
        lambda d: append(d / 'src/dispatch.rs', '\nuse pqlp::parse;\n'),
        'sdm-core/src/dispatch.rs: names pqlp, the parser',
        seed=renamed,
    )
    with_crate(
        "the parser named in sdm-core's tests",
        'sdm-core',
        lambda d: (d / 'tests/parser.rs').write_text('use sdm_pql::parse;\n'),
        'sdm-core/tests/parser.rs: names sdm_pql',
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
    ledger = l0_deps.BRIDGE_FILES[3]
    bridges(
        'a host-held semantics file without its header',
        lambda r: edit(r / ledger, '# Bridge code:', '# Code:'),
        f'{ledger}: no header naming the phase',
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
        'SDM_CORE_ROUTES read in a ledger bridge file (client.py)',
        lambda r: append(
            r / client / 'sdm_client/client.py',
            "\nimport os as _os\n\n_R = _os.environ.get('SDM_CORE_ROUTES')\n",
        ),
        'sdm_client/client.py: reads SDM_CORE_ROUTES',
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


# --- bridge_freeze.py (E50) ----------------------------------------------------


FREEZE_TREE = (
    'packages/structured-data-models-client/src',
    str(bridge_freeze.BASELINE),
    str(bridge_freeze.CHANGES),
)

# A first-party module of new rules, the seed of the dispatch cases.
FREEZE_RULES = """
def refuse(indices):
    if len(indices) > 5:
        raise ValueError('too many')
    return indices


def accept(indices):
    return indices


def strict(fn):
    def wrapped(*args, **kwargs):
        if len(args) > 2 and len(args[2]) > 5:
            raise ValueError('too many')
        return fn(*args, **kwargs)

    return wrapped


STRICT = strict
RULES = {True: refuse, False: accept}


def CHECK(indices):
    if len(indices) > 5:
        raise ValueError('too many')
"""


def freeze_cases(results: list[Result]) -> None:
    """The bridge freeze: the tripwire refuses every change to frozen code
    it is not told of, each declaration is held to its rules, the counts
    catch growth under a declared refactor, and the controls pass. The
    base is the committed baseline, declarations, decisions.md and script,
    with the tree's own sources as the base's. The review's evasions of
    rounds 1 and 2 are seeds, each checked as written (exactness) and
    after ``--write`` (the tripwire)."""
    committed = (REPO_ROOT / bridge_freeze.BASELINE).read_text()
    # The seeds are the fixture's own changes, so neither its base nor its
    # tree holds the declarations a package made of its own (T1.21d's
    # moves): a seed's message would otherwise name them, spent, in place
    # of the fault seeded.
    changes = json.dumps(
        {
            **json.loads((REPO_ROOT / bridge_freeze.CHANGES).read_text()),
            'changes': {},
        }
    )

    def based(*, spent: dict[str, Any] | None = None) -> bridge_freeze.Base:
        declared = changes
        if spent is not None:
            data = json.loads(changes)
            data['changes'] = spent
            declared = json.dumps(data)
        return bridge_freeze.Base('the fixture base', committed, declared)

    base = based()
    client = Path('packages/structured-data-models-client/src/sdm_client')
    schema = client / '_rust/schema.py'
    materialize = client / '_rust/materialize.py'
    predict = client / '_rust/predict.py'
    relational = client / 'materializers/relational.py'
    schema_py = client / 'schema.py'
    rules = client / '_rules.py'
    key_of = {
        'schema': str(schema),
        'client': str(client / 'client.py'),
        'relational': str(relational),
    }
    gather = f'{materialize}::gather'
    from_source = f'{schema}::from_source'
    gather_at = '    schema = materializer.schema\n'
    method_at = (
        '    def check_consistent(self) -> None:\n        r"""Refuse any column'
    )

    def case(
        name: str,
        change: Callable[[Path], object],
        fails_with: str | None,
        *,
        on: bridge_freeze.Base | str = base,
        rewrite: bool = False,
        declare: dict[str, dict[str, str]] | None = None,
        strict: bool = False,
    ) -> None:
        with tree(*FREEZE_TREE) as root:
            (root / bridge_freeze.CHANGES).write_text(changes)
            change(root)
            refused: list[str] = []
            if rewrite:
                try:
                    bridge_freeze.write(root)
                except bridge_freeze.Unreadable as error:
                    refused.append(f'--write refused: {error}')
            if declare is not None:
                path = root / bridge_freeze.CHANGES
                data = json.loads(path.read_text())
                data['changes'] = declare
                path.write_text(json.dumps(data))
            problems = refused + bridge_freeze.check(root, on, strict=strict)
            expect(
                results,
                f'bridge_freeze: {name}',
                problems,
                fails_with=fails_with,
            )

    def undeclared(
        name: str, change: Callable[[Path], object], key: str
    ) -> None:
        """A seed refused as written, by exactness, and after --write, as a
        change nobody declared."""
        case(name, change, 'differs from the baseline')
        case(
            f'{name}, written into the baseline',
            change,
            f'{key}: changed against the base',
            rewrite=True,
        )

    def baseline(
        root: Path, change: Callable[[dict[str, Any]], object]
    ) -> None:
        path = root / bridge_freeze.BASELINE
        data = json.loads(path.read_text())
        change(data)
        path.write_text(json.dumps(data))

    def as_format_2(data: dict[str, Any]) -> None:
        r"""The baseline in the format before E56(d), each entry with the
        counts that format held, retired."""
        data['format'] = bridge_freeze.FORMAT_2
        for functions in data['files'].values():
            for name, entry in functions.items():
                functions[name] = {
                    'fingerprint': entry['fingerprint'],
                    'raise': 0,
                    'warn': 0,
                    'branch': 0,
                    'calls': [],
                }

    @contextmanager
    def covering(changed: dict[Path, tuple[str, ...] | None]) -> Iterator[None]:
        saved = dict(bridge_freeze.COVERED)
        try:
            for rel, names in changed.items():
                if names is None:
                    del bridge_freeze.COVERED[rel]
                else:
                    bridge_freeze.COVERED[rel] = names
            bridge_freeze.LISTED = tuple(bridge_freeze.COVERED)
            bridge_freeze.FROZEN = bridge_freeze.WHOLE + bridge_freeze.LISTED
            yield
        finally:
            bridge_freeze.COVERED.clear()
            bridge_freeze.COVERED.update(saved)
            bridge_freeze.LISTED = tuple(bridge_freeze.COVERED)
            bridge_freeze.FROZEN = bridge_freeze.WHOLE + bridge_freeze.LISTED

    def at_gather(code: str) -> Callable[[Path], None]:
        return lambda r: edit(r / materialize, gather_at, code + gather_at)

    def with_rules(code: str) -> Callable[[Path], None]:
        def change(root: Path) -> None:
            (root / rules).write_text(FREEZE_RULES)
            edit(root / materialize, gather_at, code + gather_at)

        return change

    def new_schema_methods(body: str) -> Callable[[Path], None]:
        return lambda r: edit(r / schema_py, method_at, body + method_at)

    too_many = (
        '    def too_many(self, indices):\n'
        '        if len(indices) > 5:\n'
        "            raise ValueError('x')\n\n"
        '    def fine(self, indices):\n'
        '        return indices\n\n'
    )

    def schema_build_site(root: Path) -> tuple[Path, list[str], ast.Assign]:
        path = root / schema
        text = path.read_text()
        functions = [
            node
            for node in ast.parse(text).body
            if isinstance(node, ast.FunctionDef) and node.name == 'from_source'
        ]
        if len(functions) != 1:
            raise AssertionError(f'{path}: expected one from_source function')
        expected = ast.parse(
            'schema = build(cls, source, json.loads(finished.take(described.description)))'
        ).body[0]
        sites = [
            node
            for node in functions[0].body
            if isinstance(node, ast.Assign)
            and ast.dump(node) == ast.dump(expected)
        ]
        if len(sites) != 1:
            raise AssertionError(
                f'{path}: expected one schema build assignment'
            )
        return path, text.splitlines(keepends=True), sites[0]

    def added_raise(root: Path) -> None:
        path, lines, site = schema_build_site(root)
        lines.insert(
            site.lineno - 1,
            '    if described is None:\n'
            "        raise StructuredDataError('none', code='INTERNAL_ERROR')\n",
        )
        path.write_text(''.join(lines))

    def copied_description(root: Path) -> None:
        path, lines, site = schema_build_site(root)
        assert isinstance(site.value, ast.Call)
        description = site.value.args[2]
        assert description.end_lineno is not None
        assert description.end_col_offset is not None
        at = description.end_lineno - 1
        # AST columns count UTF-8 bytes, even when earlier text is non-ASCII.
        line = lines[at].encode('utf-8')
        column = description.end_col_offset
        lines[at] = (line[:column] + b'.copy()' + line[column:]).decode('utf-8')
        path.write_text(''.join(lines))

    # The tree, through git, and its copy against itself.
    expect(
        results,
        'bridge_freeze: the tree, against the merge base',
        bridge_freeze.problems(REPO_ROOT),
        fails_with=None,
    )
    case('the tree, copied, itself as the base', lambda r: None, None)
    # Each frozen file is a bridge file (F7 header, l0_deps) or holds the
    # body of a rule the design's ledger names.
    design = (REPO_ROOT / 'docs/design/rust-core/phase-1/design.md').read_text()
    ledger = design[
        design.index('### Host-held semantics') : design.index('### PQL parser')
    ]
    expect(
        results,
        'bridge_freeze: each frozen file is a bridge file or holds a ledgered rule',
        [
            f'{rel}: neither a bridge file nor the home of a ledgered rule'
            for rel in bridge_freeze.FROZEN
            if rel not in l0_deps.BRIDGE_FILES
            and '_rust' not in rel.parts
            and not any(
                f'`{name.split(".")[-1]}`' in ledger or f'`{name}`' in ledger
                for name in bridge_freeze.COVERED.get(rel, ())
            )
        ],
        fails_with=None,
    )

    # The tripwire: every evasion of review rounds 1 and 2 is a change to a
    # covered function or module block that nobody declared.
    shared_helper = (
        '\n\ndef _refuse_big(indices):\n'
        '    if len(indices) > 5:\n'
        "        raise ValueError('x')\n"
    )
    for name, change, key in (
        ('an added raise', added_raise, from_source),
        (
            'an added warning',
            at_gather(
                "    import warnings\n\n    warnings.warn('extra', stacklevel=2)\n"
            ),
            gather,
        ),
        (
            'an added branch (a shared file)',
            lambda r: edit(
                r / client / 'client.py',
                '        if not fields:\n',
                '        fields = fields if fields else ()\n        if not fields:\n',
            ),
            f'{client}/client.py::SDMClient._prepare',
        ),
        (
            'a new helper in a shared file, called',
            lambda r: (
                append(r / relational, shared_helper),
                edit(
                    r / relational,
                    '            statistic = _statistic_asked_for(options)\n',
                    '            _refuse_big(indices)\n'
                    '            statistic = _statistic_asked_for(options)\n',
                ),
            ),
            f'{relational}::RelationalMaterializer.materialize',
        ),
        (
            'a new module, called',
            with_rules(
                '    from sdm_client import _rules\n\n    _rules.refuse(indices)\n'
            ),
            gather,
        ),
        (
            'a rule inside a ledgered body (Schema.check_consistent)',
            lambda r: edit(
                r / schema_py,
                '        self._mark_key_columns()\n        for table in self.tables.values():\n',
                '        if len(self.tables) > 100:\n'
                "            raise ValueError('too many tables')\n"
                '        self._mark_key_columns()\n        for table in self.tables.values():\n',
            ),
            f'{schema_py}::Schema.check_consistent',
        ),
        (
            'a lambda',
            at_gather('    big = lambda n: n > 5 and 1 if n else 0\n'),
            gather,
        ),
        (
            'a nested def',
            at_gather(
                '    def _big(n):\n        if n > 5:\n            raise ValueError()\n\n'
                '    _big(len(indices))\n'
            ),
            gather,
        ),
        (
            'a bare decorator from a new module',
            lambda r: (
                (r / rules).write_text(FREEZE_RULES),
                edit(
                    r / materialize,
                    'from sdm_client.errors import StructuredDataError\n',
                    'from sdm_client.errors import StructuredDataError\n'
                    'from sdm_client._rules import strict\n',
                ),
                edit(
                    r / materialize,
                    '\ndef gather(\n',
                    '\n@strict\ndef gather(\n',
                ),
            ),
            gather,
        ),
        (
            'an upper-case alias as a decorator',
            lambda r: (
                (r / rules).write_text(FREEZE_RULES),
                edit(
                    r / materialize,
                    '\ndef gather(\n',
                    '\n@STRICT\ndef gather(\n',
                ),
            ),
            gather,
        ),
        (
            "a shared file's module code wrapping a covered method",
            lambda r: append(
                r / relational,
                '\n\ndef _strict(fn):\n    return fn\n\n\n'
                'RelationalMaterializer._check_entity_count = _strict(\n'
                '    RelationalMaterializer._check_entity_count\n)\n',
            ),
            f'{relational}::<module>',
        ),
        (
            "a _rust/ bridge's module code",
            lambda r: append(r / materialize, '\nif TAKEN:\n    pass\n'),
            f'{materialize}::<module>',
        ),
        (
            'dict dispatch to new helpers',
            with_rules(
                '    from sdm_client import _rules\n\n'
                '    {True: _rules.refuse, False: _rules.accept}[len(indices) > 5](indices)\n'
            ),
            gather,
        ),
        (
            'an upper-case dispatch table',
            with_rules(
                '    from sdm_client._rules import RULES\n\n'
                '    RULES[len(indices) > 5](indices)\n'
            ),
            gather,
        ),
        (
            'an upper-case function passed to map',
            with_rules(
                '    from sdm_client import _rules\n\n'
                '    list(map(_rules.CHECK, [indices]))\n'
            ),
            gather,
        ),
        (
            'bound methods in a dispatch dict',
            lambda r: (
                new_schema_methods(too_many)(r),
                at_gather(
                    '    {True: materializer.schema.too_many, False: '
                    'materializer.schema.fine}[len(indices) > 5](indices)\n'
                )(r),
            ),
            gather,
        ),
        (
            'a bound method passed to map',
            lambda r: (
                new_schema_methods(too_many)(r),
                at_gather(
                    '    list(map(materializer.schema.too_many, [indices]))\n'
                )(r),
            ),
            gather,
        ),
        (
            'getattr dispatch',
            lambda r: (
                new_schema_methods(too_many)(r),
                at_gather(
                    "    getattr(materializer.schema, 'too_many')(indices)\n"
                )(r),
            ),
            gather,
        ),
        (
            'a tuple of lambdas indexed by a condition',
            at_gather(
                "    (lambda: None, lambda: _indices([], None, ''))[len(indices) > 5]()\n"
            ),
            gather,
        ),
        (
            'a new first-party method named like one already called',
            lambda r: (
                new_schema_methods(
                    '    def get(self, indices):\n'
                    '        if len(indices) > 5:\n'
                    "            raise ValueError('x')\n\n"
                )(r),
                at_gather('    materializer.schema.get(indices)\n')(r),
            ),
            gather,
        ),
        (
            'a method with a builtin-type name',
            lambda r: (
                new_schema_methods(too_many.replace('too_many', 'update'))(r),
                at_gather('    materializer.schema.update(indices)\n')(r),
            ),
            gather,
        ),
        (
            'a swallowing try',
            lambda r: edit(
                r / materialize,
                '        choice, provider = providers.for_source(schema.source)\n',
                '        try:\n            choice, provider = providers.for_source(schema.source)\n'
                '        except StructuredDataError:\n            pass\n',
            ),
            gather,
        ),
        (
            'an aliased contextlib.suppress',
            lambda r: edit(
                r / materialize,
                '        choice, provider = providers.for_source(schema.source)\n',
                '        import contextlib\n\n        quiet = contextlib.suppress\n'
                '        with quiet(ValueError):\n'
                '            choice, provider = providers.for_source(schema.source)\n',
            ),
            gather,
        ),
        (
            'an early return under if',
            at_gather('    if len(indices) > 5:\n        return (), {}\n'),
            gather,
        ),
        (
            'a for-guarded early return',
            at_gather(
                '    for _ in range(len(indices) > 5):\n        return (), {}\n'
            ),
            gather,
        ),
        (
            "a default by 'or'",
            lambda r: edit(
                r / materialize,
                "    anchor = options.get('anchor_time')\n",
                "    anchor = options.get('anchor_time') or 0\n",
            ),
            gather,
        ),
    ):
        undeclared(name, change, key)
    # Round 3: the module's structure (Codex: scope; the subagent: dunders).
    reserved = "    _RESERVED = ('instance_id', '__kumo_anchor_time')\n"

    def relocated(root: Path) -> None:
        edit(root / relational, reserved, '')
        edit(
            root / relational,
            '\n\ndef _warn_of_a_context_the_default_mode_capped(',
            "\n\n_RESERVED = ('instance_id', '__kumo_anchor_time')\n\n\n"
            'def _warn_of_a_context_the_default_mode_capped(',
        )

    for name, change in (
        ('a class attribute moved to module scope', relocated),
        (
            'a new __setattr__ on a covered class',
            lambda r: edit(
                r / relational,
                reserved,
                reserved + '\n    def __setattr__(self, name, value):\n'
                "        if name == 'schema' and len(value.tables) > 12:\n"
                "            raise StructuredDataError('x', code='INVALID_REQUEST')\n"
                '        object.__setattr__(self, name, value)\n',
            ),
        ),
        (
            'a new __getattribute__ disabling a covered check',
            lambda r: edit(
                r / relational,
                reserved,
                reserved + '\n    def __getattribute__(self, name):\n'
                "        if name == '_check_entity_count':\n"
                '            return lambda indices, task_kind=None: None\n'
                '        return object.__getattribute__(self, name)\n',
            ),
        ),
    ):
        case(name, change, f'{relational}::<module>: differs from the baseline')
        case(
            f'{name}, written into the baseline',
            change,
            f'{relational}::<module>: changed against the base',
            rewrite=True,
        )
    rust_init = client / '_rust/__init__.py'
    undeclared(
        "the bridges' package rebinding a covered method",
        lambda r: append(
            r / rust_init,
            '\nfrom sdm_client.materializers.relational import '
            'RelationalMaterializer as _RM\n\n'
            '_RM._check_entity_count = staticmethod(lambda *a, **k: None)\n',
        ),
        f'{rust_init}::<module>',
    )

    # Round 4: what the interpreter runs, not what a UTF-8 reader sees.
    licence = '# All rights reserved.\n'
    for name, cookie, hidden in (
        (
            'a raw_unicode_escape cookie hiding a raise in a comment',
            'raw_unicode_escape',
            "    # tidy\\u000a    if len(indices) > 5: raise _not_taken('big')\n",
        ),
        (
            'a utf-7 cookie hiding a rebinding in a comment',
            'utf-7',
            '    # tidy +AAo-    materializer._check_entity_count +AD0- len\n',
        ),
    ):

        def seed(r: Path, c: str = cookie, h: str = hidden) -> None:
            edit(
                r / materialize,
                licence,
                f'# All rights reserved. -*- coding: {c} -*-\n',
            )
            at_gather(h)(r)

        case(name, seed, f'declares the source encoding {cookie}')
        case(
            f'{name}, written into the baseline',
            seed,
            '--write refused',
            rewrite=True,
        )
    case(
        'control: a utf-8 cookie',
        lambda r: edit(
            r / materialize,
            licence,
            '# All rights reserved. -*- coding: utf-8 -*-\n',
        ),
        None,
    )
    # T1.10e round 5 (subagent P3-1): a cookie behind a lone CR, which
    # detect_encoding does not see and the interpreter honours; refused
    # because the file reads otherwise as UTF-8 text.
    behind_cr = '\r# -*- coding: raw_unicode_escape -*-\r'
    hidden_rebinding = (
        '    # tidy\\u000a    materializer._check_entity_count = len\n'
    )

    def lone_cr(r: Path) -> None:
        # The hidden code first: edit() reads text, which would turn the
        # lone CRs into newlines.
        at_gather(hidden_rebinding)(r)
        path = r / materialize
        path.write_bytes(behind_cr.encode() + path.read_bytes())

    case(
        'a cookie behind a lone CR hiding a rebinding in a comment',
        lone_cr,
        'reads otherwise as UTF-8 text',
    )
    case(
        'a cookie behind a lone CR, written into the baseline',
        lone_cr,
        '--write refused',
        rewrite=True,
    )

    def bom(r: Path) -> None:
        path = r / materialize
        path.write_bytes(b'\xef\xbb\xbf' + path.read_bytes())

    case('control: a UTF-8 byte order mark', bom, None)
    case(
        'a package beside a frozen module, shadowing it',
        lambda r: append(r / client / '_rust/materialize/__init__.py', ''),
        '_rust/materialize.py: the directory',
    )

    # Codex round 2: a zero-count helper that caps the entities.
    cap = '\n\ndef _cap_indices(indices):\n    return indices[:10]\n'
    case(
        'a zero-count helper capping the entities, written in',
        lambda r: (
            append(r / materialize, cap),
            at_gather('    indices = _cap_indices(indices)\n')(r),
        ),
        f'{materialize}::_cap_indices: new against the base',
        rewrite=True,
    )
    case(
        'a new function in a _rust/ bridge, as written',
        lambda r: append(r / predict, '\n\ndef _extra(x):\n    return x\n'),
        '_rust/predict.py::_extra: covered code the baseline does not hold',
    )
    case(
        'a stale baseline',
        lambda r: baseline(
            r,
            lambda d: d['files'][key_of['schema']]['from_source'].update(
                {'fingerprint': '0' * 64}
            ),
        ),
        'from_source: differs from the baseline',
    )
    case(
        "control: a format 2 baseline's retired counts changed",
        lambda r: baseline(
            r,
            lambda d: (
                as_format_2(d),
                d['files'][key_of['schema']]['from_source'].update(
                    {'raise': 9, 'calls': ['x.y']}
                ),
            ),
        ),
        None,
    )

    # Declarations.
    def renamed_local(root: Path) -> None:
        edit(root / schema, '    spec = {\n', '    wanted = {\n')
        edit(
            root / schema, 'codec.attachment(spec)', 'codec.attachment(wanted)'
        )

    refactor = {'refactor': 'a local renamed'}
    case(
        'control: a declared refactor',
        renamed_local,
        None,
        rewrite=True,
        declare={from_source: refactor},
    )
    # A main reason (T2.0e) is judged by the base's judge, from git; the
    # judge's own rules are the T2.0e tests' (tests/test_t20e_bridge_main.py).
    from_main = {'main': f'{"ab" * 20} Rename a local (#1)'}
    case(
        'a main reason with no history to judge it by',
        renamed_local,
        'declared main, which is not a change the candidate merged from main: '
        'no git history to judge it by',
        rewrite=True,
        declare={from_source: from_main},
    )
    case(
        'a main reason its judge refuses',
        renamed_local,
        'declared main, which is not a change the candidate merged from main: '
        'the judge refuses it',
        on=replace(base, main=lambda key, text, now: 'the judge refuses it'),
        rewrite=True,
        declare={from_source: from_main},
    )
    case(
        'control: a main reason its judge accepts',
        renamed_local,
        None,
        on=replace(base, main=lambda key, text, now: None),
        rewrite=True,
        declare={from_source: from_main},
    )

    # The review's false positives of round 2, each a declared refactor.
    def shadowed(root: Path) -> None:
        path = root / relational
        text = path.read_text()
        start = text.index('def _tables_linked_to(')
        stop = text.index('return frozenset(seen)', start) + len(
            'return frozenset(seen)'
        )
        path.write_text(
            text[:start]
            + text[start:stop].replace('seen', 'dates')
            + text[stop:]
        )

    for name, change, declared_key in (
        (
            'typing.cast to a first-party class',
            at_gather(
                '    from typing import cast\n\n'
                '    materializer = cast(RelationalMaterializer, materializer)\n'
            ),
            gather,
        ),
        (
            'a local shadowing an import',
            shadowed,
            f'{relational}::_tables_linked_to',
        ),
        (
            'a copy on a value',
            copied_description,
            from_source,
        ),
    ):
        case(
            f'control: {name}, declared as a refactor',
            change,
            None,
            rewrite=True,
            declare={declared_key: {'refactor': name}},
        )
    # E56(d): the counts are retired, so a declared change that adds a
    # rule passes; review judges the reason's direction.
    case(
        'control: a declared refactor that adds a raise (the counts retired)',
        added_raise,
        None,
        rewrite=True,
        declare={from_source: {'refactor': 'a check added'}},
    )
    case(
        'a stale declaration',
        lambda r: None,
        f'{from_source}: declared in',
        declare={from_source: refactor},
    )
    case(
        'a spent declaration reused',
        renamed_local,
        'already holds, which is spent',
        on=based(spent={from_source: refactor}),
        rewrite=True,
        declare={from_source: refactor},
    )
    case(
        'control: a spent declaration of an unchanged entry, as on the phase head',
        lambda r: None,
        None,
        on=based(spent={from_source: refactor}),
        declare={from_source: refactor},
    )
    case(
        'control: a declaration with a new reason after a spent one',
        renamed_local,
        None,
        on=based(spent={from_source: refactor}),
        rewrite=True,
        declare={from_source: {'refactor': 'the local renamed again'}},
    )
    # A rule of _anchor's to move: from T1.24 the non-Timestamp refusal,
    # the anchor_entity refusal having moved into the core.
    anchor_rule = (
        '    if not isinstance(value, pd.Timestamp):\n'
        '        raise StructuredDataError(\n'
        '            f"\'anchor_time\' must be a pandas.Timestamp or the literal "\n'
        '            f"\'entity\' (got {type(value).__name__})",\n'
        "            code='INVALID_REQUEST',\n"
        '        )\n'
    )
    big = "    if len(indices) > 5:\n        raise _not_taken('big')\n"
    case(
        'a moved reason naming no task',
        lambda r: edit(r / materialize, anchor_rule, ''),
        'a moved reason names the task',
        rewrite=True,
        declare={f'{materialize}::_anchor': {'moved': 'into Rust'}},
    )
    case(
        'gone declared for an entry that changed',
        lambda r: edit(r / materialize, anchor_rule, ''),
        'declared gone, but it is changed',
        rewrite=True,
        declare={f'{materialize}::_anchor': {'gone': 'out'}},
    )
    # The batch's row cap in _check_limits (main's 1fc5a686 deleted the
    # entity count's own method, which this case moved before T2.0c).
    entity_count = 'RelationalMaterializer._check_limits'
    entity_rule = (
        '        if len(rows) > cap:\n'
        '            raise StructuredDataError(\n'
        "                f'this batch scores {len(rows)} rows and the endpoint accepts at '\n"
        "                f'most {cap} for a {kind} task',\n"
        "                code='UNSUPPORTED_REQUEST',\n"
        '            )\n'
    )
    moved_cap = {
        f'{relational}::{entity_count}': {'moved': 'the entity cap (T1.28)'},
        gather: {'moved': 'the entity cap (T1.28)'},
    }
    case(
        'control: a rule moved from relational.py into the glue, both declared moved',
        lambda r: (edit(r / relational, entity_rule, ''), at_gather(big)(r)),
        None,
        rewrite=True,
        declare=moved_cap,
    )

    statistic = (
        '            if statistic is not None:\n'
        '                _record_the_statistic(payloads, statistic)\n'
    )
    case(
        'control: a rule moved into Rust, declared moved',
        lambda r: edit(r / relational, statistic, ''),
        None,
        rewrite=True,
        declare={
            f'{relational}::RelationalMaterializer.materialize': {
                'moved': 'the statistic (T1.28)'
            }
        },
    )
    case(
        'a rule moved out, the baseline not lowered',
        lambda r: edit(r / relational, statistic, ''),
        'RelationalMaterializer.materialize: differs from the baseline',
    )
    case(
        'a declaration with no reason',
        lambda r: None,
        'a declaration holds one reason',
        declare={from_source: {'because': 'x'}},
    )
    case(
        'declarations missing',
        lambda r: (r / bridge_freeze.CHANGES).unlink(),
        'the declarations are missing',
    )
    case(
        'declarations in another format',
        lambda r: (r / bridge_freeze.CHANGES).write_text('{"changes": {}}'),
        'not declarations in format',
    )

    # Renames and coverage.
    def renamed_anchor(root: Path) -> None:
        edit(root / materialize, 'def _anchor(', 'def _anchor_of(')
        edit(
            root / materialize,
            "_anchor(options.get('anchor_time'))",
            "_anchor_of(options.get('anchor_time'))",
        )

    rename = {
        f'{materialize}::<module>': {'refactor': 'renamed _anchor_of'},
        f'{materialize}::_anchor': {'gone': 'renamed _anchor_of'},
        f'{materialize}::_anchor_of': {'refactor': 'renamed from _anchor'},
        gather: {'refactor': 'calls _anchor_of'},
    }
    case(
        'control: a rename declared as a refactor',
        renamed_anchor,
        None,
        rewrite=True,
        declare=rename,
    )
    case(
        'a rename not declared',
        renamed_anchor,
        f'{materialize}::_anchor_of: new against the base',
        rewrite=True,
    )
    case(
        'a covered function dropped from the baseline',
        lambda r: baseline(
            r,
            lambda d: d['files'][key_of['relational']].pop('_tables_linked_to'),
        ),
        '_tables_linked_to: covered code the baseline does not hold',
    )
    # T2.16a: a copy judging another tree (the gate's checkpoint copy)
    # takes covered code the tree's baseline leaves out, and a frozen file
    # it leaves out whole, when the tree declares each gone: the tree no
    # longer freezes them, and its own copy refuses a baseline entry for
    # them. Undeclared, or judged strict, each is still refused.
    linked = f'{key_of["relational"]}::_tables_linked_to'
    dropped = lambda r: baseline(  # noqa: E731
        r,
        lambda d: d['files'][key_of['relational']].pop('_tables_linked_to'),
    )
    declared_gone = {linked: {'gone': 'no longer frozen (T2.14c)'}}
    case(
        'control: a covered function left out of the baseline, declared gone',
        dropped,
        None,
        declare=declared_gone,
    )
    case(
        'a covered function left out of the baseline, declared gone, strict',
        dropped,
        '_tables_linked_to: covered code the baseline does not hold',
        declare=declared_gone,
        strict=True,
    )
    every_client = {
        f'{key_of["client"]}::{name}': {'gone': 'no longer frozen (T2.14c)'}
        for name in json.loads(committed)['files'][key_of['client']]
    }
    no_client = lambda r: baseline(  # noqa: E731
        r, lambda d: d['files'].pop(key_of['client'])
    )
    case(
        'control: a frozen file left out of the baseline, every entry gone',
        no_client,
        None,
        declare=every_client,
    )
    case(
        'a frozen file left out of the baseline, not declared',
        no_client,
        f'no entry for the frozen file {key_of["client"]}',
    )
    case(
        'a covered function renamed away',
        lambda r: edit(
            r / relational,
            'def _tables_linked_to(',
            'def _tables_reached_from(',
        ),
        '_tables_linked_to: COVERED names it, but the file defines no such',
        rewrite=True,
    )

    widened = {schema_py: (*bridge_freeze.COVERED[schema_py], '_and_list')}
    and_list = f'{schema_py}::_and_list'
    with covering(widened):
        case(
            'coverage widened over a function, not declared',
            lambda r: None,
            f'{and_list}: new against the base',
            rewrite=True,
        )
        case(
            'control: coverage widened over a function, declared',
            lambda r: None,
            None,
            rewrite=True,
            declare={and_list: {'refactor': 'coverage widened'}},
        )
    dates = client / 'core/dates.py'
    first = next(
        n.name
        for n in ast.parse((REPO_ROOT / dates).read_text()).body
        if isinstance(n, ast.FunctionDef)
    )
    with covering({dates: (first,)}):
        case(
            'a file added to the freeze, not declared',
            lambda r: None,
            f'{dates}::{first}: new against the base',
            rewrite=True,
        )
        case(
            'control: a file added to the freeze, declared',
            lambda r: None,
            None,
            rewrite=True,
            declare={
                f'{dates}::{first}': {'refactor': 'frozen'},
                f'{dates}::<module>': {'refactor': 'frozen'},
            },
        )
    adapter = client / 'adapters/relational.py'
    adapter_gone = {
        f'{adapter}::RelationalAdapter._preflight': {
            'gone': 'out of the freeze'
        },
        f'{adapter}::<module>': {'gone': 'out of the freeze'},
    }

    def preflight_rule(root: Path) -> None:
        # The review's extra5: a rule in a body that has left the freeze.
        path = root / adapter
        text = path.read_text()
        at = text.index('    def _preflight(')
        at = text.index('"""', text.index('"""', at) + 3) + 3
        at = text.index('\n', at) + 1
        path.write_text(
            text[:at]
            + "        if len(payloads) > 5:\n            raise ValueError('cap')\n"
            + text[at:]
        )

    with covering({adapter: None}):
        case(
            'a file taken out of the freeze, its entries not declared gone',
            preflight_rule,
            f'{adapter}::RelationalAdapter._preflight: gone against the base',
            rewrite=True,
        )
        case(
            'a file taken out of the freeze, its entries declared refactor',
            preflight_rule,
            'declared refactor: an entry that left is declared gone',
            rewrite=True,
            declare={k: {'refactor': 'x'} for k in adapter_gone},
        )
        case(
            'control: a file taken out of the freeze, declared gone',
            lambda r: None,
            None,
            rewrite=True,
            declare=adapter_gone,
        )

    # E56(d): the allowance reason is retired with the allowances.
    case(
        'a retired allowance reason',
        added_raise,
        'a declaration holds one reason, moved, refactor, gone',
        rewrite=True,
        declare={from_source: {'allowance': 'E66'}},
    )

    # The tree as landed: the phase branch at the landing, judged against
    # itself, its own declarations the base's (T1.24b's lesson: a
    # candidate that passes may turn the phase branch red once landed).
    landed = json.loads((REPO_ROOT / bridge_freeze.CHANGES).read_text())[
        'changes'
    ]
    case(
        'control: the tree as landed, its own declarations spent, itself as the base',
        lambda r: None,
        None,
        on=based(spent=landed),
        declare=landed,
    )
    case(
        'control: the tree as landed, written again, against itself',
        lambda r: None,
        None,
        on=based(spent=landed),
        rewrite=True,
        declare=landed,
    )

    # Missing, empty and retyped evidence.
    def write_baseline(text: str) -> Callable[[Path], None]:
        return lambda r: (r / bridge_freeze.BASELINE).write_text(text)

    case(
        'the baseline missing',
        lambda r: (r / bridge_freeze.BASELINE).unlink(),
        'the baseline is missing',
    )
    case('an empty baseline', write_baseline('{}'), 'not a baseline in format')
    case('a baseline that is not JSON', write_baseline('{'), 'not JSON')
    # Thousands of levels: refused by name, whether the reader recurses
    # (Python 3.12's, which then refuses it as too deep) or not (as not a
    # baseline); never a crash.
    deep = '[' * 100_000 + ']' * 100_000
    for reader in (bridge_freeze.load, bridge_freeze.load_changes):
        try:
            _, refused = reader(deep, 'deep')
        except RecursionError as error:  # the case's failure, not a crash
            refused = []
            print(f'{reader.__name__} raised {error!r}')
        expect(
            results,
            f'bridge_freeze: {reader.__name__} of input nested thousands deep',
            [
                p
                for p in refused
                if 'nested too deeply to read' in p
                or 'not a baseline' in p
                or 'not declarations' in p
            ],
            fails_with='deep: n',
        )
    case(
        'a very wide baseline',
        write_baseline(
            json.dumps(
                {
                    'format': bridge_freeze.FORMAT,
                    'about': '',
                    'files': {f'f{i}': {} for i in range(50_000)},
                }
            )
        ),
        # A file outside FROZEN is ignored (T2.14a1); the frozen files it
        # leaves out are refused.
        f'no entry for the frozen file {materialize}',
    )

    # T2.14a1: this copy judges only what it freezes, so a later copy may
    # freeze more and still be judged by its checkpoint's copy; everything
    # it freezes is judged as before.
    fake = {'fingerprint': 'ab' * 32}
    outside = str(client / 'not_frozen.py')
    uncovered = 'SDMClient.not_covered'
    case(
        'control: a baseline file outside FROZEN is ignored',
        lambda r: baseline(
            r, lambda d: d['files'].update({outside: {'f': dict(fake)}})
        ),
        None,
    )
    case(
        'control: an uncovered name of a covered-by-name file is ignored',
        lambda r: baseline(
            r,
            lambda d: d['files'][key_of['client']].update(
                {uncovered: dict(fake)}
            ),
        ),
        None,
    )
    case(
        'control: a declaration of an entry this copy does not freeze',
        lambda r: None,
        None,
        declare={
            f'{key_of["client"]}::{uncovered}': refactor,
            f'{outside}::f': refactor,
        },
    )
    case(
        'a frozen file missing from the baseline, beside an outside one',
        lambda r: baseline(
            r,
            lambda d: (
                d['files'].pop(key_of['schema']),
                d['files'].update({outside: {'f': dict(fake)}}),
            ),
        ),
        f'no entry for the frozen file {schema}',
    )
    case(
        'a function deleted from a WHOLE file',
        lambda r: baseline(
            r,
            lambda d: d['files'][key_of['schema']].update({'gone': dict(fake)}),
        ),
        f'{schema}::gone: in the baseline but not covered code',
    )
    case(
        'a stray declaration inside a frozen file',
        lambda r: None,
        f'{schema}::not_a_function: declared in',
        declare={f'{schema}::not_a_function': refactor},
    )
    case(
        'a covered entry is still compared exactly',
        lambda r: baseline(
            r,
            lambda d: (
                d['files'][key_of['client']]['SDMClient._prepare'].update(fake),
                d['files'].update({outside: {'f': dict(fake)}}),
            ),
        ),
        'SDMClient._prepare: differs from the baseline',
    )
    # T2.14a: strict, a copy judging its own tree (its bridge_freeze.py
    # this file, byte for byte, as prepush.sh and CI's rust-lint run it),
    # keeps every refusal T2.14a1's tolerances drop for the gate's
    # checkpoint copy: each seed passes tolerant and is refused strict.
    run_py = str(client / '_rust/run.py')
    for name, change, declared, refusal in (
        (
            'a baseline file outside FROZEN',
            lambda r: baseline(
                r, lambda d: d['files'].update({run_py: {'go': dict(fake)}})
            ),
            None,
            f'{run_py} is not a frozen bridge file',
        ),
        (
            'an uncovered client.py name in the baseline',
            lambda r: baseline(
                r,
                lambda d: d['files'][key_of['client']].update(
                    {'SDMClient.close': dict(fake)}
                ),
            ),
            None,
            f'{key_of["client"]}::SDMClient.close: in the baseline but not '
            'covered code',
        ),
        (
            'a declaration of a file outside FROZEN',
            lambda r: None,
            {f'{run_py}::go': refactor},
            f'{run_py}::go: declared in',
        ),
        (
            'a declaration of an uncovered client.py name',
            lambda r: None,
            {f'{key_of["client"]}::SDMClient.close': refactor},
            f'{key_of["client"]}::SDMClient.close: declared in',
        ),
    ):
        case(f'control: {name}, tolerant', change, None, declare=declared)
        case(f'{name}, strict', change, refusal, declare=declared, strict=True)
    case('control: the tree unchanged, tolerant', lambda r: None, None)
    case(
        'control: the tree unchanged, strict', lambda r: None, None, strict=True
    )

    # own_tree: strict only where the tree's bridge_freeze.py is this file,
    # byte for byte; problems() runs check with it.
    script = REPO_ROOT / bridge_freeze.SCRIPT
    with tree(*FREEZE_TREE) as root:
        own = root / bridge_freeze.SCRIPT
        seen: list[bool] = []
        saved = bridge_freeze.check, bridge_freeze.base_baseline
        try:
            bridge_freeze.base_baseline = lambda _root, _ref: base
            bridge_freeze.check = lambda _root, _base, *, strict=False: (
                seen.append(strict) or []
            )
            for label, text, wanted in (
                ('identical', script.read_bytes(), True),
                ('a byte differs', script.read_bytes() + b'#', False),
                ('missing', None, False),
            ):
                if own.exists():
                    own.unlink()
                if text is not None:
                    own.parent.mkdir(parents=True, exist_ok=True)
                    own.write_bytes(text)
                seen.clear()
                bridge_freeze.problems(root, 'origin/rust-core-phase-2')
                expect(
                    results,
                    f"bridge_freeze: own_tree, the tree's copy {label}",
                    []
                    if bridge_freeze.own_tree(root) is wanted
                    and seen == [wanted]
                    else [
                        f'own_tree {bridge_freeze.own_tree(root)}, problems() '
                        f'ran check with strict {seen}, not {wanted}'
                    ],
                    fails_with=None,
                )
        finally:
            bridge_freeze.check, bridge_freeze.base_baseline = saved

    case(
        'a format 2 entry without its counts',
        lambda r: baseline(
            r,
            lambda d: (
                as_format_2(d),
                d['files'][key_of['schema']]['from_source'].pop('raise'),
            ),
        ),
        'an entry holds exactly branch, calls, fingerprint, raise, warn in '
        'sdm-bridge-freeze/2',
    )

    case(
        'control: a format 2 baseline', lambda r: baseline(r, as_format_2), None
    )
    case(
        'a format 3 entry holding the retired counts',
        lambda r: baseline(
            r,
            lambda d: d['files'][key_of['schema']]['from_source'].update(
                {'raise': 1}
            ),
        ),
        'an entry holds exactly fingerprint in sdm-bridge-freeze/3',
    )
    case(
        'a baseline in a format neither copy writes',
        lambda r: baseline(r, lambda d: d.update({'format': 'x/1'})),
        'not a baseline in format sdm-bridge-freeze/3',
    )
    case(
        'a fingerprint retyped',
        lambda r: baseline(
            r,
            lambda d: d['files'][key_of['schema']]['from_source'].update(
                {'fingerprint': 'abc'}
            ),
        ),
        'fingerprint is not a SHA-256',
    )
    case(
        'a fingerprint missing',
        lambda r: baseline(
            r,
            lambda d: d['files'][key_of['schema']]['from_source'].pop(
                'fingerprint'
            ),
        ),
        'an entry holds exactly fingerprint in sdm-bridge-freeze/3',
    )
    case(
        "a frozen file's entry missing",
        lambda r: baseline(r, lambda d: d['files'].pop(key_of['schema'])),
        f'no entry for the frozen file {schema}',
    )
    case(
        'a frozen file listing no function',
        lambda r: baseline(
            r, lambda d: d['files'].update({key_of['schema']: {}})
        ),
        f'{schema} lists no function',
    )
    case(
        'a frozen file missing',
        lambda r: (r / predict).unlink(),
        '_rust/predict.py: a frozen bridge file that does not exist',
    )
    case(
        'a frozen file that does not parse',
        lambda r: append(r / predict, '\ndef (\n'),
        'does not parse',
    )
    case(
        'the base baseline unreadable',
        lambda r: None,
        'at the base (the fixture base): not JSON',
        on=bridge_freeze.Base('the fixture base', '{', changes),
    )
    case(
        'the base declarations unreadable',
        lambda r: None,
        'at the base (the fixture base): not JSON',
        on=bridge_freeze.Base('the fixture base', committed, '{'),
    )
    case(
        'no merge base',
        lambda r: None,
        'no base to ratchet against',
        on='no merge base of HEAD and origin/rust-core-phase-1',
    )
    case(
        'control: a base that predates the baseline',
        lambda r: None,
        None,
        on=bridge_freeze.Base('the fixture base', None),
    )
    case(
        'declarations against a base that predates the baseline',
        lambda r: None,
        'predates the baseline: nothing to declare',
        on=bridge_freeze.Base('the fixture base', None),
        declare={from_source: refactor},
    )

    # The fingerprint ignores what is not code, and reads alike on every
    # Python version (positions, docstrings and empty fields left out).
    plain = 'def f(x):\n    """Doc."""\n    return x\n'
    spaced = (
        '\n\ndef f(x):  # a comment\n    """Other doc."""\n\n    return (x)\n'
    )
    expect(
        results,
        'bridge_freeze: fingerprints ignore positions, comments and docstrings',
        []
        if bridge_freeze.fingerprint(ast.parse(plain).body)
        == bridge_freeze.fingerprint(ast.parse(spaced).body)
        else ['the two differ'],
        fails_with=None,
    )
    expect(
        results,
        'bridge_freeze: fingerprints see a changed constant',
        []
        if bridge_freeze.fingerprint(ast.parse(plain).body)
        != bridge_freeze.fingerprint(
            ast.parse(plain.replace('x\n', 'x[:10]\n')).body
        )
        else ['the two match'],
        fails_with=None,
    )

    # The merge base, read through git.
    with tempfile.TemporaryDirectory(prefix='sdm-freeze-git-') as tmp:
        repo = Path(tmp)

        def git(*args: str) -> None:
            subprocess.run(
                [
                    'git',
                    '-C',
                    str(repo),
                    '-c',
                    'commit.gpgsign=false',
                    '-c',
                    'core.hooksPath=/dev/null',
                    *args,
                ],
                check=True,
                capture_output=True,
                env={
                    **os.environ,
                    'GIT_AUTHOR_NAME': 'selftest',
                    'GIT_AUTHOR_EMAIL': 'selftest@example.com',
                    'GIT_COMMITTER_NAME': 'selftest',
                    'GIT_COMMITTER_EMAIL': 'selftest@example.com',
                },
            )

        git('init', '-q', '-b', 'phase')
        (repo / 'README').write_text('before\n')
        git('add', '-A')
        git('commit', '-q', '-m', 'no baseline')
        git('branch', 'old')
        at_base = {
            bridge_freeze.BASELINE: 'at the base\n',
            bridge_freeze.CHANGES: 'declarations at the base\n',
        }
        for rel, text in at_base.items():
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (repo / rel).write_text(text)
        git('add', '-A')
        git('commit', '-q', '-m', 'the baseline')
        git('checkout', '-q', '-b', 'lane')
        for rel in at_base:
            (repo / rel).write_text('on the lane\n')
        git('commit', '-q', '-am', 'the lane')
        found = bridge_freeze.base_baseline(repo, 'phase')
        expect(
            results,
            'bridge_freeze: the baseline and declarations read at the merge base',
            []
            if isinstance(found, bridge_freeze.Base)
            and (found.baseline, found.changes) == tuple(at_base.values())
            else [f'read {found!r}'],
            fails_with=None,
        )
        found = bridge_freeze.base_baseline(repo, 'old')
        expect(
            results,
            'bridge_freeze: a merge base that predates the baseline',
            []
            if isinstance(found, bridge_freeze.Base)
            and found.baseline is None
            and found.changes is None
            else [f'read {found!r}'],
            fails_with=None,
        )
        found = bridge_freeze.base_baseline(repo, 'no-such-branch')
        expect(
            results,
            'bridge_freeze: no merge base',
            [found] if isinstance(found, str) else [],
            fails_with='no merge base of HEAD and no-such-branch',
        )

        # The base branch is a setting (E56(c)): --base, else
        # SDM_FREEZE_BASE, else the phase branch; a blank one is refused.
        @contextmanager
        def freeze_base(value: str | None) -> Iterator[None]:
            saved = os.environ.pop(bridge_freeze.BASE_ENV, None)
            if value is not None:
                os.environ[bridge_freeze.BASE_ENV] = value
            try:
                yield
            finally:
                os.environ.pop(bridge_freeze.BASE_ENV, None)
                if saved is not None:
                    os.environ[bridge_freeze.BASE_ENV] = saved

        with freeze_base(None):
            chosen = bridge_freeze.base_ref()
        expect(
            results,
            'bridge_freeze: unset, the base branch is the phase branch',
            []
            if chosen
            == 'origin/'
            + phases.phase_branch(
                branch=os.environ.get('GITHUB_REF_NAME') or phases.checked_out()
            )
            else [f'chose {chosen!r}'],
            fails_with=None,
        )
        # T2.0: the phase branch, from SDM_PHASE_BRANCH, else CI's branch,
        # each a seeded value the default must follow.
        for environ, wanted in (
            ({'SDM_PHASE_BRANCH': 'rust-core-phase-2'}, 'rust-core-phase-2'),
            (
                {'GITHUB_REF_NAME': 'rust-core-phase-3-lane-a'},
                'rust-core-phase-3',
            ),
            (
                {
                    'SDM_PHASE_BRANCH': 'rust-core-phase-1',
                    'GITHUB_REF_NAME': 'rust-core-phase-2-lane-a',
                },
                'rust-core-phase-1',
            ),
        ):
            chosen = bridge_freeze.base_ref(environ=environ)
            expect(
                results,
                f'bridge_freeze: {sorted(environ.items())} names origin/{wanted}',
                [] if chosen == f'origin/{wanted}' else [f'chose {chosen!r}'],
                fails_with=None,
            )
        with freeze_base('no-such-branch'):
            found_problems = bridge_freeze.problems(REPO_ROOT)
        expect(
            results,
            'bridge_freeze: SDM_FREEZE_BASE names the base branch the check reads',
            found_problems,
            fails_with='no merge base of HEAD and no-such-branch',
        )
        with freeze_base('phase'):
            found = bridge_freeze.base_baseline(repo, bridge_freeze.base_ref())
        expect(
            results,
            'bridge_freeze: SDM_FREEZE_BASE read as a branch, its merge base read',
            []
            if isinstance(found, bridge_freeze.Base)
            and found.baseline == at_base[bridge_freeze.BASELINE]
            else [f'read {found!r}'],
            fails_with=None,
        )
        with freeze_base('no-such-branch'):
            chosen = bridge_freeze.base_ref('phase')
        expect(
            results,
            'bridge_freeze: --base wins over SDM_FREEZE_BASE',
            [] if chosen == 'phase' else [f'chose {chosen!r}'],
            fails_with=None,
        )
        for blank in ('', '  '):
            with freeze_base(blank):
                found_problems = bridge_freeze.problems(repo)
            expect(
                results,
                f'bridge_freeze: a blank SDM_FREEZE_BASE ({blank!r}) refused',
                found_problems,
                fails_with='the base branch is blank',
            )
        expect(
            results,
            'bridge_freeze: a blank --base refused',
            bridge_freeze.problems(repo, ''),
            fails_with='the base branch is blank',
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

    vocab = (
        'rust/crates',
        'scripts/rust_core/emulation-allowlist.txt',
        str(l0_scan.SCRIPT),
        str(l0_scan.DEPS_SCRIPT),
    )
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
            '\nfn f(_: sdm_pql::pyint::PyInt) {}\n',
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
        'is outside rust/crates/sdm-pql/src',
    )
    vocabulary(
        'an allowlist entry that does not exist',
        lambda r: append(
            r / l0_scan.ALLOWLIST, 'rust/crates/sdm-pql/src/gone.rs\n'
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
    sql(
        'a SELECT in sdm-core (T1.21c)',
        lambda r: append(
            r / crates / 'sdm-core/src/materialize.rs',
            '\nconst Q: &str = "SELECT expressions FROM duckdb_indexes()";\n',
        ),
        'hold `SELECT expressions FROM duckdb_indexes()`',
    )

    # The dialect guard (T1.21c, E54): one seeded fault per rule, and a
    # control of what it lets through.
    shared = ('rust/crates', 'rust/bindings')

    def dialects(
        name: str, change: Callable[[Path], object], fails_with: str | None
    ) -> None:
        scan_case(results, 'dialects', name, shared, change, fails_with)

    dialects(
        "an identifier naming DuckDB's catalog in sdm-core",
        lambda r: append(
            r / crates / 'sdm-core/src/materialize.rs',
            '\nfn duckdb_index_column(e: &str) -> &str { e }\n',
        ),
        '`duckdb_index_column` names a dialect (duckdb)',
    )
    dialects(
        "a string holding DuckDB's catalog statement in sdm-contract",
        lambda r: append(
            r / crates / 'sdm-contract/src/data.rs',
            '\npub const COLUMNS: &str = "PRAGMA table_info";\n',
        ),
        '`PRAGMA` names a dialect (pragma)',
    )
    dialects(
        'a catalog column name, as a whole word, in sdm-host',
        lambda r: append(
            r / crates / 'sdm-host/src/runs.rs',
            '\nconst POSITION: &str = "cid";\n',
        ),
        '`cid` names a dialect (cid)',
    )

    def jni(root: Path, text: str) -> None:
        source = root / 'rust/bindings/sdm-jni/src'
        # The binding exists from T3.11: its root is replaced in the copy.
        source.mkdir(parents=True, exist_ok=True)
        (source / 'lib.rs').write_text(text)

    dialects(
        "control: Java's binding, sdm-jni, as the tree holds it (absent until T3.11)",
        lambda r: None,
        None,
    )
    dialects(
        "another dialect's name in Java's binding, sdm-jni, once it exists",
        lambda r: jni(r, 'const DIALECT: &str = "Snowflake";\n'),
        '`Snowflake` names a dialect (snowflake)',
    )
    dialects(
        "control: Java's binding with no dialect word",
        lambda r: jni(r, 'const ANSWER: u8 = 42;\n'),
        None,
    )
    dialects(
        "another dialect's name in the binding, sdm-py",
        lambda r: append(
            r / 'rust/bindings/sdm-py/src/lib.rs',
            '\nconst DIALECT: &str = "Snowflake";\n',
        ),
        '`Snowflake` names a dialect (snowflake)',
    )
    dialects(
        'a dialect in camel case in sdm-schema',
        lambda r: append(
            r / crates / 'sdm-schema/src/lib.rs',
            '\npub struct DuckDbColumns;\n',
        ),
        '`DuckDbColumns` names a dialect (duckdb)',
    )
    # From T1.21d (T1.21c's review): DuckDB's constraint columns and
    # PostgreSQL's catalog by name, PRAGMA as a word, and a key's kind as
    # SQL writes it.
    for where, text, term in (
        ('sdm-core/src/describe.rs', 'constraint_type', 'constraint_type'),
        (
            'sdm-schema/src/infer.rs',
            'constraint_column_names',
            'constraint_column_names',
        ),
        (
            'sdm-engine/src/graph.rs',
            'referenced_column_names',
            'referenced_column_names',
        ),
        ('sdm-request/src/lib.rs', 'pg_catalog', 'pg_catalog'),
        ('sdm-wire/src/lib.rs', 'pragma', 'pragma'),
        ('sdm-host/src/lib.rs', 'table_info', 'table_info'),
    ):
        dialects(
            f'{term} in a string literal in {where.split("/")[0]}',
            lambda r, where=where, text=text: append(
                r / crates / where,
                f'\nconst SEEDED: &str = "{text}";\n',
            ),
            f'`{text}` names a dialect ({term})',
        )
    dialects(
        "a key's kind as SQL writes it, in sdm-core",
        lambda r: append(
            r / crates / 'sdm-core/src/describe.rs',
            '\nconst KIND: &str = "PRIMARY  KEY";\n',
        ),
        '`PRIMARY KEY` names a dialect (PRIMARY KEY)',
    )
    dialects(
        'control: pragmatic, printable_info and a primary key in prose',
        lambda r: append(
            r / crates / 'sdm-schema/src/assemble.rs',
            '\nfn pragmatic(printable_info: &str) -> &str { printable_info }\n'
            'const SAID: &str = "a relationship joins to a primary key";\n'
            'pub struct PrintableInfoPragmatic;\n',
        ),
        None,
    )
    dialects(
        'a guarded crate missing',
        lambda r: shutil.rmtree(r / crates / 'sdm-wire/src'),
        'rust/crates/sdm-wire/src: missing',
    )

    def dialect_where_allowed(root: Path) -> None:
        append(
            root / crates / 'sdm-data/src/lib.rs',
            '\nconst Q: &str = "PRAGMA table_info";\n',
        )
        append(
            root / crates / 'sdm-core/src/run.rs',
            '\n// DuckDB answers PRAGMA table_info with notnull and dflt_value\n'
            'pub fn not_null_filter(decided: bool) -> bool { decided }\n'
            '#[cfg(test)]\nmod dialect_tests {\n'
            '    const D: &str = "duckdb";\n}\n',
        )

    dialects(
        'control: sdm-data, comments, test modules, NotNull and decided',
        dialect_where_allowed,
        None,
    )

    def provider_sql(
        name: str, change: Callable[[Path], object], fails_with: str | None
    ) -> None:
        scan_case(
            results,
            'provider-sql',
            name,
            (str(l0_scan.PROVIDERS),),
            change,
            fails_with,
        )

    providers = l0_scan.PROVIDERS
    provider_sql(
        'a statement written in the DuckDB provider',
        lambda r: append(
            r / providers / 'duckdb.py',
            "\nINDEXES = 'SELECT table_name, expressions FROM duckdb_indexes()'\n",
        ),
        'a string holds `SELECT`',
    )
    provider_sql(
        'lowercase SQL in a new provider',
        lambda r: (r / providers / 'sqlite.py').write_text(
            "def tables(c):\n    return c.execute(f'select name from {c}')\n"
        ),
        'sqlite.py:2: a string holds `select name from`',
    )
    provider_sql(
        'a lowercase pragma in the DuckDB provider (T1.21d)',
        lambda r: append(
            r / providers / 'duckdb.py',
            "\nTABLES = 'pragma show_tables'\n",
        ),
        'a string holds `pragma`',
    )
    provider_sql(
        'a lowercase insert in the DuckDB provider (T1.21d)',
        lambda r: append(
            r / providers / 'duckdb.py',
            "\nKEYS = 'insert into keys values (1)'\n",
        ),
        'a string holds `insert`',
    )
    provider_sql(
        'no provider to read',
        lambda r: shutil.rmtree(r / providers),
        'no provider to read',
    )
    provider_sql(
        'control: SQL named in a docstring, and the word "select" alone',
        lambda r: append(
            r / providers / 'duckdb.py',
            '\ndef f():\n    """Runs the SELECT sdm-data rendered."""\n'
            "    return 'select a provider'\n",
        ),
        None,
    )

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
            '  installed-wheel:\n    runs-on: ubuntu-latest\n    timeout-minutes: 20\n'
            '    needs: build\n    steps:\n'
            '      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1\n',
            '  installed-wheel:\n    runs-on: ubuntu-latest\n    timeout-minutes: 20\n'
            '    needs: build\n    steps:\n'
            '      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1\n'
            f'        {ALLOWED_TO_FAIL}: true\n',
        ),
        f'`{ALLOWED_TO_FAIL}` in job installed-wheel',
    )
    skips(
        "the contract cases' step allowed to fail",
        ci(
            'contract-python-ref-pandas3',
            '        run: python -m pytest contract/tests -q',
        ),
        f'`{ALLOWED_TO_FAIL}` in job contract-python-ref-pandas3',
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
        'a language behavior step variant in the source',
        lambda r: edit(
            r / contract / 'src/run.rs',
            '    CallModel(ModelCall),\n',
            '    CallModel(ModelCall),\n    SortRows { column: String },\n',
        ),
        "Step has ['warnings', 'fetch', 'call_model', 'sort_rows', 'done']",
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

    steps(
        'a language behavior step variant in the schema',
        schema_variant,
        "'truthiness'",
    )


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


# --- the parser's name, the checked tree's (T1.28w) ---------------------------

# The name a seeded rename gives the parser crate, whatever it is called.
RENAMED = 'sdm-pql-renamed'
NAMED_TREE = (
    *RUST_TREE,
    *(str(p) for p in l0_deps.BRIDGE_FILES),
    str(l0_deps.SCRIPT),
    str(l0_scan.SCRIPT),
    str(l0_scan.ALLOWLIST),
)


def renamed_text(text: str, old: str, new: str) -> str:
    """``text`` with the crate ``old`` called ``new``, as Cargo and Rust
    name it."""
    for a, b in ((old, new), (old.replace('-', '_'), new.replace('-', '_'))):
        text = re.sub(r'(?<![\w-])' + re.escape(a) + r'(?![\w-])', b, text)
    return text


def rename_parser(root: Path, *, crates: bool, data: bool) -> None:
    """Rename the parser in the fixture at ``root``: its crate, its users
    and the allowlist (``crates``), or the guards' copies (``data``)."""
    old = l0_deps.PARSER
    paths: list[Path] = []
    if crates:
        (root / 'rust/crates' / old).rename(root / 'rust/crates' / RENAMED)
        paths += [
            p
            for p in (root / 'rust').rglob('*')
            if p.is_file() and p.suffix in ('.rs', '.toml')
        ]
        paths.append(root / l0_scan.ALLOWLIST)
    if data:
        paths += [root / l0_deps.SCRIPT, root / l0_scan.SCRIPT]
    for path in paths:
        text = path.read_text()
        new = renamed_text(text, old, RENAMED)
        if new != text:
            path.write_text(new)


def fixture_metadata(
    base: dict[str, Any], root: Path, *, crates: bool
) -> Metadata:
    """``cargo metadata`` read as if from the fixture at ``root``, the
    parser renamed when its ``crates`` are."""
    metadata = Metadata(base)
    old, real = l0_deps.PARSER, str(REPO_ROOT)
    for package in metadata.data['packages']:
        path = package['manifest_path']
        if path.startswith(real + os.sep):
            path = str(root) + path[len(real) :]
        if crates and package['name'] == old:
            package['name'] = RENAMED
            path = path.replace(
                f'{os.sep}{old}{os.sep}', f'{os.sep}{RENAMED}{os.sep}'
            )
        package['manifest_path'] = path
    metadata.data['workspace_root'] = str(root / 'rust')
    if crates:
        parser = metadata.ids.pop(old)
        metadata.ids[RENAMED] = parser
        for node in metadata.data['resolve']['nodes']:
            for dep in node['deps']:
                if dep['pkg'] == parser:
                    dep['name'] = RENAMED.replace('-', '_')
    return metadata


def parser_name_cases(results: list[Result], base: dict[str, Any]) -> None:
    def named(
        name: str,
        *,
        crates: bool,
        data: bool,
        change: Callable[[Path], object] = lambda root: None,
        deps_fails: str | None,
        scan_fails: str | None,
    ) -> None:
        with tree(*NAMED_TREE) as root:
            rename_parser(root, crates=crates, data=data)
            change(root)
            metadata = fixture_metadata(base, root, crates=crates)
            expect(
                results,
                f'l0_deps: the parser named by the tree: {name}',
                deps_problems(metadata, root),
                fails_with=deps_fails,
            )
            expect(
                results,
                f'l0_scan vocabulary: the parser named by the tree: {name}',
                l0_scan.check_vocabulary(root),
                fails_with=scan_fails,
            )

    def assign(script: Path, name: str, value: str) -> Callable[[Path], None]:
        def change(root: Path) -> None:
            path = root / script
            text, count = re.subn(
                rf'^{name} = .*$',
                f'{name} = {value}',
                path.read_text(),
                flags=re.M,
            )
            assert count == 1, f'{script}: {name} assigned {count} times'
            path.write_text(text)

        return change

    def both(value: str) -> Callable[[Path], None]:
        def change(root: Path) -> None:
            assign(l0_deps.SCRIPT, 'PARSER', value)(root)
            assign(l0_scan.SCRIPT, 'PARSER_CRATE', value)(root)

        return change

    def missing(root: Path) -> None:
        (root / l0_deps.SCRIPT).unlink()
        (root / l0_scan.SCRIPT).unlink()

    named(
        'control: the fixture as it is',
        crates=False,
        data=False,
        deps_fails=None,
        scan_fails=None,
    )
    named(
        'renamed in its crates and in the guard data, as a package renaming it is',
        crates=True,
        data=True,
        deps_fails=None,
        scan_fails=None,
    )
    named(
        'renamed in its crates, not in the guard data',
        crates=True,
        data=False,
        deps_fails=f'{RENAMED}: a workspace crate with no class',
        scan_fails=f'names {l0_scan.PARSER_CRATE}, which is no directory under '
        'rust/crates',
    )
    named(
        'renamed in the guard data, not in its crates',
        crates=False,
        data=True,
        deps_fails=f'{RENAMED}: classified, but not a workspace crate',
        scan_fails=f'names {RENAMED}, which is no directory under rust/crates',
    )
    named(
        "the tree's guard copies missing",
        crates=False,
        data=False,
        change=missing,
        deps_fails=f"the tree's {l0_deps.SCRIPT} is missing",
        scan_fails=f"the tree's {l0_scan.SCRIPT} is missing",
    )
    named(
        'the name not a literal',
        crates=False,
        data=False,
        change=both(repr(l0_deps.PARSER) + '[:]'),
        deps_fails='PARSER is not a string literal',
        scan_fails='PARSER_CRATE is not a string literal',
    )
    named(
        'the name of another crate',
        crates=False,
        data=False,
        change=both("'sdm-core'"),
        deps_fails='PARSER names sdm-core, another crate',
        scan_fails='PARSER_CRATE names sdm-core, another crate',
    )
    named(
        'the name with a trailing space',
        crates=False,
        data=False,
        change=both(repr(l0_deps.PARSER + ' ')),
        deps_fails=f'PARSER is {l0_deps.PARSER + " "!r}, not a crate',
        scan_fails=f'PARSER_CRATE is {l0_deps.PARSER + " "!r}, not a crate',
    )
    named(
        'the name a path',
        crates=False,
        data=False,
        change=both(repr('../crates/' + l0_deps.PARSER)),
        deps_fails="PARSER is '../crates/",
        scan_fails="PARSER_CRATE is '../crates/",
    )

    def scan_only(root: Path) -> None:
        assign(l0_scan.SCRIPT, 'PARSER_CRATE', repr(RENAMED))(root)

    named(
        "l0_scan.py's name naming no crate directory",
        crates=False,
        data=False,
        change=scan_only,
        deps_fails=None,
        scan_fails=f'names {RENAMED}, which is no directory under rust/crates',
    )

    def scan_differs(root: Path) -> None:
        (root / 'rust/crates' / RENAMED / 'src').mkdir(parents=True)
        scan_only(root)

    named(
        "l0_scan.py's name differing from l0_deps.py's, a crate directory",
        crates=False,
        data=False,
        change=scan_differs,
        deps_fails=None,
        scan_fails=f'but its {l0_scan.DEPS_SCRIPT} PARSER names {l0_deps.PARSER}',
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument(
        '--no-clippy', action='store_true', help='leave out the real clippy run'
    )
    args = parser.parse_args(argv)
    results: list[Result] = []
    metadata = l0_deps.cargo_metadata(REPO_ROOT)
    deps_cases(results, metadata)
    freeze_cases(results)
    scan_cases(results)
    parser_name_cases(results, metadata)
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
