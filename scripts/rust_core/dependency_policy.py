#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the current crate dependency graph and language/runtime boundaries.

Every workspace crate must be classified; dependency edges, unsafe-code policy
and runtime/driver isolation are checked from Cargo metadata."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from architecture_policy import read_rust  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path('scripts/rust_core/dependency_policy.py')

DOMAIN = 'domain'
HOST = 'host'
BINDING = 'binding'
DEV = 'dev'
SHIPPED = (DOMAIN, HOST, BINDING)

CLASSES: dict[str, str] = {
    'sdm-contract': DOMAIN,
    'sdm-core': DOMAIN,
    'sdm-pql': DOMAIN,
    'sdm-data': DOMAIN,
    'sdm-schema': DOMAIN,
    'sdm-engine': DOMAIN,
    'sdm-wire': DOMAIN,
    'sdm-request': DOMAIN,
    'sdm-host': HOST,
    'sdm-py': BINDING,
    'sdm-jni': BINDING,
    'sdm-testkit': DEV,
}
#: The crates classified ahead of their first commit: absent from the
#: workspace until they exist, and held to their class once they do.

PARSER = 'sdm-pql'
# The first-party crates each crate may depend on, directly, by any kind of
# edge: the design's graph ("Data access"), with sdm-contract the leaf every
# crate shares. A domain crate's set is the transitive closure of the chain
# drawn below it, so sdm-core may take sdm-engine directly; sdm-host, the
# binding and the test kit take only the edges drawn for them.
ALLOWED: dict[str, frozenset[str]] = {
    'sdm-contract': frozenset(),
    'sdm-pql': frozenset(),
    'sdm-wire': frozenset({'sdm-contract'}),
    'sdm-schema': frozenset({'sdm-contract'}),
    'sdm-data': frozenset({'sdm-contract'}),
    'sdm-engine': frozenset({'sdm-schema', 'sdm-contract'}),
    'sdm-request': frozenset(
        {'sdm-engine', 'sdm-schema', 'sdm-wire', 'sdm-contract'}
    ),
    'sdm-core': frozenset(
        {
            'sdm-request',
            'sdm-engine',
            'sdm-schema',
            'sdm-data',
            'sdm-wire',
            'sdm-contract',
            PARSER,
        }
    ),
    # sdm-host builds the concrete data layer for a run; it reaches
    # sdm-engine's Zones through sdm-core's re-export, never directly.
    'sdm-host': frozenset({'sdm-core', 'sdm-data', 'sdm-contract'}),
    'sdm-py': frozenset({'sdm-host', 'sdm-contract'}),
    'sdm-jni': frozenset({'sdm-host', 'sdm-contract'}),
    'sdm-testkit': frozenset({'sdm-host', 'sdm-contract'}),
}
# Edges that must be there: the binding reaches the core through
# sdm-host alone, and takes sdm-contract's types.
REQUIRED: dict[str, frozenset[str]] = {
    'sdm-py': frozenset({'sdm-host', 'sdm-contract'}),
    'sdm-jni': frozenset({'sdm-host', 'sdm-contract'}),
}
# The Rust database drivers, of DuckDB and of the dialects that may follow
# it: SQLite, Postgres, ODBC (which Databricks and others are
# reached through) and Snowflake. No shipped crate links one, whatever its
# dialect: the host runs the statements sdm-data renders through its own
# language's driver.
DRIVERS = frozenset(
    {
        'duckdb',
        'libduckdb-sys',
        'rusqlite',
        'libsqlite3-sys',
        'postgres',
        'tokio-postgres',
        'odbc-api',
        'odbc-sys',
        'snowflake-api',
        'snowflake-connector-rs',
        # SQL toolkits and Arrow's database connectors, each a driver by
        # another road (from review).
        'sqlx',
        'sqlx-postgres',
        'sqlx-sqlite',
        'adbc_core',
        'adbc_driver_manager',
        'arrow-odbc',
        'pq-sys',
        'connectorx',
    }
)
# Third-party crates no domain crate and not the host may reach: a
# language runtime or a database driver.
LANGUAGE_CRATES = (
    frozenset(
        {
            'pyo3',
            'pyo3-ffi',
            'jni',
            'jni-sys',
            'arrow-pyarrow',
            'numpy',
        }
    )
    | DRIVERS
)
# What each binding may not reach: a binding links its own language's
# runtime and no other, and no database driver. A binding not named here
# (none yet) may link no database driver.
BINDING_FORBIDDEN: dict[str, frozenset[str]] = {
    'sdm-py': DRIVERS | {'jni', 'jni-sys'},
    'sdm-jni': DRIVERS | {'pyo3', 'pyo3-ffi', 'arrow-pyarrow', 'numpy'},
}
# Where sdm-core may name the parser: its adapter, and the test-only fault
# behind the test-ops feature.
PARSER_USERS = (Path('src/pql.rs'), Path('src/test_ops.rs'))
_TEST_OPS_MODULE = re.compile(
    r'#\[cfg\(feature\s*=\s*"test-ops"\)\]\s*(?:pub\s+)?mod\s+test_ops\s*;'
)

SHIPPED_PYTHON = Path('packages/structured-data-models-client/src')


def cargo_metadata(root: Path) -> dict[str, Any]:
    out = subprocess.run(
        [
            'cargo',
            'metadata',
            '--format-version',
            '1',
            '--locked',
            '--all-features',
            '--manifest-path',
            str(root / 'rust/Cargo.toml'),
        ],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    data: dict[str, Any] = json.loads(out)
    return data


class Graph:
    """The resolved graph: package names and edges with their kinds."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.names = {p['id']: p['name'] for p in metadata['packages']}
        self.manifests = {
            p['name']: Path(p['manifest_path']) for p in metadata['packages']
        }
        self.members = sorted(
            self.names[i] for i in metadata['workspace_members']
        )
        self.workspace_root = Path(metadata['workspace_root'])
        self.edges: dict[str, list[tuple[str, set[str | None]]]] = {}
        # The name each crate's code knows a dependency by, which a
        # `package =` rename in Cargo.toml changes.
        self.externs: dict[tuple[str, str], set[str]] = {}
        for node in metadata['resolve']['nodes']:
            deps = []
            for dep in node['deps']:
                kinds = {k['kind'] for k in dep['dep_kinds']}
                deps.append((self.names[dep['pkg']], kinds))
                key = (self.names[node['id']], self.names[dep['pkg']])
                self.externs.setdefault(key, set()).add(dep['name'])
            self.edges.setdefault(self.names[node['id']], []).extend(deps)

    def direct(self, crate: str) -> list[tuple[str, set[str | None]]]:
        return self.edges.get(crate, [])

    def reach(self, crate: str) -> dict[str, list[str]]:
        """Every crate reached over normal and build edges, with a path to it."""
        paths: dict[str, list[str]] = {crate: [crate]}
        todo = [crate]
        while todo:
            current = todo.pop()
            for dep, kinds in self.direct(current):
                if kinds <= {'dev'} or dep in paths:
                    continue
                paths[dep] = [*paths[current], dep]
                todo.append(dep)
        del paths[crate]
        return paths


@dataclass(frozen=True)
class Crates:
    """The crates' data a check reads: the parser's name and the classes
    and edges, the parser's under that name."""

    parser: str
    classes: dict[str, str]
    allowed: dict[str, frozenset[str]]


def tree_crates(root: Path) -> tuple[Crates, list[str]]:
    """Use the checked-out revision's explicit architecture policy."""
    return Crates(PARSER, CLASSES, ALLOWED), []


def check_classes(graph: Graph, crates: Crates) -> list[str]:
    problems = [
        f'{crate}: a workspace crate with no class; add it to CLASSES and ALLOWED'
        for crate in graph.members
        if crate not in crates.classes
    ]
    problems += [
        f'{crate}: classified, but not a workspace crate'
        for crate in crates.classes
        if crate not in graph.members
    ]
    return problems


def check_edges(graph: Graph, crates: Crates) -> list[str]:
    problems = []
    members = set(graph.members)
    for crate in graph.members:
        allowed = crates.allowed.get(crate, frozenset())
        first_party = {dep for dep, _ in graph.direct(crate) if dep in members}
        for dep, kinds in graph.direct(crate):
            if dep in members and dep not in allowed:
                kind = '/'.join(sorted(k or 'normal' for k in kinds))
                problems.append(
                    f"{crate} -> {dep} ({kind}): not an edge of the design's graph "
                    '("Data access")'
                )
        for dep in sorted(REQUIRED.get(crate, frozenset()) - first_party):
            problems.append(f'{crate}: does not depend on {dep}')
    for crate in graph.members:
        if crate == 'sdm-core':
            continue
        if any(dep == crates.parser for dep, _ in graph.direct(crate)):
            problems.append(
                f"{crate} -> {crates.parser}: only sdm-core's adapter reaches "
                'the parser'
            )
    return problems


def check_reach(graph: Graph, crates: Crates) -> list[str]:
    problems = []
    for crate in graph.members:
        kind = crates.classes.get(crate)
        if kind not in SHIPPED:
            continue
        reached = graph.reach(crate)
        dev = [c for c in reached if crates.classes.get(c) == DEV]
        for target in dev:
            problems.append(
                f'{crate} reaches {target} ({" -> ".join(reached[target])}); '
                'nothing shipped depends on the dev-only test host'
            )
        forbidden = (
            LANGUAGE_CRATES
            if kind in (DOMAIN, HOST)
            else BINDING_FORBIDDEN.get(crate, DRIVERS)
        )
        for target in sorted(forbidden & reached.keys()):
            problems.append(
                f'{crate} reaches {target} ({" -> ".join(reached[target])}); '
                f'a {kind} crate links no language runtime or database driver'
            )
    return problems


def crate_root(graph: Graph, crate: str) -> Path:
    return graph.manifests[crate].parent


def check_parser_users(graph: Graph, parser: str) -> list[str]:
    if 'sdm-core' not in graph.manifests:
        return ['sdm-core: missing']
    root = crate_root(graph, 'sdm-core')
    problems = []
    names = {parser.replace('-', '_')} | graph.externs.get(
        ('sdm-core', parser), set()
    )
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(n) for n in sorted(names)) + r')\b'
    )
    for path in sorted(root.rglob('*.rs')):
        if 'target' in path.parts:
            continue
        rel = path.relative_to(root)
        match = pattern.search(read_rust(path.read_text()).skeleton)
        if match and rel not in PARSER_USERS:
            problems.append(
                f'sdm-core/{rel}: names {match.group(0)}, the parser; only the '
                f'adapter ({", ".join(str(p) for p in PARSER_USERS)}) uses it'
            )
    lib = root / 'src/lib.rs'
    if lib.is_file() and (root / 'src/test_ops.rs').is_file():
        if not _TEST_OPS_MODULE.search(read_rust(lib.read_text()).code):
            problems.append(
                'sdm-core/src/lib.rs: test_ops, which names the parser, is not '
                'compiled only with the test-ops feature'
            )
    return problems


def check_unsafe(graph: Graph, crates: Crates) -> list[str]:
    problems = []
    workspace = tomllib.loads((graph.workspace_root / 'Cargo.toml').read_text())
    workspace_rust = (
        workspace.get('workspace', {}).get('lints', {}).get('rust', {})
    )
    if workspace_rust.get('unsafe_code') != 'forbid':
        problems.append(
            'rust/Cargo.toml: the workspace lints do not forbid unsafe_code'
        )
    for crate in graph.members:
        kind = crates.classes.get(crate)
        manifest = tomllib.loads(graph.manifests[crate].read_text())
        lints = manifest.get('lints', {})
        if kind in (DOMAIN, HOST):
            if lints != {'workspace': True}:
                problems.append(
                    f'{crate}: does not take exactly the workspace lints'
                )
            lib = crate_root(graph, crate) / 'src/lib.rs'
            text = read_rust(lib.read_text()).skeleton if lib.is_file() else ''
            if not re.search(
                r'#!\[\s*forbid\s*\(\s*unsafe_code\s*\)\s*\]', text
            ):
                problems.append(
                    f'{crate}: its root does not forbid unsafe_code'
                )
        elif kind in (BINDING, DEV):
            level = lints.get('rust', {}).get('unsafe_code')
            if lints.get('workspace') is not True and level not in (
                'deny',
                'forbid',
            ):
                problems.append(
                    f'{crate}: unsafe_code is neither denied nor forbidden'
                )
    return problems


def shipped_files(root: Path) -> Iterable[Path]:
    yield from sorted((root / SHIPPED_PYTHON).rglob('*.py'))
    for crates in ('rust/crates', 'rust/bindings'):
        for path in sorted((root / crates).rglob('*.rs')):
            if 'target' not in path.parts:
                yield path


def check_bridges(root: Path, parser_root: Path) -> list[str]:
    """Reject the retired route switch in shipped source, without exceptions."""
    return [
        f'{path.relative_to(root)}: reads SDM_CORE_ROUTES, which is retired'
        for path in shipped_files(root)
        if 'SDM_CORE_ROUTES' in path.read_text()
    ]


def run_checks(metadata: dict[str, Any], root: Path) -> dict[str, list[str]]:
    graph = Graph(metadata)
    crates, unreadable = tree_crates(root)
    if unreadable:
        return {'classes': unreadable}
    parser = crates.parser
    parser_root = (
        crate_root(graph, parser)
        if parser in graph.manifests
        else root / 'rust/crates' / parser
    )
    return {
        'classes': check_classes(graph, crates),
        'edges': check_edges(graph, crates),
        'reach': check_reach(graph, crates),
        'parser': check_parser_users(graph, parser),
        'unsafe': check_unsafe(graph, crates),
        'bridges': check_bridges(root, parser_root),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('--root', type=Path, default=REPO_ROOT)
    parser.add_argument(
        '--metadata',
        type=Path,
        help='read cargo metadata from this file instead',
    )
    args = parser.parse_args(argv)
    metadata = (
        json.loads(args.metadata.read_text())
        if args.metadata
        else cargo_metadata(args.root)
    )
    checks = run_checks(metadata, args.root)
    failed = False
    for name, problems in checks.items():
        for problem in problems:
            print(f'{name}: {problem}')
        print(f'{name}: {"FAIL" if problems else "ok"}')
        failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
