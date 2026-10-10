#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The L0 graph guard of the Rust core (Phase 1 design, task T1.10c).

It reads the workspace's resolved graph from ``cargo metadata`` (all
features, every target platform) and checks:

- every workspace crate is classified (domain, host, binding or dev), and
  every classified crate exists;
- each crate's first-party dependencies, of any kind, are among those the
  design's "Data access" section draws: ``sdm-core`` over ``sdm-request``,
  ``sdm-engine``, ``sdm-schema``, ``sdm-data``, ``sdm-wire`` and the parser;
  nothing below ``sdm-core`` on it; ``sdm-host`` over ``sdm-core``,
  ``sdm-data`` (it builds a run's data layer, E30) and ``sdm-contract``;
  ``sdm-py`` on exactly ``sdm-host`` and ``sdm-contract``
  (E21); ``sdm-testkit`` over ``sdm-host``;
- the parser rule: only ``sdm-core`` depends on the parser crate, and only
  its adapter (``src/pql.rs``, and ``src/test_ops.rs``, compiled only with
  the ``test-ops`` feature) names it, by its own name or any name a
  ``package =`` rename gives it, read from ``cargo metadata``;
- nothing shipped (domain, host or binding) reaches ``sdm-testkit`` over
  normal or build edges;
- no domain crate, and not ``sdm-host``, reaches ``pyo3``, ``jni``,
  ``arrow-pyarrow``, ``numpy`` or ``duckdb`` over normal or build edges; a
  binding reaches no database driver and no other language's runtime
  (``sdm-py`` no ``jni``; the planned ``sdm-jni`` no ``pyo3``);
- unsafe code: each domain crate and ``sdm-host`` takes the workspace lints,
  which forbid it, and forbids it at its root; the binding and the test
  host deny it;
- bridge headers (decision F7): each bridge file carries a header naming the
  phase that deletes it, a shipped file that reads ``SDM_CORE_ROUTES`` is
  one of the seam's (``ROUTE_READERS``, a narrower list than the bridge
  files), and every ``Bridge code:`` header names a phase; which task owns
  a rule a header names is for review, not this check. The parser
  is not bridge code (E18): nothing is asked of it, and it carries no such
  header.

``selftest.py`` holds each rule to a seeded fixture it must fail on.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from l0_scan import read_rust  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

DOMAIN = 'domain'
HOST = 'host'
BINDING = 'binding'
DEV = 'dev'
SHIPPED = (DOMAIN, HOST, BINDING)

CLASSES: dict[str, str] = {
    'sdm-contract': DOMAIN,
    'sdm-core': DOMAIN,
    'sdm-pql-interim': DOMAIN,
    'sdm-data': DOMAIN,
    'sdm-schema': DOMAIN,
    'sdm-engine': DOMAIN,
    'sdm-wire': DOMAIN,
    'sdm-request': DOMAIN,
    'sdm-host': HOST,
    'sdm-py': BINDING,
    'sdm-testkit': DEV,
}

PARSER = 'sdm-pql-interim'
# The first-party crates each crate may depend on, directly, by any kind of
# edge: the design's graph ("Data access"), with sdm-contract the leaf every
# crate shares. A domain crate's set is the transitive closure of the chain
# drawn below it, so sdm-core may take sdm-engine directly; sdm-host, the
# binding and the test kit take only the edges drawn for them.
ALLOWED: dict[str, frozenset[str]] = {
    'sdm-contract': frozenset(),
    'sdm-pql-interim': frozenset(),
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
    # sdm-host builds the concrete data layer for a run (E30); it reaches
    # sdm-engine's Zones through sdm-core's re-export, never directly.
    'sdm-host': frozenset({'sdm-core', 'sdm-data', 'sdm-contract'}),
    'sdm-py': frozenset({'sdm-host', 'sdm-contract'}),
    'sdm-testkit': frozenset({'sdm-host', 'sdm-contract'}),
}
# Edges that must be there (E21): the binding reaches the core through
# sdm-host alone, and takes sdm-contract's types.
REQUIRED: dict[str, frozenset[str]] = {
    'sdm-py': frozenset({'sdm-host', 'sdm-contract'}),
}
# Third-party crates no domain crate and not the host may reach.
LANGUAGE_CRATES = frozenset(
    {
        'pyo3',
        'pyo3-ffi',
        'jni',
        'jni-sys',
        'arrow-pyarrow',
        'numpy',
        'duckdb',
        'libduckdb-sys',
    }
)
# What each binding may not reach: a binding links its own language's
# runtime and no other, and no database driver. A binding not named here
# (none yet) may link no database driver.
DRIVERS = frozenset({'duckdb', 'libduckdb-sys'})
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

# The private seam deleted in Phase 5: the only shipped files that may read
# SDM_CORE_ROUTES (decision B-D7).
_CLIENT = 'packages/structured-data-models-client/src/sdm_client/'
ROUTE_READERS = (
    Path(_CLIENT + '_routing.py'),
    Path(_CLIENT + '_rust/s0.py'),
)
# Bridge files (decision F7), each held to a header naming the phase that
# deletes it: the seam, and the files that keep a rule of the design's
# "Host-held semantics" ledger on a core route (T1.11f). Which task owns
# each rule is the ledger's, held by review, not by this check.
BRIDGE_FILES = (
    *ROUTE_READERS,
    Path(_CLIENT + '_rust/schema.py'),
    Path(_CLIENT + '_rust/materialize.py'),
    Path(_CLIENT + '_rust/predict.py'),
    Path(_CLIENT + 'materializers/relational.py'),
    Path(_CLIENT + 'client.py'),
    Path(_CLIENT + 'query.py'),
)
SHIPPED_PYTHON = Path('packages/structured-data-models-client/src')
_BRIDGE_HEADER = re.compile(r'^(?:#|//)\s*Bridge code\b.*', re.M)
_BRIDGE_PHASE = re.compile(r'deleted in Phase \d+')


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


def check_classes(graph: Graph) -> list[str]:
    problems = [
        f'{crate}: a workspace crate with no class; add it to CLASSES and ALLOWED'
        for crate in graph.members
        if crate not in CLASSES
    ]
    problems += [
        f'{crate}: classified, but not a workspace crate'
        for crate in CLASSES
        if crate not in graph.members
    ]
    return problems


def check_edges(graph: Graph) -> list[str]:
    problems = []
    members = set(graph.members)
    for crate in graph.members:
        allowed = ALLOWED.get(crate, frozenset())
        first_party = {dep for dep, _ in graph.direct(crate) if dep in members}
        for dep, kinds in graph.direct(crate):
            if dep in members and dep not in allowed:
                kind = '/'.join(sorted(k or 'normal' for k in kinds))
                problems.append(
                    f"{crate} -> {dep} ({kind}): not an edge of the design's graph "
                    '("Data access")'
                )
        for dep in sorted(REQUIRED.get(crate, frozenset()) - first_party):
            problems.append(f'{crate}: does not depend on {dep} (E21)')
    for crate in graph.members:
        if crate == 'sdm-core':
            continue
        if any(dep == PARSER for dep, _ in graph.direct(crate)):
            problems.append(
                f"{crate} -> {PARSER}: only sdm-core's adapter reaches the parser"
            )
    return problems


def check_reach(graph: Graph) -> list[str]:
    problems = []
    for crate in graph.members:
        kind = CLASSES.get(crate)
        if kind not in SHIPPED:
            continue
        reached = graph.reach(crate)
        dev = [c for c in reached if CLASSES.get(c) == DEV]
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


def check_parser_users(graph: Graph) -> list[str]:
    if 'sdm-core' not in graph.manifests:
        return ['sdm-core: missing']
    root = crate_root(graph, 'sdm-core')
    problems = []
    names = {PARSER.replace('-', '_')} | graph.externs.get(
        ('sdm-core', PARSER), set()
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


def check_unsafe(graph: Graph) -> list[str]:
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
        kind = CLASSES.get(crate)
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


def bridge_header(text: str) -> str | None:
    """The bridge header in a file's leading comments, if any."""
    head = '\n'.join(text.splitlines()[:15])
    match = _BRIDGE_HEADER.search(head)
    if not match:
        return None
    # The header may run on over the comment lines that follow it.
    lines = [match.group(0)]
    for line in head[match.end() :].splitlines()[1:]:
        if not line.startswith(('#', '//')) or line.strip() in ('#', '//'):
            break
        lines.append(line)
    return ' '.join(re.sub(r'^(?:#|//)\s*', '', line) for line in lines)


def shipped_files(root: Path) -> Iterable[Path]:
    yield from sorted((root / SHIPPED_PYTHON).rglob('*.py'))
    for crates in ('rust/crates', 'rust/bindings'):
        for path in sorted((root / crates).rglob('*.rs')):
            if 'target' not in path.parts:
                yield path


def check_bridges(root: Path, parser_root: Path) -> list[str]:
    problems = []
    for rel in BRIDGE_FILES:
        path = root / rel
        if not path.is_file():
            problems.append(f'{rel}: a bridge file that does not exist')
            continue
        header = bridge_header(path.read_text())
        if header is None or not _BRIDGE_PHASE.search(header):
            problems.append(
                f'{rel}: no header naming the phase that deletes it (F7)'
            )
    for path in shipped_files(root):
        rel = path.relative_to(root)
        text = path.read_text()
        header = bridge_header(text)
        if parser_root in path.parents:
            if header is not None:
                problems.append(
                    f'{rel}: the parser is permanent, not bridge code (E18)'
                )
            continue
        if header is not None and not _BRIDGE_PHASE.search(header):
            problems.append(
                f'{rel}: a bridge header without the phase that deletes it'
            )
        if (
            path.suffix == '.py'
            and 'SDM_CORE_ROUTES' in text
            and rel not in ROUTE_READERS
        ):
            problems.append(
                f"{rel}: reads SDM_CORE_ROUTES but is not one of the seam's "
                'ROUTE_READERS'
            )
    return problems


def run_checks(metadata: dict[str, Any], root: Path) -> dict[str, list[str]]:
    graph = Graph(metadata)
    parser_root = (
        crate_root(graph, PARSER)
        if PARSER in graph.manifests
        else root / 'rust/crates' / PARSER
    )
    return {
        'classes': check_classes(graph),
        'edges': check_edges(graph),
        'reach': check_reach(graph),
        'parser': check_parser_users(graph),
        'unsafe': check_unsafe(graph),
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
    failed = False
    for name, problems in run_checks(metadata, args.root).items():
        for problem in problems:
            print(f'{name}: {problem}')
        print(f'{name}: {"FAIL" if problems else "ok"}')
        failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
