#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Which branches of the code being ported the parity corpus reaches.

    coverage.py [--out PATH] [--rules PATH]

Runs transcript and sampling corpus replay, the fixture runner and the PQL
corpus runner under
branch coverage of ``sdm_relational_engine``, ``sdm_client``,
``sdm_connectors`` and ``predictive_query``, and writes a report (by default
``tests/parity/coverage.json``) grouped by the crate each module is ported
to. The crate comes from the porting ledger's rules: the first module rule
whose glob matches, with ``triage:`` dropped. Only modules bound for a core
crate are reported, covered or not.

The rules are the migration harness's ``ledger/rules.tsv``; ``--rules`` or
``SDM_PARITY_LEDGER_RULES`` points elsewhere. No threshold is set here; the
porting phases set theirs.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / 'tests' / 'parity' / 'coverage.json'
DEFAULT_RULES = '~/.local/share/sdm-dbx/rust-core/ledger/rules.tsv'
CORE_CRATES = (
    'wire',
    'core',
    'pql',
    'schema',
    'dialects',
    'sampling',
    'labels',
    'request',
)
PACKAGES = (
    'sdm_relational_engine',
    'sdm_client',
    'sdm_connectors',
    'predictive_query',
)
#: The ledger's source roots and the files it leaves out.
SOURCES = (
    'packages/structured-data-models-pql/src',
    'packages/structured-data-models-relational-engine/src',
    'packages/structured-data-models-client/src',
    'packages/structured-data-models-connectors/src',
    'packages/structured-data-models-api/src',
)
PACKAGE_RENAMES = {
    'sdm-api': 'structured-data-models-api',
    'sdm-client': 'structured-data-models-client',
    'sdm-connectors': 'structured-data-models-connectors',
    'sdm-pql': 'structured-data-models-pql',
    'sdm-relational-engine': 'structured-data-models-relational-engine',
}
SKIP = ('*/generated/*', '*/grammar/PQLGrammar*.py', '*/__pycache__/*')

_RCFILE = """\
[run]
branch = True
parallel = True
data_file = {data}
source_pkgs = {packages}
omit = */predictive_query/grammar/PQLGrammar*
"""


def current_package_path(path: str) -> str:
    for old, new in PACKAGE_RENAMES.items():
        prefix = f'packages/{old}/'
        if path.startswith(prefix):
            return f'packages/{new}/' + path[len(prefix) :]
    return path


def module_rules(path: Path) -> list[tuple[str, str]]:
    r"""The rules that map whole modules: ``(glob, crate)``, in order."""
    out = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        pattern, dest = line.split('\t')[:2]
        if '::' not in pattern:
            out.append(
                (current_package_path(pattern), dest.removeprefix('triage:'))
            )
    return out


def core_modules(rules: list[tuple[str, str]]) -> dict[str, str]:
    r"""Each module bound for a core crate, and that crate."""
    out = {}
    for source in SOURCES:
        for path in sorted((REPO / source).rglob('*.py')):
            rel = str(path.relative_to(REPO))
            if any(fnmatch.fnmatch(rel, glob) for glob in SKIP):
                continue
            for pattern, crate in rules:
                if fnmatch.fnmatch(rel, pattern):
                    if crate in CORE_CRATES:
                        out[rel] = crate
                    break
    return out


#: The scenarios whose recording is the answer a fix in the Rust core gives
#: (``tests/rust_core/shadow/fix_changes.json``): today's Python, measured
#: when ``SDM_CLIENT_IMPLEMENTATION=python`` chooses it, gives another, so
#: it is measured over the rest.
FIXED = REPO / 'tests' / 'rust_core' / 'shadow' / 'fix_changes.json'


def _fixed() -> set[str]:
    if os.environ.get('SDM_CLIENT_IMPLEMENTATION') != 'python':
        return set()
    if not FIXED.is_file():
        return set()
    return set(json.loads(FIXED.read_text(encoding='utf-8')))


def _runs() -> list[tuple[str, list[str]]]:
    fixed = _fixed()
    corpus = REPO / 'scripts' / 'parity' / 'run_corpus.py'
    replay = [str(corpus), 'replay']
    if fixed:
        listed = subprocess.run(
            [sys.executable, str(corpus), 'list', '--json'],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
        names = [
            entry['name']
            for entry in json.loads(listed.stdout)
            if entry.get('transcript') and entry['name'] not in fixed
        ]
        replay += ['--only', ','.join(names)]
    runs = [
        ('corpus replay', replay),
        (
            'sampling boundary replay',
            [str(corpus), 'replay', '--kind', 'sampling'],
        ),
    ]
    for name in ('test_fixtures.py', 'test_pql_corpus.py'):
        runner = REPO / 'tests' / 'parity' / name
        if runner.exists():
            deselected = [
                f'--deselect=tests/parity/{name}::test_fixture[{case}]'
                for case in sorted(fixed)
                if name == 'test_fixtures.py'
                and '/' in case
                and '#' not in case
            ]
            runs.append(
                (
                    f'tests/parity/{name}',
                    [
                        '-m',
                        'pytest',
                        str(runner),
                        '-q',
                        '-p',
                        'no:cacheprovider',
                        *deselected,
                    ],
                )
            )
    return runs


def measure(scratch: Path) -> tuple[dict[str, Any], list[str]]:
    r"""Run everything under coverage; coverage's JSON report, and the runs."""
    rcfile = scratch / 'coveragerc'
    rcfile.write_text(
        _RCFILE.format(data=scratch / 'data', packages=', '.join(PACKAGES))
    )
    names = []
    for name, args in _runs():
        proc = subprocess.run(
            [
                sys.executable,
                '-m',
                'coverage',
                'run',
                f'--rcfile={rcfile}',
                *args,
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise SystemExit(
                f'{name} failed under coverage:\n{proc.stdout[-3000:]}\n'
                f'{proc.stderr[-3000:]}'
            )
        names.append(name)
    for args in (['combine'], ['json', '-o', str(scratch / 'report.json')]):
        proc = subprocess.run(
            [sys.executable, '-m', 'coverage', *args, f'--rcfile={rcfile}'],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise SystemExit(f'coverage {args[0]} failed:\n{proc.stderr}')
    return json.loads((scratch / 'report.json').read_text()), names


def _relative(path: str) -> str:
    absolute = Path(path)
    if not absolute.is_absolute():
        absolute = REPO / absolute
    return str(absolute.resolve().relative_to(REPO.resolve()))


def report(
    measured: dict[str, Any], modules: dict[str, str], runs: list[str]
) -> dict[str, Any]:
    files = {_relative(path): body for path, body in measured['files'].items()}
    missing = sorted(set(modules) - set(files))
    if missing:
        raise SystemExit(
            f'coverage did not measure {len(missing)} modules: {missing[:10]}'
        )
    crates: dict[str, Any] = {}
    for module, crate in sorted(modules.items()):
        summary = files[module]['summary']
        entry = {
            'num_branches': summary['num_branches'],
            'covered_branches': summary['covered_branches'],
            'missing_branches': files[module].get('missing_branches', []),
            'num_statements': summary['num_statements'],
            'covered_lines': summary['covered_lines'],
        }
        body = crates.setdefault(
            crate,
            {'num_branches': 0, 'covered_branches': 0, 'modules': {}},
        )
        body['modules'][module] = entry
        body['num_branches'] += entry['num_branches']
        body['covered_branches'] += entry['covered_branches']
    return {
        'generated_by': 'scripts/parity/coverage.py',
        'branch_coverage_of': list(PACKAGES),
        'ran': runs,
        'crates': dict(sorted(crates.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        '--rules',
        type=Path,
        default=Path(
            os.environ.get('SDM_PARITY_LEDGER_RULES', DEFAULT_RULES)
        ).expanduser(),
    )
    args = parser.parse_args(argv)
    if not args.rules.is_file():
        print(f'error: no ledger rules at {args.rules}', file=sys.stderr)
        return 2
    modules = core_modules(module_rules(args.rules))
    with tempfile.TemporaryDirectory(prefix='sdm-parity-coverage-') as scratch:
        measured, runs = measure(Path(scratch))
    out = report(measured, modules, runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=False) + '\n')
    for crate, body in out['crates'].items():
        total = body['num_branches'] or 1
        print(
            f'{crate:<9} {body["covered_branches"]:>6} of '
            f'{body["num_branches"]:>6} branches '
            f'({100 * body["covered_branches"] / total:5.1f}%) in '
            f'{len(body["modules"])} modules'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
