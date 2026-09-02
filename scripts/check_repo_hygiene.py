#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three checks this repository has to pass before it can be published.

``spdx`` asserts every first-party source file carries a licence header. A file
added without one is otherwise invisible until a release audit finds it.

``retired`` asserts no shipped file sends a reader to an interface this client
dropped. An example naming one is worse than no example, and a sweep that
chooses its own roots will miss a tree -- so this enumerates from
``git ls-files``.

``internal`` asserts no internal-only reference reaches a *shipped* file. The
distinction matters: an internal URL in a provenance record is correct and
required, while the same URL in shipped source is a leak. So the scan covers
what goes into a wheel, and the documented exceptions are listed here rather
than discovered later.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SPDX_LINE = 'SPDX-License-Identifier: Apache-2.0'

# Only first-party Python. The vendored tree carries upstream's own headers and
# must not be touched; generated files get their header from the generator.
SPDX_ROOTS = (
    'scripts',
    'packages/sdm-api/src',
    'packages/sdm-api/tests',
    'packages/sdm-pql/tests',
    'packages/sdm-client/src',
    'packages/sdm-client/tests',
    'packages/sdm-connectors/src',
    'packages/sdm-connectors/tests',
    'packages/sdm-relational-engine/src',
    'packages/sdm-relational-engine/tests',
)
SPDX_EXCLUDE_PARTS = {'predictive_query', '__pycache__'}

# Anything matching these must not appear in a file that ships to a user.
INTERNAL_PATTERNS = (
    re.compile(r'gitlab-master\.nvidia\.com'),
    re.compile(r'\bnv-shared-pypi\b'),
    re.compile(r'\bnvbugs?\b', re.IGNORECASE),
    re.compile(r'\bnspect-\d', re.IGNORECASE),
    re.compile(r'\bcolossus\b', re.IGNORECASE),
    re.compile(r'\bTODO\([a-z]+\)', re.IGNORECASE),
)

# Files that legitimately name internal infrastructure. A provenance record has
# to say where the bytes came from, and a contributor guide has to say where to
# push. Neither is a leak; both are reviewed deliberately.
INTERNAL_ALLOWLIST = {
    'packages/sdm-pql/SOURCE.md',
    'packages/sdm-pql/VENDOR_MANIFEST.json',
    'packages/sdm-pql/README.md',
    'packages/sdm-api/README.md',
    'packages/sdm-api/src/sdm_api/generated/PROVENANCE.json',
    'scripts/check_repo_hygiene.py',
    'scripts/vendor_manifest.py',
    'scripts/generate_sdm_api.py',
    'README.md',
    'NOTICE',
    'CHANGELOG.md',
    '.gitlab-ci.yml',
    'docs/about/architecture.md',
}

SHIPPED_SUFFIXES = {
    '.py',
    '.md',
    '.toml',
    '.cfg',
    '.txt',
    '.json',
    '.yaml',
    '.yml',
    '.ipynb',
}

# Interfaces earlier generations of this client had. An example naming one sends
# a reader to an API that does not exist, which is why this is scanned for
# rather than left to review. The scan covers every tracked shipped file,
# including notebooks and per-package example directories -- a sweep that
# looked only at `src/` and the top-level `examples/` missed a whole tree.
RETIRED_INTERFACES = (
    'RelationalClient',
    'PredictClient',
    'client.relational(',
    'client.tabular(',
    'kumo_relational_client',
    'nemotron_predict',
)

# Where a retired name is legitimately mentioned: the checks that forbid it,
# and the record of what changed.
RETIRED_ALLOWLIST = {
    'scripts/check_repo_hygiene.py',
    'packages/sdm-client/tests/test_quickstart_notebook.py',
    'CHANGELOG.md',
    'docs/about/architecture.md',
}


def tracked_files() -> list[Path]:
    import subprocess

    out = subprocess.check_output(
        ['git', '-C', str(REPO_ROOT), 'ls-files'], text=True
    )
    # A file can be tracked and already deleted in the working tree; the scan
    # is about what is there now.
    return [
        path
        for path in (REPO_ROOT / line for line in out.splitlines() if line)
        if path.is_file()
    ]


def check_spdx() -> int:
    missing = []
    for path in tracked_files():
        relative = path.relative_to(REPO_ROOT)
        if path.suffix != '.py':
            continue
        if not any(str(relative).startswith(root) for root in SPDX_ROOTS):
            continue
        if set(relative.parts) & SPDX_EXCLUDE_PARTS:
            continue
        if SPDX_LINE not in path.read_text(encoding='utf-8', errors='replace'):
            missing.append(str(relative))

    for name in missing:
        print(f'missing SPDX header: {name}', file=sys.stderr)
    if missing:
        print(
            f'\n{len(missing)} file(s) need:\n'
            f'# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION '
            f'& AFFILIATES.\n# All rights reserved.\n# {SPDX_LINE}',
            file=sys.stderr,
        )
        return 1
    print('every first-party source file carries an SPDX header')
    return 0


def check_internal() -> int:
    findings = []
    for path in tracked_files():
        relative = str(path.relative_to(REPO_ROOT))
        if relative in INTERNAL_ALLOWLIST:
            continue
        if path.suffix not in SHIPPED_SUFFIXES:
            continue
        if 'predictive_query' in Path(relative).parts:
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern in INTERNAL_PATTERNS:
                if pattern.search(line):
                    findings.append(
                        f'{relative}:{line_number}: {line.strip()[:100]}'
                    )

    for finding in findings:
        print(
            f'internal reference in a shipped file: {finding}', file=sys.stderr
        )
    if findings:
        print(
            '\nEither remove the reference, or add the file to '
            'INTERNAL_ALLOWLIST in scripts/check_repo_hygiene.py with a reason.',
            file=sys.stderr,
        )
        return 1
    print('no internal references in shipped files')
    return 0


def check_retired() -> int:
    """No shipped file may send a reader to an interface this client dropped."""
    findings = []
    for path in tracked_files():
        relative = str(path.relative_to(REPO_ROOT))
        if relative in RETIRED_ALLOWLIST or path.suffix not in SHIPPED_SUFFIXES:
            continue
        if 'predictive_query' in Path(relative).parts:
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        for retired in RETIRED_INTERFACES:
            if retired in text:
                findings.append(f'{relative}: {retired}')

    for finding in findings:
        print(
            f'retired interface in a shipped file: {finding}', file=sys.stderr
        )
    if findings:
        print(
            '\nRewrite it against the shipped API, or remove it. An example '
            'nobody can run is worse than no example.',
            file=sys.stderr,
        )
        return 1
    print('no shipped file names a retired interface')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('check', choices=('spdx', 'internal', 'retired', 'all'))
    args = parser.parse_args()
    if args.check == 'spdx':
        return check_spdx()
    if args.check == 'internal':
        return check_internal()
    if args.check == 'retired':
        return check_retired()
    return check_spdx() | check_internal() | check_retired()


if __name__ == '__main__':
    raise SystemExit(main())
