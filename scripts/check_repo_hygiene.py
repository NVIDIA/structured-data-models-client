#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Three checks this repository has to pass before it can be published.

``spdx`` asserts every first-party source file carries a licence header. A file
added without one is otherwise invisible until a release audit finds it.

``retired`` asserts no shipped file sends a reader to an interface this client
dropped. An example naming one is worse than no example, and a sweep that
chooses its own roots will miss a tree, so this enumerates from
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

SPDX_LINE = 'SPDX-License-Identifier: LicenseRef-OpenMDW-1.1'

# Generated files get their header from the generator, so they are excluded
# below rather than listed here.
SPDX_ROOTS = (
    'scripts',
    'packages/sdm-api/src',
    'packages/sdm-api/tests',
    'packages/sdm-pql/src',
    'packages/sdm-pql/tests',
    'packages/sdm-client/src',
    'packages/sdm-client/tests',
    'packages/sdm-connectors/src',
    'packages/sdm-connectors/tests',
    'packages/sdm-relational-engine/src',
    'packages/sdm-relational-engine/tests',
    # Not shipped in a wheel, but public in the repository, and the licence
    # change reached them only because a person remembered. The gate looks
    # here now so the next one does not depend on that.
    'qa',
    '.github',
)
SPDX_EXCLUDE_PARTS = {'grammar', '__pycache__'}

# Anything matching these must not appear in a file that ships to a user.
INTERNAL_PATTERNS = (
    re.compile(r'gitlab-master\.nvidia\.com'),
    re.compile(r'\bnv-shared-pypi\b'),
    re.compile(r'\bnvbugs?\b', re.IGNORECASE),
    re.compile(r'\bnspect-\d', re.IGNORECASE),
    re.compile(r'\bcolossus\b', re.IGNORECASE),
    re.compile(r'\bTODO\([a-z]+\)', re.IGNORECASE),
    # A private workspace document. The link is dead to every reader outside
    # the company and the identifier in it is not ours to publish; two of
    # these shipped in source that was internal until this release.
    re.compile(r'\bnotion\.so\b', re.IGNORECASE),
    re.compile(r'docs\.google\.com/document', re.IGNORECASE),
    re.compile(r'\.atlassian\.net\b', re.IGNORECASE),
    re.compile(r'\bconfluence\b', re.IGNORECASE),
    # Sibling repositories that stay internal. The contract repository is not
    # here: the generated bindings record it as their source, so its name
    # ships either way.
    re.compile(r'\btfm-serving\b', re.IGNORECASE),
    re.compile(r'\bstructured-data-nims\b', re.IGNORECASE),
    # Internal build names for the servers behind a model id. A reader
    # cannot look either up, and which build answers is not the client
    # story. The public client repository's own name is spelled
    # ``structured-data-models-client``, so it has to be excluded here.
    re.compile(r'\bkumoml', re.IGNORECASE),
    re.compile(r'\bstructured-data-models\b(?!-client)', re.IGNORECASE),
)

# Files that legitimately name internal infrastructure. A provenance record has
# to say where the bytes came from, and a contributor guide has to say where to
# push. Neither is a leak; both are reviewed deliberately.
INTERNAL_ALLOWLIST = {
    # A provenance record has to name the repository the bytes came from.
    'packages/sdm-api/src/sdm_api/generated/PROVENANCE.json',
    # These two hold the patterns, so they name them by definition.
    'scripts/check_repo_hygiene.py',
    'scripts/generate_sdm_api.py',
    # Runs on the internal instance, and has to keep naming it to work.
    # `.github/workflows/ci.yml` now runs the same gates in the open; this
    # file stays only for the contract job, which clones a project that is
    # not public, and goes when that clone no longer has to happen here.
    '.gitlab-ci.yml',
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
    # The grammar itself is first-party source and is packaged, so it is read
    # like any other. Its generated neighbours are excluded by path below.
    '.g4',
}

# Shipped files that carry no suffix to recognise them by. Every distribution
# packages these two, and a licence file is the last place an internal
# reference should survive.
SHIPPED_NAMES = {'LICENSE', 'NOTICE'}

# Interfaces earlier generations of this client had. An example naming one sends
# a reader to an API that does not exist, which is why this is scanned for
# rather than left to review. The scan covers every tracked shipped file,
# including notebooks and per-package example directories. A sweep that
# looked only at `src/` and the top-level `examples/` missed a whole tree.
RETIRED_INTERFACES = (
    'RelationalClient',
    'StructuredDataClient',
    'PredictClient',
    'client.relational(',
    'client.tabular(',
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


REPOSITORY_URL = re.compile(r'https://github\.com/NVIDIA/[A-Za-z0-9._-]+')


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
        if (
            path.suffix not in SHIPPED_SUFFIXES
            and path.name not in SHIPPED_NAMES
        ):
            continue
        # Generated parser output is machine-written and enormous; the
        # grammar it is generated from is neither, and ships.
        if 'grammar' in Path(relative).parts and path.suffix != '.g4':
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
        if 'grammar' in Path(relative).parts:
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


def check_repository_url() -> int:
    """Every repository URL in a shipped file must be the one being published.

    A message that invites a bug report is only useful if the address exists.
    This repository has been renamed once already, and the rename reached the
    package metadata while five runtime messages kept sending people to the
    old name.
    """
    declared = set()
    for path in REPO_ROOT.glob('packages/*/pyproject.toml'):
        declared.update(
            REPOSITORY_URL.findall(path.read_text(encoding='utf-8'))
        )
    if len(declared) != 1:
        print(
            f'packages disagree on the repository URL: {sorted(declared)}',
            file=sys.stderr,
        )
        return 1
    expected = declared.pop()

    findings = []
    for path in tracked_files():
        relative = str(path.relative_to(REPO_ROOT))
        if (
            path.suffix not in SHIPPED_SUFFIXES
            and path.name not in SHIPPED_NAMES
        ):
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        for found in set(REPOSITORY_URL.findall(text)):
            if found != expected:
                findings.append(f'{relative}: {found}')

    for finding in sorted(findings):
        print(f'stale repository URL: {finding}', file=sys.stderr)
    if findings:
        print(
            f'\nThe packages publish to {expected!r}. A message pointing '
            'somewhere else sends a reader to a page that does not exist.',
            file=sys.stderr,
        )
        return 1
    print(f'every repository URL names {expected.rsplit("/", 1)[-1]}')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'check',
        choices=('spdx', 'internal', 'retired', 'repo-url', 'all'),
    )
    args = parser.parse_args()
    if args.check == 'spdx':
        return check_spdx()
    if args.check == 'internal':
        return check_internal()
    if args.check == 'retired':
        return check_retired()
    if args.check == 'repo-url':
        return check_repository_url()
    return (
        check_spdx()
        | check_internal()
        | check_retired()
        | check_repository_url()
    )


if __name__ == '__main__':
    raise SystemExit(main())
