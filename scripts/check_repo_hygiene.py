#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checks this repository has to pass before it can be published.

``spdx`` asserts every first-party source file carries a licence header. A file
added without one is otherwise invisible until a release audit finds it.

``retired`` asserts no shipped file sends a reader to an interface this client
dropped. An example naming one is worse than no example, and a sweep that
chooses its own roots will miss a tree, so this enumerates from
``git ls-files``.

``in-process`` asserts the skills' in-process references install the way the
client's own version allows. They build from a checkout because no release
carries ``in_process()`` yet, and the release that first does makes that
instruction wrong. Nothing else would catch it: the references are prose, and
the example harness runs them against this tree rather than against the index.

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

# Every tracked file that can carry the header is checked, rather than a list
# of directories. A root list has to be remembered, and the files it forgot
# look exactly like the files it cleared.
SPDX_SUFFIXES = {
    '.cfg',
    '.cpp',
    '.g4',
    '.h',
    '.hpp',
    '.in',
    '.py',
    '.pyi',
    '.rs',
    '.sh',
    '.toml',
    '.yaml',
    '.yml',
}

# Third-party bytes and build output. A directory of our own is not listed
# here: `grammar/` and `generated/` each hold hand-maintained files beside the
# produced ones, and excluding the directory excused those too.
SPDX_EXCLUDE_PARTS = {'__pycache__', '_vendor', 'assets', 'build'}

# A produced file says so on a line of its own, near the top, in the form the
# tool that wrote it uses. Anchoring to the start of the line matters: a
# docstring that merely mentions being generated from something would
# otherwise excuse itself from the licence check.
SPDX_GENERATED = re.compile(
    r'^[\s#/*\'"]*(?:code\s+)?generated\s+(?:by|from)\b',
    re.IGNORECASE,
)
SPDX_GENERATED_LINES = 8

# Anything matching these must not appear in a file that ships to a user.
INTERNAL_PATTERNS = (
    re.compile(r'gitlab-master\.nvidia\.com'),
    re.compile(r'urm\.nvidia\.com'),
    re.compile(r'\bkumo-tfm-nims\b'),
    # A repository this one is not, and which no reader can open.
    re.compile(r'\bkumo-relational-client\b', re.IGNORECASE),
    re.compile(r'\bnv-shared-pypi\b'),
    # Kitmaker's Portal is on the internal network, and its address shipped
    # in a workflow script until the publish moved off CI. The service is
    # named in SECURITY.md on purpose; the host a reader cannot reach is the
    # part that does not belong here.
    re.compile(r'kitmaker-portal|\bkm\.nvidia\.com\b', re.IGNORECASE),
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
    # A workspace inside that repository. The bare `rfm_nim` names the model
    # a test talks to and stays; this is a path into a tree nobody outside
    # can open, which is what shipped in a docstring until it was removed.
    re.compile(r'\bkumo_rfm_nim\b', re.IGNORECASE),
    # An internal build name for a server behind a model id. A reader cannot
    # look it up, and which build answers is not the client story. The model
    # package, structured-data-models, is public and is what the in-process
    # extra installs, so it is named like any other dependency.
    re.compile(r'\bkumoml', re.IGNORECASE),
    # The bare name is public now, and is what the in-process extra installs,
    # so it is named like any other dependency. A suffix that is not one of
    # the five distributions published here is still an internal build, such
    # as structured-data-models-nim, and is still refused.
    re.compile(
        r'\bstructured-data-models-'
        r'(?!(?:client|api|pql|connectors|relational-engine)'
        r'(?=$|[^\w-]|-(?:\d|\$?\{)))\w',
        re.IGNORECASE,
    ),
)

# Files that legitimately name internal infrastructure. A provenance record has
# to say where the bytes came from, and a contributor guide has to say where to
# push. Neither is a leak; both are reviewed deliberately.
INTERNAL_ALLOWLIST = {
    # A provenance record has to name the repository the bytes came from.
    'packages/structured-data-models-api/src/sdm_api/generated/PROVENANCE.json',
    # These two hold the patterns, so they name them by definition.
    'scripts/check_repo_hygiene.py',
    'scripts/generate_sdm_api.py',
}

SHIPPED_SUFFIXES = {
    '.py',
    # The Rust core ships inside the client wheel, so its sources are read
    # like any other.
    '.rs',
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

_OLD_BACKENDS = ('databricks', 'snowflake', 'postgres', 'sqlite', 'duckdb')

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
    # The old sdm_connectors spellings, which named a backend first:
    # connect('<backend>', ...) opened a connection for the caller, and
    # read('<backend>', ...) read a table or a file. A source now takes the
    # driver's own connection, `read` a file's path, and `read_table` an open
    # connection. The engine's per-backend `connect` functions take no backend
    # name, so they never match.
    *(
        f'{call}({quote}{backend}{quote}'
        for call, backends in (
            ('connect', _OLD_BACKENDS),
            ('read', (*_OLD_BACKENDS, 'local', 's3')),
        )
        for backend in backends
        for quote in ("'", '"')
    ),
)

# Where a retired name is legitimately mentioned, and which names: the checks
# that forbid them, and the record of what changed. Per name, so a file exempt
# for one retired interface is still checked for every other.
RETIRED_ALLOWLIST = {
    'scripts/check_repo_hygiene.py': frozenset(RETIRED_INTERFACES),
    'packages/structured-data-models-client/tests/test_local_notebooks.py': frozenset(
        {
            'RelationalClient',
            'PredictClient',
            'client.relational(',
            'client.tabular(',
            'nemotron_predict',
        }
    ),
    # The design rule that the model is an argument, never a method.
    'docs/about/architecture.md': frozenset({'client.relational('}),
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
        if path.suffix not in SPDX_SUFFIXES:
            continue
        if set(relative.parts) & SPDX_EXCLUDE_PARTS:
            continue
        body = path.read_text(encoding='utf-8', errors='replace')
        opening = body.splitlines()[:SPDX_GENERATED_LINES]
        if any(SPDX_GENERATED.match(line) for line in opening):
            continue
        if SPDX_LINE not in body:
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
        if path.suffix not in SHIPPED_SUFFIXES:
            continue
        if 'grammar' in Path(relative).parts:
            continue
        allowed = RETIRED_ALLOWLIST.get(relative, frozenset())
        text = path.read_text(encoding='utf-8', errors='replace')
        for retired in RETIRED_INTERFACES:
            if retired in text and retired not in allowed:
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


IN_PROCESS_REFERENCES = (
    'skills/tabular/references/in-process.md',
    'skills/relational/references/in-process.md',
)

# The release that first carries `in_process()`. Until the index has it the
# references have to build from a checkout; from that release on they must not,
# and the extra needs a floor no older than this. It is the feature's release
# rather than "any prerelease", so that 2.1.0rc1 does not read as a client
# without the feature and ask for the checkout back.
FIRST_IN_PROCESS_RELEASE = (2, 0, 0)
CHECKOUT_INSTALL = 'git clone -b v2-candidate'
PRERELEASE_SUFFIX = re.compile(r'(?:a|b|rc|\.post|\.dev)\d+$')
IN_PROCESS_FLOOR = re.compile(
    r'structured-data-models-client\[in-process\]>=(\d+(?:\.\d+)*)'
)
CLIENT_VERSION_FILE = (
    'packages/structured-data-models-client/src/sdm_client/_version.py'
)


def _release(version: str) -> tuple[tuple[int, ...], bool]:
    """The version's release numbers, and whether a prerelease follows them."""
    prerelease = PRERELEASE_SUFFIX.search(version)
    base = version[: prerelease.start()] if prerelease else version
    return tuple(int(part) for part in base.split('.')), prerelease is not None


def check_in_process_install() -> int:
    """The in-process references must install the way this version allows."""
    version_file = REPO_ROOT / CLIENT_VERSION_FILE
    found = re.search(
        r"__version__ = '([^']+)'", version_file.read_text(encoding='utf-8')
    )
    if found is None:
        print(f'no __version__ in {CLIENT_VERSION_FILE}', file=sys.stderr)
        return 1
    version = found.group(1)
    release, prerelease = _release(version)
    unpublished = release < FIRST_IN_PROCESS_RELEASE or (
        release == FIRST_IN_PROCESS_RELEASE and prerelease
    )
    first = '.'.join(str(part) for part in FIRST_IN_PROCESS_RELEASE)

    findings = []
    for relative in IN_PROCESS_REFERENCES:
        body = (REPO_ROOT / relative).read_text(encoding='utf-8')
        checkout = CHECKOUT_INSTALL in body
        if unpublished and not checkout:
            findings.append(
                f'{relative}: installs from the index, but {version} is '
                f'before {first}, the release that first carries in_process()'
            )
        if not unpublished and checkout:
            findings.append(
                f'{relative}: still builds from a checkout, but {version} is '
                f'{first} or later'
            )
        if not unpublished:
            floor = IN_PROCESS_FLOOR.search(body)
            if floor is None:
                findings.append(
                    f'{relative}: the extra names no version floor, so an '
                    'older client satisfies it'
                )
            elif _release(floor.group(1))[0] < FIRST_IN_PROCESS_RELEASE:
                findings.append(
                    f'{relative}: the extra floors at {floor.group(1)}, which '
                    f'a client without in_process() satisfies; {first} is the '
                    'first that carries it'
                )

    for finding in findings:
        print(f'in-process install: {finding}', file=sys.stderr)
    if findings:
        print(
            '\nA reader runs the first install that fits their client. When '
            'a release carries in_process(), drop the checkout block from '
            'both references and pin the extra to that version.',
            file=sys.stderr,
        )
        return 1
    print(f'the in-process references install the way {version} allows')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'check',
        choices=(
            'spdx',
            'internal',
            'retired',
            'repo-url',
            'in-process',
            'all',
        ),
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
    if args.check == 'in-process':
        return check_in_process_install()
    return (
        check_spdx()
        | check_internal()
        | check_retired()
        | check_repository_url()
        | check_in_process_install()
    )


if __name__ == '__main__':
    raise SystemExit(main())
