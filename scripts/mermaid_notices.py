#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attribute what is inside the vendored mermaid bundle, without guessing.

`mermaid.min.js` is upstream's published build: minified, tree-shaken, and
carrying no manifest of what went into it. Reading components out of it by hand
does not converge. Three rounds of it here each missed more, and the ones
missed last were transitive: `cose-base` reached through two Cytoscape layouts,
`chevrotain-allstar` through the parser.

So this does not read the bundle. It resolves the dependency closure mermaid
declares, and attributes all of it. Naming a component that was tree-shaken out
costs nothing; omitting one that was kept is the failure that matters.

That closure is not by itself a superset, which is the trap: `@mermaid-js/parser`
is a bundle too, and inlines langium, so the graph never reaches langium's own
dependencies, which is where chevrotain and the vscode packages live. Those
roots are named in `INLINED` and traversed like any other.

Nothing here can prove the result complete. A published bundle records nothing
about what went into it, and the rendered section says that rather than
claiming an inventory.

Two modes, because the first needs the network and CI must not:

    --refresh   read npm and rewrite MERMAID_COMPONENTS.json
    --check     render from that file and verify both LICENSE files carry it
    --write     render from that file and splice it into both LICENSE files

A mermaid upgrade is `--refresh` then `--write`, which is also what
VENDORED.md tells you to do.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = (
    REPO_ROOT
    / 'packages'
    / 'sdm-relational-engine'
    / 'src'
    / 'sdm_relational_engine'
    / 'rfm'
    / 'assets'
)
COMPONENTS = ASSETS / 'MERMAID_COMPONENTS.json'
LICENSES = (
    REPO_ROOT / 'LICENSE',
    REPO_ROOT / 'packages' / 'sdm-relational-engine' / 'LICENSE',
)

OPENS = '  The components mermaid declares, transitively.\n'
CLOSES = (
    'MIT License, as it applies to the MIT-licensed components listed above:'
)

#: Files a package may keep its license in, in the order npm tooling reads them.
LICENSE_FILES = (
    'LICENSE',
    'LICENSE.md',
    'LICENSE.txt',
    'license',
    'license.md',
    'LICENCE',
)

#: Packages whose license grants no copyright, so there is none to reproduce.
#: Requiring one would mean inventing it.
WITHOUT_A_NOTICE = {
    'chevrotain': 'ships the stock Apache-2.0 text with no NOTICE file',
    'robust-predicates': 'is released into the public domain under the Unlicense',
}

#: khroma declares no license field, so a scanner reports it as unknown. Its
#: license file at the tag grants MIT, and that is the source of this entry.
DECLARED_BY_HAND = {'khroma': 'MIT'}

#: Packages inside the bundle that no dependency graph reaches, because the
#: dependency carrying them is itself a bundle and inlines them. A graph
#: describes what npm installs, not what a build inlines.
#:
#: `@mermaid-js/parser` inlines langium, which is what brings chevrotain and
#: the vscode language-server packages; js-yaml is inlined too, and declares
#: its version inside the file. Anything else found in the bundle later is
#: added here, and its own dependencies come with it.
INLINED = {'langium': '*', 'js-yaml': '*'}


def _read(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={'User-Agent': 'sdm-client-notices'}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _version_parts(version: str) -> list[int]:
    return [int(part) for part in re.findall(r'\d+', version)[:3]] + [0, 0, 0]


def resolve(packument: dict[str, Any], spec: str) -> str | None:
    """The version npm would install for one range.

    Prereleases are excluded: npm does not pick one for a plain range, and a
    bundle built from a prerelease would be a different question.
    """
    spec = spec.split('||')[0].strip()
    released = [
        v for v in packument['versions'] if not re.search(r'[a-zA-Z-]', v)
    ]
    if not spec or spec == '*':
        return packument['dist-tags']['latest']

    floor = _version_parts(spec)
    if spec.startswith('^'):
        fits = lambda p: p[0] == floor[0] and p >= floor  # noqa: E731
    elif spec.startswith('~'):
        fits = lambda p: p[:2] == floor[:2] and p >= floor  # noqa: E731
    elif spec[0] in '><=':
        fits = lambda p: p >= floor  # noqa: E731
    elif spec in packument['versions']:
        return spec
    else:
        fits = lambda p: p[0] == floor[0]  # noqa: E731

    fitting = [v for v in released if fits(_version_parts(v))]
    return max(fitting, key=_version_parts, default=None)


def notices_of(name: str, version: str) -> list[str]:
    """The copyright lines a package's own license file carries."""
    for filename in LICENSE_FILES:
        try:
            text = _read(f'https://unpkg.com/{name}@{version}/{filename}')
        except Exception:
            continue
        found: list[str] = []
        for line in text.decode('utf-8', 'replace').splitlines():
            line = line.strip()
            if not re.match(
                r'^(Original [\w-]+ copyright: )?Copyright', line, re.I
            ):
                continue
            if re.search(
                r'COPYRIGHT HOLDER|copyright notice|copyright owner', line, re.I
            ):
                continue
            if line not in found:
                found.append(line)
        if found:
            return found
    return []


def refresh() -> int:
    """Resolve the closure and read every notice in it."""
    root = json.loads(_read('https://registry.npmjs.org/mermaid/11.15.0'))
    components: dict[str, Any] = {}
    queue = deque([*root['dependencies'].items(), *INLINED.items()])
    while queue:
        name, spec = queue.popleft()
        if name in components or name.startswith('@types/'):
            continue
        packument = json.loads(_read(f'https://registry.npmjs.org/{name}'))
        version = resolve(packument, spec)
        if version is None:
            print(f'{name}: no version fits {spec}', file=sys.stderr)
            return 1
        released = packument['versions'][version]
        components[name] = {
            'version': version,
            'license': DECLARED_BY_HAND.get(name) or released.get('license'),
            'notices': [],
        }
        for dependency, dependency_spec in (
            released.get('dependencies') or {}
        ).items():
            if dependency not in components:
                queue.append((dependency, dependency_spec))

    for name, component in components.items():
        component['notices'] = notices_of(name, component['version'])
        if not component['notices'] and name not in WITHOUT_A_NOTICE:
            print(
                f'{name}@{component["version"]} has no copyright line and is '
                f'not recorded as having none. Read its license and either add '
                f'the notice or say here why it has one.',
                file=sys.stderr,
            )
            return 1

    COMPONENTS.write_text(
        json.dumps(
            {'mermaid': '11.15.0', 'components': components},
            indent=2,
            sort_keys=True,
        )
        + '\n',
        encoding='utf-8',
    )
    print(f'recorded {len(components)} components')
    return 0


def render() -> str:
    """The attribution section, from the recorded components."""
    recorded = json.loads(COMPONENTS.read_text(encoding='utf-8'))
    lines = [
        OPENS.rstrip('\n'),
        '',
        f'  mermaid {recorded["mermaid"]} is published as a built file, minified and',
        '  tree-shaken, and carries no record of what went into it. The components',
        '  below are the ones it declares transitively, together with those its own',
        '  bundled dependencies inline, which no dependency graph reaches and which',
        '  are named in scripts/mermaid_notices.py.',
        '',
        '  This is not an inventory of the file. It is broader in one direction,',
        '  since tree-shaking drops some of what is declared, and it cannot be shown',
        '  complete in the other, since a published bundle records nothing about',
        '  what went into it. Everything reachable is attributed, because naming a',
        '  component that was dropped costs nothing and omitting one that was kept',
        '  is the failure that matters.',
        '',
        '  Versions are the ones npm resolves from what mermaid declares. The bundle',
        '  itself records a version only for a few, so these say which version each',
        '  notice was read from rather than which version upstream built with.',
        '',
    ]
    for name, component in sorted(recorded['components'].items()):
        head = f'    {name} {component["version"]}'
        lines.append(f'{head.ljust(52)}License: {component["license"]}')
        for notice in component['notices']:
            lines.append(f'      {notice}')
        if name in WITHOUT_A_NOTICE:
            lines.append(f'      {name} {WITHOUT_A_NOTICE[name]},')
            lines.append('      so it has no notice to reproduce.')
    lines.append('')
    return '\n'.join(lines) + '\n'


#: A line that only the full text of each license contains, so that carrying
#: the text can be checked rather than assumed.
LICENSE_TEXTS = {
    'MIT': 'MIT License, as it applies',
    'ISC': 'ISC License, as it applies',
    'BSD-3-Clause': 'BSD 3-Clause License, as it applies',
    'Apache-2.0': 'Apache License',
    'MPL-2.0': 'mozilla.org/MPL',
    'Unlicense': 'unencumbered software',
    'Python-2.0': 'HISTORY OF THE SOFTWARE',
}


def texts_missing_for(licence: str) -> list[str]:
    """The licenses a component is under whose text this file does not carry.

    A component arriving under a license nobody has read is how an upgrade
    would quietly ship terms the repository does not reproduce.
    """
    recorded = json.loads(COMPONENTS.read_text(encoding='utf-8'))
    wanted = set()
    for component in recorded['components'].values():
        declared = str(component['license'] or '').strip('()')
        wanted.update(re.split(r'\s+(?:OR|AND)\s+', declared))
    return [
        name
        for name in sorted(wanted - {'', 'None'})
        if LICENSE_TEXTS.get(name) is None or LICENSE_TEXTS[name] not in licence
    ]


def splice(text: str, section: str) -> str:
    start = text.index(OPENS)
    end = text.index(CLOSES)
    return text[:start] + section + '\n' + text[end:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--refresh', action='store_true', help='read npm')
    group.add_argument(
        '--write', action='store_true', help='update the LICENSE files'
    )
    group.add_argument('--check', action='store_true', help='verify them')
    args = parser.parse_args()

    if args.refresh:
        return refresh()

    section = render()
    if args.write:
        for path in LICENSES:
            path.write_text(
                splice(path.read_text(encoding='utf-8'), section),
                encoding='utf-8',
            )
        print(f'wrote the section into {len(LICENSES)} license files')
        return 0

    for path in LICENSES:
        missing = texts_missing_for(path.read_text(encoding='utf-8'))
        if missing:
            print(
                f'{path.relative_to(REPO_ROOT)} attributes components under '
                f'{missing} but does not carry those license texts.',
                file=sys.stderr,
            )
            return 1

    stale = [
        str(p.relative_to(REPO_ROOT))
        for p in LICENSES
        if section not in p.read_text(encoding='utf-8')
    ]
    if stale:
        print(
            f'{stale} do not carry the attribution rendered from '
            f'{COMPONENTS.relative_to(REPO_ROOT)}. Run '
            f'`python scripts/mermaid_notices.py --write`.',
            file=sys.stderr,
        )
        return 1
    print('both license files carry the recorded attribution and its texts')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
