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
claiming an inventory. What it does guarantee is that every version reached is
attributed separately, because two paths through the graph can want different
majors under different licenses.

Three modes. Only the first needs the network, which is why the other two
are the ones CI runs:

    --refresh   read npm and rewrite MERMAID_COMPONENTS.json
    --write     render from that file and splice it into both LICENSE files
    --check     render from that file, and verify both LICENSE files carry
                it and the text of every license it names

A mermaid upgrade is `--refresh` then `--write`, which is also what
VENDORED.md tells you to do.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
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
    'chevrotain': ('ships the stock Apache-2.0 text with no NOTICE file'),
    '@chevrotain/types': (
        'ships the stock Apache-2.0 text with no NOTICE file'
    ),
    '@chevrotain/regexp-to-ast': (
        'ships the stock Apache-2.0 text with no NOTICE file'
    ),
    '@chevrotain/gast': ('ships the stock Apache-2.0 text with no NOTICE file'),
    '@chevrotain/utils': (
        'ships the stock Apache-2.0 text with no NOTICE file'
    ),
    '@chevrotain/cst-dts-gen': (
        'ships the stock Apache-2.0 text with no NOTICE file'
    ),
    'robust-predicates': (
        'is released into the public domain under the Unlicense'
    ),
}

#: khroma declares no license field, so a scanner reports it as unknown. Its
#: license file at the tag grants MIT, and that is the source of this entry.
DECLARED_BY_HAND = {'khroma': 'MIT'}

#: Packages inside the bundle that no dependency graph reaches, because the
#: dependency carrying them is itself a bundle and inlines them. A graph
#: describes what npm installs, not what a build inlines.
#:
#: `@mermaid-js/parser` inlines langium, which is what brings chevrotain and
#: the vscode language-server packages. js-yaml is inlined too, at the version
#: the bundle states for it. Anything else found in the bundle later is added
#: here, and its own dependencies come with it.
INLINED = {'langium': '*', 'js-yaml': '4.1.1'}

#: Versions the bundle states for itself, which are therefore not resolved.
#: Reading one out of the file is better evidence than resolving a range, and
#: it matters: dompurify 3.4.16 ships a license file with no notice in it,
#: while 3.4.0, the version the bundle names, carries Cure53's.
BUNDLE_RECORDS = {
    'dompurify': '3.4.0',
    'js-yaml': '4.1.1',
    'katex': '0.16.25',
    'cytoscape': '3.33.1',
    'lodash-es': '4.18.1',
}

#: Packages whose license file grants more than their metadata declares.
#: marked is MIT, but the Markdown it derives from is BSD-3-Clause and its
#: notice travels in the same file.
ALSO_LICENSED = {
    'marked': 'MIT and BSD-3-Clause',
    'd3-geo': 'ISC and MIT',
    'd3-scale-chromatic': 'ISC and Apache-2.0',
}

#: A line only one license writes, used to notice when a package's file grants
#: more than its metadata declares. d3-geo says ISC and carries GeographicLib
#: under MIT; d3-scale-chromatic says ISC and carries ColorBrewer under
#: Apache-2.0. Neither is visible in package metadata, so `refresh` reads the
#: file for them rather than trusting the declaration.
SIGNATURES = {
    'MIT': 'Permission is hereby granted, free of charge',
    'Apache-2.0': 'Licensed under the Apache License',
    'BSD-3-Clause': 'Redistribution and use in source and binary forms',
    'ISC': 'Permission to use, copy, modify, and/or distribute',
    'MPL-2.0': 'Mozilla Public License',
    'Unlicense': 'unencumbered software released into the public domain',
    'Python-2.0': 'PYTHON SOFTWARE FOUNDATION LICENSE',
}

#: Terms a package carries that the shared license texts below do not
#: reproduce. Markdown's third clause names Markdown rather than the project
#: using it, so quoting the generic BSD-3-Clause text would not be quoting
#: what marked actually ships.
EXTRA_TERMS = {
    'marked': (
        'and the Markdown it derives from, License: BSD-3-Clause, whose',
        'third clause names Markdown rather than the project, and so is',
        'reproduced here rather than taken from the shared text below:',
        '',
        '  Redistribution and use in source and binary forms, with or without',
        '  modification, are permitted provided that the following conditions',
        '  are met:',
        '',
        '  * Redistributions of source code must retain the above copyright',
        '    notice, this list of conditions and the following disclaimer.',
        '  * Redistributions in binary form must reproduce the above copyright',
        '    notice, this list of conditions and the following disclaimer in',
        '    the documentation and/or other materials provided with the',
        '    distribution.',
        '  * Neither the name "Markdown" nor the names of its contributors may',
        '    be used to endorse or promote products derived from this software',
        '    without specific prior written permission.',
        '',
        '  This software is provided by the copyright holders and contributors',
        '  "as is" and any express or implied warranties, including, but not',
        '  limited to, the implied warranties of merchantability and fitness',
        '  for a particular purpose are disclaimed. In no event shall the',
        '  copyright owner or contributors be liable for any direct, indirect,',
        '  incidental, special, exemplary, or consequential damages',
        '  (including, but not limited to, procurement of substitute goods or',
        '  services; loss of use, data, or profits; or business interruption)',
        '  however caused and on any theory of liability, whether in contract,',
        '  strict liability, or tort (including negligence or otherwise)',
        '  arising in any way out of the use of this software, even if advised',
        '  of the possibility of such damage.',
    ),
}

#: A copyright notice, as licenses actually write one: at the start of its
#: line, and capitalised. Case matters. Apache-2.0's own body wraps onto a
#: line reading "copyright license to reproduce, prepare Derivative Works
#: of,", which a case-insensitive pattern reads as an attribution and writes
#: into this repository as though a package had asserted it.
NOTICE = re.compile(
    r'^(?:Original [\w.-]+ copyright: )?Copyright\b'
    r'|^Based on .*\bcopyright\b'
)

#: Lines that say the word and are not a notice: a license explaining what it
#: grants, the placeholder in Apache's appendix, and lodash waiving rights
#: over its sample code.
NOT_A_NOTICE = re.compile(
    r'COPYRIGHT HOLDER|copyright notice|copyright owner|copyright license'
    r'|Copyright and related rights',
    re.I,
)

#: How far a notice can run past its first line before this stops reading.
NOTICE_LINES = 4

#: Where a license stops attributing and starts granting. Most files put a
#: blank line there; import-meta-resolve puts the grant on the line directly
#: under the last notice, and without this the grant is recorded as though it
#: were part of the attribution.
BODY_OPENS = re.compile(
    r'^(?:Permission is hereby|Permission to use|Redistribution and use'
    r'|Licensed under|This software|This is free|THE SOFTWARE'
    r'|is free software|Everyone is permitted)',
    re.I,
)


def _read(url: str, attempts: int = 4) -> bytes:
    """One request, retried, because a dropped connection is not an answer.

    Treating a transient failure as "this package has no license file" would
    quietly drop a notice, which is the one outcome worth retrying for.
    """
    request = urllib.request.Request(
        url, headers={'User-Agent': 'sdm-client-notices'}
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError:
            raise
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError('unreachable')


def _version_parts(version: str) -> list[int]:
    return [int(part) for part in re.findall(r'\d+', version)[:3]] + [0, 0, 0]


def resolve(packument: dict[str, Any], spec: str) -> str | None:
    """A released version satisfying one range, the way npm picks: the highest.

    Not a semver implementation. It handles what this graph uses: exact pins,
    carets including the zero-major rule, tildes, partial versions, hyphen
    ranges, comparator pairs, and alternatives separated by ``||``. A range it
    cannot read returns None rather than a wrong answer, which is how the
    hyphen range in d3-sankey was found.
    """
    released = [
        v for v in packument['versions'] if not re.search(r'[a-zA-Z-]', v)
    ]
    fitting: list[str] = []
    for alternative in spec.split('||'):
        alternative = alternative.strip()
        if not alternative or alternative == '*':
            return packument['dist-tags']['latest']
        if alternative in packument['versions']:
            fitting.append(alternative)
            continue
        matched = _satisfying(released, alternative)
        if matched is None:
            return None
        fitting.extend(matched)
    return max(fitting, key=_version_parts, default=None)


def _satisfying(released: list[str], spec: str) -> list[str] | None:
    """Every released version one range admits, or None if it is unreadable."""
    hyphen = re.fullmatch(r'([\d.]+)\s+-\s+([\d.]+)', spec)
    if hyphen:
        # `1 - 2` is >=1.0.0 and <3.0.0: an upper bound written as a bare
        # major includes that whole major.
        low, high = (_version_parts(part) for part in hyphen.groups())
        ceiling = [high[0] + 1] if high[1:3] == [0, 0] else None
        return [
            v
            for v in released
            if _version_parts(v) >= low
            and (
                _version_parts(v)[:1] < ceiling
                if ceiling is not None
                else _version_parts(v) <= high
            )
        ]

    comparators = re.findall(r'([<>]=?)\s*([\d.]+)', spec)
    if comparators and re.fullmatch(r'[\s<>=\d.]+', spec):
        kept = released
        for operator, bound in comparators:
            edge = _version_parts(bound)
            kept = [
                v
                for v in kept
                if (_version_parts(v) >= edge if operator == '>=' else None)
                or (_version_parts(v) > edge if operator == '>' else None)
                or (_version_parts(v) <= edge if operator == '<=' else None)
                or (_version_parts(v) < edge if operator == '<' else None)
            ]
        return kept

    floor = _version_parts(spec)
    if spec.startswith('^'):
        # A caret below 1.0.0 lets only the patch move: a zero-major minor
        # bump is a breaking change by convention, and npm reads it that way.
        leading = next((i for i, part in enumerate(floor[:3]) if part), 2)
        fits = lambda p: p[: leading + 1] == floor[: leading + 1] and p >= floor  # noqa: E731
    elif spec.startswith('~'):
        fits = lambda p: p[:2] == floor[:2] and p >= floor  # noqa: E731
    elif re.fullmatch(r'\d+(\.\d+)?', spec):
        given = len(spec.split('.'))
        fits = lambda p: p[:given] == floor[:given]  # noqa: E731
    else:
        return None
    return [v for v in released if fits(_version_parts(v))]


def notices_of(name: str, version: str) -> tuple[list[str], str]:
    """The copyright notices a package's own license file carries.

    Each candidate filename is tried until one yields a notice, so a package
    that ships an empty or stock license file is not read as having none while
    another file holds the real one. A package with no notice anywhere is what
    `refresh` refuses rather than records.
    """
    for filename in LICENSE_FILES:
        try:
            text = _read(f'https://unpkg.com/{name}@{version}/{filename}')
        except Exception:
            continue
        body = text.decode('utf-8', 'replace')
        lines = body.splitlines()
        found: list[str] = []
        for position, raw in enumerate(lines):
            line = raw.strip()
            if not NOTICE.match(line) or NOT_A_NOTICE.search(line):
                continue
            # A notice runs to the blank line after it, to the next
            # notice, or to where the license starts granting, whichever
            # comes first, and no further than NOTICE_LINES. Stopping at the
            # first line costs the rest: Gruber's "All rights reserved." sits
            # under a URL, and the CWI notice wraps onto "The Netherlands.
            # All rights reserved." Running past it costs more, which
            # import-meta-resolve shows: it puts the MIT grant directly under
            # its last notice, with no blank line between.
            block = [line]
            for following in lines[position + 1 : position + NOTICE_LINES]:
                following = following.strip()
                if (
                    not following
                    or NOTICE.match(following)
                    or NOT_A_NOTICE.search(following)
                    or BODY_OPENS.match(following)
                ):
                    break
                block.append(following)
            for part in block:
                if part not in found:
                    found.append(part)
        if found:
            return found, body
    return [], ''


def refresh() -> int:
    """Resolve the closure and read every notice in it.

    Keyed by name *and* version, because two paths through the graph can want
    different majors under different licenses: d3-sankey reaches d3-array 2,
    which is BSD-3-Clause, while mermaid reaches d3-array 3, which is ISC.
    Collapsing them attributes one under the other's terms.
    """
    root = json.loads(_read('https://registry.npmjs.org/mermaid/11.15.0'))
    components: dict[str, Any] = {}
    seen: set[tuple[str, str]] = set()
    queue = deque([*root['dependencies'].items(), *INLINED.items()])
    while queue:
        name, spec = queue.popleft()
        if (name, spec) in seen or name.startswith('@types/'):
            continue
        seen.add((name, spec))
        packument = json.loads(_read(f'https://registry.npmjs.org/{name}'))
        version = BUNDLE_RECORDS.get(name) or resolve(packument, spec)
        if version is None or version not in packument['versions']:
            print(f'{name}: no released version fits {spec}', file=sys.stderr)
            return 1
        released = packument['versions'][version]
        components[f'{name}@{version}'] = {
            'name': name,
            'version': version,
            'license': (
                ALSO_LICENSED.get(name)
                or DECLARED_BY_HAND.get(name)
                or released.get('license')
            ),
            'notices': [],
        }
        for dependency, dependency_spec in (
            released.get('dependencies') or {}
        ).items():
            if (dependency, dependency_spec) not in seen:
                queue.append((dependency, dependency_spec))

    silent, undeclared = [], []
    for key, component in components.items():
        component['notices'], body = notices_of(
            component['name'], component['version']
        )
        declared = set(
            re.split(
                r'\s+(?:AND|OR|and|or)\s+',
                str(component['license'] or '').strip('()'),
            )
        )
        beyond = {
            name
            for name, signature in SIGNATURES.items()
            if signature.lower() in body.lower()
        } - declared
        if beyond:
            undeclared.append((key, sorted(beyond)))
        if (
            not component['notices']
            and component['name'] not in WITHOUT_A_NOTICE
        ):
            silent.append(key)
    if undeclared:
        print(
            f'{undeclared} carry license terms their metadata does not '
            f'declare. Read each file and record the whole expression in '
            f'ALSO_LICENSED.',
            file=sys.stderr,
        )
        return 1
    if silent:
        print(
            f'{silent} have no copyright line and are not recorded as having '
            f'none. Read each license and either take the notice from it or '
            f'record in WITHOUT_A_NOTICE why it has none.',
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
        '  A package appears once per version reached, because two paths can want',
        '  different majors under different terms: d3-array is BSD-3-Clause at 2',
        '  and ISC at 3, and both are reached.',
        '',
        '  Versions are the ones the bundle states, where it states one, and',
        '  otherwise the ones npm resolves from what mermaid declares. They say',
        '  which version each notice was read from rather than which version',
        '  upstream built with.',
        '',
    ]
    for key in sorted(recorded['components']):
        component = recorded['components'][key]
        name = component['name']
        head = f'    {name} {component["version"]}'
        lines.append(f'{head.ljust(52)}License: {component["license"]}')
        for notice in component['notices']:
            lines.append(f'      {notice}')
        if name in WITHOUT_A_NOTICE:
            lines.append(f'      {name} {WITHOUT_A_NOTICE[name]},')
            lines.append('      so it has no notice to reproduce.')
        for extra in EXTRA_TERMS.get(name, ()):
            lines.append(f'      {extra}'.rstrip())
    lines.append('')
    return '\n'.join(lines) + '\n'


#: Lines only the full text of each license contains, all of which must be
#: present. One line is not enough: `Apache License` appears in this file's
#: own prose, so checking for it passed with the whole Apache body deleted.
#: Several lines, spread through each text, is what makes a deletion show.
#:
#: MPL-2.0 is deliberately absent: its text is not here, and the only
#: component under it offers Apache-2.0 as an alternative, which is.
LICENSE_TEXTS = {
    'MIT': (
        'Permission is hereby granted, free of charge',
        'The above copyright notice and this permission notice shall be',
        'THE SOFTWARE IS PROVIDED "AS IS"',
    ),
    'ISC': (
        'Permission to use, copy, modify, and/or distribute this software',
        'THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS',
    ),
    'BSD-3-Clause': (
        'Redistribution and use in source and binary forms',
        'Neither the name of the copyright holder nor the names',
        'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS',
    ),
    'Apache-2.0': (
        'TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION',
        'Grant of Copyright License',
        'Disclaimer of Warranty',
        'END OF TERMS AND CONDITIONS',
    ),
    'Unlicense': (
        'unencumbered software released into the public domain',
        'In jurisdictions that recognize copyright laws',
    ),
    'Python-2.0': (
        'HISTORY OF THE SOFTWARE',
        'PYTHON SOFTWARE FOUNDATION LICENSE',
    ),
}


def texts_missing_for(licence: str) -> list[str]:
    """The licenses a component is under whose text this file does not carry.

    An ``OR`` is a choice, so carrying either text satisfies it: DOMPurify is
    Apache-2.0 or MPL-2.0, and this distribution relies on Apache, which the
    file says where it attributes it. An ``AND`` needs both.
    """
    recorded = json.loads(COMPONENTS.read_text(encoding='utf-8'))
    missing: list[str] = []
    for component in recorded['components'].values():
        declared = str(component['license'] or '').strip('()')
        if not declared or declared == 'None':
            continue
        if ' OR ' in declared:
            options = [part.strip() for part in declared.split(' OR ')]
            if any(_carries(licence, option) for option in options):
                continue
            missing.extend(options)
            continue
        for part in re.split(r'\s+(?:AND|and)\s+', declared):
            if not _carries(licence, part.strip()):
                missing.append(part.strip())
    return sorted(set(missing))


def _carries(licence: str, name: str) -> bool:
    markers = LICENSE_TEXTS.get(name)
    return bool(markers) and all(marker in licence for marker in markers)


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
