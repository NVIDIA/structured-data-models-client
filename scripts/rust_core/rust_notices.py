#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Writes, or checks, the notices of the Rust code the client wheel links.

The wheel carries `sdm_client._native`, built from `rust/bindings/sdm-py`
with its production features, and with it every crate that build links:
the crates `cargo tree` reaches from sdm-py over normal edges for each
shipped target (x86_64 and aarch64 Linux, decision KO2-40), proc macros and build scripts left out because nothing of
them is linked, and the workspace's own crates left out because they are
this repository's. For each, the file reproduces the licence, copyright and
notice files the crate ships, which is what its licence asks of a binary
distribution, and the Rust standard library is named after them.

    scripts/rust_core/rust_notices.py --write   regenerate the file
    scripts/rust_core/rust_notices.py --check   fail if it is out of date

`--check` runs in CI's `rust-lint`, where the crates' sources are present;
it needs `cargo` and reads `rust/Cargo.lock` without changing it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RUST = REPO / 'rust'
NOTICES = (
    REPO / 'packages/structured-data-models-client/THIRD-PARTY-NOTICES-RUST.txt'
)
#: The targets the wheel ships for (design, "Packaging"; KO2-40).
TARGETS = ('x86_64-unknown-linux-gnu', 'aarch64-unknown-linux-gnu')
#: The files a crate ships its terms in, at its root.
TERMS = re.compile(
    r'^(?:licen[cs]e|copying|copyright|notice|unlicense)(?:[-_.].*)?$',
    re.IGNORECASE,
)

#: Crates whose package leaves out the files its manifest includes, with the
#: crate of the same project and version whose files are the same: the
#: arrow-rs repository's LICENSE.txt and NOTICE.txt, which arrow-pyarrow's
#: manifest names but its published package does not carry.
SAME_TERMS_AS = {'arrow-pyarrow': 'arrow-schema'}

HEADER = f"""\
Third-party Rust code in structured-data-models-client

The client wheel carries the Rust core, the module sdm_client._native, built
from rust/bindings/sdm-py with its production features for the targets
{TARGETS[0]} and {TARGETS[1]}. These are the crates
linked into either build, other than this repository's own, each with the
licence, copyright and notice files it ships, and then the Rust standard
library. Written by scripts/rust_core/rust_notices.py from rust/Cargo.lock;
regenerate it with --write rather than editing it.
"""

STD = """\
The Rust standard library
=========================

Linked from the pinned toolchain, Rust {toolchain}, under MIT OR Apache-2.0,
Copyright (c) The Rust Project Contributors. Its copyright notices, those of
the code it bundles included, are in the Rust repository's COPYRIGHT file at
that version: https://github.com/rust-lang/rust/blob/{toolchain}/COPYRIGHT.
The texts of both licences are reproduced in the crate sections above.
"""


def _cargo(*arguments: str) -> str:
    return subprocess.run(
        ['cargo', *arguments],
        cwd=RUST,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _linked() -> list[tuple[str, str]]:
    """The third-party crates the shipped builds link, for any of
    :data:`TARGETS`, as (name, version)."""
    arguments = ['-e', 'normal,no-proc-macro', '--prefix', 'none']
    for target in TARGETS:
        arguments += ['--target', target]
    tree = _cargo(
        'tree', '--locked', '-p', 'sdm-py', *arguments, '--format', '{p}'
    )
    crates = set()
    for line in tree.splitlines():
        # `name vX.Y.Z`, then `(path)` for a workspace crate, `(*)` for one
        # already listed.
        parts = line.split()
        if not parts or '(' in line.replace('(*)', ''):
            continue
        crates.add((parts[0], parts[1].removeprefix('v')))
    return sorted(crates)


def _packages() -> dict[tuple[str, str], dict]:
    metadata = json.loads(
        _cargo('metadata', '--locked', '--format-version', '1')
    )
    return {(p['name'], p['version']): p for p in metadata['packages']}


def _toolchain() -> str:
    text = (RUST / 'rust-toolchain.toml').read_text()
    found = re.search(r'^channel = "([^"]+)"', text, re.MULTILINE)
    assert found, 'rust-toolchain.toml names no channel'
    return found.group(1)


def _text(path: pathlib.Path) -> str:
    text = path.read_text(encoding='utf-8', errors='replace')
    return '\n'.join(line.rstrip() for line in text.strip().splitlines())


def render() -> str:
    packages = _packages()
    crates = _linked()
    index = []
    sections = []
    seen: dict[str, str] = {}
    for name, version in crates:
        package = packages[(name, version)]
        root = pathlib.Path(package['manifest_path']).parent
        licence = package.get('license') or 'see its licence file'
        where = package.get('repository') or ''
        index.append(f'{name} {version}  {licence}  {where}'.rstrip())
        source = root
        if name in SAME_TERMS_AS:
            sibling = packages[(SAME_TERMS_AS[name], version)]
            source = pathlib.Path(sibling['manifest_path']).parent
        files = sorted(
            (
                p
                for p in source.iterdir()
                if p.is_file() and TERMS.match(p.name)
            ),
            key=lambda p: p.name.lower(),
        )
        if package.get('license_file'):
            named = root / package['license_file']
            if named.is_file() and named not in files:
                files.append(named)
        if not files:
            sys.exit(
                f'{name} {version} ships no licence file; its terms have to '
                f'be reproduced, so add them here before it ships'
            )
        title = f'{name} {version}'
        body = [title, '=' * len(title), f'Licence: {licence}']
        if source != root:
            body.append(
                f'Its package carries no terms files; these are the same '
                f"project's, as {SAME_TERMS_AS[name]} {version} ships them."
            )
        for path in files:
            text = _text(path)
            digest = hashlib.sha256(text.encode()).hexdigest()
            body.append('')
            body.append(f'--- {path.name} ---')
            if digest in seen:
                body.append(f'(the same text as {seen[digest]} above)')
            else:
                seen[digest] = f'{path.name} of {title}'
                body.append(text)
        sections.append('\n'.join(body))
    return (
        HEADER
        + '\nCrates\n======\n\n'
        + '\n'.join(index)
        + '\n\n\n'
        + '\n\n\n'.join(sections)
        + '\n\n\n'
        + STD.format(toolchain=_toolchain())
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    arguments = parser.parse_args()
    rendered = render()
    if arguments.write:
        NOTICES.write_text(rendered, encoding='utf-8')
        print(f'wrote {NOTICES.relative_to(REPO)}')
        return 0
    current = NOTICES.read_text(encoding='utf-8') if NOTICES.is_file() else ''
    if current != rendered:
        print(
            f'{NOTICES.relative_to(REPO)} is out of date with rust/Cargo.lock; '
            f'run scripts/rust_core/rust_notices.py --write',
            file=sys.stderr,
        )
        return 1
    print(f'{NOTICES.relative_to(REPO)} lists every linked crate')
    return 0


if __name__ == '__main__':
    sys.exit(main())
