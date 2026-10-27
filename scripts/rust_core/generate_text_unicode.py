#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render native lowercase profiles from pinned official Unicode data.

Pass --inputs with UnicodeData-VERSION.txt, SpecialCasing-VERSION.txt and
DerivedCoreProperties-VERSION.txt for each declared profile, together with
LICENSE-UNICODE.txt, then --write.
This generator performs no downloads and imports no product code.
It derives data only; the bounded lowercase algorithm remains first-party Rust.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import unicode_data

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'rust/crates/sdm-engine/src/words/casing/data.rs'
LICENSE = ROOT / 'rust/crates/sdm-engine/data/LICENSE-UNICODE.txt'
PROVENANCE = ROOT / 'rust/crates/sdm-engine/data/CASING-PROVENANCE.json'
FILES = ('UnicodeData', 'SpecialCasing', 'DerivedCoreProperties')
IDENTITY_INPUT = (0x0130, 0x03A3, 0x2C2F, 0x1C89, 0xA7CE)
INPUTS = {
    '13.0.0': (
        'bdbffbbfc8ad4d3a6d01b5891510458f3d36f7170422af4ea2bed3211a73e8bb',
        '6424312f1dc39b22e0ff9c0ffb13dfad424d9b03e6a6dc6bca941f6bf5ef1ffd',
        'a5d45f59b39deaab3c72ce8c1a2e212a5e086dff11b1f9d5bb0e352642e82248',
    ),
    '14.0.0': (
        '36018e68657fdcb3485f636630ffe8c8532e01c977703d2803f5b89d6c5feafb',
        'c667b45908fd269af25fd55d2fc5bbc157fb1b77675936e25c513ce32e080334',
        'e3eddd7d469cd1b0feed7528defad1a1cc7c6a9ceb0ae4446a6d10921ed2e7bc',
    ),
    '15.0.0': (
        '806e9aed65037197f1ec85e12be6e8cd870fc5608b4de0fffd990f689f376a73',
        '78b29c64b5840d25c11a9f31b665ee551b8a499eca6c70d770fcad7dd710f494',
        'd367290bc0867e6b484c68370530bdd1a08b6b32404601b8c7accaf83e05628d',
    ),
    '15.1.0': (
        '2fc713e6a31a87c4850a37fe2caffa4218180fadb5de86b43a143ddb4581fb86',
        '55a477efd933a52cd27e6a9bf70265bb2d8814af31aab07767abc8eb421f27ef',
        'f55d0db69123431a7317868725b1fcbf1eab6b265d756d1bd7f0f6d9f9ee108b',
    ),
    '16.0.0': (
        'ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f',
        '8d5de354eef79f2395a54c9c7dcebbaf3d30fc962d0f85611ea97aa973a0c451',
        '39d35161f2954497f69e08bdb9e701493f476a3d30222de20028feda36c1dabd',
    ),
    '17.0.0': (
        '2e1efc1dcb59c575eedf5ccae60f95229f706ee6d031835247d843c11d96470c',
        'efc25faf19de21b92c1194c111c932e03d2a5eaf18194e33f1156e96de4c9588',
        '24c7fed1195c482faaefd5c1e7eb821c5ee1fb6de07ecdbaa64b56a99da22c08',
    ),
}


def scalar(point: int) -> int:
    if not 0 <= point <= 0x10FFFF or 0xD800 <= point <= 0xDFFF:
        raise ValueError(f'non-scalar Unicode mapping: {point:X}')
    return point


def records(data: bytes) -> list[list[str]]:
    rows = []
    for raw in data.decode('utf-8').splitlines():
        line = raw.split('#', 1)[0].strip()
        if line:
            rows.append([field.strip() for field in line.split(';')])
    return rows


def mappings(data: bytes) -> tuple[dict[int, int], tuple[int, ...]]:
    simple = {}
    spaces = []
    first = None
    for row in records(data):
        if len(row) != 15:
            raise ValueError('UnicodeData record does not have 15 fields')
        point = int(row[0], 16)
        if row[1].endswith(', First>'):
            if first is not None:
                raise ValueError('nested UnicodeData range')
            first = row
            continue
        if row[1].endswith(', Last>'):
            if first is None or first[2:] != row[2:] or row[13]:
                raise ValueError('inconsistent UnicodeData range')
            points = range(int(first[0], 16), point + 1)
            first = None
        else:
            if first is not None:
                raise ValueError('unclosed UnicodeData range')
            points = (point,)
        if row[13]:
            key, value = scalar(point), scalar(int(row[13], 16))
            if key in simple or key == value:
                raise ValueError('duplicate or identity simple mapping')
            simple[key] = value
        # CPython's documented generated whitespace predicate, not the
        # potentially newer Rust standard library's whitespace database.
        if row[2] == 'Zs' or row[4] in ('WS', 'B', 'S'):
            spaces.extend(scalar(value) for value in points)
    if first is not None:
        raise ValueError('unclosed UnicodeData range')
    if spaces != sorted(set(spaces)):
        raise ValueError('Unicode whitespace is not unique and ordered')
    return simple, tuple(spaces)


def full_mappings(data: bytes, simple: dict[int, int]) -> tuple:
    full = {}
    final_sigma = []
    for row in records(data):
        # Unconditional lines have an empty final field; conditional lines
        # additionally have the semicolon after their condition list.
        if len(row) == 6 and row[-1] == '':
            row.pop()
        if len(row) != 5:
            raise ValueError('unexpected SpecialCasing record shape')
        point = scalar(int(row[0], 16))
        lowered = tuple(scalar(int(value, 16)) for value in row[1].split())
        conditions = row[4].split()
        if conditions:
            if conditions == ['Final_Sigma']:
                final_sigma.append((point, lowered))
            elif conditions[0] not in ('lt', 'tr', 'az'):
                raise ValueError(
                    f'unimplemented root casing condition: {conditions}'
                )
            continue
        if not 1 <= len(lowered) <= 3 or point in full:
            raise ValueError('invalid or duplicate full lowercase mapping')
        if lowered != (simple.get(point, point),):
            full[point] = lowered
    # The core implements this exact root context rule. Fail closed during
    # authoring if future pinned data requires another rule or mapping.
    if final_sigma != [(0x03A3, (0x03C2,))]:
        raise ValueError('root Final_Sigma data changed')
    return tuple(sorted(full.items()))


def property_ranges(data: bytes, name: str) -> tuple:
    ranges = []
    for row in records(data):
        if len(row) < 2 or row[1] != name:
            continue
        limits = row[0].split('..')
        start = scalar(int(limits[0], 16))
        end = scalar(int(limits[-1], 16))
        if len(limits) > 2 or start > end:
            raise ValueError('invalid Unicode property range')
        ranges.append((start, end))
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            raise ValueError('overlapping Unicode property ranges')
        if merged and start == merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    if not merged:
        raise ValueError(f'missing Unicode property: {name}')
    return tuple(merged)


def character(point: int) -> str:
    return f"'\\u{{{scalar(point):x}}}'"


def string(points: tuple[int, ...]) -> str:
    return '"' + ''.join(f'\\u{{{scalar(point):x}}}' for point in points) + '"'


def render(profiles: dict, spaces: tuple[int, ...]) -> bytes:
    body = [
        '// SPDX-FileCopyrightText: Copyright (c) 1991-2026 Unicode, Inc.',
        '// SPDX-License-Identifier: Unicode-3.0',
        '// Generated by scripts/rust_core/generate_text_unicode.py. Do not edit.',
        '// Inputs and hashes: ../../../data/CASING-PROVENANCE.json.',
        '',
        'use super::{Tables, UnicodeDataVersion};',
        '',
    ]
    # Keep one complete simple map. Historical profiles need only exact
    # differences, including explicit identity overrides for newer letters.
    baseline = profiles['17.0.0']['simple']
    base = dict(baseline)
    body.extend(['#[rustfmt::skip]', 'const SIMPLE: &[(char, char)] = &['])
    body.extend(
        f'    ({character(key)}, {character(value)}),'
        for key, value in baseline
    )
    body.extend(['];', ''])
    pools: dict[str, dict[tuple, str]] = {
        name: {} for name in ('overrides', 'full', 'cased', 'ignorable')
    }
    references = {}
    for version, tables in profiles.items():
        references[version] = {}
        mapping = dict(tables['simple'])
        overrides = tuple(
            (point, mapping.get(point, point))
            for point in sorted(base.keys() | mapping.keys())
            if base.get(point, point) != mapping.get(point, point)
        )
        emitted = {
            'overrides': overrides,
            **{key: value for key, value in tables.items() if key != 'simple'},
        }
        for kind, entries in emitted.items():
            pool = pools[kind]
            if entries not in pool:
                name = f'{kind.upper()}_{len(pool)}'
                pool[entries] = name
                value_type = '&str' if kind == 'full' else 'char'
                body.extend(
                    [
                        '#[rustfmt::skip]',
                        f'const {name}: &[(char, {value_type})] = &[',
                    ]
                )
                for key, value in entries:
                    rendered = (
                        string(value) if kind == 'full' else character(value)
                    )
                    body.append(f'    ({character(key)}, {rendered}),')
                body.extend(['];', ''])
            references[version][kind] = pool[entries]
    identities = identity_profiles(profiles)
    body.extend(
        [
            f'pub(super) const IDENTITY_INPUT: &str = {string(IDENTITY_INPUT)};',
            '',
            '#[rustfmt::skip]',
            'pub(super) const IDENTITIES: &[(&str, UnicodeDataVersion)] = &[',
        ]
    )
    for signature, version in identities.items():
        major, minor, _patch = version.split('.')
        body.append(
            f'    ({string(signature)}, UnicodeDataVersion::V{major}_{minor}),'
        )
    body.extend(
        ['];', '', '#[rustfmt::skip]', 'pub(super) const SPACE: &[char] = &[']
    )
    body.extend(f'    {character(point)},' for point in spaces)
    body.extend(
        [
            '];',
            '',
            'pub(super) fn tables(version: UnicodeDataVersion) -> Tables {',
            '    match version {',
        ]
    )
    for version, tables in references.items():
        major, minor, _patch = version.split('.')
        body.append(
            f'        UnicodeDataVersion::V{major}_{minor} => Tables {{'
        )
        body.append('            simple: SIMPLE,')
        body.extend(
            f'            {kind}: {name},' for kind, name in tables.items()
        )
        body.append('        },')
    body.extend(['    }', '}', ''])
    return '\n'.join(body).encode('utf-8')


def identity_profiles(profiles: dict) -> dict[tuple, str]:
    identities = {}
    tables = {}
    for version, profile in profiles.items():
        mapping = dict(profile['simple'])
        signature = tuple(mapping.get(point, point) for point in IDENTITY_INPUT)
        prior = identities.get(signature)
        if prior is not None:
            if profiles[prior]['simple'] != profile['simple']:
                raise ValueError(
                    'native identity collides across distinct simple tables'
                )
        else:
            identities[signature] = version
        prior_signature = tables.setdefault(profile['simple'], signature)
        if prior_signature != signature:
            raise ValueError(
                'equal simple tables have unequal identity signatures'
            )
    return identities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--write', action='store_true', required=True)
    args = parser.parse_args()
    inputs = []
    profiles = {}
    common_spaces = None
    for version, hashes in INPUTS.items():
        data = []
        for name, digest in zip(FILES, hashes, strict=True):
            data.append(
                unicode_data.pinned(
                    args.inputs / f'{name}-{version}.txt', digest
                )
            )
            inputs.append(
                {
                    'url': f'https://www.unicode.org/Public/{version}/ucd/{name}.txt',
                    'sha256': digest,
                }
            )
        simple, spaces = mappings(data[0])
        if common_spaces is not None and common_spaces != spaces:
            raise ValueError(
                'native whitespace requires distinct version profiles'
            )
        common_spaces = spaces
        profiles[version] = {
            'simple': tuple(sorted(simple.items())),
            'full': full_mappings(data[1], simple),
            'cased': property_ranges(data[2], 'Cased'),
            'ignorable': property_ranges(data[2], 'Case_Ignorable'),
        }
    if common_spaces is None:
        raise ValueError('no Unicode profiles declared')
    output = render(profiles, common_spaces)
    license_data = unicode_data.pinned(
        args.inputs / 'LICENSE-UNICODE.txt', unicode_data.LICENSE_SHA256
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    LICENSE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(output)
    LICENSE.write_bytes(license_data)
    PROVENANCE.write_text(
        json.dumps(
            {
                'generator': 'scripts/rust_core/generate_text_unicode.py',
                'generator_sha256': unicode_data.sha256(Path(__file__)),
                'helpers': {
                    'scripts/rust_core/unicode_data.py': unicode_data.sha256(
                        Path(unicode_data.__file__)
                    ),
                },
                'profiles': list(INPUTS),
                'inputs': inputs,
                'license': {
                    'url': unicode_data.LICENSE_URL,
                    'sha256': unicode_data.LICENSE_SHA256,
                    'spdx': 'Unicode-3.0',
                },
                'outputs': {
                    OUTPUT.relative_to(ROOT).as_posix(): hashlib.sha256(
                        output
                    ).hexdigest(),
                    LICENSE.relative_to(
                        ROOT
                    ).as_posix(): unicode_data.LICENSE_SHA256,
                },
            },
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
