#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check one candidate wheel per shipped distribution, then its installation.

This checks local install provenance, not signatures or native build identity;
the native artifact manifest checks remain responsible for those contracts.
Pass only the three wheels selected for this interpreter, not every platform
wheel in a release directory. This script uses only the standard library.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import re
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

DISTRIBUTIONS = {
    'structured-data-models-api': 'sdm_api',
    'structured-data-models-client': 'sdm_client',
    'structured-data-models-connectors': 'sdm_connectors',
}


def candidate_wheels(paths: list[Path]) -> tuple[dict[str, Path], str]:
    """Require exactly one wheel for each maintained distribution, at one version."""
    found: dict[str, Path] = {}
    versions = set()
    for path in paths:
        path = path.resolve(strict=True)
        parts = path.name.removesuffix('.whl').split('-')
        if (
            not path.is_file()
            or path.suffix != '.whl'
            or len(parts) not in (5, 6)
        ):
            raise ValueError(f'not a wheel filename: {path}')
        name = re.sub(r'[-_.]+', '-', parts[0]).lower()
        if name not in DISTRIBUTIONS:
            raise ValueError(f'unexpected distribution: {name}')
        if name in found:
            raise ValueError(f'duplicate distribution: {name}')
        found[name] = path
        versions.add(parts[1])
    missing = DISTRIBUTIONS.keys() - found.keys()
    if missing:
        raise ValueError(f'missing distributions: {sorted(missing)}')
    if len(versions) != 1:
        raise ValueError(
            f'distributions disagree on version: {sorted(versions)}'
        )
    return found, versions.pop()


def installed_wheels(wheels: dict[str, Path], version: str) -> None:
    """Check exact local wheel URLs and that imports resolve to those installs."""
    for name, wheel in wheels.items():
        distribution = importlib.metadata.distribution(name)
        if distribution.version != version:
            raise ValueError(f'{name}: installed version is not {version}')
        record = json.loads(distribution.read_text('direct_url.json') or '{}')
        parsed = urlparse(record.get('url', ''))
        if (
            parsed.scheme != 'file'
            or parsed.netloc.lower() not in ('', 'localhost')
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f'{name}: not installed from a local wheel file')
        source = Path(url2pathname(parsed.path))
        if not source.is_absolute() or source.resolve() != wheel:
            raise ValueError(
                f'{name}: installed from {source}, expected {wheel}'
            )
        module = DISTRIBUTIONS[name]
        spec = importlib.util.find_spec(module)
        expected = Path(
            str(distribution.locate_file(f'{module}/__init__.py'))
        ).resolve()
        if (
            spec is None
            or spec.origin is None
            or not expected.is_file()
            or Path(spec.origin).resolve() != expected
        ):
            raise ValueError(f'{module}: import does not resolve to {expected}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installed', action='store_true')
    parser.add_argument('--write-version', type=Path)
    parser.add_argument('wheels', nargs='+', type=Path)
    args = parser.parse_args()
    try:
        wheels, version = candidate_wheels(args.wheels)
        if args.installed:
            installed_wheels(wheels, version)
        if args.write_version:
            args.write_version.write_text(version, encoding='utf-8')
    except (
        ValueError,
        OSError,
        importlib.metadata.PackageNotFoundError,
    ) as exc:
        parser.exit(1, f'wheel provenance: {exc}\n')
    print(f'checked {len(wheels)} candidate wheels at {version}')


if __name__ == '__main__':
    main()
