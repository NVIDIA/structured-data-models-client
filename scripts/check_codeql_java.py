# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require CodeQL's source archive to contain every shipped Java source."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check(archive: Path, root: Path = ROOT) -> int:
    """Check actual extracted bytes, including optional connector adapters."""
    sources = sorted((root / 'java/src/main/java').rglob('*.java'))
    if not sources:
        raise ValueError('no shipped Java sources found')
    with zipfile.ZipFile(archive) as extracted:
        names = extracted.namelist()
        for source in sources:
            relative = source.relative_to(root).as_posix()
            matches = [
                name
                for name in names
                if name == relative or name.endswith('/' + relative)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f'{relative}: expected one extracted source, found {len(matches)}'
                )
            if extracted.read(matches[0]) != source.read_bytes():
                raise ValueError(
                    f'{relative}: extracted source differs from checkout'
                )
    return len(sources)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    count = check(args.archive)
    print(
        f'CodeQL extracted all {count} shipped Java sources from this checkout'
    )


if __name__ == '__main__':
    main()
