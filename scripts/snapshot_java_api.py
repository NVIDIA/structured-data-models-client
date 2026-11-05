#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read public Java API identities from compiled classes using the JDK's javap.

This inventories additions as well as removals; japicmp separately checks
binary/source compatibility against the target branch's compiled JAR.
No native library is loaded and no model endpoint is needed.
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def collect(classes, javap='javap'):
    classes = Path(classes)
    package = classes / 'com/nvidia/sdm'
    files = sorted(package.glob('*.class'))
    if not files:
        raise ValueError(f'no compiled Java API classes under {package}')
    result = subprocess.run(
        [
            str(javap),
            '-protected',
            '-constants',
            *(str(path) for path in files),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse(result.stdout)


def parse(output):
    symbols = set()
    owner = None
    for line in output.splitlines():
        if line.startswith('Compiled from') or line == '}':
            owner = None
        if line.startswith('public '):
            match = re.search(
                r'\b(?:class|interface) (com\.nvidia\.sdm\.[\w$]+)', line
            )
            if not match:
                raise ValueError(f'unrecognized Java declaration: {line}')
            owner = match[1]
            symbols.add(owner)
        elif owner and line.startswith(('  public ', '  protected ')):
            symbols.add(f'{owner}#{line.strip()}')
    if not symbols:
        raise ValueError('javap returned no public API')
    return sorted(symbols)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--classes', type=Path, default=Path('java/target/classes')
    )
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    javap = (
        Path(os.environ['JAVA_HOME']) / 'bin/javap'
        if 'JAVA_HOME' in os.environ
        else 'javap'
    )
    content = json.dumps(collect(args.classes, javap), indent=2) + '\n'
    if args.out:
        args.out.write_text(content)
    else:
        print(content, end='')


if __name__ == '__main__':
    main()
