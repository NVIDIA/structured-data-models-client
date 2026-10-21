# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""S1 coverage from ``coverage.py``'s own generator (T2.16's report; rule 7,
``--gate S1``, is not built). Run from the repository's root::

    python3 -I docs/design/rust-core/phase-2/evidence/coverage/s1_coverage.py REV

It lists the S0 and S1 paths and the cells whose ``expect`` is not
``unknown`` that no acceptance case covers.
"""

import sys
from pathlib import Path

sys.path.insert(0, 'scripts/rust_core')
import coverage as c  # noqa: E402


def main(rev: str) -> None:
    ratchet = c._module('ratchet')
    sha = ratchet.git(Path('.'), 'rev-parse', f'{rev}^{{commit}}').strip()
    tree = ratchet.Tree(Path('.'), sha)
    report = c.Report()
    validator = c.read_schema(tree, report)
    manifest = c.read_manifest(tree, 'python', validator, report)
    corpus = c.read_cases(tree, report)
    assert not report.problems, report.problems
    made = c.generated('python', manifest, corpus)
    print(
        f'# S1 coverage at {tree.sha}: python manifest, acceptance cases '
        f'only ({len(corpus)} cases read)'
    )
    paths = [p for p in manifest['paths'] if p['slice'] in ('S0', 'S1')]
    empty_paths = [p['id'] for p in paths if not made['paths'][str(p['id'])]]
    print(
        f'paths in S0 and S1: {len(paths)}; with an acceptance case: '
        f'{len(paths) - len(empty_paths)}; empty: {len(empty_paths)}'
    )
    for path in empty_paths:
        print(f'  empty path {path}')
    cells = manifest['cells']
    scoped = [x for x in cells if x.get('expect') != 'unknown']
    empty_cells = [x for x in scoped if not made['cells'][str(x['id'])]]
    print(
        f"cells: {len(cells)}; expect not 'unknown': {len(scoped)}; with an "
        f'acceptance case: {len(scoped) - len(empty_cells)}; empty: '
        f'{len(empty_cells)}'
    )
    for cell in empty_cells:
        print(f'  empty cell {cell["id"]} (expect {cell.get("expect")})')


if __name__ == '__main__':
    main(sys.argv[1])
