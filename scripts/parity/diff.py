#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Name the first place two transcripts, or two payload sets, differ.

    diff.py A B
    diff.py --self-test

``A`` and ``B`` are two transcripts, whose referenced Arrow files are
compared too, or two payload sets (JSON arrays of payloads). Identical inputs
print nothing and exit 0. Different inputs exit 1, printing the path of the
first difference, a space, and what changed, e.g.::

    [1].context.related_tables.orders.rows[7].price 10.0 became 10.5

Unreadable input exits 2. ``--self-test`` seeds single changes into a payload
set and a transcript of its own making and checks each is found in place.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _parity.diffing import BadInput, compare_files, render


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('a', nargs='?', type=Path)
    parser.add_argument('b', nargs='?', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args(argv)
    if args.self_test:
        from _parity.selftest import run_self_test

        return run_self_test()
    if args.a is None or args.b is None:
        parser.error('give two files to compare, or --self-test')
    try:
        found = compare_files(args.a, args.b)
    except BadInput as error:
        print(f'error: {error}', file=sys.stderr)
        return 2
    if found is None:
        return 0
    path, what = found
    print(f'{render(path)} {" ".join(what.split())}')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
