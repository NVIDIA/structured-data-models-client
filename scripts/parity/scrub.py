#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Keep credentials and private hosts out of stored recordings.

    scrub.py scrub SRC DST
    scrub.py scan [PATH...]

``scrub`` copies the JSON and Arrow files under ``SRC`` to ``DST``, keeping
their relative paths, with every secret replaced by ``<redacted:WHAT>``.
``scan`` looks through the JSON and Arrow files under each path (by default
``tests/parity/transcripts``), lists every secret it finds, and exits 1 if
it finds any, 0 if none.

Besides what is recognizable by its shape, the values of these environment
variables are secret wherever they appear, when they are set:
``DATABRICKS_SERVER_HOSTNAME``, ``DATABRICKS_TOKEN``, ``DATABRICKS_HTTP_PATH``,
``DATABRICKS_WAREHOUSE_ID``, ``DATABRICKS_HOST``, ``DATABRICKS_CLIENT_SECRET``,
``SNOWFLAKE_ACCOUNT``, ``SNOWFLAKE_USER``, ``SNOWFLAKE_PASSWORD``,
``SNOWFLAKE_TOKEN``, ``SNOWFLAKE_URL``, ``SNOWFLAKE_USERNAME``,
``SNOWFLAKE_ACCESS_TOKEN``, ``PGHOST``, ``PGUSER`` and ``PGPASSWORD``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _parity import TRANSCRIPTS
from _parity.scrubber import scan, scrub_tree


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    commands = parser.add_subparsers(dest='command', required=True)
    scrub = commands.add_parser('scrub')
    scrub.add_argument('source', type=Path)
    scrub.add_argument('target', type=Path)
    scanned = commands.add_parser('scan')
    scanned.add_argument('paths', nargs='*', type=Path)
    args = parser.parse_args(argv)

    if args.command == 'scrub':
        if not args.source.is_dir():
            print(f'error: {args.source} is not a directory', file=sys.stderr)
            return 2
        findings = scrub_tree(args.source, args.target)
        for finding in findings:
            print(f'removed {finding.what} at {finding.where}')
        print(f'{len(findings)} secret(s) removed', file=sys.stderr)
        return 0

    roots = args.paths or [TRANSCRIPTS]
    missing = [root for root in roots if not root.exists()]
    if missing:
        print(f'error: no such path: {missing}', file=sys.stderr)
        return 2
    findings = scan(roots)
    for finding in findings:
        print(f'{finding.what} at {finding.where}')
    print(f'{len(findings)} secret(s) found', file=sys.stderr)
    return 1 if findings else 0


if __name__ == '__main__':
    raise SystemExit(main())
