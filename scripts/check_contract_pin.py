#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Prove the recorded contract digest is the contract's real bytes.

The digest tests inside the package compare the generated file against
``PROVENANCE.json``, which catches a hand-edited binding. They cannot catch a
*doctored* provenance record, because both sides of that comparison live in
this repository. This script closes that by digesting a contract fetched at
the recorded revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROVENANCE = (
    REPO_ROOT
    / 'packages'
    / 'sdm-api'
    / 'src'
    / 'sdm_api'
    / 'generated'
    / 'PROVENANCE.json'
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--contract',
        required=True,
        help='Path to nim-sd.openapi.yaml, checked out at the recorded revision.',
    )
    args = parser.parse_args()

    contract = Path(args.contract)
    if not contract.is_file():
        print(f'no contract at {contract}', file=sys.stderr)
        return 1

    record = json.loads(PROVENANCE.read_text(encoding='utf-8'))
    recorded = record['source_sha256']
    actual = hashlib.sha256(contract.read_bytes()).hexdigest()

    if actual != recorded:
        print(
            f'the contract at revision {record["source_revision"]} does not '
            f'have the digest this repository recorded for it.\n'
            f'  recorded: {recorded}\n'
            f'  actual:   {actual}\n'
            f'Either the provenance record is wrong, or the revision was '
            f'moved. Regenerate the bindings.',
            file=sys.stderr,
        )
        return 1

    generated = PROVENANCE.parent / record['generated_file']
    generated_actual = hashlib.sha256(generated.read_bytes()).hexdigest()
    if generated_actual != record['generated_sha256']:
        print(
            f'{record["generated_file"]} does not match its recorded digest; '
            f'regenerate rather than editing it.',
            file=sys.stderr,
        )
        return 1

    generator = REPO_ROOT / record['generator']
    generator_actual = hashlib.sha256(generator.read_bytes()).hexdigest()
    if generator_actual != record.get('generator_sha256'):
        print(
            f'{record["generator"]} has changed since these bindings were '
            f'generated. Regenerate them so the record matches the generator '
            f'that produced them.',
            file=sys.stderr,
        )
        return 1

    print(
        f'contract {record["source_revision"][:12]} matches its recorded '
        f'digest; bindings and generator match theirs'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
