# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Append exact immutable frozen decomposed output-owner observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PRODUCER = 'd15be4baac34a30d67b2e4e18a47c6695ec8999d4c34eab4fdcb45979adc4f1e'
REFERENCE = '43e55f4f58924384c3cb52f67c8ed2e73ba45fa1'
RECEIPTS = {
    '310': '6ab01cd3cebc44646bbdfae0b7b7237b0111a9c9e52938bf4c7e978f9714414b',
    '312': '1c23268722da84c336e019aec3912b9509a65da7f9db6bedd735e6c6a0e49135',
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packets', type=Path)
    args = parser.parse_args()
    target = (
        Path(__file__).resolve().parents[2]
        / 'packages/structured-data-models-client/tests/t44_mutable_prediction_reference.json'
    )
    result = json.loads(target.read_text())
    assert result['reference'] == REFERENCE
    cases = {}
    provenance = {}
    for cell, digest in RECEIPTS.items():
        raw = (args.packets / f'result-r1-{cell}.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        receipt = json.loads(raw)
        assert receipt['reference'] == REFERENCE and receipt['cell'] == cell
        assert (
            receipt['producer_sha256'] == PRODUCER
            and not receipt['network_attempts']
        )
        assert len(receipt['records']) == 8
        cases[cell] = {record['case']: record for record in receipt['records']}
        provenance[cell] = {
            'producer_sha256': PRODUCER,
            'receipt_sha256': digest,
        }
    result['decomposed_output_cases'] = cases
    result['decomposed_output_provenance'] = provenance
    target.write_text(json.dumps(result, ensure_ascii=True, indent=2) + '\n')


if __name__ == '__main__':
    main()
