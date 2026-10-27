# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Append actual frozen model/output-owner observations without SDK imports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PRODUCER = '4aed894a8a56af2e168093d03b118cd069d7400fce047ee9c4a034e53f86aa69'
REFERENCE = '43e55f4f58924384c3cb52f67c8ed2e73ba45fa1'
RECEIPTS = {
    '310': 'b1a02ca2617c8ee04ce60cc0559fc6b3fbbbdb68edeab74fbdbbba0e64aacb1f',
    '312': '08889d11040538c43886725141d0751628afc592aa7cdc6a8f70022fe0a1b38f',
}
SHAPE_PRODUCER = (
    'ea5b0da388e02ef7a298ede77b809c1428832cf079e304e8ff2b245459c4e91b'
)
SHAPE_RECEIPTS = {
    '310': '9b52b27af19d5230d335a061bb05f11d90b24ac54d166aa3cdaa65f14ddec3a3',
    '312': 'c68c5a3c51796997b8e24372cf28b2729e5dab7679c05af4b4795937ca9466b6',
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packets', type=Path)
    args = parser.parse_args()
    target = (
        Path(__file__).resolve().parents[2]
        / 'packages/structured-data-models-client/tests/t44_mutable_prediction_reference.json'
    )
    value = json.loads(target.read_text())
    assert value['reference'] == REFERENCE
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
        assert len(receipt['records']) == 13
        cases[cell] = {record['case']: record for record in receipt['records']}
        provenance[cell] = {
            'producer_sha256': PRODUCER,
            'receipt_sha256': digest,
        }
    value['model_output_cases'] = cases
    value['model_output_provenance'] = provenance
    shapes = {}
    for cell, digest in SHAPE_RECEIPTS.items():
        raw = (args.packets / f'result-r3-{cell}.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        receipt = json.loads(raw)
        assert receipt['reference'] == REFERENCE and receipt['cell'] == cell
        assert (
            receipt['producer_sha256'] == SHAPE_PRODUCER
            and not receipt['network_attempts']
        )
        assert len(receipt['records']) == 2
        shapes[cell] = {record['case']: record for record in receipt['records']}
        provenance[f'shape-r3-{cell}'] = {
            'producer_sha256': SHAPE_PRODUCER,
            'receipt_sha256': digest,
        }
    value['model_output_shape_cases'] = shapes
    target.write_text(json.dumps(value, ensure_ascii=True, indent=2) + '\n')


if __name__ == '__main__':
    main()
