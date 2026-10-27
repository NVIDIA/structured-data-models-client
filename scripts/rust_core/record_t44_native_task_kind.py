# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Append native task preparation witnesses from immutable frozen receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REFERENCE = '43e55f4f58924384c3cb52f67c8ed2e73ba45fa1'
PRODUCER = '9702ebc0c3ef0e3d6d8de30e4ccc233046d3a7774744529bbf3d735ee1bb500e'
RECEIPTS = {
    '310': '8012fa5ea1e12600baf3fa64956a81bafa2f1d176af2ab6b20d259c0475e6fdf',
    '312': 'e8b2edc417c96378179647bf092a8eaf18ec1aa32a463828867c707a52045224',
}
REFUSAL_PRODUCER = (
    '4adb947def4520f7f269cb941721fd48fe4aa30af3b5e9a1aa7ea51dd67f7b5e'
)
REFUSAL_RECEIPTS = {
    '310': 'c72387bb8c899b7538f4db03a81edc2d90ddb1412d29103c17bb6192e0aa2318',
    '312': 'd43c3d8579e76ae52d5a82668a08e3c54f2138fc7396eed3e3cd9644dcff0db1',
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packets', type=Path)
    args = parser.parse_args()
    target = Path(__file__).resolve().parents[2] / (
        'packages/structured-data-models-client/tests/'
        't44_mutable_prediction_reference.json'
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
        assert receipt['producer_sha256'] == PRODUCER
        assert receipt['network_attempts'] == []
        observed = {}
        for record in receipt['records']:
            record = dict(record)
            if 'error' in record:
                record['error'] = {
                    key: value
                    for key, value in record['error'].items()
                    if key != 'traceback'
                }
            if 'wire' in record:
                record['wire'] = json.loads(record['wire'])
            observed[record['case']] = record
        cases[cell] = observed
        provenance[cell] = {
            'receipt_sha256': digest,
            'producer_sha256': PRODUCER,
        }
    result['native_kind_provenance'] = provenance
    result['native_kind_cases'] = cases
    refused = {}
    for cell, digest in REFUSAL_RECEIPTS.items():
        raw = (args.packets / f'result-r2-{cell}.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        receipt = json.loads(raw)
        assert receipt['reference'] == REFERENCE and receipt['cell'] == cell
        assert receipt['producer_sha256'] == REFUSAL_PRODUCER
        assert receipt['network_attempts'] == []
        refused[cell] = {
            record['case']: record for record in receipt['records']
        }
        provenance[f'refusal-{cell}'] = {
            'receipt_sha256': digest,
            'producer_sha256': REFUSAL_PRODUCER,
        }
    result['native_kind_refusal_cases'] = refused
    target.write_text(json.dumps(result, ensure_ascii=True, indent=2) + '\n')


if __name__ == '__main__':
    main()
