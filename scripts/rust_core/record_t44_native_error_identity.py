# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy immutable frozen native exception observations without SDK imports."""

import argparse
import hashlib
import json
from pathlib import Path

REFERENCE = '43e55f4f58924384c3cb52f67c8ed2e73ba45fa1'
PRODUCER = 'e3b40a39fc769e8ba2de4ebe0001d6f28dcada11e40f2b4b23a21fee4fbf4f48'
RECEIPTS = {
    '310': 'c477e4fc8bb91f7e88a9431b2ff1b8e651329eb67d1fcc0aedda516d880107b8',
    '312': '8a5fa60f1cc3bfa7562cbd19a295960e8f8627a749ae9db92c55264938a08991',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packet', type=Path)
    args = parser.parse_args()
    packet = args.packet
    assert (
        hashlib.sha256((packet / 'probe_r2.py').read_bytes()).hexdigest()
        == PRODUCER
    )
    terminal = json.loads(
        (packet / 'execution-r2-terminal.private.json').read_text()
    )
    assert terminal['exit_code'] == 0
    assert {row['cell'] for row in terminal['cells']} == set(RECEIPTS)
    assert all(row['exit_code'] == 0 for row in terminal['cells'])
    observed = []
    for cell, expected in RECEIPTS.items():
        path = packet / f'frozen-relational-error-r2-{cell}.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        value = json.loads(path.read_text())
        assert value['reference'] == REFERENCE
        assert value['producer_sha256'] == PRODUCER
        assert not value['network_attempts']
        observed.append(value['records'])
    assert observed[0] == observed[1]
    root = Path(__file__).resolve().parents[2]
    output = (
        root
        / 'packages/structured-data-models-client/tests/t44_native_error_identity_reference.json'
    )
    output.write_text(
        json.dumps(
            {
                'reference': REFERENCE,
                'producer_sha256': PRODUCER,
                'receipt_sha256': RECEIPTS,
                'records': {row['case']: row for row in observed[0]},
            },
            indent=2,
        )
        + '\n'
    )


if __name__ == '__main__':
    main()
