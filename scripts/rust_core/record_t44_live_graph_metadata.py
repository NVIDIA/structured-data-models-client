# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy complete public observations from immutable frozen metadata captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

REFERENCE = '43e55f4f58924384c3cb52f67c8ed2e73ba45fa1'
PRODUCER = 'df810a0f326fc2e8d8f6ec395bea6838dd0cbb29f73eb7a83238989daf946ee9'
DRIVER = '81d87cb88cd7152c2b2e58e6a87af6516aa981a3832fb323b3a07c585d229b8d'
CAPTURES = {
    '310': 'b9d030f97c96aa4b3bd891d2b52257db4de68150c53a08c495d8a0f50bffaca0',
    '312': '30c2ef999fbc5e5d7262d221833882ea1696075b6ac8cf806914706304f76e5a',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packet', type=Path)
    args = parser.parse_args()
    packet = args.packet
    assert digest(packet / 'probe_r1.py') == PRODUCER
    assert digest(packet / 'run_r1.private.py') == DRIVER
    terminal = json.loads(
        (packet / 'execution-terminal-r1.private.json').read_text()
    )
    assert terminal['producer_sha256'] == PRODUCER
    assert terminal['driver_sha256'] == DRIVER
    assert terminal['reference'] == REFERENCE
    assert terminal['exit_code'] == 0
    assert [cell['cell'] for cell in terminal['cells']] == ['310', '312']
    assert all(cell['exit_code'] == 0 for cell in terminal['cells'])
    tests = (
        Path(__file__).resolve().parents[2]
        / 'packages/structured-data-models-client/tests'
    )
    for cell, runtime in [('310', 'pandas2'), ('312', 'pandas3')]:
        path = packet / f'live-graph-metadata-r1-{cell}.json'
        assert digest(path) == CAPTURES[cell]
        captured = json.loads(path.read_text())
        assert captured['reference'] == REFERENCE
        assert captured['producer_sha256'] == PRODUCER
        assert captured['cell'] == cell
        assert captured['network_attempts'] == []
        records = copy.deepcopy(captured['records'])
        assert len(records) == 4
        for record in records:
            assert len(record['calls']) == 2
            for call in record['calls']:
                outcome = call['outcome']
                if outcome['kind'] == 'error':
                    # Traceback locations differ across hosts. Preserve every
                    # public error/cause field without changing SDK output.
                    del outcome['error']['traceback']
                else:
                    assert outcome['kind'] == 'success'
                    assert outcome['inspection']['payloads']
                    assert len(outcome['sent']) == 1
            assert record['calls'][1]['trace'] == []
        floats = [r for r in records if r['declared'] == 'float64']
        assert [r['mutation'] for r in floats] == [False, True]
        assert all(
            c['outcome']['kind'] == 'success'
            for r in floats
            for c in r['calls']
        )
        assert (
            floats[0]['calls'][0]['outcome']['inspection']['payloads']
            != floats[1]['calls'][0]['outcome']['inspection']['payloads']
        )
        output = {
            'reference': REFERENCE,
            'cell': cell,
            'producer_sha256': PRODUCER,
            'driver_sha256': DRIVER,
            'capture_sha256': CAPTURES[cell],
            'records': records,
        }
        target = tests / f't44_live_graph_metadata_{runtime}_reference.json'
        with target.open('x') as stream:
            json.dump(
                output, stream, indent=2, ensure_ascii=True, allow_nan=False
            )
            stream.write('\n')


if __name__ == '__main__':
    main()
