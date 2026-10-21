# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.16a fix 1 (Codex round 1, P3): ``contract/tools/l7.py report`` judges
CPU only over scenarios every repetition of which completed on both sides.
A core that refused before the stage its case runs to did less work: the
scenario is listed as incomplete and left out of the verdict, never counted
as faster.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
L7 = ROOT / 'contract/tools/l7.py'
CLIENT = ROOT / 'packages/structured-data-models-client/src/sdm_client'


def _scenario(total: float, complete: bool, core: bool) -> dict:
    found = {
        'outcomes': (
            ['success at predict'] if complete else ['refusal at materialize']
        ),
        'complete': complete,
        'cpu': {'total': total},
    }
    if core:
        found['boundary'] = {'messages': 10, 'largest': 512, 'codec_cpu': 0.0}
    return found


def _report(tmp_path: Path, scenarios: dict) -> tuple[int, str]:
    common = {
        'format': 'sdm-l7/1',
        'cell': 'ref-pandas3',
        'database': 'duckdb',
        'repeats': 5,
        'import_cpu': 0.5,
    }
    ref = {
        **common,
        'mode': 'python-ref',
        'scenarios': {
            k: _scenario(r, rc, False) for k, (r, rc, _, _) in scenarios.items()
        },
    }
    core = {
        **common,
        'mode': 'python-core',
        'crossing_cpu': 1e-7,
        'scenarios': {
            k: _scenario(c, cc, True) for k, (_, _, c, cc) in scenarios.items()
        },
    }
    (tmp_path / 'ref.json').write_text(json.dumps(ref))
    (tmp_path / 'core.json').write_text(json.dumps(core))
    out = tmp_path / 'l7.md'
    done = subprocess.run(
        [
            sys.executable,
            '-I',
            str(L7),
            'report',
            '--ref',
            str(tmp_path / 'ref.json'),
            '--core',
            str(tmp_path / 'core.json'),
            '--client',
            str(CLIENT),
            '--out',
            str(out),
        ],  # fmt: skip
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode in (0, 1), done.stderr
    return done.returncode, out.read_text()


def test_a_scenario_the_core_stopped_early_is_not_judged(tmp_path):
    # The core refused at materialize in 10 ms where the reference
    # answered in 100 ms: alone, nothing is judged, and the limit fails.
    code, text = _report(tmp_path, {'demo': (0.1, True, 0.01, False)})
    assert code == 1
    assert 'incomplete on either side' in text and '- demo:' in text
    assert '- judged: 0' in text
    assert 'every measured limit holds' not in text
    # Beside a complete scenario, it is listed and left out of the ratios.
    code, text = _report(
        tmp_path,
        {
            'demo': (0.1, True, 0.01, False),
            'done': (0.1, True, 0.09, True),
        },
    )
    assert '- judged: 1' in text
    assert '| done |' in text and '| demo |' not in text
    assert 'median core-to-reference ratio: 0.900' in text
