# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P76.derived: the parity coverage report is what ``coverage.py`` writes
over today's corpus, and it shows the link-prediction branches and the
multilabel refusal taken (``CONTRACT.md``, "Derived files").

Phase 6's ``accept_capture.py`` and ``accept_calls.py``, run beside this
file, hold the dialect captures to ``capture_dialects.py`` and the core.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from functools import cache
from pathlib import Path
from typing import Any

from _link import (
    COVERAGE_JSON,
    LINK_BRANCHES,
    MULTILABEL_BRANCH,
    PARITY_SCRIPTS,
    REPO,
    RFM,
    check_rfm_lines,
)


@cache
def _fresh() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix='sdm-p76-coverage-') as tmp:
        out = Path(tmp) / 'coverage.json'
        proc = subprocess.run(
            [
                sys.executable,
                str(PARITY_SCRIPTS / 'coverage.py'),
                '--out',
                str(out),
            ],
            cwd=REPO,
            env=dict(os.environ),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        assert proc.returncode == 0, (
            f'coverage.py failed ({proc.returncode}):\n'
            f'{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}'
        )
        return json.loads(out.read_text('utf-8'))


def _modules(report: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for crate, entry in report['crates'].items():
        for module, body in entry['modules'].items():
            key = _historical_path(module)
            assert key not in out, 'duplicate projected coverage module'
            out[key] = {'crate': crate, **body}
    return out


def test_the_report_is_what_the_script_writes() -> None:
    stored = json.loads(COVERAGE_JSON.read_text('utf-8'))
    fresh = _fresh()
    assert stored.get('ran') == fresh.get('ran'), (
        stored.get('ran'),
        fresh.get('ran'),
    )
    old, new = _modules(stored), _modules(fresh)
    assert set(old) == set(new), sorted(set(old) ^ set(new))[:20]
    stale = [module for module in new if old[module] != new[module]]
    assert not stale, (
        f'tests/parity/coverage.json is stale for {len(stale)} modules, '
        f'such as {stale[:5]}: regenerate it with scripts/parity/coverage.py'
    )
    assert stored == fresh


def test_the_report_shows_link_prediction_taken() -> None:
    check_rfm_lines()
    stored = _modules(json.loads(COVERAGE_JSON.read_text('utf-8')))
    assert RFM in stored, f'{RFM} is not in the coverage report'
    missing = {
        tuple(branch) for branch in stored[RFM].get('missing_branches', [])
    }
    untaken = sorted(missing & {*LINK_BRANCHES, MULTILABEL_BRANCH})
    assert not untaken, (
        f'the report has rfm.py link branches untaken: {untaken}'
    )


def _historical_path(path: str) -> str:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'switch'))
    from _switch import PACKAGE_RENAMES

    for old, current in PACKAGE_RENAMES.items():
        prefix = f'packages/{current}/'
        if path.startswith(prefix):
            return f'packages/{old}/' + path[len(prefix) :]
    return path
