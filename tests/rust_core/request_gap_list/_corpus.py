# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The recorded cases, run through today's Python under branch coverage
(``CONTRACT.md``, "Recorded cases").

    python _corpus.py --out PATH

Measures, over the modules ``SDM_P87_MODULES`` names (a JSON list of paths
from the repository root):

- ``held``: the branches every stored transcript (``_parity.corpus.check``)
  and every request fixture case that
  ``scripts/parity/request_fixture_exclusions.json`` does not exclude take,
  run through the Python fixture runner (``tests/parity/_fixture_ops.py``);
- ``all``: those and the branches every other fixture case under
  ``tests/parity/fixtures`` takes, whatever its operation, excluded or not.

It writes them as JSON, by module, with the transcripts and cases run and
the transcripts that did not replay.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

# ``scripts/parity/coverage.py`` would shadow the package: import it first.
import coverage

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PARITY = REPO / 'tests' / 'parity'
for _path in (HERE, REPO / 'scripts' / 'parity'):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

from _gaplist import PACKAGES, REQUEST_OPERATIONS, exclusions  # noqa: E402


def _coverage() -> coverage.Coverage:
    return coverage.Coverage(
        branch=True,
        data_file=None,
        source_pkgs=list(PACKAGES),
        omit=['*/predictive_query/grammar/PQLGrammar*'],
    )


def _executed(cov: coverage.Coverage, modules: list[str]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix='sdm-p87-corpus-') as tmp:
        out = Path(tmp) / 'report.json'
        try:
            cov.json_report(
                outfile=str(out),
                include=[str(REPO / module) for module in modules],
            )
        except coverage.exceptions.NoDataError:
            return {}
        report = json.loads(out.read_text('utf-8'))
    found: dict[str, Any] = {}
    for path, body in report.get('files', {}).items():
        absolute = Path(path)
        if not absolute.is_absolute():
            absolute = REPO / absolute
        module = absolute.resolve().relative_to(REPO.resolve()).as_posix()
        found[module] = sorted(body.get('executed_branches', []) or [])
    return found


def _run(operation: str, given: Any) -> None:
    import _fixture_ops

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            _fixture_ops.run(operation, given)
        except Exception:  # noqa: BLE001, S110 - outcomes differ
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    modules = json.loads(os.environ['SDM_P87_MODULES'])

    cases: list[tuple[str, str, Any]] = []
    for path in sorted((PARITY / 'fixtures').rglob('*.json')):
        for case in json.loads(path.read_text('utf-8'))['cases']:
            cases.append(
                (str(case['id']), str(case.get('operation', '')), case['input'])
            )
    excluded = set(exclusions())

    def held(case_id: str, operation: str) -> bool:
        return operation.startswith(REQUEST_OPERATIONS) and (
            case_id not in excluded
        )

    if str(PARITY) not in sys.path:
        sys.path.insert(0, str(PARITY))

    first = _coverage()
    first.start()
    try:
        from _parity.corpus import check, stored

        failed: dict[str, str] = {}
        transcripts = stored()
        for name, path in sorted(transcripts.items()):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                reason = check(path)
            if reason is not None:
                failed[name] = reason
        held_cases = []
        for case_id, operation, given in cases:
            if held(case_id, operation):
                _run(operation, given)
                held_cases.append(case_id)
    finally:
        first.stop()
    held_found = _executed(first, modules)

    second = _coverage()
    second.start()
    try:
        other_cases = []
        for case_id, operation, given in cases:
            if not held(case_id, operation):
                _run(operation, given)
                other_cases.append(case_id)
    finally:
        second.stop()
    other_found = _executed(second, modules)

    every = {
        module: sorted(
            {tuple(b) for b in held_found.get(module, [])}
            | {tuple(b) for b in other_found.get(module, [])}
        )
        for module in set(held_found) | set(other_found)
    }
    Path(args.out).write_text(
        json.dumps(
            {
                'held': held_found,
                'all': every,
                'transcripts': sorted(transcripts),
                'failed_transcripts': failed,
                'held_cases': held_cases,
                'other_cases': other_cases,
            }
        ),
        'utf-8',
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
