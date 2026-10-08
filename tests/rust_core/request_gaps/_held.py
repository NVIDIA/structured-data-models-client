# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The held recorded cases, run through today's Python under branch
coverage (``CONTRACT.md``, "Held cases").

    python _held.py --out PATH

Replays every stored transcript (``_parity.corpus.check``) and runs every
request fixture case that ``scripts/parity/request_fixture_exclusions.json``
does not exclude through the Python fixture runner
(``tests/parity/_fixture_ops.py``), then writes, as JSON: coverage's JSON
report of the gap entries' modules (``coverage``), the transcripts and cases
run, and the transcripts that did not replay.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

# ``scripts/parity/coverage.py`` would shadow the package: import it first.
import coverage

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parents[2] / 'scripts' / 'parity'):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

from _reqgaps import (  # noqa: E402
    PACKAGES,
    PARITY,
    REPO,
    REQUEST_OPERATIONS,
    exclusions,
    modules,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    cov = coverage.Coverage(
        branch=True,
        data_file=None,
        source_pkgs=list(PACKAGES),
        omit=['*/predictive_query/grammar/PQLGrammar*'],
    )
    cov.start()
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

        if str(PARITY) not in sys.path:
            sys.path.insert(0, str(PARITY))
        import _fixture_ops

        excluded = set(exclusions())
        ran: list[str] = []
        for path in sorted((PARITY / 'fixtures').glob('*.json')):
            for case in json.loads(path.read_text('utf-8'))['cases']:
                operation = str(case.get('operation', ''))
                if not operation.startswith(REQUEST_OPERATIONS):
                    continue
                if case.get('id') in excluded:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    try:
                        _fixture_ops.run(operation, case['input'])
                    except Exception:  # noqa: BLE001, S110 - outcomes differ
                        pass
                ran.append(case['id'])
    finally:
        cov.stop()

    with_report = Path(args.out).with_suffix('.coverage.json')
    cov.json_report(
        outfile=str(with_report),
        include=[str(REPO / module) for module in modules()],
    )
    report = json.loads(with_report.read_text('utf-8'))
    Path(args.out).write_text(
        json.dumps(
            {
                'coverage': report,
                'transcripts': sorted(transcripts),
                'failed_transcripts': failed,
                'fixture_cases': ran,
            }
        ),
        'utf-8',
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
