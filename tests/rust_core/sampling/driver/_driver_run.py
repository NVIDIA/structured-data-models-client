# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Drive the core's sampling runs from the boundary recordings alone, in a
process where today's Python can't be imported, for
``accept_driver_replay.py`` and ``accept_driver_hashseed.py``.

    python _driver_run.py RECORDINGS OUT

Before importing anything else it makes pandas, dateutil, pytz, torch,
``sdm_relational_engine``, ``sdm_client``, ``predictive_query`` and
``sdm_connectors`` unimportable, so the core answers by itself: numpy and
pyarrow are all it may lean on. For every call of every recording under
RECORDINGS it starts the run of the call's method from the recorded inputs
and tables, answers each query from the recorded statements (``drive``),
and writes to OUT, as JSON, by scenario, one entry per call:

- ``sql``: ``null``, or why the queries were not the recorded statements;
- ``outputs``: the run's ``done`` value in canonical form, or
  ``{"error": {"code", "message"}}`` where the run raised; ``null`` where
  the queries differed;
- ``queries``: how many queries the run asked;
- ``seconds``: how long the run took, start to ``done``.

It needs the wheel installed already (``SDM_P7_WHEEL_INSTALLED=1``).
"""

from __future__ import annotations

import sys

BLOCKED = (
    'pandas',
    'dateutil',
    'pytz',
    'torch',
    'sdm_relational_engine',
    'sdm_client',
    'predictive_query',
    'sdm_connectors',
)


class _Blocked:
    r"""Refuses to import :data:`BLOCKED`, as a missing module is refused.

    A ``None`` in ``sys.modules`` would block too, but pyarrow's pandas shim
    reads ``sys.modules`` and fails on it, where it takes a missing pandas.
    """

    def find_spec(self, name: str, path: object, target: object = None):
        if name.partition('.')[0] in BLOCKED:
            raise ModuleNotFoundError(f'No module named {name!r}', name=name)
        return None


def main() -> None:
    loaded = [m for m in sys.modules if m.partition('.')[0] in BLOCKED]
    assert not loaded, f'already imported: {loaded}'
    sys.meta_path.insert(0, _Blocked())  # type: ignore[arg-type]

    import json
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _driver import (
        METHODS,
        Mismatch,
        drive,
        error_of,
        native,
        read_arrow,
        read_recording,
        recorded_tables,
        recording_dirs,
        rust_outputs,
    )

    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    module = native(*METHODS, 'Outcome')
    found: dict[str, list[dict]] = {}
    for directory in recording_dirs(root):
        recording = read_recording(directory)
        entries = []
        for call in recording['calls']:
            steps = call['steps']

            def result_of(
                i: int, steps: list = steps, directory: Path = directory
            ) -> object:
                file = steps[i]['result']
                return None if file is None else read_arrow(directory / file)

            entry: dict = {'sql': None, 'outputs': None, 'queries': 0}
            started = time.perf_counter()
            asked: list = []
            try:
                run = getattr(module, call['method'])(
                    json.dumps(call['inputs'], ensure_ascii=False),
                    recorded_tables(directory, call),
                )
                value, asked = drive(module, run, steps, result_of)
                entry['outputs'] = rust_outputs(call['method'], value)
            except Mismatch as mismatch:
                entry['sql'] = str(mismatch)
            except AssertionError as bad:
                entry['outputs'] = {'unreadable': str(bad)}
            except Exception as raised:  # noqa: BLE001 - an outcome
                entry['outputs'] = {'error': error_of(raised)}
            entry['seconds'] = time.perf_counter() - started
            entry['queries'] = len(asked)
            entries.append(entry)
        found[recording['scenario']] = entries
    out.write_text(json.dumps(found, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
