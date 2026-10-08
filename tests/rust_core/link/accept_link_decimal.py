# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P76.decimal: completed link-prediction scenarios whose data stores the
listed foreign key and the primary key it links to as ``DECIMAL`` or
``NUMERIC``, on DuckDB and PostgreSQL at least, replay; on one at least
today's encoder writes labels from the decimals' own digits; and the core
samples them, and gives the payloads, exactly as today's Python does
(``CONTRACT.md``, "Decimal keys").
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Any

from _link import (
    Boundary,
    compare_runs,
    context_got,
    context_want,
    drive_recordings,
    is_refused,
    link_transcripts,
    query_of,
    read,
    recording_dir_of,
    recording_problems,
    replay_with,
    target_column,
    task_table_got,
    task_table_want,
)

#: The backends that must each have one.
BACKENDS = ('duckdb', 'postgres')


_DECIMAL_TYPE = re.compile(r'^\s*(DECIMAL|NUMERIC)\b', re.IGNORECASE)


def _target_key(path: Path, transcript: dict[str, Any]) -> str | None:
    r"""``table.column``: the primary key of the payloads' target table, as
    the boundary recording's ``sample_subgraph`` calls give it."""
    payloads = transcript.get('payloads') or []
    names = (payloads[0].get('task') or {}).get('entity_table_names') or []
    if len(names) != 2:
        return None
    recording = recording_dir_of(path) / 'boundary.json'
    if not recording.is_file():
        return None
    for call in json.loads(recording.read_text('utf-8'))['calls']:
        if call['method'] != 'sample_subgraph':
            continue
        for table in call['outputs'].get('tables', []):
            if table.get('name') == names[1] and table.get('primary_key'):
                return f'{names[1]}.{table["primary_key"]}'
    return None


def _decimal_keyed(path: Path, transcript: dict[str, Any]) -> bool:
    r"""Whether the scenario's data stores both the foreign key the query
    lists and the primary key it links to as ``DECIMAL`` or ``NUMERIC``
    (``data.column_types``)."""
    if not transcript.get('payloads'):
        return False
    query = query_of(transcript)
    assert query is not None
    listed = '.'.join(target_column(query))
    key = _target_key(path, transcript)
    types = transcript.get('data', {}).get('column_types') or {}
    return key is not None and all(
        _DECIMAL_TYPE.match(str(types.get(column, '')))
        for column in (listed, key)
    )


@cache
def found() -> dict[str, Path]:
    r"""The completed link-prediction transcripts keyed by decimals."""
    out = {}
    for name, path in link_transcripts().items():
        transcript = read(path)
        if not is_refused(transcript) and _decimal_keyed(path, transcript):
            out[name] = path
    backends = {read(path)['backend'] for path in out.values()}
    missing = [b for b in BACKENDS if b not in backends]
    assert not missing, (
        'no completed link-prediction transcript with DECIMAL or NUMERIC keys '
        f'on {missing}; found {sorted(out)}'
    )
    return out


def test_there_is_one_on_duckdb_and_postgres() -> None:
    assert found()


def test_they_replay_and_python_writes_the_decimals_digits() -> None:
    r"""Each replays, and on one at least today's encoder writes a label
    item from a decimal's own digits (``payload.py``'s
    ``_target_json_value``, the ``_decimal_text`` item of a link-prediction
    label): the branch that shows such a key reaches the request. Not every
    backend reads a ``NUMERIC`` key as a decimal; PostgreSQL's sampler reads
    it as a number."""
    import sys

    from _parity.corpus import check
    from sdm_relational_engine.rfm.inference import payload

    original = payload._decimal_text
    seen: dict[str, int] = {}
    current: list[str] = []

    def spy(value: Any) -> str | None:
        text = original(value)
        if (
            text is not None
            and current
            and sys._getframe(1).f_code.co_name == '_target_json_value'
        ):
            seen[current[0]] = seen.get(current[0], 0) + 1
        return text

    problems = []
    payload._decimal_text = spy
    try:
        for name, path in found().items():
            current[:] = [name]
            reason = check(path)
            if reason is not None:
                problems.append(f'{name}: {reason}')
    finally:
        current.clear()
        payload._decimal_text = original
    assert not problems, '\n'.join(problems[:10])
    assert seen, (
        'no label item of these was a decimal, so none reaches the '
        f'decimal branch of _target_json_value: {sorted(found())}'
    )


def test_the_core_matches_the_recordings() -> None:
    recordings, got = drive_recordings(found())
    problems, compared = recording_problems(recordings, got)
    assert not problems, (
        f"{len(problems)} decimal-key calls differ from today's Python:\n"
        + '\n'.join(problems[:25])
    )
    assert compared >= 2 * len(recordings), f'only {compared} calls compared'


def test_task_tables_and_contexts() -> None:
    from sdm_relational_engine.rfm.rfm import KumoRelational

    for method, run, want, got in (
        ('_get_task_table', 'task_table', task_table_want, task_table_got),
        ('_get_context', 'context', context_want, context_got),
    ):
        boundary = Boundary(KumoRelational, method, lambda m: m._sampler)
        reasons = replay_with(boundary, method, list(found().values()))
        assert not reasons, reasons[:3]
        calls, problems = compare_runs(run, boundary, want, got)
        assert not problems, '\n'.join(problems[:20])
        for name, path in found().items():
            mine = [call for call in calls if call['transcript'] == path]
            assert mine, f'{name}: no {method} call'
            assert all(call['core'] == 'done' for call in mine), (
                f"{name}: the core's {run} run did not finish: "
                f'{[call["core_error"] for call in mine]}'
            )


def test_payloads_from_the_core_outputs() -> None:
    r"""As Phase 7.5's ``accept_driver_payload.py``, over these: with every
    sampling call answered by the core, the replay gives the transcript's
    payloads."""
    from _driver import (
        METHODS,
        Interceptor,
        capture_module,
        native,
        replaying,
    )

    core = native(*METHODS, 'Outcome')
    capture = capture_module()
    interceptor = Interceptor(capture.describe, core, capture.to_python)
    problems = []
    with replaying(interceptor) as replay:
        for name, path in found().items():
            reason, calls = replay(path)
            if reason is not None:
                problems.append(f'{name}: {reason}')
            elif not calls:
                problems.append(f'{name}: no call went to the core')
    assert not problems, (
        f'{len(problems)} decimal-key scenarios do not reproduce over the '
        "core's sampling outputs:\n" + '\n'.join(problems[:25])
    )
