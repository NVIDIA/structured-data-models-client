# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1g's payload rule over several calls: where predict made more than
one call (a forecast's steps, ``main``'s multi-call ``predict``), a core
runner records every call's body in order (``core.predicted.calls``,
beside ``body``, the last's), and the rule holds them call by call against
the requests the endpoint received (python-core) or the reference sent
(rust-host). A request of one call records ``body`` alone and is judged as
before.

Controls: one call held as before, equal and seeded to differ; several
calls equal; a call that differs, a call missing, a call too many and the
calls out of order each fail, by the path of the call that differs.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _module(path: Path, name: str) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


COMPARE = _module(ROOT / 'scripts/rust_core/compare.py', 't21g_compare')
CASE = {'id': 'calls.probe', 'until': 'predict'}


def _call(n: int) -> dict[str, Any]:
    r"""A call's body: its rows to score, and its own request id, which
    the rule compares by presence and type only (decision E35)."""
    return {
        'model': 'kumo-relational',
        'predict': {'instance_table': {'rows': [[n], [n + 1]]}},
        'metadata': {'request_id': f'sdm-{n}'},
    }


def _failures(
    who: str, bodies: list[dict[str, Any]], sent: list[dict[str, Any]]
) -> list[str]:
    r"""The payload rule's failures for a runner whose core wrote
    ``bodies`` where the requests ``sent`` were received (python-core) or
    sent by the reference (rust-host)."""
    predicted: dict[str, Any] = {'body': bodies[-1]}
    if len(bodies) > 1:
        predicted['calls'] = bodies
    observation = {'core': {'predicted': predicted}}
    reference = None
    payloads = [{'parsed': copy.deepcopy(body)} for body in sent]
    if who == 'python-core':
        observation['public'] = {'payloads': payloads}
    else:
        reference = {'public': {'payloads': payloads}}
    report = COMPARE.Report({'payload': 'gate'})
    COMPARE._core_payload(report, CASE, who, observation, reference)
    return [text for _, text in report.failures]


def test_one_call_is_judged_as_before() -> None:
    for who in ('python-core', 'rust-host'):
        assert _failures(who, [_call(0)], [_call(0)]) == [], who
        found = _failures(who, [_call(0)], [_call(5)])
        assert len(found) == 1, (who, found)
        assert 'core.predicted.body/0/' in found[0], found


def test_several_calls_are_held_call_by_call() -> None:
    calls = [_call(0), _call(2), _call(4)]
    for who in ('python-core', 'rust-host'):
        assert _failures(who, calls, calls) == [], who
        # One call differs: named by its place among the calls.
        wrong = [_call(0), _call(2), _call(6)]
        [found] = _failures(who, wrong, calls)
        assert 'core.predicted.calls/2/' in found, found
        # A call missing, a call too many, the calls out of order.
        assert _failures(who, calls[:2], calls), who
        assert _failures(who, [*calls, _call(8)], calls), who
        assert _failures(who, [calls[1], calls[0], calls[2]], calls), who
        # Only the last call recorded, as before T2.1g: refused.
        assert _failures(who, [calls[-1]], calls), who
