# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T3.3's acceptance tests for the stand-in provider (Phase 3 design,
choice 2, "The stand-in provider"; T3.3's brief, exit 2), through the
forwarding provider python-core plugs in, ``contract/runners/standin_client.py``,
and the ``sdm-standin`` binary it talks to over a pipe. Written by the
package's test author before the implementation, from captures the capture
writer writes (``t33_fixtures.py`` gives their content: the ``shop`` base's
tables, one draw, one statement with its result).

The interface tested, named here because nothing names it before the
implementation:

- ``StandIn(capture, *, binary)``: a context manager serving the capture at
  ``capture`` (one case on one leg, as ``captures.write`` wrote it) through
  the ``sdm-standin`` binary at ``binary``;
- ``StandIn.fetch(step) -> Answer``: one fetch step as the core sends it
  (``{"id", "request", "plan"}``, the plan the leg's own dialect rendered),
  answered with ``Answer.table`` (a ``pyarrow.Table``), or ``Answer.error``
  (a provider error), or ``Answer.live_only`` (a reason code, ``B1``,
  ``B2`` or ``B4``), and ``Answer.flagged`` (the codes it was flagged with);
- ``StandInFailure``: what the stand-in refuses by name, the table, draw,
  seed or fraction named in its message, never answered.

The binary is ``SDM_STANDIN`` when set, else built by Cargo from
``rust/`` (``cargo build -p sdm-testkit --bin sdm-standin``).

Each test listed in ``contract/tests/t33-pending.json``'s ``pending`` (as
``py:<function name>``) failed when written and is held to failing
(``t33_gate.py``) until the implementer takes it out.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _load(name: str, path: Path) -> ModuleType:
    key = f't33_standin_{name}'
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[key]
        raise
    return module


fixtures = _load('fixtures', HERE / 't33_fixtures.py')


def _gate() -> ModuleType:
    key = 't33_gate'
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            key, Path(__file__).with_name('t33_gate.py')
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[key] = module
    return sys.modules[key]


#: Runs a test through the gate: it must pass, or fail while pending.
pending = _gate().gated


def _client() -> ModuleType:
    return _load('client', ROOT / 'contract/runners/standin_client.py')


def _captures() -> ModuleType:
    return _load('captures', ROOT / 'contract/runners/captures.py')


_BUILT: list[Path] = []


def _binary() -> Path:
    r"""``sdm-standin``: ``SDM_STANDIN``, else built once by Cargo."""
    named = os.environ.get('SDM_STANDIN')
    if named:
        return Path(named)
    if _BUILT:
        return _BUILT[0]
    done = subprocess.run(
        [
            'cargo',
            'build',
            '-q',
            '-p',
            'sdm-testkit',
            '--bin',
            'sdm-standin',
            '--message-format=json',
        ],
        cwd=ROOT / 'rust',
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    for line in done.stdout.splitlines():
        message = json.loads(line)
        if (
            message.get('executable')
            and message['target']['name'] == 'sdm-standin'
        ):
            _BUILT.append(Path(message['executable']))
            return _BUILT[0]
    raise AssertionError('cargo built no sdm-standin')


@pytest.fixture
def serve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    r"""``serve(capture)``: the capture written, and a stand-in serving it;
    every stand-in started is closed after the test. The writer, the binary
    and the client are reached in the test's body, where the gate holds
    what fails before they exist."""
    opened: list[Any] = []

    def serve(found: dict[str, Any]) -> Any:
        captures = _captures()
        for name in captures.PRIVATE:
            monkeypatch.delenv(name, raising=False)
        binary = _binary()
        target = tmp_path / f'capture-{len(opened)}'
        captures.write(target, found)
        stand_in = _client().StandIn(target, binary=binary)
        opened.append(stand_in)
        return stand_in.__enter__()

    yield serve
    for stand_in in reversed(opened):
        stand_in.__exit__(None, None, None)


def _keys(answer: Any) -> list[int]:
    assert answer.table is not None, answer
    return sorted(answer.table.column('user_id').to_pylist())


def _scan(table: str, column: str) -> dict[str, Any]:
    step = fixtures.scan_users()
    step['request']['table'] = {'source': [table]}
    step['request']['columns'] = [column]
    step['plan']['statements'][0]['text'] = (
        f'SELECT `{column}`\nFROM `{{catalog}}`.`{{schema}}`.`{table}`'
    )
    return step


# --- exit 2: refused by name ------------------------------------------------


@pending
def test_t33_a_dropped_recorded_table_fails_by_name(serve: Any) -> None:
    r"""Exit 2: the stand-in fails by name on a recorded table dropped,
    never answering from nothing (design, Risks: "a missing table or draw
    fails by name, never passes silently"); with the table recorded, the
    same scan is answered on the local engine from its rows."""
    whole = serve(fixtures.capture())
    answered = whole.fetch(_scan('orders', 'order_id'))
    assert answered.table is not None and answered.live_only is None
    assert sorted(answered.table.column('order_id').to_pylist()) == list(
        range(80)
    )
    dropped = serve(fixtures.without(fixtures.capture(), table='orders'))
    with pytest.raises(_client().StandInFailure, match='orders'):
        dropped.fetch(_scan('orders', 'order_id'))


@pending
def test_t33_a_dropped_random_draw_fails_by_name(serve: Any) -> None:
    r"""Exit 2: an entity-sampling request is answered only with the old
    client's recorded draw of the same table, seed and fraction text; with
    the draw dropped it fails by name, never drawn locally."""
    whole = serve(fixtures.capture())
    assert _keys(whole.fetch(fixtures.sample_users())) == [2, 5]
    dropped = serve(fixtures.without(fixtures.capture(), draws=True))
    with pytest.raises(_client().StandInFailure, match='users'):
        dropped.fetch(fixtures.sample_users())


#: (the fraction recorded, the seed asked, the fraction asked): each asked
#: draw differs from the recorded one in its seed or its fraction's text
#: alone, the same value written otherwise included.
OTHER_DRAWS = [
    ('12.5', fixtures.DRAW_SEED + 1, '12.5'),
    ('12.5', fixtures.DRAW_SEED, '12.50'),
    ('12.5', fixtures.DRAW_SEED, '12.6'),
    ('1e-05', fixtures.DRAW_SEED, '0.00001'),
    ('100.0', fixtures.DRAW_SEED, '100'),
]


@pending
@pytest.mark.parametrize(
    ('recorded', 'seed', 'fraction'),
    OTHER_DRAWS,
    ids=[f'{r}-as-{f}-seed-{s}' for r, s, f in OTHER_DRAWS],
)
def test_t33_a_draw_asked_with_another_seed_or_fraction_text_fails_by_name(
    serve: Any, recorded: str, seed: int, fraction: str
) -> None:
    r"""Exit 2: a draw asked with another seed or another fraction text fails
    by name (so a wrong percentage text or seed fails offline, RS-6),
    naming what was asked; the draw as recorded, a seed past 2**31 and a
    percentage in scientific notation included, is served."""
    found = fixtures.capture()
    found['draws'] = [{**found['draws'][0], 'fraction': recorded}]
    served = serve(found).fetch(fixtures.sample_users(fraction=recorded))
    assert _keys(served) == [2, 5]
    asked = str(seed) if seed != fixtures.DRAW_SEED else fraction
    with pytest.raises(_client().StandInFailure) as refused:
        serve(found).fetch(fixtures.sample_users(seed=seed, fraction=fraction))
    assert asked in str(refused.value), refused.value


# --- exit 2: exact replay ---------------------------------------------------


@pending
def test_t33_an_exact_replay_statement_serves_its_recorded_result(
    serve: Any,
) -> None:
    r"""Exit 2: a statement the old client sent, with its parameters, is
    answered with its recorded result, after whitespace is normalised;
    the same statement with another parameter is not replayed but answered
    on the local engine from the recorded table."""
    stand_in = serve(fixtures.capture())
    spaced = (
        'SELECT   `user_id`  FROM\n\n `{catalog}`.`{schema}`.`users`\tLIMIT  ?'
    )
    replayed = stand_in.fetch(fixtures.preview_users(spaced, 3, step_id=1))
    assert replayed.live_only is None
    assert (
        replayed.table.column('user_id').to_pylist() == fixtures.REPLAYED_ROWS
    )
    local = stand_in.fetch(
        fixtures.preview_users(fixtures.REPLAYED_TEXT, 2, step_id=2)
    )
    keys = local.table.column('user_id').to_pylist()
    assert len(keys) == 2 and set(keys) <= set(range(8)), keys
