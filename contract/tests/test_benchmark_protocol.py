# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""P1B-13's accuracy benchmark (``scripts/live/null_link_benchmark.py``)
held to its protocol, decision E73(a), offline: the T1.29 test author's
seeds, from the protocol check and both reviews of T1.29's first candidate.
No test here makes a network call or reads a Databricks configuration.

- The verdict fails closed on a non-finite result: a NaN prediction or
  score, read and scored as ``run()`` reads and scores it, or a non-finite
  metric or worsening, in any one unit of the seven the default run makes,
  fails ``verdict()`` while the other six pass (the protocol's "no more than
  2% worse" cannot be judged on a NaN, and ``NaN > 0.02`` is False).
- A zero baseline is judged in each metric's direction: AUROC or macro-F1
  rising from 0 is no degradation; MAE rising from 0 is one.
- A live run is refused without the CLI profile the brief fixes (E73(a)'s
  "the serving endpoint the owner authorised", called with a named profile),
  before any client exists; with one, the client is asked for (the control,
  so the refusal is the profile's).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _benchmark() -> Any:
    name = 'null_link_benchmark'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / 'scripts/live/null_link_benchmark.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _benchmark()

#: The default run's seven units: dataset, task, kind, whether a control.
UNITS = (
    ('generated-nulls', 'regression', 'regression', False),
    ('generated-nulls', 'binary', 'binary', False),
    ('generated-nulls', 'multiclass', 'multiclass', False),
    ('generated-control', 'regression', 'regression', True),
    ('generated-control', 'binary', 'binary', True),
    ('generated-control', 'multiclass', 'multiclass', True),
    ('adventureworks', 'regression', 'regression', False),
)
#: A passing metric of each kind, the same on both routes.
GOOD = {'regression': 1.25, 'binary': 0.75, 'multiclass': 0.5}


def unit(
    dataset: str, task: str, kind: str, control: bool, old: Any, core: Any
) -> dict[str, Any]:
    r"""A unit as ``run()`` writes it, its worsening from ``worse()``."""
    worse = None
    if old is not None and core is not None:
        worse = benchmark.worse(kind, old, core)
    return {
        'dataset': dataset,
        'task': task,
        'old': old,
        'core': core,
        'worse': worse,
        'control': control,
        'interval': (0.0, 0.0) if control else None,
    }


def seven(**changed: Any) -> list[dict[str, Any]]:
    r"""The seven units, each passing, but the one ``changed`` names by its
    index (``at``), with its ``old`` and ``core``."""
    found = []
    for at, (dataset, task, kind, control) in enumerate(UNITS):
        old = core = GOOD[kind]
        if changed.get('at') == at:
            old, core = changed['old'], changed['core']
        found.append(unit(dataset, task, kind, control, old, core))
    return found


def test_the_seven_passing_units_pass() -> None:
    r"""The control: the seeds below fail on their one change alone."""
    assert benchmark.verdict(seven()) == (True, [])


@pytest.mark.parametrize('at', range(len(UNITS)))
@pytest.mark.parametrize(
    ('old', 'core'),
    [
        (1.0, math.nan),
        (math.nan, 1.0),
        (math.nan, math.nan),
        (1.0, math.inf),
        (math.inf, 1.0),
        (1.0, -math.inf),
    ],
    ids=['core nan', 'old nan', 'both nan', 'core inf', 'old inf', 'core -inf'],
)
def test_a_non_finite_metric_in_any_unit_fails_the_verdict(
    at: int, old: float, core: float
) -> None:
    passed, reasons = benchmark.verdict(seven(at=at, old=old, core=core))
    assert passed is False, reasons
    dataset, task, _, _ = UNITS[at]
    assert any(f'{dataset}/{task}' in reason for reason in reasons), reasons


@pytest.mark.parametrize('at', range(len(UNITS)))
def test_a_non_finite_worsening_fails_the_verdict(at: int) -> None:
    units = seven()
    units[at]['worse'] = math.nan
    passed, reasons = benchmark.verdict(units)
    assert passed is False, reasons


def _frame(column: str, values: list[Any], ids: list[int]) -> Any:
    import pandas as pd

    frame = {'row_index': list(range(len(ids))), 'id': ids}
    if column == 'prediction':
        frame['prediction'] = values
    else:
        frame['prediction'] = [v == v and v > 0.5 for v in values]
        frame[column] = values
    return pd.DataFrame(frame)


@pytest.mark.parametrize(
    ('kind', 'column'),
    [('regression', 'prediction'), ('binary', 'probability_true')],
)
@pytest.mark.parametrize('bad', [math.nan, math.inf])
def test_a_non_finite_answer_read_and_scored_fails_closed(
    kind: str, column: str, bad: float
) -> None:
    r"""A regression's prediction (which the product passes through
    non-finite, E78(h)) or a binary task's positive-class probability,
    answered non-finite for one entity on both routes, so no worsening
    between them can catch it: ``read()`` refuses it, or the unit
    ``metric()`` scores from it fails the verdict, the other six passing.
    Never a metric the verdict takes. The binary task's bad score is a
    positive entity's highest, where an infinity would leave the ranks, and
    so the AUROC, as they were."""
    [task] = [t for t in benchmark.generated_tasks() if t.kind == kind]
    ids = [1, 2, 3, 4]
    truth = [0.0, 1.0, 2.0, 3.0] if kind == 'regression' else [0, 1, 0, 1]
    good = (
        [0.5, 1.5, 2.5, 3.5] if kind == 'regression' else [0.1, 0.9, 0.2, 0.8]
    )
    values = [good[0], bad, *good[2:]]
    try:
        read = benchmark.read(_frame(column, values, ids), task)
    except benchmark.smoke.SmokeError:
        return
    answers = [read[k] for k in ids]
    try:
        value = benchmark.metric(kind, truth, answers)
    except (ValueError, benchmark.smoke.SmokeError):
        return
    units = seven()
    at = [u[1] for u in UNITS].index(task.name)
    dataset, name, kind_of, control = UNITS[at]
    units[at] = unit(dataset, name, kind_of, control, value, value)
    passed, reasons = benchmark.verdict(units)
    assert passed is False, (value, reasons)


# --- a zero baseline -----------------------------------------------------------


@pytest.mark.parametrize('kind', ['binary', 'multiclass'])
@pytest.mark.parametrize('core', [0.0, 0.5, 1.0])
def test_a_classification_metric_rising_from_zero_is_no_degradation(
    kind: str, core: float
) -> None:
    assert benchmark.worse(kind, 0.0, core) <= 0
    at = [k for _, _, k, _ in UNITS].index(kind)
    assert benchmark.verdict(seven(at=at, old=0.0, core=core)) == (True, [])


def test_an_actual_auroc_from_zero_to_one_passes() -> None:
    old = benchmark.auroc([False, True], [1.0, 0.0])
    core = benchmark.auroc([False, True], [0.0, 1.0])
    assert (old, core) == (0.0, 1.0)
    assert benchmark.verdict(seven(at=1, old=old, core=core)) == (True, [])


@pytest.mark.parametrize('core', [0.01, 0.5])
def test_mae_rising_from_zero_is_degradation(core: float) -> None:
    assert benchmark.worse('regression', 0.0, core) > benchmark.TOLERANCE
    passed, reasons = benchmark.verdict(seven(at=0, old=0.0, core=core))
    assert passed is False
    assert any('generated-nulls/regression' in r for r in reasons), reasons


def test_a_zero_baseline_held_is_no_change() -> None:
    for kind in ('regression', 'binary', 'multiclass'):
        assert benchmark.worse(kind, 0.0, 0.0) == 0


def test_a_classification_falling_to_zero_is_degradation() -> None:
    assert benchmark.worse('binary', 0.5, 0.0) > benchmark.TOLERANCE
    assert benchmark.verdict(seven(at=1, old=0.5, core=0.0))[0] is False


# --- the profile ------------------------------------------------------------------


class _NoNetwork(socket.socket):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError('a socket was opened')


@pytest.fixture
def live(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    r"""A live run's surroundings with nothing live: no Databricks or
    benchmark variable, a configuration file that does not exist, no
    socket, and ``rfm_smoke._client`` recording each call and failing."""
    import os

    for name in list(os.environ):
        if name.startswith(('DATABRICKS_', 'SDM_BENCHMARK_', 'SDM_SMOKE_')):
            monkeypatch.delenv(name)
    monkeypatch.setenv('DATABRICKS_CONFIG_FILE', str(tmp_path / 'absent.cfg'))
    monkeypatch.setattr(socket, 'socket', _NoNetwork)
    asked: list[Any] = []

    def client(*args: Any, **kwargs: Any) -> Any:
        asked.append(args)
        raise benchmark.smoke.SmokeError('stubbed: no client in a test')

    monkeypatch.setattr(benchmark.smoke, '_client', client)
    # It rewires the root logger and warnings.showwarning for the process.
    monkeypatch.setattr(benchmark.smoke, '_quiet_the_world', lambda: None)
    # The run itself is never reached: the stub client fails first.
    monkeypatch.setattr(
        benchmark,
        'run',
        lambda *a, **k: pytest.fail('the run started'),
    )
    return {
        'asked': asked,
        'arguments': [
            '--endpoint',
            'an-endpoint',
            '--raw-dir',
            str(tmp_path / 'raw'),
            '--summary',
            str(tmp_path / 'summary.json'),
        ],
    }


def test_a_live_run_without_a_profile_is_refused_before_any_call(
    live: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    status = benchmark.main(live['arguments'])
    assert status == 2, capsys.readouterr()
    assert live['asked'] == [], 'a client was asked for with no profile'
    err = capsys.readouterr().err
    assert 'profile' in err, err


def test_a_live_run_with_a_profile_asks_for_its_client(
    live: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""The control: the same run with ``--profile``, or with its
    variable, reaches the client (stubbed, failing), so the refusal above
    is the missing profile's."""
    status = benchmark.main([*live['arguments'], '--profile', 'A_PROFILE'])
    assert status == 1
    assert len(live['asked']) == 1
    assert live['asked'][0][1] == 'A_PROFILE'
    monkeypatch.setenv(benchmark.PROFILE_VARIABLE, 'ANOTHER')
    assert benchmark.main(live['arguments']) == 1
    assert live['asked'][1][1] == 'ANOTHER'


# --- the whole dry run --------------------------------------------------------

#: The default run's units, by dataset and task, in the order it runs them.
SEVEN = [(dataset, task) for dataset, task, _, _ in UNITS]


def test_the_offline_dry_run_completes_all_seven_units_and_writes_the_summary(
    tmp_path: Path,
) -> None:
    r"""Codex's review of part 1: the whole ``--dry-run``, as run by hand,
    builds every request on both routes and answers it in process; it must
    reach AdventureWorks too, whatever the case files' status (part 1 keeps
    ``base.adventureworks`` diagnostic), and write its summary. Run as a
    separate process with no endpoint, profile or Databricks variable; the
    script's dry run opens no socket."""
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(('DATABRICKS_', 'SDM_BENCHMARK_', 'SDM_SMOKE_'))
        and name != 'SDM_CORE_ROUTES'
    }
    env['DATABRICKS_CONFIG_FILE'] = str(tmp_path / 'absent.cfg')
    summary = tmp_path / 'summary.json'
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'scripts/live/null_link_benchmark.py'),
            '--dry-run',
            '--summary',
            str(summary),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=tmp_path,
        timeout=900,
    )
    output = done.stdout + done.stderr
    assert done.returncode == 0, output[-4000:]
    assert 'benchmark failed' not in output, output[-4000:]
    assert summary.is_file(), output[-4000:]
    found = json.loads(summary.read_text(encoding='utf-8'))
    assert found['dry_run'] is True
    assert found['passed'] is None
    units = found['units']
    assert len(units) == 7, [(u['dataset'], u['task']) for u in units]
    assert [(u['dataset'], u['task']) for u in units] == SEVEN
    for each in units:
        assert each['old'] is not None and each['core'] is not None, each
    assert found['calls'] == 28 and found['predictions'] == 448
