# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32e part 1's real-data parity benchmark, held offline to the brief:
the independent test author's seeds, written before the benchmark exists.
No test here makes a network call, reads a Databricks configuration or
reads a dataset: every check runs on stubs, in-process data or a refused
command line.

    python -m pytest contract/tests/test_t132e_benchmark_seeds.py -q

The interface these seeds hold, which the implementer provides (the brief
leaves the script's shape open: "a new script or a mode of the existing
one"; these seeds fix the minimum a reviewer can check offline). The
module is ``scripts/live/real_data_benchmark.py``; it may import or
re-export what ``scripts/live/null_link_benchmark.py`` already has:

- ``DATASETS: tuple[str, ...]``: the datasets a run may name, exactly
  ``'adventureworks-full'`` (full-size AdventureWorks from Microsoft's
  source files) and ``'rel-hm'`` (RelBench H&M's article, customer and
  transactions), both read from local files, never committed.
- ``SEEDS: tuple[int, ...]`` and ``ENTITIES: int``: the default paired
  seeds and the entities asked about in each call.
- ``plan(names: Sequence[str], seeds: Sequence[int], entities: int) ->
  tuple[int, int]``: the calls and predictions a run makes, counted from
  the datasets' fixed tasks without reading any data: one call per
  dataset, task, seed and route (old and core), each asking about
  ``entities`` entities.
- ``PREDICTION_CEILING = 500``: the most predictions one run may send.
- ``WireBudget``, an exception that is no ``OSError``, and ``Wire(limit)``
  with ``install()``, ``uninstall()``, ``allow_one() -> int``, ``sent``
  and ``limit``, as ``null_link_benchmark`` has them: wrapping
  ``requests.Session.send`` so each request to a serving endpoint's
  ``.../serving-endpoints/<name>/invocations`` is counted where it is
  sent; each allowed prediction may send one; a second (an SDK retry) and
  any past ``limit`` are refused with ``WireBudget`` before they are sent.
- ``judge(kind, truth, old, core) -> str``: one unit's verdict,
  ``'PASS'``, ``'FAIL'`` or ``'INCONCLUSIVE'``. ``kind`` is
  ``'regression'``, ``'binary'`` or ``'multiclass'``; ``truth`` the
  entities' labels (a number, a bool, a class); ``old`` and ``core`` one
  list per seed of the entities' answers, aligned with ``truth``, each a
  dict with ``'prediction'`` and, for a binary task, ``'score'`` (the
  positive class's probability). The sanity floor, on the old route's
  metric averaged over the seeds: AUROC below 0.6; macro-F1 at or below
  chance, ``1 / k`` for the ``k`` classes ``truth`` holds (the expected
  macro-F1 of guessing by the labels' own shares); MAE no lower than that
  of predicting ``truth``'s mean for every entity. A unit under the floor
  is ``'INCONCLUSIVE'`` whatever the core answered, never a pass or a
  failure; above it, the core answering as the old route did passes and
  the core far worse fails (the rule between is the implementer's).
- ``redact``: ``rfm_smoke``'s redaction, and ``summary_text(summary:
  dict) -> str``: the text written to ``--summary``, JSON, every line
  passed through ``redact``.
- The command line: ``--dry-run``, ``--dataset NAME`` (repeatable;
  default every one of ``DATASETS``), ``--data-dir DIR`` (where the
  local files are), ``--seeds``, ``--entities``, ``--max-predictions``
  (default and most 500), ``--summary FILE``. Usage it refuses exits 2:
  a dataset not in ``DATASETS`` (every generated dataset of
  ``null_link_benchmark`` among them), its message naming the dataset;
  ``--max-predictions`` above 500 or below 1; and a plan above
  ``--max-predictions``, its message naming ``max-predictions``; each
  refused before any data is read or request built.

Added after the review of 18bd89d6 (the sanity floor bypassed when the
core refuses), against the interface the benchmark then had:

- ``unit(dataset, task, keys, truth, answers, refused, digests) -> dict``
  (``task`` with ``name`` and ``kind``; ``answers`` and ``digests`` by
  route, ``'old'`` and ``'core'``; ``refused`` each refusing route's
  refusal) and ``run(sets, seeds, count, client, recording, raw, wire)
  -> (described, units)``. A unit's ``verdict`` is its quality verdict
  alone: the sanity floor is applied to the old route's evidence first,
  so under it the verdict is ``'INCONCLUSIVE'`` even where the core
  refused. A refusal is reported apart from it, in ``refused`` (kept as
  recorded) and in a new field ``operational``: ``'FAIL'`` when any route
  refused, ``'OK'`` otherwise. Over the floor a core refusal is never a
  pass.

Added after the live run of 2026-10-13 (a server's HTTP 400 aborted the
whole run with no summary; the old and core requests differing only in
float32 label rounding, ORD-3):

- A prediction the endpoint refuses (the client's ``NimRequestError``, an
  HTTP 4xx) is an operational failure of its unit: ``operational``
  ``'FAIL'``, the reason in ``operational_errors`` naming the error's
  code and HTTP status and never its text (an endpoint wrote it); the
  other units still run; the summary is still written, with
  ``requests_sent`` and ``predictions_sent`` (the requests counted where
  they were sent, and those times the entities asked about); and the run
  exits 1. Driven through ``main`` with the client
  (``rfm_smoke._client``), the data (``dataset``, ``rows_of``,
  ``describe``) and the request (``null_link_benchmark.build``) stubbed,
  and ``requests.Session.send`` answering in process.
- ``compare_requests(old: dict, core: dict) -> str``: two request payloads
  as the client sends them, ``'equal'`` when they are one request
  (``metadata.request_id`` aside), ``'equal-within-ORD-3'`` when they
  differ only in label values (the cells of the column
  ``task.target.column_name`` names, in ``context.instance_table``), each
  pair numbers within float32 rounding (``|a - b| <= 8 * 2**-23 *
  max(|a|, |b|)``, a few float32 steps), and ``'different'`` otherwise:
  any other difference, a label beyond that, or a label that is no number
  on either side.

Brief lines proved, T1.32e part 1 scope 2 ("a real-data parity
benchmark ... with no generated data. It is a sanity check, not a gate"):

- "Sanity floor: if the old route's metric is near random (AUROC < 0.6,
  or macro-F1 at chance, or MAE no better than predicting the mean),
  report that unit INCONCLUSIVE, never pass or fail":
  ``test_a_unit_under_the_floor_is_inconclusive`` and
  ``test_a_unit_over_the_floor_is_judged`` (the control, so the floor is
  not "always inconclusive").
- "no generated data": ``test_the_datasets_are_the_real_ones`` and
  ``test_a_generated_dataset_is_refused``.
- "reuse the existing wire budget and retry refusal, at most 500
  predictions per run": ``test_the_budget_is_500_predictions``,
  ``test_a_plan_over_the_budget_is_refused_before_any_data``,
  ``test_the_default_plan_fits_the_budget``, and the wire tests.
- "the redaction (no hosts or credentials)":
  ``test_no_host_appears_in_the_summary``.

Gating (``contract/tests/t132e_gate.py``): every test here failed when it
was written (the module did not exist), so each is listed in
``t132e-pending.json``'s ``pending["exits"]`` as ``py:<test id>``; each
still runs and must still fail until the implementer takes its id out.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/live/real_data_benchmark.py'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


G = _module('t132e_gate', ROOT / 'contract/tests/t132e_gate.py')


def benchmark() -> ModuleType:
    r"""The benchmark's module; fails (inside a held check) while it does
    not exist."""
    assert SCRIPT.is_file(), f'{SCRIPT.relative_to(ROOT)} does not exist'
    return _module('real_data_benchmark', SCRIPT)


def held(request: pytest.FixtureRequest, check: Callable[[], None]) -> None:
    G.held('exits', f'py:{request.node.name}', check)


def _cli(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    r"""The script's command line, in a child, with no network to reach:
    every check here is a refusal before any call."""
    return subprocess.run(
        [sys.executable, '-I', str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
        timeout=600,
        env={'PATH': '/usr/bin:/bin', 'HOME': str(cwd)},
    )


# --- the answers ------------------------------------------------------------------

#: A binary task's labels: 4 positives, 16 negatives (64 pairs).
BINARY = [True] * 4 + [False] * 16


def binary(below: int) -> list[dict[str, Any]]:
    r"""Answers whose AUROC is ``4 * below / 64``: every positive scored
    0.5, ``below`` of the 16 negatives under it, the rest over it, no
    ties."""
    answers = [{'prediction': True, 'score': 0.5} for _ in range(4)]
    for at in range(16):
        score = 0.01 * (at + 1) if at < below else 0.6 + 0.01 * at
        answers.append({'prediction': score > 0.5, 'score': score})
    return answers


def auroc(truth: list[bool], answers: list[dict[str, Any]]) -> float:
    r"""The pairs ranked right, ties half: an independent count."""
    pos = [a['score'] for t, a in zip(truth, answers, strict=True) if t]
    neg = [a['score'] for t, a in zip(truth, answers, strict=True) if not t]
    right = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return right / (len(pos) * len(neg))


#: A multiclass task's labels: three classes, four entities each.
CLASSES = list('aaaabbbbcccc')


def classes(predicted: str) -> list[dict[str, Any]]:
    return [{'prediction': c} for c in predicted]


def macro_f1(truth: list[str], predicted: list[str]) -> float:
    found = []
    for c in sorted({*truth, *predicted}):
        tp = sum(
            t == c and p == c for t, p in zip(truth, predicted, strict=True)
        )
        fp = sum(
            t != c and p == c for t, p in zip(truth, predicted, strict=True)
        )
        fn = sum(
            t == c and p != c for t, p in zip(truth, predicted, strict=True)
        )
        found.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(found) / len(found)


#: A regression task's labels, mean 5.
NUMBERS = [1.0, 3.0, 5.0, 7.0, 9.0, 5.0]


def numbers(predicted: list[float]) -> list[dict[str, Any]]:
    return [{'prediction': p} for p in predicted]


def mae(truth: list[float], predicted: list[float]) -> float:
    return sum(abs(t - p) for t, p in zip(truth, predicted, strict=True)) / len(
        truth
    )


def test_the_seeded_answers_have_the_metrics_they_name() -> None:
    # The seeds' own arithmetic, independent of the benchmark (always run):
    # the floors below are crossed where the tests say.
    assert auroc(BINARY, binary(9)) == 36 / 64 < 0.6
    assert auroc(BINARY, binary(10)) == 40 / 64 > 0.6
    assert auroc(BINARY, binary(8)) == 0.5
    assert auroc(BINARY, binary(16)) == 1.0
    assert auroc(BINARY, binary(0)) == 0.0
    chance = 1 / 3
    assert macro_f1(CLASSES, list('aaaaaaaaaaaa')) < chance
    assert abs(macro_f1(CLASSES, list('bcacbbacbaca')) - chance) < 1e-12
    assert macro_f1(CLASSES, list('aaaabbbbcccc')) == 1.0
    assert macro_f1(CLASSES, list('aaabbbbccccb')) > 0.5
    mean = sum(NUMBERS) / len(NUMBERS)
    assert mean == 5.0
    at_mean = mae(NUMBERS, [mean] * len(NUMBERS))
    assert mae(NUMBERS, [p + 0.5 for p in NUMBERS]) < at_mean
    assert mae(NUMBERS, [0.0] * len(NUMBERS)) > at_mean


#: Units under the floor: kind, labels, the old route's answers (the same
#: on both seeds), and why.
UNDER = {
    'auroc-at-half': ('binary', BINARY, binary(8)),
    'auroc-0.5625': ('binary', BINARY, binary(9)),
    'auroc-inverted': ('binary', BINARY, binary(0)),
    'macro-f1-one-class': ('multiclass', CLASSES, classes('aaaaaaaaaaaa')),
    'macro-f1-at-chance': ('multiclass', CLASSES, classes('bcacbbacbaca')),
    'mae-at-the-mean': ('regression', NUMBERS, numbers([5.0] * 6)),
    'mae-worse-than-the-mean': ('regression', NUMBERS, numbers([0.0] * 6)),
}
#: What the core may answer: as the old route, perfectly, or worst.
CORES = {
    'binary': {
        'same': None,
        'perfect': binary(16),
        'worst': binary(0),
    },
    'multiclass': {
        'same': None,
        'perfect': classes('aaaabbbbcccc'),
        'worst': classes('bbbbccccaaaa'),
    },
    'regression': {
        'same': None,
        'perfect': numbers(list(NUMBERS)),
        'worst': numbers([100.0] * 6),
    },
}


@pytest.mark.parametrize(
    ('unit', 'core'),
    [
        pytest.param(unit, core, id=f'{unit}-core-{core}')
        for unit in UNDER
        for core in ('same', 'perfect', 'worst')
    ],
)
def test_a_unit_under_the_floor_is_inconclusive(
    unit: str, core: str, request: pytest.FixtureRequest
) -> None:
    kind, truth, old = UNDER[unit]
    answer = CORES[kind][core] or old

    def check() -> None:
        found = benchmark().judge(kind, truth, [old, old], [answer, answer])
        assert found == 'INCONCLUSIVE', (unit, core, found)

    held(request, check)


#: Units over the floor: kind, labels, the old route's answers, and the
#: core's far worse answers.
OVER = {
    'auroc-0.625': ('binary', BINARY, binary(10), binary(0)),
    'auroc-perfect': ('binary', BINARY, binary(16), binary(0)),
    'macro-f1-good': (
        'multiclass',
        CLASSES,
        classes('aaabbbbccccb'),
        classes('bbbbccccaaaa'),
    ),
    'mae-good': (
        'regression',
        NUMBERS,
        numbers([p + 0.5 for p in NUMBERS]),
        numbers([100.0] * 6),
    ),
}


@pytest.mark.parametrize('unit', [pytest.param(u, id=u) for u in OVER])
def test_a_unit_over_the_floor_is_judged(
    unit: str, request: pytest.FixtureRequest
) -> None:
    r"""The control: over the floor, the core answering as the old route
    passes and the core far worse fails; never inconclusive."""
    kind, truth, old, worse = OVER[unit]

    def check() -> None:
        judge = benchmark().judge
        assert judge(kind, truth, [old, old], [old, old]) == 'PASS'
        assert judge(kind, truth, [old, old], [worse, worse]) == 'FAIL'

    held(request, check)


def test_the_floor_reads_the_old_route_averaged_over_the_seeds(
    request: pytest.FixtureRequest,
) -> None:
    r"""One seed over the floor, the other under: the mean decides (AUROC
    (0.625 + 0.5) / 2 < 0.6, so inconclusive; (1.0 + 0.5) / 2 > 0.6, so
    judged)."""

    def check() -> None:
        judge = benchmark().judge
        low = [binary(10), binary(8)]
        assert judge('binary', BINARY, low, low) == 'INCONCLUSIVE'
        high = [binary(16), binary(8)]
        assert judge('binary', BINARY, high, high) == 'PASS'

    held(request, check)


# --- the data ---------------------------------------------------------------------


def test_the_datasets_are_the_real_ones(request: pytest.FixtureRequest) -> None:
    def check() -> None:
        found = benchmark().DATASETS
        assert isinstance(found, tuple)
        assert set(found) == {'adventureworks-full', 'rel-hm'}
        assert not any('generated' in name for name in found)

    held(request, check)


@pytest.mark.parametrize(
    'name',
    ['generated-nulls', 'generated-control', 'generated', 'no-such-dataset'],
)
def test_a_generated_dataset_is_refused(
    name: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    r"""The real-data mode takes no generated dataset (the null-link
    benchmark's two among them), refused as usage before any data is
    read."""

    def check() -> None:
        assert SCRIPT.is_file(), f'{SCRIPT.relative_to(ROOT)} does not exist'
        done = _cli(
            '--dry-run',
            '--dataset',
            name,
            '--data-dir',
            str(tmp_path),
            cwd=tmp_path,
        )
        assert done.returncode == 2, done.stdout + done.stderr
        assert name in done.stdout + done.stderr

    held(request, check)


# --- the budget -------------------------------------------------------------------


@pytest.mark.parametrize('limit', ['501', '1000', '0', '-1'])
def test_the_budget_is_500_predictions(
    limit: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    def check() -> None:
        assert benchmark().PREDICTION_CEILING == 500
        done = _cli(
            '--dry-run',
            '--max-predictions',
            limit,
            '--data-dir',
            str(tmp_path),
            cwd=tmp_path,
        )
        assert done.returncode == 2, done.stdout + done.stderr

    held(request, check)


def test_a_plan_over_the_budget_is_refused_before_any_data(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    r"""500 entities a call makes any plan of one call past 500
    predictions: refused as usage, with the data directory empty (a run
    that read data first would fail otherwise)."""

    def check() -> None:
        assert SCRIPT.is_file(), f'{SCRIPT.relative_to(ROOT)} does not exist'
        done = _cli(
            '--dry-run',
            '--entities',
            '500',
            '--data-dir',
            str(tmp_path),
            cwd=tmp_path,
        )
        assert done.returncode == 2, done.stdout + done.stderr
        assert 'max-predictions' in done.stdout + done.stderr

    held(request, check)


def test_the_default_plan_fits_the_budget(
    request: pytest.FixtureRequest,
) -> None:
    def check() -> None:
        module = benchmark()
        seeds, entities = module.SEEDS, module.ENTITIES
        calls, predictions = module.plan(module.DATASETS, seeds, entities)
        assert 0 < predictions <= module.PREDICTION_CEILING == 500
        assert predictions == calls * entities
        # Both routes, every seed: calls come in pairs per seed.
        assert len(seeds) >= 2 and calls % (2 * len(seeds)) == 0
        each = [
            module.plan([name], seeds, entities) for name in module.DATASETS
        ]
        assert all(p > 0 for _, p in each), 'every dataset asks something'
        assert sum(c for c, _ in each) == calls

    held(request, check)


INVOCATIONS = 'https://example.invalid/serving-endpoints/stub/invocations'


@pytest.fixture
def stubbed_send(monkeypatch: pytest.MonkeyPatch) -> Any:
    r"""``requests.Session.send`` replaced by a stub that records each
    request it would send and sends nothing."""
    import requests

    sent: list[str] = []

    def stub(session: Any, request: Any, **kwargs: Any) -> Any:
        sent.append(request.url)
        return 'answered'

    monkeypatch.setattr(requests.Session, 'send', stub)
    return sent, requests


def _post(requests: Any, url: str = INVOCATIONS) -> Any:
    prepared = requests.Request('POST', url, json={}).prepare()
    return requests.Session().send(prepared)


def test_a_retry_is_refused_before_it_is_sent(
    stubbed_send: Any, request: pytest.FixtureRequest
) -> None:
    sent, requests = stubbed_send

    def check() -> None:
        module = benchmark()
        assert not issubclass(module.WireBudget, OSError)
        wire = module.Wire(2)
        wire.install()
        try:
            assert wire.allow_one() == 0
            assert _post(requests) == 'answered'
            # The SDK's retry of the same prediction: refused, not sent.
            with pytest.raises(module.WireBudget):
                _post(requests)
            assert sent == [INVOCATIONS] and wire.sent == 1
            # A request with no prediction allowed: refused.
            with pytest.raises(module.WireBudget):
                _post(requests)
            assert wire.sent == 1
        finally:
            wire.uninstall()

    held(request, check)


def test_no_request_passes_the_plan(
    stubbed_send: Any, request: pytest.FixtureRequest
) -> None:
    sent, requests = stubbed_send

    def check() -> None:
        module = benchmark()
        wire = module.Wire(2)
        wire.install()
        try:
            for _ in range(wire.limit):
                wire.allow_one()
                _post(requests)
            wire.allow_one()
            with pytest.raises(module.WireBudget):
                _post(requests)
            assert wire.sent == wire.limit == len(sent) == 2
        finally:
            wire.uninstall()

    held(request, check)


# --- the summary ------------------------------------------------------------------

HOST = 'dbc-0123456789abcdef.cloud.databricks.com'
UNKNOWN_HOST = 'adb-9876543210.12.azuredatabricks.net'


def test_no_host_appears_in_the_summary(request: pytest.FixtureRequest) -> None:
    r"""A summary whose units and reasons carry the endpoint, the profile,
    a known workspace host and a host never told of: the text written
    holds none of them, and is JSON."""

    def check() -> None:
        module = benchmark()
        module.redact.know('secret-endpoint-t132e', 'secret-profile-t132e')
        module.redact.know_host(f'https://{HOST}')
        summary = {
            'format': 'any',
            'units': [
                {
                    'dataset': 'rel-hm',
                    'task': 'binary',
                    'verdict': 'INCONCLUSIVE',
                    'note': f'served by https://{HOST}/serving-endpoints/'
                    'secret-endpoint-t132e/invocations',
                }
            ],
            'reasons': [
                'profile secret-profile-t132e on '
                f'https://{UNKNOWN_HOST}/serving-endpoints/x/invocations',
                f'bare host {HOST}',
            ],
        }
        text = module.summary_text(summary)
        for secret in (
            HOST,
            'dbc-0123456789abcdef',
            UNKNOWN_HOST,
            'azuredatabricks.net',
            'secret-endpoint-t132e',
            'secret-profile-t132e',
            'https://',
        ):
            assert secret not in text, secret
        found = json.loads(text)
        assert found['units'][0]['verdict'] == 'INCONCLUSIVE'
        assert found['units'][0]['dataset'] == 'rel-hm'

    held(request, check)


# --- a refusal beside the floor (the review of 18bd89d6) --------------------

#: The review's reproduction: binary labels, the old route's scores all
#: 0.5 (AUROC 0.5, under the floor), two seeds.
FLAT_TRUTH = [False, False, True, True]
FLAT = [{'prediction': False, 'score': 0.5} for _ in FLAT_TRUTH]
#: The old route answering perfectly (AUROC 1, over the floor).
SHARP = [{'prediction': s > 0.5, 'score': s} for s in (0.1, 0.2, 0.8, 0.9)]
CORE_REFUSED = {'error': 'StructuredDataError', 'code': 'INVALID_REQUEST'}


def _task(kind: str = 'binary') -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(name=kind, kind=kind)


def _unit(
    module: Any,
    old: list[list[dict[str, Any]]],
    core: list[list[dict[str, Any]]],
    refused: dict[str, Any],
) -> dict[str, Any]:
    found: dict[str, Any] = module.unit(
        'rel-hm',
        _task(),
        [0, 1, 2, 3],
        list(FLAT_TRUTH),
        {'old': old, 'core': core},
        refused,
        {'old': ['o'] * len(old), 'core': ['c'] * len(core)},
    )
    return found


def test_a_core_refusal_under_the_floor_is_inconclusive_in_the_unit(
    request: pytest.FixtureRequest,
) -> None:
    def check() -> None:
        module = benchmark()
        found = _unit(module, [FLAT, FLAT], [], {'core': dict(CORE_REFUSED)})
        assert found['verdict'] == 'INCONCLUSIVE', found['verdict']
        assert found['refused']['core']['code'] == 'INVALID_REQUEST'
        assert found['operational'] == 'FAIL'

    held(request, check)


def test_a_core_refusal_over_the_floor_is_no_pass_in_the_unit(
    request: pytest.FixtureRequest,
) -> None:
    r"""The control: over the floor a core refusal never passes, and is
    an operational failure too."""

    def check() -> None:
        module = benchmark()
        found = _unit(module, [SHARP, SHARP], [], {'core': dict(CORE_REFUSED)})
        assert found['verdict'] != 'PASS', found['verdict']
        assert found['operational'] == 'FAIL'

    held(request, check)


def test_an_old_route_refusal_is_inconclusive_and_operational(
    request: pytest.FixtureRequest,
) -> None:
    def check() -> None:
        module = benchmark()
        found = _unit(module, [], [SHARP, SHARP], {'old': dict(CORE_REFUSED)})
        assert found['verdict'] == 'INCONCLUSIVE'
        assert found['operational'] == 'FAIL'

    held(request, check)


def test_a_unit_with_no_refusal_is_operationally_ok(
    request: pytest.FixtureRequest,
) -> None:
    def check() -> None:
        module = benchmark()
        flat = _unit(module, [FLAT, FLAT], [FLAT, FLAT], {})
        assert flat['verdict'] == 'INCONCLUSIVE'
        assert flat['operational'] == 'OK'
        sharp = _unit(module, [SHARP, SHARP], [SHARP, SHARP], {})
        assert sharp['verdict'] == 'PASS'
        assert sharp['operational'] == 'OK'

    held(request, check)


class _Refused(Exception):
    r"""A refusal as the client raises one: a class and a code."""

    code = 'INVALID_REQUEST'


def test_run_keeps_the_floor_when_the_core_refuses(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""``run`` end to end on stubs: one binary task, the old route
    building a request the stub client answers with 0.5 for every entity
    on each seed, the core refusing to build one. No data is read, no
    request leaves the process."""

    def check() -> None:
        import io
        from types import SimpleNamespace

        import pandas as pd

        module = benchmark()
        base = module.base
        labels = dict(enumerate(FLAT_TRUTH))
        task = base.Task(
            name='binary',
            kind='binary',
            pql='PREDICT stub',
            anchor=None,
            labels=lambda connection: labels,
        )
        dataset = SimpleNamespace(
            name='rel-hm', connect=lambda: None, tasks=[task]
        )

        def build(
            dataset: Any, task: Any, route: str, seed: int, keys: Any
        ) -> Any:
            if route == 'core':
                raise _Refused('refused')
            return {'route': route, 'seed': seed, 'keys': list(keys)}

        monkeypatch.setattr(base, 'build', build)
        monkeypatch.setattr(module, 'describe', lambda dataset, route: {})
        monkeypatch.setattr(module, 'rows_of', lambda connection: {})
        recording = SimpleNamespace(digests=[])

        class Client:
            def predict(self, built: Any, **_: Any) -> Any:
                recording.digests.append(f'{built["route"]}-{built["seed"]}')
                frame = pd.DataFrame(
                    {
                        'id': list(built['keys']),
                        'prediction': [False] * len(built['keys']),
                        'probability_true': [0.5] * len(built['keys']),
                    }
                )
                return SimpleNamespace(to_pandas=lambda: frame)

        _, units = module.run(
            [dataset], [0, 1], 4, Client(), recording, io.StringIO(), None
        )
        assert len(units) == 1
        found = units[0]
        assert recording.digests == ['old-0', 'old-1']
        assert found['verdict'] == 'INCONCLUSIVE', found['verdict']
        assert found['refused']['core']['code'] == 'INVALID_REQUEST'
        assert found['operational'] == 'FAIL'

    held(request, check)


# --- the live run's lessons (2026-10-13) ------------------------------------

#: The driver a child runs: the benchmark's ``main`` on stubs, the
#: endpoint refusing the core route's requests of the task ``refused``
#: with an HTTP 400 after they are sent, answering every other in process.
DRIVER = """
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import requests

root, scratch = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location(
    'real_data_benchmark', root / 'scripts/live/real_data_benchmark.py'
)
module = importlib.util.module_from_spec(spec)
sys.modules['real_data_benchmark'] = module
spec.loader.exec_module(module)
base, smoke = module.base, module.smoke
from sdm_client import NimRequestError

sent = []


def send(session, request, **kwargs):
    sent.append(request.url)
    response = requests.Response()
    response.status_code = 200
    response._content = b'{}'
    return response


requests.Session.send = send
INVOCATIONS = 'https://example.invalid/serving-endpoints/stub/invocations'


class Transport:
    def predict(self, payload):
        requests.Session().send(
            requests.Request('POST', INVOCATIONS, json={}).prepare()
        )
        if payload['task'] == 'refused' and payload['route'] == 'core':
            raise NimRequestError(
                400,
                code='TASK_SCHEMA_MISMATCH',
                message='the endpoint said secret-detail-7731',
            )
        return {}


class Client:
    def __init__(self):
        self._transport = Transport()

    def predict(self, request, model=None, outputs=None):
        self._transport.predict(request)
        frame = pd.DataFrame(
            {'id': request['keys'], 'prediction': [1.5] * len(request['keys'])}
        )
        return types.SimpleNamespace(to_pandas=lambda: frame)

    def close(self):
        pass


labels = {i: (0.0 if i < 4 else float(i)) for i in range(8)}


def task(name):
    return base.Task(
        name=name,
        kind='regression',
        pql='PREDICT stub',
        anchor=None,
        labels=lambda connection: labels,
    )


def build(dataset, task, route, seed, keys):
    return {'task': task.name, 'route': route, 'seed': seed, 'keys': list(keys)}


base.build = build
module.dataset = lambda name, *rest, **named: types.SimpleNamespace(
    name=name, connect=lambda: None, tasks=[task('refused'), task('answered')]
)
module.rows_of = lambda connection: {}
module.describe = lambda dataset, route: {}
smoke.know_profile = lambda *rest, **named: None
smoke._client = lambda *rest, **named: Client()
code = module.main(
    [
        '--endpoint', 'stub-endpoint-t132e',
        '--profile', 'stub-profile-t132e',
        '--dataset', 'adventureworks-full',
        '--data-dir', str(scratch),
        '--raw-dir', str(scratch / 'raw'),
        '--summary', str(scratch / 'summary.json'),
        '--entities', '4',
        '--seeds', '0,1',
    ]
)
(scratch / 'sent.txt').write_text(str(len(sent)))
sys.exit(code)
"""


def test_a_server_refusal_fails_its_unit_and_the_run_goes_on(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    r"""An HTTP 400 on one unit's request: that unit is an operational
    failure with its reason, the other unit runs and is operationally
    OK, the summary is written with the counts, and the run exits 1."""

    def check() -> None:
        assert SCRIPT.is_file()
        driver = tmp_path / 'driver.py'
        driver.write_text(DRIVER, encoding='utf-8')
        done = subprocess.run(
            [sys.executable, '-I', str(driver), str(ROOT), str(tmp_path)],
            capture_output=True,
            text=True,
            check=False,
            cwd=tmp_path,
            timeout=600,
        )
        output = done.stdout + done.stderr
        assert done.returncode == 1, output[-3000:]
        summary = tmp_path / 'summary.json'
        assert summary.is_file(), output[-3000:]
        text = summary.read_text(encoding='utf-8')
        found = json.loads(text)
        units = {u['task']: u for u in found['units']}
        assert set(units) == {'refused', 'answered'}, list(units)
        refused, answered = units['refused'], units['answered']
        assert refused['operational'] == 'FAIL'
        reasons = ' '.join(refused['operational_errors'])
        assert 'TASK_SCHEMA_MISMATCH' in reasons and '400' in reasons, reasons
        assert answered['operational'] == 'OK', answered['operational_errors']
        sent = int((tmp_path / 'sent.txt').read_text())
        assert sent > 0
        assert found['requests_sent'] == sent
        assert found['predictions_sent'] == sent * 4
        for secret in (
            'secret-detail-7731',
            'example.invalid',
            'stub-endpoint-t132e',
            'stub-profile-t132e',
        ):
            assert secret not in text, secret
            assert secret not in output, secret

    held(request, check)


def _payload(labels: list[Any], **changes: Any) -> dict[str, Any]:
    r"""A regression request as the client sends it: two context rows
    whose labels are ``labels``, a related table, one row to predict."""
    payload: dict[str, Any] = {
        'model': 'kumo-relational',
        'task': {
            'kind': 'regression',
            'target': {'column_name': 'TARGET', 'dtype': 'float32'},
            'entity_table_names': ['customers'],
        },
        'context': {
            'instance_table': {
                'format': 'arrays',
                'columns': ['instance_id', '__kumo_entity_ref_0', 'TARGET'],
                'rows': [[0, 40, labels[0]], [1, 41, labels[1]]],
            },
            'related_tables': {
                'transactions': {
                    'format': 'arrays',
                    'columns': ['instance_id', 'price'],
                    'rows': [[0, 0.16762712597846985], [1, 2.5]],
                }
            },
        },
        'predict': {
            'instance_table': {
                'format': 'arrays',
                'columns': ['instance_id', '__kumo_entity_ref_0'],
                'rows': [[0, 3]],
            }
        },
        'metadata': {'request_id': 'sdm-aaaa', 'num_context_examples': 2},
    }
    for path, value in changes.items():
        where: Any = payload
        *parents, last = path.split('.')
        for part in parents:
            where = where[int(part)] if part.isdigit() else where[part]
        where[int(last) if last.isdigit() else last] = value
    return payload


#: ORD-3's measured pair: one float32 step apart.
ORD3 = (0.16762712597846985, 0.16762711107730865)


def test_identical_requests_tolerate_ord_3_label_rounding_only(
    request: pytest.FixtureRequest,
) -> None:
    def check() -> None:
        compare = benchmark().compare_requests
        same = _payload([ORD3[0], 2.5])
        assert compare(same, _payload([ORD3[0], 2.5])) == 'equal'
        assert (
            compare(
                same, _payload([ORD3[0], 2.5], **{'metadata.request_id': 'x'})
            )
            == 'equal'
        )
        # Only labels, each within float32 rounding.
        assert compare(same, _payload([ORD3[1], 2.5])) == 'equal-within-ORD-3'
        assert (
            compare(_payload([ORD3[0], 2.5]), _payload([ORD3[1], 2.5000002]))
            == 'equal-within-ORD-3'
        )
        # A label beyond float32 rounding.
        assert compare(same, _payload([ORD3[0] * (1 + 1e-4), 2.5])) == (
            'different'
        )
        assert compare(same, _payload([ORD3[0], 3.5])) == 'different'
        # A label that is no number on one side.
        assert compare(same, _payload([None, 2.5])) == 'different'
        # The same rounding anywhere but a label.
        price = 'context.related_tables.transactions.rows.0.1'
        assert compare(same, _payload([ORD3[0], 2.5], **{price: ORD3[1]})) == (
            'different'
        )
        # Labels within rounding beside any other difference.
        for path, value in (
            ('metadata.num_context_examples', 3),
            ('predict.instance_table.rows.0.1', 4),
            ('task.target.column_name', 'LABEL'),
            ('context.instance_table.rows.1.1', 42),
        ):
            assert (
                compare(same, _payload([ORD3[1], 2.5], **{path: value}))
                == 'different'
            ), path
        fewer = _payload([ORD3[1], 2.5])
        fewer['context']['instance_table']['rows'].pop()
        assert compare(same, fewer) == 'different'

    held(request, check)
