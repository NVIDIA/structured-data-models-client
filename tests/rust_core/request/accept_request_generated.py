# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Generated results and rankings (``CONTRACT.md``, "Generated results").

- P85.results: ``PredictionResult``s a user can build give, through the
  host's ``flatten``, ``frame``, ``absent`` and ``merge`` (and so through the
  core's), what today's ``to_pandas``, ``absent`` and ``merge`` give.
- P85.decompose: generated rankings, ties included, order as today's
  ``_ranked`` orders them, through ``request_fixture_op('client.ranked')``.
"""

from __future__ import annotations

import json
import random
from typing import Any

import pytest
from _request import (
    PureInterpreter,
    call_absent,
    call_flatten,
    call_merge,
    frame_difference,
    json_diff,
    qualname,
    recorder,
    short,
)

RESULTS_SEED = 20261022
RESULTS = 600
MERGES = 512
RANKED_SEED = 20261021
RANKED = 3000

FIELDS = (
    'prediction',
    'probabilities',
    'quantiles',
    'scores',
    'rankings',
    'embeddings',
    'explanation',
)


def _ids(rng: random.Random, kind: str, n: int) -> list[Any]:
    import pandas as pd

    def one(k: int, kind: str) -> Any:
        if kind == 'int':
            return rng.randint(-5, 10**12)
        if kind == 'str':
            return rng.choice(['a', 'b', 'é', '10', '', 'x y']) + str(k)
        if kind == 'timestamp':
            return pd.Timestamp('2024-01-01') + pd.Timedelta(
                hours=rng.randint(0, 10_000)
            )
        if kind == 'tuple':
            return (rng.randint(0, 9), rng.choice(['p', 'q']))
        return None

    if kind == 'mixed':
        return [one(k, rng.choice(['int', 'str', 'none'])) for k in range(n)]
    return [one(k, kind) for k in range(n)]


def _result(rng: random.Random, model: str, outputs: tuple[str, ...]) -> Any:
    from sdm_api.generated.tfm_api import PredictionItem
    from sdm_client.result import PredictionResult

    kind = rng.choice(
        ['regression', 'binary', 'multiclass', 'ranking', 'forecast', 'mixed']
    )
    n = rng.choice([0, 0, 1, 2, 3, 5, 8])
    ids = _ids(
        rng,
        rng.choice(['int', 'str', 'timestamp', 'tuple', 'none', 'mixed']),
        n,
    )
    rows = list(range(n))
    rng.shuffle(rows)
    labels = rng.choice([['books', 'games'], ['0', '1', '2'], ['north']])
    items = []
    for k in range(n):
        fields: dict[str, Any] = {'row_index': rows[k], 'id': ids[k]}
        if kind == 'forecast':
            fields['forecast_step'] = rng.randint(1, 4)
        if rng.random() > 0.15:
            if kind in ('regression', 'forecast'):
                fields['prediction'] = rng.choice(
                    [rng.uniform(-1e6, 1e6), 0.1, float('nan'), 3.0]
                )
            elif kind == 'binary':
                fields['prediction'] = rng.choice([True, False, 'True'])
            elif kind == 'multiclass':
                fields['prediction'] = rng.choice(labels)
            elif kind == 'ranking':
                fields['prediction'] = rng.sample(
                    [1, 2, 3, 4], rng.randint(0, 3)
                )
            else:
                fields['prediction'] = rng.choice([1, 2.5, 'x', None, True])
        if 'probabilities' in outputs and rng.random() > 0.2:
            keys = labels if kind == 'multiclass' else ['False', 'True']
            if rng.random() < 0.2:
                keys = keys[:1]
            fields['probabilities'] = {
                key: round(rng.random(), 6) for key in keys
            }
        if 'quantiles' in outputs and rng.random() > 0.2:
            fields['quantiles'] = {
                q: rng.uniform(0, 100) for q in ('0.1', '0.5', '0.9')
            }
        if 'scores' in outputs and rng.random() > 0.2:
            fields['scores'] = tuple(
                round(rng.random(), 6) for _ in range(rng.randint(0, 3))
            )
        if 'embeddings' in outputs and rng.random() > 0.2:
            fields['embeddings'] = tuple(rng.uniform(-1, 1) for _ in range(3))
        if 'rankings' in outputs and rng.random() > 0.2:
            fields['rankings'] = tuple(
                {'candidate': c, 'score': round(rng.random(), 6)}
                for c in rng.sample([1, 2, 3, 'x'], rng.randint(0, 3))
            )
        items.append(PredictionItem(**fields))
    explanation = (
        {'by_forecast_step': {1: {'w': 0.5}}} if rng.random() < 0.2 else None
    )
    result = PredictionResult(
        model=model,
        request_id=f'req-{rng.randint(0, 999)}',
        items=items,
        metadata={'synthetic': True} if rng.random() < 0.5 else None,
        explanation=explanation,
        outputs=outputs,
        unscored_candidates=rng.sample([7, 8, 'z'], rng.randint(0, 2)),
    )
    if not items:
        result._names_its_rows = rng.random() < 0.7
        result._names_its_step = rng.random() < 0.3
    return result


def _outputs(rng: random.Random) -> tuple[str, ...]:
    return tuple(rng.sample(FIELDS, rng.randint(1, 4)))


@pytest.fixture(scope='module')
def results() -> list[Any]:
    rng = random.Random(RESULTS_SEED)
    return [
        _result(
            rng, rng.choice(['kumo-relational', 'kumo-tabular']), _outputs(rng)
        )
        for _ in range(RESULTS)
    ]


def _frame_or_error(result: Any) -> tuple[Any, str | None]:
    try:
        return result.to_pandas(), None
    except Exception as error:  # noqa: BLE001 - compared
        return None, f'{qualname(type(error))}: {error}'


def _host_frame(
    interp: PureInterpreter, result: Any
) -> tuple[Any, Any, str | None]:
    r"""The layout and frame the host gives, or why it gave none."""
    try:
        layout = call_flatten(interp, result)
        return layout, interp.call('frame', interp.host.frame, layout), None
    except Exception as error:  # noqa: BLE001 - compared
        return None, None, f'{qualname(type(error))}: {error}'


def _host_problems() -> list[str]:
    return [f'[{a}] {m}' for a, m in recorder().problems]


def test_flatten(results: list[Any]) -> None:
    r"""P85.results: ``frame(flatten(result))`` is today's ``to_pandas()``
    for every generated result, and the layout is a pyarrow table."""
    import pyarrow as pa

    interp = PureInterpreter()
    recorder().reset()
    problems = []
    compared = 0
    for i, result in enumerate(results):
        want, want_error = _frame_or_error(result)
        if want_error is not None:
            continue
        layout, got, error = _host_frame(interp, result)
        if error is not None:
            problems.append(f'result {i}: the host raised {short(error)}')
            continue
        if not isinstance(layout, (pa.Table, pa.RecordBatch)):
            problems.append(f'result {i}: the layout is not a pyarrow table')
        found = frame_difference(want, got)
        if found:
            problems.append(f'result {i}: {found}')
        compared += 1
    problems += _host_problems()
    assert compared >= RESULTS * 0.9, f'only {compared} results compared'
    assert not problems, '\n'.join(problems[:40])


def test_absent(results: list[Any]) -> None:
    r"""P85.results: ``absent(result, requested)`` is today's, a tuple."""
    rng = random.Random(RESULTS_SEED + 1)
    interp = PureInterpreter()
    recorder().reset()
    problems = []
    names = [*FIELDS, 'unknown_field', 'id', 'prediction']
    for i, result in enumerate(results):
        requested = [rng.choice(names) for _ in range(rng.randint(0, 6))]
        want = result.absent(requested)
        try:
            got = call_absent(interp, result, requested)
        except Exception as error:  # noqa: BLE001 - reported
            problems.append(f'result {i}: the host raised {short(str(error))}')
            continue
        if not isinstance(got, tuple) or list(got) != list(want):
            problems.append(f'result {i}: {got!r} instead of {want!r}')
    problems += _host_problems()
    assert not problems, '\n'.join(problems[:40])


def _attributes(result: Any) -> dict[str, Any]:
    from _parity.runner import _text_keys
    from _parity.values import encode

    return {
        'model': result.model,
        'request_id': result.request_id,
        'metadata': encode(_text_keys(dict(result.metadata))),
        'outputs': list(result.outputs),
        'unscored_candidates': encode(list(result.unscored_candidates)),
        'explanation': encode(_text_keys(result.explanation)),
    }


def test_merge(results: list[Any]) -> None:
    r"""P85.results: ``merge(parts)`` gives today's merged result, or
    refuses as today does."""
    from sdm_client.result import PredictionResult

    rng = random.Random(RESULTS_SEED + 2)
    interp = PureInterpreter()
    recorder().reset()
    problems = []
    refused = 0
    by_shape: dict[tuple[str, tuple[str, ...]], list[Any]] = {}
    for result in results:
        by_shape.setdefault((result.model, result.outputs), []).append(result)
    alike = [group for group in by_shape.values() if len(group) > 1]
    for i in range(MERGES):
        if rng.random() < 0.8 and alike:
            group = rng.choice(alike)
            parts = rng.sample(group, rng.randint(1, min(4, len(group))))
        else:
            parts = rng.sample(results, rng.randint(0, 3))
        try:
            want = PredictionResult.merge(parts)
            want_error = None
        except Exception as error:  # noqa: BLE001 - compared
            want, want_error = None, error
        try:
            got = call_merge(interp, parts)
            got_error = None
        except Exception as error:  # noqa: BLE001 - compared
            got, got_error = None, error
        if want_error is not None:
            refused += 1
            same = got_error is not None and (
                qualname(type(got_error)),
                getattr(got_error, 'code', None),
                str(got_error),
            ) == (
                qualname(type(want_error)),
                getattr(want_error, 'code', None),
                str(want_error),
            )
            if not same:
                problems.append(
                    f'merge {i}: today refuses with {want_error!r}, the host '
                    f'gives {got_error!r}'
                )
            continue
        if got_error is not None:
            problems.append(f'merge {i}: the host raised {got_error!r}')
            continue
        found = json_diff(_attributes(want), _attributes(got), f'merge {i}')
        if found:
            problems.append(found)
        frame, frame_error = _frame_or_error(want)
        if frame_error is None:
            _, got_frame, error = _host_frame(interp, got)
            found = error or frame_difference(frame, got_frame)
            if found:
                problems.append(f'merge {i}: {short(found)}')
    problems += _host_problems()
    assert refused >= 10, f'only {refused} merges were refused'
    assert not problems, '\n'.join(problems[:40])


def _ranked_inputs() -> list[dict[str, Any]]:
    rng = random.Random(RANKED_SEED)
    out = []
    pool = [0.1, 0.25, 0.5, 0.5, 0.75, 0.9, 0.0, 1.0]
    for _ in range(RANKED):
        k = rng.randint(2, 7)
        kind = rng.choice(['int', 'str', 'mixed'])
        ints = rng.sample(range(-3, 30), k)
        strs = rng.sample(['a', 'b', 'c', 'é', '10', '2', 'zz', ' '], k)
        if kind == 'int':
            candidates: list[Any] = ints
        elif kind == 'str':
            candidates = strs
        else:
            candidates = [
                rng.choice([i, s]) for i, s in zip(ints, strs, strict=True)
            ]
            if len({repr(c) for c in candidates}) < k:
                candidates = ints
        rows = rng.randint(1, 5)
        scores = [
            [
                rng.choice(pool)
                if rng.random() < 0.6
                else round(rng.random(), 6)
                for _ in range(rows)
            ]
            for _ in candidates
        ]
        out.append(
            {
                'candidates': candidates,
                'scores': scores,
                'top': rng.choice([None, *range(1, k + 2)]),
            }
        )
    return out


def test_ranked() -> None:
    r"""P85.decompose: 3,000 generated rankings, ties and all, ordered and
    cut to top-k as today's ``_ranked`` does."""
    import _fixture_ops  # tests/parity, on the path _request sets

    module = recorder().module
    problems = []
    tied = 0
    for i, given in enumerate(_ranked_inputs()):
        if any(
            len(set(row)) < len(row)
            for row in zip(*given['scores'], strict=True)
        ):
            tied += 1
        try:
            want: Any = {
                'output': json.loads(
                    json.dumps(_fixture_ops.run('client.ranked', given))
                )
            }
        except _fixture_ops.FixtureError as error:
            want = {'error': error.described()}
        try:
            got = json.loads(
                module.request_fixture_op('client.ranked', json.dumps(given))
            )
        except Exception as error:  # noqa: BLE001 - reported
            problems.append(f'ranking {i}: raised {short(str(error))}')
            continue
        if 'output' in want:
            found = json_diff(want['output'], got.get('output'), f'ranking {i}')
        else:
            found = json_diff(
                {k: want['error'][k] for k in ('class', 'code', 'message')},
                {
                    k: (got.get('error') or {}).get(k)
                    for k in ('class', 'code', 'message')
                },
                f'ranking {i}',
            )
        if found:
            problems.append(f'{found} for {short(given)}')
    assert tied >= RANKED // 4, f'only {tied} rankings hold a tie'
    assert not problems, '\n'.join(problems[:40])
