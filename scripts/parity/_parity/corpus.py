# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The corpus as a whole: what it holds, recording it, and replaying it."""

from __future__ import annotations

import json
import shutil
import tempfile
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from _parity import REPO, TRANSCRIPTS
from _parity.diffing import first_difference, render
from _parity.nim import ReplayingSession, SyntheticSession
from _parity.runner import CannotRecord, record, run
from _parity.scenarios import Scenario, by_name, every
from _parity.tape import ReplayingTape
from _parity.transcript import (
    TRANSCRIPT_FILE,
    load,
    read_arrow,
    save,
    scenario_dir,
)
from _parity.values import same

RECORDED = 'recorded'
PENDING_LIVE = 'pending-live'
#: An offline scenario with no transcript yet: the corpus is incomplete.
UNRECORDED = 'unrecorded'


def stored(corpus: Path = TRANSCRIPTS) -> dict[str, Path]:
    r"""Every transcript under ``corpus``, by scenario name."""
    out = {}
    for path in sorted(corpus.rglob('*.json')):
        name = load(path).get('name')
        if isinstance(name, str):
            out[name] = path
    return out


def listing(corpus: Path = TRANSCRIPTS) -> list[dict[str, Any]]:
    on_disk = stored(corpus)
    entries = []
    for scenario in every():
        path = on_disk.get(scenario.name)
        if path is not None:
            status = RECORDED
        elif scenario.live:
            status = PENDING_LIVE
        else:
            status = UNRECORDED
        entries.append(
            {
                'name': scenario.name,
                'backend': scenario.backend,
                'task_kind': scenario.task_kind,
                'status': status,
                'transcript': _relative(path) if path else None,
            }
        )
    return entries


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def select(names: Iterable[str] | None, live: bool) -> list[Scenario]:
    known = by_name()
    if names:
        missing = [name for name in names if name not in known]
        if missing:
            raise CannotRecord(f'no scenario named {", ".join(missing)}')
        return [known[name] for name in names]
    return [s for s in known.values() if s.live == live]


def record_one(scenario: Scenario, corpus: Path) -> Path:
    r"""Record ``scenario`` into ``corpus``, all or nothing.

    The files are written to a scratch directory first, and scrubbed there
    for a live scenario. What would be stored is then replayed, and stored
    only if it reproduces, so a recording that scrubbing broke, or that does
    not replay for any other reason, never reaches ``corpus``. A failure
    leaves ``corpus`` as it was.

    Raises:
        CannotRecord: The scenario cannot be recorded here, or its recording
            does not replay.
    """
    transcript, outcome = record(scenario.skeleton())
    with tempfile.TemporaryDirectory(prefix='sdm-parity-record-') as scratch:
        staged = Path(scratch) / 'staged'
        save(staged, transcript, outcome.tape.results, outcome.result)
        if scenario.live:
            from _parity.scrubber import scrub_tree

            scrubbed = Path(scratch) / 'scrubbed'
            scrub_tree(staged, scrubbed)
            staged = scrubbed
        reason = check(staged / TRANSCRIPT_FILE)
        if reason is not None:
            raise CannotRecord(
                f'the recording of {scenario.name} does not replay, so it is '
                f'not stored: {one_line(reason)}'
            )
        target = scenario_dir(corpus, scenario.name)
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staged, target)
    return target / TRANSCRIPT_FILE


def reanswer(path: Path) -> Path:
    r"""Answer a stored transcript's model calls again, from today's
    stand-in, without a database; all or nothing.

    The run replays the stored steps, and only what follows from the model's
    answers is written again: ``responses``, the decoded result, what the
    result holds beyond its frame, the warnings and any error. The steps and
    the payloads must come out as stored. This is how a live recording takes
    a change to the stand-in without credentials; a live recording is not
    scrubbed again, because nothing a warehouse answered changes.

    Raises:
        CannotRecord: The run took other steps or sent other payloads than
            the stored ones, or what would be stored does not replay.
    """
    transcript = load(path)
    root = path.parent
    tape = ReplayingTape(json.loads(json.dumps(transcript['steps'])), root)
    session = SyntheticSession(transcript.get('stand_in'))
    outcome = run(transcript, tape, session=session)
    problem = tape.divergence or outcome.tape.leftover()
    if problem is None:
        found = first_difference(transcript['payloads'], outcome.payloads)
        if found is not None:
            problem = f'payloads differ at payloads{_suffix(found[0])}'
    if problem is not None:
        raise CannotRecord(
            f'{transcript["name"]} does not run as stored: {one_line(str(problem))}'
        )
    renewed = {
        key: value
        for key, value in transcript.items()
        if key not in ('responses', 'error', 'answer', 'result', 'warnings')
    }
    renewed['responses'] = outcome.responses
    if outcome.error is not None:
        renewed['error'] = outcome.error
    if outcome.answer:
        renewed['answer'] = outcome.answer
    renewed['warnings'] = outcome.warnings
    with tempfile.TemporaryDirectory(prefix='sdm-parity-reanswer-') as scratch:
        staged = Path(scratch) / 'staged'
        save(staged, renewed, {}, outcome.result)
        # The steps' files are kept byte for byte: what the warehouse
        # answered is not the stand-in's to change.
        for step in transcript['steps']:
            if step.get('result'):
                shutil.copy2(root / step['result'], staged / step['result'])
        reason = check(staged / TRANSCRIPT_FILE)
        if reason is not None:
            raise CannotRecord(
                f'the answered {transcript["name"]} does not replay, so it is '
                f'not stored: {one_line(reason)}'
            )
        shutil.rmtree(root)
        shutil.copytree(staged, root)
    return path


def check(path: Path) -> str | None:
    r"""Replay one stored transcript; ``None`` if it reproduces, else why not.

    The model is answered from the stored ``responses``, never recomputed.

    The reason names the first difference: a step (``step k``), a model
    request (``response k``), a payload path, the error the run ended in,
    the decoded result, what the result holds beyond its frame, or the
    warnings.
    """
    transcript = load(path)
    root = path.parent
    if not isinstance(transcript.get('responses'), list):
        return 'the recording stores no responses to answer the model from'
    tape = ReplayingTape(json.loads(json.dumps(transcript['steps'])), root)
    session = ReplayingSession(json.loads(json.dumps(transcript['responses'])))
    outcome = None
    error: BaseException | None = None
    try:
        outcome = run(transcript, tape, session=session)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as caught:  # noqa: BLE001 - reported, not raised
        error = caught
    # The first divergence wins, however the engine handled it: code that
    # catches errors around a probe may have turned it into something else.
    if tape.divergence is not None:
        return str(tape.divergence)
    if session.divergence is not None:
        return str(session.divergence)
    if error is not None:
        last = traceback.extract_tb(error.__traceback__)[-1:]
        where = f' at {last[0].filename}:{last[0].lineno}' if last else ''
        return f'the run raised {type(error).__name__}: {error}{where}'
    assert outcome is not None
    leftover = outcome.tape.leftover() or session.leftover()
    if leftover is not None:
        return str(leftover)
    found = first_difference(transcript['payloads'], outcome.payloads)
    if found is not None:
        at, what = found
        return f'payloads differ at payloads{_suffix(at)}: {what}'
    recorded_error = transcript.get('error')
    if recorded_error != outcome.error:
        if outcome.error is None:
            return f'the run ended without the recorded error {recorded_error}'
        if recorded_error is None:
            return f'the run ended in an error the recording lacks: {outcome.error}'
        found = first_difference(recorded_error, outcome.error)
        at, what = found if found else ([], 'differ')
        return f'the error differs at error{_suffix(at)}: {what}'
    reason = _compare_result(transcript, root, outcome.result)
    if reason:
        return reason
    if transcript.get('answer', {}) != outcome.answer:
        found = first_difference(transcript.get('answer', {}), outcome.answer)
        at, what = found if found else ([], 'differ')
        return f'the answer differs at answer{_suffix(at)}: {what}'
    if transcript.get('warnings', []) != outcome.warnings:
        found = first_difference(
            transcript.get('warnings', []), outcome.warnings
        )
        at, what = found if found else ([], 'differ')
        return f'warnings differ at warnings{_suffix(at)}: {what}'
    return None


def _suffix(path: list[Any]) -> str:
    text = render(path)
    if text == '<root>':
        return ''
    return text if text.startswith('[') else '.' + text


def _compare_result(
    transcript: dict[str, Any], root: Path, result: Any
) -> str | None:
    ref = transcript.get('result')
    if not isinstance(ref, str):
        return (
            None
            if result is None
            else 'the run decoded a result, and the recording holds none'
        )
    if result is None:
        return 'the run decoded no result, and the recording holds one'
    expected = read_arrow(root / ref)
    if not expected.schema.equals(result.schema):
        return (
            f'the decoded result differs: columns {expected.schema} became '
            f'{result.schema}'.replace('\n', ', ')
        )
    a, b = expected.to_pylist(), result.to_pylist()
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        if not same(x, y):
            column = next((k for k in x if not same(x[k], y.get(k))), '?')
            return (
                f'the decoded result differs at result[{index}].{column}: '
                f'{x.get(column)!r} became {y.get(column)!r}'
            )
    if len(a) != len(b):
        return f'the decoded result had {len(a)} rows and now has {len(b)}'
    return None


def one_line(text: str) -> str:
    return ' '.join(text.split())
