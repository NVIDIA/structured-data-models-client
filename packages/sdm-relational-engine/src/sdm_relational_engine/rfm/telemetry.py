# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What a prediction has to say about itself afterwards.

A prediction that comes back wrong days later can only be explained if the run
that produced it can be reconstructed: which graph, under which engine, for which
query, at which anchor. Predictions are not deterministic and the data underneath
them moves, so re-running the question does not reveal what happened the first
time.

One record is emitted per prediction, whether it returned, refused, or raised. It
names the run rather than describing it: a graph fingerprint, versions, counts and
timings. No entity identifiers and no cell values, since both are the caller's
data. It goes to the ``sdm_relational_engine`` logger rather than to the progress
display, which belongs to whoever is watching the call rather than to whoever has
to explain it later.
"""

import json
import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger('sdm_relational_engine')


# What a caller further up knows that the call doing the work does not. The
# query-level entry point sees the parsed query and the entity list; the task
# entry point, which is where the request is actually made and where every
# caller ends up, sees neither.
_pending: ContextVar[dict[str, Any] | None] = ContextVar(
    'sdm_relational_engine.telemetry._pending', default=None
)


@contextmanager
def describing(**fields: Any) -> Iterator[None]:
    r"""Lend detail to the record the work below is about to write."""
    token = _pending.set(fields)
    try:
        yield
    finally:
        _pending.reset(token)


def pending() -> dict[str, Any]:
    r"""What a caller above lent, or nothing when it was called directly."""
    return dict(_pending.get() or {})


# Bumped when a field below is renamed or removed, so a reader can tell which
# shape it is looking at rather than discovering the change by breaking.
SCHEMA_VERSION = 1


@dataclass
class PredictionRecord:
    r"""One prediction, as it will be read back when something looks wrong."""

    schema_version: int = SCHEMA_VERSION
    graph_fingerprint: str = ''
    engine_version: str = ''
    query_type: str = ''
    entities: int = 0
    run_mode: str = ''
    batch_size: str = ''
    num_hops: int = 0
    explain: bool = False
    anchor_time: str = ''
    task_type: str = ''
    entity_table: str = ''
    prediction_id: str = ''

    outcome: str = 'raised'
    seconds: float = 0.0
    error: str = ''


def emit(record: PredictionRecord) -> None:
    r"""Write one prediction to the log, never failing the call it describes.

    The fallback is guarded too. A handler that raises raises for both calls,
    and an unguarded second one turns a prediction that succeeded into
    whatever the logger threw, which is the one thing this promises not to do.
    """
    try:
        logger.info('rfm.predict %s', json.dumps(asdict(record), default=str))
    except Exception:
        try:
            logger.debug(
                'Prediction record could not be emitted', exc_info=True
            )
        except Exception:
            pass


# Everything from the first quote in a message to the last, and a bare list of
# entity ids. What an engine or a NIM quotes back when it rejects something is
# the caller's data, and it arrives here through str(error) rather than through
# any field this record chose to carry.
#
# The span is deliberately blunt. Matching each value separately needs the
# message to say where one ends, and it does not: a backslash before a closing
# quote is either an escaped quote inside the value or the end of a value that
# ends in a backslash, and an apostrophe in prose is indistinguishable from an
# opening quote. Every rule tried for those cases left some value in the log --
# reading the escape one way leaks the rest of that value, the other way leaks
# the next one, and requiring a non-word character before an opening quote
# leaks `entity'alice@example.com'`. Taking the whole span cannot: nothing
# between the outer quotes survives, whatever the quoting meant.
#
# The cost is that a message with several values keeps only what is outside
# them, so two failures differing only in wording between two values read the
# same. The type is still there, and so is every other field on the record.
# Every character something upstream might quote with. Backticks are not
# decoration here: the Databricks backend quotes identifiers with them, so an
# error echoing a column name arrives wrapped in them, and the apostrophe is
# what Python itself quotes with.
#
# The rest are every character Unicode names as a quotation mark or a corner
# bracket. Written out rather than derived at import, because deriving it
# means naming every character in Unicode and costs about a tenth of a second
# every time this module loads. A test re-derives the set and fails if this
# literal no longer matches, so a form added by a newer Unicode cannot go in
# unnoticed -- which is the whole reason it is not a hand-kept list.
#
# Categories were the obvious alternative and are wrong twice: `Pi`/`Pf` miss
# the low-9 forms, which Unicode classes as opening punctuation, and adding
# `Ps`/`Pe` to reach them pulls in `(`, `[` and `{`, so a message naming a
# table in parentheses would lose everything after it.
#
# Opening and closing need not match, since the rule takes the span rather
# than pairing them.
_QUOTES = (
    '"\'`\u00ab\u00bb\u2018\u2019\u201a\u201b\u201c\u201d\u201e'
    '\u201f\u2039\u203a\u275b\u275c\u275d\u275e\u275f\u2760'
    '\u276e\u276f\u2e42\u300c\u300d\u300e\u300f\u301d\u301e'
    '\u301f\ufe41\ufe42\ufe43\ufe44\uff02\uff62\uff63'
    '\U0001f676\U0001f677\U0001f678\U000e0022'
)
_QUOTE_CLASS = f'[{re.escape(_QUOTES)}]'
_QUOTED_SPAN = re.compile(
    f'{_QUOTE_CLASS}.*{_QUOTE_CLASS}|{_QUOTE_CLASS}.*', re.DOTALL
)
_ID_LIST = re.compile(r'\[[^\[\]]*\]|\((?:\s*[-\w.]+\s*,)[^()]*\)')
_MARK = "'?'"


def describe_error(error: BaseException) -> str:
    r"""An error named by its type, with the values it quoted back removed.

    The sampler raises errors naming the entity it could not find, and a NIM
    can echo the request body. Both would land in this record verbatim, which
    is exactly the caller data every other field here is chosen to avoid.

    The type is always kept, since that is what a reader branches on. The
    message keeps whatever falls outside its quoted span and loses the rest.

    What this does not catch is a value that arrives unquoted, as in
    "entity alice@example.com not found". Nothing marks it as data, and
    redacting on shape would take ordinary words with it. The record's own
    fields are chosen so none of them carries caller data; this is the last
    line for the one field that quotes an upstream message back.

    Nothing here may raise. This runs inside the `except` that re-raises the
    caller's error, so an exception thrown while describing one would replace
    the failure the caller is trying to read.
    """
    name = type(error).__name__
    try:
        message = _QUOTED_SPAN.sub(_MARK, str(error))
        message = _ID_LIST.sub('(values)', message)
        return f'{name}: {message}'
    except Exception:
        return f'{name}: <message could not be read>'


@contextmanager
def recorded(record: PredictionRecord) -> Iterator[PredictionRecord]:
    r"""Time a prediction and emit its record however it ends.

    A run that raised is the one most worth having a record of, so the record is
    written on the way out rather than on success.
    """
    started = time.perf_counter()
    try:
        yield record
        if record.outcome == 'raised':
            record.outcome = 'returned'
    except BaseException as error:
        record.error = describe_error(error)
        raise
    finally:
        record.seconds = round(time.perf_counter() - started, 3)
        emit(record)
