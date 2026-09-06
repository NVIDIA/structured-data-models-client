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
    r"""Write one prediction to the log, never failing the call it describes."""
    try:
        logger.info('rfm.predict %s', json.dumps(asdict(record), default=str))
    except Exception:
        logger.debug('Prediction record could not be emitted', exc_info=True)


# A quoted value, and a bare list of entity ids. What an engine or a NIM quotes
# back when it rejects something is the caller's data, and it arrives here
# through str(error) rather than through any field this record chose to carry.
_QUOTED = re.compile(r"""(['"])(?:(?!\1).)*\1""")
_ID_LIST = re.compile(r'\[[^\[\]]*\]|\((?:\s*[-\w.]+\s*,)[^()]*\)')


def describe_error(error: BaseException) -> str:
    r"""An error named by its type, with the values it quoted back removed.

    The sampler raises errors naming the entity it could not find, and a NIM
    can echo the request body. Both would land in this record verbatim, which
    is exactly the caller data every other field here is chosen to avoid.

    The type is always kept, since that is what a reader branches on. The
    message keeps its shape and loses its values.
    """
    message = _QUOTED.sub("'?'", str(error))
    message = _ID_LIST.sub('(values)', message)
    return f'{type(error).__name__}: {message}'


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
