# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Bridge code: from T1.32a it holds no rule of the request; the entities'
# cast to the key's type, its last host-held rule (design, "Host-held
# semantics"), is the core's. It passes the call's values and the source
# to the core, and is deleted in Phase 5 with the old route.
# Frozen meanwhile (E50, scripts/rust_core/bridge_freeze.py).

r"""``materialize`` on the core: the request (T1.11c, T1.28a).

``PredictiveQuery.materialize`` with ``indices`` over the schema of a DuckDB
or, from T2.2, a SQLite source comes here on the core route, in place of the relational
materializer's own rules and gathering (``RelationalMaterializer
.materialize``), once the client has checked the option names. From T1.28a the core holds the
request's rules (design, "Host-held semantics"): the context size's bound
and the run mode it selects, the entity count, the subgraph's and the
payload's limits, the empty and reserved-name refusals, the statistic,
checked and recorded, the inference options, the context that answers the
question taken out, and the notices of what the request leaves out, each
given as a code and fields this client words (:mod:`.errors`). This module
passes the call's values as the caller gave them, each refused by the core
where the client would refuse its type, and hands back the payload, from
which ``RelationalMaterializer.materialize`` builds the ``Request``.

The core's ``materialize`` samples the source through its backend's
provider (:mod:`.providers`) and lays out the payload the engine lays out;
this module hands it the schema as it stands, as plain data, each column
with what the source said of it where the core described it (T3.16; the
core reads from the source the facts of a column that carries none); and
the entities to predict for as one Arrow column of the type pyarrow gives
their values, which the core casts to the key's type (T1.32a). From T1.21d the core holds that schema to the reference's
consistency check itself, marking the columns it joins on again from what
each column means and meant before it was joined on (decision E55); this
module checks none of it. Nothing here imports the relational engine.

As built at T1.11c the core materializes a regression over a windowed
``SUM``, with the sampling options' defaults or the anchor time and seed
given; from T1.24 it samples tables named otherwise than the source names
them, by their source names. Every option of the call passes through to
the core (decision E61): the anchor time and seed in their fields, an
unseeded sample, an anchor at each entity's own time and a context size as
the contract says them, and the names of the rest; the core refuses with
``UNSUPPORTED_REQUEST``, naming the rule, each it does not take yet, and
and any schema it does not sample yet (register entry P1B-3).
"""

from __future__ import annotations

import datetime
import json
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from sdm_client._rust import codec, errors, providers, results
from sdm_client._rust.results import CoreQuery, CoreRequest, QueryFacts
from sdm_client._rust.run import run
from sdm_client._rust.spec import held_schema
from sdm_client.errors import StructuredDataError

if TYPE_CHECKING:
    from predictive_query import ValidatedPredictiveQuery

    from sdm_client.materializers.relational import RelationalMaterializer

__all__ = ['CoreQuery', 'gather']

#: The options the contract carries in fields of their own (from T1.28a
#: the seed, the statistic and the inference options among the call's
#: values; from T2.9, contract v27, ``lag_timesteps`` among them, and from
#: T2.11 the sampling options). The core is told the names of the rest
#: (decision E61).
TAKEN = frozenset(
    {
        'anchor_time',
        'random_seed',
        'prediction_statistic',
        'inference_config',
        'top_k',
        'rows_per_call',
        'lag_timesteps',
        'context_anchor_time',
        'exclude_cols_dict',
        'max_pq_iterations',
        'num_hops',
        'num_neighbors',
        'return_embeddings',
        'use_prediction_time',
    }
)

#: The options whose values the core reads, as the caller gave them, in the
#: call's values (T1.28a; ``lag_timesteps`` from T2.9, the sampling options
#: from T2.11): their types and bounds are the core's. ``context_anchor_time``
#: is written as the anchor time is (:func:`_context_anchor`).
VALUED = frozenset(
    {
        'random_seed',
        'num_hops',
        'num_neighbors',
        'prediction_statistic',
        'inference_config',
        'top_k',
        'rows_per_call',
        'lag_timesteps',
        'exclude_cols_dict',
        'max_pq_iterations',
        'return_embeddings',
        'use_prediction_time',
    }
)


def _anchor(value: Any) -> dict[str, Any] | None:
    r"""The anchor time as the contract's ``ts`` scalar, in nanoseconds; a
    zoned one as its UTC instant with its zone, which the core reads at its
    UTC wall clock, as the engine reads it (T1.25). The literal ``'entity'``
    is carried as ``anchor_entity``, beside it.
    """
    import pandas as pd

    if value is None or (isinstance(value, str) and value == 'entity'):
        return None
    if not isinstance(value, pd.Timestamp):
        raise StructuredDataError(
            f"'anchor_time' must be a pandas.Timestamp or the literal "
            f"'entity' (got {type(value).__name__})",
            code='INVALID_REQUEST',
        )
    zone = None if value.tzinfo is None else str(value.tzinfo)
    return codec.timestamp(int(value.as_unit('ns').value), 'ns', zone)


def _context_anchor(options: dict[str, Any]) -> dict[str, Any]:
    r"""``context_anchor_time`` among the call's values, where the call names
    it (T2.11): a ``pandas.Timestamp`` as the anchor time is written, the
    literal ``'entity'`` and none as they are, and any other value as the
    text ``unreadable``, which the core refuses as the engine refuses it,
    where the engine refuses it."""
    import pandas as pd

    if 'context_anchor_time' not in options:
        return {}
    value = options['context_anchor_time']
    if isinstance(value, pd.Timestamp):
        return {'context_anchor_time': _anchor(value)}
    if value is None or (isinstance(value, str) and value == 'entity'):
        return {'context_anchor_time': value}
    return {'context_anchor_time': 'unreadable'}


def _indices(indices: Sequence[Any]) -> Any:
    r"""The entities to predict for as one Arrow column, of the type
    pyarrow infers from the values themselves (``pa.array``); where it
    infers none, as when numbers and text are mixed or an integer is past
    int64, each value as its text (``_text``: ``str``, a boolean as
    ``true`` or ``false``, as Arrow writes one, and a float as the core
    writes a float entity), a missing one (``None``, ``pd.NA``,
    ``pd.NaT`` or a float NaN) null, and missing values alone a column of
    Arrow's null type. A float NaN pyarrow infers stays a NaN, which the
    core reads as missing, as it reads this null. The core casts the
    column to the key's type under its own rule table, and refuses an
    entity that does not cast (T1.32a); this host decides nothing of it."""
    import pyarrow as pa

    values = list(indices)
    try:
        column = pa.array(values)
    except (pa.ArrowException, OverflowError, TypeError, ValueError):
        if all(_missing(value) for value in values):
            column = pa.nulls(len(values))
        else:
            column = pa.array(
                [None if _missing(value) else _text(value) for value in values],
                pa.string(),
            )
    return pa.table({'indices': column})


def _candidates(candidates: Sequence[Any]) -> list[Any]:
    r"""A ranking's candidates as the contract's given candidates (T2.8),
    each as the caller gave it. Which candidates repeat is Python's: a
    candidate equal to one given before it, by
    Python's own ``==`` and ``hash`` as the reference's ``dict.fromkeys``
    keeps the first of each, is ``repeat``, which the core leaves out
    (T2.14b fix 7); the core holds each other as given.
    One Python cannot hash, which ``dict.fromkeys`` fails on, is
    ``unreadable``, the contract's word for a value of the host's no
    contract type holds (decision E101), which the core refuses by name.
    Each other is read as the Python value it holds (a NumPy value's
    ``.item()``): a missing one (``None``, a NaN, ``pd.NA``, ``pd.NaT``)
    null; a boolean; an integer as an ``i64``, or past it as its digits; a
    finite decimal as its digits in fixed point, exactly
    (``format(value, 'f')``); a float as the double it is; text as itself;
    and any other value, a date, a timestamp or an infinite decimal among
    them, as the text the reference writes into its question (``str``),
    spelled, which the core holds equal to no other candidate. What each
    means as a candidate is the core's."""
    import contextlib
    from decimal import Decimal

    from sdm_client.wire.dtypes import _is_missing

    written: list[Any] = []
    # The candidates kept as dict.fromkeys keeps them, by Python's own ==
    # and hash: each later one equal to one given before it is a repeat
    # (T2.14b fix 7).
    kept: dict[Any, None] = {}
    # The kinds and texts of the typed candidates sent.
    spoken: set[tuple[str, str]] = set()
    for value in candidates:
        try:
            repeat = value in kept
        except TypeError:
            written.append(codec.UNREADABLE)
            continue
        if repeat:
            written.append('repeat')
            continue
        kept[value] = None
        if hasattr(value, 'item'):
            with contextlib.suppress(AttributeError, ValueError):
                value = value.item()
        if _is_missing(value):
            written.append({'value': {'t': 'null'}})
        elif isinstance(value, bool):
            written.append({'value': {'t': 'bool', 'v': value}})
        elif isinstance(value, int):
            written.append(
                {'value': codec.integer('i64', value)}
                if -(2**63) <= value < 2**63
                else {'integer': str(value)}
            )
        elif isinstance(value, Decimal) and value.is_finite():
            written.append({'decimal': format(value, 'f')})
        elif isinstance(value, float):
            written.append({'value': {'t': 'f64', 'v': repr(value)}})
        elif type(value) is str:
            written.append({'value': {'t': 'utf8', 'v': value}})
        elif isinstance(value, (datetime.date, datetime.time)):
            # A date, a time of day or a timestamp (pandas' among them), with
            # its kind; its text is the question's. The core holds two of
            # one kind and text one candidate, so one Python holds distinct
            # from an earlier one of the same kind and text (one wall time
            # in two zones) is sent spelled, which the core merges with
            # none (T2.14b fix 8).
            typed = (_kind(value), str(value))
            if typed in spoken:
                written.append({'spelled': typed[1]})
            else:
                spoken.add(typed)
                written.append({'typed': {'text': typed[1], 'type': typed[0]}})
        else:
            written.append({'spelled': str(value)})
    return written


def _kind(value: Any) -> str:
    r"""The kind of a date, a time of day or a timestamp, as the contract's
    typed candidate names it: ``datetime_zoned`` for a timestamp with a
    zone, ``datetime`` for one without (a ``pandas.Timestamp`` among
    them), ``date`` and ``time``."""
    if isinstance(value, datetime.datetime):
        return 'datetime' if value.utcoffset() is None else 'datetime_zoned'
    if isinstance(value, datetime.date):
        return 'date'
    return 'time'


def _text(value: Any) -> str:
    r"""A value as its text, a boolean as Arrow writes one and a float as
    the core writes a float entity (``sdm_engine::keys``'s ``float_text``,
    T1.36 fix 4), which is the reference's text of it: pyarrow's cast of
    the float to text, at its own width (a NumPy ``float32`` as a 32-bit
    float; a NumPy ``float64`` is a ``float``), so the core reads back the
    number the host was given. That is the shortest digits that read back
    as it, written plainly from ``1e-6`` to below ``1e10`` and with an
    exponent outside (``1e16`` is ``1e+16``, ``1.5e-07`` is ``1.5e-7``,
    ``1.0`` is ``1``, an infinity ``inf``)."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return 'true' if value else 'false'
    if isinstance(value, (float, np.float32, np.float16)):
        import pyarrow as pa

        return str(pa.array([value]).cast(pa.string())[0].as_py())
    return str(value)


def _missing(value: Any) -> bool:
    r"""Whether ``value`` is a missing value of Python, pandas or a float."""
    import pandas as pd

    return (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or (isinstance(value, float) and math.isnan(value))
    )


def gather(
    materializer: RelationalMaterializer,
    query: ValidatedPredictiveQuery,
    indices: Sequence[Any],
    num_context_samples: int | None,
    options: dict[str, Any],
    candidates: Sequence[Any] | None = None,
) -> tuple[tuple[dict[str, Any], ...], QueryFacts, CoreRequest]:
    r"""The payloads of the request, gathered on the core; what the core
    resolved of the query; and what the request keeps for predict on the
    core (:class:`~sdm_client._rust.results.CoreRequest`), with the parts of
    a request the core gathered in parts.

    The schema is handed over as it stands, as plain data
    (:func:`~sdm_client._rust.spec.held_schema`, T3.16): the core reads
    from the source what it needs of a column that carries no facts.

    ``candidates``, a ranking's, are handed over as the caller named them,
    each as the contract's given candidate of its own type
    (:func:`_candidates`, T2.8): the core asks one question per distinct
    candidate and refuses them on any other question.

    A request of more entities than one call scores is gathered by the core
    as several calls (``main``'s 1fc5a686, T2.8c): its payloads are each
    call's, in order, and the state keeps how many rows each later call
    scores. The most entities and calls one ``materialize`` holds are this
    client's (``MOST_ENTITIES``, ``MOST_CALLS``), which the core holds the
    call to.

    Raises:
        StructuredDataError: The core's refusal, as :mod:`.errors` maps it,
            ``UNSUPPORTED_REQUEST`` naming the rule for what it does not
            materialize yet; and the client's own checks of the call's
            types.
    """
    from sdm_client import decompose

    schema = materializer.schema
    anchor = options.get('anchor_time')
    choice, provider = providers.for_source(schema.source)
    operation = {
        'op': 'materialize',
        'provider': choice,
        'schema': 1,
        'query': 2,
        'indices': 3,
        'anchor_time': _anchor(options.get('anchor_time')),
        # The seed given, as its field, beside the call's values, which name
        # a seed of another type or none (an unseeded sample) for the core.
        'seed': codec.seed(options.get('random_seed')),
        'anchor_entity': isinstance(anchor, str) and anchor == 'entity',
        # The call's values the core reads (T1.28a), the seed among them: a
        # seed given as None asks for an unseeded sample, a new one each
        # call, never the default; the core says whether it takes it.
        'values': 5,
        'options': sorted(set(options) - TAKEN),
        # How many parts the caller lets a question be asked as, which the
        # core holds a forecast's steps and a ranking's candidates to
        # (T1.36 fix 21, T2.8); only the core knows which a query is.
        'most_parts': max(0, min(int(decompose.MOST_PARTS), 2**32 - 1)),
    }
    attachments = {
        1: codec.attachment(held_schema(schema)),
        2: codec.attachment(query.to_string()),
        3: _indices(indices),
        # Each as the caller gave it, a value JSON cannot hold refused as
        # the client's transport refuses it; a context size of none is none.
        5: codec.attachment(
            named=True,
            value={
                'num_context_samples': num_context_samples,
                **{
                    name: options[name]
                    for name in sorted(set(options) & VALUED)
                },
                **_context_anchor(options),
            },
        ),
    }
    # The ceilings on what one materialize holds, which the caller may set
    # as it sets the most steps of a forecast (T2.8c).
    from sdm_client.materializers import relational

    operation['most_entities'] = max(
        0, min(int(relational.MOST_ENTITIES), 2**64 - 1)
    )
    operation['most_calls'] = max(0, min(int(relational.MOST_CALLS), 2**32 - 1))
    if candidates is not None:
        operation['candidates'] = 6
        attachments[6] = codec.attachment(_candidates(candidates))
    # The sampler state the core's last run on this materializer left,
    # handed back, so a run skips what the reference's sampler reads once
    # (decision E52); the run leaves its own in its place, done or failed.
    # A semantic model's, as the core's refusal of a table or column the
    # model no longer gives names it (T3.5, plan B).
    with errors.holding(**errors.source_values(schema.source)):
        finished = run(
            operation,
            attachments,
            provider=provider,
            state=materializer._core_sampler_state,
        )
    gathered = results.expect(finished.result, results.Gathered)
    state = gathered.state
    if gathered.query is None:
        raise StructuredDataError(
            'the core gathered a query and said nothing of it',
            code='INTERNAL_ERROR',
        )
    payload = json.loads(finished.take(state.payload))
    query = json.loads(finished.take(state.query))
    payloads = [payload]
    batches = None
    if state.batches is not None:
        # A request asked in several calls (T2.8c): each later call's
        # payload, in order, and the rows the core gathered it to score.
        for batch in state.batches:
            payloads.append(json.loads(finished.take(batch.payload)))
        batches = tuple(batch.rows for batch in state.batches)
    parts = None
    if state.ranking is not None:
        # A ranking, which the core asked as one binary question per
        # candidate (T2.8): the first candidate's position, query and name
        # (the JSON text of its value, as the result names it), each later
        # one's position among the candidates given, name, payload and
        # query, and how many candidates a row keeps.
        parts = results.CoreParts(
            kind='ranking',
            query=query,
            position=state.ranking.candidate,
            name=state.ranking.value,
            top_k=state.ranking.top_k,
            later=tuple(
                results.CorePart(
                    position=part.candidate,
                    name=part.value,
                    payload=json.loads(finished.take(part.payload)),
                    query=json.loads(finished.take(part.query)),
                )
                for part in state.ranking.parts
            ),
        )
    elif state.forecast is not None:
        # A forecast, which the core asked as one regression per step (T1.36
        # fix 21): step 1's query, and each later step's number, payload
        # and query, from which the client keeps one request per step.
        parts = results.CoreParts(
            kind='forecast',
            query=query,
            position=1,
            name=None,
            top_k=None,
            later=tuple(
                results.CorePart(
                    position=step.step,
                    name=None,
                    payload=json.loads(finished.take(step.payload)),
                    query=json.loads(finished.take(step.query)),
                )
                for step in state.forecast
            ),
        )
    return (
        tuple(payloads),
        gathered.query,
        results.CoreRequest(
            version=state.version, batches=batches, parts=parts
        ),
    )
