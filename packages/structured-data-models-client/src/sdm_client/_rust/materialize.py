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
source comes here on the core route, in place of the relational
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

The core's ``materialize`` samples the source through the DuckDB provider
(:mod:`.providers.duckdb`) and lays out the payload the engine lays out;
this module hands it the description the core last answered of the
schema and the schema as it stands (from T1.24b, decision E67), which the
core reads together: what the schema says now, with what the description
said of each column that the schema does not hold (the type the source
handed over, whether it may be null, whether the source keys it); and the
entities to predict for as one Arrow column of the type pyarrow gives
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
any schema it does not sample yet (register entry P1B-3), and, from
T1.24b, a schema changed in a way no edit it takes says
(``schema_edited``), which this module no longer refuses itself.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from sdm_client._routing import RouteError
from sdm_client._rust import codec
from sdm_client._rust.providers.duckdb import DuckDBProvider
from sdm_client._rust.run import run
from sdm_client.errors import StructuredDataError

if TYPE_CHECKING:
    from predictive_query import ValidatedPredictiveQuery

    from sdm_client.materializers.relational import RelationalMaterializer
    from sdm_client.schema import Schema

#: The options the contract carries in fields of their own (from T1.28a
#: the seed, the statistic and the inference options among the call's
#: values). The core is told the names of the rest (decision E61).
TAKEN = frozenset(
    {
        'anchor_time',
        'random_seed',
        'prediction_statistic',
        'inference_config',
    }
)

#: The options whose values the core reads, as the caller gave them, in the
#: call's values (T1.28a): their types and bounds are the core's.
VALUED = frozenset(
    {
        'random_seed',
        'num_hops',
        'num_neighbors',
        'prediction_statistic',
        'inference_config',
    }
)


def description(schema: Schema) -> tuple[dict[str, Any], dict[str, Any]]:
    r"""The schema as the contract's ``Materialize`` and ``EditSchema`` take
    it: the ``SourceDescription`` the core last answered of it
    (``describe_source``'s, or the last edit's), and the schema as it
    stands (``HeldSchema``): each table by the name the schema holds it
    under, its source's name of the table, its columns' data types and
    meanings and what each meant before the schema joined on it, its key
    and time column, and the links. The core reads the two together
    (decision E67): what the schema says now, with what the description
    said of each column that the schema does not hold, and refuses by name
    a schema changed in a way it does not follow yet (``schema_edited``);
    this module checks none of it.

    Raises:
        RouteError: The schema was not described on the core, so what the
            source said of its columns is not known here.
    """
    from sdm_client.schema import _as_key_tuple

    described = getattr(schema, '_core_description', None)
    if described is None:
        raise RouteError(
            'materialize on the core needs a schema Schema.from_source '
            'described on the core'
        )
    tables = []
    for name, table in schema.tables.items():
        key = table.primary_key
        tables.append(
            {
                'name': name,
                'source': getattr(table, 'source_name', name),
                'columns': [
                    {
                        'name': column_name,
                        'dtype': column.dtype,
                        'stype': str(column.stype),
                        'stype_before_key': (
                            None
                            if column._stype_before_key is None
                            else str(column._stype_before_key)
                        ),
                    }
                    for column_name, column in table.columns.items()
                ],
                'primary_key': [] if key is None else list(_as_key_tuple(key)),
                'time_column': table.time_column,
            }
        )
    links = [
        [
            {'table': link.source_table, 'column': link.source_column},
            {'table': link.target_table, 'column': link.target_column},
        ]
        for link in schema.relationships
    ]
    return described, {'tables': tables, 'links': links}


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


def _indices(indices: Sequence[Any]) -> Any:
    r"""The entities to predict for as one Arrow column, of the type
    pyarrow infers from the values themselves (``pa.array``); where it
    infers none, as when numbers and text are mixed or an integer is past
    int64, each value as its text (``str``; a boolean as ``true`` or
    ``false``, as Arrow writes one), a missing one (``None``, ``pd.NA``,
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


def _text(value: Any) -> str:
    r"""A value as its text, a boolean as Arrow writes one and a float as
    the core writes a float entity (``sdm_engine::keys``'s ``Entity::Float``,
    Rust's ``{}`` of an ``f64``), so the core reads the same number back:
    the shortest digits that read back as it, a tie between two such digit
    strings going away from zero (Rust's choice, where ``repr`` takes the
    even one), with no exponent and a whole one with no point (``1e16`` is
    ``10000000000000000``, ``1.0`` is ``1``, ``1.5e-07`` is
    ``0.00000015``). Exact arithmetic, never a decimal context."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return 'true' if value else 'false'
    if not isinstance(value, float) or not math.isfinite(value):
        return str(value)  # an infinity is 'inf' or '-inf', as Rust's
    from decimal import Decimal
    from fractions import Fraction

    sign, digits, exponent = Decimal(repr(float(value))).as_tuple()
    assert isinstance(exponent, int)
    whole = int(''.join(map(str, digits)))
    unit = Fraction(10) ** exponent
    if 2 * (abs(Fraction(value)) - whole * unit) == unit and float(
        (whole + 1) * unit
    ) == abs(value):
        whole += 1  # a tie: Rust's shortest digits go away from zero
    text = str(whole)
    if whole == 0:
        exponent = 0
    else:
        stripped = text.rstrip('0')
        exponent += len(text) - len(stripped)
        text = stripped
    if exponent >= 0:
        written = text + '0' * exponent
    elif -exponent < len(text):
        written = f'{text[:exponent]}.{text[exponent:]}'
    else:
        written = '0.' + '0' * (-exponent - len(text)) + text
    return ('-' if sign else '') + written


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
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    r"""The payloads of the request, gathered on the core, and the request
    state's version, which the request keeps for predict on the core.

    Raises:
        StructuredDataError: The core's refusal, as :mod:`.errors` maps it,
            ``UNSUPPORTED_REQUEST`` naming the rule for what it does not
            materialize yet; and the client's own checks of the call's
            types.
    """
    schema = materializer.schema
    anchor = options.get('anchor_time')
    described, held = description(schema)
    operation = {
        'op': 'materialize',
        'provider': {'provider': 'sql', 'dialect': 'duckdb'},
        'schema': 1,
        'held': 4,
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
    }
    attachments = {
        1: codec.attachment(described),
        2: codec.attachment(query.to_string()),
        3: _indices(indices),
        4: codec.attachment(held),
        # Each as the caller gave it, a value JSON cannot hold refused as
        # the client's transport refuses it; a context size of none is none.
        5: codec.attachment(
            {
                'num_context_samples': num_context_samples,
                **{
                    name: options[name]
                    for name in sorted(set(options) & VALUED)
                },
            }
        ),
    }
    source: Any = schema.source
    # The sampler state the core's last run on this materializer left,
    # handed back, so a run skips what the reference's sampler reads once
    # (decision E52); the run leaves its own in its place, done or failed.
    finished = run(
        operation,
        attachments,
        provider=DuckDBProvider(source.connection),
        state=materializer._core_sampler_state,
    )
    state = finished.result['state']
    payload = json.loads(finished.take(state['payload']))
    finished.take(state['query'])
    return (payload,), {'version': state['version']}
