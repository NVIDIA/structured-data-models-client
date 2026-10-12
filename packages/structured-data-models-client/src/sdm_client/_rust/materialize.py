# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Bridge code: the host-held rule here (design, "Host-held semantics"),
# the entities' cast to the key's type (T1.32a, checkpoint 4), is deleted
# in Phase 1 by that task.
# Frozen meanwhile (E50, scripts/rust_core/bridge_freeze.py).

r"""``materialize`` on the core: the request (T1.11c, T1.28a).

``PredictiveQuery.materialize`` with ``indices`` over the schema of a DuckDB
source comes here on the core route, in place of the relational
materializer's own (``RelationalMaterializer.materialize``), once the
client has checked the option names. From T1.28a the core holds the
request's rules (design, "Host-held semantics"): the context size's bound
and the run mode it selects, the entity count, the subgraph's and the
payload's limits, the empty and reserved-name refusals, the statistic,
checked and recorded, the inference options, the context that answers the
question taken out, and the notices of what the request leaves out, each
given as a code and fields this client words (:mod:`.errors`). This module
converts the call's values into the contract's fields, refusing a value of
a type the contract cannot carry, as the client refuses it, and builds the
``Request`` from the payload.

The core's ``materialize`` samples the source through the DuckDB provider
(:mod:`.providers.duckdb`) and lays out the payload the engine lays out;
this module hands it the description the core last answered of the
schema and the schema as it stands (from T1.24b, decision E67), which the
core reads together: what the schema says now, with what the description
said of each column that the schema does not hold (the type the source
handed over, whether it may be null, whether the source keys it); and the
entities to predict for as one Arrow column of the key's own type. From T1.21d the core holds that schema to the reference's
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
    from sdm_client.request import Request
    from sdm_client.schema import Schema

#: The options the contract carries in fields of their own (from T1.28a
#: the statistic and the inference options too). The core is told the
#: names of the rest (decision E61).
TAKEN = frozenset(
    {
        'anchor_time',
        'random_seed',
        'prediction_statistic',
        'inference_config',
    }
)

#: The integer types a key the core samples may have, as pyarrow names
#: them, by the contract's spelling.
_ARROW_TYPES = {
    'int8': 'int8',
    'int16': 'int16',
    'int32': 'int32',
    'int64': 'int64',
    'uint8': 'uint8',
    'uint16': 'uint16',
    'uint32': 'uint32',
    'uint64': 'uint64',
    'utf8': 'string',
}


def _not_taken(rule: str) -> StructuredDataError:
    return StructuredDataError(
        'materialize on the core does not take this yet',
        code='UNSUPPORTED_REQUEST',
        details={'operation': 'materialize', 'rule': rule},
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


def _indices(indices: Sequence[Any], schema: Schema, entity: str) -> Any:
    r"""The entities to predict for, as one Arrow column of the entity key's
    own type, cast as the engine casts them: by pandas, to the dtype its
    sampled key has (``pd.Series(indices, dtype=train_pkey.dtype)``, the
    key's Arrow type as a ``pd.ArrowDtype``), so text digits, booleans and
    integral floats cast as they do there; an entity that does not cast is
    the engine's refusal (``INVALID_REQUEST``). A host's conversion of its
    own values, which the core never sees."""
    import pandas as pd
    import pyarrow as pa

    described = schema._core_description
    key = schema[entity].primary_key
    received = next(
        (
            column['received']
            for table in described['tables']
            if table['name'] == entity
            for column in table['columns']
            if column['name'] == key
        ),
        None,
    )
    arrow = _ARROW_TYPES.get((received or {}).get('type', ''))
    if arrow is None:
        raise _not_taken('key_type')
    try:
        cast = pd.Series(
            list(indices), dtype=pd.ArrowDtype(pa.type_for_alias(arrow))
        )
        column = pa.array(cast)
    except (TypeError, ValueError):
        # The engine raises these as a ValueError naming the key, which the
        # client maps to INVALID_REQUEST.
        raise StructuredDataError(
            f"'indices' must hold values matching the primary key of "
            f"'{entity}', which is {arrow}",
            code='INVALID_REQUEST',
        ) from None
    except Exception as failure:  # noqa: BLE001 - every cast failure
        # Any other failure (an integer past the key's width overflows, a
        # date has no cast) escapes the engine's own handling, and the
        # client's guard of the engine reports it as INTERNAL_ERROR.
        raise StructuredDataError(
            f'the relational engine failed while building this request: '
            f'{type(failure).__name__}',
            code='INTERNAL_ERROR',
        ) from None
    return pa.table({'indices': column})


def _types(options: dict[str, Any]) -> None:
    r"""Refuses an option whose value is of a type the contract cannot
    carry, as the client refuses it (``INVALID_REQUEST``): a seed, a hop
    count or a statistic of another type, hops that are not a sequence, and
    inference options that are not a mapping, or hold a value JSON cannot.
    The Python API's argument types; what the values mean is the core's."""
    from numbers import Integral

    def whole(name: str) -> None:
        value = options.get(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral)
        ):
            raise StructuredDataError(
                f'{name} must be a whole number, got '
                f'{type(value).__name__} {value!r}',
                code='INVALID_REQUEST',
            )

    whole('random_seed')
    whole('num_hops')
    neighbors = options.get('num_neighbors')
    if neighbors is not None and (
        not isinstance(neighbors, Sequence)
        or isinstance(neighbors, (str, bytes))
    ):
        raise StructuredDataError(
            'num_neighbors must be a sequence of hop sizes',
            code='INVALID_REQUEST',
        )
    statistic = options.get('prediction_statistic')
    if 'prediction_statistic' in options and not isinstance(statistic, str):
        raise StructuredDataError(
            f'prediction_statistic is {statistic!r}, and a regression reports '
            f"'mean' or 'median' of its predicted distribution",
            code='INVALID_REQUEST',
        )


def _inference(given: Any) -> bytes | None:
    r"""The caller's inference options as the contract's attachment: a
    mapping's keys and values as JSON, which the core reads by the engine's
    rules; none for none. One of another type, which the engine takes as its
    own config object, is refused by name (register P1B-3); one holding a
    value JSON cannot hold, as the client's transport refuses it."""
    from collections.abc import Mapping

    if given is None:
        return None
    if not isinstance(given, Mapping):
        raise _not_taken('inference_config')
    try:
        return codec.attachment({str(k): v for k, v in given.items()})
    except (TypeError, ValueError, RecursionError):
        raise StructuredDataError(
            'inference_config holds a value that cannot be sent as JSON',
            code='INVALID_REQUEST',
        ) from None


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
    from predictive_query.validator.utils import table_name

    schema = materializer.schema
    _types(options)
    entity = table_name(query.entity_column)
    seed = options.get('random_seed')
    anchor = options.get('anchor_time')
    described, held = description(schema)
    config = _inference(options.get('inference_config'))
    operation = {
        'op': 'materialize',
        'provider': {'provider': 'sql', 'dialect': 'duckdb'},
        'schema': 1,
        'held': 4,
        'query': 2,
        'indices': 3,
        'anchor_time': _anchor(options.get('anchor_time')),
        'seed': None if seed is None else codec.seed(int(seed)),
        # A seed given as None asks for an unseeded sample, a new one each
        # call, never the default; the core says whether it takes it.
        'unseeded': 'random_seed' in options and seed is None,
        'anchor_entity': isinstance(anchor, str) and anchor == 'entity',
        # Validated as a whole number by the client, of any size; the core
        # bounds it (T1.28a).
        'num_context_samples': (
            None
            if num_context_samples is None
            else codec.seed(int(num_context_samples))
        ),
        # The hops the sampling options ask, which the core bounds before it
        # refuses the options by name: the count given, and how many hop
        # sizes the sequence holds.
        'num_hops': (
            None
            if options.get('num_hops') is None
            else codec.seed(int(options['num_hops']))
        ),
        'num_neighbors': (
            None
            if options.get('num_neighbors') is None
            else codec.seed(len(options['num_neighbors']))
        ),
        'prediction_statistic': options.get('prediction_statistic'),
        'inference_config': None if config is None else 5,
        'options': sorted(set(options) - TAKEN),
    }
    attachments = {
        1: codec.attachment(described),
        2: codec.attachment(query.to_string()),
        3: _indices(indices, schema, entity),
        4: codec.attachment(held),
    }
    if config is not None:
        attachments[5] = config
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


def materialize(
    materializer: RelationalMaterializer,
    query: ValidatedPredictiveQuery,
    indices: Sequence[Any],
    num_context_samples: int | None,
    options: dict[str, Any],
) -> Request:
    r"""The request, gathered and laid out on the core (:func:`gather`),
    as the client's ``Request`` of its payload."""
    from predictive_query.validator.utils import table_name

    from sdm_client.request import Request

    payloads, core_state = gather(
        materializer, query, indices, num_context_samples, options
    )
    request = Request.materialized(
        payloads=payloads,
        # The Request's own copy of whether the target is windowed, which
        # the client reads of the query (design, "Host-held semantics").
        windowed=getattr(query.target_ast, 'date_offset_range', None)
        is not None,
        schema=materializer.schema.snapshot(),
        query=query.to_string(),
        indices=list(indices),
        entity_table=table_name(query.entity_column),
        requested_inference={
            'inference_config': options.get('inference_config') or {}
        },
    )
    # The request state's version, which predict on the core hands back
    # with the payload (T1.11d).
    request._core_state = core_state
    return request
