# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Bridge code: the host-held rules here (design, "Host-held semantics"),
# the option refusals (T1.24), the zoned anchor's conversion (T1.25), the
# run mode, entity count, shape, limits and capped-context notice (T1.28a),
# the index once-flag (T1.26, by E52) and the entities' cast to the key's
# type (T1.32a, checkpoint 4), are deleted in Phase 1 by those tasks.
# Frozen meanwhile (E50, scripts/rust_core/bridge_freeze.py).

r"""``materialize`` on the core: the request's gathering (T1.11c).

``PredictiveQuery.materialize`` with ``indices`` over the schema of a DuckDB
source comes here on the core route, in place of the relational engine's
gathering (``RelationalMaterializer._gather``). Everything around that
step stays the client's own, as on the old route: the checks of the call
and the notices before it, and, after it, the notices of what the payload
sends and leaves out, the context that answers the question taken out, and
the ``Request`` built from the payload.

The core's ``materialize`` samples the source through the DuckDB provider
(:mod:`.providers.duckdb`) and lays out the payload the engine lays out;
this module hands it what the schema says now, with what the core's
description of the source said of each column that the schema does not
hold (the type the source handed over, whether it may be null, whether the
source keys it), and the entities to predict for as one Arrow column of the
key's own type. From T1.21d the core holds that schema to the reference's
consistency check itself, marking the columns it joins on again from what
each column means and meant before it was joined on (decision E55); this
module checks none of it. Nothing here imports the relational engine.

As built at T1.11c the core materializes a regression over a windowed
``SUM``, with the sampling options' defaults or the anchor time and seed
given; an option it does not take yet, or a schema it does not sample yet,
is refused with ``UNSUPPORTED_REQUEST`` naming the rule (register entry
P1B-3), here for an option and in the core for the rest.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from types import SimpleNamespace
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

#: The options the core takes; ``prediction_statistic`` is the client's own,
#: recorded on the payload after the gathering, as on the old route.
TAKEN = frozenset({'anchor_time', 'random_seed', 'prediction_statistic'})

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


def description(schema: Schema) -> dict[str, Any]:
    r"""The schema as the contract's ``SourceDescription``: its tables, their
    columns' data and semantic types (what each means, and meant before the
    schema joined on it), keys and time columns, and its relationships, as
    they stand; of each column, what the core's description of the source
    said that the schema does not hold.

    Raises:
        RouteError: The schema was not described on the core, so what the
            source said of its columns is not known here.
        StructuredDataError: ``UNSUPPORTED_REQUEST`` naming the rule, for a
            schema edited in a way the description cannot follow yet.
    """
    from sdm_client.materializers.sources import wire_dtype
    from sdm_client.schema import _as_key_tuple

    described = getattr(schema, '_core_description', None)
    if described is None:
        raise RouteError(
            'materialize on the core needs a schema Schema.from_source '
            'described on the core'
        )
    by_name = {table['name']: table for table in described['tables']}
    tables = []
    for name, table in schema.tables.items():
        source = by_name.get(name)
        # The description names each table by the source table it read; a
        # table mapped to another source table since is not followed yet.
        if source is None or getattr(table, 'source_name', name) != name:
            raise _not_taken('schema_edited')
        columns = {column['name']: column for column in source['columns']}
        shaped = []
        for column_name, column in table.columns.items():
            read = columns.get(column_name)
            if read is None or wire_dtype(read['dtype']) != column.dtype:
                raise _not_taken('schema_edited')
            shaped.append(
                {
                    **read,
                    'stype': str(column.stype),
                    'stype_before_key': column._stype_before_key,
                }
            )
        key = table.primary_key
        tables.append(
            {
                'name': name,
                'columns': shaped,
                'primary_key': [] if key is None else list(_as_key_tuple(key)),
                'time_column': table.time_column,
                'edit_stamp_refused': source['edit_stamp_refused'],
                'duplicate_keys': source['duplicate_keys'],
            }
        )
    links = [
        [
            {'table': link.source_table, 'column': link.source_column},
            {'table': link.target_table, 'column': link.target_column},
        ]
        for link in schema.relationships
    ]
    return {'tables': tables, 'links': links}


def _anchor(value: Any) -> dict[str, Any] | None:
    r"""The anchor time as the contract's ``ts`` scalar, in nanoseconds with
    no zone; a zoned one read at its UTC wall clock, as the engine reads it.
    """
    import pandas as pd

    if value is None:
        return None
    if isinstance(value, str) and value == 'entity':
        raise _not_taken('anchor_entity')
    if not isinstance(value, pd.Timestamp):
        raise StructuredDataError(
            f"'anchor_time' must be a pandas.Timestamp or the literal "
            f"'entity' (got {type(value).__name__})",
            code='INVALID_REQUEST',
        )
    if value.tzinfo is not None:
        value = value.tz_convert('UTC').tz_localize(None)
    return codec.timestamp(int(value.as_unit('ns').value), 'ns', None)


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
            materialize yet; and the client's own checks of the call.
    """
    from predictive_query.validator.utils import table_name

    from sdm_client.materializers.relational import (
        _warn_of_a_context_the_default_mode_capped,
    )
    from sdm_client.models import RELATIONAL_LIMITS

    for name in sorted(options):
        if name not in TAKEN:
            raise _not_taken(name)
    # A seed given as None asks the reference for an unseeded sample, a new
    # one each call; the core samples under a seed, so it is refused, never
    # taken for the default.
    if 'random_seed' in options and options['random_seed'] is None:
        raise _not_taken('unseeded')
    if num_context_samples is not None:
        raise _not_taken('num_context_samples')
    schema = materializer.schema
    materializer._run_mode_for(num_context_samples)
    materializer._check_entity_count(indices)
    materializer._check_shape(options, RELATIONAL_LIMITS)
    entity = table_name(query.entity_column)
    seed = options.get('random_seed')
    operation = {
        'op': 'materialize',
        'provider': {'provider': 'sql', 'dialect': 'duckdb'},
        'schema': 1,
        'query': 2,
        'indices': 3,
        'anchor_time': _anchor(options.get('anchor_time')),
        'seed': None if seed is None else codec.seed(int(seed)),
    }
    attachments = {
        1: codec.attachment(description(schema)),
        2: codec.attachment(query.to_string()),
        3: _indices(indices, schema, entity),
    }
    source: Any = schema.source
    # The reference checks the source's indexes once per materializer, when
    # it builds its sampler, and warns of missing ones then; the core reads
    # them on every run, so the warning is shown once per materializer here
    # (decision E41(j)).
    # Checked means a run gave the warning or finished, so read the
    # indexes; a run refused before its index read leaves it unchecked.
    checked = materializer._core_indexes_checked
    given: set[str] = set()
    try:
        finished = run(
            operation,
            attachments,
            provider=DuckDBProvider(source.connection),
            quiet=frozenset({'MISSING_INDICES'}) if checked else frozenset(),
            given=given,
        )
    finally:
        if 'MISSING_INDICES' in given:
            materializer._core_indexes_checked = True
    materializer._core_indexes_checked = True
    state = finished.result['state']
    payload = json.loads(finished.take(state['payload']))
    finished.take(state['query'])
    # What the client reads of the engine's task and its part: the context
    # gathered, and the payload.
    task: Any = SimpleNamespace(
        num_context_examples=payload['metadata']['num_context_examples']
    )
    _warn_of_a_context_the_default_mode_capped(task, num_context_samples)
    part: Any = SimpleNamespace(payload=payload)
    materializer._check_limits(part, RELATIONAL_LIMITS)
    return (payload,), {'version': state['version']}
