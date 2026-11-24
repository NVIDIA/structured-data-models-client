# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""The one translation between the core's wire format and the public
source types (:mod:`sdm_client.provider`), both ways, and the run loop's
provider over every :class:`~sdm_client.provider.Source`.

Read: a fetch's request (the contract's ``DataRequest``) into the public
request it is, each key set it names one of the core's Arrow attachments;
the schema its answer must have; a source's capabilities and failures.
Written: a public request back into the contract's JSON, its key sets by
attachment id, as a built-in source hands it to its SQL kit; a source's
:class:`~sdm_client.provider.Capabilities` and
:class:`~sdm_client.provider.ProviderFailure` as the core reads them. Each
way is the other's inverse: a request read and written again is the
request the core sent.

It decides nothing: which requests a source is sent, which filters, and a
key set too large for one request split into several, are the core's,
alike for every host and every source.
"""

from __future__ import annotations

import base64
import datetime
import decimal
import math
import struct
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import pyarrow as pa

from sdm_client import provider as p
from sdm_client._rust import codec
from sdm_client.errors import StructuredDataError

__all__ = [
    'HostProvider',
    'capabilities',
    'chosen',
    'expected_schema',
    'failure',
    'operation',
    'read_capabilities',
    'read_failure',
    'read_request',
    'refused',
    'value',
    'write_request',
]


def chosen(source: p.Source) -> tuple[dict[str, Any], HostProvider]:
    r"""The core's provider choice for one operation over ``source``, and
    the run loop's provider answering its fetches. Its capabilities are
    read once, at the start of the operation, from the source the
    operation asks (:func:`operation`)."""
    source = operation(source)
    declared = source.capabilities()
    if not isinstance(declared, p.Capabilities):
        raise TypeError(
            'Source.capabilities() must return a Capabilities, got '
            f'{type(declared).__name__}'
        )
    choice = {'provider': 'host', 'capabilities': capabilities(declared)}
    return choice, HostProvider(source)


def operation(source: p.Source) -> p.Source:
    r"""The source one operation over ``source`` asks: a built-in source's
    own answers for that operation (a fresh SQL kit, so what it keeps
    within an operation is that operation's alone), or any other source
    itself."""
    from sdm_client.sources import _BuiltInSource

    if isinstance(source, _BuiltInSource):
        return source._operation()
    return source


class HostProvider:
    r"""The run loop's provider over a source: each fetch step read into a
    request and its expected schema, and asked of it."""

    def __init__(self, source: p.Source) -> None:
        self._source = source

    def __call__(
        self, step: dict[str, Any], keys: Mapping[int, Any] | None = None
    ) -> p.ArrowStream:
        request = read_request(step['request'], keys or {})
        expected = expected_schema(step['plan'].get('expect'))
        answer = self._source.answer(request, expected)
        if not isinstance(answer, p.ArrowStream):
            raise TypeError(
                f'Source.answer() returned a {type(answer).__name__}; '
                'it returns an object exporting an Arrow stream '
                '(__arrow_c_stream__), such as a pyarrow Table'
            )
        return answer

    def error_fields(self, error: Exception) -> dict[str, Any] | None:
        r"""What ``error`` tells the core, by the source's ``failure``: the
        run loop reads the failure's attributes where it says nothing, as
        it does where the hook fails or answers with anything but a
        :class:`~sdm_client.provider.ProviderFailure`, so the source's own
        failure is always the one the operation reports."""
        try:
            told = self._source.failure(error)
        except Exception:  # noqa: BLE001 - the source's failure stands
            return None
        if not isinstance(told, p.ProviderFailure):
            return None
        return failure(told)


# --- refusals ----------------------------------------------------------------


def refused(error: BaseException) -> dict[str, Any] | None:
    r"""The core's error a source refused a request with, as the contract
    writes it (``FetchOutcome::Refused``), where ``error`` is such a
    refusal: one a built-in source's SQL kit gave, raised by its
    ``answer`` as the :class:`~sdm_client.errors.StructuredDataError` the
    operation fails with; ``None`` for any other failure."""
    if isinstance(error, StructuredDataError):
        return error._refusal
    return None


# --- capabilities --------------------------------------------------------------


def capabilities(declared: p.Capabilities) -> dict[str, Any]:
    r"""The contract's ``ProviderCapabilities`` of ``declared``, written as
    the core writes them: each field left out where the core leaves it
    out at its default."""
    wire: dict[str, Any] = {
        'optional': sorted(request.value for request in declared.optional),
        'max_keys_per_request': declared.max_keys_per_request,
        'sampling': declared.sampling.value,
        'max_seed': declared.max_seed,
        'text_times': _named(declared.text_times),
        'null_order': _named(declared.null_order),
        'index_advice': _named(declared.index_advice),
    }
    optional: dict[str, Any] = {
        'filters': sorted(kind.value for kind in declared.filters),
        'fallback': None
        if declared.fallback is None
        else {
            'max_rows': declared.fallback.max_rows,
            'max_bytes': declared.fallback.max_bytes,
        },
        'fallback_text_order': _named(declared.fallback_text_order),
        'derived_relations': declared.derived_relations,
        'sample_rows': declared.sample_by_row_count,
        'neighbor_lookback': declared.neighbor_lookback,
        'keys_optional': declared.keys_optional,
        'key_enforcement': declared.key_enforcement,
        'opaque_vendor': declared.opaque_vendor,
        'compares_stored_times': declared.compares_stored_times,
        'samples_by_key_dtype': declared.sample_by_key_type,
        'reads_nan_as_missing': declared.nan_is_missing,
        'database': declared.engine,
        'column_affinities': declared.column_types_are_affinities,
        'reads_tables_as_they_stand': declared.reread_tables_every_run,
        'namespace': list(declared.namespace),
        'tries_upper_case_names': declared.retry_upper_case_names,
        'namespace_first': declared.ask_namespace_first,
        'view': None
        if declared.semantic_model is None
        else {
            'name': declared.semantic_model.name,
            'trust_expressions': declared.semantic_model.trust_expressions,
        },
    }
    wire.update(
        (name, given)
        for name, given in optional.items()
        if given not in (None, False, [])
    )
    return wire


def read_capabilities(wire: Mapping[str, Any]) -> p.Capabilities:
    r"""The contract's ``ProviderCapabilities`` as the public
    :class:`~sdm_client.provider.Capabilities` it declares."""
    fallback = wire.get('fallback')
    view = wire.get('view')
    return p.Capabilities(
        optional=frozenset(
            p.OptionalRequest(name) for name in wire['optional']
        ),
        max_keys_per_request=wire.get('max_keys_per_request'),
        sampling=p.Sampling(wire['sampling']),
        max_seed=wire['max_seed'],
        null_order=_member(p.NullOrder, wire.get('null_order')),
        fallback=None
        if fallback is None
        else p.Fallback(fallback['max_rows'], fallback['max_bytes']),
        filters=frozenset(
            p.FilterKind(name) for name in wire.get('filters') or ()
        ),
        fallback_text_order=_member(
            p.TextOrder, wire.get('fallback_text_order')
        ),
        text_times=_member(p.TextTimes, wire.get('text_times')),
        index_advice=_member(p.IndexAdvice, wire.get('index_advice')),
        engine=wire.get('database'),
        namespace=tuple(wire.get('namespace') or ()),
        ask_namespace_first=wire.get('namespace_first') is True,
        retry_upper_case_names=wire.get('tries_upper_case_names') is True,
        column_types_are_affinities=wire.get('column_affinities') is True,
        reread_tables_every_run=wire.get('reads_tables_as_they_stand') is True,
        semantic_model=None
        if view is None
        else p.SemanticModel(view['name'], view['trust_expressions'] is True),
        derived_relations=wire.get('derived_relations') is True,
        sample_by_row_count=wire.get('sample_rows') is True,
        sample_by_key_type=wire.get('samples_by_key_dtype') is True,
        neighbor_lookback=wire.get('neighbor_lookback') is True,
        keys_optional=wire.get('keys_optional') is True,
        key_enforcement=wire.get('key_enforcement') is True,
        compares_stored_times=wire.get('compares_stored_times') is True,
        nan_is_missing=wire.get('reads_nan_as_missing') is True,
        opaque_vendor=wire.get('opaque_vendor'),
    )


def _named(member: Any) -> str | None:
    return None if member is None else member.value


def _member(kind: Any, name: str | None) -> Any:
    return None if name is None else kind(name)


# --- failures ------------------------------------------------------------------


def failure(told: p.ProviderFailure) -> dict[str, Any]:
    r"""The fields of the contract's ``ProviderError`` ``told`` says, each
    where it says one: the facts the core reads the failure by among
    them."""
    fields = {
        'sqlstate': told.sqlstate,
        'vendor_code': told.vendor_code,
        'error_class': told.error_class,
        'driver_message': told.message,
        'facts': None if told.facts is None else _facts(told.facts),
    }
    return {name: given for name, given in fields.items() if given is not None}


def read_failure(wire: Mapping[str, Any]) -> p.ProviderFailure:
    r"""The contract's ``ProviderError`` as the public failure it tells."""
    facts = wire.get('facts')
    return p.ProviderFailure(
        sqlstate=wire.get('sqlstate'),
        vendor_code=wire.get('vendor_code'),
        error_class=wire.get('error_class'),
        message=wire.get('driver_message'),
        facts=None if facts is None else _read_facts(facts),
    )


def _facts(facts: p.FailureFacts) -> dict[str, Any]:
    return {
        'not_found': facts.not_found,
        'refuses_expression': facts.refuses_expression,
        'unusable': _named(facts.unusable),
        'denied': facts.denied,
        'reason': _named(facts.reason),
    }


def _read_facts(wire: Mapping[str, Any]) -> p.FailureFacts:
    return p.FailureFacts(
        not_found=wire.get('not_found') is True,
        refuses_expression=wire.get('refuses_expression') is True,
        unusable=_member(p.Unusable, wire.get('unusable')),
        denied=wire.get('denied') is True,
        reason=_member(p.FailureReason, wire.get('reason')),
    )


# --- requests, read --------------------------------------------------------------


def read_request(
    wire: Mapping[str, Any], keys: Mapping[int, Any]
) -> p.DataRequest:
    r"""The contract's ``DataRequest`` JSON as the public request it is,
    each key set it names one of ``keys``, by attachment."""
    kind = wire['request']
    if kind == 'describe':
        return p.Describe(
            tables=tuple(_relation(t) for t in wire['tables']),
            part=p.DescribePart(wire['part']),
        )
    if kind == 'scan':
        return p.Scan(
            table=_relation(wire['table']),
            columns=tuple(wire['columns']),
            filters=_filters(wire['filters'], keys),
            limit=wire.get('limit'),
        )
    if kind == 'preview':
        return p.Preview(
            table=_relation(wire['table']),
            columns=tuple(wire['columns']),
            limit=wire['limit'],
        )
    if kind == 'count_rows':
        return p.CountRows(
            table=_relation(wire['table']),
            filters=_filters(wire['filters'], keys),
        )
    if kind == 'key_counts':
        return p.KeyCounts(
            keys=tuple(
                p.TableKey(
                    table=_relation(key['table']),
                    columns=tuple(key['columns']),
                    label=at if key.get('label') is None else key['label'],
                    unique=key.get('unique') is True,
                )
                for at, key in enumerate(wire['keys'])
            )
        )
    if kind == 'time_bounds':
        return p.TimeBounds(
            columns=tuple(
                p.TableColumn(_relation(c['table']), c['column'])
                for c in wire['columns']
            )
        )
    if kind == 'column_ranges':
        return p.ColumnRanges(
            table=_relation(wire['table']), columns=tuple(wire['columns'])
        )
    if kind == 'repeated_key':
        return p.RepeatedKey(
            table=_relation(wire['table']), key=tuple(wire['key'])
        )
    if kind == 'sample_entities':
        return p.SampleEntities(
            table=_relation(wire['table']),
            key=tuple(wire['key']),
            n=wire['n'],
            seed=_value(wire['seed']),
            columns=tuple(wire['columns']),
            filters=_filters(wire['filters'], keys),
            rows=wire.get('rows'),
            key_dtype=wire.get('key_dtype'),
        )
    if kind == 'neighbors':
        window = wire.get('window')
        return p.Neighbors(
            table=_relation(wire['table']),
            key=tuple(wire['key']),
            columns=tuple(wire['columns']),
            seeds=_key_set(wire['seeds'], keys),
            window=None if window is None else _window(window),
            per_seed=_limit(wire.get('per_seed')),
            order=tuple(_term(term) for term in wire.get('order') or ()),
        )
    if kind == 'lookup':
        return p.Lookup(
            table=_relation(wire['table']),
            key=tuple(wire['key']),
            columns=tuple(wire['columns']),
            seeds=_key_set(wire['seeds'], keys),
        )
    raise ValueError(
        f'the core sent a data request this host does not know: {kind!r}'
    )


def _relation(wire: Mapping[str, Any]) -> p.Relation:
    return p.Relation(
        parts=tuple(wire['source']),
        columns=tuple(
            p.RelationColumn(
                name=column['name'],
                column=column.get('column'),
                expression=column.get('expression'),
            )
            for column in wire.get('columns') or ()
        ),
        filter=wire.get('filter'),
        found=wire.get('found') is True,
    )


def _filters(wire: Any, keys: Mapping[int, Any]) -> tuple[p.Filter, ...]:
    return tuple(_filter(each, keys) for each in wire or ())


def _filter(wire: Mapping[str, Any], keys: Mapping[int, Any]) -> p.Filter:
    kind = wire['filter']
    if kind == 'key_in':
        return p.KeyIn(tuple(wire['columns']), _key_set(wire['keys'], keys))
    if kind == 'time_range':
        return p.TimeRange(
            wire['column'], _bound(wire.get('lower')), _bound(wire.get('upper'))
        )
    if kind == 'not_null':
        return p.NotNull(wire['column'])
    if kind == 'compare':
        return p.Compare(
            wire['column'], p.CompareOp(wire['op']), _value(wire['value'])
        )
    if kind == 'in':
        return p.In(wire['column'], tuple(_value(v) for v in wire['values']))
    if kind == 'is_null':
        return p.IsNull(wire['column'])
    if kind == 'any_of':
        return p.AnyOf(_filters(wire['filters'], keys))
    if kind == 'all_of':
        return p.AllOf(_filters(wire['filters'], keys))
    if kind == 'has_rows':
        return p.HasRows(
            column=wire['column'],
            table=_relation(wire['table']),
            key=wire['key'],
            filters=_filters(wire['filters'], keys),
            at_least=wire['at_least'],
        )
    raise ValueError(
        f'the core sent a filter this host does not know: {kind!r}'
    )


def _bound(wire: Mapping[str, Any] | None) -> p.Bound | None:
    if wire is None:
        return None
    return p.Bound(_value(wire['value']), wire['inclusive'] is True)


def _key_set(wire: Mapping[str, Any], keys: Mapping[int, Any]) -> p.KeySet:
    attachment = wire['attachment']
    if attachment not in keys:
        raise ValueError('a request names a key set the core did not give')
    return p.KeySet(
        schema=pa.schema([_field(column) for column in wire['columns']]),
        reader=_reader(keys[attachment]),
    )


def _reader(stream: Any) -> pa.RecordBatchReader:
    r"""A key set the core sent, an object exporting an Arrow stream, as a
    pyarrow reader, through the PyCapsule interface pyarrow 14.0.1 reads
    (``RecordBatchReader.from_stream`` is 15's)."""
    if isinstance(stream, pa.RecordBatchReader):
        return stream
    reader: pa.RecordBatchReader = pa.RecordBatchReader._import_from_c_capsule(
        stream.__arrow_c_stream__()
    )
    return reader


def _window(wire: Mapping[str, Any]) -> p.Window:
    def seed_bound(bound: Mapping[str, Any] | None) -> p.SeedBound | None:
        if bound is None:
            return None
        return p.SeedBound(bound['seed_column'], bound['inclusive'] is True)

    return p.Window(
        wire['column'],
        seed_bound(wire.get('lower')),
        seed_bound(wire.get('upper')),
        source_type=wire.get('source_type'),
    )


def _limit(wire: Any) -> p.NeighborLimit | None:
    if wire is None:
        return None
    if isinstance(wire, int) and not isinstance(wire, bool):
        return p.Count(wire)
    if isinstance(wire, Mapping) and wire.get('negative') is not None:
        return p.AllBut(wire['negative'])
    if isinstance(wire, Mapping) and wire.get('sql_expression') is not None:
        return p.SqlLimit(wire['sql_expression'])
    raise ValueError('the core sent a per-seed limit this host does not know')


def _term(wire: Mapping[str, Any]) -> p.TieTerm:
    return p.TieTerm(
        column=wire['column'],
        as_timestamp=wire.get('as_timestamp') is True,
        descending=wire.get('descending') is True,
        nulls_last=wire.get('nulls_last') is True,
        string_like=wire.get('string_like') is True,
    )


# --- requests, written -----------------------------------------------------------


class _KeySets:
    r"""The key sets a written request names, each by the attachment id it
    is given in order, the same key set once."""

    def __init__(self) -> None:
        self.given: dict[int, pa.RecordBatchReader] = {}
        self._ids: dict[int, int] = {}

    def write(self, keys: p.KeySet) -> dict[str, Any]:
        at = self._ids.get(id(keys.reader))
        if at is None:
            at = len(self.given) + 1
            self._ids[id(keys.reader)] = at
            self.given[at] = keys.reader
        return {
            'attachment': at,
            'columns': [_write_field(field) for field in keys.schema],
        }


def write_request(
    request: p.DataRequest,
) -> tuple[dict[str, Any], dict[int, pa.RecordBatchReader]]:
    r"""``request`` as the contract's ``DataRequest`` JSON, and the key sets
    it names, by attachment id: what :func:`read_request` reads back as
    ``request``."""
    keys = _KeySets()
    filters = _write_filters
    wire: dict[str, Any]
    if isinstance(request, p.Describe):
        wire = {
            'tables': [_write_relation(t) for t in request.tables],
            'part': request.part.value,
        }
    elif isinstance(request, p.Scan):
        wire = {
            'table': _write_relation(request.table),
            'columns': list(request.columns),
            'filters': filters(request.filters, keys),
            'limit': request.limit,
        }
    elif isinstance(request, p.Preview):
        wire = {
            'table': _write_relation(request.table),
            'columns': list(request.columns),
            'limit': request.limit,
        }
    elif isinstance(request, p.CountRows):
        wire = {
            'table': _write_relation(request.table),
            'filters': filters(request.filters, keys),
        }
    elif isinstance(request, p.KeyCounts):
        wire = {
            'keys': [_write_key(key, at) for at, key in enumerate(request.keys)]
        }
    elif isinstance(request, p.TimeBounds):
        wire = {
            'columns': [
                {'table': _write_relation(c.table), 'column': c.column}
                for c in request.columns
            ]
        }
    elif isinstance(request, p.ColumnRanges):
        wire = {
            'table': _write_relation(request.table),
            'columns': list(request.columns),
        }
    elif isinstance(request, p.RepeatedKey):
        wire = {
            'table': _write_relation(request.table),
            'key': list(request.key),
        }
    elif isinstance(request, p.SampleEntities):
        wire = {
            'table': _write_relation(request.table),
            'key': list(request.key),
            'n': request.n,
            'seed': write_value(request.seed),
            'columns': list(request.columns),
            'filters': filters(request.filters, keys),
        }
        if request.key_dtype is not None:
            wire['key_dtype'] = request.key_dtype
        if request.rows is not None:
            wire['rows'] = request.rows
    elif isinstance(request, p.Neighbors):
        wire = {
            'table': _write_relation(request.table),
            'key': list(request.key),
            'columns': list(request.columns),
            'seeds': keys.write(request.seeds),
            'window': None
            if request.window is None
            else _write_window(request.window),
            'per_seed': _write_limit(request.per_seed),
            'order': [_write_term(term) for term in request.order] or None,
        }
    elif isinstance(request, p.Lookup):
        wire = {
            'table': _write_relation(request.table),
            'key': list(request.key),
            'columns': list(request.columns),
            'seeds': keys.write(request.seeds),
        }
    else:
        raise TypeError(f'not a data request: {type(request).__name__}')
    return {'request': request.name, **wire}, keys.given


def _write_relation(relation: p.Relation) -> dict[str, Any]:
    wire: dict[str, Any] = {'source': list(relation.parts)}
    if relation.columns:
        wire['columns'] = [
            {
                name: given
                for name, given in (
                    ('name', column.name),
                    ('column', column.column),
                    ('expression', column.expression),
                )
                if given is not None
            }
            for column in relation.columns
        ]
    if relation.filter is not None:
        wire['filter'] = relation.filter
    if relation.found:
        wire['found'] = True
    return wire


def _write_key(key: p.TableKey, at: int) -> dict[str, Any]:
    wire: dict[str, Any] = {
        'table': _write_relation(key.table),
        'columns': list(key.columns),
    }
    # A key's label is its position unless the request names another.
    if key.label != at:
        wire['label'] = key.label
    if key.unique:
        wire['unique'] = True
    return wire


def _write_filters(
    filters: tuple[p.Filter, ...], keys: _KeySets
) -> list[dict[str, Any]]:
    return [_write_filter(each, keys) for each in filters]


def _write_filter(filter: p.Filter, keys: _KeySets) -> dict[str, Any]:
    if isinstance(filter, p.KeyIn):
        return {
            'filter': 'key_in',
            'columns': list(filter.columns),
            'keys': keys.write(filter.keys),
        }
    if isinstance(filter, p.TimeRange):
        return {
            'filter': 'time_range',
            'column': filter.column,
            'lower': _write_bound(filter.lower),
            'upper': _write_bound(filter.upper),
        }
    if isinstance(filter, p.NotNull):
        return {'filter': 'not_null', 'column': filter.column}
    if isinstance(filter, p.Compare):
        return {
            'filter': 'compare',
            'column': filter.column,
            'op': filter.op.value,
            'value': write_value(filter.value),
        }
    if isinstance(filter, p.In):
        return {
            'filter': 'in',
            'column': filter.column,
            'values': [write_value(v) for v in filter.values],
        }
    if isinstance(filter, p.IsNull):
        return {'filter': 'is_null', 'column': filter.column}
    if isinstance(filter, (p.AnyOf, p.AllOf)):
        return {
            'filter': 'any_of' if isinstance(filter, p.AnyOf) else 'all_of',
            'filters': _write_filters(filter.filters, keys),
        }
    if isinstance(filter, p.HasRows):
        return {
            'filter': 'has_rows',
            'column': filter.column,
            'table': _write_relation(filter.table),
            'key': filter.key,
            'filters': _write_filters(filter.filters, keys),
            'at_least': filter.at_least,
        }
    raise TypeError(f'not a filter: {type(filter).__name__}')


def _write_bound(bound: p.Bound | None) -> dict[str, Any] | None:
    if bound is None:
        return None
    return {'value': write_value(bound.value), 'inclusive': bound.inclusive}


def _write_window(window: p.Window) -> dict[str, Any]:
    def seed_bound(bound: p.SeedBound | None) -> dict[str, Any] | None:
        if bound is None:
            return None
        return {'seed_column': bound.seed_column, 'inclusive': bound.inclusive}

    wire: dict[str, Any] = {'column': window.column}
    if window.source_type is not None:
        wire['source_type'] = window.source_type
    wire['lower'] = seed_bound(window.lower)
    wire['upper'] = seed_bound(window.upper)
    return wire


def _write_limit(limit: p.NeighborLimit | None) -> Any:
    if limit is None:
        return None
    if isinstance(limit, p.Count):
        return limit.count
    if isinstance(limit, p.AllBut):
        return {'negative': limit.value}
    return {'sql_expression': limit.expression}


def _write_term(term: p.TieTerm) -> dict[str, Any]:
    wire: dict[str, Any] = {}
    if term.string_like:
        wire['string_like'] = True
    wire.update(
        column=term.column,
        as_timestamp=term.as_timestamp,
        descending=term.descending,
        nulls_last=term.nulls_last,
    )
    return wire


# --- values --------------------------------------------------------------------


def _value(wire: Mapping[str, Any]) -> p.Value:
    return p.Value(value(wire), _scalar_type(wire))


def value(wire: Mapping[str, Any]) -> Any:
    r"""A contract scalar as the plain Python value the public types hold,
    exactly, through the codec's reader.

    Temporal values keep every unit they carry: a microsecond timestamp or
    duration is a :class:`datetime.datetime` or :class:`datetime.timedelta`,
    a nanosecond one a :class:`pandas.Timestamp` or
    :class:`pandas.Timedelta`, which are those types keeping nanoseconds; a
    date is a :class:`datetime.date`; a time of day a :class:`datetime.time`,
    or, with nanoseconds, a :class:`pandas.Timedelta` since midnight. A date
    or timestamp outside the years :mod:`datetime` holds is a
    :class:`numpy.datetime64`. A timestamp with a zone is in that zone.
    """
    read = codec.decode_scalar(wire)
    if not isinstance(read, codec.Temporal):
        return read
    if read.tag == 'date32':
        try:
            return _EPOCH + datetime.timedelta(days=read.value)
        except OverflowError:
            import numpy as np

            return np.datetime64(read.value, 'D')
    if read.unit == 'ns':
        import pandas as pd

        if read.tag == 'ts':
            instant = pd.Timestamp(read.value, unit='ns')
            if read.zone is None:
                return instant
            return instant.tz_localize('UTC').tz_convert(_zone(read.zone))
        since = pd.Timedelta(read.value, unit='ns')
        if read.tag == 'duration' or read.value % 1000:
            return since
        return _time_of_day(read.value // 1000)
    if read.tag == 'duration':
        return datetime.timedelta(microseconds=read.value)
    if read.tag == 'time64':
        return _time_of_day(read.value)
    try:
        wall = _UNIX + datetime.timedelta(microseconds=read.value)
    except OverflowError:
        import numpy as np

        return np.datetime64(read.value, 'us')
    if read.zone is None:
        return wall
    return wall.replace(tzinfo=datetime.timezone.utc).astimezone(
        _zone(read.zone)
    )


_SCALAR_TYPES: dict[str, pa.DataType] = {
    'null': pa.null(),
    'bool': pa.bool_(),
    'i8': pa.int8(),
    'i16': pa.int16(),
    'i32': pa.int32(),
    'i64': pa.int64(),
    'u8': pa.uint8(),
    'u16': pa.uint16(),
    'u32': pa.uint32(),
    'u64': pa.uint64(),
    'f32': pa.float32(),
    'f64': pa.float64(),
    'utf8': pa.string(),
    'binary': pa.binary(),
    'date32': pa.date32(),
}
_SCALAR_TAGS = {kind: tag for tag, kind in _SCALAR_TYPES.items()}


def _scalar_type(wire: Mapping[str, Any]) -> pa.DataType:
    tag = wire['t']
    if tag in _SCALAR_TYPES:
        return _SCALAR_TYPES[tag]
    if tag == 'd128':
        return pa.decimal128(wire['p'], wire['s'])
    if tag == 'd256':
        return pa.decimal256(wire['p'], wire['s'])
    if tag == 'ts':
        return pa.timestamp(wire['unit'], tz=wire.get('zone'))
    if tag == 'duration':
        return pa.duration(wire['unit'])
    if tag == 'time64':
        return pa.time64(wire['unit'])
    raise ValueError(
        f'the core sent a value of a type this host does not know: {tag!r}'
    )


def write_value(given: p.Value) -> dict[str, Any]:
    r"""``given`` as the contract scalar of its type, exactly: the inverse
    of reading one."""
    kind, plain = given.type, given.value
    tag = _SCALAR_TAGS.get(kind)
    if tag == 'null':
        return {'t': 'null'}
    if tag == 'bool':
        return {'t': 'bool', 'v': bool(plain)}
    if tag in ('i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64'):
        return codec.integer(tag, int(plain))
    if tag in ('f32', 'f64'):
        return {'t': tag, 'v': _float_text(float(plain), tag == 'f32')}
    if tag == 'utf8':
        return {'t': 'utf8', 'v': str(plain)}
    if tag == 'binary':
        return {'t': 'binary', 'v': base64.b64encode(bytes(plain)).decode()}
    if tag == 'date32':
        return {'t': 'date32', 'v': str(_days(plain))}
    if pa.types.is_decimal(kind):
        # Exactly, in a context wide enough for every decimal's digits.
        with decimal.localcontext(decimal.Context(prec=_DIGITS)):
            unscaled = int(Decimal(plain).scaleb(kind.scale))
        return {
            't': 'd128' if pa.types.is_decimal128(kind) else 'd256',
            'p': kind.precision,
            's': kind.scale,
            'v': str(unscaled),
        }
    if pa.types.is_timestamp(kind):
        return codec.timestamp(
            _since_epoch(plain, kind.unit), kind.unit, kind.tz
        )
    if pa.types.is_duration(kind):
        return {
            't': 'duration',
            'unit': kind.unit,
            'v': str(_span(plain, kind.unit)),
        }
    if pa.types.is_time64(kind):
        return {
            't': 'time64',
            'unit': kind.unit,
            'v': str(_of_day(plain, kind.unit)),
        }
    raise ValueError(f'a value of a type the contract does not hold: {kind}')


def _float_text(number: float, single: bool) -> str:
    r"""``number`` as the core writes a float: the shortest digits that
    read back as it (as a float32 where ``single``), plain from 1e-4 up to
    1e16 and in exponent form outside, as Rust's ``{:?}`` writes them."""
    if math.isnan(number):
        return 'NaN'
    if math.isinf(number):
        return 'inf' if number > 0 else '-inf'
    text = repr(number)
    if single:
        for places in range(1, 10):
            candidate = f'{number:.{places}g}'
            if _as_float32(float(candidate)) == number:
                text = candidate
                break
    sign, digits, exponent = Decimal(text).as_tuple()
    assert isinstance(exponent, int)
    shown = ''.join(map(str, digits)).rstrip('0') or '0'
    # The power of ten of the leading digit.
    point = len(digits) + exponent - 1 if any(digits) else 0
    minus = '-' if sign else ''
    if not any(digits) or 1e-4 <= abs(number) < 1e16:
        whole = point + 1
        if whole <= 0:
            return f'{minus}0.{"0" * -whole}{shown}'
        if whole >= len(shown):
            return f'{minus}{shown}{"0" * (whole - len(shown))}.0'
        return f'{minus}{shown[:whole]}.{shown[whole:]}'
    mantissa = shown[0] if len(shown) == 1 else f'{shown[0]}.{shown[1:]}'
    return f'{minus}{mantissa}e{point}'


def _as_float32(number: float) -> float:
    return float(struct.unpack('>f', struct.pack('>f', number))[0])


_EPOCH = datetime.date(1970, 1, 1)
_UNIX = datetime.datetime(1970, 1, 1)
#: More digits than any decimal the contract holds (76) carries.
_DIGITS = 100
_PER_SECOND = {'s': 1, 'ms': 1_000, 'us': 1_000_000, 'ns': 1_000_000_000}


def _is_pandas(plain: Any) -> bool:
    return (type(plain).__module__ or '').startswith('pandas')


def _numpy_count(plain: Any, unit: str) -> int | None:
    r"""A NumPy date, time or span counted in ``unit``; ``None`` for any
    other value."""
    if (type(plain).__module__ or '') != 'numpy':
        return None
    kind = (
        'datetime64' if 'datetime64' in type(plain).__name__ else 'timedelta64'
    )
    return int(plain.astype(f'{kind}[{unit}]').astype('int64'))


def _days(plain: Any) -> int:
    counted = _numpy_count(plain, 'D')
    if counted is not None:
        return counted
    return (plain - _EPOCH).days


def _since_epoch(plain: Any, unit: str) -> int:
    counted = _numpy_count(plain, unit)
    if counted is not None:
        return counted
    if _is_pandas(plain):
        return int(plain.value) * _PER_SECOND[unit] // _PER_SECOND['ns']
    if plain.tzinfo is not None:
        plain = plain.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    micros = (plain - _UNIX) // datetime.timedelta(microseconds=1)
    return micros * _PER_SECOND[unit] // _PER_SECOND['us']


def _span(plain: Any, unit: str) -> int:
    counted = _numpy_count(plain, unit)
    if counted is not None:
        return counted
    if _is_pandas(plain):
        return int(plain.value) * _PER_SECOND[unit] // _PER_SECOND['ns']
    micros = plain // datetime.timedelta(microseconds=1)
    return micros * _PER_SECOND[unit] // _PER_SECOND['us']


def _of_day(plain: Any, unit: str) -> int:
    if isinstance(plain, datetime.time):
        micros = (
            (plain.hour * 60 + plain.minute) * 60 + plain.second
        ) * 1_000_000 + plain.microsecond
        return micros * _PER_SECOND[unit] // _PER_SECOND['us']
    return _span(plain, unit)


def _time_of_day(microseconds: int) -> datetime.time:
    seconds, micro = divmod(microseconds, 1_000_000)
    minutes, second = divmod(seconds, 60)
    hour, minute = divmod(minutes, 60)
    return datetime.time(hour, minute, second, micro)


def _zone(name: str) -> datetime.tzinfo:
    r"""A zone the contract names: ``UTC``, a fixed offset (``+05:30``), or
    an IANA name."""
    if name == 'UTC':
        return datetime.timezone.utc
    if len(name) == 6 and name[0] in '+-' and name[3] == ':':
        sign = -1 if name[0] == '-' else 1
        offset = datetime.timedelta(
            hours=int(name[1:3]), minutes=int(name[4:6])
        )
        return datetime.timezone(sign * offset)
    from zoneinfo import ZoneInfo

    return ZoneInfo(name)


# --- schemas -------------------------------------------------------------------


def expected_schema(wire: Mapping[str, Any] | None) -> pa.Schema | None:
    r"""The schema an answer must have, where its plan fixes one
    (``ExpectedSchema::Schema``), as a pyarrow schema; ``None`` where the
    answer holds the source's own columns."""
    if wire is None or wire.get('expect') != 'schema':
        return None
    return pa.schema([_field(f) for f in wire['fields']])


def _field(wire: Mapping[str, Any]) -> pa.Field:
    return pa.field(
        wire['name'],
        _type(wire['type']),
        nullable=wire['nullable'] is True,
        metadata=wire.get('metadata') or None,
    )


_SIMPLE_TYPES: dict[str, pa.DataType] = {
    'null': pa.null(),
    'bool': pa.bool_(),
    'int8': pa.int8(),
    'int16': pa.int16(),
    'int32': pa.int32(),
    'int64': pa.int64(),
    'uint8': pa.uint8(),
    'uint16': pa.uint16(),
    'uint32': pa.uint32(),
    'uint64': pa.uint64(),
    'float32': pa.float32(),
    'float64': pa.float64(),
    'utf8': pa.string(),
    'large_utf8': pa.large_string(),
    'binary': pa.binary(),
    'large_binary': pa.large_binary(),
    'date32': pa.date32(),
}
_SIMPLE_NAMES = {kind: name for name, kind in _SIMPLE_TYPES.items()}


def _type(wire: Mapping[str, Any]) -> pa.DataType:
    kind = wire['type']
    if kind in _SIMPLE_TYPES:
        return _SIMPLE_TYPES[kind]
    if kind == 'decimal128':
        return pa.decimal128(wire['precision'], wire['scale'])
    if kind == 'decimal256':
        return pa.decimal256(wire['precision'], wire['scale'])
    if kind == 'timestamp':
        return pa.timestamp(wire['unit'], tz=wire.get('zone'))
    if kind == 'duration':
        return pa.duration(wire['unit'])
    if kind == 'time64':
        return pa.time64(wire['unit'])
    if kind == 'list':
        return pa.list_(_field(wire['item']))
    if kind == 'struct':
        return pa.struct([_field(f) for f in wire['fields']])
    if kind == 'map':
        key, item = _field(wire['key']), _field(wire['value'])
        return pa.map_(key, item, keys_sorted=wire['sorted'] is True)
    if kind == 'dictionary_utf8':
        index = getattr(pa, wire['index'])()
        return pa.dictionary(index, pa.string())
    raise ValueError(
        f'the core expects a column type this host does not know: {kind!r}'
    )


def _write_field(field: pa.Field) -> dict[str, Any]:
    wire: dict[str, Any] = {
        'name': field.name,
        'type': _write_type(field.type),
        'nullable': field.nullable,
    }
    if field.metadata:
        wire['metadata'] = {
            key.decode(): given.decode()
            for key, given in field.metadata.items()
        }
    return wire


def _write_type(kind: pa.DataType) -> dict[str, Any]:
    if kind in _SIMPLE_NAMES:
        return {'type': _SIMPLE_NAMES[kind]}
    if pa.types.is_decimal128(kind) or pa.types.is_decimal256(kind):
        name = 'decimal128' if pa.types.is_decimal128(kind) else 'decimal256'
        return {'type': name, 'precision': kind.precision, 'scale': kind.scale}
    if pa.types.is_timestamp(kind):
        return {'type': 'timestamp', 'unit': kind.unit, 'zone': kind.tz}
    if pa.types.is_duration(kind):
        return {'type': 'duration', 'unit': kind.unit}
    if pa.types.is_time64(kind):
        return {'type': 'time64', 'unit': kind.unit}
    if pa.types.is_map(kind):
        return {
            'type': 'map',
            'entries': 'entries',
            'key': _write_field(kind.key_field),
            'value': _write_field(kind.item_field),
            'sorted': kind.keys_sorted,
        }
    if pa.types.is_list(kind):
        return {'type': 'list', 'item': _write_field(kind.value_field)}
    if pa.types.is_struct(kind):
        return {
            'type': 'struct',
            'fields': [
                _write_field(kind.field(i)) for i in range(kind.num_fields)
            ],
        }
    if pa.types.is_dictionary(kind) and pa.types.is_string(kind.value_type):
        return {'type': 'dictionary_utf8', 'index': str(kind.index_type)}
    raise ValueError(f'a column type the contract does not hold: {kind}')
