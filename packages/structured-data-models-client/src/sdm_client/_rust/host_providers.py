# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""The bridge between the core's host-provider messages and a caller's
:class:`~sdm_client.provider.DataProvider`: the provider choice a
:class:`~sdm_client.provider.ProviderSource` names, each fetch's request
read into the public request types with the key sets it carries, the
schema its answer must have, and what a failure tells the core.

It translates and decides nothing: which requests a provider is sent,
which filters, and a key set too large for one request split into several,
are the core's, alike for every host.
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any

from sdm_client import provider as p
from sdm_client._rust import codec

__all__ = [
    'HostProvider',
    'capabilities',
    'chosen',
    'expected_schema',
    'read_request',
]


def chosen(provider: p.DataProvider) -> tuple[dict[str, Any], HostProvider]:
    r"""The core's provider choice for ``provider``, and the run loop's
    provider answering its fetches. Its capabilities are read once."""
    declared = provider.capabilities()
    if not isinstance(declared, p.Capabilities):
        raise TypeError(
            'DataProvider.capabilities() must return a Capabilities, got '
            f'{type(declared).__name__}'
        )
    choice = {'provider': 'host', 'capabilities': capabilities(declared)}
    return choice, HostProvider(provider)


def capabilities(declared: p.Capabilities) -> dict[str, Any]:
    r"""The contract's ``ProviderCapabilities`` of ``declared``."""
    wire: dict[str, Any] = {
        'optional': sorted(request.value for request in declared.optional),
        'max_keys_per_request': declared.max_keys_per_request,
        'sampling': declared.sampling.value,
        'max_seed': declared.max_seed,
        'text_times': None,
        'null_order': (
            None if declared.null_order is None else declared.null_order.value
        ),
        'index_advice': None,
    }
    if declared.filters:
        wire['filters'] = sorted(kind.value for kind in declared.filters)
    if declared.fallback is not None:
        wire['fallback'] = {
            'max_rows': declared.fallback.max_rows,
            'max_bytes': declared.fallback.max_bytes,
        }
    return wire


class HostProvider:
    r"""The run loop's provider over a caller's provider: each fetch step
    read into a request and its expected schema, and asked of it."""

    def __init__(self, provider: p.DataProvider) -> None:
        self._provider = provider

    def __call__(
        self, step: dict[str, Any], keys: Mapping[int, Any] | None = None
    ) -> Any:
        request = read_request(step['request'], keys or {})
        expected = expected_schema(step['plan'].get('expect'))
        answer = self._provider.answer(request, expected)
        if not hasattr(answer, '__arrow_c_stream__'):
            raise TypeError(
                f'DataProvider.answer() returned a {type(answer).__name__}; '
                'it returns an object exporting an Arrow stream '
                '(__arrow_c_stream__), such as a pyarrow Table'
            )
        return answer

    def error_fields(self, error: Exception) -> dict[str, Any] | None:
        r"""What ``error`` tells the core, by the provider's ``failure``,
        where it has one: the run loop reads the failure's attributes
        otherwise, as it does where the hook fails or answers with anything
        but a :class:`~sdm_client.provider.ProviderFailure`, so the
        provider's own failure is always the one the operation reports."""
        read = getattr(self._provider, 'failure', None)
        if not callable(read):
            return None
        try:
            told = read(error)
        except Exception:  # noqa: BLE001 - the provider's failure stands
            return None
        if not isinstance(told, p.ProviderFailure):
            return None
        fields = {
            'sqlstate': told.sqlstate,
            'vendor_code': told.vendor_code,
            'error_class': told.error_class,
            'driver_message': told.message,
        }
        return {
            name: value for name, value in fields.items() if value is not None
        }


# --- the request -----------------------------------------------------------


def read_request(wire: Mapping[str, Any], keys: Mapping[int, Any]) -> Any:
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
            seed=value(wire['seed']),
            columns=tuple(wire['columns']),
            filters=_filters(wire['filters'], keys),
            rows=wire.get('rows'),
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
    if wire.get('columns') or wire.get('filter') is not None:
        raise ValueError(
            'the core sent a derived relation to a provider that declares none'
        )
    return p.Relation(tuple(wire['source']))


def _filters(wire: Any, keys: Mapping[int, Any]) -> tuple[Any, ...]:
    return tuple(_filter(each, keys) for each in wire or ())


def _filter(wire: Mapping[str, Any], keys: Mapping[int, Any]) -> Any:
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
            wire['column'], p.CompareOp(wire['op']), value(wire['value'])
        )
    if kind == 'in':
        return p.In(wire['column'], tuple(value(v) for v in wire['values']))
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
    return p.Bound(value(wire['value']), wire['inclusive'] is True)


def _key_set(wire: Mapping[str, Any], keys: Mapping[int, Any]) -> p.KeySet:
    attachment = wire['attachment']
    if attachment not in keys:
        raise ValueError('a request names a key set the core did not give')
    return p.KeySet(
        columns=tuple(column['name'] for column in wire['columns']),
        reader=_reader(keys[attachment]),
    )


def _reader(stream: Any) -> Any:
    r"""A key set the core sent, an object exporting an Arrow stream, as a
    pyarrow reader, through the PyCapsule interface pyarrow 14.0.1 reads
    (``RecordBatchReader.from_stream`` is 15's)."""
    import pyarrow as pa

    if isinstance(stream, pa.RecordBatchReader):
        return stream
    return pa.RecordBatchReader._import_from_c_capsule(
        stream.__arrow_c_stream__()
    )


def _window(wire: Mapping[str, Any]) -> p.Window:
    def seed_bound(bound: Mapping[str, Any] | None) -> p.SeedBound | None:
        if bound is None:
            return None
        return p.SeedBound(bound['seed_column'], bound['inclusive'] is True)

    return p.Window(
        wire['column'],
        seed_bound(wire.get('lower')),
        seed_bound(wire.get('upper')),
    )


def _limit(wire: Any) -> p.Count | p.AllBut | None:
    if wire is None:
        return None
    if isinstance(wire, int) and not isinstance(wire, bool):
        return p.Count(wire)
    if isinstance(wire, Mapping) and wire.get('negative') is not None:
        return p.AllBut(wire['negative'])
    raise ValueError(
        'the core sent a per-seed limit a host provider cannot answer'
    )


def _term(wire: Mapping[str, Any]) -> p.TieTerm:
    return p.TieTerm(
        column=wire['column'],
        as_timestamp=wire.get('as_timestamp') is True,
        descending=wire.get('descending') is True,
        nulls_last=wire.get('nulls_last') is True,
        string_like=wire.get('string_like') is True,
    )


# --- values ------------------------------------------------------------------


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


_EPOCH = datetime.date(1970, 1, 1)
_UNIX = datetime.datetime(1970, 1, 1)


# --- the expected schema -----------------------------------------------------


def expected_schema(wire: Mapping[str, Any] | None) -> Any:
    r"""The schema an answer must have, where its plan fixes one
    (``ExpectedSchema::Schema``), as a pyarrow schema; ``None`` where the
    answer holds the source's own columns."""
    if wire is None or wire.get('expect') != 'schema':
        return None
    import pyarrow as pa

    return pa.schema([_field(f) for f in wire['fields']])


def _field(wire: Mapping[str, Any]) -> Any:
    import pyarrow as pa

    return pa.field(
        wire['name'],
        _type(wire['type']),
        nullable=wire['nullable'] is True,
        metadata=wire.get('metadata') or None,
    )


def _type(wire: Mapping[str, Any]) -> Any:
    import pyarrow as pa

    kind = wire['type']
    simple = {
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
    if kind in simple:
        return simple[kind]
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
