# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Answers between a host and the core.

A prediction's result crosses the binding as JSON text (its *plain* form):
its items, metadata and explanation, every value written as the parity
fixtures write one (``tests/parity/FORMAT.md``), with a mapping whose keys
are not all text written ``{"$map": [[key, value], ...]}``.

The core builds a result's frame as a *layout*: a ``pyarrow.Table`` of the
frame's columns whose field metadata says how pandas holds each one
(``sdm.dtype``) and, for an ``object`` column, its cells (``sdm.cells``).
:func:`frame` builds the frame back from the layout, restoring call-local
host references when a public result holds values outside the wire types.

Public presentation uses operation-local inferred pandas columns instead:
the core expands cells, requests stable ordering and validates/applies the
returned permutation to every column before :func:`frame` materializes it.

Nothing here decides anything about a result; it only writes and reads it.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import json
import math
import uuid
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from ._containers import ancestors

#: Integers an ``int64`` holds; wider ones are written as their digits.
_INT64 = (-(2**63), 2**63 - 1)

#: The field metadata keys of a layout.
DTYPE_KEY = b'sdm.dtype'
CELLS_KEY = b'sdm.cells'


class HeldValues(dict[int, Any]):
    def __init__(self, values: dict[int, Any] | None = None) -> None:
        super().__init__(values or {})
        self.scope = (
            values.scope if isinstance(values, HeldValues) else uuid.uuid4().hex
        )
        self.error: BaseException | None = None
        self._indices = {id(value): key for key, value in self.items()}

    def retain(self, value: Any) -> int:
        identity = id(value)
        if identity not in self._indices:
            reference = len(self)
            self[reference] = value
            self._indices[identity] = reference
        return self._indices[identity]

    def clear(self) -> None:
        super().clear()
        self._indices.clear()
        self.error = None

    def protocol(self, operation: str, operands_json: str) -> str:
        try:
            return _host_protocol(operation, operands_json, references=self)
        except Exception as error:
            self.error = error
            raise


def _held(value: Any, view: Any, references: dict[int, Any]) -> Any:
    if isinstance(references, HeldValues):
        reference = references.retain(value)
    else:
        reference = next(
            (key for key, item in references.items() if item is value),
            len(references),
        )
        references[reference] = value
    body = {
        'reference': {'$host_ref': str(reference)},
        'value': view,
        'type': type.__getattribute__(type(value), '__name__'),
    }
    if isinstance(references, HeldValues):
        body['owner'] = references.scope
    return {'$held': body}


def _has_reference(value: Any) -> bool:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, dict):
            if len(item) == 1 and '$held' in item:
                return True
            pending.extend(item.values())
    return False


def _host_protocol(
    operation: str,
    operands_json: str,
    *,
    references: dict[int, Any] | None = None,
) -> str:
    r"""Generic language facts requested by the core at an operation's use site."""
    operands = decode(json.loads(operands_json), references)
    if operation == 'truth':
        result = bool(operands)
    elif operation == 'hashable':
        hash(operands)
        result = None
    elif operation == 'repr':
        result = repr(operands)
    elif operation == 'set_cardinality':
        result = len(set(operands))
    elif operation == 'mapping_get':
        mapping, key, *defaults = operands
        result = mapping.get(key, *defaults)
    elif operation == 'getitem':
        value, key = operands
        result = value[key]
    elif operation == 'is_instance':
        value, family = operands
        result = isinstance(
            value,
            {'dict': dict, 'mapping': Mapping, 'list_tuple': (list, tuple)}[
                family
            ],
        )
    elif operation in ('mapping_set', 'mapping_copy', 'mapping_pop'):
        if references is None:
            raise ValueError('mapping assignment requires its operation owner')
        if operation == 'mapping_copy':
            dictionary = dict(operands)
        elif operation == 'mapping_pop':
            dictionary, key, default = operands
            dictionary.pop(key, default)
        else:
            dictionary, key, value = operands
            if dictionary is None:
                dictionary = {}
            dictionary[key] = value
        view = encode(dictionary, references=references)
        if isinstance(view, dict) and '$held' in view:
            view = view['$held']['value']
        return json.dumps(_held(dictionary, view, references))
    elif operation == 'sorted_set_difference':
        left, right = operands
        result = sorted(set(left) - frozenset(right))
    elif operation == 'sorted_set_intersection':
        left, right = operands
        result = sorted(set(left) & set(right))
    elif operation == 'dataclass_nondefault_fields':
        result = None
        if dataclasses.is_dataclass(operands) and not isinstance(
            operands, type
        ):
            result = []
            for field in dataclasses.fields(operands):
                value = getattr(operands, field.name)
                default = (
                    field.default_factory()
                    if field.default_factory is not dataclasses.MISSING
                    else field.default
                )
                if value != default:
                    result.append(field.name)
    else:
        raise ValueError(f'unknown host protocol {operation}')
    return json.dumps(encode(result, references=references))


def encode(
    value: Any,
    _holding: frozenset[int] = frozenset(),
    *,
    max_depth: int | None = None,
    references: dict[int, Any] | None = None,
) -> Any:
    r"""A Python value as plain JSON, tagged where JSON has no form for it.

    A container met again inside itself is written
    ``{"$refers_back": TYPE}``, which the core refuses to send as JSON as
    ``json_ready`` does; plain JSON cannot hold the cycle itself.
    ``_holding`` is the ids of the containers holding ``value``.
    The iterative walk preserves deeply nested values without consuming
    Python's call stack.
    A caller may bound an argument whose contract forbids deep nesting,
    retaining its outer shape through the client's no-plain-form path.
    A NumPy item that remains a NumPy scalar goes directly to scalar
    encoding or the no-plain-form fallback, never back on the work queue.
    """
    try:
        import numpy as np
        import pandas as pd
    except ImportError:  # the binding on its own, without the data stack
        np = pd = None  # type: ignore[assignment]

    root: list[Any] = [None]
    stack: list[tuple[Any, frozenset[int], Any, Any, bool]] = [
        (value, _holding, root, 0, False)
    ]

    def sequence(values: Any, holding: frozenset[int]) -> list[Any]:
        found = list(values)
        result: list[Any] = [None] * len(found)
        stack.extend(
            (item, holding, result, position, False)
            for position, item in reversed(list(enumerate(found)))
        )
        return result

    while stack:
        item, holding, target, key, complete = stack.pop()
        if complete:
            view = target[key]
            if references is not None and any(
                _has_reference(pair[0]) for pair in view['$map']
            ):
                target[key] = _held(item, view, references)
            continue
        if references is not None and type(item) not in (
            type(None),
            bool,
            int,
            float,
            str,
            dict,
            list,
            tuple,
        ):
            opaque = {
                '$object': {
                    'type': type.__getattribute__(type(item), '__name__'),
                    'repr': '',
                }
            }
            if issubclass(type(item), str):
                view = encode(str.__str__(item))
            elif issubclass(type(item), int):
                view = encode(int.__index__(item))
            elif issubclass(type(item), float):
                view = encode(float.__float__(item))
            elif type(item) in (
                bytes,
                Decimal,
                dt.datetime,
                dt.date,
                uuid.UUID,
            ):
                view = encode(item)
            elif pd is not None and type(item) is pd.Timestamp:
                view = encode(item)
            elif (
                np is not None
                and any(
                    type(item) is scalar for scalar in np.sctypeDict.values()
                )
                and issubclass(type(item), np.generic)
                and item.dtype.kind in 'biufU'
            ):
                converted = item.item()
                if type(converted) not in (bool, int, float, str):
                    view = opaque
                else:
                    view = (
                        encode(converted)
                        if isinstance(item, np.str_)
                        else {
                            '$np': {
                                'type': item.dtype.name,
                                'value': encode(converted),
                            }
                        }
                    )
            else:
                view = opaque
            target[key] = _held(item, view, references)
            continue

        if id(item) in holding:
            view = {'$refers_back': type(item).__name__}
            target[key] = (
                _held(item, view, references)
                if references is not None
                else view
            )
            continue
        holding = ancestors(item, holding, max_depth)
        if isinstance(item, tuple):
            target[key] = {'$tuple': sequence(item, holding)}
        elif np is not None and isinstance(item, np.ndarray):
            target[key] = {
                '$ndarray': {
                    'dtype': 'str'
                    if item.dtype.kind == 'U'
                    else item.dtype.name,
                    'values': sequence(item.tolist(), holding),
                }
            }
        elif isinstance(item, Mapping):
            if all(type(name) is str for name in item):
                mapping = dict.fromkeys(item)
                target[key] = (
                    {'$dict': mapping}
                    if len(mapping) == 1 and next(iter(mapping)).startswith('$')
                    else mapping
                )
                stack.extend(
                    (child, holding, mapping, name, False)
                    for name, child in reversed(list(item.items()))
                )
            else:
                stack.append((item, holding, target, key, True))
                target[key] = {
                    '$map': [sequence(pair, holding) for pair in item.items()]
                }
        elif isinstance(item, (list, set, frozenset)):
            target[key] = sequence(item, holding)
        elif (
            np is not None
            and isinstance(item, np.generic)
            and not isinstance(item, np.str_)
        ):
            if isinstance(item, np.longdouble):
                stack.append((float(item), holding, target, key, False))
            elif isinstance(item, np.datetime64):
                stack.append((pd.Timestamp(item), holding, target, key, False))
            elif isinstance(item, np.timedelta64):
                stack.append((pd.Timedelta(item), holding, target, key, False))
            else:
                converted = (
                    item.tobytes() if item.dtype.kind == 'S' else item.item()
                )
                if isinstance(converted, np.generic):
                    target[key] = _encode_scalar(converted, np, pd)
                    continue
                scalar = {
                    'type': item.dtype.str
                    if item.dtype.kind in 'SV'
                    else item.dtype.name,
                    'value': None,
                }
                target[key] = {'$np': scalar}
                stack.append((converted, holding, scalar, 'value', False))
        else:
            target[key] = _encode_scalar(item, np, pd)
    return root[0]


def _encode_scalar(value: Any, np: Any, pd: Any) -> Any:

    if value is None or isinstance(value, (bool, str)):
        return value
    if pd is not None and value is pd.NA:
        return {'$na': None}
    if pd is not None and value is pd.NaT:
        return {'$nat': None}
    if np is not None and isinstance(value, np.str_):
        return str(value)
    if isinstance(value, int):
        if _INT64[0] <= value <= _INT64[1]:
            return value
        return {'$int': str(value)}
    if isinstance(value, float):
        if math.isnan(value):
            return {'$float': 'nan'}
        if math.isinf(value):
            return {'$float': 'inf' if value > 0 else '-inf'}
        return value
    if isinstance(value, Decimal):
        return {'$decimal': str(value)}
    if pd is not None and isinstance(value, pd.Timestamp):
        return {'$timestamp': value.isoformat()}
    if isinstance(value, dt.datetime):
        return {'$datetime': value.isoformat()}
    if isinstance(value, dt.date):
        return {'$date': value.isoformat()}
    if pd is not None and isinstance(value, (pd.Timedelta, dt.timedelta)):
        return {'$timedelta_us': int(pd.Timedelta(value).value // 1000)}
    if isinstance(value, dt.timedelta):
        return {'$timedelta_us': value // dt.timedelta(microseconds=1)}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {'$bytes': base64.b64encode(bytes(value)).decode('ascii')}
    if isinstance(value, uuid.UUID):
        return {'$uuid': str(value)}
    raise TypeError(f'no plain form for {type(value).__name__}')


def decode(
    value: Any, references: dict[int, Any] | ResultValues | None = None
) -> Any:
    r"""Plain JSON as the Python value it stands for; the inverse of
    :func:`encode`.
    Cyclic references require the same host-owned registry; their graph is
    never inferred from a container's type name.
    """
    if isinstance(value, list):
        return [decode(item, references) for item in value]
    if not isinstance(value, dict):
        return value
    if len(value) == 1:
        ((tag, body),) = value.items()
        if tag == '$host_ref' and isinstance(references, ResultValues):
            return references.values[int(body)]
        if tag in ('$host_ref', '$held'):
            if references is None or isinstance(references, ResultValues):
                raise ValueError('held values require their operation owner')
            if (
                tag == '$held'
                and 'owner' in body
                and (
                    not isinstance(references, HeldValues)
                    or references.scope != body['owner']
                )
            ):
                raise ValueError(
                    'held reference is outside its operation owner'
                )
            reference = (
                body if tag == '$host_ref' else body['reference']['$host_ref']
            )
            try:
                return references[int(reference)]
            except (KeyError, ValueError, TypeError) as error:
                raise ValueError(
                    'held reference is outside its operation owner'
                ) from error
        if (
            tag == '$refers_back'
            and isinstance(body, dict)
            and references is not None
        ):
            return references[body['reference']]
        found = _decode_tag(tag, body, references)
        if found is not _NOT_A_TAG:
            return found
    return {key: decode(item, references) for key, item in value.items()}


_NOT_A_TAG = object()


def _decode_tag(
    tag: str, body: Any, references: dict[int, Any] | ResultValues | None = None
) -> Any:
    import numpy as np
    import pandas as pd

    if tag == '$int':
        return int(body)
    if tag == '$float':
        return float(body)
    if tag == '$decimal':
        return Decimal(body)
    if tag == '$timestamp':
        return pd.Timestamp(body)
    if tag == '$datetime':
        return dt.datetime.fromisoformat(body)
    if tag == '$date':
        return dt.date.fromisoformat(body)
    if tag == '$timedelta_us':
        return dt.timedelta(microseconds=body)
    if tag == '$bytes':
        return base64.b64decode(body)
    if tag == '$uuid':
        return uuid.UUID(body)
    if tag == '$na':
        return pd.NA
    if tag == '$nat':
        return pd.NaT
    if tag == '$tuple':
        return tuple(decode(item, references) for item in body)
    if tag == '$np':
        dtype = body['type']
        for prefix, code in (('bytes', 'S'), ('void', 'V')):
            if dtype.startswith(prefix) and dtype[len(prefix) :].isdigit():
                dtype = f'{code}{int(dtype[len(prefix) :]) // 8}'
                break
        value = decode(body['value'])
        if np.dtype(dtype).kind == 'S':
            return np.bytes_(value)
        return np.array([value], dtype=dtype)[0]
    if tag == '$ndarray':
        return np.array(decode(body['values']), dtype=body['dtype'])
    if tag == '$map':
        return {
            decode(key, references): decode(value, references)
            for key, value in body
        }
    if tag == '$dict':
        return {key: decode(item, references) for key, item in body.items()}
    return _NOT_A_TAG


_ITEM_FIELDS = (
    'row_index',
    'id',
    'forecast_step',
    'prediction',
    'probabilities',
    'scores',
    'rankings',
    'embeddings',
    'quantiles',
    'explanation',
    'metadata',
)


def _item(item: Any, fields: Sequence[str] = _ITEM_FIELDS) -> dict[str, Any]:
    if isinstance(item, Mapping):
        found = item
    else:
        found = {name: getattr(item, name, None) for name in fields}
    return {
        name: encode(found[name])
        for name in fields
        if found.get(name) is not None
    }


def absence_json(
    result: Any,
    requested: Sequence[str],
    references: ResultValues,
) -> str:
    """Hold original objects without evaluating their presence protocols."""
    return json.dumps(
        {
            'items': [
                {'metadata': references.hold(item, none_is_value=True)}
                for item in result.items
            ],
            'owner': references.hold(result),
            'requested': references.hold(requested),
        }
    )


class ResultValues:
    """Synchronous result presentation values, held only for one native call.

    The core owns field expansion and ordering; references preserve host
    values it need not interpret, including arbitrary Python objects.
    Reference numbers belong to this instance, never a wire request.
    """

    def __init__(self) -> None:
        self.values: list[Any] = []
        self._indices: dict[int, int] = {}

    def hold(self, value: Any, *, none_is_value: bool = False) -> Any:
        if value is None and not none_is_value:
            return None
        identity = id(value)
        if identity not in self._indices:
            self._indices[identity] = len(self.values)
            self.values.append(value)
        return {'$host_ref': str(self._indices[identity])}

    def equal(self, left: str, right: str) -> bool:
        existing, candidate = self.values[int(left)], self.values[int(right)]
        return existing is candidate or bool(existing == candidate)

    def truth(self, reference: str) -> bool:
        return bool(self.values[int(reference)])

    def protocol(self, operation: str, operands_json: str) -> str:
        operands = decode(json.loads(operands_json), self)
        if operation == 'set':
            collection = set(operands[0])
            found = [self.hold(collection), len(collection)]
        elif operation == 'sorted_repr':
            found = repr(sorted(operands[0]))
        elif operation == 'dict_keys':
            found = [
                self.hold(value, none_is_value=True)
                for value in dict.fromkeys(operands[0])
            ]
        elif operation == 'contains':
            found = operands[1] in operands[0]
        elif operation == 'equal':
            found = bool(operands[0] == operands[1])
        elif operation == 'getattr_default_none':
            found = self.hold(getattr(operands[0], operands[1], None))
        else:
            raise ValueError(f'unknown result protocol: {operation}')
        return json.dumps(found)

    def metadata_entry(self, reference: str, key: str, encoded: str) -> Any:
        """Copy and assign the entry chosen by the core, without rehashing keys."""
        mapping = dict(self.values[int(reference)])
        mapping[key] = decode(json.loads(encoded), self)
        return json.dumps(self.hold(mapping))

    def item(self, item: Any) -> dict[str, Any]:
        found = {}
        for name in ('row_index', 'id', 'forecast_step', 'prediction'):
            value = getattr(item, name, None)
            if value is not None:
                found[name] = self.hold(value)
        for name in ('probabilities', 'quantiles'):
            mapping = getattr(item, name, None)
            if mapping:
                found[name] = {
                    '$map': [
                        [format(key, ''), self.hold(value)]
                        for key, value in mapping.items()
                    ]
                }
        for name in ('scores', 'embeddings'):
            sequence = getattr(item, name, None)
            if sequence is not None:
                found[name] = [self.hold(value) for value in list(sequence)]
        if item.rankings:
            found['rankings'] = [
                {
                    '$map': [
                        [self.hold(key), self.hold(value)]
                        for key, value in dict(entry).items()
                    ]
                }
                for entry in item.rankings
            ]
        return found

    def infer_column(self, encoded: str) -> tuple[str, int]:
        """Infer one supplied column with pandas' record-construction semantics."""
        import pandas as pd

        cells = decode(json.loads(encoded), self)
        column = (
            pd.DataFrame([{0: value} for value in cells])[0]
            if cells
            else pd.Series([], dtype=object)
        )
        return self.hold(column)['$host_ref'], len(column)

    def column_order(
        self, reference: str, stable: bool, ascending: bool, missing_last: bool
    ) -> list[int]:
        """Pandas' vector ordering primitive under the core's explicit policy."""
        column = self.values[int(reference)]
        return column.sort_values(
            kind='stable' if stable else 'quicksort',
            ascending=ascending,
            na_position='last' if missing_last else 'first',
        ).index.tolist()


def result_json(
    result: Any, *, references: ResultValues | None = None, layout: bool = False
) -> str:
    r"""A result, a ``PredictionResult`` or an :class:`Answer`, as the plain
    JSON the core reads."""
    if isinstance(result, Answer):
        return result.plain
    found = {
        'model': None if layout or references is not None else result.model,
        'request_id': None
        if layout or references is not None
        else result.request_id,
        'items': [references.item(item) for item in result.items]
        if references is not None and layout
        else [
            _item(item)
            if references is None
            else {'metadata': references.hold(item, none_is_value=True)}
            for item in result.items
        ],
        'metadata': {}
        if layout
        else (
            encode(dict(result.metadata or {}))
            if references is None
            else references.hold(result.metadata)
        ),
        'explanation': None
        if layout
        else (
            encode(result.explanation)
            if references is None
            else references.hold(result.explanation)
        ),
        'outputs': []
        if references is not None or layout and result.items
        else list(result.outputs),
        'unscored_candidates': [
            encode(c)
            if references is None
            else references.hold(c, none_is_value=True)
            for c in (() if layout else result.unscored_candidates)
        ],
        'names_its_rows': False
        if layout and result.items
        else bool(getattr(result, '_names_its_rows', True)),
        'names_its_step': False
        if layout and result.items
        else bool(getattr(result, '_names_its_step', False)),
    }
    if references is not None and (not layout or not result.items):
        found['host_values'] = {
            'model': None
            if layout
            else references.hold(result.model, none_is_value=True),
            'request_id': None
            if layout
            else references.hold(result.request_id, none_is_value=True),
            'outputs': references.hold(result.outputs),
        }
    return json.dumps(found)


class Answer:
    r"""A result the core gave, as plain JSON, with the attributes a
    ``PredictionResult`` has."""

    def __init__(self, plain: str) -> None:
        self.plain = plain
        found = json.loads(plain)
        self.model = found['model']
        self.request_id = found['request_id']
        self.metadata = decode(found.get('metadata') or {})
        self.outputs = tuple(found.get('outputs') or ())
        self.unscored_candidates = tuple(
            decode(c) for c in found.get('unscored_candidates') or ()
        )
        self.explanation = decode(found.get('explanation'))
        self.items = tuple(found.get('items') or ())
        self._names_its_rows = found.get('names_its_rows', True)
        self._names_its_step = found.get('names_its_step', False)

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return f'Answer({self.model!r}, {len(self.items)} rows)'


def _column(
    field: Any, array: Any, references: ResultValues | None = None
) -> Any:
    import numpy as np
    import pandas as pd

    metadata = field.metadata or {}
    dtype = metadata.get(DTYPE_KEY, b'object').decode()
    if dtype == 'object':
        cells = json.loads(metadata.get(CELLS_KEY, b'[]').decode())
        values = np.empty(len(cells), dtype=object)
        for at, cell in enumerate(cells):
            values[at] = decode(cell, references)
        if references is not None and any(
            isinstance(cell, dict) and '$host_ref' in cell for cell in cells
        ):
            return pd.Series(list(values))
        return pd.Series(values, dtype=object)
    if dtype == 'str':
        return pd.Series(array.to_pylist(), dtype='str')
    if dtype.startswith('datetime64'):
        return pd.Series(array.to_pandas()).astype(dtype)
    return pd.Series(array.to_numpy(zero_copy_only=False)).astype(dtype)


def frame(layout: Any, references: ResultValues | None = None) -> Any:
    r"""The frame ``to_pandas()`` gives, restoring optional host values."""
    import pandas as pd
    import pyarrow as pa

    if isinstance(layout, str):
        if references is None:
            raise ValueError('result presentation requires host column facts')
        found = json.loads(layout)
        columns = {
            column['name']: references.values[int(column['reference'])]
            .take(column['positions'])
            .reset_index(drop=True)
            for column in found['columns']
        }
        return pd.DataFrame(
            columns,
            index=pd.RangeIndex(found['rows']),
            columns=pd.Index([], dtype=object)
            if found['rows'] and not columns
            else None,
        )
    if isinstance(layout, pa.RecordBatch):
        layout = pa.Table.from_batches([layout])
    columns = {}
    for at, field in enumerate(layout.schema):
        array = layout.column(at).combine_chunks()
        columns[field.name] = _column(field, array, references).reset_index(
            drop=True
        )
    built = pd.DataFrame(columns)
    built.index = pd.RangeIndex(layout.num_rows)
    return built


def requested(names: Sequence[str]) -> list[str]:
    r"""The names asked for, as a list of text."""
    return [str(name) for name in names]
