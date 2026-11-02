# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify synthetic/source node-key equivalence against the shared fixture.

Only wire identity representation changes. Instances, row order, features,
labels and every sampled foreign-key edge remain checked. Ambiguous mappings
are refused, never guessed. Raw envelopes stay in the observation.
"""

import copy
import datetime
import math
import re
from decimal import Decimal, InvalidOperation


class IdentityError(ValueError):
    pass


def same(value, expected, dtype):
    # These are source-identity proofs, never payload/result normalization.
    if value is None and expected == {'float': 'nan'}:
        return dtype in {'float32', 'float64'}
    if dtype == 'int64' and isinstance(value, str) and type(expected) is int:
        return -(2**63) <= expected < 2**63 and value == str(expected)
    if (
        dtype == 'float64'
        and type(value) is float
        and isinstance(expected, dict)
        and set(expected) == {'decimal'}
    ):
        try:
            rounded = float(Decimal(expected['decimal']))
        except (ValueError, TypeError, InvalidOperation, OverflowError):
            return False
        return (
            math.isfinite(value) and math.isfinite(rounded) and value == rounded
        )
    if dtype == 'stringlist' and expected is None and value == []:
        return True
    if dtype == 'stringlist' and isinstance(expected, str):
        # Tokenization is not a stable source identity. Other fields or the
        # original key must still identify exactly one row. The complete token
        # list remains in the canonical payload for exact old/new comparison.
        return isinstance(value, list) and all(
            isinstance(token, str) for token in value
        )
    if (
        dtype.startswith('timestamp')
        and isinstance(value, str)
        and isinstance(expected, str)
    ):

        def instant(text, source=False):
            fraction = re.search(r'[T ]\d{2}:\d{2}:\d{2}[.,](\d+)', text)
            # Parse whole seconds separately: Python 3.10 rejects nanoseconds.
            # Decimal below retains every fractional digit for the comparison.
            whole = (
                text[: fraction.start(1) - 1] + text[fraction.end(1) :]
                if fraction
                else text
            )
            parsed = datetime.datetime.fromisoformat(
                whole.replace('Z', '+00:00')
            )
            remainder = Decimal('0.' + fraction[1]) if fraction else Decimal(0)
            if source:
                digits = {
                    'timestamp[s]': 0,
                    'timestamp[ms]': 3,
                    'timestamp[us]': 6,
                    'timestamp[ns]': 9,
                }.get(dtype)
                if digits is not None:
                    # Compare the source instant at the declared wire precision.
                    scale = 10**digits
                    remainder = Decimal(int(remainder * scale)) / scale
            parsed = parsed.replace(microsecond=0)
            return (
                parsed.replace(tzinfo=datetime.timezone.utc)
                if parsed.tzinfo is None
                else parsed,
                remainder,
            )

        return instant(value) == instant(expected, source=True)
    return (
        type(value) is type(expected)
        and value == expected
        or isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and value == expected
    )


def canonical(payloads, fixture):
    out = copy.deepcopy(payloads)
    for payload in out:
        definitions = payload.get('schema', {}).get('related_tables', {})
        keys = fixture['primary_keys']
        foreign = {name: [] for name in definitions}
        for source, target in fixture['links']:
            source_table, source_column = source.split('.')
            target_table, target_column = target.split('.')
            if source_table in foreign:
                foreign[source_table].append(
                    (source_column, target_table, target_column)
                )
        wire_keys = {}
        for name, definition in definitions.items():
            if name not in keys:
                raise IdentityError('unknown sampled fixture table: ' + name)
            primary = definition['primary_key']
            if (
                not isinstance(primary, list)
                or len(primary) != 2
                or primary[0] != 'instance_id'
            ):
                raise IdentityError('unexpected sampled composite key')
            wire_keys[name] = primary[1]
        for split in ('context', 'predict'):
            related = payload.get(split, {}).get('related_tables', {})
            mappings, originals = {}, {}
            for name, table in related.items():
                if name not in definitions:
                    raise IdentityError('sampled table has no schema')
                source = fixture['tables'][name]
                source_rows = [
                    dict(zip(source['columns'], row, strict=True))
                    for row in source['rows']
                ]
                pk = keys[name]
                wire_pk = wire_keys[name]
                foreign_columns = {field for field, _, _ in foreign[name]}
                features = [
                    column
                    for column in table['columns']
                    if column not in {'instance_id', wire_pk, *foreign_columns}
                ]
                if any(column not in source['columns'] for column in features):
                    raise IdentityError('unknown sampled feature')
                seen_source = set()
                mapping, row_sources = {}, []
                for values in table['rows']:
                    row = dict(zip(table['columns'], values, strict=True))
                    matches = [
                        candidate
                        for candidate in source_rows
                        if all(
                            same(
                                row[column],
                                candidate[column],
                                definitions[name]['columns'][column]['dtype'],
                            )
                            for column in features
                        )
                        and (
                            wire_pk == '__node_id'
                            or candidate[pk] == row[wire_pk]
                        )
                    ]
                    if len(matches) != 1:
                        raise IdentityError(
                            'sampled identity is not uniquely evidenced: '
                            + name
                        )
                    original = matches[0]
                    wire_id = (row['instance_id'], row[wire_pk])
                    source_id = (row['instance_id'], original[pk])
                    if wire_id in mapping or source_id in seen_source:
                        raise IdentityError(
                            'sampled node mapping is not bijective'
                        )
                    mapping[wire_id] = original[pk]
                    seen_source.add(source_id)
                    row_sources.append((row, original))
                mappings[name], originals[name] = mapping, row_sources
            for name, table in related.items():
                normalized = []
                for row, original in originals[name]:
                    values = {
                        column: value
                        for column, value in row.items()
                        if column not in ('instance_id', wire_keys[name])
                    }
                    for column, target, _target_column in foreign[name]:
                        if column not in row:
                            continue
                        wire_value = row[column]
                        if wire_value is None:
                            if original[column] is not None:
                                raise IdentityError(
                                    'sampled foreign-key identity was lost'
                                )
                            continue
                        destination = mappings.get(target, {}).get(
                            (row['instance_id'], wire_value)
                        )
                        if (
                            destination is None
                            or destination != original[column]
                        ):
                            raise IdentityError(
                                'sampled foreign-key edge differs from fixture'
                            )
                        values[column] = destination
                    normalized.append(
                        {
                            'instance_id': row['instance_id'],
                            'source_key': original[keys[name]],
                            'features': values,
                        }
                    )
                table['columns'] = ['instance_id', 'source_key', 'features']
                table['rows'] = normalized
        for name, definition in definitions.items():
            wire_pk = wire_keys[name]
            pk = keys[name]
            key_type = definition['columns'].pop(wire_pk)
            # A primary key is structural ID data under either spelling;
            # older envelopes additionally wrote the redundant ID stype.
            key_type.pop('stype', None)
            definition['columns'][pk] = key_type
            definition['primary_key'] = ['instance_id', pk]
        for link in payload.get('schema', {}).get('relationships', []):
            target = link.get('target_table')
            if target in wire_keys:
                link['target_columns'] = [
                    keys[target] if column == wire_keys[target] else column
                    for column in link['target_columns']
                ]
    return out


def canonical_scoring_payload(payload, pinned_schema, fixture):
    """Inspect a scoring body only with the verified actual pinned schema."""
    if 'schema' in payload:
        raise IdentityError(
            'scoring request unexpectedly redeclares pinned schema'
        )
    result = canonical([{**payload, 'schema': pinned_schema}], fixture)[0]
    del result['schema']
    return result


def equivalent_legacy_decimal_cast(old_payloads, new_payloads, fixture):
    """Prove the frozen Arrow decimal cast bug against exact source decimals.

    Only the caller's named DuckDB decimal-feature case uses this policy.
    Replace proven old rounding on copies; never change results or source rows.
    """
    from decimal import localcontext

    source = fixture['tables']['users']
    if source.get('types', {}).get('fraction') != 'decimal128(20,6)':
        return False
    key = fixture['primary_keys']['users']
    key_at, at = (source['columns'].index(c) for c in (key, 'fraction'))
    values = {row[key_at]: row[at] for row in source['rows']}
    if len(values) != len(source['rows']) or len(old_payloads) != len(
        new_payloads
    ):
        return False
    changes = []
    with localcontext() as context:
        context.prec = 80
        for old, new in zip(old_payloads, new_payloads, strict=True):
            for payload in (old, new):
                schema = payload['schema']['related_tables']['users']
                if (
                    schema['primary_key'] != ['instance_id', key]
                    or schema['columns']['fraction']['dtype'] != 'float64'
                ):
                    return False
            for half in ('context', 'predict'):
                a = old[half]['related_tables']['users']
                b = new[half]['related_tables']['users']
                if len(a['rows']) != len(b['rows']):
                    return False
                ai, bi = (t['columns'].index('fraction') for t in (a, b))
                ak, bk = (t['columns'].index(key) for t in (a, b))
                ax, bx = (t['columns'].index('instance_id') for t in (a, b))
                for row, other in zip(a['rows'], b['rows'], strict=True):
                    identity = row[ak]
                    if (
                        type(identity) is not int
                        or type(other[bk]) is not int
                        or type(row[ax]) is not int
                        or type(other[bx]) is not int
                        or identity not in values
                        or identity != other[bk]
                        or row[ax] != other[bx]
                    ):
                        return False
                    value = values[identity]
                    if value is None:
                        if row[ai] is not None or other[bi] is not None:
                            return False
                        continue
                    if (
                        type(row[ai]) is not float
                        or type(other[bi]) is not float
                    ):
                        return False
                    exact = Decimal(value)
                    coefficient = exact.scaleb(6)
                    if (
                        not exact.is_finite()
                        or coefficient != coefficient.to_integral_value()
                        or abs(coefficient) >= 10**20
                    ):
                        return False
                    # Frozen PyArrow24 decimal128->double: scale by multiplication.
                    legacy = float(int(coefficient)) * 1e-6
                    nearest = float(exact)
                    if row[ai] != legacy or other[bi] != nearest:
                        return False
                    if legacy != nearest:
                        if abs(Decimal.from_float(nearest) - exact) >= abs(
                            Decimal.from_float(legacy) - exact
                        ):
                            return False
                        changes.append((row, ai, nearest))
    for row, at, nearest in changes:
        row[at] = nearest
    return bool(changes)
