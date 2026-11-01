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


class IdentityError(ValueError):
    pass


def same(value, expected, dtype):
    if (
        dtype.startswith('timestamp')
        and isinstance(value, str)
        and isinstance(expected, str)
    ):

        def instant(text):
            parsed = datetime.datetime.fromisoformat(
                text.replace('Z', '+00:00')
            )
            return (
                parsed.replace(tzinfo=datetime.timezone.utc)
                if parsed.tzinfo is None
                else parsed
            )

        return instant(value) == instant(expected)
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
