# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sampled graph links must resolve within their own prediction instance."""

from collections import Counter
from decimal import Decimal


def number(value):
    if isinstance(value, dict) and set(value) == {'number'}:
        return Decimal(value['number'])
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return Decimal(str(value))
    return value


def rows(table):
    return [
        dict(zip(table['columns'], row, strict=True)) for row in table['rows']
    ]


def assert_local_links(body):
    relationships = body['schema']['relationships']
    assert relationships
    # A shared source column is overwritten in emitted relationship order.
    # Other declared destinations are not guaranteed to resolve.
    effective = {}
    for relation in relationships:
        effective[
            relation.get('source_table'), tuple(relation['source_columns'])
        ] = relation
    checked = 0
    for half in ('context', 'predict'):
        section = body[half]
        tables = {**section['related_tables'], None: section['instance_table']}
        for relation in effective.values():
            source = tables.get(relation.get('source_table'))
            if source is None:
                continue
            target = tables.get(relation['target_table'])
            targets = (
                Counter(
                    (
                        number(row['instance_id']),
                        *(number(row[c]) for c in relation['target_columns']),
                    )
                    for row in rows(target)
                )
                if target is not None
                else Counter()
            )
            for row in rows(source):
                values = [row[c] for c in relation['source_columns']]
                if any(value is None for value in values):
                    continue
                assert not any(isinstance(value, bool) for value in values)
                key = (
                    number(row['instance_id']),
                    *(number(value) for value in values),
                )
                assert targets[key] == 1, (half, relation, key, targets[key])
                checked += 1
    assert checked > 0


def assert_diamond_links_retained(body):
    # The fixture stores order id in tag and sets orders.cid = order id % 10.
    witnessed = 0
    for half in ('context', 'predict'):
        tables = body[half]['related_tables']
        customers = {
            (number(row['instance_id']), number(row['tag'])): number(
                row['__node_id']
            )
            for row in rows(tables['customers'])
        }
        for row in rows(tables['orders']):
            target = customers.get(
                (number(row['instance_id']), number(row['tag']) % 10)
            )
            if target is not None:
                assert row['cid'] is not None
                assert number(row['cid']) == target
                witnessed += 1
    assert witnessed > 0
