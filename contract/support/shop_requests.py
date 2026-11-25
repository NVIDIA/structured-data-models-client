#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The relational requests a shop case should send, derived from its authored inputs.

Each call's context is every user the entity filter and ASSUMING admit at
each anchor, walking back one target window at a time from the anchor the
case gives while there is data, labelled by the target the call asks; its
related tables hold each instance's user and the facts up to its anchor.
Nothing here reads a recorded request.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

#: A window over orders: an aggregation and its days.
_SUM = re.compile(r'^PREDICT SUM\(orders\.price, (-?\d+), (-?\d+), days\)')
_ANY = re.compile(r'^PREDICT COUNT\(orders\.\*, (-?\d+), (-?\d+), days\) > 0')
_RANK = re.compile(
    r'^PREDICT LIST_DISTINCT\(buys\.item_id, (-?\d+), (-?\d+), days\) RANK '
)
_FORECAST = re.compile(r' FORECAST (\d+) TIMEFRAMES ')
_COUNT_ORDERS = re.compile(r'^COUNT\(orders\.\*, (-?\d+), (-?\d+), days\) > 0$')
_STATIC = re.compile(r"^users\.(\w+) (>|<|=) ('?[\w ]+'?)$")
#: The rows sent for a call, at most.
_ROWS_PER_CALL = 10_000


def _stamp(value: datetime) -> str:
    return value.strftime('%Y-%m-%dT%H:%M:%S.%fZ')


class Shop:
    r"""A case's database read once: users, orders and, where its setup
    makes them, buys and items."""

    def __init__(self, case: dict[str, Any], bases: dict[str, list[str]]):
        import duckdb

        connection = duckdb.connect()
        try:
            for statement in [
                *bases[case['input']['base']],
                *case['input'].get('setup', []),
            ]:
                connection.execute(statement)
            tables = {
                row[0]
                for row in connection.execute(
                    'SELECT table_name FROM duckdb_tables()'
                ).fetchall()
            }
            self.tables = tables
            self.users = {
                row[0]: row[1:]
                for row in connection.execute(
                    'SELECT user_id, age, country FROM users'
                ).fetchall()
            }
            self.orders = connection.execute(
                'SELECT user_id, price, ts FROM orders'
            ).fetchall()
            self.buys = (
                connection.execute(
                    'SELECT user_id, item_id, ts FROM buys'
                ).fetchall()
                if 'buys' in tables
                else []
            )
            self.items = (
                dict(
                    connection.execute(
                        'SELECT item_id, kind FROM items'
                    ).fetchall()
                )
                if 'items' in tables
                else {}
            )
            self._static = connection
            self.earliest = min(ts for *_, ts in [*self.orders, *self.buys])
        except BaseException:
            connection.close()
            raise

    def close(self) -> None:
        self._static.close()

    def admitted(self, condition: str | None, at: datetime) -> set[int]:
        r"""The users ``condition`` admits at ``at``: a column of the user
        compared with a literal, or a count of orders in a window of it."""
        if condition is None:
            return set(self.users)
        window = _COUNT_ORDERS.match(condition)
        if window:
            start, end = (int(n) for n in window.groups())
            return {
                user
                for user, _, ts in self.orders
                if at + timedelta(days=start) < ts <= at + timedelta(days=end)
            }
        static = _STATIC.match(condition)
        assert static, f'no rule derives the users {condition!r} admits'
        column, operator, literal = static.groups()
        return {
            row[0]
            for row in self._static.execute(
                f'SELECT user_id FROM users WHERE {column} {operator} {literal}'
            ).fetchall()
        }

    def orders_until(self, user: int, at: datetime) -> list[tuple[Any, ...]]:
        return sorted(
            (price, _stamp(ts), user)
            for owner, price, ts in self.orders
            if owner == user and ts <= at
        )

    def buys_until(self, user: int, at: datetime) -> list[tuple[Any, ...]]:
        return sorted(
            (_stamp(ts), user, self.items[item])
            for owner, item, ts in self.buys
            if owner == user and ts <= at
        )


def _target(pql: str) -> tuple[str, int, int]:
    body = re.sub(r'^(EVALUATE|EXPLAIN) ', '', pql)
    for kind, pattern in (('sum', _SUM), ('any', _ANY), ('member', _RANK)):
        found = pattern.match(body)
        if found:
            start, end = (int(n) for n in found.groups())
            return kind, start, end
    raise AssertionError(f'no rule derives the labels of {pql!r}')


def _clauses(pql: str) -> tuple[str | None, str | None]:
    r"""The entity filter and the ASSUMING condition ``pql`` gives."""
    assuming = None
    if ' ASSUMING ' in pql:
        pql, assuming = pql.split(' ASSUMING ', 1)
    _, _, condition = pql.partition(' FOR EACH users.user_id WHERE ')
    return condition or None, assuming


def _label(
    shop: Shop,
    kind: str,
    user: int,
    at: datetime,
    window: tuple[int, int],
    candidate: Any,
) -> Any:
    low, high = (at + timedelta(days=d) for d in window)
    if kind == 'sum':
        return float(
            sum(
                price
                for owner, price, ts in shop.orders
                if owner == user and low < ts <= high
            )
        )
    if kind == 'any':
        return any(
            owner == user and low < ts <= high for owner, _, ts in shop.orders
        )
    return any(
        owner == user and item == candidate and low < ts <= high
        for owner, item, ts in shop.buys
    )


def expected_calls(
    case: dict[str, Any], bases: dict[str, list[str]]
) -> list[dict[str, Any]]:
    r"""Each call the case should make, in order: its context, by anchor
    and entity, each with its label; the entities it scores; the anchor
    they are scored at; and its related tables. A ranking is asked one
    candidate at a time, each candidate once, and one whose context holds
    one class is not asked; a forecast is asked a step at a time."""
    given = case['input']
    spec = given['materialize']
    anchor = datetime.fromisoformat(spec['anchor_time'])
    kind, start, end = _target(given['pql'])
    condition, assuming = _clauses(given['pql'])
    steps = _FORECAST.search(given['pql'])
    parts: list[tuple[Any, tuple[int, int]]]
    if kind == 'member':
        parts = [(c, (start, end)) for c in dict.fromkeys(spec['candidates'])]
    elif steps:
        width = end - start
        parts = [
            (None, (start + width * k, end + width * k))
            for k in range(int(steps.group(1)))
        ]
    else:
        parts = [(None, (start, end))]
    shop = Shop(case, bases)
    try:
        calls = []
        for candidate, window in parts:
            width = window[1] - window[0]
            at = anchor - timedelta(days=window[1])
            context = {}
            while at >= shop.earliest:
                users = shop.admitted(condition, at) & shop.admitted(
                    assuming, at
                )
                for user in users:
                    context[(_stamp(at), user)] = _label(
                        shop, kind, user, at, window, candidate
                    )
                at -= timedelta(days=width)
            if kind == 'member' and len(set(context.values())) < 2:
                continue
            batch = spec.get('rows_per_call') or _ROWS_PER_CALL
            indices = spec['indices']
            calls += [
                {
                    'context': context,
                    'predict': indices[first : first + batch],
                    'anchor': _stamp(anchor),
                    'related': related(shop),
                }
                for first in range(0, len(indices), batch)
            ]
        return calls
    finally:
        shop.close()


def related(shop: Shop) -> dict[str, Any]:
    r"""What each related table holds for an instance of ``user`` at
    ``at``, by table."""
    out: dict[str, Any] = {
        'users': lambda user, at: [(user, *shop.users[user])],
        'orders': shop.orders_until,
    }
    if 'buys' in shop.tables:
        out['buys'] = shop.buys_until
        out['items'] = lambda user, at: sorted(
            {kind for *_, kind in shop.buys_until(user, at)}
        )
    return out


def plain(value: Any) -> Any:
    r"""A parsed request's value with each number as the number it
    writes (the runner keeps a number as ``{"number": <its text>}``)."""
    if isinstance(value, dict):
        if set(value) == {'number'} and isinstance(value['number'], str):
            text = value['number']
            return float(text) if any(c in text for c in '.eE') else int(text)
        return {name: plain(item) for name, item in value.items()}
    if isinstance(value, list):
        return [plain(item) for item in value]
    return value


def _rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(zip(table['columns'], row, strict=True)) for row in table['rows']
    ]


def assert_sent(
    case: dict[str, Any], bases: dict[str, list[str]], sent: list[Any]
) -> None:
    r"""Each request in ``sent`` (parsed, numbers as numbers) is the call
    :func:`expected_calls` derives for ``case``: its context's entities,
    anchors and labels, the entities it scores, and every related table's
    rows for each instance."""
    wanted = expected_calls(case, bases)
    sent = [plain(body) for body in sent]
    assert len(sent) == len(wanted), (case['id'], len(sent), len(wanted))
    for call, (body, want) in enumerate(zip(sent, wanted, strict=True)):
        where = f'{case["id"]} call {call}'
        target = body['task']['target']['column_name']
        context = _rows(body['context']['instance_table'])
        predict = _rows(body['predict']['instance_table'])
        got = {
            (row['__kumo_anchor_time'], row['__kumo_entity_ref_0']): row[target]
            for row in context
        }
        assert len(got) == len(context), f'{where}: an instance twice'
        missing = sorted(set(want['context']) - set(got))
        extra = sorted(set(got) - set(want['context']))
        assert not missing and not extra, (
            f'{where}: context lacks {missing[:4]} and holds {extra[:4]}'
        )
        wrong = [
            (each, got[each], label)
            for each, label in sorted(want['context'].items())
            if got[each] != label or type(got[each]) is not type(label)
        ]
        assert not wrong, f'{where}: labels (sent, derived): {wrong[:4]}'
        assert [row['__kumo_entity_ref_0'] for row in predict] == (
            want['predict']
        ), f'{where}: scores the wrong entities'
        assert {row['__kumo_anchor_time'] for row in predict} == {
            want['anchor']
        }, f'{where}: scored at the wrong anchor'
        assert len({row['instance_id'] for row in [*context, *predict]}) == (
            len(context) + len(predict)
        ), f'{where}: an instance id twice'
        for half, instances in (('context', context), ('predict', predict)):
            tables = body[half]['related_tables']
            assert set(tables) == set(want['related']), (
                f'{where}: {half} related tables {sorted(tables)}'
            )
            held = {name: _rows(table) for name, table in tables.items()}
            for instance in instances:
                user = instance['__kumo_entity_ref_0']
                at = datetime.fromisoformat(instance['__kumo_anchor_time'][:-1])
                for name, rule in want['related'].items():
                    rows = [
                        row
                        for row in held[name]
                        if row['instance_id'] == instance['instance_id']
                    ]
                    assert _related(name, rows, held) == rule(user, at), (
                        f'{where}: {half} {name} of {instance}'
                    )


def _related(
    name: str, rows: list[dict[str, Any]], held: dict[str, Any]
) -> list[Any]:
    if name == 'users':
        return [(r['user_id'], r['age'], r['country']) for r in rows]
    if name == 'orders':
        return sorted((r['price'], r['ts'], r['user_id']) for r in rows)
    if name == 'items':
        return sorted(r['kind'] for r in rows)
    kinds = {
        (r['instance_id'], r['__node_id']): r['kind'] for r in held['items']
    }
    return sorted(
        (r['ts'], r['user_id'], kinds[(r['instance_id'], r['item_id'])])
        for r in rows
    )
