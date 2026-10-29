# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK-free comparison of actual bound fetch owners and checked SQL plans."""

from __future__ import annotations

import copy
from typing import Any


class BindingError(ValueError):
    """The observation does not prove its own bound attachments."""


def require(condition: bool) -> None:
    if not condition:
        raise BindingError('fetch bindings do not name complete actual owners')


def _id(value: Any) -> bool:
    return type(value) is int and 0 <= value < 2**32


def _ordinal(value: Any) -> bool:
    return type(value) is int and value >= 0


def _digest(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(c in '0123456789abcdef' for c in value)
    )


def _params(fetch: dict[str, Any]) -> list[dict[str, Any]]:
    plan = fetch['plan']
    require(
        isinstance(plan, dict)
        and plan.get('plan') == 'sql'
        and {'plan', 'expect', 'statements'}
        <= set(plan)
        <= {'plan', 'expect', 'statements', 'result_conversions'}
        and isinstance(plan['statements'], list)
    )
    params = []
    for statement in plan['statements']:
        require(
            isinstance(statement, dict)
            and set(statement) == {'dialect', 'text', 'params'}
            and isinstance(statement['text'], str)
            and isinstance(statement['dialect'], str)
            and isinstance(statement['params'], list)
        )
        for param in statement['params']:
            require(
                isinstance(param, dict)
                and param.get('param') in ('key_set', 'text', 'value')
            )
            require(
                set(param) == {'param', 'keys', 'relation'}
                if param['param'] == 'key_set'
                else set(param) == {'param', 'attachment'}
                if param['param'] == 'text'
                else set(param) == {'param', 'value'}
            )
            params.append(param)
    return params


def _bound_id(param: dict[str, Any]) -> int | None:
    if param['param'] == 'key_set':
        return param['keys']['attachment']
    if param['param'] == 'text':
        return param['attachment']
    return None


def _request(request: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(request)
    if result.get('request') in ('lookup', 'neighbors'):
        require(_id(result['seeds']['attachment']))
        result['seeds']['attachment'] = 0
    return result


def _normal_plan(plan: dict[str, Any], ids: dict[int, int]) -> dict[str, Any]:
    result = copy.deepcopy(plan)
    for statement in result['statements']:
        for param in statement['params']:
            at = _bound_id(param)
            if at is not None:
                require(at in ids)
                if param['param'] == 'key_set':
                    param['keys']['attachment'] = ids[at]
                else:
                    param['attachment'] = ids[at]
    return result


def _checked(
    fetch: dict[str, Any],
    held: dict[tuple[int, int], Any],
    seen: set[tuple[int, int]],
) -> Any:
    require(
        isinstance(fetch, dict)
        and {'request', 'plan'}
        <= set(fetch)
        <= {'request', 'plan', 'purpose', 'bindings'}
    )
    facts = fetch.get('bindings')
    if facts is None:
        return None
    require(
        isinstance(facts, dict)
        and set(facts) == {'available', 'run', 'fetch', 'owners', 'witness'}
    )
    require(
        type(facts['available']) is bool
        and _ordinal(facts['run'])
        and _id(facts['fetch'])
        and facts['fetch'] > 0
    )
    identity = facts['run'], facts['fetch']
    require(identity not in seen)
    seen.add(identity)
    if not facts['available']:
        require(facts['owners'] == [] and facts['witness'] is None)
        return None
    params = _params(fetch)
    ids = list(
        dict.fromkeys(_bound_id(p) for p in params if _bound_id(p) is not None)
    )
    require(all(_id(at) for at in ids))
    owners = facts['owners']
    require(isinstance(owners, list) and len(owners) == len(ids))
    by_id = {}
    for owner in owners:
        require(
            isinstance(owner, dict)
            and set(owner)
            == {'attachment', 'fields', 'metadata', 'rows', 'values'}
        )
        at = owner['attachment']
        require(_id(at) and at not in by_id and at in ids)
        require(
            _ordinal(owner['rows'])
            and isinstance(owner['fields'], list)
            and isinstance(owner['metadata'], dict)
        )
        require(
            all(
                type(k) is str and type(v) is str
                for k, v in owner['metadata'].items()
            )
        )
        for field in owner['fields']:
            require(
                isinstance(field, dict)
                and {'name', 'type', 'nullable'}
                <= set(field)
                <= {'name', 'type', 'nullable', 'metadata'}
            )
            require(
                type(field['name']) is str
                and isinstance(field['type'], dict)
                and type(field['nullable']) is bool
            )
            require(
                'metadata' not in field
                or isinstance(field['metadata'], dict)
                and all(
                    type(k) is str and type(v) is str
                    for k, v in field['metadata'].items()
                )
            )
        value = owner['values']
        require(
            isinstance(value, dict)
            and set(value)
            == {'columns', 'types', 'ordered', 'unordered', 'unencodable'}
        )
        require(
            value['columns'] == [field['name'] for field in owner['fields']]
            and value['types'] is None
        )
        require(
            _digest(value['ordered'])
            and _digest(value['unordered'])
            and value['unencodable'] == []
        )
        semantic = {
            key: value for key, value in owner.items() if key != 'attachment'
        }
        previous = held.setdefault((facts['run'], at), semantic)
        require(previous == semantic)
        by_id[at] = semantic
    require(set(by_id) == set(ids))
    for param in params:
        at = _bound_id(param)
        if param['param'] == 'key_set':
            require(param['keys']['columns'] == by_id[at]['fields'])
        elif param['param'] == 'text':
            fields = by_id[at]['fields']
            require(
                len(fields) == 1
                and fields[0]['name'] == 'keys'
                and fields[0]['type'] == {'type': 'utf8'}
                and fields[0]['nullable'] is False
            )
            require(by_id[at]['rows'] == 1)
    canonical_request = _request(fetch['request'])
    canonical_plan = _normal_plan(
        fetch['plan'], dict(zip(ids, range(len(ids)), strict=True))
    )
    witness = facts['witness']
    if witness is not None:
        require(
            isinstance(witness, dict)
            and set(witness)
            == {
                'issued_request',
                'issued_plan',
                'canonical_request',
                'canonical_plan',
            }
        )
        require(
            witness['issued_request'] == fetch['request']
            and witness['issued_plan'] == fetch['plan']
        )
        require(witness['canonical_request'] == canonical_request)
        require(
            fetch['request'].get('request') in ('lookup', 'neighbors')
            and len(ids) == 1
        )
        proposed = witness['canonical_plan']
        require(
            isinstance(proposed, dict) and set(proposed) == set(fetch['plan'])
        )
        require(len(proposed['statements']) == len(fetch['plan']['statements']))
        # Only the independent renderer owns SQL/name differences. All
        # nontext plan data and all typed parameter facts remain exact.
        raw = copy.deepcopy(canonical_plan)
        for actual, selected in zip(
            raw['statements'], proposed['statements'], strict=True
        ):
            require(
                actual['dialect'] in ('sqlite', 'duckdb')
                and actual['dialect'] == selected['dialect']
            )
            require(
                type(selected['text']) is str
                and len(actual['params']) == len(selected['params'])
            )
            actual['text'] = selected['text']
            for a, b in zip(actual['params'], selected['params'], strict=True):
                require(a['param'] != 'text' and a['param'] == b['param'])
                if a['param'] == 'key_set':
                    require(type(b['relation']) is str)
                    a['relation'] = b['relation']
        require(raw == proposed)
        canonical_plan = copy.deepcopy(proposed)
    return {
        'request': canonical_request,
        'plan': canonical_plan,
        'purpose': fetch.get('purpose'),
        'owners': [by_id[at] for at in ids],
    }


def agree(left: list[Any], right: list[Any]) -> bool:
    """Validate new owner evidence; old observations keep exact raw IDs."""
    if len(left) != len(right):
        return False
    caches = ({}, {})
    seen = (set(), set())
    for a, b in zip(left, right, strict=True):
        ca, cb = (
            _checked(value, caches[i], seen[i])
            for i, value in enumerate((a, b))
        )
        if ca is None or cb is None:
            if {k: v for k, v in a.items() if k != 'bindings'} != {
                k: v for k, v in b.items() if k != 'bindings'
            }:
                return False
        elif ca != cb:
            return False
    return True
