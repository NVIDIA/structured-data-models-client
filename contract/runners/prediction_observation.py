# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate actual prediction evidence before projecting common host meaning.

Actual operations, typed inputs and observations stay in the artifact.
Only their checked common role/model is projected for comparison or replay.
This module reads JSON data and never evaluates SDK or user-owned objects.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any


class ObservationError(ValueError):
    """Missing or contradictory prediction evidence, without user values."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ObservationError(message)


def _ordinal(value: Any) -> bool:
    return type(value) is int and value >= 0


def _sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in '0123456789abcdef' for character in value)
    )


def _materialization_bytes(value: Any) -> tuple[bytes, Any]:
    require(
        isinstance(value, dict)
        and set(value) == {'attachment', 'base64', 'sha256', 'parsed'}
        and _ordinal(value['attachment'])
        and _sha256(value['sha256']),
        'materialization bytes observation has no closed identity',
    )
    raw = base64.b64decode(value['base64'], validate=True)
    require(
        hashlib.sha256(raw).hexdigest() == value['sha256'],
        'materialization bytes digest differs',
    )
    parsed = json.loads(raw)
    require(
        _same_jsonl_snapshot(parsed, value['parsed']),
        'materialization parsed data differs from its actual bytes',
    )
    return raw, parsed


def materialization_owner_ids(kind: str, data: Any) -> list[int]:
    """Opaque slots of the finite materialization products, without SDK code."""
    require(
        isinstance(data, dict), 'materialization product has no data object'
    )
    if kind in ('column_meaning', 'lookup_rows', 'request'):
        return []
    require(
        kind
        in (
            'source',
            'population',
            'label_inputs',
            'target_column',
            'label_groups',
            'label_batch',
            'lookup',
        ),
        'unknown materialization owner product',
    )
    ids = [data['owner']]
    if kind == 'target_column':
        ids = materialization_owner_ids('label_inputs', data['input']) + ids
    history = (
        data.get('eligibility')
        if kind == 'label_inputs'
        else data.get('what_if')
        if kind == 'label_groups'
        else None
    )
    if history is not None:
        require(
            isinstance(history, dict)
            and set(history) == {'columns', 'values'}
            and isinstance(history['columns'], list)
            and isinstance(history['values'], list),
            'materialization condition history has no closed shape',
        )
        for column in history['columns']:
            ids.extend(materialization_owner_ids('label_inputs', column))
        for value in history['values']:
            require(
                isinstance(value, dict) and set(value) == {'column', 'owner'},
                'completed condition has no typed owner',
            )
            ids.append(value['owner'])
    require(
        all(_ordinal(key) and key < 2**32 for key in ids),
        'materialization owner is not an attachment id',
    )
    return list(dict.fromkeys(ids))


def _materialization_components(value: dict[str, Any]) -> list[tuple[str, Any]]:
    stage = value['stage']
    if stage == 'task':
        return []
    if stage in (
        'population',
        'source_column',
        'source_fallback',
        'source_failure',
    ):
        return [('source', value['source'])]
    if stage == 'lookup_rows':
        return [('lookup', value['lookup'])]
    if stage == 'label_inputs':
        return [('population', value['population'])] + (
            []
            if value.get('previous') is None
            else [('label_batch', value['previous'])]
        )
    if stage == 'target_column':
        return [('label_inputs', value['inputs'])]
    if stage == 'eligibility_column':
        return [
            ('label_inputs', value['inputs']),
            ('target_column', value['column']),
        ]
    if stage in (
        'condition_column',
        'filtered_target',
        'windowed_target',
        'native_label_groups',
    ):
        return [('target_column', value['column'])]
    if stage == 'restore_label_groups':
        return [('label_groups', value['groups'])]
    if stage == 'what_if_column':
        return [
            ('label_groups', value['groups']),
            ('target_column', value['column']),
        ]
    if stage == 'complete_label_batch':
        return [
            ('population', value['population']),
            ('label_groups', value['groups']),
        ]
    if stage == 'request':
        return [('population', value['population'])] + [
            ('label_batch', batch) for batch in value['batches']
        ]
    raise ObservationError('unknown actual materialization stage')


def _materialization_slots(
    kind: str, data: Any
) -> list[tuple[int, str, Any, bool]]:
    """Actual resource kinds and the metadata each published slot exposes."""
    materialization_owner_ids(kind, data)
    if kind in ('column_meaning', 'lookup_rows', 'request'):
        return []
    slots = [(data['owner'], kind, data, False)]
    if kind == 'target_column':
        slots.extend(_materialization_slots('label_inputs', data['input']))
    history = (
        data.get('eligibility')
        if kind == 'label_inputs'
        else data.get('what_if')
        if kind == 'label_groups'
        else None
    )
    if history is not None:
        for column in history['columns']:
            slots.extend(_materialization_slots('label_inputs', column))
        slots.extend(
            (value['owner'], 'target_column', value['column'], True)
            for value in history['values']
        )
    return slots


def _materialization_metadata(kind: str, data: Any) -> Any:
    """Remove only explicitly rebound IDs; keep every observed meaning field."""
    out = {key: value for key, value in data.items() if key != 'owner'}
    if kind == 'target_column':
        out['input'] = _materialization_metadata('label_inputs', data['input'])
        out['condition_values'] = [
            {
                'column': value['column'],
                'converted': {'encoding': value['converted']['encoding']},
            }
            for value in data['condition_values']
        ]
    name = (
        'eligibility'
        if kind == 'label_inputs'
        else 'what_if'
        if kind == 'label_groups'
        else None
    )
    if name is not None and data.get(name) is not None:
        history = data[name]
        out[name] = {
            'columns': [
                _materialization_metadata('label_inputs', column)
                for column in history['columns']
            ],
            'values': [
                {'column': value['column']} for value in history['values']
            ],
        }
    if kind == 'label_groups' and data.get('restored') is not None:
        out['restored'] = {'encoding': data['restored']['encoding']}
    return out


def _materialization_same_slot(
    left: tuple[Any, ...], right: tuple[Any, ...]
) -> bool:
    _, kind, data, partial = left
    _, other_kind, other, other_partial = right
    if kind != other_kind:
        return False
    if partial or other_partial:
        return kind == 'target_column' and _same_jsonl_snapshot(
            data if partial else data['input']['target'],
            other if other_partial else other['input']['target'],
        )
    return _same_jsonl_snapshot(
        _materialization_metadata(kind, data),
        _materialization_metadata(kind, other),
    )


def _materialization_lifecycle(record: dict[str, Any]) -> None:
    events = record['events']
    require(isinstance(events, list), 'materialization lifecycle is absent')
    terminal = {
        'done': 'done',
        'error': 'core_error',
        'cancelled': 'cancelled',
        'cancel_failed': 'cancel_failed',
    }.get(record['completion'])
    terminals = [
        (at, event['event'])
        for at, event in enumerate(events)
        if event['event']
        in ('done', 'core_error', 'cancelled', 'cancel_failed')
    ]
    require(
        terminals == []
        if terminal is None
        else len(terminals) == 1 and terminals[0][1] == terminal,
        'materialization lifecycle has missing or contradictory completion',
    )
    require(
        (record['result'] is not None) == (terminal == 'done')
        and (record['error'] is not None) == (terminal == 'core_error'),
        'materialization lifecycle disagrees with its actual result/error',
    )
    reads = [
        (at, event['attachment'])
        for at, event in enumerate(events)
        if event['event'] == 'value_taken'
    ]
    require(
        [key for _, key in reads]
        == [value['attachment'] for value in record['value_reads']]
        and all(terminal == 'done' and at > terminals[0][0] for at, _ in reads),
        'materialization value take precedes or differs from its actual Done',
    )


def _materialization_state(value: Any) -> tuple[dict[str, Any], list[int]]:
    require(
        isinstance(value, dict)
        and set(value) <= {'state', 'query'}
        and 'state' in value,
        'nested materialized result has no closed shape',
    )
    state = value['state']
    require(
        isinstance(state, dict)
        and {'version', 'payload', 'query'}
        <= set(state)
        <= {'version', 'payload', 'query', 'forecast', 'batches', 'ranking'}
        and _ordinal(state['version']),
        'nested materialized state has no closed shape',
    )
    ids = [state['payload'], state['query']]
    for batch in state.get('batches') or []:
        require(
            isinstance(batch, dict)
            and {'payload', 'rows'}
            <= set(batch)
            <= {'payload', 'rows', 'request_id'}
            and _ordinal(batch['rows']),
            'materialized batch is malformed',
        )
        ids.append(batch['payload'])
    for part in state.get('forecast') or []:
        require(
            isinstance(part, dict)
            and {'step', 'payload', 'query'}
            <= set(part)
            <= {'step', 'payload', 'query', 'request_id'}
            and _ordinal(part['step']),
            'materialized forecast is malformed',
        )
        ids.extend([part['payload'], part['query']])
    ranking = state.get('ranking')
    if ranking is not None:
        require(
            isinstance(ranking, dict)
            and set(ranking) == {'top_k', 'candidate', 'value', 'parts'},
            'materialized ranking is malformed',
        )
        for part in ranking['parts']:
            require(
                isinstance(part, dict)
                and {'candidate', 'value', 'payload', 'query'}
                <= set(part)
                <= {'candidate', 'value', 'payload', 'query', 'request_id'},
                'materialized ranking part is malformed',
            )
            ids.extend([part['payload'], part['query']])
    require(
        all(_ordinal(key) and key < 2**32 for key in ids),
        'materialized state names no attachment',
    )
    return state, list(dict.fromkeys(ids))


def _materialization_read_origin(
    record: dict[str, Any], owned: dict[int, Any], run: int
) -> Any:
    origin = record.get('origin')
    if origin is None:
        return None
    require(
        isinstance(origin, dict)
        and set(origin) == {'run', 'first', 'failed'}
        and origin['run'] in owned
        and origin['run'] < run,
        'materialization diagnostic has no original source Run',
    )
    original = owned[origin['run']]
    require(
        original['completion'] in ('cancelled', 'cancel_failed'),
        'materialization diagnostic source Run was not cancelled',
    )
    partition = [
        fetch
        for fetch in original['fetches']
        if fetch.get('partition', {}).get('first') == origin['first']
    ]
    require(
        partition
        and all(fetch.get('purpose') == 'source_rows' for fetch in partition),
        'materialization diagnostic names no actual source partition',
    )
    if origin['failed'] is not None:
        require(
            any(fetch['id'] == origin['failed'] for fetch in partition),
            'materialization diagnostic names no actual failed member',
        )
    return origin


def _materializations(
    core: dict[str, Any], operations: list[str]
) -> dict[str, Any]:
    records, guards = {}, {}
    for name, operation, into in (
        ('materialization_preparations', 'prepare_materialize', records),
        ('materialization_guards', 'prepare_frames', guards),
    ):
        for record in core.get(name, []):
            run = record['run']
            require(
                _ordinal(run)
                and run < len(operations)
                and operations[run] == operation
                and run not in into,
                'materialization Run is absent or repeated',
            )
            require(
                record['operation']['op'] == operation
                and record['operation']['input']
                == record['input']['attachment'],
                'materialization input has another operation owner',
            )
            _raw, parsed = _materialization_bytes(record['input'])
            require(
                isinstance(parsed, dict),
                'materialization input is not a domain object',
            )
            require(
                not (
                    record['error'] is not None and record['result'] is not None
                ),
                'materialization both answered and errored',
            )
            into[run] = {**record, '_input': parsed, '_value': None}
            values = record['value_reads']
            require(
                isinstance(values, list) and len(values) <= 1,
                'materialization product was read repeatedly',
            )
            if values:
                result = record['result']
                require(
                    isinstance(result, dict)
                    and result['op'] == operation
                    and result['value'] == values[0]['attachment']
                    and record['completion'] == 'done',
                    'materialization product lacks its actual Done',
                )
                _, product = _materialization_bytes(values[0])
                require(
                    isinstance(product, dict)
                    and set(product) == {'value', 'data'},
                    'materialization product has no tagged data',
                )
                into[run]['_value'] = product
    require(
        set(records)
        == {
            at
            for at, op in enumerate(operations)
            if op == 'prepare_materialize'
        },
        'materialization operation has no raw input journal',
    )
    entries = core.get('materialization_entries', [])
    require(
        isinstance(entries, list), 'materialization entries are not observed'
    )
    owners, inputs, projected = {}, {}, []
    fetches = [dict(fetch) for fetch in core['fetches']]
    fetch_positions: set[int] = set()
    all_records = {**records, **guards}
    for at, entry in enumerate(entries):
        require(
            isinstance(entry, dict)
            and set(entry)
            == {
                'ordinal',
                'parent',
                'prediction_parent',
                'runs',
                'completion',
                'terminal',
                'failure',
            }
            and _ordinal(entry['ordinal'])
            and entry['ordinal'] == at
            and (
                entry['parent'] is None
                or _ordinal(entry['parent'])
                and entry['parent'] < at
            ),
            'materialization entry has no closed nesting identity',
        )
        require(
            entry['prediction_parent'] is None
            or _ordinal(entry['prediction_parent'])
            and entry['prediction_parent'] < len(core.get('predictions', [])),
            'materialization has no actual prediction parent',
        )
        runs = entry['runs']
        require(
            isinstance(runs, list)
            and all(_ordinal(run) and run < len(operations) for run in runs)
            and runs == sorted(set(runs)),
            'materialization entry has incoherent Run order',
        )
        owned = {
            run: record
            for run, record in all_records.items()
            if record.get('entry') == at
        }
        require(
            set(owned) <= set(runs),
            'materialization record is outside its entry',
        )
        for run in owned:
            require(
                run not in owners, 'materialization Run belongs to two entries'
            )
            owners[run] = at
        tasks = [
            run
            for run, record in owned.items()
            if run in records and record['_input'].get('stage') == 'task'
        ]
        if not tasks:
            require(
                not owned
                and entry['terminal'] is None
                and entry['failure'] is None
                and entry['completion'] == 'raised',
                'materialization entry has no actual Task',
            )
            projected.append(
                {**entry, 'native': False, 'position': None, 'through': None}
            )
            continue
        require(
            len(tasks) == 1 and tasks[0] == min(owned),
            'materialization entry has multiple or late Tasks',
        )
        task = records[tasks[0]]['_input']['request']
        snapshots = {}
        for observation in records[tasks[0]]['input_reads']:
            key = observation['attachment']
            require(
                key not in snapshots, 'materialization input was observed twice'
            )
            if 'scalars' in observation:
                require(
                    set(observation) == {'attachment', 'scalars'}
                    and isinstance(observation['scalars'], list)
                    and all(
                        isinstance(item, dict)
                        for item in observation['scalars']
                    ),
                    'materialization indices are not actual scalar observations',
                )
                snapshots[key] = observation['scalars']
            else:
                _, snapshots[key] = _materialization_bytes(observation)
        for name in (
            'schema',
            'query',
            'indices',
            'values',
            'candidates',
            'held',
        ):
            if task.get(name) is not None:
                require(
                    task[name] in snapshots,
                    'materialization Task omits its actual referenced input',
                )
        inputs[at] = {
            'op': 'materialize',
            'provider': task['provider'],
            'schema': snapshots[task['schema']],
            'query': snapshots[task['query']],
            'indices': None
            if task.get('indices') is None
            else snapshots[task['indices']],
            'anchor_time': task['anchor_time'],
            'seed': task['seed'],
        }
        for run, record in owned.items():
            _materialization_lifecycle(record)
            origin = _materialization_read_origin(record, owned, run)
            if run in guards:
                require(
                    record['_input'].get('stage')
                    in ('assess_failure', 'report_failure'),
                    'unrelated frame preparation claimed as materialization guard',
                )
                if record['_input']['stage'] == 'report_failure':
                    assessment = record['_input']['report']['assessment']
                    require(
                        any(
                            earlier < run
                            and earlier in guards
                            and previous['_value'] is not None
                            and previous['_value']['value']
                            == 'failure_assessment'
                            and _same_jsonl_snapshot(
                                previous['_value']['data'], assessment
                            )
                            for earlier, previous in owned.items()
                        ),
                        'materialization report has no actual prior assessment',
                    )
                continue
            stage = record['_input']['stage']
            product = record['_value']
            if product is not None:
                allowed = {
                    'task': ('source', 'request'),
                    'population': ('population',),
                    'lookup_rows': ('lookup_rows',),
                    'source_column': ('column_meaning',),
                    'source_fallback': ('column_meaning',),
                    'label_inputs': ('label_inputs',),
                    'eligibility_column': ('label_inputs',),
                    'target_column': ('target_column',),
                    'condition_column': ('target_column',),
                    'filtered_target': ('target_column',),
                    'windowed_target': ('target_column',),
                    'native_label_groups': ('label_groups',),
                    'restore_label_groups': ('label_groups',),
                    'what_if_column': ('label_groups',),
                    'complete_label_batch': ('label_batch',),
                    'request': ('request',),
                }
                require(
                    product['value'] in allowed.get(stage, ()),
                    'materialization stage returned another product kind',
                )
            if (
                stage == 'request'
                and record['_input'].get('values') is not None
            ):
                late = record['_input']['values']
                require(
                    any(
                        item['attachment'] == late and 'base64' in item
                        for item in record['input_reads']
                    ),
                    'materialization request omits its actual late values',
                )
            for item in record['input_reads']:
                if 'base64' in item:
                    _materialization_bytes(item)
            for take in record['owner_takes']:
                if take['origin'] == 'result':
                    require(
                        product is not None
                        and take['attachment']
                        in materialization_owner_ids(
                            product['value'], product['data']
                        ),
                        'materialization take is not a declared opaque output',
                    )
                else:
                    require(
                        take['origin'] == 'fetch'
                        and any(
                            fetch['id'] == take['fetch_id']
                            and fetch.get('lookup', {}).get('owner')
                            == take['attachment']
                            for fetch in record['fetches']
                        ),
                        'materialization take has no actual lookup fetch',
                    )
            if stage == 'source_failure':
                require(
                    origin is not None and origin['failed'] is not None,
                    'source diagnostic has no exact failed read association',
                )
                failed = next(
                    fetch
                    for fetch in owned[origin['run']]['fetches']
                    if fetch['id'] == origin['failed']
                )['request']
                read = record['_input']['read']
                require(
                    read['kind'] == failed['request']
                    and read['source'] == failed['table']
                    and read['columns'] == failed['columns'],
                    'source diagnostic differs from its actual failed request',
                )
            expected = set(
                key
                for kind, data in _materialization_components(record['_input'])
                for key in materialization_owner_ids(kind, data)
            )
            binds = record['owner_binds']
            require(
                isinstance(binds, list)
                and len(binds) == len(expected)
                and {bind['attachment'] for bind in binds} == expected,
                'materialization has no complete actual owner binds',
            )
            for bind in binds:
                producer = bind['producer']
                require(
                    isinstance(producer, dict)
                    and producer['run'] in owned
                    and producer['run'] < run,
                    'materialization resource has a foreign or future producer',
                )
                source = owned[producer['run']]
                take = {
                    key: value
                    for key, value in producer.items()
                    if key != 'run'
                }
                require(
                    take in source['owner_takes'],
                    'materialization bind has no actual producer take',
                )
                if producer['origin'] == 'result':
                    product = source['_value']
                    require(
                        product is not None
                        and producer['attachment']
                        in materialization_owner_ids(
                            product['value'], product['data']
                        ),
                        'materialization producer does not declare this owner',
                    )
                    producer_slots = _materialization_slots(
                        product['value'], product['data']
                    )
                else:
                    require(
                        producer['origin'] == 'fetch',
                        'unknown materialization resource origin',
                    )
                    fetched = next(
                        (
                            fetch
                            for fetch in source['fetches']
                            if fetch['id'] == producer['fetch_id']
                        ),
                        None,
                    )
                    require(
                        fetched is not None
                        and fetched.get('purpose') == 'source_rows'
                        and fetched.get('lookup', {}).get('owner')
                        == producer['attachment']
                        and fetched['lookup']['read'] == producer['fetch_id'],
                        'lookup owner has no actual issuing fetch',
                    )
                    producer_slots = _materialization_slots(
                        'lookup', fetched['lookup']
                    )
                wanted = [
                    slot
                    for kind, data in _materialization_components(
                        record['_input']
                    )
                    for slot in _materialization_slots(kind, data)
                    if slot[0] == bind['attachment']
                ]
                actual = [
                    slot
                    for slot in producer_slots
                    if slot[0] == producer['attachment']
                ]
                require(
                    actual
                    and wanted
                    and all(
                        any(
                            _materialization_same_slot(slot, produced)
                            for produced in actual
                        )
                        for slot in wanted
                    ),
                    'materialization resource kind or meaning differs from its actual producer',
                )
            cursor = 0
            for event in record['events']:
                if event['event'] != 'fetch':
                    continue
                positions = event['positions']
                issued = record['fetches'][
                    cursor : cursor + len(event['requests'])
                ]
                require(
                    [fetch['id'] for fetch in issued] == event['requests']
                    and len(positions) == len(issued),
                    'materialization fetch event does not retain its actual partition',
                )
                for position, fetched in zip(positions, issued, strict=True):
                    require(
                        _ordinal(position)
                        and position < len(fetches)
                        and position not in fetch_positions,
                        'materialization fetch has no unique actual global position',
                    )
                    fetch_positions.add(position)
                    actual = {
                        key: fetched[key]
                        for key in ('request', 'plan', 'purpose')
                        if key in fetched
                    }
                    require(
                        core['fetches'][position] == actual,
                        'materialization fetch differs from its actual global trace',
                    )
                    if fetched.get('purpose') in (
                        'source_rows',
                        'describe_result',
                    ):
                        fetches[position] = {
                            key: value
                            for key, value in actual.items()
                            if key != 'purpose'
                        }
                cursor += len(issued)
            require(
                cursor == len(record['fetches']),
                'materialization fetches have no lifecycle events',
            )
        terminal, failure = entry['terminal'], entry['failure']
        materialized, complete_product, error = None, None, None
        if entry['completion'] == 'returned':
            require(
                failure is None
                and isinstance(terminal, dict)
                and set(terminal) == {'run'}
                and terminal['run'] in owned,
                'materialization returned without its actual terminal Run',
            )
            record = owned[terminal['run']]
            product = record['_value']
            require(
                record['completion'] == 'done'
                and product is not None
                and product['value'] == 'request',
                'materialization terminal is not an actual nested request',
            )
            require(
                record['_input']['stage'] in ('task', 'request'),
                'a helper cannot stand in for the materialized request',
            )
            state, ids = _materialization_state(product['data'])
            outputs = record['output_reads']
            require(
                len(outputs) == len(ids)
                and {item['attachment'] for item in outputs} == set(ids),
                'materialization terminal state has incomplete actual takes',
            )
            taken = {
                item['attachment']: _materialization_bytes(item)
                for item in outputs
            }
            payload = json.loads(
                taken[state['payload']][0],
                parse_int=lambda token: {'number': token},
                parse_float=lambda token: {'number': token},
            )
            materialized = {
                'version': state['version'],
                'query': taken[state['query']][1],
                'payload': payload,
            }

            def resolved(part: dict[str, Any]) -> dict[str, Any]:
                out = dict(part)
                out['payload'] = json.loads(
                    taken[part['payload']][0],
                    parse_int=lambda token: {'number': token},
                    parse_float=lambda token: {'number': token},
                )
                if 'query' in part:
                    out['query'] = taken[part['query']][1]
                return out

            completed_state = resolved(state)
            for field in ('batches', 'forecast'):
                if state.get(field) is not None:
                    completed_state[field] = [
                        resolved(part) for part in state[field]
                    ]
            if state.get('ranking') is not None:
                completed_state['ranking'] = {
                    **state['ranking'],
                    'parts': [
                        resolved(part) for part in state['ranking']['parts']
                    ],
                }
            complete_product = {**product['data'], 'state': completed_state}
        else:
            require(
                entry['completion'] == 'raised'
                and terminal is None
                and (
                    failure is None
                    or isinstance(failure, dict)
                    and set(failure) == {'run', 'relation'}
                    and failure['run'] in owned
                ),
                'materialization error has no selected actual terminal association',
            )
            record = None if failure is None else owned[failure['run']]
            if failure is None:
                # Actual Task, input, resource and lifecycle observations
                # above still prove the work reached. An escaping host
                # failure need not be a successfully presented Core error.
                # Its independent public outcome remains authoritative.
                pass
            elif failure['relation'] == 'native':
                product = record['_value']
                require(
                    failure['run'] in guards
                    and record['completion'] == 'done'
                    and product is not None
                    and product['value'] == 'failure_assessment'
                    and product['data']['category'] == 'pass_through',
                    'selected native materialization failure has no actual pass-through assessment',
                )
            else:
                require(
                    failure['relation'] in ('selected', 'cause', 'refused')
                    and record['completion'] == 'error'
                    and record['error'] is not None,
                    'materialization failure lacks its actual Core error',
                )
                error = record['error']
        projected.append(
            {
                **entry,
                'native': True,
                'position': None,
                'through': None,
                '_task': tasks[0],
                '_materialized': materialized,
                '_product': complete_product,
                '_error': error,
            }
        )
    for run, record in all_records.items():
        if 'entry' in record:
            require(
                owners.get(run) == record['entry'],
                'materialization record has no entry owner',
            )
    return {
        'records': all_records,
        'entries': projected,
        'owners': owners,
        'inputs': inputs,
        'fetches': fetches,
    }


def _ordered_json(value: bytes | str) -> Any:
    def nonfinite(_token: str) -> Any:
        raise ObservationError('prediction input has a nonfinite JSON value')

    return json.loads(
        value,
        parse_int=lambda token: ('number', token),
        parse_float=lambda token: ('number', token),
        object_pairs_hook=lambda members: ('object', members),
        parse_constant=nonfinite,
    )


def _same_snapshot(left: Any, right: Any) -> bool:
    return _compare_snapshot(left, right, ordered_objects=True)


def _same_jsonl_snapshot(left: Any, right: Any) -> bool:
    """Canonical JSONL can sort DTO object keys, but never its arrays."""
    return _compare_snapshot(left, right, ordered_objects=False)


def _compare_snapshot(left: Any, right: Any, *, ordered_objects: bool) -> bool:
    pending = [(left, right)]
    while pending:
        left, right = pending.pop()
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            if (
                list(left) != list(right)
                if ordered_objects
                else left.keys() != right.keys()
            ):
                return False
            pending.extend((left[key], right[key]) for key in left)
        elif isinstance(left, list):
            if len(left) != len(right):
                return False
            pending.extend(zip(left, right, strict=True))
        elif left != right:
            return False
    return True


def _reply_obligation(reply: Any) -> int:
    if reply is None:
        return 0
    require(isinstance(reply, dict), 'model reply observation is malformed')
    outcome = reply.get('outcome')
    if outcome in ('response', 'failed'):
        require(
            set(reply) == {'outcome', 'status'}
            and _ordinal(reply['status'])
            and reply['status'] <= 65535,
            'reported model response has no actual status',
        )
        return int(outcome == 'response')
    require(
        reply == {'outcome': 'transport_error'},
        'unknown model reply observation',
    )
    return 0


def model_reply(outcome: dict[str, Any]) -> dict[str, Any]:
    """The actual reported outcome/status, without provider diagnostics."""
    kind = outcome['outcome']
    fact = {
        'outcome': kind,
        **(
            {'status': outcome['status']}
            if kind == 'response'
            else {'status': outcome['answer']['status']}
            if kind == 'failed'
            else {}
        ),
    }
    _reply_obligation(fact)
    return fact


def check_receipts(
    attempts: list[dict[str, Any]], receipts: list[dict[str, Any]]
) -> None:
    """Require a monotone receipt partition, without inventing retry counts."""
    bodies: dict[str, Any] = {}
    required = []
    for attempt in attempts:
        digest = attempt['body_sha256']
        require(_sha256(digest), 'attempted body hash is unobserved')
        if digest in bodies:
            require(
                _same_snapshot(bodies[digest], attempt['body']),
                'one posted hash carries contradictory body observations',
            )
        else:
            bodies[digest] = attempt['body']
        required.append(_reply_obligation(attempt['reply']))
    hashes = []
    for receipt in receipts:
        digest = receipt.get('sha256')
        require(
            _sha256(digest)
            and digest in bodies
            and _same_snapshot(bodies[digest], receipt['parsed']),
            'endpoint receipt differs from every actual attempted body/hash',
        )
        hashes.append(digest)
    # reachable[j] means earlier attempts can account for exactly the first
    # j receipts. A response needs one; other observed outcomes may have none.
    reachable = [True] + [False] * len(hashes)
    for attempt, minimum in zip(attempts, required, strict=True):
        following = [minimum == 0 and reachable[0]]
        for at, digest in enumerate(hashes, start=1):
            following.append(
                (minimum == 0 and reachable[at])
                or (
                    digest == attempt['body_sha256']
                    and (reachable[at - 1] or following[at - 1])
                )
            )
        reachable = following
    require(
        reachable[-1],
        'endpoint receipts have no ordered allocation satisfying model responses',
    )


def _known_endpoint(endpoint: Any) -> None:
    require(
        endpoint is None
        or (
            isinstance(endpoint, dict)
            and set(endpoint) == {'endpoint', 'name', 'max_request_bytes'}
            and endpoint['endpoint'] == 'serving_endpoint'
            and type(endpoint['name']) is str
            and _ordinal(endpoint['max_request_bytes'])
            and endpoint['max_request_bytes'] <= 2**64 - 1
        ),
        'prediction has an unknown model endpoint observation',
    )


def observed_call_body(call: dict[str, Any], encoded: dict[str, Any]) -> None:
    """Bind independently observed input bytes and the actual posted JSON.

    The standard decoder's internal tuple tags distinguish number tokens
    from all JSON objects, including duplicate and ordered object members.
    No serialized request is reconstructed from this comparison product.
    """
    original = base64.b64decode(call['input_body_base64'], validate=True)
    require(
        hashlib.sha256(original).hexdigest() == call['input_body_sha256'],
        'prediction input bytes differ from their observed hash',
    )
    body = call['body']
    if body is None:
        return
    require(
        isinstance(body, dict) and _sha256(call['body_sha256']),
        'prediction posted body/hash is unobserved',
    )
    endpoint = encoded.get('model_endpoint')
    _known_endpoint(endpoint)
    if endpoint is None:
        require(
            call['body_sha256'] == call['input_body_sha256']
            and _same_jsonl_snapshot(
                json.loads(
                    original,
                    parse_int=lambda token: {'number': token},
                    parse_float=lambda token: {'number': token},
                ),
                body,
            ),
            'prediction direct body differs from its actual input bytes',
        )
        payload = body
    else:
        rows = body.get('dataframe_records')
        require(
            set(body) == {'dataframe_records'}
            and isinstance(rows, list)
            and len(rows) == 1
            and isinstance(rows[0], dict)
            and set(rows[0]) == {'request_json'}
            and type(rows[0]['request_json']) is str,
            'prediction serving body has no single request_json row',
        )
        inner = rows[0]['request_json']
        require(
            _ordered_json(original) == _ordered_json(inner),
            'prediction serving request differs from its actual input JSON',
        )
        payload = json.loads(inner)
    metadata = payload.get('metadata') if isinstance(payload, dict) else None
    require(
        isinstance(metadata, dict)
        and metadata.get('request_id') == call['request_id'],
        'prediction posted request ID differs from its actual typed input',
    )


def _model(value: Any) -> str:
    require(
        isinstance(value, dict)
        and value.get('phase')
        in {
            'model_resolved',
            'outputs_resolved',
            'shape_accepted',
            'task_accepted',
            'task_admitted',
            'bound',
        },
        'prediction has no actual typed model preparation',
    )
    model = value.get('model')
    require(
        type(model) is str and bool(model), 'prediction model is unobserved'
    )
    return model


def _parts(prepared: dict[str, Any]) -> list[dict[str, Any]]:
    aggregation = prepared['input']['aggregation']
    kind = aggregation['kind']
    if kind == 'calls':
        groups = [aggregation]
    else:
        require(
            kind in ('steps', 'ranking'), 'unknown final prediction aggregation'
        )
        groups = aggregation['children']
    parts = []
    for group in groups:
        require(
            isinstance(group['parts'], list),
            'prediction parts are not observed',
        )
        if group.get('known_empty', False):
            require(
                not group['parts'] and not group['counted'],
                'known-empty prediction carries model calls',
            )
        else:
            require(
                len(group['parts']) == len(group['counted'])
                and bool(group['parts']),
                'prediction response counts are incomplete',
            )
        parts.extend(group['parts'])
    return parts


def _selected(
    final: dict[str, Any], calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {call['request_id']: call for call in calls}
    selected = []
    for part in _parts(final['input']):
        scope = part['data']['scope']
        call = by_id.get(scope['call_request_id'])
        require(
            call is not None
            and call['scope'] == scope
            and call['body'] is not None,
            'final prediction references an unobserved response scope',
        )
        require(
            part['attribution']['scope'] == scope,
            'final prediction attribution changed its response scope',
        )
        selected.append(call)
    runs = [call['run'] for call in selected]
    require(
        runs == sorted(set(runs)),
        'final prediction reorders or repeats actual calls',
    )
    return selected


def _final(
    final: dict[str, Any],
    records: dict[int, dict[str, Any]],
    calls: list[dict[str, Any]],
    *,
    complete: bool,
) -> dict[str, Any]:
    record = records.get(final['run'])
    require(
        record is not None
        and record['input']['stage'] == 'finish_relational_response'
        and record['input'] == final['input']
        and record['result'] == final['data']
        and record['error'] is None,
        'final prediction differs from its actual Run result',
    )
    selected = _selected(final, calls)
    scopes = {call['request_id']: call['scope'] for call in selected}
    lineage = final['data']['lineage']
    require(isinstance(lineage, list), 'final prediction lineage is unobserved')
    for source in lineage:
        scope = scopes.get(source['call_request_id'])
        require(
            scope is not None
            and _ordinal(source['answer_ordinal'])
            and source['answer_ordinal'] < scope['response_rows'],
            'final prediction lineage names an unobserved call row',
        )
        require(
            source['source_row'] is None or _ordinal(source['source_row']),
            'final prediction native source position is malformed',
        )
    require(
        final['metadata'] is None or isinstance(final['metadata'], dict),
        'final prediction metadata observation is malformed',
    )
    table = final['result']
    if complete:
        require(
            isinstance(final['metadata'], dict) and isinstance(table, dict),
            'final prediction metadata or Arrow result is unobserved',
        )
    if table is not None:
        require(
            isinstance(table, dict),
            'final prediction Arrow observation is malformed',
        )
        names = [column['name'] for column in table['columns']]
        require(
            len(names) == len(set(names)) and 'answer_index' in names,
            'final prediction Arrow columns are incomplete',
        )
        rows = table['rows']
        require(
            len(rows) == len(lineage)
            and all(len(row) == len(names) for row in rows),
            'final prediction Arrow rows disagree with lineage',
        )
        at = names.index('answer_index')
        require(
            sorted(int(row[at]['i64']) for row in rows)
            == list(range(len(rows))),
            'final prediction Arrow indices disagree with lineage',
        )
    predicted = final['data']['predicted']
    bodies = [call['body'] for call in selected]
    return {
        'body': bodies[-1] if bodies else None,
        **({'calls': bodies} if len(bodies) > 1 else {}),
        'model': predicted['model'],
        'request_id': predicted['request_id'],
        'metadata': final['metadata'],
        'result': table,
    }


def _below(child: Any, parent: int, entries: list[dict[str, Any]]) -> bool:
    seen = set()
    while child is not None:
        require(
            _ordinal(child) and child < len(entries) and child not in seen,
            'prediction parent chain is malformed',
        )
        if child == parent:
            return True
        seen.add(child)
        child = entries[child]['parent']
    return False


def _entry(
    entry: dict[str, Any],
    ordinal: int,
    records: dict[int, dict[str, Any]],
    operations: list[str],
    entries: list[dict[str, Any]],
    materialization_errors: dict[int, tuple[int, dict[str, Any]]],
) -> dict[str, Any]:
    require(
        entry['ordinal'] == ordinal,
        'prediction entries are not in invocation order',
    )
    parent = entry['parent']
    require(
        parent is None or (_ordinal(parent) and parent < ordinal),
        'prediction parent is not an earlier actual entry',
    )
    argument = entry['argument']
    require(
        argument == {'form': 'unobserved'}
        or (
            set(argument) == {'form', 'value'}
            and argument['form'] == 'exact_string'
            and type(argument['value']) is str
        ),
        'prediction model argument observation is malformed',
    )
    require(
        entry['completion'] in ('returned', 'raised') and 'carried' in entry,
        'prediction entry remains active',
    )
    runs = entry['runs']
    require(
        isinstance(runs, list)
        and all(_ordinal(run) and run < len(operations) for run in runs)
        and runs == sorted(set(runs)),
        'prediction actual Run membership is malformed',
    )
    owned = {
        run: record
        for run, record in records.items()
        if record['entry'] == ordinal
    }
    require(
        set(owned)
        == {run for run in runs if operations[run] == 'prepare_request'},
        'prediction entry and actual preparation owners disagree',
    )
    models = set()
    calls = entry['calls']
    require(isinstance(calls, list), 'prediction actual calls are unobserved')
    call_runs = [call['run'] for call in calls]
    require(
        call_runs == sorted(set(call_runs)),
        'prediction calls repeat or change actual order',
    )
    require(
        set(call_runs)
        == {
            run
            for run, record in owned.items()
            if record['input']['stage'] == 'call_relational'
        },
        'prediction call journal differs from actual typed call starts',
    )
    ids = [call['request_id'] for call in calls]
    require(
        all(type(value) is str and bool(value) for value in ids)
        and len(ids) == len(set(ids)),
        'prediction call request IDs are malformed or repeated',
    )
    for record in owned.values():
        value = record['result']
        if isinstance(value, dict) and 'phase' in value:
            models.add(_model(value))
    for call in calls:
        record = owned[call['run']]
        encoded = record['input']['input']
        require(
            encoded['request_id'] == call['request_id'],
            'prediction input request ID differs from its actual call',
        )
        body, scope = call['body'], call['scope']
        _reply_obligation(call['reply'])
        observed_call_body(call, encoded)
        if body is None:
            require(
                entry['completion'] == 'raised'
                and scope is None
                and call['body_sha256'] is None
                and call['reply'] is None
                and record['result'] is None,
                'prediction call body is unobserved',
            )
            continue
        models.add(_model(encoded['call']))
        if scope is None:
            require(
                entry['completion'] == 'raised' and record['result'] is None,
                'prediction completed call has no actual response scope',
            )
        else:
            require(
                isinstance(record['result'], dict)
                and record['result']['scope'] == scope
                and scope['call_request_id'] == call['request_id']
                and _ordinal(scope['response_rows'])
                and call['reply'] is not None
                and call['reply']['outcome'] == 'response'
                and 200 <= call['reply']['status'] < 300,
                'prediction response scope differs from its actual call result',
            )
    results = entry['results']
    result_runs = [final['run'] for final in results]
    require(
        result_runs == sorted(set(result_runs)),
        'prediction result Runs repeat or change order',
    )
    require(
        set(result_runs)
        == {
            run
            for run, record in owned.items()
            if record['input']['stage'] == 'finish_relational_response'
            and record['result'] is not None
        },
        'prediction final result journal differs from actual typed completions',
    )
    rendered = []
    for final in results:
        rendered.append(
            _final(
                final, owned, calls, complete=entry['completion'] == 'returned'
            )
        )
        models.add(_model(final['input']['input']['bound']))
    require(
        len(models) <= 1,
        'prediction canonical model preparations contradict each other',
    )
    failure = entry.get('failure')
    if failure is not None:
        failed = records.get(failure['run'])
        supplied = materialization_errors.get(failure['run'])
        failure_owner = (
            failed['entry']
            if failed is not None
            else supplied[0]
            if supplied is not None
            else None
        )
        failure_error = (
            failed['error']
            if failed is not None
            else supplied[1]
            if supplied is not None
            else None
        )
        require(
            entry['completion'] == 'raised'
            and failure_owner is not None
            and _below(failure_owner, ordinal, entries)
            and failure['relation'] in ('selected', 'cause', 'refused')
            and failure_error is not None,
            'prediction terminal error has no actual native presentation association',
        )
    native = bool(models) or (failure is not None and failure['run'] in owned)
    final = entry['final']
    if entry['completion'] == 'returned' and native:
        require(
            _ordinal(final) and final == len(results) - 1,
            'prediction returned without its final typed result',
        )
        require(
            [call['request_id'] for call in _selected(results[final], calls)]
            == ids,
            'prediction final result omits actual paid calls',
        )
    else:
        require(
            final is None,
            'a failed or unprepared entry claims a final prediction',
        )
    return {
        'ordinal': ordinal,
        'parent': parent,
        'model': next(iter(models)) if models else None,
        'native': native,
        'completion': entry['completion'],
        'failure': failure,
        '_rendered': rendered[final] if final is not None else None,
        '_calls': calls,
        '_results': rendered,
    }


def _partial_entry(
    entry: dict[str, Any],
    rendered: list[dict[str, Any]],
    carried: tuple[dict[str, Any], dict[str, Any]] | None,
) -> dict[str, Any]:
    results = []
    positions = {
        call['request_id']: at for at, call in enumerate(entry['calls'])
    }
    for raw, value in zip(entry['results'], rendered, strict=True):
        selected = _selected(raw, entry['calls'])
        results.append(
            {
                **value,
                'aggregation': raw['input']['input']['aggregation']['kind'],
                'outputs': raw['data']['predicted']['outputs'],
                'call_indices': [
                    positions[call['request_id']] for call in selected
                ],
                'scopes': [
                    {
                        'call_index': positions[call['request_id']],
                        **{
                            key: call['scope'][key]
                            for key in ('model', 'request_id', 'response_rows')
                        },
                    }
                    for call in selected
                ],
                'lineage': [
                    {
                        'call_index': positions[source['call_request_id']],
                        'answer_ordinal': source['answer_ordinal'],
                        'source_row': source['source_row'],
                    }
                    for source in raw['data']['lineage']
                ],
            }
        )
    answered = None
    if carried is not None:
        raw, value = carried
        aggregation = raw['input']['input']['aggregation']
        groups = (
            [aggregation]
            if aggregation['kind'] == 'calls'
            else aggregation['children']
        )
        rows = 0
        for group in groups:
            require(
                all(
                    item is None or _ordinal(item) for item in group['counted']
                ),
                'carried prediction has malformed actual row counts',
            )
            rows += sum(item for item in group['counted'] if item is not None)
        # This describes the proved successful SDK handover, not a second
        # readback of arbitrary public error.details or the public result.
        answered = {
            'rows': rows,
            **{
                key: value[key]
                for key in ('result', 'model', 'request_id', 'metadata')
            },
            'outputs': raw['data']['predicted']['outputs'],
        }
    return {'observed': True, 'answered': answered, 'results': results}


def _legacy_partial(answered: Any, observed: bool) -> dict[str, Any]:
    if answered is not None:
        require(
            isinstance(answered, dict)
            and set(answered)
            == {'rows', 'result', 'model', 'request_id', 'metadata', 'outputs'}
            and _ordinal(answered['rows'])
            and isinstance(answered['result'], dict)
            and isinstance(answered['result']['columns'], list)
            and isinstance(answered['result']['rows'], list)
            and type(answered['model']) is str
            and type(answered['request_id']) is str
            and isinstance(answered['metadata'], dict)
            and isinstance(answered['outputs'], list)
            and all(type(value) is str for value in answered['outputs']),
            'legacy carried partial result is malformed',
        )
    return {'observed': observed, 'answered': answered, 'results': None}


def _legacy_product(value: Any) -> None:
    if value is None:
        return
    require(
        isinstance(value, dict)
        and {'body', 'model', 'request_id', 'metadata', 'result'} <= set(value)
        and set(value)
        <= {
            'body',
            'calls',
            'model',
            'request_id',
            'metadata',
            'result',
            'outputs',
        }
        and (value['body'] is None or isinstance(value['body'], dict))
        and type(value['model']) is str
        and type(value['request_id']) is str
        and (value['metadata'] is None or isinstance(value['metadata'], dict))
        and (value['result'] is None or isinstance(value['result'], dict)),
        'legacy result is not an observed typed prediction product',
    )
    if 'outputs' in value:
        require(
            isinstance(value['outputs'], list)
            and all(type(item) is str for item in value['outputs']),
            'legacy result outputs are malformed',
        )
    if 'calls' in value:
        require(
            isinstance(value['calls'], list)
            and len(value['calls']) >= 2
            and all(isinstance(item, dict) for item in value['calls'])
            and value['calls'][-1] == value['body'],
            'legacy result call order is malformed',
        )


def _legacy_events(
    core: dict[str, Any],
    operations: list[str],
    inputs: list[dict[str, Any]],
    run_positions: dict[int, int],
    attempts: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Read each legacy Run's products without borrowing its successor."""
    legacy_runs = [run for run, op in enumerate(operations) if op == 'predict']
    legacy_inputs = [record for record in inputs if record['op'] == 'predict']
    require(
        len(legacy_inputs) == len(legacy_runs),
        'legacy prediction inputs lack actual Run owners',
    )
    journal = core.get('legacy_predictions')
    control_observed = (
        'answered' in core
        if journal is None
        else (core.get('capture_capabilities') or {}).get(
            'native_error_control', True
        )
    )
    if journal is None:
        require(
            len(legacy_runs) <= 1,
            'multiple legacy predictions lack Run-local products',
        )
        journal = [
            {
                'run': run,
                'completion': 'error'
                if core.get('error') is not None
                else 'done',
                'result': core.get('predicted'),
                'error': core.get('error'),
                **(
                    {'answered': core['answered']} if 'answered' in core else {}
                ),
            }
            for run in legacy_runs
        ]
    require(
        isinstance(journal, list)
        and all(
            isinstance(item, dict) and _ordinal(item.get('run'))
            for item in journal
        )
        and [item['run'] for item in journal] == legacy_runs,
        'legacy prediction journal changes actual Run order or ownership',
    )
    events = []
    for record, observed in zip(legacy_inputs, journal, strict=True):
        require(
            isinstance(observed, dict)
            and set(observed)
            == (
                {'run', 'completion', 'result', 'error'}
                | ({'answered'} if control_observed else set())
            )
            and observed['completion'] in ('done', 'error', 'unobserved')
            and (
                observed['error'] is None
                or (
                    isinstance(observed['error'], dict)
                    and set(observed['error']) == {'kind', 'code'}
                    and all(
                        type(value) is str
                        for value in observed['error'].values()
                    )
                )
            )
            and (observed['completion'] == 'done' or observed['result'] is None)
            and (observed['completion'] == 'error' or observed['error'] is None)
            and (
                observed['completion'] != 'error'
                or observed['error'] is not None
            )
            and (
                observed['completion'] == 'error'
                or observed.get('answered') is None
            ),
            'legacy prediction journal has contradictory completion products',
        )
        _legacy_product(observed['result'])
        require(
            core.get('capture_capabilities') is None
            or observed['result'] is None
            or 'outputs' in observed['result'],
            'current native completion omits its observed output supplement',
        )
        events.append(
            {
                'position': run_positions[observed['run']],
                'through': run_positions[observed['run']],
                'model': record['model'],
                'attempts': None
                if attempts is None
                else [
                    item for item in attempts if item['run'] == observed['run']
                ],
                'calls': [
                    item['body']
                    for item in attempts
                    if item['run'] == observed['run']
                ]
                if attempts is not None
                else (
                    observed['result'].get(
                        'calls', [observed['result']['body']]
                    )
                    if observed['result'] is not None
                    and observed['result']['body'] is not None
                    else []
                    if observed['result'] is not None
                    else None
                ),
                'result': observed['result'],
                'error': observed['error'],
                'partial': _legacy_partial(
                    observed.get('answered'),
                    control_observed and observed['completion'] == 'error',
                ),
            }
        )
    return events


def _view(core: dict[str, Any], operations: list[str]) -> dict[str, Any]:
    materialization = _materializations(core, operations)
    capabilities = core.get('capture_capabilities')
    require(
        capabilities is None
        or (
            isinstance(capabilities, dict)
            and set(capabilities) == {'model_replies', 'native_error_control'}
            and all(type(value) is bool for value in capabilities.values())
        ),
        'capture interception capabilities are malformed',
    )
    if capabilities is not None:
        require(
            ('answered' in core) == capabilities['native_error_control'],
            'carried observation availability contradicts capture interception',
        )
    inputs = core['inputs']
    records = {}
    for record in inputs:
        if record['op'] != 'prepare_request':
            continue
        run = record['run']
        require(
            _ordinal(run)
            and run < len(operations)
            and operations[run] == 'prepare_request'
            and run not in records,
            'actual preparation Run is missing or duplicated',
        )
        require(
            not (record['result'] is not None and record['error'] is not None),
            'actual preparation has both a result and a native error',
        )
        records[run] = record
    require(
        set(records)
        == {
            at
            for at, operation in enumerate(operations)
            if operation == 'prepare_request'
        },
        'actual preparation operations have incomplete input evidence',
    )
    entries = core.get('predictions', [])
    require(
        isinstance(entries, list),
        'prediction entries are not observed as a list',
    )
    materialization_errors = {}
    for entry in materialization['entries']:
        failure = entry['failure']
        if (
            entry['native']
            and entry['prediction_parent'] is not None
            and failure is not None
            and failure['relation'] != 'native'
        ):
            materialization_errors[failure['run']] = (
                entry['prediction_parent'],
                entry['_error'],
            )
    selected_errors = {run: record['error'] for run, record in records.items()}
    selected_errors.update(
        {run: error for run, (_, error) in materialization_errors.items()}
    )
    projected = [
        _entry(entry, at, records, operations, entries, materialization_errors)
        for at, entry in enumerate(entries)
    ]
    products = {}
    for entry, projected_entry in zip(entries, projected, strict=True):
        for raw, value in zip(
            entry['results'], projected_entry['_results'], strict=True
        ):
            require(
                raw['run'] not in products,
                'a native result belongs to multiple entries',
            )
            products[raw['run']] = (raw, value)
    carries = {}
    for entry in entries:
        carry = entry['carried']
        if carry is None:
            carries[entry['ordinal']] = None
            continue
        require(
            entry['completion'] == 'raised'
            and isinstance(carry, dict)
            and set(carry) == {'run', 'relation'}
            and _ordinal(carry['run'])
            and carry['run'] in products
            and carry['relation'] in ('selected', 'cause')
            and _below(
                records[carry['run']]['entry'], entry['ordinal'], entries
            ),
            'carried prediction has no owned successful native result association',
        )
        raw, value = products[carry['run']]
        require(
            raw['metadata'] is not None and raw['result'] is not None,
            'carried prediction result was not completely observed',
        )
        carries[entry['ordinal']] = (raw, value)
    owners = {}
    for entry in entries:
        for run in entry['runs']:
            require(
                run not in owners,
                'an actual Run belongs to more than one prediction entry',
            )
            owners[run] = entry['ordinal']
    for run, record in records.items():
        require(
            record['entry'] == owners.get(run),
            'actual preparation has a missing prediction owner',
        )
    # Only proved native preparation earns a common prediction role. Actual
    # nonprediction operations and unrelated preparation remain fully visible.
    native = {entry['ordinal']: entry for entry in projected if entry['native']}
    emitted = set()
    logical_operations = []
    run_positions = {}
    entry_positions = {}
    materialization_positions = {}
    materialization_native = {
        entry['ordinal']: entry
        for entry in materialization['entries']
        if entry['native']
    }
    for run, operation in enumerate(operations):
        materialize = materialization_native.get(
            materialization['owners'].get(run)
        )
        entry = native.get(owners.get(run))
        if materialize is not None:
            ordinal = materialize['ordinal']
            if ordinal not in materialization_positions:
                materialization_positions[ordinal] = len(logical_operations)
                logical_operations.append('materialize')
            run_positions[run] = materialization_positions[ordinal]
        elif operation == 'prepare_request' and entry is not None:
            if entry['ordinal'] not in emitted:
                entry_positions[entry['ordinal']] = len(logical_operations)
                logical_operations.append('predict')
                emitted.add(entry['ordinal'])
            run_positions[run] = entry_positions[entry['ordinal']]
        else:
            run_positions[run] = len(logical_operations)
            logical_operations.append(operation)
    logical_inputs = []
    input_positions = []
    emitted = set()
    cursor = 0
    for record in inputs:
        if record['op'] == 'prepare_request':
            run = record['run']
        else:
            run = next(
                (
                    at
                    for at in range(cursor, len(operations))
                    if operations[at] == record['op']
                ),
                None,
            )
            require(
                run is not None,
                'an actual input has no matching native operation',
            )
        require(
            run >= cursor, 'actual input observations change native Run order'
        )
        cursor = run + 1
        if record['op'] == 'predict':
            state = record['state']
            if core.get('materialized') is not None:
                require(
                    state == core['materialized'],
                    'legacy prediction state differs from its own materialized answer',
                )
            logical_inputs.append({'op': 'predict', 'model': record['model']})
            input_positions.append(run_positions[run])
        elif record['op'] == 'prepare_request' and record['entry'] in native:
            entry = native[record['entry']]
            if entry['ordinal'] not in emitted:
                logical_inputs.append(
                    {'op': 'predict', 'model': entry['model']}
                )
                input_positions.append(run_positions[run])
                emitted.add(entry['ordinal'])
        else:
            logical_inputs.append(record)
            input_positions.append(run_positions[run])
    for ordinal, entry in materialization_native.items():
        logical_inputs.append(materialization['inputs'][ordinal])
        input_positions.append(run_positions[entry['_task']])
    ordered_inputs = sorted(
        zip(input_positions, logical_inputs, strict=True),
        key=lambda item: item[0],
    )
    input_positions = [position for position, _ in ordered_inputs]
    logical_inputs = [value for _, value in ordered_inputs]
    materialized = core.get('materialized')
    selected_error = core.get('error')
    last_materialization = next(
        (
            entry
            for entry in reversed(materialization['entries'])
            if entry['parent'] is None and entry['native']
        ),
        None,
    )
    if last_materialization is not None:
        materialized = last_materialization['_materialized']
    last = next(
        (entry for entry in reversed(projected) if entry['parent'] is None),
        None,
    )
    if last is not None:
        require(
            core.get('predicted') == last['_rendered'],
            'common prediction differs from its actual final Arrow/metadata/result observations',
        )
        error = None
        if last['failure'] is not None:
            error = selected_errors[last['failure']['run']]
        if last_materialization is None or max(
            entries[last['ordinal']]['runs'], default=-1
        ) >= max(last_materialization['runs'], default=-1):
            require(
                core.get('error') == error,
                'common prediction error lacks an actual selected native owner',
            )
            selected_error = error
    if last_materialization is not None and (
        last is None
        or max(last_materialization['runs'], default=-1)
        > max(entries[last['ordinal']]['runs'], default=-1)
    ):
        selected_error = last_materialization['_error']
        require(
            core.get('error') == selected_error,
            'common materialization error lacks its actual selected owner',
        )
    legacy_calls = core.get('model_calls')
    require(
        legacy_calls is None or isinstance(legacy_calls, list),
        'legacy actual model calls are malformed',
    )
    taken = []
    for call in legacy_calls or []:
        require(
            set(call)
            == {'run', 'body', 'body_sha256', 'reply', 'model_endpoint'}
            and _ordinal(call['run'])
            and call['run'] < len(operations)
            and operations[call['run']] == 'predict'
            and isinstance(call['body'], dict)
            and _sha256(call['body_sha256']),
            'legacy model call lacks an actual prediction Run/body/hash',
        )
        _reply_obligation(call['reply'])
        _known_endpoint(call['model_endpoint'])
        if capabilities is not None:
            if capabilities['model_replies']:
                require(
                    not taken
                    or taken[-1]['run'] != call['run']
                    or taken[-1]['reply'] is not None,
                    'a full-observer model call precedes its pending reply',
                )
            else:
                require(
                    call['reply'] is None,
                    'a sparse observer claims an unavailable model reply',
                )
        taken.append(dict(call))
    require(
        [call['run'] for call in taken]
        == sorted(call['run'] for call in taken),
        'legacy model call observations change actual Run order',
    )
    for entry in projected:
        taken.extend(
            {
                'run': call['run'],
                'body': call['body'],
                'body_sha256': call['body_sha256'],
                'reply': call['reply'],
                'model_endpoint': records[call['run']]['input']['input'][
                    'model_endpoint'
                ],
            }
            for call in entry['_calls']
            if call['body'] is not None
        )
    taken.sort(key=lambda item: item['run'])
    attempts = taken
    calls = [call['body'] for call in taken]
    hashes = [call['body_sha256'] for call in taken]
    if not entries and legacy_calls is None:
        predicted = core.get('predicted')
        calls = (
            predicted.get('calls', [predicted['body']])
            if predicted is not None and predicted.get('body') is not None
            else None
        )
        hashes = None
        attempts = None
    partial = _legacy_partial(
        core.get('answered'),
        'answered' in core and core.get('error') is not None,
    )
    if last is not None and last['completion'] == 'raised':
        partial = _partial_entry(
            entries[last['ordinal']], last['_results'], carries[last['ordinal']]
        )
    events = _legacy_events(core, operations, inputs, run_positions, attempts)
    for entry in projected:
        if not entry['native']:
            continue
        raw = entries[entry['ordinal']]
        failure = entry['failure']
        rendered = entry['_rendered']
        if rendered is not None:
            rendered = {
                **rendered,
                'outputs': raw['results'][raw['final']]['data']['predicted'][
                    'outputs'
                ],
            }
        events.append(
            {
                'position': entry_positions[entry['ordinal']],
                # A shared prefix cannot cut a still-owned nested invocation
                # away from its parent's terminal product or handover.
                'through': max(
                    run_positions[run]
                    for descendant in entries
                    if _below(descendant['ordinal'], entry['ordinal'], entries)
                    for run in descendant['runs']
                ),
                'model': entry['model'],
                'attempts': [
                    item for item in attempts if item['run'] in raw['runs']
                ],
                'calls': [
                    item['body']
                    for item in entry['_calls']
                    if item['body'] is not None
                ],
                'result': rendered,
                'error': selected_errors[failure['run']]
                if failure is not None
                else None,
                'partial': {
                    **_partial_entry(
                        raw, entry['_results'], carries[entry['ordinal']]
                    ),
                    'observed': entry['completion'] == 'raised',
                },
            }
        )
    events.sort(key=lambda item: item['position'])
    for entry in projected:
        del entry['_rendered']
        del entry['_calls']
        del entry['_results']
    for entry in materialization['entries']:
        if entry['native']:
            entry['position'] = materialization_positions[entry['ordinal']]
            related = set(entry['runs'])
            for child in materialization['entries']:
                parent = child['parent']
                while parent is not None:
                    if parent == entry['ordinal']:
                        related.update(child['runs'])
                        break
                    parent = materialization['entries'][parent]['parent']
            entry['through'] = max(
                (run_positions[run] for run in related),
                default=entry['position'],
            )
            del entry['_task']
            entry['materialized'] = entry.pop('_materialized')
            entry['product'] = entry.pop('_product')
            entry['error'] = entry.pop('_error')
    return {
        'operations': logical_operations,
        'inputs': logical_inputs,
        'input_positions': input_positions,
        'predictions': projected,
        'calls': calls,
        'call_hashes': hashes,
        'attempts': attempts,
        'events': events,
        'partial': partial,
        'materializations': materialization['entries'],
        'materialized': materialized,
        'error': selected_error,
        'fetches': materialization['fetches'],
    }


def view(core: dict[str, Any], operations: list[str]) -> dict[str, Any]:
    """Read only actual evidence; malformed evidence never receives a role."""
    try:
        return _view(core, operations)
    except ObservationError:
        raise
    except (
        KeyError,
        TypeError,
        ValueError,
        IndexError,
        AttributeError,
    ) as error:
        raise ObservationError(
            'prediction evidence has an invalid closed shape'
        ) from error
