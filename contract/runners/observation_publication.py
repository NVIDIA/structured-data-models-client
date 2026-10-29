# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private raw evidence owns its separately derived warehouse comparison.

A published derivative is an inspection product, never a raw observation.
The existing placeholder policy operates only after strict raw validation.
"""

from __future__ import annotations

import base64
import copy
import functools
import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WAREHOUSES = frozenset(
    ('databricks-sql', 'databricks-spark', 'snowflake', 'lakebase')
)
PUBLIC_FORMAT = 'sdm-published-observation/1'
MAP_FORMAT = 'sdm-private-name-map/1'
RAW_JOURNALS = frozenset(
    (
        'materialization_preparations',
        'materialization_guards',
        'materialization_entries',
        'predictions',
        'legacy_predictions',
        'model_calls',
    )
)


class PublicationError(ValueError):
    """An invalid evidence/projection relationship, without private values."""


@functools.cache
def runner() -> Any:
    path = ROOT / 'contract/runners/run_python.py'
    spec = importlib.util.spec_from_file_location(
        'private_publication_runner', path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _names(database: str, names: Any) -> dict[str, str]:
    if database not in WAREHOUSES or type(names) is not dict:
        raise PublicationError('no declared warehouse name mapping')
    if any(
        type(key) is not str
        or re.fullmatch(r'[a-z][a-z0-9_]*', key) is None
        or type(value) is not str
        or not value
        for key, value in names.items()
    ):
        raise PublicationError('malformed warehouse name mapping')
    if len({value.casefold() for value in names.values()}) != len(names):
        raise PublicationError('warehouse names have ambiguous aliases')
    return dict(names)


def _serving_record(value: dict[str, Any]) -> dict[str, Any] | None:
    records = value.get('dataframe_records')
    if (
        set(value) == {'dataframe_records'}
        and isinstance(records, list)
        and len(records) == 1
        and isinstance(records[0], dict)
        and set(records[0]) == {'request_json'}
    ):
        return records[0]
    return None


def _project_json_text(held: Any, text: str) -> tuple[str, bool]:
    """Map decoded names using the existing lossless ordered JSON reader."""
    reader = runner()._sibling('prediction_observation')
    try:
        decoded = reader._ordered_json(text)
    except (ValueError, RecursionError):
        raise PublicationError('nested request JSON is unreadable') from None
    pending = [('value', decoded)]
    written = []
    changed = False
    while pending:
        action, value = pending.pop()
        if action == 'literal':
            written.append(value)
        elif type(value) is tuple and value[0] == 'number':
            # The decoder already validated this exact token as JSON.
            written.append(value[1])
        elif type(value) is tuple and value[0] == 'object':
            pairs = value[1]
            # Preserve the existing finite nested Serving boundary too.
            if len(pairs) == 1 and pairs[0][0] == 'dataframe_records':
                rows = pairs[0][1]
                if type(rows) is list and len(rows) == 1:
                    row = rows[0]
                    if (
                        type(row) is tuple
                        and row[0] == 'object'
                        and len(row[1]) == 1
                    ):
                        key, nested = row[1][0]
                        if key == 'request_json' and type(nested) is str:
                            mapped, altered = _project_json_text(held, nested)
                            if altered:
                                pairs = [
                                    (
                                        'dataframe_records',
                                        [
                                            (
                                                'object',
                                                [('request_json', mapped)],
                                            )
                                        ],
                                    )
                                ]
                                changed = True
            members = []
            originals = {}
            for key, item in pairs:
                mapped = held.text(key)
                if mapped in originals and originals[mapped] != key:
                    raise PublicationError(
                        'warehouse names collide with object members'
                    )
                originals[mapped] = key
                changed |= mapped != key
                members.append((mapped, item))
            written.append('{')
            pending.append(('literal', '}'))
            for at in range(len(members) - 1, -1, -1):
                key, item = members[at]
                pending.extend(
                    [
                        ('value', item),
                        ('literal', ':'),
                        ('literal', json.dumps(key, ensure_ascii=True)),
                    ]
                )
                if at:
                    pending.append(('literal', ','))
        elif type(value) is list:
            written.append('[')
            pending.append(('literal', ']'))
            for at in range(len(value) - 1, -1, -1):
                pending.append(('value', value[at]))
                if at:
                    pending.append(('literal', ','))
        elif type(value) is str:
            mapped = held.text(value)
            changed |= mapped != value
            written.append(json.dumps(mapped, ensure_ascii=True))
        elif value is None:
            written.append('null')
        elif type(value) is bool:
            written.append('true' if value else 'false')
        else:
            raise PublicationError('nested request JSON has no typed value')
    return ''.join(written) if changed else text, changed


def _project(held: Any, value: Any) -> Any:
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            keys = [held.text(key) for key in current]
            if len(set(keys)) != len(keys):
                raise PublicationError(
                    'warehouse names collide with object members'
                )
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    shown = held.value(value)
    # request_json is a decoded publication field, not the original wire.
    # Escaped member values must not hide a private name from Placeheld.
    pending = [shown]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            nested_owner = _serving_record(current)
            nested = (
                None if nested_owner is None else nested_owner['request_json']
            )
            if isinstance(nested, str):
                mapped, changed = _project_json_text(held, nested)
                if changed:
                    nested_owner['request_json'] = mapped
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    pending = [shown]
    while pending:
        current = pending.pop()
        if isinstance(current, str) and held.text(current) != current:
            raise PublicationError('warehouse name projection is not stable')
        if isinstance(current, dict):
            pending.extend(current)
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return shown


class ComparisonObservation(Mapping[str, Any]):
    """An owned original and its checked decoded comparison projections.

    Only prepare_comparison constructs this in the comparison entry path.
    Properties detach returned evidence so callers cannot edit the owner.
    No serialized cached view can supply or replace the original evidence.
    """

    __slots__ = ('_raw', '_raw_view', '_view', '_shown', '_held')

    def __init__(
        self, raw: dict[str, Any], raw_view: dict[str, Any], held: Any
    ):
        self._raw = raw
        self._raw_view = raw_view
        self._held = held
        self._view = _project(held, raw_view)
        ordinary = {key: value for key, value in raw.items() if key != 'core'}
        core = raw.get('core')
        if core is not None:
            ordinary['core'] = {
                **{
                    key: value
                    for key, value in core.items()
                    if key not in RAW_JOURNALS
                },
                'inputs': raw_view['inputs'],
                'fetches': raw_view.get('fetches', core.get('fetches')),
                'materialized': raw_view.get(
                    'materialized', core.get('materialized')
                ),
                'error': raw_view.get('error', core.get('error')),
            }
        else:
            ordinary['core'] = None
        self._shown = _project(held, ordinary)

    def __getitem__(self, key: str) -> Any:
        return self._shown[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._shown)

    def __len__(self) -> int:
        return len(self._shown)

    @property
    def raw(self) -> dict[str, Any]:
        return copy.deepcopy(self._raw)

    @property
    def raw_view(self) -> dict[str, Any]:
        return copy.deepcopy(self._raw_view)

    @property
    def view(self) -> dict[str, Any]:
        return copy.deepcopy(self._view)

    def project(self, value: Any) -> Any:
        return _project(self._held, value)


def prepare_comparison(
    raw: dict[str, Any], database: str, names: dict[str, str]
) -> ComparisonObservation:
    """Validate the actual raw source before applying existing named placeholders."""
    if type(raw) is not dict or raw.get('format') != 'sdm-observation/1':
        raise PublicationError('comparison requires a raw observation')
    names = _names(database, names)
    raw = copy.deepcopy(raw)
    run = runner()
    try:
        if run._invalid(run.validator('observation'), raw) is not None:
            raise PublicationError('raw observation shape is invalid')
        if raw['case'].partition('@')[2] != database:
            raise PublicationError(
                'raw observation belongs to another warehouse'
            )
        core = raw.get('core')
        reader = run._sibling('prediction_observation')
        view = (
            reader.view(core, raw['identity'].get('operations') or [])
            if core is not None
            else {
                'operations': [],
                'inputs': [],
                'predictions': [],
                'materializations': [],
            }
        )
        if (
            raw['identity']['runner'] == 'python-core'
            and view.get('attempts') is not None
        ):
            reader.check_receipts(
                view['attempts'], raw['public'].get('payloads', [])
            )
    except PublicationError:
        raise
    except (ValueError, TypeError, KeyError, RecursionError):
        raise PublicationError(
            'raw observation evidence is inconsistent'
        ) from None
    return ComparisonObservation(
        raw, view, run.Placeheld(names, names.values())
    )


def load_name_maps(
    paths: list[Path],
    database: str,
) -> dict[str, dict[str, str]]:
    """Closed names-only maps, each identifying its own actual input bytes."""
    if not paths or any(not isinstance(path, Path) for path in paths):
        raise PublicationError('private source maps are unreadable')
    sources = {}
    try:
        for path in paths:
            value = json.loads(path.read_text())
            if type(value) is not dict or set(value) != {
                'format',
                'database',
                'names',
                'source_sha256',
            }:
                raise PublicationError(
                    'private name map has no closed identity'
                )
            if value['format'] != MAP_FORMAT or value['database'] != database:
                raise PublicationError(
                    'private name map belongs to another warehouse'
                )
            digest = value['source_sha256']
            if (
                type(digest) is not str
                or re.fullmatch(r'[0-9a-f]{64}', digest) is None
                or digest in sources
            ):
                raise PublicationError(
                    'private source maps repeat or lack a source identity'
                )
            sources[digest] = _names(database, value['names'])
    except PublicationError:
        raise
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        raise PublicationError('private name map is unreadable') from None
    return sources


def load_name_map(
    path: Path,
    database: str,
    source: Path,
    *,
    source_bytes: bytes | None = None,
) -> dict[str, str]:
    """One explicit map bound to the actual input file bytes."""
    sources = load_name_maps([path], database)
    try:
        digest = hashlib.sha256(
            source.read_bytes() if source_bytes is None else source_bytes
        ).hexdigest()
    except OSError:
        raise PublicationError('private source file is unreadable') from None
    if digest not in sources:
        raise PublicationError('private name map names different source bytes')
    return sources[digest]


def public_derivative(
    prepared: ComparisonObservation, *, source_sha256: str, line: int
) -> dict[str, Any]:
    """A labelled safe summary, with no original encoded evidence or journals."""
    if (
        type(prepared) is not ComparisonObservation
        or type(source_sha256) is not str
        or re.fullmatch(r'[0-9a-f]{64}', source_sha256) is None
        or type(line) is not int
        or line < 1
    ):
        raise PublicationError('publication has no original source identity')
    view = prepared.view
    error = view.get('error')
    return {
        'format': PUBLIC_FORMAT,
        'source': {
            'sha256': source_sha256,
            'line': line,
            'case': prepared['case'],
        },
        'observation': copy.deepcopy(
            {key: value for key, value in prepared.items() if key != 'core'}
        ),
        'core_summary': {
            'operations': view['operations'],
            'materialized': view.get('materialized'),
            'error': None
            if error is None
            else {'kind': error['kind'], 'code': error['code']},
        },
    }


def encoded_operands(raw: dict[str, Any]) -> list[bytes]:
    """The finite actual byte carriers produced by the current collectors."""
    core = raw.get('core') or {}
    result = []
    for record in core.get('materialization_preparations', []):
        snapshots = [
            record['input'],
            *record.get('value_reads', []),
            *record.get('input_reads', []),
            *record.get('output_reads', []),
        ]
        result.extend(
            base64.b64decode(item['base64'], validate=True)
            for item in snapshots
            if 'base64' in item
        )
    for record in core.get('materialization_guards', []):
        snapshots = [record['input'], *record.get('value_reads', [])]
        result.extend(
            base64.b64decode(item['base64'], validate=True)
            for item in snapshots
            if 'base64' in item
        )
    for entry in core.get('predictions', []):
        result.extend(
            base64.b64decode(call['input_body_base64'], validate=True)
            for call in entry.get('calls', [])
        )
    return result


def private_leak(prepared: ComparisonObservation, held: Any) -> bool:
    """Check decoded actual operands and nested Serving JSON before publication."""
    pending = [
        prepared.raw,
        *[json.loads(value) for value in encoded_operands(prepared.raw)],
    ]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value)
            pending.extend(value.values())
            nested_owner = _serving_record(value)
            nested = (
                None if nested_owner is None else nested_owner['request_json']
            )
            if isinstance(nested, str):
                try:
                    pending.append(json.loads(nested))
                except (ValueError, RecursionError):
                    raise PublicationError(
                        'nested request JSON is unreadable'
                    ) from None
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str) and held.leaked(held.text(value)):
            return True
    return False
