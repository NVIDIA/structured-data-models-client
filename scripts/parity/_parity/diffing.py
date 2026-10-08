# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The first place two payload sets or two transcripts differ.

A path names it the way it would be written to reach the value:
``[1].context.related_tables.orders.rows[7].price``. Keys that are not plain
names are quoted, ``["two words"]``, and a cell of a table's ``rows`` is named
by its column, since that is what a reader looks for.

Values compare by JSON type as well as value: ``3`` and ``3.0`` differ, as do
``1`` and ``true``. Key order and layout do not count. ``metadata.request_id``
is the one normalized field: it differs on every request by design.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

_NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_#\-]*$')

Path_ = list[Any]


class BadInput(ValueError):
    r"""An input that is not a payload set or a transcript."""


def render(path: Path_) -> str:
    r"""``path`` written out; the root is ``<root>``."""
    out = ''
    for part in path:
        if isinstance(part, int):
            out += f'[{part}]'
        elif _NAME.match(part):
            out += f'.{part}' if out else part
        else:
            out += f'[{json.dumps(part)}]'
    return out or '<root>'


def _kind(value: Any) -> str:
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'boolean'
    if isinstance(value, int):
        return 'integer'
    if isinstance(value, float):
        return 'number'
    if isinstance(value, str):
        return 'string'
    if isinstance(value, list):
        return 'array'
    if isinstance(value, dict):
        return 'object'
    return type(value).__name__


def _show(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= 80 else text[:77] + '...'


def _normalized(path: Path_) -> bool:
    return len(path) >= 2 and path[-2:] == ['metadata', 'request_id']


def first_difference(
    a: Any, b: Any, path: Path_ | None = None
) -> tuple[Path_, str] | None:
    r"""The path of the first difference between ``a`` and ``b``, and what
    it is; ``None`` when they are the same."""
    return _walk(a, b, path or [], None)


def _walk(
    a: Any, b: Any, path: Path_, columns: list[Any] | None
) -> tuple[Path_, str] | None:
    if _normalized(path):
        return None
    if _kind(a) != _kind(b):
        return path, f'{_kind(a)} {_show(a)} became {_kind(b)} {_show(b)}'
    if isinstance(a, dict):
        return _walk_object(a, b, path)
    if isinstance(a, list):
        return _walk_array(a, b, path, columns)
    if isinstance(a, float) and math.isnan(a) and math.isnan(b):
        return None
    if a != b:
        return path, f'{_show(a)} became {_show(b)}'
    return None


def _walk_object(
    a: dict[str, Any], b: dict[str, Any], path: Path_
) -> tuple[Path_, str] | None:
    # A table's column list is read before its rows, so a reordered column
    # is reported as itself rather than as every cell under it.
    keys = list(a)
    if 'columns' in keys and 'rows' in keys:
        keys.remove('columns')
        keys.insert(0, 'columns')
    columns = a.get('columns') if isinstance(a.get('columns'), list) else None
    for key in keys:
        if key not in b:
            if _normalized([*path, key]):
                continue
            return [*path, key], f'the key {key!r} was removed'
        found = _walk(
            a[key], b[key], [*path, key], columns if key == 'rows' else None
        )
        if found:
            return found
    for key in b:
        if key not in a and not _normalized([*path, key]):
            return [*path, key], f'the key {key!r} was added'
    return None


def _walk_array(
    a: list[Any], b: list[Any], path: Path_, columns: list[Any] | None
) -> tuple[Path_, str] | None:
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        if columns is not None and isinstance(x, list) and isinstance(y, list):
            found = _walk_row(x, y, [*path, index], columns)
        else:
            found = _walk(x, y, [*path, index], None)
        if found:
            return found
    if len(a) != len(b):
        index = min(len(a), len(b))
        what = 'missing' if len(a) > len(b) else 'added'
        return (
            [*path, index],
            f'{len(a)} items became {len(b)}: item {index} is {what}',
        )
    return None


def _walk_row(
    a: list[Any], b: list[Any], path: Path_, columns: list[Any]
) -> tuple[Path_, str] | None:
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        name = columns[index] if index < len(columns) else index
        found = _walk(x, y, [*path, name], None)
        if found:
            return found
    if len(a) != len(b):
        return path, f'a row of {len(a)} cells became {len(b)}'
    return None


# Transcripts.

#: The order a transcript's top-level keys are compared in.
TRANSCRIPT_ORDER = (
    'format_version',
    'name',
    'backend',
    'task_kind',
    'status',
    'data',
    'calls',
    'steps',
    'payloads',
    'result',
    'warnings',
)


def _table_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    from _parity.transcript import read_arrow

    table = read_arrow(path)
    schema = [f'{field.name}: {field.type}' for field in table.schema]
    return schema, table.to_pylist()


def _walk_table(
    a_root: Path, a_ref: Any, b_root: Path, b_ref: Any, path: Path_
) -> tuple[Path_, str] | None:
    r"""Two referenced Arrow files, compared by what they hold."""
    if not (isinstance(a_ref, str) and isinstance(b_ref, str)):
        return _walk(a_ref, b_ref, path, None)
    a_schema, a_rows = _table_rows(a_root / a_ref)
    b_schema, b_rows = _table_rows(b_root / b_ref)
    if a_schema != b_schema:
        return path, f'the columns {a_schema} became {b_schema}'
    return _walk(_plain(a_rows), _plain(b_rows), path, None)


def _plain(value: Any) -> Any:
    r"""Arrow values as JSON-like ones, so they compare by the same rules."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return f'{type(value).__name__}:{value!r}'


def first_transcript_difference(
    a: dict[str, Any], a_root: Path, b: dict[str, Any], b_root: Path
) -> tuple[Path_, str] | None:
    keys = [k for k in TRANSCRIPT_ORDER if k in a or k in b]
    keys += [k for k in a if k not in keys]
    keys += [k for k in b if k not in keys]
    for key in keys:
        if key not in a or key not in b:
            what = 'removed' if key in a else 'added'
            return [key], f'the key {key!r} was {what}'
        if key == 'steps':
            found = _walk_steps(a[key], a_root, b[key], b_root)
        elif key == 'result':
            found = _walk_table(a_root, a[key], b_root, b[key], [key])
        else:
            found = _walk(a[key], b[key], [key], None)
        if found:
            return found
    return None


def _walk_steps(
    a: Any, a_root: Path, b: Any, b_root: Path
) -> tuple[Path_, str] | None:
    if not (isinstance(a, list) and isinstance(b, list)):
        return _walk(a, b, ['steps'], None)
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        path = ['steps', index]
        if not (isinstance(x, dict) and isinstance(y, dict)):
            found = _walk(x, y, path, None)
        else:
            plain_x = {k: v for k, v in x.items() if k != 'result'}
            plain_y = {k: v for k, v in y.items() if k != 'result'}
            found = _walk(plain_x, plain_y, path, None) or _walk_table(
                a_root,
                x.get('result'),
                b_root,
                y.get('result'),
                [*path, 'result'],
            )
        if found:
            return found
    if len(a) != len(b):
        index = min(len(a), len(b))
        return ['steps', index], f'{len(a)} steps became {len(b)}'
    return None


def load_input(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError as error:
        raise BadInput(f'{path} does not exist') from error
    except (OSError, ValueError) as error:
        raise BadInput(f'{path} is not JSON: {error}') from error


def is_transcript(value: Any) -> bool:
    return isinstance(value, dict) and 'steps' in value


def compare_files(a_path: Path, b_path: Path) -> tuple[Path_, str] | None:
    r"""The first difference between two files of the same kind."""
    a, b = load_input(a_path), load_input(b_path)
    if is_transcript(a) and is_transcript(b):
        return first_transcript_difference(
            a, Path(a_path).parent, b, Path(b_path).parent
        )
    if isinstance(a, list) and isinstance(b, list):
        return first_difference(a, b)
    raise BadInput(
        'compare two transcripts, or two payload sets (JSON arrays of payloads)'
    )
