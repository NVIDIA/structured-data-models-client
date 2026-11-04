# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata presentation only; no execution or schema decisions live here."""

from __future__ import annotations

import base64
import html
import io
import json
import warnings
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sdm_client.schema import Schema


def _label(value: str) -> str:
    # Encode every code point: names cannot close quotes or create Mermaid
    # directives, edges, HTML, or script. Rendering decodes these as text.
    return ''.join(f'#{ord(c)};' for c in value)


def mermaid(schema: Schema, show_columns: bool) -> str:
    if type(show_columns) is not bool:
        raise TypeError('show_columns must be a boolean')
    identifiers = {name: f't{i}' for i, name in enumerate(schema.tables)}
    lines = ['flowchart LR']
    if not identifiers:
        lines.append('  empty["Empty schema"]')
    for name, table in schema.tables.items():
        primary = table.primary_key
        keys = {primary} if isinstance(primary, str) else set(primary or ())
        foreign = {
            r.source_column
            for r in schema.relationships
            if r.source_table == name
        }
        labels = [_label(name)]
        for column_name, column in table.columns.items():
            marks = []
            if column_name in keys:
                marks.append('PK')
            if column_name in foreign:
                marks.append('FK')
            if column_name == table.time_column:
                marks.append('TIME')
            if show_columns or marks:
                description = f'{column_name}: {column.stype or column.dtype or "unknown"}'
                if marks:
                    description += ' [' + ', '.join(marks) + ']'
                labels.append(_label(description))
        lines.append(f'  {identifiers[name]}["' + '<br/>'.join(labels) + '"]')
    for link in schema.relationships:
        description = (
            f'{link.source_column} → {link.target_column} (many to one)'
        )
        lines.append(
            f'  {identifiers[link.source_table]} -->|"{_label(description)}"| {identifiers[link.target_table]}'
        )
    return '\n'.join(lines)


@lru_cache(maxsize=2)
def _asset(name: str) -> str:
    return (
        files('sdm_client').joinpath('assets', name).read_text(encoding='utf-8')
    )


def document(source: str) -> str:
    # Replace source LAST: its text cannot be interpreted as template tokens.
    return (
        _asset('schema.html')
        .replace('@@RENDERER@@', _asset('mermaid.min.js'))
        .replace('@@SOURCE@@', html.escape(source))
    )


def image(source: str, extension: str) -> bytes:
    import requests

    state = json.dumps(
        {
            'code': source,
            'mermaid': {'theme': 'neutral', 'securityLevel': 'strict'},
        }
    )
    encoded = base64.urlsafe_b64encode(state.encode()).decode()
    route = 'svg' if extension == '.svg' else 'img'
    try:
        response = requests.get(
            f'https://mermaid.ink/{route}/base64:{encoded}'
            + ('?type=png' if extension == '.png' else ''),
            timeout=30,
        )
        response.raise_for_status()
        return response.content
    except requests.RequestException as error:
        # requests exceptions can embed the full URL, including schema text.
        status = (
            error.response.status_code if error.response is not None else None
        )
        detail = (
            f'HTTP {status}' if status is not None else type(error).__name__
        )
        raise RuntimeError(
            f'schema image export failed at the external renderer ({detail})'
        ) from None


def visualize(
    schema: Schema,
    path: Path | str | io.BytesIO | None,
    show_columns: bool,
    height: int,
) -> None:
    if type(height) is not int:
        raise TypeError('height must be an integer')
    if height <= 0:
        raise ValueError('height must be positive')
    source = mermaid(schema, show_columns)
    if isinstance(path, io.BytesIO):
        path.write(image(source, '.png'))
        return
    if path is not None:
        if not isinstance(path, (str, Path)):
            raise TypeError('path must be a path, BytesIO or None')
        target = Path(path)
        extension = target.suffix.lower()
        if extension == '.mmd':
            target.write_text(source + '\n', encoding='utf-8')
        elif extension == '.html':
            target.write_text(document(source), encoding='utf-8')
        elif extension in ('.png', '.svg'):
            target.write_bytes(image(source, extension))
        else:
            raise ValueError('use an .mmd, .html, .png or .svg path')
        return
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        pass
    else:
        if get_script_run_ctx(suppress_warning=True) is not None:
            try:
                st.components.v1.html(
                    document(source), height=height, scrolling=True
                )
            except Exception:  # Restricted hosts may disable custom components.
                st.code(source)
            return
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
    except ImportError:
        shell = None
    else:
        shell = get_ipython()
    if shell is not None and getattr(shell, 'kernel', None) is not None:
        markup = html.escape(document(source), quote=True)
        display(
            HTML(
                f'<iframe sandbox="allow-scripts" srcdoc="{markup}" width="100%" height="{height}"></iframe>'
            )
        )
    else:
        warnings.warn(
            'No notebook display available; printing Mermaid source.',
            stacklevel=3,
        )
        print(source)
