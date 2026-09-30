# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The notebooks run only in a workspace, so they are held to the public API
here."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[3] / 'examples' / 'databricks'
NOTEBOOKS = sorted(EXAMPLES.glob('*.ipynb'))


def _cells(notebook: Path, kind: str) -> list[str]:
    return [
        ''.join(cell['source'])
        for cell in json.loads(notebook.read_text(encoding='utf-8'))['cells']
        if cell['cell_type'] == kind
    ]


def _python(source: str) -> str:
    r"""A cell without its IPython magics, which are not Python."""
    return '\n'.join(
        '' if line.lstrip().startswith('%') else line
        for line in source.splitlines()
    )


def test_there_are_notebooks_for_the_catalog_and_for_lakebase() -> None:
    assert {path.name for path in NOTEBOOKS} >= {
        'sdm_client_adventureworks.ipynb',
        'sdm_client_lakebase.ipynb',
    }


@pytest.mark.parametrize('notebook', NOTEBOOKS, ids=lambda path: path.stem)
def test_every_code_cell_parses(notebook: Path) -> None:
    for index, source in enumerate(_cells(notebook, 'code')):
        try:
            ast.parse(_python(source))
        except SyntaxError as error:  # pragma: no cover - the failure is it
            pytest.fail(f'{notebook.name} cell {index} does not parse: {error}')


@pytest.mark.parametrize('notebook', NOTEBOOKS, ids=lambda path: path.stem)
def test_every_name_imported_from_this_project_exists(notebook: Path) -> None:
    missing = []
    for source in _cells(notebook, 'code'):
        for node in ast.walk(ast.parse(_python(source))):
            if not isinstance(node, ast.ImportFrom) or node.module not in (
                'sdm_client',
                'sdm_connectors',
            ):
                continue
            module = importlib.import_module(node.module)
            missing += [
                f'{node.module}.{alias.name}'
                for alias in node.names
                if not hasattr(module, alias.name)
            ]
    assert not missing, f'{notebook.name} imports {missing}'


@pytest.mark.parametrize('notebook', NOTEBOOKS, ids=lambda path: path.stem)
def test_the_tables_are_sampled_in_place(notebook: Path) -> None:
    body = '\n'.join(_cells(notebook, 'code'))
    assert 'Schema.from_source(' in body
    assert 'Schema.from_data(' not in body
    assert 'SDMClient.for_databricks_serving(' in body
    assert 'model=KumoRelational()' in body
