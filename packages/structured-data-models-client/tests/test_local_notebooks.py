# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The local notebooks must keep matching the shipped API.

A notebook is the first thing most people run, and it is the easiest artefact
to leave behind when an interface moves. This does not execute them (that needs
two NIMs), but it does hold them to the public surface, which is what actually
rots.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import sdm_client

EXAMPLES = Path(__file__).parents[3] / 'examples' / 'local'
NOTEBOOKS = sorted(EXAMPLES.glob('*.ipynb'))

# Interfaces earlier generations of this client had. None of them exist here,
# and each has appeared in a stale example at least once.
RETIRED = (
    'RelationalClient',
    'PredictClient',
    'client.relational(',
    'client.tabular(',
    'nemotron_predict',
    'run_mode=',
)


def cells(notebook: Path, kind: str) -> list[str]:
    return [
        ''.join(cell['source'])
        for cell in json.loads(notebook.read_text(encoding='utf-8'))['cells']
        if cell['cell_type'] == kind
    ]


def test_there_is_a_notebook_for_each_model():
    assert {path.name for path in NOTEBOOKS} == {
        'tabular.ipynb',
        'relational.ipynb',
    }


@pytest.mark.parametrize('notebook', NOTEBOOKS, ids=lambda p: p.name)
def test_the_notebook_is_well_formed(notebook):
    assert json.loads(notebook.read_text(encoding='utf-8'))['nbformat'] >= 4
    assert cells(notebook, 'code'), f'{notebook.name} runs nothing'
    assert cells(notebook, 'markdown'), f'{notebook.name} explains nothing'


@pytest.mark.parametrize('notebook', NOTEBOOKS, ids=lambda p: p.name)
def test_every_code_cell_parses(notebook):
    for index, source in enumerate(cells(notebook, 'code')):
        try:
            ast.parse(source)
        except (
            SyntaxError
        ) as error:  # pragma: no cover - the failure is the point
            pytest.fail(
                f'{notebook.name} code cell {index} does not parse: {error}'
            )


@pytest.mark.parametrize('notebook', NOTEBOOKS, ids=lambda p: p.name)
@pytest.mark.parametrize('retired', RETIRED)
def test_the_notebook_uses_no_interface_this_client_retired(retired, notebook):
    body = '\n'.join(cells(notebook, 'code') + cells(notebook, 'markdown'))
    assert retired not in body, (
        f'{notebook.name} still shows {retired!r}, which this client does not '
        f'have'
    )


@pytest.mark.parametrize('notebook', NOTEBOOKS, ids=lambda p: p.name)
def test_every_name_the_notebook_imports_is_exported(notebook):
    """A name that moved out of ``__all__`` would break the first cell."""
    imported: set[str] = set()
    for source in cells(notebook, 'code'):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.module == 'sdm_client':
                imported.update(alias.name for alias in node.names)
    assert imported, f'{notebook.name} imports nothing from sdm_client'
    missing = sorted(name for name in imported if not hasattr(sdm_client, name))
    assert not missing, f'{notebook.name} imports {missing}, which do not exist'


def test_each_notebook_asks_its_own_model_as_an_argument():
    """The one idea, and the reason these are two files rather than one."""
    tabular = '\n'.join(cells(EXAMPLES / 'tabular.ipynb', 'code'))
    relational = '\n'.join(cells(EXAMPLES / 'relational.ipynb', 'code'))
    assert 'model=KumoTabular()' in tabular
    assert 'model=KumoRelational()' not in tabular
    assert 'model=KumoRelational()' in relational
