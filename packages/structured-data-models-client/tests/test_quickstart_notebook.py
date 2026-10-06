# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The quickstart notebook must keep matching the shipped API.

A notebook is the first thing most people run, and it is the easiest artefact
to leave behind when an interface moves. This does not execute it (that needs
two NIMs), but it does hold it to the public surface, which is what actually
rots.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import sdm_client

NOTEBOOK = (
    Path(__file__).parents[3] / 'examples' / 'sdm_client_quickstart.ipynb'
)

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


def cells(kind: str) -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    return [
        ''.join(cell['source'])
        for cell in notebook['cells']
        if cell['cell_type'] == kind
    ]


def test_the_notebook_is_present_and_well_formed():
    assert NOTEBOOK.exists(), f'no quickstart at {NOTEBOOK}'
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    assert notebook['nbformat'] >= 4
    assert cells('code'), 'the quickstart runs nothing'
    assert cells('markdown'), 'the quickstart explains nothing'


def test_every_code_cell_parses():
    for index, source in enumerate(cells('code')):
        try:
            ast.parse(source)
        except (
            SyntaxError
        ) as error:  # pragma: no cover - the failure is the point
            pytest.fail(f'code cell {index} does not parse: {error}')


@pytest.mark.parametrize('retired', RETIRED)
def test_the_notebook_uses_no_interface_this_client_retired(retired):
    body = '\n'.join(cells('code') + cells('markdown'))
    assert retired not in body, (
        f'the quickstart still shows {retired!r}, which this client does not '
        f'have'
    )


def test_every_name_the_notebook_imports_is_exported():
    """A name that moved out of ``__all__`` would break the first cell."""
    imported: set[str] = set()
    for source in cells('code'):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.module == 'sdm_client':
                imported.update(alias.name for alias in node.names)
    assert imported, 'the quickstart imports nothing from sdm_client'
    missing = sorted(name for name in imported if not hasattr(sdm_client, name))
    assert not missing, f'the quickstart imports {missing}, which do not exist'


def test_the_notebook_shows_the_model_as_an_argument():
    """The one idea; if the quickstart stops showing it, it has drifted."""
    body = '\n'.join(cells('code'))
    assert 'model=KumoTabular()' in body
    assert 'model=KumoRelational()' in body
