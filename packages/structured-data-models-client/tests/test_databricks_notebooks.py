# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The notebooks run only in a workspace, so they are held to the public API
here."""

from __future__ import annotations

import ast
import importlib
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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


def test_there_is_a_notebook_for_each_way_in() -> None:
    assert {path.name for path in NOTEBOOKS} >= {
        'kumo_relational_quickstart.ipynb',
        'kumo_tabular_quickstart.ipynb',
        'lakebase_relational.ipynb',
    }


def _ours(module: str | None) -> bool:
    return module is not None and (
        module in ('sdm_client', 'sdm_connectors')
        or module.startswith(('sdm_client.', 'sdm_connectors.'))
    )


@pytest.mark.parametrize('notebook', NOTEBOOKS, ids=lambda path: path.stem)
def test_every_name_imported_from_this_project_is_public(
    notebook: Path,
) -> None:
    # Public, not merely present: a notebook is copied from, so a private name
    # it reaches for is one a user comes to depend on.
    imported: list[str] = []
    private: list[str] = []
    for source in _cells(notebook, 'code'):
        for node in ast.walk(ast.parse(_python(source))):
            if not isinstance(node, ast.ImportFrom) or not _ours(node.module):
                continue
            module = importlib.import_module(node.module)
            for alias in node.names:
                imported.append(f'{node.module}.{alias.name}')
                if alias.name not in getattr(module, '__all__', ()):
                    private.append(f'{node.module}.{alias.name}')
    assert imported, f'{notebook.name} imports nothing from this project'
    assert not private, (
        f"{notebook.name} imports {private}, which are not in their module's "
        f'__all__'
    )


@pytest.mark.parametrize(
    ('active_model', 'ready_model', 'ready_version', 'accepted'),
    [
        ('catalog.schema.tabular', 'catalog.schema.tabular', '1', True),
        ('catalog.schema.other', 'catalog.schema.tabular', '1', True),
        ('catalog.schema.tabular', 'CATALOG.SCHEMA.TABULAR', '1', True),
        ('catalog.schema.tabular', 'catalog.schema.other', '1', False),
        ('catalog.schema.tabular', 'catalog.schema.tabular', '2', False),
    ],
)
def test_tabular_notebook_checks_the_model_after_a_pending_update(
    active_model: str, ready_model: str, ready_version: str, accepted: bool
) -> None:
    cells = _cells(EXAMPLES / 'kumo_tabular_quickstart.ipynb', 'code')
    deploy = next(
        cell for cell in cells if 'workspace = WorkspaceClient()' in cell
    )
    connect = next(
        cell
        for cell in cells
        if 'wait_get_serving_endpoint_not_updating(' in cell
    )

    def config(name: str, version: str) -> SimpleNamespace:
        return SimpleNamespace(
            served_entities=[
                SimpleNamespace(entity_name=name, entity_version=version)
            ]
        )

    workspace = Mock()
    workspace.serving_endpoints.get.return_value = SimpleNamespace(
        config=config(active_model, '1'),
        pending_config=config('catalog.schema.tabular', '1'),
    )
    workspace.serving_endpoints.wait_get_serving_endpoint_not_updating.return_value = SimpleNamespace(
        config=config(ready_model, ready_version),
        state=SimpleNamespace(ready=SimpleNamespace(value='READY')),
    )
    client_factory = Mock()
    namespace = {
        'WorkspaceClient': lambda: workspace,
        'NotFound': LookupError,
        'MODEL_NAME': 'catalog.schema.tabular',
        'MODEL_VERSION': '1',
        'ENDPOINT_NAME': 'tabular-demo',
        'timedelta': timedelta,
        'SDMClient': SimpleNamespace(for_databricks_serving=client_factory),
        'KumoTabular': Mock(),
    }
    exec(deploy, namespace)
    workspace.serving_endpoints.create.assert_not_called()
    if accepted:
        exec(connect, namespace)
        client_factory.assert_called_once_with('tabular-demo', timeout=900)
    else:
        with pytest.raises(
            ValueError, match='not catalog.schema.tabular version 1'
        ):
            exec(connect, namespace)
        client_factory.assert_not_called()


@pytest.mark.parametrize(
    ('active', 'pending', 'accepted'),
    [
        ('tabular', None, True),
        ('other', None, False),
        ('tabular', 'other', False),
    ],
)
def test_tabular_deploy_checks_active_and_pending_config(
    active, pending, accepted
):
    cell = next(
        c
        for c in _cells(EXAMPLES / 'kumo_tabular_quickstart.ipynb', 'code')
        if 'workspace = WorkspaceClient()' in c
    )

    def config(model):
        return SimpleNamespace(
            served_entities=[
                SimpleNamespace(entity_name=model, entity_version='1')
            ]
        )

    workspace = Mock()
    workspace.serving_endpoints.get.return_value = SimpleNamespace(
        config=config(active),
        pending_config=config(pending) if pending else None,
    )
    namespace = dict(
        WorkspaceClient=lambda: workspace,
        NotFound=LookupError,
        MODEL_NAME='tabular',
        MODEL_VERSION='1',
        ENDPOINT_NAME='demo',
    )
    if accepted:
        exec(cell, namespace)
    else:
        with pytest.raises(ValueError, match='already exists'):
            exec(cell, namespace)
    workspace.serving_endpoints.create.assert_not_called()
    workspace.serving_endpoints.wait_get_serving_endpoint_not_updating.assert_not_called()


@pytest.mark.parametrize('failure', ['UPDATE_FAILED', 'UPDATE_CANCELED'])
@pytest.mark.parametrize(
    ('ready', 'model', 'version', 'expected'),
    [
        ('READY', 'tabular', '1', None),
        ('NOT_READY', 'tabular', '1', RuntimeError),
        ('READY', 'other', '1', ValueError),
        ('READY', 'tabular', '2', ValueError),
    ],
)
def test_tabular_failed_update_checks_active_deployment(
    failure, ready, model, version, expected
):
    from databricks.sdk.errors import OperationFailed

    cell = next(
        c
        for c in _cells(EXAMPLES / 'kumo_tabular_quickstart.ipynb', 'code')
        if 'wait_get_serving_endpoint_not_updating(' in c
    )
    workspace = Mock()
    error = OperationFailed(failure)
    workspace.serving_endpoints.wait_get_serving_endpoint_not_updating.side_effect = error
    workspace.serving_endpoints.get.return_value = SimpleNamespace(
        state=SimpleNamespace(ready=SimpleNamespace(value=ready)),
        config=SimpleNamespace(
            served_entities=[
                SimpleNamespace(entity_name=model, entity_version=version)
            ]
        ),
    )
    factory = Mock()
    namespace = dict(
        workspace=workspace,
        OperationFailed=OperationFailed,
        ENDPOINT_NAME='demo',
        MODEL_NAME='tabular',
        MODEL_VERSION='1',
        timedelta=timedelta,
        SDMClient=SimpleNamespace(for_databricks_serving=factory),
        KumoTabular=Mock(),
    )
    if expected:
        with pytest.raises(expected) as caught:
            exec(cell, namespace)
        assert caught.value.__cause__ is error
        factory.assert_not_called()
    else:
        exec(cell, namespace)
        factory.assert_called_once_with('demo', timeout=900)


def test_tabular_sampling_is_independent_of_source_order():
    import pandas as pd

    cells = _cells(EXAMPLES / 'kumo_tabular_quickstart.ipynb', 'code')
    prepare = next(c for c in cells if c.startswith('products ='))
    classify = next(c for c in cells if c.startswith("TARGET = 'Color'")).split(
        'request = Request('
    )[0]
    regress = next(c for c in cells if c.startswith('priced =')).split(
        'request = Request('
    )[0]
    frame = pd.DataFrame(
        {
            'ProductID': range(100),
            'Color': ['red', 'blue'] * 50,
            'ListPrice': range(1, 101),
        }
    )
    for name in [
        'rowguid',
        'ModifiedDate',
        'Name',
        'ProductNumber',
        'SellStartDate',
        'SellEndDate',
        'DiscontinuedDate',
    ]:
        frame[name] = ''
    results = []
    for source in [frame, frame.sample(frac=1, random_state=42)]:
        spark = Mock()
        spark.table.return_value.toPandas.return_value = source
        namespace = dict(spark=spark, PRODUCTS='products')
        exec(prepare, namespace)
        exec(classify, namespace)
        exec(regress, namespace)
        results.append(namespace)
    for name in ['held_out', 'context', 'to_price', 'priced']:
        pd.testing.assert_frame_equal(results[0][name], results[1][name])
