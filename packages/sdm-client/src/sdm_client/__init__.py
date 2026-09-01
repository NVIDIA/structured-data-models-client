# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""One Python client for NVIDIA's structured-data foundation models.

Structured-data foundation models predict in context: you give the model
labelled rows and the rows you want scored, and it answers in one pass. There
is no training loop and no per-dataset artifact.

>>> from sdm_client import Request, StructuredDataClient, KumoTabular
>>> request = Request(context=labelled, predict=to_score,
...                   target='churn', task='classification')
>>> with StructuredDataClient(url='http://localhost:8000') as client:
...     print(client.predict(request, model=KumoTabular()).to_pandas())

The model is an argument, not a method name, so the same request runs on any
model whose profile accepts it.
"""

from sdm_client._version import __version__
from sdm_client.base import ModelCapabilities, ModelSpec, RuntimeContract
from sdm_client.client import FittedModel, StructuredDataClient
from sdm_client.errors import (
    MissingExtraError,
    NimRequestError,
    StructuredDataError,
    UnknownModelError,
    UnsupportedRequestError,
)
from sdm_client.models import KumoRelational, KumoTabular, Model
from sdm_client.request import Request, Split, Target, Task
from sdm_client.result import PredictionResult
from sdm_client.schema import Column, Relationship, Schema, Table

__all__ = [
    'Column',
    'FittedModel',
    'KumoRelational',
    'KumoTabular',
    'MissingExtraError',
    'Model',
    'ModelCapabilities',
    'ModelSpec',
    'NimRequestError',
    'PredictionResult',
    'Relationship',
    'Request',
    'RuntimeContract',
    'Schema',
    'Split',
    'StructuredDataClient',
    'StructuredDataError',
    'Table',
    'Target',
    'Task',
    'UnknownModelError',
    'UnsupportedRequestError',
    '__version__',
]
