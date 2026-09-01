# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The models this client can send to.

Pass one as an argument rather than calling a differently-named method:

>>> client.predict(request, model=KumoTabular())
>>> client.predict(request, model=KumoRelational())

Each is a thin handle over a :class:`~sdm_client.base.ModelSpec`. A handle
carries no state and no configuration -- gathering knobs live on the request,
output fields on the call -- so constructing one is free and two of the same
kind are interchangeable.

The declarations below are **read from the deployments**, not from a design
document: ``tfm-serving``'s tabular adapter and the relational NIM's universal
adapter each publish their own task and output lists, and these mirror them.
Where a source document disagreed, the servers won.
"""

from __future__ import annotations

from predictive_query import RELATIONAL, TABULAR
from sdm_api import TFM_MODEL_KUMO_RELATIONAL, TFM_MODEL_KUMO_TABULAR

from sdm_client.base import ModelSpec, RuntimeContract

# Sizes the relational endpoint enforces. It refuses past them, so the client
# refuses first with a message that names the limit. See the architecture note
# on inference knobs: `run_mode` is derived from these rather than being set.
RELATIONAL_LIMITS = {
    'context_rows_fast': 1_000,
    'context_rows_max': 10_000,
    'predict_rows_max': 1_000,
    'predict_rows_temporal_link_prediction': 200,
    'explanation_predict_rows': 1,
    'hops_max': 6,
    'tables_max': 15,
}

KUMO_TABULAR = ModelSpec(
    model_id=TFM_MODEL_KUMO_TABULAR,
    profile=TABULAR,
    runtime=RuntimeContract(
        # The endpoint takes the coarse kind only and decides binary versus
        # multiclass itself, so this client never narrows it here.
        tasks=frozenset({'classification', 'regression'}),
        outputs=frozenset({'prediction', 'probabilities', 'quantiles'}),
    ),
)

KUMO_RELATIONAL = ModelSpec(
    model_id=TFM_MODEL_KUMO_RELATIONAL,
    profile=RELATIONAL,
    runtime=RuntimeContract(
        tasks=frozenset(
            {
                'classification',
                'binary_classification',
                'multiclass_classification',
                'regression',
                'forecasting',
                'temporal_link_prediction',
            }
        ),
        outputs=frozenset(
            {
                'prediction',
                'probabilities',
                'scores',
                'rankings',
                'embeddings',
                'quantiles',
                'explanation',
            }
        ),
        explanation_formats=('feature_importance', 'subgraph'),
        limits=RELATIONAL_LIMITS,
        # What the deployment accepts. The engine was written against a
        # different backend and defaults to sending `output_type` and
        # `target_transforms` as well, which this one refuses outright.
        inference_options=frozenset(
            {
                'kind',
                'num_estimators',
                'column_shuffle',
                'category_shuffle',
                'hop_shuffle',
                'class_shuffle',
            }
        ),
    ),
)


class Model:
    r"""Base for the model handles. Not useful on its own."""

    spec: ModelSpec

    @property
    def name(self) -> str:
        r"""The model id sent on the wire."""
        return self.spec.model_id

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Model) and other.spec is self.spec

    def __hash__(self) -> int:
        return hash(self.spec.model_id)

    def __repr__(self) -> str:
        return f'{type(self).__name__}()'


class KumoTabular(Model):
    r"""One table in, a prediction per row out.

    Takes ``classification`` or ``regression``, and produces ``prediction``,
    ``probabilities`` and ``quantiles``. It reads no relationships, so a
    request carrying related tables is refused before it is sent.

    Needs no extra: this model does no client-side work beyond shaping the
    request, so it ships in the base install and runs anywhere.
    """

    spec = KUMO_TABULAR


class KumoRelational(Model):
    r"""Several related tables, joined as a graph.

    Takes every task kind the contract carries except ``ranking``, and produces
    all seven output fields including ``explanation``.

    Needs ``pip install sdm-client[relational]``: building the graph and
    sampling it is heavy client-side work that lives in a native package.
    """

    spec = KUMO_RELATIONAL


#: Every model this client can send to, by id.
MODELS = {
    KUMO_TABULAR.model_id: KUMO_TABULAR,
    KUMO_RELATIONAL.model_id: KUMO_RELATIONAL,
}
