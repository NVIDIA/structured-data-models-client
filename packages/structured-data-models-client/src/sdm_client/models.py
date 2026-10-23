# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The models this client can send to.

Pass one as an argument rather than calling a differently-named method:

>>> client.predict(request, model=KumoTabular())
>>> client.predict(request, model=KumoRelational())

Each is a thin handle over a :class:`~sdm_client.base.ModelSpec`. A handle
carries no state and no configuration. Gathering knobs live on the request and
output fields on the call, so constructing one is free and two of the same
kind are interchangeable.

The declarations below mirror what deployed endpoints actually answer,
rather than what any design document says they should.

**A runtime contract describes a deployment, and one model id may have more
than one.** The relational declaration below is the broader of those in
service: six task kinds and all seven output fields. A narrower build of the
same id serves only classification and regression, and produces only
``prediction``, ``probabilities`` and ``quantiles``. Asking that one for a
forecast passes the checks here and is refused by the endpoint.

Reading this from the endpoint instead would settle it, but the Universal TFM
contract has no capabilities route and ``/v1/models`` returns an id and nothing
else. So this is the broader of the two on purpose: refusing locally what one
deployment can answer would be worse than letting the other say no.
"""

from __future__ import annotations

from sdm_api import TFM_MODEL_KUMO_RELATIONAL, TFM_MODEL_KUMO_TABULAR

from sdm_client.base import ModelSpec, RuntimeContract
from sdm_client.errors import StructuredDataError, UnknownModelError

# Sizes the relational endpoint enforces. `predict_rows_max` is the serving
# layer's own `TFM_MAX_TEST_ROWS`, measured at the boundary rather than read
# from a document: 10,000 rows to score are answered, and 10,001 come back
#
#   [422 UNSUPPORTED_SHAPE] N_test 10001 exceeds limit 10000.
#
# A deployment may be configured to hold to less, which is what the
# `rows_per_call` option on materialize() is for.
#
# The endpoint refuses past these, so the client refuses first with a message
# that names the limit. See the architecture note on inference knobs:
# `run_mode` is derived from these rather than being set.
#
# `predict_rows_max` and `predict_rows_temporal_link_prediction` restate the
# engine's own `_MAX_PRED_SIZE`, which defaults to the first and holds the
# second for `TaskType.TEMPORAL_LINK_PREDICTION`
# (sdm_relational_engine/rfm/rfm.py). The duplication predates this client,
# and it is load-bearing: this value is what the client asks the engine to
# batch at, so a drift would gather batches the endpoint refuses, or ask for
# smaller ones than it would take, and the engine's own refusal names a
# `batch_size=` argument this client's callers have not got. Change them
# together; `test_the_two_ceilings_are_the_engine_s_own` fails if they move
# apart.
RELATIONAL_LIMITS = {
    'context_rows_fast': 1_000,
    'context_rows_max': 10_000,
    'predict_rows_max': 10_000,
    'predict_rows_temporal_link_prediction': 200,
    'explanation_predict_rows': 1,
    'hops_max': 6,
    'tables_max': 15,
    'classes_max': 10,
}

# A relational regression is answered from a predicted distribution, and the
# endpoint reports its mean unless told otherwise; the engine reports the
# median, so that is asked for by default.
DEFAULT_PREDICTION_STATISTIC = 'median'
PREDICTION_STATISTICS = frozenset({'mean', 'median'})
SUMMARISED_TASK_KINDS = frozenset({'regression', 'forecasting'})

# The tabular endpoint refuses a context past this, exactly:
#
#   [422 UNSUPPORTED_SHAPE] N_train 10001 exceeds limit 10000.
#
# Measured against the deployment rather than read from a document, at the
# boundary: 10,000 rows answer, 10,001 refuse. Declared so the client refuses
# first, because the alternative is encoding and transmitting an arbitrarily
# large context to be told the limit at the far end.
TABULAR_LIMITS = {
    'context_rows_max': 10_000,
}

KUMO_TABULAR = ModelSpec(
    model_id=TFM_MODEL_KUMO_TABULAR,
    profile='TABULAR',
    runtime=RuntimeContract(
        # The endpoint takes the coarse kind only and decides binary versus
        # multiclass itself, so this client never narrows it here.
        tasks=frozenset({'classification', 'regression'}),
        outputs=frozenset({'prediction', 'probabilities', 'quantiles'}),
        limits=TABULAR_LIMITS,
    ),
)

KUMO_RELATIONAL = ModelSpec(
    model_id=TFM_MODEL_KUMO_RELATIONAL,
    profile='RELATIONAL',
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
        # Empty because nothing here takes a format argument: the contract's
        # `OutputSpec` is only `{fields: [...]}`, and the endpoint names the
        # shape it returns. Two names declared here appeared in no response.
        explanation_formats=(),
        limits=RELATIONAL_LIMITS,
        unread_feature_stypes=frozenset({'text'}),
        # Inference-config keys this client forwards, rather than keys the
        # deployment is known to serve. The engine also sends `output_type`,
        # whose choice travels as `output.prediction_statistic` instead, and
        # `target_transforms`, which does nothing on this model: the endpoint
        # accepts only `quantile`, reads it nowhere, and says the model emits
        # its quantiles natively. A deployment may still refuse a value given
        # for a key that is forwarded, and the shipping one refuses every
        # shuffle and every `num_estimators` but its own, so this list is not
        # a menu.
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

    _spec: ModelSpec

    @property
    def name(self) -> str:
        r"""The model id sent on the wire."""
        return self._spec.model_id

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Model) and other._spec is self._spec

    def __hash__(self) -> int:
        return hash(self._spec.model_id)

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

    _spec = KUMO_TABULAR


class KumoRelational(Model):
    r"""Several related tables, joined as a graph.

    Takes every task kind the contract carries except ``ranking``, and produces
    all seven output fields including ``explanation``.

    Needs ``pip install structured-data-models-client[relational]``: building the graph and
    sampling it is heavy client-side work that lives in a native package.
    """

    _spec = KUMO_RELATIONAL


#: Every model this client can send to, by id.
MODELS = {
    KUMO_TABULAR.model_id: KUMO_TABULAR,
    KUMO_RELATIONAL.model_id: KUMO_RELATIONAL,
}


def spec_for(model: str | Model) -> ModelSpec:
    r"""The specification behind a model handle or a model id.

    Shared by the two places a model is named, gathering and sending a request,
    so both accept the same arguments and refuse the same ones the same way.

    A handle names one of the models below, not a new one. It is worth exactly
    the spec it carries. That spec chooses the route a request is gathered for
    and the limits it is checked against. The id
    it names is a closed enum on the wire. One this module did not declare
    is refused here rather than gathering work that cannot be sent. ``Model``
    itself declares ``_spec`` without defining one, so a bare ``Model()`` gets
    that same refusal rather than an ``AttributeError``.

    Args:
        model: A model handle such as ``KumoTabular()``, or a model id.

    Raises:
        UnknownModelError: If ``model`` is an id or handle the contract does
            not carry.
        StructuredDataError: If ``model`` is neither a handle nor an id.
    """
    if isinstance(model, Model):
        spec = getattr(model, '_spec', None)
        if isinstance(spec, ModelSpec) and MODELS.get(spec.model_id) is spec:
            return spec
        named = getattr(spec, 'model_id', None) or type(model).__name__
        raise UnknownModelError(named, list(MODELS))
    if isinstance(model, str):
        spec = MODELS.get(model)
        if spec is None:
            raise UnknownModelError(model, list(MODELS))
        return spec
    raise StructuredDataError(
        f'model must be a model handle or a model id, got '
        f'{type(model).__name__}',
        code='INVALID_REQUEST',
    )
