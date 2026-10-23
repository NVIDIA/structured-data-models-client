# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What a model is, to this client.

A model is described by two things that are deliberately kept apart.

A **profile** says what the model may be *asked*. It is named here (the
language's ``RELATIONAL`` or ``TABULAR``) and comes from ``predictive_query``
where an old-route check reads it, never on the core route (T3.16, the
structure design's rule 11); it is data, not code: a frozen set of flags obeying one
normative rule: a capability may only reject. The language accepts a fixed
set of queries and every model accepts a subset of it, so declaring a model can
never widen what the language allows.

A **runtime contract** says what this *deployment* can do: which task kinds and
output fields it serves, whether it holds sessions, and the limits it enforces.
A profile cannot know whether a checkpoint for regression was packaged, so
those facts live beside it and are never duplicated into it.

An **adapter** does what neither can describe: shape the wire request,
normalize the response, choose a session strategy, release resources. It does
not own query-shape validation. The profile does.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Protocol

from sdm_client.errors import UnsupportedRequestError

if TYPE_CHECKING:  # pragma: no cover - imported for typing only.
    from predictive_query import Capabilities, Profile

    from sdm_client.request import Request
    from sdm_client.result import PredictionResult

# The capability flags that describe the *shape* of a request, as opposed to
# the deployment concerns in the same dataclass. Only these are derivable from
# a request, so only these are compared.
SHAPE_CAPABILITIES = (
    'relationships',
    'multiple_tables',
    'link_prediction',
    'forecasting',
    'static',
    'temporal',
)


@dataclass(frozen=True)
class RuntimeContract:
    r"""What a deployment of the model serves.

    Attributes:
        tasks: The task kinds the endpoint accepts. Narrower than the
            contract's enum: a tabular model takes only the coarse
            ``'classification'``.
        outputs: The output fields the model produces.
        inference_options: Inference-config keys the endpoint accepts.
        explanation_formats: Named explanation shapes, when it explains.
        sessions: Whether the endpoint holds a pinned context.
        limits: Sizes the endpoint enforces, so the client can refuse locally
            rather than eat a 500.
        unread_feature_stypes: Semantic types the endpoint accepts and the
            model does not read. A column carrying one is answered around
            rather than refused, so the caller is told it carried nothing.
    """

    tasks: frozenset[str]
    outputs: frozenset[str]
    explanation_formats: tuple[str, ...] = ()
    sessions: bool = True
    limits: Mapping[str, int] = field(default_factory=dict)
    #: Inference-config keys this deployment accepts. Empty means it takes no
    #: inference config at all. A driver built for a different backend may
    #: default to sending more than this, which the endpoint refuses.
    inference_options: frozenset[str] = frozenset()
    #: Semantic types this deployment takes and the model never reads. The
    #: request succeeds and the column contributes nothing, which is worth
    #: saying out loud.
    unread_feature_stypes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ModelSpec:
    r"""A model id, what it may be asked, and what its deployment serves."""

    model_id: str
    #: The language's profile the model is asked through, by its name
    #: (``RELATIONAL`` or ``TABULAR``): the name the core validates a query
    #: for, and :func:`language_profile` the old PQL package's object.
    profile: str
    runtime: RuntimeContract


#: How each of the language's profiles names the model it is asked of
#: (``Profile.capabilities.model``), which a refusal of a query says, held
#: here so the core route words it with nothing of the old PQL package
#: (T3.16, rule 11); ``tests/test_t316_the_schema_as_it_stands.py`` holds
#: it to the language's.
PROFILE_MODELS = {
    'FINE_TUNING': 'fine-tuning',
    'RELATIONAL': 'a relational model',
    'TABULAR': 'a tabular model',
    'TIMESERIES': 'a time-series model',
    'ANOMALY': 'an anomaly-detection model',
}


def language_profile(name: str) -> Profile:
    r"""The old PQL package's profile named ``name``, for the old route's
    validator and the capability checks that read it."""
    import predictive_query.profiles as profiles

    found: Profile = getattr(profiles, name)
    return found


@dataclass(frozen=True)
class ModelCapabilities:
    r"""What :meth:`~sdm_client.SDMClient.capabilities` reports.

    Describes the client side. Nothing here is read from the endpoint, and
    ``source`` says so, so a caller cannot mistake it for the endpoint's own
    answer.
    """

    model: str
    tasks: tuple[str, ...]
    outputs: tuple[str, ...]
    single_table: bool
    multiple_tables: bool
    relationships: bool
    explanation_formats: tuple[str, ...] = ()
    sessions: bool = True
    source: str = 'sdk'


class ModelAdapter(Protocol):
    r"""The per-model half of a prediction.

    Deliberately small. Everything a profile or a runtime contract can already
    state is left to them; what remains is the work that genuinely differs
    between models.
    """

    spec: ModelSpec

    def default_output(self, request: Request) -> tuple[str, ...]:
        r"""The output fields to ask for when the caller named none."""
        ...

    def prediction_plan(
        self, request: Request, outputs: tuple[str, ...]
    ) -> Iterable[dict[str, Any]]:
        r"""The wire payloads for one request, in the order to send them.

        More than one when the request exceeds the endpoint's per-call cap.
        Batching lives here because it is transport, not part of the question.
        """
        ...

    def session_plan(
        self, request: Request, outputs: tuple[str, ...]
    ) -> dict[str, Any]:
        r"""The payload that pins this request's context on the endpoint."""
        ...

    def session_prediction_plan(
        self,
        request: Request,
        outputs: tuple[str, ...],
        pinned: Mapping[str, Any] | None = None,
    ) -> Iterable[dict[str, Any]]:
        r"""The payloads that score against an already-pinned context.

        ``pinned`` is the envelope the session opened with, minus the context
        itself. The schema is not sent again on a scoring call, so the rows
        have to be written to describe the one that was pinned.
        """
        ...

    def normalize(
        self,
        response: dict[str, Any],
        outputs: tuple[str, ...],
        payload: dict[str, Any],
    ) -> PredictionResult:
        r"""Turn one endpoint response into the client's result type.

        ``payload`` is the request that produced it. A batch's answers are
        indexed within that batch, so mapping them back to what the caller
        asked about needs the request beside the response.
        """
        ...

    def explain_error(
        self, error: Exception, payload: dict[str, Any]
    ) -> Exception:
        r"""A chance to add what the endpoint's message left out.

        A NIM sometimes reports a rule without saying which column broke it.
        The request that was just refused knows, and the adapter is holding it,
        so the answer is one lookup away. Returns the error to raise, the one
        it was given, unless it has something to add.
        """
        return error

    def close(self) -> None:
        r"""Release anything this adapter opened that the client cannot see."""
        ...


def required_capabilities(request: Request) -> Capabilities:
    r"""What answering ``request`` would require of a model.

    Built with :meth:`Capabilities.nothing` and switched on, so a flag this
    request does not need stays off and cannot cause a spurious rejection.
    """
    from predictive_query import Capabilities

    kind = request.task.kind
    temporal = bool(request.task.options.get('anchor_time_column')) or kind in {
        'forecasting',
        'temporal_link_prediction',
    }
    return Capabilities.nothing('this request').also(
        relationships=bool(request.schema.relationships),
        multiple_tables=len(request.schema) > 1,
        link_prediction=kind == 'temporal_link_prediction',
        forecasting=kind == 'forecasting',
        temporal=temporal,
        static=not temporal,
    )


def check_request_against_model(
    request: Request, spec: ModelSpec, outputs: tuple[str, ...]
) -> None:
    r"""Refuse, locally, a request the model cannot answer.

    Three checks, in the order that produces the most useful message: the
    shape the profile rejects, then the task kind, then the output fields.
    A shape mismatch explains the other two, so it is reported first.

    Raises:
        UnsupportedRequestError: With the model named, and what it does
            support.
    """
    required = required_capabilities(request)
    supported = language_profile(spec.profile).capabilities
    unsupported = [
        name
        for name in SHAPE_CAPABILITIES
        if getattr(required, name) and not getattr(supported, name)
    ]
    if unsupported:
        raise UnsupportedRequestError(
            f'{spec.model_id} cannot answer a request needing '
            f'{", ".join(sorted(unsupported))}',
            details={
                'model': spec.model_id,
                'unsupported': sorted(unsupported),
            },
        )

    if request.task.kind not in spec.runtime.tasks:
        raise UnsupportedRequestError(
            f'{spec.model_id} does not serve the task kind '
            f'{request.task.kind!r}; it serves '
            f'{sorted(spec.runtime.tasks)}',
            details={
                'model': spec.model_id,
                'task': request.task.kind,
                'supported': sorted(spec.runtime.tasks),
            },
        )

    unknown = [name for name in outputs if name not in spec.runtime.outputs]
    if unknown:
        raise UnsupportedRequestError(
            f'{spec.model_id} does not produce {sorted(unknown)}; it produces '
            f'{sorted(spec.runtime.outputs)}',
            details={
                'model': spec.model_id,
                'unsupported_outputs': sorted(unknown),
                'supported_outputs': sorted(spec.runtime.outputs),
            },
        )

    cap = spec.runtime.limits.get('context_rows_max')
    rows = request._context_rows_per_call()
    if cap is not None and rows > cap:
        raise UnsupportedRequestError(
            f'this request carries {rows:,} context rows and {spec.model_id} '
            f'reads at most {cap:,}. A request built directly is sent as it '
            f'was given, so choose the rows you want as context, or gather it '
            f'with materialize(), which keeps the context under the cap and '
            f'says how',
            details={
                'model': spec.model_id,
                'context_rows': rows,
                'context_rows_max': cap,
            },
        )


def describe(spec: ModelSpec) -> ModelCapabilities:
    r"""The public description of a model, for ``capabilities()``."""
    supported = language_profile(spec.profile).capabilities
    return ModelCapabilities(
        model=spec.model_id,
        tasks=tuple(sorted(spec.runtime.tasks)),
        outputs=tuple(sorted(spec.runtime.outputs)),
        single_table=True,
        multiple_tables=supported.multiple_tables,
        relationships=supported.relationships,
        explanation_formats=spec.runtime.explanation_formats,
        sessions=spec.runtime.sessions,
        source='sdk',
    )


def capability_names() -> tuple[str, ...]:
    r"""Every flag ``Capabilities`` carries, for tests that assert coverage."""
    from predictive_query import Capabilities

    return tuple(f.name for f in fields(Capabilities) if f.name != 'model')
