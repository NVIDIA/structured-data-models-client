# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model metadata and the host transport-adapter protocol.

Rust owns query validation, prediction planning, sessions and result merging.
Profiles name the Rust validation policy; runtime contracts describe deployment
capabilities. Host adapters add context to endpoint errors and release resources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

# The capability flags that describe the *shape* of a request, as opposed to
# the deployment concerns in the same dataclass. Only these are derivable from
# a request, so only these are compared.


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
