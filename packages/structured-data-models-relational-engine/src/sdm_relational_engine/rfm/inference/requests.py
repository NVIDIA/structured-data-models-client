# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Client-owned, not upstream's. Upstream also defined validate/parse/evaluate
# request and response types; this client only ever POSTs a prediction, so only
# the prediction request and response are kept.

from typing import Any

from pydantic.dataclasses import dataclass
from sdm_relational_engine.rfm.inference.config import (
    InferenceConfig,
    RunMode,
)
from sdm_relational_engine.rfm.inference.context import PredictionContext


@dataclass
class RFMPredictRequest:
    context: PredictionContext
    run_mode: RunMode
    use_prediction_time: bool = False
    query: str = ''
    inference_config: InferenceConfig | None = None
    return_embeddings: bool = False


@dataclass
class RFMPredictResponse:
    prediction: dict[str, Any]
