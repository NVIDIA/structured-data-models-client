# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

r"""Runtime modes retained from the former ``kumo-api`` package.

They are used across the relational engine, so they live in this focused
module rather than in the removed upstream API tree.
"""

from dataclasses import field
from typing import Literal

from predictive_query.schema.common import StrEnum
from predictive_query.schema.task import TaskType
from pydantic import ConfigDict, conint
from pydantic.dataclasses import dataclass


class RunMode(StrEnum):
    r"""How much in-context data a prediction is given.

    The mode caps the number of in-context examples the client collects and
    selects the per-hop neighbour fanout, trading accuracy against latency and
    payload size. It is sent as ``inference.run_mode``.

    The caps are 100 (``debug``), 1,000 (``fast``), 5,000 (``normal``) and
    10,000 (``best``). The exact counts live in
    ``sdm_relational_engine.rfm.rfm``. Temporal link prediction always uses the
    ``fast`` budget regardless of the mode.
    """

    FAST = 'fast'
    NORMAL = 'normal'
    BEST = 'best'
    DEBUG = 'debug'


TargetTransform = Literal['clip', 'power', 'quantile']
RegressionOutput = Literal['mean', 'median', 'quantiles']

# Pydantic's default is `extra='ignore'`, which drops a misspelled key without
# a word and silently substitutes the default: `output_typ='mean'` returns the
# median. Nothing forwards an unknown key. The wire payload is built from the
# declared fields alone, so there is no forward compatibility to preserve,
# and a bad *value* for a known key is already rejected.
_CONFIG = ConfigDict(extra='forbid')


@dataclass(config=_CONFIG)
class InferenceConfig:
    num_estimators: conint(gt=0, le=4) = 1  # type: ignore
    column_shuffle: bool = False
    category_shuffle: bool = False
    hop_shuffle: bool = False

    @classmethod
    def from_task_type(cls, task_type: TaskType) -> 'InferenceConfig':
        if task_type.is_classification:
            return ClassificationInferenceConfig()
        if task_type in {TaskType.REGRESSION, TaskType.FORECASTING}:
            return RegressionInferenceConfig()
        return InferenceConfig()


@dataclass(config=_CONFIG)
class ClassificationInferenceConfig(InferenceConfig):
    class_shuffle: bool = False


@dataclass(config=_CONFIG)
class RegressionInferenceConfig(InferenceConfig):
    target_transforms: list[TargetTransform | None] = field(
        default_factory=lambda: ['quantile']
    )
    output_type: RegressionOutput = 'median'
