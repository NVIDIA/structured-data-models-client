# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import Literal

from predictive_query.schema.task import TaskType
from pydantic import ConfigDict, conint
from pydantic.dataclasses import dataclass

TargetTransform = Literal['clip', 'power', 'quantile']
RegressionOutput = Literal['mean', 'median', 'quantiles']

# Pydantic's default is `extra='ignore'`, which drops a misspelled key without
# a word and silently substitutes the default: `output_typ='mean'` returns the
# median. Nothing forwards an unknown key -- the wire payload is built from the
# declared fields alone -- so there is no forward compatibility to preserve,
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
