# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``inference_config`` must not swallow a misspelled key.

pydantic's default is
``extra='ignore'``, so ``output_typ='mean'`` used to return the *median* with
no diagnostic -- on a zero-inflated target, a column of ``0.0`` where the
caller asked for a mean. Nothing forwards an unknown key (the wire payload is
built from the declared fields alone), so there is no forward compatibility to
weigh against saying so.
"""

from typing import Any

import pytest
from predictive_query.schema.task import TaskType
from pydantic import ValidationError
from sdm_relational_engine.api.rfm.inference import (
    ClassificationInferenceConfig,
    InferenceConfig,
    RegressionInferenceConfig,
)


@pytest.mark.parametrize(
    ('config_class', 'kwargs'),
    [
        (RegressionInferenceConfig, {'output_typ': 'mean'}),
        (RegressionInferenceConfig, {'target_transform': ['clip']}),
        (InferenceConfig, {'num_estimatorz': 4}),
        (InferenceConfig, {'column_shufle': True}),
        (ClassificationInferenceConfig, {'class_shufle': True}),
    ],
)
def test_an_unknown_key_is_named(config_class: Any, kwargs: Any) -> None:
    with pytest.raises(ValidationError) as excinfo:
        config_class(**kwargs)
    assert next(iter(kwargs)) in str(excinfo.value)


@pytest.mark.parametrize(
    ('config_class', 'kwargs'),
    [
        (InferenceConfig, {}),
        (InferenceConfig, {'num_estimators': 4, 'column_shuffle': True}),
        (InferenceConfig, {'category_shuffle': True, 'hop_shuffle': True}),
        (
            ClassificationInferenceConfig,
            {'class_shuffle': True, 'num_estimators': 2},
        ),
        (RegressionInferenceConfig, {'output_type': 'mean'}),
        (
            RegressionInferenceConfig,
            {'output_type': 'quantiles', 'target_transforms': ['clip', None]},
        ),
    ],
)
def test_every_declared_key_is_still_accepted(
    config_class: Any, kwargs: Any
) -> None:
    config = config_class(**kwargs)
    for name, value in kwargs.items():
        assert getattr(config, name) == value


def test_a_bad_value_for_a_known_key_is_still_rejected() -> None:
    with pytest.raises(ValidationError, match='output_type'):
        RegressionInferenceConfig(output_type='meen')


@pytest.mark.parametrize('task_type', list(TaskType))
def test_from_task_type_still_builds_every_default(task_type: Any) -> None:
    assert isinstance(
        InferenceConfig.from_task_type(task_type), InferenceConfig
    )
