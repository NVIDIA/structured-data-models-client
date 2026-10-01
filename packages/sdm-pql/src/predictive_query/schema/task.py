# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1


from predictive_query.schema.common import StrEnum


class TaskType(StrEnum):
    r"""What a predictive query asks the model to do.

    The ``get_*`` staticmethods name the families these fall into, and
    each has a matching ``is_*`` property for asking about one task.
    """

    BINARY_CLASSIFICATION = 'binary_classification'
    MULTICLASS_CLASSIFICATION = 'multiclass_classification'
    MULTILABEL_CLASSIFICATION = 'multilabel_classification'
    MULTILABEL_RANKING = 'multilabel_ranking'
    REGRESSION = 'regression'
    TEMPORAL_LINK_PREDICTION = 'temporal_link_prediction'
    STATIC_LINK_PREDICTION = 'static_link_prediction'
    FORECASTING = 'forecasting'

    LINK_PREDICTION = 'link_prediction'  # Deprecated.

    @staticmethod
    def get_node_pred_tasks() -> list['TaskType']:
        r"""Every task that predicts a property of one entity."""
        return [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
            TaskType.MULTILABEL_RANKING,
            TaskType.REGRESSION,
            TaskType.FORECASTING,
        ]

    @property
    def is_node_pred(self) -> bool:
        return self in self.get_node_pred_tasks()

    @staticmethod
    def get_link_pred_tasks() -> list['TaskType']:
        r"""Every task that predicts a link between two entities."""
        return [
            TaskType.TEMPORAL_LINK_PREDICTION,
            TaskType.STATIC_LINK_PREDICTION,
            TaskType.LINK_PREDICTION,
        ]

    @property
    def is_link_pred(self) -> bool:
        return self in self.get_link_pred_tasks()

    @staticmethod
    def get_classification_tasks() -> list['TaskType']:
        r"""Every task whose answer is a class rather than a number."""
        return [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        ]

    @property
    def is_classification(self) -> bool:
        return self in self.get_classification_tasks()

    @staticmethod
    def get_multilabel_tasks() -> list['TaskType']:
        r"""Every task that may answer with more than one label at once."""
        return [
            TaskType.MULTILABEL_CLASSIFICATION,
            TaskType.MULTILABEL_RANKING,
        ]

    @property
    def is_multilabel(self) -> bool:
        return self in self.get_multilabel_tasks()

    @staticmethod
    def get_ranking_tasks() -> list['TaskType']:
        r"""Every task that answers with an ordered list rather than one value."""
        return TaskType.get_link_pred_tasks() + [TaskType.MULTILABEL_RANKING]

    @property
    def is_ranking(self) -> bool:
        return self in self.get_ranking_tasks()
