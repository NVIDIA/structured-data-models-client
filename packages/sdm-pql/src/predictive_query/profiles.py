# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

r"""What a query is for, and what the model on the other end can do.

Two things vary between callers of this package, and only one of them is a
capability.

**Mode** is what a query is *for*. ``PREDICT SUM(orders.price, 0, 30, days) FOR
EACH users.user_id`` defines a training target under :attr:`Mode.TRAINING` and asks
a question under :attr:`Mode.INFERENCE`. That is a fork in meaning, so it stays
explicit.

**Capabilities** are what a given model can answer. A single-table model cannot
aggregate across a relationship; a model without link prediction cannot rank a
foreign key. Those are not properties of the language, and a query one model
refuses may be perfectly good for another.

The rules a capability obeys are normative. The one that matters most:
**a capability may only reject**. It can turn a
query the language accepts into one this model will not answer. It can never do
the reverse, so the set of queries the language accepts is fixed and every model
accepts a subset of it.
"""

from dataclasses import dataclass, replace

from predictive_query.schema.common import StrEnum


class Mode(StrEnum):
    r"""Whether the query is defining a task or asking one.

    The same text means different things on either side of that line,
    which is why this is a mode rather than a capability.
    """

    #: The query defines what a model should learn. Naming particular
    #: entities is meaningless here: the task is over all of them.
    TRAINING = 'TRAINING'

    #: The query asks a model that already exists, possibly about named
    #: entities. Not spelled ``PREDICT``: every query starts with that
    #: word, and the two would mean different things.
    INFERENCE = 'INFERENCE'


@dataclass(frozen=True)
class Capabilities:
    r"""What a model can answer, as data rather than as code.

    Two kinds of thing live here and are deliberately kept apart, because
    merging them is how this rots. **Shape** is what the query touches, and is
    read straight off the resolved tree. **Task** is what the query asks for.
    A field belongs to exactly one group; if a new one seems to belong to both,
    it is two fields.

    Every field defaults to permitting. That is a **compatibility** default,
    for a caller adapting existing code. It is the wrong default for a model
    declaring itself: a model that forgets ``forecasting=False`` would
    silently accept forecasts it cannot answer. Declare a new model with
    :meth:`nothing` and switch on what it supports.

    Args:
        model: How the model is named in a diagnostic.
        key_columns_in_filters: *Deployment.* May a filter compare a primary
            or foreign key column?
        timestamp_columns_in_filters: *Deployment.* May a filter compare a
            timestamp column?
        relationships: *Shape.* May a query reach across a foreign key?
        multiple_tables: *Shape.* May a query name more than one table?
        temporal: *Shape.* May a target carry a time window?
        static: *Shape.* Does this model add a refusal of its own for a
            target with no time window? ``True`` means it adds none, which is
            not the same as every static target being answerable: the language
            refuses a static aggregation before any capability is consulted,
            so a model with ``static=True`` still takes only a static
            **column**. Read as a promise about static aggregations it is
            wrong, and on a schema with no time column that is most of them.
        entity_filters: *Shape.* May the entity carry a ``WHERE`` filter?
        counterfactuals: *Shape.* May a query carry an ``ASSUMING`` clause?
        derived_targets: *Shape.* May the target be a condition, aggregation or
            logical operation instead of a single column?
        link_prediction: *Task.* May a target rank a foreign key?
        forecasting: *Task.* May a query carry a ``FORECAST`` clause?
    """

    model: str = 'this model'

    # shape
    relationships: bool = True
    multiple_tables: bool = True
    temporal: bool = True
    static: bool = True
    entity_filters: bool = True
    counterfactuals: bool = True
    derived_targets: bool = True

    # task
    link_prediction: bool = True
    forecasting: bool = True

    # deployment. Not a property of the language nor of a model, but of the
    # stack serving it: one deployment could not serve key columns at all.
    key_columns_in_filters: bool = True
    timestamp_columns_in_filters: bool = True

    @classmethod
    def nothing(cls, model: str) -> 'Capabilities':
        r"""A model that supports nothing until it says otherwise.

        The base for declaring a real model, so that supporting something is
        an act rather than an omission::

            Capabilities.nothing('my-model').also(temporal=True, static=True)
        """
        return cls(
            model=model,
            relationships=False,
            multiple_tables=False,
            temporal=False,
            static=False,
            entity_filters=False,
            counterfactuals=False,
            derived_targets=False,
            link_prediction=False,
            forecasting=False,
            key_columns_in_filters=False,
            timestamp_columns_in_filters=False,
        )

    def also(self, **supported: bool) -> 'Capabilities':
        r"""The same capabilities, with each named one set as given."""
        unknown = set(supported) - {
            field for field in vars(self) if field != 'model'
        }
        if unknown:
            raise ValueError(f'not a capability: {sorted(unknown)}')
        # `replace` is typed against the declared fields, and `model` among
        # them is a str, so a bool-only mapping does not widen to it.
        return replace(self, **supported)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Profile:
    r"""A mode and a set of capabilities: everything a caller declares."""

    mode: Mode
    capabilities: Capabilities = Capabilities()

    def named(self, model: str) -> 'Profile':
        r"""The same profile, naming a particular model in its diagnostics."""
        return replace(
            self, capabilities=replace(self.capabilities, model=model)
        )


#: Fine-tuning a model on the query. Nothing about the language is
#: restricted, because the model does not exist yet; what applies is
#: :attr:`Mode.TRAINING`, which is about defining a task rather than
#: asking one.
FINE_TUNING = Profile(
    mode=Mode.TRAINING, capabilities=Capabilities(model='fine-tuning')
)

#: A pretrained relational model: many tables, relationships, link prediction.
RELATIONAL = Profile(
    mode=Mode.INFERENCE, capabilities=Capabilities(model='a relational model')
)

#: A single-table model. It scores one column of one table as it stands, so
#: an aggregation over a second table is something it cannot answer.
TABULAR = Profile(
    mode=Mode.INFERENCE,
    capabilities=Capabilities(
        model='a tabular model',
        relationships=False,
        multiple_tables=False,
        temporal=False,
        entity_filters=False,
        counterfactuals=False,
        derived_targets=False,
        link_prediction=False,
        forecasting=False,
    ),
)

#: A forecasting model. One table, and a target that must carry a window.
TIMESERIES = Profile(
    mode=Mode.INFERENCE,
    capabilities=Capabilities(
        model='a time-series model',
        relationships=False,
        multiple_tables=False,
        static=False,
        link_prediction=False,
    ),
)

#: An anomaly-detection model. One table, no window, nothing to rank.
ANOMALY = Profile(
    mode=Mode.INFERENCE,
    capabilities=Capabilities(
        model='an anomaly-detection model',
        relationships=False,
        multiple_tables=False,
        temporal=False,
        link_prediction=False,
        forecasting=False,
    ),
)

__all__ = [
    'ANOMALY',
    'FINE_TUNING',
    'RELATIONAL',
    'TABULAR',
    'TIMESERIES',
    'Capabilities',
    'Mode',
    'Profile',
]
