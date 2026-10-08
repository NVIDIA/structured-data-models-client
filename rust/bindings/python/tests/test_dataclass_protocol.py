# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from sdm_core.answers import HeldValues, encode


def nondefault(value, references=None):
    references = HeldValues() if references is None else references
    return json.loads(
        references.protocol(
            'dataclass_nondefault_fields',
            json.dumps(encode(value, references=references)),
        )
    )


def test_dataclass_defaults_inheritance_and_factories_are_observed():
    @dataclass
    class Base:
        count: int = 1

    @dataclass
    class Config(Base):
        values: list[int] = field(default_factory=list)

    assert nondefault(Config()) == []
    assert nondefault(Config(count=2, values=[3])) == ['count', 'values']
    assert nondefault({'count': 2}) is None
    assert nondefault(Config) is None


def test_dataclass_comparison_stops_before_later_field_access():
    events = []
    failure = RuntimeError('comparison stopped')

    class Compared:
        def __ne__(self, other):
            events.append('compare')
            raise failure

    def default():
        events.append('factory')
        return Compared()

    @dataclass
    class Config:
        first: object = field(default_factory=default)
        second: int = 1

        def __getattribute__(self, name):
            if name in ('first', 'second'):
                events.append(name)
            return object.__getattribute__(self, name)

    config = Config()
    events.clear()
    references = HeldValues()
    with pytest.raises(RuntimeError) as caught:
        nondefault(config, references)
    assert caught.value is failure
    assert references.error is failure
    assert events == ['first', 'factory', 'compare']
