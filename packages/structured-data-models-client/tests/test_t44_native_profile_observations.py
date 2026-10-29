# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite native observer effects stay separate from SDK notice assertions."""

from __future__ import annotations

import copy
import warnings
from contextlib import contextmanager

import pytest


class OptionEffects:
    """Attribute only warnings emitted inside an explicit native option call.

    The original complete warning and event journals remain authoritative.
    This helper supplies indices for the separate semantic schedule assertion.
    """

    def __init__(self):
        self.phase = None
        self.notices = []

    @contextmanager
    def at(self, phase):
        assert phase in (
            'option-read',
            'option-write',
            'option-entry',
            'option-exit',
        )
        previous = self.phase
        self.phase = phase
        try:
            yield
        finally:
            self.phase = previous

    def warning(self, message, category, filename, event=None):
        if self.phase is not None:
            self.notices.append(
                (message, category, filename, self.phase, event)
            )

    def snapshot(self, caught):
        remaining = set(range(len(caught)))
        records = []
        for message, category, filename, phase, event in self.notices:
            matches = [
                at
                for at in sorted(remaining)
                if caught[at].message is message
                and caught[at].category is category
            ]
            assert matches, 'an attributed warning was not forwarded to capture'
            at = matches[0]
            remaining.remove(at)
            assert caught[at].filename == filename
            records.append(
                {
                    'warning': at,
                    'event': event,
                    'phase': phase,
                    'category': f'{category.__module__}.{category.__qualname__}',
                    'message': str(message),
                    'filename': filename,
                }
            )
        return records


def product_notices(observed):
    indices = {record['warning'] for record in observed['option_effects']}
    assert len(indices) == len(observed['option_effects'])
    assert all(0 <= at < len(observed['warnings']) for at in indices)
    for record in observed['option_effects']:
        notice = observed['warnings'][record['warning']]
        assert record['message'] == notice['message']
        if 'category' in notice:
            assert record['category'] == notice['category']
    return [
        notice
        for at, notice in enumerate(observed['warnings'])
        if at not in indices
    ]


def semantic_events(observed):
    indices = [
        record['event']
        for record in observed['option_effects']
        if record['event'] is not None
    ]
    assert len(set(indices)) == len(indices)
    for at in indices:
        assert observed['events'][at] == 'warning'
    return [
        event
        for at, event in enumerate(observed['events'])
        if at not in indices
    ]


def test_option_effect_attribution_keeps_complete_raw_records():
    effects = OptionEffects()
    events = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        original = warnings.showwarning

        def recording(
            message, category, filename, lineno, file=None, line=None
        ):
            effects.warning(message, category, filename, len(events))
            events.append('warning')
            original(message, category, filename, lineno, file=file, line=line)

        warnings.showwarning = recording
        warnings.warn('SDK advice', UserWarning)
        with effects.at('option-read'):
            warnings.warn('native option read', DeprecationWarning)
        events.append('converted')
    observed = {
        'events': events,
        'warnings': [
            {
                'message': str(item.message),
                'category': f'{item.category.__module__}.{item.category.__qualname__}',
            }
            for item in caught
        ],
        'option_effects': effects.snapshot(caught),
    }
    original = copy.deepcopy(observed)
    assert product_notices(observed) == [observed['warnings'][0]]
    assert semantic_events(observed) == ['warning', 'converted']
    assert observed == original
    changed = copy.deepcopy(observed)
    changed['warnings'][1]['message'] += ' changed'
    with pytest.raises(AssertionError):
        product_notices(changed)
    changed = copy.deepcopy(observed)
    changed['events'][1] = 'missing callback'
    with pytest.raises(AssertionError):
        semantic_events(changed)
