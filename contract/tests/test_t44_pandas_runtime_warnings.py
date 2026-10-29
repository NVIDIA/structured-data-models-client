# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The owner's general pandas-warning policy preserves nonwarning checks.

These are authored comparator inputs, not native runtime recordings.
Historical HOST-5 controls remain separate and unchanged.
"""

from __future__ import annotations

import copy

import pytest
from test_runtime_deprecations import _observation, compare, host

NOTICE = ('UserWarning', 'sdm_client')
RULES = dict.fromkeys(('schema', 'warnings', 'result', 'payload'), 'gate')


def compared(mine, theirs, *, approved=True, against='python-old', stop=None):
    report = compare.Report(RULES)
    report.runtime_warnings = approved
    compare._host_fields(
        report,
        {'id': 't44.runtime-warnings'},
        'python-core',
        mine,
        theirs,
        against,
        stop,
    )
    return report


@pytest.mark.parametrize('origin', ['pandas', 'pyarrow'])
@pytest.mark.parametrize('against', ['python-old', 'python-ref'])
@pytest.mark.parametrize(
    ('old', 'core'),
    [
        (['Pandas4Warning'], []),
        ([], ['FutureWarning']),
        (['DeprecationWarning'], ['RuntimeWarning', 'UserWarning']),
    ],
)
def test_runtime_warning_classes_counts_and_sides_do_not_block(
    origin, against, old, core
):
    mine = _observation(host(*[(name, origin) for name in core], NOTICE))
    theirs = _observation(host(NOTICE, *[(name, origin) for name in old]))
    before = copy.deepcopy((mine, theirs))
    report = compared(mine, theirs, against=against)
    assert not report.failures
    assert any('HOST-6' in note for note in report.notes)
    assert (mine, theirs) == before


def test_general_warning_policy_is_inactive_without_owner_approval():
    mine = _observation(host(NOTICE))
    theirs = _observation(host(('Pandas4Warning', 'pyarrow'), NOTICE))
    report = compared(mine, theirs, approved=False)
    assert {rule for rule, _ in report.failures} == {'warnings'}


@pytest.mark.parametrize('origin', ['sdm_client', 'numpy', None])
def test_another_origin_is_not_a_pandas_runtime_warning(origin):
    mine = _observation(host(NOTICE))
    theirs = _observation(host(('Pandas4Warning', origin), NOTICE))
    report = compared(mine, theirs)
    assert {rule for rule, _ in report.failures} == {'warnings'}


@pytest.mark.parametrize('which', ['mine', 'theirs'])
@pytest.mark.parametrize('broken', ['missing', 'short', 'wrong-class'])
def test_missing_or_inconsistent_origin_cannot_classify_a_warning(
    which, broken
):
    mine = _observation(host(('Pandas4Warning', 'pandas'), NOTICE))
    theirs = _observation(host(('Pandas4Warning', 'pyarrow'), NOTICE))
    target = (mine if which == 'mine' else theirs)['host']
    if broken == 'missing':
        del target['origins']
    elif broken == 'short':
        target['origins'].pop()
    else:
        target['origins'][0]['class'] = 'another-class'
    report = compared(mine, theirs)
    assert {rule for rule, _ in report.failures} == {'warnings'}


@pytest.mark.parametrize('which', ['mine', 'theirs'])
def test_unobserved_historical_origins_keep_their_warning_classes(which):
    # Historical python-ref deliberately omitted origins. Its retained
    # notice needs no invented origin to match the other route's notice.
    historical = _observation(host(NOTICE))
    del historical['host']['origins']
    observed = _observation(host(('Pandas4Warning', 'pyarrow'), NOTICE))
    mine, theirs = (
        (historical, observed) if which == 'mine' else (observed, historical)
    )
    before = copy.deepcopy((mine, theirs))
    assert not compared(mine, theirs, against='python-ref').failures
    assert (mine, theirs) == before


def test_nonruntime_warnings_keep_their_order_and_occurrences():
    first = _observation(host(NOTICE, ('SDMWarning', 'sdm_client')))
    second = _observation(
        host(('Pandas4Warning', 'pandas'), ('SDMWarning', 'sdm_client'), NOTICE)
    )
    assert {rule for rule, _ in compared(first, second).failures} == {
        'warnings'
    }
    assert {
        rule
        for rule, _ in compared(
            _observation(host()), _observation(host(NOTICE))
        ).failures
    } == {'warnings'}


def test_held_stop_keeps_the_nonruntime_warning_prefix():
    mine = _observation(host(NOTICE, ('SDMWarning', 'sdm_client')))
    theirs = _observation(host(('Pandas4Warning', 'pyarrow'), NOTICE))
    assert not compared(mine, theirs, stop='materialize').failures
    assert compared(mine, theirs).failures


@pytest.mark.parametrize('field', ['schema', 'result', 'payloads'])
def test_runtime_warning_policy_cannot_hide_other_public_differences(field):
    mine = _observation(host(NOTICE))
    theirs = _observation(host(('Pandas4Warning', 'pyarrow'), NOTICE))
    mine['public'][field] = (
        [
            {
                'method': 'POST',
                'path': '/v1/predictions',
                'parsed': {'changed': True},
            }
        ]
        if field == 'payloads'
        else {'changed': True}
    )
    report = compared(mine, theirs)
    expected = 'payload' if field == 'payloads' else field
    assert {rule for rule, _ in report.failures} == {expected}


def test_the_general_policy_is_recorded_as_an_actual_owner_decision():
    assert compare.owner_approved('HOST-6')
