# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local IDs need actual owner facts; legacy data retains exact equality."""

import copy
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).parents[1] / 'runners/fetch_bindings.py'
SPEC = importlib.util.spec_from_file_location('checked_fetch_bindings', PATH)
assert SPEC is not None and SPEC.loader is not None
bindings = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bindings)


def authored_fetch(run, attachment):
    fields = [{'name':'keys', 'type':{'type':'utf8'}, 'nullable':False}]
    return {
        'request': {'request':'lookup', 'table':{'source':['users']},
                    'key':['id'], 'columns':['id'], 'seeds':{'attachment':attachment+1,'columns':[]}},
        'plan': {'plan':'sql', 'expect':{'schema':'discover'},
                 'statements':[{'dialect':'postgres','text':'SELECT actual_bound_keys',
                                'params':[{'param':'text','attachment':attachment}]}]},
        'bindings': {'available':True,'run':run,'fetch':1,'owners':[
            {'attachment':attachment,'fields':fields,'metadata':{},'rows':1,
             'values':{'columns':['keys'],'types':None,'ordered':'a'*64,
                       'unordered':'a'*64,'unencodable':[]}}], 'witness':None},
    }


def test_local_ids_match_only_with_complete_observed_bound_content():
    left,right = authored_fetch(3,41), authored_fetch(7,4)
    before = copy.deepcopy((left,right))
    assert bindings.agree([left],[right])
    assert (left,right) == before
    del right['bindings']
    assert not bindings.agree([left],[right])


@pytest.mark.parametrize('changed',['value','rows','metadata','sql','table','missing','foreign','type'])
def test_corrupt_content_or_business_read_is_never_an_alpha_rename(changed):
    left,right = authored_fetch(3,41), authored_fetch(7,4)
    owner = right['bindings']['owners'][0]
    if changed == 'value': owner['values']['ordered'] = 'b'*64
    elif changed == 'rows': owner['rows'] = 2
    elif changed == 'metadata': owner['fields'][0]['metadata'] = {'actual':'different'}
    elif changed == 'sql': right['plan']['statements'][0]['text'] += ' WHERE false'
    elif changed == 'table': right['request']['table']['source'] = ['other_users']
    elif changed == 'missing': right['bindings']['owners'] = []
    elif changed == 'foreign': owner['attachment'] = 900
    else: owner['fields'][0]['type'] = {'type':'int64'}
    try:
        same = bindings.agree([left],[right])
    except bindings.BindingError:
        same = False
    assert not same


def test_reused_local_attachment_cannot_claim_different_content_in_one_run():
    first = authored_fetch(3,41)
    later = copy.deepcopy(first)
    later['bindings']['fetch'] = 2
    later['bindings']['owners'][0]['values']['ordered'] = 'b'*64
    with pytest.raises(bindings.BindingError):
        bindings.agree([first,later],copy.deepcopy([first,later]))


def test_legacy_exact_ids_still_match_and_different_ids_still_fail():
    left = authored_fetch(3,41)
    del left['bindings']
    assert bindings.agree([left],copy.deepcopy([left]))
    right = authored_fetch(3,4)
    del right['bindings']
    assert not bindings.agree([left],[right])
