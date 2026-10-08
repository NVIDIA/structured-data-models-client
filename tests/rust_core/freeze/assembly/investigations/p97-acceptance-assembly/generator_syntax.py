# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DV12's syntax-only projection, with independently executed parser witnesses."""

from __future__ import annotations

import ast
import inspect
import json
import sys
import warnings
from pathlib import Path

from generator_primitives import module
from support import contract, json_bytes, require, same_json, sha

PARSER = (
    'packages/structured-data-models-pql/src/predictive_query/parser/parser.py'
)
HELPER = 'tests/rust_core/pql/_pql.py'
SOURCES = (
    PARSER,
    HELPER,
    'rust/crates/pql/tests/custom_profiles.py',
    'rust/crates/pql/src/api.rs',
    'rust/crates/pql/src/visitor.rs',
)
INTERVENTION = 'DV12 source-authenticated PQL syntax diagnostics'


def plain(value):
    require(type(value) is dict and len(value) == 1, 'pql.syntax_encoding')
    if 'scalar' in value:
        require(
            type(value['scalar']) in (str, bool, int, type(None)),
            'pql.syntax_scalar',
        )
        return value['scalar']
    if 'list' in value:
        return [plain(item) for item in value['list']]
    require('mapping' in value, 'pql.syntax_mapping')
    result = {}
    for key, item in value['mapping']:
        name = plain(key)
        require(type(name) is str and name not in result, 'pql.syntax_key')
        result[name] = plain(item)
    return result


def syntax_result(value):
    return (
        type(value) is dict
        and set(value) == {'errors', 'warnings'}
        and value['warnings'] == []
        and type(value['errors']) is list
        and len(value['errors']) == 1
        and type(value['errors'][0]) is dict
        and set(value['errors'][0]) == {'code', 'span', 'message'}
        and value['errors'][0]['code'] == 'PQL_PARSE_SYNTAX'
        and value['errors'][0]['span'] is None
        and type(value['errors'][0]['message']) is str
        and bool(value['errors'][0]['message'].strip())
    )


def witness(root, value, outcome):
    operation = value.get('operation')
    if operation not in {'pql.validate', 'pql.profile', 'pql.options'}:
        return None
    if operation == 'pql.options':
        error = outcome['error']
        if not (
            outcome['status'] == 'refused'
            and type(error) is dict
            and error['class'] == 'builtins.ValueError'
            and error['code'] is None
            and error['details'] is None
        ):
            return None
    else:
        result = outcome['result']
        if not (
            type(result) is dict
            and 'mapping' in result
            and any(
                key == {'scalar': 'errors'} for key, item in result['mapping']
            )
        ):
            return None
        if not syntax_result(plain(result)):
            return None
    import predictive_query
    import sdm_core._native as native
    from generator_inputs import typed
    from generator_oracles import OriginalCalls
    from predictive_query.parser.parser import PQLParser

    arguments = value['arguments']
    mode = arguments.get('mode')
    if mode is None:
        mode = getattr(predictive_query, arguments['profile']).mode.value
    parser = PQLParser(mode=predictive_query.Mode(mode))
    require(
        Path(parser.to_parsed_predictive_query.__code__.co_filename).resolve()
        == (root / PARSER).resolve(),
        'pql.syntax_parser_source',
    )
    original = OriginalCalls(root)
    with warnings.catch_warnings(record=True) as old_warnings:
        warnings.simplefilter('always')
        try:
            original.invoke(
                parser.to_parsed_predictive_query, arguments['query']
            )
        except ValueError as error:
            old_message = str(error)
        else:
            return None
    require(not old_warnings, 'pql.syntax_parser_warnings')
    function = native.pql_validate
    require(inspect.isbuiltin(function), 'pql.syntax_native_builtin')
    calls = []

    def profile(frame, event, argument):
        if event == 'c_call' and argument is function:
            calls.append('pql_validate')

    require(sys.getprofile() is None, 'pql.syntax_profile_busy')
    sys.setprofile(profile)
    try:
        with warnings.catch_warnings(record=True) as native_warnings:
            warnings.simplefilter('always')
            supplied = (
                arguments['query'],
                json.dumps(arguments['schema']),
                'RELATIONAL',
                mode,
            )
            actual = json.loads(function(*supplied))
    finally:
        sys.setprofile(None)
    require(calls == ['pql_validate'], 'pql.syntax_native_call')
    require(not native_warnings, 'pql.syntax_native_warnings')
    if not syntax_result(actual):
        return None
    return {
        'input_sha256': sha(json_bytes(typed(value))),
        'sources': {name: sha((root / name).read_bytes()) for name in SOURCES},
        'original_parser_events': original.events,
        'original_parser_error': {
            'class': 'builtins.ValueError',
            'code': None,
            'details': None,
            'message': old_message,
        },
        'native_function': 'pql_validate',
        'native_calls': len(calls),
        'native_sha256': sha(Path(native.__file__).read_bytes()),
        'native_diagnostic': actual,
    }


def validate(root, inventory, recipe, case, original, native_sha256):
    from generator_inputs import typed

    require(
        recipe['generator']['name'] == 'pql.queries'
        and case['consumer_kind'] == 'original-python',
        'pql.syntax_family',
    )
    value = plain(recipe['input'])
    require(
        value['operation'] in {'pql.validate', 'pql.profile', 'pql.options'},
        'pql.syntax_operation',
    )
    evidence = original.get('pql_syntax_witness')
    require(
        type(evidence) is dict
        and set(evidence)
        == {
            'input_sha256',
            'sources',
            'original_parser_events',
            'original_parser_error',
            'native_function',
            'native_calls',
            'native_sha256',
            'native_diagnostic',
        },
        'pql.syntax_witness',
    )
    sources = {
        name: inventory['generator_adapter_sources'][name] for name in SOURCES
    }
    require(
        same_json(evidence['sources'], sources)
        and all(
            sha((root / name).read_bytes()) == digest
            for name, digest in sources.items()
        ),
        'pql.syntax_sources',
    )
    require(
        evidence['input_sha256'] == sha(json_bytes(recipe['input'])),
        'pql.syntax_input',
    )
    require(
        native_sha256 is not None
        and evidence['native_sha256'] == native_sha256
        and evidence['native_function'] == 'pql_validate'
        and type(evidence['native_calls']) is int
        and evidence['native_calls'] == 1,
        'pql.syntax_native_origin',
    )
    events = evidence['original_parser_events']
    require(
        type(events) is list and len(events) == 1, 'pql.syntax_original_origin'
    )
    event = events[0]
    parser_class = next(
        node
        for node in ast.parse((root / PARSER).read_bytes()).body
        if isinstance(node, ast.ClassDef) and node.name == 'PQLParser'
    )
    parser_function = next(
        node
        for node in parser_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == 'to_parsed_predictive_query'
    )
    require(
        set(event)
        == {
            'path',
            'function',
            'line',
            'source_sha256',
            'operation_calls',
            'native_calls',
        }
        and event['path'] == PARSER
        and event['function']
        in {
            'PQLParser.to_parsed_predictive_query',
            'to_parsed_predictive_query',
        }
        and event['source_sha256'] == sources[PARSER]
        and type(event['line']) is int
        and event['line'] == parser_function.lineno
        and type(event['operation_calls']) is int
        and event['operation_calls'] == 1
        and event['native_calls'] == 0,
        'pql.syntax_original_origin',
    )
    old_error = evidence['original_parser_error']
    require(
        type(old_error) is dict
        and set(old_error) == {'class', 'code', 'details', 'message'}
        and old_error['class'] == 'builtins.ValueError'
        and old_error['code'] is None
        and old_error['details'] is None
        and type(old_error['message']) is str
        and bool(old_error['message']),
        'pql.syntax_original_error',
    )
    actual = evidence['native_diagnostic']
    require(syntax_result(actual), 'pql.syntax_diagnostic')
    helper = module(root, HELPER)
    projected_old = {
        'errors': [
            {
                'code': 'PQL_PARSE_SYNTAX',
                'span': None,
                'message': old_error['message'],
            }
        ],
        'warnings': [],
    }
    require(
        helper.mismatch(projected_old, actual) is None,
        'pql.syntax_original_policy',
    )
    old, expected = case['reference'], case['expected']
    replacement = dict(old)
    if value['operation'] == 'pql.options':
        require(
            old['status'] == 'refused' and same_json(old['error'], old_error),
            'pql.syntax_wrapper_origin',
        )
        replacement['error'] = {
            **old_error,
            'message': actual['errors'][0]['message'],
        }
    else:
        old_result = plain(old['result'])
        require(
            old['status'] == 'success'
            and old['error'] is None
            and syntax_result(old_result)
            and old_result['errors'][0]['message'] == old_error['message']
            and helper.mismatch(old_result, actual) is None,
            'pql.syntax_returned_origin',
        )
        replacement['result'] = typed(
            {'warnings': actual['warnings'], 'errors': actual['errors']}
        )
    require(
        same_json(replacement, expected), 'pql.syntax_exact_native_observation'
    )
    correction = case['corrected_reference']
    require(
        correction['dv'] == 'DV12'
        and correction['intervention'] == INTERVENTION
        and correction['source_sha256'] == sources[HELPER],
        'pql.syntax_correction_scope',
    )
    contract.validate_corrected_reference(
        correction, [*inventory['accepted_dvs'], 'DV12']
    )
