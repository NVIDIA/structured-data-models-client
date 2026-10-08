# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""scripts/generate_wire_rs.py: what it writes, and what it refuses.

``edge/contract.json`` holds every shape the generator models, and
``edge.rs`` tests how the types it writes for them behave. Here: that the
committed ``edge/generated.rs`` is current, how the command line writes and
checks, and that every keyword the generator cannot express stops it rather
than being dropped. The ``rust-wire`` CI job runs this file.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SCRIPT = REPO / 'scripts' / 'generate_wire_rs.py'
EDGE = HERE / 'edge'


def _load_generator() -> Any:
    spec = importlib.util.spec_from_file_location('generate_wire_rs', SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Its dataclasses look their module up while being defined.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gen = _load_generator()


def _run(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )


def _generate(schemas: dict[str, Any]) -> str:
    spec = {'components': {'schemas': schemas}}
    return gen.generate(spec, source='test.json', source_sha256='0' * 64)


def _item(code: str, name: str) -> str:
    r"""The lines of the item ``name``, with the attributes above it."""
    blocks = code.split('\n\n')
    found = [b for b in blocks if f' {name} ' in b or f' {name};' in b]
    assert found, f'{name} is not in the output'
    return found[0]


def test_the_edge_fixture_is_what_the_generator_writes():
    result = _run(
        '--spec',
        EDGE / 'contract.json',
        '--output',
        EDGE / 'generated.rs',
        '--no-record',
        '--check',
    )
    assert result.returncode == 0, (
        f'{result.stderr}regenerate edge/generated.rs with the same command '
        'without --check'
    )


def test_without_a_record_it_writes_and_checks_only_the_output(tmp_path):
    output = tmp_path / 'generated.rs'
    args = ('--spec', EDGE / 'contract.json', '--output', output, '--no-record')
    assert _run(*args).returncode == 0
    assert [p.name for p in tmp_path.iterdir()] == ['generated.rs']
    assert output.read_text() == (EDGE / 'generated.rs').read_text()
    assert _run(*args, '--check').returncode == 0

    output.write_text(output.read_text() + '// edited\n')
    result = _run(*args, '--check')
    assert result.returncode == 1
    assert 'not up to date' in result.stderr


def test_the_record_names_the_output_the_generator_and_the_contract(tmp_path):
    output = tmp_path / 'generated.rs'
    args = ('--spec', EDGE / 'contract.json', '--output', output)
    assert _run(*args).returncode == 0
    record = json.loads((tmp_path / 'PROVENANCE.json').read_text())
    assert set(record) == {
        'generated_file',
        'generated_sha256',
        'generator',
        'generator_sha256',
        'source_path',
        'source_repository',
        'source_revision',
        'source_sha256',
    }
    assert record['generated_file'] == 'generated.rs'
    assert record['source_path'] == 'contract.json'
    assert _run(*args, '--check').returncode == 0

    record['source_revision'] = 'f' * 40
    (tmp_path / 'PROVENANCE.json').write_text(json.dumps(record))
    result = _run(*args, '--check')
    assert result.returncode == 1
    assert 'PROVENANCE.json' in result.stderr


def test_an_empty_schema_for_other_keys_leaves_an_object_open():
    code = _generate(
        {
            'Open': {
                'type': 'object',
                'additionalProperties': {},
                'properties': {'a': {'type': 'string'}},
            },
            'Closed': {
                'type': 'object',
                'additionalProperties': False,
                'properties': {'a': {'type': 'string'}},
            },
        }
    )
    open_item = _item(code, 'Open')
    assert 'deny_unknown_fields' not in open_item
    assert 'pub additional_properties: Map<String, Value>,' in open_item
    closed_item = _item(code, 'Closed')
    assert 'deny_unknown_fields' in closed_item
    assert 'additional_properties' not in closed_item


def test_a_required_nullable_field_must_be_present():
    code = _generate(
        {
            'Thing': {
                'type': 'object',
                'required': ['a'],
                'properties': {'a': {'type': 'string', 'nullable': True}},
            }
        }
    )
    assert (
        '    #[serde(deserialize_with = "fields::nullable")]\n'
        '    pub a: Option<String>,'
    ) in code


def _edge() -> dict[str, Any]:
    return json.loads((EDGE / 'contract.json').read_text())['components'][
        'schemas'
    ]


def _with(path: list[str], value: Any) -> dict[str, Any]:
    r"""The edge contract's schemas, with ``value`` set at ``path``."""
    schemas = copy.deepcopy(_edge())
    node = schemas
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return schemas


def _cat_ref() -> dict[str, str]:
    return {'$ref': '#/components/schemas/Cat'}


REFUSALS = {
    'an unmodelled keyword': (
        _with(['Count', 'not'], {'type': 'string'}),
        "Count: unsupported keywords ['not']",
    ),
    'an unmodelled keyword deep inside': (
        _with(['Closed', 'properties', 'inner', 'allOf'], []),
        "ClosedInner: unsupported keywords ['allOf']",
    ),
    'a discriminator outside a union': (
        _with(['Cat', 'discriminator'], {'propertyName': 'kind'}),
        'Cat: a discriminator without oneOf or anyOf',
    ),
    'items on a string': (
        _with(['Closed', 'properties', 'type', 'items'], {}),
        'ClosedType: items on a schema that is no array',
    ),
    'properties on an integer': (
        _with(['Count', 'properties'], {}),
        "Count: ['properties'] on a integer schema",
    ),
    'object keywords without a type': (
        _with(['Names'], {'required': ['a']}),
        "Names: ['required'] without type object",
    ),
    'properties without a type': (
        {'Thing': {'properties': {'a': {'type': 'string'}}}},
        "Thing: ['properties'] without type object",
    ),
    'additionalProperties that is no schema': (
        _with(['OpenEmpty', 'additionalProperties'], 'yes'),
        "OpenEmpty: additionalProperties is 'yes'",
    ),
    'a schema that is no object': (
        _with(['Closed', 'properties', 'plain'], True),
        'ClosedPlain: a schema is an object, not True',
    ),
    'required names that are no list': (
        _with(['Closed', 'required'], 'type'),
        "Closed: required is 'type', not a list of names",
    ),
    'an enum that is no list': (
        _with(['Mode', 'enum'], 'ab'),
        "Mode: enum is 'ab', not a list of values",
    ),
    'an empty discriminator': (
        _with(['Pet', 'discriminator'], {}),
        'Pet: discriminator is {}, not a propertyName',
    ),
    'a schema name that is no Rust type': (
        {'lower': {'type': 'string'}},
        "schema name 'lower' is not a Rust type",
    ),
    'an inline type named like a schema': (
        {
            'Thing': {
                'type': 'object',
                'properties': {'a': {'type': 'string', 'enum': ['x']}},
            },
            'ThingA': {'type': 'string'},
        },
        'two items would be named ThingA',
    ),
    'two properties with one Rust name': (
        _with(['Closed', 'properties', 'camel_case'], {'type': 'string'}),
        "Closed: 'camel_case' and 'camelCase' would both be camel_case",
    ),
    'two enum values with one Rust name': (
        _with(['Mode', 'enum'], ['a-b', 'a_b']),
        'Mode: two values would be AB',
    ),
    'an enum of integers': (
        _with(['Mode'], {'type': 'integer', 'enum': [1, 2]}),
        'Mode: only string enums are supported',
    ),
    'an unresolved reference': (
        _with(['Closed', 'properties', 'mode', '$ref'], '#/components/x'),
        'unresolved reference #/components/x',
    ),
    'an array without items': (
        _with(['Names'], {'type': 'array'}),
        'Names: an array without items',
    ),
    'an unknown type': (
        _with(['Names'], {'type': 'file'}),
        "Names: unknown type 'file'",
    ),
    'a nullable struct': (
        _with(['Cat', 'nullable'], True),
        'Cat: a nullable named type',
    ),
    'both oneOf and anyOf': (
        _with(['Either', 'oneOf'], [_cat_ref()]),
        'Either: both oneOf and anyOf',
    ),
    'a union with its own shape': (
        _with(['Either', 'type'], 'object'),
        'Either: a union with its own shape',
    ),
    'a union member that is no schema': (
        _with(['Either', 'anyOf'], [True]),
        'Either: a schema is an object, not True',
    ),
    'a tagged union member that is no schema': (
        _with(['Pet', 'oneOf'], [_cat_ref(), 'Dog']),
        "Pet: a schema is an object, not 'Dog'",
    ),
    'a union without members': (
        _with(['Either', 'anyOf'], []),
        'Either: a union is a list of schemas',
    ),
    'two union members with one name': (
        _with(['Either', 'anyOf'], [{'type': 'string'}, {'type': 'string'}]),
        'Either: two members would be String',
    ),
    'a tagged member that is no reference': (
        _with(['Pet', 'oneOf'], [_cat_ref(), {'type': 'object'}]),
        'Pet: a tagged member is not a $ref',
    ),
    'a tag mapped outside the union': (
        _with(
            ['Pet', 'discriminator', 'mapping', 'other'],
            '#/components/schemas/Count',
        ),
        'Pet: other maps outside oneOf',
    ),
}


@pytest.mark.parametrize('case', REFUSALS)
def test_what_it_cannot_express_stops_it(case):
    schemas, message = REFUSALS[case]
    with pytest.raises(gen.ContractError) as error:
        _generate(schemas)
    assert message in str(error.value)


def test_a_refusal_fails_the_command_and_writes_nothing(tmp_path):
    spec = tmp_path / 'contract.json'
    contract = json.loads((EDGE / 'contract.json').read_text())
    contract['components']['schemas']['Cat']['discriminator'] = {
        'propertyName': 'kind'
    }
    spec.write_text(json.dumps(contract))
    result = _run('--spec', spec, '--output', tmp_path / 'out' / 'generated.rs')
    assert result.returncode == 1
    assert 'a discriminator without oneOf or anyOf' in result.stderr
    assert not (tmp_path / 'out').exists()
