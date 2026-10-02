#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the recorded request bodies against the contract they are sent to.

The rest of the contract gate works on digests and on the response models: it
proves the binding was generated from the pinned spec, and that the spec has
not moved under it. Nothing read a request *body*. The client builds those as
plain mappings rather than through the generated models, so a key renamed, a
section dropped, or a value off an enum would reach the wire and only be
refused by a deployment.

This closes that over the payloads in ``tests/golden/wire_payloads.json``,
which are recorded from real calls and already pinned against drift.

It reads what the contract declares and no more. Six of the seventeen schemas
a request reaches set ``additionalProperties: true``, so a key invented inside
``output``, ``inference``, ``task``, ``target``, ``metadata`` or a column is
contractual by construction. Rules the spec states in prose, such as which
columns a prediction may name, are not expressible as a schema and are not
read either.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PAYLOADS = REPO_ROOT / 'tests' / 'golden' / 'wire_payloads.json'

#: Keywords OpenAPI 3.0 adds to JSON Schema that carry no constraint.
ANNOTATIONS = frozenset(
    {'discriminator', 'example', 'externalDocs', 'xml', 'deprecated'}
)

#: The three request schemas, told apart by the sections only one of them has.
#: A session id is in the path rather than the body, so it cannot be used.
BY_SHAPE = {
    ('context', 'predict'): 'PredictionRequest',
    ('context',): 'SessionCreateRequest',
    ('predict',): 'SessionPredictionRequest',
}


def as_json_schema(node: Any) -> Any:
    r"""One OpenAPI 3.0 schema as the JSON Schema it means.

    ``$ref`` is rebased onto ``$defs``, annotations are dropped, and
    ``nullable`` becomes the null type it stands for, which is the one
    difference that would otherwise reject a body the contract allows.
    """
    if isinstance(node, list):
        return [as_json_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    converted: dict[str, Any] = {}
    for key, value in node.items():
        if key == '$ref':
            converted['$ref'] = value.replace(
                '#/components/schemas/', '#/$defs/'
            )
        elif key in ANNOTATIONS or key == 'nullable':
            continue
        else:
            converted[key] = as_json_schema(value)

    if node.get('nullable') and 'type' in converted:
        declared = converted['type']
        types = declared if isinstance(declared, list) else [declared]
        if 'null' not in types:
            converted['type'] = [*types, 'null']
    return converted


class Unreadable(Exception):
    r"""The recording could not be read, which is not a contract failure."""


def schema_for(body: dict[str, Any]) -> str | None:
    r"""Which request this body is, or ``None`` if it is none of them.

    Guessing would read a body against a schema it was never sent as, and
    report every field that schema wants as missing.
    """
    sections = tuple(
        section for section in ('context', 'predict') if section in body
    )
    return BY_SHAPE.get(sections)


def recorded_bodies(payloads: Path) -> list[tuple[str, dict[str, Any]]]:
    r"""The bodies in a recording, as ``name`` and body.

    Raises ``Unreadable`` if the file cannot be read or is not a recording,
    so a broken file reads as the mistake it is rather than as a contract
    failure or a traceback.
    """
    try:
        text = payloads.read_text(encoding='utf-8')
    except OSError as error:
        raise Unreadable(f'{payloads} cannot be read: {error}') from error
    except UnicodeDecodeError as error:
        raise Unreadable(f'{payloads} is not UTF-8: {error}') from error

    try:
        recorded = json.loads(text)
    except json.JSONDecodeError as error:
        raise Unreadable(f'{payloads} is not JSON: {error}') from error
    except (ValueError, RecursionError) as error:
        # JSON a parser will not finish: an integer past the digit limit, or
        # nesting past the recursion one. Still the file's fault, not the
        # contract's.
        raise Unreadable(f'{payloads} cannot be parsed: {error}') from error

    if not isinstance(recorded, dict):
        raise Unreadable(
            f'{payloads} holds {type(recorded).__name__}, not calls by name'
        )

    bodies = []
    for name, entry in sorted(recorded.items()):
        calls = entry if isinstance(entry, list) else [entry]
        for position, body in enumerate(calls):
            if not isinstance(body, dict):
                raise Unreadable(
                    f'{name}[{position}] is {type(body).__name__}, not a '
                    f'request body'
                )
            bodies.append((f'{name}[{position}]', body))
    return bodies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--contract',
        required=True,
        help='Path to nim-sd.openapi.yaml, checked out at the pinned revision.',
    )
    parser.add_argument(
        '--payloads',
        default=PAYLOADS,
        type=Path,
        help='The recorded bodies to read. Defaults to the golden file.',
    )
    args = parser.parse_args()

    try:
        import yaml
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as error:  # pragma: no cover - a CI setup mistake
        print(f'needs pyyaml and jsonschema: {error}', file=sys.stderr)
        return 1

    contract = Path(args.contract)
    if not contract.is_file():
        print(f'no contract at {contract}', file=sys.stderr)
        return 1

    spec = yaml.safe_load(contract.read_text(encoding='utf-8'))
    defs = {
        name: as_json_schema(schema)
        for name, schema in spec['components']['schemas'].items()
    }

    try:
        bodies = recorded_bodies(args.payloads)
    except Unreadable as error:
        print(error, file=sys.stderr)
        return 1
    if not bodies:
        print(
            f'{args.payloads} records no request bodies, so nothing was '
            f'read. A recording that lost its contents would otherwise '
            f'report success.',
            file=sys.stderr,
        )
        return 1

    failed = 0
    for name, body in bodies:
        wanted = schema_for(body)
        if wanted is None:
            failed += 1
            print(
                f'{name} is none of {", ".join(sorted(BY_SHAPE.values()))}: '
                f'it carries neither a context nor rows to predict.',
                file=sys.stderr,
            )
            continue
        # The formats this contract uses are OpenAPI's own: int32, int64,
        # float and byte. None is a JSON Schema assertion, so the checker
        # reads nothing today and is here for formats the contract may add.
        validator = Draft202012Validator(
            {**defs[wanted], '$defs': defs}, format_checker=FormatChecker()
        )
        errors = sorted(validator.iter_errors(body), key=lambda e: list(e.path))
        if not errors:
            continue
        failed += 1
        print(f'{name} is not a {wanted}:', file=sys.stderr)
        for error in errors[:5]:
            where = '.'.join(str(part) for part in error.path) or '(body)'
            print(f'    {where}: {error.message}', file=sys.stderr)

    if failed:
        print(
            f'\n{failed} of {len(bodies)} recorded bodies do not match the '
            f'contract at this revision. Either the client builds something '
            f'the contract does not declare, or the contract moved and the '
            f'client has not followed.',
            file=sys.stderr,
        )
        return 1

    counted = (
        '1 recorded request body matches'
        if len(bodies) == 1
        else f'{len(bodies)} recorded request bodies match'
    )
    print(f'{counted} the contract')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
