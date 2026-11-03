#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record native binding vectors shared by Python and Java tests.

Run with a test-ops Python installation:
    python -I contract/tools/t311_java_record.py binding

Only the native binding section is updated.
Existing historical reference metadata is preserved, never regenerated.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/t311-java-vectors.json'
FORMAT = 'sdm-t311-java-vectors/1'
S0 = ROOT / 'contract/cases/s0'
#: The host facts every recorded start carries, as the binding's tests
#: write them; the Java test sends the same.
FACTS: dict[str, Any] = {
    'now': {'t': 'ts', 'unit': 'us', 'zone': 'UTC', 'v': '1767225600000000'},
    'local_zone': None,
    'zone_roots': [],
    'platform': {'os': 'linux', 'strftime_pads_year': False},
}
#: Texts no S0 case holds, each on the binary-classification case's schema:
#: a NUL inside a string literal (JNI's modified UTF-8 writes it as two
#: bytes), a character outside the BMP (a surrogate pair in Java, four
#: bytes in UTF-8, six in modified UTF-8), non-ASCII letters and a single
#: deprecated escape whose warning must cross both bindings.
TEXTS = {
    'text.nul': "PREDICT users.country = 'a\x00b' FOR EACH users.user_id",
    'text.escape-warning': r"PREDICT users.country = 'a\/' FOR EACH users.user_id",
    'text.non-bmp': "PREDICT users.country = '\U0001f600' FOR EACH users.user_id",
    'text.non-ascii': (
        "PREDICT users.country = 'Zürich 東京' FOR EACH users.user_id"
    ),
}
SCHEMA_CASE = 's0.accept.binary-classification'
#: Blank texts the reference's constructor refuses.
BLANKS = {'blank.empty': '', 'blank.spaces': '   ', 'blank.whitespace': '\t\n '}
#: The contract's error kinds, in the order tried.
KINDS = (
    'invalid_request',
    'unsupported',
    'not_found',
    'model_endpoint',
    'internal',
    'data_source',
    'cancelled',
    'protocol',
)


def _cases() -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding='utf-8'))
        for path in sorted(S0.glob('*.json'))
    ]


def _texts() -> dict[str, str]:
    r"""Every query text the vectors cover, by vector name: each S0 case's
    by its id, then :data:`TEXTS`."""
    found = {case['id']: case['input']['pql'] for case in _cases()}
    return {**found, **TEXTS}


def _load() -> dict[str, Any]:
    if OUT.is_file():
        found: dict[str, Any] = json.loads(OUT.read_text(encoding='utf-8'))
        return found
    return {
        'format': FORMAT,
        'about': (
            "T3.11's vectors: the reference's PredictiveQuery constructor on "
            "each S0 query text and on blank texts (reference), and Python's "
            "binding's answers to each S0 case's validate_query and to the "
            'test-only fail operation for every contract error code '
            '(binding). java/src/test/java/com/nvidia/sdm/ holds the Java '
            'host to them.'
        ),
        'recorded_by': 'contract/tools/t311_java_record.py',
    }


def _write(document: dict[str, Any]) -> None:
    OUT.write_text(
        json.dumps(document, indent=1, ensure_ascii=True, sort_keys=True)
        + '\n',
        encoding='utf-8',
    )


# --- binding -----------------------------------------------------------------


def _start(native: Any, operation: dict[str, Any]) -> bytes:
    return json.dumps(
        {
            'message': 'start',
            'contract': native.CONTRACT_VERSION,
            'facts': FACTS,
            'operation': operation,
        }
    ).encode()


def _attachment(value: Any) -> bytes:
    r"""An attachment as Python's host writes it (``codec.attachment``)."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate(native: Any) -> dict[str, Any]:
    cases = {case['id']: case for case in _cases()}
    answers: dict[str, Any] = {}
    for name, text in _texts().items():
        recorded = cases.get(name, cases[SCHEMA_CASE])['core']['validate_query']
        operation = {
            'op': 'validate_query',
            'query': 1,
            'schema': 2,
            'profile': recorded['profile'],
        }
        attachments = {
            1: _attachment(text),
            2: _attachment(recorded['schema']),
        }
        entry: dict[str, Any] = {
            'operation': json.dumps(operation, sort_keys=True),
            'attachments': {
                str(key): base64.b64encode(value).decode('ascii')
                for key, value in attachments.items()
            },
            'text': text,
        }
        entry['warnings'] = []
        run = None
        try:
            run = native.Run.start(_start(native, operation), attachments)
            control, carried = run.next()
        except native.CoreError as error:
            entry.update(
                error=error.code,
                digest=_digest(error.control),
                accepted=None,
                codes=None,
            )
        else:
            step = json.loads(control)
            if step['message'] == 'attached':
                # A done step past the control cap travels as its bytes
                # attachment (decision H7); the digest stays the control's.
                step = json.loads(carried[step['attachment']])
            if (step.get('message'), step.get('step')) != ('step', 'done'):
                raise ValueError('validate_query did not return a done step')
            result = step['result']
            if result.get('op') != 'validate_query':
                raise ValueError(
                    'validate_query did not return its done result'
                )
            entry.update(
                error=None,
                digest=_digest(control),
                accepted=result['accepted'],
                codes=[
                    diagnostic['code']
                    for diagnostic in result['diagnostics']
                    if diagnostic['severity'] == 'error'
                ],
            )
        finally:
            # Warnings are a separate binding channel, also available when
            # next() raises. Keep their full ordered control beside the
            # terminal result/error, rather than treating them as diagnostics.
            taken = None if run is None else run.take_warnings()
            if taken is not None:
                warning_control, carried = taken
                warning = json.loads(warning_control)
                if warning['message'] == 'attached':
                    warning = json.loads(carried[warning['attachment']])
                if warning.get('message') != 'warnings':
                    raise ValueError(
                        'validate_query returned a non-warning control'
                    )
                entry['warnings'].append(
                    {'digest': _digest(warning_control), 'control': warning}
                )
        answers[name] = entry
    return answers


def _sentences(native: Any) -> dict[str, dict[str, Any]]:
    r"""Each error code's sentence the vectors use: its first in the core's
    table that names no field, else its first."""
    chosen: dict[str, dict[str, Any]] = {}
    for entry in json.loads(native.templates())['sentences']:
        if entry['kind'] != 'error':
            continue
        held = chosen.get(entry['code'])
        if held is None or (
            '{' in held['template'] and '{' not in entry['template']
        ):
            chosen[entry['code']] = entry
    return chosen


def _errors(native: Any) -> dict[str, Any]:
    operation = {'op': 'test', 'test': 'fail', 'error': 1}
    sentences = _sentences(native)
    answers: dict[str, Any] = {}
    for code in native.ERROR_CODES:
        sentence = sentences[code]
        for kind in KINDS:
            error = {
                'kind': kind,
                'code': code,
                'sentence': sentence['name'],
                'message': 't311',
                'retryable': False,
            }
            try:
                # A kind the code does not allow is refused at the start.
                run = native.Run.start(
                    _start(native, operation), {1: _attachment(error)}
                )
            except native.CoreError:
                continue
            try:
                run.next()
            except native.CoreError as raised:
                if raised.code != code:
                    continue
                answers[code] = {
                    'error': json.dumps(error, sort_keys=True),
                    'kind': raised.kind,
                    'code': raised.code,
                    'retryable': raised.retryable,
                    'digest': _digest(raised.control),
                    'template': sentence['template'],
                }
                break
            raise SystemExit(f'the fail operation with {code} did not fail')
        else:
            raise SystemExit(f'no kind the core takes with {code}')
    return answers


def record_binding() -> dict[str, Any]:
    from sdm_client import _native as native

    if not native.TEST_OPS:
        raise SystemExit(
            'the binding is not the test-ops build: run with $SDM_KIT_PYTHON'
        )
    return {
        'facts': FACTS,
        'contract': native.CONTRACT_VERSION,
        'validate': _validate(native),
        'errors': _errors(native),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sections = parser.add_subparsers(dest='section', required=True)
    sections.add_parser('binding')
    parser.parse_args(argv)
    document = _load()
    document['binding'] = record_binding()
    _write(document)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
