# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 9.7 entrypoints over the independently reviewed acceptance assembly."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMANDS = {
    'generate': 'generate',
    'sweep': 'sweep',
    'match': 'verify-recordings',
    'frozen': 'frozen',
    'manifest': 'manifest',
    'names': 'names',
}


class Unavailable(Exception):
    pass


def verify_sources():
    manifest = json.loads((HERE / 'source-manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        path = HERE / relative
        if (
            path.is_symlink()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise Unavailable('acceptance.source_binding')


def bound_file(inputs, key, base):
    item = inputs.get(key)
    if not isinstance(item, dict) or set(item) != {'path', 'sha256'}:
        raise Unavailable('acceptance.pending_input.' + key)
    path = Path(item['path']).expanduser()
    path = path if path.is_absolute() else base / path
    if (
        not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']
    ):
        raise Unavailable('acceptance.input_binding.' + key)
    return path


def required_path(inputs, key, base):
    value = inputs.get(key)
    if not isinstance(value, str) or not value:
        raise Unavailable('acceptance.pending_input.' + key)
    path = Path(value).expanduser()
    path = path if path.is_absolute() else base / path
    if not path.is_file():
        raise Unavailable('acceptance.unavailable_input.' + key)
    return path


def inventory_policy(root, inventory, policy, same_json):
    import name_signatures

    required = name_signatures.required(root, policy)
    if any(
        not same_json(inventory.get(key), value)
        for key, value in required.items()
    ):
        raise Unavailable('acceptance.inventory_policy')
    scope = policy['required']['approved_difference_scopes']['DV87']
    path = root / scope['source']
    if (
        not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest()
        != scope['source_sha256']
    ):
        raise Unavailable('acceptance.dv87_source_binding')
    actual = inventory.get('generators')
    expected = policy['generator_policy']
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise Unavailable('acceptance.inventory_policy')
    for name, settings in expected.items():
        bound = actual[name]
        if (
            not isinstance(bound, dict)
            or set(bound) != set(settings) | {'source_sha256'}
            or any(
                not same_json(bound.get(key), value)
                for key, value in settings.items()
            )
        ):
            raise Unavailable('acceptance.inventory_policy')
        path = root / settings['source']
        if (
            not path.is_file()
            or bound['source_sha256']
            != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise Unavailable('acceptance.generator_source_binding')


def dispatch(command, root, harness, inputs_path):
    verify_sources()
    sys.dont_write_bytecode = True
    sys.path.insert(
        0, str(HERE / 'assembly/investigations/p97-acceptance-assembly')
    )
    import run as assembly
    from support import same_json, unique_json

    if (
        command in {'generate', 'sweep'}
        and not (root / 'scripts/parity/p97_generate.py').is_file()
    ):
        raise Unavailable('generator.absent')
    if (
        command == 'frozen'
        and not (
            root / 'rust/bindings/python/python/sdm_core/replay.py'
        ).is_file()
    ):
        raise Unavailable('replay.binding_target_absent')
    inputs = unique_json(inputs_path.read_bytes())
    inventory_path = bound_file(inputs, 'inventory', inputs_path.parent)
    inventory = unique_json(inventory_path.read_bytes())
    policy = unique_json((HERE / 'inventory-policy.json').read_bytes())
    inventory_policy(root, inventory, policy, same_json)
    if (
        inventory.get('base')
        != (harness / 'phases/phase-9.7/BASE').read_text().strip()
    ):
        raise Unavailable('acceptance.phase_base')
    head = (
        subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, stderr=subprocess.DEVNULL
        )
        .decode()
        .strip()
    )
    output = harness / 'phases/phase-9.7/evidence' / head[:12] / 'p97'
    output.mkdir(parents=True, exist_ok=True)
    arguments = [
        str(assembly.__file__),
        COMMANDS[command],
        '--root',
        str(root),
        '--inventory',
        str(inventory_path),
        '--out',
        str(output / (command + '.json')),
    ]
    if command != 'names':
        candidate = bound_file(inputs, 'candidate_binding', inputs_path.parent)
        arguments += ['--candidate-binding', str(candidate)]
        if command in {'generate', 'sweep'}:
            binding = unique_json(candidate.read_bytes())
            python = (
                Path(binding['lanes']['3']['runtime']).expanduser()
                / 'bin/python'
            )
            arguments += ['--python', str(python)]
    if command == 'sweep':
        generated = list(arguments)
        generated[1] = 'generate'
        generated[generated.index('--out') + 1] = str(output / 'recipes.json')
        if invoke(assembly, generated, harness) != 0:
            return 1
        arguments += ['--recipes', str(output / 'recipes.json')]
    if command in {'sweep', 'match', 'frozen'}:
        provider = bound_file(inputs, 'postgres_provider', inputs_path.parent)
        arguments += [
            '--postgres-provider',
            str(provider),
            '--postgres-provider-sha256',
            inputs['postgres_provider']['sha256'],
        ]
    if command in {'match', 'frozen', 'manifest'}:
        arguments += [
            '--manifest',
            str(root / 'tests/parity/frozen-public/manifest.json'),
        ]
    if command in {'frozen', 'manifest'}:
        review = bound_file(inputs, 'native_review', inputs_path.parent)
        binary = required_path(
            inputs,
            'instrumented_rust_binary'
            if command == 'manifest'
            else 'rust_binary',
            inputs_path.parent,
        )
        arguments += [
            '--native-review',
            str(review),
            '--rust-binary',
            str(binary),
        ]
    if command == 'frozen':
        python = required_path(inputs, 'oracle_free_python', inputs_path.parent)
        arguments += [
            '--python',
            str(python),
            '--case-index',
            str(root / 'tests/parity/frozen-public/case-index.json'),
        ]
    if command == 'manifest':
        arguments += ['--artifacts', str(output / 'scaffold')]
    return invoke(assembly, arguments, harness)


def invoke(assembly, arguments, harness):
    previous_arguments = sys.argv
    previous_harness = os.environ.get('P97_PHASE_HARNESS')
    try:
        sys.argv = arguments
        os.environ['P97_PHASE_HARNESS'] = str(harness)
        return assembly.main()
    finally:
        sys.argv = previous_arguments
        if previous_harness is None:
            os.environ.pop('P97_PHASE_HARNESS', None)
        else:
            os.environ['P97_PHASE_HARNESS'] = previous_harness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=COMMANDS)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--harness', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    arguments = parser.parse_args()
    try:
        return dispatch(
            arguments.command,
            arguments.root.resolve(),
            arguments.harness.resolve(),
            arguments.inputs.resolve(),
        )
    except Unavailable as error:
        print('REJECTED: ' + str(error))
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
    ):
        print('REJECTED: acceptance.invalid_or_unavailable_input')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
