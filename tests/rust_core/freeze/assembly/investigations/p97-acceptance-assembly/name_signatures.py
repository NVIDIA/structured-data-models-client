# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve approved line identities without approving newly discovered names."""

from pathlib import Path

from support import checked_path, contract, require, sha, unique_json


def resolve(root, signatures):
    require(isinstance(signatures, list), 'names.signature_list')
    identities = set()
    exemptions = []
    for signature in signatures:
        require(
            isinstance(signature, dict)
            and set(signature)
            == {'path', 'line_sha256', 'count', 'reason', 'dv'},
            'names.signature_shape',
        )
        require(
            signature['dv'] == 'DV112'
            and isinstance(signature['reason'], str)
            and bool(signature['reason'].strip())
            and type(signature['count']) is int
            and signature['count'] > 0,
            'names.signature_approval',
        )
        identity = signature['path'], signature['line_sha256']
        require(identity not in identities, 'names.duplicate_signature')
        identities.add(identity)
        path = checked_path(root, signature['path'])
        require(path.is_file(), 'names.signature_source_missing')
        lines = path.read_text().splitlines()
        matches = [
            number
            for number, line in enumerate(lines, 1)
            if contract.digest(line) == signature['line_sha256']
            and any(name in line for name in contract.LEGACY)
        ]
        require(
            len(matches) == signature['count'], 'names.signature_occurrences'
        )
        exemptions.extend(
            {
                'path': signature['path'],
                'line': number,
                'line_sha256': signature['line_sha256'],
                'reason': signature['reason'],
                'dv': signature['dv'],
            }
            for number in matches
        )
    return sorted(exemptions, key=lambda entry: (entry['path'], entry['line']))


def required(root, policy):
    result = dict(policy['required'])
    result['name_exemptions'] = resolve(root, policy['name_signatures'])
    observed = {}
    for name, binding in policy['dialect_key_bindings'].items():
        raw = checked_path(root, name).read_bytes()
        require(sha(raw) == binding['sha256'], 'names.dialect_bytes')
        count = raw.decode().count('_Keys')
        require(count == binding['count'], 'names.dialect_count')
        observed[name] = count
    require(
        observed == result['dialect_keys_disposition'],
        'names.dialect_policy',
    )
    return result


def validate(root, sources, exemptions, accepted_dvs):
    policy = unique_json(
        (
            Path(__file__).resolve().parents[3] / 'inventory-policy.json'
        ).read_bytes()
    )
    require(
        exemptions == resolve(root, policy['name_signatures']),
        'names.signature_binding',
    )
    contract.validate_names(sources, exemptions, [*accepted_dvs, 'DV112'])
