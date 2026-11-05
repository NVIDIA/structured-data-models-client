# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run every registered live API requirement with private diagnostics."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from api_contracts import ROOT, groups, validate_live_targets

from contract import manifest

CONFIG_VARIABLES = (
    'SDM_LABELS_WAREHOUSE_CONFIG',
    'SDM_END_TIME_CONFIG',
    'SDM_SQL_COLUMN_CONFIG',
    'SDM_EXAMPLES_CONFIG',
)
REQUIRED_VARIABLES = (
    *CONFIG_VARIABLES,
    'SDM_RELATIONAL_URL',
    'SDM_TABULAR_LIVE_URL',
    'SDM_TEST_S3_READER_URI',
    'SDM_GRADIENT_URL',
    'SDM_RUN_LOCAL_MODEL',
)


def environment(config_path, config, inherited):
    """Use existing test contracts; explicit private overrides take precedence."""
    env = dict(inherited)
    for name in CONFIG_VARIABLES:
        env.setdefault(name, str(config_path.resolve()))
    for model, variable in [
        ('relational', 'SDM_RELATIONAL_URL'),
        ('tabular', 'SDM_TABULAR_LIVE_URL'),
        ('link', 'SDM_LINK_URL'),
    ]:
        url = (
            config.get('endpoints', {})
            .get(model, {})
            .get('client', {})
            .get('url')
        )
        if url:
            env.setdefault(variable, url)
    overrides = config.get('live_api', {}).get('environment', {})
    if set(overrides) - {*REQUIRED_VARIABLES, 'SDM_LINK_URL'}:
        raise ValueError('unknown live API environment key')
    if any(not isinstance(value, str) for value in overrides.values()):
        raise ValueError('live API environment values must be strings')
    env.update(overrides)
    link_url = (
        config.get('endpoints', {}).get('link', {}).get('client', {}).get('url')
    )
    if env.get('SDM_LINK_URL') and env['SDM_LINK_URL'] != link_url:
        raise ValueError('link URL must match the recorded configured endpoint')
    return env


def preflight(env):
    missing = [name for name in REQUIRED_VARIABLES if not env.get(name)]
    missing.extend(
        name
        for name in CONFIG_VARIABLES
        if env.get(name) and not Path(env[name]).is_file()
    )
    if env.get('SDM_RUN_LOCAL_MODEL') != '1':
        missing.append('SDM_RUN_LOCAL_MODEL=1')
    return sorted(set(missing))


def artifact_identities(
    commit, provenance, java_provenance, java_jar, java_native
):
    """Reuse the installed-wheel proof and verify the exact Java inputs."""
    from python_runner import artifact

    python = artifact(commit, 'candidate', provenance)
    java = json.loads(java_provenance.read_text())
    if (
        java.get('commit') != commit
        or java.get('contract') != python['contract']
        or java.get('package_hash')
        != hashlib.sha256(java_jar.read_bytes()).hexdigest()
        or java.get('native_hash')
        != hashlib.sha256(java_native.read_bytes()).hexdigest()
    ):
        raise ValueError('Java artifact identity mismatch')
    # Maven compiles the API test sources and production classes from this checkout.
    head = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ['git', 'status', '--porcelain'], cwd=ROOT, text=True
    ).strip()
    if head != commit or dirty:
        raise ValueError(
            'Java tests require the requested clean source checkout'
        )
    return {
        'python': python,
        'pretrained_runtime': {
            'distribution': 'structured-data-models',
            'version': importlib.metadata.version('structured-data-models'),
        },
        'java': {
            'execution': 'source-equivalent',
            'source_commit': head,
            'tree_clean': True,
            'native_hash': java['native_hash'],
            'contract': java['contract'],
            'verified_reference_jar_hash': java['package_hash'],
        },
    }


def model_identities(config):
    """Record caller-pinned versions, never endpoint addresses or credentials."""
    models = {}
    endpoints = config.get('endpoints', {})
    for model in dict.fromkeys(('tabular', 'relational', *endpoints)):
        endpoint = endpoints.get(model, {})
        version = endpoint.get('version')
        transport = endpoint.get('transport')
        if (
            not isinstance(version, str)
            or not version
            or transport not in ('nim', 'databricks', 'private-bridge')
        ):
            raise ValueError('live endpoint version/transport is missing')
        models[model] = {'version': version, 'transport': transport}
        if transport == 'private-bridge':
            route = (
                config.get('live_api', {})
                .get('bridge_routes', {})
                .get(model, {})
            )
            url = endpoint.get('client', {}).get('url')
            entity = endpoint.get('entity')
            if (
                not isinstance(url, str)
                or not url
                or not isinstance(entity, str)
                or not entity
                or route.get('url') != url
                or route.get('entity') != entity
                or route.get('version') != version
                or route.get('transport') != 'databricks'
                or not isinstance(route.get('endpoint'), str)
                or not route['endpoint']
            ):
                raise ValueError(
                    'private bridge route identity is missing or inconsistent'
                )
            declared = {
                key: route[key]
                for key in ('url', 'entity', 'version', 'transport', 'endpoint')
            }
            models[model].update(
                identity_basis='configured-route',
                route_sha256=hashlib.sha256(
                    json.dumps(
                        declared, sort_keys=True, separators=(',', ':')
                    ).encode()
                ).hexdigest(),
            )
    case_routes = {}
    for case, overrides in config.get('endpoints_by_case', {}).items():
        case_routes[case] = {}
        for model, override in overrides.items():
            matches = [
                name
                for name, endpoint in endpoints.items()
                if endpoint == override
            ]
            if len(matches) != 1:
                raise ValueError(
                    'case endpoint must identify one recorded primary route'
                )
            case_routes[case][model] = matches[0]
    if case_routes:
        models['case_routes'] = case_routes
    for model in ('pretrained', 'gradients'):
        version = (
            config.get('live_api', {}).get('model_versions', {}).get(model)
        )
        if not isinstance(version, str) or not version:
            raise ValueError('pretrained/gradient model versions are missing')
        models[model] = {
            'version': version,
            'transport': 'in-process' if model == 'pretrained' else 'http',
        }
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--private-dir', required=True, type=Path)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--provenance', type=Path)
    parser.add_argument('--java-provenance', type=Path)
    parser.add_argument('--java-jar', type=Path)
    parser.add_argument('--java-native', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    data = manifest()
    validate_live_targets(data, {item['id'] for item in data['api_contracts']})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.private_dir.chmod(0o700)
    config = json.loads(args.config.read_text())
    env = environment(args.config, config, os.environ)
    missing = preflight(env)
    report = {
        'candidate_commit': args.commit,
        'status': 'incomplete',
        'requirements': data['live_api_tests'],
        'missing': missing,
        'results': {},
    }
    try:
        report['models'] = model_identities(config)
    except ValueError:
        missing.append('immutable model version/transport metadata')
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    if missing:
        print('Live API requirements unavailable: ' + ', '.join(missing))
        return 1
    if args.preflight:
        print('Live API configuration present; execution is still required.')
        return 0
    if any(
        value is None
        for value in (
            args.provenance,
            args.java_provenance,
            args.java_jar,
            args.java_native,
        )
    ):
        parser.error(
            'execution requires Python and Java artifact provenance, JAR and JNI paths'
        )
    try:
        report['artifacts'] = artifact_identities(
            args.commit,
            args.provenance,
            args.java_provenance,
            args.java_jar,
            args.java_native,
        )
    except Exception:
        # Diagnostic details may include private paths; keep the public failure generic.
        print('Live API artifact or model identity verification failed.')
        return 1
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    for language in ('python', 'java'):
        output = args.private_dir / (language + '-results.json')
        output.unlink(missing_ok=True)
        script = (
            'api_contracts.py' if language == 'python' else 'check_junit.py'
        )
        command = [
            sys.executable,
            str(ROOT / 'tests/compatibility' / script),
            '--gate',
            'live',
            '--run',
            '--out',
            str(output),
        ]
        if language == 'java':
            command.append(
                '--maven-arg=-Dsdm.native.library='
                + str(args.java_native.resolve())
            )
            command.append(
                '--maven-arg=-Dsdm.snowflake.config='
                + str(args.config.resolve())
            )
        log = args.private_dir / (language + '.log')
        with log.open('w') as stream:
            status = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        evidence = json.loads(output.read_text()) if output.is_file() else {}
        report['results'][language] = {
            'status': 'passed'
            if status == 0 and evidence.get('status') == 'passed'
            else 'incomplete',
            'exit_code': status,
            'required_methods': sorted(
                {
                    node
                    for group in groups(data, 'live')
                    for node in group['tests'].get(language, [])
                }
            ),
            'completed_tests': evidence.get('completed_tests', []),
        }
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    complete = all(
        item['status'] == 'passed' for item in report['results'].values()
    )
    report['status'] = 'passed' if complete else 'incomplete'
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print('Live API acceptance: ' + report['status'])
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
