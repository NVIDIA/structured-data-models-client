#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Submits and collects one-time Databricks validation runs of a notebook.

On demand and local only: it never runs in CI and gates nothing. It drives
the Databricks CLI (``databricks ... -o json``) and only ever:

- submits a one-time run (``jobs submit``, the runs/submit API), never a
  saved job: each run gets its own single-node job cluster (``new_cluster``,
  ``num_workers`` 0, under the given cluster policy) or a serverless
  environment, never an existing cluster;
- runs a notebook already in the given workspace folder, which it never
  imports, overwrites or deletes (import a new notebook yourself, without
  ``--overwrite``, so a name that exists fails rather than being replaced);
- reads a run's state and its notebook's exit value.

    dbx_runs.py spec --template FILE --out FILE --target TARGET
                     --notebook NAME [--sha SHA] [--set KEY=VALUE ...]
    dbx_runs.py submit --target TARGET --notebook NAME
                       [--param KEY=VALUE ...] [--timeout SECONDS]
    dbx_runs.py submit --json FILE
    dbx_runs.py wait RUN_ID [RUN_ID ...] [--poll SECONDS] [--max-wait SECONDS]
                     [--show-url]

``spec`` fills a job template's ``<KEY>`` placeholders (``<FOLDER>``,
``<NOTEBOOK>``, ``<VOLUME_DIR>``, ``<SHA>``, ``<SHA7>``, ``<TARGET>``,
``<SPARK_VERSION>``, ``<NODE_TYPE>``, ``<POLICY_ID>``, ``<USER>``,
``<UUID>`` and any ``--set``), sets the run's and each task's timeout from
the template's ``_timeout_seconds`` by ``--set LEG=...`` where it has one
(refusing a leg it does not time), merges the cluster extras into
``new_cluster``, checks the result as ``submit`` would, and writes it to a
new file, mode 0600, that it never prints; a file that exists is refused.
``submit`` prints the run ID; ``wait`` polls until every run ends and prints
each run's state, error and notebook exit value as JSON. That output is
private: a state message, error or trace can name workspace paths, hosts or
users, so read it before quoting any of it in public evidence. The run's
page URL (the workspace host) is left out unless ``--show-url`` asks for
it. A resubmission of the same payload reuses its ``idempotency_token``, so
a retried submit never starts a second run.

Every workspace value comes from an argument or the environment, with no
default, and none is written in this repository:

    --profile      SDM_DBX_PROFILE           the Databricks CLI profile
    --folder       SDM_DBX_WORKSPACE_FOLDER  workspace folder of the notebooks
    --policy-id    SDM_DBX_POLICY_ID         cluster policy of the job cluster
    --user         SDM_DBX_USER              single_user_name of the cluster
    --volume-dir   SDM_DBX_VOLUME_DIR        volume folder of the wheels (spec)
    --cluster-extra SDM_DBX_CLUSTER_EXTRA    optional: a JSON file of more
                                             new_cluster fields (cloud
                                             attributes, log delivery, tags)

TARGET is ``serverless-env<N>`` (serverless environment version N) or one of
the classic single-node targets in :data:`CLASSIC` (AWS node types).
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

#: Classic targets: (Spark version, AWS node type). The client's native
#: wheel is built for x86_64 only, so an ``-arm`` target needs an arm build.
CLASSIC = {
    'dbr14.3-x86': ('14.3.x-scala2.12', 'm5d.large'),
    'dbr14.3-arm': ('14.3.x-scala2.12', 'm6gd.large'),
    'dbr15.4-x86': ('15.4.x-scala2.12', 'm5d.large'),
    'dbr15.4-arm': ('15.4.x-scala2.12', 'm6gd.large'),
    'dbr16.4-x86': ('16.4.x-scala2.12', 'm5d.large'),
    'dbr16.4-arm': ('16.4.x-scala2.12', 'm6gd.large'),
    'dbr17.3-x86': ('17.3.x-scala2.13', 'm5d.large'),
    'dbr17.3-arm': ('17.3.x-scala2.13', 'm6gd.large'),
}
SETTINGS = {
    'profile': 'SDM_DBX_PROFILE',
    'folder': 'SDM_DBX_WORKSPACE_FOLDER',
    'policy_id': 'SDM_DBX_POLICY_ID',
    'user': 'SDM_DBX_USER',
    'volume_dir': 'SDM_DBX_VOLUME_DIR',
    'cluster_extra': 'SDM_DBX_CLUSTER_EXTRA',
}
PLACEHOLDER = re.compile(r'<([A-Z][A-Z0-9_]*)>')
ENDED = ('TERMINATED', 'INTERNAL_ERROR', 'SKIPPED')


class Refused(Exception):
    """A payload this tool will not submit, or a setting it lacks."""


def setting(arguments: argparse.Namespace, name: str) -> str:
    value = getattr(arguments, name, None) or os.environ.get(SETTINGS[name])
    if not value:
        flag = '--' + name.replace('_', '-')
        raise Refused(f'give {flag} or set {SETTINGS[name]}')
    return value


def cli(profile: str, *args: str) -> dict[str, Any]:
    for attempt in range(6):
        out = subprocess.run(
            ['databricks', '--profile', profile, *args, '-o', 'json'],
            capture_output=True,
            text=True,
        )
        if out.returncode == 0:
            return json.loads(out.stdout) if out.stdout.strip() else {}
        if attempt == 5:
            raise RuntimeError(
                f'databricks {args[0]} {args[1]}: {out.stderr[-500:]}'
            )
        time.sleep(15 * (attempt + 1))
    raise AssertionError


def new_cluster(
    target: str, policy_id: str, user: str, extra: dict[str, Any]
) -> dict[str, Any]:
    if target not in CLASSIC:
        raise Refused(f'unknown target {target!r}; one of {sorted(CLASSIC)}')
    spark_version, node_type = CLASSIC[target]
    cluster: dict[str, Any] = {
        'spark_version': spark_version,
        'node_type_id': node_type,
        'num_workers': 0,
        'policy_id': policy_id,
        'data_security_mode': 'SINGLE_USER',
        'single_user_name': user,
        'runtime_engine': 'STANDARD',
        'spark_conf': {
            'spark.databricks.cluster.profile': 'singleNode',
            'spark.master': 'local[*, 4]',
        },
        'custom_tags': {'ResourceClass': 'SingleNode'},
    }
    return merge_cluster(cluster, extra)


def merge_cluster(
    cluster: dict[str, Any], extra: dict[str, Any]
) -> dict[str, Any]:
    """The cluster with the extras merged one level deep, so extra tags or
    Spark settings join the single-node ones rather than replacing them."""
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(cluster.get(key), dict):
            cluster[key] = {**cluster[key], **value}
        else:
            cluster[key] = value
    return cluster


def cluster_extra(arguments: argparse.Namespace) -> dict[str, Any]:
    path = getattr(arguments, 'cluster_extra', None) or os.environ.get(
        SETTINGS['cluster_extra']
    )
    if not path:
        return {}
    extra = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(extra, dict):
        raise Refused('the cluster extras are not a JSON object')
    return extra


def check(payload: dict[str, Any], folder: str, policy_id: str) -> None:
    """Refuses anything but a one-time run of notebooks in ``folder`` on new
    single-node clusters under ``policy_id`` or on serverless. The checks
    run on the merged cluster, so the extras cannot undo them."""
    root = posixpath.normpath(folder.rstrip('/') or '/')
    if not payload.get('idempotency_token'):
        raise Refused('the payload has no idempotency_token')
    tasks = payload.get('tasks') or []
    if not tasks:
        raise Refused('the payload has no tasks')
    left = PLACEHOLDER.findall(json.dumps(payload))
    if left:
        raise Refused(f'unfilled placeholders: {sorted(set(left))}')
    environments = {
        e.get('environment_key') for e in payload.get('environments', [])
    }
    for task in tasks:
        name = task.get('task_key', '?')
        for forbidden in ('existing_cluster_id', 'job_cluster_key'):
            if forbidden in task:
                raise Refused(f'task {name}: {forbidden} (new clusters only)')
        notebook = (task.get('notebook_task') or {}).get('notebook_path', '')
        # A path with '..', '.' or '//' is refused outright, not resolved.
        if notebook != posixpath.normpath(notebook) or not notebook.startswith(
            root.rstrip('/') + '/'
        ):
            raise Refused(f'task {name}: notebook outside the workspace folder')
        cluster = task.get('new_cluster')
        if cluster is not None:
            if 'autoscale' in cluster:
                raise Refused(f'task {name}: autoscale (single node only)')
            if cluster.get('num_workers') != 0:
                raise Refused(f'task {name}: num_workers is not 0')
            conf = cluster.get('spark_conf') or {}
            if conf.get('spark.databricks.cluster.profile') != 'singleNode':
                raise Refused(f'task {name}: not the singleNode profile')
            if not str(conf.get('spark.master', '')).startswith('local'):
                raise Refused(f'task {name}: spark.master is not local')
            if cluster.get('policy_id') != policy_id:
                raise Refused(f'task {name}: not under the cluster policy')
        elif task.get('environment_key') not in environments:
            raise Refused(f'task {name}: neither new_cluster nor serverless')


def payload_for(
    arguments: argparse.Namespace, parameters: dict[str, str]
) -> dict[str, Any]:
    folder = setting(arguments, 'folder')
    task: dict[str, Any] = {
        'task_key': 'validate',
        'notebook_task': {
            'notebook_path': f'{folder.rstrip("/")}/{arguments.notebook}',
            'source': 'WORKSPACE',
            'base_parameters': parameters,
        },
        'timeout_seconds': arguments.timeout,
    }
    payload: dict[str, Any] = {
        'run_name': f'sdm-client-validation {arguments.notebook} {arguments.target}',
        'idempotency_token': str(uuid.uuid4()),
        'timeout_seconds': arguments.timeout,
        'tasks': [task],
    }
    if arguments.target.startswith('serverless-env'):
        version = arguments.target.removeprefix('serverless-env')
        task['environment_key'] = 'env'
        payload['environments'] = [
            {'environment_key': 'env', 'spec': {'environment_version': version}}
        ]
    else:
        task['new_cluster'] = new_cluster(
            arguments.target,
            setting(arguments, 'policy_id'),
            setting(arguments, 'user'),
            cluster_extra(arguments),
        )
    return payload


def fill(value: Any, values: dict[str, str]) -> Any:
    if isinstance(value, str):
        return PLACEHOLDER.sub(
            lambda m: values.get(m.group(1), m.group(0)), value
        )
    if isinstance(value, list):
        return [fill(v, values) for v in value]
    if isinstance(value, dict):
        return {k: fill(v, values) for k, v in value.items()}
    return value


def run_spec(arguments: argparse.Namespace) -> int:
    template = json.loads(arguments.template.read_text(encoding='utf-8'))
    template.pop('_comment', None)
    folder = setting(arguments, 'folder')
    policy_id = setting(arguments, 'policy_id')
    values = {
        'FOLDER': folder.rstrip('/'),
        'NOTEBOOK': arguments.notebook,
        'TARGET': arguments.target,
        'POLICY_ID': policy_id,
        'USER': setting(arguments, 'user'),
        'UUID': str(uuid.uuid4()),
    }
    if 'VOLUME_DIR' in PLACEHOLDER.findall(json.dumps(template)):
        values['VOLUME_DIR'] = setting(arguments, 'volume_dir').rstrip('/')
    if arguments.target in CLASSIC:
        values['SPARK_VERSION'], values['NODE_TYPE'] = CLASSIC[arguments.target]
    if arguments.sha:
        values['SHA'], values['SHA7'] = arguments.sha, arguments.sha[:7]
    for pair in arguments.set:
        key, _, value = pair.partition('=')
        values[key] = value
    timeouts = template.pop('_timeout_seconds', None)
    payload = fill(template, values)
    if timeouts is not None:
        # A template timing each leg (capture.job.template.json): the
        # run's and each task's timeout are the leg's.
        leg = values.get('LEG')
        if leg not in timeouts:
            raise Refused(f'the template names no timeout for the leg {leg!r}')
        payload['timeout_seconds'] = int(timeouts[leg])
        for task in payload.get('tasks', []):
            task['timeout_seconds'] = int(timeouts[leg])
    extra = cluster_extra(arguments)
    for task in payload.get('tasks', []):
        if 'new_cluster' in task:
            task['new_cluster'] = merge_cluster(
                task['new_cluster'], dict(extra)
            )
    check(payload, folder, policy_id)
    try:
        descriptor = os.open(
            arguments.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError:
        raise Refused(
            f'{arguments.out} exists; nothing is overwritten'
        ) from None
    with os.fdopen(descriptor, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')
    print(f'filled (not printed): {arguments.out}')
    return 0


def run_submit(arguments: argparse.Namespace) -> int:
    profile = setting(arguments, 'profile')
    folder = setting(arguments, 'folder')
    if arguments.json:
        payload = json.loads(arguments.json.read_text(encoding='utf-8'))
    else:
        if not (arguments.target and arguments.notebook):
            raise Refused('give --json FILE, or --target and --notebook')
        parameters = dict(p.split('=', 1) for p in arguments.param)
        payload = payload_for(arguments, parameters)
    uses_cluster = any('new_cluster' in t for t in payload.get('tasks', []))
    policy_id = setting(arguments, 'policy_id') if uses_cluster else ''
    check(payload, folder, policy_id)
    result = cli(
        profile, 'jobs', 'submit', '--no-wait', '--json', json.dumps(payload)
    )
    print(int(result['run_id']))
    return 0


def collect(profile: str, run_id: int, show_url: bool) -> dict[str, Any]:
    run = cli(profile, 'jobs', 'get-run', str(run_id))
    state = run.get('state', {})
    out: dict[str, Any] = {
        'run_id': run_id,
        'life_cycle': state.get('life_cycle_state'),
        'result': state.get('result_state'),
        'message': state.get('state_message', '')[:500],
    }
    if show_url:
        out['url'] = run.get('run_page_url')
    if state.get('life_cycle_state') in ENDED:
        task = run['tasks'][0]
        output = cli(profile, 'jobs', 'get-run-output', str(task['run_id']))
        notebook = output.get('notebook_output', {}).get('result')
        out['error'] = (output.get('error') or '')[:2000]
        out['error_trace'] = (output.get('error_trace') or '')[-3000:]
        try:
            out['output'] = json.loads(notebook) if notebook else None
        except json.JSONDecodeError:
            out['output'] = notebook
    return out


def run_wait(arguments: argparse.Namespace) -> int:
    profile = setting(arguments, 'profile')
    pending = set(arguments.run_ids)
    results: dict[int, dict[str, Any]] = {}
    deadline = time.monotonic() + arguments.max_wait
    while pending:
        for run_id in sorted(pending):
            status = collect(profile, run_id, arguments.show_url)
            if 'output' in status:
                results[run_id] = status
                pending.discard(run_id)
        if pending:
            if time.monotonic() > deadline:
                for run_id in pending:
                    results[run_id] = {
                        'run_id': run_id,
                        'life_cycle': 'still running',
                    }
                break
            time.sleep(arguments.poll)
    json.dump(results, sys.stdout, indent=1, default=str)
    print()
    passed = all(r.get('result') == 'SUCCESS' for r in results.values())
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('\n', 2)[2],
    )
    common = argparse.ArgumentParser(add_help=False)
    for name in SETTINGS:
        common.add_argument('--' + name.replace('_', '-'), dest=name)
    sub = parser.add_subparsers(dest='command', required=True)

    spec = sub.add_parser('spec', parents=[common], help='fill a job template')
    spec.add_argument('--template', type=Path, required=True)
    spec.add_argument('--out', type=Path, required=True)
    spec.add_argument('--target', required=True)
    spec.add_argument('--notebook', required=True)
    spec.add_argument('--sha')
    spec.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')

    submit = sub.add_parser('submit', parents=[common], help='submit one run')
    submit.add_argument('--json', type=Path, help='a filled payload (spec)')
    submit.add_argument('--target')
    submit.add_argument('--notebook')
    submit.add_argument(
        '--param', action='append', default=[], metavar='KEY=VALUE'
    )
    submit.add_argument('--timeout', type=int, default=3600)

    wait = sub.add_parser('wait', parents=[common], help='collect runs')
    wait.add_argument('run_ids', nargs='+', type=int)
    wait.add_argument('--poll', type=int, default=30)
    wait.add_argument('--max-wait', type=int, default=4 * 3600)
    wait.add_argument(
        '--show-url',
        action='store_true',
        help="also print each run's page URL (names the workspace host)",
    )

    arguments = parser.parse_args(argv)
    handlers = {'spec': run_spec, 'submit': run_submit, 'wait': run_wait}
    try:
        return handlers[arguments.command](arguments)
    except Refused as error:
        print(f'refused: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
