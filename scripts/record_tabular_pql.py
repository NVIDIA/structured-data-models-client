#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record public PQL expectations from an explicitly pinned frozen checkout.

Use that checkout on PYTHONPATH, in the same runtime as the candidate tests.
This writer never imports candidate product modules or computes expected orders.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'packages/structured-data-models-client/tests'
sys.path.insert(0, str(TESTS))

from tabular_pql_fixtures import (  # noqa: E402
    INDEX_FAMILIES,
    INDEX_SAMPLE_COUNTS,
    LABEL_FAMILIES,
    indexed_frame,
    label_frame,
)
from tabular_pql_reference import runtime_identity  # noqa: E402


class Endpoint:
    url = 'http://localhost'

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(self, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(body)
        return {
            'model': 'kumo-tabular',
            'id': 'frozen-pql-reference',
            'predictions': [{'row_index': 0, 'prediction': 0.0}],
            'metadata': {},
        }

    def predict_body(self, body: bytes) -> bytes:
        return json.dumps(self.predict(json.loads(body))).encode()

    def close(self) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-checkout', type=Path, required=True)
    parser.add_argument('--reference-revision', required=True)
    parser.add_argument(
        '--output-dir', type=Path, default=TESTS / 'data/tabular_pql_reference'
    )
    parser.add_argument(
        '--group',
        choices=('labels', 'counts', 'numeric', 'decimal'),
        default='labels',
    )
    args = parser.parse_args()
    reference = args.reference_checkout.resolve()
    assert reference != ROOT, (
        'candidate checkout cannot supply reference expectations'
    )
    revision = subprocess.check_output(
        ['git', '-C', str(reference), 'rev-parse', 'HEAD'], text=True
    ).strip()
    assert (
        len(args.reference_revision) == 40
        and revision == args.reference_revision
    )
    dirty = subprocess.check_output(
        [
            'git',
            '-C',
            str(reference),
            'status',
            '--porcelain',
            '--untracked-files=no',
        ],
        text=True,
    )
    assert not dirty.strip(), 'frozen checkout has tracked modifications'
    assert not os.environ.get('SDM_CORE_ROUTES'), (
        'reference execution must disable core routes'
    )
    import sdm_client
    from sdm_client import KumoTabular, PredictiveQuery, SDMClient

    assert Path(sdm_client.__file__).resolve().is_relative_to(reference), (
        sdm_client.__file__
    )
    if args.group in ('counts', 'numeric', 'decimal'):
        if args.group == 'counts':
            from tabular_pql_count_cases import CASES, observe_case
        elif args.group == 'numeric':
            from tabular_numeric_pure_cases import CASES, observe_case
        else:
            from tabular_decimal_cases import CASES, observe_case

        assert (
            args.output_dir.resolve()
            != (TESTS / 'data/tabular_pql_reference').resolve()
        ), 'non-label receipts require a separate --output-dir'
        runtime = runtime_identity()
        receipt = {
            'format': 1,
            'generated_by': 'scripts/record_tabular_pql.py',
            'group': args.group,
            'reference': {
                'revision': revision,
                'client_module': str(
                    Path(sdm_client.__file__).resolve().relative_to(reference)
                ),
            },
            'runtime': runtime,
            'cases': {name: observe_case(name) for name in CASES},
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / (runtime['fingerprint'] + '.json')
        if output.exists():
            previous = json.loads(output.read_text())
            assert previous['reference']['revision'] == revision
            assert previous['group'] == args.group
        output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)
            + '\n'
        )
        print(
            json.dumps(
                {
                    'receipt': str(output),
                    'journeys': len(CASES),
                    'runtime': runtime['identity'],
                },
                indent=2,
            )
        )
        return
    cases: dict[str, Any] = {}
    for group, families, builder in [
        ('indices', INDEX_FAMILIES, indexed_frame),
        ('labels', LABEL_FAMILIES, label_frame),
    ]:
        cases[group] = {}
        for family in families:
            cases[group][family] = {}
            for entrypoint in ('indices', 'predict'):
                frame = builder(family)
                arguments = (
                    {
                        'indices': [len(frame) - 1],
                        'num_context_samples': INDEX_SAMPLE_COUNTS.get(
                            family, 40
                        ),
                    }
                    if entrypoint == 'indices'
                    else {'predict': frame.iloc[[-1]].drop(columns=['target'])}
                )
                endpoint = Endpoint()
                try:
                    request = PredictiveQuery(
                        'PREDICT data.target FOR EACH data.id', {'data': frame}
                    ).materialize(**arguments, task='regression')
                except (TypeError, OverflowError, ValueError) as error:
                    cases[group][family][entrypoint] = {
                        'error': {
                            'type': type(error).__name__,
                            'module': type(error).__module__,
                            'message': str(error),
                        }
                    }
                    continue
                context_ids = request.context.instance_table['id'].tolist()
                client = SDMClient.__new__(SDMClient)
                client._start(endpoint)
                try:
                    result = client.predict(request, model=KumoTabular())
                    encoded_ids = [
                        row[0]
                        for row in endpoint.calls[0]['context'][
                            'instance_table'
                        ]['rows']
                    ]
                    result_frame = result.to_pandas()
                    assert encoded_ids == context_ids
                    assert len(endpoint.calls) == 1 and len(result_frame) == 1
                    cases[group][family][entrypoint] = {
                        'context_ids': context_ids,
                        'prediction': {
                            'encoded_context_ids': encoded_ids,
                            'columns': result_frame.columns.tolist(),
                            'rows': result_frame.to_dict(orient='records'),
                        },
                    }
                finally:
                    client.close()
    from sdm_client.errors import StructuredDataError

    frame = indexed_frame('range')
    try:
        PredictiveQuery(
            'PREDICT data.target FOR EACH data.id', {'data': frame}
        ).materialize(
            predict=frame.iloc[[80]].drop(columns=['target']),
            num_context_samples=40,
            task='regression',
        )
    except StructuredDataError as error:
        cases['validation'] = {
            'supplied_frame_cap': {
                'predict': {
                    'error': {
                        'type': type(error).__name__,
                        'module': type(error).__module__,
                        'message': str(error),
                        'code': error.code,
                    }
                }
            }
        }
    else:
        raise AssertionError(
            'frozen supplied-frame path unexpectedly accepted a context cap'
        )
    runtime = runtime_identity()
    receipt = {
        'format': 1,
        'generated_by': 'scripts/record_tabular_pql.py',
        'reference': {
            'revision': revision,
            'client_module': str(
                Path(sdm_client.__file__).resolve().relative_to(reference)
            ),
        },
        'runtime': runtime,
        'cases': cases,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = runtime['fingerprint'] + '.json'
    output = args.output_dir / filename
    if output.exists():
        assert (
            json.loads(output.read_text())['reference']['revision'] == revision
        ), 'do not overwrite a receipt from another frozen revision'
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n'
    )
    print(
        json.dumps(
            {
                'receipt': filename,
                'runtime': runtime['identity'],
                'journeys': sum(
                    len(entries)
                    for group in cases.values()
                    for entries in group.values()
                ),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
