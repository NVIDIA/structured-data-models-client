# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Original input and consumer execution in one explicitly owned old lane."""

import os
import sys
from pathlib import Path

from generator_inputs import construct, observe_samples, source_bindings, typed
from generator_native import NATIVE_STREAMS
from support import Rejected, json_bytes, require, same_json, sha, unique_json


def activate_provider(provider):
    import sdm_relational_engine

    provider_path = Path(provider['path']).expanduser().resolve()
    require(
        sha(provider_path.read_bytes()) == provider['sha256'],
        'generator.extension_provider',
    )
    sdm_relational_engine.__path__.append(str(provider_path.parent))
    from sdm_relational_engine import relationallib

    require(
        Path(relationallib.__file__).resolve() == provider_path,
        'generator.extension_origin',
    )
    return provider_path


def worker(root, request):
    import pandas as pd
    import sdm_core._native as native

    provider = request['extension_provider']
    provider_path = activate_provider(provider)
    from sdm_client import _impl

    require(_impl.OLD is True, 'generator.original_selector')
    result = {}
    failure = None
    require(
        same_json(source_bindings(root), request['sources']),
        'generator.worker_sources',
    )
    mode = request.get('mode', 'original')
    require(
        mode in {'original', 'native-input', 'native-consumer'},
        'generator.worker_mode',
    )
    for item in request['streams']:
        key = f'{item["name"]}/{item["seed"]}'
        require(key not in result, 'generator.duplicate_stream')
        try:
            if mode == 'native-consumer':
                require(
                    item['name'] in NATIVE_STREAMS,
                    'generator.worker_native_family',
                )
                supplied = request['native_inputs'][key]
                require(
                    set(supplied)
                    == {str(ordinal) for ordinal in item['ordinals']},
                    'generator.worker_native_coordinates',
                )
                require(
                    all(
                        same_json(typed(sample['raw_input']), sample['input'])
                        for sample in supplied.values()
                    ),
                    'generator.worker_native_complete_input',
                )
                result[key] = observe_samples(
                    root,
                    item['name'],
                    {
                        int(ordinal): sample['raw_input']
                        for ordinal, sample in supplied.items()
                    },
                    native_variant=request['native_variant'],
                )
            else:
                require(
                    mode != 'native-input' or item['name'] in NATIVE_STREAMS,
                    'generator.worker_native_family',
                )
                result[key] = construct(
                    root,
                    item['name'],
                    item['seed'],
                    item['ordinals'],
                    request['sources'],
                    array_count=item.get('array_count'),
                    observe_outcomes=mode == 'original',
                    native_variant=request.get('native_variant'),
                )
        except Exception as error:
            failure = {
                'stream': key,
                'exception_class': type(error).__name__,
                'category': str(error) if isinstance(error, Rejected) else None,
            }
            break
    require(
        sha(provider_path.read_bytes()) == provider['sha256'],
        'generator.extension_changed',
    )

    require(
        same_json(source_bindings(root), request['sources']),
        'generator.worker_sources_changed',
    )
    return {
        'process_id': os.getpid(),
        'extension_provider_sha256': provider['sha256'],
        'pandas_major': pd.__version__.split('.')[0],
        'pandas_version': pd.__version__,
        'python_version': sys.version.split()[0],
        'native_sha256': sha(Path(native.__file__).read_bytes()),
        'native_variant': request.get('native_variant'),
        'hashseed': os.environ.get('PYTHONHASHSEED'),
        'streams': result,
        'failure': failure,
    }


if __name__ == '__main__':
    request = unique_json(sys.stdin.buffer.read())
    sys.stdout.buffer.write(json_bytes(worker(Path(sys.argv[1]), request)))
