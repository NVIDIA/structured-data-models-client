# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture semantic input witnesses in an isolated genuine core-selector worker."""

import os
import sys
import tempfile
from pathlib import Path

from generator_inputs import source_bindings, typed
from generator_primitives import schema
from generator_semantic import FAMILY, assets, validate
from generator_semantic_capture import collect
from support import json_bytes, require, same_json, sha, unique_json


def worker(root, request):
    import pandas as pd
    import sdm_core._native as native
    import sdm_relational_engine
    from sdm_client import _impl

    provider = request['extension_provider']
    provider_path = Path(provider['path']).resolve()
    require(
        sha(provider_path.read_bytes()) == provider['sha256'],
        'semantic.extension_provider',
    )
    sdm_relational_engine.__path__.append(str(provider_path.parent))
    from sdm_relational_engine import relationallib

    require(
        Path(relationallib.__file__).resolve() == provider_path,
        'semantic.extension_origin',
    )

    require(_impl.OLD is False, 'semantic.core_selector')
    require(
        same_json(source_bindings(root), request['sources']),
        'semantic.worker_sources',
    )
    stream = request['stream']
    require(stream['name'] == FAMILY, 'semantic.worker_family')
    generated = schema(root, FAMILY, stream['seed'], stream['ordinals'])
    require(
        set(generated) == set(stream['ordinals']), 'semantic.worker_coordinates'
    )
    samples = {}
    for ordinal, value in generated.items():
        original = typed(value)
        with tempfile.TemporaryDirectory(
            prefix='p97-semantic-inputs-'
        ) as temporary:
            captured = collect(root, value, Path(temporary))
        require(not Path(temporary).exists(), 'semantic.asset_cleanup')
        require(same_json(original, typed(value)), 'semantic.input_changed')
        validate(captured['host_inputs'], captured['host_inputs'], original)
        require(
            set(captured['assets'])
            == assets(captured['host_inputs']['external']),
            'semantic.worker_assets',
        )
        samples[str(ordinal)] = {'input': original, **captured}
    require(
        same_json(source_bindings(root), request['sources']),
        'semantic.worker_sources_changed',
    )
    require(
        sha(provider_path.read_bytes()) == provider['sha256'],
        'semantic.extension_changed',
    )
    return {
        'process_id': os.getpid(),
        'pandas_version': pd.__version__,
        'python_version': sys.version.split()[0],
        'native_sha256': sha(Path(native.__file__).read_bytes()),
        'extension_provider_sha256': provider['sha256'],
        'samples': samples,
    }


if __name__ == '__main__':
    sys.stdout.buffer.write(
        json_bytes(
            worker(Path(sys.argv[1]), unique_json(sys.stdin.buffer.read()))
        )
    )
