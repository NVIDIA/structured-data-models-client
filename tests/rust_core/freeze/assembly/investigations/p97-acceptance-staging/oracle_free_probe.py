# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run inside the future isolated binding environment before frozen replay."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json


def probe() -> dict[str, list[str]]:
    modules = ['predictive_query', 'relationallib', 'sdm_relational_engine']
    distributions = [
        'sdm-pql',
        'sdm-relational-engine',
        'structured-data-models-pql',
        'structured-data-models-relational-engine',
    ]
    for module in modules:
        if importlib.util.find_spec(module) is not None:
            raise RuntimeError('oracle module remains importable')
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as error:
            if error.name == module:
                continue
        raise RuntimeError('oracle import did not fail at its module boundary')
    for distribution in distributions:
        try:
            importlib.metadata.distribution(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        raise RuntimeError('oracle distribution remains installed')
    return {'absent_modules': modules, 'absent_distributions': distributions}


if __name__ == '__main__':
    print(json.dumps(probe(), sort_keys=True))
