# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from setuptools import setup

# ``kumo-api`` and ``kumo-pql`` are internalized under ``sdm_relational_engine.api`` and
# ``predictive_query`` (see their SOURCE.md files), so the client carries no external
# ``kumo-api`` dependency. ``pydantic`` used to arrive transitively via
# ``kumo-api``; it is now declared directly.

if bool(int(os.getenv('WITH_RELATIONALLIB', '1'))):
    kwargs = dict(cmake_source_dir='.')
else:
    kwargs = dict()

setup(
    install_requires=[
        'numpy',
        'pandas',
        'pyarrow>=8.0.0',
        'python-dateutil',
        'requests>=2.28.2',
        'urllib3',
        'typing_extensions>=4.5.0',
        'pydantic>=2.7',
        'sdm-connectors==0.1.0',
        'antlr4-python3-runtime==4.9.3',
        'rich>=9.0.0',
        'jinja2',
        'tabulate',
    ],
    **kwargs,
)
