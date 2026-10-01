# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

import os

from setuptools import setup

# The retained wire types are first-party code and the query language comes
# from ``predictive_query``, so the client carries no external ``kumo-api``
# dependency. ``pydantic`` used to arrive transitively through it; it is now
# declared directly.

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
        'sdm-api==1.0.0',
        'sdm-pql==1.0.0',
        'sdm-connectors==1.0.0',
        'antlr4-python3-runtime==4.13.2',
        'rich>=9.0.0',
        'jinja2',
        'tabulate',
    ],
    **kwargs,
)
