# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The Python host's data providers: each answers the fetch steps of a core
run, one request at a time, with an object that exports Arrow through
``__arrow_c_stream__``.

A provider reads no pandas and converts nothing: the driver's Arrow, or,
from a driver that gives rows (psycopg), the Arrow pyarrow builds of its
values, is what crosses to the core, where admission and the schema checks
hold it (design, "Core, host and binding"). A test holds every module here
to that, by its imports and by running one with pandas unimportable.
"""
