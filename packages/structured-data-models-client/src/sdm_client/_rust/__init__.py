# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The Python host of the Rust core.

Only a core route reaches this package, and only it imports
``sdm_client._native``: the control-message codec (:mod:`.codec`), the host
facts a run starts with (:mod:`.facts`), the run loop (:mod:`.run`), the one
table from the core's error codes to this client's exceptions
(:mod:`.errors`), and the S0 route (:mod:`.s0`). Importing this package
imports nothing native; :func:`.run.native` does, when a run starts.
"""
