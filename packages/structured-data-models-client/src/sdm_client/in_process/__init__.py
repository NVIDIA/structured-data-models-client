# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The models served in this Python process rather than by a NIM.

Behind :meth:`SDMClient.in_process <sdm_client.SDMClient.in_process>`, and not
public API itself. Needs the ``in-process`` extra, which brings
structured-data-models and PyTorch.
"""

from sdm_client.in_process.backend import InProcessBackend

__all__ = ['InProcessBackend']
