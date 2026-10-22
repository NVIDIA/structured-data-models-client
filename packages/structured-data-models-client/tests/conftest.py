# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small random-weight models, standing in for the published ones so that
nothing is downloaded and no GPU is needed."""

from __future__ import annotations

import pytest


@pytest.fixture(scope='module')
def sdm():
    return pytest.importorskip('sdm', reason='needs the in-process extra')


@pytest.fixture(scope='module')
def tabular(sdm):
    import torch

    torch.manual_seed(0)
    return sdm.models.KumoTabular(size='small', pretrained=False, device='cpu')


@pytest.fixture(scope='module')
def relational(sdm):
    pytest.importorskip('sdm_relational_engine', reason='needs [relational]')
    import torch

    torch.manual_seed(0)
    return sdm.models.KumoRelational(pretrained=False, device='cpu')
