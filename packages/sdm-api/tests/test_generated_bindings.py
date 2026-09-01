# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The generated bindings must match the contract they claim to come from.

Two failures are worth catching separately. Someone editing the generated file
by hand is caught without needing the contract checked out, so it runs
everywhere. Drift against the contract itself needs the sibling repository, so
it skips when that is absent -- which is everywhere except a maintainer's
machine and the pipeline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
GENERATED = PACKAGE_ROOT / 'src' / 'sdm_api' / 'generated'
PROVENANCE = GENERATED / 'PROVENANCE.json'
TFM_API = GENERATED / 'tfm_api.py'

REPO_ROOT = PACKAGE_ROOT.parent.parent
CONTRACT = REPO_ROOT.parent / 'structured-data-api' / 'nim-sd.openapi.yaml'


def provenance() -> dict:
    return json.loads(PROVENANCE.read_text(encoding='utf-8'))


def test_the_generated_file_was_not_edited_by_hand():
    recorded = provenance()['generated_sha256']
    actual = hashlib.sha256(TFM_API.read_bytes()).hexdigest()
    assert actual == recorded, (
        'tfm_api.py does not match the digest recorded when it was generated. '
        'Regenerate with scripts/generate_sdm_api.py rather than editing it.'
    )


def test_the_provenance_names_a_contract_revision():
    record = provenance()
    assert len(record['source_revision']) == 40, 'not a full git revision'
    assert len(record['source_sha256']) == 64
    assert record['source_repository'] == 'structured-data-api'


def test_the_banner_agrees_with_the_recorded_source_digest():
    banner = TFM_API.read_text(encoding='utf-8').splitlines()[6]
    assert provenance()['source_sha256'] in banner


@pytest.mark.skipif(
    not CONTRACT.exists(), reason='the contract repository is not checked out'
)
def test_the_recorded_contract_digest_still_matches_the_contract():
    actual = hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    assert actual == provenance()['source_sha256'], (
        'the canonical contract has moved since these bindings were '
        'generated. Regenerate them, and re-read the architecture plan on what '
        'else pins this contract.'
    )


def test_both_shipping_models_are_reachable():
    from sdm_api import (
        TFM_MODEL_KUMO_RELATIONAL,
        TFM_MODEL_KUMO_TABULAR,
        TFM_MODEL_VALUES,
    )

    assert TFM_MODEL_KUMO_RELATIONAL == 'kumo-relational'
    assert TFM_MODEL_KUMO_TABULAR == 'kumo-tabular'
    assert set(TFM_MODEL_VALUES) == {'kumo-relational', 'kumo-tabular'}


def test_the_model_field_is_a_closed_enum_which_bounds_the_escape_hatch():
    """A new model id cannot be *sent* until the contract carries it.

    This is the constraint the architecture plan calls out in section 0. The
    client refuses an unknown id locally and names the contract change; this
    test is what tells us the day that stops being necessary.
    """
    from sdm_api import TFM_MODEL_VALUES

    assert len(TFM_MODEL_VALUES) == 2, (
        'the contract now carries a different set of model ids. If it has been '
        'opened to an unbounded string, the client may stop refusing unknown '
        'ids locally -- see section 0 of the architecture plan.'
    )


def test_every_operation_the_client_needs_is_generated():
    from sdm_api import TFM_ENDPOINTS_BY_OPERATION_ID

    for operation in (
        'runPrediction',
        'createSession',
        'runSessionPrediction',
        'deleteSession',
    ):
        assert operation in TFM_ENDPOINTS_BY_OPERATION_ID, (
            f'{operation} is missing; the client cannot reach the endpoint'
        )


def test_the_prediction_and_session_paths_are_what_the_client_posts_to():
    from sdm_api import TFM_ENDPOINTS_BY_OPERATION_ID

    endpoints = TFM_ENDPOINTS_BY_OPERATION_ID
    assert endpoints['runPrediction'].get_path() == '/v1/predictions'
    assert endpoints['createSession'].get_path() == '/v1/sessions'
    assert (
        endpoints['runSessionPrediction'].get_path()
        == '/v1/sessions/{session_id}/predictions'
    )
    assert endpoints['deleteSession'].get_path() == '/v1/sessions/{session_id}'


def test_the_seven_output_fields_and_the_task_kinds_are_carried():
    from sdm_api import TFM_OUTPUT_FIELD_VALUES, TFM_TASK_KIND_VALUES

    assert set(TFM_OUTPUT_FIELD_VALUES) == {
        'prediction',
        'probabilities',
        'scores',
        'rankings',
        'embeddings',
        'quantiles',
        'explanation',
    }
    # Tabular sends only the coarse 'classification'; relational may narrow it.
    # Both narrowings have to exist on the wire for that to be possible.
    assert 'classification' in TFM_TASK_KIND_VALUES
    assert 'binary_classification' in TFM_TASK_KIND_VALUES
    assert 'multiclass_classification' in TFM_TASK_KIND_VALUES


def test_the_package_carries_no_third_party_dependency():
    """Anything needing only the wire vocabulary must not inherit a runtime."""
    text = (PACKAGE_ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    body = text.split('dependencies = [', 1)[1].split(']', 1)[0]
    assert not body.strip(), f'sdm-api grew a dependency: {body.strip()}'
