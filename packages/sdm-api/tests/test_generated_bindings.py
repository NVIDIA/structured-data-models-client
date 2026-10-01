# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""The generated bindings must match the contract they claim to come from.

Two failures are worth catching separately. Someone editing the generated file
by hand is caught without needing the contract checked out, so it runs
everywhere. Drift against the contract itself needs the sibling repository, so
it skips when that is absent, which is everywhere except a maintainer's
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
        'ids locally; see docs/about/architecture.md.'
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


def test_a_response_item_without_row_index_is_refused():
    """``row_index`` is the join key callers are told to rely on.

    The contract makes it required. If the binding accepted an item without
    one, a caller would receive a row whose key is silently ``None`` and join
    against it.
    """
    from sdm_api.generated.tfm_api import PredictionItem, ResponseContractError

    with pytest.raises(ResponseContractError):
        PredictionItem.from_dict({})
    with pytest.raises(ResponseContractError):
        PredictionItem.from_dict({'prediction': 1})


def test_a_null_row_index_is_refused_as_well_as_a_missing_one():
    from sdm_api.generated.tfm_api import PredictionItem, ResponseContractError

    with pytest.raises(ResponseContractError):
        PredictionItem.from_dict({'row_index': None})


@pytest.mark.parametrize('omitted', ['id', 'model', 'predictions', 'metadata'])
def test_a_response_missing_any_required_field_is_refused(omitted):
    from sdm_api.generated.tfm_api import (
        PredictionResponse,
        ResponseContractError,
    )

    body = {
        'id': 'r-1',
        'model': 'kumo-tabular',
        'predictions': [{'row_index': 0}],
        'metadata': {},
    }
    del body[omitted]
    with pytest.raises(ResponseContractError):
        PredictionResponse.from_dict(body)


def test_a_well_formed_response_parses_and_keeps_its_join_key():
    from sdm_api.generated.tfm_api import PredictionResponse

    response = PredictionResponse.from_dict(
        {
            'id': 'r-1',
            'model': 'kumo-relational',
            'predictions': [
                {'row_index': 0, 'prediction': 'yes'},
                {'row_index': 1, 'prediction': 'no', 'id': 'user-7'},
            ],
            'metadata': {'elapsed_ms': 12},
        }
    )
    assert response.model == 'kumo-relational'
    assert [item.row_index for item in response.predictions] == [0, 1]
    assert response.predictions[1].id == 'user-7'
    # An optional field stays absent rather than being invented.
    assert response.predictions[0].id is None
    assert response.metadata == {'elapsed_ms': 12}


def test_every_spec_required_field_is_generated_without_a_default():
    """The generator derives requiredness from the spec; prove it landed.

    Read off the dataclass rather than the source, so this fails if a future
    regeneration turns a required field back into a defaulted one.
    """
    import dataclasses

    from sdm_api.generated.tfm_api import PredictionItem, PredictionResponse

    required = {
        PredictionItem: {'row_index'},
        PredictionResponse: {'id', 'model', 'predictions', 'metadata'},
    }
    for model, names in required.items():
        defaulted = {
            field.name
            for field in dataclasses.fields(model)
            if field.default is not dataclasses.MISSING
            or field.default_factory is not dataclasses.MISSING
        }
        assert not (names & defaulted), (
            f'{model.__name__} generates spec-required fields with defaults: '
            f'{sorted(names & defaulted)}'
        )


def test_the_recorded_revision_is_the_one_the_digest_belongs_to():
    """Provenance must pin a revision *and* that revision's bytes.

    Checking the shape of the revision string alone would pass against a
    record that names one revision and carries another's digest.
    """
    record = provenance()
    if not CONTRACT.exists():
        pytest.skip('the contract repository is not checked out')

    import subprocess

    head = subprocess.check_output(
        ['git', '-C', str(CONTRACT.parent), 'rev-parse', 'HEAD'], text=True
    ).strip()
    if head != record['source_revision']:
        pytest.skip(
            'the local contract checkout is not at the recorded revision'
        )
    assert (
        hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
        == record['source_sha256']
    )


def test_the_generator_that_produced_these_bindings_is_recorded():
    """A changed generator changes the output, so it belongs in provenance."""
    generator = REPO_ROOT / provenance()['generator']
    assert generator.is_file(), 'provenance names a generator that is missing'
    assert (
        hashlib.sha256(generator.read_bytes()).hexdigest()
        == provenance()['generator_sha256']
    ), 'the generator changed; regenerate so the record matches it'


def test_the_formatter_does_not_own_the_generated_file():
    """Formatting the generated file would break its provenance on every run.

    Its bytes are recorded in PROVENANCE.json, so whoever writes them has to be
    the generator alone. This is a regression guard: the exclusion was added
    after a formatter run silently invalidated the record.
    """
    root = (REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    assert 'sdm_api/generated/tfm_api.py' in root, (
        'the generated bindings are not excluded from ruff; a format run will '
        'rewrite them and break PROVENANCE.json'
    )


def test_a_number_too_large_for_a_float_is_a_contract_error_not_an_overflow():
    """`float(10**400)` raises OverflowError, which no adapter catches.

    The point of routing every numeric field through `_finite` is that a
    malformed response fails as a response-contract problem; an escaping
    OverflowError reads as a client crash instead.
    """
    from sdm_api.generated.tfm_api import ResponseContractError, _finite

    with pytest.raises(ResponseContractError):
        _finite(10**400)


def test_a_number_spelled_as_a_string_is_still_accepted():
    """Servers do send one, and refusing a value that parses exactly would
    be a regression with nothing gained."""
    from sdm_api.generated.tfm_api import _finite

    assert _finite('0.6') == 0.6
