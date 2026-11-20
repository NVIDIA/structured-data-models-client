# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``kumo-anomaly`` asked for a score, with nothing labelled to learn from.

A small random-weight model stands in for the published one, so nothing is
downloaded and no GPU is needed. The sampling is turned down for the same
reason: reconstructions are combined by their median, so fewer samples is a
noisier score rather than a different kind of answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdm_client import (
    KumoAnomaly,
    Request,
    SDMClient,
    StructuredDataError,
)

pytestmark = pytest.mark.filterwarnings('ignore')

FAST = {'num_samples': 2, 'num_steps': 4}
#: What the backend seeds every run with, named here so the direct call this
#: is compared against draws the same samples.
SEED = 42
DISTURBED = slice(80, 90)


def readings(rows: int = 120) -> pd.DataFrame:
    """A two-channel series with one stretch pushed off its own shape."""
    step = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            'reading_id': np.arange(rows),
            'flow': np.sin(step / 9),
            'pressure': np.cos(step / 14),
        }
    )
    frame.loc[DISTURBED, ['flow', 'pressure']] += 4.0
    return frame


def scored(model, frame: pd.DataFrame, **inference) -> pd.DataFrame:
    with SDMClient.in_process(model) as client:
        request = Request.anomaly(frame, inference={**FAST, **inference})
        return client.predict(request, model=KumoAnomaly()).to_pandas()


def test_a_score_and_a_label_per_row_come_back(anomaly):
    """The answer names the row, its score, and each thresholder's verdict."""
    frame = readings()
    answer = scored(anomaly, frame)

    assert list(answer.columns) == [
        'row_index',
        'id',
        'prediction',
        'probability_scs',
        'probability_macs',
    ]
    assert len(answer) == len(frame)
    assert answer['row_index'].tolist() == list(range(len(frame)))
    assert np.isfinite(answer['prediction']).all()
    for name in ('probability_scs', 'probability_macs'):
        assert set(answer[name].unique()) <= {0.0, 1.0}


def test_only_the_fields_that_were_asked_for_come_back(anomaly):
    """Asking for the score alone is a smaller answer, not the same one with
    the labels computed and sent anyway."""
    with SDMClient.in_process(anomaly) as client:
        request = Request.anomaly(readings(), inference=FAST)
        answer = client.predict(
            request, model=KumoAnomaly(), outputs=['prediction']
        ).to_pandas()

    assert 'prediction' in answer.columns
    assert not [name for name in answer.columns if name.startswith('prob')]


def test_the_answer_is_the_one_the_model_gives_when_called_directly(
    sdm, anomaly
):
    """Nothing is computed on the way through: the client shapes the request
    and reads the answer back, and the numbers are the model's own, with or
    without a key beside the readings. They track the stretch that was
    disturbed rather than merely being finite."""
    import torch

    frame = readings()
    answer = scored(anomaly, frame)
    channels = frame[['flow', 'pressure']]
    keyless = scored(anomaly, channels)
    direct = anomaly(
        sdm.TableTensor.from_pandas(
            channels, dict.fromkeys(channels.columns, 'numerical')
        ),
        generator=torch.Generator().manual_seed(SEED),
        **FAST,
    )

    for each in (answer, keyless):
        torch.testing.assert_close(
            torch.tensor(each['prediction'].to_numpy()),
            direct['mae'].numerical[..., 0].double(),
        )
    disturbed = answer['prediction'].iloc[DISTURBED]
    assert disturbed.mean() > answer['prediction'].drop(disturbed.index).mean()


def test_each_answer_comes_back_under_the_row_s_own_key(anomaly):
    """`sdm_client.result` tells callers to join on the key, so a key that
    joins for one model and not another is a difference nobody expects."""
    frame = readings()
    frame['reading_id'] += 1000
    answer = scored(anomaly, frame)

    assert answer['id'].tolist() == frame['reading_id'].tolist()


def test_readings_are_answered_though_the_schema_was_built_from_nulls(
    anomaly,
):
    """A column with nothing in it is declared categorical because a schema
    has to say something. The readings then decide its dtype, and refusing
    them for the placeholder's sake refuses a question that is answerable."""
    from sdm_client import Schema

    frame = readings()[['flow', 'pressure']]
    blank = Schema.from_data({'instance': frame.iloc[:0].astype(object)})

    with SDMClient.in_process(anomaly) as client:
        request = Request.anomaly(frame, schema=blank, inference=FAST)
        answer = client.predict(request, model=KumoAnomaly()).to_pandas()

    assert len(answer) == len(frame)


def test_readings_that_leave_out_a_composite_key_are_answered(anomaly):
    """A key names the row and is not a channel, so a caller may leave it
    out. Then there is no value to answer under and rows match by position."""
    from sdm_client import Schema

    frame = readings()
    frame['site'] = 'north'
    schema = Schema.from_data({'instance': frame})
    schema['instance'].primary_key = ['reading_id', 'site']

    channels = frame[['flow', 'pressure']]
    with SDMClient.in_process(anomaly) as client:
        answer = client.predict(
            Request.anomaly(channels, schema=schema, inference=FAST),
            model=KumoAnomaly(),
        ).to_pandas()

    assert len(answer) == len(frame)
    assert answer['row_index'].tolist() == list(range(len(frame)))


@pytest.mark.parametrize('held', [True, False])
def test_a_meaning_the_caller_gave_a_column_is_kept(anomaly, held):
    """The placeholder is a meaning nobody chose. One a caller typed by hand
    is a claim, whether or not the column it is on held anything to read."""
    from sdm_client import Schema

    frame = readings()[['flow', 'pressure']]
    source = frame if held else frame.iloc[:0].astype(object)
    schema = Schema.from_data({'instance': source})
    schema['instance']['flow'].stype = 'categorical'

    with pytest.raises(StructuredDataError):
        with SDMClient.in_process(anomaly) as client:
            client.predict(
                Request.anomaly(frame, schema=schema, inference=FAST),
                model=KumoAnomaly(),
            )


def test_a_score_that_is_not_a_number_is_refused(anomaly):
    """A row scored NaN is not a row that is unremarkable, and a frame of
    them cannot be thresholded or plotted."""
    frame = pd.DataFrame({'flow': [1e100, 2e100] * 50})

    with pytest.raises(StructuredDataError) as caught:
        scored(anomaly, frame)
    assert 'malformed' in str(caught.value)


def test_a_column_the_model_cannot_reconstruct_is_refused(anomaly):
    """A reading it cannot reconstruct has no error to score."""
    frame = readings()
    frame['site'] = 'north'

    with pytest.raises(StructuredDataError) as caught:
        scored(anomaly, frame)
    assert 'site' in str(caught.value)


def test_there_is_no_session_to_pin(anomaly):
    """A session pins a fitted context, and this model fits none."""
    with SDMClient.in_process(anomaly) as client:
        with pytest.raises(StructuredDataError) as caught:
            client.fit(Request.anomaly(readings()), model=KumoAnomaly())
    assert 'session' in str(caught.value)


def test_the_supervised_constructor_will_not_build_this_question(anomaly):
    """It takes a context and a target, and nothing reads either here. Built
    that way the request came back with a whole-row score under a name the
    caller had asked about, which is a different question answered quietly."""
    frame = readings()

    with pytest.raises(StructuredDataError) as caught:
        Request(
            context=frame,
            predict=frame.drop(columns=['flow']),
            target='flow',
            task='anomaly_detection',
        )
    assert 'Request.anomaly' in str(caught.value)


def test_readings_that_are_not_what_the_schema_says_are_refused(anomaly):
    """A schema the caller brought says what a column holds. Readings that
    hold something else are a different series under that name, and were
    widened into one rather than refused."""
    from sdm_client import Schema

    frame = readings()[['flow', 'pressure']]
    schema = Schema.from_data({'instance': frame})
    flipped = frame.assign(flow=[True, False] * (len(frame) // 2))

    with pytest.raises(StructuredDataError):
        with SDMClient.in_process(anomaly) as client:
            client.predict(
                Request.anomaly(flipped, schema=schema, inference=FAST),
                model=KumoAnomaly(),
            )


def test_a_structured_data_models_without_this_model_still_answers(
    sdm, monkeypatch
):
    """Every published release predates `KumoAnomaly`, and this is asked of
    every model handed to the client, including ones it does not serve. It
    has to answer no there rather than raise the import."""
    monkeypatch.delattr(sdm.models, 'KumoAnomaly', raising=False)

    with pytest.raises(StructuredDataError) as caught:
        SDMClient.in_process(object())
    assert caught.value.code == 'INVALID_CONFIGURATION'


def test_a_model_that_is_not_installed_is_not_offered(sdm, monkeypatch):
    """`models()` says what this backend answers for. Naming one it cannot
    load sends a caller to a refusal they were told to expect an answer to."""
    monkeypatch.delattr(sdm.models, 'KumoAnomaly', raising=False)

    with SDMClient.in_process() as client:
        assert 'kumo-anomaly' not in client.models()


@pytest.mark.parametrize('option', ['num_samples', 'num_steps'])
@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_the_sampling_knobs_take_a_count(anomaly, option, value):
    with pytest.raises(StructuredDataError) as caught:
        scored(anomaly, readings(), **{option: value})
    assert option in str(caught.value)
