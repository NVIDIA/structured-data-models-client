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
    KumoTabular,
    Request,
    SDMClient,
    StructuredDataError,
    UnsupportedRequestError,
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


def test_the_same_readings_score_the_same_twice(anomaly):
    """Reconstruction is sampled, so the answer is seeded. Without that an
    unchanged request comes back with different numbers every call."""
    frame = readings()

    assert scored(anomaly, frame)['prediction'].tolist() == pytest.approx(
        scored(anomaly, frame)['prediction'].tolist()
    )


def test_the_disturbed_stretch_scores_above_the_rest(anomaly):
    """The score is what the model was asked for, so it has to track the
    thing that was done to the data rather than merely be finite."""
    answer = scored(anomaly, readings())
    disturbed = answer['prediction'].iloc[DISTURBED]
    rest = answer['prediction'].drop(answer.index[DISTURBED])

    assert disturbed.mean() > rest.mean()


def test_the_answer_is_the_one_the_model_gives_when_called_directly(
    sdm, anomaly
):
    """Nothing is computed on the way through: the client shapes the request
    and reads the answer back, and the numbers are the model's own."""
    import torch

    frame = readings()
    answer = scored(anomaly, frame)
    channels = frame[['flow', 'pressure']]
    direct = anomaly(
        sdm.TableTensor.from_pandas(
            channels, dict.fromkeys(channels.columns, 'numerical')
        ),
        generator=torch.Generator().manual_seed(SEED),
        **FAST,
    )

    torch.testing.assert_close(
        torch.tensor(answer['prediction'].to_numpy()),
        direct['mae'].numerical[..., 0].double(),
    )


def test_a_key_column_is_not_read_as_a_channel(anomaly):
    """`reading_id` names the row. Reconstructing it would score a row by how
    surprising its identifier is, which rises with every row."""
    frame = readings()
    without = frame.drop(columns=['reading_id'])

    assert scored(anomaly, frame)['prediction'].tolist() == pytest.approx(
        scored(anomaly, without)['prediction'].tolist()
    )


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


def test_a_column_the_model_cannot_reconstruct_is_refused(anomaly):
    """A reading it cannot reconstruct has no error to score."""
    frame = readings()
    frame['site'] = 'north'

    with pytest.raises(StructuredDataError) as caught:
        scored(anomaly, frame)
    assert "'site'" in str(caught.value) or 'site' in str(caught.value)


def test_there_is_no_session_to_pin(anomaly):
    """A session pins a fitted context, and this model fits none."""
    with SDMClient.in_process(anomaly) as client:
        with pytest.raises(
            (UnsupportedRequestError, StructuredDataError)
        ) as caught:
            client.fit(Request.anomaly(readings()), model=KumoAnomaly())
    assert 'session' in str(caught.value)


def test_another_model_is_not_asked_this_question(tabular):
    """`anomaly_detection` is a task only one model here answers."""
    with SDMClient.in_process(tabular) as client:
        with pytest.raises(StructuredDataError):
            client.predict(Request.anomaly(readings()), model=KumoTabular())


@pytest.mark.parametrize('option', ['num_samples', 'num_steps'])
@pytest.mark.parametrize('value', [0, -1, 1.5])
def test_the_sampling_knobs_take_a_count(anomaly, option, value):
    with pytest.raises(StructuredDataError) as caught:
        scored(anomaly, readings(), **{option: value})
    assert option in str(caught.value)
