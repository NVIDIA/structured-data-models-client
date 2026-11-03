# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An anomaly answer put back over the readings it was about, and drawn."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdm_client import KumoAnomaly, Request, SDMClient, StructuredDataError

pytest.importorskip('matplotlib', reason='needs the report extra')

from sdm_client.reporting import anomaly_report  # noqa: E402

pytestmark = pytest.mark.filterwarnings('ignore')

FAST = {'num_samples': 2, 'num_steps': 4}
DISTURBED = slice(80, 90)


def readings(rows: int = 120) -> pd.DataFrame:
    step = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            'ts': pd.date_range('2026-01-01', periods=rows, freq='h'),
            'flow': np.sin(step / 9),
            'pressure': np.cos(step / 14),
        }
    )
    frame.loc[DISTURBED, ['flow', 'pressure']] += 4.0
    return frame


def scored(model, frame: pd.DataFrame):
    channels = frame.select_dtypes(include=[np.number])
    with SDMClient.in_process(model) as client:
        return client.predict(
            Request.anomaly(channels, inference=FAST), model=KumoAnomaly()
        )


def pages(path) -> int:
    raw = path.read_bytes()
    return raw.count(b'/Type /Page') - raw.count(b'/Type /Pages')


def test_a_report_is_drawn_from_an_answer_and_its_readings(anomaly, tmp_path):
    """Three pages: the score, the signals, and what the score means."""
    frame = readings()
    written = anomaly_report(
        scored(anomaly, frame), frame, tmp_path / 'run.pdf'
    )

    assert written.exists()
    assert pages(written) == 3


def test_the_readings_may_carry_what_the_model_was_not_sent(anomaly, tmp_path):
    """The model refuses a timestamp, and a report wants one to plot
    against, so the frame drawn is wider than the frame scored. Dropping a
    column does not move a row, so the positions still line up."""
    frame = readings()
    result = scored(anomaly, frame)

    assert 'ts' not in result.to_pandas().columns
    assert anomaly_report(result, frame, tmp_path / 'run.pdf').exists()


def test_readings_that_are_not_the_ones_scored_are_refused(anomaly, tmp_path):
    """A filtered frame lines up with nothing, and drawing it would put the
    right labels under the wrong rows."""
    frame = readings()
    result = scored(anomaly, frame)

    with pytest.raises(StructuredDataError) as caught:
        anomaly_report(result, frame.iloc[:50], tmp_path / 'run.pdf')
    assert 'scored' in str(caught.value)


def test_rows_are_matched_by_position_and_not_by_index_label(anomaly, tmp_path):
    """`row_index` is a position in the frame that was sent, not a label of
    this frame's index. A frame indexed from 100 lines up with nothing if
    the two are confused."""
    frame = readings()
    relabelled = frame.set_index(np.arange(100, 100 + len(frame)))

    assert anomaly_report(
        scored(anomaly, frame), relabelled, tmp_path / 'run.pdf'
    ).exists()


def test_a_reading_already_called_prediction_is_kept(anomaly, tmp_path):
    """The score and the label are put onto the frame to draw it. A reading
    of that name would be replaced by one of them and plotted as the score."""
    from sdm_client.reporting.anomaly import _rejoined

    frame = readings()
    frame['prediction'] = np.arange(len(frame), dtype=float) * 10
    frame['detected'] = 0

    drawn, score, detected = _rejoined(scored(anomaly, frame), frame, 'scs')

    assert score != 'prediction' and detected != 'detected'
    assert drawn['prediction'].tolist() == frame['prediction'].tolist()
    assert drawn['detected'].tolist() == frame['detected'].tolist()


def test_a_label_that_is_not_a_label_is_refused(anomaly, tmp_path):
    """0.9 is not a verdict. Rounding it to one before anything looked would
    have made it a detection nobody reported."""
    frame = readings()
    result = scored(anomaly, frame)
    answer = result.to_pandas()
    answer.loc[0, 'probability_scs'] = 0.9
    result.to_pandas = lambda: answer

    with pytest.raises(ValueError, match='binary'):
        anomaly_report(result, frame, tmp_path / 'run.pdf')


def test_a_timestamp_index_is_the_axis_it_is_drawn_against(anomaly, tmp_path):
    """Upstream plots against a DatetimeIndex when there is one. Giving the
    readings a fresh index drew dates a month apart as even steps."""
    from sdm_client.reporting.anomaly import _rejoined

    frame = readings().set_index('ts')

    drawn, _, _ = _rejoined(scored(anomaly, frame), frame, 'scs')

    assert isinstance(drawn.index, pd.DatetimeIndex)


def test_the_report_says_which_thresholder_marked_it(anomaly, tmp_path):
    """They disagree, so a reader has to be told which one this is."""
    frame = readings()
    written = anomaly_report(
        scored(anomaly, frame),
        frame,
        tmp_path / 'run.pdf',
        detected_by='macs',
    )

    assert b'MACS' in written.read_bytes()


def test_the_score_and_the_label_are_not_drawn_as_signals(anomaly, tmp_path):
    """They are put on the frame to draw it, and reading every numerical
    column off that frame would plot them as though they were readings."""
    from sdm_client.reporting.anomaly import _signals

    frame = readings()

    assert _signals(
        frame, None, timestamp_column='ts', ground_truth_column=None
    ) == ['flow', 'pressure']


def test_a_thresholder_that_was_not_asked_for_is_refused(anomaly, tmp_path):
    with pytest.raises(StructuredDataError) as caught:
        anomaly_report(
            scored(anomaly, readings()),
            readings(),
            tmp_path / 'run.pdf',
            detected_by='percentile',
        )
    assert 'scs' in str(caught.value)


def test_an_answer_without_the_labels_says_what_to_ask_for(anomaly, tmp_path):
    frame = readings()
    channels = frame.select_dtypes(include=[np.number])
    with SDMClient.in_process(anomaly) as client:
        result = client.predict(
            Request.anomaly(channels, inference=FAST),
            model=KumoAnomaly(),
            outputs=['prediction'],
        )

    with pytest.raises(StructuredDataError) as caught:
        anomaly_report(result, frame, tmp_path / 'run.pdf')
    assert 'probabilities' in str(caught.value)


@pytest.mark.parametrize('detected_by', ['scs', 'macs'])
def test_either_thresholder_may_mark_the_detections(
    anomaly, tmp_path, detected_by
):
    """They disagree, so a report marks one of them and says which."""
    frame = readings()
    written = anomaly_report(
        scored(anomaly, frame),
        frame,
        tmp_path / f'{detected_by}.pdf',
        detected_by=detected_by,
    )

    assert written.exists()


def test_a_signal_the_readings_do_not_carry_is_refused(anomaly, tmp_path):
    frame = readings()

    with pytest.raises(StructuredDataError) as caught:
        anomaly_report(
            scored(anomaly, frame),
            frame,
            tmp_path / 'run.pdf',
            feature_columns=['nothing'],
        )
    assert 'nothing' in str(caught.value)


def test_the_caller_may_mark_the_detections_themselves(anomaly, tmp_path):
    """Both thresholders read the whole run and decide for themselves where
    the line falls, so a reader who wants it elsewhere has to supply it."""
    from sdm_client import percentile_thresholds

    frame = readings()
    result = scored(anomaly, frame)
    scores = result.to_pandas().sort_values('row_index')['prediction']
    _, upper = percentile_thresholds(scores, upper_percentile=90.0)

    written = anomaly_report(
        result,
        frame,
        tmp_path / 'run.pdf',
        detected_by=(scores.to_numpy() > upper),
    )

    assert b'supplied by the caller' in written.read_bytes()


def test_supplied_detections_are_read_by_position_not_by_label(
    anomaly, tmp_path
):
    """They mark one row of the readings each, in the order the readings are
    in. Aligned on an index instead, a detection lands on whichever row
    happened to share a label, which is usually none of them."""
    from sdm_client.reporting.anomaly import _rejoined

    frame = readings()
    marks = pd.Series(
        [False] * len(frame), index=np.arange(500, 500 + len(frame))
    )
    marks.iloc[3] = True

    drawn, _, detected = _rejoined(scored(anomaly, frame), frame, marks)

    assert drawn[detected].to_numpy().nonzero()[0].tolist() == [3]


def test_supplied_detections_that_are_scores_are_refused(anomaly, tmp_path):
    """Drawn as detections they mark every row whose score rounds up, which
    is a different report that looks exactly like the one that was asked
    for."""
    frame = readings()

    with pytest.raises(StructuredDataError) as caught:
        anomaly_report(
            scored(anomaly, frame),
            frame,
            tmp_path / 'run.pdf',
            detected_by=np.linspace(0.0, 1.0, len(frame)),
        )
    assert 'threshold' in str(caught.value)


def test_supplied_detections_that_are_not_one_per_row_are_refused(
    anomaly, tmp_path
):
    frame = readings()

    with pytest.raises(StructuredDataError) as caught:
        anomaly_report(
            scored(anomaly, frame),
            frame,
            tmp_path / 'run.pdf',
            detected_by=np.zeros(len(frame) - 1, dtype=bool),
        )
    assert 'one row each' in str(caught.value)


def test_supplied_detections_need_no_thresholder_in_the_answer(
    anomaly, tmp_path
):
    """The labels are the caller's, so the answer does not have to carry
    any, and a report of the score alone is then a report."""
    frame = readings()
    channels = frame.select_dtypes(include=[np.number])
    with SDMClient.in_process(anomaly) as client:
        result = client.predict(
            Request.anomaly(channels, inference=FAST),
            model=KumoAnomaly(),
            outputs=['prediction'],
        )

    assert anomaly_report(
        result,
        frame,
        tmp_path / 'run.pdf',
        detected_by=np.zeros(len(frame), dtype=bool),
    ).exists()
