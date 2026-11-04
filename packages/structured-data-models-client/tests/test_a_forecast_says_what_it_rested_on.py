# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Why a forecast is the forecast it is, and the report that draws it.

An explanation walks a window per step of the model's context, so it costs
tens of seconds where a forecast costs a fraction of one. It is computed once
here and read by every test that needs one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdm_client import KumoForecast, Request, SDMClient, StructuredDataError

pytest.importorskip('matplotlib', reason='needs the report extra')
pytest.importorskip('torch', reason='the attribution runs on torch')

from sdm_client.reporting import forecast_report  # noqa: E402

pytestmark = pytest.mark.filterwarnings('ignore')


def history(rows: int = 600) -> pd.DataFrame:
    step = np.arange(rows, dtype=float)
    return pd.DataFrame({'flow': np.sin(step / 9), 'temp': np.cos(step / 7)})


@pytest.fixture(scope='module')
def explained(forecast):
    with SDMClient.in_process(forecast) as client:
        return client.predict(
            Request.forecast(history(), target='flow', horizon=4),
            model=KumoForecast(),
            outputs=['prediction', 'explanation'],
        )


class Carrying:
    """An answer carrying an explanation, for drawing one that was edited.

    A result's explanation is what it came back with and cannot be assigned
    over, and a page is drawn or not drawn by whether its block is there.
    """

    def __init__(self, explanation):
        self.explanation = explanation


def without(result, *names, under=None):
    """What `result` carries, less the block(s) `names` asks for."""
    carried = dict(result.explanation)
    block = dict(carried['explanation'])
    if under is None:
        for name in names:
            block.pop(name, None)
    else:
        block[under] = {
            key: value
            for key, value in block[under].items()
            if key not in names
        }
    carried['explanation'] = block
    return Carrying(carried)


def pages(path) -> int:
    raw = path.read_bytes()
    return raw.count(b'/Type /Page') - raw.count(b'/Type /Pages')


def test_an_explanation_comes_back_when_it_is_asked_for(explained):
    """It reads the model's gradients and its latent trajectory, so it is
    produced where the model runs and nowhere else."""
    carried = explained.explanation

    assert isinstance(carried, dict)
    assert set(carried) >= {'forecast', 'explanation', 'metadata'}


def test_the_attribution_covers_every_lag_and_every_step(explained):
    block = explained.explanation['explanation']
    attributions = np.asarray(block['lag_horizon_attributions'])

    assert attributions.shape == (block['shapes']['K_lags'], 4)
    assert np.isfinite(attributions).all()


def test_an_explanation_is_data_rather_than_an_object(explained):
    """It travels as the contract's `explanation` field, so it has to be
    what that field carries."""
    import json

    assert json.loads(json.dumps(explained.explanation))


def test_a_forecast_asked_without_one_does_not_pay_for_it(forecast):
    """It costs far more than the forecast, so it is not computed unless it
    was asked for."""
    with SDMClient.in_process(forecast) as client:
        result = client.predict(
            Request.forecast(history(), target='flow', horizon=4),
            model=KumoForecast(),
            outputs=['prediction'],
        )

    assert result.explanation is None


def test_a_flat_series_has_almost_nowhere_to_move(forecast):
    """The trajectory builder pads its windows and says which steps are
    padding. Reading the padding as observed zeros made a constant series
    look like it moved."""
    flat = pd.DataFrame({'flow': np.full(600, 10.0)})
    with SDMClient.in_process(forecast) as client:
        result = client.predict(
            Request.forecast(flat, target='flow', horizon=4),
            model=KumoForecast(),
            outputs=['prediction', 'explanation'],
        )

    flow = np.asarray(result.explanation['explanation']['flow_magnitudes'])
    assert float(flow.sum()) < 0.5


def test_a_forecast_retrieval_had_a_hand_in_is_not_explained(forecast):
    """The attribution describes the model's own forward pass. Where
    retrieval contributes, explaining that pass describes a number the
    caller was never given."""
    history_long = history(600)
    with SDMClient.in_process(forecast) as client:
        with pytest.raises(StructuredDataError) as caught:
            client.predict(
                Request.forecast(
                    history_long,
                    target='flow',
                    horizon=4,
                    inference={'alpha': 0.0},
                ),
                model=KumoForecast(),
                outputs=['prediction', 'explanation'],
            )
    assert 'alpha' in str(caught.value)


def test_the_attribution_reads_the_model_s_own_scale(forecast):
    """The model standardises before it reads anything, so the attribution
    works there too. Handing it raw readings attributed a model being fed
    numbers it never sees."""
    from sdm_client.explain._explain import _standardised

    raw = np.array([[1.0, 2.0, 3.0]])
    standardised = _standardised(forecast, raw)

    assert not np.allclose(standardised, raw)
    assert np.allclose(
        standardised, (raw + 5.671202659606934) / 8.693312644958496, atol=1e-4
    )


def test_what_is_drawn_is_in_the_readings_own_scale(explained):
    """The attribution works standardised; a reader does not."""
    baseline = np.asarray(
        explained.explanation['explanation']['baseline_forecast']
    )

    assert np.abs(baseline).max() < 5.0


def test_the_feature_axis_needs_a_horizon_one_pass_reaches(forecast):
    """It reads a single forward pass, which covers the model's own horizon
    and no more."""
    with SDMClient.in_process(forecast) as client:
        with pytest.raises(StructuredDataError) as caught:
            client.predict(
                Request.forecast(
                    history(600),
                    target='flow',
                    horizon=80,
                    inference={'feature_axis': True},
                ),
                model=KumoForecast(),
                outputs=['prediction', 'explanation'],
            )
    assert 'feature-axis' in str(caught.value)


def test_the_diagnostics_survive_being_written_and_read(explained):
    """They are computed, serialised, and read back for the page that prints
    them, which printed n/a for every one of them."""
    from sdm_client.reporting.forecast import _explanation

    restored = _explanation(explained.explanation)

    assert restored.flow_ratio_forecast_vs_history is not None
    assert restored.curvature_ratio_forecast_vs_history is not None


def test_the_top_lags_are_ranked_from_the_scores(explained, tmp_path):
    """That helper differences a cumulative score into a per-step weight, so
    handing it the normalised attribution ranks the lags near-backwards.

    Asserted on what the report passes. Comparing the helper against itself
    said nothing about the call that draws the page, and passed while the
    report still handed over the wrong array.
    """
    from sdm_client.explain._attribution import pages
    from sdm_client.reporting import forecast as module

    seen = []
    original = pages._topk_lag_steps_per_horizon

    def watched(scores, **kwargs):
        seen.append(np.asarray(scores))
        return original(scores, **kwargs)

    pages._topk_lag_steps_per_horizon = watched
    try:
        module.forecast_report(explained, tmp_path / 'f.pdf')
    finally:
        pages._topk_lag_steps_per_horizon = original

    block = explained.explanation['explanation']
    assert len(seen) == 1
    assert np.allclose(seen[0], np.asarray(block['lag_horizon_scores']))


def test_the_history_the_page_is_told_about_excludes_the_forecast(explained):
    """The trajectory runs the context and then the horizon. Reading its
    length as the context said every transition was history."""
    from sdm_client.reporting.forecast import _context_len

    block = explained.explanation['explanation']
    steps = block['diagnostics']['latent_trajectory_shape'][0]

    assert _context_len(explained.explanation, 4) == steps - 4


def test_a_gap_outside_the_window_is_not_a_gap_in_it(forecast):
    """The attribution reads the last 512 steps, so a reading missing before
    them is not missing from what it reads. Checking the whole history
    refused a question identical to one it answers."""
    frame = history(600)
    frame.loc[0, 'temp'] = np.nan

    with SDMClient.in_process(forecast) as client:
        result = client.predict(
            Request.forecast(
                frame,
                target='flow',
                horizon=4,
                inference={'feature_axis': True},
            ),
            model=KumoForecast(),
            outputs=['prediction', 'explanation'],
        )

    assert result.explanation is not None


def test_the_files_the_pages_name_are_written(explained, tmp_path):
    """A page that names a file beside a report a reader cannot find points
    them at nothing."""
    artifacts = tmp_path / 'artifacts'
    forecast_report(explained, tmp_path / 'f.pdf', artifacts_dir=artifacts)
    written = {path.name for path in artifacts.iterdir()}

    assert 'lag_horizon_heatmap.png' in written
    assert 'semantic_flow.csv' in written
    assert 'explanation.json' in written


def test_a_value_json_cannot_spell_does_not_cost_the_report(
    explained, tmp_path
):
    """The explanation is written out before any page is drawn, so one
    value no JSON reader accepts took the whole report with it."""
    import json

    artifacts = tmp_path / 'artifacts'
    carried = dict(explained.explanation)
    carried['diagnostics'] = {'top': [{'share': float('nan')}]}

    forecast_report(
        Carrying(carried), tmp_path / 'f.pdf', artifacts_dir=artifacts
    )
    written = json.loads((artifacts / 'explanation.json').read_text())

    assert written['diagnostics']['top'][0]['share'] is None


def test_the_series_own_channel_is_not_ranked_among_the_measured(
    forecast, tmp_path
):
    """Its effect is set aside rather than measured, so it arrives as a zero.
    Left among real numbers the page ranked it, and with small effects
    elsewhere it called the excluded channel the most influential of all."""
    from sdm_client.reporting.forecast import (
        _channels,
        _explanation,
        _without_the_unmeasured,
    )

    with SDMClient.in_process(forecast) as client:
        result = client.predict(
            Request.forecast(
                history(600),
                target='flow',
                horizon=4,
                inference={'feature_axis': True},
            ),
            model=KumoForecast(),
            outputs=['prediction', 'explanation'],
        )

    carried = result.explanation
    assert carried['metadata']['feature_axis_excludes'] == 'flow'
    _, channels = _without_the_unmeasured(
        _explanation(carried), _channels(carried), carried
    )
    assert channels == ['temp']


def test_a_forecaster_that_cannot_use_a_covariate_does_not_attribute_one(
    sdm,
):
    """Without channel mixing a column measured alongside the series reaches
    the forecast not at all, while the embedding still pools it, so an
    attribution moves for something that changes nothing."""
    import torch

    torch.manual_seed(0)
    flat = sdm.models.KumoForecast(
        cross_channel=False, pretrained=False, device='cpu'
    )
    with SDMClient.in_process(flat) as client:
        with pytest.raises(StructuredDataError) as caught:
            client.predict(
                Request.forecast(history(600), target='flow', horizon=2),
                model=KumoForecast(),
                outputs=['prediction', 'explanation'],
            )
    assert 'mix channels' in str(caught.value)


def test_the_heatmap_shows_what_its_caption_says(explained, tmp_path):
    """Its rows are single past steps, and it was handed the cumulative
    scores those are differenced out of, so a flat run read as the most
    influential part of the window while the table ranked the newest first.
    """
    from sdm_client.explain._attribution import pages

    seen = []
    original = pages._save_lag_horizon_artifacts

    def watched(out_dir, attributions, scores=None, **kwargs):
        seen.append(np.asarray(attributions))
        return original(out_dir, attributions, scores, **kwargs)

    pages._save_lag_horizon_artifacts = watched
    try:
        forecast_report(explained, tmp_path / 'f.pdf')
    finally:
        pages._save_lag_horizon_artifacts = original

    block = explained.explanation['explanation']
    marginal, _ = pages._topk_lag_steps_per_horizon(
        np.asarray(block['lag_horizon_scores']), top_k=5
    )
    assert len(seen) == 1
    assert np.allclose(seen[0], np.asarray(marginal))


def test_the_channel_effects_are_measured_not_differentiated(forecast):
    """This forecaster normalises each window by statistics it takes without
    gradients, so a gradient through it holds that normalisation still and
    reports a sensitivity hundreds of times what a perturbation produces.
    The forward pass is not differentiable, which is what selects the
    difference the attribution would otherwise skip."""
    import torch

    from sdm_client.explain._model import Forecaster

    shim = Forecaster(forecast)
    window = torch.zeros(1, 2, shim.context_size, requires_grad=True)

    assert not shim(x_enc=window).forecast.requires_grad


def test_the_attribution_heatmap_is_drawn(explained, tmp_path):
    """It is a page drawn from an image on disk, and hardcoding no image
    dropped it from every report although the matrix was right there."""
    artifacts = tmp_path / 'artifacts'
    written = forecast_report(
        explained, tmp_path / 'f.pdf', artifacts_dir=artifacts
    )

    assert (artifacts / 'lag_horizon_heatmap.png').exists()
    assert pages(written) >= 4


def test_the_artifacts_are_kept_only_where_a_place_was_named(
    explained, tmp_path
):
    """They are written either way, because a page reads one of them."""
    before = set(tmp_path.iterdir())
    forecast_report(explained, tmp_path / 'f.pdf')

    assert set(tmp_path.iterdir()) - before == {tmp_path / 'f.pdf'}


def test_a_channel_with_a_gap_is_not_called_uninfluential(forecast):
    """The feature axis replaces a channel with its own average, which a gap
    makes undefined; the channel then reads as having no influence at all."""
    frame = history(600)
    frame.loc[599, 'temp'] = np.nan

    with SDMClient.in_process(forecast) as client:
        with pytest.raises(StructuredDataError) as caught:
            client.predict(
                Request.forecast(
                    frame,
                    target='flow',
                    horizon=4,
                    inference={'feature_axis': True},
                ),
                model=KumoForecast(),
                outputs=['prediction', 'explanation'],
            )
    assert 'gap' in str(caught.value)


def test_a_gradient_over_a_gap_is_not_an_attribution(forecast):
    """Integrated gradients measure against a baseline of noise over the
    window, and one gap makes the whole baseline NaN: every attribution
    taken against it reads as no influence at all."""
    frame = history(600)
    frame.loc[599, 'temp'] = np.nan

    with SDMClient.in_process(forecast) as client:
        with pytest.raises(StructuredDataError) as caught:
            client.predict(
                Request.forecast(
                    frame,
                    target='flow',
                    horizon=4,
                    inference={'integrated_gradients': True},
                ),
                model=KumoForecast(),
                outputs=['prediction', 'explanation'],
            )
    assert 'integrated gradients' in str(caught.value)


def test_a_report_is_drawn_from_an_explanation(explained, tmp_path):
    written = forecast_report(explained, tmp_path / 'forecast.pdf')

    assert written.exists()
    assert pages(written) >= 3


def test_the_report_names_the_series_and_the_lag_window(explained, tmp_path):
    """Read off the data the page is drawn from. The text in this PDF is
    compressed, so sniffing its bytes for a word tests the encoder rather
    than the report."""
    metadata = explained.explanation['metadata']

    assert metadata['target_column'] == 'flow'
    assert metadata['channel_labels'] == ['flow', 'temp']
    assert forecast_report(
        explained, tmp_path / 'forecast.pdf', dataset_name='flow'
    ).exists()


def test_a_forecast_with_no_explanation_says_how_to_ask_for_one(
    forecast, tmp_path
):
    with SDMClient.in_process(forecast) as client:
        result = client.predict(
            Request.forecast(history(), target='flow', horizon=4),
            model=KumoForecast(),
            outputs=['prediction'],
        )

    with pytest.raises(StructuredDataError) as caught:
        forecast_report(result, tmp_path / 'forecast.pdf')
    assert 'explanation' in str(caught.value)


def test_a_history_shorter_than_the_model_reads_cannot_be_attributed(
    forecast,
):
    """The attribution is about the window the model read, and a history
    shorter than that window is not one it read."""
    with SDMClient.in_process(forecast) as client:
        with pytest.raises(StructuredDataError) as caught:
            client.predict(
                Request.forecast(history(100), target='flow', horizon=2),
                model=KumoForecast(),
                outputs=['prediction', 'explanation'],
            )
    assert '512' in str(caught.value)


def test_only_this_model_is_asked_to_explain_a_forecast(anomaly):
    """`explanation` is a field the forecaster produces, not every model."""
    from sdm_client import KumoAnomaly

    frame = pd.DataFrame({'flow': np.sin(np.arange(60) / 9)})
    with SDMClient.in_process(anomaly) as client:
        with pytest.raises(StructuredDataError) as caught:
            client.predict(
                Request.anomaly(frame, inference={'num_samples': 1}),
                model=KumoAnomaly(),
                outputs=['prediction', 'explanation'],
            )
    assert 'explanation' in str(caught.value)


def test_the_smoothness_of_the_latent_state_is_measured(explained):
    """Reported as null for every run before it was computed, so its page
    was never drawn."""
    block = explained.explanation['explanation']['trajectory_stability']

    assert block is not None
    assert block['n_time_steps'] == 512
    assert 0.0 <= block['zero_crossing_rate_mean'] <= 1.0


def test_the_smoothness_is_the_history_walked_once(explained, forecast):
    """It is read off the trajectory the attribution already walked rather
    than walked again, so it has to be what a second walk would say."""
    import torch

    from sdm_client.explain._attribution.interpretability import (
        compute_trajectory_stability,
    )
    from sdm_client.explain._explain import _standardised
    from sdm_client.explain._model import Forecaster

    carried = explained.explanation
    window = _standardised(
        forecast, history().to_numpy().T[:, -512:].astype('float32')
    )
    walked = compute_trajectory_stability(
        Forecaster(forecast),
        np.ascontiguousarray(window, dtype='float32'),
        seq_len=512,
        input_mask_t=np.ones(512, dtype='float32'),
        device=torch.device('cpu'),
    )

    read = carried['explanation']['trajectory_stability']
    assert read['relative_jitter_mean'] == pytest.approx(
        walked.relative_jitter_mean, rel=1e-5
    )
    assert read['zero_crossing_rate_mean'] == pytest.approx(
        walked.zero_crossing_rate_mean, rel=1e-5
    )


def test_the_smoothness_page_is_drawn(explained, tmp_path):
    before = pages(forecast_report(explained, tmp_path / 'f.pdf'))
    stripped = without(explained, 'trajectory_stability')

    after = pages(forecast_report(stripped, tmp_path / 'g.pdf'))

    assert before == after + 1


@pytest.fixture(scope='module')
def over_channels(forecast):
    with SDMClient.in_process(forecast) as client:
        return client.predict(
            Request.forecast(
                history(),
                target='flow',
                horizon=4,
                inference={'feature_axis': True},
            ),
            model=KumoForecast(),
            outputs=['prediction', 'explanation'],
        )


def test_the_embedding_is_measured_for_every_channel(over_channels):
    """Asked for alongside the feature axis, because it answers the question
    the feature axis raises: an influence attributed through a derivative
    that is not there is not an influence."""
    block = over_channels.explanation['explanation']['feature_axis']
    measured = block['embedding_stability']

    assert measured['channel_labels'] == ['flow', 'temp']
    assert all(count > 0 for count in measured['n_trials_per_channel'])
    assert all(np.isfinite(measured['lip_ratio_mean']))


def test_the_embedding_is_not_measured_without_the_feature_axis(explained):
    """It costs a window per channel per trial, which is the feature axis's
    own order of cost and not the lag attribution's."""
    block = explained.explanation['explanation'].get('feature_axis') or {}

    assert 'embedding_stability' not in block


def test_the_embedding_stability_page_is_drawn(over_channels, tmp_path):
    before = pages(forecast_report(over_channels, tmp_path / 'f.pdf'))
    stripped = without(
        over_channels, 'embedding_stability', under='feature_axis'
    )

    after = pages(forecast_report(stripped, tmp_path / 'g.pdf'))

    assert before == after + 1


def test_the_stability_csv_names_the_channels_it_measured(
    over_channels, tmp_path
):
    """The channels it measured include the series being forecast, which the
    channel attribution beside it sets aside. Drawn under the attribution's
    labels the two lists were different lengths, and the page numbered the
    channels instead of naming them."""
    import pandas as pd

    artifacts = tmp_path / 'artifacts'
    forecast_report(over_channels, tmp_path / 'f.pdf', artifacts_dir=artifacts)
    written = pd.read_csv(artifacts / 'feature_axis_embedding_stability.csv')

    assert written['feature'].tolist() == ['flow', 'temp']


@pytest.mark.parametrize('reference', [0.0, 1.5, None])
def test_a_reference_scale_of_zero_is_a_reference_scale(reference):
    """A trajectory that does not move has a mean step of zero, which read
    as a measurement that was never taken."""
    from sdm_client.reporting.forecast import _stability

    report, labels = _stability(
        {
            'explanation': {
                'feature_axis': {
                    'embedding_stability': {
                        'channel_labels': ['flow', 'temp'],
                        'lip_ratio_mean': [1.0, 2.0],
                        'lip_ratio_max': [1.0, 2.0],
                        'lip_ratio_p50': [1.0, 2.0],
                        'lip_ratio_p95': [1.0, 2.0],
                        'n_trials_per_channel': [3, 3],
                        'n_unique_windows': 4,
                        'step_delta_norm_mean': reference,
                    }
                }
            }
        }
    )

    assert labels == ['flow', 'temp']
    read = report.step_delta_norm_mean
    assert np.isnan(read) if reference is None else read == reference


def test_too_many_channels_leave_the_page_out_rather_than_the_rest(forecast):
    """The probes are a window per channel per trial and a copy of each, so
    their memory grows with the square of the channels. Past the cap the
    page is left out and the feature axis it was asked beside is not."""
    import warnings

    from sdm_client.core.warnings import SDMWarning
    from sdm_client.explain import _explain

    frame = history()
    with SDMClient.in_process(forecast) as client:
        request = Request.forecast(
            frame,
            target='flow',
            horizon=4,
            inference={'feature_axis': True},
        )
        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter('always')
            kept = _explain._STABILITY_BYTES
            _explain._STABILITY_BYTES = 1
            try:
                result = client.predict(
                    request,
                    model=KumoForecast(),
                    outputs=['prediction', 'explanation'],
                )
            finally:
                _explain._STABILITY_BYTES = kept

    block = result.explanation['explanation']['feature_axis']
    assert 'embedding_stability' not in block
    assert block['channel_horizon_attributions'] is not None
    assert any(
        issubclass(one.category, SDMWarning)
        and 'MiB of probes' in str(one.message)
        for one in raised
    )


@pytest.fixture(scope='module')
def by_gradient(forecast):
    with SDMClient.in_process(forecast) as client:
        return client.predict(
            Request.forecast(
                history(),
                target='flow',
                horizon=4,
                inference={'integrated_gradients': True},
            ),
            model=KumoForecast(),
            outputs=['prediction', 'explanation'],
        )


def test_the_gradient_attribution_covers_the_window_it_read(by_gradient):
    block = by_gradient.explanation['explanation']['integrated_gradients']

    assert np.asarray(block['attribution']).shape == (2, 512)
    assert block['channel_labels'] == ['flow', 'temp']
    assert sum(block['channel_abs_share']) == pytest.approx(1.0, rel=1e-5)


@pytest.mark.parametrize(
    ('differentiable', 'reads'),
    [(False, False), (True, True)],
)
def test_the_gradient_is_taken_through_the_normalisation(
    sdm, differentiable, reads
):
    """Each window is divided by statistics taken over the window itself,
    and those are taken without gradients. A gradient through the model as it
    stands therefore describes a pass with each window's own mean and spread
    held still, which is not the pass that answered.

    Measured against an invariant rather than against the model, so it holds
    whatever the weights are: a normalised window has the same sum of squares
    whatever it was made from, so moving the window cannot change it and the
    true derivative is exactly zero.
    """
    import torch
    from sdm.nn import RevIN

    from sdm_client.explain._model import _statistics

    torch.manual_seed(0)
    revin = RevIN(eps=1e-5)
    window = torch.randn(1, 3, 16, dtype=torch.float32)
    direction = torch.randn_like(window)
    direction /= direction.norm()

    moved = window.clone().requires_grad_(True)
    statistics = _statistics if differentiable else RevIN.stats
    mean, scale = statistics(moved, correction=0)
    revin(moved, mean, scale).square().sum().backward()
    along = float((moved.grad * direction).sum())

    assert (abs(along) < 1e-4) is reads


def test_the_normalisation_is_left_as_it_was_found(forecast):
    """It is patched for the length of one attribution. Left patched, every
    forecast after it runs through a normaliser this client replaced."""
    from sdm_client.explain._model import Forecaster

    shim = Forecaster(forecast)
    with shim.differentiable_normalisation():
        patched = [
            'stats' in vars(module)
            for module in shim.modules()
            if hasattr(module, 'eps') and hasattr(module, 'stats')
        ]

    assert any(patched)
    assert not any(
        'stats' in vars(module)
        for module in shim.modules()
        if hasattr(module, 'eps') and hasattr(module, 'stats')
    )


def test_the_normalisation_survives_being_held_twice(forecast):
    """The inner one deleted the outer one's patch on its way out, and the
    outer one then raised on a name that was no longer there."""
    from sdm_client.explain._model import Forecaster

    shim = Forecaster(forecast)
    revins = [
        module
        for module in shim.modules()
        if hasattr(module, 'eps') and hasattr(module, 'stats')
    ]
    with shim.differentiable_normalisation():
        with shim.differentiable_normalisation():
            pass
        inner = [vars(module).get('stats') for module in revins]

    assert all(held is not None for held in inner)
    assert not any('stats' in vars(module) for module in revins)


def test_a_normalisation_of_its_own_is_put_back(forecast):
    """A module that carried its own statistics before any of this lost
    them outright: the patch was deleted rather than replaced."""
    from sdm_client.explain._model import Forecaster

    shim = Forecaster(forecast)
    revin = next(
        module
        for module in shim.modules()
        if hasattr(module, 'eps') and hasattr(module, 'stats')
    )
    theirs = revin.stats
    revin.stats = theirs

    with shim.differentiable_normalisation():
        pass

    assert vars(revin).get('stats') is theirs


def test_the_length_of_the_embedding_is_not_what_is_explained(forecast):
    """This forecaster's embedding is a retrieval key, scaled to unit length
    before it is returned, so its length is one whatever it was given.
    Explaining that explains a constant, and every reading came back zero."""
    import torch

    from sdm_client.explain._explain import _standardised
    from sdm_client.explain._model import Forecaster

    shim = Forecaster(forecast)
    window = _standardised(
        forecast, history().to_numpy().T[:, -512:].astype('float32')
    )
    embedded = shim.embed(
        x_enc=torch.as_tensor(window, dtype=torch.float32)[None],
        input_mask=torch.ones(1, 512),
    ).embeddings

    assert float(embedded.norm()) == pytest.approx(1.0, rel=1e-5)


def test_a_forecast_asked_without_gradients_does_not_pay_for_them(explained):
    """They walk a path of sixty-four windows and take a gradient at each."""
    assert 'integrated_gradients' not in explained.explanation['explanation']


def test_the_gradient_page_is_drawn(by_gradient, tmp_path):
    before = pages(forecast_report(by_gradient, tmp_path / 'f.pdf'))
    stripped = without(by_gradient, 'integrated_gradients')

    after = pages(forecast_report(stripped, tmp_path / 'g.pdf'))

    assert before == after + 1


def test_the_files_the_gradient_page_names_are_written(by_gradient, tmp_path):
    artifacts = tmp_path / 'artifacts'
    forecast_report(by_gradient, tmp_path / 'f.pdf', artifacts_dir=artifacts)
    written = {path.name for path in artifacts.iterdir()}

    assert 'integrated_gradients_attributions.csv' in written
    assert 'integrated_gradients_channel_summary.csv' in written
