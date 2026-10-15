"""Adversarial cases for observation-based view/privacy kernels."""
import json

import pytest

from attacker.attacks import view_privacy as vp
from attacker.evidence import verify_evidence


def test_mixup_nonzero_decoy_mean_requires_calibration():
    data = dict(releases=[[[2.0]]*16], truth=[[1.0]], calibration_mean=[5.0], **{"lambda": .75})
    rows = vp.evaluate_mixup(data)
    calibrated = [r for r in rows if r["arm"] == "calibrated_mean"]
    naive = [r for r in rows if r["arm"] == "naive_zero_mean"]
    assert all(r["mse"] == 0 for r in calibrated)
    assert all(r["mse"] > 2 for r in naive)
    assert calibrated[-1]["conditional_mean_variance"] == 0


def test_unknown_mean_is_nonidentifiable_and_lambda_sensitive():
    releases = [[[2.0]]*8]
    assert vp.demix(releases, .75, [5.0]) == [[1.0]]
    assert vp.demix(releases, .75, [2.0]) == [[2.0]]
    assert vp.demix(releases, .5, [5.0]) != [[1.0]]


def test_noise_sampling_uncertainty_computed():
    data = dict(releases=[[[0.0], [2.0]]], truth=[[1.0]], calibration_mean=[0.0], **{"lambda": 1.0})
    row = next(r for r in vp.evaluate_mixup(data) if r["arm"] == "calibrated_mean" and r["releases"] == 2)
    assert row["mse"] == 0
    assert row["conditional_mean_variance"] == 1


def test_single_fragment_leaks_property_union_reconstructs():
    data = dict(train=[[-2, -1], [-1, 1], [1, -1], [2, 1]], train_labels=[0, 0, 1, 1],
                evaluation=[[-1.5, 3], [1.5, -3]], eval_labels=[0, 1], views=[[0], [1]])
    rows = vp.evaluate_views(data)
    assert rows[1]["property_accuracy"] == 1
    assert rows[1]["reconstruction_mse"] > 0
    assert rows[-1]["reconstruction_mse"] == 0
    assert rows[0]["property_accuracy"] == .5


def test_overlapping_complementary_views_do_not_invent_threshold():
    data = dict(train=[[0, 0, 0], [1, 1, 1]], train_labels=[0, 1],
                evaluation=[[3, 4, 5]], eval_labels=[1], views=[[0], [0, 1]])
    union = vp.evaluate_views(data)[-1]
    assert union["observed_fraction"] == 2/3
    assert union["reconstruction_mse"] > 0


def test_epoch_reset_prevents_cross_key_cancellation():
    data = dict(timestamps=[0, 3, 6, 9], epochs=[0, 0, 1, 1], releases=[[10], [10], [-10], [-10]],
                truth_by_epoch={"0": [10], "1": [-10]}, window=10, max_count=4)
    rows = vp.evaluate_tripwire(data)
    assert all(row["accepted"] == 2 and row["accumulated_mse"] == 0 for row in rows)
    assert all(row["pooling_scope"] == "same_epoch_only" for row in rows)


def test_low_slow_satisfies_sliding_budget_but_accumulates():
    times = [i*2.6 for i in range(100)]
    assert all(vp.monitor_acceptance(times, 10, 4, "sliding"))
    assert sum(vp.monitor_acceptance([i*.1 for i in range(100)], 10, 4, "sliding")) == 4


def test_window_boundary_distinguishes_monitors():
    times = [9.8, 9.9, 10.0, 10.1]
    assert vp.monitor_acceptance(times, 10, 2, "fixed") == [True]*4
    assert vp.monitor_acceptance(times, 10, 2, "sliding") == [True, True, False, False]


@pytest.mark.parametrize("mutation", [
    lambda b: b.update(extra=True),
    lambda b: b.update(schema="wrong"),
    lambda b: b["data"].update(unknown=True),
    lambda b: b["data"].update(calibration_mean=[float("nan")]),
    lambda b: b["data"].update(releases=[[[1.0, 2.0]]]),
    lambda b: b["data"].update(**{"lambda": 0}),
    lambda b: b["data"].update(assumed_lambdas=[True]),
    lambda b: b["data"].update(clean_outputs=[[1.0]]),
])
def test_strict_mixup_schema(mutation):
    bundle = dict(schema=vp.SCHEMA_VERSION, case="mixup", data=dict(releases=[[[1.0]]], truth=[[1.0]], calibration_mean=[0.0], **{"lambda": .75}))
    mutation(bundle)
    with pytest.raises(ValueError):
        vp.evaluate_bundle(bundle)


def test_synthetic_runs_are_reproducible_and_nonlinearity_measured():
    bundles = list(vp.synthetic_bundles(7, True))
    assert bundles == list(vp.synthetic_bundles(7, True))
    rows = vp.evaluate_bundle(bundles[0])
    assert any(r["arm"] == "nonlinear_corrected_output" and r["mse"] > 0 for r in rows)


def test_cli_supplied_arrays_and_evidence(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(next(vp.synthetic_bundles(7, True))))
    output = tmp_path / "result.json"
    args = vp.build_parser().parse_args(["--bundle", str(source), "--output", str(output)])
    assert vp.run(args) == 0
    assert verify_evidence(str(output) + ".evidence.json")[0]
    result = json.loads(output.read_text())
    assert result["source"]["kind"] == "supplied_arrays"
    assert result["results"]


def test_malformed_coordinate_and_epoch_inputs_fail_closed():
    bundles = list(vp.synthetic_bundles(7, True))
    view = bundles[1]
    view["data"]["views"] = [[[0]]]
    with pytest.raises(ValueError):
        vp.evaluate_bundle(view)
    tripwire = bundles[3]
    tripwire["data"]["epochs"][0] = 1
    with pytest.raises(ValueError):
        vp.evaluate_bundle(tripwire)


def test_preserves_existing_evidence(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("preserve")
    args = vp.build_parser().parse_args(["--toy", "--quick", "--output", str(output)])
    with pytest.raises(ValueError, match="fresh"):
        vp.run(args)
    assert output.read_text() == "preserve"


def test_centroid_prediction_does_not_consume_eval_labels():
    train = [[-1.0], [1.0]]
    assert vp.centroid_predictions(train, [0, 1], [[.5]], [0]) == [1]
    assert vp.centroid_predictions(train, [1, 0], [[.5]], [0]) == [0]


def test_three_complementary_views_have_measured_two_coalition_coverage():
    bundle = list(vp.synthetic_bundles(7, True))[2]
    rows = vp.evaluate_bundle(bundle)
    singles = [r for r in rows if len(r["coalition"]) == 1]
    pairs = [r for r in rows if len(r["coalition"]) == 2]
    assert len(singles) == len(pairs) == 3
    assert all(0 < r["observed_fraction"] < 1 and r["reconstruction_mse"] > 0 for r in singles)
    assert all(r["observed_fraction"] == 1 and r["reconstruction_mse"] == 0 for r in pairs)
    assert all(r["num_views"] == 3 and r["minimum_coalition_full_coordinate_coverage"] == 2 for r in rows)
    assert all(r["layout"] == "overlapping_coordinate_views" for r in rows)


def test_finite_calibration_and_oracle_are_separate():
    bundle = next(vp.synthetic_bundles(7, True))
    data = bundle["data"]
    assert data["calibration_mean"] != data["oracle_pool_mean"]
    rows = vp.evaluate_bundle(bundle)
    calibrated = [r for r in rows if r["arm"] == "calibrated_mean"]
    oracle = [r for r in rows if r["arm"] == "oracle_pool_mean"]
    assert calibrated and oracle
    assert all(r["calibration_source"] == "finite_auxiliary_samples" and r["calibration_sample_count"] == 64 for r in calibrated)
    assert all(r["calibration_source"] == "oracle_entire_pool" for r in oracle)
    data["calibration_samples"][0][0] += 1
    with pytest.raises(ValueError, match="match calibration_mean"):
        vp.evaluate_bundle(bundle)
