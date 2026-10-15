import copy

import pytest

from attacker.attacks import representation_privacy as rp


def metric(report, family, target="membership", control="fitted"):
    return next(r for r in report["results"] if r["family"] == family and r["target"] == target and r["control"] == control)


def test_coordinate_signal_and_null():
    positive = rp.probe_audit(rp.make_fixture())
    null = rp.probe_audit(rp.make_fixture(signal=False))
    assert metric(positive, "coordinate")["auc"] > .95
    assert .3 < metric(null, "coordinate")["auc"] < .7
    assert .25 < metric(positive, "coordinate", control="shuffled_calibration_labels")["auc"] < .75


def test_flattened_gram_can_hide_norm_but_keep_coordinates():
    result = rp.probe_audit(rp.make_fixture(flattened=True))
    assert metric(result, "coordinate")["auc"] > .95
    assert .35 < metric(result, "gram")["auc"] < .65
    assert .35 < metric(result, "norm")["auc"] < .65


@pytest.mark.parametrize("mutation", ["overlap", "nan", "dimension", "missing_provenance"])
def test_invalid_contract(mutation):
    bundle = rp.make_fixture()
    if mutation == "overlap":
        bundle["evaluation"][0]["document_id"] = bundle["calibration"][0]["document_id"]
    elif mutation == "nan":
        bundle["evaluation"][0]["released"][0][0] = float("nan")
    elif mutation == "dimension":
        bundle["evaluation"][0]["released"][0].pop()
    else:
        del bundle["provenance"]
    with pytest.raises(ValueError):
        rp.probe_audit(bundle)


@pytest.mark.parametrize("nonlinear", [False, True])
def test_truth_never_changes_predictions(nonlinear):
    if nonlinear:
        pytest.importorskip("torch")
    bundle = rp.make_fixture()
    first = rp.probe_audit(bundle, nonlinear=nonlinear)
    for row in bundle["evaluation"]:
        row["label"] = 1 - row["label"]
        row["membership"] = 1 - row["membership"]
    second = rp.probe_audit(bundle, nonlinear=nonlinear)
    assert [r["predictions"] for r in first["results"]] == [r["predictions"] for r in second["results"]]


def test_phase_signal_null_and_matched_strata():
    assert all(r["auc"] == 1 for r in rp.phase_audit(rp.make_fixture(), "training")["results"])
    assert all(r["auc"] == .5 for r in rp.phase_audit(rp.make_fixture(signal=False), "training")["results"])


@pytest.mark.parametrize("mutation", ["provenance", "references", "exposure", "mode", "overlap"])
def test_phase_fail_closed(mutation):
    bundle = rp.make_fixture()
    mode = "training"
    if mutation == "provenance":
        del bundle["phase_provenance"]
    elif mutation == "references":
        del bundle["phase_records"][0]["ref_in"]
    elif mutation == "exposure":
        bundle["phase_records"] = [r for r in bundle["phase_records"] if r["phase"] != "public_only"]
    elif mutation == "mode":
        mode = "inference"
    else:
        bundle["phase_records"][0]["document_id"] = bundle["calibration"][0]["document_id"]
    with pytest.raises(ValueError):
        rp.phase_audit(bundle, mode)


def test_phase_truth_does_not_fit_reference_models():
    bundle = rp.make_fixture()
    changed = copy.deepcopy(bundle)
    for row in changed["phase_records"]:
        row["phase"] = ("public_only" if row["public_exposed"] else "never_seen") if row["phase"] == "private_training_member" else "private_training_member"
    assert all(r["auc"] == 0 for r in rp.phase_audit(changed, "training")["results"])


def test_cli_real_bundle_and_sealed_evidence(tmp_path):
    import json
    bundle = tmp_path / "releases.json"
    bundle.write_text(json.dumps(rp.make_fixture(flattened=True)))
    output = tmp_path / "audit.json"
    args = rp.build_parser().parse_args(["--bundle", str(bundle), "--output", str(output), "--evidence", "--arms", "gram_flattening", "private_phase"])
    args.mode = "training"
    assert rp.run(args) == 0
    result = json.loads(output.read_text())
    assert len(result["input_sha256"]) == 64
    assert result["results"]["support"].startswith("released-feature adapter")
    seal = json.loads((tmp_path / "audit.json.evidence.json").read_text())
    from attacker.evidence import file_sha256
    assert seal["artifact"]["sha256"] == file_sha256(output)


def test_multitoken_gram_shape_and_rotation_invariance():
    import numpy as np
    rng = np.random.default_rng(44)
    x = rng.normal(size=(3, 64))
    q, _ = np.linalg.qr(rng.normal(size=(64, 64)))
    original = rp.features([{"released": x.tolist()}], "gram")
    rotated = rp.features([{"released": (x @ q).tolist()}], "gram")
    assert original.shape == (1, 6)
    np.testing.assert_allclose(original, rotated, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(original[0], (x @ x.T)[np.triu_indices(3)])


def test_nonlinear_decoder_recovers_xor_where_ridge_fails():
    pytest.importorskip("torch")
    import numpy as np
    x = np.zeros((80, 64))
    x[:, :2] = np.tile([[-1, -1], [-1, 1], [1, -1], [1, 1]], (20, 1))
    y = (x[:, 0] != x[:, 1]).astype(int)
    linear, _, _ = rp.fit_predict(x, y, x)
    pred, score, _ = rp.mlp_fit_predict(x, y, x, seed=17)
    assert np.mean(linear == y) == .5
    assert np.mean(pred == y) == 1
    repeated, repeated_score, _ = rp.mlp_fit_predict(x, y, x, seed=17)
    np.testing.assert_array_equal(pred, repeated)
    np.testing.assert_array_equal(score, repeated_score)


def test_numpy_optional_for_parser_and_explicit_numeric_error(monkeypatch):
    monkeypatch.setattr(rp, "np", None)
    assert rp.build_parser().parse_args(["--toy"]).toy
    with pytest.raises(RuntimeError, match="requires numpy"):
        rp.make_fixture()


def test_output_is_fresh_and_parent_created(tmp_path):
    output = tmp_path / "nested" / "audit.json"
    args = rp.build_parser().parse_args(["--toy", "--output", str(output), "--evidence"])
    assert rp.run(args) == 0
    before = output.read_bytes()
    with pytest.raises(ValueError, match="fresh"):
        rp.run(args)
    assert output.read_bytes() == before


def test_real_label_only_bundle_needs_no_membership(tmp_path):
    import json
    bundle = rp.make_fixture()
    for part in ("calibration", "evaluation"):
        for row in bundle[part]:
            del row["membership"]
    path = tmp_path / "label-only.json"
    path.write_text(json.dumps(bundle))
    output = tmp_path / "audit.json"
    args = rp.build_parser().parse_args(["--bundle", str(path), "--output", str(output)])
    assert rp.run(args) == 0
    result = json.loads(output.read_text())["results"]["arms"]["bottleneck"]
    assert len(result["results"]) == 6
    assert {r["target"] for r in result["results"]} == {"label"}
    assert metric(result, "coordinate", target="label")["accuracy"] > .95


@pytest.mark.parametrize("partition_only", [False, True])
def test_mixed_membership_presence_rejected(partition_only):
    bundle = rp.make_fixture()
    rows = bundle["evaluation"] if partition_only else bundle["evaluation"][:1]
    for row in rows:
        del row["membership"]
    with pytest.raises(ValueError, match="all rows"):
        rp.probe_audit(bundle)


def test_shadow_toy_flag_parsing_and_torch_gate(monkeypatch):
    assert rp.build_parser().parse_args(["--shadow-toy"]).shadow_toy
    for argv in ([], ["--toy", "--shadow-toy"], ["--toy", "--bundle", "x.json"]):
        args = rp.build_parser().parse_args(argv)
        with pytest.raises(ValueError, match="exactly one of"):
            rp.run(args)
    from attacker.attacks import common
    monkeypatch.setattr(common, "torch", None)
    with pytest.raises(SystemExit, match="requires torch"):
        rp.run(rp.build_parser().parse_args(["--shadow-toy"]))


def test_module_self_test_passes():
    assert rp.self_test() == 0


def test_phase_only_bundle_needs_no_released_representation(tmp_path):
    import json
    fixture = rp.make_fixture()
    bundle = {key: fixture[key] for key in ("phase_provenance", "phase_records")}
    path = tmp_path / "phase-only.json"
    path.write_text(json.dumps(bundle))
    output = tmp_path / "audit.json"
    args = rp.build_parser().parse_args(["--bundle", str(path), "--output", str(output), "--arms", "private_phase"])
    args.mode = "training"
    assert rp.run(args) == 0
    result = json.loads(output.read_text())["results"]["arms"]["private_phase"]
    assert all(row["auc"] == 1 for row in result["results"])
