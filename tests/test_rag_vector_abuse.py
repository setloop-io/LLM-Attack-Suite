"""rag-vector-abuse: arm coverage, defended/vulnerable contrast, negative
control, store-dump errors, determinism, artifact layout."""

import json

import pytest

from attacker import threat_model
from attacker.attacks import rag_vector_abuse as rv


def _run(tmp_path, extra=()):
    out = tmp_path / "rv.json"
    args = rv.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(out), *extra])
    args.mode = "inference"
    assert rv.run(args) == 0
    return out, json.loads(out.read_text())


def test_contract_members():
    assert rv.EXPERIMENT_ID == "rag_vector_abuse"
    assert rv.MODES == ("inference",)
    assert rv.REQUIRES_LABELS is False
    assert callable(rv.build_parser) and callable(rv.run)


def test_toy_artifact_layout_and_journal(tmp_path):
    out, art = _run(tmp_path)
    assert art["schema"] == rv.ARTIFACT_SCHEMA
    threat_model.validate(art["threat_model"])
    assert art["config"]["attack"] == rv.EXPERIMENT_ID
    lines = (tmp_path / "rv.json.jsonl").read_text().strip().splitlines()
    assert len(lines) == len(art["results"])
    assert all(json.loads(l)["experiment"] == rv.EXPERIMENT_ID
               for l in lines)


def test_every_arm_recorded_in_both_worlds(tmp_path):
    _, art = _run(tmp_path)
    for arm in rv.ARMS:
        worlds = {r["world"] for r in art["results"] if r["arm"] == arm}
        assert worlds == {"vulnerable", "defended"}, arm


def test_defended_zero_hits_vulnerable_nonzero(tmp_path):
    _, art = _run(tmp_path)
    s = art["summary"][0]
    assert s["vulnerable_total_hits"] > 0
    assert s["defended_total_hits"] == 0
    assert s["control_false_flags"] == 0
    assert s["control_stale_hits_defended"] == 0


def test_negative_control_cell(tmp_path):
    _, art = _run(tmp_path)
    ctrl = [r for r in art["results"] if r["arm"] == "negative-control"]
    assert len(ctrl) == 1 and ctrl[0]["false_flags"] == 0


def test_membership_metrics_shape(tmp_path):
    _, art = _run(tmp_path)
    mb = [r for r in art["results"] if r["arm"] == "membership-linkability"
          and r["world"] == "vulnerable"]
    assert mb and 0.0 <= mb[0]["membership_auc"] <= 1.0
    assert 0.0 <= mb[0]["tpr_at_1pct_fpr"] <= 1.0


def test_inversion_reports_precision_recall_vs_baseline(tmp_path):
    _, art = _run(tmp_path)
    inv = [r for r in art["results"] if r["arm"] == "embedding-inversion"]
    assert inv and "token_precision" in inv[0]
    assert "majority_baseline" in inv[0]


def test_store_dump_validation(tmp_path):
    good = tmp_path / "s.jsonl"
    good.write_text(json.dumps({"id": "d1", "text": "hello world",
                                "metadata": {"tenant": "t1"}}) + "\n")
    out, art = _run(tmp_path, ["--store", str(good)])
    audit = [r for r in art["results"] if r["arm"] == "store-audit"]
    assert audit and audit[0]["documents"] == 1
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x"}\n')
    args = rv.build_parser().parse_args(
        ["--toy", "--quick", "--store", str(bad)])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="bad.jsonl"):
        rv.run(args)
    with pytest.raises(SystemExit, match="missing.jsonl"):
        args = rv.build_parser().parse_args(
            ["--toy", "--quick", "--store", str(tmp_path / "missing.jsonl")])
        args.mode = "inference"
        rv.run(args)


def test_determinism(tmp_path):
    _, a = _run(tmp_path)
    out2 = tmp_path / "rv2.json"
    args = rv.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(out2)])
    args.mode = "inference"
    rv.run(args)
    assert a["results"] == json.loads(out2.read_text())["results"]


def test_self_test_passes():
    assert rv.self_test() == 0
