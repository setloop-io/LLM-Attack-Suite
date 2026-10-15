"""lira-membership: LIRA-style scoring core, offline/toy modes, gate
verdicts, and registry dispatch via attacks.register_attack (the parent
wires the built-in registry entry separately). Torch-free throughout."""

import json

import pytest

from attacker import attacks
from attacker.attacks import lira_membership as lm


def parse(*argv):
    args = lm.build_parser().parse_args(list(argv))
    args.mode = "training"
    return args


@pytest.fixture
def cleanup_registry():
    before = set(attacks.ATTACKS)
    yield
    for name in set(attacks.ATTACKS) - before:
        attacks.ATTACKS.pop(name, None)


def write_jsonl(path, records):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return str(path)


def separable_records(n=60, shift=3.0, seed=0):
    import random
    rng = random.Random(seed)
    recs = []
    for split in ("in", "out"):
        for i in range(n):
            s = rng.gauss(0, 1) - (shift if split == "in" else 0.0)
            recs.append({"example_id": f"{split}-{i}", "split": split,
                         "statistic": s})
    return recs


def test_contract_and_dispatch_via_register_attack(cleanup_registry):
    attacks.validate_contract(lm, "lira-membership")
    assert lm.MODES == ("training",)
    assert lm.REQUIRES_LABELS is False
    if "lira-membership" not in attacks.ATTACKS:
        attacks.register_attack("lira-membership", lm)
    assert attacks.load("lira-membership") is lm
    assert "lira-membership" in attacks.for_mode("training")
    assert "lira-membership" not in attacks.for_mode("inference")


def test_parser_builds_torchless():
    args = lm.build_parser().parse_args([])
    assert args.statistic == "raw" and args.worlds == "both"
    assert args.toy_n == 100000 and args.toy_shift == 2.0
    assert args.online is False and args.evidence is False
    assert args.n_shadows == 4 and args.shadow_steps == 200


def test_log_gauss_matches_normaldist():
    import math
    from statistics import NormalDist
    nd = NormalDist(mu=0.3, sigma=1.7)
    for x in (-2.0, 0.0, 0.3, 5.0):
        assert math.exp(lm._log_gauss(x, 0.3, 1.7 ** 2)) == \
            pytest.approx(nd.pdf(x))
    assert math.isfinite(lm._log_gauss(1.0, 0.0, 0.0))


def test_rank_auc_fixtures():
    assert lm.rank_auc([2.0, 3.0], [0.0, 1.0]) == 1.0
    assert lm.rank_auc([0.0, 1.0], [2.0, 3.0]) == 0.0
    assert lm.rank_auc([1.0, 1.0], [1.0, 1.0]) == 0.5
    assert 0.0 < lm.rank_auc([0.5, 2.5], [1.0, 2.0]) < 1.0


def test_tpr_at_fpr_conservative_grid():
    r = lm.tpr_at_fpr([3.0, 2.0, 1.5], [2.5, 1.6, 0.5], 1.0 / 3.0)
    assert r["tpr"] == pytest.approx(2.0 / 3.0)
    assert r["fpr"] == pytest.approx(1.0 / 3.0)
    assert r["resolvable"] and r["min_fpr_step"] == pytest.approx(1.0 / 3.0)
    r = lm.tpr_at_fpr([3.0, 2.0], [1.0, 0.0], 0.001)
    assert r["resolvable"] is False
    assert r["tpr"] == 1.0 and r["fpr"] == 0.0


def test_loo_excludes_the_scored_record():
    recs = [{"example_id": f"i{i}", "split": "in", "statistic": v}
            for i, v in enumerate((1.0, 2.0, 3.0))] + \
           [{"example_id": f"o{i}", "split": "out", "statistic": v}
            for i, v in enumerate((4.0, 5.0, 6.0))]
    scored = lm.score_records(recs)
    assert scored[0]["model"] == "global_loo"
    assert scored[0]["mu_in"] == 2.5 and scored[0]["mu_out"] == 5.0
    assert scored[3]["mu_in"] == 2.0 and scored[3]["mu_out"] == 5.5


def test_per_example_reference_lists_switch_model():
    recs = [{"example_id": "x", "split": "in", "statistic": 0.0,
             "ref_in": [0.0, 0.1, -0.1], "ref_out": [2.0, 2.1, 1.9]}]
    recs += separable_records(10)
    scored = lm.score_records(recs)
    assert scored[0]["model"] == "per_example"
    assert scored[0]["mu_in"] == pytest.approx(0.0)
    assert scored[0]["mu_out"] == pytest.approx(2.0)
    assert all(r["model"] == "global_loo" for r in scored[1:])


def test_statistic_transform():
    import math
    assert lm._transform(0.5, "logit") == 0.0
    assert lm._transform(0.75, "logit") == pytest.approx(math.log(3.0))
    assert math.isfinite(lm._transform(0.0, "logit"))
    assert math.isfinite(lm._transform(1.0, "logit"))
    assert lm._transform(-3.2, "raw") == -3.2


def test_score_records_requires_two_per_split():
    with pytest.raises(SystemExit, match=">=2 IN and >=2 OUT"):
        lm.score_records([{"example_id": "a", "split": "in",
                           "statistic": 1.0}])


def test_load_records_validation(tmp_path):
    good = write_jsonl(tmp_path / "good.jsonl", separable_records(5))
    assert len(lm.load_records([good])) == 10
    bad_split = write_jsonl(tmp_path / "bad.jsonl", [
        {"example_id": "x", "split": "maybe", "statistic": 1.0}])
    with pytest.raises(SystemExit, match="split must be"):
        lm.load_records([bad_split])
    missing = write_jsonl(tmp_path / "missing.jsonl", [
        {"example_id": "x", "split": "in"}])
    with pytest.raises(SystemExit, match="missing field 'statistic'"):
        lm.load_records([missing])
    bad_stat = write_jsonl(tmp_path / "badstat.jsonl", [
        {"example_id": "x", "split": "in", "statistic": "high"}])
    with pytest.raises(SystemExit, match="statistic must be a number"):
        lm.load_records([bad_stat])
    bad_ref = write_jsonl(tmp_path / "badref.jsonl", [
        {"example_id": "x", "split": "in", "statistic": 1.0,
         "ref_in": [1.0]}])
    with pytest.raises(SystemExit, match="ref_in must be a list"):
        lm.load_records([bad_ref])


def test_offline_mode_separable_and_null(tmp_path):
    out = str(tmp_path / "off.json")
    path = write_jsonl(tmp_path / "recs.jsonl", separable_records())
    assert lm.run(parse("--records", path, "--output", out)) == 0
    artifact = json.load(open(out))
    rec = artifact["results"][0]
    assert rec["world"] == "offline"
    assert rec["auc"] > 0.85
    assert rec["gate"]["verdict"] == "FAIL"
    assert set(rec["tpr_at_fpr"]) == {"0.01", "0.001"}
    assert len(open(out + ".jsonl").read().splitlines()) == 1

    null_path = write_jsonl(tmp_path / "null.jsonl",
                            separable_records(shift=0.0, seed=7))
    null_out = str(tmp_path / "null.json")
    assert lm.run(parse("--records", null_path, "--output", null_out)) == 0
    null_rec = json.load(open(null_out))["results"][0]
    assert abs(null_rec["auc"] - 0.5) < 0.15


def test_toy_run_both_worlds_quick(tmp_path):
    out = str(tmp_path / "toy.json")
    assert lm.run(parse("--toy", "--quick", "--output", out)) == 0
    artifact = json.load(open(out))
    assert sorted(r["world"] for r in artifact["results"]) == \
        ["null", "separable"]
    for rec in artifact["results"]:
        for key in ("auc", "tpr_at_fpr", "best_balanced_accuracy",
                    "gate", "n_in", "n_out"):
            assert key in rec
        assert rec["gate"]["verdict"] in ("PASS", "FAIL", "INSUFFICIENT_EVIDENCE")
    null = [r for r in artifact["results"] if r["world"] == "null"][0]
    assert abs(null["auc"] - 0.5) < 0.1
    sep = [r for r in artifact["results"]
           if r["world"] == "separable"][0]
    assert sep["auc"] > 0.7


def test_toy_gate_verdicts_at_gate_resolving_n(tmp_path):
    out = str(tmp_path / "toyfull.json")
    assert lm.run(parse("--toy", "--seeds", "0", "--output", out)) == 0
    artifact = json.load(open(out))
    verdicts = {r["world"]: r["gate"]["verdict"]
                for r in artifact["results"]}
    assert verdicts == {"separable": "FAIL", "null": "INSUFFICIENT_EVIDENCE"}
    null = [r for r in artifact["results"] if r["world"] == "null"][0]
    assert abs(null["auc"] - 0.5) < 0.02


def test_worlds_flag_selects(tmp_path):
    out = str(tmp_path / "nullonly.json")
    assert lm.run(parse("--toy", "--quick", "--worlds", "null",
                        "--output", out)) == 0
    worlds = [r["world"] for r in json.load(open(out))["results"]]
    assert worlds == ["null"]


def test_online_is_torch_gated_and_validates_shadow_count(monkeypatch):
    from attacker.attacks import common
    monkeypatch.setattr(common, "torch", None)
    with pytest.raises(SystemExit, match="requires torch"):
        lm.run(parse("--online"))
    args = parse("--online", "--n-shadows", "1")
    with pytest.raises(SystemExit, match="--n-shadows must be >=2"):
        lm.run(args)


def test_evidence_flag_wires_evidence_logger(tmp_path):
    from attacker.evidence import verify_evidence
    out = str(tmp_path / "ev.json")
    assert lm.run(parse("--toy", "--quick", "--evidence",
                        "--output", out)) == 0
    ok, problems = verify_evidence(out + ".evidence.json")
    assert ok and problems == []
    with pytest.raises(SystemExit, match="--evidence requires --output"):
        lm.run(parse("--toy", "--quick", "--evidence"))


def test_module_self_test_passes():
    assert lm.self_test() == 0
