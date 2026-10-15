"""lira-membership RMIA scoring (--scoring rmia): the 2024 SOTA robust
membership inference (Zarifzadeh et al., "Low-Cost High-Power Membership
Inference by Boosting Relativity"), implemented as the offline Bayes-rule
pairwise likelihood ratio (eq. 3) composed into the quantile score
(eq. 5, gamma=1) over a non-member reference population.

Torch-free throughout: scoring-core fixtures, the toy separable/null
sanity (rmia AUC >= lira AUC, null ~ 0.5), journal/artifact/evidence, and
the planted heavy-tailed-difficulty population where RMIA's relativity
recovers the TPR@low-FPR edge the absolute LIRA score loses (the paper's
headline result)."""

import json
import random

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


def heavy_tail_records(n=1500, shift=2.5, tau=3.0, n_ref=4, sigma=0.7,
                       seed=0):
    """Planted per-example records on a HEAVY-TAILED difficulty
    distribution (a tail of intrinsically easy AND hard examples on top of
    a moderate body). The statistic is a loss-like NLL (members shift
    lower); ref_in/ref_out are the example's NLLs under IN/OUT shadow
    models (few reference models — RMIA's documented low-cost regime).
    This is the regime where the absolute LIRA score is dominated by
    difficulty and RMIA's relativity wins at low FPR."""
    rng = random.Random(seed)
    recs = []
    for split in ("in", "out"):
        for i in range(n):
            r = rng.random()
            if r < 0.075:
                th = -abs(rng.gauss(0.0, tau))
            elif r < 0.15:
                th = abs(rng.gauss(0.0, tau))
            else:
                th = rng.gauss(0.0, 2.0)
            s = th - (shift if split == "in" else 0.0) + rng.gauss(0.0, sigma)
            recs.append({
                "example_id": f"{split}-{i}", "split": split, "statistic": s,
                "ref_in": [th - shift + rng.gauss(0.0, sigma)
                           for _ in range(n_ref)],
                "ref_out": [th + rng.gauss(0.0, sigma)
                            for _ in range(n_ref)]})
    rng.shuffle(recs)
    return recs


def test_contract_and_dispatch(cleanup_registry):
    attacks.validate_contract(lm, "lira-membership")
    if "lira-membership" not in attacks.ATTACKS:
        attacks.register_attack("lira-membership", lm)
    assert attacks.load("lira-membership") is lm


def test_parser_builds_torchless():
    args = lm.build_parser().parse_args([])
    assert args.scoring == "lira"
    assert args.toy_ref == 8
    for s in ("lira", "rmia"):
        assert lm.build_parser().parse_args(["--scoring", s]).scoring == s
    with pytest.raises(SystemExit):
        lm.build_parser().parse_args(["--scoring", "nope"])
    with pytest.raises(SystemExit):
        lm.build_parser().parse_args(["--help"])


def test_log_rho_fixtures():
    import math
    assert lm._log_rho(1.0, [2.0, 2.0]) == pytest.approx(1.0)
    assert lm._log_rho(0.0, [0.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert lm._log_rho(0.0, [3.0, 3.0]) > lm._log_rho(0.0, [0.5, 0.5])
    assert math.isfinite(lm._logsumexp([0.0, -100.0, 100.0]))


def test_rmia_scores_separable_fixture():
    recs = [{"example_id": f"i{i}", "split": "in", "statistic": 0.0,
             "ref_out": [2.0, 2.0, 2.0]} for i in range(6)] + \
           [{"example_id": f"o{i}", "split": "out", "statistic": 0.0,
             "ref_out": [c, c, c]}
            for i, c in enumerate((-0.6, -0.4, -0.2, 0.0, 0.2, 0.4))]
    rs = lm.rmia_scores(recs)
    assert all(r["model"] == "rmia_offline" for r in rs)
    assert all(0.0 <= r["rmia"] <= 1.0 for r in rs)
    assert all(r["n_ref_population"] == 6 for r in rs)
    assert lm.rank_auc([r["log_rho"] for r in rs if r["split"] == "in"],
                       [r["log_rho"] for r in rs if r["split"] == "out"]) \
        == 1.0
    assert sum(r["rmia"] for r in rs if r["split"] == "in") > \
        sum(r["rmia"] for r in rs if r["split"] == "out")
    assert sum(r["rmia"] == 1.0 for r in rs if r["split"] == "out") == 1


def test_rmia_requires_ref_out():
    with pytest.raises(SystemExit, match="ref_out"):
        lm.rmia_scores([{"example_id": "x", "split": "in", "statistic": 1.0},
                        {"example_id": "y", "split": "out", "statistic": 2.0,
                         "ref_out": [2.0, 2.1]}])
    with pytest.raises(SystemExit, match="ref_out"):
        lm.rmia_scores([{"example_id": "x", "split": "in", "statistic": 1.0,
                         "ref_out": [1.0]},
                        {"example_id": "y", "split": "out", "statistic": 2.0,
                         "ref_out": [2.0, 2.1]}])
    with pytest.raises(SystemExit, match="reference population"):
        lm.rmia_scores([{"example_id": "x", "split": "in", "statistic": 1.0,
                         "ref_out": [1.0, 1.1]}])


def test_toy_rmia_ge_lira_and_null_honest():
    for world, shift in (("separable", 2.0), ("null", 0.0)):
        for seed in (0, 1):
            recs = lm.make_toy_records(world, 3000, seed, shift, n_ref=8)
            rs = lm.rmia_scores(recs)
            auc_r = lm.rank_auc([r["rmia"] for r in rs if r["split"] == "in"],
                                [r["rmia"] for r in rs if r["split"] == "out"])
            ls = lm.score_records(recs)
            auc_l = lm.rank_auc([r["llr"] for r in ls if r["split"] == "in"],
                                [r["llr"] for r in ls if r["split"] == "out"])
            assert auc_r >= auc_l - 0.02
            if world == "null":
                assert abs(auc_r - 0.5) < 0.05
                assert abs(auc_l - 0.5) < 0.05
            else:
                assert auc_r > 0.8


def test_toy_run_rmia_journals_artifact_and_comparison(tmp_path):
    out = str(tmp_path / "rmia.json")
    assert lm.run(parse("--toy", "--scoring", "rmia", "--quick",
                        "--output", out)) == 0
    art = json.load(open(out))
    assert sorted(r["world"] for r in art["results"]) == ["null",
                                                          "separable"]
    for rec in art["results"]:
        assert rec["scoring"] == "rmia"
        for key in ("auc", "tpr_at_fpr", "gate", "lira_global_auc",
                    "n_in", "n_out"):
            assert key in rec
        assert rec["auc"] >= rec["lira_global_auc"] - 0.02
    null = [r for r in art["results"] if r["world"] == "null"][0]
    assert abs(null["auc"] - 0.5) < 0.1
    journal = [json.loads(l) for l in open(out + ".jsonl")]
    assert len(journal) == len(art["results"])


def test_toy_run_lira_default_unchanged(tmp_path):
    out = str(tmp_path / "lira.json")
    assert lm.run(parse("--toy", "--quick", "--output", out)) == 0
    art = json.load(open(out))
    for rec in art["results"]:
        assert rec["scoring"] == "lira"
        assert "lira_global_auc" not in rec


def test_evidence_flag_wires_evidence_logger(tmp_path):
    from attacker.evidence import verify_evidence
    out = str(tmp_path / "ev.json")
    assert lm.run(parse("--toy", "--scoring", "rmia", "--quick", "--evidence",
                        "--output", out)) == 0
    ok, problems = verify_evidence(out + ".evidence.json")
    assert ok and problems == []
    with pytest.raises(SystemExit, match="--evidence requires --output"):
        lm.run(parse("--toy", "--scoring", "rmia", "--quick", "--evidence"))


def test_records_rmia_separable_and_null(tmp_path):
    recs = heavy_tail_records(n=400, shift=3.0, n_ref=8, seed=0)
    path = write_jsonl(tmp_path / "recs.jsonl", recs)
    out = str(tmp_path / "off.json")
    assert lm.run(parse("--records", path, "--scoring", "rmia",
                        "--output", out)) == 0
    rec = json.load(open(out))["results"][0]
    assert rec["scoring"] == "rmia"
    assert rec["auc"] > 0.85
    assert set(rec["tpr_at_fpr"]) == {"0.01", "0.001"}
    assert len(open(out + ".jsonl").read().splitlines()) == 1

    null = heavy_tail_records(n=400, shift=0.0, n_ref=8, seed=7)
    null_path = write_jsonl(tmp_path / "null.jsonl", null)
    null_out = str(tmp_path / "null.json")
    assert lm.run(parse("--records", null_path, "--scoring", "rmia",
                        "--output", null_out)) == 0
    null_rec = json.load(open(null_out))["results"][0]
    assert abs(null_rec["auc"] - 0.5) < 0.15


def test_rmia_beats_lira_tpr_at_low_fpr_on_heavy_tail(tmp_path):
    for seed in (0, 1):
        recs = heavy_tail_records(n=1500, shift=2.5, tau=3.0, n_ref=4,
                                  sigma=0.7, seed=seed)
        path = write_jsonl(tmp_path / f"heavy{seed}.jsonl", recs)
        out_l = str(tmp_path / f"l{seed}.json")
        out_r = str(tmp_path / f"r{seed}.json")
        assert lm.run(parse("--records", path, "--scoring", "lira",
                            "--output", out_l)) == 0
        assert lm.run(parse("--records", path, "--scoring", "rmia",
                            "--output", out_r)) == 0
        lira = json.load(open(out_l))["results"][0]
        rmia = json.load(open(out_r))["results"][0]
        lira_tpr = lira["tpr_at_fpr"]["0.001"]
        rmia_tpr = rmia["tpr_at_fpr"]["0.001"]
        assert lira_tpr["resolvable"] and rmia_tpr["resolvable"]
        assert rmia_tpr["tpr"] > lira_tpr["tpr"]
        assert rmia["auc"] >= lira["auc"] - 0.02


def test_module_self_test_passes():
    assert lm.self_test() == 0
