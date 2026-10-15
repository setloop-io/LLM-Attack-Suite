"""poisoning-backdoor (integrity loss via data/gradient/adapter/checkpoint/
RAG poisoning, OWASP LLM04/LLM05): module contract, toy end-to-end +
artifact/journal layout, the mandatory negative controls (zero-poison
world at the false-activation rate; random-corruption baseline separating
targeted success from utility damage), the vulnerable/informative worlds
(every arm measurably works undefended; defenses bite where they should
and FAIL where documented — clean-label), the honest-majority control for
model-replacement, the real-target integration point (full validation,
then a descriptive SystemExit), malformed input, determinism, and the
self-test. All torch-free."""

import json

import pytest

from attacker import threat_model
from attacker.attacks import poisoning_backdoor as pb


def _toy_args(tmp_path, *extra, name="pb.json"):
    args = pb.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(tmp_path / name), *extra])
    args.mode = "training"
    assert pb.run(args) == 0
    art = json.loads((tmp_path / name).read_text())
    journal = [json.loads(l) for l in
               (tmp_path / (name + ".jsonl")).read_text().splitlines()]
    return art, journal


def _cells(art, arm, world=None, role=None):
    out = [r for r in art["results"] if r["arm"] == arm]
    if world is not None:
        out = [r for r in out if r["world"] == world]
    if role is not None:
        out = [r for r in out if r["role"] == role]
    return out


def test_contract_and_torchless_help():
    from attacker.attacks import validate_contract
    assert validate_contract(pb) is pb
    assert pb.EXPERIMENT_ID == "poisoning_backdoor"
    assert pb.MODES == ("training",)
    assert isinstance(pb.REQUIRES_LABELS, str) and pb.REQUIRES_LABELS
    assert isinstance(pb.DESCRIPTION, str) and pb.DESCRIPTION
    ap = pb.build_parser()
    with pytest.raises(SystemExit) as exc:
        ap.parse_args(["--help"])
    assert exc.value.code == 0
    args = ap.parse_args(["--toy", "--quick", "--arms", "clean-label",
                          "rag-poison", "--budgets", "0.1", "0.2",
                          "--clients", "6"])
    assert args.arms == ["clean-label", "rag-poison"]
    assert args.budgets == [0.1, 0.2] and args.clients == 6
    args = ap.parse_args(["--dataset", "d.jsonl", "--poison-plan",
                          "p.jsonl"])
    assert args.dataset == "d.jsonl" and args.poison_plan == "p.jsonl"


def test_self_test_passes():
    assert pb.self_test() == 0


def test_toy_quick_end_to_end(tmp_path):
    art, journal = _toy_args(tmp_path)
    for key in ("schema", "config", "threat_model", "provenance",
                "results", "summary"):
        assert key in art
    assert art["schema"] == "dtraining.attacker.poisoning_backdoor.v1"
    assert art["config"]["attack"] == pb.EXPERIMENT_ID
    threat_model.validate(art["threat_model"])
    assert len(journal) == len(art["results"]) > 0
    for arm in pb.ARMS:
        recs = _cells(art, arm)
        assert recs, f"arm {arm} produced no result records"
        assert all(r["experiment"] == pb.EXPERIMENT_ID for r in recs)
    assert all(r["experiment"] == pb.EXPERIMENT_ID for r in art["results"])
    ctl = _cells(art, pb.NEGATIVE_CONTROL)
    assert ctl and all(r["role"] == "negative-control" for r in ctl)
    arms_in_summary = {s["arm"] for s in art["summary"]}
    assert set(pb.ARMS) | {pb.NEGATIVE_CONTROL} == arms_in_summary
    for r in art["results"]:
        if r["role"] in ("attack", "baseline"):
            for key in ("asr", "asr_wilson_lo", "asr_wilson_hi",
                        "false_activation_rate", "trigger_specificity",
                        "clean_acc", "clean_acc_delta",
                        "asr_after_clean_ft", "persistence",
                        "asr_transfer"):
                assert key in r, f"{r['arm']}/{r['world']} missing {key}"


def test_zero_poison_control_sits_at_false_activation_rate(tmp_path):
    art, _ = _toy_args(tmp_path)
    ctl = _cells(art, pb.NEGATIVE_CONTROL)
    assert len(ctl) == 2 * len(art["config"]["seeds"])
    for r in ctl:
        assert r["budget"] == 0.0
        assert r["asr"] == 0.0
        assert r["false_activation_rate"] == 0.0
        assert r["clean_acc"] >= 0.9
    summ = [s for s in art["summary"] if s["arm"] == pb.NEGATIVE_CONTROL][0]
    assert summ["asr_mean"] == 0.0


def test_random_corruption_baseline_separates_targeted_from_utility():
    import random
    data = pb.make_examples(24, 5)
    test = pb.make_examples(10, 17)
    transfer = pb.make_examples(8, 23)
    corrupted, _ = pb.corrupt_random_labels(data, 0.25, random.Random(7))
    assert not any(pb.TRIGGER_ID in ex["tokens"] for ex in corrupted)
    m = pb.train(pb.new_model(), pb._xy(corrupted), 12, 0.8, 31)
    ev = pb.eval_model(m, test, transfer, 0, 3)
    assert ev["asr"] == 0.0
    assert ev["clean_acc"] < 1.0


def test_vulnerable_world_shows_the_attack(tmp_path):
    """Otherwise the controls are vacuous: undefended, every targeted arm
    must bite at the largest quick budget."""
    art, _ = _toy_args(tmp_path)

    def und_at_top(arm):
        cs = _cells(art, arm, world="undefended")
        top = max(r["budget"] for r in cs)
        return [r for r in cs if r["budget"] == top]

    for arm in ("clean-label", "dirty-label", "trigger-backdoor",
                "model-replacement", "adapter-backdoor",
                "checkpoint-backdoor"):
        for r in und_at_top(arm):
            assert r["asr"] >= 0.75, (arm, r["asr"])
            assert r["asr_wilson_lo"] > r["false_activation_rate"]
    for r in und_at_top("rag-poison"):
        assert r["asr"] >= 0.9
    gs = und_at_top("gradient-sign")
    assert all(r["clean_acc_delta"] <= -0.2 for r in gs)
    assert all(r["trigger_specificity"] <= 0.05 for r in gs)
    gs_base = _cells(art, "gradient-sign", world=pb.RANDOM_BASELINE)
    top = max(r["budget"] for r in gs_base)
    assert all(r["clean_acc_delta"] > -0.15 for r in gs_base
               if r["budget"] == top)


def test_defended_world_bites_where_it_should(tmp_path):
    art, _ = _toy_args(tmp_path)
    for arm in ("dirty-label", "trigger-backdoor"):
        for r in _cells(art, arm, world="defended"):
            assert r["asr"] == 0.0
            assert r["detection_recall"] == 1.0
    for r in _cells(art, "trigger-backdoor", world="defended"):
        assert r["collateral_flag_rate"] > 0.0
    cl = _cells(art, "clean-label", world="defended")
    assert all(r["detection_recall"] == 0.0 for r in cl)
    top = max(r["budget"] for r in cl)
    assert all(r["asr"] >= 0.75 for r in cl if r["budget"] == top)
    for arm in ("adapter-backdoor", "checkpoint-backdoor"):
        dfd = _cells(art, arm, world="defended")
        assert all(r["gate_accepted"] is (r["budget"] <= 0.125)
                   for r in dfd)
        assert all(r["asr"] == 0.0 for r in dfd if not r["gate_accepted"])
    rag_lo = [r for r in _cells(art, "rag-poison") if r["budget"] == 0.125]
    und = [r for r in rag_lo if r["world"] == "undefended"]
    dfd = [r for r in rag_lo if r["world"] == "defended"]
    assert und and dfd
    assert all(r["asr"] < u["asr"] for r in dfd for u in und
               if r["seed"] == u["seed"])
    summ = {s["arm"]: s for s in art["summary"]}
    assert summ["adapter-backdoor"]["first_gate_failure_budget"] == 0.125
    assert summ["dirty-label"]["first_gate_failure_budget"] is None


def test_model_replacement_honest_majority_control(tmp_path):
    art, _ = _toy_args(tmp_path)
    ctl = [r for r in _cells(art, "model-replacement", world="defended")
           if r["role"] == "control"]
    assert ctl and all(r["budget"] == 0.5 for r in ctl)
    assert all(r["asr"] >= 0.75 for r in ctl)
    honest = [r for r in _cells(art, "model-replacement", world="defended")
              if r["role"] == "attack"]
    assert honest and all(r["n_malicious"] < r["n_clients"] / 2
                          for r in honest)
    assert all(r["asr"] == 0.0 for r in honest)


def test_live_requires_both_files(tmp_path):
    args = pb.build_parser().parse_args(["--dataset", "d.jsonl"])
    with pytest.raises(SystemExit, match="--poison-plan"):
        pb.run(args)
    args = pb.build_parser().parse_args(["--poison-plan", "p.jsonl"])
    with pytest.raises(SystemExit, match="--dataset"):
        pb.run(args)


def test_live_and_toy_are_mutually_exclusive(tmp_path):
    args = pb.build_parser().parse_args(
        ["--toy", "--dataset", "d.jsonl", "--poison-plan", "p.jsonl"])
    with pytest.raises(SystemExit, match="mutually exclusive"):
        pb.run(args)


def test_no_target_path_is_descriptive():
    args = pb.build_parser().parse_args([])
    with pytest.raises(SystemExit, match="--toy"):
        pb.run(args)


def test_live_validates_then_stops_at_the_integration_point(tmp_path):
    ds = tmp_path / "clean.jsonl"
    ds.write_text('{"tokens": ["a", "b"], "label": "x"}\n'
                  '{"tokens": ["c"], "label": "y"}\n')
    plan = tmp_path / "plan.jsonl"
    plan.write_text('{"arm": "dirty-label", "budget": 0.1}\n'
                    '{"arm": "rag-poison", "budget": 0.2, '
                    '"tokens": ["kw"], "label": "pay"}\n')
    out = tmp_path / "live.json"
    args = pb.build_parser().parse_args(
        ["--dataset", str(ds), "--poison-plan", str(plan),
         "--output", str(out)])
    with pytest.raises(SystemExit, match="INTEGRATION POINT"):
        pb.run(args)
    journal = [json.loads(l)
               for l in (tmp_path / "live.json.jsonl")
               .read_text().splitlines()]
    assert len(journal) == 1
    rec = journal[0]
    assert rec["experiment"] == pb.EXPERIMENT_ID
    assert rec["arm"] == "live-plan" and rec["n_clean_examples"] == 2
    assert rec["planned_cells"] == [{"arm": "dirty-label", "budget": 0.1},
                                    {"arm": "rag-poison", "budget": 0.2}]


def test_malformed_input_is_descriptive(tmp_path):
    ds = tmp_path / "bad.jsonl"
    ds.write_text('{"tokens": [], "label": "x"}\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl:1.*tokens"):
        pb.read_dataset(str(ds))
    with pytest.raises(ValueError, match="does not exist"):
        pb.read_dataset(str(tmp_path / "missing.jsonl"))
    plan = tmp_path / "plan.jsonl"
    plan.write_text('{"arm": "sideways", "budget": 0.1}\n')
    with pytest.raises(ValueError, match="sideways"):
        pb.read_poison_plan(str(plan))
    plan.write_text('{"arm": "dirty-label", "budget": 1.5}\n')
    with pytest.raises(ValueError, match="budget"):
        pb.read_poison_plan(str(plan))
    plan.write_text('{"arm": "dirty-label"}\n')
    with pytest.raises(ValueError, match=r"arm\+budget"):
        pb.read_poison_plan(str(plan))
    with pytest.raises(ValueError, match="does not exist"):
        pb.read_poison_plan(str(tmp_path / "missing.jsonl"))
    args = pb.build_parser().parse_args(
        ["--dataset", str(ds), "--poison-plan", str(plan)])
    with pytest.raises(ValueError, match=r"bad\.jsonl:1.*tokens"):
        pb.run(args)
    args = pb.build_parser().parse_args(
        ["--toy", "--quick", "--budgets", "-0.1",
         "--output", str(tmp_path / "bad.json")])
    with pytest.raises(SystemExit, match="--budgets"):
        pb.run(args)
    assert not (tmp_path / "bad.json").exists()
    args = pb.build_parser().parse_args(["--toy", "--clients", "2"])
    with pytest.raises(SystemExit, match="--clients"):
        pb.run(args)


def test_determinism_same_seed(tmp_path):
    art1, _ = _toy_args(tmp_path, name="a.json")
    art2, _ = _toy_args(tmp_path, name="b.json")
    assert art1["results"] == art2["results"]
    assert art1["summary"] == art2["summary"]


def test_toy_without_output_writes_nothing(tmp_path):
    args = pb.build_parser().parse_args(["--toy", "--quick"])
    args.mode = "training"
    assert pb.run(args) == 0
    assert list(tmp_path.iterdir()) == []


def test_arm_subset_runs_only_those_arms(tmp_path):
    art, _ = _toy_args(tmp_path, "--arms", "dirty-label", "rag-poison",
                       name="sub.json")
    arms = {r["arm"] for r in art["results"]}
    assert arms == {"dirty-label", "rag-poison", pb.NEGATIVE_CONTROL}


def test_summary_reports_distinct_n_seeds_alongside_n_cells(tmp_path):
    """An attack arm holds one record per (seed, budget, world), so n_cells
    is a record count and must not be read as a seed count. The zero-poison
    negative control emits exactly one record per seed, so only the attack
    arms carry more records than seeds."""
    art, _ = _toy_args(tmp_path, "--arms", "dirty-label", name="ns.json")
    by = {}
    for r in art["results"]:
        by.setdefault(r["arm"], []).append(r)
    assert art["summary"]
    for row in art["summary"]:
        recs = by[row["arm"]]
        assert row["n_seeds"] == len({r["seed"] for r in recs})
        assert row["n_cells"] == len(recs)
    atk = next(r for r in art["summary"] if r["arm"] == "dirty-label")
    assert atk["n_cells"] > atk["n_seeds"]


def test_duplicate_seeds_are_not_double_counted(tmp_path):
    """--quick keeps seeds[:2]; a repeated value therefore yields duplicate
    records per seed and n_seeds must stay the distinct count."""
    art, _ = _toy_args(tmp_path, "--arms", "dirty-label",
                       "--seeds", "0", "0", "1", name="dup.json")
    assert art["config"]["seeds"] == [0, 0]
    by = {}
    for r in art["results"]:
        by.setdefault(r["arm"], []).append(r)
    assert art["summary"]
    for row in art["summary"]:
        recs = by[row["arm"]]
        assert {r["seed"] for r in recs} == {0}
        assert row["n_seeds"] == 1
        assert row["n_cells"] == len(recs) > 1
