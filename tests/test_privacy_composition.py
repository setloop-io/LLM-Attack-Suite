"""privacy-composition: contract + parser, the toy multi-surface world end
to end (negative control + vulnerable world), determinism, the real
--artifacts composition path (node keying, counted skips, cumulative
cross-epoch axis, UNMEASURED gaps), and fail-closed malformed-input
handling. All torch-free."""

import json

import pytest

from attacker import threat_model
from attacker.attacks import validate_contract
from attacker.attacks import privacy_composition as pc


def parse(*argv):
    return pc.build_parser().parse_args(list(argv))


def test_contract():
    assert validate_contract(pc) is pc
    assert pc.EXPERIMENT_ID == "privacy_composition"
    assert pc.MODES == ("training", "inference")
    assert pc.REQUIRES_LABELS is False
    assert isinstance(pc.DESCRIPTION, str) and pc.DESCRIPTION


def test_help_works_torchless():
    with pytest.raises(SystemExit) as exc:
        parse("--help")
    assert exc.value.code == 0


def test_parser_flags():
    args = parse("--toy", "--quick", "--budgets", "1", "4", "16",
                 "--trials", "4096", "--control-rate", "0.05",
                 "--carryover", "0.7", "--retention", "0.4",
                 "--synergy", "0.2", "--artifacts", "a.json", "b.json",
                 "--evidence")
    assert args.toy and args.quick and args.evidence
    assert args.budgets == [1, 4, 16]
    assert args.trials == 4096
    assert args.control_rate == 0.05
    assert args.carryover == 0.7 and args.retention == 0.4
    assert args.synergy == 0.2
    assert args.artifacts == ["a.json", "b.json"]
    assert hasattr(args, "output") and hasattr(args, "seed") \
        and hasattr(args, "seeds")


def test_invalid_args_are_descriptive():
    with pytest.raises(SystemExit, match="--control-rate"):
        pc.run(parse("--toy", "--control-rate", "0.9"))
    with pytest.raises(SystemExit, match="--budgets"):
        pc.run(parse("--toy", "--budgets", "0"))
    with pytest.raises(SystemExit, match="--trials"):
        pc.run(parse("--toy", "--trials", "10"))


def test_no_input_path_is_a_descriptive_error():
    with pytest.raises(SystemExit, match="pick an input path"):
        pc.run(parse())


@pytest.fixture
def toy_run(tmp_path):
    output = str(tmp_path / "pc.json")
    args = parse("--toy", "--quick", "--output", output)
    assert pc.run(args) == 0
    with open(output) as f:
        return output, json.load(f)


def test_toy_artifact_layout(toy_run):
    output, art = toy_run
    for key in ("schema", "config", "threat_model", "provenance",
                "results", "summary"):
        assert key in art
    assert art["schema"] == "dtraining.attacker.privacy_composition.v1"
    assert art["config"]["attack"] == pc.EXPERIMENT_ID
    assert art["config"]["toy"] is True
    assert art["summary"]
    threat_model.validate(art["threat_model"])
    assert art["threat_model"]["schema"] == threat_model.SCHEMA_V2
    assert art["threat_model"]["actor"]["intent"] == "passive"
    assert art["threat_model"]["actor"]["persistence"] == "cross-epoch"
    assert art["threat_model"]["mappings"]["owasp_llm_2026"] == ["LLM02"]
    with open(output + ".jsonl") as f:
        journal = f.read().strip().splitlines()
    assert len(journal) == len(art["results"])
    assert all(json.loads(l)["experiment"] == pc.EXPERIMENT_ID
               for l in journal)


def test_every_arm_recorded(toy_run):
    _, art = toy_run
    assert art["results"]
    assert all(r["experiment"] == pc.EXPERIMENT_ID for r in art["results"])
    arms = {r["arm"] for r in art["results"]}
    assert set(pc.ARMS) <= arms
    seeds = {r.get("seed") for r in art["results"]}
    assert seeds == {0, 1}


def test_negative_control(toy_run):
    """Each surface alone at its minimum budget plus the null surface:
    advantage ~0, gate PASS."""
    _, art = toy_run
    neg = [r for r in art["results"] if r["arm"] == "negative-control"]
    assert len(neg) == 2 * 5
    for r in neg:
        assert r["role"] == "negative-control"
        e = r["curve"][0]
        assert e["budget"] == art["config"]["budgets"][0]
        assert e["advantage_pp"] < 0.5
        assert e["verdict"] == "INSUFFICIENT_EVIDENCE"
    null = [r for r in art["results"] if r["arm"] == "in-session"
            and r["surfaces"] == [pc.NULL_SURFACE]]
    assert null
    for r in null:
        assert r["first_gate_failure_budget"] is None
        assert all(e["advantage_pp"] == 0.0 and e["verdict"] == "INSUFFICIENT_EVIDENCE"
                   for e in r["curve"])


def test_vulnerable_world_composes(toy_run):
    """The toy world is deliberately vulnerable: joint dimensions must
    fail the frozen gate early and beat every individual surface —
    otherwise the negative control above is vacuous."""
    _, art = toy_run
    res = art["results"]
    bmax = max(art["config"]["budgets"])
    for seed in (0, 1):
        ms = next(r for r in res if r["arm"] == "multi-surface"
                  and r["seed"] == seed)
        assert ms["surfaces"] == list(pc.TOY_SURFACES)
        assert ms["first_gate_failure_budget"] is not None
        assert ms["first_gate_failure_budget"] <= 4
        ms_adv = next(e["advantage_pp"] for e in ms["curve"]
                      if e["budget"] == bmax)
        individuals = [next(e["advantage_pp"]
                            for r in res if r["arm"] == "in-session"
                            and r["seed"] == seed and r["surfaces"] == [s]
                            for e in r["curve"] if e["budget"] == bmax)
                       for s in pc.TOY_SURFACES]
        assert ms_adv > max(individuals)
        for arm in ("cross-session", "cross-epoch", "fwd-bwd-timing"):
            r = next(r for r in res if r["arm"] == arm
                     and r["seed"] == seed)
            assert r["first_gate_failure_budget"] is not None
        syn = next(r for r in res if r["arm"] == "synergy"
                   and r["seed"] == seed)
        assert syn["max_synergy_pp"] > 0.0
        marg = next(r for r in res if r["arm"] == "marginal-leakage"
                    and r["seed"] == seed)
        assert marg["delta_pp"]["S1"] > 0.0
        assert marg["delta_pp"]["S2"] > 0.0
        assert marg["delta_pp"][pc.NULL_SURFACE] == 0.0
        ub = next(r for r in res if r["arm"] == "adaptive-upper-bound"
                  and r["seed"] == seed)
        assert ub["max_excess_pp"] > 1.0
        assert ub["at_budget"] == bmax


def test_determinism(tmp_path):
    runs = []
    for i in (0, 1):
        output = tmp_path / f"d{i}.json"
        args = parse("--toy", "--quick", "--output", str(output))
        assert pc.run(args) == 0
        with open(output) as f:
            art = json.load(f)
        runs.append((art["results"], art["summary"]))
    assert runs[0] == runs[1]


def _wire_shape_artifact():
    return {"schema": "dtraining.attacker.wire_shape.v1", "config": {},
            "threat_model": "t", "provenance": {},
            "results": [
                {"experiment": "wire_shape_analysis", "arm": "length-leak",
                 "top1": 4.5, "budget": 1, "n_records": 2000},
                {"experiment": "wire_shape_analysis", "arm": "length-leak",
                 "top1": 5.5, "budget": 4, "n_records": 2000},
                {"experiment": "wire_shape_analysis", "arm": "length-leak",
                 "top1": 8.0, "budget": 16, "n_records": 2000},
                {"experiment": "wire_shape_analysis", "arm": "phase-detect",
                 "f1": 0.9},
                {"experiment": "mystery_exp", "top1": 5.0},
            ], "summary": []}


def _leak_artifact():
    return {"schema": "dtraining.attacker.leak.v1", "config": {},
            "threat_model": "t", "provenance": {},
            "results": [
                {"experiment": "leak_accumulation", "leak_per_epoch": 4,
                 "epoch": t, "cumulative_leaked_rows": 4 * (t + 1),
                 "composite_decode_top1": 4.2 + 0.8 * (t + 1)}
                for t in range(4)
            ] + [{"experiment": "leak_accumulation", "epoch": 4,
                  "error": "solve failed"}],
            "summary": []}


@pytest.fixture
def real_run(tmp_path):
    p1, p2 = tmp_path / "ws.json", tmp_path / "la.json"
    p1.write_text(json.dumps(_wire_shape_artifact()))
    p2.write_text(json.dumps(_leak_artifact()))
    output = str(tmp_path / "pc_real.json")
    args = parse("--artifacts", str(p1), str(p2), "--output", output)
    assert pc.run(args) == 0
    with open(output) as f:
        return output, json.load(f)


def test_real_artifact_layout_and_audit(real_run):
    output, art = real_run
    for key in ("schema", "config", "threat_model", "provenance",
                "results", "summary"):
        assert key in art
    threat_model.validate(art["threat_model"])
    assert art["config"]["source"] == "artifacts"
    with open(output + ".jsonl") as f:
        journal = f.read().strip().splitlines()
    assert len(journal) == len(art["results"])
    audit = next(r for r in art["results"] if r["arm"] == "input-audit")
    assert audit["audit"]["records_total"] == 10
    assert audit["audit"]["points"] == 7
    assert audit["audit"]["skipped_no_metric"] == 1
    assert audit["audit"]["skipped_unknown_experiment"] == 1
    assert audit["audit"]["skipped_error"] == 1
    nodes = {(n["attack"], n["surface"]) for n in audit["nodes"]}
    assert nodes == {("wire-shape", "S7"), ("leak-accumulation", "S1")}


def test_real_composition_cells(real_run):
    _, art = real_run
    res = art["results"]
    assert all(r["experiment"] == pc.EXPERIMENT_ID for r in res)
    insession = [r for r in res if r["arm"] == "in-session"]
    assert {(r["attack"], tuple(r["surfaces"])) for r in insession} == \
        {("wire-shape", ("S7",)), ("leak-accumulation", ("S1",))}
    ws = next(r for r in insession if r["attack"] == "wire-shape")
    assert [e["budget"] for e in ws["curve"]] == [1.0, 4.0, 16.0]
    assert ws["curve"][-1]["verdict"] == "UNMEASURED"
    xe = next(r for r in res if r["arm"] == "cross-epoch")
    cums = [e["budget"] for e in xe["curve"]]
    assert cums == sorted(cums) and len(cums) == 4
    ms = next(r for r in res if r["arm"] == "multi-surface")
    assert ms["surfaces"] == ["S1", "S7"]
    assert ms["max_advantage_pp"] > max(
        r["max_advantage_pp"] for r in insession)
    fbt = next(r for r in res if r["arm"] == "fwd-bwd-timing")
    assert fbt["verdict"] == "UNMEASURED"
    la = next(r for r in insession if r["attack"] == "leak-accumulation")
    assert all(e["verdict"] == "UNMEASURED" for e in la["curve"])
    assert "first_gate_failure_budget" not in la
    assert any(r["arm"] == "marginal-leakage" for r in res)
    assert any(r["arm"] == "synergy" for r in res)
    ub = next(r for r in res if r["arm"] == "adaptive-upper-bound")
    assert "max_excess_pp" not in ub


def test_real_path_fail_closed(tmp_path):
    missing = str(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="missing.json"):
        pc.load_artifact_documents([missing])
    with pytest.raises(SystemExit, match="missing.json"):
        pc.run(parse("--artifacts", missing))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="bad.json"):
        pc.load_artifact_documents([str(bad)])
    with pytest.raises(SystemExit, match="bad.json"):
        pc.run(parse("--artifacts", str(bad)))
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"results": []}))
    with pytest.raises(ValueError, match="empty.json"):
        pc.load_artifact_documents([str(empty)])
    with pytest.raises(SystemExit, match="empty.json"):
        pc.run(parse("--artifacts", str(empty)))


def test_self_test_passes():
    assert pc.self_test() == 0
