"""trace-inversion: arithmetic world, order conventions, retrieval +
composition, verifier-guided search, refinement, replay calibration,
capability-excess gating, and registry dispatch via
attacks.register_attack (the parent wires the built-in registry entry
separately). Torch-free throughout."""

import json
import random

import pytest

from attacker import attacks
from attacker.attacks import trace_inversion as ti


def parse(*argv):
    args = ti.build_parser().parse_args(list(argv))
    args.mode = "inference"
    return args


@pytest.fixture
def cleanup_registry():
    before = set(attacks.ATTACKS)
    yield
    for name in set(attacks.ATTACKS) - before:
        attacks.ATTACKS.pop(name, None)


def test_contract_and_dispatch_via_register_attack(cleanup_registry):
    attacks.validate_contract(ti, "trace-inversion")
    assert ti.EXPERIMENT_ID == "trace_inversion"
    assert ti.MODES == ("inference",)
    assert ti.REQUIRES_LABELS is False
    if "trace-inversion" not in attacks.ATTACKS:
        attacks.register_attack("trace-inversion", ti)
    assert attacks.load("trace-inversion") is ti
    assert "trace-inversion" in attacks.for_mode("inference")


def test_parser_builds_torchless():
    args = ti.build_parser().parse_args([])
    assert args.setting == ["summary", "no-summary"]
    assert args.candidates == 1 and args.refine is False
    assert args.calibrate_with is None


def test_order_conventions():
    p = {"a": 3, "op1": "+", "b": 4, "op2": "*", "c": 5}
    assert ti.final_answer(p, ti.ORDER_LTR) == 35
    assert ti.final_answer(p, ti.ORDER_PRECEDENCE) == 23
    same = {"a": 3, "op1": "*", "b": 4, "op2": "-", "c": 5}
    assert ti.final_answer(same, ti.ORDER_LTR) \
        == ti.final_answer(same, ti.ORDER_PRECEDENCE)
    assert ti.parse_problem(ti.problem_text(p)) == p


def test_victim_example_self_consistent():
    rng = random.Random(0)
    for _ in range(20):
        ex = ti.victim_example(rng)
        assert str(ti.final_answer(ex["problem"], ti.ORDER_LTR)) \
            == ex["answer"]
        assert "Step" in ex["trace"] and "Step" not in ex["summary"]
        assert ex["answer"] in ex["summary"]


def test_metric_anchors():
    assert ti.token_f1("a b c", "a b c") == 1.0
    assert ti.token_f1("a b", "c d") == 0.0
    assert ti.rouge_l_f1("a b c", "a b c") == 1.0
    assert ti.rouge_l_f1("b a", "a b") < ti.token_f1("b a", "a b")
    assert ti.summary_coverage("Let's compute 1. Answer: 2.",
                               "Let's compute 1. Work. Answer: 2.") == 1.0


def test_verifier_guided_search_fixes_order_hypothesis():
    p = {"a": 3, "op1": "+", "b": 4, "op2": "*", "c": 5}
    x = ti.problem_text(p)
    summary = f"{ti.VICTIM_STYLE['opening'].format(**p)} " \
              f"{ti.VICTIM_STYLE['final'].format(y=35)}"
    corpus = ti.make_surrogate_corpus(random.Random(1), n=20)
    one = ti.best_of_k(ti.synthesize_candidates(
        x, summary, "35", ti.SURROGATE_STYLE, 1, True, corpus=corpus))
    assert not one["consistent"]
    four = ti.best_of_k(ti.synthesize_candidates(
        x, summary, "35", ti.SURROGATE_STYLE, 4, True, corpus=corpus))
    assert four["consistent"] and four["order"] == ti.ORDER_LTR
    assert four["coverage"] == 1.0


def test_refinement_repairs_inconsistent_draft():
    p = {"a": 3, "op1": "+", "b": 4, "op2": "*", "c": 5}
    x = ti.problem_text(p)
    draft = ti.synthesize(x, None, "35", ti.SURROGATE_STYLE,
                          ti.ORDER_PRECEDENCE, use_summary=False)
    assert not draft["consistent"]
    fixed, did = ti.refine(draft, x, None, "35", use_summary=False)
    assert did and fixed["consistent"] and fixed["order"] == ti.ORDER_LTR
    ok, did_ok = ti.refine(fixed, x, None, "35", use_summary=False)
    assert not did_ok and ok is fixed


def _write_decoded(path, texts):
    with open(path, "w") as f:
        for t in texts:
            f.write(json.dumps({"decoded_text": t}) + "\n")
    return str(path)


def test_learn_style_distinguishes_families():
    victim_texts = [ti.victim_example(random.Random(i))["trace"]
                    for i in range(10)]
    style, support = ti.learn_style(victim_texts)
    assert style["opening"] == ti.VICTIM_STYLE["opening"]
    assert style["step"] == ti.VICTIM_STYLE["step"]
    assert style["final"] == ti.VICTIM_STYLE["final"]
    assert style["closing"] == ti.VICTIM_STYLE["closing"]
    assert support["n_traces"] == 10
    sur = ti.make_surrogate_corpus(random.Random(0), n=12)
    s_style, _ = ti.learn_style([e["text"] for e in sur])
    assert s_style["step"] == ti.SURROGATE_STYLE["step"]
    assert s_style["closing"] is None


def test_learn_style_robust_to_decode_corruption():
    from attacker.attacks import reasoning_replay as rr
    texts = []
    for i in range(8):
        ex = ti.victim_example(random.Random(i))
        blob = rr.encode_blob(ex["trace"], "claude-like")
        texts.append(rr.toy_decode(blob, "claude-like", "current-turn",
                                   seed=i)["text"])
    style, _ = ti.learn_style(texts)
    assert style["opening"] == ti.VICTIM_STYLE["opening"]
    assert style["step"] == ti.VICTIM_STYLE["step"]
    assert style["final"] == ti.VICTIM_STYLE["final"]


@pytest.fixture
def toy_run(tmp_path):
    out = str(tmp_path / "ti.json")
    assert ti.run(parse("--toy", "--quick", "--candidates", "4", "--refine",
                        "--output", out)) == 0
    return out, json.load(open(out))


def _cell(art, setting, calibrated, k, refined):
    return [r for r in art["results"]
            if r["setting"] == setting and r["calibrated"] == calibrated
            and r["candidates"] == k and r["refined"] == refined][0]


def test_summary_beats_no_summary_tf1(toy_run):
    _, art = toy_run
    for k in (1, 4):
        s = _cell(art, "summary", False, k, False)
        n = _cell(art, "no-summary", False, k, False)
        assert s["tf1"] > n["tf1"] + 0.05
        assert s["summary_coverage"] > n["summary_coverage"]


def test_best_of_k_uplift(toy_run):
    _, art = toy_run
    one = _cell(art, "summary", False, 1, False)
    four = _cell(art, "summary", False, 4, False)
    assert one["answer_consistency"] < 1.0
    assert four["answer_consistency"] > one["answer_consistency"]
    assert four["tf1"] >= one["tf1"]


def test_refinement_reduces_inconsistency(toy_run):
    _, art = toy_run
    plain = _cell(art, "summary", False, 1, False)
    refined = _cell(art, "summary", False, 1, True)
    assert refined["answer_consistency"] > plain["answer_consistency"]
    assert refined["n_refined"] > 0


def test_capability_excess_cells_carry_gate_verdicts(toy_run):
    _, art = toy_run
    for rec in art["results"]:
        assert rec["gate"]["verdict"] in ("PASS", "FAIL")
        assert rec["gate"]["control_definition"] == "label_free_majority"
        assert "capability_excess_pp" in rec
    main = _cell(art, "summary", False, 4, False)
    assert main["student"]["accuracy"] > 0.9
    assert main["control"]["accuracy"] < 0.2
    assert main["capability_excess_pp"] > 50.0
    assert main["gate"]["verdict"] == "FAIL"
    assert main["student"]["order_learned"] == ti.ORDER_LTR


def test_calibration_improves_tf1(toy_run, tmp_path):
    decoded = _write_decoded(
        tmp_path / "decoded.jsonl",
        [ti.victim_example(random.Random(i))["trace"] for i in range(10)])
    out = str(tmp_path / "cal.json")
    assert ti.run(parse("--toy", "--quick", "--candidates", "4",
                        "--calibrate-with", decoded,
                        "--output", out)) == 0
    art = json.load(open(out))
    for setting in ("summary", "no-summary"):
        un = _cell(art, setting, False, 4, False)
        cal = _cell(art, setting, True, 4, False)
        assert cal["tf1"] > un["tf1"] + 0.05
        assert cal["rouge_l"] > un["rouge_l"]
    support = art["config"]["calibration_support"]
    assert support["n_traces"] == 10


def test_calibrate_with_schema_drift_is_descriptive(toy_run, tmp_path):
    bad = _write_decoded(tmp_path / "bad.jsonl", ["x"])
    with open(bad, "w") as f:
        f.write(json.dumps({"text": "no decoded_text key"}) + "\n")
    with pytest.raises(SystemExit, match="decoded_text"):
        ti.run(parse("--toy", "--quick", "--calibrate-with", bad,
                     "--output", str(tmp_path / "z.json")))


def test_non_toy_is_the_documented_integration_point():
    with pytest.raises(SystemExit, match="--toy"):
        ti.run(parse())


def test_evidence_flag_wires_evidence_logger(tmp_path):
    from attacker.evidence import verify_evidence
    out = str(tmp_path / "ev.json")
    assert ti.run(parse("--toy", "--quick", "--evidence",
                        "--output", out)) == 0
    ok, problems = verify_evidence(out + ".evidence.json")
    assert ok and problems == []
    with pytest.raises(SystemExit, match="--evidence requires --output"):
        ti.run(parse("--toy", "--quick", "--evidence"))


def test_journal_lines_match_result_cells(toy_run):
    out, art = toy_run
    lines = open(out + ".jsonl").read().strip().splitlines()
    assert len(lines) == len(art["results"])
    assert all("error" not in json.loads(l) for l in lines)


def test_module_self_test_passes():
    assert ti.self_test() == 0
