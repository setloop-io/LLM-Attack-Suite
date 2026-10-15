"""The combined design, end to end (README.md):
reasoning-replay --toy decodes genuine victim traces into
<output>.decoded.jsonl; trace-inversion --calibrate-with consumes them and
must measurably beat the uncalibrated surrogate-style baseline. Runs both
module run() functions into tmp_path, torch-free, < 60 s total."""

import json
import time

import pytest

from attacker.attacks import reasoning_replay as rr
from attacker.attacks import trace_inversion as ti
from attacker.evidence import verify_evidence


def _run_replay(out):
    args = rr.build_parser().parse_args(
        ["--toy", "--quick", "--evidence", "--output", out])
    args.mode = "inference"
    assert rr.run(args) == 0
    return out


def _run_inversion(out, decoded):
    args = ti.build_parser().parse_args(
        ["--toy", "--quick", "--candidates", "4", "--refine",
         "--calibrate-with", decoded, "--evidence", "--output", out])
    args.mode = "inference"
    assert ti.run(args) == 0
    return out


def _cell(art, setting, calibrated, k, refined):
    hits = [r for r in art["results"]
            if r["setting"] == setting and r["calibrated"] == calibrated
            and r["candidates"] == k and r["refined"] == refined]
    assert len(hits) == 1
    return hits[0]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    d = tmp_path_factory.mktemp("combined")
    rr_out, ti_out = str(d / "rr.json"), str(d / "ti.json")
    t0 = time.monotonic()
    _run_replay(rr_out)
    _run_inversion(ti_out, rr_out + ".decoded.jsonl")
    elapsed = time.monotonic() - t0
    return {"rr_out": rr_out, "ti_out": ti_out, "elapsed": elapsed,
            "rr": json.load(open(rr_out)), "ti": json.load(open(ti_out))}


def test_pipeline_completes_well_under_a_minute(pipeline):
    assert pipeline["elapsed"] < 60.0


def test_replay_writes_the_calibration_artifact(pipeline):
    decoded_path = pipeline["rr_out"] + ".decoded.jsonl"
    decoded = [json.loads(l) for l in open(decoded_path)]
    assert len(decoded) == 18
    assert all(isinstance(r["decoded_text"], str) and r["decoded_text"]
               for r in decoded)
    assert pipeline["ti"]["config"]["calibrate_with"] == decoded_path
    assert pipeline["ti"]["config"]["calibration_support"]["n_traces"] == 18


def test_calibrated_beats_uncalibrated_tf1_by_margin(pipeline):
    art = pipeline["ti"]
    for setting in ("summary", "no-summary"):
        for k in (1, 4):
            un = _cell(art, setting, False, k, False)
            cal = _cell(art, setting, True, k, False)
            assert cal["tf1"] > un["tf1"] + 0.05, (
                setting, k, un["tf1"], cal["tf1"])
            assert cal["rouge_l"] >= un["rouge_l"]


def test_best_of_4_beats_best_of_1_answer_consistency(pipeline):
    art = pipeline["ti"]
    one = _cell(art, "summary", False, 1, False)
    four = _cell(art, "summary", False, 4, False)
    assert four["answer_consistency"] > one["answer_consistency"]
    assert four["answer_consistency"] == 1.0
    refined = _cell(art, "summary", False, 1, True)
    assert refined["answer_consistency"] > one["answer_consistency"]


def test_summary_setting_beats_no_summary(pipeline):
    art = pipeline["ti"]
    for k in (1, 4):
        s = _cell(art, "summary", False, k, False)
        n = _cell(art, "no-summary", False, k, False)
        assert s["tf1"] > n["tf1"]


def test_capability_excess_gated_everywhere(pipeline):
    for rec in pipeline["ti"]["results"]:
        assert rec["gate"]["verdict"] in ("PASS", "FAIL")
    main = _cell(pipeline["ti"], "summary", True, 4, False)
    assert main["capability_excess_pp"] > 50.0
    assert main["gate"]["verdict"] == "FAIL"


def test_both_evidence_chains_verify(pipeline):
    for out in (pipeline["rr_out"], pipeline["ti_out"]):
        ok, problems = verify_evidence(out + ".evidence.json")
        assert ok and problems == []


def test_journals_match_results(pipeline):
    for out, art in ((pipeline["rr_out"], pipeline["rr"]),
                     (pipeline["ti_out"], pipeline["ti"])):
        lines = [json.loads(l) for l in open(out + ".jsonl")]
        assert len(lines) == len(art["results"])
        assert all("error" not in l for l in lines)
