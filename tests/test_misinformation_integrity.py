"""misinformation-integrity: arm coverage, defended/vulnerable contrast,
corpus/answer scoring, errors, determinism, artifact layout."""

import json

import pytest

from attacker import threat_model
from attacker.attacks import misinformation_integrity as mi


def _run(tmp_path, extra=()):
    out = tmp_path / "mi.json"
    args = mi.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(out), *extra])
    args.mode = "inference"
    assert mi.run(args) == 0
    return out, json.loads(out.read_text())


def test_contract_members():
    assert mi.EXPERIMENT_ID == "misinformation_integrity"
    assert mi.MODES == ("inference",)
    assert mi.REQUIRES_LABELS


def test_toy_artifact_layout_and_journal(tmp_path):
    out, art = _run(tmp_path)
    assert art["schema"] == mi.ARTIFACT_SCHEMA
    threat_model.validate(art["threat_model"])
    lines = (tmp_path / "mi.json.jsonl").read_text().strip().splitlines()
    assert len(lines) == len(art["results"])
    assert all(json.loads(l)["experiment"] == mi.EXPERIMENT_ID
               for l in lines)


def test_every_arm_recorded_in_both_worlds(tmp_path):
    _, art = _run(tmp_path)
    for arm in mi.ARMS:
        worlds = {r["world"] for r in art["results"] if r["arm"] == arm}
        assert worlds == {"vulnerable", "defended"}, arm


def test_defended_world_holds(tmp_path):
    _, art = _run(tmp_path)
    for r in art["results"]:
        if r.get("world") != "defended" or "error" in r:
            continue
        for key in ("premise_acceptance_rate", "fabricated_citation_rate",
                    "stale_answer_rate", "contradiction_rate",
                    "action_harm_rate"):
            if key in r:
                assert r[key] == 0.0, (r["arm"], key, r[key])


def test_vulnerable_world_shows_failures(tmp_path):
    _, art = _run(tmp_path)
    vuln = [r for r in art["results"] if r.get("world") == "vulnerable"]
    assert any(r.get("stale_answer_rate", 0) > 0 for r in vuln)
    assert any(r.get("fabricated_citation_rate", 0) > 0 for r in vuln)
    assert any(r.get("contradiction_rate", 0) > 0 for r in vuln)


def test_negative_control_no_false_refusals(tmp_path):
    _, art = _run(tmp_path)
    ctrl = [r for r in art["results"] if r["arm"] == "negative-control"]
    assert len(ctrl) == 1
    assert ctrl[0]["defended_false_refusals"] == 0
    assert ctrl[0]["benign_trials"] > 0


def test_real_corpus_and_answer_scoring(tmp_path):
    corpus = tmp_path / "c.jsonl"
    corpus.write_text(json.dumps(
        {"subject": "itemX", "attribute": "price", "value": "42",
         "valid_from": 0, "valid_to": 10**9,
         "citation": "DocX states itemX has value 42."}) + "\n")
    answers = tmp_path / "a.jsonl"
    answers.write_text(
        json.dumps({"question": "what is the value of itemX?",
                    "answer": "itemX has value 42.", "confidence": 0.9,
                    "citation": "DocX states itemX has value 42."}) + "\n"
        + json.dumps({"question": "what is the value of itemX?",
                      "answer": "itemX has value 7.", "confidence": 0.95,
                      "citation": "DocY (fabricated)"}) + "\n")
    out, art = _run(tmp_path, ["--corpus", str(corpus),
                               "--system", str(answers)])
    real = [r for r in art["results"] if r["arm"] == "real-corpus"]
    assert real and real[0]["scored"]["answers"] == 2
    assert real[0]["scored"]["accuracy"] == 0.5
    assert real[0]["scored"]["fabricated_citation_rate"] == 0.5


def test_corpus_and_answer_drift(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"subject": "x"}\n')
    args = mi.build_parser().parse_args(
        ["--toy", "--quick", "--corpus", str(bad)])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="bad.jsonl"):
        mi.run(args)
    args = mi.build_parser().parse_args(
        ["--toy", "--quick", "--system", str(bad)])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="requires --corpus"):
        mi.run(args)


def test_determinism(tmp_path):
    _, a = _run(tmp_path)
    out2 = tmp_path / "mi2.json"
    args = mi.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(out2)])
    args.mode = "inference"
    mi.run(args)
    assert a["results"] == json.loads(out2.read_text())["results"]


def test_self_test_passes():
    assert mi.self_test() == 0
