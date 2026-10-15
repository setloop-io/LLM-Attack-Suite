"""reasoning-replay: blob codec, toy compatibility-matrix scanner,
self-consistency extraction + faithfulness, secrets pack, inject arm, live
adapter degradation, api-transcript surface drift errors, MITM blob
harvesting, and registry dispatch via attacks.register_attack (the parent
wires the built-in registry entry separately). Torch-free throughout."""

import json

import pytest

from attacker import attacks, transcripts
from attacker.attacks import reasoning_replay as rr
from attacker.attacks import trace_inversion as ti


def parse(*argv):
    args = rr.build_parser().parse_args(list(argv))
    args.mode = "inference"
    return args


@pytest.fixture
def cleanup_registry():
    before = set(attacks.ATTACKS)
    yield
    for name in set(attacks.ATTACKS) - before:
        attacks.ATTACKS.pop(name, None)


def test_contract_and_dispatch_via_register_attack(cleanup_registry):
    attacks.validate_contract(rr, "reasoning-replay")
    assert rr.EXPERIMENT_ID == "reasoning_replay"
    assert rr.MODES == ("inference",)
    assert rr.REQUIRES_LABELS is False
    if "reasoning-replay" not in attacks.ATTACKS:
        attacks.register_attack("reasoning-replay", rr)
    assert attacks.load("reasoning-replay") is rr
    assert "reasoning-replay" in attacks.for_mode("inference")
    assert "reasoning-replay" not in attacks.for_mode("training")


def test_parser_builds_torchless():
    args = rr.build_parser().parse_args([])
    assert args.arm == ["all"] and args.reps == 4
    assert args.transcripts is None and args.evidence is False


def test_surface_registered_and_validates():
    from attacker.surfaces import get_surface, list_surfaces
    prov = get_surface("api-transcript")
    assert "api-transcript" in list_surfaces()
    assert prov.modes == ("inference",)
    rec = transcripts.make_record("s", 0, "assistant", "hi",
                                  blobs=["b"], thinking_tokens=3,
                                  ts=1700000000)
    assert rec["ts"] == 1700000000.0


@pytest.mark.parametrize("bad,match", [
    ({"session_id": "s", "turn": 0, "role": "user", "text": "x",
      "blobs": [], "usage": {"thinking_tokens": None}}, "record keys"),
    ({"session_id": "s", "turn": 0, "role": "admin", "text": "x",
      "blobs": [], "usage": {"thinking_tokens": None}, "ts": 1.0},
     "role"),
    ({"session_id": "s", "turn": 0, "role": "user", "text": "x",
      "blobs": [7], "usage": {"thinking_tokens": None}, "ts": 1.0},
     "blobs"),
    ({"session_id": "s", "turn": 0, "role": "user", "text": "x",
      "blobs": [], "usage": {}, "ts": 1.0}, "thinking_tokens"),
    ({"session_id": "s", "turn": 0, "role": "user", "text": "x",
      "blobs": [], "usage": {"thinking_tokens": None}, "ts": "x"},
     "ts"),
])
def test_transcript_schema_drift_is_descriptive(tmp_path, bad, match):
    path = tmp_path / "drift.jsonl"
    path.write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match=match) as exc:
        transcripts.load_transcripts(str(path))
    assert f"{path}:1" in str(exc.value)


def test_load_transcripts_sorts_and_extract_blobs(tmp_path):
    path = tmp_path / "t.jsonl"
    recs = [transcripts.make_record("s0", 1, "assistant", "The answer is 3.",
                                    blobs=["B"], thinking_tokens=5, ts=2.0),
            transcripts.make_record("s0", 0, "user", "Compute 1 + 2.",
                                    ts=1.0)]
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    loaded = transcripts.load_transcripts(str(path))
    assert [r["turn"] for r in loaded] == [0, 1]
    flat = transcripts.extract_blobs(loaded)
    assert len(flat) == 1 and flat[0]["blob"] == "B"
    assert flat[0]["thinking_tokens"] == 5


def test_blob_codec_roundtrip_and_malformed():
    blob = rr.encode_blob("Step 1: add 2 and 3 to get 5.", "gpt-like")
    assert rr.decode_blob(blob) == ("gpt-like",
                                    "Step 1: add 2 and 3 to get 5.")
    with pytest.raises(ValueError, match="malformed toy blob"):
        rr.decode_blob("garbage")
    with pytest.raises(ValueError, match="undecodable"):
        rr.decode_blob("RTBLOB1.x.###notb64###")


def test_toy_decode_honors_planted_matrix():
    genuine = "Let's compute 3 + 4 * 5. Step 1: add 3 and 4 to get 7. " \
              "Step 2: multiply 7 and 5 to get 35. The final answer is 35."
    blob = rr.encode_blob(genuine, "claude-like")
    good = rr.toy_decode(blob, "claude-like", "current-turn", seed=0)
    assert rr.transcription_rate(good["text"], genuine) > 0.9
    blob_g = rr.encode_blob(genuine, "gpt-like")
    refused = rr.toy_decode(blob_g, "claude-like", "current-turn", seed=0)
    assert refused["refused"] and refused["text"] == rr.REFUSAL_TEXT
    partial = rr.toy_decode(rr.encode_blob(genuine, "gemini-like"),
                            "gemini-like", "current-turn", seed=0)
    assert len(partial["text"].split()) < len(genuine.split())


@pytest.fixture
def scan_artifact(tmp_path):
    out = str(tmp_path / "scan.json")
    assert rr.run(parse("--toy", "--quick", "--arm", "scan",
                        "--output", out)) == 0
    return json.load(open(out))


def test_scan_recovers_planted_matrix(scan_artifact):
    cells = [r for r in scan_artifact["results"] if r["arm"] == "scan"]
    assert len(cells) == 27
    zero = [c for c in cells if c["planted_fidelity"] == 0.0]
    full = [c for c in cells if c["planted_fidelity"] >= 0.9
            and c["planted_coverage"] == 1.0]
    partial = [c for c in cells if 0.0 < c["planted_coverage"] < 1.0]
    mean = lambda cs: sum(c["transcription_rate"] for c in cs) / len(cs)
    assert mean(full) > 0.85 and mean(partial) > 0.3
    assert mean(zero) < 0.01
    assert mean(full) > mean(partial) > mean(zero) + 0.2
    assert all(c["gate"]["verdict"] == "FAIL" for c in full + partial)
    assert all(c["gate"]["verdict"] == "INSUFFICIENT_EVIDENCE" for c in zero)
    assert all(c["gate"]["at_floor"] for c in zero)


def test_scan_matrix_shape_matches_paper_table1(scan_artifact):
    cells = [r for r in scan_artifact["results"] if r["arm"] == "scan"]
    rate = {(c["source"], c["decoder"], c["scheme"]):
            c["transcription_rate"] for c in cells}
    assert all(rate[("claude-like", d, s)] > 0.8
               for d in rr.DECODERS for s in rr.SCHEMES)
    assert rate[("gpt-like", "gpt-like", "current-turn")] > 0.8
    assert max(rate[("gpt-like", d, s)]
               for d in rr.DECODERS for s in rr.SCHEMES
               if (d, s) != ("gpt-like", "current-turn")) < 0.01
    assert all(0.2 < rate[("gemini-like", d, s)] < 0.9
               for d in rr.DECODERS for s in rr.SCHEMES
               if rr.compat("gemini-like", d, s)[0] > 0)


@pytest.fixture
def extract_run(tmp_path):
    out = str(tmp_path / "ext.json")
    assert rr.run(parse("--toy", "--quick", "--arm", "extract",
                        "--output", out)) == 0
    return out, json.load(open(out))


def test_extract_faithfulness_orders_fidelity(extract_run):
    _, art = extract_run
    cells = {c["source"]: c for c in art["results"] if c["arm"] == "extract"}
    hi, lo = cells["claude-like"], cells["gemini-like"]
    assert hi["kfold_agreement_mean"] > lo["kfold_agreement_mean"]
    assert hi["billing_match_rate"] == 1.0
    assert lo["billing_match_rate"] == 0.0
    assert hi["cross_decoder_agreement_mean"] is not None


def test_extract_selects_best_decoder_without_ground_truth(extract_run):
    out, _ = extract_run
    decoded = [json.loads(l) for l in open(out + ".decoded.jsonl")]
    assert len(decoded) == 18
    for rec in decoded:
        for key in ("session_id", "turn", "source_model", "decoder",
                    "scheme", "decoded_text", "thinking_tokens",
                    "faithfulness", "problem_text", "answer"):
            assert key in rec
    by_source = {}
    for rec in decoded:
        by_source.setdefault(rec["source_model"], set()).add(rec["decoder"])
    assert by_source["claude-like"] == {"claude-like"}
    assert by_source["gpt-like"] == {"gpt-like"}


def test_decoded_traces_are_near_verbatim_for_portable_sources(extract_run):
    out, _ = extract_run
    decoded = [json.loads(l) for l in open(out + ".decoded.jsonl")]
    claude = [r for r in decoded if r["source_model"] == "claude-like"]
    f1 = [ti.token_f1(r["decoded_text"], _genuine_of(r)) for r in claude]
    assert sum(f1) / len(f1) > 0.9


def _genuine_of(decoded_rec):
    """Reconstruct the genuine trace of a decoded record (toy ground truth
    travels inside the blob)."""
    p = ti.parse_problem(decoded_rec["problem_text"])
    steps = ti.compute_steps(p, ti.ORDER_LTR)
    return ti.render_trace(p, steps, ti.VICTIM_STYLE)


def test_secrets_find_canaries_zero_false_positives(tmp_path):
    out = str(tmp_path / "sec.json")
    assert rr.run(parse("--toy", "--quick", "--arm", "secrets",
                        "--output", out)) == 0
    rec = [r for r in json.load(open(out))["results"]
           if r["arm"] == "secrets"][0]
    found = set(rec["per_detector"])
    assert {"api_key", "email"} <= found
    assert rec["yield_per_10k_blocks"] > 0
    import random
    clean = [ti.victim_example(random.Random(i))["trace"] for i in range(30)]
    assert all(rr.scan_text(t) == [] for t in clean)
    rep = rr.secrets_report(clean)
    assert rep["n_detections"] == 0 and rep["yield_per_10k_blocks"] == 0.0


def test_entropy_and_password_detectors():
    assert any(h["detector"] == "password" for h in
               rr.scan_text("the config has password: hunter2 ok"))
    assert any(h["detector"] == "aws_access_key" for h in
               rr.scan_text("aws credential AKIAIOSFODNN7EXAMPLE inside"))
    high = "x" + "aB3dE5fG7hI9jK1lM3nO5pQ7rS9"
    assert rr.shannon_entropy(high) > rr.ENTROPY_MIN_BITS


def test_inject_reports_acceptance(tmp_path):
    out = str(tmp_path / "inj.json")
    assert rr.run(parse("--toy", "--quick", "--arm", "inject",
                        "--output", out)) == 0
    cells = [r for r in json.load(open(out))["results"]
             if r["arm"] == "inject"]
    assert len(cells) == 3
    by_dec = {c["decoder"]: c for c in cells}
    assert by_dec["claude-like"]["acceptance_rate"] > 0.5
    assert by_dec["claude-like"]["acceptance_rate"] \
        > by_dec["gemini-like"]["acceptance_rate"]
    assert all(c["gate"]["verdict"] in ("PASS", "FAIL", "INSUFFICIENT_EVIDENCE") for c in cells)
    assert by_dec["claude-like"]["gate"]["verdict"] == "FAIL"


def test_live_mode_adapter_statuses(tmp_path):
    journal = tmp_path / "live.jsonl"
    recs = [transcripts.make_record("s0", 0, "user", "Compute 1 + 2 * 3.",
                                    ts=1.0),
            transcripts.make_record("s0", 1, "assistant", "The answer is 7.",
                                    blobs=["RTBLOB1.x.y"],
                                    thinking_tokens=10, ts=2.0)]
    journal.write_text("".join(json.dumps(r) + "\n" for r in recs))
    out = str(tmp_path / "live.json")
    assert rr.run(parse("--transcripts", str(journal), "--arm", "scan",
                        "extract", "inject", "--output", out)) == 0
    cells = json.load(open(out))["results"]
    assert cells and all(c["verdict"] == "INFO" for c in cells)
    assert all(c["status"] in ("no credentials", "integration point")
               for c in cells)
    assert all("API_KEY" in c["detail"] or "endpoint" in c["detail"]
               for c in cells)


def test_no_input_path_is_a_descriptive_error():
    with pytest.raises(SystemExit, match="--toy"):
        rr.run(parse("--arm", "scan"))


def test_transcript_drift_live_is_a_clean_exit(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"session_id": "s"}) + "\n")
    with pytest.raises(SystemExit, match="record keys"):
        rr.run(parse("--transcripts", str(bad), "--output",
                     str(tmp_path / "x.json")))


def test_mitm_extract_blobs_from_payload_sidecars(tmp_path):
    from attacker.mitm import (MitmProxy, _EchoServer, _echo_roundtrip,
                               extract_blobs)
    journal = str(tmp_path / "cap.jsonl")
    envelope = json.dumps({"type": "thinking",
                           "thinking": "Step 1: add 2 and 3.",
                           "signature": "RTBLOB1.claude-like.ABC"})
    payload = (f"data: {envelope}\n\n".encode()
               + b"data: {\"type\": \"text\", \"text\": \"done\"}\n\n"
               + b"not json at all\n")
    with _EchoServer() as echo:
        with MitmProxy("127.0.0.1", 0, "127.0.0.1", echo.port, journal,
                       frame="raw") as proxy:
            assert _echo_roundtrip(proxy.port, payload) == payload
    hits = extract_blobs(journal)
    assert len(hits) == 2
    assert all(h["blob"] == "RTBLOB1.claude-like.ABC" for h in hits)
    assert {h["dir"] for h in hits} == {"c2s", "s2c"}
    assert all("signature" in h["key_path"] for h in hits)


def test_evidence_flag_wires_evidence_logger(tmp_path):
    from attacker.evidence import verify_evidence
    out = str(tmp_path / "ev.json")
    assert rr.run(parse("--toy", "--quick", "--evidence",
                        "--output", out)) == 0
    ok, problems = verify_evidence(out + ".evidence.json")
    assert ok and problems == []
    with pytest.raises(SystemExit, match="--evidence requires --output"):
        rr.run(parse("--toy", "--quick", "--evidence"))


def test_module_self_test_passes():
    assert rr.self_test() == 0


def test_transcripts_module_self_test_passes():
    assert transcripts.self_test() == 0
