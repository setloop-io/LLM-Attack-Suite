"""kv-timing spec-decode arm (speculative-decoding acceptance-pattern
membership channel): toy planted/null worlds end-to-end, offline journal
path + schema drift, LIVE localhost servers that expose (or hide)
per-token acceptance, --member ground truth, dispatch, and the graceful
INFO-cell contract. All torch-free."""

import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from attacker.attacks import kv_timing


@pytest.fixture
def registered():
    """Dispatch through the real registry without editing ATTACKS."""
    from attacker import attacks
    mod = importlib.import_module("attacker.attacks.kv_timing")
    registered_here = "kv-timing" not in attacks.ATTACKS
    if registered_here:
        attacks.register_attack("kv-timing", mod)
    yield mod
    if registered_here:
        attacks.ATTACKS.pop("kv-timing", None)


def _spec_payload(prompt):
    if prompt.startswith("member"):
        accepted = [True] * 14 + [False] * 2
    else:
        accepted = [i % 2 == 0 for i in range(16)]
    latencies = [1.0 if a else 7.0 for a in accepted]
    return accepted, latencies


class _SpecDecodeHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            body = {}
        accepted, latencies = _spec_payload(body.get("prompt", ""))
        payload = json.dumps({
            "id": "cmpl-spec", "choices": [{"text": " ok"}],
            "spec_decode": {"accepted": accepted,
                            "latencies_ms": latencies}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class _ChoicesConventionHandler(_SpecDecodeHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            body = {}
        accepted, _ = _spec_payload(body.get("prompt", ""))
        payload = json.dumps({
            "id": "cmpl-spec",
            "choices": [{"text": " ok",
                         "accepted_tokens": accepted}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _PlainHandler(_SpecDecodeHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        payload = json.dumps({"id": "cmpl-plain",
                              "choices": [{"text": " ok"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture(params=("_SpecDecodeHandler", "_ChoicesConventionHandler",
                        "_PlainHandler"))
def spec_server(request):
    handler = globals()[request.param]
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield request.param, \
            f"http://127.0.0.1:{srv.server_address[1]}/v1/completions"
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def _run(args_list, tmp_path, name="spec.json"):
    args = kv_timing.build_parser().parse_args(
        [*args_list, "--output", str(tmp_path / name)])
    args.mode = "inference"
    assert kv_timing.run(args) == 0
    art = json.loads((tmp_path / name).read_text())
    journal = [json.loads(l) for l in
               (tmp_path / (name + ".jsonl")).read_text().splitlines()]
    return art, journal


def test_arm_flag_and_help_torchless():
    ap = kv_timing.build_parser()
    with pytest.raises(SystemExit) as exc:
        ap.parse_args(["--help"])
    assert exc.value.code == 0
    args = ap.parse_args([])
    assert args.arm == "prefix-cache"
    args = ap.parse_args(["--arm", "spec-decode"])
    assert args.arm == "spec-decode"
    assert args.spec_candidates == 16 and args.members == 8
    assert args.positions == 4096 and args.span == 2048


def test_self_test_passes():
    assert kv_timing.self_test() == 0


def test_toy_end_to_end(tmp_path):
    art, journal = _run(["--arm", "spec-decode", "--toy", "--quick"],
                        tmp_path)
    assert art["config"]["arm"] == "spec-decode"
    assert art["config"]["source"] == "toy"
    recs = art["results"]
    assert len(recs) == len(journal) == 4
    planted = [r for r in recs if r["world"] == "planted"]
    null = [r for r in recs if r["world"] == "null"]
    for r in planted:
        assert r["acceptance_auc"] == 1.0
        assert r["member_acceptance_mean"] \
            > r["nonmember_acceptance_mean"] + 0.4
        assert r["latency_auc"] > 0.9
        assert r["gate"]["verdict"] == "FAIL"
        assert r["membership_channel"] == "open"
        assert r["gate"]["n"] == r["n_gate_tokens"]
    for r in null:
        assert r["gate"]["verdict"] == "INSUFFICIENT_EVIDENCE"
        assert r["membership_channel"] == "unmeasured"
        assert abs(r["member_acceptance_mean"]
                   - r["nonmember_acceptance_mean"]) < 0.05
    summ = {(s["world"]): s for s in art["summary"]}
    assert summ["planted"]["membership_channel"] == "open"
    assert summ["planted"]["gate_verdicts"] == ["FAIL", "FAIL"]
    assert summ["null"]["membership_channel"] == "unmeasured"
    sidecar = art["spec_journal"]
    assert sidecar["schema"] == kv_timing.SPEC_JOURNAL_SCHEMA
    lines = (tmp_path / "spec.json.spec.jsonl").read_text().splitlines()
    assert len(lines) == sidecar["records"] == 2 * 2 * 16 * 2048
    first = json.loads(lines[0])
    assert {"candidate", "position", "accepted", "latency_ms", "member"} \
        <= set(first)


def test_offline_journal_path(tmp_path):
    records = kv_timing.toy_spec_records(
        candidates=6, members=2, positions=512, span=256, p_hi=0.95,
        p_lo=0.2, base_ms=2.0, penalty_ms=6.0, noise_ms=0.5, seed=7)
    jf = tmp_path / "acceptance.jsonl"
    jf.write_text("".join(json.dumps(r) + "\n" for r in records))
    art, journal = _run(["--arm", "spec-decode", "--journal", str(jf)],
                        tmp_path, name="j.json")
    assert art["config"]["source"] == "journal"
    stats = [r for r in art["results"] if "acceptance_rate" in r]
    assert len(stats) == 6
    by_cand = {r["candidate"]: r for r in stats}
    assert by_cand["candidate-0"]["member"] is True
    assert by_cand["candidate-0"]["acceptance_rate"] \
        > by_cand["candidate-5"]["acceptance_rate"] + 0.3
    cell = art["results"][-1]
    assert cell["acceptance_auc"] == 1.0
    assert cell["gate"]["verdict"] == "FAIL"
    assert art["summary"][0]["membership_scored"] is True
    assert art["summary"][0]["membership_channel"] == "open"


def test_offline_journal_without_labels_is_info_cell(tmp_path):
    records = kv_timing.toy_spec_records(
        candidates=4, members=2, positions=64, span=32, p_hi=0.9,
        p_lo=0.2, base_ms=2.0, penalty_ms=6.0, noise_ms=0.5, seed=3)
    for r in records:
        del r["member"]
    jf = tmp_path / "unlabeled.jsonl"
    jf.write_text("".join(json.dumps(r) + "\n" for r in records))
    art, _ = _run(["--arm", "spec-decode", "--journal", str(jf)],
                  tmp_path, name="u.json")
    cell = art["results"][-1]
    assert "info" in cell and "gate" not in cell
    assert "member labels" in cell["info"]
    assert art["summary"][0]["membership_scored"] is False
    for r in records:
        r["latency_ms"] = None
    jf.write_text("".join(json.dumps(r) + "\n" for r in records))
    art2, _ = _run(["--arm", "spec-decode", "--journal", str(jf)],
                   tmp_path, name="u2.json")
    stat = art2["results"][0]
    assert stat["mean_latency_ms"] is None
    assert stat["acceptance_rate"] > 0.0


def test_journal_schema_drift_is_descriptive(tmp_path):
    jf = tmp_path / "bad.jsonl"
    jf.write_text(json.dumps({"candidate": "x", "position": 0}) + "\n")
    with pytest.raises(ValueError, match="accepted"):
        kv_timing.read_spec_journal(str(jf))
    jf.write_text(json.dumps({"candidate": "x", "position": 0,
                              "accepted": "yes"}) + "\n")
    with pytest.raises(ValueError, match="bool"):
        kv_timing.read_spec_journal(str(jf))
    jf.write_text(json.dumps({"candidate": "x", "position": 0,
                              "accepted": True, "member": "y"}) + "\n")
    with pytest.raises(ValueError, match="member"):
        kv_timing.read_spec_journal(str(jf))
    jf.write_text("{not json\n")
    with pytest.raises(ValueError, match="not JSON"):
        kv_timing.read_spec_journal(str(jf))


def test_extract_acceptance_conventions():
    assert kv_timing.extract_acceptance(
        {"spec_decode": {"accepted": [True, False],
                         "latencies_ms": [1.0, 7.0]}}) \
        == ([True, False], [1.0, 7.0])
    assert kv_timing.extract_acceptance(
        {"spec_decode": {"accepted": [True]}}) == ([True], None)
    assert kv_timing.extract_acceptance(
        {"choices": [{"accepted_tokens": [False, True]}]}) \
        == ([False, True], None)
    assert kv_timing.extract_acceptance({"choices": [{"text": "x"}]}) is None
    assert kv_timing.extract_acceptance("garbage") is None


def test_live_servers(registered, spec_server, tmp_path):
    """The genuine live path: real HTTP POSTs against servers that expose
    per-token acceptance (two documented conventions) or hide it."""
    kind, endpoint = spec_server
    out = tmp_path / f"live_{kind}.json"
    from attacker.__main__ import main
    rc = main(["--mode", "inference", "--attack", "kv-timing",
               "--arm", "spec-decode", "--endpoint", endpoint,
               "--candidate", "member boilerplate system prompt",
               "--candidate", "member second boilerplate",
               "--candidate", "novel prefix alpha",
               "--candidate", "novel prefix beta",
               "--member", "member boilerplate system prompt",
               "--member", "member second boilerplate",
               "--output", str(out)])
    assert rc == 0
    art = json.loads(out.read_text())
    assert art["config"]["source"] == "live"
    stats = [r for r in art["results"] if "acceptance_rate" in r]
    if kind == "_PlainHandler":
        infos = [r for r in art["results"] if "info" in r]
        assert len(infos) >= 4
        assert all(r["acceptance_exposed"] is False for r in infos[:4])
        assert "not measurable" in infos[0]["info"]
        assert not (tmp_path / f"live_{kind}.json.spec.jsonl").exists()
        assert art["summary"][0]["n_exposed"] == 0
        assert art["summary"][0]["membership_scored"] is False
        return
    assert len(stats) == 4
    by_cand = {r["candidate"]: r for r in stats}
    assert by_cand["member boilerplate system prompt"]["member"] is True
    assert by_cand["member boilerplate system prompt"]["acceptance_rate"] \
        == pytest.approx(0.875)
    assert by_cand["novel prefix alpha"]["acceptance_rate"] == 0.5
    cell = art["results"][-1]
    assert cell["acceptance_auc"] == 1.0
    assert cell["gate"]["verdict"] == "FAIL"
    assert cell["membership_channel"] == "open"
    if kind == "_SpecDecodeHandler":
        assert cell["latency_auc"] == 1.0
        assert by_cand["novel prefix beta"]["mean_latency_ms"] is not None
    else:
        assert cell["latency_auc"] is None
    sidecar = art["spec_journal"]
    assert sidecar["schema"] == kv_timing.SPEC_JOURNAL_SCHEMA
    assert sidecar["records"] == 4 * 16
    assert (tmp_path / f"live_{kind}.json.spec.jsonl").is_file()


def test_live_without_member_labels_is_stats_only(registered, tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _SpecDecodeHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{srv.server_address[1]}/v1/completions"
        art, _ = _run(["--arm", "spec-decode", "--endpoint", endpoint,
                       "--candidate", "member anything"], tmp_path,
                      name="nolabel.json")
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)
    cell = art["results"][-1]
    assert "info" in cell and "gate" not in cell
    assert art["summary"][0]["membership_scored"] is False


def test_live_unreachable_endpoint_journals_error(registered, tmp_path):
    out = tmp_path / "dead.json"
    from attacker.__main__ import main
    rc = main(["--mode", "inference", "--attack", "kv-timing",
               "--arm", "spec-decode",
               "--endpoint", "http://127.0.0.1:9/v1/completions",
               "--timeout", "0.5", "--candidate", "anything",
               "--output", str(out)])
    assert rc == 0
    art = json.loads(out.read_text())
    assert all("acceptance_rate" not in r for r in art["results"])
    journal = [json.loads(l) for l in
               (tmp_path / "dead.json.jsonl").read_text().splitlines()]
    assert any("error" in j for j in journal)


def test_member_must_name_a_candidate(tmp_path):
    args = kv_timing.build_parser().parse_args(
        ["--arm", "spec-decode", "--endpoint", "http://127.0.0.1:9/",
         "--candidate", "a", "--member", "b",
         "--output", str(tmp_path / "x.json")])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="--candidate"):
        kv_timing.run(args)


def test_members_must_be_fewer_than_candidates(tmp_path):
    args = kv_timing.build_parser().parse_args(
        ["--arm", "spec-decode", "--toy", "--spec-candidates", "2",
         "--members", "2", "--output", str(tmp_path / "x.json")])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="--members"):
        kv_timing.run(args)


def test_spec_arm_requires_an_input_path():
    args = kv_timing.build_parser().parse_args(["--arm", "spec-decode"])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="--endpoint"):
        kv_timing.run(args)


def test_default_arm_unaffected(tmp_path):
    """No --arm: the prefix-cache toy behaves exactly as before."""
    art, _ = _run(["--toy", "--quick"], tmp_path, name="pc.json")
    assert art["config"]["arm"] == "prefix-cache"
    assert art["config"]["source"] == "toy"
    assert art["summary"][0]["toy_member_detected"] is True


def test_dispatch_toy_end_to_end(registered, tmp_path):
    from attacker.__main__ import main
    out = tmp_path / "disp.json"
    rc = main(["--mode", "inference", "--attack", "kv-timing",
               "--arm", "spec-decode", "--toy", "--quick",
               "--output", str(out)])
    assert rc == 0
    art = json.loads(out.read_text())
    assert art["config"]["arm"] == "spec-decode"
    assert art["results"] and art["summary"]
    journal = (tmp_path / "disp.json.jsonl").read_text().strip().splitlines()
    assert len(journal) == len(art["results"])
    assert all(json.loads(l)["experiment"] == kv_timing.EXPERIMENT_ID
               for l in journal)


def test_evidence_chain(tmp_path):
    import os
    from attacker.evidence import verify_evidence
    out = tmp_path / "ev.json"
    args = kv_timing.build_parser().parse_args(
        ["--arm", "spec-decode", "--toy", "--quick", "--evidence",
         "--output", str(out)])
    args.mode = "inference"
    assert kv_timing.run(args) == 0
    assert os.path.isfile(str(out) + ".evidence.json")
    ok, problems = verify_evidence(str(out) + ".evidence.json")
    assert ok and problems == []
