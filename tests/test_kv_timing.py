"""kv-timing (KV/prefix-cache timing side channel): stats core anchors,
toy end-to-end, offline journal path, and a LIVE localhost server with an
artificial prefix cache exercising the real HTTP measurement path.
All torch-free."""

import importlib
import json
import threading
import time
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


class _PrefixCacheHandler(BaseHTTPRequestHandler):
    seen_prompts = set()
    warm_delay_s = 0.005
    cold_delay_s = 0.045

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            body = {}
        prompt = body.get("prompt", "")
        delay = (self.warm_delay_s if prompt in type(self).seen_prompts
                 else self.cold_delay_s)
        type(self).seen_prompts.add(prompt)
        time.sleep(delay)
        payload = json.dumps({"id": "cmpl-test",
                              "choices": [{"text": " ok"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def prefix_cache_server():
    _PrefixCacheHandler.seen_prompts = set()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _PrefixCacheHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/v1/completions"
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def _run(args_list, tmp_path, name="kv.json", mode="inference"):
    args = kv_timing.build_parser().parse_args(
        [*args_list, "--output", str(tmp_path / name)])
    args.mode = mode
    assert kv_timing.run(args) == 0
    art = json.loads((tmp_path / name).read_text())
    journal = [json.loads(l) for l in
               (tmp_path / (name + ".jsonl")).read_text().splitlines()]
    return art, journal


def test_contract_and_torchless_help():
    from attacker.attacks import validate_contract
    assert validate_contract(kv_timing) is kv_timing
    assert kv_timing.MODES == ("inference",)
    assert kv_timing.REQUIRES_LABELS is False
    ap = kv_timing.build_parser()
    with pytest.raises(SystemExit) as exc:
        ap.parse_args(["--help"])
    assert exc.value.code == 0


def test_self_test_passes():
    assert kv_timing.self_test() == 0


def test_t_test_against_t_table():
    assert abs(kv_timing.t_two_sided_p(2.0, 10.0) - 0.0734) < 5e-4
    assert kv_timing.t_two_sided_p(0.0, 15.0) == pytest.approx(1.0)
    assert kv_timing.t_two_sided_p(30.0, 8.0) < 1e-8
    ps = [kv_timing.t_two_sided_p(t, 20.0) for t in (0.5, 1.0, 2.0, 3.0)]
    assert ps == sorted(ps, reverse=True)


def test_welch_warm_cold_separation():
    warm = [18.0 + (i % 5) * 0.3 for i in range(20)]
    cold = [32.0 + (i % 5) * 0.3 for i in range(20)]
    rec = kv_timing.analyze_candidate("c", warm, cold, 0.05, "pytest")
    assert rec["verdict"] == "channel_open"
    assert rec["p_value"] < 1e-10
    assert rec["median_delta_ms"] == pytest.approx(14.0, abs=1.0)
    assert rec["cohens_d"] > 5.0
    flipped = kv_timing.analyze_candidate("c", cold, warm, 0.05, "pytest")
    assert flipped["verdict"] == "channel_closed"


def test_toy_end_to_end(tmp_path):
    art, journal = _run(["--toy", "--quick"], tmp_path)
    assert art["schema"] == "dtraining.attacker.kv_timing.v1"
    assert art["config"]["source"] == "toy"
    recs = art["results"]
    assert len(recs) == len(journal) == 4
    member = [r for r in recs if r["candidate"] == "candidate-0"]
    others = [r for r in recs if r["candidate"] != "candidate-0"]
    assert all(r["verdict"] == "channel_open" for r in member)
    assert all(r["verdict"] == "channel_closed" for r in others)
    assert all(r["median_delta_ms"] > 10.0 for r in member)
    summ = art["summary"][0]
    assert summ["toy_member_detected"] is True
    assert summ["toy_false_positive_cells"] == 0
    assert summ["channel"] == "open"
    sidecar = art["toy_journal"]
    assert sidecar["schema"] == kv_timing.JOURNAL_SCHEMA
    lines = (tmp_path / "kv.json.timings.jsonl").read_text().splitlines()
    assert len(lines) == sidecar["records"] == 2 * 2 * 12 * 2
    first = json.loads(lines[0])
    assert {"candidate", "condition", "ttft_ms", "total_ms", "rep"} \
        <= set(first)


def test_offline_journal_path(tmp_path):
    """A hand-written warm/cold journal drives the same analyzer."""
    records = kv_timing.toy_timing_records(
        candidates=2, reps=20, member=0, warm_ms=15.0, cold_ms=40.0,
        noise_ms=2.0, seed=7)
    jf = tmp_path / "timings.jsonl"
    jf.write_text("".join(json.dumps(r) + "\n" for r in records))
    art, journal = _run(["--journal", str(jf)], tmp_path, name="j.json")
    assert art["config"]["source"] == "journal"
    assert art["config"]["journal"] == str(jf)
    by_cand = {r["candidate"]: r for r in art["results"]}
    assert by_cand["candidate-0"]["verdict"] == "channel_open"
    assert by_cand["candidate-1"]["verdict"] == "channel_closed"
    assert by_cand["candidate-0"]["n_warm"] == 20


def test_journal_schema_drift_is_descriptive(tmp_path):
    jf = tmp_path / "bad.jsonl"
    jf.write_text(json.dumps({"candidate": "x", "condition": "lukewarm",
                              "ttft_ms": 1.0}) + "\n")
    with pytest.raises(ValueError, match="condition"):
        kv_timing.read_timing_journal(str(jf))
    jf.write_text(json.dumps({"candidate": "x"}) + "\n")
    with pytest.raises(ValueError, match="ttft_ms"):
        kv_timing.read_timing_journal(str(jf))


def test_live_localhost_server(registered, prefix_cache_server, tmp_path):
    """The genuine live path: real HTTP POSTs against a prefix-caching
    server on 127.0.0.1 — warm probes must be measurably faster."""
    out = tmp_path / "live.json"
    from attacker.__main__ import main
    rc = main(["--mode", "inference", "--attack", "kv-timing",
               "--endpoint", prefix_cache_server, "--reps", "8",
               "--candidate", "the victim deployment's secret system prompt",
               "--output", str(out)])
    assert rc == 0
    art = json.loads(out.read_text())
    assert art["config"]["source"] == "live"
    (rec,) = art["results"]
    assert rec["n_warm"] == 8 and rec["n_cold"] == 8
    assert rec["verdict"] == "channel_open"
    assert rec["median_delta_ms"] > 15.0
    assert rec["p_value"] < 0.05
    assert rec["median_warm_ms"] < rec["median_cold_ms"]
    assert art["summary"][0]["channel"] == "open"


def test_live_unreachable_endpoint_journals_error(registered, tmp_path):
    out = tmp_path / "dead.json"
    from attacker.__main__ import main
    rc = main(["--mode", "inference", "--attack", "kv-timing",
               "--endpoint", "http://127.0.0.1:9/v1/completions",
               "--reps", "3", "--timeout", "0.5",
               "--candidate", "anything", "--output", str(out)])
    assert rc == 0
    art = json.loads(out.read_text())
    assert art["results"] == []
    journal = [json.loads(l) for l in
               (tmp_path / "dead.json.jsonl").read_text().splitlines()]
    assert len(journal) == 1 and "error" in journal[0]


def test_no_input_path_is_descriptive():
    args = kv_timing.build_parser().parse_args([])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="--endpoint"):
        kv_timing.run(args)
