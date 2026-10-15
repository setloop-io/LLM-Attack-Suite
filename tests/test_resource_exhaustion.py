"""resource-exhaustion (availability/cost/fairness stress tests, OWASP
LLM06): module contract, toy end-to-end + artifact/journal layout, the
mandatory negative control (benign-only load: parity ~ 1, ~0 fairness
degradation), the vulnerable world (attack arms measurably work), hard
caps (memory invariant + billing auto-stop), the --live safety envelope
and the live network driver (measures real cost/latency amplification
against a local stdlib http.server stub, enforces the hard caps as
auto-stops), malformed input, determinism, and the self-test. All
torch-free."""

import contextlib
import http.server
import json
import socket
import threading
import time

import pytest

from attacker import threat_model
from attacker.attacks import resource_exhaustion as rx


def _make_stub(per_token_ms=0.0, status_code=200):
    """Return (handler_cls, counter): the stub sleeps per requested output
    token (a decode-cost proxy) and echoes `usage.completion_tokens =
    max_tokens`, so a bigger --max-tokens request costs proportionally more
    latency AND tokens — a real, deterministic amplification signal."""
    counter = {"n": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {}
            max_tokens = int(payload.get("max_tokens") or 1)
            counter["n"] += 1
            if per_token_ms:
                time.sleep(max_tokens * per_token_ms / 1000.0)
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            if status_code == 200:
                body = json.dumps({
                    "choices": [{"text": "x"}],
                    "usage": {"prompt_tokens": len(payload.get("prompt", "")),
                              "completion_tokens": max_tokens,
                              "total_tokens": max_tokens}}).encode()
            else:
                body = b'{"error": "stub"}'
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler, counter


@contextlib.contextmanager
def _serve(handler_cls):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/v1/completions"
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2)


def _closed_port():
    """A loopback port with nothing listening (bound then released)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _toy_args(tmp_path, *extra, name="rx.json"):
    args = rx.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(tmp_path / name), *extra])
    args.mode = "inference"
    assert rx.run(args) == 0
    art = json.loads((tmp_path / name).read_text())
    journal = [json.loads(l) for l in
               (tmp_path / (name + ".jsonl")).read_text().splitlines()]
    return art, journal


def test_contract_and_torchless_help():
    from attacker.attacks import validate_contract
    assert validate_contract(rx) is rx
    assert rx.EXPERIMENT_ID == "resource_exhaustion"
    assert rx.MODES == ("training", "inference")
    assert rx.REQUIRES_LABELS is False
    assert isinstance(rx.DESCRIPTION, str) and rx.DESCRIPTION
    ap = rx.build_parser()
    with pytest.raises(SystemExit) as exc:
        ap.parse_args(["--help"])
    assert exc.value.code == 0
    args = ap.parse_args(["--toy", "--quick", "--arms", "long-context",
                          "training-skew", "--budgets", "4", "16",
                          "--rate", "3.0"])
    assert args.arms == ["long-context", "training-skew"]
    assert args.budgets == [4, 16] and args.rate == 3.0
    args = ap.parse_args(["--live", "--endpoint", "http://h:8000/v1",
                          "--max-requests", "10", "--max-seconds", "30",
                          "--max-cost-units", "100"])
    assert args.live and args.max_requests == 10


def test_self_test_passes():
    assert rx.self_test() == 0


def test_toy_quick_end_to_end(tmp_path):
    art, journal = _toy_args(tmp_path)
    for key in ("schema", "config", "threat_model", "provenance",
                "results", "summary"):
        assert key in art
    assert art["schema"] == "dtraining.attacker.resource_exhaustion.v1"
    assert art["config"]["attack"] == rx.EXPERIMENT_ID
    threat_model.validate(art["threat_model"])
    assert len(journal) == len(art["results"]) > 0
    for arm in rx.ARMS:
        recs = [r for r in art["results"]
                if r["arm"] == arm and r["experiment"] == rx.EXPERIMENT_ID]
        assert recs, f"arm {arm} produced no result records"
    ctl = [r for r in art["results"] if r["arm"] == rx.NEGATIVE_CONTROL]
    assert ctl and all(r["role"] == "negative-control" for r in ctl)
    assert all(r["experiment"] == rx.EXPERIMENT_ID for r in art["results"])
    arms_in_summary = {s["arm"] for s in art["summary"]}
    assert set(rx.ARMS) | {rx.NEGATIVE_CONTROL} == arms_in_summary


def test_negative_control_behaves(tmp_path):
    art, _ = _toy_args(tmp_path)
    ctl = [r for r in art["results"] if r["arm"] == rx.NEGATIVE_CONTROL]
    assert ctl
    for r in ctl:
        assert 0.8 <= r["amplification_parity"] <= 1.25
        assert r["control_p95_degradation"] == 0.0
        assert r["error_rate"] == 0.0
        assert r["cap_tripped"] is None
        assert r["manual_intervention_required"] is False
        assert r["recovered"] is True
    summ = [s for s in art["summary"] if s["arm"] == rx.NEGATIVE_CONTROL][0]
    assert 0.8 <= summ["amplification_parity_mean"] <= 1.25


def test_vulnerable_world_shows_the_attack(tmp_path):
    """Otherwise the negative control is vacuous: in the undefined
    (unprotected) simulated world the arms must measurably bite."""
    art, _ = _toy_args(tmp_path)

    def cells(arm):
        return [r for r in art["results"] if r["arm"] == arm]

    def max_budget_cells(arm):
        cs = cells(arm)
        top = max(r["budget"] for r in cs)
        return [r for r in cs if r["budget"] == top]

    for r in cells("tokenizer-amplification"):
        assert r["amplification_vs_control"] >= 3.0
    for r in cells("decode-amplification"):
        assert r["amplification_vs_control"] >= 3.0
    for r in max_budget_cells("long-context"):
        assert r["control_p95_degradation"] >= 0.3
    for r in max_budget_cells("cache-thrash"):
        assert r["cache_evictions"] > 0
        assert r["eviction_impact"] > 0.05
        assert r["control_cache_hit_rate"] < 1.0
    for r in cells("retry-storm"):
        assert r["wasted_share"] >= 0.2
        assert r["errors"]["abandoned"] > 0
    for r in max_budget_cells("memory-pressure"):
        assert r["mem_utilization"] > 0.5
    for r in cells("protocol-oversize"):
        assert r["errors"]["intake_reject"] == r["budget"]
        assert r["amplification_vs_control"] >= 2.0
    for r in cells("training-skew"):
        assert r["step_time_inflation"] > 1.0
        assert r["control_throughput_drop"] > 0.0
    for r in max_budget_cells("training-skew"):
        assert r["step_time_inflation"] >= 2.0


def test_hard_caps_hold_and_auto_stop(tmp_path):
    art, _ = _toy_args(tmp_path)
    sim_cells = [r for r in art["results"] if r["arm"] != "training-skew"]
    for r in sim_cells:
        assert r["peak_mem_units"] <= r["mem_cap_units"]
    art2, _ = _toy_args(tmp_path, "--billing-cap-units", "50000",
                        name="capped.json")
    tripped = [r for r in art2["results"] if r["cap_tripped"] == "billing"]
    assert tripped
    assert all(r["manual_intervention_required"] for r in tripped)


def test_live_without_caps_is_refused(tmp_path):
    args = rx.build_parser().parse_args(["--live"])
    with pytest.raises(SystemExit, match="--endpoint"):
        rx.run(args)
    args = rx.build_parser().parse_args(
        ["--live", "--endpoint", "http://h:8000/v1/completions"])
    with pytest.raises(SystemExit, match="--max-requests"):
        rx.run(args)
    args = rx.build_parser().parse_args(
        ["--live", "--endpoint", "http://h:8000/v1/completions",
         "--max-requests", "10"])
    with pytest.raises(SystemExit, match="--max-seconds"):
        rx.run(args)
    args = rx.build_parser().parse_args(
        ["--live", "--endpoint", "ftp://h", "--max-requests", "10",
         "--max-seconds", "5"])
    with pytest.raises(SystemExit, match="ftp://h"):
        rx.run(args)


def test_live_driver_journals_plan_then_measures(tmp_path):
    """The bounded plan is journaled BEFORE any network I/O (unchanged), then
    the driver measures against the live target and writes the artifact."""
    handler, _ = _make_stub(per_token_ms=0.02)
    out = tmp_path / "live.json"
    with _serve(handler) as url:
        args = rx.build_parser().parse_args(
            ["--live", "--endpoint", url, "--arms", "decode-amplification",
             "--budgets", "8", "--max-requests", "20", "--max-seconds", "30",
             "--max-cost-units", "500000", "--output", str(out)])
        args.mode = "inference"
        assert rx.run(args) == 0
    journal = [json.loads(l) for l in
               (tmp_path / "live.json.jsonl").read_text().splitlines()]
    plan = journal[0]
    assert plan["experiment"] == rx.EXPERIMENT_ID
    assert plan["arm"] == "live-plan"
    assert plan["caps"]["max_requests"] == 20
    assert 0 < plan["n_planned_requests"] <= 20
    assert len(journal) > 1
    assert {j.get("role") for j in journal[1:]} >= {"negative-control",
                                                    "attack"}


def test_live_driver_measures_amplification(tmp_path):
    """The honest ceiling-breaker: a decode-amplification request (max output
    tokens) costs the endpoint far more per request than the benign control —
    an objective latency/token ratio, no classifier."""
    handler, counter = _make_stub(per_token_ms=0.05)
    out = tmp_path / "live.json"
    with _serve(handler) as url:
        args = rx.build_parser().parse_args(
            ["--live", "--endpoint", url, "--arms", "decode-amplification",
             "--budgets", "8", "--max-requests", "25", "--max-seconds", "60",
             "--output", str(out)])
        args.mode = "inference"
        assert rx.run(args) == 0
    art = json.loads(out.read_text())
    for key in ("schema", "config", "threat_model", "provenance",
                "results", "summary"):
        assert key in art
    assert art["schema"] == "dtraining.attacker.resource_exhaustion.v1"
    threat_model.validate(art["threat_model"])
    assert art["config"]["live"] is True and art["config"]["toy"] is False
    arms = {r["arm"] for r in art["results"]}
    assert "decode-amplification" in arms and rx.NEGATIVE_CONTROL in arms
    dec = [r for r in art["results"]
           if r["arm"] == "decode-amplification"][0]
    assert dec["token_amplification"] == 64.0
    assert dec["amplification_vs_control"] is not None
    assert dec["amplification_vs_control"] > 2.0
    assert dec["latency_amplification"] == dec["amplification_vs_control"]
    assert dec["error_rate"] == 0.0
    assert dec["cap_tripped"] is None and dec["recovered"] is True
    assert dec["n_completed"] == dec["n_requests"] > 0
    assert dec["peak_mem_units"] is None and dec["cache_evictions"] is None
    assert dec["control_p95_degradation"] is None
    nc = [r for r in art["results"] if r["arm"] == rx.NEGATIVE_CONTROL][0]
    assert abs(nc["amplification_vs_control"] - 1.0) < 1e-6
    assert nc["control_p95_degradation"] == 0.0
    assert {s["arm"] for s in art["summary"]} == arms
    dec_row = [s for s in art["summary"]
               if s["arm"] == "decode-amplification"][0]
    assert dec_row["amplification_vs_control_mean"] > 2.0


def test_live_driver_enforces_max_requests_hard_cap(tmp_path):
    """--max-requests is a HARD stop: the total requests that ever reach the
    endpoint never exceed it."""
    handler, counter = _make_stub(per_token_ms=0.0)
    out = tmp_path / "cap.json"
    with _serve(handler) as url:
        args = rx.build_parser().parse_args(
            ["--live", "--endpoint", url, "--arms", "decode-amplification",
             "--budgets", "64", "--max-requests", "9", "--max-seconds", "60",
             "--output", str(out)])
        args.mode = "inference"
        assert rx.run(args) == 0
    assert counter["n"] <= 9
    art = json.loads(out.read_text())
    dec = [r for r in art["results"]
           if r["arm"] == "decode-amplification"][0]
    assert dec["n_requests"] <= 4


def test_live_driver_health_cap_auto_stops_on_errors(tmp_path):
    """An endpoint erroring on every request trips the health threshold and
    auto-stops — recorded as a FAILED test (manual intervention), never a
    silent success."""
    handler, counter = _make_stub(status_code=500)
    out = tmp_path / "health.json"
    with _serve(handler) as url:
        args = rx.build_parser().parse_args(
            ["--live", "--endpoint", url, "--arms", "decode-amplification",
             "--budgets", "64", "--max-requests", "50", "--max-seconds", "60",
             "--output", str(out)])
        args.mode = "inference"
        assert rx.run(args) == 0
    assert counter["n"] < 20
    art = json.loads(out.read_text())
    recs = art["results"]
    assert any(r["cap_tripped"] == "health" for r in recs)
    assert all(r["manual_intervention_required"]
               for r in recs if r["cap_tripped"])
    assert all(not r["recovered"] for r in recs if r["cap_tripped"])


def test_live_driver_cost_cap_auto_stops(tmp_path):
    """--max-cost-units caps cumulative estimated cost (prompt+completion
    tokens): a stream of max-output requests trips it almost immediately."""
    handler, counter = _make_stub(per_token_ms=0.0)
    out = tmp_path / "cost.json"
    with _serve(handler) as url:
        args = rx.build_parser().parse_args(
            ["--live", "--endpoint", url, "--arms", "decode-amplification",
             "--budgets", "64", "--max-requests", "50", "--max-seconds", "60",
             "--max-cost-units", "1500", "--output", str(out)])
        args.mode = "inference"
        assert rx.run(args) == 0
    assert counter["n"] < 12
    art = json.loads(out.read_text())
    assert any(r["cap_tripped"] == "max_cost" for r in art["results"])


def test_live_driver_skips_non_completion_arms(tmp_path):
    """training-skew's coordinator microbatch has no `prompt` — it is
    journaled as skipped-live, not POSTed to a completions endpoint."""
    handler, _ = _make_stub(per_token_ms=0.0)
    out = tmp_path / "skip.json"
    with _serve(handler) as url:
        args = rx.build_parser().parse_args(
            ["--live", "--endpoint", url, "--arms", "training-skew",
             "decode-amplification", "--budgets", "4", "--max-requests", "20",
             "--max-seconds", "60", "--output", str(out)])
        args.mode = "inference"
        assert rx.run(args) == 0
    journal = [json.loads(l) for l in
               (tmp_path / "skip.json.jsonl").read_text().splitlines()]
    skipped = [j for j in journal if j.get("role") == "skipped-live"]
    assert any(j["arm"] == "training-skew" for j in skipped)
    art = json.loads(out.read_text())
    assert "training-skew" not in {r["arm"] for r in art["results"]}
    assert "decode-amplification" in {r["arm"] for r in art["results"]}


def test_live_against_unreachable_endpoint_records_failure(tmp_path):
    """The driver no longer stubs out at an integration point: pointed at a
    closed port it actually attempts to connect, the health cap auto-stops
    on the connection failures, and it writes an honest artifact (never
    raises 'wire your harness')."""
    out = tmp_path / "dead.json"
    url = f"http://127.0.0.1:{_closed_port()}/v1/completions"
    args = rx.build_parser().parse_args(
        ["--live", "--endpoint", url, "--arms", "decode-amplification",
         "--budgets", "8", "--max-requests", "20", "--max-seconds", "30",
         "--max-cost-units", "500", "--output", str(out)])
    args.mode = "inference"
    assert rx.run(args) == 0
    journal = [json.loads(l) for l in
               (tmp_path / "dead.json.jsonl").read_text().splitlines()]
    plan = journal[0]
    assert plan["arm"] == "live-plan" and plan["caps"]["max_requests"] == 20
    assert 0 < plan["n_planned_requests"] <= 20
    art = json.loads(out.read_text())
    recs = art["results"]
    assert recs
    assert any(r["cap_tripped"] == "health" for r in recs)
    attack = [r for r in recs if r["role"] == "attack"]
    assert attack and all(r["error_rate"] == 1.0 for r in attack)
    assert all(r["manual_intervention_required"]
               for r in recs if r["cap_tripped"])


def test_live_and_toy_are_mutually_exclusive():
    args = rx.build_parser().parse_args(
        ["--toy", "--live", "--endpoint", "http://h:8000",
         "--max-requests", "10", "--max-seconds", "5"])
    with pytest.raises(SystemExit, match="mutually exclusive"):
        rx.run(args)


def test_no_target_path_is_descriptive():
    args = rx.build_parser().parse_args([])
    with pytest.raises(SystemExit, match="--toy"):
        rx.run(args)


def test_live_cap_guard_auto_stop():
    now = [0.0]
    guard = rx.LiveCapGuard(5, 10.0, max_cost_units=50.0,
                            clock=lambda: now[0]).start()
    assert guard.check(5, n_errors=0, est_cost_units=49.0) is True
    with pytest.raises(rx.LiveCapTripped, match="max-requests"):
        guard.check(6)
    now[0] = 11.0
    with pytest.raises(rx.LiveCapTripped, match="max-seconds"):
        guard.check(5)
    with pytest.raises(rx.LiveCapTripped, match="max-cost"):
        rx.LiveCapGuard(50, 100.0, max_cost_units=1.0,
                        clock=lambda: 0.0).start().check(
                            1, est_cost_units=1.0)
    with pytest.raises(rx.LiveCapTripped, match="health"):
        rx.LiveCapGuard(50, 100.0, clock=lambda: 0.0).start().check(
            10, n_errors=6)
    with pytest.raises(ValueError, match="max_requests"):
        rx.LiveCapGuard(None, 10.0)
    with pytest.raises(ValueError, match="max_seconds"):
        rx.LiveCapGuard(10, None)


def test_live_plan_is_bounded():
    args = rx.build_parser().parse_args(
        ["--live", "--endpoint", "http://h:8000/v1/completions",
         "--max-requests", "7", "--max-seconds", "30"])
    caps = rx.check_live_caps(args)
    plan = rx.build_live_plan(args, caps)
    assert plan["n_requests"] <= 7
    for item in plan["requests"]:
        text = json.dumps(item["payload"])
        assert len(text) <= 12000
        assert item["delay_ms"] <= 30 * 1000 / 2
    with pytest.raises(ValueError, match="unknown arm"):
        rx.render_live_request("no-such-arm", 0)


def test_malformed_input_is_descriptive(tmp_path):
    args = rx.build_parser().parse_args(
        ["--toy", "--quick", "--budgets", "-3",
         "--output", str(tmp_path / "bad.json")])
    with pytest.raises(SystemExit, match="--budgets"):
        rx.run(args)
    assert not (tmp_path / "bad.json").exists()
    args = rx.build_parser().parse_args(["--toy", "--rate", "0"])
    with pytest.raises(SystemExit, match="--rate"):
        rx.run(args)


def test_determinism_same_seed(tmp_path):
    art1, _ = _toy_args(tmp_path, name="a.json")
    art2, _ = _toy_args(tmp_path, name="b.json")
    assert art1["results"] == art2["results"]
    assert art1["summary"] == art2["summary"]


def test_toy_without_output_writes_nothing(tmp_path):
    args = rx.build_parser().parse_args(["--toy", "--quick"])
    args.mode = "inference"
    assert rx.run(args) == 0
    assert list(tmp_path.iterdir()) == []


def test_summary_reports_distinct_n_seeds_alongside_n_cells(tmp_path):
    """Each arm holds one record per (seed, budget), so n_cells is a record
    count and must not be read as a seed count; n_seeds carries the seeds."""
    art, _ = _toy_args(tmp_path)
    by = {}
    for r in art["results"]:
        by.setdefault(r["arm"], []).append(r)
    assert art["summary"]
    for row in art["summary"]:
        recs = by[row["arm"]]
        assert row["n_seeds"] == len({r["seed"] for r in recs})
        assert row["n_cells"] == len(recs) > row["n_seeds"]


def test_duplicate_seeds_are_not_double_counted(tmp_path):
    """--quick keeps seeds[:2]; a repeated value therefore yields duplicate
    records per seed and n_seeds must stay the distinct count."""
    art, _ = _toy_args(tmp_path, "--seeds", "0", "0", "1")
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
