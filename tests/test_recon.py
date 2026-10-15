"""Recon scanner (attacker/recon.py): config classification, degraded
offline paths, endpoint stub probing, and the --recon CLI."""

import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import urllib.error

from attacker import remote
from attacker.recon import (SCHEMA, classify_config, estimate_params,
                            recon_endpoint, recon_model, recon_node,
                            run_recon, seam_candidates)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GPT2_CFG = {"model_type": "gpt2", "n_layer": 12, "n_embd": 768,
            "n_head": 12, "vocab_size": 50257, "n_positions": 1024,
            "n_inner": 3072, "tie_word_embeddings": True}

MOE_CFG = {"model_type": "mixtral", "num_hidden_layers": 32,
           "hidden_size": 4096, "intermediate_size": 14336,
           "num_attention_heads": 32, "vocab_size": 32000,
           "num_local_experts": 8, "num_experts_per_tok": 2,
           "max_position_embeddings": 32768}


def test_classify_dense_gpt2():
    cls = classify_config(GPT2_CFG)
    assert cls["model_type"] == "gpt2" and cls["is_moe"] is False
    assert (cls["n_layers"], cls["hidden_size"], cls["vocab_size"]) \
        == (12, 768, 50257)
    assert cls["tie_word_embeddings"] is True
    est = cls["param_estimate"]
    assert 110e6 < est["total"] < 140e6
    assert "ESTIMATE" in est["note"] and est["formula"]
    assert cls["seam_candidates"] == [4, 6, 8]
    attacks = {a["attack"] for a in cls["attack_arms"]}
    assert {"seam-inversion", "gradient-inversion", "label-scoring",
            "output-inversion"} <= attacks
    assert cls["recommended_battery"]["quick"]


def test_classify_moe():
    cls = classify_config(MOE_CFG)
    assert cls["is_moe"] is True and cls["num_experts"] == 8
    assert cls["experts_per_token"] == 2
    assert cls["param_estimate"]["total"] > 40e9
    assert any("routing" in n for n in cls["notes"])


def test_classify_degenerate_configs_never_raise():
    cls = classify_config({})
    assert cls["model_type"] == "unknown" and cls["notes"]
    assert cls["param_estimate"]["total"] is None
    assert classify_config({"model_type": "moe-ish"})["is_moe"] is True
    assert seam_candidates(1) == [] and seam_candidates(None) == []


def test_estimate_params_formula_documented():
    total, formula = estimate_params(GPT2_CFG)
    assert isinstance(total, int) and "embed" in formula


def test_recon_model_local_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(GPT2_CFG))
    rec = recon_model(str(tmp_path))
    assert rec["status"] == "ok" and rec["resolved_from"] == "local-path"
    assert rec["classification"]["n_layers"] == 12


def test_recon_model_offline_degrades():
    def dead(url, timeout):
        raise urllib.error.URLError("no network (test)")
    rec = recon_model("no-such-org/no-such-model", fetcher=dead)
    assert rec["status"] == "degraded"
    assert "offline" in rec["error"] and rec["target_kind"] == "model"


def test_recon_node_unknown_name():
    rec = recon_node("no-such-node",
                     config={"schema": remote.SCHEMA, "nodes": []})
    assert rec["status"] == "degraded" and "no-such-node" in rec["error"]


def test_recon_node_unreachable_fast():
    rec = recon_node("dead",
                     config={"schema": remote.SCHEMA,
                             "nodes": [{"name": "dead", "host": "192.0.2.1",
                                        "role": "cpu"}]},
                     timeout=1)
    assert rec["status"] == "degraded" and rec["reachable"] is False
    assert "capabilities" in rec


class _Stub(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({"object": "list",
                               "data": [{"id": "m1"}, {"id": "m2"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_recon_endpoint_stub():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        rec = recon_endpoint(f"127.0.0.1:{srv.server_address[1]}",
                             timeout=2, samples=3)
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2)
    assert rec["status"] == "ok" and rec["http_status"] == 200
    assert rec["server_header"]
    assert rec["openai_style"] is True and rec["models"] == ["m1", "m2"]
    assert rec["timing_sample_ms"]["n"] == 3
    assert "eligible" in rec["attack_applicability"]["kv-timing"]


def test_recon_endpoint_refused_degrades():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    rec = recon_endpoint(f"127.0.0.1:{port}", timeout=2)
    assert rec["status"] == "degraded" and "error" in rec


def test_run_recon_writes_house_artifact_and_report(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text(json.dumps(GPT2_CFG))
    out = run_recon("model", str(tmp_path / "model"),
                    output=str(tmp_path / "recon.json"))
    art = json.loads(open(out).read())
    assert art["schema"] == SCHEMA
    assert all(k in art for k in ("config", "threat_model", "provenance",
                                  "results", "summary"))
    assert art["summary"][0]["model_type"] == "gpt2"
    assert art["summary"][0]["param_estimate_total"] > 1e8
    assert os.path.isfile(out + ".report.md")
    md = open(out + ".report.md").read()
    assert "# Execution report" in md and "Corpus caveat" in md


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "attacker", *args],
                          cwd=ROOT, capture_output=True, text=True,
                          timeout=300)


def test_cli_recon_model(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(GPT2_CFG))
    out = tmp_path / "r.json"
    r = run_cli("--recon", "--model", str(tmp_path), "--output", str(out),
                "--evidence")
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    assert "gpt2" in r.stdout and "ESTIMATE" in r.stdout
    assert out.is_file() and os.path.isfile(str(out) + ".report.md")
    assert os.path.isfile(str(out) + ".evidence.json")


def test_cli_recon_requires_one_target():
    r = run_cli("--recon")
    assert r.returncode != 0
