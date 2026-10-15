#!/usr/bin/env python3
"""recon — target reconnaissance before attack selection (pure stdlib,
torch-free; every probe is timeout-guarded and NEVER raises — a probe
failure is a degraded record, not a traceback).

    python -m attacker --recon --model gpt2 [--output recon.json] [--evidence]
    python -m attacker --recon --node toula
    python -m attacker --recon --endpoint http://127.0.0.1:8000

Three target kinds:

  --model NAME_OR_PATH   read config.json (a local directory/file, else the
      HF hub when the network answers): architecture family (model_type),
      dense vs MoE (num_experts/num_local_experts/n_routed_experts/moe_*
      keys), depth/width/vocab/heads, tie_word_embeddings, an ESTIMATE of
      the parameter count from config arithmetic (labeled ESTIMATE — the
      formula is in the record), the seam-split candidates (N/3 spacing),
      which attack arms apply, and the recommended battery subset.
  --node NAME            probe a fleet node (attacker/nodes.json) via
      attacker.remote + short-timeout ssh probes: reachability, python,
      torch presence/CUDA, GPUs, memory, HF-cache models -> which tiers of
      gpu-memory / kv-timing / splitbench the node can host.
  --endpoint URL         light HTTP probe: server header, OpenAI-style
      /v1/models listing when open, a small timing sample -> which
      API-surface attacks apply (kv-timing, spec-decode, reasoning-replay
      eligibility).

Output: a recon artifact in the house schema
(schema/config/threat_model/provenance/results/summary, schema
"dtraining.attacker.recon.v1"), optionally evidence-sealed, plus the
markdown execution report via attacker.report conventions.
"""

import argparse
import json
import os
import re
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request

from . import artifacts, remote
from .report import write_report

SCHEMA = "dtraining.attacker.recon.v1"

THREAT_MODEL = (
    "reconnaissance: passive fingerprinting of the target environment "
    "BEFORE attack selection — reads public model configs, fleet-node "
    "probe responses, and unauthenticated HTTP surface only; no attack "
    "cell is executed by this module")

_MOE_COUNT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts",
                   "num_local_experts_per_tok")
_MOE_HINT_KEYS = ("moe_intermediate_size", "shared_expert_intermediate_size",
                  "first_k_dense_replace", "num_experts_per_tok",
                  "moe_top_k", "router_aux_loss_coef")

_LEARNED_POS_FAMILIES = ("gpt2",)


def _first(cfg, *keys):
    for k in keys:
        v = cfg.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v:
            return v
    return None


def _moe_experts(cfg):
    for k in ("num_experts", "num_local_experts", "n_routed_experts"):
        v = cfg.get(k)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return None


def estimate_params(cfg):
    """Parameter-count ESTIMATE from config arithmetic (documented formula;
    labeled ESTIMATE in every consumer — norm/embedding-norm params and
    GQA's smaller KV projections are ignored; MoE counts every routed
    expert's full MLP). Returns (total, formula-str)."""
    hidden = _first(cfg, "n_embd", "hidden_size")
    vocab = _first(cfg, "vocab_size")
    n_layers = _first(cfg, "n_layer", "num_hidden_layers")
    if not (hidden and vocab and n_layers):
        return None, "insufficient config keys (need hidden, vocab, layers)"
    attn = 4 * hidden * hidden
    n_experts = _moe_experts(cfg)
    tie = bool(cfg.get("tie_word_embeddings"))
    embed = vocab * hidden * (1 if tie else 2)
    pos = (cfg.get("n_positions", 0) or 0) * hidden \
        if cfg.get("model_type") in _LEARNED_POS_FAMILIES else 0
    if n_experts:
        moe_inter = _first(cfg, "moe_intermediate_size",
                           "intermediate_size") or 4 * hidden
        shared = _first(cfg, "shared_expert_intermediate_size") or 0
        n_dense = cfg.get("first_k_dense_replace", 0) or 0
        dense_mlp = 3 * hidden * (_first(cfg, "intermediate_size")
                                  or 4 * hidden)
        per_moe = attn + n_experts * 3 * hidden * moe_inter \
            + (3 * hidden * shared if shared else 0)
        total = embed + pos + n_dense * (attn + dense_mlp) \
            + (n_layers - n_dense) * per_moe
        formula = (f"embed({vocab}x{hidden}x{1 if tie else 2}) + "
                   f"{n_dense} dense + {n_layers - n_dense} MoE layers x "
                   f"(attn 4h^2 + {n_experts} experts x 3x{hidden}x"
                   f"{moe_inter})")
    else:
        inter = _first(cfg, "n_inner") or _first(cfg, "intermediate_size")
        if inter is None:
            inter = 4 * hidden
            mlp = 8 * hidden * hidden
        elif "n_inner" in cfg and cfg.get("n_inner"):
            mlp = 2 * hidden * inter
        else:
            mlp = 3 * hidden * inter
        total = embed + pos + n_layers * (attn + mlp)
        formula = (f"embed({vocab}x{hidden}x{1 if tie else 2}) + "
                   f"{n_layers} layers x (attn 4h^2 + mlp "
                   f"{'2' if 'n_inner' in cfg and cfg.get('n_inner') else '3'}"
                   f"x{hidden}x{inter})")
    return int(total), formula


def seam_candidates(n_layers):
    """Split-boundary suggestions at N/3 spacing (SplitPlan-ready ints in
    [1, N-1], deduped): the head should own >=1 block and the tail too."""
    if not isinstance(n_layers, int) or n_layers < 2:
        return []
    cands = sorted({max(1, min(n_layers - 1, round(k * n_layers / 3)))
                    for k in (1, 1.5, 2)})
    return cands


def classify_config(cfg):
    """Classify one HF config.json dict into the recon record (pure,
    never raises)."""
    rec = {"model_type": cfg.get("model_type", "unknown")}
    n_layers = _first(cfg, "n_layer", "num_hidden_layers")
    hidden = _first(cfg, "n_embd", "hidden_size")
    vocab = _first(cfg, "vocab_size")
    heads = _first(cfg, "n_head", "num_attention_heads")
    rec["n_layers"] = n_layers
    rec["hidden_size"] = hidden
    rec["vocab_size"] = vocab
    rec["n_heads"] = heads
    rec["tie_word_embeddings"] = bool(cfg.get("tie_word_embeddings", False))
    n_experts = _moe_experts(cfg)
    rec["is_moe"] = bool(n_experts or "moe" in str(rec["model_type"]).lower()
                         or any(k in cfg for k in _MOE_HINT_KEYS
                                if cfg.get(k)))
    rec["num_experts"] = n_experts
    rec["experts_per_token"] = _first(cfg, "num_experts_per_tok", "moe_top_k")
    rec["context_window"] = _first(cfg, "n_positions",
                                   "max_position_embeddings")
    total, formula = estimate_params(cfg)
    rec["param_estimate"] = {
        "total": total,
        "total_human": (f"~{total / 1e9:.2f}B" if total and total >= 1e9
                        else f"~{total / 1e6:.1f}M" if total else None),
        "formula": formula,
        "note": "ESTIMATE from config arithmetic (attn 4h^2 at MHA widths; "
                "gated MLP 3h*inter / GPT-2 fc+proj 2h*n_inner; MoE counts "
                "every routed expert's full MLP; norms ignored)"}

    arms, notes = [], []
    seams = seam_candidates(n_layers)
    if seams:
        rec["seam_candidates"] = seams
        two_seam = f"{seams[0]},{seams[-1]}" if len(seams) > 1 \
            else str(seams[0])
        notes.append(f"split-point candidates at N/3 spacing: "
                     f"{', '.join(str(s) for s in seams)} "
                     f"(e.g. --split {seams[0]} or the two-seam "
                     f"--split {two_seam})")
    if hidden and vocab:
        arms.append({"surface": "S1", "attack": "seam-inversion",
                     "why": "embedding matrix present -> the VMI/"
                            "linearization inversion family applies "
                            "(white-box E grant)"})
        arms.append({"surface": "S1", "attack": "vec2text-inversion",
                     "why": "learned inversion over the same seam rows"})
        arms.append({"surface": "S2", "attack": "gradient-inversion",
                     "arm": "tag",
                     "why": "tied/known embedding places the label "
                            "direction in E's space for the bwd seam "
                            "gradient direction decode"})
        arms.append({"surface": "S9", "attack": "label-scoring",
                     "why": "per-sequence gradient norms leak sequence "
                            "properties"})
    if rec["tie_word_embeddings"]:
        arms.append({"surface": "S4", "attack": "output-inversion",
                     "why": "tied unembedding grants the output head"})
        notes.append("tie_word_embeddings=true: the output head IS the "
                     "embedding — output-side inversion arms apply")
    if rec["is_moe"]:
        notes.append("MoE: per-token expert routing of a body slice crosses "
                     "the seam — a routing-leak side channel (which experts "
                     "fired per token) on top of the value leak; no "
                     "dedicated module yet, note it in the threat model")
    if rec["model_type"] == "unknown":
        notes.append("unknown model_type: classify by hand before trusting "
                     "the arm suggestions")
    rec["attack_arms"] = arms
    rec["notes"] = notes
    rec["recommended_battery"] = {
        "quick": ["seam-inversion (vmi)", "label-scoring",
                  "gpu-memory (scrub-check)"],
        "full": [a["attack"] for a in arms] + ["gpu-memory (scrub-check)"],
        "command": "python -m attacker --benchmark quick --target-config "
                   "target.json --workdir out/bench"}
    return rec


def _hf_fetch(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent":
                                               "llm-attacker-recon"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def recon_model(source, fetcher=None, timeout=8):
    """Resolve + classify a model's config.json. `source` is a local
    directory (containing config.json), a config.json file, or an HF hub
    name (fetched via `fetcher(url, timeout) -> bytes`, injectable for
    tests; default urllib). Never raises: offline/unknown -> a degraded
    record with a descriptive error."""
    fetcher = fetcher or _hf_fetch
    rec = {"target_kind": "model", "source": str(source)}
    try:
        if os.path.isdir(source):
            path = os.path.join(source, "config.json")
            rec["resolved_from"] = "local-path"
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
        elif os.path.isfile(source):
            rec["resolved_from"] = "local-path"
            with open(source, encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            url = f"https://huggingface.co/{source}/resolve/main/config.json"
            rec["resolved_from"] = "hf-hub"
            rec["config_url"] = url
            cfg = json.loads(fetcher(url, timeout).decode("utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError("config.json is not a JSON object")
        rec["classification"] = classify_config(cfg)
        rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "degraded"
        rec["degraded"] = True
        rec["error"] = (f"config unavailable: {type(e).__name__}: {e} — "
                        "offline or unknown model; pass a local path or "
                        "check the name")
    return rec


def _ssh_run(node, shell_cmd, timeout):
    """One raw ssh command on a fleet node (remote.py conventions:
    BatchMode, ConnectTimeout, hard timeout). Returns (rc, stdout tail);
    rc 124 on timeout, 127 when ssh cannot launch. Never raises."""
    target = (f"{node['user']}@{node['host']}" if node.get("user")
              else node["host"])
    cmd = ["ssh", "-o", "BatchMode=yes",
           "-o", f"ConnectTimeout={max(1, int(timeout))}",
           "-p", str(int(node.get("port", 22))), target, shell_cmd]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout + 5)
        out = (proc.stdout or "").strip()
        return proc.returncode, out[-2000:]
    except subprocess.TimeoutExpired:
        return 124, f"error: ssh command timed out after {timeout + 5}s"
    except Exception as e:
        return 127, f"error: could not launch ssh: {e}"


def recon_node(name, config=None, timeout=4):
    """Probe one fleet node (attacker/nodes.json) -> capabilities report.
    Never raises; unreachable nodes degrade to status records (remote.py
    doctrine)."""
    rec = {"target_kind": "node", "node": name}
    try:
        cfg = config or remote.load_config()
        node = remote.get_node(cfg, name)
    except (OSError, ValueError, KeyError) as e:
        rec["status"] = "degraded"
        rec["error"] = str(e)
        return rec
    rec["host"] = node["host"]
    rec["role"] = node["role"]
    status = remote.probe_node(node, timeout=timeout)
    rec.update(status.to_dict())
    if not status.ssh_ok:
        rec["status"] = "degraded"
        rec["capabilities"] = {"note": "node unreachable or ssh failed — "
                                       "capabilities unknown"}
        return rec

    def probe(cmd):
        rc, out = _ssh_run(node, cmd, timeout)
        return out if rc == 0 else None

    py = probe("python3 --version 2>&1")
    torch_probe = probe(
        "python3 -c \"import torch; print(torch.__version__, "
        "torch.cuda.is_available())\" 2>&1")
    torch_version, cuda = None, False
    if torch_probe:
        parts = torch_probe.split()
        if len(parts) >= 2 and parts[0][0].isdigit():
            torch_version, cuda = parts[0], parts[1] == "True"
    gpus = probe("nvidia-smi -L 2>/dev/null") or ""
    mem = probe("python3 -c \"import os; print(os.sysconf('SC_PAGE_SIZE') *"
                " os.sysconf('SC_PHYS_PAGES'))\" 2>/dev/null")
    hf = probe("ls ~/.cache/huggingface/hub 2>/dev/null")
    models = sorted(m for m in (hf or "").splitlines()
                    if m.startswith("models--"))
    rec["python"] = py
    rec["torch"] = {"installed": torch_version is not None,
                    "version": torch_version, "cuda": cuda}
    rec["gpus"] = [g for g in gpus.splitlines() if g.strip()]
    rec["memory_bytes"] = int(mem) if mem and mem.isdigit() else None
    rec["hf_cache_models"] = models
    rec["status"] = "ok"
    rec["capabilities"] = {
        "gpu-memory": ("torch-cuda tier (real)" if cuda
                       else "sim tier only (--toy): no CUDA visible"),
        "kv-timing": ("live endpoint probing + offline journals"
                      if py else "unavailable: no python3"),
        "splitbench": (f"runs on {'cuda' if cuda else 'cpu'} (torch "
                       f"{torch_version})" if torch_version
                       else "unavailable: torch not installed"),
        "benchmark": ("full battery on this node"
                      if torch_version else "torch-less: cells degrade to "
                                            "error records")}
    return rec


def _http_get(url, timeout):
    """One GET -> (status, headers-dict, body-bytes, elapsed_s); raises on
    transport errors (caller degrades)."""
    t0 = time.monotonic()
    req = urllib.request.Request(url, headers={"User-Agent":
                                               "llm-attacker-recon"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read(65536)
        return r.status, dict(r.headers.items()), body, \
            time.monotonic() - t0


def recon_endpoint(url, timeout=5, samples=3):
    """Light HTTP probe of an inference endpoint. Never raises; every
    failure mode degrades to a noted record."""
    rec = {"target_kind": "endpoint", "endpoint": str(url)}
    base = str(url)
    if "://" not in base:
        base = "http://" + base
    base = base.rstrip("/")
    try:
        status, headers, body, elapsed = _http_get(base, timeout)
        rec["status"] = "ok"
        rec["http_status"] = status
        rec["server_header"] = headers.get("Server") or headers.get("server")
        rec["base_latency_ms"] = round(1000.0 * elapsed, 2)
    except Exception as e:
        rec["status"] = "degraded"
        rec["error"] = f"endpoint probe failed: {type(e).__name__}: {e}"
        return rec

    root = base[:-3] if base.endswith("/v1") else base
    models = None
    try:
        mstatus, _, mbody, _ = _http_get(root + "/v1/models", timeout)
        if mstatus == 200:
            data = json.loads(mbody.decode("utf-8", "replace"))
            if isinstance(data, dict) and isinstance(data.get("data"), list):
                models = [m.get("id") for m in data["data"]
                          if isinstance(m, dict) and m.get("id")]
    except Exception as e:
        rec["models_error"] = f"/v1/models: {type(e).__name__}: {e}"
    rec["openai_style"] = models is not None
    rec["models"] = models

    timings = []
    for _ in range(max(1, int(samples))):
        try:
            _, _, _, elapsed = _http_get(base, timeout)
            timings.append(round(1000.0 * elapsed, 2))
        except Exception:
            break
    if timings:
        rec["timing_sample_ms"] = {"n": len(timings), "min": min(timings),
                                   "median": statistics.median(timings),
                                   "max": max(timings)}
    rec["attack_applicability"] = {
        "kv-timing": ("eligible: endpoint answers — measure the "
                      "prefix-cache TTFT channel with kv-timing --endpoint "
                      "(POST /v1/completions path, unverified by this GET "
                      "probe)" ),
        "spec-decode": ("unknown: needs a completion probe to see whether "
                        "acceptance patterns/metadata are exposed"),
        "reasoning-replay": ("unknown: needs an authenticated completion "
                             "carrying reasoning blobs (S8b) to assess"),
        "wire-shape": ("applicable at the link layer regardless (mitm-proxy"
                       " journal)")}
    if models:
        rec["attack_applicability"]["note"] = (
            f"OpenAI-style listing open: {len(models)} model id(s) visible "
            f"({', '.join(models[:5])})")
    return rec


def _summary_row(rec):
    """Flat scalar subset for the artifact's summary cell (report table /
    scorecard headline; the full record stays in results[0])."""
    row = {"target_kind": rec.get("target_kind"),
           "status": rec.get("status", "degraded")}
    cls = rec.get("classification") or {}
    for key in ("model_type", "is_moe", "num_experts", "n_layers",
                "hidden_size", "vocab_size", "tie_word_embeddings"):
        if key in cls:
            row[key] = cls[key]
    est = (cls.get("param_estimate") or {}).get("total")
    if est is not None:
        row["param_estimate_total"] = est
    for key in ("node", "reachable", "ssh_ok", "endpoint", "http_status",
                "openai_style"):
        if key in rec:
            row[key] = rec[key]
    if rec.get("error"):
        row["error"] = rec["error"]
    return row


def _slug(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text)).strip("-") or "target"


def run_recon(kind, target, output=None, evidence=False, timeout=None,
              fetcher=None):
    """Run one recon probe, write the artifact (+ evidence + MD report),
    print the findings. Returns the artifact path; never raises."""
    if kind == "model":
        rec = recon_model(target, fetcher=fetcher,
                          timeout=timeout or 8)
    elif kind == "node":
        rec = recon_node(target, timeout=timeout or 4)
    else:
        rec = recon_endpoint(target, timeout=timeout or 5)
    output = output or f"recon_{kind}_{_slug(target)}.json"
    out = artifacts.make_artifact(
        SCHEMA, {"target_kind": kind, "target": str(target)},
        THREAT_MODEL,
        interpretation="results[0] is the recon record; summary[0] its "
                       "flat scalar subset. status=degraded means the "
                       "probe could not answer (offline / unreachable / "
                       "refused) — degradation is a result, not a crash.")
    out["results"].append(rec)
    out["summary"].append(_summary_row(rec))
    if evidence:
        from .evidence import EvidenceLogger
        with EvidenceLogger(output, node_role=f"recon/{kind}") as ev:
            ev.event("recon_result", _summary_row(rec))
            artifacts.write_artifact(output, out, evidence=ev)
    else:
        artifacts.write_artifact(output, out)
    report_path = write_report(output)

    print(f"[recon] {kind} {target}: status={rec.get('status')}")
    if rec.get("error"):
        print(f"[recon]   degraded: {rec['error']}")
    cls = rec.get("classification") or {}
    if cls:
        est = cls.get("param_estimate") or {}
        print(f"[recon]   {cls.get('model_type')} "
              f"{'MoE x' + str(cls.get('num_experts')) if cls.get('is_moe') else 'dense'} "
              f"L={cls.get('n_layers')} h={cls.get('hidden_size')} "
              f"V={cls.get('vocab_size')} params={est.get('total_human')} "
              f"(ESTIMATE)")
        for n in cls.get("notes", []):
            print(f"[recon]   note: {n}")
    caps = rec.get("capabilities") or {}
    for k, v in caps.items():
        print(f"[recon]   {k}: {v}")
    app = rec.get("attack_applicability") or {}
    for k, v in app.items():
        print(f"[recon]   {k}: {v}")
    print(f"[recon]   artifact: {output} (+ {report_path})")
    return output


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m attacker --recon",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    tgt = ap.add_mutually_exclusive_group(required=True)
    tgt.add_argument("--model", help="HF name or local path (config.json)")
    tgt.add_argument("--node", help="fleet node name (attacker/nodes.json)")
    tgt.add_argument("--endpoint", help="inference endpoint URL")
    ap.add_argument("--output", default=None,
                    help="recon artifact path (default: "
                         "recon_<kind>_<target>.json); the MD report lands "
                         "at <output>.report.md")
    ap.add_argument("--evidence", action="store_true",
                    help="seal the recon artifact with an "
                         "attacker.evidence chain")
    ap.add_argument("--timeout", type=float, default=None,
                    help="per-probe timeout seconds (defaults: model 8, "
                         "node 4, endpoint 5)")
    args = ap.parse_args(argv)
    kind = "model" if args.model else "node" if args.node else "endpoint"
    target = args.model or args.node or args.endpoint
    run_recon(kind, target, output=args.output, evidence=args.evidence,
              timeout=args.timeout)
    return 0


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import tempfile
    print("recon — model config classification (pure python):")
    gpt2 = {"model_type": "gpt2", "n_layer": 12, "n_embd": 768, "n_head": 12,
            "vocab_size": 50257, "n_positions": 1024, "n_inner": 3072,
            "tie_word_embeddings": True}
    cls = classify_config(gpt2)
    check("dense GPT-2-like config classifies dense with the core dims",
          cls["is_moe"] is False and cls["n_layers"] == 12
          and cls["hidden_size"] == 768 and cls["vocab_size"] == 50257
          and cls["tie_word_embeddings"] is True)
    est = cls["param_estimate"]["total"]
    check("param ESTIMATE lands in the GPT-2 band (~124M)",
          110e6 < est < 140e6 and "ESTIMATE" in cls["param_estimate"]["note"])
    check("seam candidates are N/3-spaced SplitPlan-ready ints",
          cls["seam_candidates"] == [4, 6, 8])
    check("embedding + tied-head arms apply (S1 inversion, S4 output)",
          any(a["attack"] == "seam-inversion" for a in cls["attack_arms"])
          and any(a["attack"] == "output-inversion"
                  for a in cls["attack_arms"]))
    check("the recommended battery names the standard quick subset",
          cls["recommended_battery"]["quick"]
          == ["seam-inversion (vmi)", "label-scoring",
              "gpu-memory (scrub-check)"])

    mixtralish = {"model_type": "mixtral", "num_hidden_layers": 32,
                  "hidden_size": 4096, "intermediate_size": 14336,
                  "num_attention_heads": 32, "vocab_size": 32000,
                  "num_local_experts": 8, "num_experts_per_tok": 2,
                  "max_position_embeddings": 32768}
    cls_m = classify_config(mixtralish)
    est_m = cls_m["param_estimate"]["total"]
    check("MoE config classifies MoE with expert counts",
          cls_m["is_moe"] is True and cls_m["num_experts"] == 8
          and cls_m["experts_per_token"] == 2)
    check("MoE param ESTIMATE lands in the 8x7B band (>40B)",
          est_m > 40e9)
    check("MoE routing-leak surface is noted",
          any("routing" in n for n in cls_m["notes"]))
    check("MoE never counts fake positional params",
          "1024" not in cls_m["param_estimate"]["formula"])
    check("empty config degrades to unknown with a note, never raises",
          classify_config({})["model_type"] == "unknown"
          and classify_config({})["notes"])

    print("recon — model resolution (local + offline-degraded):")
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(gpt2, f)
        rec = recon_model(d)
        check("local dir resolves and classifies",
              rec["status"] == "ok" and rec["resolved_from"] == "local-path"
              and rec["classification"]["n_layers"] == 12)
        rec = recon_model(os.path.join(d, "config.json"))
        check("a bare config.json file resolves too",
              rec["status"] == "ok")

        def dead_fetcher(url, timeout):
            raise urllib.error.URLError("no network (self-test)")

        rec = recon_model("no-such-org/no-such-model", fetcher=dead_fetcher)
        check("offline hub fetch degrades with a descriptive error",
              rec["status"] == "degraded" and "offline" in rec["error"]
              and rec["target_kind"] == "model")

        out = run_recon("model", d, output=os.path.join(d, "recon.json"))
        check("run_recon writes artifact + report",
              os.path.isfile(out)
              and os.path.isfile(out + ".report.md"))
        art = json.load(open(out))
        check("recon artifact carries the house schema + recon schema",
              art["schema"] == SCHEMA
              and all(k in art for k in ("config", "threat_model",
                                         "provenance", "results", "summary")))
        check("summary row is the flat scalar subset",
              art["summary"][0]["model_type"] == "gpt2"
              and art["summary"][0]["param_estimate_total"] == est)
        md = open(out + ".report.md").read()
        check("the MD report renders via report.py conventions",
              "# Execution report" in md and "Corpus caveat" in md)

    print("recon — node probing (degraded paths, short timeouts):")
    rec = recon_node("no-such-node",
                     config={"schema": remote.SCHEMA, "nodes": []})
    check("unknown node is a degraded record, not a crash",
          rec["status"] == "degraded" and "no-such-node" in rec["error"])
    t0 = time.monotonic()
    rec = recon_node("dead",
                     config={"schema": remote.SCHEMA,
                             "nodes": [{"name": "dead", "host": "192.0.2.1",
                                        "role": "cpu"}]},
                     timeout=1)
    dt = time.monotonic() - t0
    check(f"unreachable node degrades fast ({dt:.2f}s) with capabilities "
          "noted unknown",
          rec["status"] == "degraded" and rec["reachable"] is False
          and "capabilities" in rec and dt < 10)

    print("recon — endpoint probing (localhost stub + refused):")
    import http.server
    import threading

    class Stub(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/v1/models":
                body = json.dumps({"object": "list", "data": [
                    {"id": "stub-model-1"}, {"id": "stub-model-2"}]}).encode()
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

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        rec = recon_endpoint(f"127.0.0.1:{port}", timeout=2, samples=3)
        check("stub endpoint probes ok with a server header",
              rec["status"] == "ok" and rec["http_status"] == 200
              and rec["server_header"])
        check("OpenAI-style /v1/models listing is parsed",
              rec["openai_style"] is True
              and rec["models"] == ["stub-model-1", "stub-model-2"])
        check("the timing sample landed",
              rec["timing_sample_ms"]["n"] == 3
              and rec["timing_sample_ms"]["min"] > 0)
        check("attack applicability names kv-timing eligibility",
              "kv-timing" in rec["attack_applicability"]
              and "eligible" in rec["attack_applicability"]["kv-timing"])
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    dead_port = sock.getsockname()[1]
    sock.close()
    t0 = time.monotonic()
    rec = recon_endpoint(f"127.0.0.1:{dead_port}", timeout=2)
    check("a refused endpoint degrades fast with an error record",
          rec["status"] == "degraded" and "error" in rec
          and time.monotonic() - t0 < 5)

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
