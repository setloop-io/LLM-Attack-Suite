#!/usr/bin/env python3
"""resource-exhaustion — availability, cost, and fairness stress tests of an
LLM serving stack under adversarial but valid or near-valid workloads
(OWASP LLM06; THREAT_MODEL_REVIEW P0 "resource-exhaustion"). Where
max-effort optimizes the *evaluator*, this module tests the *service*:
tenant DoS, cache thrashing, long-context amplification, starvation, and
cost exhaustion. Label-free; applies to inference serving and to
split-training microbatch intake (training-skew arm).

SAFETY (hard requirement)
-------------------------
The default and ONLY built-in world is a local SIMULATED serving stack
(SimServer) with hard compute, memory, duration, request, and billing
caps — no network, no torch, deterministic. A real target requires BOTH
--live AND --endpoint plus the --max-requests / --max-seconds hard caps
(--max-cost-units optional); the live path validates the safety envelope
(check_live_caps), builds a bounded request plan (render_live_request /
build_live_plan), then DRIVES the endpoint (run_live / _drive_live):
each attack request is interleaved with a benign control and sent through
measure_live_completion under the auto-stop guard (LiveCapGuard: raises
LiveCapTripped at the first crossed request/wall-clock/cost cap or an
unhealthy error rate). The driver measures REAL cost/latency amplification
(attack server-service-time per request vs the control's — objective,
latency/tokens, no classifier) and emits the SAME artifact schema as the
toy tier. A --live run without caps is refused. A run MUST auto-stop when
a health or cost threshold is crossed; an uncontrolled outage is a FAILED
test, never a successful one (cap_tripped + manual_intervention_required
in the cell).

THE SIMULATED STACK
-------------------
SimServer(seed, caps, warm_cache=...): intake parser (per-request header cost; oversized
declarations rejected before payload intake), FIFO queue (cap -> 429-style
rejects), continuous batcher (max_batch), LRU prefix cache (cache_tokens),
KV memory model (1 unit/token; cap -> head-of-line blocking), token cost
model (prefill 1 unit/token, decode 2 units/token x decode-cost
multiplier, fixed per-step overhead when busy — batching amortizes it), a
server-side per-request deadline, and a billing meter (hard cap stops the
run). A benign CONTROL TENANT always runs concurrently at --rate req/s
with moderate prompts drawn from a small pool of shared boilerplate
prefixes; its stream is IDENTICAL across the worlds of a (seed, budget)
pair so degradation is measured against the same background.

ARMS (per-cell journals; each arm sweeps the --budgets request grid)
--------------------------------------------------------------------
  long-context              max / near-max context requests
  tokenizer-amplification   small byte inputs expanding to many tokens
  cache-thrash              unique large prefixes evict useful cached ones
  batch-fragmentation       staggered tiny/medium requests defeat batching
                            (fixed per-step overhead dominates)
  decode-amplification      max output caps + expensive decoding settings
                            (decode cost multiplier)
  retry-storm               short client timeouts + retries duplicate work
  memory-pressure           concurrent large-footprint requests approach
                            the allocator cap
  training-skew             max-length microbatches inflate the whole
                            synchronous step (barrier model; co-workers are
                            the control tenant)
  protocol-oversize         bounded oversized declarations rejected before
                            payload intake, after burning parse budget

METRICS (per cell; see JOURNAL SCHEMA below)
--------------------------------------------
attacker vs defender cost and the amplification ratio (work-based, and
relative to the control tenant: amplification_vs_control); throughput and
p50/p95/p99 latency; error rate and control-tenant error rate (starvation
margin); peak memory vs the cap; cache hit rate, evictions, and eviction
impact on the control tenant; fairness: control-tenant p95 degradation vs
the negative-control world; wasted work share (retry-storm); recovery
time and manual_intervention_required. NEGATIVE CONTROL (mandatory): the
benign-control cell — benign-only load at the same total rate — must show
amplification_parity ~= 1 (the benign-extra tenant costs the server the
same per unit as the control tenant) and ~0 fairness degradation; it is
also the degradation baseline for its (seed, budget) pair.

JOURNAL SCHEMA (per-cell records, <output>.jsonl + artifact results)
--------------------------------------------------------------------
  experiment, arm, role ("attack" | "negative-control"), seed, budget,
  n_requests, n_control, n_completed, attacker_cost_units,
  control_cost_units, defender_work_units, defender_overhead_units,
  defender_total_units, amplification, amplification_work,
  control_amplification_work, amplification_vs_control, throughput_rps,
  p50_ms, p95_ms, p99_ms, control_p50_ms, control_p95_ms,
  control_p95_degradation, amplification_parity (negative control only),
  error_rate, control_error_rate, peak_mem_units, mem_cap_units,
  mem_utilization, cache_hit_rate, control_cache_hit_rate,
  cache_evictions, eviction_impact, wasted_units, wasted_share,
  recovery_time_ms, recovered, cap_tripped,
  manual_intervention_required, errors{intake_reject, queue_reject,
  timeout, abandoned, unfinished}
training-skew cells instead carry: n_attacker_microbatches, workers,
steps, step_time_inflation, cost_amplification, control_throughput_drop,
defender_total_units, defender_benign_baseline_units.

LIMITATIONS / EXCLUDED VARIANTS
-------------------------------
Abstract compute/memory/billing units — NOT calibrated to a specific GPU,
model, or tokenizer; single node (no replicas, autoscaling, preemption,
chunked prefill, or paged KV; KV memory freed at completion); network and
TLS not modeled; training-skew models only a synchronous data-parallel
barrier (no pipeline parallelism, no async SGD); volumetric L3/L4 DDoS,
colluding tenants, and adaptive rate-limit evasion are out of scope; the
live driver measures cost/latency amplification with a bounded closed-loop
(one outstanding request) sender under LiveCapGuard — it does NOT generate
concurrent load, so live control-tenant fairness/queueing degradation
(control_p95_degradation) is not wire-observable and is reported null on
live cells (the toy stack measures it); only completion-shaped arms are
exercised on a /v1/completions endpoint (training-skew's coordinator
microbatch is toy-only and is journaled as skipped-live).

Usage:
    python -m attacker --mode inference --attack resource-exhaustion --help
    python -m attacker --mode inference --attack resource-exhaustion \
        --toy --quick --output /tmp/rx.json
    python -m attacker --mode training --attack resource-exhaustion \
        --toy --arms training-skew --output /tmp/rx-train.json
    python -m attacker --mode inference --attack resource-exhaustion \
        --live --endpoint http://host:8000/v1/completions \
        --max-requests 200 --max-seconds 60 --max-cost-units 5000
"""

import argparse
import json
import random
import socket
import statistics
import time
import urllib.error
import urllib.request

from .. import artifacts, threat_model
from .common import add_common_args, journal_error

EXPERIMENT_ID = "resource_exhaustion"
MODES = ("training", "inference")
REQUIRES_LABELS = False
DESCRIPTION = ("availability/cost/fairness stress tests (OWASP LLM06): a "
               "simulated serving stack (intake parser, queue, LRU prefix "
               "cache, batcher, token cost + memory model, hard caps) "
               "under 9 adversarial workload arms vs a benign control "
               "tenant; --live path is a cap-checked integration point")

ARMS = ("long-context", "tokenizer-amplification", "cache-thrash",
        "batch-fragmentation", "decode-amplification", "retry-storm",
        "memory-pressure", "training-skew", "protocol-oversize")
NEGATIVE_CONTROL = "benign-control"

LIVE_PROMPT_CHAR_CAP = 8192
LIVE_MAX_TOKENS_CAP = 1024

LIVE_CONTROL_MAX_TOKENS = 16
LIVE_REQUEST_TIMEOUT_S = 120.0

_EPS = 1e-9


def _pct(vals, q):
    """Linear-interpolated percentile (q in [0, 100]); nan for empty."""
    vals = sorted(vals)
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return float(vals[0])
    k = (len(vals) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return float(vals[f])
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def _r6(x):
    """Round for the journal; nan/inf -> None (strict-JSON safe)."""
    if x is None or x != x or x in (float("inf"), float("-inf")):
        return None
    return round(float(x), 6)


def _new_tstat():
    return {"work_units": 0.0, "cost_units": 0.0, "n_intake": 0,
            "n_completed": 0, "out_tokens": 0, "latencies": [],
            "hit_tokens": 0.0, "prefix_tokens": 0.0,
            "errors": {"intake_reject": 0, "queue_reject": 0, "timeout": 0,
                       "abandoned": 0, "unfinished": 0}}


CONTROL_PREFIX_POOL = 8
CONTROL_PREFIX_TOKENS = 384


def control_warm_keys():
    """The control pool pre-warmed in every world's cache (a real serving
    stack runs warm): cold-start misses would otherwise be attributed to
    whichever tenant draws a key first and pollute the eviction metric."""
    return {f"ctl-{i}": CONTROL_PREFIX_TOKENS
            for i in range(CONTROL_PREFIX_POOL)}


class SimServer:
    """A deterministic simulated serving stack with hard caps. run(requests)
    steps the scheduler at caps["step_ms"] granularity and returns per-tenant
    stats plus world metrics. Deterministic given the seed and requests."""

    def __init__(self, seed, caps, warm_cache=None):
        self.seed = seed
        self.caps = dict(caps)
        self.warm_cache = dict(warm_cache or {})

    def run(self, requests):
        c = self.caps
        step = c["step_ms"]
        reqs = sorted(requests, key=lambda r: r["arrival_ms"])
        injected = []
        idx = 0
        queue, running = [], []
        cache = dict(self.warm_cache)
        cache_used = sum(cache.values())
        mem_used = 0.0
        peak_mem = 0.0
        intake_backlog = 0.0
        overhead_units = 0.0
        wasted_units = 0.0
        evictions = 0
        n_intake = 0
        cap_tripped = None
        last_noncontrol_depart = None
        recovery_t = None
        tenants = {}

        def tstat(tenant):
            if tenant not in tenants:
                tenants[tenant] = _new_tstat()
            return tenants[tenant]

        def depart(req, t):
            nonlocal last_noncontrol_depart
            if req["tenant"] != "control":
                last_noncontrol_depart = t

        t = 0.0
        while cap_tripped is None:
            if not queue and not running:
                cand = []
                if idx < len(reqs):
                    cand.append(reqs[idx]["arrival_ms"])
                if injected:
                    cand.append(min(r["arrival_ms"] for r in injected))
                if not cand:
                    break
                nxt = min(cand)
                if nxt > t:
                    t = nxt
            budget = float(c["step_budget_units"])

            due = []
            while idx < len(reqs) and reqs[idx]["arrival_ms"] <= t:
                due.append(reqs[idx])
                idx += 1
            if injected:
                keep = []
                for r in injected:
                    (due if r["arrival_ms"] <= t else keep).append(r)
                injected = keep
            for req in due:
                n_intake += 1
                st = tstat(req["tenant"])
                st["n_intake"] += 1
                if n_intake > c["max_requests"]:
                    cap_tripped = "requests"
                    break
                hdr = req.get("header_units") or c["header_parse_units"]
                intake_backlog += hdr
                st["work_units"] += hdr
                st["cost_units"] += (c["request_fee_units"]
                                     + req["input_bytes"]
                                     * c["byte_fee_units"])
                declared = req.get("declared_tokens")
                if declared is not None and declared > c["context_window"]:
                    st["errors"]["intake_reject"] += 1
                    depart(req, t)
                    continue
                if req["prompt_tokens"] > c["context_window"]:
                    st["errors"]["intake_reject"] += 1
                    depart(req, t)
                    continue
                if len(queue) >= c["queue_max"]:
                    st["errors"]["queue_reject"] += 1
                    depart(req, t)
                    continue
                queue.append(req)

            while queue and len(running) < c["max_batch"]:
                req = queue[0]
                out_tok = min(req["output_tokens"], c["max_output_tokens"])
                footprint = req["prompt_tokens"] + out_tok
                if mem_used + footprint > c["mem_units"]:
                    break
                queue.pop(0)
                st = tstat(req["tenant"])
                hit = 0
                key = req.get("prefix_key")
                pt = (min(req.get("prefix_tokens", 0), req["prompt_tokens"])
                      if key else 0)
                if key and pt > 0:
                    st["prefix_tokens"] += pt
                    if key in cache:
                        hit = pt
                        cache[key] = cache.pop(key)
                    else:
                        cache[key] = pt
                        cache_used += pt
                        while cache_used > c["cache_tokens"] and cache:
                            k0 = next(iter(cache))
                            cache_used -= cache.pop(k0)
                            evictions += 1
                    st["hit_tokens"] += hit
                req["prefill_left"] = float(req["prompt_tokens"] - hit)
                req["decode_left"] = out_tok
                req["footprint"] = footprint
                req["work_done"] = 0.0
                req["out_done"] = 0
                mem_used += footprint
                peak_mem = max(peak_mem, mem_used)
                running.append(req)

            served = min(budget, intake_backlog)
            budget -= served
            intake_backlog -= served
            if running:
                budget = max(0.0, budget - c["fixed_overhead_units"])
                overhead_units += c["fixed_overhead_units"]
                for req in running:
                    if budget <= 0:
                        break
                    if req["prefill_left"] > 0:
                        do = min(req["prefill_left"],
                                 budget / c["prefill_unit_cost"])
                        units = do * c["prefill_unit_cost"]
                        req["prefill_left"] -= do
                        budget -= units
                        req["work_done"] += units
                        tstat(req["tenant"])["work_units"] += units
                for req in running:
                    if req["prefill_left"] == 0 and req["decode_left"] > 0:
                        cost = (c["decode_unit_cost"]
                                * req.get("decode_cost_mult", 1.0))
                        if budget >= cost:
                            req["decode_left"] -= 1
                            req["out_done"] += 1
                            budget -= cost
                            req["work_done"] += cost
                            st = tstat(req["tenant"])
                            st["work_units"] += cost
                            st["out_tokens"] += 1

            if running:
                still = []
                for req in running:
                    if req["prefill_left"] <= 0 and req["decode_left"] <= 0:
                        st = tstat(req["tenant"])
                        st["n_completed"] += 1
                        st["latencies"].append(t - req["arrival_ms"])
                        st["cost_units"] += (req.get("out_done", 0)
                                             * c["out_token_fee_units"])
                        mem_used -= req["footprint"]
                        depart(req, t)
                    else:
                        still.append(req)
                running = still

            if queue or running:
                for coll_name in ("queue", "running"):
                    coll = queue if coll_name == "queue" else running
                    kept = []
                    for req in coll:
                        cto = req.get("client_timeout_ms")
                        if cto is not None and t - req["arrival_ms"] >= cto:
                            st = tstat(req["tenant"])
                            wasted_units += req.get("work_done", 0.0)
                            if coll_name == "running":
                                mem_used -= req["footprint"]
                            if req.get("retries_left", 0) > 0:
                                dup = dict(req)
                                dup["arrival_ms"] = t + 100.0
                                dup["retries_left"] = req["retries_left"] - 1
                                for k in ("prefill_left", "decode_left",
                                          "footprint", "work_done",
                                          "out_done"):
                                    dup.pop(k, None)
                                injected.append(dup)
                            else:
                                st["errors"]["abandoned"] += 1
                            depart(req, t)
                        else:
                            kept.append(req)
                    if coll_name == "queue":
                        queue = kept
                    else:
                        running = kept

            if queue or running:
                for coll_name in ("queue", "running"):
                    coll = queue if coll_name == "queue" else running
                    kept = []
                    for req in coll:
                        if t - req["arrival_ms"] >= c["deadline_ms"]:
                            st = tstat(req["tenant"])
                            st["errors"]["timeout"] += 1
                            wasted_units += req.get("work_done", 0.0)
                            if coll_name == "running":
                                mem_used -= req["footprint"]
                            depart(req, t)
                        else:
                            kept.append(req)
                    if coll_name == "queue":
                        queue = kept
                    else:
                        running = kept

            spent = overhead_units + sum(s["work_units"]
                                         for s in tenants.values())
            if spent > c["billing_cap_units"]:
                cap_tripped = "billing"

            if recovery_t is None and last_noncontrol_depart is not None \
                    and t >= last_noncontrol_depart and not queue \
                    and len(running) <= max(1, c["max_batch"] // 2):
                recovery_t = t
            t += step

        end_t = t
        for coll in (queue, running):
            for req in coll:
                st = tstat(req["tenant"])
                st["errors"]["unfinished"] += 1
                wasted_units += req.get("work_done", 0.0)
        if last_noncontrol_depart is None:
            recovery_ms, recovered = 0.0, True
        elif recovery_t is not None:
            recovery_ms = max(0.0, recovery_t - last_noncontrol_depart)
            recovered = True
        else:
            recovery_ms = max(0.0, end_t - last_noncontrol_depart)
            recovered = False
        return {"tenants": tenants, "overhead_units": overhead_units,
                "wasted_units": wasted_units, "peak_mem_units": peak_mem,
                "mem_cap_units": c["mem_units"],
                "cache_evictions": evictions, "n_intake_total": n_intake,
                "sim_ms": end_t, "cap_tripped": cap_tripped,
                "recovery_time_ms": recovery_ms, "recovered": recovered}


def _base_request(tenant, arrival_ms, prompt_tokens, output_tokens,
                  input_bytes=None):
    return {"tenant": tenant, "arrival_ms": float(arrival_ms),
            "prompt_tokens": int(prompt_tokens),
            "output_tokens": int(output_tokens),
            "input_bytes": int(input_bytes if input_bytes is not None
                               else prompt_tokens * 4),
            "prefix_key": None, "prefix_tokens": 0, "decode_cost_mult": 1.0,
            "header_units": None, "declared_tokens": None,
            "client_timeout_ms": None, "retries_left": 0}


def make_control_stream(seed, caps, rate):
    """The benign CONTROL TENANT: fixed load at `rate` req/s over the whole
    horizon; moderate prompts sharing a small pool of boilerplate prefixes
    (the prefix cache should serve these well). Identical in every world of
    a (seed, budget) pair — the fairness baseline."""
    rng = random.Random(seed)
    reqs = []
    t = 250.0
    while True:
        t += rng.expovariate(rate / 1000.0)
        if t >= caps["duration_ms"]:
            break
        prompt = rng.randint(384, 640)
        r = _base_request("control", t, prompt, rng.randint(32, 96))
        r["prefix_key"] = f"ctl-{rng.randrange(CONTROL_PREFIX_POOL)}"
        r["prefix_tokens"] = CONTROL_PREFIX_TOKENS
        reqs.append(r)
    return reqs


def make_benign_extra(seed, budget, caps):
    """The NEGATIVE CONTROL's extra load: `budget` requests of exactly the
    control tenant's shape (same prefix pool, same sizes), arrivals uniform
    over the horizon like the control tenant — benign-only load at the same
    total rate as an attack cell (uniform so cold-cache first-draws are
    attributed symmetrically between the tenants). Amplification parity
    must be ~= 1 and degradation ~ 0."""
    rng = random.Random(seed + 5)
    reqs = []
    for _ in range(budget):
        prompt = rng.randint(384, 640)
        r = _base_request("benign-extra",
                          rng.uniform(500.0, caps["duration_ms"] - 500.0),
                          prompt, rng.randint(32, 96))
        r["prefix_key"] = f"ctl-{rng.randrange(CONTROL_PREFIX_POOL)}"
        r["prefix_tokens"] = CONTROL_PREFIX_TOKENS
        reqs.append(r)
    return reqs


def make_attack_stream(arm, seed, budget, caps):
    """`budget` attacker requests shaped per arm (arrivals inside the attack
    window). All shapes are valid or near-valid: nothing here exceeds the
    context window or output caps (protocol-oversize declarations are
    bounded at < 2x the window). Shapes are fully determined by
    (arm, seed, budget, caps)."""
    cw, mo = caps["context_window"], caps["max_output_tokens"]
    dur, step = caps["duration_ms"], caps["step_ms"]
    w0, w1 = dur * 0.05, dur * 0.30

    def spread(i):
        return w0 if budget <= 1 else w0 + (w1 - w0) * i / (budget - 1)

    reqs = []
    for i in range(budget):
        if arm == "long-context":
            r = _base_request("attacker", w0 + (i // 4) * step,
                              cw if i % 2 == 0 else int(cw * 0.9), 128)
        elif arm == "tokenizer-amplification":
            r = _base_request("attacker", spread(i), 2048, 128,
                              input_bytes=256)
        elif arm == "cache-thrash":
            r = _base_request("attacker", w0 + i * step, 2048, 32)
            r["prefix_key"] = f"thrash-{seed}-{i}"
            r["prefix_tokens"] = 1900
        elif arm == "batch-fragmentation":
            if i % 2 == 0:
                r = _base_request("attacker", w0 + i * 1.5 * step, 24, 1)
            else:
                r = _base_request("attacker", w0 + i * 1.5 * step, 700, 48)
        elif arm == "decode-amplification":
            r = _base_request("attacker", spread(i), 128, mo)
            r["decode_cost_mult"] = 4.0
        elif arm == "retry-storm":
            r = _base_request("attacker", spread(i), 2048, 256)
            r["client_timeout_ms"] = 1200.0
            r["retries_left"] = 3
            r["prefix_key"] = f"retry-{seed}-{i}"
            r["prefix_tokens"] = 1500
        elif arm == "memory-pressure":
            r = _base_request("attacker", w0 + (i // 8) * step, 2200, 384)
        elif arm == "protocol-oversize":
            r = _base_request("attacker", w0 + (i // 6) * step, 64, 1,
                              input_bytes=400)
            r["declared_tokens"] = cw + 1 + (i % 3) * 512
            r["header_units"] = 2000.0
        else:
            raise ValueError(f"unknown arm {arm!r} (known: {list(ARMS)})")
        reqs.append(r)
    return reqs


def simulate_training_skew(seed, budget, caps):
    """Synchronous data-parallel training: `workers` slots consume one
    microbatch per step each and the step closes at a BARRIER, so the step
    time is the MAX microbatch cost plus barrier overhead — one skewed
    microbatch inflates the step for every worker (the control tenant is
    the co-workers' throughput). Microbatch cost is superlinear in sequence
    length (attention): cost(L) = L + (L / 64)^2 units. The attacker
    submits max-length microbatches (valid but disproportionately costly);
    benign microbatches draw L ~ U(512, 1536). Deterministic given seed."""
    rng = random.Random(seed)
    workers, steps = 8, 24
    lmax = caps["context_window"]
    barrier = 100.0
    fee = 50.0

    def cost(length):
        return length + (length / 64.0) ** 2

    total_slots = workers * steps
    n_attack = min(budget, total_slots)
    attack_slots = set(rng.sample(range(total_slots), n_attack))
    benign_steps, attack_steps = [], []
    for s in range(steps):
        lens = [rng.randint(512, 1536) for _ in range(workers)]
        has_att = any(s * workers + w in attack_slots for w in range(workers))
        base = max(cost(L) for L in lens)
        benign_steps.append(base + barrier)
        attack_steps.append(max(base, cost(lmax) if has_att else 0.0)
                            + barrier)
    mean_benign = statistics.fmean(benign_steps)
    mean_attack = statistics.fmean(attack_steps)
    benign_gpu = sum(benign_steps) * workers
    attack_gpu = sum(attack_steps) * workers
    attacker_cost = n_attack * fee
    return {"n_attack": n_attack, "workers": workers, "steps": steps,
            "attacker_cost": attacker_cost,
            "benign_gpu_units": benign_gpu, "attack_gpu_units": attack_gpu,
            "cost_amplification": (attack_gpu - benign_gpu)
            / max(attacker_cost, _EPS),
            "step_time_inflation": mean_attack / max(mean_benign, _EPS),
            "control_throughput_drop": 1.0 - mean_benign
            / max(mean_attack, _EPS),
            "mean_benign_step_units": mean_benign}


def world_cell_record(arm, role, seed, budget, stats, caps, baseline=None):
    """One simulated world -> one journal record. `baseline` is the
    negative-control world's stats (same seed+budget): fairness degradation
    and eviction impact are measured against it."""
    tenants = stats["tenants"]
    ctrl = tenants.get("control") or _new_tstat()
    nonc = [s for t, s in tenants.items() if t != "control"]
    nc_work = sum(s["work_units"] for s in nonc)
    nc_cost = sum(s["cost_units"] for s in nonc)
    all_lat = [l for s in tenants.values() for l in s["latencies"]]
    work_total = sum(s["work_units"] for s in tenants.values())
    defender_total = stats["overhead_units"] + work_total
    errors = {"intake_reject": 0, "queue_reject": 0, "timeout": 0,
              "abandoned": 0, "unfinished": 0}
    for s in tenants.values():
        for k in errors:
            errors[k] += s["errors"][k]
    n_err = sum(errors.values())
    ctrl_err = sum(ctrl["errors"].values())
    amp_work = nc_work / max(nc_cost, _EPS)
    ctrl_amp = ctrl["work_units"] / max(ctrl["cost_units"], _EPS)
    hit_total = sum(s["hit_tokens"] for s in tenants.values())
    pfx_total = sum(s["prefix_tokens"] for s in tenants.values())
    n_completed = sum(s["n_completed"] for s in tenants.values())
    rec = {"experiment": EXPERIMENT_ID, "arm": arm, "role": role,
           "seed": seed, "budget": budget,
           "n_requests": stats["n_intake_total"],
           "n_control": ctrl["n_intake"], "n_completed": n_completed,
           "attacker_cost_units": _r6(nc_cost),
           "control_cost_units": _r6(ctrl["cost_units"]),
           "defender_work_units": _r6(work_total),
           "defender_overhead_units": _r6(stats["overhead_units"]),
           "defender_total_units": _r6(defender_total),
           "amplification": _r6(defender_total / max(nc_cost, _EPS)),
           "amplification_work": _r6(amp_work),
           "control_amplification_work": _r6(ctrl_amp),
           "amplification_vs_control": _r6(amp_work / max(ctrl_amp, _EPS)),
           "throughput_rps": _r6(n_completed
                                 / max(stats["sim_ms"] / 1000.0, _EPS)),
           "p50_ms": _r6(_pct(all_lat, 50)),
           "p95_ms": _r6(_pct(all_lat, 95)),
           "p99_ms": _r6(_pct(all_lat, 99)),
           "control_p50_ms": _r6(_pct(ctrl["latencies"], 50)),
           "control_p95_ms": _r6(_pct(ctrl["latencies"], 95)),
           "error_rate": _r6(n_err / max(1, stats["n_intake_total"])),
           "control_error_rate": _r6(ctrl_err / max(1, ctrl["n_intake"])),
           "peak_mem_units": _r6(stats["peak_mem_units"]),
           "mem_cap_units": _r6(stats["mem_cap_units"]),
           "mem_utilization": _r6(stats["peak_mem_units"]
                                  / max(stats["mem_cap_units"], _EPS)),
           "cache_hit_rate": (_r6(hit_total / pfx_total)
                              if pfx_total else None),
           "control_cache_hit_rate": (_r6(ctrl["hit_tokens"]
                                          / ctrl["prefix_tokens"])
                                      if ctrl["prefix_tokens"] else None),
           "cache_evictions": stats["cache_evictions"],
           "wasted_units": _r6(stats["wasted_units"]),
           "wasted_share": _r6(stats["wasted_units"]
                               / max(work_total, _EPS)),
           "recovery_time_ms": _r6(stats["recovery_time_ms"]),
           "recovered": stats["recovered"],
           "cap_tripped": stats["cap_tripped"],
           "manual_intervention_required": stats["cap_tripped"] is not None,
           "errors": errors}
    if role == "negative-control":
        rec["amplification_parity"] = rec["amplification_vs_control"]
        rec["control_p95_degradation"] = 0.0
        rec["eviction_impact"] = 0.0
    else:
        bctrl = ((baseline or {}).get("tenants") or {}).get("control") \
            or _new_tstat()
        bp95 = _pct(bctrl["latencies"], 95)
        p95 = _pct(ctrl["latencies"], 95)
        rec["control_p95_degradation"] = (
            _r6(p95 / bp95 - 1.0)
            if bp95 == bp95 and bp95 > 0 and p95 == p95 else None)
        bhit = (bctrl["hit_tokens"] / bctrl["prefix_tokens"]
                if bctrl["prefix_tokens"] else None)
        chit = (ctrl["hit_tokens"] / ctrl["prefix_tokens"]
                if ctrl["prefix_tokens"] else None)
        rec["eviction_impact"] = (_r6(bhit - chit)
                                  if bhit is not None and chit is not None
                                  else None)
    return rec


def training_skew_cell(cell_seed, seed, budget, caps):
    """The training-skew cell (own barrier model; see simulate_training_skew).
    The control tenant is the co-workers' throughput; `cost_amplification`
    is the extra defender GPU-units per attacker cost unit."""
    ts = simulate_training_skew(cell_seed, budget, caps)
    return {"experiment": EXPERIMENT_ID, "arm": "training-skew",
            "role": "attack", "seed": seed, "budget": budget,
            "n_attacker_microbatches": ts["n_attack"],
            "workers": ts["workers"], "steps": ts["steps"],
            "attacker_cost_units": _r6(ts["attacker_cost"]),
            "defender_total_units": _r6(ts["attack_gpu_units"]),
            "defender_benign_baseline_units": _r6(ts["benign_gpu_units"]),
            "amplification_vs_control": _r6(ts["cost_amplification"]),
            "cost_amplification": _r6(ts["cost_amplification"]),
            "step_time_inflation": _r6(ts["step_time_inflation"]),
            "control_throughput_drop": _r6(ts["control_throughput_drop"]),
            "control_p95_degradation": _r6(ts["control_throughput_drop"]),
            "error_rate": 0.0, "recovered": True,
            "recovery_time_ms": _r6(ts["mean_benign_step_units"]),
            "cap_tripped": None, "manual_intervention_required": False,
            "note": "synchronous-barrier model: one max-length microbatch "
                    "inflates the whole step; control tenant = co-workers"}


class LiveCapTripped(RuntimeError):
    """A hard safety cap was crossed during a live run: the driver MUST
    stop; the test is FAILED, never successful."""


class LiveCapGuard:
    """The hard safety envelope for live runs: request, wall-clock, and
    (optional) estimated-cost caps plus a health threshold. check() raises
    LiveCapTripped naming the crossed cap — the run auto-stops there."""

    def __init__(self, max_requests, max_seconds, max_cost_units=None,
                 max_error_rate=0.5, clock=time.monotonic):
        if max_requests is None or max_requests <= 0:
            raise ValueError("LiveCapGuard needs a positive max_requests cap")
        if max_seconds is None or max_seconds <= 0:
            raise ValueError("LiveCapGuard needs a positive max_seconds cap")
        if max_cost_units is not None and max_cost_units <= 0:
            raise ValueError("max_cost_units must be positive when given")
        self.max_requests = int(max_requests)
        self.max_seconds = float(max_seconds)
        self.max_cost_units = max_cost_units
        self.max_error_rate = float(max_error_rate)
        self.clock = clock
        self._t0 = None

    def start(self):
        self._t0 = self.clock()
        return self

    def elapsed_s(self):
        return 0.0 if self._t0 is None else self.clock() - self._t0

    def check(self, n_sent, n_errors=0, est_cost_units=0.0):
        """Return True while inside the envelope; raise LiveCapTripped at
        the FIRST crossed cap. Call around every request (auto-stop)."""
        if n_sent > self.max_requests:
            raise LiveCapTripped(
                f"max-requests cap crossed ({n_sent} > "
                f"{self.max_requests})")
        if self._t0 is not None and self.elapsed_s() > self.max_seconds:
            raise LiveCapTripped(
                f"max-seconds cap crossed ({self.elapsed_s():.1f}s > "
                f"{self.max_seconds}s)")
        if self.max_cost_units is not None \
                and est_cost_units >= self.max_cost_units:
            raise LiveCapTripped(
                f"max-cost cap crossed ({est_cost_units} >= "
                f"{self.max_cost_units} units)")
        if n_sent >= 8 and n_errors / n_sent > self.max_error_rate:
            raise LiveCapTripped(
                f"health threshold crossed ({n_errors}/{n_sent} errors > "
                f"{self.max_error_rate}); an uncontrolled outage is a "
                f"FAILED test, never a success")
        return True


def check_live_caps(args):
    """Refuse a live run without the hard safety caps. Returns the
    validated caps dict; SystemExit naming every missing/invalid flag."""
    problems = []
    ep = getattr(args, "endpoint", None)
    if not ep:
        problems.append("--endpoint URL")
    elif not (ep.startswith("http://") or ep.startswith("https://")):
        problems.append(f"--endpoint must be an http(s):// URL, got {ep!r}")
    mr = getattr(args, "max_requests", None)
    if mr is None or mr <= 0:
        problems.append("--max-requests N (positive int)")
    ms = getattr(args, "max_seconds", None)
    if ms is None or ms <= 0:
        problems.append("--max-seconds S (positive)")
    mc = getattr(args, "max_cost_units", None)
    if mc is not None and mc <= 0:
        problems.append("--max-cost-units must be positive when given")
    if problems:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] --live refused: missing/invalid safety "
            f"caps: {'; '.join(problems)}. A live availability test "
            f"without hard caps is never run.")
    return {"endpoint": ep, "max_requests": int(mr),
            "max_seconds": float(ms),
            "max_cost_units": float(mc) if mc is not None else None}


def render_live_request(arm, seq):
    """One bounded request template for the live driver (OpenAI-style
    /v1/completions payload). Sizes are HARD-BOUNDED by LIVE_PROMPT_CHAR_
    CAP / LIVE_MAX_TOKENS_CAP so a misconfigured arm cannot exceed the
    declared test envelope."""
    if arm == "long-context":
        return {"prompt": "context filler " * (LIVE_PROMPT_CHAR_CAP // 15),
                "max_tokens": 8, "stream": False}
    if arm == "tokenizer-amplification":
        return {"prompt": "🙂" * 512, "max_tokens": 128, "stream": False}
    if arm == "cache-thrash":
        return {"prompt": f"unique-prefix-{seq} " + "pad " * 200,
                "max_tokens": 1, "stream": False}
    if arm == "batch-fragmentation":
        return ({"prompt": "tiny", "max_tokens": 1, "stream": False}
                if seq % 2 == 0 else
                {"prompt": "medium " + "pad " * 120, "max_tokens": 48,
                 "stream": False})
    if arm == "decode-amplification":
        return {"prompt": "Count slowly, one word per line, starting at 1.",
                "max_tokens": LIVE_MAX_TOKENS_CAP, "stream": False}
    if arm == "retry-storm":
        return {"prompt": "Summarize a long document " + "pad " * 150,
                "max_tokens": 512, "stream": False,
                "client_timeout_ms": 1500, "retries": 3}
    if arm == "memory-pressure":
        return {"prompt": "memory " * (LIVE_PROMPT_CHAR_CAP // 14),
                "max_tokens": LIVE_MAX_TOKENS_CAP, "stream": False}
    if arm == "training-skew":
        return {"microbatch": {"seq_len": 4096, "samples": 8}}
    if arm == "protocol-oversize":
        return {"headers": {"X-Declared-Prompt-Tokens": str(1 << 20)},
                "prompt": "x", "max_tokens": 1, "stream": False}
    raise ValueError(f"unknown arm {arm!r} (known: {list(ARMS)})")


def build_live_plan(args, caps):
    """The bounded live request plan: per selected arm and budget point,
    request templates with schedule offsets, truncated at --max-requests
    and scheduled inside the first half of --max-seconds."""
    plan = []
    for arm in args.arms:
        for budget in args.budgets:
            for i in range(budget):
                if len(plan) >= caps["max_requests"]:
                    break
                plan.append({"arm": arm, "seq": i,
                             "payload": render_live_request(arm, i)})
    if not plan:
        spacing = 0.0
    else:
        spacing = max(50.0, 1000.0 / max(args.rate, 0.1))
        horizon = caps["max_seconds"] * 1000.0 / 2.0
        if len(plan) * spacing > horizon:
            spacing = horizon / len(plan)
    for i, item in enumerate(plan):
        item["delay_ms"] = round(i * spacing, 3)
    return {"caps": {k: v for k, v in caps.items() if k != "endpoint"},
            "n_requests": len(plan), "schedule_ms": spacing,
            "requests": plan}


def measure_live_request(endpoint, payload, timeout=10.0):
    """One timed JSON POST to the endpoint (fully implemented; the harness
    calls it under a LiveCapGuard). Returns (elapsed_ms, http_status)."""
    req = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        status = e.code
    return (time.perf_counter() - t0) * 1000.0, status


def render_live_control(seq):
    """The benign control tenant's request: a modest prompt + small output
    (the fair baseline). A unique per-seq prefix avoids a prefix-cache-hit
    confound on the control's measured service time."""
    return {"prompt": f"[benign {seq}] In one short sentence, say hello.",
            "max_tokens": LIVE_CONTROL_MAX_TOKENS, "stream": False}


def _parse_usage(body):
    """OpenAI-style `usage` object from a completion response body (bytes),
    or {} when absent/unparseable. Never raises."""
    if not body:
        return {}
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        return {}
    if not isinstance(obj, dict):
        return {}
    usage = obj.get("usage")
    return usage if isinstance(usage, dict) else {}


def measure_live_completion(endpoint, payload, timeout=LIVE_REQUEST_TIMEOUT_S):
    """One timed JSON POST that READS the response body: returns a dict with
    elapsed_ms (server service time — the objective defender-work proxy),
    http_status, ok (2xx), error ("timeout"|"network"|None), prompt_bytes,
    and completion_tokens/prompt_tokens (from `usage`, 0 if the server hides
    them). Network/HTTP failures are captured, never raised — the driver
    records them and the health cap decides when to stop."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    status, body, err = 0, b"", None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            status = getattr(resp, "status", 200) or 200
    except urllib.error.HTTPError as e:
        status = e.code
        try:
            body = e.read()
        except Exception:
            body = b""
    except (TimeoutError, socket.timeout):
        err = "timeout"
    except urllib.error.URLError as e:
        err = ("timeout" if isinstance(e.reason, (TimeoutError, socket.timeout))
               else "network")
    except Exception:
        err = "network"
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    usage = _parse_usage(body)
    return {"elapsed_ms": elapsed_ms, "http_status": status,
            "ok": 200 <= status < 300, "error": err,
            "prompt_bytes": len(data),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0)}


def _cap_name(exc):
    """Short cap label for the `cap_tripped` journal field."""
    s = str(exc)
    for needle, name in (("max-requests", "max_requests"),
                         ("max-seconds", "max_seconds"),
                         ("max-cost", "max_cost"), ("health", "health")):
        if needle in s:
            return name
    return "cap"


def _live_errors(measurements):
    """Map live per-request failures onto the toy error sub-schema (same
    five keys) so a live cell's `errors` is schema-identical."""
    errors = {"intake_reject": 0, "queue_reject": 0, "timeout": 0,
              "abandoned": 0, "unfinished": 0}
    for m in measurements:
        if m["ok"]:
            continue
        if m["error"] == "timeout":
            errors["timeout"] += 1
        elif m["error"] is not None:
            errors["abandoned"] += 1
        elif m["http_status"] == 429:
            errors["queue_reject"] += 1
        elif 400 <= m["http_status"] < 500:
            errors["intake_reject"] += 1
        else:
            errors["abandoned"] += 1
    return errors


def _ok_latencies(measurements):
    return [m["elapsed_ms"] for m in measurements if m["ok"]]


def _ok_tokens(measurements):
    return [m["completion_tokens"] for m in measurements if m["ok"]]


def _mean(vals):
    return statistics.fmean(vals) if vals else float("nan")


def _parity(vals):
    """Benign-vs-benign parity: split the control samples in half; a stable
    server shows ~1 (no self-amplification). None with <2 samples."""
    if len(vals) < 2:
        return None
    mid = len(vals) // 2
    a, b = _mean(vals[:mid]), _mean(vals[mid:])
    return a / b if b > 0 else None


def live_cell_record(arm, role, arm_ms, ctrl_ms, wall_s, cap_tripped, seed):
    """One live cell -> one journal record, mirroring world_cell_record's
    keys. Wire-invisible fields (memory, cache internals, concurrent
    fairness) are explicitly null. amplification_vs_control = attack server
    service-time per completed request / the benign control's (objective)."""
    lat, toks = _ok_latencies(arm_ms), _ok_tokens(arm_ms)
    clat, ctoks = _ok_latencies(ctrl_ms), _ok_tokens(ctrl_ms)
    n_req, n_ok = len(arm_ms), len(lat)
    n_ctrl, n_ctrl_ok = len(ctrl_ms), len(clat)
    work, ctrl_work = sum(lat), sum(clat)
    cost = float(n_ok) if n_ok else _EPS
    ctrl_cost = float(n_ctrl_ok) if n_ctrl_ok else _EPS
    amp_work = work / cost
    ctrl_amp = ctrl_work / ctrl_cost
    amp_vs_ctrl = amp_work / ctrl_amp if (clat and ctrl_amp > 0) else None
    mean_tok, mean_ctok = _mean(toks), _mean(ctoks)
    amp_tok = (mean_tok / mean_ctok
               if (ctoks and mean_ctok > 0 and mean_tok == mean_tok) else None)
    rec = {"experiment": EXPERIMENT_ID, "arm": arm, "role": role,
           "source": "live", "seed": seed, "budget": n_req,
           "n_requests": n_req, "n_control": n_ctrl, "n_completed": n_ok,
           "attacker_cost_units": _r6(cost),
           "control_cost_units": _r6(ctrl_cost),
           "defender_work_units": _r6(work),
           "defender_overhead_units": None,
           "defender_total_units": _r6(work),
           "amplification": _r6(amp_vs_ctrl),
           "amplification_work": _r6(amp_work),
           "control_amplification_work": _r6(ctrl_amp),
           "amplification_vs_control": _r6(amp_vs_ctrl),
           "latency_amplification": _r6(amp_vs_ctrl),
           "token_amplification": _r6(amp_tok),
           "mean_latency_ms": _r6(_mean(lat)),
           "control_mean_latency_ms": _r6(_mean(clat)),
           "mean_completion_tokens": _r6(mean_tok),
           "control_mean_completion_tokens": _r6(mean_ctok),
           "throughput_rps": _r6(n_ok / max(wall_s, _EPS)),
           "p50_ms": _r6(_pct(lat, 50)), "p95_ms": _r6(_pct(lat, 95)),
           "p99_ms": _r6(_pct(lat, 99)),
           "control_p50_ms": _r6(_pct(clat, 50)),
           "control_p95_ms": _r6(_pct(clat, 95)),
           "control_p95_degradation": None,
           "error_rate": _r6((n_req - n_ok) / max(1, n_req)),
           "control_error_rate": _r6((n_ctrl - n_ctrl_ok) / max(1, n_ctrl)),
           "peak_mem_units": None, "mem_cap_units": None,
           "mem_utilization": None,
           "cache_hit_rate": None, "control_cache_hit_rate": None,
           "cache_evictions": None,
           "wasted_units": None, "wasted_share": None,
           "recovery_time_ms": None, "recovered": cap_tripped is None,
           "cap_tripped": cap_tripped,
           "manual_intervention_required": cap_tripped is not None,
           "errors": _live_errors(arm_ms),
           "note": "live cost/latency amplification vs the benign control; "
                   "memory/cache/fairness fields are not wire-observable "
                   "(null)"}
    if role == "negative-control":
        rec["amplification_parity"] = _r6(_parity(clat))
        rec["control_p95_degradation"] = 0.0
    return rec


def _drive_live(caps, attack_items, args):
    """Send an interleaved benign-control + attack workload to the endpoint,
    strictly under LiveCapGuard. A warm-up request absorbs the cold model
    load. Returns (by_arm, control_measurements, wall_seconds, cap_tripped);
    cap_tripped is set only when a hard cap/health threshold auto-stops the
    run (exhausting the bounded plan is a normal, cap_tripped=None finish)."""
    endpoint = caps["endpoint"]
    timeout = min(caps["max_seconds"], LIVE_REQUEST_TIMEOUT_S)
    guard = LiveCapGuard(caps["max_requests"], caps["max_seconds"],
                         caps["max_cost_units"]).start()
    state = {"n_sent": 0, "n_err": 0, "est_cost": 0.0, "tripped": None}

    def send(payload):
        try:
            guard.check(state["n_sent"] + 1, state["n_err"], state["est_cost"])
        except LiveCapTripped as e:
            state["tripped"] = _cap_name(e)
            return None
        m = measure_live_completion(endpoint, payload, timeout=timeout)
        state["n_sent"] += 1
        if not m["ok"]:
            state["n_err"] += 1
        state["est_cost"] += m["completion_tokens"] + m["prompt_tokens"]
        try:
            guard.check(state["n_sent"], state["n_err"], state["est_cost"])
        except LiveCapTripped as e:
            state["tripped"] = _cap_name(e)
        return m

    by_arm, ctrl_ms, ctrl_seq = {}, [], 0
    send(render_live_control(ctrl_seq))
    ctrl_seq += 1
    for item in attack_items:
        if state["tripped"]:
            break
        cm = send(render_live_control(ctrl_seq))
        ctrl_seq += 1
        if cm is not None:
            ctrl_ms.append(cm)
        if state["tripped"]:
            break
        am = send(item["payload"])
        if am is not None:
            by_arm.setdefault(item["arm"], []).append(am)
    return by_arm, ctrl_ms, guard.elapsed_s(), state["tripped"]


LIVE_INTERP = (
    "live cost/latency amplification (OWASP LLM06): amplification_vs_control "
    "= attack server-service-time per completed request / the benign "
    "control's; > 1 means the endpoint spends more compute per request on "
    "the attack workload than on benign load (objective — latency/tokens, no "
    "classifier). token_amplification is the output-token analog. "
    "cap_tripped != null + manual_intervention_required means a hard safety "
    "cap or health threshold auto-stopped the run — an uncontrolled outage, "
    "a FAILED test, never a success. Memory/cache/fairness fields are not "
    "wire-observable (null); concurrent fairness-degradation is out of scope "
    "for this bounded closed-loop driver.")


def _make_live_artifact(args, caps, arms_run):
    """The standard artifact skeleton for a live run — SAME schema string as
    the toy tier; config marks live=True and carries the safety caps."""
    config = {"attack": EXPERIMENT_ID, "mode": getattr(args, "mode", None),
              "toy": False, "live": True, "endpoint": caps["endpoint"],
              "arms": list(arms_run), "budgets": list(args.budgets),
              "seeds": [args.seed], "seed": args.seed, "rate": args.rate,
              "caps": {"max_requests": caps["max_requests"],
                       "max_seconds": caps["max_seconds"],
                       "max_cost_units": caps["max_cost_units"]}}
    return artifacts.make_artifact(
        "dtraining.attacker.resource_exhaustion.v1", config,
        _threat_model_doc(), interpretation=LIVE_INTERP)


def run_live(args):
    """Drive a REAL OpenAI-style endpoint: validate + journal the bounded
    plan, then measure attack-vs-control cost/latency amplification under the
    hard safety envelope, emitting the toy tier's artifact schema."""
    caps = check_live_caps(args)
    plan = build_live_plan(args, caps)
    artifacts.append_jsonl(args.output, {
        "experiment": EXPERIMENT_ID, "arm": "live-plan", "role": "plan",
        "caps": plan["caps"], "n_planned_requests": plan["n_requests"],
        "schedule_ms": plan["schedule_ms"],
        "note": "bounded plan validated; driving the live target under "
                "LiveCapGuard"})
    completion_items = [it for it in plan["requests"] if "prompt" in it["payload"]]
    for skipped in sorted({it["arm"] for it in plan["requests"]
                           if "prompt" not in it["payload"]}):
        artifacts.append_jsonl(args.output, {
            "experiment": EXPERIMENT_ID, "arm": skipped, "role": "skipped-live",
            "note": "not a completions request; measurable only on the toy "
                    "serving stack, not a live /v1/completions endpoint"})
    budget = max(0, (caps["max_requests"] - 1) // 2)
    completion_items = completion_items[:budget]
    by_arm, ctrl_ms, wall_s, cap_tripped = _drive_live(caps, completion_items,
                                                       args)

    out = _make_live_artifact(args, caps, sorted(by_arm))
    per_arm = {}
    ctl_rec = live_cell_record(NEGATIVE_CONTROL, "negative-control",
                               ctrl_ms, ctrl_ms, wall_s, cap_tripped,
                               args.seed)
    artifacts.append_jsonl(args.output, ctl_rec)
    out["results"].append(ctl_rec)
    per_arm[NEGATIVE_CONTROL] = [ctl_rec]
    cap_txt = f" CAP-TRIPPED={cap_tripped}" if cap_tripped else ""
    cp = ctl_rec["control_mean_latency_ms"]
    cp_txt = f"{cp:.1f}ms" if cp is not None else "n/a"
    print(f"[resource-exhaustion/live] {NEGATIVE_CONTROL}: "
          f"n_control={ctl_rec['n_control']} ctrl_mean={cp_txt} "
          f"parity={ctl_rec.get('amplification_parity')}{cap_txt}")
    for arm in sorted(by_arm):
        rec = live_cell_record(arm, "attack", by_arm[arm], ctrl_ms, wall_s,
                               cap_tripped, args.seed)
        artifacts.append_jsonl(args.output, rec)
        out["results"].append(rec)
        per_arm.setdefault(arm, []).append(rec)
        amp = rec["amplification_vs_control"]
        amp_txt = f"{amp:.2f}x" if amp is not None else "n/a"
        print(f"[resource-exhaustion/live] {arm}: "
              f"n={rec['n_completed']}/{rec['n_requests']} "
              f"amp_vs_ctrl={amp_txt} "
              f"tok_amp={rec['token_amplification']} "
              f"err={rec['error_rate']:.1%}{cap_txt}")
    _summary_rows(out, per_arm)
    artifacts.write_artifact(args.output, out, evidence=_make_evidence(args))
    if cap_tripped:
        print(f"[resource-exhaustion/live] SAFETY STOP: {cap_tripped} cap/"
              f"threshold crossed — recorded as a FAILED test (uncontrolled "
              f"outage is never a success).")
    return 0


def _make_evidence(args):
    """An EvidenceLogger when --evidence and --output are set (None
    otherwise, with the repo-convention warning)."""
    if not getattr(args, "evidence", False):
        return None
    if not args.output:
        print(f"[{EXPERIMENT_ID}] --evidence needs --output; continuing "
              "without an evidence chain")
        return None
    from ..evidence import EvidenceLogger
    return EvidenceLogger(args.output, node_role=f"attacker/{EXPERIMENT_ID}")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--evidence", action="store_true",
                    help="write a tamper-evident evidence chain alongside "
                         "the artifact (attacker.evidence)")
    ap.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS),
                    help="attack arm(s) to run (default: all nine)")
    ap.add_argument("--budgets", type=int, nargs="+", default=[8, 24, 64],
                    help="attacker requests (or microbatches) per cell; "
                         "--quick keeps the first 2")
    ap.add_argument("--rate", type=float, default=2.0,
                    help="benign control-tenant load, req/s")
    ap.add_argument("--duration-ms", type=int, default=20000,
                    help="simulation horizon (hard duration cap)")
    ap.add_argument("--step-ms", type=int, default=20,
                    help="scheduler step granularity (ms)")
    ap.add_argument("--context-window", type=int, default=4096,
                    help="max prompt tokens; longer prompts rejected")
    ap.add_argument("--max-output-tokens", type=int, default=512,
                    help="server-side output cap")
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--mem-units", type=float, default=40000.0,
                    help="KV memory cap in units (1 unit/token; hard)")
    ap.add_argument("--cache-tokens", type=int, default=24000,
                    help="prefix-cache capacity in tokens (LRU)")
    ap.add_argument("--queue-max", type=int, default=64,
                    help="queue cap; excess requests are rejected (429)")
    ap.add_argument("--billing-cap-units", type=float, default=20000000.0,
                    help="hard cost cap; crossing it stops the run")
    ap.add_argument("--deadline-ms", type=float, default=12000.0,
                    help="server-side per-request deadline")
    ap.add_argument("--live", action="store_true",
                    help="real target. Requires --endpoint plus the hard "
                         "caps --max-requests and --max-seconds "
                         "(--max-cost-units recommended)")
    ap.add_argument("--endpoint", default=None,
                    help="target URL (OpenAI-style /v1/completions or any "
                         "JSON POST endpoint)")
    ap.add_argument("--max-requests", type=int, default=None,
                    help="hard request cap for --live (required)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="hard wall-clock cap for --live (required)")
    ap.add_argument("--max-cost-units", type=float, default=None,
                    help="hard estimated-cost cap for --live (optional)")
    return ap


def server_caps(args):
    """The toy world's hard caps from CLI args (fixed cost/fee schedule)."""
    return {"duration_ms": args.duration_ms, "step_ms": args.step_ms,
            "max_requests": 100000, "billing_cap_units":
                args.billing_cap_units,
            "context_window": args.context_window,
            "max_output_tokens": args.max_output_tokens,
            "max_batch": args.max_batch, "mem_units": args.mem_units,
            "cache_tokens": args.cache_tokens, "queue_max": args.queue_max,
            "step_budget_units": 12000.0, "fixed_overhead_units": 2000.0,
            "prefill_unit_cost": 1.0, "decode_unit_cost": 2.0,
            "header_parse_units": 20.0, "deadline_ms": args.deadline_ms,
            "request_fee_units": 5.0, "byte_fee_units": 0.01,
            "out_token_fee_units": 0.05}


def _threat_model_doc():
    return threat_model.skeleton(
        "active API client / authenticated tenant (or malicious split "
        "client) stresses availability, cost, and fairness of the serving "
        "stack with valid or near-valid workloads; measured on a simulated "
        "stack with hard caps against a concurrent benign control tenant.",
        scope={"use_case": "split-inference", "deployment": "multi-tenant",
               "in_scope": ["inference serving stack: intake/parser, "
                            "request queue, prefix cache, batcher, KV "
                            "memory allocator, billing meter",
                            "split-training coordinator: synchronous "
                            "microbatch intake (training-skew arm)",
                            "adversarial but schema-valid / near-valid "
                            "requests within a declared rate budget"],
               "out_of_scope": ["volumetric L3/L4 DDoS and botnet floods",
                                "credential theft or unauthenticated access",
                                "live targets without an operator-wired "
                                "harness (integration point only)"]},
        actor={"role": "API user", "intent": "active",
               "persistence": "session",
               "authorization": "valid tenant credentials; operates inside "
                                "the declared request-rate budget unless "
                                "explicitly testing rate limits",
               "knowledge": ["public API schema and limits",
                             "approximate context window / output caps"],
               "capabilities": ["craft request shapes: length, token "
                                "expansion, prefix reuse, decoding "
                                "settings, retries, header declarations",
                                "sustain a bounded request rate for one "
                                "session"],
               "resources": ["one API key / tenant session",
                             "modest request budget (the --budgets grid)"],
               "vantage": ["public inference API",
                           "training microbatch intake (split client)"]},
        target={"assets": ["service availability",
                           "compute and billing budget",
                           "per-tenant fairness", "prefix-cache utility"],
                "components": ["request queue", "LRU prefix cache",
                               "continuous batcher", "tokenizer/parser "
                               "intake", "KV memory allocator",
                               "billing meter"],
                "trust_boundaries": ["tenant <-> shared serving "
                                     "infrastructure"],
                "entry_points": ["public completions endpoint",
                                 "microbatch intake (split-training "
                                 "client)"]},
        scenario={"preconditions": ["multi-tenant or shared serving stack",
                                    "attacker holds valid tenant access"],
                  "lifecycle_stage": ["inference", "operations",
                                      "training"],
                  "security_properties": ["availability"],
                  "attack_steps": ["shape adversarial-but-valid workloads "
                                   "per arm",
                                   "sweep the request-budget grid",
                                   "measure amplification, latency, "
                                   "fairness, cache, memory, and recovery "
                                   "against the benign control tenant"],
                  "technical_impact": ["cost/compute amplification",
                                       "control-tenant latency "
                                       "degradation and starvation",
                                       "cache thrash and evictions",
                                       "allocator pressure",
                                       "duplicate work via retries"],
                  "business_impact": ["unexpected billing",
                                      "SLA violations for benign tenants",
                                      "manual incident response"]},
        controls={"assumed": ["authentication and per-tenant rate limits "
                              "exist (not under test here)"],
                  "tested": ["hard CPU/memory/duration/request/billing "
                             "caps",
                             "auto-stop on crossed health or cost "
                             "thresholds"],
                  "detective": ["billing and health thresholds trip the "
                                "run (cap_tripped)"],
                  "response": ["the run auto-stops; an uncontrolled "
                               "outage is recorded as a failed test, "
                               "never a success"],
                  "residual_risk": "simulated stack: absolute numbers are "
                                   "abstract units, not a calibrated "
                                   "deployment",
                  "status": "open"},
        mappings={"owasp_llm_2026": ["LLM06"], "cwe": ["CWE-400"]},
        evaluation={"success_metrics": ["amplification vs the control "
                                        "tenant",
                                        "control-tenant p95 degradation / "
                                        "starvation",
                                        "cache hit-rate and eviction "
                                        "impact",
                                        "peak memory vs cap",
                                        "recovery time / cap_tripped"],
                    "baseline": "benign-only load at the same total rate "
                                "(the benign-control cell: amplification "
                                "parity ~= 1, fairness degradation ~ 0)",
                    "budget": {"requests_per_cell": "budgets grid "
                               "(default 8/24/64; quick: first 2)",
                               "duration_ms": 20000,
                               "live": "--max-requests / --max-seconds / "
                                       "--max-cost-units"},
                    "limitations": ["abstract compute/memory/billing "
                                    "units — not calibrated to a specific "
                                    "GPU/model/tokenizer",
                                    "single node: no replicas, "
                                    "autoscaling, preemption, chunked "
                                    "prefill, or paged KV",
                                    "training-skew models only a "
                                    "synchronous data-parallel barrier",
                                    "live driver is an integration "
                                    "point: cap-check + plan + timing "
                                    "helpers are implemented and "
                                    "unit-tested; the send loop must be "
                                    "wired to an operator harness under "
                                    "LiveCapGuard"],
                    "coverage_status": "partial"})


def _make_artifact(args, caps):
    config = {"attack": EXPERIMENT_ID, "mode": getattr(args, "mode", None),
              "toy": True, "arms": list(args.arms),
              "budgets": list(args.budgets), "seeds": list(args.seeds),
              "seed": args.seed, "rate": args.rate, "caps": caps,
              "live": False}
    return artifacts.make_artifact(
        "dtraining.attacker.resource_exhaustion.v1", config,
        _threat_model_doc(),
        interpretation="amplification_vs_control >> 1 means the arm burns "
                       "defender compute per unit attacker cost far beyond "
                       "benign load; control_p95_degradation > 0 means "
                       "co-tenant fairness suffers; eviction_impact > 0 "
                       "means the attacker evicts useful cached prefixes; "
                       "cap_tripped + manual_intervention_required=true "
                       "means the safety stop fired — an uncontrolled "
                       "outage, recorded as a FAILED test, never a "
                       "success. The benign-control cell must show "
                       "amplification_parity ~= 1 and ~0 degradation "
                       "(mandatory negative control).")


def _check_toy_args(args):
    if not args.budgets or any(b <= 0 for b in args.budgets):
        raise SystemExit(f"[{EXPERIMENT_ID}] --budgets must be positive "
                         f"integers, got {args.budgets}")
    if args.rate <= 0:
        raise SystemExit(f"[{EXPERIMENT_ID}] --rate must be > 0 req/s, "
                         f"got {args.rate}")
    if args.step_ms <= 0 or args.duration_ms < 20 * args.step_ms:
        raise SystemExit(f"[{EXPERIMENT_ID}] --duration-ms must be >= 20x "
                         f"--step-ms (got {args.duration_ms} / "
                         f"{args.step_ms})")
    if args.mem_units < 2 * (args.context_window + args.max_output_tokens):
        raise SystemExit(f"[{EXPERIMENT_ID}] --mem-units must hold at "
                         f"least two full-context sequences (got "
                         f"{args.mem_units}; need >= "
                         f"{2 * (args.context_window + args.max_output_tokens)})")


def run_toy(args):
    _check_toy_args(args)
    if args.quick:
        args.budgets = args.budgets[:2]
        args.seeds = args.seeds[:2]
    caps = server_caps(args)
    out = _make_artifact(args, caps)
    per_arm = {}
    for seed in args.seeds:
        for bi, budget in enumerate(args.budgets):
            base_seed = args.seed + 1000 * seed + 97 * bi
            try:
                control = make_control_stream(base_seed * 31 + 7, caps,
                                              args.rate)
                base_stats = SimServer(base_seed, caps,
                                       warm_cache=control_warm_keys()).run(
                    control + make_benign_extra(base_seed, budget, caps))
                rec = world_cell_record(NEGATIVE_CONTROL,
                                        "negative-control", seed, budget,
                                        base_stats, caps)
            except Exception as e:
                journal_error(args.output, EXPERIMENT_ID,
                              {"arm": NEGATIVE_CONTROL, "seed": seed,
                               "budget": budget}, e)
                continue
            artifacts.append_jsonl(args.output, rec)
            out["results"].append(rec)
            per_arm.setdefault(NEGATIVE_CONTROL, []).append(rec)
            p95_txt = (f"{rec['control_p95_ms']:.0f}ms"
                       if rec["control_p95_ms"] is not None else "n/a")
            print(f"[resource-exhaustion] {NEGATIVE_CONTROL} seed={seed} "
                  f"budget={budget}: parity="
                  f"{rec['amplification_parity']:.3f} "
                  f"ctrl_p95={p95_txt} "
                  f"err={rec['error_rate']:.1%}"
                  + (f" CAP-TRIPPED={rec['cap_tripped']}"
                     if rec["cap_tripped"] else ""))
            for arm in args.arms:
                cell_seed = base_seed + 7919 * (ARMS.index(arm) + 1)
                try:
                    if arm == "training-skew":
                        rec = training_skew_cell(cell_seed, seed, budget,
                                                 caps)
                    else:
                        world = make_control_stream(base_seed * 31 + 7,
                                                    caps, args.rate) \
                            + make_attack_stream(arm, cell_seed, budget,
                                                 caps)
                        stats = SimServer(cell_seed, caps,
                                          warm_cache=control_warm_keys()
                                          ).run(world)
                        rec = world_cell_record(arm, "attack", seed,
                                                budget, stats, caps,
                                                baseline=base_stats)
                except Exception as e:
                    journal_error(args.output, EXPERIMENT_ID,
                                  {"arm": arm, "seed": seed,
                                   "budget": budget}, e)
                    continue
                artifacts.append_jsonl(args.output, rec)
                out["results"].append(rec)
                per_arm.setdefault(arm, []).append(rec)
                if arm == "training-skew":
                    print(f"[resource-exhaustion] {arm} seed={seed} "
                          f"budget={budget}: step_infl="
                          f"{rec['step_time_inflation']:.2f}x "
                          f"cost_amp={rec['cost_amplification']:.1f}x "
                          f"ctrl_tp_drop="
                          f"{rec['control_throughput_drop']:.1%}")
                else:
                    deg = rec["control_p95_degradation"]
                    deg_txt = f"{deg:+.0%}" if deg is not None else "n/a"
                    print(f"[resource-exhaustion] {arm} seed={seed} "
                          f"budget={budget}: amp_vs_ctrl="
                          f"{rec['amplification_vs_control']:.2f}x "
                          f"ctrl_p95_deg={deg_txt} "
                          f"err={rec['error_rate']:.1%} "
                          f"peak_mem={rec['peak_mem_units']:.0f}")
    _summary_rows(out, per_arm)
    artifacts.write_artifact(args.output, out,
                         evidence=_make_evidence(args))
    return 0


def _summary_rows(out, per_arm):
    def mean_of(recs, key):
        vals = [r[key] for r in recs
                if isinstance(r.get(key), (int, float))]
        return _r6(statistics.fmean(vals)) if vals else None

    for arm, recs in sorted(per_arm.items()):
        row = {"experiment": EXPERIMENT_ID, "arm": arm,
               "n_seeds": len({r["seed"] for r in recs}),
               "n_cells": len(recs)}
        if arm == NEGATIVE_CONTROL:
            row["amplification_parity_mean"] = mean_of(
                recs, "amplification_parity")
            row["error_rate_mean"] = mean_of(recs, "error_rate")
        elif arm == "training-skew":
            row["cost_amplification_mean"] = mean_of(
                recs, "cost_amplification")
            row["step_time_inflation_mean"] = mean_of(
                recs, "step_time_inflation")
            row["control_throughput_drop_mean"] = mean_of(
                recs, "control_throughput_drop")
        else:
            row["amplification_vs_control_mean"] = mean_of(
                recs, "amplification_vs_control")
            row["control_p95_degradation_mean"] = mean_of(
                recs, "control_p95_degradation")
            row["error_rate_mean"] = mean_of(recs, "error_rate")
            row["peak_mem_units_max"] = max(
                (r["peak_mem_units"] or 0) for r in recs)
            row["recovered_fraction"] = _r6(
                sum(1 for r in recs if r["recovered"]) / len(recs))
        out["summary"].append(row)


def run(args):
    if getattr(args, "live", False) and getattr(args, "toy", False):
        raise SystemExit(f"[{EXPERIMENT_ID}] --live and --toy are mutually "
                         f"exclusive (the simulated stack vs a real "
                         f"target).")
    if getattr(args, "live", False):
        return run_live(args)
    if args.toy:
        return run_toy(args)
    raise SystemExit(f"[{EXPERIMENT_ID}] pick a target: --toy (local "
                     f"simulated serving stack, hard caps) or --live "
                     f"--endpoint URL --max-requests N --max-seconds S "
                     f"[--max-cost-units C] (real target, hard safety "
                     f"envelope).")


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("resource-exhaustion availability/cost/fairness simulator "
          "(pure python):")

    check("v2 threat model validates",
          threat_model.validate(_threat_model_doc()) is not None)
    check("percentile anchors",
          _pct([1, 2, 3, 4], 50) == 2.5
          and _pct([1, 2, 3, 4], 100) == 4.0
          and _pct([], 95) != _pct([], 95))

    import types
    args = types.SimpleNamespace(
        duration_ms=20000, step_ms=20, context_window=4096,
        max_output_tokens=512, max_batch=16, mem_units=40000.0,
        cache_tokens=24000, queue_max=64, billing_cap_units=20000000.0,
        deadline_ms=12000.0)
    caps = server_caps(args)
    seed, budget = 0, 24
    base_seed = 1234 + 1000 * seed
    control = make_control_stream(base_seed * 31 + 7, caps, 2.0)

    benign = SimServer(base_seed, caps,
                       warm_cache=control_warm_keys()).run(
        control + make_benign_extra(base_seed, budget, caps))
    ctl_rec = world_cell_record(NEGATIVE_CONTROL, "negative-control",
                                seed, budget, benign, caps)
    check("negative control: zero errors, no cap tripped, recovered",
          ctl_rec["error_rate"] == 0.0 and ctl_rec["cap_tripped"] is None
          and ctl_rec["recovered"])
    check("negative control: amplification parity ~= 1",
          0.85 <= ctl_rec["amplification_parity"] <= 1.18)
    check("negative control: control tenant latency finite",
          ctl_rec["control_p95_ms"] is not None
          and ctl_rec["control_p95_ms"] > 0)

    benign2 = SimServer(base_seed, caps,
                        warm_cache=control_warm_keys()).run(
        make_control_stream(base_seed * 31 + 7, caps, 2.0)
        + make_benign_extra(base_seed, budget, caps))
    check("deterministic world: identical peak_mem/overhead/evictions",
          benign["peak_mem_units"] == benign2["peak_mem_units"]
          and benign["overhead_units"] == benign2["overhead_units"]
          and benign["cache_evictions"] == benign2["cache_evictions"])

    def attack_cell(arm):
        cs = base_seed + 7919 * (ARMS.index(arm) + 1)
        world = make_control_stream(base_seed * 31 + 7, caps, 2.0) \
            + make_attack_stream(arm, cs, budget, caps)
        return world_cell_record(arm, "attack", seed, budget,
                                 SimServer(cs, caps,
                                           warm_cache=control_warm_keys()
                                           ).run(world),
                                 caps, baseline=benign)

    tok = attack_cell("tokenizer-amplification")
    check("tokenizer-amplification: cost amplification >> control",
          tok["amplification_vs_control"] >= 3.0)
    dec = attack_cell("decode-amplification")
    check("decode-amplification: cost amplification >> control",
          dec["amplification_vs_control"] >= 3.0)
    lc = attack_cell("long-context")
    check("long-context: control-tenant fairness degrades",
          lc["control_p95_degradation"] is not None
          and lc["control_p95_degradation"] >= 0.3)
    ct = attack_cell("cache-thrash")
    check("cache-thrash: evictions and control cache impact",
          ct["cache_evictions"] > 0
          and ct["eviction_impact"] is not None
          and ct["eviction_impact"] > 0.05)
    rs = attack_cell("retry-storm")
    check("retry-storm: duplicate work shows as wasted share",
          rs["wasted_share"] >= 0.2)
    mp = attack_cell("memory-pressure")
    check("memory-pressure: hard memory cap never exceeded",
          mp["peak_mem_units"] <= mp["mem_cap_units"])
    po = attack_cell("protocol-oversize")
    check("protocol-oversize: all attacker requests rejected at intake, "
          "parse budget burned",
          po["errors"]["intake_reject"] == budget
          and po["amplification_vs_control"] >= 2.0)

    ts = training_skew_cell(base_seed + 7919 * 8, seed, budget, caps)
    check("training-skew: skewed microbatch inflates the barrier step",
          ts["step_time_inflation"] >= 2.0
          and ts["control_throughput_drop"] > 0.0)

    tiny = dict(caps, billing_cap_units=50000.0)
    tripped = SimServer(base_seed, tiny,
                        warm_cache=control_warm_keys()).run(
        make_control_stream(base_seed * 31 + 7, tiny, 2.0)
        + make_attack_stream("decode-amplification", base_seed + 7919 * 5,
                             budget, tiny))
    check("billing cap auto-stops the run (never a success)",
          tripped["cap_tripped"] == "billing")

    try:
        check_live_caps(types.SimpleNamespace(
            endpoint=None, max_requests=10, max_seconds=5,
            max_cost_units=None))
        check("--live without --endpoint refused", False)
    except SystemExit as e:
        check("--live without --endpoint refused", "--endpoint" in str(e))
    try:
        check_live_caps(types.SimpleNamespace(
            endpoint="http://h:8000/v1/completions", max_requests=None,
            max_seconds=5, max_cost_units=None))
        check("--live without --max-requests refused", False)
    except SystemExit as e:
        check("--live without --max-requests refused",
              "--max-requests" in str(e))
    try:
        check_live_caps(types.SimpleNamespace(
            endpoint="ftp://h", max_requests=10, max_seconds=5,
            max_cost_units=None))
        check("non-http endpoint refused naming the value", False)
    except SystemExit as e:
        check("non-http endpoint refused naming the value",
              "ftp://h" in str(e))
    caps_live = check_live_caps(types.SimpleNamespace(
        endpoint="http://h:8000/v1/completions", max_requests=10,
        max_seconds=5, max_cost_units=100.0))
    check("valid live caps accepted",
          caps_live["max_requests"] == 10
          and caps_live["max_cost_units"] == 100.0)

    now = [0.0]
    guard = LiveCapGuard(5, 10.0, max_cost_units=50.0,
                         clock=lambda: now[0]).start()
    check("guard allows requests inside the envelope",
          guard.check(5, n_errors=0, est_cost_units=49.0))
    try:
        guard.check(6)
        check("guard trips at max-requests", False)
    except LiveCapTripped as e:
        check("guard trips at max-requests", "max-requests" in str(e))
    now[0] = 11.0
    try:
        guard.check(5)
        check("guard trips at max-seconds", False)
    except LiveCapTripped as e:
        check("guard trips at max-seconds", "max-seconds" in str(e))
    try:
        LiveCapGuard(5, 10.0, max_cost_units=1.0,
                     clock=lambda: 0.0).start().check(
                         1, est_cost_units=1.0)
        check("guard trips at max-cost", False)
    except LiveCapTripped as e:
        check("guard trips at max-cost", "max-cost" in str(e))
    try:
        LiveCapGuard(50, 10.0, clock=lambda: 0.0).start().check(
            10, n_errors=6)
        check("guard trips on unhealthy error rate", False)
    except LiveCapTripped as e:
        check("guard trips on unhealthy error rate", "health" in str(e))
    try:
        LiveCapGuard(None, 10.0)
        check("guard refuses a missing request cap", False)
    except ValueError:
        check("guard refuses a missing request cap", True)

    for arm in ARMS:
        p = render_live_request(arm, 0)
        json.dumps(p)
        text = json.dumps(p)
        check(f"live template for {arm} is bounded and serializable",
              len(text) <= 12000)
    try:
        render_live_request("no-such-arm", 0)
        check("unknown live arm rejected", False)
    except ValueError:
        check("unknown live arm rejected", True)

    def _mm(ms, ok=True, status=200, tok=0, err=None):
        return {"elapsed_ms": ms, "http_status": status, "ok": ok,
                "error": err, "prompt_bytes": 100, "completion_tokens": tok,
                "prompt_tokens": 5}

    check("live control template is bounded + benign",
          "prompt" in render_live_control(3)
          and render_live_control(3)["max_tokens"] == LIVE_CONTROL_MAX_TOKENS
          and len(json.dumps(render_live_control(3))) <= 12000)
    check("usage parsed from a completion body (0 when absent)",
          _parse_usage(json.dumps(
              {"usage": {"completion_tokens": 42, "prompt_tokens": 7}}
          ).encode())["completion_tokens"] == 42
          and _parse_usage(b"") == {} and _parse_usage(b"not json") == {})
    ctrl_m = [_mm(10.0, tok=16), _mm(12.0, tok=16), _mm(11.0, tok=16),
              _mm(9.0, tok=16)]
    atk_m = [_mm(200.0, tok=1024), _mm(220.0, tok=1024), _mm(180.0, tok=1024)]
    rec = live_cell_record("decode-amplification", "attack", atk_m, ctrl_m,
                           5.0, None, 0)
    check("live amplification_vs_control == attack/control latency ratio",
          rec["amplification_vs_control"] is not None
          and abs(rec["amplification_vs_control"]
                  - _mean(_ok_latencies(atk_m)) / _mean(_ok_latencies(ctrl_m)))
          < 1e-6)
    check("live token_amplification == output-token ratio (64x)",
          abs(rec["token_amplification"] - 64.0) < 1e-6)
    check("live cell mirrors the toy schema keys",
          all(k in rec for k in
              ("experiment", "arm", "role", "seed", "budget",
               "amplification_vs_control", "throughput_rps", "error_rate",
               "p95_ms", "errors", "cap_tripped",
               "manual_intervention_required")))
    check("wire-invisible fields are explicitly null on live cells",
          rec["peak_mem_units"] is None and rec["cache_evictions"] is None
          and rec["control_p95_degradation"] is None
          and rec["defender_overhead_units"] is None)
    check("healthy live cell: recovered, no manual intervention",
          rec["recovered"] is True
          and rec["manual_intervention_required"] is False)
    nc = live_cell_record(NEGATIVE_CONTROL, "negative-control", ctrl_m,
                          ctrl_m, 5.0, None, 0)
    check("live negative control: vs_control == 1, parity ~1",
          abs(nc["amplification_vs_control"] - 1.0) < 1e-6
          and nc["amplification_parity"] is not None
          and 0.8 <= nc["amplification_parity"] <= 1.25
          and nc["control_p95_degradation"] == 0.0)
    errs = _live_errors([_mm(1, ok=False, status=429),
                         _mm(1, ok=False, status=400),
                         _mm(1, ok=False, err="timeout", status=0),
                         _mm(1, ok=False, err="network", status=0),
                         _mm(1, ok=True, status=200, tok=8)])
    check("live errors map onto the toy error sub-schema (5 keys)",
          set(errs) == {"intake_reject", "queue_reject", "timeout",
                        "abandoned", "unfinished"}
          and errs["queue_reject"] == 1 and errs["intake_reject"] == 1
          and errs["timeout"] == 1 and errs["abandoned"] == 1)
    check("a tripped cap marks the live cell FAILED (manual intervention)",
          live_cell_record("decode-amplification", "attack", atk_m, ctrl_m,
                           5.0, "health", 0)["manual_intervention_required"]
          is True)
    check("cap-name labels the crossed cap",
          _cap_name(LiveCapTripped("max-seconds cap crossed (x)"))
          == "max_seconds"
          and _cap_name(LiveCapTripped("health threshold crossed"))
          == "health")

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
