#!/usr/bin/env python3
"""router-side-channel — passive exploitation of MoE / dynamic-routing
metadata (surface S12, THREAT_MODEL_REVIEW P1): selected expert ids, top-k
routing weights, per-expert queue depth / capacity overflows, and
cross-tenant contention on shared experts leak token properties, sequence
domains and lengths, session linkability, victim activity intervals, and
the token->expert affinity map — no payloads and no token ids needed.
Label-free; applies to inference serving and to MoE (fine-)training
routing metadata (both modes produce the same router trace shape).

ROUTER-TRACE JOURNAL SCHEMA (llm_attacker.router_trace.v1)
==========================================================
One JSON object per line, one record per routed token:

    {"expert_ids": [2, 3], "weights": [0.62, 0.38], "queue_depth": 1,
     "overflow": false, "bytes": 271, "ts": 1712345678.123,
     "session_id": "s0"}

    expert_ids   list of int >= 0, one per selected expert (top-k, >= 1)
    weights      list of numbers, same length as expert_ids, each in
                 [0, 1], summing to 1.0 +/- 1e-3 (top-k normalized)
    queue_depth  int >= 0: tokens in flight at the busiest selected expert
    overflow     bool: a selected expert exceeded capacity and the load
                 balancer spilled / dropped work
    bytes        int >= 0: bytes moved for this routing decision
    ts           float, epoch seconds of the observed record
    session_id   str, the serving session the token belongs to

All keys are required. Schema drift (missing/extra keys, bad types,
negative depths, weights that do not normalize) is a descriptive
ValueError naming the file and line — never a silent skip. The schema is
registered as the "router-trace" SurfaceProvider (attacker.surfaces) at
module import, so future attacks consume router traces through the
provider registry: rows(mode, source=<journal path>) yields
(expert_id_list, metadata) pairs.

TOY WORLDS (--toy, deterministic, seeded, torch-free)
=====================================================
make_toy_moe builds a synthetic MoE world: VOCAB=64 token types, each with
a hidden PROPERTY bit ("sensitive": 4 of every 16) and a DOMAIN label
(t // 16, four domains); E=8 experts, top-k=2 routing.

  informative   affinity router: seeded token->expert affinities;
                sensitive tokens get a +1.5 bonus on SENS_EXPERTS={0,1}
                and domain d gets +1.0 on DOMAIN_EXPERTS[d], so properties
                and domains correlate with expert choice. The uneven load
                forms queues (per-expert in-flight load with a hard
                per-step drain) and, at high load, capacity overflows.
  null          NEGATIVE CONTROL, always run alongside: uniform random
                top-2 routing. A seeded hidden map exists (for scoring the
                extraction arm) but does NOT influence routing. The
                uniform spread keeps every expert below its service rate,
                so queue telemetry is flat by construction — the
                balanced-routing control.

The attacker never sees token ids: per-sequence/session features are
expert-id histograms, routing-weight sums, queue integrals, overflow
counts, byte counts, and timing. Calibration (per-expert sensitivity,
per-domain profiles) comes from CHOSEN-INPUT PROBES — the attacker's own
API traffic, whose properties it knows.

ARMS (each runs informative vs null over a seeds grid)
======================================================
  property-classification   per-sequence mean routed sensitivity (probes
                            give per-expert sensitivity) -> binary AUC +
                            TPR@1%FPR with Wilson 95% CI.
  length-domain-inference   length from the queue integral + overflow
                            count (linear model fit on the first half,
                            MAE + Spearman on the held-out half vs the
                            mean-predictor baseline); domain from
                            expert-mix histograms (probe-calibrated
                            profiles; one-vs-rest AUC + argmax accuracy).
  linkability               same-user sessions share expert-mix
                            fingerprints: cosine-similarity AUC of
                            same-user vs cross-user session pairs.
  contention-fingerprinting cross-tenant: victim load on shared experts
                            shifts the attacker's observed queue depths —
                            recover victim-active slots (AUC + TPR@1%FPR).
  expert-map-extraction     single-token probes read back the
                            token->top-expert map; fidelity vs the 1/E
                            random baseline.

LIMITATIONS / EXCLUDED VARIANTS
===============================
  * simulator, not a live MoE stack; the real-target path is the
    --journal router-trace analyzer (schema above).
  * when per-token traces are visible, record count / total bytes
    trivially leak sequence length; the SCORED length channel is
    queue/overflow telemetry (the aggregate-telemetry observer). Record
    counts are reported as context, never as the scored metric.
  * the only defense modeled is the null (uniform) router control;
    adaptive defenses (routing differential privacy, expert padding,
    fixed-shape batching) are out of scope.
  * a single routing layer is simulated; multi-layer MoE router
    composition is excluded.
  * probes are the attacker's own traffic; no victim-query manipulation.
  * Wilson CIs (95%) are reported on TPR proportions; AUCs are reported
    per seed over the seeds grid, without a closed-form CI.

Usage:
    python -m attacker --mode inference --attack router-side-channel --help
    python -m attacker --mode inference --attack router-side-channel \
        --toy --quick --output /tmp/rsc.json
    python -m attacker --mode inference --attack router-side-channel \
        --journal captures/router_trace.jsonl --output rsc.json
"""

import argparse
import json
import math
import os
import random
import statistics

from .. import artifacts, threat_model
from ..surfaces import SurfaceProvider, register_surface
from .common import add_common_args, journal_error

EXPERIMENT_ID = "router_side_channel"
DISPLAY = "router-side-channel"
MODES = ("training", "inference")
REQUIRES_LABELS = False
DESCRIPTION = ("passive MoE router/scheduling metadata analysis (S12): "
               "expert ids, routing weights, queue depth, and capacity "
               "overflows leak token properties, domains, lengths, session "
               "linkability, cross-tenant activity, and the token->expert "
               "map")

SCHEMA_VERSION = "dtraining.attacker.router_side_channel.v1"
JOURNAL_SCHEMA = "llm_attacker.router_trace.v1"
ARMS = ("property-classification", "length-domain-inference", "linkability",
        "contention-fingerprinting", "expert-map-extraction")
JOURNAL_ARMS = ("length-domain-inference", "linkability",
                "contention-fingerprinting")
WORLDS = ("informative", "null")
CONTROL_WORLD = "null"

VOCAB = 64
N_DOMAINS = 4
N_EXPERTS = 8
TOP_K = 2
SENS_EXPERTS = (0, 1)
DOMAIN_EXPERTS = {0: (2, 3), 1: (4, 5), 2: (6, 7), 3: (2, 7)}
ROUTE_NOISE = 0.3
BASE_BYTES = 256
SEQ_SERVICE = 0.25
SEQ_CAPACITY = 3.0
CONTENTION_SERVICE = 9.0
CONTENTION_CAPACITY = 6.0
CONTENTION_VICTIM_RATE = 16

PRIMARY_METRIC = {"property-classification": "auc",
                  "length-domain-inference": "mean_domain_auc",
                  "linkability": "auc",
                  "contention-fingerprinting": "auc",
                  "expert-map-extraction": "fidelity"}
METRIC_KEYS = {"property-classification": ("auc", "tpr_at_1pct_fpr"),
               "length-domain-inference": ("length_mae",
                                           "length_mae_baseline",
                                           "mean_domain_auc",
                                           "domain_accuracy"),
               "linkability": ("auc", "tpr_at_1pct_fpr",
                               "mean_cosine_same", "mean_cosine_cross"),
               "contention-fingerprinting": ("auc", "tpr_at_1pct_fpr"),
               "expert-map-extraction": ("fidelity", "lift_pp")}


class RouterTraceProvider(SurfaceProvider):
    """The router-trace journal as an observation surface: one record per
    routed token of {expert_ids, weights, queue_depth, overflow, bytes, ts,
    session_id} (schema llm_attacker.router_trace.v1). rows(mode,
    source=<journal path>) yields (expert_id_list, metadata) pairs with
    metadata the full normalized record. Pure python, torch-free."""

    name = "router-trace"
    schema = {"expert_ids": list, "weights": list, "queue_depth": int,
              "overflow": bool, "bytes": int, "ts": float,
              "session_id": str}
    modes = ("training", "inference")

    def validate(self, meta, path="<record>"):
        """Key-set + type validation; weights must be top-k normalized
        (sum 1 +/- 1e-3). ValueError on drift."""
        keys = set(meta)
        if keys != set(self.schema):
            raise ValueError(
                f"{path}: record keys {sorted(keys)} != surface "
                f"'router-trace' schema {sorted(self.schema)} "
                f"(schema {JOURNAL_SCHEMA})")
        eids = meta["expert_ids"]
        if (not isinstance(eids, list) or not eids
                or any(isinstance(e, bool) or not isinstance(e, int)
                       or e < 0 for e in eids)):
            raise ValueError(f"{path}: expert_ids must be a non-empty list "
                             f"of int >= 0, got {eids!r}")
        ws = meta["weights"]
        if (not isinstance(ws, list) or len(ws) != len(eids)
                or any(isinstance(w, bool)
                       or not isinstance(w, (int, float))
                       or not 0.0 <= w <= 1.0 for w in ws)):
            raise ValueError(f"{path}: weights must be a list of numbers in "
                             f"[0, 1] matching expert_ids, got {ws!r}")
        if abs(sum(ws) - 1.0) > 1e-3:
            raise ValueError(f"{path}: weights must sum to 1.0 (top-k "
                             f"normalized), got sum={sum(ws)!r}")
        qd = meta["queue_depth"]
        if isinstance(qd, bool) or not isinstance(qd, int) or qd < 0:
            raise ValueError(f"{path}: queue_depth must be an int >= 0, "
                             f"got {qd!r}")
        if not isinstance(meta["overflow"], bool):
            raise ValueError(f"{path}: overflow must be a bool, got "
                             f"{meta['overflow']!r}")
        nbytes = meta["bytes"]
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) \
                or nbytes < 0:
            raise ValueError(f"{path}: bytes must be an int >= 0, got "
                             f"{nbytes!r}")
        ts = meta["ts"]
        if isinstance(ts, bool) or not isinstance(ts, (int, float)):
            raise ValueError(f"{path}: ts must be a number, got {ts!r}")
        if not isinstance(meta["session_id"], str):
            raise ValueError(f"{path}: session_id must be a str, got "
                             f"{meta['session_id']!r}")
        return {"expert_ids": list(eids),
                "weights": [float(w) for w in ws],
                "queue_depth": int(qd), "overflow": meta["overflow"],
                "bytes": int(nbytes), "ts": float(ts),
                "session_id": meta["session_id"]}

    def rows(self, mode=None, source=None):
        self.check_mode(mode)
        if source is None:
            raise ValueError("surface 'router-trace' needs source=<router "
                             f"trace journal path ({JOURNAL_SCHEMA})>")

        def gen():
            with open(source) as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    meta = self.validate(json.loads(line),
                                         f"{source}:{lineno}")
                    yield list(meta["expert_ids"]), meta
        return gen()


register_surface(RouterTraceProvider())


def read_journal(path):
    """Parse a router-trace journal -> {session_id: [records sorted by
    ts]}. Records are validated through the RouterTraceProvider; schema
    drift and a missing file are descriptive ValueErrors naming the file
    (and line)."""
    if not os.path.isfile(path):
        raise ValueError(f"{path}: router-trace journal not found "
                         f"(schema {JOURNAL_SCHEMA})")
    provider = RouterTraceProvider()
    sessions = {}
    for _, meta in provider.rows(source=path):
        sessions.setdefault(meta["session_id"], []).append(meta)
    for records in sessions.values():
        records.sort(key=lambda r: r["ts"])
    return sessions


def rank_auc(pos, neg):
    """Mann-Whitney AUC = P(pos > neg), ties count 1/2. No distributional
    assumption on the scores; nan when a class is empty."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else (0.5 if p == q else 0.0)
    return wins / (len(pos) * len(neg))


def tpr_at_fpr(pos, neg, target=0.01):
    """Conservative TPR at FPR <= target: the max TPR over the achievable
    threshold grid whose FPR never exceeds the target (predict positive iff
    score >= threshold)."""
    out = {"target_fpr": target, "tpr": float("nan"), "fpr": float("nan"),
           "n_pos": len(pos), "n_neg": len(neg)}
    if not pos or not neg:
        return out
    best_tpr, best_fpr = 0.0, 0.0
    for thr in sorted(set(pos) | set(neg), reverse=True):
        fpr = sum(1 for q in neg if q >= thr) / len(neg)
        if fpr > target + 1e-15:
            continue
        tpr = sum(1 for p in pos if p >= thr) / len(pos)
        if tpr > best_tpr:
            best_tpr, best_fpr = tpr, fpr
    out["tpr"], out["fpr"] = best_tpr, best_fpr
    return out


def wilson(k, n, z=1.959964):
    """Wilson score interval for a binomial proportion k/n -> (lo, hi)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n
                         + z * z / (4.0 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a, b):
    """Spearman rank correlation (average ranks on ties); nan for n < 2
    or a constant input."""
    if len(a) != len(b):
        raise ValueError("spearman: inputs differ in length")
    n = len(a)
    if n < 2:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    if va <= 0.0 or vb <= 0.0:
        return float("nan")
    return cov / math.sqrt(va * vb)


def linfit(xs, ys):
    """Least-squares y ~= a*x + b -> (a, b); (0.0, mean(ys)) when xs is
    constant (the null world's flat queue telemetry)."""
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0, my
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return a, my - a * mx


def cosine(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _sanitize(obj):
    """Recursively map non-finite floats to None so artifacts stay strict
    JSON (NaN/Infinity are not valid JSON literals)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _r6(x):
    return round(x, 6) if isinstance(x, float) and math.isfinite(x) else x


def make_toy_moe(seed, world):
    """Synthetic MoE world. Returns {"world", "tokens", "affinity",
    "hidden_map", "expert_overhead"}. tokens: V=64 types with a domain
    (t // 16) and a sensitive bit (t % 16 < 4). affinity (informative
    world): base noise + 1.5 on SENS_EXPERTS for sensitive tokens + 1.0 on
    DOMAIN_EXPERTS[domain]. hidden_map (null world): a seeded top-expert
    per token used ONLY as extraction ground truth — the null router never
    consults it. expert_overhead: per-expert byte-cost factor."""
    rng = random.Random(seed)
    per_domain = VOCAB // N_DOMAINS
    tokens = [{"id": t, "domain": t // per_domain,
               "sensitive": 1 if t % per_domain < per_domain // 4 else 0}
              for t in range(VOCAB)]
    affinity = [[rng.uniform(0.0, 0.5) for _ in range(N_EXPERTS)]
                for _ in range(VOCAB)]
    for tok in tokens:
        if tok["sensitive"]:
            for e in SENS_EXPERTS:
                affinity[tok["id"]][e] += 1.5
        for e in DOMAIN_EXPERTS[tok["domain"]]:
            affinity[tok["id"]][e] += 1.0
    return {"world": world, "tokens": tokens, "affinity": affinity,
            "hidden_map": [rng.randrange(N_EXPERTS) for _ in range(VOCAB)],
            "expert_overhead": [rng.uniform(-0.3, 0.5)
                                for _ in range(N_EXPERTS)]}


def route(moe, token_id, r):
    """Route one token top-2. Informative: affinity + uniform noise; null:
    uniform scores (the hidden map is never consulted). Returns
    (expert_ids, softmax-normalized top-k weights)."""
    if moe["world"] == "informative":
        aff = moe["affinity"][token_id]
        scores = [aff[e] + r.uniform(0.0, ROUTE_NOISE)
                  for e in range(N_EXPERTS)]
    else:
        scores = [r.uniform(0.0, 1.0) for _ in range(N_EXPERTS)]
    top = sorted(range(N_EXPERTS), key=lambda e: (-scores[e], e))[:TOP_K]
    m = max(scores[e] for e in top)
    ex = [math.exp(scores[e] - m) for e in top]
    s = sum(ex)
    return top, [x / s for x in ex]


def truth_top_expert(moe, token_id):
    """Ground-truth top expert: affinity argmax (informative) or the
    seeded hidden map (null — unroutable, so extraction sits at 1/E)."""
    if moe["world"] == "informative":
        aff = moe["affinity"][token_id]
        return max(range(N_EXPERTS), key=lambda e: (aff[e], -e))
    return moe["hidden_map"][token_id]


class ExpertQueues:
    """Per-expert in-flight load with a hard per-step drain. The null
    router's uniform spread keeps every expert below the service rate, so
    queue telemetry is flat there by construction (the balanced-routing
    control); the informative router's concentrated load forms queues.

    load() adds one routing decision; step() advances time one unit (one
    token-step in the sequence arms, one slot in the contention arm) and
    drains; observe() is the sequence-arm convenience (read depth, load,
    step)."""

    def __init__(self, service, capacity):
        self.service, self.capacity = service, capacity
        self.in_flight = [0.0] * N_EXPERTS

    def load(self, expert_ids, weights):
        for e, w in zip(expert_ids, weights):
            self.in_flight[e] += w

    def step(self):
        """Advance one time step: hard-drain `service` load per expert;
        returns the post-drain backlog (the per-interval telemetry the
        contention arm reads)."""
        self.in_flight = [max(0.0, x - self.service)
                          for x in self.in_flight]
        return list(self.in_flight)

    def observe(self, expert_ids, weights):
        """Record one routing decision -> (queue_depth, overflow): the
        depth at the busiest selected expert BEFORE this token is added,
        and whether any selected expert is above capacity."""
        depth = int(max(self.in_flight[e] for e in expert_ids))
        overflow = any(self.in_flight[e] > self.capacity
                       for e in expert_ids)
        self.load(expert_ids, weights)
        self.step()
        return depth, overflow


def _token_bytes(moe, expert_ids, weights, r):
    overhead = sum(w * moe["expert_overhead"][e]
                   for e, w in zip(expert_ids, weights))
    return max(1, int(BASE_BYTES * (1.0 + overhead)
                      + r.uniform(-16.0, 16.0)))


def _draw_domain(r, mix):
    """Draw a domain from a probability mixture over N_DOMAINS."""
    x = r.random()
    acc = 0.0
    for d, p in enumerate(mix):
        acc += p
        if x <= acc:
            return d
    return N_DOMAINS - 1


def gen_sequence(r, moe, length, domain, sensitive_rate, session_id, t0,
                 queues):
    """One synthetic request: `length` tokens, each sensitive with
    probability `sensitive_rate`, routed with queueing. `domain` is either
    an int (fixed domain) or a probability mixture over domains drawn per
    token (the linkability arm's per-user topical profile). Returns
    (records, t_end). The attacker never sees token ids — only the trace
    records (schema llm_attacker.router_trace.v1)."""
    per_domain = VOCAB // N_DOMAINS
    mix = domain if isinstance(domain, (list, tuple)) else None
    pools = {(d, s): [t for t in range(d * per_domain, (d + 1) * per_domain)
                      if moe["tokens"][t]["sensitive"] == s]
             for d in range(N_DOMAINS) for s in (0, 1)}
    records = []
    t = t0
    for _ in range(length):
        dom = _draw_domain(r, mix) if mix is not None else domain
        pool = pools[(dom, 1 if r.random() < sensitive_rate else 0)]
        tok = pool[r.randrange(len(pool))]
        eids, ws = route(moe, tok, r)
        depth, ovf = queues.observe(eids, ws)
        records.append({"expert_ids": eids, "weights": ws,
                        "queue_depth": depth, "overflow": ovf,
                        "bytes": _token_bytes(moe, eids, ws, r),
                        "ts": t, "session_id": session_id})
        t += r.uniform(0.005, 0.015)
    return records, t + r.uniform(0.2, 0.5)


def calibrate(r, moe, repeats):
    """Chosen-input probes: the attacker routes every token type `repeats`
    times (its own traffic, known properties) and accumulates routing
    weight per expert. Returns {"sensitivity": [E], "profiles": {d: [E]}}
    — per-expert sensitive-weight share and per-domain (non-sensitive)
    weight profiles, both L1-normalized where defined."""
    w_sens = [0.0] * N_EXPERTS
    w_tot = [0.0] * N_EXPERTS
    w_dom = {d: [0.0] * N_EXPERTS for d in range(N_DOMAINS)}
    for tok in moe["tokens"]:
        for _ in range(repeats):
            eids, ws = route(moe, tok["id"], r)
            for e, w in zip(eids, ws):
                w_tot[e] += w
                if tok["sensitive"]:
                    w_sens[e] += w
                else:
                    w_dom[tok["domain"]][e] += w
    sens = [(w_sens[e] / w_tot[e] if w_tot[e] > 1e-12 else 0.5)
            for e in range(N_EXPERTS)]
    profiles = {}
    for d, wvec in w_dom.items():
        s = sum(wvec)
        profiles[d] = ([x / s for x in wvec] if s > 1e-12
                       else [1.0 / N_EXPERTS] * N_EXPERTS)
    return {"sensitivity": sens, "profiles": profiles}


def seq_hist(records):
    """L1-normalized routing-weight histogram over experts."""
    hist = [0.0] * N_EXPERTS
    for rec in records:
        for e, w in zip(rec["expert_ids"], rec["weights"]):
            hist[e] += w
    s = sum(hist)
    return [h / s for h in hist] if s > 1e-12 else hist


def seq_property_score(records, sens):
    """Mean routed sensitivity: the average over tokens of the
    weight-sum of per-expert sensitivities of the selected experts."""
    return statistics.fmean(
        sum(w * sens[e] for e, w in zip(rec["expert_ids"], rec["weights"]))
        for rec in records)


def _cell_rng(args, seed, world, arm):
    return random.Random((args.seed + 1) * 1000033 + seed * 10007
                         + ARMS.index(arm) * 101 + WORLDS.index(world))


def _tpr_block(pos, neg):
    t = tpr_at_fpr(pos, neg, 0.01)
    lo, hi = wilson(int(round(t["tpr"] * t["n_pos"])), t["n_pos"]) \
        if t["tpr"] == t["tpr"] else (float("nan"), float("nan"))
    return {"tpr_at_1pct_fpr": _r6(t["tpr"]), "tpr_fpr": _r6(t["fpr"]),
            "tpr_wilson_lo": _r6(lo), "tpr_wilson_hi": _r6(hi)}


def cell_property_classification(args, seed, world):
    arm = "property-classification"
    r = _cell_rng(args, seed, world, arm)
    moe = make_toy_moe(r.getrandbits(30), world)
    cal = calibrate(r, moe, args.n_probe_repeats)
    pos, neg = [], []
    n = args.n_sequences
    for i in range(n):
        prop = i % 2
        domain = r.randrange(N_DOMAINS)
        records, _ = gen_sequence(r, moe, r.randint(8, 24), domain,
                                  0.8 if prop else 0.05,
                                  f"{world}-pc-{seed}-{i}", 1000.0,
                                  ExpertQueues(SEQ_SERVICE, SEQ_CAPACITY))
        score = seq_property_score(records, cal["sensitivity"])
        (pos if prop else neg).append(score)
    rec = {"experiment": EXPERIMENT_ID, "arm": arm, "world": world,
           "control": world == CONTROL_WORLD, "seed": seed,
           "n_sequences": n, "n_pos": len(pos), "n_neg": len(neg),
           "auc": _r6(rank_auc(pos, neg)),
           **_tpr_block(pos, neg),
           "mean_score_pos": _r6(statistics.fmean(pos)),
           "mean_score_neg": _r6(statistics.fmean(neg)),
           "expert_sensitivity": [_r6(x) for x in cal["sensitivity"]]}
    return _sanitize(rec)


def cell_length_domain(args, seed, world):
    arm = "length-domain-inference"
    r = _cell_rng(args, seed, world, arm)
    moe = make_toy_moe(r.getrandbits(30), world)
    cal = calibrate(r, moe, args.n_probe_repeats)
    n = args.n_sequences
    feats, lengths, domains = [], [], []
    for i in range(n):
        domain = i % N_DOMAINS
        records, _ = gen_sequence(r, moe, r.randint(6, 30), domain, 0.05,
                                  f"{world}-ld-{seed}-{i}", 1000.0,
                                  ExpertQueues(SEQ_SERVICE, SEQ_CAPACITY))
        q_int = sum(rec["queue_depth"] for rec in records)
        n_ovf = sum(1 for rec in records if rec["overflow"])
        feats.append({"x": q_int + 4.0 * n_ovf, "hist": seq_hist(records)})
        lengths.append(len(records))
        domains.append(domain)
    half = n // 2
    xs, ys = [f["x"] for f in feats], lengths
    a, b = linfit(xs[:half], ys[:half])
    est = [a * x + b for x in xs[half:]]
    true = ys[half:]
    mae = statistics.fmean(abs(t - e) for t, e in zip(true, est))
    base = statistics.fmean(ys[:half])
    mae_base = statistics.fmean(abs(t - base) for t in true)
    per_domain_auc = {}
    correct = 0
    for d in range(N_DOMAINS):
        pos = [sum(h * p for h, p in zip(f["hist"], cal["profiles"][d]))
               for f, dom in zip(feats, domains) if dom == d]
        neg = [sum(h * p for h, p in zip(f["hist"], cal["profiles"][d]))
               for f, dom in zip(feats, domains) if dom != d]
        per_domain_auc[str(d)] = _r6(rank_auc(pos, neg))
    for f, dom in zip(feats, domains):
        pred = max(range(N_DOMAINS),
                   key=lambda d: sum(h * p for h, p in
                                     zip(f["hist"], cal["profiles"][d])))
        correct += pred == dom
    rec = {"experiment": EXPERIMENT_ID, "arm": arm, "world": world,
           "control": world == CONTROL_WORLD, "seed": seed,
           "n_sequences": n,
           "length_feature": "queue_integral + 4 * n_overflows",
           "length_mae": _r6(mae),
           "length_mae_baseline": _r6(mae_base),
           "length_mae_lift": _r6(mae_base - mae),
           "length_spearman": _r6(spearman(xs[half:], true)),
           "mean_domain_auc": _r6(statistics.fmean(
               per_domain_auc.values())),
           "domain_auc": per_domain_auc,
           "domain_accuracy": _r6(correct / n)}
    return _sanitize(rec)


def cell_linkability(args, seed, world):
    """Same-user sessions share an expert-mix fingerprint. Each user has a
    seeded topical profile (a sparse random mixture over domains) plus a
    seeded sensitive-token rate; sessions are independent draws from the
    profile. Fingerprint = the session's routing-weight histogram. AUC of
    cosine similarity: same-user pairs vs cross-user pairs."""
    arm = "linkability"
    r = _cell_rng(args, seed, world, arm)
    moe = make_toy_moe(r.getrandbits(30), world)
    fps, users = [], []
    for u in range(args.users):
        mix_raw = [r.uniform(0.1, 1.0) ** 2 for _ in range(N_DOMAINS)]
        tot = sum(mix_raw)
        mix = [x / tot for x in mix_raw]
        rate = r.uniform(0.0, 0.4)
        for s in range(args.sessions_per_user):
            sid = f"{world}-lk-{seed}-u{u}-s{s}"
            t = 1000.0
            session_records = []
            for _ in range(4):
                records, t = gen_sequence(r, moe, r.randint(8, 20), mix,
                                          rate, sid, t,
                                          ExpertQueues(SEQ_SERVICE,
                                                       SEQ_CAPACITY))
                session_records.extend(records)
            fps.append(seq_hist(session_records))
            users.append(u)
    pos, neg = [], []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            c = cosine(fps[i], fps[j])
            (pos if users[i] == users[j] else neg).append(c)
    rec = {"experiment": EXPERIMENT_ID, "arm": arm, "world": world,
           "control": world == CONTROL_WORLD, "seed": seed,
           "n_users": args.users, "n_sessions": len(fps),
           "n_pairs": len(pos) + len(neg),
           "n_pos": len(pos), "n_neg": len(neg),
           "auc": _r6(rank_auc(pos, neg)),
           **_tpr_block(pos, neg),
           "mean_cosine_same": _r6(statistics.fmean(pos)),
           "mean_cosine_cross": _r6(statistics.fmean(neg))}
    return _sanitize(rec)


def cell_contention(args, seed, world):
    """Cross-tenant contention: a victim tenant bursts load in intervals;
    the attacker sends a constant probe stream (its own domain-0 tokens)
    and reads the post-slot queue backlog on its routed experts. Victim
    bursts on shared experts saturate the per-slot service rate and leave
    a growing backlog only when the router concentrates load
    (informative); the null router's uniform spread keeps every expert
    below service, so the attacker's telemetry stays flat."""
    arm = "contention-fingerprinting"
    r = _cell_rng(args, seed, world, arm)
    moe = make_toy_moe(r.getrandbits(30), world)
    T = args.slots
    active = [False] * T
    t = 2
    while t < T - 2:
        if r.random() < 0.5:
            blen = r.randint(3, 6)
            for k in range(t, min(T, t + blen)):
                active[k] = True
            t += blen + r.randint(3, 6)
        else:
            t += 1
    pool = [tok["id"] for tok in moe["tokens"]
            if tok["domain"] == 0 and not tok["sensitive"]]
    queues = ExpertQueues(CONTENTION_SERVICE, CONTENTION_CAPACITY)
    scores, n_records = [], 0
    for slot in range(T):
        if active[slot]:
            for _ in range(CONTENTION_VICTIM_RATE):
                eids, ws = route(moe, pool[r.randrange(len(pool))], r)
                queues.load(eids, ws)
                n_records += 1
        atk_experts = set()
        for _ in range(4):
            eids, ws = route(moe, pool[r.randrange(len(pool))], r)
            queues.load(eids, ws)
            atk_experts.update(eids)
            n_records += 1
        backlog = queues.step()
        scores.append(max(backlog[e] for e in atk_experts))
    pos = [s for s, a in zip(scores, active) if a]
    neg = [s for s, a in zip(scores, active) if not a]
    rec = {"experiment": EXPERIMENT_ID, "arm": arm, "world": world,
           "control": world == CONTROL_WORLD, "seed": seed,
           "n_slots": T, "n_active_slots": sum(active),
           "n_records": n_records,
           "auc": _r6(rank_auc(pos, neg)),
           **_tpr_block(pos, neg),
           "mean_depth_active": _r6(statistics.fmean(pos)) if pos else None,
           "mean_depth_idle": _r6(statistics.fmean(neg)) if neg else None,
           "slot_scores": [round(s, 3) for s in scores]}
    return _sanitize(rec)


def cell_expert_map(args, seed, world):
    arm = "expert-map-extraction"
    r = _cell_rng(args, seed, world, arm)
    moe = make_toy_moe(r.getrandbits(30), world)
    recovered = []
    for tok in range(VOCAB):
        w_obs = [0.0] * N_EXPERTS
        for _ in range(args.n_probe_repeats):
            eids, ws = route(moe, tok, r)
            for e, w in zip(eids, ws):
                w_obs[e] += w
        recovered.append(max(range(N_EXPERTS),
                             key=lambda e: (w_obs[e], -e)))
    hits = sum(1 for t in range(VOCAB)
               if recovered[t] == truth_top_expert(moe, t))
    fidelity = hits / VOCAB
    rec = {"experiment": EXPERIMENT_ID, "arm": arm, "world": world,
           "control": world == CONTROL_WORLD, "seed": seed,
           "n_tokens": VOCAB, "n_probes": VOCAB * args.n_probe_repeats,
           "fidelity": _r6(fidelity),
           "random_baseline": _r6(1.0 / N_EXPERTS),
           "lift_pp": _r6(100.0 * (fidelity - 1.0 / N_EXPERTS))}
    return _sanitize(rec)


ARM_FUNCS = {"property-classification": cell_property_classification,
             "length-domain-inference": cell_length_domain,
             "linkability": cell_linkability,
             "contention-fingerprinting": cell_contention,
             "expert-map-extraction": cell_expert_map}


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS),
                    help="attack arm(s) to run (default: all five)")
    ap.add_argument("--journal", default=None,
                    help=f"router-trace journal to analyze "
                         f"({JOURNAL_SCHEMA}); measurable arms: "
                         f"{', '.join(JOURNAL_ARMS)}")
    ap.add_argument("--n-sequences", type=int, default=48,
                    help="toy: sequences per classification cell")
    ap.add_argument("--n-probe-repeats", type=int, default=3,
                    help="toy: chosen-input probe repeats per token type")
    ap.add_argument("--users", type=int, default=8,
                    help="toy: linkability users")
    ap.add_argument("--sessions-per-user", type=int, default=3,
                    help="toy: linkability sessions per user")
    ap.add_argument("--slots", type=int, default=48,
                    help="toy: contention time slots")
    ap.add_argument("--slot-ms", type=float, default=1000.0,
                    help="journal: contention slot width in ms")
    ap.add_argument("--evidence", action="store_true",
                    help="wrap the run in an EvidenceLogger (requires "
                         "--output): the artifact write is recorded as a "
                         "chained event and the evidence manifest is "
                         "finalized alongside")
    return ap


def _threat_model_doc():
    return threat_model.validate(threat_model.skeleton(
        "passive observer of MoE / dynamic-routing metadata (surface S12) "
        "— expert ids, top-k routing weights, queue depths, and capacity "
        "overflows — inferring token properties, sequence domains and "
        "lengths, session linkability, cross-tenant activity, and the "
        "token->expert affinity map without payloads or token ids",
        scope={"use_case": "split-inference", "deployment": "multi-tenant",
               "in_scope": ["MoE router and scheduling metadata: selected "
                            "expert ids, top-k routing weights, per-expert "
                            "queue depth, capacity overflows, load-"
                            "balancer decisions (S12)",
                            "router-trace journals "
                            f"({JOURNAL_SCHEMA})",
                            "chosen-input probes through the attacker's "
                            "own API traffic"],
               "out_of_scope": ["payload content and outputs (S4)",
                                "seam activations / gradients (S1/S2)",
                                "adaptive routing defenses (routing DP, "
                                "expert padding, fixed-shape batching)",
                                "multi-layer MoE router composition"]},
        actor={"role": "cloud",
               "vantage": ["server", "co-tenant"],
               "intent": "passive",
               "authorization": "authorized-red-team",
               "privileges": ["honest-but-curious control-plane trace "
                              "access", "one low-privilege tenant API key"],
               "knowledge": ["expert count and top-k (public architecture)",
                             "its own probe tokens' properties"],
               "capabilities": ["observe router/scheduling metadata "
                                "(expert ids, weights, queue depth, "
                                "overflows, bytes, timing, session ids)",
                                "send chosen single-token probes as its "
                                "own API traffic"],
               "resources": ["one tenant API key", "router-trace access"],
               "persistence": "cross-session",
               "collusion": []},
        target={"assets": ["token property bits (e.g. sensitive class)",
                           "sequence lengths and domains",
                           "user session linkability",
                           "cross-tenant victim activity intervals",
                           "the token->expert affinity map"],
                "components": ["MoE router / gating layer",
                               "expert workers", "load balancer",
                               "scheduler queues"],
                "data_stores": ["router-trace logs", "scheduling metrics"],
                "data_flows": ["token -> router -> expert queue -> "
                               "expert worker"],
                "trust_boundaries": ["control plane (routing metadata) vs "
                                     "tenant data plane (payloads)",
                                     "tenant boundary on shared experts"],
                "entry_points": ["router-trace / scheduling telemetry"],
                "exit_points": ["observed expert ids, weights, queue "
                                "depths, overflows"]},
        scenario={"preconditions": ["MoE or dynamically-routed deployment",
                                    "attacker observes router metadata "
                                    "(S12 vantage)",
                                    "shared experts across tenants for "
                                    "the contention arm"],
                  "lifecycle_stage": ["inference", "operations"],
                  "security_properties": ["confidentiality", "privacy"],
                  "attack_steps": ["calibrate per-expert sensitivity and "
                                   "domain profiles with chosen-input "
                                   "probes",
                                   "collect per-sequence expert-mix / "
                                   "queue / overflow features",
                                   "classify properties and domains, "
                                   "fingerprint sessions, recover "
                                   "cross-tenant activity intervals",
                                   "extract the token->expert map from "
                                   "single-token probes"],
                  "technical_impact": ["property inference from routing "
                                       "metadata",
                                       "session linkability across "
                                       "requests",
                                       "cross-tenant activity "
                                       "reconstruction",
                                       "router-map extraction enabling "
                                       "targeted follow-on attacks"],
                  "business_impact": ["privacy breach via infrastructure "
                                      "metadata on shared MoE serving"]},
        controls={"assumed": ["routing metadata is visible to the "
                              "control-plane observer"],
                  "tested": ["null (uniform-random) router as the "
                             "balanced-routing control, run alongside "
                             "every cell"],
                  "status": "open",
                  "residual_risk": "adaptive defenses (routing DP, "
                                   "expert padding, fixed-shape batching) "
                                   "are not modeled"},
        mappings={"owasp_llm_2026": ["LLM02", "LLM09"],
                  "cwe": ["CWE-200"]},
        evaluation={"success_metrics": ["property AUC and TPR@1%FPR with "
                                        "Wilson 95% CI",
                                        "recovery above the null-router "
                                        "control (pp)",
                                        "linkability AUC across sessions",
                                        "contention slot AUC",
                                        "expert-map extraction fidelity "
                                        "vs the 1/E baseline"],
                    "baseline": "null (uniform-random) router world, run "
                                "alongside every cell (AUC ~ 0.5, "
                                "fidelity ~ 1/E, flat queue telemetry)",
                    "budget": {"probes": "one chosen-input probe per "
                                         "token type x --n-probe-repeats",
                               "sequences_per_cell": "--n-sequences",
                               "seeds": "--seeds"},
                    "limitations": ["simulated MoE stack; live integration "
                                    "via the router-trace journal "
                                    "analyzer",
                                    "per-token record count leaks length "
                                    "trivially and is reported as "
                                    "context, not scored",
                                    "single routing layer; no adaptive "
                                    "defenses"],
                    "coverage_status": "covered"}))


def _make_artifact(args, source, extra_config=None, evidence=None):
    config = {"attack": EXPERIMENT_ID, "mode": getattr(args, "mode", None),
              "source": source, "arms": args.arms, "seed": args.seed,
              "journal_schema": JOURNAL_SCHEMA,
              "toy_world": {"vocab": VOCAB, "experts": N_EXPERTS,
                            "top_k": TOP_K, "domains": N_DOMAINS},
              "control_world": CONTROL_WORLD,
              "n_probe_repeats": args.n_probe_repeats}
    if extra_config:
        config.update(extra_config)
    return artifacts.make_artifact(
        SCHEMA_VERSION, config, _threat_model_doc(),
        interpretation="the informative router must show the attack "
                       "working (property/linkability/contention AUC well "
                       "above 0.5, length MAE well below the mean-"
                       "predictor baseline, extraction fidelity well above "
                       "1/E); the null (uniform) router control must sit "
                       "at baseline (AUC ~ 0.5, fidelity ~ 1/E, flat "
                       "queue telemetry). Routing metadata alone — no "
                       "payloads, no token ids — suffices.",
        evidence=evidence)


def _progress(rec):
    parts = []
    for key in ("auc", "mean_domain_auc", "fidelity", "length_mae",
                "tpr_at_1pct_fpr", "lift_pp", "n_sequences", "n_pairs",
                "n_active_slots"):
        if key in rec and rec[key] is not None:
            parts.append(f"{key}={rec[key]}")
    return (f"[{DISPLAY}] {rec.get('world', 'journal')}/{rec['arm']} "
            f"seed={rec.get('seed', '-')}: " + " ".join(parts))


def run_toy(args, evidence=None):
    if args.quick:
        args.seeds = args.seeds[:2]
        args.n_sequences = min(args.n_sequences, 24)
        args.slots = min(args.slots, 24)
    worlds = list(WORLDS)
    out = _make_artifact(args, "toy",
                         {"toy": True, "worlds": worlds,
                          "seeds": args.seeds,
                          "n_sequences": args.n_sequences,
                          "slots": args.slots}, evidence=evidence)
    per = {}
    for seed in args.seeds:
        for world in worlds:
            for arm in args.arms:
                try:
                    rec = ARM_FUNCS[arm](args, seed, world)
                except Exception as e:
                    journal_error(args.output, EXPERIMENT_ID,
                                  {"arm": arm, "world": world,
                                   "seed": seed}, e)
                    continue
                artifacts.append_jsonl(args.output, rec)
                out["results"].append(rec)
                per.setdefault((arm, world), []).append(rec)
                print(_progress(rec))
    means = {}
    for (arm, world), recs in sorted(per.items()):
        row = {"experiment": EXPERIMENT_ID, "arm": arm, "world": world,
               "control": world == CONTROL_WORLD, "n_seeds": len(recs)}
        for key in METRIC_KEYS[arm]:
            vals = [r[key] for r in recs
                    if isinstance(r.get(key), (int, float))]
            if vals:
                row[key + "_mean"] = statistics.fmean(vals)
                means[(arm, world, key)] = row[key + "_mean"]
        out["summary"].append(row)
    for arm in args.arms:
        key = PRIMARY_METRIC[arm]
        if (arm, CONTROL_WORLD, key) not in means:
            continue
        row = {"experiment": EXPERIMENT_ID, "arm": arm,
               "check": "negative-control", "metric": key,
               "null_control": means[(arm, CONTROL_WORLD, key)]}
        if (arm, "informative", key) in means:
            row["informative"] = means[(arm, "informative", key)]
            row["attack_signal"] = row["informative"] - row["null_control"]
        out["summary"].append(row)
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def _segment_sequences(records):
    """Split one session's ts-sorted records into sequences: a boundary
    where the inter-record gap exceeds 3x the median gap."""
    if len(records) < 2:
        return [records] if records else []
    gaps = [records[i + 1]["ts"] - records[i]["ts"]
            for i in range(len(records) - 1)]
    thr = 3.0 * statistics.median(gaps)
    seqs, start = [], 0
    for i, g in enumerate(gaps):
        if g > thr:
            seqs.append(records[start:i + 1])
            start = i + 1
    seqs.append(records[start:])
    return seqs


def _journal_length_domain(sessions):
    seq_lens, q_integrals, n_ovf, top_shares = [], [], 0, []
    for records in sessions.values():
        for seq in _segment_sequences(records):
            seq_lens.append(len(seq))
            q_integrals.append(sum(r["queue_depth"] for r in seq))
            n_ovf += sum(1 for r in seq if r["overflow"])
            top_shares.append(max(seq_hist(seq)))
    return {"experiment": EXPERIMENT_ID, "arm": "length-domain-inference",
            "source": "journal", "n_sessions": len(sessions),
            "n_sequences": len(seq_lens),
            "records_per_sequence":
                {"mean": _r6(statistics.fmean(seq_lens)),
                 "min": min(seq_lens), "max": max(seq_lens)},
            "queue_integral":
                {"mean": _r6(statistics.fmean(q_integrals)),
                 "max": max(q_integrals)},
            "n_overflows": n_ovf,
            "mean_top_expert_share": _r6(statistics.fmean(top_shares)),
            "note": "unlabeled journal: descriptive features, no ground "
                    "truth; per-token record counts leak length trivially "
                    "and are context, the scored channel is queue/"
                    "overflow telemetry"}


def _journal_linkability(sessions):
    fps = {sid: seq_hist(records) for sid, records in sessions.items()}
    sids = sorted(fps)
    cosines = [cosine(fps[a], fps[b])
               for i, a in enumerate(sids) for b in sids[i + 1:]]
    return {"experiment": EXPERIMENT_ID, "arm": "linkability",
            "source": "journal", "n_sessions": len(sessions),
            "n_pairs": len(cosines),
            "cosine_mean": _r6(statistics.fmean(cosines))
            if cosines else None,
            "cosine_max": _r6(max(cosines)) if cosines else None,
            "n_candidate_linked_pairs":
                sum(1 for c in cosines if c >= 0.99),
            "note": "unlabeled journal: pairwise session-fingerprint "
                    "similarity; pairs >= 0.99 are linkability candidates"}


def _journal_contention(sessions, slot_ms):
    all_records = [r for records in sessions.values() for r in records]
    t0 = min(r["ts"] for r in all_records)
    width = slot_ms / 1000.0
    slots = {}
    for r in all_records:
        i = int((r["ts"] - t0) / width)
        d, o = slots.get(i, (0.0, 0))
        slots[i] = (d + r["queue_depth"], o + (1 if r["overflow"] else 0))
    n_slots = max(slots) + 1 if slots else 0
    scores = [slots.get(i, (0.0, 0))[0] + 2.0 * slots.get(i, (0, 0))[1]
              for i in range(n_slots)]
    med = statistics.median(scores) if scores else 0.0
    mad = statistics.median([abs(s - med) for s in scores]) \
        if scores else 0.0
    thr = med + 3.0 * mad
    active = [i for i, s in enumerate(scores) if s > thr]
    intervals = [[s, e] for s, e in
                 zip([a for a in active if a - 1 not in active],
                     [a for a in active if a + 1 not in active])]
    return {"experiment": EXPERIMENT_ID, "arm": "contention-fingerprinting",
            "source": "journal", "n_slots": n_slots,
            "slot_ms": slot_ms,
            "threshold": _r6(thr), "n_active_slots": len(active),
            "active_intervals": intervals,
            "note": "unlabeled journal: high-activity slots = queue/"
                    "overflow score above median + 3*MAD"}


def run_journal(args, evidence=None):
    out = _make_artifact(args, "journal", {"journal": args.journal,
                                           "slot_ms": args.slot_ms},
                         evidence=evidence)
    try:
        sessions = read_journal(args.journal)
    except ValueError as e:
        raise SystemExit(f"[{DISPLAY}] {e}") from None
    funcs = {"length-domain-inference":
             lambda: _journal_length_domain(sessions),
             "linkability":
             lambda: _journal_linkability(sessions),
             "contention-fingerprinting":
             lambda: _journal_contention(sessions, args.slot_ms)}
    for arm in args.arms:
        if arm not in JOURNAL_ARMS:
            print(f"[{DISPLAY}] {arm}: not measurable from a "
                  f"{JOURNAL_SCHEMA} journal (needs chosen-input probes / "
                  f"ground truth); skipped")
            continue
        try:
            rec = funcs[arm]()
        except Exception as e:
            journal_error(args.output, EXPERIMENT_ID, {"arm": arm}, e)
            continue
        artifacts.append_jsonl(args.output, rec)
        out["results"].append(rec)
        print(_progress(rec))
    for rec in out["results"]:
        out["summary"].append(
            {"experiment": EXPERIMENT_ID, "arm": rec["arm"],
             "source": "journal",
             "n_sessions": rec.get("n_sessions"),
             "n_records_total": sum(len(v) for v in sessions.values())})
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def _run(args, evidence=None):
    if args.journal:
        return run_journal(args, evidence=evidence)
    if args.toy:
        return run_toy(args, evidence=evidence)
    raise SystemExit(f"[{DISPLAY}] pick an input path: --toy (synthetic "
                     f"MoE worlds with ground truth and the null-router "
                     f"negative control) or --journal FILE (real router "
                     f"traces, schema {JOURNAL_SCHEMA}).")


def run(args):
    if getattr(args, "evidence", False):
        if not args.output:
            raise SystemExit(f"[{DISPLAY}] --evidence requires --output "
                             "(the evidence manifest is written next to "
                             "the artifact)")
        from ..evidence import EvidenceLogger
        with EvidenceLogger(args.output,
                            node_role=f"attack/{EXPERIMENT_ID}") as ev:
            return _run(args, evidence=ev)
    return _run(args, evidence=None)


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("router-side-channel (pure python):")

    from ..surfaces import get_surface, list_surfaces
    prov = get_surface("router-trace")
    check("router-trace surface registered at import",
          "router-trace" in list_surfaces()
          and isinstance(prov, RouterTraceProvider))
    good = {"expert_ids": [2, 3], "weights": [0.6, 0.4], "queue_depth": 1,
            "overflow": False, "bytes": 256, "ts": 1.0, "session_id": "s"}
    check("a conforming record validates", prov.validate(dict(good))
          ["queue_depth"] == 1)
    for bad, why in [({**good, "x": 1}, "extra key"),
                     ({k: v for k, v in good.items() if k != "ts"},
                      "missing key"),
                     ({**good, "weights": [0.6, 0.6]},
                      "weights not normalized"),
                     ({**good, "queue_depth": -1}, "negative depth"),
                     ({**good, "queue_depth": True}, "bool depth"),
                     ({**good, "expert_ids": [-1, 2]}, "bad expert id")]:
        try:
            prov.validate(bad)
            check(f"schema drift rejected ({why})", False)
        except ValueError:
            check(f"schema drift rejected ({why})", True)

    check("rank_auc anchors (separated / reversed / tied)",
          rank_auc([3.0, 4.0], [1.0, 2.0]) == 1.0
          and rank_auc([1.0, 2.0], [3.0, 4.0]) == 0.0
          and rank_auc([1.0, 1.0], [1.0, 1.0]) == 0.5)
    check("tpr_at_fpr anchors (separated / indistinguishable)",
          tpr_at_fpr([3.0, 4.0], [1.0, 2.0], 0.01)["tpr"] == 1.0
          and tpr_at_fpr([1.0, 1.0], [1.0, 1.0], 0.01)["tpr"] == 0.0)
    lo, hi = wilson(0, 10)
    check("wilson anchors (0/10 -> lo 0, hi < 0.5; 5/10 inside)",
          lo == 0.0 and hi < 0.5
          and wilson(5, 10)[0] < 0.5 < wilson(5, 10)[1])
    check("spearman / linfit / cosine anchors",
          spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
          and linfit([1, 2, 3], [2, 4, 6]) == (2.0, 0.0)
          and cosine([1, 0], [0, 1]) == 0.0
          and abs(cosine([1, 1], [1, 1]) - 1.0) < 1e-12)

    moe = make_toy_moe(7, "informative")
    check("toy world: 64 tokens, 16 sensitive, 4 domains",
          len(moe["tokens"]) == VOCAB
          and sum(t["sensitive"] for t in moe["tokens"]) == 16
          and len({t["domain"] for t in moe["tokens"]}) == N_DOMAINS)
    r = random.Random(3)
    hits = sum(1 for _ in range(40)
               if any(e in SENS_EXPERTS
                      for e in route(moe, 0, r)[0]))
    check("informative router: sensitive tokens route to SENS_EXPERTS "
          ">= 90% of the time", hits >= 36)
    moe_null = make_toy_moe(7, "null")
    r = random.Random(3)
    spread = {e for _ in range(40) for e in route(moe_null, 0, r)[0]}
    check("null router: no affinity (a token reaches many experts)",
          len(spread) >= 6)

    args = build_parser().parse_args([])
    args.n_sequences, args.slots = 24, 24
    prop_inf = cell_property_classification(args, 0, "informative")
    prop_nul = cell_property_classification(args, 0, "null")
    check("property-classification: informative AUC >= 0.9",
          prop_inf["auc"] >= 0.9)
    check("property-classification: null control at baseline "
          "(|AUC - 0.5| <= 0.2)", abs(prop_nul["auc"] - 0.5) <= 0.2)
    ld_inf = cell_length_domain(args, 0, "informative")
    ld_nul = cell_length_domain(args, 0, "null")
    check("length-domain: informative length MAE well below baseline",
          ld_inf["length_mae"] < 0.6 * ld_inf["length_mae_baseline"]
          and ld_inf["mean_domain_auc"] >= 0.8)
    check("length-domain: null length estimate IS the baseline",
          ld_nul["length_mae"] == ld_nul["length_mae_baseline"]
          and abs(ld_nul["mean_domain_auc"] - 0.5) <= 0.25)
    lk_inf = cell_linkability(args, 0, "informative")
    lk_nul = cell_linkability(args, 0, "null")
    check("linkability: informative AUC >= 0.9, null ~ 0.5",
          lk_inf["auc"] >= 0.9 and abs(lk_nul["auc"] - 0.5) <= 0.2)
    ct_inf = cell_contention(args, 0, "informative")
    ct_nul = cell_contention(args, 0, "null")
    check("contention: informative AUC >= 0.9",
          ct_inf["auc"] >= 0.9 and ct_inf["n_active_slots"] > 0)
    check("contention: null telemetry flat (AUC == 0.5)",
          ct_nul["auc"] == 0.5)
    em_inf = cell_expert_map(args, 0, "informative")
    em_nul = cell_expert_map(args, 0, "null")
    check("expert-map: informative fidelity >= 0.85",
          em_inf["fidelity"] >= 0.85)
    check("expert-map: null fidelity near the 1/E baseline",
          em_nul["fidelity"] <= 0.35)
    check("determinism: identical cell inputs give identical metrics",
          cell_property_classification(args, 0, "informative") == prop_inf)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        jf = os.path.join(td, "traces.jsonl")
        recs, t_end = gen_sequence(random.Random(5), moe, 8, 0, 0.05,
                                   "s0", 1.0,
                                   ExpertQueues(SEQ_SERVICE, SEQ_CAPACITY))
        with open(jf, "w") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        sessions = read_journal(jf)
        check("toy records round-trip through the journal schema",
              len(sessions.get("s0", [])) == len(recs))
        with open(jf, "a") as f:
            f.write(json.dumps({"expert_ids": [1]}) + "\n")
        try:
            read_journal(jf)
            check("journal drift is a descriptive ValueError", False)
        except ValueError as e:
            check("journal drift names file and line", f"{jf}:9" in str(e))
        try:
            read_journal(os.path.join(td, "missing.jsonl"))
            check("missing journal raises", False)
        except ValueError as e:
            check("missing journal names the file",
                  "missing.jsonl" in str(e))
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
