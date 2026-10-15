#!/usr/bin/env python3
"""kv-timing — KV-cache / prefix-cache timing side channel for inference
servers (the arXiv:2409.20002 line).

A serving stack with a shared prefix cache answers a request whose prefix
is already cached measurably faster (cache-warm TTFT) than an identical-
length request with a novel prefix (cache-cold TTFT, full prefill). The
attacker probes candidate prefixes — e.g. "did anyone recently serve a
prompt starting with <secret candidate>?" — and reads membership off the
warm/cold timing gap. Label-free; inference mode only.

Two arms (--arm; default prefix-cache):

  prefix-cache   the warm/cold TTFT channel above.
  spec-decode    speculative-decoding acceptance-pattern channel: when the
                 server runs speculative decoding, per-token acceptance
                 correlates with how predictable the continuation is — a
                 candidate prefix whose continuation the draft model
                 predicts well (boilerplate / a repeated, already-seen
                 prefix) shows a long high-acceptance span, a membership
                 signal for that candidate prefix. Inputs: --journal FILE
                 (offline acceptance journal, schema below), --toy
                 (planted acceptance world: member candidates get a
                 high-acceptance span, non-members chance acceptance
                 everywhere; plus a null world with the span disabled),
                 or --endpoint URL (reuses the live measurement path,
                 extended to record per-token acceptance WHEN the server
                 reports it — a graceful INFO cell when it does not).
                 Metric: candidate-prefix membership AUC (rank-based, vs
                 member labels) with a shuffled-label control; the frozen
                 gate is scored at the TOKEN level (pooled member vs
                 non-member acceptance — the honest large-n unit, cf.
                 gpu_memory's byte level: candidate counts cannot resolve
                 the 1.0 pp gate at realistic sizes).

Three input paths (prefix-cache arm):

  --endpoint URL   live measurement against an OpenAI-style /v1/completions
                   endpoint (or any raw URL accepting {"prompt": ...} JSON
                   POSTs). With max_tokens=1 and stream=false the whole-
                   response time approximates TTFT; it is recorded as both
                   ttft_ms and total_ms. Per candidate: one priming request
                   (warms the cache), then --reps warm probes of the exact
                   prefix and --reps cold probes of equal-length NOVEL
                   prefixes (never sharing the candidate, so they cannot
                   hit its cache entry).
  --journal FILE   offline analysis of a timing journal (schema below).
  --toy            synthetic warm/cold timing journals (known ground truth:
                   candidate 0 is the cache member) fed through the same
                   analyzer — the torch-free machinery check.

TIMING JOURNAL SCHEMA (llm_attacker.kv_timing.v1) — one JSON object/line:
  candidate   str, identifier of the candidate prefix under test
  condition   "warm" (cache hit expected) | "cold" (novel prefix)
  ttft_ms     float, time-to-first-token in milliseconds
  total_ms    float, total request latency in milliseconds
  rep         optional int, repetition index
Extra keys are tolerated; unknown conditions are rejected.

SPECULATIVE-DECODE JOURNAL SCHEMA (llm_attacker.kv_spec_decode.v1) — one
JSON object/line:
  candidate   str, identifier of the candidate prefix under test
  position    int, token position in the continuation
  accepted    bool (or 0/1): the speculated token at this position was
              accepted by the verifier (requires a server that exposes
              per-token speculative acceptance)
  latency_ms  optional float (may be null): per-token latency when the
              server reports it; accepted tokens are typically faster
  member      optional bool: GROUND TRUTH for scoring — the candidate is
              a known boilerplate/member prefix. Attacker-side the
              acceptance rate is the SIGNAL; the label is evaluation-only.
Extra keys are tolerated; a candidate's member labels must not conflict.

Live spec-decode response conventions (what extract_acceptance reads):
  {"spec_decode": {"accepted": [bool, ...], "latencies_ms": [...]?}}  or
  {"choices": [{"accepted_tokens": [bool, ...], ...}, ...]}
A server exposing neither gets a graceful INFO cell (channel not
measurable), never an error.

Statistics (pure python, torch-free): Welch's two-sample t-test between the
warm and cold TTFT samples per candidate (p-value via the regularized
incomplete beta function), Cohen's d effect size, and median-delta in ms.
Verdict per candidate: channel_open iff p < --alpha AND the warm median is
faster (median_delta_ms > 0); else channel_closed.

Usage:
    python -m attacker --mode inference --attack kv-timing --help
    python -m attacker --mode inference --attack kv-timing --toy --quick \
        --output /tmp/kv.json
    python -m attacker --mode inference --attack kv-timing \
        --endpoint http://host:8000/v1/completions --reps 30 \
        --candidate "System prompt of the victim deployment" --output kv.json
    python -m attacker --mode inference --attack kv-timing \
        --journal captures/timings.jsonl --output kv.json
    python -m attacker --mode inference --attack kv-timing \
        --arm spec-decode --toy --quick --output /tmp/spec.json
    python -m attacker --mode inference --attack kv-timing \
        --arm spec-decode --journal captures/acceptance.jsonl \
        --output spec.json
"""

import argparse
import json
import math
import random
import statistics
import time
import urllib.request
import uuid

from .. import artifacts, gate
from .common import add_common_args, journal_error
from .lira_membership import rank_auc

EXPERIMENT_ID = "kv_prefix_cache_timing"
MODES = ("inference",)
REQUIRES_LABELS = False
DESCRIPTION = ("KV/prefix-cache timing side channel (arXiv:2409.20002 "
               "line): warm-vs-cold TTFT separation -> prefix-membership "
               "inference against inference servers (live endpoint, offline "
               "journal, or toy analyzer paths); spec-decode arm: "
               "speculative-decoding acceptance-pattern membership channel "
               "(journal/toy/live, graceful INFO when the server hides "
               "acceptance)")

JOURNAL_SCHEMA = "llm_attacker.kv_timing.v1"
CONDITIONS = ("warm", "cold")

ARMS = ("prefix-cache", "spec-decode")
SPEC_JOURNAL_SCHEMA = "llm_attacker.kv_spec_decode.v1"
SPEC_ACCEPT_MS = 2.0
SPEC_REJECT_PENALTY_MS = 6.0

DEFAULT_CANDIDATES = ("shared system prompt of the deployment",)


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (Numerical
    Recipes betacf)."""
    max_it, eps, fpmin = 200, 3.0e-14, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_it + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betai(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    """Two-sided p-value of Student's t statistic at df degrees of freedom:
    p = I_{df/(df+t^2)}(df/2, 1/2)."""
    if df <= 0 or math.isnan(t) or math.isnan(df):
        return float("nan")
    if math.isinf(t):
        return 0.0
    return min(1.0, max(0.0, betai(df / 2.0, 0.5, df / (df + t * t))))


def _var(xs):
    return statistics.variance(xs) if len(xs) >= 2 else float("nan")


def welch_t_test(warm, cold):
    """Welch's two-sample t-test (does NOT assume equal variance). t is
    signed: positive means the cold sample is SLOWER (cache signal)."""
    na, nb = len(warm), len(cold)
    ma, mb = statistics.fmean(warm), statistics.fmean(cold)
    if na < 2 or nb < 2:
        return {"t": float("nan"), "df": float("nan"),
                "p_value": float("nan"), "mean_warm_ms": ma,
                "mean_cold_ms": mb}
    va, vb = _var(warm), _var(cold)
    se2 = va / na + vb / nb
    if se2 <= 0.0:
        t = 0.0 if ma == mb else math.copysign(float("inf"), mb - ma)
        df = float(na + nb - 2)
    else:
        t = (mb - ma) / math.sqrt(se2)
        df = se2 * se2 / ((va / na) ** 2 / (na - 1)
                          + (vb / nb) ** 2 / (nb - 1))
    return {"t": t, "df": df, "p_value": t_two_sided_p(t, df),
            "mean_warm_ms": ma, "mean_cold_ms": mb}


def cohens_d(warm, cold):
    """Pooled-SD effect size, signed: positive = cold slower than warm."""
    na, nb = len(warm), len(cold)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = _var(warm), _var(cold)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled <= 0.0:
        return 0.0 if statistics.fmean(warm) == statistics.fmean(cold) \
            else float("inf")
    return (statistics.fmean(cold) - statistics.fmean(warm)) / pooled


def analyze_candidate(candidate, warm, cold, alpha, source):
    """One candidate's warm/cold samples -> per-cell journal record with the
    channel verdict."""
    med_w, med_c = statistics.median(warm), statistics.median(cold)
    rec = {"experiment": EXPERIMENT_ID, "candidate": candidate,
         "source": source, "n_warm": len(warm), "n_cold": len(cold),
         "median_warm_ms": med_w, "median_cold_ms": med_c,
         "median_delta_ms": med_c - med_w, "alpha": alpha,
         **welch_t_test(warm, cold), "cohens_d": cohens_d(warm, cold)}
    rec["verdict"] = ("channel_open"
                      if (not math.isnan(rec["p_value"])
                          and rec["p_value"] < alpha
                          and rec["median_delta_ms"] > 0)
                      else "channel_closed")
    return rec


def measure_request(endpoint, prompt, model=None, timeout=10.0):
    """One timed completion request; returns elapsed ms. With max_tokens=1 /
    stream=false the whole-response time approximates TTFT."""
    payload = {"prompt": prompt, "max_tokens": 1, "stream": False}
    if model:
        payload["model"] = model
    req = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()
    return (time.perf_counter() - t0) * 1000.0


def _novel_prefix(candidate, rep):
    """An equal-length NOVEL prefix: never shares a token with the candidate
    (hex noise), so it cannot hit the candidate's cache entry; equal length
    controls the prompt-length confound on prefill time."""
    noise = f"x{uuid.uuid4().hex} y{uuid.uuid4().hex}"
    return (noise * (len(candidate) // len(noise) + 1))[:len(candidate)]


def measure_candidate(endpoint, candidate, reps, model=None, timeout=10.0,
                      suffix=""):
    """Prime the cache with the candidate, then interleave warm probes of
    the exact prefix with cold probes of novel prefixes."""
    measure_request(endpoint, candidate + suffix, model, timeout)
    warm, cold = [], []
    for rep in range(reps):
        warm.append(measure_request(endpoint, candidate + suffix, model,
                                    timeout))
        cold.append(measure_request(endpoint, _novel_prefix(candidate, rep)
                                    + suffix, model, timeout))
    return warm, cold


def read_timing_journal(path):
    """Parse a llm_attacker.kv_timing.v1 journal -> {candidate: {"warm":
    [...ms], "cold": [...ms]}} keyed on ttft_ms. Raises ValueError on
    schema drift (a malformed line names the journal, the line number, and
    the schema)."""
    groups = {}
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: not JSON ({e}); "
                                 f"expected schema {JOURNAL_SCHEMA}")
            missing = [k for k in ("candidate", "condition", "ttft_ms")
                       if k not in rec]
            if missing:
                raise ValueError(f"{path}:{lineno}: missing {missing}; "
                                 f"expected schema {JOURNAL_SCHEMA}")
            if rec["condition"] not in CONDITIONS:
                raise ValueError(f"{path}:{lineno}: condition "
                                 f"{rec['condition']!r} not in {CONDITIONS}")
            groups.setdefault(rec["candidate"], {"warm": [], "cold": []})[
                rec["condition"]].append(float(rec["ttft_ms"]))
    return groups


def toy_timing_records(candidates, reps, member, warm_ms, cold_ms, noise_ms,
                       seed):
    """Synthetic warm/cold TTFT records (schema llm_attacker.kv_timing.v1).
    The member candidate's warm probes hit the cache (warm_ms); every other
    probe — including warm-labeled probes of NON-members, whose prefixes
    were never cached — pays the full prefill (cold_ms)."""
    rng = random.Random(seed)
    records = []
    for ci in range(candidates):
        name = f"candidate-{ci}"
        for rep in range(reps):
            warm_mu = warm_ms if ci == member else cold_ms
            for condition, mu in (("warm", warm_mu), ("cold", cold_ms)):
                ttft = max(0.1, rng.gauss(mu, noise_ms))
                records.append({"candidate": name, "condition": condition,
                                "ttft_ms": ttft, "total_ms": ttft
                                + abs(rng.gauss(2.0, 0.5)), "rep": rep,
                                "seed": seed})
    return records


def toy_spec_records(candidates, members, positions, span, p_hi, p_lo,
                     base_ms, penalty_ms, noise_ms, seed, null=False):
    """Synthetic per-token acceptance records (schema
    llm_attacker.kv_spec_decode.v1). Member candidates (candidate-0 ..
    candidate-{members-1}) have a high-acceptance span over the first
    `span` positions — the boilerplate/repeated-prefix signal: the draft
    model predicts the continuation, the verifier keeps accepting. Every
    other position (and every position of a non-member, and EVERYTHING in
    the null world) accepts at the chance rate p_lo. Latency: accepted
    tokens are fast (base_ms), rejected pay the fallback (penalty_ms)."""
    rng = random.Random(seed)
    records = []
    for ci in range(candidates):
        name = f"candidate-{ci}"
        member = ci < members
        for pos in range(positions):
            p = p_hi if (member and not null and pos < span) else p_lo
            accepted = rng.random() < p
            lat = max(0.01, rng.gauss(
                base_ms + (0.0 if accepted else penalty_ms), noise_ms))
            records.append({"candidate": name, "position": pos,
                            "accepted": accepted, "latency_ms": lat,
                            "member": member, "seed": seed})
    return records


def read_spec_journal(path):
    """Parse a llm_attacker.kv_spec_decode.v1 journal -> {candidate:
    {"rows": [(position, accepted, latency_ms|None)], "member": bool|
    None}}. Raises ValueError on schema drift (naming the journal, the
    line number, and the schema)."""
    groups = {}
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: not JSON ({e}); "
                                 f"expected schema {SPEC_JOURNAL_SCHEMA}")
            missing = [k for k in ("candidate", "position", "accepted")
                       if k not in rec]
            if missing:
                raise ValueError(f"{path}:{lineno}: missing {missing}; "
                                 f"expected schema {SPEC_JOURNAL_SCHEMA}")
            if not isinstance(rec["accepted"], (bool, int)) \
                    or rec["accepted"] not in (0, 1, False, True):
                raise ValueError(f"{path}:{lineno}: accepted "
                                 f"{rec['accepted']!r} is not bool/0/1; "
                                 f"expected schema {SPEC_JOURNAL_SCHEMA}")
            member = rec.get("member")
            if member is not None and not isinstance(member, bool):
                raise ValueError(f"{path}:{lineno}: member {member!r} is "
                                 "not a bool; expected schema "
                                 f"{SPEC_JOURNAL_SCHEMA}")
            g = groups.setdefault(rec["candidate"],
                                  {"rows": [], "member": None})
            if member is not None:
                if g["member"] is not None and g["member"] != member:
                    raise ValueError(f"{path}:{lineno}: conflicting member "
                                     f"labels for candidate "
                                     f"{rec['candidate']!r}")
                g["member"] = member
            lat = rec.get("latency_ms")
            g["rows"].append((int(rec["position"]), bool(rec["accepted"]),
                              None if lat is None else float(lat)))
    return groups


def analyze_spec_candidate(candidate, rows, member, source):
    """One candidate's per-token rows -> per-candidate acceptance stats.
    Latency stats need EVERY row to carry latency_ms, else None (a server
    that reports acceptance without timing is a valid input)."""
    n = len(rows)
    n_acc = sum(1 for _, a, _ in rows if a)
    lats = [l for _, _, l in rows if l is not None]
    complete = n > 0 and len(lats) == n
    lat_a = [l for _, a, l in rows if a and l is not None]
    lat_r = [l for _, a, l in rows if not a and l is not None]
    return {"experiment": EXPERIMENT_ID, "arm": "spec-decode",
            "candidate": candidate, "source": source, "member": member,
            "n_tokens": n, "accepted_tokens": n_acc,
            "acceptance_rate": n_acc / max(1, n),
            "mean_latency_ms": (sum(lats) / len(lats)) if complete else None,
            "mean_latency_accepted_ms": (sum(lat_a) / len(lat_a))
            if complete and lat_a else None,
            "mean_latency_rejected_ms": (sum(lat_r) / len(lat_r))
            if complete and lat_r else None}


def spec_membership_cell(stats, source, seed, n_seeds):
    """The membership-AUC cell over per-candidate stats: rank AUC of the
    acceptance-rate score vs the member labels, a shuffled-label control
    AUC, the latency-based AUC (secondary signal; None when latencies are
    incomplete), and the frozen-gate verdict scored at the TOKEN level
    (pooled member vs non-member acceptance — the honest large-n unit,
    cf. gpu_memory's byte level; candidate counts cannot resolve the
    1.0 pp gate at realistic sizes). Returns None when labels are absent
    or one class is empty (the caller emits a graceful INFO cell)."""
    members = [s for s in stats if s["member"] is True]
    nonmembers = [s for s in stats if s["member"] is False]
    if not members or not nonmembers:
        return None
    auc = rank_auc([s["acceptance_rate"] for s in members],
                   [s["acceptance_rate"] for s in nonmembers])
    labels = [s["member"] for s in stats]
    shuffled = list(labels)
    random.Random(seed).shuffle(shuffled)
    sm = [s["acceptance_rate"] for s, m in zip(stats, shuffled) if m]
    sn = [s["acceptance_rate"] for s, m in zip(stats, shuffled) if not m]
    ctl_auc = rank_auc(sm, sn) if sm and sn else float("nan")
    lat_auc = None
    if all(s["mean_latency_ms"] is not None for s in stats):
        lat_auc = rank_auc([-s["mean_latency_ms"] for s in members],
                           [-s["mean_latency_ms"] for s in nonmembers])
    m_acc = sum(s["accepted_tokens"] for s in members)
    m_tok = sum(s["n_tokens"] for s in members)
    n_acc = sum(s["accepted_tokens"] for s in nonmembers)
    n_tok = sum(s["n_tokens"] for s in nonmembers)
    g = gate.verdict(m_acc / max(1, m_tok), n_acc / max(1, n_tok),
                     min(m_tok, n_tok), n_seeds=n_seeds)
    return {"experiment": EXPERIMENT_ID, "arm": "spec-decode",
            "source": source, "n_candidates": len(stats),
            "n_members": len(members), "n_nonmembers": len(nonmembers),
            "acceptance_auc": auc, "control_auc": ctl_auc,
            "latency_auc": lat_auc,
            "member_acceptance_mean": m_acc / max(1, m_tok),
            "nonmember_acceptance_mean": n_acc / max(1, n_tok),
            "n_gate_tokens": min(m_tok, n_tok), "gate": g,
            "membership_channel": "open" if g["verdict"] == "FAIL"
            else ("closed" if g["verdict"] == "PASS" else "unmeasured")}


def extract_acceptance(resp):
    """Per-token speculative acceptance from a completion response, when
    the server reports it. Supported conventions (module docstring):
    resp["spec_decode"]["accepted"] (+ optional ["latencies_ms"]), or
    resp["choices"][0]["accepted_tokens"]. Returns (accepted_list,
    latencies_list|None) or None when the server exposes neither."""
    if not isinstance(resp, dict):
        return None
    sd = resp.get("spec_decode")
    if isinstance(sd, dict) and isinstance(sd.get("accepted"), list):
        acc = list(sd["accepted"])
        lats = sd.get("latencies_ms")
        return (acc, list(lats) if isinstance(lats, list)
                and len(lats) == len(acc) else None)
    try:
        acc = resp["choices"][0]["accepted_tokens"]
    except (KeyError, IndexError, TypeError):
        return None
    return (list(acc), None) if isinstance(acc, list) else None


def measure_spec_request(endpoint, prompt, model=None, timeout=10.0,
                         max_tokens=16):
    """One completion probe that READS the response body (the prefix-cache
    path only times it): returns (parsed_json, elapsed_ms)."""
    payload = {"prompt": prompt, "max_tokens": max_tokens, "stream": False}
    if model:
        payload["model"] = model
    req = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    dt = (time.perf_counter() - t0) * 1000.0
    try:
        return json.loads(body), dt
    except json.JSONDecodeError:
        return {}, dt


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--arm", choices=ARMS, default="prefix-cache",
                    help="prefix-cache: the warm/cold TTFT channel "
                         "(default, all paths below). spec-decode: the "
                         "speculative-decoding acceptance-pattern "
                         "membership channel (same --endpoint/--journal/"
                         "--toy paths, spec journal schema)")
    src = ap.add_argument_group("input (pick one)")
    src.add_argument("--endpoint", default=None,
                     help="OpenAI-style /v1/completions URL (or raw URL "
                          "accepting {'prompt': ...} JSON POSTs)")
    src.add_argument("--journal", default=None,
                     help="offline journal (prefix-cache: "
                          f"{JOURNAL_SCHEMA}; spec-decode: "
                          f"{SPEC_JOURNAL_SCHEMA})")
    ap.add_argument("--candidate", action="append", default=None,
                    help="candidate prefix to probe (repeatable; live path)")
    ap.add_argument("--suffix", default="",
                    help="probe suffix appended after the candidate prefix "
                         "(live path)")
    ap.add_argument("--model", default=None,
                    help="model field for the OpenAI-style request body")
    ap.add_argument("--reps", type=int, default=30,
                    help="warm/cold repetitions per candidate")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="significance level for the channel-open verdict")
    ap.add_argument("--timeout", type=float, default=10.0,
                    help="per-request timeout in seconds (live path)")
    ap.add_argument("--candidates", type=int, default=3,
                    help="prefix-cache toy: number of candidate prefixes "
                         "(candidate-0 is the cache member)")
    ap.add_argument("--warm-ms", type=float, default=18.0,
                    help="toy: cache-warm TTFT mean")
    ap.add_argument("--cold-ms", type=float, default=32.0,
                    help="toy: cache-cold TTFT mean (full prefill)")
    ap.add_argument("--noise-ms", type=float, default=2.0,
                    help="toy: TTFT noise stddev")
    ap.add_argument("--spec-candidates", type=int, default=16,
                    help="spec-decode toy: candidate prefixes")
    ap.add_argument("--members", type=int, default=8,
                    help="spec-decode toy: member candidates (candidate-0 "
                         ".. candidate-{members-1}); must be < "
                         "--spec-candidates")
    ap.add_argument("--positions", type=int, default=4096,
                    help="spec-decode toy: continuation tokens per "
                         "candidate (default sizes resolve the 1.0 pp "
                         "gate at the token level, so a true null PASSes "
                         "with margin — cf. lira's --toy-n)")
    ap.add_argument("--span", type=int, default=2048,
                    help="spec-decode toy: high-acceptance boilerplate "
                         "span (positions) for member candidates")
    ap.add_argument("--p-hi", type=float, default=0.9,
                    help="spec-decode toy: acceptance probability inside "
                         "the boilerplate span")
    ap.add_argument("--p-lo", type=float, default=0.15,
                    help="spec-decode toy: chance acceptance probability "
                         "outside the span / for non-members")
    ap.add_argument("--max-tokens", type=int, default=16,
                    help="spec-decode live: tokens to request per probe")
    ap.add_argument("--member", action="append", default=None,
                    help="spec-decode live: name a candidate KNOWN to be "
                         "a member (repeatable; scoring ground truth, "
                         "must name a --candidate)")
    ap.add_argument("--evidence", action="store_true",
                    help="wrap the run in an EvidenceLogger (requires "
                         "--output): the artifact write is recorded as a "
                         "chained event and the evidence manifest is "
                         "finalized alongside")
    return ap


def _make_artifact(args, source, extra_config=None, evidence=None):
    arm = getattr(args, "arm", "prefix-cache")
    config = {"attack": EXPERIMENT_ID, "mode": getattr(args, "mode", None),
              "arm": arm, "source": source, "reps": args.reps,
              "alpha": args.alpha, "seed": args.seed}
    if extra_config:
        config.update(extra_config)
    threat = ("remote or co-located attacker measuring TTFT / total-latency "
              "differences between cache-warm (shared prefix) and cache-cold "
              "(novel prefix) requests against an inference server with a prefix "
              "cache — prefix-membership inference, label-free (the attacker "
              "needs only the candidate prefixes and a timing source).")
    interpretation = ("channel_open: the warm/cold timing gap is "
                      "statistically significant at alpha (Welch t-test, "
                      "warm faster) — the prefix cache leaks membership. "
                      "channel_closed: no exploitable separation measured "
                      "at this alpha/reps budget.")
    if arm == "spec-decode":
        threat = ("attacker probing candidate prefixes against an "
                  "inference server that runs SPECULATIVE DECODING and "
                  "exposes per-token acceptance: a candidate whose "
                  "continuation the draft model predicts well (boilerplate "
                  "/ repeated already-seen prefix) shows a high-acceptance "
                  "span — a membership signal for that candidate prefix. "
                  "Label-free: member labels are evaluation ground truth, "
                  "never an attack input.")
        interpretation = ("membership_channel=open (gate FAIL): member "
                          "candidates' acceptance pattern separates from "
                          "non-members — candidate-level membership AUC "
                          "above the shuffled-label control, and the "
                          "token-level pooled-acceptance excess beyond the "
                          "frozen gate (the honest large-n unit; candidate "
                          "counts cannot resolve 1.0 pp). latency_auc is "
                          "the secondary signal (accepted tokens are "
                          "faster). An INFO cell (no gate) means the "
                          "server/journal did not expose per-token "
                          "acceptance — the channel is not measurable "
                          "there, which is not a leak verdict either way.")
    return artifacts.make_artifact(
        "dtraining.attacker.kv_timing.v1", config, threat,
        interpretation=interpretation,
        evidence=evidence)


def _analyze_groups(groups, source, alpha, output, out, extra=None):
    """Run analyze_candidate over a {candidate: {warm, cold}} grouping;
    journal + collect each cell. Returns the per-candidate records."""
    recs = []
    for candidate in sorted(groups):
        g = groups[candidate]
        if len(g["warm"]) < 2 or len(g["cold"]) < 2:
            journal_error(output, EXPERIMENT_ID,
                          {"candidate": candidate, "source": source,
                           **(extra or {})},
                          "need >= 2 warm and 2 cold samples per candidate")
            continue
        rec = analyze_candidate(candidate, g["warm"], g["cold"], alpha,
                                source)
        if extra:
            rec.update(extra)
        artifacts.append_jsonl(output, rec)
        out["results"].append(rec)
        recs.append(rec)
        print(f"[kv-timing] {candidate} ({source}): "
              f"median delta={rec['median_delta_ms']:.2f}ms "
              f"p={rec['p_value']:.3g} d={rec['cohens_d']:.2f} "
              f"-> {rec['verdict']}")
    return recs


def _summarize(out, recs):
    n_open = sum(1 for r in recs if r["verdict"] == "channel_open")
    out["summary"].append({
        "experiment": EXPERIMENT_ID,
        "n_candidates": len(recs), "n_channel_open": n_open,
        "median_delta_ms_mean": (statistics.fmean(
            r["median_delta_ms"] for r in recs) if recs else float("nan")),
        "channel": "open" if n_open else "closed"})


def run_live(args, evidence=None):
    out = _make_artifact(args, "live", {
        "endpoint": args.endpoint, "candidates":
        args.candidate or list(DEFAULT_CANDIDATES), "model": args.model,
        "suffix": args.suffix}, evidence=evidence)
    groups = {}
    for candidate in args.candidate or list(DEFAULT_CANDIDATES):
        try:
            warm, cold = measure_candidate(args.endpoint, candidate,
                                           args.reps, model=args.model,
                                           timeout=args.timeout,
                                           suffix=args.suffix)
        except Exception as e:
            journal_error(args.output, EXPERIMENT_ID,
                          {"candidate": candidate, "source": "live",
                           "endpoint": args.endpoint}, e)
            continue
        groups[candidate] = {"warm": warm, "cold": cold}
    recs = _analyze_groups(groups, "live", args.alpha, args.output, out)
    _summarize(out, recs)
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def run_journal(args, evidence=None):
    out = _make_artifact(args, "journal", {"journal": args.journal},
                         evidence=evidence)
    groups = read_timing_journal(args.journal)
    recs = _analyze_groups(groups, "journal", args.alpha, args.output, out)
    _summarize(out, recs)
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def run_toy(args, evidence=None):
    if args.quick:
        args.reps = min(args.reps, 12)
        args.candidates = min(args.candidates, 2)
        args.seeds = args.seeds[:2]
    out = _make_artifact(args, "toy", {
        "toy": True, "candidates": args.candidates, "member": "candidate-0",
        "warm_ms": args.warm_ms, "cold_ms": args.cold_ms,
        "noise_ms": args.noise_ms, "seeds": args.seeds}, evidence=evidence)
    all_records = []
    member_open, nonmember_open = 0, 0
    for seed in args.seeds:
        records = toy_timing_records(args.candidates, args.reps, member=0,
                                     warm_ms=args.warm_ms,
                                     cold_ms=args.cold_ms,
                                     noise_ms=args.noise_ms,
                                     seed=args.seed + 1000 * seed)
        all_records.extend(records)
        groups = {}
        for r in records:
            groups.setdefault(r["candidate"], {"warm": [], "cold": []})[
                r["condition"]].append(r["ttft_ms"])
        recs = _analyze_groups(groups, "toy", args.alpha, args.output, out,
                               extra={"seed": seed})
        member_open += sum(1 for r in recs
                           if r["candidate"] == "candidate-0"
                           and r["verdict"] == "channel_open")
        nonmember_open += sum(1 for r in recs
                              if r["candidate"] != "candidate-0"
                              and r["verdict"] == "channel_open")
    if args.output and all_records:
        sidecar = args.output + ".timings.jsonl"
        with open(sidecar, "w") as f:
            for r in all_records:
                f.write(json.dumps(r) + "\n")
        out["toy_journal"] = {"path": sidecar, "schema": JOURNAL_SCHEMA,
                              "records": len(all_records)}
    _summarize(out, [r for r in out["results"]])
    n_member = len(args.seeds)
    out["summary"][-1].update({
        "toy_member": "candidate-0",
        "toy_member_detected": member_open == n_member,
        "toy_member_open_seeds": member_open,
        "toy_false_positive_cells": nonmember_open})
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def _run(args, evidence=None):
    if getattr(args, "arm", "prefix-cache") == "spec-decode":
        if args.journal:
            return run_spec_journal(args, evidence=evidence)
        if args.endpoint:
            return run_spec_live(args, evidence=evidence)
        if args.toy:
            return run_spec_toy(args, evidence=evidence)
        raise SystemExit(
            f"[{EXPERIMENT_ID}] spec-decode: pick an input path: "
            "--endpoint URL (live; a graceful INFO cell when the server "
            "hides per-token acceptance), --journal FILE (offline "
            f"acceptance journal, schema {SPEC_JOURNAL_SCHEMA}), or --toy "
            "(planted acceptance world).")
    if args.journal:
        return run_journal(args, evidence=evidence)
    if args.endpoint:
        return run_live(args, evidence=evidence)
    if args.toy:
        return run_toy(args, evidence=evidence)
    raise SystemExit(f"[{EXPERIMENT_ID}] pick an input path: --endpoint URL "
                     "(live measurement), --journal FILE (offline timing "
                     f"journal, schema {JOURNAL_SCHEMA}), or --toy "
                     "(synthetic machinery check).")


def _spec_sidecar(args, all_records, out):
    """Crash-safe raw acceptance journal sidecar (<output>.spec.jsonl) —
    the rows the analyzer consumed, reusable via --journal."""
    if args.output and all_records:
        sidecar = args.output + ".spec.jsonl"
        with open(sidecar, "w") as f:
            for r in all_records:
                f.write(json.dumps(r) + "\n")
        out["spec_journal"] = {"path": sidecar,
                               "schema": SPEC_JOURNAL_SCHEMA,
                               "records": len(all_records)}


def _spec_groups(records):
    groups = {}
    for r in records:
        groups.setdefault(r["candidate"], {"rows": [], "member": None})
        groups[r["candidate"]]["rows"].append(
            (r["position"], r["accepted"], r.get("latency_ms")))
        if r.get("member") is not None:
            groups[r["candidate"]]["member"] = r["member"]
    return groups


def run_spec_toy(args, evidence=None):
    if args.members >= args.spec_candidates:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] --members ({args.members}) must be < "
            f"--spec-candidates ({args.spec_candidates}) — the AUC "
            "contrast needs at least one non-member candidate")
    if args.quick:
        args.positions = min(args.positions, 2048)
        args.seeds = args.seeds[:2]
    out = _make_artifact(args, "toy", {
        "toy": True, "spec_candidates": args.spec_candidates,
        "members": args.members, "positions": args.positions,
        "span": args.span, "p_hi": args.p_hi, "p_lo": args.p_lo,
        "seeds": args.seeds}, evidence=evidence)
    all_records = []
    for seed in args.seeds:
        for world in ("planted", "null"):
            records = toy_spec_records(
                args.spec_candidates, args.members, args.positions,
                args.span, args.p_hi, args.p_lo, SPEC_ACCEPT_MS,
                SPEC_REJECT_PENALTY_MS, args.noise_ms,
                args.seed + 1000 * seed, null=(world == "null"))
            all_records.extend(records)
            stats = [analyze_spec_candidate(c, g["rows"], g["member"],
                                            "toy")
                     for c, g in sorted(_spec_groups(records).items())]
            cell = spec_membership_cell(stats, "toy",
                                        args.seed + 77 + 1000 * seed,
                                        n_seeds=len(args.seeds))
            cell.update({"world": world, "seed": seed})
            artifacts.append_jsonl(args.output, cell)
            out["results"].append(cell)
            print(f"[kv-timing/spec] {world} seed={seed}: "
                  f"acceptance AUC={cell['acceptance_auc']:.4f} "
                  f"(control {cell['control_auc']:.4f}) "
                  f"gate={cell['gate']['verdict']} "
                  f"channel={cell['membership_channel']}")
    _spec_sidecar(args, all_records, out)
    for world in ("planted", "null"):
        recs = [r for r in out["results"] if r["world"] == world]
        if recs:
            out["summary"].append({
                "experiment": EXPERIMENT_ID, "arm": "spec-decode",
                "source": "toy", "world": world, "n_seeds": len(recs),
                "acceptance_auc_mean": statistics.fmean(
                    r["acceptance_auc"] for r in recs),
                "member_acceptance_mean": statistics.fmean(
                    r["member_acceptance_mean"] for r in recs),
                "nonmember_acceptance_mean": statistics.fmean(
                    r["nonmember_acceptance_mean"] for r in recs),
                "gate_verdicts": [r["gate"]["verdict"] for r in recs],
                "membership_channel": recs[-1]["membership_channel"]})
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def run_spec_journal(args, evidence=None):
    out = _make_artifact(args, "journal",
                         {"journal": args.journal}, evidence=evidence)
    groups = read_spec_journal(args.journal)
    stats = []
    for candidate in sorted(groups):
        g = groups[candidate]
        s = analyze_spec_candidate(candidate, g["rows"], g["member"],
                                   "journal")
        artifacts.append_jsonl(args.output, s)
        out["results"].append(s)
        stats.append(s)
        print(f"[kv-timing/spec] {candidate} (journal): "
              f"acceptance={s['acceptance_rate']:.4f} "
              f"n={s['n_tokens']} member={s['member']}")
    cell = spec_membership_cell(stats, "journal", args.seed,
                                n_seeds=len(args.seeds))
    if cell is None:
        cell = {"experiment": EXPERIMENT_ID, "arm": "spec-decode",
                "source": "journal",
                "info": "journal carries no usable member labels (need "
                        ">= 1 member and >= 1 non-member): per-candidate "
                        "stats only — attacker-side the acceptance rate "
                        "IS the signal; labels are scoring ground truth"}
    artifacts.append_jsonl(args.output, cell)
    out["results"].append(cell)
    out["summary"].append({
        "experiment": EXPERIMENT_ID, "arm": "spec-decode",
        "source": "journal", "n_candidates": len(stats),
        "membership_scored": "gate" in cell,
        **({"membership_channel": cell["membership_channel"],
            "acceptance_auc": cell["acceptance_auc"]}
           if "gate" in cell else {})})
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def run_spec_live(args, evidence=None):
    candidates = args.candidate or list(DEFAULT_CANDIDATES)
    members = set(args.member or [])
    unknown = members - set(candidates)
    if unknown:
        raise SystemExit(f"[{EXPERIMENT_ID}] --member names {sorted(unknown)} "
                         "which are not probed: pass them as --candidate too "
                         "(member labels are scoring ground truth)")
    out = _make_artifact(args, "live", {
        "endpoint": args.endpoint, "candidates": candidates,
        "members": sorted(members), "model": args.model,
        "max_tokens": args.max_tokens}, evidence=evidence)
    stats, exposed = [], []
    for candidate in candidates:
        try:
            resp, elapsed = measure_spec_request(
                args.endpoint, candidate, model=args.model,
                timeout=args.timeout, max_tokens=args.max_tokens)
        except Exception as e:
            journal_error(args.output, EXPERIMENT_ID,
                          {"candidate": candidate, "source": "live",
                           "arm": "spec-decode",
                           "endpoint": args.endpoint}, e)
            continue
        found = extract_acceptance(resp)
        if found is None:
            rec = {"experiment": EXPERIMENT_ID, "arm": "spec-decode",
                   "candidate": candidate, "source": "live",
                   "acceptance_exposed": False,
                   "total_ms": elapsed,
                   "info": "server response carries no per-token "
                           "speculative acceptance (checked conventions: "
                           "spec_decode.accepted, "
                           "choices[].accepted_tokens); the channel is "
                           "not measurable against this endpoint"}
            artifacts.append_jsonl(args.output, rec)
            out["results"].append(rec)
            print(f"[kv-timing/spec] {candidate} (live): INFO — server "
                  "does not expose per-token acceptance")
            continue
        accepted, latencies = found
        rows = [(i, bool(a),
                 None if latencies is None else float(latencies[i]))
                for i, a in enumerate(accepted)]
        for i, a, l in rows:
            exposed.append({"candidate": candidate, "position": i,
                            "accepted": a, "latency_ms": l,
                            "member": candidate in members})
        s = analyze_spec_candidate(candidate, rows,
                                   candidate in members, "live")
        s["acceptance_exposed"] = True
        s["total_ms"] = elapsed
        artifacts.append_jsonl(args.output, s)
        out["results"].append(s)
        stats.append(s)
        print(f"[kv-timing/spec] {candidate} (live): "
              f"acceptance={s['acceptance_rate']:.4f} "
              f"n={s['n_tokens']} member={s['member']}")
    _spec_sidecar(args, exposed, out)
    cell = spec_membership_cell(stats, "live", args.seed,
                                n_seeds=len(args.seeds))
    if cell is None:
        cell = {"experiment": EXPERIMENT_ID, "arm": "spec-decode",
                "source": "live",
                "info": "no membership labels (--member) or no exposed "
                        "acceptance: per-candidate stats only"}
    artifacts.append_jsonl(args.output, cell)
    out["results"].append(cell)
    out["summary"].append({
        "experiment": EXPERIMENT_ID, "arm": "spec-decode",
        "source": "live", "n_candidates": len(candidates),
        "n_exposed": len(stats),
        "membership_scored": "gate" in cell,
        **({"membership_channel": cell["membership_channel"],
            "acceptance_auc": cell["acceptance_auc"]}
           if "gate" in cell else {})})
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def run(args):
    if getattr(args, "evidence", False):
        if not args.output:
            raise SystemExit(f"[{EXPERIMENT_ID}] --evidence requires "
                             "--output (the evidence manifest is written "
                             "next to the artifact)")
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

    print("kv-timing prefix-cache side channel (pure python):")

    p = t_two_sided_p(2.0, 10.0)
    check("t two-sided p matches the t-table (t=2.0, df=10 -> 0.0734)",
          abs(p - 0.0734) < 5e-4)
    check("t=0 gives p=1", abs(t_two_sided_p(0.0, 15.0) - 1.0) < 1e-12)
    check("huge t gives p ~ 0", t_two_sided_p(30.0, 8.0) < 1e-8)
    check("betai anchors", betai(2.0, 3.0, 0.0) == 0.0
          and betai(2.0, 3.0, 1.0) == 1.0)

    rng = random.Random(0)
    warm = [rng.gauss(18.0, 2.0) for _ in range(30)]
    cold = [rng.gauss(32.0, 2.0) for _ in range(30)]
    rec = analyze_candidate("c", warm, cold, 0.05, "selftest")
    check("separated distributions open the channel",
          rec["verdict"] == "channel_open" and rec["p_value"] < 1e-10
          and rec["median_delta_ms"] > 10.0 and rec["cohens_d"] > 3.0)
    same = analyze_candidate("c", warm, list(warm), 0.05, "selftest")
    check("identical distributions stay closed",
          same["verdict"] == "channel_closed")
    noise = analyze_candidate("c", warm[:30],
                              [rng.gauss(18.0, 2.0) for _ in range(30)],
                              0.05, "selftest")
    check("same-mean noise stays closed", noise["verdict"] == "channel_closed")
    inv = analyze_candidate("c", cold, warm, 0.05, "selftest")
    check("warm SLOWER than cold is not a cache channel (sign check)",
          inv["verdict"] == "channel_closed")

    recs = toy_timing_records(3, 12, member=0, warm_ms=18.0, cold_ms=32.0,
                              noise_ms=2.0, seed=1)
    check("toy journal matches the declared schema",
          len(recs) == 3 * 12 * 2 and all(
              set(("candidate", "condition", "ttft_ms", "total_ms", "rep"))
              <= set(r) and r["condition"] in CONDITIONS for r in recs))
    groups = {}
    for r in recs:
        groups.setdefault(r["candidate"], {"warm": [], "cold": []})[
            r["condition"]].append(r["ttft_ms"])
    member = analyze_candidate("candidate-0", *map(groups["candidate-0"].get,
                                                   CONDITIONS), 0.05, "toy")
    other = analyze_candidate("candidate-1", *map(groups["candidate-1"].get,
                                                  CONDITIONS), 0.05, "toy")
    check("toy: the cache member is detected open",
          member["verdict"] == "channel_open")
    check("toy: non-members stay closed (no false positive)",
          other["verdict"] == "channel_closed")

    check("equal-length novel prefixes never share the candidate",
          all(_novel_prefix("abc", i) != "abc"
              and len(_novel_prefix("abc", i)) == 3 for i in range(4)))

    spec = toy_spec_records(4, 2, 64, 32, 0.9, 0.15, 2.0, 6.0, 0.5, 1)
    check("spec toy journal matches the declared schema",
          len(spec) == 4 * 64 and all(
              set(("candidate", "position", "accepted", "latency_ms",
                   "member")) <= set(r) for r in spec))
    mem_rows = [r for r in spec if r["candidate"] == "candidate-0"]
    check("spec toy: member span accepts high, tail at chance",
          statistics.fmean(r["accepted"] for r in mem_rows[:32]) > 0.75
          and statistics.fmean(r["accepted"] for r in mem_rows[32:]) < 0.35)
    non_rows = [r for r in spec if r["candidate"] == "candidate-3"]
    check("spec toy: non-members accept at chance everywhere",
          statistics.fmean(r["accepted"] for r in non_rows) < 0.35)
    null_spec = toy_spec_records(4, 2, 64, 32, 0.9, 0.15, 2.0, 6.0, 0.5, 1,
                                 null=True)
    check("spec toy: the null world disables the member span",
          statistics.fmean(r["accepted"] for r in null_spec
                           if r["candidate"] == "candidate-0") < 0.35)

    args = build_parser().parse_args(["--arm", "spec-decode"])
    planted = toy_spec_records(args.spec_candidates, args.members,
                               args.positions, args.span, args.p_hi,
                               args.p_lo, SPEC_ACCEPT_MS,
                               SPEC_REJECT_PENALTY_MS, args.noise_ms, 11)
    stats = [analyze_spec_candidate(c, g["rows"], g["member"], "toy")
             for c, g in sorted(_spec_groups(planted).items())]
    cell = spec_membership_cell(stats, "toy", 99, n_seeds=1)
    check("spec planted: membership AUC at ceiling, control at chance",
          cell["acceptance_auc"] == 1.0
          and 0.3 <= cell["control_auc"] <= 0.7)
    check("spec planted: accepted tokens are faster (latency AUC)",
          cell["latency_auc"] is not None and cell["latency_auc"] > 0.95)
    check("spec planted: the token-level gate FAILs (channel open)",
          cell["gate"]["verdict"] == "FAIL"
          and cell["membership_channel"] == "open")
    null_stats = [analyze_spec_candidate(c, g["rows"], g["member"], "toy")
                  for c, g in sorted(_spec_groups(
                      toy_spec_records(args.spec_candidates, args.members,
                                       args.positions, args.span, args.p_hi,
                                       args.p_lo, SPEC_ACCEPT_MS,
                                       SPEC_REJECT_PENALTY_MS,
                                       args.noise_ms, 11,
                                       null=True)).items())]
    null_cell = spec_membership_cell(null_stats, "toy", 99, n_seeds=1)
    check("spec null: the token-level gate PASSes at the default sizes "
          "(floor resolved with margin, cf. lira --toy-n)",
          null_cell["gate"]["verdict"] == "INSUFFICIENT_EVIDENCE"
          and null_cell["membership_channel"] == "unmeasured")
    check("spec null: no member-label classes -> no cell (INFO path)",
          spec_membership_cell(
              [dict(s, member=None) for s in stats], "toy", 1,
              n_seeds=1) is None)

    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        jf = os.path.join(d, "spec.jsonl")
        with open(jf, "w") as f:
            for r in spec:
                if r["position"] < 4:
                    f.write(json.dumps(r) + "\n")
        parsed = read_spec_journal(jf)
        check("spec journal round-trips",
              len(parsed) == 4 and len(parsed["candidate-0"]["rows"]) == 4
              and parsed["candidate-0"]["member"] is True)
        with open(jf, "w") as f:
            f.write(json.dumps({"candidate": "x", "position": 0}) + "\n")
        try:
            read_spec_journal(jf)
            check("spec journal drift is descriptive", False)
        except ValueError as e:
            check("spec journal drift is descriptive",
                  "accepted" in str(e) and SPEC_JOURNAL_SCHEMA in str(e))
        with open(jf, "w") as f:
            f.write(json.dumps({"candidate": "x", "position": 0,
                                "accepted": True, "member": True}) + "\n")
            f.write(json.dumps({"candidate": "x", "position": 1,
                                "accepted": False, "member": False}) + "\n")
        try:
            read_spec_journal(jf)
            check("conflicting member labels rejected", False)
        except ValueError as e:
            check("conflicting member labels rejected",
                  "conflicting" in str(e))

    check("extract_acceptance reads the spec_decode convention",
          extract_acceptance({"spec_decode": {"accepted": [True, False],
                                              "latencies_ms": [1.0, 7.0]}})
          == ([True, False], [1.0, 7.0]))
    check("extract_acceptance reads the choices convention",
          extract_acceptance({"choices": [{"accepted_tokens": [False]}]})
          == ([False], None))
    check("extract_acceptance: no convention -> None (INFO cell path)",
          extract_acceptance({"choices": [{"text": " ok"}]}) is None
          and extract_acceptance({}) is None)
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
