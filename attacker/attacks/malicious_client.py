#!/usr/bin/env python3
"""malicious-client — an untrusted AUTHENTICATED CLIENT against the shared
cloud computation of a split/distributed LLM (THREAT_MODEL_REVIEW P1
"malicious-client"). The malicious-server / active-cloud attacks cover the
untrusted CLOUD; this module flips the vantage: the attacker is a client
with valid credentials who controls only its own inputs and updates, but
shares the cloud body (and, in training, the shared head) with honest
co-clients. Label-free; applies to inference (query arms) and to
split-training (update arms).

THE TOY WORLD (simulator, pure python — no torch, no numpy)
===========================================================
The cloud holds a shared linear body slice y = W x + b with W 12x8 and
b 12 (seeded; D_OUT x D_IN). Honest clients send seeded benign inputs
(x ~ N(0, I)); one client is SHIFTED-BUT-BENIGN (x ~ N(0.4, 1.5^2 I)) and
must NEVER trip the detectors (the false-positive control). The attacker
client sends crafted ones. Small linear-algebra helpers (matvec, Gaussian
elimination with partial pivoting, ridge least squares) live below —
pure python lists.

WORLDS
======
  vulnerable   no server-side checks at all
  defended     all six mitigations on (the NEGATIVE CONTROL):
               input_validation (reject non-finite / |x|_inf > norm_cap),
               update_clipping (per-client update norm <= clip_tau),
               budget_limiter (per-TENANT query cap = D_IN, one below the
               D_IN+1 identifiability threshold of the linear body),
               anomaly_detection (per-tenant duplicate-probe registry;
               update rejected when its cosine to the coordinate-wise
               median update of the round < reject_cosine),
               response_rounding (query responses rounded to
               response_quantum — the API-logit-rounding defense),
               response_perturbation (Gaussian noise sigma=0.5 on query
               responses — the output-perturbation defense; kills
               variance readout even within budget)
  isolated     ISOLATED CONTROL: the attacker alone against a private,
               undefended body instance — no shared state, no co-clients.
               Cross-client harm metrics are 0 there by construction, so
               the shared-world excess over isolation is exactly the harm
               that SHARING enables.

ARMS (per-cell journals; see JOURNAL SCHEMA below)
==================================================
  crafted-extraction   functional body extraction: query with the
                       zero+axes design (x=0 reads b, then unit axes read
                       columns of W), fit (W_hat, b_hat) by ridge least
                       squares; relative Frobenius error / mean row cosine
                       / held-out R^2 per query budget (--budgets). Exact
                       (rel ~ 1e-15) once the granted budget reaches
                       D_IN+1; the defended per-tenant cap (D_IN) sits one
                       probe below identifiability.
  neuron-observability chosen inputs maximizing per-neuron (sensitive
                       neuron / router analogue) observability: K seeded
                       random probes, per-neuron scale estimate
                       std(y_j) (x ~ N(0, I) => Var(y_j) = ||W_j||^2) vs
                       the true row norms ||W_j||;
                       mean relative error + Spearman rank correlation.
                       The defended cap leaves K=8 probes (sampling error
                       ~27%) and response perturbation inflates the small
                       scales — the estimate lands far outside the gates.
  gradient-poisoning   bounded malicious gradients against shared
                       training of the linear head h over the shared body:
                       the attacker submits -tau * g_hat (sign-flipped,
                       norm tau = 8) where g_hat is the unit honest mean
                       gradient (worst-case oracle: the direction is
                       estimable from the observed shared-head drift).
                       Vulnerable averaging lets the shared head settle at
                       a displaced equilibrium (honest held-out MSE
                       explodes vs the no-attack baseline); the defended
                       world rejects the attacker every round (cosine ~ -1
                       to the round median) — final MSE == baseline
                       EXACTLY — and never rejects the shifted-benign
                       client (fp_rate = 0).
  instability-bomb     numerical-instability inputs: bounded activation
                       bombs (|x|=7.9, just under the cap), huge finite
                       probes (1e6..1e308), and non-finite probes
                       (inf, nan). Responses feed a shared running-mean
                       calibration state consumed by ALL clients' head
                       readout; a non-finite grant poisons it permanently.
                       Metrics: probes accepted/rejected, detection rate,
                       honest outputs corrupted post-bomb, compute and
                       memory amplification vs the benign stream, and a
                       benign-parity client (same query count, ordinary
                       inputs) that must sit at amplification ~1.
  budget-extraction    extraction fidelity at a FIXED budget
                       (--extract-budget = D_IN = 8): naive random design
                       vs the optimal zero+axes design, with the limiter
                       off (vulnerable: the 9th, overspending probe is
                       granted — exact extraction) vs on (defended: capped
                       at 8 — one below identifiability, rounding on top).
  collusion            C colluding sybil clients (--colluders) under ONE
                       tenant average their noisy observation channels
                       (per-response noise --noise-sigma): identical probe
                       sets, responses averaged -> sigma/sqrt(C) ->
                       fidelity gain ~ sqrt(C) over a single client.
                       Defended: the per-TENANT budget cap + duplicate-
                       probe registry give sybils nothing a single client
                       does not have (gain == 1.0); a cross-tenant honest
                       duplicate (two honest clients sending the same
                       canonical health-check probe) must stay accepted.

METRICS / JOURNAL SCHEMA (per-cell records, <output>.jsonl + artifact)
======================================================================
Every record carries: experiment, arm, world (vulnerable|defended|
isolated), role (attack|negative-control|isolated-control), seed, plus
per-arm fields:
  crafted-extraction / budget-extraction:
    budget, design (budget-extraction), queries_requested,
    queries_granted, determinate (granted >= D_IN+1), rel_frobenius,
    mean_row_cosine, bias_rel_err, r2_heldout, isolated_rel_frobenius,
    fidelity_delta_vs_isolated, n_rejected, attack_succeeded
  neuron-observability:
    n_probes_granted, mean_rel_err, max_rel_err, rank_corr,
    isolated_mean_rel_err, fidelity_delta_vs_isolated, attack_succeeded
  gradient-poisoning:
    baseline_mse, final_mse, honest_mse_delta, mse_ratio,
    detection_rate, false_positive_rate, n_clipped, privacy_cosine_mean
    (attacker readout of the honest mean-gradient direction from the
    shared-head drift — shared training leaks it in BOTH worlds; clipping
    protects integrity, not privacy), isolated_honest_mse_delta (0.0),
    cross_client_excess, attack_succeeded
  instability-bomb:
    probes_accepted, probes_rejected, rejection_worthy, detection_rate,
    shifted_benign_accepted/total (false-positive check),
    server_corrupted (own-instance calibration state non-finite),
    pre_bomb_honest_ok, honest_outputs_corrupted, peak_activation_log10,
    compute_amplification, memory_amplification, benign_parity,
    isolated_honest_corrupted (0), cross_client_excess, attack_succeeded
  collusion:
    rel_err_single, rel_err_colluded, gain (single/colluded),
    expected_gain_sqrt_c, queries_granted, queries_rejected,
    budget_limit, cross_tenant_duplicate_accepted,
    isolated_rel_err_single, fidelity_delta_vs_isolated, attack_succeeded
attack_succeeded is the per-arm security gate (rel_frobenius <= 0.1;
mean_rel_err <= 0.18 AND rank_corr >= 0.9; honest_mse_delta >
max(0.05, 0.5*baseline); honest_outputs_corrupted > 0; gain >= 1.5).
Isolated-control cells always carry attack_succeeded = False (isolation
removes the cross-client harm by construction); their metric values are
the capability reference the shared worlds are compared against.

REAL TARGETS (integration points)
=================================
--probe-file FILE replays a captured client-stream journal (schema
llm_attacker.malicious_client.probe.v1: one JSON object per line,
{"client": str, "kind": "query"|"update", "x": [D_IN floats] |
"g": [D_OUT floats], "tenant": str optional, "ts": float optional})
through the defended policy stack OFFLINE: per-tenant budget accounting,
duplicate-probe registry, input validation, update clipping, and the
median-cosine anomaly detector (reference = the stream's own median
update). Schema drift is a descriptive ValueError naming file and line.
The LIVE driver is a documented integration point (not wired here): wrap
a real split server's query/update intake with SharedBodyServer's policy
stack and drive the same arm functions against real body responses.

LIMITATIONS / EXCLUDED VARIANTS
===============================
The body slice is LINEAR (12x8): no nonlinearities, no LayerNorm, no
attention — extraction/observability results are upper bounds on a real
stack. The poisoning attacker is handed the honest-gradient direction
(worst case); a real attacker estimates it with error. The detector's
reference is the round median of submitted updates (no adaptive attacker
that poisons the median itself, no Sybil-majority evasion). Collusion
assumes all sybils authenticate under ONE tenant (cross-tenant sybil
attribution is out of scope). No multi-epoch persistence, no cryptographic
privacy (DP/secure aggregation) layers, abstract compute/memory units (not
GPU-calibrated), and the real-target live driver is an integration point.

Usage:
    python -m attacker --mode training --attack malicious-client --help
    python -m attacker --mode training --attack malicious-client \
        --toy --quick --output /tmp/mc.json
    python -m attacker --mode inference --attack malicious-client --toy \
        --arms crafted-extraction budget-extraction collusion \
        --output /tmp/mc-extract.json
    python -m attacker --mode training --attack malicious-client \
        --probe-file captures/client_stream.jsonl --output /tmp/mc-replay.json
"""

import argparse
import json
import math
import random
import statistics

from .. import artifacts, threat_model
from .common import add_common_args, journal_error

EXPERIMENT_ID = "malicious_client"
MODES = ("training", "inference")
REQUIRES_LABELS = False
DESCRIPTION = ("untrusted authenticated client against the shared cloud "
               "body: crafted-activation extraction, neuron observability, "
               "gradient poisoning, instability bombs, budget-limited "
               "extraction, and sybil collusion (vulnerable vs defended "
               "vs isolated worlds)")

SCHEMA_VERSION = "dtraining.attacker.malicious_client.v1"
PROBE_SCHEMA = "llm_attacker.malicious_client.probe.v1"

D_IN, D_OUT = 8, 12

DEFENSES = ("input_validation", "update_clipping", "budget_limiter",
            "anomaly_detection", "response_rounding", "response_perturbation")
VULNERABLE = {d: False for d in DEFENSES}
DEFENDED = {d: True for d in DEFENSES}
WORLDS = ("vulnerable", "defended")

PARAMS = {"budget_limit": D_IN, "norm_cap": 8.0, "clip_tau": 1.0,
          "reject_cosine": 0.2, "response_quantum": 0.5,
          "response_noise": 0.5}

ARMS = ("crafted-extraction", "neuron-observability", "gradient-poisoning",
        "instability-bomb", "budget-extraction", "collusion")

_HEADLINE = {"crafted-extraction": "rel_frobenius",
             "neuron-observability": "mean_rel_err",
             "gradient-poisoning": "honest_mse_delta",
             "instability-bomb": "honest_outputs_corrupted",
             "budget-extraction": "rel_frobenius",
             "collusion": "gain"}

EXTRACT_GATE = 0.1
OBS_REL_GATE = 0.18
OBS_RANK_GATE = 0.9
POISON_GATE_ABS = 0.05
COLLUSION_GAIN_GATE = 1.5

PARSE_COST = 1.0
COST_LOG_CAP = 700.0
ARENA_CAP = 1e6

POISON_LR, POISON_TAU = 0.05, 8.0


def matvec(W, x):
    return [sum(wij * xj for wij, xj in zip(row, x)) for row in W]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm2(v):
    return math.sqrt(sum(x * x for x in v))


def cosine(a, b):
    na, nb = norm2(a), norm2(b)
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    return dot(a, b) / (na * nb)


def gauss_solve(M, B):
    """Solve M X = B (M n x n, B n x m) by Gauss-Jordan elimination with
    partial pivoting. Raises ValueError on a (near-)singular pivot."""
    n = len(M)
    m = len(B[0])
    A = [list(M[i]) + list(B[i]) for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) < 1e-12:
            raise ValueError("gauss_solve: matrix is (near-)singular")
        A[col], A[piv] = A[piv], A[col]
        inv = 1.0 / A[col][col]
        A[col] = [v * inv for v in A[col]]
        for r in range(n):
            if r != col:
                f = A[r][col]
                if f:
                    A[r] = [a - f * p for a, p in zip(A[r], A[col])]
    return [row[n:n + m] for row in A]


def solve_ls(A, Y, ridge=1e-9):
    """Least squares theta = argmin ||A theta - Y||^2 + ridge||theta||^2 via
    the normal equations (A rows are samples, Y rows are targets). With
    ridge ~ 0 an exactly-determined system is recovered to float precision;
    an underdetermined one lands near the minimum-norm solution."""
    n_cols = len(A[0])
    M = [[sum(row[i] * row[j] for row in A) for j in range(n_cols)]
         for i in range(n_cols)]
    for i in range(n_cols):
        M[i][i] += ridge
    B = [[sum(row[i] * yrow[j] for row, yrow in zip(A, Y))
          for j in range(len(Y[0]))] for i in range(n_cols)]
    return gauss_solve(M, B)


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


def _fin(x):
    """JSON-safe: non-finite floats become None (never a NaN literal in a
    journal record)."""
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def make_body(seed):
    """The seeded shared linear body slice y = W x + b (D_OUT x D_IN).
    Rows carry a planted per-neuron scale profile (0.35 .. 1.6) — the toy
    analogue of sensitive neurons / high-gain router paths, well separated
    so the observability arm's ranking metric is informative."""
    rng = random.Random(seed)
    W = [[rng.gauss(0.0, 0.5) for _ in range(D_IN)] for _ in range(D_OUT)]
    for j in range(D_OUT):
        s = 0.35 + 1.25 * j / (D_OUT - 1)
        W[j] = [v * s for v in W[j]]
    b = [rng.gauss(0.0, 0.25) for _ in range(D_OUT)]
    return W, b


def make_eval_set(seed, n=24):
    """Held-out probe set for extraction R^2 (seeded; identical for every
    world of a seed so the contrast is policy-only)."""
    W, b = make_body(seed)
    rng = random.Random(seed + 99)
    xs = [[rng.gauss(0.0, 1.0) for _ in range(D_IN)] for _ in range(n)]
    ys = [matvec(W, x) for x in xs]
    for x, y in zip(xs, ys):
        for j in range(D_OUT):
            y[j] += b[j]
    return xs, ys


class SharedBodyServer:
    """The cloud's shared body slice under a defense config. Grants or
    rejects client queries per the policy stack (input validation ->
    per-tenant duplicate-probe registry -> per-tenant budget), applies
    response noise (collusion arm) and response rounding, and meters
    abstract compute units and the peak activation magnitude (the
    amplification metrics). Rejections cost PARSE_COST (the intake work is
    already spent)."""

    def __init__(self, seed, defenses, params=None, noise_sigma=0.0):
        self.W, self.b = make_body(seed)
        self.defenses = dict(defenses)
        self.p = dict(PARAMS)
        self.p.update(params or {})
        self.noise_sigma = noise_sigma
        self._noise = random.Random(seed + 777)
        self.budget_used = {}
        self.seen_probes = {}
        self.n_rejected = {}
        self.compute_units = 0.0
        self.peak_activation = 0.0
        self.cost_log = []

    def _meter(self, tenant, granted, y):
        if granted:
            mag = max(abs(v) for v in y)
            if math.isfinite(mag):
                self.peak_activation = max(self.peak_activation, mag)
                cost = 1.0 + min(math.log1p(mag), COST_LOG_CAP)
            else:
                self.peak_activation = float("inf")
                cost = 1.0 + COST_LOG_CAP
        else:
            cost = PARSE_COST
        self.compute_units += cost
        self.cost_log.append((tenant, granted, cost))
        return cost

    def query(self, client, x, tenant=None):
        """One client query. Returns (y, None) on grant, (None, reason) on
        rejection. Never raises on adversarial input — rejection is the
        defended behavior."""
        tenant = tenant or client
        if self.defenses["input_validation"]:
            if any(not math.isfinite(v) for v in x):
                return self._reject(tenant, "invalid-input")
            if max(abs(v) for v in x) > self.p["norm_cap"]:
                return self._reject(tenant, "invalid-input")
        if self.defenses["anomaly_detection"]:
            if tuple(x) in self.seen_probes.get(tenant, ()):
                return self._reject(tenant, "duplicate-probe")
        if self.defenses["budget_limiter"]:
            if self.budget_used.get(tenant, 0) >= self.p["budget_limit"]:
                return self._reject(tenant, "budget-exceeded")
        y = matvec(self.W, x)
        for j in range(D_OUT):
            y[j] += self.b[j]
            if self.noise_sigma:
                y[j] += self._noise.gauss(0.0, self.noise_sigma)
            if self.defenses["response_perturbation"]:
                y[j] += self._noise.gauss(0.0, self.p["response_noise"])
        if self.defenses["response_rounding"]:
            q = self.p["response_quantum"]
            y = [round(v / q) * q for v in y]
        self.seen_probes.setdefault(tenant, set()).add(tuple(x))
        self.budget_used[tenant] = self.budget_used.get(tenant, 0) + 1
        self._meter(tenant, True, y)
        return y, None

    def _reject(self, tenant, reason):
        self.n_rejected[reason] = self.n_rejected.get(reason, 0) + 1
        self._meter(tenant, False, None)
        return None, reason


def design_zero_axes(n_extra, seed):
    """The optimal linear-body design: x = 0 (reads b), then the D_IN unit
    axes (each reads one column of W), then seeded Gaussian extras."""
    rng = random.Random(seed + 21)
    probes = [[0.0] * D_IN]
    for i in range(D_IN):
        e = [0.0] * D_IN
        e[i] = 1.0
        probes.append(e)
    for _ in range(n_extra):
        probes.append([rng.gauss(0.0, 1.0) for _ in range(D_IN)])
    return probes


def design_random(n, seed, tag=22):
    rng = random.Random(seed + tag)
    return [[rng.gauss(0.0, 1.0) for _ in range(D_IN)] for _ in range(n)]


def query_all(server, probes, client="atk", tenant="atk"):
    """Run probes through the server; returns (granted_pairs, n_rejected)."""
    pairs, n_rejected = [], 0
    for x in probes:
        y, _ = server.query(client, x, tenant=tenant)
        if y is None:
            n_rejected += 1
        else:
            pairs.append((x, y))
    return pairs, n_rejected


def fit_body(pairs, ridge=None):
    """Ridge least-squares fit of (W, b) from granted (x, y) pairs. The
    ridge is adaptive: an (over-)determined system gets ~0 (recovery to
    float precision); an underdetermined one gets 1e-9 and lands near the
    minimum-norm solution."""
    if not pairs:
        raise ValueError("fit_body: no granted (query, response) pairs "
                         "to fit")
    if ridge is None:
        ridge = 1e-12 if len(pairs) >= D_IN + 1 else 1e-9
    A = [list(x) + [1.0] for x, _ in pairs]
    Y = [list(y) for _, y in pairs]
    theta = solve_ls(A, Y, ridge=ridge)
    W_hat = [[theta[i][j] for i in range(D_IN)] for j in range(D_OUT)]
    b_hat = [theta[D_IN][j] for j in range(D_OUT)]
    return W_hat, b_hat


def extraction_metrics(W_hat, b_hat, W, b, eval_set):
    """rel Frobenius error on W, mean per-row cosine, bias relative error,
    and held-out R^2 on the seeded eval set."""
    num = sum((a - bb) ** 2 for ra, rb in zip(W_hat, W)
              for a, bb in zip(ra, rb))
    den = sum(v * v for row in W for v in row)
    rel_fro = math.sqrt(num / den) if den > 0 else float("nan")
    row_cos = [c for c in (cosine(a, bb) for a, bb in zip(W_hat, W))
               if c == c]
    mean_cos = statistics.fmean(row_cos) if row_cos else float("nan")
    bnum = math.sqrt(sum((a - bb) ** 2 for a, bb in zip(b_hat, b)))
    bden = norm2(b)
    bias_rel = bnum / bden if bden > 0 else float("nan")
    xs, ys = eval_set
    ss_res, ss_tot = 0.0, 0.0
    y_mean = [statistics.fmean(col) for col in zip(*ys)]
    for x, y in zip(xs, ys):
        y_hat = matvec(W_hat, x)
        for j in range(D_OUT):
            y_hat[j] += b_hat[j]
            ss_res += (y[j] - y_hat[j]) ** 2
            ss_tot += (y[j] - y_mean[j]) ** 2
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rel_frobenius": rel_fro, "mean_row_cosine": _fin(mean_cos),
            "bias_rel_err": _fin(bias_rel), "r2_heldout": _fin(r2)}


def _extraction_cell(world, seed, probes, n_requested, extra):
    """One extraction cell: query the (world-configured) shared server with
    probes, fit, score; the isolated reference is the same probes against a
    private undefended body (no co-clients)."""
    server = SharedBodyServer(seed, DEFENDED if world == "defended"
                              else VULNERABLE)
    pairs, n_rejected = query_all(server, probes)
    W, b = server.W, server.b
    eval_set = make_eval_set(seed)
    W_hat, b_hat = fit_body(pairs)
    iso = SharedBodyServer(seed, VULNERABLE)
    iso_pairs, _ = query_all(iso, probes)
    W_iso, b_iso = fit_body(iso_pairs)
    met = extraction_metrics(W_hat, b_hat, W, b, eval_set)
    iso_rel = extraction_metrics(W_iso, b_iso, iso.W, iso.b,
                                 eval_set)["rel_frobenius"]
    cell = {"queries_requested": n_requested,
            "queries_granted": len(pairs),
            "determinate": len(pairs) >= D_IN + 1,
            **met, "isolated_rel_frobenius": iso_rel,
            "fidelity_delta_vs_isolated": rel_fro_delta(met["rel_frobenius"],
                                                        iso_rel),
            "n_rejected": n_rejected,
            "attack_succeeded": met["rel_frobenius"] <= EXTRACT_GATE}
    cell.update(extra)
    return cell


def rel_fro_delta(rel, iso_rel):
    return rel - iso_rel


def arm_crafted_extraction(world, seed, budgets):
    if world == "isolated":
        server = SharedBodyServer(seed, VULNERABLE)
        probes = design_zero_axes(max(budgets) - D_IN - 1, seed)[:max(budgets)]
        pairs, _ = query_all(server, probes)
        W_hat, b_hat = fit_body(pairs)
        met = extraction_metrics(W_hat, b_hat, server.W, server.b,
                                 make_eval_set(seed))
        return [{"budget": max(budgets), "queries_granted": len(pairs),
                 **met, "attack_succeeded": False,
                 "note": "isolated capability reference — extraction works "
                         "solo; no cross-client harm exists here"}]
    cells = []
    full = design_zero_axes(max(budgets) - D_IN - 1, seed)
    for B in budgets:
        cells.append(_extraction_cell(world, seed, full[:B], B,
                                      {"budget": B}))
    return cells


def arm_budget_extraction(world, seed, extract_budget):
    if world == "isolated":
        server = SharedBodyServer(seed, VULNERABLE)
        probes = design_zero_axes(0, seed)[:D_IN + 1]
        pairs, _ = query_all(server, probes)
        W_hat, b_hat = fit_body(pairs)
        met = extraction_metrics(W_hat, b_hat, server.W, server.b,
                                 make_eval_set(seed))
        return [{"budget": extract_budget, "design": "optimal",
                 "queries_granted": len(pairs), **met,
                 "attack_succeeded": False,
                 "note": "isolated capability reference (no limiter on a "
                         "private instance)"}]
    cells = []
    naive = design_random(extract_budget, seed)
    cells.append(_extraction_cell(world, seed, naive, extract_budget,
                                  {"budget": extract_budget,
                                   "design": "naive-random"}))
    optimal = design_zero_axes(0, seed)[:D_IN + 1]
    cells.append(_extraction_cell(world, seed, optimal, len(optimal),
                                  {"budget": extract_budget,
                                   "design": "optimal-zero-axes",
                                   "limiter": world != "vulnerable"}))
    return cells


def _observability_fit(server, probes):
    pairs, n_rejected = query_all(server, probes)
    if len(pairs) < 2:
        return None, n_rejected
    ys = [y for _, y in pairs]
    est = []
    for j in range(D_OUT):
        col = [y[j] for y in ys]
        mu = statistics.fmean(col)
        var = sum((v - mu) ** 2 for v in col) / (len(col) - 1)
        est.append(math.sqrt(var))
    return est, n_rejected


def _observability_metrics(server, probes):
    est, n_rejected = _observability_fit(server, probes)
    true = [norm2(row) for row in server.W]
    if est is None:
        return {"n_probes_granted": 0, "mean_rel_err": float("nan"),
                "max_rel_err": float("nan"), "rank_corr": float("nan")}, 0
    rel = [abs(e - t) / t if t > 0 else float("nan")
           for e, t in zip(est, true)]
    return {"n_probes_granted": len(probes) - n_rejected,
            "mean_rel_err": statistics.fmean(rel),
            "max_rel_err": max(rel),
            "rank_corr": spearman(est, true)}, n_rejected


def arm_neuron_observability(world, seed, probes):
    rng = random.Random(seed + 31)
    probe_xs = [[rng.gauss(0.0, 1.0) for _ in range(D_IN)]
                for _ in range(probes)]
    server = SharedBodyServer(seed, DEFENDED if world == "defended"
                              else VULNERABLE)
    met, n_rejected = _observability_metrics(server, probe_xs)
    iso = SharedBodyServer(seed, VULNERABLE)
    iso_met, _ = _observability_metrics(iso, probe_xs)
    cell = {**met, "n_rejected": n_rejected,
            "isolated_mean_rel_err": _fin(iso_met["mean_rel_err"]),
            "fidelity_delta_vs_isolated":
                _fin(met["mean_rel_err"] - iso_met["mean_rel_err"])
                if met["mean_rel_err"] == met["mean_rel_err"] else None}
    if world == "isolated":
        cell.update({"attack_succeeded": False,
                     "note": "isolated capability reference — scale readout "
                             "works solo; no cross-client harm exists here"})
    else:
        cell["attack_succeeded"] = (
            met["mean_rel_err"] <= OBS_REL_GATE
            and met["rank_corr"] == met["rank_corr"]
            and met["rank_corr"] >= OBS_RANK_GATE)
    return [cell]


def _poison_dataset(seed, n_clients, n_local=16, n_test=64, noise=0.1):
    """Shared-training fixture: body (seed), true head, per-client local
    data (client 0 is SHIFTED-BUT-BENIGN: N(0.4, 1.5^2) — the detector
    false-positive control), and a held-out honest test set. Everything is
    derived from fixed sub-seeds so the baseline and the attacked runs see
    IDENTICAL data (the defended delta is exactly 0)."""
    W, b = make_body(seed)
    rng_h = random.Random(seed + 11)
    h_star = [rng_h.gauss(0.0, 1.0) for _ in range(D_OUT)]

    def samples(rng, n, mu, sigma):
        out = []
        for _ in range(n):
            x = [rng.gauss(mu, sigma) for _ in range(D_IN)]
            y = matvec(W, x)
            for j in range(D_OUT):
                y[j] += b[j]
            t = dot(h_star, y) + rng.gauss(0.0, noise)
            out.append((y, t))
        return out

    clients = []
    for i in range(n_clients):
        rng = random.Random(seed + 100 + i)
        mu, sigma = (0.4, 1.5) if i == 0 else (0.0, 1.0)
        clients.append({"id": f"honest-{i}", "shifted": i == 0,
                        "data": samples(rng, n_local, mu, sigma)})
    rng_t = random.Random(seed + 555)
    test = samples(rng_t, n_test, 0.0, 1.0)
    return {"W": W, "b": b, "h_star": h_star, "clients": clients,
            "test": test}


def _grad(h, data):
    g = [0.0] * D_OUT
    for y, t in data:
        r = dot(h, y) - t
        for j in range(D_OUT):
            g[j] += r * y[j]
    return [v / len(data) for v in g]


def _median_vec(vecs):
    return [statistics.median(col) for col in zip(*vecs)]


def _mse(h, test):
    return statistics.fmean((dot(h, y) - t) ** 2 for y, t in test)


def _train_shared(ds, rounds, attack, defended, params):
    """One shared-training run. Vulnerable: plain mean over all submitted
    updates. Defended: reject any update whose cosine to the round's
    coordinate-wise median is below reject_cosine (the median is computed
    over ALL submissions — no oracle needed with a client majority), then
    clip accepted updates to clip_tau. The attacker (when attack=True)
    submits -POISON_TAU * g_hat with g_hat the unit honest mean gradient.
    Returns (final_mse, counters, privacy_cosine_mean)."""
    h = [0.0] * D_OUT
    clients = ds["clients"]
    n_sub = len(clients) + (1 if attack else 0)
    cnt = {"attacker_rejected": 0, "shifted_rejected": 0, "clipped": 0}
    cos_sum, cos_n = 0.0, 0
    for _ in range(rounds):
        gs = [_grad(h, c["data"]) for c in clients]
        honest_mean = [statistics.fmean(col) for col in zip(*gs)]
        subs = list(gs)
        shifted_idx = {i for i, c in enumerate(clients) if c["shifted"]}
        g_att = None
        if attack:
            ng = norm2(honest_mean)
            g_att = ([-POISON_TAU * v / ng for v in honest_mean]
                     if ng > 1e-12 else [0.0] * D_OUT)
            subs.append(g_att)
        median_dir = _median_vec(subs)
        keep, kept_attacker = [], False
        for i, g in enumerate(subs):
            if defended and cosine(g, median_dir) < params["reject_cosine"]:
                if attack and i == len(subs) - 1:
                    cnt["attacker_rejected"] += 1
                elif i in shifted_idx:
                    cnt["shifted_rejected"] += 1
                continue
            if attack and i == len(subs) - 1:
                kept_attacker = True
            if defended:
                ng = norm2(g)
                if ng > params["clip_tau"]:
                    g = [v * params["clip_tau"] / ng for v in g]
                    cnt["clipped"] += 1
            keep.append(g)
        m = [statistics.fmean(col) for col in zip(*keep)] if keep \
            else [0.0] * D_OUT
        h = [hj - POISON_LR * mj for hj, mj in zip(h, m)]
        if attack and norm2(honest_mean) > 1e-9:
            if kept_attacker and g_att is not None:
                est = [n_sub * mj - gj for mj, gj in zip(m, g_att)]
            else:
                est = m
            c = cosine(est, honest_mean)
            if c == c:
                cos_sum += c
                cos_n += 1
    return _mse(h, ds["test"]), cnt, (cos_sum / cos_n if cos_n else None)


def arm_gradient_poisoning(world, seed, clients, rounds):
    defended = world == "defended"
    if world == "isolated":
        ds_self = _poison_dataset(seed + 900, 1)
        base_self, _, _ = _train_shared(ds_self, rounds, False, False,
                                        PARAMS)
        atk_self, _, _ = _train_shared(ds_self, rounds, True, False, PARAMS)
        return [{"baseline_mse": base_self, "final_mse": base_self,
                 "honest_mse_delta": 0.0, "mse_ratio": 1.0,
                 "detection_rate": 0.0, "false_positive_rate": 0.0,
                 "n_clipped": 0, "privacy_cosine_mean": None,
                 "isolated_honest_mse_delta": 0.0,
                 "cross_client_excess": 0.0,
                 "private_self_harm_delta": atk_self - base_self,
                 "attack_succeeded": False,
                 "note": "isolated control — the payload diverges the "
                         "attacker's own head (self_harm_delta > 0) but no "
                         "honest client exists to harm"}]
    ds = _poison_dataset(seed, clients)
    base_mse, _, _ = _train_shared(ds, rounds, False, defended, PARAMS)
    atk_mse, cnt, priv_cos = _train_shared(ds, rounds, True, defended,
                                           PARAMS)
    delta = atk_mse - base_mse
    return [{"baseline_mse": base_mse, "final_mse": atk_mse,
             "honest_mse_delta": delta,
             "mse_ratio": _fin(atk_mse / base_mse) if base_mse > 0 else None,
             "detection_rate": cnt["attacker_rejected"] / rounds,
             "false_positive_rate": cnt["shifted_rejected"] / rounds,
             "n_clipped": cnt["clipped"],
             "privacy_cosine_mean": _fin(priv_cos),
             "isolated_honest_mse_delta": 0.0,
             "cross_client_excess": delta,
             "attack_succeeded":
                 delta > max(POISON_GATE_ABS, 0.5 * base_mse)}]


BOMB_PROBES = (("bounded-hi", 7.9), ("bounded-lo", -7.9),
               ("huge-1e6", 1e6), ("huge-1e12", 1e12), ("huge-1e300", 1e300),
               ("overflow-1e308", 1e308), ("inf", float("inf")),
               ("nan", float("nan")))


def arm_instability_bomb(world, seed, clients):
    defended = world == "defended"
    n_honest = max(4, clients) if world != "isolated" else 0
    server = SharedBodyServer(seed, DEFENDED if defended else VULNERABLE)
    rng_h = random.Random(seed + 41)
    rng_s = random.Random(seed + 42)
    rng_p = random.Random(seed + 44)
    rng_c = random.Random(seed + 43)
    h_cal = [rng_c.gauss(0.0, 1.0) for _ in range(D_OUT)]
    nc = norm2(h_cal)
    h_cal = [v / nc for v in h_cal]
    stats = {"sum": [0.0] * D_OUT, "n": 0}

    def honest_phase(n, rng, mu, sigma, prefix, tenant_prefix):
        ok, zs = 0, []
        for i in range(n):
            x = [rng.gauss(mu, sigma) for _ in range(D_IN)]
            y, _ = server.query(f"{prefix}-{i}", x,
                                tenant=f"{tenant_prefix}-{i}")
            if y is None:
                continue
            for j in range(D_OUT):
                stats["sum"][j] += y[j]
            stats["n"] += 1
            mean = [s / stats["n"] for s in stats["sum"]]
            z = dot(h_cal, [yj - mj for yj, mj in zip(y, mean)])
            zs.append(z)
            if math.isfinite(z):
                ok += 1
        return ok, zs

    pre_ok, _ = honest_phase(n_honest, rng_h, 0.0, 1.0, "honest-pre",
                             "honest")
    shifted_ok, _ = honest_phase(n_honest, rng_s, 0.4, 1.5, "shifted-pre",
                                 "shifted")
    benign_cost = statistics.fmean(
        c for _, granted, c in server.cost_log if granted) \
        if server.cost_log else PARSE_COST
    benign_peak = server.peak_activation
    parity_cost0 = server.compute_units
    honest_phase(len(BOMB_PROBES), rng_p, 0.0, 1.0, "parity", "parity")
    parity_cost = server.compute_units - parity_cost0
    cost0 = server.compute_units
    accepted = rejected = worthy = worthy_rejected = 0
    for label, v in BOMB_PROBES:
        x = [v] * D_IN
        bad = (not math.isfinite(v)) or abs(v) > PARAMS["norm_cap"]
        worthy += bool(bad)
        y, _ = server.query("atk", x, tenant="atk")
        if y is None:
            rejected += 1
            worthy_rejected += bool(bad)
            continue
        accepted += 1
        for j in range(D_OUT):
            stats["sum"][j] += y[j]
        stats["n"] += 1
    attacker_cost = server.compute_units - cost0
    server_corrupted = any(not math.isfinite(s) for s in stats["sum"])
    post_ok, post_zs = honest_phase(n_honest, rng_h, 0.0, 1.0,
                                    "honest-post", "honest")
    corrupted = len(post_zs) - post_ok
    peak = server.peak_activation
    peak_log = 300.0 if not math.isfinite(peak) else \
        math.log10(max(peak, 1e-300))
    arena = ARENA_CAP if not math.isfinite(peak) else min(peak, ARENA_CAP)
    benign_arena = min(max(benign_peak, 1.0), ARENA_CAP)
    cell = {"probes_accepted": accepted, "probes_rejected": rejected,
            "rejection_worthy": worthy,
            "detection_rate": (worthy_rejected / worthy) if worthy else 0.0,
            "shifted_benign_accepted": shifted_ok,
            "shifted_benign_total": n_honest,
            "server_corrupted": server_corrupted,
            "pre_bomb_honest_ok": pre_ok,
            "honest_outputs_corrupted": corrupted,
            "peak_activation_log10": peak_log,
            "compute_amplification":
                attacker_cost / (len(BOMB_PROBES) * benign_cost),
            "memory_amplification": arena / benign_arena,
            "benign_parity": _fin(parity_cost /
                                  (len(BOMB_PROBES) * benign_cost)),
            "isolated_honest_corrupted": 0,
            "cross_client_excess": corrupted}
    if world == "isolated":
        cell.update({"attack_succeeded": False,
                     "note": "isolated control — the bombs corrupt the "
                             "attacker's own private instance "
                             f"(server_corrupted={server_corrupted}) but no "
                             "honest client exists"})
    else:
        cell["attack_succeeded"] = corrupted > 0
    return [cell]


def arm_collusion(world, seed, colluders, probes, noise_sigma):
    defended = world == "defended"
    server = SharedBodyServer(seed, DEFENDED if defended else VULNERABLE,
                              noise_sigma=noise_sigma)
    probe_xs = design_random(probes, seed, tag=51)
    per_colluder = []
    for k in range(colluders):
        pairs, _ = query_all(server, probe_xs, client=f"atk-{k}",
                             tenant="atk")
        per_colluder.append({tuple(x): y for x, y in pairs})
    eval_set = make_eval_set(seed)

    def rel_err_of(pair_map):
        W_hat, b_hat = fit_body(list(pair_map.items()))
        return extraction_metrics(W_hat, b_hat, server.W, server.b,
                                  eval_set)["rel_frobenius"]

    single = per_colluder[0]
    rel_single = rel_err_of(single)
    averaged = {}
    for x in probe_xs:
        key = tuple(x)
        obs = [cm[key] for cm in per_colluder if key in cm]
        if obs:
            averaged[key] = [statistics.fmean(col) for col in zip(*obs)]
    rel_coll = rel_err_of(averaged)
    gain = (rel_single / rel_coll) if rel_coll > 0 else float("inf")
    cross_ok = True
    for t in ("honest-a", "honest-b"):
        y, _ = server.query(t, [0.0] * D_IN, tenant=t)
        cross_ok = cross_ok and y is not None
    iso = SharedBodyServer(seed, VULNERABLE, noise_sigma=noise_sigma)
    iso_pairs, _ = query_all(iso, probe_xs)
    W_hat, b_hat = fit_body(iso_pairs)
    rel_iso = extraction_metrics(W_hat, b_hat, iso.W, iso.b,
                                 eval_set)["rel_frobenius"]
    cell = {"rel_err_single": rel_single, "rel_err_colluded": rel_coll,
            "gain": _fin(gain),
            "expected_gain_sqrt_c": math.sqrt(colluders),
            "queries_granted": sum(len(cm) for cm in per_colluder),
            "queries_rejected": sum(server.n_rejected.values()),
            "budget_limit": (PARAMS["budget_limit"] if defended else None),
            "cross_tenant_duplicate_accepted": cross_ok,
            "isolated_rel_err_single": rel_iso,
            "fidelity_delta_vs_isolated": rel_coll - rel_iso}
    if world == "isolated":
        cell.update({"attack_succeeded": False,
                     "note": "isolated control — one tenant, one private "
                             "body; there is no one to collude against"})
    else:
        cell["attack_succeeded"] = gain >= COLLUSION_GAIN_GATE
    return [cell]


def _cells_for(arm, world, seed, args):
    if arm == "crafted-extraction":
        return arm_crafted_extraction(world, seed, args.budgets)
    if arm == "neuron-observability":
        return arm_neuron_observability(world, seed, args.probes)
    if arm == "gradient-poisoning":
        return arm_gradient_poisoning(world, seed, args.clients,
                                      args.rounds)
    if arm == "instability-bomb":
        return arm_instability_bomb(world, seed, args.clients)
    if arm == "budget-extraction":
        return arm_budget_extraction(world, seed, args.extract_budget)
    return arm_collusion(world, seed, args.colluders, args.collusion_probes,
                         args.noise_sigma)


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
                    help="attack arm(s) to run (default: all six)")
    ap.add_argument("--world", choices=(*WORLDS, "both"), default="both",
                    help="vulnerable (no checks) / defended (all six "
                         "mitigations — the negative control) / both "
                         "(default: the contrast IS the result); the "
                         "isolated attacker-alone control always runs")
    ap.add_argument("--clients", type=int, default=4,
                    help="toy: honest clients sharing the body (client 0 is "
                         "the shifted-but-benign FP control)")
    ap.add_argument("--colluders", type=int, default=4,
                    help="toy: sybil clients under one tenant (collusion)")
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[4, 8, 9, 16],
                    help="crafted-extraction query budgets (identifiability "
                         f"threshold: {D_IN + 1})")
    ap.add_argument("--extract-budget", type=int, default=D_IN,
                    help="budget-extraction fixed budget (defended limiter "
                         f"caps at {D_IN})")
    ap.add_argument("--probes", type=int, default=96,
                    help="neuron-observability probe count")
    ap.add_argument("--collusion-probes", type=int, default=16,
                    help="collusion probes per colluder")
    ap.add_argument("--rounds", type=int, default=40,
                    help="gradient-poisoning shared-training rounds")
    ap.add_argument("--noise-sigma", type=float, default=0.8,
                    help="collusion per-response observation noise")
    ap.add_argument("--probe-file", default=None,
                    help="REAL integration point: captured client-stream "
                         f"journal ({PROBE_SCHEMA}) replayed offline "
                         "through the defended policy stack")
    return ap


def _threat_model_doc():
    return threat_model.skeleton(
        summary="malicious authenticated client of a shared split "
                "learning/inference service (S1/S2/S3): controls only its "
                "own inputs and updates but shares the cloud body with "
                "honest clients — crafted-activation body extraction, "
                "sensitive-neuron observability, gradient poisoning, "
                "numerical-instability bombs, budget-limited extraction, "
                "and sybil collusion",
        scope={"use_case": "split-training", "deployment": "multi-tenant",
               "in_scope": ["shared cloud body slice (seam activations "
                            "S1 fwd, seam gradients S2 bwd, body weights "
                            "S3 via functional extraction)",
                            "shared-head training over honest co-client "
                            "updates",
                            "query/update intake policy stack (validation, "
                            "budgets, anomaly detection)",
                            "captured client-stream journals (offline "
                            "policy replay)"],
               "out_of_scope": ["nonlinear bodies, attention, LayerNorm "
                                "(linear 12x8 slice only)",
                                "cross-tenant sybil attribution (colluders "
                                "share one tenant)",
                                "adaptive attackers that poison the "
                                "detector's median reference",
                                "cryptographic privacy layers (DP / secure "
                                "aggregation), multi-epoch persistence"]},
        actor={"role": "client", "intent": "malicious",
               "persistence": "cross-session",
               "vantage": ["authenticated split-learning client / API user "
                           "on the shared service"],
               "authorization": "valid client credentials; no server-side "
                                "privileges",
               "knowledge": ["own query responses and the shared head "
                             "state", "the split architecture and body "
                             "dims (not the weights)",
                             "honest-gradient direction from shared-head "
                             "drift (worst-case oracle, poisoning arm)"],
               "capabilities": ["arbitrary crafted activation uploads "
                                "(S1)", "bounded malicious gradient "
                                "updates (S2)", "non-finite / extreme-"
                                "magnitude inputs", "budget-aware query "
                                "design"],
               "collusion": ["c sybil clients registered under one tenant "
                             "(collusion arm)"]},
        target={"assets": ["shared body weights W,b (S3)",
                           "honest clients' updates and query distribution "
                           "(S1/S2)", "shared-head training integrity",
                           "serving availability for honest co-clients"],
                "components": ["cloud body slice", "shared head trainer / "
                               "update aggregator", "query intake and "
                               "validation", "rate/budget limiter",
                               "shared calibration state"],
                "trust_boundaries": ["client<->cloud seam", "tenant "
                                     "boundary between co-clients"],
                "entry_points": ["seam activation upload (fwd)",
                                 "gradient/update upload (bwd)",
                                 "query endpoint"]},
        scenario={"preconditions": ["multiple tenants share one cloud "
                                    "body / head", "the attacker holds "
                                    "valid client credentials",
                                    "query responses and shared-head "
                                    "updates are visible to all clients"],
                  "lifecycle_stage": ["training", "inference"],
                  "security_properties": ["confidentiality", "integrity",
                                          "availability"],
                  "attack_steps": list(ARMS),
                  "technical_impact": ["functional theft of the shared "
                                       "body slice",
                                       "honest-client quality loss via "
                                       "poisoned shared training",
                                       "cross-client service corruption "
                                       "via poisoned shared state",
                                       "limiter evasion via sybil "
                                       "collusion"],
                  "business_impact": ["model-IP loss", "tenant SLA "
                                      "violations", "integrity loss for "
                                      "all co-tenants"]},
        controls={"assumed": ["clients authenticate per tenant"],
                  "tested": list(DEFENSES),
                  "detective": ["per-tenant duplicate-probe registry",
                                "median-cosine update anomaly detection "
                                "with a shifted-benign FP budget"],
                  "response": ["per-tenant budget revocation"],
                  "residual_risk": "shared training leaks the honest "
                                   "aggregate gradient direction in both "
                                   "worlds (clipping protects integrity, "
                                   "not privacy)",
                  "status": "open"},
        mappings={"owasp_llm_2026": ["LLM02", "LLM05", "LLM06"]},
        evaluation={"success_metrics": [
                        "extraction fidelity per query budget "
                        "(rel-Frobenius / cosine / held-out R^2)",
                        "honest-client MSE delta vs the no-attack "
                        "baseline (shared-model integrity loss)",
                        "cross-client harm excess over the isolated "
                        "attacker-alone control",
                        "compute/memory amplification vs the benign-parity "
                        "client",
                        "detection rate; false positives on the "
                        "shifted-but-benign client (must be 0)",
                        "collusion fidelity gain vs sqrt(C)"],
                    "baseline": "defended world (all six mitigations on) "
                                "+ isolated attacker-alone control + "
                                "shifted-benign and benign-parity load "
                                "controls",
                    "budget": {"queries_per_arm": "<= 96 toy probes",
                               "training_rounds": 40,
                               "colluders": 4,
                               "poison_update_norm": POISON_TAU},
                    "limitations": ["linear body slice — extraction/"
                                    "observability results are upper "
                                    "bounds",
                                    "worst-case oracle for the poisoning "
                                    "direction",
                                    "single-tenant sybil assumption",
                                    "abstract compute/memory units, not "
                                    "GPU-calibrated"],
                    "coverage_status": "covered"})


def _make_artifact(args, source, extra_config=None):
    config = {"attack": EXPERIMENT_ID, "mode": getattr(args, "mode", None),
              "source": source, "arms": args.arms, "world": args.world,
              "dims": [D_OUT, D_IN], "clients": args.clients,
              "colluders": args.colluders, "budgets": args.budgets,
              "extract_budget": args.extract_budget, "probes": args.probes,
              "collusion_probes": args.collusion_probes,
              "rounds": args.rounds, "noise_sigma": args.noise_sigma,
              "defenses": {"vulnerable": VULNERABLE, "defended": DEFENDED},
              "defense_params": dict(PARAMS), "seed": args.seed}
    if extra_config:
        config.update(extra_config)
    return artifacts.make_artifact(
        SCHEMA_VERSION, config, _threat_model_doc(),
        interpretation="EXPECTED SECURE BEHAVIOR: the defended world "
                       "defeats every arm (attack_succeeded=false "
                       "everywhere): extraction is capped one probe below "
                       "identifiability, poisoned updates are rejected "
                       "with zero false positives on the shifted-benign "
                       "client, non-finite/oversized bombs are rejected "
                       "before they touch shared state, and sybils under "
                       "one tenant gain nothing over a single client "
                       "(gain == 1). The isolated control shows no "
                       "cross-client harm by construction. The vulnerable "
                       "world must show the opposite — exact extraction at "
                       "budget >= D_IN+1, honest-MSE explosion, corrupted "
                       "honest outputs, ~sqrt(C) collusion gain — else "
                       "the control is vacuous.")


def run_toy(args):
    if args.quick:
        args.seeds = args.seeds[:1]
        args.budgets = sorted(set(args.budgets[:1]) | {D_IN + 1})
        args.probes = min(args.probes, 64)
        args.collusion_probes = min(args.collusion_probes, 12)
        args.rounds = min(args.rounds, 24)
    worlds = WORLDS if args.world == "both" else (args.world,)
    out = _make_artifact(args, "toy", {"toy": True, "seeds": args.seeds})
    per_cell = {}
    for seed in args.seeds:
        world_seed = args.seed + 1000 * seed
        for arm in args.arms:
            for world in (*worlds, "isolated"):
                cell_id = {"arm": arm, "world": world, "seed": seed}
                try:
                    cells = _cells_for(arm, world, world_seed, args)
                except Exception as e:
                    journal_error(args.output, EXPERIMENT_ID, cell_id, e)
                    continue
                for cell in cells:
                    rec = {"experiment": EXPERIMENT_ID, **cell_id,
                           "role": ("attack" if world == "vulnerable"
                                    else "negative-control"
                                    if world == "defended"
                                    else "isolated-control"),
                           **cell}
                    artifacts.append_jsonl(args.output, rec)
                    out["results"].append(rec)
                    per_cell.setdefault((arm, world), []).append(rec)
                    sub = " ".join(f"{k}={rec[k]}" for k in
                                   ("budget", "design") if k in rec)
                    head = _HEADLINE[arm]
                    hv = rec.get(head)
                    hv_s = (f"{hv:.6g}" if isinstance(hv, float)
                            else str(hv))
                    print(f"[{EXPERIMENT_ID}] {arm} {world} seed={seed}"
                          + (f" {sub}" if sub else "")
                          + f": attack_succeeded={rec['attack_succeeded']}"
                          + f" {head}={hv_s}")
    for (arm, world), recs in sorted(per_cell.items()):
        row = {"experiment": EXPERIMENT_ID, "arm": arm, "world": world,
               "n_seeds": len({r["seed"] for r in recs}),
               "n_cells": len(recs),
               "attack_success_rate": round(statistics.fmean(
                   float(r["attack_succeeded"]) for r in recs), 6)}
        head = _HEADLINE[arm]
        vals = [r[head] for r in recs
                if isinstance(r.get(head), (int, float))]
        if vals:
            row[f"{head}_mean"] = round(statistics.fmean(vals), 6)
        out["summary"].append(row)
    artifacts.write_artifact(args.output, out,
                         evidence=_make_evidence(args))
    return 0


def read_probe_file(path):
    """Parse a client-stream journal (schema llm_attacker."
    "malicious_client.probe.v1): one JSON object per line,
    {"client": str, "kind": "query"|"update", "x": [D_IN floats] for
    queries | "g": [D_OUT floats] for updates, "tenant": str optional
    (default: client), "ts": float optional}. Schema drift is a
    descriptive ValueError naming file and line; an unreadable file is a
    SystemExit naming the path."""
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError as e:
        raise SystemExit(f"[{EXPERIMENT_ID}] --probe-file: cannot read "
                         f"{path}: {e}")
    records = []
    for lineno, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {e}")
        where = f"{path}:{lineno}"
        if not isinstance(rec, dict):
            raise ValueError(f"{where}: record must be an object, got "
                             f"{type(rec).__name__}")
        unknown = sorted(set(rec) - {"client", "tenant", "kind", "x", "g",
                                     "ts"})
        if unknown:
            raise ValueError(f"{where}: unknown keys {unknown} (schema "
                             f"{PROBE_SCHEMA})")
        if not isinstance(rec.get("client"), str) or not rec["client"]:
            raise ValueError(f"{where}: 'client' must be a non-empty str")
        if rec.get("tenant") is not None and not isinstance(rec["tenant"],
                                                            str):
            raise ValueError(f"{where}: 'tenant' must be a str")
        kind = rec.get("kind")
        if kind not in ("query", "update"):
            raise ValueError(f"{where}: 'kind' must be 'query'|'update', "
                             f"got {kind!r}")
        vec_key, dim = ("x", D_IN) if kind == "query" else ("g", D_OUT)
        vec = rec.get(vec_key)
        if not isinstance(vec, list) or len(vec) != dim or \
                any(isinstance(v, bool) or not isinstance(v, (int, float))
                    for v in vec):
            raise ValueError(f"{where}: '{vec_key}' must be a list of "
                             f"{dim} numbers for kind={kind!r}")
        if rec.get("ts") is not None and \
                (isinstance(rec["ts"], bool)
                 or not isinstance(rec["ts"], (int, float))):
            raise ValueError(f"{where}: 'ts' must be a number")
        records.append({"client": rec["client"],
                        "tenant": rec.get("tenant") or rec["client"],
                        "kind": kind, vec_key: [float(v) for v in vec],
                        "ts": rec.get("ts")})
    if not records:
        raise ValueError(f"{path}: no records found (schema "
                         f"{PROBE_SCHEMA})")
    return records


def run_replay(args):
    """--probe-file FILE: replay a captured client stream through the
    defended policy stack OFFLINE (per-tenant budgets, duplicate-probe
    registry, input validation, update clipping, median-cosine anomaly
    detection with the stream's own median update as reference). No
    ground truth -> no attack gate; the record is a policy-replay report.
    The live driver (enforcing this stack on a real split server) is the
    documented integration point."""
    records = read_probe_file(args.probe_file)
    p = dict(PARAMS)
    budget_used, seen = {}, {}
    per_tenant = {}
    reasons = {}
    n_clipped = n_flagged = 0
    updates = [r["g"] for r in records if r["kind"] == "update"]
    median_dir = _median_vec(updates) if updates else None

    def bump(tenant, key):
        per_tenant.setdefault(tenant, {"records": 0, "accepted": 0,
                                       "rejected": 0, "budget_used": 0})
        per_tenant[tenant][key] += 1

    for r in records:
        t = r["tenant"]
        bump(t, "records")
        if r["kind"] == "query":
            x = r["x"]
            if any(not math.isfinite(v) for v in x) or \
                    max(abs(v) for v in x) > p["norm_cap"]:
                reasons["invalid-input"] = reasons.get("invalid-input", 0) + 1
                bump(t, "rejected")
                continue
            if tuple(x) in seen.get(t, ()):
                reasons["duplicate-probe"] = reasons.get("duplicate-probe",
                                                         0) + 1
                bump(t, "rejected")
                continue
            if budget_used.get(t, 0) >= p["budget_limit"]:
                reasons["budget-exceeded"] = reasons.get("budget-exceeded",
                                                         0) + 1
                bump(t, "rejected")
                continue
            seen.setdefault(t, set()).add(tuple(x))
            budget_used[t] = budget_used.get(t, 0) + 1
            per_tenant[t]["budget_used"] += 1
            bump(t, "accepted")
        else:
            g = r["g"]
            if any(not math.isfinite(v) for v in g):
                reasons["invalid-update"] = reasons.get("invalid-update",
                                                        0) + 1
                bump(t, "rejected")
                continue
            if median_dir is not None and len(updates) >= 3 and \
                    cosine(g, median_dir) < p["reject_cosine"]:
                n_flagged += 1
                reasons["anomalous-update"] = reasons.get("anomalous-update",
                                                          0) + 1
                bump(t, "rejected")
                continue
            if norm2(g) > p["clip_tau"]:
                n_clipped += 1
            bump(t, "accepted")
    n_acc = sum(v["accepted"] for v in per_tenant.values())
    n_rej = sum(v["rejected"] for v in per_tenant.values())
    out = _make_artifact(args, "probe-file",
                         {"probe_file": args.probe_file})
    rec = {"experiment": EXPERIMENT_ID, "arm": "policy-replay",
           "world": "real", "role": "policy-replay",
           "probe_file": args.probe_file, "probe_schema": PROBE_SCHEMA,
           "n_records": len(records),
           "n_queries": sum(1 for r in records if r["kind"] == "query"),
           "n_updates": len(updates),
           "n_accepted": n_acc, "n_rejected": n_rej,
           "rejection_reasons": reasons, "per_tenant": per_tenant,
           "updates_clipped": n_clipped, "updates_flagged": n_flagged,
           "attack_succeeded": False,
           "note": "offline policy replay of a captured client stream "
                   "under the defended stack — no ground truth, no attack "
                   "gate; the live enforcement driver is the integration "
                   "point (see module docstring)"}
    artifacts.append_jsonl(args.output, rec)
    out["results"].append(rec)
    out["summary"].append({"experiment": EXPERIMENT_ID,
                           "arm": "policy-replay", "world": "real",
                           "n_records": len(records), "n_rejected": n_rej})
    artifacts.write_artifact(args.output, out,
                         evidence=_make_evidence(args))
    print(f"[{EXPERIMENT_ID}] policy-replay {args.probe_file}: "
          f"{len(records)} records, {n_rej} rejected {reasons}, "
          f"{n_flagged} updates flagged, {n_clipped} would be clipped")
    return 0


def run(args):
    if getattr(args, "probe_file", None):
        return run_replay(args)
    if not args.toy:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] pick an input path: --toy (simulated "
            "vulnerable/defended worlds + isolated control) or "
            f"--probe-file FILE (offline policy replay, schema "
            f"{PROBE_SCHEMA}). Integration point for the live driver: "
            "wrap a real split server's query/update intake with "
            "SharedBodyServer's policy stack (input validation, "
            "duplicate-probe registry, per-tenant budgets, update "
            "clipping, median-cosine anomaly detection) and drive the "
            "same arm functions against real body responses (see module "
            "docstring).")
    return run_toy(args)


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("malicious-client shared-body simulator (pure python):")

    M = [[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]]
    x_true = [[1.0], [-2.0], [0.5]]
    B = [[sum(row[k] * x_true[k][0] for k in range(3))] for row in M]
    X = gauss_solve(M, B)
    check("gauss_solve recovers a known 3x3 solution",
          max(abs(X[i][0] - x_true[i][0]) for i in range(3)) < 1e-9)
    try:
        gauss_solve([[1.0, 2.0], [2.0, 4.0]], [[1.0], [2.0]])
        check("gauss_solve rejects a singular matrix", False)
    except ValueError:
        check("gauss_solve rejects a singular matrix", True)
    check("cosine anchors", cosine([1, 0], [1, 0]) == 1.0
          and abs(cosine([1, 0], [0, 1])) < 1e-12)

    seed = 7
    W, b = make_body(seed)
    probes = design_zero_axes(0, seed)[:D_IN + 1]
    ys = [matvec(W, x) for x in probes]
    for y in ys:
        for j in range(D_OUT):
            y[j] += b[j]
    W_hat, b_hat = fit_body(list(zip(probes, ys)))
    met = extraction_metrics(W_hat, b_hat, W, b, make_eval_set(seed))
    check("least squares: zero+axes design extracts a frozen body exactly",
          met["rel_frobenius"] < 1e-9 and abs(met["r2_heldout"] - 1.0)
          < 1e-9)
    check("defended budget cap sits below identifiability",
          PARAMS["budget_limit"] == D_IN < D_IN + 1)

    cv = arm_crafted_extraction("vulnerable", seed, [4, 9])
    cd = arm_crafted_extraction("defended", seed, [4, 9])
    ci = arm_crafted_extraction("isolated", seed, [4, 9])
    check("crafted-extraction: vulnerable exact at budget 9, "
          "underdetermined at 4",
          cv[1]["rel_frobenius"] < 1e-9 and cv[1]["attack_succeeded"]
          and not cv[0]["attack_succeeded"] and not cv[0]["determinate"])
    check("crafted-extraction: defended capped below identifiability at "
          "every budget",
          all(not c["attack_succeeded"] for c in cd)
          and all(c["queries_granted"] <= PARAMS["budget_limit"]
                  for c in cd)
          and cd[1]["rel_frobenius"] > 0.2)
    check("crafted-extraction: isolated control extracts its private body",
          ci[0]["rel_frobenius"] < 1e-9
          and not ci[0]["attack_succeeded"])

    bv = arm_budget_extraction("vulnerable", seed, D_IN)
    bd = arm_budget_extraction("defended", seed, D_IN)
    check("budget-extraction: vulnerable overspends to exactness; naive "
          "design fails at the same budget",
          bv[1]["attack_succeeded"] and bv[1]["rel_frobenius"] < 1e-9
          and not bv[0]["attack_succeeded"])
    check("budget-extraction: defended limiter holds both designs off",
          all(not c["attack_succeeded"] for c in bd)
          and min(c["rel_frobenius"] for c in bd) > 0.2)

    ov = arm_neuron_observability("vulnerable", seed, 96)[0]
    od = arm_neuron_observability("defended", seed, 96)[0]
    check("neuron-observability: vulnerable reads per-neuron scales",
          ov["attack_succeeded"] and ov["mean_rel_err"] <= OBS_REL_GATE
          and ov["rank_corr"] >= OBS_RANK_GATE)
    check("neuron-observability: defended cap (8 probes) defeats the "
          "readout",
          not od["attack_succeeded"]
          and od["n_probes_granted"] == PARAMS["budget_limit"]
          and od["mean_rel_err"] > OBS_REL_GATE)

    pv = arm_gradient_poisoning("vulnerable", seed, 4, 24)[0]
    pd_ = arm_gradient_poisoning("defended", seed, 4, 24)[0]
    pi = arm_gradient_poisoning("isolated", seed, 4, 24)[0]
    check("gradient-poisoning: vulnerable explodes the honest MSE",
          pv["attack_succeeded"]
          and pv["honest_mse_delta"] > POISON_GATE_ABS
          and pv["detection_rate"] == 0.0)
    check("gradient-poisoning: defended rejects every attacker round with "
          "zero FP and exactly zero integrity loss",
          not pd_["attack_succeeded"] and pd_["honest_mse_delta"] == 0.0
          and pd_["detection_rate"] == 1.0
          and pd_["false_positive_rate"] == 0.0)
    check("gradient-poisoning: isolated control harms no honest client "
          "(but the payload is real)",
          not pi["attack_succeeded"] and pi["honest_mse_delta"] == 0.0
          and pi["private_self_harm_delta"] > 0.0)

    iv = arm_instability_bomb("vulnerable", seed, 4)[0]
    id_ = arm_instability_bomb("defended", seed, 4)[0]
    ii = arm_instability_bomb("isolated", seed, 4)[0]
    check("instability-bomb: vulnerable poisons the shared calibration "
          "state and corrupts every honest output",
          iv["attack_succeeded"] and iv["server_corrupted"]
          and iv["honest_outputs_corrupted"] == iv["pre_bomb_honest_ok"]
          and iv["compute_amplification"] > 10.0
          and iv["benign_parity"] < 2.0)
    check("instability-bomb: defended rejects all rejection-worthy probes, "
          "no corruption, shifted-benign untouched",
          not id_["attack_succeeded"] and id_["detection_rate"] == 1.0
          and id_["honest_outputs_corrupted"] == 0
          and id_["shifted_benign_accepted"]
          == id_["shifted_benign_total"] > 0)
    check("instability-bomb: isolated control corrupts only the attacker's "
          "own instance",
          not ii["attack_succeeded"] and ii["server_corrupted"]
          and ii["honest_outputs_corrupted"] == 0)

    kv = arm_collusion("vulnerable", seed, 4, 12, 0.8)[0]
    kd = arm_collusion("defended", seed, 4, 12, 0.8)[0]
    check("collusion: vulnerable sybils gain ~sqrt(C) over a single "
          "client",
          kv["attack_succeeded"] and kv["gain"] >= COLLUSION_GAIN_GATE
          and kv["rel_err_colluded"] < kv["rel_err_single"])
    check("collusion: defended per-tenant cap makes sybils useless "
          "(gain == 1) while cross-tenant honest duplicates stay accepted",
          not kd["attack_succeeded"] and kd["gain"] == 1.0
          and kd["cross_tenant_duplicate_accepted"]
          and kd["queries_granted"] == PARAMS["budget_limit"])

    check("threat model is a valid v2 document",
          threat_model.validate(_threat_model_doc()) is not None)

    check("determinism: same seed, same arm metrics",
          arm_crafted_extraction("defended", seed, [4, 9])
          == arm_crafted_extraction("defended", seed, [4, 9])
          and arm_collusion("vulnerable", seed, 4, 12, 0.8)
          == arm_collusion("vulnerable", seed, 4, 12, 0.8))

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
