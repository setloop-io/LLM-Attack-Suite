#!/usr/bin/env python3
"""Citation scope (reference audit): the rmia arm implements a restricted
OUT-reference relative-likelihood quantile, with gamma fixed to one and
a known non-member population. It omits the published offline calibration;
toy sanity checks do not reproduce RMIA's reported superiority. The cited
title is Low-Cost High-Power Membership Inference Attacks (ICML 2024).
See docs/reference-audit-2026-09-22.md.

lira-membership (training mode) — LIRA-style likelihood-ratio membership
inference on boundary-feature statistics.

Carlini et al. 2022 ("Membership Inference Attacks From First Principles")
adapted to the split-boundary setting. For each target example the attacker
models a scalar statistic s (per-example loss, or logit-scaled confidence of
the true label — see --statistic) as Gaussian under the IN hypothesis (the
example was in the training set) and the OUT hypothesis, and scores the log
likelihood ratio

    Λ(s) = log N(s; μ_in, σ_in²) − log N(s; μ_out, σ_out²)

Reference distributions stand in for LIRA's shadow models (shadow BOUNDARY
captures instead of shadow networks). Two estimation modes, chosen per
record:

  per-example  the record carries "ref_in" / "ref_out" lists — the example's
               own statistic under shadow runs trained with/without it (the
               online-LIRA per-example Gaussian model, fed offline);
  global       the record carries only its own statistic; (μ, σ) come from
               the reference pools with LEAVE-ONE-OUT for the scored record
               (a scored IN record is excluded from its own μ_in estimate) —
               the global-variance LIRA variant.

Metrics: rank-based (Mann–Whitney) AUC = P(Λ_in > Λ_out) with tie-averaged
ranks (documented choice — no normality assumption on the score); TPR at
FPR ≤ 1% and ≤ 0.1% by a conservative threshold sweep (max TPR over
thresholds whose FPR is ≤ the target, never interpolated past the
achievable grid; when 1/n_out > target the cell is flagged
"resolvable": false — the target FPR is below the sample resolution); and
an attacker.gate verdict with recovery = AUC vs the 0.5 chance control
(membership analogue of the label-free majority control; AUC is not
binomial, so the Wilson bound is a scoring convention — documented, n =
min(n_in, n_out)).

RMIA scoring (--scoring rmia, Zarifzadeh et al. 2024, "Low-Cost
High-Power Membership Inference Attacks") motivates a restricted
relative-likelihood alternative. Where LIRA scores the ABSOLUTE Gaussian likelihood ratio of
the example's statistic, RMIA scores it RELATIVE to a population of
reference examples, which cancels the per-example difficulty that
dominates the absolute score on heavy-tailed difficulty distributions.
The implementation composes a pairwise relative likelihood with a
quantile score at gamma = 1. It uses the OUT mean directly (equivalent to
a = 1 in the paper's Eq. 10), without fitting the offline calibration:

    rho(t)   = Pr(t | theta) / Pr(t)
    Pr(t|theta) = exp(-s_t)                    (the record's statistic s_t
                                                read as the per-example
                                                negative log-likelihood
                                                under the target model)
    Pr(t)       = mean_i exp(-ref_out_i)       (empirical mean likelihood
                                                over the example's OUT
                                                reference models — the
                                                paper's offline Pr_OUT)
    RMIA(x)     = (1/|Z|) sum_{z in Z} 1[ log rho(x) >= log rho(z) ]

i.e. the QUANTILE of the example's relative likelihood rho(x) in the
reference population Z (the split="out" records, the paper's "all
non-members of the target model, excluding the query itself"). Higher =
more likely a member. Offline RMIA needs ONLY ref_out (no IN shadow
models) — its documented low-cost regime — and reuses the existing record
schema unchanged ({example_id, split, statistic[, ref_in, ref_out]}); the
reference population is the OUT records, so no schema extension is
required. The tests exercise this restricted score on planted populations; they
do not reproduce the paper's benchmark or establish superiority over LiRA.

Modes:
  --toy       generates two worlds: "separable" (members shifted by
              --toy-shift σ) and "null" (shift exactly 0 — the honesty arm:
              AUC must land at ≈0.5 and the gate must PASS, or the
              machinery is broken). With --scoring rmia the toy records
              additionally carry reference (OUT) NLLs (--toy-ref draws, no
              membership shift) so the relative score is well defined, and
              each cell also reports the LIRA-global AUC on the SAME
              records for the rmia >= lira sanity comparison.
  --records F.jsonl ...
              offline scoring of {"example_id", "split": "in"|"out",
              "statistic": float [, "ref_in": [...], "ref_out": [...]]}.
              --scoring rmia requires ref_out on every record.
  --online    trains the internal shadow-model ensemble
              (attacker/shadow_train.py: --n-shadows leave-one-out
              shadows on a synthetic corpus, --shadow-steps Adam updates
              each) and scores the generated records through the SAME
              scoring path as --records; requires torch. Deployed-defender
              reference models must still be captured externally and fed
              via --records.

Usage:
    python -m attacker --mode training --attack lira-membership --help
    python -m attacker --mode training --attack lira-membership --toy \
        --quick --output /tmp/lira.json
    python -m attacker --mode training --attack lira-membership --toy \
        --scoring rmia --quick --output /tmp/rmia.json
    python -m attacker --mode training --attack lira-membership \
        --records run.jsonl --scoring rmia --output rmia.json
"""

import argparse
import json
import math
import random

from .. import artifacts, gate
from .common import add_common_args

EXPERIMENT_ID = "lira_membership"
MODES = ("training",)
REQUIRES_LABELS = False
DESCRIPTION = ("LIRA-style (Carlini et al. 2022) per-example Gaussian "
               "likelihood-ratio membership inference on boundary-feature "
               "statistics; AUC + TPR@low-FPR + gate verdict, with a "
               "null-world honesty arm")

SCHEMA = "dtraining.attacker.lira_membership.v1"
VAR_FLOOR = 1e-6
FPR_TARGETS = (0.01, 0.001)


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--scoring", choices=("lira", "rmia"), default="lira",
                    help="membership scoring: lira (default, the existing "
                         "per-example Gaussian likelihood ratio) or rmia "
                         "(Zarifzadeh et al. 2024, offline relative "
                         "likelihood-ratio quantile over a reference "
                         "population; requires ref_out per record)")
    ap.add_argument("--records", nargs="+", default=None, metavar="JSONL",
                    help="offline records: {example_id, split, statistic} "
                         "per line (optional ref_in/ref_out shadow lists; "
                         "ref_out required for --scoring rmia)")
    ap.add_argument("--statistic", choices=("raw", "logit"), default="raw",
                    help="statistic transform: raw (a loss-like statistic) "
                         "or logit (log(p/(1-p)) for confidences in (0,1))")
    ap.add_argument("--worlds", choices=("both", "separable", "null"),
                    default="both", help="toy worlds to generate")
    ap.add_argument("--toy-n", type=int, default=100000,
                    help="toy examples per split (default resolves the "
                         "1.0 pp gate at the 0.5 control: the Wilson "
                         "floor is ~0.4 pp, so a true null PASSes with "
                         "margin)")
    ap.add_argument("--toy-shift", type=float, default=2.0,
                    help="separable world member shift, in noise-sigma "
                         "units (null world is always exactly 0)")
    ap.add_argument("--toy-ref", type=int, default=8,
                    help="reference (OUT) NLL draws per toy record "
                         "(--scoring rmia only; 0 = none)")
    ap.add_argument("--online", action="store_true",
                    help="online mode: train the internal shadow-model "
                         "ensemble (attacker/shadow_train.py) on its "
                         "synthetic corpus and score the generated "
                         "records through the --records path; requires "
                         "torch. Deployed-defender reference models must "
                         "still be supplied via --records.")
    ap.add_argument("--n-shadows", type=int, default=4,
                    help="--online: leave-one-out shadow models (>=2)")
    ap.add_argument("--shadow-steps", type=int, default=200,
                    help="--online: full-batch Adam updates per model")
    ap.add_argument("--evidence", action="store_true",
                    help="seal the run with an attacker.evidence chain "
                         "(<output>.evidence.json[l]); requires --output")
    return ap


def _log_gauss(x, mu, var):
    """log N(x; mu, var) with a variance floor (degenerate references)."""
    var = max(var, VAR_FLOOR)
    return -0.5 * (x - mu) ** 2 / var - 0.5 * math.log(2.0 * math.pi * var)


def _transform(x, statistic):
    if statistic == "logit":
        p = min(max(x, 1e-6), 1.0 - 1e-6)
        return math.log(p / (1.0 - p))
    return float(x)


def _mean_var(values):
    n = len(values)
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / max(1, n - 1)
    return mu, max(var, VAR_FLOOR)


def load_records(paths):
    """Read + validate the offline JSONL schema; descriptive errors."""
    records = []
    for path in paths:
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise SystemExit(f"[{EXPERIMENT_ID}] {path}:{lineno}: "
                                     f"not JSON: {e}")
                for field in ("example_id", "split", "statistic"):
                    if field not in rec:
                        raise SystemExit(
                            f"[{EXPERIMENT_ID}] {path}:{lineno}: missing "
                            f"field {field!r} (schema: {{example_id, split, "
                            "statistic[, ref_in, ref_out]}})")
                if rec["split"] not in ("in", "out"):
                    raise SystemExit(f"[{EXPERIMENT_ID}] {path}:{lineno}: "
                                     f"split must be \"in\"|\"out\", got "
                                     f"{rec['split']!r}")
                if isinstance(rec["statistic"], bool) or not isinstance(
                        rec["statistic"], (int, float)):
                    raise SystemExit(f"[{EXPERIMENT_ID}] {path}:{lineno}: "
                                     "statistic must be a number")
                for key in ("ref_in", "ref_out"):
                    if key in rec:
                        vals = rec[key]
                        if not isinstance(vals, list) or len(vals) < 2 \
                                or any(isinstance(v, bool) or not isinstance(
                                    v, (int, float)) for v in vals):
                            raise SystemExit(
                                f"[{EXPERIMENT_ID}] {path}:{lineno}: {key} "
                                "must be a list of >=2 numbers (shadow "
                                "statistics for the per-example model)")
                records.append(rec)
    return records


def score_records(records, statistic="raw"):
    """Score every record with the Gaussian likelihood ratio.

    Returns one dict per record: example_id, split, statistic (post
    transform), llr, model ("per_example"|"global_loo"), and the (mu, sd)
    used under each hypothesis (auditable). Needs >=2 records per split."""
    ins = [_transform(r["statistic"], statistic)
           for r in records if r["split"] == "in"]
    outs = [_transform(r["statistic"], statistic)
            for r in records if r["split"] == "out"]
    if len(ins) < 2 or len(outs) < 2:
        raise SystemExit(f"[{EXPERIMENT_ID}] need >=2 IN and >=2 OUT "
                         f"records (got {len(ins)}/{len(outs)})")
    sum_in = sum(ins)
    sq_in = sum(v * v for v in ins)
    sum_out = sum(outs)
    sq_out = sum(v * v for v in outs)
    pool_in = _mean_var(ins)
    pool_out = _mean_var(outs)

    def loo(total, sq, x, n_pool):
        n = n_pool - 1
        mu = (total - x) / n
        var = (sq - x * x) / n - mu * mu
        return mu, max(var, VAR_FLOOR)

    scored = []
    for rec in records:
        s = _transform(rec["statistic"], statistic)
        if "ref_in" in rec and "ref_out" in rec:
            mu_in, var_in = _mean_var([_transform(v, statistic)
                                       for v in rec["ref_in"]])
            mu_out, var_out = _mean_var([_transform(v, statistic)
                                         for v in rec["ref_out"]])
            model = "per_example"
        elif rec["split"] == "in":
            mu_in, var_in = loo(sum_in, sq_in, s, len(ins))
            mu_out, var_out = pool_out
            model = "global_loo"
        else:
            mu_in, var_in = pool_in
            mu_out, var_out = loo(sum_out, sq_out, s, len(outs))
            model = "global_loo"
        llr = _log_gauss(s, mu_in, var_in) - _log_gauss(s, mu_out, var_out)
        scored.append({"example_id": rec["example_id"], "split": rec["split"],
                       "statistic": s, "llr": llr, "model": model,
                       "mu_in": mu_in, "sd_in": math.sqrt(var_in),
                       "mu_out": mu_out, "sd_out": math.sqrt(var_out)})
    return scored


def _logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _log_rho(s, ref_out):
    """log rho(t) = log Pr(t|theta) - log Pr(t), the RMIA per-example
    relative likelihood (eq. 3). The statistic s is read as the
    per-example NLL under the target model (Pr(t|theta) = exp(-s)); Pr(t)
    is the empirical mean likelihood over the OUT reference models
    (mean_i exp(-ref_out_i), the paper's offline Pr_OUT). Log-domain for
    numerical stability."""
    return -s - (_logsumexp([-r for r in ref_out]) - math.log(len(ref_out)))


def rmia_scores(records, statistic="raw"):
    """Score every record with the offline RMIA relative-likelihood
    quantile (Zarifzadeh et al. 2024, eq. 3 + eq. 5 at gamma = 1).

    RMIA(x) = fraction of the reference population Z (the split="out"
    records — the paper's "all non-members of the target model, excluding
    the query itself") whose relative likelihood log rho(z) is met or
    exceeded by log rho(x). Every record must carry ref_out (>=2 OUT
    reference NLLs); a descriptive SystemExit otherwise. Returns one dict
    per record: example_id, split, statistic (post transform), log_rho,
    rmia (the quantile score in [0, 1]), and n_ref_population
    (auditable)."""
    from bisect import bisect_right
    for rec in records:
        ref = rec.get("ref_out")
        if not isinstance(ref, list) or len(ref) < 2:
            raise SystemExit(
                f"[{EXPERIMENT_ID}] --scoring rmia needs per-example "
                "reference (OUT) statistics: every record must carry "
                f"'ref_out' (>=2 numbers); example "
                f"{rec.get('example_id')!r} has "
                f"{type(ref).__name__ if ref is not None else 'none'}. "
                "RMIA's offline Pr_OUT is the empirical mean likelihood "
                "over the example's OUT reference models.")
    if not any(r["split"] == "out" for r in records):
        raise SystemExit(f"[{EXPERIMENT_ID}] --scoring rmia needs >=1 OUT "
                         "record as the reference population")
    log_rhos = [(_transform(rec["statistic"], statistic),
                 _log_rho(_transform(rec["statistic"], statistic),
                          [_transform(v, statistic)
                           for v in rec["ref_out"]]))
                for rec in records]
    ref_population = sorted(lr for rec, (_, lr) in zip(records, log_rhos)
                            if rec["split"] == "out")
    m = len(ref_population)
    scored = []
    for rec, (s, lr) in zip(records, log_rhos):
        rank = bisect_right(ref_population, lr)
        if rec["split"] == "out":
            q = (rank - 1) / max(1, m - 1)
        else:
            q = rank / m
        scored.append({"example_id": rec["example_id"], "split": rec["split"],
                       "statistic": s,
                       "log_rho": lr, "rmia": q, "model": "rmia_offline",
                       "n_ref_population": m})
    return scored


def rank_auc(scores_in, scores_out):
    """Mann–Whitney AUC = P(in > out) with tie-averaged ranks; ties count
    ½. Rank-based: no distributional assumption on the score."""
    n_in, n_out = len(scores_in), len(scores_out)
    if n_in == 0 or n_out == 0:
        return float("nan")
    order = sorted([(v, 1) for v in scores_in]
                   + [(v, 0) for v in scores_out])
    rank_sum_in = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][0] == order[i][0]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        rank_sum_in += avg_rank * sum(tag for _, tag in order[i:j + 1])
        i = j + 1
    u = rank_sum_in - n_in * (n_in + 1) / 2.0
    return u / (n_in * n_out)


def _roc_sweep(scores_in, scores_out):
    """(threshold, tpr, fpr) over the achievable grid, thresholds swept
    high -> low; prediction = llr >= threshold."""
    sin = sorted(scores_in, reverse=True)
    sout = sorted(scores_out, reverse=True)
    n_in, n_out = len(sin), len(sout)
    points = [(float("inf"), 0.0, 0.0)]
    i = j = 0
    for thr in sorted(set(sin) | set(sout), reverse=True):
        while i < n_in and sin[i] >= thr:
            i += 1
        while j < n_out and sout[j] >= thr:
            j += 1
        points.append((thr, i / n_in, j / n_out))
    return points


def tpr_at_fpr(scores_in, scores_out, target):
    """Conservative TPR at FPR <= target: the max TPR over the achievable
    grid whose FPR never exceeds the target (no interpolation past the
    sample resolution; resolvable=False when 1/n_out > target)."""
    best_tpr, best_fpr = 0.0, 0.0
    for _, tpr, fpr in _roc_sweep(scores_in, scores_out):
        if fpr <= target + 1e-15 and tpr > best_tpr:
            best_tpr, best_fpr = tpr, fpr
    return {"target_fpr": target, "tpr": best_tpr, "fpr": best_fpr,
            "min_fpr_step": 1.0 / max(1, len(scores_out)),
            "resolvable": 1.0 / max(1, len(scores_out)) <= target}


def best_balanced_accuracy(scores_in, scores_out):
    """max over the achievable threshold grid of (tpr + tnr) / 2."""
    best, best_thr = 0.0, float("inf")
    for thr, tpr, fpr in _roc_sweep(scores_in, scores_out):
        ba = 0.5 * (tpr + 1.0 - fpr)
        if ba > best:
            best, best_thr = ba, thr
    return best, best_thr


def lira_metrics(scored, n_seeds=None, score_key="llr"):
    """The full metric bundle for one scored population. score_key is the
    per-record score field ("llr" for LIRA, "rmia" for RMIA); the metrics
    (rank AUC, TPR@low-FPR, gate) are scoring-agnostic."""
    scores_in = [r[score_key] for r in scored if r["split"] == "in"]
    scores_out = [r[score_key] for r in scored if r["split"] == "out"]
    auc = rank_auc(scores_in, scores_out)
    bal_acc, bal_thr = best_balanced_accuracy(scores_in, scores_out)
    verdict = gate.verdict(auc, 0.5, min(len(scores_in), len(scores_out)),
                           n_seeds=n_seeds)
    return {"n_in": len(scores_in), "n_out": len(scores_out),
            "auc": round(auc, 6),
            "tpr_at_fpr": {str(t): tpr_at_fpr(scores_in, scores_out, t)
                           for t in FPR_TARGETS},
            "best_balanced_accuracy": round(bal_acc, 6),
            "best_balanced_accuracy_threshold": bal_thr,
            "gate": verdict}


def make_toy_records(world, n_per_split, seed, shift, n_ref=0):
    """Toy boundary-statistic world. Per-example difficulty theta_i ~
    N(0,1) makes the statistic heterogeneous across examples (real
    per-example losses are); the observed statistic is a loss-like value
    s_i = theta_i + eps_i with eps_i ~ N(0,1), and MEMBERS shift lower by
    `shift` (training reduces a member's loss). The null world passes
    shift = 0 exactly — same distribution both sides, so AUC must land at
    0.5: the honesty arm.

    n_ref > 0 additionally attaches `ref_out`: n_ref reference (OUT)
    negative-log-likelihoods of the same example (theta_i + eps, NO
    membership shift — reference models never trained on it). RMIA reads
    these as Pr_OUT; LIRA ignores a lone ref_out (it needs ref_in too, so
    it stays on its global model — the offline comparison)."""
    rng = random.Random(seed)
    records = []
    for split in ("in", "out"):
        member_shift = shift if split == "in" else 0.0
        for i in range(n_per_split):
            theta = rng.gauss(0.0, 1.0)
            s = theta - member_shift + rng.gauss(0.0, 1.0)
            rec = {"example_id": f"{world}-{split}-{i:06d}",
                   "split": split, "statistic": s}
            if n_ref > 0:
                rec["ref_out"] = [theta + rng.gauss(0.0, 1.0)
                                  for _ in range(n_ref)]
            records.append(rec)
    rng.shuffle(records)
    return records


def _make_evidence(args):
    """--evidence wiring: one EvidenceLogger threaded through
    make_artifact/write_artifact (contract: default None = no change)."""
    if not getattr(args, "evidence", False):
        return None
    if not args.output:
        raise SystemExit(f"[{EXPERIMENT_ID}] --evidence requires --output")
    from ..evidence import EvidenceLogger
    return EvidenceLogger(args.output, node_role="attacker/lira-membership")


def run(args):
    ev = _make_evidence(args)
    online_records = None
    if args.online:
        if args.n_shadows < 2:
            raise SystemExit(
                f"[{EXPERIMENT_ID}] --n-shadows must be >=2 (the record "
                "schema needs >=2 reference NLLs per hypothesis)")
        from .common import require_torch
        require_torch(EXPERIMENT_ID)
        from ..shadow_train import ShadowConfig, build_lira_records
        cfg = ShadowConfig(n_shadows=args.n_shadows,
                           steps=args.shadow_steps, seed=args.seed)
        online_records = build_lira_records(cfg)
    if args.quick:
        args.seeds = [0]
        args.toy_n = 500
    out = artifacts.make_artifact(
        SCHEMA,
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": args.toy,
         "records": args.records, "scoring": args.scoring,
         "statistic": args.statistic,
         "worlds": args.worlds, "toy_n": args.toy_n,
         "toy_shift": args.toy_shift, "toy_ref": args.toy_ref,
         "online": args.online, "n_shadows": args.n_shadows,
         "shadow_steps": args.shadow_steps,
         "seeds": args.seeds,
         "fpr_targets": list(FPR_TARGETS)},
        "honest-but-curious cloud with shadow/reference boundary-feature "
        "distributions (the LIRA shadow-model assumption adapted to the "
        "split seam); per-example Gaussian IN/OUT models of a loss-like "
        "statistic, likelihood-ratio membership scores. Split ground "
        "truth on the records is evaluation truth, not an attack input "
        "(label-free attack).",
        interpretation="AUC is rank-based (Mann–Whitney, tie-averaged); "
                       "TPR@FPR is the conservative achievable-grid value; "
                       "the gate verdict uses recovery = AUC vs the 0.5 "
                       "chance control with n = min(n_in, n_out) (a "
                       "scoring convention — AUC is not binomial). The "
                       "null toy world must score AUC ~= 0.5 and PASS the "
                       "gate; anything else is a machinery bug.",
        evidence=ev)
    cells = []
    if args.toy:
        worlds = ("separable", "null") if args.worlds == "both" \
            else (args.worlds,)
        for world in worlds:
            for seed in args.seeds:
                shift = args.toy_shift if world == "separable" else 0.0
                n_ref = args.toy_ref if args.scoring == "rmia" else 0
                records = make_toy_records(world, args.toy_n,
                                           args.seed + 1000 * seed, shift,
                                           n_ref=n_ref)
                rec = {"experiment": EXPERIMENT_ID, "world": world,
                       "seed": seed, "shift": shift,
                       "scoring": args.scoring}
                if args.scoring == "rmia":
                    scored = rmia_scores(records, args.statistic)
                    rec.update(lira_metrics(scored,
                                            n_seeds=len(args.seeds),
                                            score_key="rmia"))
                    lira_scored = score_records(records, args.statistic)
                    rec["lira_global_auc"] = rank_auc(
                        [r["llr"] for r in lira_scored if r["split"] == "in"],
                        [r["llr"] for r in lira_scored
                         if r["split"] == "out"])
                else:
                    scored = score_records(records, args.statistic)
                    rec.update(lira_metrics(scored,
                                            n_seeds=len(args.seeds)))
                cells.append(rec)
                extra = (f" lira_global_auc={rec['lira_global_auc']:.4f}"
                         if args.scoring == "rmia" else "")
                print(f"[{args.scoring}-toy] world={world} seed={seed}: "
                      f"auc={rec['auc']} tpr@1%="
                      f"{rec['tpr_at_fpr']['0.01']['tpr']}{extra} "
                      f"gate={rec['gate']['verdict']}")
    elif args.records or online_records is not None:
        loaded = online_records if online_records is not None \
            else load_records(args.records)
        if args.scoring == "rmia":
            scored = rmia_scores(loaded, args.statistic)
            metrics = lira_metrics(scored, score_key="rmia")
        else:
            scored = score_records(loaded, args.statistic)
            metrics = lira_metrics(scored)
        cells.append({"experiment": EXPERIMENT_ID,
                      "world": ("online_shadow" if online_records is not None
                                else "offline"),
                      "seed": args.seed if online_records is not None else None,
                      "scoring": args.scoring, **metrics})
        print(f"[{args.scoring}] "
              f"{'online_shadow' if online_records is not None else 'offline'}"
              f": n={len(scored)} "
              f"auc={cells[0]['auc']} gate={cells[0]['gate']['verdict']}")
    else:
        raise SystemExit(f"[{EXPERIMENT_ID}] pass --toy, --records "
                         "FILE.jsonl ..., or --online (internal shadow-model "
                         "trainer, requires torch).")
    for rec in cells:
        artifacts.append_jsonl(args.output, rec)
        if ev is not None:
            ev.event("cell_result", rec)
        out["results"].append(rec)
    for world in sorted(set(r["world"] for r in cells)):
        vals = [r for r in cells if r["world"] == world]
        out["summary"].append({
            "experiment": EXPERIMENT_ID, "world": world,
            "auc_mean": sum(r["auc"] for r in vals) / len(vals),
            "gate_verdicts": [r["gate"]["verdict"] for r in vals],
            "n_cells": len(vals)})
    artifacts.write_artifact(args.output, out, evidence=ev)
    return 0


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import tempfile
    import os
    from statistics import NormalDist

    print("lira-membership scoring core (pure python):")
    nd = NormalDist(mu=1.0, sigma=2.0)
    check("log-gaussian matches statistics.NormalDist.pdf",
          abs(math.exp(_log_gauss(0.5, 1.0, 4.0)) - nd.pdf(0.5)) < 1e-12)
    check("logit transform: 0.5 -> 0, clamped at the ends",
          _transform(0.5, "logit") == 0.0
          and math.isfinite(_transform(0.0, "logit"))
          and math.isfinite(_transform(1.0, "logit")))

    check("AUC: perfect separation = 1, reversed = 0, ties = 0.5",
          rank_auc([2.0, 3.0], [0.0, 1.0]) == 1.0
          and rank_auc([0.0, 1.0], [2.0, 3.0]) == 0.0
          and rank_auc([1.0, 1.0], [1.0, 1.0]) == 0.5)
    r = tpr_at_fpr([3.0, 2.0, 1.5], [2.5, 1.6, 0.5], 1.0 / 3.0)
    check("TPR@FPR fixture: conservative achievable-grid value",
          r["tpr"] == 2.0 / 3.0 and r["fpr"] == 1.0 / 3.0
          and r["resolvable"] and r["min_fpr_step"] == 1.0 / 3.0)
    r2 = tpr_at_fpr([3.0, 2.0], [1.0, 0.0], 0.001)
    check("TPR@FPR flags an unresolvable target honestly",
          not r2["resolvable"] and r2["tpr"] == 1.0 and r2["fpr"] == 0.0)

    recs = [{"example_id": f"i{i}", "split": "in", "statistic": v}
            for i, v in enumerate((1.0, 2.0, 3.0))] + \
           [{"example_id": f"o{i}", "split": "out", "statistic": v}
            for i, v in enumerate((4.0, 5.0, 6.0))]
    scored = score_records(recs)
    first = scored[0]
    check("global LOO excludes the scored record from its own pool",
          first["model"] == "global_loo"
          and first["mu_in"] == 2.5 and first["mu_out"] == 5.0)
    wide = [{"example_id": f"i{i}", "split": "in", "statistic": v}
            for i, v in enumerate((0.0, 0.1, -0.1, 0.05, -0.05, 0.02))] + \
           [{"example_id": f"o{i}", "split": "out", "statistic": v}
            for i, v in enumerate((2.0, 2.1, 1.9, 2.05, 1.95, 1.98))]
    scored_wide = score_records(wide)
    check("members score above non-members on a separable fixture",
          rank_auc([r["llr"] for r in scored_wide if r["split"] == "in"],
                   [r["llr"] for r in scored_wide if r["split"] == "out"])
          == 1.0)
    pe = score_records([{"example_id": "x", "split": "in", "statistic": 0.0,
                         "ref_in": [0.0, 0.1, -0.1],
                         "ref_out": [2.0, 2.1, 1.9]}] + recs[1:])
    check("per-example ref lists switch the model used",
          pe[0]["model"] == "per_example"
          and abs(pe[0]["mu_in"] - 0.0) < 1e-12
          and abs(pe[0]["mu_out"] - 2.0) < 1e-12)

    null = score_records(make_toy_records("null", 2000, 0, 0.0))
    check("null toy world lands at AUC ~= 0.5 (honesty arm)",
          abs(rank_auc([r["llr"] for r in null if r["split"] == "in"],
                       [r["llr"] for r in null if r["split"] == "out"])
              - 0.5) < 0.05)

    check("parser builds torch-less", build_parser() is not None)
    from . import common as _common
    saved_torch = _common.torch
    _common.torch = None
    try:
        run(build_parser().parse_args(["--online"]))
        check("--online raises the require_torch SystemExit torch-less",
              False)
    except SystemExit as e:
        check("--online raises the require_torch SystemExit torch-less",
              "requires torch" in str(e))
    finally:
        _common.torch = saved_torch
    with tempfile.TemporaryDirectory() as d:
        ns = build_parser().parse_args(
            ["--toy", "--quick", "--output", os.path.join(d, "l.json")])
        ns.mode = "training"
        check("toy run end to end (both worlds) torch-free",
              run(ns) == 0)
        art = json.load(open(os.path.join(d, "l.json")))
        check("artifact has separable + null cells with gate verdicts",
              sorted(r["world"] for r in art["results"])
              == ["null", "separable"]
              and all(r["gate"]["verdict"] in ("PASS", "FAIL")
                      for r in art["results"]))
        check("journal lines match result cells",
              len(open(os.path.join(d, "l.json.jsonl")).read().splitlines())
              == len(art["results"]))

    print("rmia scoring core (pure python):")
    check("log rho fixture: -s - log-mean-exp(-ref)",
          abs(_log_rho(1.0, [2.0, 2.0]) - 1.0) < 1e-12
          and abs(_log_rho(0.0, [0.0, 0.0, 0.0]) - 0.0) < 1e-12)
    rmia_fix = [{"example_id": f"i{i}", "split": "in", "statistic": 0.0,
                 "ref_out": [2.0, 2.0, 2.0]} for i in range(6)] + \
               [{"example_id": f"o{i}", "split": "out", "statistic": 0.0,
                 "ref_out": [c, c, c]}
                for i, c in enumerate((-0.6, -0.4, -0.2, 0.0, 0.2, 0.4))]
    rs = rmia_scores(rmia_fix)
    check("rmia: members separate non-members in the relative likelihood",
          rank_auc([r["log_rho"] for r in rs if r["split"] == "in"],
                   [r["log_rho"] for r in rs if r["split"] == "out"]) == 1.0
          and sum(r["rmia"] for r in rs if r["split"] == "in")
          > sum(r["rmia"] for r in rs if r["split"] == "out"))
    check("rmia score is a quantile in [0, 1] with audit fields",
          all(0.0 <= r["rmia"] <= 1.0 for r in rs)
          and all(r["model"] == "rmia_offline"
                  and r["n_ref_population"] == 6 and "log_rho" in r
                  for r in rs))
    check("rmia leave-one-out: only the top OUT query reaches quantile 1",
          sum(r["rmia"] == 1.0 for r in rs if r["split"] == "out") == 1
          and all(r["rmia"] == 1.0 for r in rs if r["split"] == "in"))
    try:
        rmia_scores([{"example_id": "x", "split": "in", "statistic": 1.0},
                     {"example_id": "y", "split": "out", "statistic": 2.0,
                      "ref_out": [2.0, 2.1]}])
        check("rmia without ref_out on every record is a descriptive error",
              False)
    except SystemExit as e:
        check("rmia without ref_out on every record is a descriptive error",
              "ref_out" in str(e))
    for world, shift in (("separable", 2.0), ("null", 0.0)):
        recs_w = make_toy_records(world, 3000, 0, shift, n_ref=8)
        rs_w = rmia_scores(recs_w)
        auc_r = rank_auc([r["rmia"] for r in rs_w if r["split"] == "in"],
                         [r["rmia"] for r in rs_w if r["split"] == "out"])
        ls_w = score_records(recs_w)
        auc_l = rank_auc([r["llr"] for r in ls_w if r["split"] == "in"],
                         [r["llr"] for r in ls_w if r["split"] == "out"])
        check(f"rmia toy {world}: rmia AUC >= lira AUC, honest level",
              auc_r >= auc_l - 0.02
              and (abs(auc_r - 0.5) < 0.05 if world == "null"
                   else auc_r > 0.8))
    check("parser builds torch-less with --scoring",
          build_parser().parse_args([]).scoring == "lira"
          and build_parser().parse_args(["--scoring", "rmia"]).scoring
          == "rmia")
    with tempfile.TemporaryDirectory() as d:
        ns = build_parser().parse_args(
            ["--toy", "--scoring", "rmia", "--quick",
             "--output", os.path.join(d, "r.json")])
        ns.mode = "training"
        check("rmia toy run end to end (both worlds) torch-free",
              run(ns) == 0)
        art = json.load(open(os.path.join(d, "r.json")))
        check("rmia artifact has separable+null cells with lira comparison",
              sorted(r["world"] for r in art["results"]) == ["null",
                                                             "separable"]
              and all("lira_global_auc" in r for r in art["results"])
              and all(r["scoring"] == "rmia" for r in art["results"]))
        check("rmia journal lines match result cells",
              len(open(os.path.join(d, "r.json.jsonl")).read().splitlines())
              == len(art["results"]))

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
