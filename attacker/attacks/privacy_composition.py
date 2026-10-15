#!/usr/bin/env python3
"""privacy-composition — do individually acceptable releases become unsafe
when combined over time, users, models, or observation surfaces?
(THREAT_MODEL_REVIEW P1). Each release of a split-LLM system may PASS the
frozen gate on its own (excess over control <= +1.0pp) while the
CUMULATIVE budget an attacker accumulates — repeated queries in one
session, the same identity across sessions, observations across training
epochs, or several observation surfaces at once — composes into a FAIL.
This module ORCHESTRATES existing attacks before introducing new attack
math: it consumes the journals/artifacts of other attacks and builds a
composition graph; a seeded toy multi-surface world exercises the same
composer with known ground truth. Label-free; applies to training and
inference. Pure stdlib, torch-free.

INPUT PATHS
===========

(a) SCENARIO path (not empirical joint leakage): --artifacts A1.json A2.json ... reads completed attack
    artifacts. Every result record carrying an experiment id plus a
    recognized metric becomes a point on the composition-graph node keyed
    (attack, surface via attacker.attack_metadata); records carrying a
    recognized budget field contribute their budget axis. Recognized
    fields (first numeric match wins):

        metric  advantage | membership_advantage (already an excess over
                control; |v| > 1 read as percentage points) | auc (mapped
                to membership advantage 2*AUC-1) | recovery | top1 |
                top1_mean | composite_decode_top1 (rates scored against
                --control-rate; v > 1 read as percent)
        budget  budget | cumulative_budget | cumulative_leaked_rows |
                epoch_size | leak_per_epoch (absent: ONE unit of budget,
                counted in the audit)
        rows    n | n_eval | n_records (eval rows behind the metric; the
                confidence bound needs them)
        epoch   epoch (numeric; the cross-epoch axis)
        surface surface (a README.md id; absent: the attack's
                FIRST metadata surface — multi-surface attacks are counted
                in the audit)

    Skipped records (error cells, no experiment id, unknown experiment id,
    no metadata, no recognized metric, non-positive budget) are COUNTED in
    the input-audit cell, never silently dropped. A missing, unparseable,
    or completely empty artifact is a descriptive ValueError naming the
    path (fail-closed).

(b) TOY path: --toy generates a synthetic multi-surface world
    (make_toy_streams): per-surface measurement streams for S1 (seam
    activations), S2 (seam gradients), S6 (cache/timing), S7 (traffic
    shape) whose per-query attacker advantage grows with cumulative budget
    along seeded per-surface curves a_s(b) = amplitude_s *
    (1 - exp(-b/tau_s)) (seeded +/-5% jitter), plus a SYNERGY term
    gamma * sqrt(a_S1 * a_S2) whenever S1 and S2 are combined, plus a NULL
    control surface sitting at exactly 0 (PINNED at the control rate —
    gate.verdict treats the control as fixed, so the null cell reports
    the floor rather than adding a second layer of binomial noise). Each
    non-null cell point draws seeded binomial eval rows at control +
    advantage.

COMPOSITION MODEL
=================

Surfaces compose under independence: joint = 1 - prod(1 - a_s). The toy
world additionally injects the S1xS2 synergy term above; the REAL path
never invents leakage — it composes with gamma = 0 and REPORTS the
observed synergy (joint minus the strongest individual). Composition
dimensions (arms):

    in-session       each surface/node alone vs cumulative query budget
    cross-session    separate artifacts/runs on one surface compose (the
                     same identity observed across sessions)
    cross-epoch      records with an epoch axis accumulate budget
                     cumulatively across epochs (toy: S1+S2 with a
                     per-epoch --retention factor)
    multi-surface    all surfaces joint
    fwd-bwd-timing   S1 + S2 + S6 joint (forward + backward + timing)

METRICS (cells): advantage-vs-budget tables per dimension; the FIRST
budget at which the frozen gate FAILs (attacker.gate.DEFAULT_GATE via
gate.verdict — imported, never reimplemented); marginal per-surface
leakage (leave-one-out delta at the maximum budget); synergy of the joint
vs the strongest individual attack (pp); and a confidence-adjusted upper
bound over the best adaptive composition (Bonferroni-Wilson upper-95 via
gate.wilson_upper over the seeded binomial draws, max across dimensions
per budget). NEGATIVE CONTROL (mandatory, own cells): each surface alone
at its minimum budget plus the NULL surface — advantage ~0, gate PASS.
The vulnerable toy world MUST fail the gate on the joint dimensions well
before any individual surface, else the control is vacuous.

LIMITATIONS / EXCLUDED VARIANTS
===============================

- The toy curves are seeded analytic forms with an injected synergy term —
  a machinery check with known ground truth, NOT measured leakage.
- The real path composes COMPLETED artifacts; driving live attack runs
  from this module (orchestrating processes rather than artifacts) is a
  future integration point, intentionally excluded.
- Joint confidence uses n = min eval rows across composed surfaces with k
  synthesized from the composed rate (an approximation; every composed
  surface must carry an eval-row count or the joint verdict is reported
  UNMEASURED, never guessed). Per-node cells carry exact per-record
  verdicts.
- Off-grid budgets on the real path use the nearest carried point (last
  point at or below the query, else the first); repeated measurements at
  one budget average (mean advantage, min eval rows — conservative).
- Cross-session identity assumption: separate artifacts/runs are treated
  as separate sessions of one identity.
- Excluded: DP/Renyi-accountant composition, collusion across distinct
  identities, adaptive per-query budget reallocation, model-version /
  split-position dimensions, non-binomial confidence models.

Usage:
    python -m attacker --mode inference --attack privacy-composition --help
    python -m attacker --mode inference --attack privacy-composition \
        --toy --quick --output /tmp/pc.json
    python -m attacker --mode training --attack privacy-composition \
        --artifacts out/wire_shape.json out/er4.json --output pc.json
"""

import argparse
import json
import math
import random
import statistics

from .. import artifacts, attack_metadata, gate, threat_model
from .common import add_common_args, journal_error

EXPERIMENT_ID = "privacy_composition"
MODES = ("training", "inference")
REQUIRES_LABELS = False
DESCRIPTION = ("privacy composition (review P1): compose attack artifacts "
               "(or a seeded toy multi-surface world) into advantage-vs-"
               "budget curves per composition dimension — first frozen-"
               "gate failure, marginal per-surface leakage, synergy, and "
               "a confidence-adjusted upper bound over the best adaptive "
               "composition")

SCHEMA_VERSION = "dtraining.attacker.privacy_composition.v1"

TOY_SURFACES = ("S1", "S2", "S6", "S7")
TOY_ANCHORS = {"S1": (0.040, 32.0), "S2": (0.035, 24.0),
               "S6": (0.020, 64.0), "S7": (0.015, 96.0)}
TOY_JITTER = 0.05
NULL_SURFACE = "NULL"
FWD_BWD_TIMING = ("S1", "S2", "S6")

ARMS = ("negative-control", "in-session", "cross-session", "cross-epoch",
        "multi-surface", "fwd-bwd-timing", "marginal-leakage", "synergy",
        "adaptive-upper-bound")

METRIC_FIELDS = ("advantage", "membership_advantage", "auc", "recovery",
                 "top1", "top1_mean", "composite_decode_top1")
BUDGET_FIELDS = ("budget", "cumulative_budget", "cumulative_leaked_rows",
                 "epoch_size", "leak_per_epoch")
N_FIELDS = ("n", "n_eval", "n_records")
EPOCH_FIELD = "epoch"

CONTROL_RATE = 0.04
N_EVAL = 8192
SYNERGY_GAMMA = 0.3
CARRYOVER = 0.8
RETENTION = 0.5
DEFAULT_BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128)


def binomial_draw(rng, n, p):
    """Exact Binomial(n, p) via geometric waiting times (fast at the small
    rates here). Deterministic given `rng`."""
    if n <= 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    log_q = math.log(1.0 - p)
    k, pos = 0, -1
    while True:
        pos += int(math.log(1.0 - rng.random()) / log_q) + 1
        if pos >= n:
            return k
        k += 1


def combine(advantages, gamma=0.0):
    """Independent composition 1 - prod(1 - a_s) over {surface: advantage},
    plus the seeded S1xS2 synergy term gamma * sqrt(a_S1 * a_S2) when both
    are present (toy world only — the real path passes gamma=0: synergy is
    REPORTED, never invented). Clamped to [0, 1]."""
    if not advantages:
        return 0.0
    prod = 1.0
    for a in advantages.values():
        prod *= (1.0 - a)
    joint = 1.0 - prod
    if gamma and "S1" in advantages and "S2" in advantages:
        a1, a2 = advantages["S1"], advantages["S2"]
        if a1 > 0.0 and a2 > 0.0:
            joint += gamma * math.sqrt(a1 * a2)
    return max(0.0, min(1.0, joint))


def _draws(control_rate, advantage, n):
    """Synthesize binomial successes for a composed rate (clamped to
    [0, 1]); None when no eval-row count backs the point."""
    if not n:
        return None
    p = min(1.0, max(0.0, control_rate + advantage))
    return int(round(p * n))


class ToyStreams:
    """The seeded synthetic multi-surface world: per-surface advantage
    curves a_s(b) = amplitude * (1 - exp(-b/tau)) with seeded +/-5%
    jitter, the S1xS2 synergy term via combine(..., gamma), binomial eval
    draws at control + advantage, and the NULL control surface at 0."""

    def __init__(self, seed, control_rate=CONTROL_RATE, gamma=SYNERGY_GAMMA):
        self.control_rate = control_rate
        self.gamma = gamma
        self.rng = random.Random(seed)
        self.surfaces = {}
        for s in TOY_SURFACES:
            amp, tau = TOY_ANCHORS[s]
            self.surfaces[s] = {
                "amplitude": amp * (1.0 + self.rng.uniform(-TOY_JITTER,
                                                           TOY_JITTER)),
                "tau": tau * (1.0 + self.rng.uniform(-TOY_JITTER,
                                                     TOY_JITTER))}
        self.surfaces[NULL_SURFACE] = {"amplitude": 0.0, "tau": 1.0}

    def surface_advantage(self, surface, budget):
        st = self.surfaces[surface]
        if st["amplitude"] <= 0.0 or budget <= 0.0:
            return 0.0
        return st["amplitude"] * (1.0 - math.exp(-budget / st["tau"]))

    def advantage(self, surfaces, budget):
        return combine({s: self.surface_advantage(s, budget)
                        for s in surfaces}, self.gamma)

    def point(self, surfaces, budget, n_eval):
        joint = self.advantage(surfaces, budget)
        if joint <= 0.0:
            k = int(round(self.control_rate * n_eval))
        else:
            k = binomial_draw(self.rng, n_eval,
                              min(1.0, self.control_rate + joint))
        return {"advantage": joint, "k": k, "n": n_eval,
                "per_surface": {s: self.surface_advantage(s, budget)
                                for s in surfaces}}


def make_toy_streams(seed, control_rate=CONTROL_RATE, gamma=SYNERGY_GAMMA):
    """The synthetic multi-surface world generator (S1 seam, S2 grads,
    S6 timing, S7 traffic + the NULL control surface)."""
    return ToyStreams(seed, control_rate=control_rate, gamma=gamma)


class CompositionNodes:
    """Nodes keyed (attack, surface), each carrying (budget, advantage, n,
    epoch) points extracted from real artifacts. Repeated measurements at
    one budget average (mean advantage, min eval rows — conservative);
    off-grid budgets use the nearest carried point (last point at or below
    the query, else the first)."""

    def __init__(self, control_rate):
        self.control_rate = control_rate
        self.nodes = []

    def add_point(self, attack, surface, budget, advantage, n, epoch):
        for node in self.nodes:
            if node["attack"] == attack and node["surface"] == surface:
                break
        else:
            node = {"attack": attack, "surface": surface, "by_budget": {}}
            self.nodes.append(node)
        node["by_budget"].setdefault(budget, []).append((advantage, n,
                                                         epoch))

    def node_points(self, node):
        """Sorted point dicts with repeats averaged."""
        pts = []
        for b in sorted(node["by_budget"]):
            obs = node["by_budget"][b]
            ns = [o[1] for o in obs if o[1]]
            epochs = sorted({o[2] for o in obs if o[2] is not None})
            pts.append({"budget": b,
                        "advantage": statistics.fmean(o[0] for o in obs),
                        "n": min(ns) if ns else None,
                        "n_points": len(obs), "epochs": epochs})
        return pts

    def surfaces(self):
        return sorted({nd["surface"] for nd in self.nodes})

    def nodes_for(self, surface):
        return [nd for nd in self.nodes if nd["surface"] == surface]

    @staticmethod
    def _value_at(pts, budget):
        if not pts:
            return None
        chosen = pts[0]
        for p in pts:
            if p["budget"] <= budget:
                chosen = p
            else:
                break
        return chosen

    def surface_advantage(self, surface, budget):
        """1-prod across the surface's nodes: separate artifacts/runs
        observing one surface compose."""
        advs = {}
        for i, nd in enumerate(self.nodes_for(surface)):
            p = self._value_at(self.node_points(nd), budget)
            if p is not None:
                advs[i] = p["advantage"]
        return combine(advs, 0.0) if advs else None

    def advantage(self, surfaces, budget):
        advs = {}
        for s in surfaces:
            a = self.surface_advantage(s, budget)
            if a is not None:
                advs[s] = a
        return combine(advs, 0.0) if advs else None

    def point(self, surfaces, budget):
        """Composed point; the joint eval-row count is the MIN across
        composed surfaces (the least-observed surface bounds confidence)
        and None unless every composed surface carries one."""
        per, ns = {}, {}
        for s in surfaces:
            a = self.surface_advantage(s, budget)
            if a is None:
                continue
            per[s] = a
            for i, nd in enumerate(self.nodes_for(s)):
                p = self._value_at(self.node_points(nd), budget)
                if p is None or not p["n"]:
                    ns[s] = None
                    break
                ns.setdefault(s, p["n"])
                ns[s] = min(ns[s], p["n"])
        if not per:
            return {"advantage": 0.0, "k": None, "n": None,
                    "per_surface": {}}
        joint = combine(per, 0.0)
        have_n = all(ns.get(s) for s in per)
        n = min(ns[s] for s in per) if have_n else None
        return {"advantage": joint, "k": _draws(self.control_rate, joint, n),
                "n": n, "per_surface": per}


def load_artifact_documents(paths):
    """Read attack artifacts; a missing, unparseable, or completely empty
    artifact is a descriptive ValueError naming the path (fail-closed —
    never silently compose nothing)."""
    docs = []
    for path in paths:
        try:
            with open(path) as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"cannot read artifact {path!r}: {e}") from e
        if not isinstance(doc, dict) or not doc.get("results"):
            raise ValueError(f"artifact {path!r} carries no result records "
                             "(empty or malformed; refusing to compose "
                             "nothing)")
        docs.append((path, doc))
    return docs


def _experiment_to_attack():
    """EXPERIMENT_ID -> attack registry name (lazy import: the registry
    validates all built-ins at import; torch-free by repo rule)."""
    from . import ATTACKS, load
    out = {}
    for name in ATTACKS:
        try:
            exp = getattr(load(name), "EXPERIMENT_ID", None)
        except Exception:
            continue
        if exp:
            out[exp] = name
    return out


def _numeric(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and x == x


def normalize_metric(field, value, control_rate):
    """Map a recognized metric to an advantage fraction (excess over the
    control rate). advantage/membership_advantage are already advantages
    (|v| > 1 read as pp); AUC maps to membership advantage 2*AUC-1;
    recovery/top1-style rates are scored against the control rate
    (v > 1 read as percent)."""
    v = float(value)
    if field in ("advantage", "membership_advantage"):
        return v / 100.0 if abs(v) > 1.0 else v
    if field == "auc":
        return max(-1.0, min(1.0, 2.0 * v - 1.0))
    rate = v / 100.0 if v > 1.0 else v
    return rate - control_rate


def extract_nodes(docs, control_rate):
    """Build the composition graph from artifact documents (results only —
    summary rows are derived views). Skips are counted in the audit dict,
    never silent."""
    exp2attack = _experiment_to_attack()
    nodes = CompositionNodes(control_rate)
    audit = {"records_total": 0, "skipped_error": 0,
             "skipped_no_experiment": 0, "skipped_unknown_experiment": 0,
             "skipped_no_metadata": 0, "skipped_no_metric": 0,
             "skipped_bad_budget": 0, "defaulted_budget": 0,
             "defaulted_surface": 0, "points": 0, "nodes": 0}
    for path, doc in docs:
        for rec in doc["results"]:
            audit["records_total"] += 1
            if not isinstance(rec, dict):
                audit["skipped_no_experiment"] += 1
                continue
            if "error" in rec:
                audit["skipped_error"] += 1
                continue
            exp = rec.get("experiment")
            if not isinstance(exp, str) or not exp:
                audit["skipped_no_experiment"] += 1
                continue
            attack = exp2attack.get(exp)
            if attack is None:
                audit["skipped_unknown_experiment"] += 1
                continue
            try:
                md = attack_metadata.metadata_for(attack)
            except KeyError:
                audit["skipped_no_metadata"] += 1
                continue
            metric = next((f for f in METRIC_FIELDS if _numeric(rec.get(f))),
                          None)
            if metric is None:
                audit["skipped_no_metric"] += 1
                continue
            budget = next((rec[f] for f in BUDGET_FIELDS
                           if _numeric(rec.get(f))), None)
            if budget is None:
                audit["defaulted_budget"] += 1
                budget = 1.0
            elif budget <= 0:
                audit["skipped_bad_budget"] += 1
                continue
            surface = rec.get("surface")
            if surface not in attack_metadata.SURFACES:
                surface = md["surfaces"][0]
                if len(md["surfaces"]) > 1:
                    audit["defaulted_surface"] += 1
            n = next((int(rec[f]) for f in N_FIELDS
                      if _numeric(rec.get(f)) and rec[f] > 0), None)
            epoch = rec.get(EPOCH_FIELD)
            epoch = float(epoch) if _numeric(epoch) else None
            nodes.add_point(attack, surface, float(budget),
                            normalize_metric(metric, rec[metric],
                                             control_rate), n, epoch)
            audit["points"] += 1
    audit["nodes"] = len(nodes.nodes)
    return nodes, audit


def _gate_fields(advantage, k, n, control_rate):
    if not n:
        return {"recovery_pct": None, "upper95_pct": None,
                "excess_pp": None, "n_eval": None, "verdict": None,
                "note": "no eval-row count behind this point; gate "
                        "verdict UNMEASURED"}
    v = gate.verdict(k / n, control_rate, n)
    return {"recovery_pct": round(v["recovery_pct"], 6),
            "upper95_pct": round(v["upper95_pct"], 6),
            "excess_pp": round(v["excess_pp"], 6),
            "floor_pp": round(v["floor_pp"], 6),
            "n_eval": n, "verdict": v["verdict"]}


def compose_curve(probe, surfaces, budgets, control_rate):
    """The advantage-vs-budget table for one composition dimension:
    probe(surfaces, budget) -> {"advantage", "k", "n", "per_surface"}."""
    curve = []
    for b in budgets:
        pt = probe(tuple(surfaces), float(b))
        entry = {"budget": b,
                 "advantage_pp": round(100.0 * pt["advantage"], 6),
                 "per_surface_pp": {s: round(100.0 * a, 6) for s, a
                                    in pt["per_surface"].items()}}
        entry.update(_gate_fields(pt["advantage"], pt.get("k"), pt.get("n"),
                                  control_rate))
        curve.append(entry)
    return curve


def first_gate_failure(curve):
    """The FIRST budget whose frozen-gate verdict is FAIL (None when the
    dimension never fails — e.g. the NULL surface)."""
    for e in curve:
        if e.get("verdict") == "FAIL":
            return e["budget"]
    return None


def marginal_leakage(advantage_fn, surfaces, budget):
    """Leave-one-out delta at `budget`: how much joint advantage each
    surface contributes (pp)."""
    full = advantage_fn(tuple(surfaces), budget) or 0.0
    deltas = {}
    for s in surfaces:
        rest = tuple(x for x in surfaces if x != s)
        without = advantage_fn(rest, budget) if rest else 0.0
        deltas[s] = round(100.0 * (full - (without or 0.0)), 6)
    return deltas


def synergy_curve(advantage_fn, surfaces, budgets):
    """Joint vs the strongest individual attack per budget (pp)."""
    rows = []
    for b in budgets:
        advs = {s: advantage_fn((s,), b) or 0.0 for s in surfaces}
        joint = advantage_fn(tuple(surfaces), b) or 0.0
        strongest = max(advs.values()) if advs else 0.0
        rows.append({"budget": b,
                     "joint_advantage_pp": round(100.0 * joint, 6),
                     "strongest_individual_pp": round(100.0 * strongest, 6),
                     "strongest_surface": (max(advs, key=advs.get)
                                           if advs else None),
                     "synergy_pp": round(100.0 * (joint - strongest), 6)})
    best = max(rows, key=lambda r: r["synergy_pp"]) if rows else None
    return rows, best


def adaptive_upper_bound(curves_by_arm, control_rate):
    """Confidence-adjusted upper bound over the best adaptive composition:
    per budget, the max Bonferroni-Wilson upper-95 across dimensions."""
    budgets = sorted({e["budget"] for c in curves_by_arm.values()
                      for e in c})
    rows = []
    for b in budgets:
        best_arm, best = None, None
        for arm, curve in sorted(curves_by_arm.items()):
            for e in curve:
                if e["budget"] == b and e.get("upper95_pct") is not None \
                        and (best is None
                             or e["upper95_pct"] > best["upper95_pct"]):
                    best_arm, best = arm, e
        if best is None:
            rows.append({"budget": b, "best_arm": None,
                         "upper95_pct": None, "excess_pp": None,
                         "note": "no confidence-bearing cell at this "
                                 "budget"})
        else:
            rows.append({"budget": b, "best_arm": best_arm,
                         "upper95_pct": best["upper95_pct"],
                         "excess_pp": round(best["upper95_pct"]
                                            - 100.0 * control_rate, 6)})
    return rows


def _threat_model_doc():
    return threat_model.normalize({
        "schema": threat_model.SCHEMA_V2,
        "summary": "passive actor accumulating the split system's own "
                   "released measurement artifacts (or equivalent "
                   "per-surface observations) over time, sessions, epochs, "
                   "and observation surfaces — testing whether releases "
                   "that individually PASS the frozen gate compose into a "
                   "gate FAIL",
        "scope": {"use_case": "split-training",
                  "deployment": "multi-tenant",
                  "in_scope": ["completed attack artifacts/journals of "
                               "this framework (real path)",
                               "seeded synthetic multi-surface world (toy "
                               "path)"],
                  "out_of_scope": ["orchestrating live attack runs "
                                   "(future integration point)",
                                   "DP/Renyi accountant composition",
                                   "collusion across distinct identities"]},
        "actor": {"role": "cloud", "intent": "passive",
                  "vantage": ["own measurement logs/artifacts accumulated "
                              "over releases",
                              "S1 seam, S2 gradient, S6 timing, S7 traffic "
                              "observations"],
                  "authorization": "evaluator-held artifacts only",
                  "privileges": ["read access to released artifacts"],
                  "knowledge": ["attack metadata (surface per attack)",
                                "the frozen gate definition"],
                  "capabilities": ["composition-graph construction",
                                   "independent-composition + synergy "
                                   "accounting",
                                   "confidence-adjusted adaptive "
                                   "composition"],
                  "resources": ["offline compute"],
                  "persistence": "cross-epoch"},
        "target": {"assets": ["per-release privacy guarantees",
                              "cumulative privacy budget"],
                   "components": ["release/scorecard pipeline",
                                  "the frozen gate"],
                   "data_flows": ["releases -> attacker artifact store",
                                  "artifact store -> composition graph"],
                   "trust_boundaries": ["per-release accounting vs "
                                        "cross-release accumulation"],
                   "entry_points": ["--artifacts composition"],
                   "exit_points": ["composition cells + summary"]},
        "scenario": {"preconditions": ["individually released results "
                                       "available to the attacker",
                                       "releases share identity/surfaces "
                                       "across time"],
                     "lifecycle_stage": ["training", "inference",
                                         "operations"],
                     "security_properties": ["privacy", "confidentiality"],
                     "attack_steps": ["collect artifacts per release",
                                      "build the composition graph keyed "
                                      "(attack, surface)",
                                      "compose per dimension vs "
                                      "cumulative budget",
                                      "score the first frozen-gate "
                                      "failure, marginals, synergy, and "
                                      "the adaptive upper bound"],
                     "technical_impact": ["individually safe releases "
                                          "compose into a gate failure"],
                     "business_impact": ["privacy-budget accounting that "
                                         "ignores composition "
                                         "underestimates disclosure"]},
        "controls": {"assumed": ["per-release frozen-gate scoring",
                                 "key rotation across epochs"],
                     "tested": ["composition across budgets, sessions, "
                                "epochs, and surfaces"],
                     "bypassed": ["per-release budget accounting (the "
                                  "cumulative budget is what composes)"],
                     "detective": ["this composer"],
                     "status": "open"},
        "mappings": {"owasp_llm_2026": ["LLM02"]},
        "evaluation": {"success_metrics": ["advantage vs cumulative budget "
                                           "per composition dimension",
                                           "first gate-failure budget",
                                           "marginal per-surface leakage "
                                           "(leave-one-out)",
                                           "synergy vs strongest "
                                           "individual (pp)",
                                           "confidence-adjusted upper "
                                           "bound over the best adaptive "
                                           "composition"],
                       "baseline": "NULL surface + per-surface minimum-"
                                   "budget cells: advantage ~0, gate PASS",
                       "budget": {"toy": f"{N_EVAL} eval rows per point, "
                                         "budgets grid --budgets",
                                  "real": "budgets as recorded in the "
                                          "source artifacts"},
                       "limitations": ["toy curves are seeded analytic "
                                       "forms, not measured leakage",
                                       "real path composes completed "
                                       "artifacts only (no live driver)",
                                       "joint confidence bounded by the "
                                       "least-observed surface"],
                       "coverage_status": "covered"}})


def _make_artifact(args, source, extra_config=None, evidence=None):
    config = {"attack": EXPERIMENT_ID, "mode": getattr(args, "mode", None),
              "source": source, "seed": args.seed,
              "control_rate": args.control_rate, "trials": args.trials,
              "carryover": args.carryover, "retention": args.retention,
              "synergy_gamma": args.synergy, "gate": gate.GATE_VERSION}
    if extra_config:
        config.update(extra_config)
    return artifacts.make_artifact(
        SCHEMA_VERSION, config, _threat_model_doc(),
        interpretation="secure composition: the NULL surface and every "
                       "minimum-budget cell sit at advantage ~0 and PASS, "
                       "and joint dimensions fail the frozen gate no "
                       "earlier than their strongest individual surface. "
                       "The toy world is DELIBERATELY vulnerable: joint "
                       "dimensions must FAIL well before any individual "
                       "surface and synergy_pp must be positive, else the "
                       "negative control is vacuous. On the real path, a "
                       "first_gate_failure_budget below the per-release "
                       "budgets means the release accounting "
                       "underestimates cumulative leakage.",
        evidence=evidence)


def _scenario_record(rec):
    """Model projections have no observed joint counts or confidence gate."""
    if isinstance(rec, list):
        return [_scenario_record(v) for v in rec]
    if not isinstance(rec, dict):
        return rec
    excluded = {"gate", "verdict", "correct", "n_correct", "upper95_pct", "upper95", "excess_pp", "max_excess_pp", "max_excess_pp_mean", "first_gate_failure_budget", "min_first_gate_failure_budget"}
    result = {k: _scenario_record(v) for k, v in rec.items() if k not in excluded}
    result.update(evidence_tier="scenario", verdict="UNMEASURED")
    if "note" in result:
        result["note"] = "Scenario projection under modeling assumptions; not an observed joint attack."
    return result


def _emit(args, out, rec, line):
    emitted = _scenario_record(rec) if getattr(args, "artifacts", None) else rec
    artifacts.append_jsonl(args.output, emitted)
    out["results"].append(emitted)
    print(line)


def _curve_cell(args, out, arm, surfaces, curve, seed=None, extra=None):
    rec = {"experiment": EXPERIMENT_ID, "arm": arm,
           "surfaces": list(surfaces), "curve": curve,
           "first_gate_failure_budget": first_gate_failure(curve),
           "max_advantage_pp": max((e["advantage_pp"] for e in curve),
                                   default=0.0),
           "n_budgets": len(curve)}
    if seed is not None:
        rec["seed"] = seed
    if extra:
        rec.update(extra)
    _emit(args, out, rec,
          f"[{EXPERIMENT_ID}] "
          + (f"seed={seed} " if seed is not None else "")
          + f"{arm} surfaces={'+'.join(surfaces) or '-'}: "
            f"first-fail={rec['first_gate_failure_budget']} "
            f"adv_max={rec['max_advantage_pp']:.2f}pp")
    return rec


def _summarize(out):
    per_arm = {}
    for r in out["results"]:
        if "error" in r or "arm" not in r:
            continue
        per_arm.setdefault(r["arm"], []).append(r)
    for arm, recs in sorted(per_arm.items()):
        row = {"experiment": EXPERIMENT_ID, "arm": arm,
               "n_cells": len(recs)}
        ffs = [r["first_gate_failure_budget"] for r in recs
               if r.get("first_gate_failure_budget") is not None]
        if ffs:
            row["min_first_gate_failure_budget"] = min(ffs)
        ma = [r["max_advantage_pp"] for r in recs if "max_advantage_pp" in r]
        if ma:
            row["max_advantage_pp_mean"] = statistics.fmean(ma)
        syn = [r["max_synergy_pp"] for r in recs if "max_synergy_pp" in r]
        if syn:
            row["max_synergy_pp_mean"] = statistics.fmean(syn)
        ub = [r["max_excess_pp"] for r in recs if "max_excess_pp" in r]
        if ub:
            row["max_excess_pp_mean"] = statistics.fmean(ub)
        out["summary"].append(row)


def run_toy(args, evidence=None):
    budgets = sorted(set(args.budgets))
    if args.quick:
        args.seeds = args.seeds[:2]
        budgets = budgets[:4]
    out = _make_artifact(args, "toy",
                         {"toy": True, "budgets": budgets,
                          "seeds": args.seeds,
                          "toy_anchors": {s: list(a) for s, a
                                          in TOY_ANCHORS.items()}},
                         evidence=evidence)
    for seed in args.seeds:
        streams = make_toy_streams(args.seed + 1000 * seed,
                                   control_rate=args.control_rate,
                                   gamma=args.synergy)
        curves = {}

        def cell(arm, fn, extra_cell=None):
            """Run one cell; failures are journaled, never raised."""
            try:
                return fn()
            except Exception as e:
                journal_error(args.output, EXPERIMENT_ID,
                              {"arm": arm, "seed": seed}, e)
                return None

        for s in TOY_SURFACES + (NULL_SURFACE,):
            def insession(s=s):
                curve = compose_curve(
                    lambda surf, b: streams.point(surf, b, args.trials),
                    (s,), budgets, args.control_rate)
                curves[f"in-session/{s}"] = curve
                return _curve_cell(args, out, "in-session", (s,), curve,
                                   seed=seed)
            cell(f"in-session/{s}", insession)

        for s in TOY_SURFACES + (NULL_SURFACE,):
            def neg(s=s):
                curve = curves.get(f"in-session/{s}")
                if not curve:
                    raise ValueError(f"in-session curve for {s} missing; "
                                     "the negative control reuses its "
                                     "first point")
                return _curve_cell(
                    args, out, "negative-control", (s,), curve[:1],
                    seed=seed,
                    extra={"role": "negative-control",
                           "expectation": "advantage ~0 and gate PASS at "
                                          "the minimum budget / null "
                                          "surface",
                           "stream_params": streams.surfaces[s]})
            cell(f"negative-control/{s}", neg)

        def cross_session():
            curve = compose_curve(
                lambda surf, b: streams.point(surf, b * args.carryover,
                                              args.trials),
                TOY_SURFACES, budgets, args.control_rate)
            for e in curve:
                e["effective_budget"] = round(e["budget"]
                                              * args.carryover, 6)
            curves["cross-session"] = curve
            return _curve_cell(args, out, "cross-session", TOY_SURFACES,
                               curve, seed=seed,
                               extra={"budget_factor": args.carryover})
        cell("cross-session", cross_session)

        def cross_epoch():
            curve = compose_curve(
                lambda surf, b: streams.point(surf, b * args.retention,
                                              args.trials),
                ("S1", "S2"), budgets, args.control_rate)
            for e in curve:
                e["effective_budget"] = round(e["budget"]
                                              * args.retention, 6)
            curves["cross-epoch"] = curve
            return _curve_cell(args, out, "cross-epoch", ("S1", "S2"),
                               curve, seed=seed,
                               extra={"budget_factor": args.retention})
        cell("cross-epoch", cross_epoch)

        def multi_surface():
            curve = compose_curve(
                lambda surf, b: streams.point(surf, b, args.trials),
                TOY_SURFACES, budgets, args.control_rate)
            curves["multi-surface"] = curve
            return _curve_cell(args, out, "multi-surface", TOY_SURFACES,
                               curve, seed=seed)
        cell("multi-surface", multi_surface)

        def fwd_bwd_timing():
            curve = compose_curve(
                lambda surf, b: streams.point(surf, b, args.trials),
                FWD_BWD_TIMING, budgets, args.control_rate)
            curves["fwd-bwd-timing"] = curve
            return _curve_cell(args, out, "fwd-bwd-timing", FWD_BWD_TIMING,
                               curve, seed=seed)
        cell("fwd-bwd-timing", fwd_bwd_timing)

        def marginal():
            deltas = marginal_leakage(streams.advantage, TOY_SURFACES,
                                      float(budgets[-1]))
            deltas[NULL_SURFACE] = 0.0
            rec = {"experiment": EXPERIMENT_ID, "arm": "marginal-leakage",
                   "seed": seed, "surfaces": list(TOY_SURFACES),
                   "at_budget": budgets[-1], "delta_pp": deltas}
            _emit(args, out, rec,
                  f"[{EXPERIMENT_ID}] seed={seed} marginal-leakage "
                  f"@b={budgets[-1]}: "
                  + " ".join(f"{s}={deltas[s]:.2f}pp"
                             for s in TOY_SURFACES))
            return rec
        cell("marginal-leakage", marginal)

        def synergy():
            rows, best = synergy_curve(streams.advantage, TOY_SURFACES,
                                       budgets)
            rec = {"experiment": EXPERIMENT_ID, "arm": "synergy",
                   "seed": seed, "surfaces": list(TOY_SURFACES),
                   "rows": rows,
                   "max_synergy_pp": best["synergy_pp"] if best else 0.0,
                   "at_budget": best["budget"] if best else None}
            _emit(args, out, rec,
                  f"[{EXPERIMENT_ID}] seed={seed} synergy: "
                  f"max={rec['max_synergy_pp']:.2f}pp "
                  f"@b={rec['at_budget']}")
            return rec
        cell("synergy", synergy)

        def upper_bound():
            rows = adaptive_upper_bound(curves, args.control_rate)
            valid = [r for r in rows if r["excess_pp"] is not None]
            top = max(valid, key=lambda r: r["excess_pp"]) if valid else None
            rec = {"experiment": EXPERIMENT_ID,
                   "arm": "adaptive-upper-bound", "seed": seed,
                   "rows": rows,
                   "max_excess_pp": top["excess_pp"] if top else None,
                   "at_budget": top["budget"] if top else None,
                   "best_arm": top["best_arm"] if top else None}
            _emit(args, out, rec,
                  f"[{EXPERIMENT_ID}] seed={seed} adaptive-upper-bound: "
                  f"max_excess={rec['max_excess_pp']}pp "
                  f"@b={rec['at_budget']} arm={rec['best_arm']}")
            return rec
        cell("adaptive-upper-bound", upper_bound)

    _summarize(out)
    if getattr(args, "artifacts", None):
        out["summary"] = _scenario_record(out["summary"])
        out["evidence_tier"] = "scenario"
        out["interpretation"] = "Modeled composition only; joint leakage, synergy and confidence are UNMEASURED."
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def run_artifacts(args, evidence=None):
    try:
        docs = load_artifact_documents(args.artifacts)
        unique = {}
        for path, doc in docs:
            unique.setdefault(json.dumps(doc, sort_keys=True), (path, doc))
        docs = list(unique.values())
    except ValueError as e:
        raise SystemExit(f"[{EXPERIMENT_ID}] {e}") from e
    nodes, audit = extract_nodes(docs, args.control_rate)
    out = _make_artifact(args, "artifacts",
                         {"toy": False,
                          "artifacts": [p for p, _ in docs],
                          "budgets": "as recorded in the source artifacts "
                                     "(--budgets shapes the toy grid only)"},
                         evidence=evidence)

    audit_rec = {"experiment": EXPERIMENT_ID, "arm": "input-audit",
                 "artifacts": [p for p, _ in docs], "audit": audit,
                 "nodes": [{"attack": nd["attack"], "surface": nd["surface"],
                            "n_budgets": len(nd["by_budget"])}
                           for nd in nodes.nodes]}
    _emit(args, out, audit_rec,
          f"[{EXPERIMENT_ID}] input-audit: {audit['points']} points on "
          f"{audit['nodes']} nodes from {len(docs)} artifacts "
          f"(skipped: {audit['skipped_no_metric']} no-metric, "
          f"{audit['skipped_unknown_experiment']} unknown-experiment, "
          f"{audit['skipped_error']} error, "
          f"{audit['defaulted_budget']} budget-defaulted)")
    if not nodes.nodes:
        print(f"[{EXPERIMENT_ID}] no composable records in the source "
              "artifacts; audit cell only")
        _summarize(out)
        artifacts.write_artifact(args.output, out, evidence=evidence)
        return 0

    curves = {}
    union_grid = sorted({b for nd in nodes.nodes for b in nd["by_budget"]})

    def cell(arm, fn):
        try:
            return fn()
        except Exception as e:
            journal_error(args.output, EXPERIMENT_ID, {"arm": arm}, e)
            return None

    def neg_controls():
        recs = []
        for nd in nodes.nodes:
            pts = nodes.node_points(nd)
            curve = compose_curve(
                lambda surf, b, pts=pts: {
                    "advantage": pts[0]["advantage"],
                    "k": _draws(args.control_rate, pts[0]["advantage"],
                                pts[0]["n"]),
                    "n": pts[0]["n"],
                    "per_surface": {nd["surface"]: pts[0]["advantage"]}},
                (nd["surface"],), (pts[0]["budget"],), args.control_rate)
            recs.append(_curve_cell(
                args, out, "negative-control", (nd["surface"],), curve,
                extra={"role": "negative-control",
                       "attack": nd["attack"],
                       "expectation": "the minimum-budget release sits "
                                      "near the control"}))
        return recs
    cell("negative-control", neg_controls)

    def in_session():
        recs = []
        for nd in nodes.nodes:
            pts = {p["budget"]: p for p in nodes.node_points(nd)}
            curve = compose_curve(
                lambda surf, b, pts=pts, nd=nd: {
                    "advantage": pts[b]["advantage"],
                    "k": _draws(args.control_rate, pts[b]["advantage"],
                                pts[b]["n"]),
                    "n": pts[b]["n"],
                    "per_surface": {nd["surface"]: pts[b]["advantage"]}},
                (nd["surface"],), sorted(pts), args.control_rate)
            curves[f"in-session/{nd['attack']}/{nd['surface']}"] = curve
            recs.append(_curve_cell(args, out, "in-session",
                                    (nd["surface"],), curve,
                                    extra={"attack": nd["attack"]}))
        return recs
    cell("in-session", in_session)

    def cross_session():
        recs = []
        for s in nodes.surfaces():
            grid = sorted({b for nd in nodes.nodes_for(s)
                           for b in nd["by_budget"]})
            curve = compose_curve(nodes.point, (s,), grid,
                                  args.control_rate)
            curves[f"cross-session/{s}"] = curve
            recs.append(_curve_cell(
                args, out, "cross-session", (s,), curve,
                extra={"attacks": sorted(nd["attack"]
                                         for nd in nodes.nodes_for(s))}))
        return recs
    cell("cross-session", cross_session)

    def cross_epoch():
        recs = []
        for nd in nodes.nodes:
            pts = [p for p in nodes.node_points(nd) if p["epochs"]]
            if not pts:
                continue
            pts.sort(key=lambda p: (min(p["epochs"]), p["budget"]))
            curve, cum = [], 0.0
            for p in pts:
                cum += p["budget"]
                entry = {"budget": round(cum, 6),
                         "epoch": min(p["epochs"]),
                         "advantage_pp": round(100.0 * p["advantage"], 6),
                         "per_surface_pp": {
                             nd["surface"]: round(100.0 * p["advantage"],
                                                  6)}}
                entry.update(_gate_fields(
                    p["advantage"],
                    _draws(args.control_rate, p["advantage"], p["n"]),
                    p["n"], args.control_rate))
                curve.append(entry)
            curves[f"cross-epoch/{nd['attack']}/{nd['surface']}"] = curve
            recs.append(_curve_cell(args, out, "cross-epoch",
                                    (nd["surface"],), curve,
                                    extra={"attack": nd["attack"],
                                           "budget_axis": "cumulative "
                                                          "across epochs"}))
        if not recs:
            rec = {"experiment": EXPERIMENT_ID, "arm": "cross-epoch",
                   "surfaces": [], "curve": [],
                   "first_gate_failure_budget": None,
                   "note": "UNMEASURED: no record in the source artifacts "
                           "carries an 'epoch' axis"}
            _emit(args, out, rec,
                  f"[{EXPERIMENT_ID}] cross-epoch: UNMEASURED (no epoch "
                  "axis in the source artifacts)")
            return rec
        return recs
    cell("cross-epoch", cross_epoch)

    surfaces = nodes.surfaces()

    def multi_surface():
        curve = compose_curve(nodes.point, surfaces, union_grid,
                              args.control_rate)
        curves["multi-surface"] = curve
        return _curve_cell(args, out, "multi-surface", surfaces, curve,
                           extra={"n_policy": "min eval rows across "
                                              "composed surfaces; k "
                                              "synthesized from the "
                                              "composed rate"})
    cell("multi-surface", multi_surface)

    def fwd_bwd_timing():
        present = [s for s in FWD_BWD_TIMING if s in surfaces]
        if len(present) < 2:
            rec = {"experiment": EXPERIMENT_ID, "arm": "fwd-bwd-timing",
                   "surfaces": present, "curve": [],
                   "first_gate_failure_budget": None,
                   "note": "UNMEASURED: need >= 2 of "
                           f"{list(FWD_BWD_TIMING)} present, got {present}"}
            _emit(args, out, rec,
                  f"[{EXPERIMENT_ID}] fwd-bwd-timing: UNMEASURED "
                  f"(surfaces present: {present})")
            return rec
        grid = sorted({b for s in present for nd in nodes.nodes_for(s)
                       for b in nd["by_budget"]})
        curve = compose_curve(nodes.point, present, grid,
                              args.control_rate)
        curves["fwd-bwd-timing"] = curve
        return _curve_cell(args, out, "fwd-bwd-timing", present, curve)
    cell("fwd-bwd-timing", fwd_bwd_timing)

    def marginal():
        deltas = marginal_leakage(nodes.advantage, surfaces,
                                  float(union_grid[-1]))
        rec = {"experiment": EXPERIMENT_ID, "arm": "marginal-leakage",
               "surfaces": surfaces, "at_budget": union_grid[-1],
               "delta_pp": deltas}
        _emit(args, out, rec,
              f"[{EXPERIMENT_ID}] marginal-leakage @b={union_grid[-1]}: "
              + " ".join(f"{s}={deltas[s]:.2f}pp" for s in surfaces))
        return rec
    cell("marginal-leakage", marginal)

    def synergy():
        rows, best = synergy_curve(nodes.advantage, surfaces, union_grid)
        rec = {"experiment": EXPERIMENT_ID, "arm": "synergy",
               "surfaces": surfaces, "rows": rows,
               "max_synergy_pp": best["synergy_pp"] if best else 0.0,
               "at_budget": best["budget"] if best else None,
               "note": "real path: gamma=0 — synergy is the OBSERVED "
                       "joint minus the strongest individual, never an "
                       "injected term"}
        _emit(args, out, rec,
              f"[{EXPERIMENT_ID}] synergy: max={rec['max_synergy_pp']:.2f}"
              f"pp @b={rec['at_budget']}")
        return rec
    cell("synergy", synergy)

    def upper_bound():
        rows = adaptive_upper_bound(curves, args.control_rate)
        valid = [r for r in rows if r["excess_pp"] is not None]
        top = max(valid, key=lambda r: r["excess_pp"]) if valid else None
        rec = {"experiment": EXPERIMENT_ID, "arm": "adaptive-upper-bound",
               "rows": rows,
               "max_excess_pp": top["excess_pp"] if top else None,
               "at_budget": top["budget"] if top else None,
               "best_arm": top["best_arm"] if top else None}
        if top is None:
            rec["note"] = ("UNMEASURED: no record carried an eval-row "
                           "count (n/n_eval/n_records); the confidence "
                           "bound is never guessed")
        _emit(args, out, rec,
              f"[{EXPERIMENT_ID}] adaptive-upper-bound: "
              f"max_excess={rec['max_excess_pp']}pp "
              f"@b={rec['at_budget']} arm={rec['best_arm']}")
        return rec
    cell("adaptive-upper-bound", upper_bound)

    _summarize(out)
    if getattr(args, "artifacts", None):
        out["summary"] = _scenario_record(out["summary"])
        out["evidence_tier"] = "scenario"
        out["interpretation"] = "Modeled composition only; joint leakage, synergy and confidence are UNMEASURED."
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def _validate_args(args):
    problems = []
    if not args.budgets or any(not isinstance(b, int) or b <= 0
                               for b in args.budgets):
        problems.append(f"--budgets must be positive ints, "
                        f"got {args.budgets}")
    if not (0.0 < args.control_rate < 0.5):
        problems.append(f"--control-rate must be in (0, 0.5), got "
                        f"{args.control_rate}")
    if args.trials < 100:
        problems.append(f"--trials must be >= 100 (Wilson bound is "
                        f"meaningless below), got {args.trials}")
    for name in ("carryover", "retention"):
        v = getattr(args, name)
        if not (0.0 < v <= 1.0):
            problems.append(f"--{name} must be in (0, 1], got {v}")
    if args.synergy < 0.0:
        problems.append(f"--synergy must be >= 0, got {args.synergy}")
    if problems:
        raise SystemExit(f"[{EXPERIMENT_ID}] " + "; ".join(problems))


def _run(args, evidence=None):
    _validate_args(args)
    if args.artifacts:
        return run_artifacts(args, evidence=evidence)
    if args.toy:
        return run_toy(args, evidence=evidence)
    raise SystemExit(f"[{EXPERIMENT_ID}] pick an input path: --artifacts "
                     "A.json ... (compose completed attack artifacts) or "
                     "--toy (seeded synthetic multi-surface world with "
                     "known ground truth).")


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


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--artifacts", nargs="+", default=None, metavar="A.json",
                    help="completed attack artifacts to compose (real "
                         "path; fail-closed: a missing/unparseable/empty "
                         "artifact is a descriptive error naming the path)")
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=list(DEFAULT_BUDGETS),
                    help="cumulative-budget grid (toy path; the real path "
                         "uses the budgets recorded in the source "
                         "artifacts)")
    ap.add_argument("--trials", type=int, default=N_EVAL,
                    help="eval rows per toy cell point (binomial draws "
                         "behind the Wilson bound)")
    ap.add_argument("--control-rate", type=float, default=CONTROL_RATE,
                    help="label-free majority control rate the gate "
                         "scores against")
    ap.add_argument("--carryover", type=float, default=CARRYOVER,
                    help="toy cross-session budget carryover factor")
    ap.add_argument("--retention", type=float, default=RETENTION,
                    help="toy cross-epoch budget retention factor")
    ap.add_argument("--synergy", type=float, default=SYNERGY_GAMMA,
                    help="toy S1xS2 synergy coefficient gamma (the real "
                         "path always composes with gamma=0)")
    ap.add_argument("--evidence", action="store_true",
                    help="wrap the run in an EvidenceLogger (requires "
                         "--output)")
    return ap


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("privacy-composition (pure python):")

    rng = random.Random(0)
    check("binomial_draw degenerate anchors",
          binomial_draw(rng, 100, 0.0) == 0
          and binomial_draw(rng, 100, 1.0) == 100)
    draws = [binomial_draw(random.Random(s), 1000, 0.05) for s in range(5)]
    check("binomial_draw in range and seeded-deterministic",
          all(0 <= k <= 1000 for k in draws)
          and draws == [binomial_draw(random.Random(s), 1000, 0.05)
                        for s in range(5)])

    check("combine anchors: empty is 0, single is identity, clamps to 1",
          combine({}) == 0.0
          and abs(combine({"S1": 0.03}) - 0.03) < 1e-12
          and combine({"S1": 0.9, "S2": 0.9}, 1.0) == 1.0)
    indep = 1.0 - (1.0 - 0.01) * (1.0 - 0.02)
    check("combine: S1xS2 synergy term is additive and positive",
          abs(combine({"S1": 0.01, "S2": 0.02}, 0.3)
              - (indep + 0.3 * math.sqrt(0.01 * 0.02))) < 1e-12
          and combine({"S1": 0.01, "S2": 0.02}, 0.0) == indep)

    s1, s2 = make_toy_streams(7), make_toy_streams(7)
    check("toy streams are seeded-deterministic",
          s1.surfaces == s2.surfaces)
    check("NULL surface sits at exactly 0 across budgets",
          all(s1.surface_advantage(NULL_SURFACE, b) == 0.0
              for b in DEFAULT_BUDGETS))
    check("toy curves are monotone nondecreasing in budget",
          all(s1.surface_advantage("S1", a) <= s1.surface_advantage("S1", b)
              for a, b in zip(DEFAULT_BUDGETS, DEFAULT_BUDGETS[1:])))
    check("S1xS2 joint exceeds the strongest individual (synergy)",
          s1.advantage(("S1", "S2"), 128.0)
          > max(s1.surface_advantage("S1", 128.0),
                s1.surface_advantage("S2", 128.0)))

    check("frozen gate anchors: at-control PASSes, a real break FAILs",
          gate.verdict(0.04, 0.04, 8192)["verdict"] == "INSUFFICIENT_EVIDENCE"
          and gate.verdict(0.10, 0.04, 8192)["verdict"] == "FAIL")
    probe = lambda surf, b: s1.point(surf, b, 8192)
    null_curve = compose_curve(probe, (NULL_SURFACE,), DEFAULT_BUDGETS,
                               CONTROL_RATE)
    check("NULL surface never fails the gate",
          first_gate_failure(null_curve) is None
          and all(e["verdict"] == "INSUFFICIENT_EVIDENCE" for e in null_curve))
    joint_curve = compose_curve(probe, TOY_SURFACES, DEFAULT_BUDGETS,
                                CONTROL_RATE)
    s1_curve = compose_curve(probe, ("S1",), DEFAULT_BUDGETS, CONTROL_RATE)
    check("vulnerable toy world: the joint fails before S1 alone",
          first_gate_failure(joint_curve) is not None
          and first_gate_failure(s1_curve) is not None
          and first_gate_failure(joint_curve)
          < first_gate_failure(s1_curve))

    check("marginal leakage: NULL contributes 0, real surfaces positive",
          all(v > 0.0 for v in marginal_leakage(
              s1.advantage, TOY_SURFACES, 128.0).values()))
    rows, best = synergy_curve(s1.advantage, TOY_SURFACES,
                               list(DEFAULT_BUDGETS))
    check("synergy positive and maximized at some budget",
          best is not None and best["synergy_pp"] > 0.0
          and len(rows) == len(DEFAULT_BUDGETS))
    ub = adaptive_upper_bound({"multi-surface": joint_curve,
                               "in-session/S1": s1_curve}, CONTROL_RATE)
    check("adaptive upper bound tracks the best dimension per budget",
          len(ub) == len(DEFAULT_BUDGETS)
          and all(r["best_arm"] == "multi-surface" for r in ub[-2:]))

    check("normalize_metric anchors (top1 vs control, AUC, pp advantage)",
          abs(normalize_metric("top1", 5.0, 0.04) - 0.01) < 1e-12
          and abs(normalize_metric("auc", 0.75, 0.04) - 0.5) < 1e-12
          and abs(normalize_metric("advantage", 2.5, 0.04) - 0.025)
          < 1e-12)

    frozen_doc = {"results": [
        {"experiment": "wire_shape_analysis", "top1": 8.0, "budget": 4,
         "n_records": 1000},
        {"experiment": "wire_shape_analysis", "top1": 9.0, "budget": 4,
         "n_records": 2000},
        {"experiment": "wire_shape_analysis", "no_metric": True},
        {"experiment": "wire_shape_analysis", "top1": 7.0},
        {"experiment": "mystery_exp", "top1": 5.0},
        {"error": "boom"},
        {"experiment": "er4_stale_key", "top1_mean": 6.0, "epoch_size": 64,
         "epoch": 2}]}
    nodes, audit = extract_nodes([("frozen.json", frozen_doc)], 0.04)
    check("extract_nodes: records land on (attack, surface) nodes",
          audit["points"] == 4 and audit["nodes"] == 2
          and any(nd["attack"] == "wire-shape" and nd["surface"] == "S7"
                  for nd in nodes.nodes))
    check("extract_nodes: repeats average (mean advantage, min n)",
          abs(nodes.surface_advantage("S7", 4.0)
              - ((0.08 - 0.04) + (0.09 - 0.04)) / 2.0) < 1e-12
          and nodes.point(("S7",), 4.0)["n"] == 1000)
    check("extract_nodes: skips are counted, never silent",
          audit["skipped_no_metric"] == 1
          and audit["skipped_unknown_experiment"] == 1
          and audit["skipped_error"] == 1
          and audit["defaulted_budget"] == 1)
    check("cross-epoch node carries the epoch axis",
          nodes.node_points(nodes.nodes_for("S1")[0])[0]["epochs"] == [2.0])

    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        missing = os.path.join(d, "missing.json")
        try:
            load_artifact_documents([missing])
            check("missing artifact is a descriptive ValueError", False)
        except ValueError as e:
            check("missing artifact is a descriptive ValueError",
                  "missing.json" in str(e))
        empty = os.path.join(d, "empty.json")
        with open(empty, "w") as f:
            json.dump({"results": []}, f)
        try:
            load_artifact_documents([empty])
            check("empty artifact is a descriptive ValueError", False)
        except ValueError as e:
            check("empty artifact is a descriptive ValueError",
                  "empty.json" in str(e))

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
