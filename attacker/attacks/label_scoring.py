#!/usr/bin/env python3
"""label-scoring (training mode) — norm-based property inference from bwd
seam gradients (the norm-based scoring line, arXiv:2102.08504 "Label
Leakage and Protection in Two-party Split Learning", transplanted to the
split-LLM seam).

Attack fact: the per-row L2 norm of the gradients the server-side body
sends back across the seam carries information about the underlying
tokens. The original paper scores classification labels; the honest LM
analogue implemented here is PROPERTY inference over sequences:
--property "contains-token:<id>" labels each captured sequence by whether
its canonical row contains the token, and the per-SEQUENCE gradient norm
(sum of the per-position L2 norms over the sequence) is the attack score.

Scoring convention (identical to lira_membership, documented): rank-based
(Mann–Whitney) AUC = P(norm_present > norm_absent) with tie-averaged
ranks; TPR at FPR <= 1% / 0.1% by the conservative achievable-grid sweep;
and an attacker.gate verdict with recovery = AUC vs the 0.5 chance control
(the membership analogue of the label-free majority control; AUC is not
binomial, so the Wilson bound is a scoring convention — n =
min(n_present, n_absent)). Direction convention: higher norm = property
present; an AUC materially BELOW 0.5 is equally a leak (flip the score).

Modes:
  --toy            synthetic norms from two mixtures (property present /
                   absent with a planted norm gap --toy-gap), including the
                   null world (gap exactly 0 — the honesty arm: AUC must
                   land at ~0.5 and PASS the gate).
  --capture-dir D  real bwd seam captures (wire-training schema). Per-step
                   INFO cells report the norm distribution (mean/std/min/
                   max). With --canonical-pt C --property contains-token:ID
                   the run additionally scores one pooled property-inference
                   cell (AUC + TPR@low-FPR + gate).

Usage:
    python -m attacker --mode training --attack label-scoring --help
    python -m attacker --mode training --attack label-scoring --toy \
        --quick --output /tmp/ls.json
    python -m attacker --mode training --attack label-scoring \
        --capture-dir CAPS --canonical-pt CAPS/canonical.pt \
        --property contains-token:220 --output ls.json --evidence
"""

import argparse
import math
import random

try:
    import torch
except ImportError:
    torch = None

from .. import artifacts, gate
from .common import add_common_args, journal_error, require_torch
from .lira_membership import FPR_TARGETS, rank_auc, tpr_at_fpr

EXPERIMENT_ID = "norm_based_scoring"
MODES = ("training",)
REQUIRES_LABELS = False
DESCRIPTION = ("Norm-based scoring (arXiv:2102.08504 lineage): per-sequence "
               "L2 norm of bwd seam gradients leaks sequence properties "
               "(contains-token:<id>); rank AUC + TPR@low-FPR + gate "
               "verdict over the 0.5 AUC control, with a null-world "
               "honesty arm")

SCHEMA = "dtraining.attacker.norm_based_scoring.v1"
PROPERTY_KINDS = ("contains-token",)

THREAT_MODEL = (
    "honest-but-curious seam observer of bwd seam gradients (the tensors "
    "the server-side body returns across the split seam during "
    "fine-tuning). The attack score is the per-sequence gradient norm; "
    "canonical tokens label rows for evaluation only — evaluation truth, "
    "never an attack input (label-free attack).")

INTERPRETATION = (
    "AUC is rank-based (Mann–Whitney, tie-averaged): P(norm_present > "
    "norm_absent); TPR@FPR is the conservative achievable-grid value "
    "(never interpolated past the sample resolution); the gate verdict "
    "uses recovery = AUC vs the 0.5 chance control with n = "
    "min(n_present, n_absent) — a scoring convention (AUC is not "
    "binomial), the same one lira_membership documents. Direction: "
    "higher norm = property present. The null toy world (gap exactly 0) "
    "must score AUC ~= 0.5 and PASS the gate; anything else is a "
    "machinery bug.")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--capture-dir", default=None,
                    help="wire-capture directory (wire-training schema, "
                         "phase='bwd' seam gradients)")
    ap.add_argument("--canonical-pt", default=None,
                    help="canonical.pt ground-truth input_ids (row-aligned "
                         "to the fwd captures step-major; required by "
                         "--property, used for evaluation labeling only)")
    ap.add_argument("--property", default=None, metavar="SPEC",
                    help="sequence property to infer: 'contains-token:<id>' "
                         "(label = the canonical row contains the token). "
                         "Without it, only norm-distribution INFO cells "
                         "are reported.")
    ap.add_argument("--worlds", choices=("both", "separable", "null"),
                    default="both", help="toy worlds to generate")
    ap.add_argument("--toy-n", type=int, default=100000,
                    help="toy sequences per group (present/absent); the "
                         "default resolves the 1.0 pp gate at the 0.5 "
                         "control (Wilson floor ~0.4 pp), so a true null "
                         "PASSes with margin")
    ap.add_argument("--toy-gap", type=float, default=4.0,
                    help="separable world planted norm gap, in noise-sigma "
                         "units (null world is always exactly 0)")
    ap.add_argument("--evidence", action="store_true",
                    help="seal the run with an attacker.evidence chain "
                         "(<output>.evidence.json[l]); requires --output")
    return ap


def _make_evidence(args):
    """--evidence wiring: one EvidenceLogger threaded through
    make_artifact/write_artifact (contract: default None = no change)."""
    if not getattr(args, "evidence", False):
        return None
    if not args.output:
        raise SystemExit(f"[{EXPERIMENT_ID}] --evidence requires --output")
    from ..evidence import EvidenceLogger
    return EvidenceLogger(args.output, node_role="attacker/label-scoring")


def parse_property(spec):
    """'contains-token:<id>' -> ("contains-token", id). Descriptive
    ValueError on any drift (run() surfaces it as SystemExit)."""
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(f"invalid property {spec!r}: expected "
                         "'contains-token:<id>' (registered kinds: "
                         f"{', '.join(PROPERTY_KINDS)})")
    kind, _, arg = spec.partition(":")
    if kind not in PROPERTY_KINDS:
        raise ValueError(f"unknown property kind {kind!r} (registered: "
                         f"{', '.join(PROPERTY_KINDS)})")
    try:
        tok = int(arg)
    except ValueError:
        raise ValueError(f"property {spec!r}: token id {arg!r} is not an "
                         "integer") from None
    if tok < 0:
        raise ValueError(f"property {spec!r}: token id must be >= 0")
    return kind, tok


def label_rows(rows, prop):
    """Boolean property label per canonical row (list of token-id lists)."""
    kind, tok = prop
    if kind == "contains-token":
        return [tok in row for row in rows]
    raise ValueError(f"unknown property kind {kind!r}")


def sequence_norm(positions):
    """Per-SEQUENCE norm: the sum over positions of the per-position L2
    norm (positions = the sequence's rows in one bwd capture)."""
    return sum(math.sqrt(sum(x * x for x in row)) for row in positions)


def norm_metrics(norms_present, norms_absent, n_seeds=None):
    """The full metric bundle for one scored population: rank AUC,
    TPR@low-FPR, gate verdict over the 0.5 AUC control (documented
    scoring convention: n = min group size)."""
    if len(norms_present) < 1 or len(norms_absent) < 1:
        raise ValueError(f"need >=1 present and >=1 absent sequences "
                         f"(got {len(norms_present)}/{len(norms_absent)})")
    auc = rank_auc(norms_present, norms_absent)
    return {"n_present": len(norms_present), "n_absent": len(norms_absent),
            "auc": round(auc, 6),
            "tpr_at_fpr": {str(t): tpr_at_fpr(norms_present, norms_absent, t)
                           for t in FPR_TARGETS},
            "gate": gate.verdict(auc, 0.5,
                                 min(len(norms_present), len(norms_absent)),
                                 n_seeds=n_seeds)}


def norm_stats(norms):
    """Norm-distribution INFO bundle (mean/std/min/max; sample std)."""
    n = len(norms)
    mu = sum(norms) / max(1, n)
    var = sum((v - mu) ** 2 for v in norms) / max(1, n - 1)
    return {"n_seq": n, "norm_mean": round(mu, 6),
            "norm_std": round(math.sqrt(var), 6),
            "norm_min": round(min(norms), 6) if norms else None,
            "norm_max": round(max(norms), 6) if norms else None}


def make_toy_norms(world, n_per_group, seed, gap):
    """Synthetic per-sequence norms: present ~ |N(gap, 1)|, absent ~
    |N(0, 1)|. Norms are non-negative magnitudes, hence the fold; the null
    world passes gap = 0 exactly (same distribution both sides, AUC must
    land at ~0.5 — the honesty arm)."""
    rng = random.Random(seed)
    present = [abs(rng.gauss(gap, 1.0)) for _ in range(n_per_group)]
    absent = [abs(rng.gauss(0.0, 1.0)) for _ in range(n_per_group)]
    return present, absent


def toy_cell(args, world, seed):
    gap = args.toy_gap if world == "separable" else 0.0
    present, absent = make_toy_norms(world, args.toy_n,
                                     args.seed + 1000 * seed, gap)
    return {"world": world, "seed": seed, "gap": gap,
            **norm_metrics(present, absent, n_seeds=len(args.seeds))}


def _run_toy(args, ev):
    out = artifacts.make_artifact(
        SCHEMA,
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "worlds": args.worlds, "toy_n": args.toy_n,
         "toy_gap": args.toy_gap, "seeds": args.seeds,
         "fpr_targets": list(FPR_TARGETS)},
        THREAT_MODEL + " Toy tier: synthetic two-mixture norms with a "
        "planted gap; validates the AUC/TPR/gate plumbing (including the "
        "null world), not a real model.",
        interpretation=INTERPRETATION,
        evidence=ev)
    worlds = ("separable", "null") if args.worlds == "both" \
        else (args.worlds,)
    cells = []
    for world in worlds:
        for seed in args.seeds:
            cell = {"experiment": EXPERIMENT_ID, "kind": "toy_property"}
            try:
                cell.update(toy_cell(args, world, seed))
            except Exception as e:
                journal_error(args.output, EXPERIMENT_ID, cell, e)
                continue
            artifacts.append_jsonl(args.output, cell)
            if ev is not None:
                ev.event("cell_result", cell)
            out["results"].append(cell)
            cells.append(cell)
            print(f"[norm-toy] world={world} seed={seed} gap={cell['gap']}: "
                  f"auc={cell['auc']} gate={cell['gate']['verdict']}")
    for world in sorted(set(c["world"] for c in cells)):
        vals = [c for c in cells if c["world"] == world]
        out["summary"].append({
            "experiment": EXPERIMENT_ID, "world": world,
            "auc_mean": sum(c["auc"] for c in vals) / len(vals),
            "excess_pp": max(c["gate"]["excess_pp"] for c in vals),
            "gate_verdicts": [c["gate"]["verdict"] for c in vals],
            "n_cells": len(vals)})
    artifacts.write_artifact(args.output, out, evidence=ev)
    return 0


def _sequence_norms_torch(t):
    """Per-sequence norms of one bwd capture [b, s, H] (or [s, H]):
    sum over positions of the per-position L2 norm — the tensor twin of
    sequence_norm()."""
    if t.dim() == 2:
        t = t.unsqueeze(0)
    return t.float().norm(dim=-1).sum(-1).tolist()


def _run_real(args, ev, prop):
    require_torch(EXPERIMENT_ID)
    from .. import captures
    records = [(m, p) for m, p in
               captures.scan_captures(args.capture_dir, "training")
               if m["phase"] == "bwd"]
    if not records:
        raise SystemExit(f"[{EXPERIMENT_ID}] no phase='bwd' captures in "
                         f"{args.capture_dir} (norm-based scoring reads "
                         "the training-mode seam gradients)")
    canonical = None
    if prop is not None:
        if not args.canonical_pt:
            raise SystemExit(f"[{EXPERIMENT_ID}] --property requires "
                             "--canonical-pt (evaluation labeling only)")
        canonical = torch.load(args.canonical_pt, map_location="cpu").long()

    out = artifacts.make_artifact(
        SCHEMA,
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": False,
         "capture_dir": args.capture_dir, "canonical_pt": args.canonical_pt,
         "property": args.property,
         "fpr_targets": list(FPR_TARGETS)},
        THREAT_MODEL, interpretation=INTERPRETATION, evidence=ev)

    cursor = 0
    all_norms = []
    pairs = []
    n_info = 0
    for meta, pt in records:
        cell = {"experiment": EXPERIMENT_ID, "kind": "norm_stats",
                "step": meta.get("step"), "epoch": meta.get("epoch")}
        try:
            norms = _sequence_norms_torch(captures.load_tensor(pt))
            if not norms:
                raise ValueError("empty bwd capture")
            cell.update(norm_stats(norms))
            if canonical is not None:
                b = len(norms)
                rows = canonical[cursor:cursor + b]
                if rows.shape[0] != b:
                    raise ValueError(
                        f"canonical/capture misalignment: bwd capture at "
                        f"step {meta.get('step')} has batch {b} but "
                        f"canonical has {rows.shape[0]} rows left at "
                        f"cursor {cursor}")
                labels = label_rows(rows.tolist(), prop)
                pairs.extend(zip(norms, labels))
                cursor += b
            all_norms.extend(norms)
        except Exception as e:
            journal_error(args.output, EXPERIMENT_ID, cell, e)
            continue
        artifacts.append_jsonl(args.output, cell)
        if ev is not None:
            ev.event("cell_result", cell)
        out["results"].append(cell)
        n_info += 1
        print(f"[norm] step={meta.get('step')}: n_seq={cell['n_seq']} "
              f"mean={cell['norm_mean']} std={cell['norm_std']}")

    if canonical is not None:
        cell = {"experiment": EXPERIMENT_ID, "kind": "property_score",
                "property": args.property}
        try:
            present = [n for n, l in pairs if l]
            absent = [n for n, l in pairs if not l]
            cell.update(norm_metrics(present, absent))
        except Exception as e:
            journal_error(args.output, EXPERIMENT_ID, cell, e)
        else:
            artifacts.append_jsonl(args.output, cell)
            if ev is not None:
                ev.event("cell_result", cell)
            out["results"].append(cell)
            print(f"[norm] property={args.property}: auc={cell['auc']} "
                  f"tpr@1%={cell['tpr_at_fpr']['0.01']['tpr']} "
                  f"gate={cell['gate']['verdict']}")
            out["summary"].append({
                "experiment": EXPERIMENT_ID, "kind": "property_score",
                "property": args.property, "auc": cell["auc"],
                "excess_pp": cell["gate"]["excess_pp"],
                "gate_verdicts": [cell["gate"]["verdict"]],
                "n_cells": 1})
    out["summary"].append({"experiment": EXPERIMENT_ID,
                           "kind": "norm_stats", "n_info_cells": n_info,
                           **norm_stats(all_norms)})
    artifacts.write_artifact(args.output, out, evidence=ev)
    return 0


def run(args):
    ev = _make_evidence(args)
    prop = None
    if args.property is not None:
        try:
            prop = parse_property(args.property)
        except ValueError as e:
            raise SystemExit(f"[{EXPERIMENT_ID}] {e}") from None
    if args.toy:
        if args.quick:
            args.seeds = [0]
            args.toy_n = min(args.toy_n, 500)
        return _run_toy(args, ev)
    if not args.capture_dir:
        raise SystemExit(f"[{EXPERIMENT_ID}] pass --toy, or --capture-dir "
                         "D [--canonical-pt C --property contains-token:ID].")
    return _run_real(args, ev, prop)


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import json
    import os
    import tempfile

    print("label-scoring norm core (pure python):")
    check("parse_property accepts contains-token:<id>",
          parse_property("contains-token:220") == ("contains-token", 220))
    for bad in ("contains-token", "has-token:1", "contains-token:x",
                "contains-token:-1", "", None):
        try:
            parse_property(bad)
            check(f"parse_property({bad!r}) raises", False)
        except ValueError:
            check(f"parse_property({bad!r}) raises", True)
    check("label_rows marks rows containing the token",
          label_rows([[1, 2, 3], [4, 5], [2, 7]],
                     ("contains-token", 2)) == [True, False, True])
    check("sequence_norm sums per-position L2 norms",
          abs(sequence_norm([[3.0, 4.0], [0.0, 0.0], [1.0, 2.0]])
              - (5.0 + 0.0 + math.sqrt(5.0))) < 1e-12)

    present, absent = make_toy_norms("separable", 2000, 0, 4.0)
    check("separable toy world scores AUC > 0.99",
          norm_metrics(present, absent)["auc"] > 0.99)
    present, absent = make_toy_norms("null", 100000, 1, 0.0)
    null = norm_metrics(present, absent)
    check("null toy world lands at AUC ~= 0.5 (honesty arm)",
          abs(null["auc"] - 0.5) < 0.01)
    check("null world PASSes the gate at the default toy-n "
          "(floor resolved)", null["gate"]["verdict"] == "INSUFFICIENT_EVIDENCE")
    check("norm_metrics refuses an empty group (journaled, never raised "
          "from run)",
          _raises_valueerror(lambda: norm_metrics([], [1.0])))

    check("parser builds torch-less",
          build_parser().parse_args([]).property is None)
    ns = build_parser().parse_args([])
    ns.mode = "training"
    try:
        run(ns)
        check("no --toy/--capture-dir is a descriptive error", False)
    except SystemExit as e:
        check("no --toy/--capture-dir is a descriptive error",
              "--capture-dir" in str(e))
    try:
        run(build_parser().parse_args(["--toy", "--property", "bogus"]))
        check("a malformed --property is a descriptive error", False)
    except SystemExit as e:
        check("a malformed --property is a descriptive error",
              "contains-token" in str(e))
    if torch is None:
        ns2 = build_parser().parse_args(["--capture-dir", "/nonexistent"])
        ns2.mode = "training"
        try:
            run(ns2)
            check("real tier is torch-guarded", False)
        except SystemExit as e:
            check("real tier is torch-guarded", "requires torch" in str(e))

    with tempfile.TemporaryDirectory() as d:
        ns = build_parser().parse_args(
            ["--toy", "--quick", "--evidence",
             "--output", os.path.join(d, "ls.json")])
        ns.mode = "training"
        check("toy run end to end torch-free", run(ns) == 0)
        art = json.load(open(os.path.join(d, "ls.json")))
        check("toy artifact has separable + null cells with gate verdicts",
              sorted(c["world"] for c in art["results"])
              == ["null", "separable"]
              and all(c["gate"]["verdict"] in ("PASS", "FAIL", "INSUFFICIENT_EVIDENCE")
                      for c in art["results"]))
        check("journal lines match result cells",
              len(open(os.path.join(d, "ls.json.jsonl")).read()
                  .splitlines()) == len(art["results"]))
        from ..evidence import verify_evidence
        good, problems = verify_evidence(
            os.path.join(d, "ls.json.evidence.json"))
        check("evidence chain verifies", good and problems == [])

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def _raises_valueerror(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(self_test())
