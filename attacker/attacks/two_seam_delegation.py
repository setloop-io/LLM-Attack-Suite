#!/usr/bin/env python3
"""two-seam-delegation (training + inference modes) — chained reconstruction
across the TWO exposed seams of a two-segment delegation topology
(cloud A / private island / cloud B; SplitPlan "4,8" -> head | body | tail),
measuring the COMPOUNDED reconstruction error versus the single-segment
(one-seam) baseline.

Threat model (WHITE-BOX toy note, mirrored in every artifact): the victim
runs three chained segments — H1 (cloud A head), M (private-island middle
nonlinearity), H2 (cloud B tail input stage) — and the attacker observes
BOTH wire seams: seam1 = H1(x) at the A->island boundary and
seam2 = H2(M(H1(x))) at the island->B boundary. As in seam-inversion, the
attacker is granted the token embedding matrix E (public base weights) and,
in the toy tier, the public segment maps H1/H2 and the middle codebook
(the toy analogue of a white-box view of the delegation contract); the
deployed middle itself is private. Canonical tokens are evaluation truth,
never an attack input (label-free attack).

Three decodes are compared per cell:
  (a) direct-seam2   single-seam decode AT seam2: ridge-invert the assumed
                     composed linear map H1 @ H2 (middle approximated as
                     identity) and cosine-NN decode over E. The private
                     nonlinearity M is unmodelled, so the error COMPOUNDS
                     across both seams — the measurement of interest.
  (b) chained        two-hop decode: ridge-invert H2 at seam2 to estimate
                     the middle activation m_hat, then cosine-NN decode
                     m_hat against the middle codebook {M(E[v] @ H1)}.
                     Uses both seams; should dominate (a) whenever M is
                     nonlinear.
  (c) control-seam1  single-segment control: decode at seam1 only
                     (ridge-invert H1, NN over E) — the one-seam baseline
                     the compounded error is reported against.

Honesty/null arm (--middle identity): with M = identity the topology
degenerates to one composed linear map and the chained error MUST equal
the single-seam (control) error within tolerance; a gap there is a bug in
the chaining machinery, not an attack effect.

Reported per cell: position-aligned token accuracy, decoded-embedding MSE,
linear-estimate MSE where the arm has one, the majority-token control, and
the frozen-gate Bonferroni-Wilson excess verdict over that control
(n = scored tokens), all via attacker.gate.

Real tier (--capture-dir): NOT producible today. splitbench's CaptureWriter
(attacker/splitbench.py) names captures wire_{seq:05d}.{pt,json} keyed off
a single seam per writer/mode — there is no seam index in the sidecar
schema, so a dual-seam (seam1 + seam2) capture layout does not exist yet.
The real tier therefore validates this descriptively and raises a
descriptive SystemExit stating the expected layout; wiring the actual
dual-seam decode is blocked on splitbench gaining per-seam capture naming
(tracked in experiments/defense_gap_scope.json).

Usage:
    python -m attacker --mode training --attack two-seam-delegation --help
    python -m attacker --mode training --attack two-seam-delegation \
        --toy --quick --output /tmp/tsd.json
"""

import argparse

from .. import artifacts, gate
from . import seam_inversion as si
from .common import add_common_args, journal_error, require_torch

EXPERIMENT_ID = "two_seam_delegation"
MODES = ("training", "inference")
REQUIRES_LABELS = False
DESCRIPTION = ("two-segment delegation (cloud A / private island / cloud B): "
               "chained inversion across BOTH exposed seams; measures "
               "compounded seam2 reconstruction error (direct vs chained "
               "decode) against the single-seam control; identity-middle "
               "null arm pins chained error == single-seam error")
SCHEMA = "dtraining.attacker.two_seam_delegation.v1"

TOY_RIDGE_LAMBDA = 1e-6
MNULL_TOL = 1e-6

THREAT_MODEL = (
    "honest-but-curious observer of BOTH delegation seams of a two-segment "
    "split (cloud A / private island / cloud B): seam1 = H1(x) and "
    "seam2 = H2(M(H1(x))). Granted: the victim embedding matrix E (public "
    "base weights, the VFLAIR-LLM VMI assumption) and, in the toy tier, the "
    "public segment maps H1/H2 and the middle codebook (white-box toy "
    "analogue of the delegation contract); the deployed middle M is "
    "private. Canonical tokens are evaluation truth, never an attack input "
    "(label-free attack).")

INTERPRETATION = (
    "direct-seam2 is the single-seam decode at the far seam with the "
    "private middle unmodelled (compounded error); chained inverts seam2 "
    "to a middle estimate and decodes against the middle codebook using "
    "both seams; control-seam1 is the single-segment baseline. The "
    "security-relevant ordering is chained >= direct on the planted "
    "nonlinear middle, with identity-middle equality as the null arm. "
    "Toy tier is a machinery check on synthetic KNOWN-map segments, not a "
    "real-model security claim; no dual-seam deployed capture exists yet "
    "(real tier is a descriptive error).")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, hidden=8)
    ap.add_argument("--capture-dir", default=None,
                    help="dual-seam wire-capture directory (NOT producible "
                         "today: splitbench CaptureWriter names captures "
                         "wire_NNNNN.{pt,json} with a single seam per "
                         "writer and no seam index in the sidecar; the "
                         "real tier validates the expected layout and "
                         "exits descriptively)")
    ap.add_argument("--middle", choices=("planted", "identity", "both"),
                    default="both",
                    help="toy middle arm: planted nonlinear MLP, identity "
                         "null arm (chained error must equal single-seam "
                         "error), or both (default)")
    ap.add_argument("--vocab", type=int, default=40, help="toy vocab size")
    ap.add_argument("--seq-len", type=int, default=16,
                    help="toy victim sequence length")
    ap.add_argument("--toy-seqs", type=int, default=32,
                    help="toy victim sequences per cell")
    ap.add_argument("--toy-cal-tokens", type=int, default=4000,
                    help="toy calibration-corpus tokens (control mode)")
    ap.add_argument("--toy-sigmas", type=float, nargs="+",
                    default=(0.0, 0.05, 0.2, 0.5),
                    help="toy seam-noise sweep applied to BOTH seams "
                         "(0 = noiseless)")
    ap.add_argument("--evidence", action="store_true",
                    help="seal the run with an attacker.evidence chain "
                         "(<output>.evidence.json[l]); requires --output")
    return ap


def _make_evidence(args):
    if not getattr(args, "evidence", False):
        return None
    if not args.output:
        raise SystemExit(f"[{EXPERIMENT_ID}] --evidence requires --output")
    from ..evidence import EvidenceLogger
    return EvidenceLogger(args.output,
                          node_role="attacker/two-seam-delegation")


def _tanh_mat(X):
    import math
    return [[math.tanh(v) for v in row] for row in X]


def toy_world(seed, vocab=40, dim=8, n_seq=32, seq_len=16, sigma=0.0,
              cal_tokens=4000, middle="planted"):
    """Synthetic two-seam world. E [vocab, dim] random; H1/H2 [dim, dim]
    random KNOWN maps (public head/tail segments); the private middle M is
    a one-hidden-layer MLP tanh(. @ Wm1) @ Wm2 ('planted') or the identity
    (null arm). Victim sequences are uniform token draws; the observed
    seams are seam1 = E[tok] @ H1 and seam2 = M(seam1) @ H2, each plus
    N(0, sigma^2) element noise. The middle codebook {M(E[v] @ H1)} is the
    white-box toy grant used by the chained decode."""
    import random
    rng = random.Random(seed)
    E = si._randn_mat(vocab, dim, rng)
    H1 = si._randn_mat(dim, dim, rng)
    H2 = si._randn_mat(dim, dim, rng)
    Wm1 = si._randn_mat(dim, dim, rng, scale=0.5)
    Wm2 = si._randn_mat(dim, dim, rng, scale=0.5)
    victim = [[rng.randrange(vocab) for _ in range(seq_len)]
              for _ in range(n_seq)]
    cal = [rng.randrange(vocab) for _ in range(cal_tokens)]

    def apply_middle(h1_rows):
        if middle == "identity":
            return [list(r) for r in h1_rows]
        return si._matmul(_tanh_mat(si._matmul(h1_rows, Wm1)), Wm2)

    def noise(rng_):
        return rng_.gauss(0.0, sigma) if sigma > 0.0 else 0.0

    seam1, seam2, mid = [], [], []
    for seq in victim:
        h1 = si._matmul([E[t] for t in seq], H1)
        m = apply_middle(h1)
        h2 = si._matmul(m, H2)
        seam1.append([[v + noise(rng) for v in row] for row in h1])
        seam2.append([[v + noise(rng) for v in row] for row in h2])
        mid.append(m)
    codebook = apply_middle(si._matmul(E, H1))
    return {"E": E, "H1": H1, "H2": H2, "victim": victim, "seam1": seam1,
            "seam2": seam2, "mid": mid, "codebook": codebook, "cal": cal,
            "vocab": vocab, "dim": dim, "sigma": sigma, "middle": middle}


def _embedding_mse(preds, trues, E):
    """Mean ||E[pred] - E[true]||^2 over decoded positions."""
    tot, n = 0.0, 0
    for p, t in zip(preds, trues):
        for a, b in zip(p, t):
            tot += sum((x - y) ** 2 for x, y in zip(E[a], E[b]))
            n += 1
    return round(tot / max(1, n), 8)


def _estimate_mse(ests, trues, E):
    """Mean ||e_hat - E[true]||^2 for arms with a linear estimate."""
    tot, n = 0.0, 0
    for rows, t in zip(ests, trues):
        for row, b in zip(rows, t):
            tot += sum((x - y) ** 2 for x, y in zip(row, E[b]))
            n += 1
    return round(tot / max(1, n), 8)


def toy_cell(args, sigma, seed, middle):
    """One toy cell: run all three decodes on the same world and score each
    with the shared seam-inversion plumbing + a gate verdict."""
    world = toy_world(args.seed + 3971 * seed, vocab=args.vocab,
                      dim=args.hidden, n_seq=args.toy_seqs,
                      seq_len=args.seq_len, sigma=sigma,
                      cal_tokens=args.toy_cal_tokens, middle=middle)
    E, victim = world["E"], world["victim"]
    control = si.majority_control_token([world["cal"]])

    D_comp = si.ridge_decode_map(si._matmul(world["H1"], world["H2"]),
                                 TOY_RIDGE_LAMBDA)
    est_direct = [si._matmul(rows, D_comp) for rows in world["seam2"]]
    pred_direct = [si.cosine_nn_decode(rows, E) for rows in est_direct]

    D2 = si.ridge_decode_map(world["H2"], TOY_RIDGE_LAMBDA)
    mid_hat = [si._matmul(rows, D2) for rows in world["seam2"]]
    pred_chained = [si.cosine_nn_decode(rows, world["codebook"])
                    for rows in mid_hat]

    D1 = si.ridge_decode_map(world["H1"], TOY_RIDGE_LAMBDA)
    est_ctrl = [si._matmul(rows, D1) for rows in world["seam1"]]
    pred_ctrl = [si.cosine_nn_decode(rows, E) for rows in est_ctrl]

    arms = {
        "direct-seam2": (pred_direct, _estimate_mse(est_direct, victim, E)),
        "chained": (pred_chained, None),
        "control-seam1": (pred_ctrl, _estimate_mse(est_ctrl, victim, E)),
    }
    cells = {}
    for arm, (preds, est_mse) in arms.items():
        cell = {"experiment": EXPERIMENT_ID, "arm": arm, "middle": middle,
                "sigma": sigma, "seed": seed,
                "embedding_mse": _embedding_mse(preds, victim, E),
                **si.score_sequences(preds, victim, control,
                                     n_seeds=len(args.seeds))}
        if est_mse is not None:
            cell["estimate_mse"] = est_mse
        cells[arm] = cell
    if middle == "identity":
        chained_acc = cells["chained"]["seq_accuracy"]
        ctrl_acc = cells["control-seam1"]["seq_accuracy"]
        cells["chained"]["null_arm_delta_accuracy"] = round(
            abs(chained_acc - ctrl_acc), 8)
        cells["chained"]["null_arm_ok"] = (
            cells["chained"]["null_arm_delta_accuracy"] <= MNULL_TOL)
    return cells


def _run_toy(args, ev):
    middles = ("planted", "identity") if args.middle == "both" \
        else (args.middle,)
    out = artifacts.make_artifact(
        SCHEMA,
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "vocab": args.vocab, "dim": args.hidden, "seq_len": args.seq_len,
         "toy_seqs": args.toy_seqs, "toy_cal_tokens": args.toy_cal_tokens,
         "toy_sigmas": list(args.toy_sigmas), "seeds": args.seeds,
         "middles": list(middles), "ridge_lambda": TOY_RIDGE_LAMBDA,
         "null_arm_tol": MNULL_TOL},
        THREAT_MODEL + " Toy tier: synthetic KNOWN-map chained segments "
        "(H1/H2 and the middle codebook granted); measures decode/scoring/"
        "gate plumbing and the compounded-vs-chained error ordering, not a "
        "real model.",
        interpretation=INTERPRETATION,
        evidence=ev)
    cells = []
    for middle in middles:
        for sigma in args.toy_sigmas:
            for seed in args.seeds:
                try:
                    arm_cells = toy_cell(args, sigma, seed, middle)
                except Exception as e:
                    journal_error(args.output, EXPERIMENT_ID,
                                  {"middle": middle, "sigma": sigma,
                                   "seed": seed}, e)
                    continue
                for cell in arm_cells.values():
                    artifacts.append_jsonl(args.output, cell)
                    if ev is not None:
                        ev.event("cell_result", cell)
                    out["results"].append(cell)
                    cells.append(cell)
                print(f"[two-seam-toy] middle={middle} sigma={sigma} "
                      f"seed={seed}: direct="
                      f"{arm_cells['direct-seam2']['seq_accuracy']} chained="
                      f"{arm_cells['chained']['seq_accuracy']} control="
                      f"{arm_cells['control-seam1']['seq_accuracy']}")
    for middle in middles:
        for sigma in sorted({c["sigma"] for c in cells
                             if c["middle"] == middle}):
            vals = [c for c in cells
                    if c["middle"] == middle and c["sigma"] == sigma]
            by_arm = {}
            for arm in ("direct-seam2", "chained", "control-seam1"):
                av = [c for c in vals if c["arm"] == arm]
                if av:
                    by_arm[arm] = {
                        "seq_accuracy_mean":
                            sum(c["seq_accuracy"] for c in av) / len(av),
                        "embedding_mse_mean":
                            sum(c["embedding_mse"] for c in av) / len(av),
                        "gate_verdicts": [c["gate"]["verdict"] for c in av]}
            out["summary"].append({
                "experiment": EXPERIMENT_ID, "middle": middle,
                "sigma": sigma, "arms": by_arm, "n_cells": len(vals)})
    artifacts.write_artifact(args.output, out, evidence=ev)
    return 0


def _run_real(args, ev):
    require_torch(EXPERIMENT_ID)
    raise SystemExit(
        f"[{EXPERIMENT_ID}] dual-seam captures are not producible today: "
        "splitbench CaptureWriter names captures wire_{seq:05d}.{pt,json} "
        "with a single seam per writer/mode and NO seam index in the "
        "sidecar schema, so a two-segment delegation run (head | island | "
        "tail, SplitPlan '4,8') cannot record seam1 and seam2 as distinct, "
        "row-aligned capture streams. Expected layout once splitbench "
        "gains per-seam naming: one wire pair per seam per step, e.g. "
        "wire_{seq:05d}.seam1.{pt,json} and wire_{seq:05d}.seam2.{pt,json} "
        "(or a 'seam' field in the sidecar), step-major row-aligned with "
        "canonical.pt. The chained decode is ready to wire against that "
        "layout; until then --toy is the supported tier.")


def run(args):
    ev = _make_evidence(args)
    if args.toy:
        if args.quick:
            args.seeds = [0]
            args.toy_sigmas = (0.0, 0.5)
            args.toy_seqs = min(args.toy_seqs, 16)
            args.toy_cal_tokens = min(args.toy_cal_tokens, 1000)
        return _run_toy(args, ev)
    if not args.capture_dir:
        raise SystemExit(f"[{EXPERIMENT_ID}] pass --toy, or --capture-dir "
                         "D for a dual-seam capture (not producible today; "
                         "see --help).")
    return _run_real(args, ev)


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import json
    import os
    import tempfile

    print("two-seam-delegation toy core (pure python):")

    w1 = toy_world(0, vocab=20, dim=6, n_seq=4, seq_len=5, sigma=0.1)
    w2 = toy_world(0, vocab=20, dim=6, n_seq=4, seq_len=5, sigma=0.1)
    check("toy world is deterministic",
          w1["victim"] == w2["victim"] and w1["seam2"] == w2["seam2"])
    check("seam shapes: [n_seq][seq_len][dim] on both seams",
          len(w1["seam1"]) == 4 and len(w1["seam1"][0]) == 5
          and len(w1["seam1"][0][0]) == 6
          and len(w1["seam2"][0][0]) == 6)
    check("middle codebook has one row per vocab token",
          len(w1["codebook"]) == 20 and len(w1["codebook"][0]) == 6)

    args = build_parser().parse_args(["--toy"])
    args.mode = "training"
    planted = toy_cell(args, 0.0, 0, "planted")
    check("planted middle: chained beats the compounded direct decode",
          planted["chained"]["seq_accuracy"]
          > planted["direct-seam2"]["seq_accuracy"])
    check("planted middle: single-seam control is strongest",
          planted["control-seam1"]["seq_accuracy"]
          >= planted["chained"]["seq_accuracy"])
    check("cells carry control + gate fields",
          all(c["gate"]["verdict"] in ("PASS", "FAIL")
              for c in planted.values()))

    ident = toy_cell(args, 0.0, 0, "identity")
    check("identity middle null arm: chained == single-seam within tol",
          ident["chained"]["null_arm_ok"] is True)
    check("identity middle: direct decode recovers (linear world)",
          ident["direct-seam2"]["seq_accuracy"] > 0.9)

    check("parser builds torch-less",
          build_parser().parse_args([]).middle == "both")
    ns = build_parser().parse_args([])
    ns.mode = "training"
    try:
        run(ns)
        check("no --toy/--capture-dir is a descriptive error", False)
    except SystemExit as e:
        check("no --toy/--capture-dir is a descriptive error",
              "--capture-dir" in str(e))
    ns2 = build_parser().parse_args(["--capture-dir", "/nonexistent"])
    ns2.mode = "training"
    try:
        run(ns2)
        check("real tier explains the missing dual-seam layout", False)
    except SystemExit as e:
        check("real tier explains the missing dual-seam layout",
              "dual-seam" in str(e) or "requires torch" in str(e))

    with tempfile.TemporaryDirectory() as d:
        ns = build_parser().parse_args(
            ["--toy", "--quick", "--evidence",
             "--output", os.path.join(d, "tsd.json")])
        ns.mode = "training"
        check("toy run end to end torch-free", run(ns) == 0)
        art = json.load(open(os.path.join(d, "tsd.json")))
        check("toy artifact has per-arm cells with gate verdicts",
              len(art["results"]) == 2 * 2 * 3
              and all(c["gate"]["verdict"] in ("PASS", "FAIL")
                      for c in art["results"]))
        check("artifact schema is versioned", art["schema"] == SCHEMA)

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
