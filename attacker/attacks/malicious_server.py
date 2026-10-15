#!/usr/bin/env python3
"""malicious-server (training mode) — the FL malicious-server analogue for
split training (the Boenisch / Fowl "robbing the fed" line transplanted to
the split-LLM seam): the cloud CRAFTS the body-side computation so the
seam gradient it returns leaks the client batch far more strongly than an
honest-but-curious body's gradient ever does.

Threat model (mirrored in every artifact): a MALICIOUS server — not
honest-but-curious. It serves the crafted body computation of splitbench's
--server-attack passive-amplify hook,

    y = Body(h) + alpha * h ,        logits = lm_head(ln_f(y))

so the seam gradient it returns is dL/dh = alpha * J_lnf (p - onehot) E +
(honest body backprop): an alpha-scaled, PER-ROW, near-isometric copy of
the label/token direction that skips the body blocks' attention mixing and
MLP shear. The craft is mathematically real (autograd on the modified
computation, not a doctored tensor), minimal, and public (the deployed
body is the server's own computation; alpha is a public scalar). The
attacker is granted the public token embedding matrix E (the TAG/LAMP
white-box assumption, tied unembedding lm_head == wte); canonical tokens
are evaluation truth, never an attack input (label-free attack).

Attack: the seam bwd-gradient inversion — the tag-style per-token
direction decode (candidate score C[i, v] = cos(-g*_i, E[v]) + a light
assignment) imported from gradient_inversion's tag pieces, run against
captures produced with --server-attack passive-amplify (the crafted world)
vs none (the honest world) and reporting recovery in BOTH worlds. The
published result class is malicious server >> honest-but-curious at equal
budget (identical budget-free decode over the same number of captures; no
optimization loop anywhere).

Where the gap lives (measured, tests/test_malicious_server.py): on a tiny
random GPT-2 (2L/32/V=99) the honest direction decode is ALREADY ~1.0 —
random-init bodies are near-isometries and the residual stream preserves
the per-row label direction (the committed tag test's 100%). The honest
decode degrades only when the body genuinely distorts per-row directions
(attention mixing + MLP shear through several blocks) AND the candidate
space is dense (rival embeddings crowd the direction space): the torch
tier constructs that regime with a dense-vocab distorting tiny world
(amplified attention/output projections, V=32000 — a stand-in for a
trained/deep body at realistic vocab density), where honest drops to
~0.43 while the crafted alpha path restores ~1.0. The toy tier (below)
isolates the same mechanism in a torch-free linear world where the
cross-position mixing is exact.

Toy tier (torch-free, consistent linear world): E [V, d] raw Gaussian
embeddings; A orthogonal public seam map; U = E @ A the tied unembedding
at the seam. The honest body is the cross-position-mixing linear
computation

    y_i = M h_i + mix * sum_{j != i} M h_j ,   logits_i = U y_i / temp

(the attention analogue: every position's content leaks into every row's
output) and the crafted body adds the identity path:

    y_i = (alpha I + M) h_i + mix * sum_{j != i} M h_j .

The captured seam gradient is the world's TRUE CE gradient of each
computation: g_i = (delta_i D + mix * sum_{k != i} delta_k M) / temp with
D = M (honest) | alpha I + M (crafted) and delta_k = sum_v (p_kv -
1{v=y_k}) U[v] — the per-token direction argument of the tag arm. In the
honest world every row's gradient is a mixture of ALL positions' label
directions (mixing drowns the per-row signal: recovery sits at the
majority control); the crafted world's alpha * delta_i term re-dominates
each row with its own label direction (recovery ~1.0). Same attacker
knowledge (U) in both worlds; only the server's computation differs.

Usage:
    python -m attacker --mode training --attack malicious-server --help
    python -m attacker --mode training --attack malicious-server --toy \
        --quick --output /tmp/ms.json
    python -m attacker --mode training --attack malicious-server \
        --capture-dir CAPS_H --canonical-pt CAPS_H/canonical.pt \
        --crafted-dir CAPS_C --crafted-canonical-pt CAPS_C/canonical.pt \
        --model gpt2 --split 6 --output ms.json --evidence

(splitbench captures: python3 -m attacker.splitbench ... --save-canonical
with --server-attack none for CAPS_H and --server-attack passive-amplify
--server-attack-alpha 8 for CAPS_C, same model/split/data.)
"""

import argparse
import math
import random
import types

try:
    import torch
except ImportError:
    torch = None

from .. import artifacts
from .common import add_common_args, journal_error, require_torch
from .gradient_inversion import (_cosine, _majority_control_token,
                                 _matmul, _orthogonal_mat, _randn_mat,
                                 _score_tag, _softmax, _tag_decode)

EXPERIMENT_ID = "malicious_server_steering"
MODES = ("training",)
REQUIRES_LABELS = False
DESCRIPTION = ("malicious-server steering (Boenisch/Fowl line): the cloud "
               "crafts the body computation (passive-amplify: y = Body(h) "
               "+ alpha*h) so the returned seam gradient carries an "
               "alpha-scaled per-row copy of the token signal; bwd "
               "tag-style direction decode against honest vs crafted "
               "captures (crafted >> honest at equal budget)")

SCHEMA = "dtraining.attacker.malicious_server.v1"
WORLDS = ("honest", "crafted")

THREAT_MODEL = (
    "MALICIOUS split-training server (not honest-but-curious): it crafts "
    "the body-side computation (splitbench --server-attack "
    "passive-amplify, y = Body(h) + alpha*h through the shared tail) so "
    "the seam gradient it returns carries an alpha-scaled per-row "
    "near-isometric copy of the label/token direction (dL/dh = alpha * "
    "J_lnf (p - onehot) E + honest backprop). The wire observer sees the "
    "fwd seam tensor and the returned gradient; it is granted the public "
    "token embedding matrix E (tied unembedding). Canonical tokens are "
    "evaluation truth, never an attack input (label-free attack).")

INTERPRETATION = (
    "seq_accuracy / token-set P/R are position-aligned / orderless vs "
    "canonical tokens[1:] (the gradient at row i points at the label of "
    "position i+1 — the tag convention; position 0 is nobody's label and "
    "is not scored). The control decodes the calibration-corpus majority "
    "token at every position; the gate is the frozen-gate "
    "Bonferroni-Wilson excess over that control, n = scored tokens. "
    "honest cells use --server-attack none captures, crafted cells use "
    "--server-attack passive-amplify captures, decoded with the identical "
    "budget-free direction decode (equal budget); "
    "crafted_minus_honest_pp is the published result class (malicious "
    "server >> honest-but-curious) made concrete. Toy tier is a "
    "consistent linear world with exact cross-position mixing — a "
    "machinery check, not a security claim; real numbers come from "
    "splitbench captures.")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, hidden=32)
    ap.add_argument("--capture-dir", default=None,
                    help="HONEST world: wire-capture directory produced "
                         "with --server-attack none (wire-training "
                         "schema; reads phase='bwd' seam gradients)")
    ap.add_argument("--canonical-pt", default=None,
                    help="canonical.pt of the HONEST capture run "
                         "(splitbench --save-canonical)")
    ap.add_argument("--crafted-dir", default=None,
                    help="CRAFTED world: wire-capture directory produced "
                         "with --server-attack passive-amplify (same "
                         "model/split/data as the honest run)")
    ap.add_argument("--crafted-canonical-pt", default=None,
                    help="canonical.pt of the CRAFTED capture run")
    ap.add_argument("--model", default=None,
                    help="HF name/path of the PUBLIC base model — grants "
                         "the token embedding matrix E (tied unembedding)")
    ap.add_argument("--split", default="6",
                    help="seam boundary of the capture harness split "
                         "(default '6')")
    ap.add_argument("--samples", type=int, default=8,
                    help="cap on scored sequences per world (scan order)")
    ap.add_argument("--assignment", choices=("greedy", "hungarian"),
                    default="greedy",
                    help="direction decode over the per-position candidate "
                         "matrix: greedy per-position argmax (default) or "
                         "hungarian one-to-one")
    ap.add_argument("--calibration-seqs", type=int, default=64,
                    help="public calibration sequences for the majority "
                         "control (synthetic uniform draws unless "
                         "--calibration-file)")
    ap.add_argument("--calibration-file", default=None,
                    help="public text tokenized with the --model tokenizer "
                         "for the control corpus (default: synthetic "
                         "uniform draws — no public text assumed)")
    ap.add_argument("--vocab", type=int, default=32, help="toy vocab size")
    ap.add_argument("--seq-len", type=int, default=8,
                    help="toy victim sequence length")
    ap.add_argument("--toy-seqs", type=int, default=8,
                    help="toy victim sequences per cell")
    ap.add_argument("--tag-temp", type=float, default=8.0,
                    help="toy body softmax temperature (high = the "
                         "near-uniform confidence of a random-init body, "
                         "the step-0 capture regime)")
    ap.add_argument("--toy-mix", type=float, default=1.0,
                    help="toy honest-body cross-position mixing weight "
                         "(the attention analogue drowning the per-row "
                         "direction)")
    ap.add_argument("--server-attack-alpha", type=float, default=8.0,
                    help="the public amplification scalar of the crafted "
                         "body (toy tier: functional; real tier: "
                         "documentary record of the crafted capture run's "
                         "--server-attack-alpha)")
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
    return EvidenceLogger(args.output,
                          node_role="attacker/malicious-server")


def toy_ms_world(seed, vocab=32, dim=32, seq_len=8, n_seq=8, temp=8.0,
                 alpha=8.0, mix=1.0):
    """The malicious-server toy world (module docstring): E raw Gaussian,
    A orthogonal public seam map, U = E @ A the tied unembedding, M the
    body's per-row linear map (~ N(0, 1/d) — a moderate shear), mix the
    cross-position mixing weight, alpha the crafted identity-path scalar.
    Victim sequences are uniform token draws; labels = tokens[1:]."""
    rng = random.Random(seed)
    E = _randn_mat(vocab, dim, rng)
    A = _orthogonal_mat(dim, rng)
    U = _matmul(E, A)
    M = _randn_mat(dim, dim, rng, scale=1.0 / math.sqrt(dim))
    seqs = [[rng.randrange(vocab) for _ in range(seq_len)]
            for _ in range(n_seq)]
    return {"vocab": vocab, "dim": dim, "seq_len": seq_len, "temp": temp,
            "alpha": alpha, "mix": mix, "E": E, "A": A, "U": U, "M": M,
            "seqs": seqs}


def toy_ms_grads(world, ids, crafted):
    """The world's TRUE CE seam gradients for one sequence (module
    docstring): the forward mixes positions (mix) and optionally carries
    the crafted identity path (alpha, crafted=True); g_i = (delta_i D +
    mix sum_{k != i} delta_k M) with D = M | alpha I + M and delta_k =
    sum_v (p_kv - 1{v=y_k}) U[v], p from the SAME world's forward."""
    U, M, temp, V = world["U"], world["M"], world["temp"], world["vocab"]
    mix, T, d = world["mix"], len(ids), world["dim"]
    alpha = world["alpha"] if crafted else 0.0
    hM = [_matmul([U[t]], M)[0] for t in ids]
    ys = []
    for i in range(T):
        y = [alpha * U[ids[i]][k] + hM[i][k] for k in range(d)]
        for j in range(T):
            if j != i:
                y = [a + mix * b for a, b in zip(y, hM[j])]
        ys.append(y)
    deltas = []
    for i in range(T - 1):
        logits = [sum(a * b for a, b in zip(ys[i], U[v])) / temp
                  for v in range(V)]
        p = _softmax(logits)
        lab = ids[i + 1]
        deltas.append([sum((p[v] - (1.0 if v == lab else 0.0)) * U[v][k]
                           for v in range(V)) for k in range(d)])
    grads = []
    for i in range(T - 1):
        g = _matmul([deltas[i]], M)[0]
        g = [alpha * deltas[i][k] + g[k] for k in range(d)]
        for k in range(T - 1):
            if k != i:
                dkM = _matmul([deltas[k]], M)[0]
                g = [a + mix * b for a, b in zip(g, dkM)]
        grads.append(g)
    return grads


def toy_ms_candidates(world, g):
    """C[i][v] = cos(-g*_i, U[v]) — the tag cosine objective per token
    (the attacker decodes against the public seam embedding U, identical
    knowledge in both worlds)."""
    U = world["U"]
    return [[_cosine([-x for x in gi], U[v])
             for v in range(world["vocab"])] for gi in g]


def toy_ms_cell(args, world_kind, seed):
    """One toy cell: decode the world's true captured gradients with the
    tag greedy decode and score with the shared tag plumbing."""
    if world_kind not in WORLDS:
        raise ValueError(f"unknown world {world_kind!r} "
                         f"(valid: {', '.join(WORLDS)})")
    world = toy_ms_world(args.seed + 557 * seed, vocab=args.vocab,
                         dim=args.hidden, seq_len=args.seq_len,
                         n_seq=args.toy_seqs, temp=args.tag_temp,
                         alpha=args.server_attack_alpha, mix=args.toy_mix)
    crafted = world_kind == "crafted"
    preds, trues = [], []
    for ids in world["seqs"]:
        g = toy_ms_grads(world, ids, crafted)
        preds.append(_tag_decode(toy_ms_candidates(world, g), "greedy"))
        trues.append(ids[1:])
    cal_rng = random.Random(args.seed + 41249 + seed)
    cal = [cal_rng.randrange(args.vocab) for _ in range(4000)]
    control = _majority_control_token([cal])
    return {"world": world_kind, "seed": seed,
            "server_attack_alpha": world["alpha"], "toy_mix": world["mix"],
            **_score_tag(preds, trues, control, n_seeds=len(args.seeds))}


def _run_toy(args, ev):
    out = artifacts.make_artifact(
        SCHEMA,
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "hidden": args.hidden, "vocab": args.vocab,
         "seq_len": args.seq_len, "toy_seqs": args.toy_seqs,
         "tag_temp": args.tag_temp, "toy_mix": args.toy_mix,
         "server_attack_alpha": args.server_attack_alpha,
         "seeds": args.seeds},
        THREAT_MODEL + " Toy tier: the consistent linear mixing world "
        "(E raw Gaussian, A orthogonal public seam map, U = E @ A tied "
        "unembedding, M the body's per-row map, mix the cross-position "
        "mixing weight); gradients are the world's true CE gradients.",
        interpretation=INTERPRETATION,
        evidence=ev)
    cells = []
    for world_kind in WORLDS:
        for seed in args.seeds:
            cell = {"experiment": EXPERIMENT_ID, "arm": "bwd-direction",
                    "submode": f"toy-{world_kind}"}
            try:
                cell.update(toy_ms_cell(args, world_kind, seed))
            except Exception as e:
                journal_error(args.output, EXPERIMENT_ID, cell, e)
                continue
            artifacts.append_jsonl(args.output, cell)
            if ev is not None:
                ev.event("cell_result", cell)
            out["results"].append(cell)
            cells.append(cell)
            print(f"[ms-toy] {world_kind} seed={seed}: "
                  f"acc={cell['seq_accuracy']} control="
                  f"{cell['control_accuracy']} "
                  f"gate={cell['gate']['verdict']}")
    means = {}
    for world_kind in WORLDS:
        vals = [c for c in cells if c["world"] == world_kind]
        if vals:
            means[world_kind] = sum(c["seq_accuracy"] for c in vals) \
                / len(vals)
            out["summary"].append({
                "experiment": EXPERIMENT_ID, "world": world_kind,
                "seq_accuracy_mean": means[world_kind],
                "control_accuracy_mean":
                    sum(c["control_accuracy"] for c in vals) / len(vals),
                "gate_verdicts": [c["gate"]["verdict"] for c in vals],
                "n_seeds": len({c["seed"] for c in vals}),
                "n_cells": len(vals)})
    if len(means) == 2:
        out["summary"].append({
            "experiment": EXPERIMENT_ID,
            "crafted_minus_honest_pp":
                round(100.0 * (means["crafted"] - means["honest"]), 4)})
    artifacts.write_artifact(args.output, out, evidence=ev)
    return 0


def _control_token(args, vocab, seq_len, device):
    """The attacker-side majority control: public calibration tokens
    (synthetic uniform draws unless --calibration-file); canonical never
    consulted."""
    if args.calibration_file:
        if not args.model:
            raise SystemExit(f"[{EXPERIMENT_ID}] --calibration-file needs "
                             "--model (its tokenizer)")
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise SystemExit(f"[{EXPERIMENT_ID}] --calibration-file "
                             "requires transformers") from None
        tok = AutoTokenizer.from_pretrained(args.model)
        with open(args.calibration_file) as f:
            ids = tok(f.read(), return_tensors="pt").input_ids[0].tolist()
        cal = ids[:args.calibration_seqs * seq_len]
    else:
        cal = torch.randint(0, vocab, (args.calibration_seqs * seq_len,),
                            generator=torch.Generator().manual_seed(
                                args.seed + 4242)).tolist()
    return _majority_control_token([cal])


def _world_cells(args, world_kind, capture_dir, canonical_pt, E, control,
                 device):
    """Decode one world's bwd captures with the tag direction decode;
    returns per-step cells (scored vs tokens[1:], the tag convention)."""
    from .gradient_inversion import (_tag_candidates_torch,
                                     _tag_load_sequences)
    shim = types.SimpleNamespace(capture_dir=capture_dir,
                                 canonical_pt=canonical_pt,
                                 samples=args.samples)
    picked, seq_len, h_dim = _tag_load_sequences(shim)
    groups = {}
    order = []
    for i, p in enumerate(picked):
        key = (p["step"], p["epoch"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)
    cells = []
    for key in order:
        idx = groups[key]
        g_star = torch.stack([picked[i]["g"] for i in idx]).to(device)
        trues = [picked[i]["true"][1:] for i in idx]
        cell = {"experiment": EXPERIMENT_ID, "arm": "bwd-direction",
                "submode": f"real-{args.assignment}",
                "world": world_kind, "mode": args.mode,
                "step": key[0], "epoch": key[1],
                "capture_dir": capture_dir}
        C = _tag_candidates_torch(g_star, E)[:, :-1, :]
        cand = C.cpu().tolist()
        preds = [_tag_decode(c, args.assignment) for c in cand]
        cell.update(_score_tag(preds, trues, control))
        cells.append(cell)
    return cells


def _run_real(args, ev):
    require_torch(EXPERIMENT_ID)
    from .gradient_inversion import _tag_load_hf
    missing = [name for name, val in
               (("--capture-dir", args.capture_dir),
                ("--canonical-pt", args.canonical_pt),
                ("--crafted-dir", args.crafted_dir),
                ("--crafted-canonical-pt", args.crafted_canonical_pt),
                ("--model", args.model))
               if not val]
    if missing:
        raise SystemExit(f"[{EXPERIMENT_ID}] real tier needs "
                         + ", ".join(missing)
                         + " (splitbench captures with --server-attack "
                         "none / passive-amplify; see module docstring)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _tag_load_hf(args.model, device)
    E = model.get_input_embeddings().weight.detach().to(device)
    vocab = E.shape[0]
    shim_probe = types.SimpleNamespace(capture_dir=args.capture_dir,
                                       canonical_pt=args.canonical_pt,
                                       samples=1)
    from .gradient_inversion import _tag_load_sequences
    _, seq_len, _ = _tag_load_sequences(shim_probe)
    control = _control_token(args, vocab, seq_len, device)

    out = artifacts.make_artifact(
        SCHEMA,
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": False,
         "capture_dir": args.capture_dir,
         "canonical_pt": args.canonical_pt,
         "crafted_dir": args.crafted_dir,
         "crafted_canonical_pt": args.crafted_canonical_pt,
         "model": args.model, "split": args.split,
         "server_attack_alpha": args.server_attack_alpha,
         "samples": args.samples, "assignment": args.assignment,
         "calibration_seqs": args.calibration_seqs,
         "calibration_file": args.calibration_file, "device": device,
         "seq_len": seq_len},
        THREAT_MODEL, interpretation=INTERPRETATION, evidence=ev)

    cells = []
    for world_kind, cap_dir, can_pt in (
            ("honest", args.capture_dir, args.canonical_pt),
            ("crafted", args.crafted_dir, args.crafted_canonical_pt)):
        try:
            world_cells = _world_cells(args, world_kind, cap_dir, can_pt,
                                       E, control, device)
        except Exception as e:
            journal_error(args.output, EXPERIMENT_ID,
                          {"world": world_kind}, e)
            continue
        for cell in world_cells:
            artifacts.append_jsonl(args.output, cell)
            if ev is not None:
                ev.event("cell_result", cell)
            out["results"].append(cell)
            cells.append(cell)
            print(f"[ms] {world_kind} step={cell['step']} "
                  f"({args.assignment}): acc={cell['seq_accuracy']} "
                  f"control={cell['control_accuracy']} "
                  f"gate={cell['gate']['verdict']}")

    means = {}
    for world_kind in WORLDS:
        vals = [c for c in cells if c["world"] == world_kind]
        if vals:
            n_tok = sum(c["n_tokens"] for c in vals)
            pooled = sum(c["n_correct"] for c in vals) / max(1, n_tok)
            pooled_ctl = sum(c["n_control_correct"] for c in vals) \
                / max(1, n_tok)
            means[world_kind] = pooled
            from .. import gate
            out["summary"].append({
                "experiment": EXPERIMENT_ID, "world": world_kind,
                "n_cells": len(vals), "n_tokens": n_tok,
                "seq_accuracy": round(pooled, 6),
                "control_accuracy": round(pooled_ctl, 6),
                "token_set_precision_mean":
                    sum(c["token_set_precision"] for c in vals) / len(vals),
                "token_set_recall_mean":
                    sum(c["token_set_recall"] for c in vals) / len(vals),
                "gate": gate.verdict(pooled, pooled_ctl, n_tok),
                "gate_verdicts": [c["gate"]["verdict"] for c in vals]})
    if len(means) == 2:
        out["summary"].append({
            "experiment": EXPERIMENT_ID,
            "crafted_minus_honest_pp":
                round(100.0 * (means["crafted"] - means["honest"]), 4)})
    artifacts.write_artifact(args.output, out, evidence=ev)
    return 0


def run(args):
    ev = _make_evidence(args)
    if args.samples < 1:
        raise SystemExit(f"[{EXPERIMENT_ID}] --samples must be >= 1")
    if args.toy:
        if args.quick:
            args.seeds = [0, 1]
        return _run_toy(args, ev)
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

    print("malicious-server toy core (pure python):")
    w = toy_ms_world(0, vocab=16, dim=16, seq_len=6, n_seq=3)
    check("world is deterministic and shaped",
          len(w["seqs"]) == 3 and len(w["U"]) == 16 and len(w["U"][0]) == 16
          and toy_ms_world(0, 16, 16, 6, 3)["seqs"] == w["seqs"])
    gh = toy_ms_grads(w, w["seqs"][0], crafted=False)
    gc = toy_ms_grads(w, w["seqs"][0], crafted=True)
    check("gradients are [seq_len-1, dim] real rows",
          len(gh) == 5 and len(gh[0]) == 16 and len(gc) == 5)
    check("the crafted gradient differs from the honest one (alpha path)",
          gh != gc)
    C = toy_ms_candidates(w, gc)
    check("candidate matrix is [seq_len-1, vocab] of cosines",
          len(C) == 5 and len(C[0]) == 16
          and all(-1.0 - 1e-9 <= c <= 1.0 + 1e-9 for row in C for c in row))

    args = build_parser().parse_args(
        ["--toy", "--vocab", "32", "--hidden", "32", "--seq-len", "8",
         "--toy-seqs", "8", "--seeds", "0", "1", "2"])
    args.mode = "training"
    honest_accs, crafted_accs = [], []
    for seed in (0, 1, 2):
        h_cell = toy_ms_cell(args, "honest", seed)
        c_cell = toy_ms_cell(args, "crafted", seed)
        honest_accs.append(h_cell["seq_accuracy"])
        crafted_accs.append(c_cell["seq_accuracy"])
        check(f"toy seed {seed}: crafted >> honest, honest at control, "
              "gate fields present",
              c_cell["seq_accuracy"] > 0.90
              and h_cell["seq_accuracy"]
              < h_cell["control_accuracy"] + 0.10
              and c_cell["gate"]["verdict"] == "FAIL"
              and "token_set_precision" in h_cell)
    check("toy mean gap is the published result class",
          sum(crafted_accs) / 3 - sum(honest_accs) / 3 > 0.8)

    check("parser builds torch-less",
          build_parser().parse_args([]).assignment == "greedy"
          and build_parser().parse_args([]).server_attack_alpha == 8.0
          and build_parser().parse_args([]).toy_mix == 1.0)
    ns = build_parser().parse_args([])
    ns.mode = "training"
    try:
        run(ns)
        check("real tier without capture args is a descriptive error",
              False)
    except SystemExit as e:
        check("real tier without capture args is a descriptive error",
              "--capture-dir" in str(e) or "requires torch" in str(e))
    if torch is None:
        ns2 = build_parser().parse_args(
            ["--capture-dir", "/nonexistent",
             "--canonical-pt", "/nonexistent",
             "--crafted-dir", "/nonexistent",
             "--crafted-canonical-pt", "/nonexistent",
             "--model", "x"])
        ns2.mode = "training"
        try:
            run(ns2)
            check("real tier is torch-guarded", False)
        except SystemExit as e:
            check("real tier is torch-guarded", "requires torch" in str(e))

    with tempfile.TemporaryDirectory() as d:
        ns = build_parser().parse_args(
            ["--toy", "--quick", "--evidence",
             "--output", os.path.join(d, "ms.json")])
        ns.mode = "training"
        check("toy run end to end torch-free", run(ns) == 0)
        art = json.load(open(os.path.join(d, "ms.json")))
        check("toy artifact has honest+crafted cells with gate verdicts",
              sorted(c["world"] for c in art["results"])
              == ["crafted", "crafted", "honest", "honest"]
              and all(c["gate"]["verdict"] in ("PASS", "FAIL")
                      for c in art["results"]))
        check("toy summary carries the crafted-minus-honest margin",
              any("crafted_minus_honest_pp" in s
                  for s in art["summary"]))
        check("journal lines match result cells",
              len(open(os.path.join(d, "ms.json.jsonl")).read()
                  .splitlines()) == len(art["results"]))
        from ..evidence import verify_evidence
        good, problems = verify_evidence(
            os.path.join(d, "ms.json.evidence.json"))
        check("evidence chain verifies", good and problems == [])
        try:
            run(build_parser().parse_args(["--toy", "--evidence"]))
            check("--evidence without --output raises", False)
        except SystemExit as e:
            check("--evidence without --output raises",
                  "--evidence requires --output" in str(e))

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
