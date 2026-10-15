#!/usr/bin/env python3
"""Citation scope (reference audit): this is a local body/seam-gradient matching
adaptation. Yin et al., See Through Gradients, is CVPR 2021, not IEEE S&P
2022. ExPLoit is related label-inference work, not the algorithm implemented
here. See docs/reference-audit-2026-09-22.md.

batch-label-recon (training mode) — Batch Label Reconstruction (BLI),
a local batch-gradient matching adaptation for the split-LLM seam,
motivated by label recovery from gradients and by
VFLAIR-LLM's BatchLabelReconstruction_LLM.

Threat model (mirrored in every artifact): the honest-but-curious server
observes the FIRST training batch's backward pass on the seam —

  * the seam gradient h_wire.grad (splitbench captures it natively as
    phase="bwd", step=0 — the first batch is always captured), AND
  * the per-parameter BODY-slice gradients (splitbench
    --body-grad-capture dumps them to <capture_dir>/body_grads_step0.pt),

and is granted the body slice itself (white-box: --model + --split at the
same architecture as the capture; at step 0 the body is the pre-update
body, so the public/initial weights ARE the captured body — no finetuned
state is assumed). Canonical tokens are evaluation truth, never an
attack input: the attack is LABEL-FREE (REQUIRES_LABELS=False) — it
infers the labels from the gradients.

Classification-to-LM correspondence (honest notes): classic BLI recovers
the multiset of classification labels in a batch from the sign structure
of the last-layer gradients (for cross-entropy, dL/db_c = p_c - y_c).
The split-LLM analogue of "the batch's labels" is the multiset of
NEXT-TOKEN targets of the first training batch — i.e. the batch's input
tokens shifted by one. This module therefore implements TOKEN
RECONSTRUCTION of the first batch: a dummy soft-label distribution Z
[b, s-1, V] (the relaxed one-hot next-token targets) plus a dummy seam
input h_d [b, s, H] are optimized (Adam, --steps) so the SIMULATED body
gradients match the captured ones (L2, plus --seam-weight * L2 on the
seam gradient). Two honest consequences: (1) the first token of each
sequence is nobody's label and is not scored; (2) the recovered bag of
tokens (token-set precision/recall) is the direct analogue of BLI's
recovered label multiset, while position-aligned accuracy additionally
measures assignment — BLI's harder step.

Scoring (per cell, shared with the toy tier): orderless token-set
precision/recall (bag-of-tokens overlap — BLI's native metric),
position-aligned token accuracy vs canonical, a bag-mode control (the
mode of the attacker's OWN recovered token multiset decoded at every
position — attacker-computable, canonical never consulted), and the
frozen-gate Bonferroni-Wilson excess over that control, n = scored
tokens.

Tiers:
  --toy          torch-free machinery check. Synthetic linear "body"
                 y = W x (W [V, d] KNOWN — the white-box toy analogue of
                 the granted body) over a batch of B = n_seq*seq_len
                 positions with KNOWN inputs X (the toy grant; the LM
                 tier must co-optimize h_d, which is why it needs Adam).
                 Captured per-param grads G_W = sum_i r_i x_i^T with
                 r_i = softmax(W x_i) - onehot(t_i). Recovery is REAL
                 closed-form least squares: R = G_W X^T (X X^T)^-1, then
                 t_i = argmin_v (r_i[v] - p_i[v]) (the label is the
                 negative residual spike — the iDLG sign argument made
                 concrete). Planted (real grads) recovers >95%; the null
                 cell (random grads at matched scale) sits at control.
                 Validates decode/scoring/gate plumbing, not a security
                 claim.
  (real tier)    --capture-dir D --canonical-pt C --model M --split S
                 [--body-grads-pt P] [--steps N] — the gradient-matching
                 attack above on splitbench captures (finetune with
                 --body-grad-capture --save-canonical).

Usage:
    python -m attacker --mode training --attack batch-label-recon --help
    python -m attacker --mode training --attack batch-label-recon --toy \
        --quick --output /tmp/blr.json
    python -m attacker --mode training --attack batch-label-recon \
        --capture-dir CAPS --canonical-pt CAPS/canonical.pt \
        --model tinygpt2/ --split 1 --steps 200 --output blr.json \
        --evidence
"""

import argparse
import collections
import math
import random

try:
    import torch
except ImportError:
    torch = None

from .. import artifacts, gate
from .common import add_common_args, journal_error, require_torch

EXPERIMENT_ID = "batch_label_reconstruction"
MODES = ("training",)
REQUIRES_LABELS = False
DESCRIPTION = ("Batch Label Reconstruction (local gradient-matching adaptation; "
               "VFLAIR-LLM BatchLabelReconstruction_LLM clean-room): the "
               "LM analogue is first-batch token reconstruction — dummy "
               "soft labels + dummy seam input optimized so simulated "
               "body gradients match the captured step-0 body grads "
               "(+ seam bwd), L2 + Adam; label-free. Toy tier: "
               "closed-form least-squares label recovery on a linear "
               "body (planted >95%, null at control)")

SCHEMA = "dtraining.attacker.batch_label_reconstruction.v1"
TOY_RIDGE_LAMBDA = 1e-8

THREAT_MODEL = (
    "honest-but-curious server observing the FIRST training batch's "
    "backward pass: the seam gradient (phase='bwd', step=0 wire "
    "capture) AND the per-parameter body-slice gradients "
    "(body_grads_step0.pt, splitbench --body-grad-capture); the body "
    "slice itself is granted white-box via --model + --split at the "
    "capture's architecture (at step 0 the body is the pre-update body, "
    "so initial/public weights are the captured body — no finetuned "
    "state assumed). Canonical tokens are evaluation truth, never an "
    "attack input (label-free attack: labels are INFERRED from the "
    "gradients).")

INTERPRETATION = (
    "LM analogue of classification BLI: the 'batch labels' are the "
    "next-token targets of the first training batch, so the attack "
    "reconstructs the batch's tokens shifted by one (position 0 is "
    "nobody's label and is not scored). token_set_precision/recall are "
    "orderless bag-of-tokens overlaps (BLI's native multiset metric); "
    "seq_accuracy is position-aligned vs canonical (BLI's harder "
    "assignment step); the control decodes the mode of the attacker's "
    "OWN recovered bag at every position (attacker-computable without "
    "canonical); the gate verdict is the frozen-gate Bonferroni-Wilson "
    "excess over that control with n = scored tokens. Toy tier is a "
    "machinery check of the least-squares/decode/scoring/gate plumbing "
    "on a KNOWN linear body with KNOWN inputs (grants the LM tier "
    "co-optimizes), not a security claim; real numbers come from "
    "--capture-dir runs.")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, hidden=16)
    ap.add_argument("--capture-dir", default=None,
                    help="wire-capture directory (wire-training schema; "
                         "needs the phase='bwd' step=0 seam capture and, "
                         "unless --body-grads-pt redirects it, "
                         "body_grads_step0.pt from splitbench "
                         "--body-grad-capture)")
    ap.add_argument("--canonical-pt", default=None,
                    help="canonical.pt ground-truth input_ids (splitbench "
                         "--save-canonical); the first-batch rows are "
                         "evaluation truth for the scoring cells")
    ap.add_argument("--model", default=None,
                    help="HF name/path supplying the body slice (white-box "
                         "grant; the step-0 body is the pre-update body) "
                         "and the vocab for the soft-label dummy")
    ap.add_argument("--split", default="6",
                    help="seam boundary of the body slice (must match the "
                         "capture harness split; default '6')")
    ap.add_argument("--body-grads-pt", default=None,
                    help="captured per-parameter body grads {name: tensor} "
                         "(default <capture-dir>/body_grads_step0.pt)")
    ap.add_argument("--body-state-pt", default=None,
                    help="optional saved body state dict loaded onto the "
                         "--model arch (strict=False) when the captured "
                         "body is not the public init (splitbench does "
                         "not export one — step-0 bodies need none)")
    ap.add_argument("--steps", type=int, default=200,
                    help="Adam steps for the gradient-matching "
                         "optimization (default 200; the honest budget "
                         "for above-control recovery on tiny models is "
                         "a few hundred steps — 30-step smoke runs "
                         "validate machinery, not recovery)")
    ap.add_argument("--lr", type=float, default=0.1,
                    help="Adam lr for the dummy soft labels / seam input")
    ap.add_argument("--seam-weight", type=float, default=1.0,
                    help="weight of the seam-gradient L2 match term "
                         "(0 = match body param grads only)")
    ap.add_argument("--device", default=None, choices=("cpu", "cuda"),
                    help="torch device (default: cuda when available)")
    ap.add_argument("--vocab", type=int, default=12, help="toy vocab size")
    ap.add_argument("--seq-len", type=int, default=4,
                    help="toy victim sequence length (labels per seq)")
    ap.add_argument("--toy-seqs", type=int, default=4,
                    help="toy victim sequences per cell (toy batch B = "
                         "toy-seqs * seq-len positions)")
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
                          node_role="attacker/batch-label-recon")


def _softmax(row):
    m = max(row)
    exps = [math.exp(v - m) for v in row]
    z = sum(exps)
    return [e / z for e in exps]


def _gauss_solve(M, B):
    """Solve M X = B by Gauss-Jordan elimination with partial pivoting.
    M [n][n], B [n][k]; returns X [n][k]. Raises ValueError on a
    (near-)singular pivot — a solve failure the caller journals as a
    per-cell error record, never a crash."""
    n = len(M)
    aug = [list(M[i]) + list(B[i]) for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            raise ValueError("least-squares normal equations are "
                             "(near-)singular; check the toy geometry "
                             "(batch positions vs dim) or the ridge")
        aug[col], aug[piv] = aug[piv], aug[col]
        inv = 1.0 / aug[col][col]
        aug[col] = [v * inv for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                f = aug[r][col]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def set_precision_recall(pred, true):
    """Orderless per-sequence token-set overlap — the bag-of-tokens
    metric, the direct analogue of BLI's recovered label multiset."""
    ps, ts = set(pred), set(true)
    inter = len(ps & ts)
    return inter / max(1, len(ps)), inter / max(1, len(ts))


def seq_accuracy(preds, trues):
    """Position-aligned per-token accuracy, pooled over sequences.
    Returns (accuracy, n_correct, n_tokens). Length mismatches are a
    scoring bug and raise (journaled as a cell error by the caller)."""
    n_correct = 0
    n_tokens = 0
    for p, t in zip(preds, trues):
        if len(p) != len(t):
            raise ValueError(f"pred/true length mismatch ({len(p)} vs "
                             f"{len(t)}): position-aligned scoring needs "
                             "aligned sequences")
        n_correct += sum(1 for a, b in zip(p, t) if a == b)
        n_tokens += len(t)
    return (n_correct / max(1, n_tokens), n_correct, n_tokens)


def bag_mode_control(pred_lists):
    """The label-free control token: the mode of the attacker's OWN
    recovered bag (decoded at every position). Canonical is never
    consulted; an empty bag is a scoring bug and raises."""
    counts = collections.Counter(t for seq in pred_lists for t in seq)
    if not counts:
        raise ValueError("empty recovered bag: no control token")
    return counts.most_common(1)[0][0]


def score_batch(preds, trues, control_tok, n_seeds=None):
    """The full per-cell metric bundle (both tiers share this): bag
    precision/recall, position-aligned accuracy, bag-mode control
    accuracy, and the frozen-gate verdict (excess over control,
    n = scored tokens)."""
    prs = [set_precision_recall(p, t) for p, t in zip(preds, trues)]
    acc, n_correct, n_tokens = seq_accuracy(preds, trues)
    c_acc, c_correct, _ = seq_accuracy([[control_tok] * len(t)
                                        for t in trues], trues)
    return {"n_seq": len(trues),
            "n_tokens": n_tokens,
            "token_set_precision": round(sum(p for p, _ in prs)
                                         / max(1, len(prs)), 6),
            "token_set_recall": round(sum(r for _, r in prs)
                                      / max(1, len(prs)), 6),
            "seq_accuracy": round(acc, 6),
            "n_correct": n_correct,
            "control_token": control_tok,
            "control_accuracy": round(c_acc, 6),
            "n_control_correct": c_correct,
            "gate": gate.verdict(acc, c_acc, n_tokens, n_seeds=n_seeds)}


def toy_world(seed, vocab=12, dim=16, n_seq=4, seq_len=4, w_scale=0.5):
    """Synthetic linear-body batch world. W [vocab, dim] random and KNOWN
    to the attacker (the white-box toy analogue of the granted body
    slice); X [B, dim] random and KNOWN (the toy grant — the LM tier
    co-optimizes the seam input instead); labels uniform per position.
    The captured per-param grads are computed from the world's actual
    forward math: p_i = softmax(W x_i), r_i = p_i - onehot(t_i),
    G_W = sum_i r_i x_i^T (and G_b = sum_i r_i, reported but unused by
    the recovery)."""
    rng = random.Random(seed)
    W = [[w_scale * rng.gauss(0.0, 1.0) for _ in range(dim)]
         for _ in range(vocab)]
    B = n_seq * seq_len
    X = [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(B)]
    labels = [[rng.randrange(vocab) for _ in range(seq_len)]
              for _ in range(n_seq)]
    flat = [t for seq in labels for t in seq]
    P, R = [], []
    for x, t in zip(X, flat):
        p = _softmax([sum(w * xi for w, xi in zip(W[v], x))
                      for v in range(vocab)])
        P.append(p)
        R.append([pv - (1.0 if v == t else 0.0)
                  for v, pv in enumerate(p)])
    GW = [[sum(R[i][v] * X[i][k] for i in range(B)) for k in range(dim)]
          for v in range(vocab)]
    GB = [sum(R[i][v] for i in range(B)) for v in range(vocab)]
    return {"W": W, "X": X, "P": P, "GW": GW, "GB": GB, "labels": labels,
            "vocab": vocab, "dim": dim, "seq_len": seq_len}


def null_grads_like(GW, rng):
    """The null observation: random grads at the captured grad's matched
    per-entry scale (no batch information — the negative control)."""
    n = sum(len(row) for row in GW)
    var = sum(v * v for row in GW for v in row) / max(1, n)
    s = math.sqrt(var)
    return [[rng.gauss(0.0, s) for _ in row] for row in GW]


def toy_recover(W, X, P, GW, lam=TOY_RIDGE_LAMBDA):
    """Closed-form least-squares label recovery (the linear case).

    G_W = R X with R [V, B] the per-position residual columns, so
    R = G_W X^T (X X^T + lam I)^-1 — solved via the normal equations on
    the [B, B] Gram (B <= dim in the default geometry, so the planted
    case is exact). The label is the negative residual spike:
    t_i = argmin_v (r_i[v] - p_i[v]) — for CE, r_i = p_i - onehot(t_i),
    so r_i - p_i is -1 exactly at the label (the iDLG sign argument).
    Returns the per-sequence predicted label lists."""
    B = len(X)
    dim = len(X[0])
    vocab = len(W)
    gram = [[sum(X[i][k] * X[j][k] for k in range(dim))
             + (lam if i == j else 0.0) for j in range(B)]
            for i in range(B)]
    CT = [[sum(GW[v][k] * X[i][k] for k in range(dim))
           for v in range(vocab)] for i in range(B)]
    RT = _gauss_solve(gram, CT)
    preds = [min(range(vocab), key=lambda v: RT[i][v] - P[i][v])
             for i in range(B)]
    return preds, RT


def _regroup(flat, n_seq, seq_len):
    return [flat[i * seq_len:(i + 1) * seq_len] for i in range(n_seq)]


def toy_cell(args, source, seed):
    """One toy cell: least-squares recover, decode, score with the
    shared plumbing. source 'planted' uses the world's true captured
    grads; source 'null' uses scale-matched random grads (negative
    control)."""
    world = toy_world(args.seed + 977 * seed, vocab=args.vocab,
                      dim=args.hidden, n_seq=args.toy_seqs,
                      seq_len=args.seq_len)
    GW = world["GW"] if source == "planted" else null_grads_like(
        world["GW"], random.Random(args.seed + 31337 + seed))
    preds_flat, _ = toy_recover(world["W"], world["X"], world["P"], GW)
    preds = _regroup(preds_flat, args.toy_seqs, args.seq_len)
    control = bag_mode_control(preds)
    return {"source": source, "seed": seed,
            **score_batch(preds, world["labels"], control,
                          n_seeds=len(args.seeds))}


def _run_toy(args, ev):
    out = artifacts.make_artifact(
        SCHEMA,
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "vocab": args.vocab, "dim": args.hidden, "seq_len": args.seq_len,
         "toy_seqs": args.toy_seqs, "seeds": args.seeds,
         "ridge_lambda": TOY_RIDGE_LAMBDA},
        THREAT_MODEL + " Toy tier: synthetic KNOWN linear body with KNOWN "
        "inputs (grants the LM tier co-optimizes); measures "
        "least-squares/decode/scoring/gate plumbing, not a real model.",
        interpretation=INTERPRETATION,
        evidence=ev)
    cells = []
    for source in ("planted", "null"):
        for seed in args.seeds:
            cell = {"experiment": EXPERIMENT_ID, "arm": "toy-least-squares",
                    "submode": "linear-body-closed-form"}
            try:
                cell.update(toy_cell(args, source, seed))
            except Exception as e:
                journal_error(args.output, EXPERIMENT_ID, cell, e)
                continue
            artifacts.append_jsonl(args.output, cell)
            if ev is not None:
                ev.event("cell_result", cell)
            out["results"].append(cell)
            cells.append(cell)
            print(f"[blr-toy] {source} seed={seed}: "
                  f"acc={cell['seq_accuracy']} control="
                  f"{cell['control_accuracy']} setP="
                  f"{cell['token_set_precision']} "
                  f"gate={cell['gate']['verdict']}")
    for source in ("planted", "null"):
        vals = [c for c in cells if c["source"] == source]
        if vals:
            out["summary"].append({
                "experiment": EXPERIMENT_ID, "source": source,
                "seq_accuracy_mean": sum(c["seq_accuracy"] for c in vals)
                / len(vals),
                "control_accuracy_mean":
                    sum(c["control_accuracy"] for c in vals) / len(vals),
                "token_set_precision_mean":
                    sum(c["token_set_precision"] for c in vals) / len(vals),
                "gate_verdicts": [c["gate"]["verdict"] for c in vals],
                "n_cells": len(vals)})
    artifacts.write_artifact(args.output, out, evidence=ev)
    return 0


def _load_hf_model(name_or_path, device):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise SystemExit(f"[{EXPERIMENT_ID}] --model requires transformers "
                         "(pip install transformers)") from None
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path, torch_dtype=torch.float32,
        attn_implementation="eager")
    return model.to(device).eval()


def _load_body_grads(args):
    path = args.body_grads_pt
    if path is None:
        import os
        path = os.path.join(args.capture_dir, "body_grads_step0.pt")
    try:
        grads = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError) as e:
        raise SystemExit(f"[{EXPERIMENT_ID}] cannot load body grads "
                         f"{path}: {e} (run splitbench finetune with "
                         "--body-grad-capture)") from None
    if not isinstance(grads, dict) or not grads \
            or not all(torch.is_tensor(v) for v in grads.values()):
        raise SystemExit(f"[{EXPERIMENT_ID}] {path} is not a {{name: "
                         "tensor}} body-grads dump (splitbench "
                         "--body-grad-capture)")
    return {name: g.float() for name, g in grads.items()}, path


def _load_seam_bwd(args):
    """The step-0 phase='bwd' seam capture: supplies the batch geometry
    [b, s, H] (wire-visible metadata) and the seam-gradient match
    target. Part of the threat model; a descriptive error when absent."""
    from .. import captures
    records = [(m, p) for m, p in
               captures.scan_captures(args.capture_dir, "training")
               if m["phase"] == "bwd" and m["step"] == 0]
    if not records:
        raise SystemExit(f"[{EXPERIMENT_ID}] no phase='bwd' step=0 seam "
                         f"capture in {args.capture_dir} (the seam "
                         "gradient is part of the observation; splitbench "
                         "always captures step 0)")
    return captures.load_tensor(records[0][1]).float()


def _run_real(args, ev):
    require_torch(EXPERIMENT_ID)
    from ..splitbench import SplitPlan, build_split
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    body_grads, grads_path = _load_body_grads(args)
    seam_bwd = _load_seam_bwd(args).to(device)
    if seam_bwd.dim() == 2:
        seam_bwd = seam_bwd.unsqueeze(0)
    b, s, h_dim = seam_bwd.shape

    canonical = torch.load(args.canonical_pt, map_location="cpu",
                           weights_only=True).long()
    if canonical.shape[0] < b or canonical.shape[1] < s:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] canonical/capture misalignment: the "
            f"step-0 batch is [{b}, {s}] but canonical has shape "
            f"{tuple(canonical.shape)} (canonical.pt is row-aligned to "
            "the fwd captures, step-major; the first b rows are the "
            "first batch)")
    trues = canonical[:b, 1:s].tolist()

    model = _load_hf_model(args.model, device)
    if args.body_state_pt:
        state = torch.load(args.body_state_pt, map_location="cpu",
                           weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise SystemExit(f"[{EXPERIMENT_ID}] --body-state-pt has "
                             f"{len(unexpected)} unexpected keys "
                             f"(e.g. {unexpected[0]!r}); the body state "
                             "must match the --model arch")
    split = build_split(model, SplitPlan.parse(args.split))
    vocab = model.get_input_embeddings().weight.shape[0]

    named = dict(model.named_parameters())
    names = [n for n in body_grads if n in named]
    if not names:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] none of the {len(body_grads)} captured "
            "body-grad names match the --model parameters (e.g. "
            f"{next(iter(body_grads))!r}); --model + --split must be the "
            "SAME architecture/wrapping as the capture")
    params = [named[n] for n in names]
    targets = [body_grads[n].to(device) for n in names]

    out = artifacts.make_artifact(
        SCHEMA,
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": False,
         "capture_dir": args.capture_dir, "canonical_pt": args.canonical_pt,
         "model": args.model, "split": args.split,
         "body_grads_pt": grads_path, "body_state_pt": args.body_state_pt,
         "steps": args.steps, "lr": args.lr, "seam_weight": args.seam_weight,
         "device": device, "batch": b, "seq_len": s, "h_dim": h_dim,
         "n_body_grads_captured": len(body_grads),
         "n_body_grads_matched": len(names)},
        THREAT_MODEL, interpretation=INTERPRETATION, evidence=ev)

    cell = {"experiment": EXPERIMENT_ID, "arm": "grad-match",
            "submode": "soft-label-dlg", "mode": args.mode,
            "step": 0, "epoch": 0,
            "surface": f"body-grads:{len(names)}/{len(body_grads)}"
                       f"+seam-bwd(w={args.seam_weight})"}
    try:
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        h_d = (0.1 * torch.randn(b, s, h_dim, generator=g)).to(device)
        Z = (0.01 * torch.randn(b, s - 1, vocab, generator=g)).to(device)
        h_d.requires_grad_(True)
        Z.requires_grad_(True)
        opt = torch.optim.Adam([h_d, Z], lr=args.lr)
        match = None
        for _ in range(args.steps):
            opt.zero_grad()
            logits = split.rest_forward(h_d)
            logp = torch.nn.functional.log_softmax(
                logits[:, :-1].float(), dim=-1)
            soft = torch.nn.functional.softmax(Z.float(), dim=-1)
            loss_sim = -(soft * logp).sum(-1).mean()
            grads_sim = torch.autograd.grad(
                loss_sim, params + [h_d], create_graph=True)
            match = sum(((gs - t) ** 2).sum()
                        for gs, t in zip(grads_sim[:-1], targets))
            if args.seam_weight > 0.0:
                match = match + args.seam_weight \
                    * ((grads_sim[-1] - seam_bwd) ** 2).sum()
            match.backward()
            opt.step()
        pred = Z.detach().argmax(-1).cpu().tolist()
        control = bag_mode_control(pred)
        cell.update(score_batch(pred, trues, control))
        cell["final_match_loss"] = round(float(match.item()), 8)
    except Exception as e:
        journal_error(args.output, EXPERIMENT_ID, cell, e)
        artifacts.write_artifact(args.output, out, evidence=ev)
        return 0
    artifacts.append_jsonl(args.output, cell)
    if ev is not None:
        ev.event("cell_result", cell)
    out["results"].append(cell)
    out["summary"].append({
        "experiment": EXPERIMENT_ID, "arm": "grad-match",
        "n_cells": 1, "n_tokens": cell["n_tokens"],
        "token_set_precision_mean": cell["token_set_precision"],
        "token_set_recall_mean": cell["token_set_recall"],
        "seq_accuracy": cell["seq_accuracy"],
        "control_accuracy": cell["control_accuracy"],
        "gate": cell["gate"],
        "gate_verdicts": [cell["gate"]["verdict"]]})
    artifacts.write_artifact(args.output, out, evidence=ev)
    print(f"[batch-label-recon] step=0: acc={cell['seq_accuracy']} "
          f"control={cell['control_accuracy']} setP="
          f"{cell['token_set_precision']} match_loss="
          f"{cell['final_match_loss']} gate={cell['gate']['verdict']}")
    return 0


def run(args):
    ev = _make_evidence(args)
    if args.toy:
        if args.quick:
            args.seeds = [0]
            args.toy_seqs = min(args.toy_seqs, 4)
        if args.toy_seqs < 1 or args.seq_len < 1:
            raise SystemExit(f"[{EXPERIMENT_ID}] --toy-seqs and "
                             "--seq-len must be >= 1")
        return _run_toy(args, ev)
    missing = [flag for flag, val in (("--capture-dir", args.capture_dir),
                                      ("--canonical-pt", args.canonical_pt),
                                      ("--model", args.model)) if not val]
    if missing:
        raise SystemExit(f"[{EXPERIMENT_ID}] pass --toy, or the real-tier "
                         f"arguments (missing: {', '.join(missing)}): "
                         "--capture-dir D --canonical-pt C --model M "
                         "(+ --split at the capture's boundary)")
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

    print("batch-label-recon toy core (pure python):")

    p = _softmax([1.0, 2.0, 3.0])
    check("softmax normalizes and orders",
          abs(sum(p) - 1.0) < 1e-12 and p[0] < p[1] < p[2])
    X = _gauss_solve([[2.0, 1.0], [1.0, 3.0]], [[1.0], [2.0]])
    check("Gauss-Jordan solve matches the hand solution",
          abs(X[0][0] - 0.2) < 1e-12 and abs(X[1][0] - 0.6) < 1e-12)
    try:
        _gauss_solve([[1.0, 2.0], [2.0, 4.0]], [[1.0], [2.0]])
        check("singular system raises (journaled, never a crash)", False)
    except ValueError as e:
        check("singular system raises (journaled, never a crash)",
              "singular" in str(e))

    p_, r_ = set_precision_recall([1, 2, 3], [2, 3, 4])
    check("bag precision/recall fixture",
          abs(p_ - 2 / 3) < 1e-12 and abs(r_ - 2 / 3) < 1e-12)
    acc, nc, nt = seq_accuracy([[1, 2], [3, 4]], [[1, 9], [3, 4]])
    check("position-aligned pooled accuracy fixture",
          nc == 3 and nt == 4 and acc == 0.75)
    check("bag-mode control is the recovered bag's mode",
          bag_mode_control([[5, 5, 7], [5, 7]]) == 5)

    args = build_parser().parse_args(["--toy"])
    args.mode = "training"
    w1 = toy_world(0, vocab=12, dim=16, n_seq=4, seq_len=4)
    w2 = toy_world(0, vocab=12, dim=16, n_seq=4, seq_len=4)
    check("toy world is deterministic and shaped",
          w1["labels"] == w2["labels"] and len(w1["GW"]) == 12
          and len(w1["GW"][0]) == 16)
    for seed in (0, 1, 2):
        cell = toy_cell(args, "planted", seed)
        check(f"planted toy recovers >95% of labels (seed {seed})",
              cell["seq_accuracy"] > 0.95
              and cell["token_set_recall"] > 0.95
              and cell["gate"]["verdict"] == "FAIL")
    null = toy_cell(args, "null", 0)
    planted = toy_cell(args, "planted", 0)
    check("null (random grads) sits at control, at the gate floor",
          null["seq_accuracy"] < 0.5
          and abs(null["seq_accuracy"] - null["control_accuracy"]) < 0.25
          and null["gate"]["at_floor"]
          and null["gate"]["excess_pp"] < planted["gate"]["excess_pp"]
          and not planted["gate"]["at_floor"])

    check("parser builds torch-less",
          build_parser().parse_args([]).split == "6")
    ns = build_parser().parse_args([])
    ns.mode = "training"
    try:
        run(ns)
        check("no --toy/real args is a descriptive error", False)
    except SystemExit as e:
        check("no --toy/real args is a descriptive error",
              "--capture-dir" in str(e))
    if torch is None:
        ns2 = build_parser().parse_args(
            ["--capture-dir", "/nonexistent", "--canonical-pt",
             "/nonexistent", "--model", "x"])
        ns2.mode = "training"
        try:
            run(ns2)
            check("real tier is torch-guarded", False)
        except SystemExit as e:
            check("real tier is torch-guarded", "requires torch" in str(e))

    with tempfile.TemporaryDirectory() as d:
        ns = build_parser().parse_args(
            ["--toy", "--quick", "--evidence",
             "--output", os.path.join(d, "blr.json")])
        ns.mode = "training"
        check("toy run end to end torch-free", run(ns) == 0)
        art = json.load(open(os.path.join(d, "blr.json")))
        check("toy artifact has planted+null cells with gate verdicts",
              len(art["results"]) == 2
              and {c["source"] for c in art["results"]}
              == {"planted", "null"}
              and all(c["gate"]["verdict"] in ("PASS", "FAIL")
                      for c in art["results"]))
        check("journal lines match result cells",
              len(open(os.path.join(d, "blr.json.jsonl")).read()
                  .splitlines()) == len(art["results"]))
        from ..evidence import verify_evidence
        good, problems = verify_evidence(
            os.path.join(d, "blr.json.evidence.json"))
        check("evidence chain verifies", good and problems == [])

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
