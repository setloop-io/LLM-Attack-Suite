#!/usr/bin/env python3
"""Citation scope (reference audit): these are local VFLAIR-LLM-inspired
adaptations. In particular, public-head ridge calibration and the joint
beam search do not establish published-setting VMI/WBI fidelity. The
gradient-direction score is not TAG or LAMP's published algorithm.
See docs/reference-audit-2026-09-22.md.

seam-inversion (training + inference modes) — model-inversion recovery of
the victim's input tokens from fwd seam activations h* (VFLAIR-LLM's VMI /
WBI line, transplanted to the split-LLM seam).

Threat model (WHITE-BOX note, mirrored in every artifact): the attacker
observes the seam tensor h* crossing to the server-side body and is granted
the victim's TOKEN EMBEDDING MATRIX E — public base weights in practice
(--model gpt2 gives wte) — exactly the VFLAIR-LLM VMI assumption. With
--head-pt the victim's head transform itself is also granted (the strongest
variant). With only --model, calibration forwards run through the PUBLIC
(pre-finetune) head: the victim's finetuned head is NOT assumed, and the
resulting mismatch is part of the honest measurement. Canonical tokens are
evaluation truth, never an attack input (label-free attack).

Arms (--arm):
  vmi   Vanilla Model Inversion. Two sub-modes:
        (a) --head-pt FILE  — optimize a dummy per-position embedding
            sequence Z [seq, d] through the KNOWN victim head transform,
            min_Z ||Head(Z + pos) - h*||^2 (the seam sits AFTER the head,
            so the head's computation is the inversion target). head-pt is
            a saved head state + arch spec (llm_attacker.seam_head.v1, see
            save_head_pt); splitbench will export it natively later.
        (b) --linearization — the default when no --head-pt is given:
            estimate the local linear map seam-row -> token-embedding
            directly. Fit a ridge regression on a small PUBLIC calibration
            set (synthetic or --calibration-file public text pushed through
            the same public head) mapping seam rows to token-embedding
            space, then cosine nearest-neighbor decode over E. This is the
            honest black-box-ish arm and runs on today's splitbench GPT-2
            captures with nothing but the public embedding matrix. With
            --embedding-pt only (no head at all) it degrades to direct NN
            decode of h* rows over E (documented weakest variant).
  wbi   soft one-hot relaxation (arXiv:2004.00053): optimize Z [seq, vocab]
        with softmax(Z/T) @ E as the relaxed embedding sequence, toward the
        same target — through the --head-pt head when given, else through
        the forward ridge map e -> h fit on the same calibration set.
        Torch tier.
  joint fwd+bwd joint inversion (training mode; needs both capture
        phases): decode per-position CANDIDATE SETS from the bwd seam
        gradient's per-token direction signal (the TAG/LAMP cosine score
        cos(-g*_i, E[v]); position 0 has no label gradient, so its
        candidates come from the fwd linearization decode), then
        re-rank/verify by FORWARD-CONSISTENCY: a candidate sequence's
        public-head forward must reproduce the captured h*. The head is
        causal, so the residual ||Head(ids)_<=i - h*_<=i||^2 accumulates
        left-to-right and a beam search (--joint-beam) over the candidate
        sets (--joint-candidates) uses approximate finite-width pruning. It
        uses both captures; every cell also reports the fwd-only
        (linearization) and bwd-only (direction decode + control token at
        position 0) diagnostics. These have different compute costs; use
        experiments/compare_baselines.py for measured, controlled comparisons.

Defense-aware adaptivity (--adaptive {none,noise-whitening}, Tramèr
doctrine: the gaussian seam defense and its sigma are PUBLIC, passed as
--defense-sigma):
  * linearization arm: the plain ridge fit is already NOISE-NAIVE by
    construction — the attacker runs the calibration forwards locally, so
    the defense (which lives at the victim's seam) never touched them;
    the fit maps CLEAN head rows to embeddings while the deployed wire
    rows are h* + n, and the decode noise n @ W_dec is amplified by
    ||W_dec||. The adaptive variant fits the ridge on NOISE-AUGMENTED
    calibration rows (h_cal + n_k paired with the same e_cal targets,
    --noise-samples K antithetic draws at the public sigma): in the
    K*N_cal -> infinity limit this converges to the noise-marginalized
    (Wiener) linear estimator (H' H + sigma^2 N I)^{-1} H' E — ridge
    lambda_eff = sigma^2 * N_cal — which is exactly the toy tier's
    closed-form adaptive map (lambda = sigma^2).
  * vmi/wbi optimizer arms: the plain objective min_Z ||Head(Z) - h_wire||^2
    fits the single captured noise realization n_a inside h_wire = h* + n_a.
    The adaptive objective marginalizes the defense noise INSIDE the
    inversion: min_Z E_n[||Head(Z) - (h_wire - n)||^2] estimated with K
    antithetic draws per step (n, -n pairs — variance reduction). CANDID
    NOTE (measured, see tests/test_adaptive_inversion.py): for a SINGLE
    captured h_wire this expectation has the SAME population minimizer as
    the plain objective (the cross term vanishes since E[n] = 0), so the
    vmi arm's gain is the target-jitter regularization of the trajectory,
    not a different optimum — parity with plain is the honest expectation
    there; the strict denoising gain lives in the linearization arm
    (Wiener shrinkage of the noise-amplified decode map).

Resumability: --resume DIR checkpoints the vmi-head optimizer loop (Z +
Adam moments) every --checkpoint-every steps via attacker.checkpoint and
resumes mid-loop after a kill (one stream per cell under DIR; same-device
restart). Arms/sub-modes without an optimizer loop ignore the flag.

Decode: cosine nearest neighbor over E (--model HF name/path or
--embedding-pt FILE); wbi decodes as argmax_v Z (the relaxation's own
convention).

Scoring (ALL reported per cell, VFLAIR conventions): orderless token-set
precision/recall (set overlap per sequence), sequence-level token accuracy
(position-aligned vs canonical, pooled per token), and a majority-token
control (the calibration-corpus mode token decoded at every position —
attacker-computable, no canonical access) with an attacker.gate verdict of
the Bonferroni-Wilson excess over that control, n = scored tokens. Per-step
cells + a pooled summary.

Usage:
    python -m attacker --mode training --attack seam-inversion --help
    python -m attacker --mode training --attack seam-inversion --toy \
        --quick --output /tmp/si.json
    python -m attacker --mode training --attack seam-inversion \
        --capture-dir CAPS --canonical-pt CAPS/canonical.pt --model gpt2 \
        --arm vmi --steps 200 --samples 8 --output si.json --evidence

Toy tier (torch-free, honest): tiny synthetic seam — vocab 40, d=8, random
E; h = E[tok] @ A + gaussian noise with KNOWN A (the white-box toy analogue
of the victim head). Inversion is a pure-python ridge solve (normal
equations via Gaussian elimination) + NN decode — no optimizer loop (the
optimizer loop is torch-tier). Recovers >90% of tokens at low noise and
degrades gracefully as noise grows; this validates the decode / scoring /
gate plumbing that both arms share.
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

EXPERIMENT_ID = "seam_inversion"
MODES = ("training", "inference")
REQUIRES_LABELS = False
DESCRIPTION = ("VMI/WBI seam inversion of victim input tokens from fwd "
               "seam activations (VFLAIR-LLM line); white-box note: the "
               "threat model grants the victim's embedding matrix E "
               "(public base weights), with --head-pt also the head "
               "transform. Ridge-linearization + soft-relaxation arms, "
               "cosine-NN decode, majority-control gate verdicts")

SCHEMA = "dtraining.attacker.seam_inversion.v1"
HEAD_PT_FORMAT = "llm_attacker.seam_head.v1"
ARMS = ("vmi", "wbi", "joint")
ADAPTIVE_MODES = ("none", "noise-whitening")
TOY_RIDGE_LAMBDA = 1e-6

THREAT_MODEL = (
    "honest-but-curious seam observer of fwd seam activations h* (training: "
    "the tensor the body receives; inference: prefill seam rows), granted "
    "the victim's token embedding matrix E (public base weights — the "
    "VFLAIR-LLM VMI white-box assumption); --head-pt additionally grants "
    "the victim head transform. With --model alone, calibration forwards "
    "use the PUBLIC head — the victim's finetuned head is not assumed and "
    "the mismatch is part of the measurement. Canonical tokens are "
    "evaluation truth, never an attack input (label-free attack).")

INTERPRETATION = (
    "token-set precision/recall are orderless per-sequence set overlaps "
    "(VFLAIR convention); seq_accuracy is position-aligned vs canonical, "
    "pooled per token; the control decodes the calibration-corpus majority "
    "token at every position (attacker-computable without canonical); the "
    "gate verdict is the frozen-gate Bonferroni-Wilson excess over that "
    "control with n = scored tokens. Toy tier is a machinery check of the "
    "shared ridge/decode/scoring/gate plumbing on a synthetic KNOWN-map "
    "seam, not a security claim; real numbers come from --capture-dir runs.")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, hidden=8)
    ap.add_argument("--calibration-pt", default=None, help="public calibration token tensor [documents, sequence]; distinct from scoring truth")
    ap.add_argument("--arm", choices=ARMS, default="vmi",
                    help="attack arm (default vmi); joint = fwd+bwd joint "
                         "beam inversion (training mode only, needs both "
                         "capture phases)")
    ap.add_argument("--adaptive", choices=ADAPTIVE_MODES, default="none",
                    help="defense-aware adaptivity (Tramer doctrine: the "
                         "defense and its sigma are public). none = the "
                         "historical behavior (default). noise-whitening: "
                         "model the gaussian seam noise INSIDE the "
                         "inversion — noise-marginalized vmi/wbi objective "
                         "(--noise-samples antithetic draws) and "
                         "noise-augmented ridge calibration at the public "
                         "--defense-sigma")
    ap.add_argument("--noise-samples", type=int, default=8,
                    help="noise draws per objective/calibration batch for "
                         "--adaptive noise-whitening (default 8; drawn as "
                         "antithetic +/- pairs for variance reduction)")
    ap.add_argument("--defense-sigma", type=float, default=0.0,
                    help="PUBLIC sigma of the deployed gaussian seam "
                         "defense (Tramer adaptive doctrine: the defense "
                         "parameters are known). 0 = no deployed noise "
                         "(adaptive degenerates to plain, noted in the "
                         "run)")
    ap.add_argument("--joint-candidates", type=int, default=8,
                    help="joint arm: per-position candidate-set size from "
                         "the bwd direction signal (fwd linearization at "
                         "position 0)")
    ap.add_argument("--joint-beam", type=int, default=4,
                    help="joint arm: beam width of the forward-consistency "
                         "rerank search")
    ap.add_argument("--joint-grid", type=int, default=4,
                    help="joint toy tier: grid side k of the k x k "
                         "complementary-rank world (vocab = k^2)")
    ap.add_argument("--capture-dir", default=None,
                    help="wire-capture directory (wire-training schema in "
                         "training mode: fwd seam activations; inference "
                         "mode scores prefill seam rows)")
    ap.add_argument("--canonical-pt", default=None,
                    help="canonical.pt ground-truth input_ids, LongTensor "
                         "[n_captured*batch, seq] row-aligned to the fwd "
                         "captures step-major (splitbench --save-canonical)")
    ap.add_argument("--model", default=None,
                    help="HF name/path of the PUBLIC base model: supplies "
                         "the embedding matrix E and the public head for "
                         "calibration forwards (split at --split)")
    ap.add_argument("--embedding-pt", default=None,
                    help=".pt with the embedding matrix E ([V, d] tensor, a "
                         "state dict carrying wte/embed_tokens, or a "
                         "head-pt blob) — the no-head variant")
    ap.add_argument("--head-pt", default=None,
                    help="saved victim head state + arch spec "
                         f"({HEAD_PT_FORMAT}; see save_head_pt). Grants the "
                         "KNOWN head transform: vmi optimizes through it, "
                         "wbi relaxes through it")
    ap.add_argument("--linearization", action="store_true",
                    help="force the ridge-linearization sub-mode even when "
                         "--head-pt is given (default: linearization "
                         "whenever no --head-pt is given)")
    ap.add_argument("--split", default="6",
                    help="seam boundary of the PUBLIC head used for "
                         "calibration forwards (must match the capture "
                         "harness split; default '6')")
    ap.add_argument("--steps", type=int, default=200,
                    help="optimization steps for the vmi-head / wbi "
                         "optimizer paths (linearization decode has no "
                         "loop and ignores this)")
    ap.add_argument("--resume", default=None, metavar="DIR",
                    help="checkpoint directory for the vmi-head optimizer "
                         "loop: state (Z + Adam moments) is checkpointed "
                         "every --checkpoint-every steps via "
                         "attacker.checkpoint (one stream per cell under "
                         "DIR) and the loop resumes from the latest "
                         "checkpoint found there — kill-and-restart safe "
                         "(same-device restart). Sub-modes without an "
                         "optimizer loop ignore this")
    ap.add_argument("--checkpoint-every", type=int, default=50,
                    help="vmi-head optimizer checkpoint cadence in steps "
                         "(default 50; step 0 is always checkpointed)")
    ap.add_argument("--samples", type=int, default=8,
                    help="cap on scored sequences (scan order)")
    ap.add_argument("--lr", type=float, default=0.1,
                    help="Adam lr for the optimizer paths")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="softmax temperature T for the wbi relaxation")
    ap.add_argument("--ridge-lambda", type=float, default=0.0,
                    help="ridge lambda; <= 0 = auto "
                         "(1e-3 * mean diagonal of the Gram matrix)")
    ap.add_argument("--calibration-seqs", type=int, default=64,
                    help="public calibration sequences for the ridge fit")
    ap.add_argument("--calibration-file", default=None,
                    help="public text file tokenized with the --model "
                         "tokenizer for calibration (default: synthetic "
                         "uniform token draws — no public text assumed)")
    ap.add_argument("--device", default=None, choices=("cpu", "cuda"),
                    help="torch device (default: cuda when available)")
    ap.add_argument("--vocab", type=int, default=40, help="toy vocab size")
    ap.add_argument("--seq-len", type=int, default=16,
                    help="toy victim sequence length")
    ap.add_argument("--toy-seqs", type=int, default=32,
                    help="toy victim sequences per cell")
    ap.add_argument("--toy-cal-tokens", type=int, default=4000,
                    help="toy calibration-corpus tokens (control mode)")
    ap.add_argument("--toy-sigmas", type=float, nargs="+",
                    default=(0.0, 0.05, 0.2, 0.5, 1.0),
                    help="toy seam-noise sweep (0 = noiseless)")
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
    return EvidenceLogger(args.output, node_role="attacker/seam-inversion")


def _randn_mat(rows, cols, rng, scale=1.0):
    return [[scale * rng.gauss(0.0, 1.0) for _ in range(cols)]
            for _ in range(rows)]


def _matmul(A, B):
    """A [n,k] @ B [k,m] -> [n,m] on lists of rows (toy sizes only)."""
    Bt = list(zip(*B))
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt]
            for row in A]


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
            raise ValueError("ridge normal equations are (near-)singular; "
                             "increase the ridge lambda")
        aug[col], aug[piv] = aug[piv], aug[col]
        inv = 1.0 / aug[col][col]
        aug[col] = [v * inv for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                f = aug[r][col]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def ridge_decode_map(A, lam):
    """White-box linearization inversion of the toy seam h = e @ A.

    Row-wise ridge: min_e ||e A - h||^2 + lam ||e||^2 has the closed form
    e_hat = h A^T (A A^T + lam I)^{-1}. A is [d][H] nested lists; returns
    D = A^T (A A^T + lam I)^{-1} as [H][d] lists so that e_hat = h @ D.
    """
    d, H = len(A), len(A[0])
    gram = [[sum(A[i][k] * A[j][k] for k in range(H))
             + (lam if i == j else 0.0) for j in range(d)]
            for i in range(d)]
    solved = _gauss_solve(gram, A)
    return [list(col) for col in zip(*solved)]


def cosine_nn_decode(rows, E):
    """Cosine nearest neighbor over embedding rows E [V][d]; rows [n][d].
    Zero-norm rows/embedding entries never win a cosine (skipped)."""
    enorms = [math.sqrt(sum(v * v for v in e)) for e in E]
    preds = []
    for row in rows:
        rn = math.sqrt(sum(v * v for v in row))
        best, best_id = -2.0, 0
        if rn >= 1e-12:
            for i, e in enumerate(E):
                if enorms[i] < 1e-12:
                    continue
                sim = sum(a * b for a, b in zip(row, e)) / (rn * enorms[i])
                if sim > best:
                    best, best_id = sim, i
        preds.append(best_id)
    return preds


def set_precision_recall(pred, true):
    """Orderless per-sequence token-set overlap (VFLAIR convention)."""
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


def majority_control_token(token_lists):
    """The corpus-mode token — the majority-control decode (mode of an
    attacker-side calibration corpus; canonical is never consulted)."""
    counts = collections.Counter(t for seq in token_lists for t in seq)
    if not counts:
        raise ValueError("empty control corpus")
    return counts.most_common(1)[0][0]


def score_sequences(preds, trues, control_tok, n_seeds=None):
    """The full per-cell metric bundle (both tiers share this): orderless
    set precision/recall, position-aligned token accuracy, majority-token
    control accuracy, and the frozen-gate verdict (excess over control,
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


def toy_world(seed, vocab=40, dim=8, n_seq=32, seq_len=16, sigma=0.0,
              cal_tokens=4000):
    """Tiny synthetic seam world. E [vocab, dim] random; A [dim, dim]
    random and KNOWN to the attacker (white-box toy analogue of the victim
    head transform); victim sequences are uniform token draws with seam
    rows h = E[tok] @ A + N(0, sigma^2) per element. The calibration corpus
    is a separate uniform draw — the attacker-side analogue of public text
    (it sets the majority control and never touches the victim rows)."""
    rng = random.Random(seed)
    E = _randn_mat(vocab, dim, rng)
    A = _randn_mat(dim, dim, rng)
    victim = [[rng.randrange(vocab) for _ in range(seq_len)]
              for _ in range(n_seq)]
    cal = [rng.randrange(vocab) for _ in range(cal_tokens)]
    rows = []
    for seq in victim:
        hs = []
        for tok in seq:
            e = E[tok]
            hs.append([sum(e[j] * A[j][k] for j in range(dim))
                       + (rng.gauss(0.0, sigma) if sigma > 0.0 else 0.0)
                       for k in range(dim)])
        rows.append(hs)
    return {"E": E, "A": A, "victim": victim, "h": rows, "cal": cal,
            "vocab": vocab, "dim": dim, "sigma": sigma}


def toy_cell(args, sigma, seed):
    """One toy cell: ridge-invert the KNOWN map and NN-decode, then score
    with the shared plumbing. This is exactly the linearization arm with an
    oracle map — the optimizer loop itself is torch-tier."""
    world = toy_world(args.seed + 791 * seed, vocab=args.vocab,
                      dim=args.hidden, n_seq=args.toy_seqs,
                      seq_len=args.seq_len, sigma=sigma,
                      cal_tokens=args.toy_cal_tokens)
    D = ridge_decode_map(world["A"], TOY_RIDGE_LAMBDA)
    preds = [cosine_nn_decode(_matmul(hs, D), world["E"])
             for hs in world["h"]]
    control = majority_control_token([world["cal"]])
    return {"sigma": sigma, "seed": seed,
            **score_sequences(preds, world["victim"], control,
                              n_seeds=len(args.seeds))}


def _run_toy(args, ev):
    out = artifacts.make_artifact(
        SCHEMA,
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "vocab": args.vocab, "dim": args.hidden, "seq_len": args.seq_len,
         "toy_seqs": args.toy_seqs, "toy_cal_tokens": args.toy_cal_tokens,
         "toy_sigmas": list(args.toy_sigmas), "seeds": args.seeds,
         "ridge_lambda": TOY_RIDGE_LAMBDA},
        THREAT_MODEL + " Toy tier: synthetic KNOWN-map seam (A granted); "
        "measures decode/scoring/gate plumbing and noise robustness of the "
        "ridge inversion, not a real model.",
        interpretation=INTERPRETATION,
        evidence=ev)
    cells = []
    for sigma in args.toy_sigmas:
        for seed in args.seeds:
            cell = {"experiment": EXPERIMENT_ID, "arm": "toy-ridge",
                    "submode": "known-map-linearization"}
            try:
                cell.update(toy_cell(args, sigma, seed))
            except Exception as e:
                journal_error(args.output, EXPERIMENT_ID, cell, e)
                continue
            artifacts.append_jsonl(args.output, cell)
            if ev is not None:
                ev.event("cell_result", cell)
            out["results"].append(cell)
            cells.append(cell)
            print(f"[seam-toy] sigma={sigma} seed={seed}: "
                  f"acc={cell['seq_accuracy']} control="
                  f"{cell['control_accuracy']} setP="
                  f"{cell['token_set_precision']} "
                  f"gate={cell['gate']['verdict']}")
    for sigma in sorted(set(c["sigma"] for c in cells)):
        vals = [c for c in cells if c["sigma"] == sigma]
        out["summary"].append({
            "experiment": EXPERIMENT_ID, "sigma": sigma,
            "seq_accuracy_mean": sum(c["seq_accuracy"] for c in vals)
            / len(vals),
            "control_accuracy_mean": sum(c["control_accuracy"] for c in vals)
            / len(vals),
            "gate_verdicts": [c["gate"]["verdict"] for c in vals],
            "n_cells": len(vals)})
    artifacts.write_artifact(args.output, out, evidence=ev)
    return 0


def toy_cell_adaptive(args, sigma, seed, variant):
    """One adaptive-toy cell: the toy world's seam rows carry gaussian
    noise N(0, sigma^2) (the deployed defense). plain decodes with the
    historical map D = A^T (A A^T + lam I)^{-1} (lam = TOY_RIDGE_LAMBDA —
    noise-naive). adaptive decodes with the NOISE-MARGINALIZED (Wiener)
    map D = A^T (A A^T + (lam + sigma^2) I)^{-1}: for the linear-Gaussian
    world (E rows ~ N(0, I), wire noise ~ N(0, sigma^2 I)) this is the
    minimum-MSE linear estimator of e given h = eA + n, and the exact
    closed form the real tier's --noise-samples augmented ridge fit
    converges to as K * N_cal -> infinity (the augmentation makes the fit
    minimize E_n[||(H + N) W - E||^2] = ||H W - E||^2 + sigma^2 N ||W||^2,
    i.e. ridge with lambda_eff = sigma^2 * N_cal)."""
    if variant not in ("plain", "adaptive"):
        raise ValueError(f"unknown adaptive-toy variant {variant!r}")
    world = toy_world(args.seed + 791 * seed, vocab=args.vocab,
                      dim=args.hidden, n_seq=args.toy_seqs,
                      seq_len=args.seq_len, sigma=sigma,
                      cal_tokens=args.toy_cal_tokens)
    lam = TOY_RIDGE_LAMBDA + (sigma ** 2 if variant == "adaptive" else 0.0)
    D = ridge_decode_map(world["A"], lam)
    preds = [cosine_nn_decode(_matmul(hs, D), world["E"])
             for hs in world["h"]]
    control = majority_control_token([world["cal"]])
    return {"sigma": sigma, "seed": seed, "variant": variant,
            "ridge_lambda_effective": lam,
            **score_sequences(preds, world["victim"], control,
                              n_seeds=len(args.seeds))}


def _run_toy_adaptive(args, ev):
    """Toy adaptive sweep: plain vs noise-marginalized cells at matched
    sigma (full gate fields per variant), summary margin per sigma."""
    out = artifacts.make_artifact(
        SCHEMA,
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "adaptive": args.adaptive, "noise_samples": args.noise_samples,
         "defense_sigma": args.defense_sigma,
         "vocab": args.vocab, "dim": args.hidden, "seq_len": args.seq_len,
         "toy_seqs": args.toy_seqs, "toy_cal_tokens": args.toy_cal_tokens,
         "toy_sigmas": list(args.toy_sigmas), "seeds": args.seeds,
         "ridge_lambda": TOY_RIDGE_LAMBDA},
        THREAT_MODEL + " Toy ADAPTIVE tier: synthetic KNOWN-map seam with "
        "gaussian seam noise (the deployed defense); the plain decode is "
        "the noise-naive ridge, the adaptive decode is the "
        "noise-marginalized (Wiener) map with lambda_eff = lambda + "
        "sigma^2 — the closed form of the real tier's noise-augmented "
        "calibration fit.",
        interpretation=INTERPRETATION + " Adaptive tier: per (sigma, seed) "
                       "one cell per variant (plain / adaptive); the "
                       "summary's adaptive_margin_pp is the strict "
                       "denoising gain of modeling the public defense "
                       "inside the inversion.",
        evidence=ev)
    cells = []
    for sigma in args.toy_sigmas:
        for seed in args.seeds:
            for variant in ("plain", "adaptive"):
                cell = {"experiment": EXPERIMENT_ID, "arm": "toy-ridge",
                        "submode": f"adaptive-{variant}"}
                try:
                    cell.update(toy_cell_adaptive(args, sigma, seed,
                                                  variant))
                except Exception as e:
                    journal_error(args.output, EXPERIMENT_ID, cell, e)
                    continue
                artifacts.append_jsonl(args.output, cell)
                if ev is not None:
                    ev.event("cell_result", cell)
                out["results"].append(cell)
                cells.append(cell)
                print(f"[seam-toy-adaptive] sigma={sigma} seed={seed} "
                      f"{variant}: acc={cell['seq_accuracy']} control="
                      f"{cell['control_accuracy']} "
                      f"gate={cell['gate']['verdict']}")
    for sigma in sorted(set(c["sigma"] for c in cells)):
        vals = [c for c in cells if c["sigma"] == sigma]
        by_variant = {v: [c for c in vals if c["variant"] == v]
                      for v in ("plain", "adaptive")}
        means = {v: (sum(c["seq_accuracy"] for c in cs) / len(cs)
                     if cs else None)
                 for v, cs in by_variant.items()}
        margin = (100.0 * (means["adaptive"] - means["plain"])
                  if None not in means.values() else None)
        out["summary"].append({
            "experiment": EXPERIMENT_ID, "sigma": sigma,
            "plain_seq_accuracy_mean": means["plain"],
            "adaptive_seq_accuracy_mean": means["adaptive"],
            "adaptive_margin_pp": margin,
            "control_accuracy_mean":
                sum(c["control_accuracy"] for c in vals) / len(vals),
            "gate_verdicts": [f"{c['variant']}:{c['gate']['verdict']}"
                              for c in vals],
            "n_cells": len(vals)})
    artifacts.write_artifact(args.output, out, evidence=ev)
    return 0


def toy_joint_world(seed, grid=4, dim=16, n_seq=32, seq_len=16,
                    cal_tokens=4000):
    """The k x k grid world. V = k^2 tokens; each token v = (row, col) has
    embedding e_v = concat(r_row, c_col) in R^{2m} (m = dim//2) with random
    per-row / per-column direction vectors. The fwd seam map A = [I_m; 0]
    projects onto the row block (rank m < 2m; null(A) = the column block);
    the bwd seam map C = [0; I_m] projects onto the column block (rank m;
    null(C) = the row block). null(A) intersect null(C) = {0}: the JOINT
    observation (h, g) = (r_row, c_col) identifies the cell — and hence
    the token — while each single observation identifies only its row
    (resp. column) class of k tokens. Noiseless: the tier measures
    identifiability, not noise robustness (the noise story is the
    adaptive tier's)."""
    if dim % 2:
        raise ValueError(f"toy joint world needs an even dim, got {dim}")
    rng = random.Random(seed)
    m = dim // 2
    vocab = grid * grid
    rows = _randn_mat(grid, m, rng)
    cols = _randn_mat(grid, m, rng)
    E = [list(rows[v // grid]) + list(cols[v % grid])
         for v in range(vocab)]
    victim = [[rng.randrange(vocab) for _ in range(seq_len)]
              for _ in range(n_seq)]
    cal = [rng.randrange(vocab) for _ in range(cal_tokens)]
    return {"E": E, "vocab": vocab, "dim": dim, "m": m, "grid": grid,
            "victim": victim, "cal": cal}


def toy_joint_cell(args, variant, seed):
    """One joint-toy cell. variant 'fwd-only' decodes the row projection,
    'bwd-only' the column projection, 'joint' the concatenation (the
    jointly-identifying observation). NN ties inside a row/column class
    break deterministically (cosine_nn_decode's strict-> scan), so each
    single sits at its information-theoretic floor 1/grid; the joint
    decode recovers."""
    if variant not in ("fwd-only", "bwd-only", "joint"):
        raise ValueError(f"unknown joint-toy variant {variant!r}")
    world = toy_joint_world(args.seed + 431 * seed, grid=args.joint_grid,
                            dim=2 * args.hidden, n_seq=args.toy_seqs,
                            seq_len=args.seq_len,
                            cal_tokens=args.toy_cal_tokens)
    E, m = world["E"], world["m"]
    fwd_E = [e[:m] for e in E]
    bwd_E = [e[m:] for e in E]
    preds = []
    for seq in world["victim"]:
        h_rows = [E[t][:m] for t in seq]
        g_rows = [E[t][m:] for t in seq]
        if variant == "fwd-only":
            preds.append(cosine_nn_decode(h_rows, fwd_E))
        elif variant == "bwd-only":
            preds.append(cosine_nn_decode(g_rows, bwd_E))
        else:
            preds.append(cosine_nn_decode([E[t] for t in seq], E))
    control = majority_control_token([world["cal"]])
    return {"seed": seed, "variant": variant, "grid": world["grid"],
            **score_sequences(preds, world["victim"], control,
                              n_seeds=len(args.seeds))}


def _run_toy_joint(args, ev):
    """Toy joint sweep: fwd-only / bwd-only / joint cells (full gate
    fields), summary margins of the joint over each single."""
    out = artifacts.make_artifact(
        SCHEMA,
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "arm": "joint", "joint_grid": args.joint_grid,
         "dim": 2 * args.hidden, "seq_len": args.seq_len,
         "toy_seqs": args.toy_seqs, "toy_cal_tokens": args.toy_cal_tokens,
         "seeds": args.seeds},
        THREAT_MODEL + " Toy JOINT tier: the complementary-rank grid "
        "world — the fwd seam observation is rank-deficient along the "
        "column block, the bwd observation along the row block "
        "(complementary null spaces, trivial intersection); only the "
        "joint observation identifies the token.",
        interpretation=INTERPRETATION + " Joint tier: singles decode at "
                       "their single-observation ambiguity floor (1/grid "
                       "= the row/column class size), the joint decode "
                       "recovers — the honesty centerpiece: high recovery "
                       "only from using BOTH captures.",
        evidence=ev)
    cells = []
    for seed in args.seeds:
        for variant in ("fwd-only", "bwd-only", "joint"):
            cell = {"experiment": EXPERIMENT_ID, "arm": "joint",
                    "submode": f"toy-{variant}"}
            try:
                cell.update(toy_joint_cell(args, variant, seed))
            except Exception as e:
                journal_error(args.output, EXPERIMENT_ID, cell, e)
                continue
            artifacts.append_jsonl(args.output, cell)
            if ev is not None:
                ev.event("cell_result", cell)
            out["results"].append(cell)
            cells.append(cell)
            print(f"[seam-toy-joint] seed={seed} {variant}: "
                  f"acc={cell['seq_accuracy']} control="
                  f"{cell['control_accuracy']} "
                  f"gate={cell['gate']['verdict']}")
    if cells:
        means = {v: [c["seq_accuracy"] for c in cells
                     if c["variant"] == v]
                 for v in ("fwd-only", "bwd-only", "joint")}
        means = {v: sum(xs) / len(xs) for v, xs in means.items() if xs}
        out["summary"].append({
            "experiment": EXPERIMENT_ID, "arm": "joint",
            "joint_seq_accuracy_mean": means.get("joint"),
            "fwd_only_seq_accuracy_mean": means.get("fwd-only"),
            "bwd_only_seq_accuracy_mean": means.get("bwd-only"),
            "joint_minus_fwd_pp": (100.0 * (means["joint"]
                                            - means["fwd-only"])
                                   if "joint" in means
                                   and "fwd-only" in means else None),
            "joint_minus_bwd_pp": (100.0 * (means["joint"]
                                            - means["bwd-only"])
                                   if "joint" in means
                                   and "bwd-only" in means else None),
            "control_accuracy_mean":
                sum(c["control_accuracy"] for c in cells) / len(cells),
            "gate_verdicts": [f"{c['variant']}:{c['gate']['verdict']}"
                              for c in cells],
            "n_cells": len(cells)})
    artifacts.write_artifact(args.output, out, evidence=ev)
    return 0


def _embedding_of_model(model):
    return model.get_input_embeddings().weight.detach()


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


def save_head_pt(path, model, boundary, arch="gpt2"):
    """Export the seam-head state in the head-pt v1 format (what
    splitbench will emit natively later): the arch spec + enough config to
    rebuild the stack + the state of wte/wpe/blocks[0:boundary). Loading
    reconstructs a full model with only the head slice's weights restored
    (the rest stays random and is never run by the inversion)."""
    cfg = model.config
    keep = ("n_layer", "n_embd", "n_head", "n_positions", "vocab_size",
            "bos_token_id", "eos_token_id")
    state = {}
    prefixes = ("transformer.wte.", "transformer.wpe.")
    block_prefixes = tuple(f"transformer.h.{i}." for i in range(boundary))
    for k, v in model.state_dict().items():
        if k.startswith(prefixes + block_prefixes):
            state[k] = v.detach().cpu()
    torch.save({"format": HEAD_PT_FORMAT, "arch": arch,
                "boundary": int(boundary),
                "config": {k: getattr(cfg, k) for k in keep},
                "state_dict": state}, path)
    return path


def _load_head_pt(path, device):
    """Load a head-pt blob -> (split, E). v1 supports arch 'gpt2' via the
    splitbench Gpt2Split interface."""
    from ..splitbench import SplitPlan, build_split
    blob = torch.load(path, map_location="cpu")
    if not isinstance(blob, dict) or blob.get("format") != HEAD_PT_FORMAT:
        raise SystemExit(f"[{EXPERIMENT_ID}] {path} is not a "
                         f"{HEAD_PT_FORMAT} head export (see save_head_pt)")
    if blob.get("arch") != "gpt2":
        raise SystemExit(f"[{EXPERIMENT_ID}] head-pt arch "
                         f"{blob.get('arch')!r} unsupported in v1 "
                         "(registered: gpt2)")
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError:
        raise SystemExit(f"[{EXPERIMENT_ID}] --head-pt requires "
                         "transformers (pip install transformers)") from None
    cfg = GPT2Config(**blob["config"])
    model = GPT2LMHeadModel(cfg)
    model.load_state_dict(blob["state_dict"], strict=False)
    model = model.to(device).eval()
    split = build_split(model, SplitPlan.parse(str(blob["boundary"]),
                                               n_layers=cfg.n_layer))
    return split, _embedding_of_model(model)


def _load_embedding_pt(path):
    """E from a .pt: a raw [V, d] tensor, a state dict carrying a wte /
    embed_tokens weight, or a head-pt blob."""
    blob = torch.load(path, map_location="cpu")
    if torch.is_tensor(blob):
        if blob.dim() == 2:
            return blob.float()
        raise SystemExit(f"[{EXPERIMENT_ID}] {path}: expected a 2-D "
                         f"[V, d] embedding tensor, got shape "
                         f"{tuple(blob.shape)}")
    if isinstance(blob, dict):
        if blob.get("format") == HEAD_PT_FORMAT:
            blob = blob["state_dict"]
        for key in ("embedding", "wte.weight", "transformer.wte.weight",
                    "embed_tokens.weight",
                    "model.embed_tokens.weight"):
            if key in blob and torch.is_tensor(blob[key]):
                return blob[key].float()
        for k, v in blob.items():
            if torch.is_tensor(v) and v.dim() == 2 \
                    and ("wte" in k or "embed_tokens" in k):
                return v.float()
    raise SystemExit(f"[{EXPERIMENT_ID}] {path}: cannot locate an "
                     "embedding matrix (expected a [V, d] tensor or a "
                     "state dict with wte.weight / embed_tokens.weight)")


def _head_embed_forward(split, Z):
    """The victim head transform on a dummy EMBEDDING sequence Z
    [n, seq, d] (Gpt2Split interface; head-pt v1 is the GPT-2 family):
    drop(Z + wpe(pos)) -> blocks[0:boundary)."""
    pos = torch.arange(Z.shape[1], device=Z.device).unsqueeze(0)
    hidden = split.drop(Z + split.wpe(pos))
    return split._run_blocks(hidden, split.ranges[0], None)


def _calibration_ids(args, vocab, seq_len, device):
    """Public calibration sequences [n_cal, seq_len]: --calibration-file
    public text through the --model tokenizer, else synthetic uniform draws
    (no public text assumed)."""
    if getattr(args, "calibration_pt", None):
        if args.calibration_file:
            raise ValueError("calibration-pt and calibration-file are mutually exclusive")
        ids = torch.load(args.calibration_pt, map_location="cpu", weights_only=True)
        if ids.ndim != 2 or ids.shape[1] != seq_len or ids.numel() == 0 or ids.min() < 0 or ids.max() >= vocab:
            raise ValueError("invalid public calibration token tensor")
        return ids[:args.calibration_seqs].long().to(device)
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
            ids = tok(f.read(), return_tensors="pt").input_ids[0]
        n_win = ids.numel() // seq_len
        if n_win < 1:
            raise SystemExit(f"[{EXPERIMENT_ID}] calibration file too "
                             f"small: {ids.numel()} tokens < seq_len "
                             f"{seq_len}")
        windows = ids[:n_win * seq_len].view(n_win, seq_len)
        return windows[:args.calibration_seqs].to(device)
    g = torch.Generator().manual_seed(args.seed + 4242)
    return torch.randint(0, vocab, (args.calibration_seqs, seq_len),
                         generator=g).to(device)


def _head_rows(split, ids, batch=16):
    """Batched public-head forwards -> seam rows [n, seq, H] (no grad)."""
    rows = []
    with torch.no_grad():
        for i in range(0, ids.shape[0], batch):
            rows.append(split.head_forward(ids[i:i + batch]))
    return torch.cat(rows)


def _ridge_fit(X, Y, lam):
    """Ridge map W [d_in, d_out] minimizing ||X W - Y||^2 + lam ||W||^2.
    lam <= 0 selects the scale-aware auto value (1e-3 * mean Gram
    diagonal). Returns (W, lam_used)."""
    d_in = X.shape[1]
    gram = X.T @ X
    if lam <= 0.0:
        lam = 1e-3 * (torch.diagonal(gram).mean().item())
    eye = torch.eye(d_in, dtype=X.dtype, device=X.device)
    W = torch.linalg.solve(gram + lam * eye, X.T @ Y)
    return W, lam


def _cosine_nn_torch(rows, E):
    """Batched cosine-NN decode: rows [n, d], E [V, d] -> LongTensor [n]."""
    rn = torch.nn.functional.normalize(rows.float(), dim=-1)
    en = torch.nn.functional.normalize(E.float(), dim=-1)
    return (rn @ en.T).argmax(1)


def _antithetic_noise(shape, sigma, samples, seed, device):
    """K antithetic noise draws N(0, sigma^2) for the noise-whitening
    adaptive objective: K/2 base draws plus their negations (variance
    reduction; K=1 -> a single draw). Drawn on CPU from a dedicated
    generator (device-agnostic) and moved. Returns [K, *shape]."""
    k = max(1, int(samples))
    n_base = (k + 1) // 2
    gen = torch.Generator().manual_seed(seed)
    base = torch.randn(n_base, *shape, generator=gen) * sigma
    return torch.cat([base, -base])[:k].to(device)


def _noise_augmented_rows(h_cal, e_cal, sigma, samples, seed):
    """Noise-augmented ridge calibration for --adaptive noise-whitening:
    K antithetic copies of the clean calibration rows with fresh
    N(0, sigma^2) draws at the PUBLIC defense sigma, paired with the same
    embedding targets. The fit then minimizes E_n[||(h + n) W - e||^2] —
    the Wiener / noise-marginalized decode map (module docstring)."""
    nz = _antithetic_noise(tuple(h_cal.shape), sigma, samples, seed,
                           h_cal.device)
    k = nz.shape[0]
    h_aug = (h_cal.unsqueeze(0) + nz).reshape(-1, h_cal.shape[-1])
    e_aug = e_cal.repeat(k, 1)
    return h_aug, e_aug


def _optimize_vmi_head(split, target, E, args, warm_ids, checkpointer=None,
                       adaptive=None):
    """vmi sub-mode (a): optimize dummy embeddings Z [n, seq, d] through
    the KNOWN head, min ||Head(Z + pos) - h*||^2. Warm start from the
    linearization decode when available. Returns (pred [n, seq], loss).

    adaptive= (sigma, K): the noise-whitening objective — per step, K
    antithetic defense-noise draws n_k and the marginalized loss
    (1/K) sum_k ||Head(Z) - (h_wire - n_k)||^2 (Tramer doctrine: the
    gaussian defense sigma is public). Candid note (module docstring):
    for a single captured h_wire this has the SAME population minimizer
    as the plain objective; the gain mechanism is trajectory
    regularization, so parity with plain is the honest expectation. The
    draws reseed per ABSOLUTE step index, so a checkpoint-resumed loop
    replays the identical noise stream — resume-deterministic.

    With checkpointer= (attacker.checkpoint.Checkpointer) the loop state
    (Z + Adam moments) is checkpointed every checkpointer.every steps; a
    latest() hit restores that state and continues the loop after the
    checkpointed step — kill-and-restart safe. The cadence itself (step 0
    always included) is the Checkpointer's; an already-finished loop
    (checkpointed step >= args.steps - 1) just decodes the restored Z."""
    n, seq, _ = target.shape
    start = 0
    Z = opt = None
    if checkpointer is not None:
        found = checkpointer.latest()
        if found is not None:
            done, payload = found
            Z = payload["Z"].to(target.device).detach().clone()
            Z.requires_grad_(True)
            opt = torch.optim.Adam([Z], lr=args.lr)
            opt.load_state_dict(payload["opt"])
            start = done + 1
    if Z is None:
        if warm_ids is not None:
            Z = E[warm_ids].detach().clone()
        else:
            Z = 0.1 * torch.randn(n, seq, E.shape[1],
                                  device=target.device)
        Z.requires_grad_(True)
        opt = torch.optim.Adam([Z], lr=args.lr)
    loss = None
    for step in range(start, args.steps):
        opt.zero_grad()
        h_hat = _head_embed_forward(split, Z)
        if adaptive is not None:
            sigma, samples = adaptive
            nz = _antithetic_noise(tuple(target.shape), sigma, samples,
                                   args.seed + 104729 * step,
                                   target.device)
            loss = (h_hat.unsqueeze(0) - (target.unsqueeze(0) - nz)) \
                .pow(2).mean()
        else:
            loss = torch.nn.functional.mse_loss(h_hat, target)
        loss.backward()
        opt.step()
        if checkpointer is not None:
            checkpointer.save(step, {"step": step, "Z": Z.detach(),
                                     "opt": opt.state_dict()})
    with torch.no_grad():
        pred = _cosine_nn_torch(Z.detach().reshape(-1, E.shape[1]),
                                E).reshape(n, seq)
    return pred, None if loss is None else float(loss.item())


def _optimize_wbi(split, fwd_map, target, E, args, warm_ids, adaptive=None):
    """wbi: optimize relaxed one-hots Z [n, seq, V]; the relaxed embedding
    softmax(Z/T) @ E goes through the KNOWN head (split) when given, else
    through the forward ridge map e -> h. Warm start puts logit mass on
    the linearization decode. Returns (pred [n, seq], loss).

    adaptive= (sigma, K): the same noise-whitening marginalized target as
    the vmi optimizer (antithetic draws at the public defense sigma; see
    _optimize_vmi_head's candid note — parity with plain is the honest
    expectation for a single capture)."""
    n, seq, _ = target.shape
    V = E.shape[0]
    Z = torch.zeros(n, seq, V, device=target.device)
    if warm_ids is not None:
        Z.scatter_(2, warm_ids.unsqueeze(-1), 2.0)
    Z.requires_grad_(True)
    opt = torch.optim.Adam([Z], lr=args.lr)
    loss = None
    for step in range(args.steps):
        opt.zero_grad()
        emb = torch.nn.functional.softmax(Z / args.temperature, dim=-1) @ E
        h_hat = _head_embed_forward(split, emb) if split is not None \
            else emb @ fwd_map
        if adaptive is not None:
            sigma, samples = adaptive
            nz = _antithetic_noise(tuple(target.shape), sigma, samples,
                                   args.seed + 130363 * step,
                                   target.device)
            loss = (h_hat.unsqueeze(0) - (target.unsqueeze(0) - nz)) \
                .pow(2).mean()
        else:
            loss = torch.nn.functional.mse_loss(h_hat, target)
        loss.backward()
        opt.step()
    pred = Z.detach().argmax(-1)
    return pred, float(loss.item())


def _load_scored_sequences(args):
    """Pick up to --samples scored sequences, row-aligned to canonical.
    Training mode reads phase='fwd' captures; inference mode scores
    phase='prefill' rows (decode rows are model-generated tokens, not
    victim-private data — documented). Returns (picked, seq_len, h_dim)."""
    from .. import captures
    if not args.canonical_pt:
        raise SystemExit(f"[{EXPERIMENT_ID}] pass --canonical-pt "
                         "(splitbench --save-canonical); canonical tokens "
                         "are evaluation truth for the scoring cells.")
    canonical = torch.load(args.canonical_pt, map_location="cpu").long()
    phase = "fwd" if args.mode == "training" else "prefill"
    records = [(m, p) for m, p in
               captures.scan_captures(args.capture_dir, args.mode)
               if m["phase"] == phase]
    if not records:
        raise SystemExit(f"[{EXPERIMENT_ID}] no phase={phase!r} captures "
                         f"in {args.capture_dir} (mode {args.mode})")
    picked = []
    cursor = 0
    for meta, pt in records:
        t = captures.load_tensor(pt)
        if t.dim() == 2:
            t = t.unsqueeze(0)
        b, s, h = t.shape
        rows = canonical[cursor:cursor + b]
        if rows.shape[0] != b:
            raise SystemExit(
                f"[{EXPERIMENT_ID}] canonical/capture misalignment: capture "
                f"at step {meta.get('step', meta.get('request_seq'))} has "
                f"batch {b} but canonical has {rows.shape[0]} rows left "
                f"at cursor {cursor} (canonical.pt is row-aligned to the "
                f"{phase!r} captures, step-major)")
        if rows.shape[1] != s:
            raise SystemExit(f"[{EXPERIMENT_ID}] canonical seq_len "
                             f"{rows.shape[1]} != capture seq_len {s}")
        for r in range(b):
            if len(picked) >= args.samples:
                break
            picked.append({"step": meta.get("step",
                                            meta.get("request_seq")),
                           "epoch": meta.get("epoch"),
                           "h": t[r], "true": rows[r].tolist()})
        cursor += b
        if len(picked) >= args.samples:
            break
    if not picked:
        raise SystemExit(f"[{EXPERIMENT_ID}] --samples {args.samples} "
                         "selected no sequences")
    return picked, picked[0]["h"].shape[0], picked[0]["h"].shape[-1]


def _load_scored_sequences_joint(args):
    """Joint arm loader: up to --samples scored sequences carrying BOTH
    the phase='fwd' seam activation h and the phase='bwd' seam gradient g
    of the same step, row-aligned to canonical (the splitbench convention:
    canonical.pt is row-aligned to the fwd captures, step-major; the bwd
    capture of a captured step always exists). Training mode only (the
    bwd channel). Returns (picked, seq_len, h_dim)."""
    from .. import captures
    if not args.canonical_pt:
        raise SystemExit(f"[{EXPERIMENT_ID}] pass --canonical-pt "
                         "(splitbench --save-canonical); canonical tokens "
                         "are evaluation truth for the scoring cells.")
    canonical = torch.load(args.canonical_pt, map_location="cpu").long()
    records = captures.scan_captures(args.capture_dir, "training")
    by_step = {}
    for meta, pt in records:
        by_step.setdefault(meta["step"], {})[meta["phase"]] = (meta, pt)
    if not any("bwd" in ph for ph in by_step.values()):
        raise SystemExit(f"[{EXPERIMENT_ID}] joint arm: no phase='bwd' "
                         f"seam captures in {args.capture_dir} (the joint "
                         "arm needs BOTH phases of the training seam)")
    if not all("bwd" in ph for ph in by_step.values()):
        raise SystemExit(f"[{EXPERIMENT_ID}] joint arm: a captured step "
                         "lacks its phase='bwd' capture (canonical.pt is "
                         "row-aligned to the fwd captures — every fwd "
                         "step must carry its bwd pair; splitbench "
                         "captures both every captured step)")
    picked = []
    cursor = 0
    for step in sorted(by_step):
        ph = by_step[step]
        if "bwd" not in ph:
            continue
        h = captures.load_tensor(ph["fwd"][1]).float()
        g = captures.load_tensor(ph["bwd"][1]).float()
        if h.dim() == 2:
            h = h.unsqueeze(0)
        if g.dim() == 2:
            g = g.unsqueeze(0)
        b, s, _ = h.shape
        rows = canonical[cursor:cursor + b]
        if rows.shape[0] != b or rows.shape[1] != s:
            raise SystemExit(
                f"[{EXPERIMENT_ID}] canonical/capture misalignment at step "
                f"{step}: fwd batch [{b}, {s}] vs canonical rows "
                f"{tuple(rows.shape)} at cursor {cursor} (canonical.pt is "
                "row-aligned to the fwd captures, step-major)")
        for r in range(b):
            if len(picked) >= args.samples:
                break
            picked.append({"step": step,
                           "epoch": ph["bwd"][0].get("epoch"),
                           "h": h[r], "g": g[r],
                           "batch": b,
                           "true": rows[r].tolist()})
        cursor += b
        if len(picked) >= args.samples:
            break
    if not picked:
        raise SystemExit(f"[{EXPERIMENT_ID}] --samples {args.samples} "
                         "selected no sequences")
    return picked, picked[0]["h"].shape[0], picked[0]["h"].shape[-1]


def _joint_beam_decode(split, rows_h, rows_g, E, W_dec, n_candidates,
                       beam_width):
    """The joint fwd+bwd decode (module docstring): per-position candidate
    sets from the bwd per-token direction signal (the gradient at row p-1
    points at the LABEL of position p — the tag convention), position 0
    from the fwd linearization decode (no label gradient there), then a
    beam search whose score is the FORWARD-CONSISTENCY residual
    sum_{j<=p} ||Head(prefix)_j - h*_j||^2. The public head is causal, so
    each extension's residual at position p is final for its prefix.
    Finite-width beam pruning remains approximate; it is not globally exact. rows_h/rows_g [n, seq, H]; returns
    LongTensor [n, seq]."""
    from .gradient_inversion import _tag_candidates_torch
    n, s, H = rows_h.shape
    d = E.shape[1]
    C = _tag_candidates_torch(rows_g, E)
    e_hat = (rows_h.reshape(-1, H) @ W_dec).reshape(n, s, d)
    fwd_c = torch.nn.functional.normalize(e_hat, dim=-1) \
        @ torch.nn.functional.normalize(E.float(), dim=-1).T
    k = min(n_candidates, E.shape[0])
    preds = []
    with torch.no_grad():
        for i in range(n):
            cand = [fwd_c[i, 0].topk(k).indices.tolist()]
            cand += [C[i, p - 1].topk(k).indices.tolist()
                     for p in range(1, s)]
            beam = [([], 0.0)]
            for p in range(s):
                exts = [(pref + [v], res) for pref, res in beam
                        for v in cand[p]]
                ids_batch = torch.tensor([x[0] for x in exts],
                                         dtype=torch.long,
                                         device=rows_h.device)
                hh = split.head_forward(ids_batch)[:, p, :]
                r = ((hh - rows_h[i, p].unsqueeze(0)) ** 2).sum(-1) \
                    .tolist()
                beam = sorted(((pref, r0 + r1) for (pref, r0), r1
                               in zip(exts, r)),
                              key=lambda t: t[1])[:beam_width]
            preds.append(beam[0][0])
    return torch.tensor(preds, dtype=torch.long)


def _run_real(args, ev):
    require_torch(EXPERIMENT_ID)
    from ..splitbench import SplitPlan, build_split
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.arm == "joint":
        if args.mode != "training":
            raise SystemExit(f"[{EXPERIMENT_ID}] the joint arm needs the "
                             "bwd seam gradient — training mode only")
        picked, seq_len, h_dim = _load_scored_sequences_joint(args)
    else:
        picked, seq_len, h_dim = _load_scored_sequences(args)

    split = None
    head_note = "none"
    if args.head_pt:
        split, E = _load_head_pt(args.head_pt, device)
        head_note = f"head-pt:{args.head_pt}"
    else:
        if args.embedding_pt:
            E = _load_embedding_pt(args.embedding_pt).to(device)
        elif args.model:
            model = _load_hf_model(args.model, device)
            E = _embedding_of_model(model)
            from ..splitbench import _n_layers_of
            split = build_split(model, SplitPlan.parse(
                args.split, n_layers=_n_layers_of(model)))
            head_note = f"public-head:{args.model}@{args.split}"
        else:
            raise SystemExit(
                f"[{EXPERIMENT_ID}] pass --head-pt (KNOWN victim head), "
                "--model (public base weights: E + public calibration "
                "head), or --embedding-pt (E only, weakest variant).")
    E = E.to(device)
    if args.arm == "wbi" and split is None:
        raise SystemExit(f"[{EXPERIMENT_ID}] wbi needs a forward model: "
                         "--head-pt (KNOWN head) or --model (public head "
                         "for the calibration-fitted forward ridge map)")
    if args.arm == "joint" and split is None:
        raise SystemExit(f"[{EXPERIMENT_ID}] the joint arm needs a head "
                         "for the forward-consistency rerank: --head-pt "
                         "(KNOWN head) or --model (public head)")

    use_head_opt = args.head_pt and not args.linearization
    if args.resume and not (args.arm == "vmi" and use_head_opt):
        print(f"[{EXPERIMENT_ID}] note: --resume checkpoints only the "
              "vmi-head optimizer loop; this sub-mode has no loop — "
              "flag ignored")
    W_dec = None
    fwd_map = None
    lam_used = None
    adaptive_active = args.adaptive == "noise-whitening" \
        and args.defense_sigma > 0.0
    if args.adaptive == "noise-whitening" and not adaptive_active:
        print(f"[{EXPERIMENT_ID}] note: --adaptive noise-whitening with "
              "--defense-sigma 0 models no deployed noise — degenerates "
              "to the plain arms")
    adaptive_obj = ((args.defense_sigma, args.noise_samples)
                    if adaptive_active else None)
    cal_ids = _calibration_ids(args, E.shape[0], seq_len, device)
    if split is not None:
        h_cal = _head_rows(split, cal_ids).reshape(-1, h_dim)
        e_cal = E[cal_ids.reshape(-1)]
        if adaptive_active:
            h_fit, e_fit = _noise_augmented_rows(
                h_cal, e_cal, args.defense_sigma, args.noise_samples,
                args.seed + 911)
        else:
            h_fit, e_fit = h_cal, e_cal
        W_dec, lam_used = _ridge_fit(h_fit, e_fit, args.ridge_lambda)
        if args.arm == "wbi" and not use_head_opt:
            fwd_map, _ = _ridge_fit(e_cal, h_cal, args.ridge_lambda)
    elif E.shape[1] != h_dim:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] identity decode needs dim(E) == seam dim "
            f"({E.shape[1]} != {h_dim}); pass --model or --head-pt for a "
            "calibration map")
    control = majority_control_token(cal_ids.tolist())

    submode = {"vmi": "vmi-head" if use_head_opt else
               ("vmi-linearization" if W_dec is not None
                else "vmi-identity-nn"),
               "wbi": "wbi-head" if use_head_opt else "wbi-map",
               "joint": "joint-beam"}[args.arm]
    if adaptive_active:
        submode += "+noise-whitening"
    rows = torch.stack([p["h"] for p in picked]).to(device)
    grows = None
    if args.arm == "joint":
        grows = torch.stack([p["g"] for p in picked]).to(device)
    warm_ids = None
    if W_dec is not None:
        warm_ids = _cosine_nn_torch(
            rows.reshape(-1, h_dim) @ W_dec, E).reshape(len(picked), seq_len)
    groups = {}
    order = []
    for i, p in enumerate(picked):
        key = (p["step"], p["epoch"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)

    out = artifacts.make_artifact(
        SCHEMA,
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": False,
         "arm": args.arm, "submode": submode,
         "capture_dir": args.capture_dir, "canonical_pt": args.canonical_pt,
         "model": args.model, "embedding_pt": args.embedding_pt,
         "head_pt": args.head_pt, "split": args.split,
         "linearization_forced": bool(args.linearization),
         "steps": args.steps, "samples": args.samples, "lr": args.lr,
         "temperature": args.temperature,
         "ridge_lambda": args.ridge_lambda, "ridge_lambda_used": lam_used,
         "calibration_seqs": args.calibration_seqs,
         "calibration_file": args.calibration_file,
         "calibration_pt": getattr(args, "calibration_pt", None), "device": device,
         "resume": args.resume,
         "checkpoint_every": args.checkpoint_every,
         "adaptive": args.adaptive, "noise_samples": args.noise_samples,
         "defense_sigma": args.defense_sigma,
         "joint_candidates": args.joint_candidates,
         "joint_beam": args.joint_beam,
         "seeds": list(args.seeds)},
        THREAT_MODEL, interpretation=INTERPRETATION, evidence=ev)

    cells = []
    for key in order:
        idx = groups[key]
        cell = {"experiment": EXPERIMENT_ID, "arm": args.arm,
                "submode": submode, "mode": args.mode,
                "step": key[0], "epoch": key[1], "surface": head_note}
        try:
            if args.arm == "joint":
                pred = _joint_beam_decode(
                    split, rows[idx], grows[idx], E, W_dec,
                    args.joint_candidates, args.joint_beam)
                final_loss = None
            elif args.arm == "vmi" and not use_head_opt:
                flat = rows[idx].reshape(-1, h_dim)
                ehat = flat @ W_dec if W_dec is not None else flat
                pred = _cosine_nn_torch(ehat, E).reshape(len(idx), seq_len)
                final_loss = None
            elif args.arm == "vmi":
                ckpt = None
                if args.resume:
                    from ..checkpoint import Checkpointer
                    stream = f"vmi-step{key[0]}" + (
                        f"-epoch{key[1]}" if key[1] is not None else "")
                    ckpt = Checkpointer(args.resume, stream,
                                        args.checkpoint_every)
                pred, final_loss = _optimize_vmi_head(
                    split, rows[idx], E, args,
                    warm_ids[idx] if warm_ids is not None else None,
                    checkpointer=ckpt, adaptive=adaptive_obj)
            else:
                pred, final_loss = _optimize_wbi(
                    split if use_head_opt else None, fwd_map, rows[idx], E,
                    args, warm_ids[idx] if warm_ids is not None else None,
                    adaptive=adaptive_obj)
            preds = pred.reshape(len(idx), seq_len).cpu().tolist()
            trues = [picked[i]["true"] for i in idx]
            cell.update(score_sequences(preds, trues, control,
                                        n_seeds=len(args.seeds)))
            cell["per_sequence"] = [{"index": i, "prediction": p, "truth": t,
                "n_tokens": len(t), "correct": sum(a == b for a, b in zip(p, t)),
                "control_correct": sum(control == b for b in t)}
                for i, p, t in zip(idx, preds, trues)]
            if final_loss is not None:
                cell["final_objective"] = round(final_loss, 8)
            if args.arm == "joint":
                from .gradient_inversion import (_tag_candidates_torch,
                                                 _tag_decode)
                fwd_flat = rows[idx].reshape(-1, h_dim)
                fwd_pred = _cosine_nn_torch(fwd_flat @ W_dec, E) \
                    .reshape(len(idx), seq_len).cpu().tolist()
                C = _tag_candidates_torch(grows[idx], E)
                bwd_preds = []
                for r_i in range(len(idx)):
                    cand = C[r_i, :-1, :].cpu().tolist()
                    bwd_preds.append([control]
                                     + _tag_decode(cand, "greedy"))
                fwd_acc, _, _ = seq_accuracy(fwd_pred, trues)
                bwd_acc, _, _ = seq_accuracy(bwd_preds, trues)
                cell["fwd_linearization_seq_accuracy"] = round(fwd_acc, 6)
                cell["bwd_direction_seq_accuracy"] = round(bwd_acc, 6)
                cell["joint_minus_fwd_pp"] = round(
                    100.0 * (cell["seq_accuracy"] - fwd_acc), 4)
                cell["joint_minus_bwd_pp"] = round(
                    100.0 * (cell["seq_accuracy"] - bwd_acc), 4)
        except Exception as e:
            journal_error(args.output, EXPERIMENT_ID, cell, e)
            out["results"].append({**cell, "error": f"{type(e).__name__}: {e}"})
            continue
        artifacts.append_jsonl(args.output, cell)
        if ev is not None:
            ev.event("cell_result", cell)
        out["results"].append(cell)
        cells.append(cell)
        msg = (f"[seam-inversion] arm={args.arm}/{submode} step={key[0]}: "
               f"acc={cell['seq_accuracy']} control="
               f"{cell['control_accuracy']} setP="
               f"{cell['token_set_precision']} "
               f"gate={cell['gate']['verdict']}")
        if args.arm == "joint":
            msg += (f" | fwd-only={cell['fwd_linearization_seq_accuracy']}"
                    f" bwd-only={cell['bwd_direction_seq_accuracy']}")
        print(msg)

    if cells:
        n_corr = sum(c["n_correct"] for c in cells)
        n_tok = sum(c["n_tokens"] for c in cells)
        c_corr = sum(c["n_control_correct"] for c in cells)
        pooled_acc = n_corr / max(1, n_tok)
        pooled_ctl = c_corr / max(1, n_tok)
        summary = {
            "experiment": EXPERIMENT_ID, "arm": args.arm,
            "submode": submode,
            "n_cells": len(cells), "n_tokens": n_tok,
            "token_set_precision_mean":
                sum(c["token_set_precision"] for c in cells) / len(cells),
            "token_set_recall_mean":
                sum(c["token_set_recall"] for c in cells) / len(cells),
            "seq_accuracy": round(pooled_acc, 6),
            "control_accuracy": round(pooled_ctl, 6),
            "gate": gate.verdict(pooled_acc, pooled_ctl, n_tok,
                                 n_seeds=len(args.seeds)),
            "gate_verdicts": [c["gate"]["verdict"] for c in cells]}
        if args.arm == "joint":
            summary["fwd_linearization_seq_accuracy_mean"] = \
                sum(c["fwd_linearization_seq_accuracy"] for c in cells) \
                / len(cells)
            summary["bwd_direction_seq_accuracy_mean"] = \
                sum(c["bwd_direction_seq_accuracy"] for c in cells) \
                / len(cells)
            summary["joint_minus_fwd_pp_mean"] = \
                sum(c["joint_minus_fwd_pp"] for c in cells) / len(cells)
            summary["joint_minus_bwd_pp_mean"] = \
                sum(c["joint_minus_bwd_pp"] for c in cells) / len(cells)
        out["summary"].append(summary)
    artifacts.write_artifact(args.output, out, evidence=ev)
    return 0


def run(args):
    ev = _make_evidence(args)
    if args.samples < 1:
        raise SystemExit(f"[{EXPERIMENT_ID}] --samples must be >= 1")
    if args.checkpoint_every < 1:
        raise SystemExit(f"[{EXPERIMENT_ID}] --checkpoint-every must be "
                         ">= 1")
    if args.noise_samples < 1:
        raise SystemExit(f"[{EXPERIMENT_ID}] --noise-samples must be >= 1")
    if args.joint_candidates < 1 or args.joint_beam < 1:
        raise SystemExit(f"[{EXPERIMENT_ID}] --joint-candidates and "
                         "--joint-beam must be >= 1")
    if args.toy:
        if args.quick:
            args.seeds = [0]
            args.toy_sigmas = (0.0, 0.5, 1.0)
            args.toy_seqs = min(args.toy_seqs, 16)
            args.toy_cal_tokens = min(args.toy_cal_tokens, 1000)
        if args.arm == "joint":
            return _run_toy_joint(args, ev)
        if args.adaptive != "none":
            return _run_toy_adaptive(args, ev)
        return _run_toy(args, ev)
    if not args.capture_dir:
        raise SystemExit(f"[{EXPERIMENT_ID}] pass --toy, or --capture-dir "
                         "D --canonical-pt C plus an attack surface "
                         "(--model / --embedding-pt / --head-pt).")
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

    print("seam-inversion toy core (pure python):")

    X = _gauss_solve([[2.0, 1.0], [1.0, 3.0]], [[1.0], [2.0]])
    check("Gauss-Jordan solve matches the hand solution",
          abs(X[0][0] - 0.2) < 1e-12 and abs(X[1][0] - 0.6) < 1e-12)
    try:
        _gauss_solve([[1.0, 2.0], [2.0, 4.0]], [[1.0], [2.0]])
        check("singular system raises (journaled, never a crash)", False)
    except ValueError as e:
        check("singular system raises (journaled, never a crash)",
              "singular" in str(e))

    rng = random.Random(7)
    A = _randn_mat(4, 4, rng)
    D = ridge_decode_map(A, 1e-9)
    e = [0.5, -1.0, 2.0, 0.25]
    e_hat = _matmul([_matmul([e], A)[0]], D)[0]
    check("ridge decode map inverts a known noiseless map",
          max(abs(a - b) for a, b in zip(e, e_hat)) < 1e-6)

    check("cosine NN decode picks the matching row, zero-norm safe",
          cosine_nn_decode([[0.9, 0.1], [0.0, 0.0]],
                           [[1.0, 0.0], [0.0, 1.0]]) == [0, 0])
    p, r = set_precision_recall([1, 2, 3], [2, 3, 4])
    check("orderless set precision/recall fixture",
          abs(p - 2 / 3) < 1e-12 and abs(r - 2 / 3) < 1e-12)
    acc, nc, nt = seq_accuracy([[1, 2], [3, 4]], [[1, 9], [3, 4]])
    check("position-aligned pooled accuracy fixture",
          nc == 3 and nt == 4 and acc == 0.75)
    check("majority control token is the corpus mode",
          majority_control_token([[5, 5, 7], [5, 7]]) == 5)

    args = build_parser().parse_args(["--toy"])
    args.mode = "training"
    cell0 = toy_cell(args, 0.0, 0)
    check("noiseless toy recovers >90% of tokens",
          cell0["seq_accuracy"] > 0.90)
    check("toy cell carries control + gate fields",
          "control_accuracy" in cell0 and cell0["gate"]["verdict"]
          in ("PASS", "FAIL"))
    noisy = toy_cell(args, 1.0, 0)
    check("heavy noise degrades recovery (graceful)",
          noisy["seq_accuracy"] < cell0["seq_accuracy"])

    check("parser builds torch-less",
          build_parser().parse_args([]).arm == "vmi")
    ns = build_parser().parse_args([])
    ns.mode = "training"
    try:
        run(ns)
        check("no --toy/--capture-dir is a descriptive error", False)
    except SystemExit as e:
        check("no --toy/--capture-dir is a descriptive error",
              "--capture-dir" in str(e))
    if torch is None:
        ns2 = build_parser().parse_args(["--capture-dir", "/nonexistent",
                                         "--canonical-pt", "/nonexistent"])
        ns2.mode = "training"
        try:
            run(ns2)
            check("real tier is torch-guarded", False)
        except SystemExit as e:
            check("real tier is torch-guarded", "requires torch" in str(e))

    with tempfile.TemporaryDirectory() as d:
        ns = build_parser().parse_args(
            ["--toy", "--quick", "--evidence",
             "--output", os.path.join(d, "si.json")])
        ns.mode = "training"
        check("toy run end to end torch-free", run(ns) == 0)
        art = json.load(open(os.path.join(d, "si.json")))
        check("toy artifact has sigma cells with gate verdicts",
              len(art["results"]) == 3
              and all(c["gate"]["verdict"] in ("PASS", "FAIL")
                      for c in art["results"]))
        check("journal lines match result cells",
              len(open(os.path.join(d, "si.json.jsonl")).read()
                  .splitlines()) == len(art["results"]))
        from ..evidence import verify_evidence
        good, problems = verify_evidence(
            os.path.join(d, "si.json.evidence.json"))
        check("evidence chain verifies", good and problems == [])

    print("seam-inversion adaptive + joint toy cores (pure python):")
    check("parser carries the adaptive/joint flags torch-less",
          build_parser().parse_args([]).adaptive == "none"
          and build_parser().parse_args([]).noise_samples == 8
          and build_parser().parse_args([]).defense_sigma == 0.0
          and build_parser().parse_args([]).joint_candidates == 8
          and build_parser().parse_args([]).joint_beam == 4
          and build_parser().parse_args(["--arm", "joint"]).arm == "joint"
          and build_parser().parse_args(
              ["--adaptive", "noise-whitening"]).adaptive
          == "noise-whitening")
    args_a = build_parser().parse_args(["--toy", "--seeds", "0", "1"])
    args_a.mode = "training"
    for sigma, min_margin in ((0.5, 0.02), (1.0, 0.10)):
        plains, adapts = [], []
        for seed in (0, 1):
            plains.append(toy_cell_adaptive(args_a, sigma, seed, "plain")
                          ["seq_accuracy"])
            adapts.append(toy_cell_adaptive(args_a, sigma, seed,
                                            "adaptive")["seq_accuracy"])
        plain_m = sum(plains) / 2
        adapt_m = sum(adapts) / 2
        check(f"adaptive toy sigma={sigma}: adaptive > plain by a margin "
              f"({plain_m:.3f} -> {adapt_m:.3f})",
              adapt_m >= plain_m + min_margin)
    cell_a = toy_cell_adaptive(args_a, 1.0, 0, "adaptive")
    check("adaptive toy cell carries the full gate fields + variant",
          cell_a["variant"] == "adaptive"
          and "control_accuracy" in cell_a
          and cell_a["gate"]["verdict"] in ("PASS", "FAIL")
          and cell_a["ridge_lambda_effective"]
          == TOY_RIDGE_LAMBDA + 1.0)

    args_j = build_parser().parse_args(["--toy", "--arm", "joint",
                                        "--seeds", "0", "1"])
    args_j.mode = "training"
    wj = toy_joint_world(0, grid=4, dim=16, n_seq=4, seq_len=6)
    check("joint toy world is deterministic and shaped",
          len(wj["E"]) == 16 and len(wj["E"][0]) == 16
          and toy_joint_world(0, grid=4, dim=16, n_seq=4,
                              seq_len=6)["victim"] == wj["victim"])
    means = {}
    for variant in ("fwd-only", "bwd-only", "joint"):
        means[variant] = sum(toy_joint_cell(args_j, variant, s)
                             ["seq_accuracy"] for s in (0, 1)) / 2
    check("joint toy: singles sit at the 1/grid floor, joint recovers",
          means["joint"] > 0.90
          and means["fwd-only"] < 0.6 and means["bwd-only"] < 0.6
          and means["joint"] > means["fwd-only"] + 0.4
          and means["joint"] > means["bwd-only"] + 0.4)
    with tempfile.TemporaryDirectory() as d:
        ns = build_parser().parse_args(
            ["--toy", "--adaptive", "noise-whitening", "--quick",
             "--evidence", "--output", os.path.join(d, "sia.json")])
        ns.mode = "training"
        check("adaptive toy run end to end torch-free", run(ns) == 0)
        art = json.load(open(os.path.join(d, "sia.json")))
        check("adaptive toy artifact has paired variant cells + margins",
              len(art["results"]) == 6
              and all("variant" in c and "gate" in c
                      for c in art["results"])
              and all("adaptive_margin_pp" in s for s in art["summary"]))
        good, problems = verify_evidence(
            os.path.join(d, "sia.json.evidence.json"))
        check("adaptive evidence chain verifies", good and problems == [])
        ns = build_parser().parse_args(
            ["--toy", "--arm", "joint", "--quick",
             "--output", os.path.join(d, "sij.json")])
        ns.mode = "training"
        check("joint toy run end to end torch-free", run(ns) == 0)
        art = json.load(open(os.path.join(d, "sij.json")))
        check("joint toy artifact has 3 variants x 1 seed + margin summary",
              len(art["results"]) == 3
              and art["summary"][0]["joint_minus_fwd_pp"] > 0
              and art["summary"][0]["joint_minus_bwd_pp"] > 0)
        ns2 = build_parser().parse_args(["--arm", "joint"])
        ns2.mode = "training"
        try:
            run(ns2)
            check("joint real tier without --capture-dir is descriptive",
                  False)
        except SystemExit as e:
            check("joint real tier without --capture-dir is descriptive",
                  "--capture-dir" in str(e))
        ns2 = build_parser().parse_args(["--arm", "joint",
                                         "--capture-dir", "/nonexistent",
                                         "--canonical-pt", "/nonexistent"])
        ns2.mode = "inference"
        if torch is not None:
            try:
                run(ns2)
                check("joint arm refuses inference mode", False)
            except SystemExit as e:
                check("joint arm refuses inference mode",
                      "training mode only" in str(e))

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
