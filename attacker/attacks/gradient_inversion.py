#!/usr/bin/env python3
"""Citation scope (reference audit): `tag` is a local cosine-gradient-direction
decoder, not an implementation of TAG or LAMP's optimisation algorithms.
The sequence prior is an analogue of image-prior regularisation, not a
GradInversion replication; Geiping et al. (2020) and Yin et al. (2021) are
distinct works. See docs/reference-audit-2026-09-22.md.

gradient-inversion (training mode) — DLG++ boundary inversion, plus the
iDLG and sequence-prior arms (Phase C literature breadth).

Superset of split-training/gradient_inversion.py (weak DLG) and
split-training/dlgpp.py (DLG++: multi-restart, cosine LR, L2+cosine
matching on boundary activation AND gradient, token-level refinement,
optional LM seq-prior).

Arms (--arm, default dlgpp — the historical behavior, unchanged):
  dlgpp      the existing optimization attacker (below).
  idlg       analytic label recovery BEFORE the optimization loop (iDLG,
             Zhao et al. 2020): for a cross-entropy head the last-layer
             bias-gradient row is p - onehot(y), so argmin_c db[c] = y by
             the sign argument. The toy world is a REAL random linear
             classifier (embeddings -> tall boundary map -> classification
             head; the bias-gradient row is computed from the model's
             actual forward math, not a lookup), so the exactness is the
             published theorem made concrete, not a rigged fixture. The
             cell reports label accuracy and the downstream reconstruction
             improvement over the DLG++ optimization arm run on the SAME
             world (the boundary map is tall/underdetermined, so
             activation matching alone does not pin the tokens).
  prior-arm  Sequence-prior regularized inversion (Geiping et al.
             2020 add a total-variation image prior to the fidelity
             objective). The candidates here are 1-D embedding SEQUENCES,
             not images; the documented analogue of TV is the group-lasso
             smoothness prior R(z) = sum_i ||z_{i+1} - z_i||_2 (isotropic
             TV over adjacent embedding rows; --prior tv is the
             anisotropic l1 version). The toy world for this arm draws
             locally coherent victim sequences (a geometric random walk
             over the embedding lattice), so the smoothness prior is a
             TRUE prior of the data distribution — the PSNR/L2 delta with
             and without the prior is then an honest measurement, not a
             foregone conclusion.
  tag        local per-token direction decoder; TAG/LAMP are related work. Three modern pieces replace the DLG-era free
             L2 optimization: (a) a COSINE-similarity gradient-match
             objective (Geiping et al. 2020) realized per-token — the
             candidate score is the cosine between the captured seam
             gradient direction and each vocab token's direction, not an
             L2 distance; (b) attention-aware decomposition — at the seam
             the per-position gradient contributions are approximately
             separable, and each position's NEGATIVE gradient direction
             points at that position's next-token LABEL embedding (the
             tied unembedding lm_head == wte makes this exact: for CE,
             dL/d(hidden)_i = (p_i - onehot(y_i)) @ E, so -dL/d(hidden)_i
             = sum_v (1{v=y_i} - p_iv) E[v] is dominated by E[y_i]). The
             attack scores every vocab token's direction E[v] against the
             captured gradient per position (the "gradient carries
             per-token direction" argument); (c) a light combinatorial
             assignment over the candidate matrix (greedy per-position
             argmax, or Hungarian one-to-one) INSTEAD of free
             optimization. Like BLI, position 0 is nobody's label and is
             not scored: tag recovers tokens[1:]. Torch tier: splitbench
             bwd captures (--capture-dir/--canonical-pt/--model/--split,
             public head grants E); a dlgpp-style activation-match
             optimization baseline runs at the same budget for the
             improvement measurement. Toy tier torch-free: a linear world
             with an ORTHOGONAL seam map + tied unembedding where the
             per-token direction argument is exact; planted recovery
             >90%, null at control.
  all        idlg + prior-arm on the same seeds (unchanged; tag is a
             separate arm with its own toy + real tiers).

Paths:
  --toy          synthetic world. The dlgpp arm's toy loop needs torch
                 (historical); the idlg / prior-arm toy cores are PURE
                 PYTHON so the new arms are machinery-checked torch-free.
  --dlgpp "--model M --corpus-file C ..."
                 delegation: runs split-training/dlgpp.py as a subprocess
                 with the given args (full model-driven attack, untouched
                 script) and normalizes its JSON into the framework
                 artifact. This is how real DLG++ runs are driven.
  real idlg/prior-arm capture path: integration point — feed per-row head
                 gradients; currently a descriptive error (below).

Usage:
    python -m attacker --mode training --attack gradient-inversion --help
    python -m attacker --mode training --attack gradient-inversion --toy \
        --quick --output /tmp/dlg.json
    python -m attacker --mode training --attack gradient-inversion --toy \
        --arm idlg --output /tmp/idlg.json          # torch-free
    python -m attacker --mode training --attack gradient-inversion --toy \
        --arm prior-arm --output /tmp/prior.json    # torch-free
    python -m attacker --mode training --attack gradient-inversion \
        --dlgpp "--model /models/qwen3-0.6b --toy --quick" --output d.json
"""

import argparse
import json
import math
import os
import random
import shlex
import subprocess
import sys

try:
    import torch
except ImportError:
    torch = None

from .. import artifacts, gate
from .common import add_common_args, journal_error, require_torch

EXPERIMENT_ID = "dlgpp_gradient_inversion"
MODES = ("training",)
REQUIRES_LABELS = False
DESCRIPTION = ("DLG++ optimization attacker on the training boundary "
               "(h*, g* = dL/dh) — multi-restart, L2+cosine match, "
               "nearest-embedding refinement; arms: iDLG analytic label "
               "recovery, sequence-prior regularization, and "
               "the local transformer gradient-direction arm (cosine per-token "
               "gradient-direction scoring + light assignment, torch tier "
               "on splitbench bwd captures)")

ARMS = ("dlgpp", "idlg", "prior-arm", "tag", "all")
PRIORS = ("none", "tv", "group-lasso")
ASSIGNMENTS = ("greedy", "hungarian")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--dlgpp", default=None, metavar='"ARGS"',
                    help="delegate to split-training/dlgpp.py with these "
                         "args (quoted string); real model-driven runs")
    ap.add_argument("--arm", choices=ARMS, default="dlgpp",
                    help="attack arm (default dlgpp = the historical "
                         "behavior); idlg/prior-arm toy cores are "
                         "torch-free")
    ap.add_argument("--seq-len", type=int, default=16,
                    help="toy victim sequence length")
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=300,
                    help="optimization rounds per restart")
    ap.add_argument("--prior", choices=PRIORS, default="none",
                    help="candidate regularizer: none (default), tv "
                         "(anisotropic, sum |z_{i+1}-z_i|), group-lasso "
                         "(isotropic, sum ||z_{i+1}-z_i||_2) — the "
                         "sequence analogue of GradInversion's image TV")
    ap.add_argument("--prior-lambda", type=float, default=0.3,
                    help="regularizer weight for --prior")
    ap.add_argument("--smooth", type=float, default=6.0,
                    help="prior-arm toy world: random-walk temperature; "
                         "P(x_{i+1}=c) ∝ exp(-||e_c - e_xi||^2/smooth^2)")
    ap.add_argument("--vocab", type=int, default=64, help="toy vocab size")
    ap.add_argument("--assignment", choices=ASSIGNMENTS, default="greedy",
                    help="tag decode over the per-position candidate "
                         "matrix: greedy per-position argmax (default, "
                         "valid when tokens repeat) or hungarian one-to-one "
                         "assignment (exact when a sequence's labels are "
                         "distinct; documented to underperform greedy on "
                         "repeating text)")
    ap.add_argument("--capture-dir", default=None,
                    help="tag real tier: wire-capture directory "
                         "(wire-training schema; reads phase='bwd' seam "
                         "gradients and phase='fwd' seam activations for "
                         "the dlgpp-style optimization baseline)")
    ap.add_argument("--canonical-pt", default=None,
                    help="canonical.pt ground-truth input_ids (splitbench "
                         "--save-canonical); evaluation truth, never an "
                         "attack input")
    ap.add_argument("--model", default=None,
                    help="tag real tier: HF name/path of the PUBLIC base "
                         "model — grants the token embedding matrix E and "
                         "the public head (the tied unembedding places the "
                         "label direction in E's space; the head runs the "
                         "dlgpp-style baseline's calibration forwards)")
    ap.add_argument("--split", default="6",
                    help="seam boundary of the PUBLIC head (must match the "
                         "capture harness split; default '6')")
    ap.add_argument("--samples", type=int, default=8,
                    help="tag real tier: cap on scored sequences (scan "
                         "order)")
    ap.add_argument("--calibration-seqs", type=int, default=64,
                    help="public calibration sequences for the majority "
                         "control (synthetic uniform draws unless "
                         "--calibration-file)")
    ap.add_argument("--calibration-file", default=None,
                    help="public text tokenized with the --model tokenizer "
                         "for the control corpus (default: synthetic "
                         "uniform draws — no public text assumed)")
    ap.add_argument("--lr", type=float, default=0.05,
                    help="Adam lr for the dlgpp-style optimization baseline "
                         "(tag itself runs no optimization loop)")
    ap.add_argument("--toy-seqs", type=int, default=8,
                    help="tag toy: victim sequences per cell")
    ap.add_argument("--tag-temp", type=float, default=8.0,
                    help="tag toy: body softmax temperature — high values "
                         "give the near-uniform confidence of a random-init "
                         "body (the step-0 capture regime), where the "
                         "per-token direction argument is exact")
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
                          node_role="attacker/gradient-inversion")


def toy_dlg(args, seed):
    """Synthetic DLG on a random linear boundary. Returns dict of
    metrics. --prior adds the sequence-TV analogue regularizer to the
    optimization (none = the historical objective, unchanged)."""
    g = torch.Generator().manual_seed(args.seed + seed)
    H, V, T = args.hidden, 64, args.seq_len
    embed = torch.randn(V, H, generator=g)
    head = torch.randn(H, H, generator=g) / (H ** 0.5)
    true_ids = torch.randint(0, V, (T,), generator=g)
    h_star = (embed[true_ids] @ head).detach()
    target = torch.randn(T, H, generator=g)
    g_star = 2.0 * (h_star - target)

    def prior_term(z):
        if args.prior == "tv":
            return (z[1:] - z[:-1]).abs().sum()
        if args.prior == "group-lasso":
            return (z[1:] - z[:-1]).norm(dim=1).sum()
        return None

    def objs(z):
        h_hat = z @ head
        g_hat = 2.0 * (h_hat - target)
        l_h = ((h_hat - h_star) ** 2).sum()
        l_g = ((g_hat - g_star) ** 2).sum()
        loss = l_h + l_g
        reg = prior_term(z)
        if reg is not None:
            loss = loss + args.prior_lambda * reg
        return loss

    best_z, best_l = None, None
    for r in range(args.restarts):
        z = torch.randn(T, H, generator=torch.Generator().manual_seed(
            args.seed + 900 * seed + r), requires_grad=True)
        opt = torch.optim.Adam([z], lr=0.05)
        for step in range(args.rounds):
            opt.zero_grad()
            loss = objs(z)
            loss.backward()
            opt.step()
        if best_l is None or loss.item() < best_l:
            best_l, best_z = loss.item(), z.detach()
    d = torch.cdist(best_z.float(), embed.float())
    rec_ids = d.argmin(1)
    acc = (rec_ids == true_ids).float().mean().item()
    return {"token_recovery": round(100.0 * acc, 4),
            "random_baseline": round(100.0 / V, 4),
            "final_objective": round(best_l, 6),
            "prior": args.prior,
            "prior_lambda": args.prior_lambda if args.prior != "none"
            else 0.0,
            "restarts": args.restarts, "rounds": args.rounds}


def _randn_mat(rows, cols, rng, scale=1.0):
    return [[scale * rng.gauss(0.0, 1.0) for _ in range(cols)]
            for _ in range(rows)]


def _matmul(A, B):
    """A [n,k] @ B [k,m] -> [n,m] on lists of rows (toy sizes only)."""
    Bt = list(zip(*B))
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt]
            for row in A]


def _softmax(row):
    m = max(row)
    exps = [math.exp(v - m) for v in row]
    s = sum(exps)
    return [e / s for e in exps]


def _sq_dist(u, v):
    return sum((a - b) ** 2 for a, b in zip(u, v))


def _weighted_choice(weights, rng):
    total = sum(weights)
    pick = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if pick <= acc:
            return i
    return len(weights) - 1


def toy_classification_world(seed, hidden, vocab, seq_len, smooth=0.0):
    """The shared idlg/prior-arm toy world (docstring at the section
    header). smooth > 0 draws x as a geometric random walk over the
    embedding lattice — locally coherent sequences, so the group-lasso
    smoothness prior is a true prior of the data distribution."""
    rng = random.Random(seed)
    E, B, V, T = hidden, max(2, hidden // 2), vocab, seq_len
    embed = _randn_mat(V, E, rng)
    head = _randn_mat(E, B, rng, scale=1.0 / math.sqrt(E))
    weights_out = _randn_mat(B, V, rng, scale=1.0 / math.sqrt(B))
    bias = [0.1 * rng.gauss(0.0, 1.0) for _ in range(V)]

    x = [rng.randrange(V)]
    for _ in range(T - 1):
        if smooth > 0.0:
            w = [math.exp(-_sq_dist(embed[c], embed[x[-1]])
                          / (smooth * smooth)) for c in range(V)]
            x.append(_weighted_choice(w, rng))
        else:
            x.append(rng.randrange(V))
    y = list(x)

    h_star = _matmul([list(embed[xi]) for xi in x], head)
    logits = [[row[j] + bias[j] for j in range(V)]
              for row in _matmul(h_star, weights_out)]
    probs = [_softmax(row) for row in logits]
    db = [[probs[i][c] - (1.0 if c == y[i] else 0.0) for c in range(V)]
          for i in range(T)]
    g_star = _matmul(db, [list(col) for col in zip(*weights_out)])
    return {"hidden": E, "boundary": B, "vocab": V, "seq_len": T,
            "smooth": smooth, "embed": embed, "head": head,
            "weights_out": weights_out, "bias": bias, "x": x, "y": y,
            "h_star": h_star, "g_star": g_star, "db": db}


def idlg_labels(world):
    """iDLG (Zhao et al. 2020): the label of row i is argmin_c db_i[c].
    Exact for a cross-entropy head by the sign argument — the toy measures
    the machinery and the downstream value, not a novel claim."""
    return [min(range(len(row)), key=lambda c: row[c])
            for row in world["db"]]


def _prior_value(z, prior):
    """R(z): tv = sum |Δz| (anisotropic), group-lasso = sum ||Δz||_2
    (isotropic sequence TV — the GradInversion image-TV analogue)."""
    if prior == "tv":
        return sum(abs(z[i + 1][j] - z[i][j])
                   for i in range(len(z) - 1) for j in range(len(z[0])))
    if prior == "group-lasso":
        return sum(math.sqrt(_sq_dist(z[i + 1], z[i]))
                   for i in range(len(z) - 1))
    return 0.0


def _add_prior_subgradient(grad, z, prior, lam):
    """In-place grad += lam * ∂R(z) (subgradient; zero subgradient at
    Δz = 0 for tv, unit-direction pull otherwise)."""
    T, E = len(z), len(z[0])
    for i in range(T):
        for j in range(E):
            d_prev = z[i][j] - (z[i - 1][j] if i > 0 else 0.0)
            d_next = (z[i + 1][j] if i < T - 1 else 0.0) - z[i][j]
            if prior == "tv":
                s = ((d_prev > 0) - (d_prev < 0)) if i > 0 else 0.0
                s -= ((d_next > 0) - (d_next < 0)) if i < T - 1 else 0.0
            else:
                s = 0.0
                if i > 0:
                    n = math.sqrt(_sq_dist(z[i], z[i - 1]))
                    if n > 1e-12:
                        s += d_prev / n
                if i < T - 1:
                    n = math.sqrt(_sq_dist(z[i + 1], z[i]))
                    if n > 1e-12:
                        s -= d_next / n
            grad[i][j] += lam * s


def _toy_optimize(world, rounds, lr, restarts, seed, prior="none",
                  prior_lambda=0.0):
    """Minimize sum_i ||z_i @ head - h*_i||^2 + prior_lambda * R(z) over
    z [T x E] with a small pure-python Adam; returns the best final z.
    Fidelity-only is the DLG++ toy arm on this world; the underdetermined
    (tall) head leaves an (E-B)-dim solution manifold, which is exactly
    what the prior / the iDLG labels resolve."""
    head, h_star = world["head"], world["h_star"]
    T, E, B = world["seq_len"], world["hidden"], world["boundary"]
    head_t = [list(col) for col in zip(*head)]
    best = None
    for r in range(restarts):
        rng = random.Random(1000 * seed + r + 7)
        z = _randn_mat(T, E, rng)
        m = [[0.0] * E for _ in range(T)]
        v = [[0.0] * E for _ in range(T)]
        for step in range(1, rounds + 1):
            diff = _matmul(z, head)
            for i in range(T):
                for j in range(B):
                    diff[i][j] -= h_star[i][j]
            grad = _matmul(diff, head_t)
            for i in range(T):
                for j in range(E):
                    grad[i][j] *= 2.0
            if prior != "none" and prior_lambda > 0.0:
                _add_prior_subgradient(grad, z, prior, prior_lambda)
            b1t = 1.0 - 0.9 ** step
            b2t = 1.0 - 0.999 ** step
            for i in range(T):
                for j in range(E):
                    m[i][j] = 0.9 * m[i][j] + 0.1 * grad[i][j]
                    v[i][j] = 0.999 * v[i][j] + 0.001 * grad[i][j] ** 2
                    z[i][j] -= (lr * (m[i][j] / b1t)
                                / (math.sqrt(v[i][j] / b2t) + 1e-8))
        fid = sum(_sq_dist(a, b) for a, b in zip(_matmul(z, head), h_star))
        obj = fid + prior_lambda * _prior_value(z, prior)
        if best is None or obj < best[0]:
            best = (obj, z)
    return best[1]


def _snap(world, z):
    """Nearest-embedding snap (the toy's token-level refinement)."""
    embed = world["embed"]
    return [min(range(world["vocab"]),
                key=lambda c: _sq_dist(row, embed[c])) for row in z]


def _rmse_psnr(z, z_ref):
    """RMSE and PSNR (dB) of z against z_ref; PSNR peak is the reference
    range, capped at 99 dB for a perfect reconstruction (json-safe)."""
    T, E = len(z_ref), len(z_ref[0])
    mse = sum((z[i][j] - z_ref[i][j]) ** 2
              for i in range(T) for j in range(E)) / (T * E)
    rmse = math.sqrt(mse)
    peak = max(max(r) for r in z_ref) - min(min(r) for r in z_ref)
    psnr = 99.0 if rmse <= 1e-12 else 20.0 * math.log10(peak / rmse)
    return rmse, psnr


def _pct(rec_ids, true_ids):
    return round(100.0 * sum(a == b for a, b in zip(rec_ids, true_ids))
                 / max(1, len(true_ids)), 4)


def toy_idlg(args, seed):
    """iDLG arm cell: analytic labels, then the DLG++-arm optimization on
    the SAME world as the baseline; report label accuracy and the
    downstream reconstruction improvement."""
    world = toy_classification_world(args.seed + seed, args.hidden,
                                     args.vocab, args.seq_len, smooth=0.0)
    labels = idlg_labels(world)
    label_acc = _pct(labels, world["y"])
    z_ref = [list(world["embed"][xi]) for xi in world["x"]]

    z_guided = [list(world["embed"][c]) for c in labels]
    guided_acc = _pct(_snap(world, z_guided), world["x"])
    guided_rmse, guided_psnr = _rmse_psnr(z_guided, z_ref)

    z_opt = _toy_optimize(world, args.rounds, 0.05, args.restarts, seed)
    opt_acc = _pct(_snap(world, z_opt), world["x"])
    opt_rmse, opt_psnr = _rmse_psnr(z_opt, z_ref)

    return {"label_accuracy": label_acc,
            "guided_token_recovery": guided_acc,
            "dlgpp_arm_token_recovery": opt_acc,
            "improvement_pp": round(guided_acc - opt_acc, 4),
            "guided_rmse": round(guided_rmse, 6),
            "dlgpp_arm_rmse": round(opt_rmse, 6),
            "guided_psnr_db": round(guided_psnr, 4),
            "dlgpp_arm_psnr_db": round(opt_psnr, 4),
            "random_baseline": round(100.0 / world["vocab"], 4),
            "restarts": args.restarts, "rounds": args.rounds}


def toy_prior(args, seed, prior):
    """prior-arm cell: GradInversion-style regularized inversion on a
    locally coherent sequence world; report RMSE/PSNR + token recovery."""
    world = toy_classification_world(args.seed + seed, args.hidden,
                                     args.vocab, args.seq_len,
                                     smooth=args.smooth)
    z_ref = [list(world["embed"][xi]) for xi in world["x"]]
    z = _toy_optimize(world, args.rounds, 0.05, args.restarts, seed,
                      prior=prior, prior_lambda=args.prior_lambda)
    rmse, psnr = _rmse_psnr(z, z_ref)
    return {"prior": prior,
            "prior_lambda": args.prior_lambda if prior != "none" else 0.0,
            "token_recovery": _pct(_snap(world, z), world["x"]),
            "rmse": round(rmse, 6), "psnr_db": round(psnr, 4),
            "smooth": args.smooth,
            "restarts": args.restarts, "rounds": args.rounds}


def _orthogonal_mat(n, rng):
    """A random orthogonal [n, n] matrix (Gram–Schmidt over Gaussian
    rows). The toy's public seam map: orthogonality makes backprop an
    isometry, so the per-token direction argument is EXACT in the toy."""
    rows = [[rng.gauss(0.0, 1.0) for _ in range(n)] for _ in range(n)]
    basis = []
    for row in rows:
        v = list(row)
        for q in basis:
            d = sum(a * b for a, b in zip(v, q))
            v = [a - d * b for a, b in zip(v, q)]
        nrm = math.sqrt(sum(a * a for a in v))
        basis.append([a / nrm for a in v])
    return [list(col) for col in zip(*basis)]


def _cosine(u, v):
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(a * a for a in v))
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return sum(a * b for a, b in zip(u, v)) / (nu * nv)


def _hungarian_assign(cost):
    """One-to-one assignment minimizing total cost (the Hungarian
    algorithm, O(n^2 m), n_rows <= n_cols). cost[i][j]; returns
    assign[i] = column assigned to row i. Used by the tag 'hungarian'
    decode: exact when a sequence's labels are distinct, documented to
    underperform greedy when labels repeat (LM text repeats)."""
    n, m = len(cost), len(cost[0])
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    assign = [-1] * n
    for j in range(1, m + 1):
        if p[j] > 0:
            assign[p[j] - 1] = j - 1
    return assign


def _tag_decode(candidates, assignment):
    """Decode the per-position candidate matrix C[i][v] (cosine scores)
    into one token per position. greedy: per-position argmax (default,
    valid when tokens repeat). hungarian: one-to-one assignment (exact
    for distinct labels)."""
    if assignment == "hungarian":
        cost = [[-c for c in row] for row in candidates]
        return _hungarian_assign(cost)
    return [max(range(len(row)), key=lambda v: row[v]) for row in candidates]


def _set_precision_recall(pred, true):
    ps, ts = set(pred), set(true)
    inter = len(ps & ts)
    return inter / max(1, len(ps)), inter / max(1, len(ts))


def _seq_accuracy(preds, trues):
    n_correct = sum(1 for p, t in zip(preds, trues) for a, b in zip(p, t)
                    if a == b)
    n_tokens = sum(len(t) for t in trues)
    return n_correct / max(1, n_tokens), n_correct, n_tokens


def _majority_control_token(token_lists):
    import collections
    counts = collections.Counter(t for seq in token_lists for t in seq)
    if not counts:
        raise ValueError("empty control corpus")
    return counts.most_common(1)[0][0]


def _score_tag(preds, trues, control_tok, n_seeds=None):
    """The per-cell metric bundle (both tag tiers share this): orderless
    token-set precision/recall, position-aligned accuracy, majority-token
    control accuracy, and the frozen-gate verdict (excess over control,
    n = scored tokens)."""
    prs = [_set_precision_recall(p, t) for p, t in zip(preds, trues)]
    acc, n_correct, n_tokens = _seq_accuracy(preds, trues)
    c_acc, c_correct, _ = _seq_accuracy([[control_tok] * len(t)
                                         for t in trues], trues)
    return {"n_seq": len(trues), "n_tokens": n_tokens,
            "token_set_precision": round(sum(p for p, _ in prs)
                                         / max(1, len(prs)), 6),
            "token_set_recall": round(sum(r for _, r in prs)
                                      / max(1, len(prs)), 6),
            "seq_accuracy": round(acc, 6), "n_correct": n_correct,
            "control_token": control_tok,
            "control_accuracy": round(c_acc, 6),
            "n_control_correct": c_correct,
            "gate": gate.verdict(acc, c_acc, n_tokens, n_seeds=n_seeds)}


def toy_tag_world(seed, vocab=64, dim=32, seq_len=8, temp=8.0, n_seq=8):
    """The tag toy world — a REAL linear model where the per-token
    direction argument is exact.

      E [V, d] raw Gaussian embeddings (mirrors wte); A [d, d] orthogonal
      public seam map (backprop isometry); U = E @ A the TIED unembedding
      at the seam (mirrors lm_head == wte); seam acts h_i = U[ids_i]; the
      body is the linear head logits_i = U h_i / temp with next-token CE
      (labels y_i = ids_{i+1}). The captured seam gradient is the world's
      true CE gradient g*_i = U^T (p_i - onehot(y_i)) = sum_v (p_iv -
      1{v=y_i}) U[v], so -g*_i is a weighted sum of per-token directions
      U[v] with the label U[y_i] carrying the unique positive weight —
      the "gradient carries per-token direction" argument, exact here
      because A is an isometry. High temp = the near-uniform confidence
      of a random-init body (the step-0 capture regime). n_seq victim
      sequences share the one model (E, A, U)."""
    rng = random.Random(seed)
    V, d, T = vocab, dim, seq_len
    E = _randn_mat(V, d, rng)
    A = _orthogonal_mat(d, rng)
    U = _matmul(E, A)
    seqs = []
    for _ in range(n_seq):
        ids = [rng.randrange(V) for _ in range(T)]
        labels = ids[1:]
        g = []
        for i in range(T - 1):
            logits = [sum(Uv * hv for Uv, hv in zip(U[v], U[ids[i]])) / temp
                      for v in range(V)]
            p = _softmax(logits)
            g.append([sum((p[v] - (1.0 if v == labels[i] else 0.0)) * U[v][k]
                          for v in range(V)) for k in range(d)])
        seqs.append({"ids": ids, "labels": labels, "g": g})
    return {"vocab": V, "dim": d, "seq_len": T, "temp": temp, "E": E,
            "A": A, "U": U, "seqs": seqs}


def tag_candidates(world, g):
    """C[i][v] = cos(-g*_i, U[v]) — the cosine objective per token."""
    U = world["U"]
    return [[_cosine([-x for x in gi], U[v]) for v in range(world["vocab"])]
            for gi in g]


def _tag_null_grads(world, g, rng):
    """The null observation: random gradients at the captured gradient's
    matched per-entry scale (no label information — the negative control)."""
    flat = [x for gi in g for x in gi]
    scale = math.sqrt(sum(x * x for x in flat) / max(1, len(flat)))
    return [[rng.gauss(0.0, scale) for _ in range(world["dim"])]
            for _ in g]


def toy_tag_cell(args, source, seed):
    """One tag toy cell: decode the candidate matrix (greedy/hungarian)
    and score with the shared plumbing. source 'planted' uses the world's
    true captured gradients; 'null' uses scale-matched random gradients."""
    world = toy_tag_world(args.seed + 557 * seed, vocab=args.vocab,
                          dim=args.hidden, seq_len=args.seq_len,
                          temp=args.tag_temp, n_seq=args.toy_seqs)
    rng = random.Random(args.seed + 9137 + seed)
    preds, trues = [], []
    for s in world["seqs"]:
        g = s["g"] if source == "planted" else _tag_null_grads(world, s["g"],
                                                               rng)
        preds.append(_tag_decode(tag_candidates(world, g), args.assignment))
        trues.append(s["labels"])
    cal_rng = random.Random(args.seed + 41249 + seed)
    cal = [cal_rng.randrange(args.vocab) for _ in range(4000)]
    control = _majority_control_token([cal])
    return {"source": source, "seed": seed,
            "assignment": args.assignment,
            **_score_tag(preds, trues, control, n_seeds=len(args.seeds))}


def _run_tag_toy(args, ev):
    out = artifacts.make_artifact(
        "dtraining.attacker.dlgpp.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "arm": "tag", "assignment": args.assignment, "hidden": args.hidden,
         "vocab": args.vocab, "seq_len": args.seq_len,
         "toy_seqs": args.toy_seqs, "tag_temp": args.tag_temp,
         "seeds": args.seeds},
        "semi-honest cloud observes the per-position seam gradient "
        "g* = dL/dh at the split depth and is granted the public token "
        "embedding matrix E (tied unembedding). tag arm: cosine per-token "
        "gradient-direction scoring (the label's embedding dominates the "
        "negative gradient direction) + a light assignment (greedy / "
        "Hungarian) instead of free optimization; recovers tokens[1:] "
        "(position 0 is nobody's label). Toy tier: linear world with an "
        "orthogonal seam map where the direction argument is exact.",
        interpretation="toy tier is a machinery check on a REAL linear "
                       "model (pure python, torch-free): the planted cell "
                       "recovers >90% by the exact per-token direction "
                       "argument; the null cell (scale-matched random "
                       "gradients) must sit at the majority control. Real "
                       "numbers come from --capture-dir runs.",
        evidence=ev)
    cells = []
    for source in ("planted", "null"):
        for seed in args.seeds:
            cell = {"experiment": EXPERIMENT_ID, "arm": "tag",
                    "submode": f"toy-{args.assignment}"}
            try:
                cell.update(toy_tag_cell(args, source, seed))
            except Exception as e:
                journal_error(args.output, EXPERIMENT_ID, cell, e)
                continue
            artifacts.append_jsonl(args.output, cell)
            if ev is not None:
                ev.event("cell_result", cell)
            out["results"].append(cell)
            cells.append(cell)
            print(f"[tag-toy] {source} seed={seed} ({args.assignment}): "
                  f"acc={cell['seq_accuracy']} control="
                  f"{cell['control_accuracy']} gate={cell['gate']['verdict']}")
    for source in ("planted", "null"):
        vals = [c for c in cells if c["source"] == source]
        if vals:
            out["summary"].append({
                "experiment": EXPERIMENT_ID, "arm": "tag", "source": source,
                "seq_accuracy_mean": sum(c["seq_accuracy"] for c in vals)
                / len(vals),
                "control_accuracy_mean": sum(c["control_accuracy"]
                                             for c in vals) / len(vals),
                "gate_verdicts": [c["gate"]["verdict"] for c in vals],
                "n_cells": len(vals)})
    artifacts.write_artifact(args.output, out, evidence=ev)
    return 0


def _tag_load_hf(name_or_path, device):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise SystemExit(f"[{EXPERIMENT_ID}] --model requires transformers "
                         "(pip install transformers)") from None
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path, dtype=torch.float32, attn_implementation="eager")
    return model.to(device).eval()


def _tag_load_sequences(args):
    """Load up to --samples scored sequences: the phase='bwd' seam
    gradients (the tag observation) row-aligned to canonical, plus the
    matching phase='fwd' seam activations for the dlgpp-style baseline.
    Returns (picked, seq_len, h_dim)."""
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
        raise SystemExit(f"[{EXPERIMENT_ID}] no phase='bwd' seam captures "
                         f"in {args.capture_dir} (the seam gradient is the "
                         "tag observation; splitbench captures it every "
                         "step)")
    picked = []
    cursor = 0
    for step in sorted(by_step):
        ph = by_step[step]
        if "bwd" not in ph:
            continue
        g = captures.load_tensor(ph["bwd"][1]).float()
        h = captures.load_tensor(ph["fwd"][1]).float() if "fwd" in ph \
            else None
        if g.dim() == 2:
            g = g.unsqueeze(0)
        if h is not None and h.dim() == 2:
            h = h.unsqueeze(0)
        b, s, _ = g.shape
        rows = canonical[cursor:cursor + b]
        if rows.shape[0] != b or rows.shape[1] != s:
            raise SystemExit(
                f"[{EXPERIMENT_ID}] canonical/capture misalignment at step "
                f"{step}: bwd batch [{b}, {s}] vs canonical rows "
                f"{tuple(rows.shape)} at cursor {cursor} (canonical.pt is "
                "row-aligned to the captures, step-major)")
        for r in range(b):
            if len(picked) >= args.samples:
                break
            picked.append({"step": step,
                           "epoch": ph["bwd"][0].get("epoch"),
                           "g": g[r],
                           "h": (h[r] if h is not None and h.shape[0] == b
                                 else None),
                           "true": rows[r].tolist()})
        cursor += b
        if len(picked) >= args.samples:
            break
    if not picked:
        raise SystemExit(f"[{EXPERIMENT_ID}] --samples {args.samples} "
                         "selected no sequences")
    return picked, picked[0]["g"].shape[0], picked[0]["g"].shape[-1]


def _tag_candidates_torch(g_star, E):
    """C [n, s, V]: cos(-g*_i, E[v]) per sequence/position (batched)."""
    rn = torch.nn.functional.normalize(-g_star.float(), dim=-1)
    en = torch.nn.functional.normalize(E.float(), dim=-1)
    return rn @ en.T


def _tag_dlgpp_baseline(split, h_star, E, args, device):
    """The dlgpp-style baseline at equal budget: free optimization of a
    dummy embedding sequence Z through the KNOWN public head to match the
    captured seam ACTIVATION h* (L2 — the DLG-era objective), multi-restart
    + nearest-embedding snap. Gradient matching is UNAVAILABLE here (the
    body is not granted, so a simulated gradient cannot be computed) — the
    baseline is activation-only, which is exactly the DLG-era limitation
    the tag arm overcomes by exploiting the gradient direction. Returns
    (pred [n, s], final_loss)."""
    n, s, h_dim = h_star.shape
    d = E.shape[1]

    def head_on_emb(Z):
        if hasattr(split, "drop") and hasattr(split, "wpe"):
            pos = torch.arange(s, device=device).unsqueeze(0)
            hidden = split.drop(Z + split.wpe(pos))
        else:
            hidden = Z
        return split._run_blocks(hidden, split.ranges[0], None)

    best = None
    for r in range(args.restarts):
        Z = torch.randn(n, s, d, generator=torch.Generator(
            device=device).manual_seed(args.seed + 900 + r),
            device=device, requires_grad=True)
        opt = torch.optim.Adam([Z], lr=args.lr)
        loss = None
        for _ in range(args.rounds):
            opt.zero_grad()
            loss = ((head_on_emb(Z) - h_star) ** 2).sum()
            loss.backward()
            opt.step()
        if best is None or loss.item() < best[0]:
            best = (loss.item(), Z.detach())
    with torch.no_grad():
        pred = torch.nn.functional.cosine_similarity(
            best[1].unsqueeze(2).float(),
            E.view(1, 1, *E.shape).float(), dim=-1).argmax(-1)
    return pred, best[0]


def _run_tag_real(args, ev):
    require_torch(EXPERIMENT_ID)
    from ..splitbench import SplitPlan, build_split, _n_layers_of
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.model:
        raise SystemExit(f"[{EXPERIMENT_ID}] tag real tier needs --model "
                         "(public base weights: grants E and the public "
                         "head)")
    picked, seq_len, h_dim = _tag_load_sequences(args)
    model = _tag_load_hf(args.model, device)
    split = build_split(model, SplitPlan.parse(
        args.split, n_layers=_n_layers_of(model)))
    E = model.get_input_embeddings().weight.detach().to(device)
    vocab = E.shape[0]
    if args.calibration_file:
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
    control = _majority_control_token([cal])

    out = artifacts.make_artifact(
        "dtraining.attacker.dlgpp.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": False,
         "arm": "tag", "assignment": args.assignment,
         "capture_dir": args.capture_dir, "canonical_pt": args.canonical_pt,
         "model": args.model, "split": args.split, "samples": args.samples,
         "calibration_seqs": args.calibration_seqs,
         "calibration_file": args.calibration_file,
         "rounds": args.rounds, "restarts": args.restarts, "lr": args.lr,
         "device": device, "seq_len": seq_len, "h_dim": h_dim},
        "semi-honest cloud observing the per-position seam gradient "
        "g* = dL/dh (phase='bwd' captures), granted the public token "
        "embedding matrix E and public head (--model at --split; the tied "
        "unembedding places the label direction in E's space). tag arm: "
        "cosine per-token gradient-direction scoring + a light assignment "
        "(greedy / Hungarian) instead of free optimization; recovers "
        "tokens[1:] (position 0 is nobody's label, BLI convention). A "
        "dlgpp-style activation-match optimization baseline runs at the "
        "same budget (rounds/restarts) for the improvement measurement. "
        "Canonical tokens are evaluation truth, never an attack input "
        "(label-free attack).",
        interpretation="the candidate score C[i,v] = cos(-g*_i, E[v]) is "
                       "the Geiping cosine objective realized per-token; "
                       "the decode is greedy per-position argmax (default) "
                       "or Hungarian one-to-one (exact for distinct labels, "
                       "underperforms greedy on repeating text). seq_accuracy/"
                       "set-P/R are position-aligned/orderless vs canonical "
                       "tokens[1:]; the control decodes the calibration-mode "
                       "token at every position; the gate is the frozen-gate "
                       "excess over that control, n = scored tokens. The "
                       "dlgpp-style baseline is activation-only (the body is "
                       "not granted, so gradient matching is unavailable) — "
                       "the DLG-era limitation tag overcomes.",
        evidence=ev)

    cells = []
    groups = {}
    order = []
    for i, p in enumerate(picked):
        key = (p["step"], p["epoch"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)
    for key in order:
        idx = groups[key]
        g_star = torch.stack([picked[i]["g"] for i in idx]).to(device)
        trues = [picked[i]["true"][1:] for i in idx]
        cell = {"experiment": EXPERIMENT_ID, "arm": "tag",
                "submode": f"real-{args.assignment}", "mode": args.mode,
                "step": key[0], "epoch": key[1]}
        try:
            C = _tag_candidates_torch(g_star, E)[:, :-1, :]
            cand = C.cpu().tolist()
            preds = [_tag_decode(c, args.assignment) for c in cand]
            cell.update(_score_tag(preds, trues, control))
            if all(picked[i]["h"] is not None for i in idx):
                h_star = torch.stack([picked[i]["h"] for i in idx]).to(device)
                base_pred, base_loss = _tag_dlgpp_baseline(
                    split, h_star, E, args, device)
                base_trues = [picked[i]["true"] for i in idx]
                base_acc, _, _ = _seq_accuracy(
                    [bp[1:].tolist() for bp in base_pred.cpu()],
                    [t[1:] for t in base_trues])
                cell["dlgpp_baseline_seq_accuracy"] = round(base_acc, 6)
                cell["dlgpp_baseline_objective"] = round(base_loss, 8)
                cell["improvement_pp"] = round(
                    100.0 * (cell["seq_accuracy"] - base_acc), 4)
        except Exception as e:
            journal_error(args.output, EXPERIMENT_ID, cell, e)
            continue
        artifacts.append_jsonl(args.output, cell)
        if ev is not None:
            ev.event("cell_result", cell)
        out["results"].append(cell)
        cells.append(cell)
        msg = (f"[tag] step={key[0]} ({args.assignment}): acc="
               f"{cell['seq_accuracy']} control={cell['control_accuracy']} "
               f"gate={cell['gate']['verdict']}")
        if "dlgpp_baseline_seq_accuracy" in cell:
            msg += (f" | dlgpp-baseline={cell['dlgpp_baseline_seq_accuracy']}"
                    f" (+{cell['improvement_pp']} pp)")
        print(msg)
    if cells:
        n_tok = sum(c["n_tokens"] for c in cells)
        n_corr = sum(c["n_correct"] for c in cells)
        c_corr = sum(c["n_control_correct"] for c in cells)
        pooled = n_corr / max(1, n_tok)
        pooled_ctl = c_corr / max(1, n_tok)
        base = [c["dlgpp_baseline_seq_accuracy"] for c in cells
                if "dlgpp_baseline_seq_accuracy" in c]
        out["summary"].append({
            "experiment": EXPERIMENT_ID, "arm": "tag",
            "assignment": args.assignment, "n_cells": len(cells),
            "n_tokens": n_tok, "seq_accuracy": round(pooled, 6),
            "control_accuracy": round(pooled_ctl, 6),
            "dlgpp_baseline_seq_accuracy_mean": (
                sum(base) / len(base) if base else None),
            "gate": gate.verdict(pooled, pooled_ctl, n_tok),
            "gate_verdicts": [c["gate"]["verdict"] for c in cells]})
    artifacts.write_artifact(args.output, out, evidence=ev)
    return 0


def _run_tag(args, ev):
    if args.toy:
        if args.quick:
            args.seeds = [0, 1]
            args.seq_len = 8
        return _run_tag_toy(args, ev)
    if not args.capture_dir:
        raise SystemExit(f"[{EXPERIMENT_ID}] tag arm: pass --toy (torch-free "
                         "machinery check), or the real-tier arguments "
                         "--capture-dir D --canonical-pt C --model M "
                         "[--split S] (splitbench bwd captures).")
    return _run_tag_real(args, ev)


def _run_new_arms(args, ev):
    arms = ("idlg", "prior-arm") if args.arm == "all" else (args.arm,)
    out = artifacts.make_artifact(
        "dtraining.attacker.dlgpp.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "arm": args.arm, "hidden": args.hidden, "vocab": args.vocab,
         "seq_len": args.seq_len, "restarts": args.restarts,
         "rounds": args.rounds, "prior_lambda": args.prior_lambda,
         "smooth": args.smooth, "seeds": args.seeds},
        "semi-honest cloud observes per-row boundary activations h* and "
        "head gradients (g* = dL/dh, last-layer bias rows db) at the "
        "split depth. idlg arm: analytic iDLG label recovery + downstream "
        "reconstruction vs the DLG++ optimization arm on the same world. "
        "prior-arm: GradInversion-style group-lasso/TV regularized "
        "inversion on locally coherent sequences.",
        interpretation="toy worlds are machinery checks on REAL random "
                       "linear classifiers (pure python, torch-free): "
                       "iDLG label exactness is the published CE sign "
                       "argument, not a rigged lookup; the prior-arm PSNR "
                       "delta is measured on sequences where smoothness is "
                       "a true data prior. Real-capture runs use --dlgpp.",
        evidence=ev)
    for seed in args.seeds:
        for arm in arms:
            if arm == "idlg":
                rec = {"experiment": EXPERIMENT_ID, "arm": arm,
                       "seed": seed, **toy_idlg(args, seed)}
                print(f"[idlg-toy] seed={seed}: label acc="
                      f"{rec['label_accuracy']}% guided="
                      f"{rec['guided_token_recovery']}% vs dlgpp-arm "
                      f"{rec['dlgpp_arm_token_recovery']}% "
                      f"(+{rec['improvement_pp']} pp)")
            else:
                recs = []
                for prior in ("none", "group-lasso"):
                    rec = {"experiment": EXPERIMENT_ID, "arm": arm,
                           "seed": seed, **toy_prior(args, seed, prior)}
                    artifacts.append_jsonl(args.output, rec)
                    if ev is not None:
                        ev.event("cell_result", rec)
                    out["results"].append(rec)
                    recs.append(rec)
                    print(f"[prior-toy] seed={seed} prior={prior}: "
                          f"psnr={rec['psnr_db']} dB rmse={rec['rmse']} "
                          f"recovery={rec['token_recovery']}%")
                gain = (recs[1]["psnr_db"] - recs[0]["psnr_db"])
                out["summary"].append({
                    "experiment": EXPERIMENT_ID, "arm": arm, "seed": seed,
                    "psnr_gain_db": round(gain, 4),
                    "recovery_gain_pp": round(
                        recs[1]["token_recovery"]
                        - recs[0]["token_recovery"], 4)})
                continue
            artifacts.append_jsonl(args.output, rec)
            if ev is not None:
                ev.event("cell_result", rec)
            out["results"].append(rec)
    for arm in arms:
        vals = [r for r in out["results"] if r["arm"] == arm]
        if arm == "idlg" and vals:
            out["summary"].append({
                "experiment": EXPERIMENT_ID, "arm": arm,
                "label_accuracy_mean": sum(r["label_accuracy"]
                                           for r in vals) / len(vals),
                "improvement_pp_mean": sum(r["improvement_pp"]
                                           for r in vals) / len(vals),
                "n_cells": len(vals)})
    artifacts.write_artifact(args.output, out, evidence=ev)
    return 0


def run(args):
    ev = _make_evidence(args)
    if args.dlgpp:
        here = os.path.dirname(os.path.abspath(__file__))
        script = os.path.join(here, "..", "..", "split-training",
                              "dlgpp.py")
        if not os.path.exists(script):
            raise SystemExit(f"dlgpp.py not found at {script}")
        cmd = [sys.executable, script] + shlex.split(args.dlgpp)
        if "--output" not in args.dlgpp:
            if not args.output:
                raise SystemExit("pass --output (or include --output in "
                                 "--dlgpp args)")
            cmd += ["--output", args.output + ".dlgpp.json"]
        print(f"[dlgpp] delegating: {' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            if ev is not None:
                ev.event("error", {"error": f"dlgpp delegation rc={rc}"})
                ev.finalize()
            return rc
        dlg_out = cmd[cmd.index("--output") + 1]
        out = artifacts.make_artifact(
            "dtraining.attacker.dlgpp.v1",
            {"attack": EXPERIMENT_ID, "delegated": True,
             "dlgpp_args": args.dlgpp, "mode": args.mode},
            "semi-honest cloud observes boundary activation h* and "
            "gradient g* = dL/dh at the split depth (DLG++ A2 attacker, "
            "delegated to split-training/dlgpp.py).",
            evidence=ev)
        with open(dlg_out) as f:
            out["results"] = [json.load(f)]
        artifacts.write_artifact(args.output, out, evidence=ev)
        return 0
    if args.arm == "tag":
        return _run_tag(args, ev)
    if args.arm != "dlgpp":
        if not args.toy:
            raise SystemExit(
                f"[{EXPERIMENT_ID}] --arm {args.arm} real-capture path "
                "(torch-guarded): feed per-row boundary activations h*, "
                "head gradients g* = dL/dh and last-layer bias-gradient "
                "rows db from a split-training capture; --toy exercises "
                "the full math torch-free.")
        if args.quick:
            args.seeds, args.restarts, args.rounds = [0, 1], 2, 100
            args.seq_len = 8
        return _run_new_arms(args, ev)
    require_torch(EXPERIMENT_ID)
    if not args.toy:
        raise SystemExit(f"[{EXPERIMENT_ID}] pass --toy, or --dlgpp for "
                         "the model-driven delegation path.")
    if args.quick:
        args.seeds, args.restarts, args.rounds = [0, 1], 2, 100
        args.seq_len = 8
    out = artifacts.make_artifact(
        "dtraining.attacker.dlgpp.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "hidden": args.hidden, "seq_len": args.seq_len,
         "restarts": args.restarts, "rounds": args.rounds,
         "prior": args.prior, "seeds": args.seeds},
        "semi-honest cloud observes boundary activation h* and gradient "
        "g* = dL/dh at the split depth; optimization inversion "
        "(multi-restart, L2 match on h and g, nearest-embedding snap).",
        interpretation="toy path is a machinery check; real numbers come "
                       "from the --dlgpp delegation (split-training/"
                       "dlgpp.py, depths 1/4/8, Qwen3-0.6B).",
        evidence=ev)
    for seed in args.seeds:
        rec = {"experiment": EXPERIMENT_ID, "seed": seed,
               **toy_dlg(args, seed)}
        artifacts.append_jsonl(args.output, rec)
        if ev is not None:
            ev.event("cell_result", rec)
        out["results"].append(rec)
        print(f"[dlgpp-toy] seed={seed}: token recovery="
              f"{rec['token_recovery']}% (random "
              f"{rec['random_baseline']}%)")
    out["summary"].append({
        "experiment": EXPERIMENT_ID,
        "token_recovery_mean": sum(r["token_recovery"]
                                   for r in out["results"])
        / max(1, len(out["results"]))})
    artifacts.write_artifact(args.output, out, evidence=ev)
    return 0


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import tempfile
    print("gradient-inversion idlg/prior-arm toy cores (pure python):")
    world = toy_classification_world(0, hidden=16, vocab=8, seq_len=4)
    check("world carries ground truth + real gradient rows",
          len(world["x"]) == 4 and len(world["db"]) == 4
          and len(world["db"][0]) == 8 and len(world["g_star"][0]) == 8)
    check("world generation is deterministic",
          toy_classification_world(0, 16, 8, 4)["x"] == world["x"])
    labels = idlg_labels(world)
    check("iDLG argmin recovers the toy labels exactly (CE sign argument)",
          labels == world["y"])
    check("iDLG label accuracy fixture is 100%",
          toy_idlg(build_parser().parse_args(
              ["--toy", "--arm", "idlg", "--seq-len", "4", "--hidden", "16",
               "--vocab", "8", "--rounds", "30", "--restarts", "1",
               "--seeds", "0"]), 0)["label_accuracy"] == 100.0)

    z = [[0.0, 1.0], [0.4, 0.6], [0.8, 0.2]]
    check("group-lasso prior = sum of adjacent-row norms",
          abs(_prior_value(z, "group-lasso")
              - (math.sqrt(0.4 ** 2 + 0.4 ** 2)
                 + math.sqrt(0.4 ** 2 + 0.4 ** 2))) < 1e-12)
    check("tv prior = sum of adjacent |diffs|",
          abs(_prior_value(z, "tv") - (0.8 + 0.8)) < 1e-12)
    grad = [[0.0, 0.0] for _ in z]
    _add_prior_subgradient(grad, z, "group-lasso", 0.1)
    z_step = [[z[i][j] - 0.05 * grad[i][j] for j in range(2)]
              for i in range(3)]
    check("group-lasso subgradient step lowers the prior value",
          _prior_value(z_step, "group-lasso")
          < _prior_value(z, "group-lasso"))
    rmse, psnr = _rmse_psnr(z, z)
    check("psnr of a perfect reconstruction is the capped 99 dB",
          rmse == 0.0 and psnr == 99.0)

    check("parser builds torch-less with the new flags",
          build_parser().parse_args([]).arm == "dlgpp"
          and build_parser().parse_args([]).prior == "none")
    with tempfile.TemporaryDirectory() as d:
        ns = build_parser().parse_args(
            ["--toy", "--arm", "idlg", "--quick", "--hidden", "16",
             "--vocab", "8", "--output", os.path.join(d, "idlg.json")])
        ns.mode = "training"
        check("idlg arm runs end to end torch-free",
              run(ns) == 0 and os.path.isfile(
                  os.path.join(d, "idlg.json")))
        check("idlg artifact journal written",
              os.path.isfile(os.path.join(d, "idlg.json.jsonl")))

    print("gradient-inversion tag arm toy core (pure python):")
    tw = toy_tag_world(0, vocab=16, dim=32, seq_len=4, n_seq=2)
    check("tag world is deterministic and shaped",
          len(tw["seqs"]) == 2 and len(tw["seqs"][0]["g"]) == 3
          and len(tw["seqs"][0]["g"][0]) == 32
          and toy_tag_world(0, 16, 32, 4, n_seq=2)["seqs"][0]["ids"]
          == tw["seqs"][0]["ids"])
    C = tag_candidates(tw, tw["seqs"][0]["g"])
    check("candidate matrix is [seq_len-1, vocab] of cosines",
          len(C) == 3 and len(C[0]) == 16
          and all(-1.0 - 1e-9 <= c <= 1.0 + 1e-9 for row in C for c in row))
    check("planted candidates peak at the label (exact direction argument; "
          "near-orthogonal dim >= 2*vocab regime)",
          all(max(range(16), key=lambda v: C[i][v])
              == tw["seqs"][0]["labels"][i] for i in range(3)))
    cost = [[-1.0 if j == i else 0.0 for j in range(4)] for i in range(3)]
    check("hungarian solves the identity assignment",
          _hungarian_assign(cost) == [0, 1, 2])
    args_tag = build_parser().parse_args(
        ["--toy", "--arm", "tag", "--vocab", "32", "--hidden", "16",
         "--seq-len", "6", "--toy-seqs", "6", "--seeds", "0", "1"])
    for seed in (0, 1):
        planted = toy_tag_cell(args_tag, "planted", seed)
        null = toy_tag_cell(args_tag, "null", seed)
        check(f"tag toy seed {seed}: planted >90%, null at control",
              planted["seq_accuracy"] > 0.90
              and abs(null["seq_accuracy"] - null["control_accuracy"]) < 0.05
              and planted["gate"]["verdict"] == "FAIL")
    with tempfile.TemporaryDirectory() as d:
        ns = build_parser().parse_args(
            ["--toy", "--arm", "tag", "--quick",
             "--output", os.path.join(d, "tag.json")])
        ns.mode = "training"
        check("tag toy arm runs end to end torch-free",
              run(ns) == 0 and os.path.isfile(os.path.join(d, "tag.json")))
        check("tag toy artifact has planted+null cells with gate verdicts",
              len(json.load(open(os.path.join(d, "tag.json")))["results"])
              == 4)
        check("tag toy journal written",
              os.path.isfile(os.path.join(d, "tag.json.jsonl")))
        ns2 = build_parser().parse_args(["--arm", "tag"])
        ns2.mode = "training"
        try:
            run(ns2)
            check("tag real tier without --capture-dir is descriptive", False)
        except SystemExit as e:
            check("tag real tier without --capture-dir is descriptive",
                  "--capture-dir" in str(e))
    check("parser builds torch-less with the tag flags",
          build_parser().parse_args([]).assignment == "greedy"
          and build_parser().parse_args(["--arm", "tag"]).arm == "tag"
          and build_parser().parse_args(["--assignment", "hungarian"])
          .assignment == "hungarian")

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
