#!/usr/bin/env python3
"""body-extraction — query-driven functional extraction of the SERVER's
body slice in a split deployment: MODEL-IP theft, distinct from data
privacy. The attacker submits inputs through the victim head (or drives a
public head of the same architecture) and observes the body slice's
seam-facing outputs; the goal is a STUDENT body functionally matching the
victim body — not the recovery of any training example.

Threat model (mirrored in every artifact): the attacker can (a) choose
hidden-state queries — random draws, corpus-driven rows pushed through the
public head, or offline probe rows from a wire capture (--capture-dir:
the wire fwd/prefill/decode rows are exactly what the body received in a
recorded session) — and (b) observe the body's outputs on them (the
serving API of a body-exposing deployment, or the downstream seam of a
3-slice split). The attacker is granted the slice CONFIG (arch, layer
count, widths) but NOT the body weights. Label-free.

Two arms:

  distill     functional distillation: query the victim body on
              attacker-chosen hidden states, train a student slice (same
              arch, smaller or equal) to match outputs (MSE on hidden
              states). Metrics: held-out functional similarity (R² / mean
              cosine on fresh probes) and a downstream-transfer check —
              the student replaces the body behind the PUBLIC tail
              (ln_f + lm_head; tied embeddings make the unembedding
              public and ln_f ships with the slice config) and we measure
              top-1 logit agreement with the victim path. Control: the
              UNTRAINED student (same fresh init draw, zero queries)
              scored on the same probes. Gate: frozen-gate
              Bonferroni-Wilson excess over the control, scored at the
              ELEMENT level (fraction of held-out hidden entries the
              student reproduces within 5% of the victim output std —
              the honest large-n match unit, cf. gpu_memory's byte
              level); R²/cosine/logit-agreement stay interpretive.
  weightsim   weights-space proximity ESTIMATE (granted arch match):
              per-layer output drift between victim and student on shared
              probes, plus per-layer weight drift as simulation ground
              truth. HONESTY NOTE: true weight recovery needs
              cryptanalytic access we do NOT claim — functional
              equivalence is the honest extraction metric, and the toy
              tier demonstrates exactly why: for linear stacks the
              per-layer decomposition is non-identifiable from I/O, so a
              student can match the victim FUNCTIONALLY (R² ~ 1) while
              its per-layer weights drift arbitrarily far.

Tiers:

  --toy       torch-free machinery check. Victim body = random linear map
              (distill) / 2-layer linear stack (weightsim); the student
              is fit in CLOSED FORM (ridge least squares; rank-limited
              ALS when --student-dim < --hidden). The planted closed-form
              fit reaches R² > 0.99 with ~perfect transfer agreement; the
              random (untrained) student sits at control.
  (real)      --model M --split S: local victim sim — a tiny random GPT-2
              (or any registered arch) split via attacker.splitbench; the
              victim body is the blocks after the seam (hidden-state
              output), the student a reinitialized same-arch slice
              trained with Adam (--steps, --lr) on --queries query rows
              (random + head-driven + --capture-dir rows). The victim
              body is planted off init by --victim-jitter (default 1.0):
              a random-init GPT-2 body is near-identity (residual
              stream), so ANY same-arch init would match it and the
              untrained control would beat the attack — the jitter is the
              TRAINED-body stand-in (0 disables, e.g. against a
              pretrained checkpoint). Documented approximation: this
              queries a LOCAL body instance standing in for the victim's
              serving API — the query plumbing is identical, the
              responses are not exfiltrated over a network.

Offline probe rows: --capture-dir supplies the QUERY distribution only
(corpus-realistic rows the body once consumed); responses still come from
the local victim sim — the wire schema records what the body RECEIVES,
never its outputs, so a pure-capture run cannot pair responses (a 3-slice
deployment capturing the body|tail seam could; not implemented here).

Usage:
    python -m attacker --mode training --attack body-extraction --help
    python -m attacker --mode inference --attack body-extraction --toy \
        --quick --output /tmp/be.json
    python -m attacker --mode training --attack body-extraction \
        --model tinygpt2/ --split 1 --queries 512 --steps 200 \
        --output be.json --evidence
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

EXPERIMENT_ID = "body_slice_extraction"
MODES = ("training", "inference")
REQUIRES_LABELS = False
DESCRIPTION = ("query-driven functional extraction of the server's body "
               "slice (model-IP theft, not data privacy): distill a "
               "student body from attacker-chosen hidden-state queries "
               "(MSE on hidden states; held-out R²/cosine + public-tail "
               "logit-agreement transfer check) + weightsim per-layer "
               "drift proxy (functional equivalence is the honest metric; "
               "no weight-recovery claim)")

SCHEMA = "dtraining.attacker.body_extraction.v1"
ARMS = ("distill", "weightsim")
TOY_RIDGE_LAMBDA = 1e-6

THREAT_MODEL = (
    "attacker with query access to the victim body slice (hidden-state "
    "queries submitted through the victim head or a public head of the "
    "same architecture; the body's seam-facing outputs are observable) "
    "and the slice CONFIG (arch/layers/widths) but NOT the body weights; "
    "the goal is a student body FUNCTIONALLY matching the victim body "
    "(model-IP theft), not the recovery of training data. Label-free: "
    "held-out probe ground truth is evaluation-only, never an attack "
    "input.")

INTERPRETATION = (
    "extraction_channel=open (gate FAIL): the distilled student matches "
    "the victim body FUNCTIONALLY above the untrained-student control — "
    "the gate is scored at the element level (fraction of held-out "
    "hidden entries reproduced within 5% of the victim output std); "
    "held-out R²/cosine and top-1 logit agreement through the public "
    "tail (student replaces the body, same ln_f + lm_head) are the "
    "interpretive metrics. The "
    "weightsim arm reports per-layer output drift as the "
    "extraction-quality proxy and per-layer weight drift as simulation "
    "ground truth: functional extraction does NOT recover weights — true "
    "weight recovery needs cryptanalytic access this module does not "
    "claim. Toy tier is a machinery check of the fit/scoring/gate "
    "plumbing on a KNOWN linear body, not a security claim; real numbers "
    "come from --model/--split runs.")


def _eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _transpose(A):
    return [list(col) for col in zip(*A)]


def _matmul(A, B):
    """A [r][k] @ B [k][c] -> [r][c] (row lists)."""
    Bt = _transpose(B)
    return [[sum(a * b for a, b in zip(ra, cb)) for cb in Bt] for ra in A]


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
                             "(queries vs hidden dim) or the ridge")
        aug[col], aug[piv] = aug[piv], aug[col]
        inv = 1.0 / aug[col][col]
        aug[col] = [v * inv for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                f = aug[r][col]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def _ridge_fit(X, Y, lam):
    """Closed-form ridge least squares: W = (X^T X + lam I)^-1 X^T Y.
    X [Q][H], Y [Q][H] -> W [H][H] with pred(x) = x @ W (right-apply)."""
    h = len(X[0])
    gram = _eye(h)
    for i in range(h):
        gram[i][i] = lam
    cross = [[0.0] * h for _ in range(h)]
    for x, y in zip(X, Y):
        for i, xi in enumerate(x):
            if xi == 0.0:
                continue
            gi, ci = gram[i], cross[i]
            for j in range(h):
                gi[j] += xi * x[j]
                ci[j] += xi * y[j]
    return _gauss_solve(gram, cross)


def _als_rank_fit(W_star, rank, steps, seed):
    """Rank-limited student: alternating least squares W* ~ U V with
    U [H][rank], V [rank][H]. The "smaller student" knob: fitting the
    composed closed-form map at reduced rank (reduced-rank regression
    without a full SVD; converges to the same subspace)."""
    h = len(W_star)
    rng = random.Random(seed)
    U = [[rng.gauss(0.0, 1.0) for _ in range(rank)] for _ in range(h)]
    for _ in range(max(1, steps)):
        V = _gauss_solve(_matmul(_transpose(U), U),
                         _matmul(_transpose(U), W_star))
        Vt = _transpose(V)
        Ut = _gauss_solve(_matmul(V, Vt),
                          _matmul(V, _transpose(W_star)))
        U = _transpose(Ut)
    return _matmul(U, V)


def _fro(A):
    return math.sqrt(sum(v * v for row in A for v in row))


def _r2(pred, true):
    """Global coefficient of determination over all entries (<= 1)."""
    ss_res = ss_tot = 0.0
    n = 0
    mean = 0.0
    for row in true:
        for v in row:
            mean += v
            n += 1
    mean /= max(1, n)
    for pr, tr in zip(pred, true):
        for p, t in zip(pr, tr):
            ss_res += (p - t) * (p - t)
            ss_tot += (t - mean) * (t - mean)
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _mean_cosine(pred, true):
    """Mean per-row cosine similarity."""
    total, n = 0.0, 0
    for pr, tr in zip(pred, true):
        np_ = math.sqrt(sum(v * v for v in pr))
        nt = math.sqrt(sum(v * v for v in tr))
        if np_ > 0.0 and nt > 0.0:
            total += sum(p * t for p, t in zip(pr, tr)) / (np_ * nt)
            n += 1
    return total / max(1, n)


def _argmax(row):
    return max(range(len(row)), key=lambda i: row[i])


def _element_match(pred, true, tol_frac=0.05):
    """The honest large-n functional-match unit for the frozen gate (cf.
    gpu_memory's byte level): fraction of held-out ENTRIES with
    |pred - true| < tol_frac * global-std(true). A closed-form /
    well-distilled student matches ~every entry; an untrained student
    matches ~2*tol_frac of them. Returns (match_fraction, n_elements)."""
    n = 0
    ss = 0.0
    mean = 0.0
    for row in true:
        for v in row:
            mean += v
            n += 1
    mean /= max(1, n)
    for row in true:
        for v in row:
            ss += (v - mean) * (v - mean)
    tol = tol_frac * math.sqrt(ss / max(1, n))
    match = 0
    for pr, tr in zip(pred, true):
        for p, t in zip(pr, tr):
            match += 1 if abs(p - t) < tol else 0
    return match / max(1, n), n


def _transfer_metrics(pred, true, tail):
    """Downstream-transfer check through the public toy tail T [V][H]
    (logits = h @ T^T): top-1 agreement fraction between the student and
    victim paths."""
    agree = n = 0
    for pr, tr in zip(pred, true):
        lp = [sum(p * t for p, t in zip(pr, tv)) for tv in tail]
        lt = [sum(p * t for p, t in zip(tr, tv)) for tv in tail]
        agree += 1 if _argmax(lp) == _argmax(lt) else 0
        n += 1
    return agree / max(1, n), n


def toy_world(seed, hidden, vocab, n_layers=1):
    """Deterministic toy victim: `n_layers` right-apply linear maps
    [H][H] (the body stack), a public tail [V][H] (toy ln_f+lm_head), and
    8 corpus prototypes [H] (the structured directions head outputs
    cluster around)."""
    rng = random.Random(seed)
    layers = [[[rng.gauss(0.0, 1.0) for _ in range(hidden)]
               for _ in range(hidden)] for _ in range(n_layers)]
    tail = [[rng.gauss(0.0, 1.0) for _ in range(hidden)]
            for _ in range(vocab)]
    protos = [[rng.gauss(0.0, 2.0) for _ in range(hidden)]
              for _ in range(8)]
    return {"layers": layers, "tail": tail, "prototypes": protos,
            "hidden": hidden, "vocab": vocab}


def toy_queries(world, n_rows, corpus_frac, seed):
    """Attacker-chosen query rows: a mix of corpus-driven rows (a random
    prototype direction + small noise — the toy stand-in for pushing real
    tokens through the public head) and plain gaussian draws."""
    rng = random.Random(seed)
    h = world["hidden"]
    rows = []
    n_corpus = int(n_rows * corpus_frac)
    for i in range(n_rows):
        if i < n_corpus:
            p = world["prototypes"][rng.randrange(len(world["prototypes"]))]
            rows.append([v + rng.gauss(0.0, 0.1) for v in p])
        else:
            rows.append([rng.gauss(0.0, 1.0) for _ in range(h)])
    return rows


def _stack_apply(X, layers):
    for W in layers:
        X = _matmul(X, W)
    return X


def _random_map(hidden, seed):
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 1.0) for _ in range(hidden)]
            for _ in range(hidden)]


def _functional_metrics(pred, true, tail):
    match, n_elem = _element_match(pred, true)
    agreement, n_rows = _transfer_metrics(pred, true, tail)
    return {"r2": _r2(pred, true), "cosine": _mean_cosine(pred, true),
            "element_match": match, "n_eval_elements": n_elem,
            "logit_agreement": agreement, "n_eval_rows": n_rows}


def toy_distill_cell(args, seed):
    """One toy distill cell: closed-form student fit on attacker-chosen
    queries; planted vs the untrained (random-map) control."""
    world = toy_world(args.seed + 77 + 1000 * seed, args.hidden,
                      args.vocab, n_layers=1)
    W_v = world["layers"][0]
    X = toy_queries(world, args.queries, args.corpus_frac,
                    args.seed + 1000 * seed)
    Y = _matmul(X, W_v)
    W_s = _ridge_fit(X, Y, TOY_RIDGE_LAMBDA * max(1, len(X)))
    student = "closed-form"
    if args.student_dim and args.student_dim < args.hidden:
        W_s = _als_rank_fit(W_s, args.student_dim, args.steps,
                            args.seed + 555 + 1000 * seed)
        student = f"rank-{args.student_dim}-als"
    X_te = toy_queries(world, args.eval_rows, args.corpus_frac,
                       args.seed + 4242 + 1000 * seed)
    Y_te = _matmul(X_te, W_v)
    P_te = _matmul(X_te, W_s)
    ctl = _random_map(args.hidden, args.seed + 999 + 1000 * seed)
    C_te = _matmul(X_te, ctl)
    rec = {"tier": "sim", "student": student, "queries": len(X),
           "student_dim": args.student_dim or args.hidden,
           **_functional_metrics(P_te, Y_te, world["tail"])}
    ctl_m = _functional_metrics(C_te, Y_te, world["tail"])
    rec["control_r2"] = ctl_m["r2"]
    rec["control_cosine"] = ctl_m["cosine"]
    rec["control_logit_agreement"] = ctl_m["logit_agreement"]
    rec["control_element_match"] = ctl_m["element_match"]
    return rec


def toy_weightsim_cell(args, seed):
    """One toy weightsim cell: 2-layer linear victim body; the composed
    map is fit in closed form and re-decomposed with a random invertible
    A — the student matches the FUNCTION (R² ~ 1) while its per-layer
    weights drift arbitrarily (linear stacks are non-identifiable from
    I/O). Per-layer output/weight drift is the extraction-quality proxy;
    functional equivalence is the honest metric."""
    world = toy_world(args.seed + 78 + 1000 * seed, args.hidden,
                      args.vocab, n_layers=2)
    W1, W2 = world["layers"]
    X = toy_queries(world, args.queries, args.corpus_frac,
                    args.seed + 1000 * seed)
    Y = _stack_apply(X, (W1, W2))
    W_star = _ridge_fit(X, Y, TOY_RIDGE_LAMBDA * max(1, len(X)))
    rng = random.Random(args.seed + 555 + 1000 * seed)
    h = args.hidden
    A = _eye(h)
    for i in range(h):
        for j in range(i + 1, h):
            A[i][j] = rng.gauss(0.0, 0.5)
    A_inv = _gauss_solve(A, _eye(h))
    S1 = _matmul(W_star, A)
    S2 = A_inv
    X_te = toy_queries(world, args.eval_rows, args.corpus_frac,
                       args.seed + 4242 + 1000 * seed)
    Y_te = _stack_apply(X_te, (W1, W2))
    L1_v = _matmul(X_te, W1)
    L1_s = _matmul(X_te, S1)
    P_te = _matmul(L1_s, S2)
    ctl1 = _random_map(h, args.seed + 999 + 1000 * seed)
    ctl2 = _random_map(h, args.seed + 998 + 1000 * seed)
    C_te = _stack_apply(X_te, (ctl1, ctl2))
    rec = {"tier": "sim", "student": "closed-form+gauge",
           "queries": len(X), "student_dim": args.hidden,
           **_functional_metrics(P_te, Y_te, world["tail"]),
           "functional_r2": _r2(P_te, Y_te),
           "per_layer_output_drift": [
               _fro([[s - v for s, v in zip(rs, rv)]
                     for rs, rv in zip(L1_s, L1_v)]) / max(_fro(L1_v), 1e-12),
               _fro([[s - v for s, v in zip(rs, rv)]
                     for rs, rv in zip(P_te, Y_te)]) / max(_fro(Y_te), 1e-12)],
           "per_layer_weight_drift": [
               _fro([[s - v for s, v in zip(rs, rv)]
                     for rs, rv in zip(S1, W1)]) / max(_fro(W1), 1e-12),
               _fro([[s - v for s, v in zip(rs, rv)]
                     for rs, rv in zip(S2, W2)]) / max(_fro(W2), 1e-12)],
           "note": "functional equivalence achieved with arbitrary "
                   "per-layer weight drift (linear stacks are "
                   "non-identifiable from I/O): functional match is the "
                   "extraction; weight recovery is NOT claimed (needs "
                   "cryptanalytic access)"}
    ctl_m = _functional_metrics(C_te, Y_te, world["tail"])
    rec["control_r2"] = ctl_m["r2"]
    rec["control_cosine"] = ctl_m["cosine"]
    rec["control_logit_agreement"] = ctl_m["logit_agreement"]
    rec["control_element_match"] = ctl_m["element_match"]
    rec["control_final_output_drift"] = _fro(
        [[s - v for s, v in zip(rs, rv)] for rs, rv in zip(C_te, Y_te)]
    ) / max(_fro(Y_te), 1e-12)
    return rec


def _load_hf_model(name_or_path, device):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] --model requires transformers "
            "(pip install transformers); --toy works torch-less.") from None
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path, torch_dtype=torch.float32,
        attn_implementation="eager")
    return model.to(device).eval()


def _hidden_size(config):
    h = getattr(config, "n_embd", None) or getattr(config, "hidden_size",
                                                   None)
    if h is None:
        raise ValueError(f"cannot determine the hidden size of "
                         f"{type(config).__name__}")
    return int(h)


def _torch_capture_batches(args, split, gen):
    """Offline probe rows -> [1, seq_len, H] query batches. The QUERY
    distribution only; responses still come from the local victim sim
    (module docstring)."""
    from .. import captures
    mode = getattr(args, "mode", None) or "training"
    records = captures.scan_captures(args.capture_dir, mode)
    phases = ("fwd",) if mode == "training" else ("prefill", "decode")
    rows = [captures.rows_2d(captures.load_tensor(pt))
            for meta, pt in records if meta.get("phase") in phases]
    if not rows:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] --capture-dir {args.capture_dir} has no "
            f"{phases} rows for mode {mode!r} (need splitbench wire "
            "captures; the fwd rows are what the body received)")
    rows = torch.cat(rows).float()
    h = _hidden_size(split.model.config)
    if rows.shape[1] != h:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] capture rows have hidden {rows.shape[1]} "
            f"but --model has hidden {h}: offline probe rows must come "
            "from the SAME architecture as the local victim sim")
    s = args.seq_len
    n = rows.shape[0] // s
    if n == 0:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] --capture-dir has {rows.shape[0]} rows, "
            f"need at least --seq-len {s}")
    perm = torch.randperm(n, generator=gen).tolist()
    return [rows[i * s:(i + 1) * s].unsqueeze(0) for i in perm]


def _torch_train_student(args, seed):
    """The shared real-tier engine (both arms report off ONE trained
    student per seed): build the local victim sim, harvest query/response
    pairs, evaluate the untrained student (control), distill with Adam,
    re-evaluate, and measure per-layer drift."""
    require_torch(EXPERIMENT_ID)
    import torch.nn.functional as F
    from .. import splitbench
    device = torch.device(args.device)
    torch.manual_seed(args.seed + 1000 * seed)
    model = _load_hf_model(args.model, device)
    plan = splitbench.SplitPlan.parse(
        args.split, n_layers=splitbench._n_layers_of(model))
    vsplit = splitbench.build_split(model, plan)
    body_range = vsplit.ranges[-1]
    gen = torch.Generator().manual_seed(args.seed + 31 + 1000 * seed)
    h = _hidden_size(model.config)
    vocab = int(model.config.vocab_size)
    s = args.seq_len
    if args.victim_jitter > 0:
        with torch.no_grad():
            for blk in vsplit.blocks[body_range[0]:body_range[1]]:
                for p in blk.parameters():
                    scale = float(p.std())
                    p.add_(torch.randn(p.shape, generator=gen)
                           .to(p.device, p.dtype)
                           * (args.victim_jitter
                              * (scale if scale > 1e-8 else 0.02)))
    if args.capture_dir:
        pool_x = _torch_capture_batches(args, vsplit, gen)
        query_source = "capture-dir"
    else:
        n_batches = max(2, args.queries // (4 * s))
        pool_x = []
        with torch.no_grad():
            for i in range(n_batches):
                if i % 2 == 0:
                    ids = torch.randint(0, vocab, (4, s),
                                        generator=gen).to(device)
                    pool_x.append(vsplit.head_forward(ids).detach())
                else:
                    pool_x.append(torch.randn(4, s, h,
                                              generator=gen).to(device))
        query_source = "random+head"
    with torch.no_grad():
        pool = [(x, vsplit.body_forward(x.to(device), body_range).detach())
                for x in pool_x]
    n_query_rows = sum(x.numel() // h for x, _ in pool)
    n_eval = max(2, len(pool) // 3)
    with torch.no_grad():
        probes = []
        for _ in range(n_eval):
            ids = torch.randint(0, vocab, (4, s), generator=gen).to(device)
            x = vsplit.head_forward(ids).detach()
            probes.append((x, vsplit.body_forward(x, body_range).detach()))
    torch.manual_seed(args.seed + 777 + 1000 * seed)
    student = type(model)(model.config).to(device).eval()
    ssplit = splitbench.build_split(student, plan)

    def tail(hidden):
        return vsplit.lm_head(vsplit.ln_f(hidden))

    def evaluate():
        ss_res = ss_tot = cos_sum = 0.0
        agree = n_rows = 0
        n_elem = match = 0
        var_sum = 0.0
        with torch.no_grad():
            mean = sum(float(y.sum()) for _, y in probes) / max(
                1, sum(y.numel() for _, y in probes))
            for _, y in probes:
                var_sum += float(((y - mean) ** 2).sum())
            tol = 0.05 * math.sqrt(var_sum / max(
                1, sum(y.numel() for _, y in probes)))
            for x, y in probes:
                pred = ssplit.body_forward(x, body_range)
                ss_res += float(((pred - y) ** 2).sum())
                ss_tot += float(((y - y.mean()) ** 2).sum())
                cos_sum += float(F.cosine_similarity(pred, y,
                                                     dim=-1).mean())
                agree += int((tail(pred).argmax(-1)
                              == tail(y).argmax(-1)).sum())
                match += int(((pred - y).abs() < tol).sum())
                n_elem += y.numel()
                n_rows += y.numel() // h
        return {"r2": 1.0 - ss_res / max(ss_tot, 1e-12),
                "cosine": cos_sum / max(1, len(probes)),
                "logit_agreement": agree / max(1, n_rows),
                "element_match": match / max(1, n_elem),
                "n_eval_rows": n_rows, "n_eval_elements": n_elem}

    def drift():
        """Per-layer output drift on the probes (victim prefix vs student
        prefix) + per-layer weight drift (simulation ground truth)."""
        out_drifts = []
        with torch.no_grad():
            for k in range(1, body_range[1] - body_range[0] + 1):
                rng_k = (body_range[0], body_range[0] + k)
                num = den = 0.0
                for x, _ in probes:
                    v = vsplit.body_forward(x, rng_k)
                    u = ssplit.body_forward(x, rng_k)
                    num += float(((v - u) ** 2).sum())
                    den += float((v ** 2).sum())
                out_drifts.append(math.sqrt(num / max(den, 1e-12)))
        w_drifts = []
        with torch.no_grad():
            for k in range(body_range[1] - body_range[0]):
                vb = vsplit.blocks[body_range[0] + k]
                sb = ssplit.blocks[body_range[0] + k]
                num = sum(float(((sp.detach() - vp.detach()) ** 2).sum())
                          for sp, vp in zip(sb.parameters(),
                                            vb.parameters()))
                den = sum(float((vp.detach() ** 2).sum())
                          for vp in vb.parameters())
                w_drifts.append(math.sqrt(num / max(den, 1e-12)))
        return out_drifts, w_drifts

    control = evaluate()
    control_out_drift, control_w_drift = drift()
    params = [p for blk in ssplit.blocks[body_range[0]:body_range[1]]
              for p in blk.parameters()]
    opt = torch.optim.Adam(params, lr=args.lr)
    final_loss = float("nan")
    for step in range(args.steps):
        x, y = pool[step % len(pool)]
        loss = F.mse_loss(ssplit.body_forward(x, body_range), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = float(loss.item())
    trained = evaluate()
    out_drift, w_drift = drift()
    return {"seed": seed, "query_source": query_source,
            "body_layers": [body_range[0], body_range[1]],
            "n_query_rows": n_query_rows, "steps": args.steps,
            "lr": args.lr, "victim_jitter": args.victim_jitter,
            "final_train_loss": final_loss,
            "trained": trained, "control": control,
            "per_layer_output_drift": out_drift,
            "control_per_layer_output_drift": control_out_drift,
            "per_layer_weight_drift": w_drift,
            "control_per_layer_weight_drift": control_w_drift}


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, hidden=64)
    ap.add_argument("--arm", choices=ARMS, nargs="+", default=list(ARMS),
                    help="attack arm(s) to run (default: both)")
    ap.add_argument("--capture-dir", default=None,
                    help="offline probe rows (wire captures; the fwd/"
                         "prefill/decode rows are the QUERY distribution — "
                         "responses still come from the --model victim "
                         "sim)")
    ap.add_argument("--model", default=None,
                    help="HF name/path of the local victim sim (real tier)")
    ap.add_argument("--split", default="6",
                    help="SplitPlan spec, e.g. '6'; the victim body is the "
                         "blocks after the first boundary")
    ap.add_argument("--student-dim", type=int, default=None,
                    help="toy tier only: student width (default --hidden = "
                         "exact closed form; smaller fits a rank-limited "
                         "ALS student). The torch-tier student is always "
                         "an arch-equal reinitialized slice")
    ap.add_argument("--queries", type=int, default=512,
                    help="attacker query budget in hidden-state rows")
    ap.add_argument("--eval-rows", type=int, default=256,
                    help="toy tier: held-out probe rows per cell")
    ap.add_argument("--steps", type=int, default=200,
                    help="real tier: Adam distillation steps; toy tier: "
                         "ALS iterations when --student-dim < --hidden")
    ap.add_argument("--lr", type=float, default=1e-2,
                    help="real tier: student Adam learning rate")
    ap.add_argument("--seq-len", type=int, default=8,
                    help="query sequence length (real tier)")
    ap.add_argument("--vocab", type=int, default=64,
                    help="toy tier: public tail vocab size")
    ap.add_argument("--corpus-frac", type=float, default=0.5,
                    help="fraction of corpus-driven (vs gaussian) query "
                         "rows")
    ap.add_argument("--device", default="cpu",
                    help="real tier device ('cpu' | 'cuda')")
    ap.add_argument("--victim-jitter", type=float, default=1.0,
                    help="real tier: std (relative to each parameter's own "
                         "std) of the gaussian perturbation planting the "
                         "victim body off init — the TRAINED-body stand-in "
                         "(a random-init GPT-2 body is near-identity and "
                         "the control would beat the attack). 0 disables")
    ap.add_argument("--evidence", action="store_true",
                    help="wrap the run in an EvidenceLogger (requires "
                         "--output): the artifact write is recorded as a "
                         "chained event and the evidence manifest is "
                         "finalized alongside")
    return ap


def _gate_cell(rec, n_seeds):
    """Attach the frozen-gate verdict, scored at the ELEMENT level (the
    honest large-n functional-match unit, cf. gpu_memory's byte level):
    extraction channel OPEN (FAIL) when the fraction of held-out hidden
    entries the student reproduces within tolerance exceeds the
    untrained-student control beyond the gate. Held-out R²/cosine and
    top-1 logit agreement stay as the interpretive metrics."""
    g = gate.verdict(rec["element_match"], rec["control_element_match"],
                     rec["n_eval_elements"], n_seeds=n_seeds)
    rec["gate"] = g
    rec["extraction_channel"] = "open" if g["verdict"] == "FAIL" \
        else ("closed" if g["verdict"] == "PASS" else "unmeasured")
    return rec


def _summarize_arm(out, arm, tier, recs):
    out["summary"].append({
        "experiment": EXPERIMENT_ID, "arm": arm, "tier": tier,
        "r2_mean": sum(r["r2"] for r in recs) / len(recs),
        "logit_agreement_mean": sum(r["logit_agreement"]
                                    for r in recs) / len(recs),
        "control_logit_agreement_mean": sum(
            r["control_logit_agreement"] for r in recs) / len(recs),
        "n_seeds": len(recs),
        "extraction_channel": recs[-1]["extraction_channel"],
        "gate_verdict": recs[-1]["gate"]["verdict"]})


def run_toy(args, evidence=None):
    if args.quick:
        args.queries = min(args.queries, 192)
        args.eval_rows = min(args.eval_rows, 128)
        args.seeds = args.seeds[:2]
    out = artifacts.make_artifact(
        SCHEMA,
        {"attack": EXPERIMENT_ID, "mode": getattr(args, "mode", None),
         "toy": True, "arms": args.arm, "hidden": args.hidden,
         "student_dim": args.student_dim or args.hidden,
         "queries": args.queries, "eval_rows": args.eval_rows,
         "corpus_frac": args.corpus_frac, "vocab": args.vocab,
         "seeds": args.seeds, "seed": args.seed},
        THREAT_MODEL + " Toy tier: the victim body is a KNOWN random "
        "linear map / 2-layer linear stack — a machinery check of the "
        "fit/scoring/gate plumbing, not a security claim.",
        interpretation=INTERPRETATION, evidence=evidence)
    for arm in args.arm:
        recs = []
        for seed in args.seeds:
            cell = {"experiment": EXPERIMENT_ID, "arm": arm, "seed": seed}
            try:
                rec = (toy_distill_cell(args, seed) if arm == "distill"
                       else toy_weightsim_cell(args, seed))
            except Exception as e:
                journal_error(args.output, EXPERIMENT_ID, cell, e)
                continue
            rec = {**cell, **rec}
            _gate_cell(rec, n_seeds=len(args.seeds))
            artifacts.append_jsonl(args.output, rec)
            out["results"].append(rec)
            recs.append(rec)
            print(f"[bodyex] arm={arm} seed={seed}: "
                  f"r2={rec['r2']:.4f} "
                  f"agree={rec['logit_agreement']:.4f} "
                  f"(control {rec['control_logit_agreement']:.4f}) "
                  f"channel={rec['extraction_channel']}")
        if recs:
            _summarize_arm(out, arm, "sim", recs)
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def run_real(args, evidence=None):
    require_torch(EXPERIMENT_ID)
    out = artifacts.make_artifact(
        SCHEMA,
        {"attack": EXPERIMENT_ID, "mode": getattr(args, "mode", None),
         "toy": False, "arms": args.arm, "model": args.model,
         "split": args.split, "capture_dir": args.capture_dir,
         "queries": args.queries, "steps": args.steps, "lr": args.lr,
         "seq_len": args.seq_len, "device": args.device,
         "victim_jitter": args.victim_jitter,
         "seeds": args.seeds, "seed": args.seed},
        THREAT_MODEL + " Local-sim tier: the victim body is instantiated "
        "LOCALLY from --model/--split (splitbench split) and queried "
        "directly — a stand-in for the victim's serving API; the query "
        "plumbing is identical, the responses are not exfiltrated over a "
        "network.",
        interpretation=INTERPRETATION, evidence=evidence)
    trained = {}
    per_arm = {}
    for arm in args.arm:
        for seed in args.seeds:
            cell = {"experiment": EXPERIMENT_ID, "arm": arm, "seed": seed}
            try:
                if seed not in trained:
                    trained[seed] = _torch_train_student(args, seed)
                bundle = trained[seed]
                tr, ctl = bundle["trained"], bundle["control"]
                rec = {"tier": "local-sim",
                       "query_source": bundle["query_source"],
                       "body_layers": bundle["body_layers"],
                       "queries": bundle["n_query_rows"],
                       "steps": bundle["steps"], "lr": bundle["lr"],
                       "final_train_loss": bundle["final_train_loss"],
                       "r2": tr["r2"], "cosine": tr["cosine"],
                       "logit_agreement": tr["logit_agreement"],
                       "element_match": tr["element_match"],
                       "control_r2": ctl["r2"],
                       "control_cosine": ctl["cosine"],
                       "control_logit_agreement":
                           ctl["logit_agreement"],
                       "control_element_match": ctl["element_match"],
                       "n_eval_rows": tr["n_eval_rows"],
                       "n_eval_elements": tr["n_eval_elements"]}
                if arm == "weightsim":
                    rec.update({
                        "functional_r2": tr["r2"],
                        "per_layer_output_drift":
                            bundle["per_layer_output_drift"],
                        "control_per_layer_output_drift":
                            bundle["control_per_layer_output_drift"],
                        "per_layer_weight_drift":
                            bundle["per_layer_weight_drift"],
                        "note": "per-layer output drift is the "
                                "extraction-quality proxy; weight drift "
                                "is simulation ground truth — functional "
                                "equivalence is the extraction, weight "
                                "recovery is NOT claimed"})
            except Exception as e:
                journal_error(args.output, EXPERIMENT_ID, cell, e)
                continue
            rec = {**cell, **rec}
            _gate_cell(rec, n_seeds=len(args.seeds))
            if arm == "weightsim":
                rec["functional_equivalence"] = \
                    rec["extraction_channel"] == "open"
            artifacts.append_jsonl(args.output, rec)
            out["results"].append(rec)
            per_arm.setdefault(arm, []).append(rec)
            print(f"[bodyex] arm={arm} seed={seed}: "
                  f"r2={rec['r2']:.4f} "
                  f"agree={rec['logit_agreement']:.4f} "
                  f"(control {rec['control_logit_agreement']:.4f}) "
                  f"channel={rec['extraction_channel']}")
    for arm, recs in sorted(per_arm.items()):
        _summarize_arm(out, arm, "local-sim", recs)
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def _run(args, evidence=None):
    if args.toy:
        return run_toy(args, evidence=evidence)
    if args.model:
        return run_real(args, evidence=evidence)
    raise SystemExit(f"[{EXPERIMENT_ID}] pick a tier: --toy (torch-free "
                     "closed-form machinery check) or --model M [--split "
                     "S] [--capture-dir D] (local victim sim; responses "
                     "always come from the local sim — the wire schema "
                     "records what the body RECEIVES, never its outputs)")


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

    print("body-extraction body-slice distillation attack (pure python):")

    sol = _gauss_solve([[2.0, 1.0], [1.0, 3.0]], [[1.0], [2.0]])
    check("Gauss-Jordan solve matches the hand solution",
          abs(sol[0][0] - 0.2) < 1e-9 and abs(sol[1][0] - 0.6) < 1e-9)

    rng = random.Random(0)
    W = [[rng.gauss(0.0, 1.0) for _ in range(8)] for _ in range(8)]
    X = [[rng.gauss(0.0, 1.0) for _ in range(8)] for _ in range(64)]
    W_hat = _ridge_fit(X, _matmul(X, W), 1e-8)
    check("closed-form ridge fit recovers a planted linear map",
          _r2(_matmul(X, W_hat), _matmul(X, W)) > 0.999)

    w1 = toy_world(7, 12, 10)
    check("toy world is deterministic",
          toy_world(7, 12, 10) == w1 and toy_world(8, 12, 10) != w1)

    args = build_parser().parse_args(
        ["--hidden", "24", "--queries", "96", "--eval-rows", "128",
         "--vocab", "12"])
    cell = toy_distill_cell(args, 0)
    check("toy distill: planted closed-form fit is functionally exact",
          cell["r2"] > 0.99 and cell["logit_agreement"] == 1.0
          and cell["element_match"] > 0.99)
    check("toy distill: the untrained control sits at chance",
          cell["control_logit_agreement"] <= 0.25
          and cell["control_r2"] < 0.3
          and cell["control_element_match"] < 0.2)
    g = gate.verdict(cell["element_match"], cell["control_element_match"],
                     cell["n_eval_elements"])
    check("toy distill: planted match FAILs the gate (channel open)",
          g["verdict"] == "FAIL")
    check("toy distill: recovery at control is at the floor and PASSes",
          gate.verdict(cell["control_element_match"],
                       cell["control_element_match"],
                       cell["n_eval_elements"])["verdict"] == "INSUFFICIENT_EVIDENCE")

    args_small = build_parser().parse_args(
        ["--hidden", "24", "--queries", "96", "--eval-rows", "64",
         "--vocab", "12", "--student-dim", "12", "--steps", "15"])
    cell_als = toy_distill_cell(args_small, 0)
    check("rank-limited ALS student beats the untrained control",
          cell_als["r2"] > cell_als["control_r2"]
          and cell_als["student"] == "rank-12-als")

    wcell = toy_weightsim_cell(args, 0)
    check("toy weightsim: functional extraction succeeds (R2 > 0.99)",
          wcell["functional_r2"] > 0.99
          and wcell["logit_agreement"] == 1.0)
    check("toy weightsim: final-layer output drift vanishes",
          wcell["per_layer_output_drift"][-1] < 0.01)
    check("toy weightsim: per-layer weights are NOT recovered "
          "(non-identifiability demonstrated)",
          wcell["per_layer_weight_drift"][0] > 0.2
          and wcell["per_layer_weight_drift"][1] > 0.2)
    check("toy weightsim: control stack is functionally far",
          wcell["control_r2"] < 0.3
          and wcell["control_final_output_drift"] > 0.5)

    check("build_parser works torch-less and defaults are sane",
          build_parser().parse_args([]).split == "6")
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
