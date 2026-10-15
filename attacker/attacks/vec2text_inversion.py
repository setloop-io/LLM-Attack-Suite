#!/usr/bin/env python3
"""vec2text-inversion (training + inference modes) — LEARNED embedding
inversion in the vec2text line (Morris et al., "Text Embeddings Reveal
(Almost) As Much As Text", arXiv:2310.06816), transplanted to the
split-LLM seam. A small conditional language model is TRAINED to generate
the victim's text from captured seam rows, instead of decoding each
position independently against the embedding table (seam-inversion's NN
decode). Generation couples the positions, so the attack exploits both
the seam geometry and the statistics of text — the vec2text edge over
per-position decoding.

Threat model (mirrored in every artifact): identical to seam-inversion —
honest-but-curious observer of fwd seam activations h*, granted the
victim's PUBLIC base weights (--model: embedding E + public head, split
at --split) for TRAINING forwards only. The inverter is trained on PUBLIC
text (--train-file, or synthetic uniform draws when no public text is
assumed) pushed through the PUBLIC head; the victim's finetuned head is
never assumed and any mismatch is part of the honest measurement.
Canonical tokens are evaluation truth, never an attack input (label-free
attack). The wire surface itself reveals the sequence LENGTH (the number
of seam rows), so the inverter is trained and run at that length — no
private information.

CLI (framework --mode stays the training/inference wire surface; the
vec2text stage is --phase — the dispatcher owns args.mode and would
overwrite a second --mode flag):

    python -m attacker --mode training --attack vec2text-inversion --help
    # train (torch tier): public text/draws -> public head -> seam rows;
    # fit the inverter to reconstruct the public tokens from the rows
    $PY -m attacker --mode training --attack vec2text-inversion \
        --phase train --model gpt2 --split 6 --seq-len 128 \
        --train-file public.txt --train-steps 3000 --corrections 1 \
        --inverter-pt inv.pt --output train.json --evidence
    # attack: seam captures + canonical -> generated text -> scoring
    $PY -m attacker --mode training --attack vec2text-inversion \
        --phase attack --capture-dir CAPS --canonical-pt CAPS/canonical.pt \
        --inverter-pt inv.pt --corrections 1 --samples 8 \
        --model gpt2 --output v2t.json --evidence
    # torch-free toy tier
    python -m attacker --mode training --attack vec2text-inversion \
        --toy --quick --output /tmp/v2t.json

Architecture (torch tier, deliberately small and fast): a compact
transformer encoder-decoder (defaults 2+2 layers, d_model=128, ff=512,
nhead=4) trained from scratch. The encoder consumes the projected seam
rows; the decoder generates exactly seq_len tokens (BOS-started, greedy).
--corrections N implements the vec2text two-step: the SAME conditional
net doubles as the correction model — a hypothesis joins the encoder
memory as a second typed token segment, and attack-time decoding
iterates hyp <- Inverter(h, hyp) N times. vec2text trains a dedicated
corrector once and reuses it across iterations; we train one shared
conditional net (base mode + correction mode) and likewise reuse it —
documented simplification: one network, two call patterns, iterated.

Training (torch tier): phase 1 trains the base inverter with teacher
forcing; when --corrections >= 1 the last third of --train-steps is a
correction phase on the net's own greedy hypotheses (refreshed every 4
steps; base loss kept so the base mode is not forgotten). A few thousand
steps of this tiny net is minutes on a GPU node. The checkpoint
(llm_attacker.vec2text_inverter.v1) carries the arch spec, the
attacker-side majority-control token (computed from the PUBLIC corpus —
canonical is never consulted), and a corrector_trained flag; attack with
--corrections >= 1 on a base-only checkpoint is a descriptive error.

Scoring (per cell + pooled summary): sequence-level token accuracy
(position-aligned vs canonical), smoothed corpus BLEU-4 (add-one
smoothing on n >= 2 — tiny-sample safe), orderless token-set P/R
(VFLAIR convention), a majority-token control (attacker-computable from
the public corpus, stored in the checkpoint) and the frozen-gate verdict
(attacker.gate.verdict, n = scored tokens). With --model/--embedding-pt
at attack time the artifact ALSO carries the position-independent
identity-NN baseline (seam-inversion's weakest variant) under
"baseline_nn" for a head-to-head against the learned decode.

Toy tier (torch-free, honest): synthetic vocab 60, d=16 world whose seam
applies a POSITION-DEPENDENT linear map h = E[tok] @ A_p + noise — the
toy analogue of a position-conditioned head (positional encodings make
real heads position-dependent). The learned toy inverter is a
nearest-centroid classifier trained on a public split (per-position
centroids, position-pooled fallback): it estimates the actual
position-conditioned seam geometry from public (h, tok) pairs alone. The
baseline is position-independent cosine-NN over E (one global geometry —
seam-inversion's decode). Why learned wins here: random per-position
maps are non-orthogonal across positions, so a single global table
mis-scores rows whose geometry it cannot undo, while per-position
centroids absorb each A_p from public data. The toy text is uniform (no
bigram structure) ON PURPOSE: the toy win is purely geometric; the
LM-statistics edge is the torch tier's contribution, not smuggled into
the toy. Toy tier validates decode/scoring/gate plumbing and the
learned > NN-baseline ordering; real numbers come from --capture-dir
runs.
"""

import argparse
import collections
import math
import random
import time

try:
    import torch
except ImportError:
    torch = None

from .. import artifacts, gate
from .common import add_common_args, journal_error, require_torch
from .seam_inversion import (_matmul, _randn_mat, cosine_nn_decode,
                             majority_control_token, score_sequences)

EXPERIMENT_ID = "vec2text_inversion"
MODES = ("training", "inference")
REQUIRES_LABELS = False
DESCRIPTION = ("vec2text-line LEARNED seam inversion: a small transformer "
               "encoder-decoder trained on public text (through the public "
               "head) GENERATES victim text from seam rows, with optional "
               "iterative correction rounds; strictly stronger than "
               "seam-inversion's per-position NN decode. Seq-accuracy + "
               "smoothed BLEU-4 + token-set P/R + majority-control "
               "frozen-gate verdicts")

SCHEMA = "dtraining.attacker.vec2text_inversion.v1"
CKPT_FORMAT = "llm_attacker.vec2text_inverter.v1"
BLEU_MAX_N = 4

THREAT_MODEL = (
    "honest-but-curious seam observer of fwd seam activations h* (training: "
    "the tensor the body receives; inference: prefill seam rows), granted "
    "the victim's PUBLIC base weights (--model: embedding E + public head "
    "at --split) for training forwards on public text only — the victim's "
    "finetuned head is never assumed (mismatch is part of the "
    "measurement). Canonical tokens are evaluation truth, never an attack "
    "input (label-free attack). The wire surface reveals the sequence "
    "length (row count), so the inverter is trained/run at that length — "
    "no private information.")

INTERPRETATION = (
    "seq_accuracy is position-aligned vs canonical, pooled per token; "
    "bleu is smoothed corpus BLEU-4 (add-one smoothing on n>=2); "
    "token-set precision/recall are orderless per-sequence set overlaps "
    "(VFLAIR convention); the control decodes the public-corpus majority "
    "token at every position (attacker-computable; stored in the "
    "checkpoint at train time — canonical is never consulted); the gate "
    "verdict is the frozen-gate Bonferroni-Wilson excess over that "
    "control with n = scored tokens. baseline_nn (when --model/"
    "--embedding-pt is given at attack time) is the position-independent "
    "identity-NN decode over E — seam-inversion's weakest variant — the "
    "head-to-head the learned inverter must beat. Toy tier is a machinery "
    "check of the shared decode/scoring/gate plumbing on a synthetic "
    "position-dependent seam (nearest-centroid learned inverter vs "
    "position-independent NN), not a security claim; real numbers come "
    "from --capture-dir runs.")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, hidden=16)
    ap.add_argument("--phase", choices=("train", "attack"), default="attack",
                    help="vec2text stage: 'train' fits an inverter on "
                         "public text (torch tier) and saves --inverter-pt; "
                         "'attack' (default) loads --inverter-pt and scores "
                         "--capture-dir against --canonical-pt. The "
                         "framework's --mode (training/inference wire "
                         "surface) is separate and unchanged")
    ap.add_argument("--capture-dir", default=None,
                    help="wire-capture directory (attack phase; fwd seam "
                         "activations in training mode, prefill rows in "
                         "inference mode — same conventions as "
                         "seam-inversion)")
    ap.add_argument("--canonical-pt", default=None,
                    help="canonical.pt ground-truth input_ids, LongTensor "
                         "[n_captured*batch, seq] row-aligned to the fwd "
                         "captures step-major (splitbench --save-canonical)")
    ap.add_argument("--model", default=None,
                    help="HF name/path of the PUBLIC base model: train "
                         "phase supplies E + the public head for training "
                         "forwards; attack phase it is OPTIONAL and only "
                         "adds the identity-NN baseline over E")
    ap.add_argument("--embedding-pt", default=None,
                    help="attack-phase baseline alternative to --model: a "
                         ".pt with the embedding matrix E ([V, d] tensor "
                         "or a state dict carrying wte/embed_tokens)")
    ap.add_argument("--split", default="6",
                    help="seam boundary of the PUBLIC head used for "
                         "training forwards (must match the capture "
                         "harness split; default '6')")
    ap.add_argument("--inverter-pt", default=None,
                    help="vec2text inverter checkpoint "
                         f"({CKPT_FORMAT}): SAVED by --phase train, "
                         "LOADED by --phase attack")
    ap.add_argument("--train-file", default=None,
                    help="public text file tokenized with the --model "
                         "tokenizer for training (default: synthetic "
                         "uniform token draws — no public text assumed)")
    ap.add_argument("--train-seqs", type=int, default=2048,
                    help="public training sequences (windows of --seq-len)")
    ap.add_argument("--train-steps", type=int, default=2000,
                    help="optimizer steps; with --corrections >= 1 the "
                         "last third is the correction phase")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="training batch size (sequences)")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="AdamW lr for the inverter")
    ap.add_argument("--d-model", type=int, default=128,
                    help="inverter transformer width")
    ap.add_argument("--nhead", type=int, default=4,
                    help="inverter attention heads")
    ap.add_argument("--enc-layers", type=int, default=2,
                    help="inverter encoder layers")
    ap.add_argument("--dec-layers", type=int, default=2,
                    help="inverter decoder layers")
    ap.add_argument("--ff", type=int, default=512,
                    help="inverter feed-forward width")
    ap.add_argument("--corrections", type=int, default=0,
                    help="iterative-correction rounds (vec2text two-step). "
                         "train: >= 1 adds the correction phase; attack: "
                         "apply that many refinement rounds (requires a "
                         "checkpoint trained with --corrections >= 1)")
    ap.add_argument("--samples", type=int, default=8,
                    help="cap on scored sequences (scan order)")
    ap.add_argument("--device", default=None, choices=("cpu", "cuda"),
                    help="torch device (default: cuda when available)")
    ap.add_argument("--seq-len", type=int, default=12,
                    help="toy sequence length; torch train: public window "
                         "length (must match the capture harness seq_len)")
    ap.add_argument("--vocab", type=int, default=60, help="toy vocab size")
    ap.add_argument("--toy-seqs", type=int, default=24,
                    help="toy victim sequences per cell")
    ap.add_argument("--toy-pub-seqs", type=int, default=300,
                    help="toy public training sequences (centroid fit + "
                         "control corpus)")
    ap.add_argument("--toy-sigmas", type=float, nargs="+",
                    default=(0.0, 0.1, 0.3, 1.0, 2.0),
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
    return EvidenceLogger(args.output,
                          node_role="attacker/vec2text-inversion")


def _ngrams(tokens, n):
    return collections.Counter(tuple(tokens[i:i + n])
                               for i in range(len(tokens) - n + 1))


def corpus_bleu(preds, trues, max_n=BLEU_MAX_N):
    """Smoothed corpus BLEU (Papineni et al.) with add-one smoothing on
    n-gram orders >= 2 (Lin & Och's +1: tiny-sample safe — an unsmoothed
    BLEU-4 collapses to 0 on small cells). Unigram precision stays exact;
    brevity penalty from pooled candidate/reference lengths. Deterministic;
    a perfect match scores exactly 1.0."""
    if max_n < 1:
        raise ValueError("max_n must be >= 1")
    p_n = []
    for n in range(1, max_n + 1):
        clipped = 0
        total = 0
        for p, t in zip(preds, trues):
            pc, tc = _ngrams(p, n), _ngrams(t, n)
            total += sum(pc.values())
            clipped += sum(min(c, tc[g]) for g, c in pc.items())
        if n == 1:
            p_n.append(clipped / max(1, total))
        else:
            p_n.append((clipped + 1) / (total + 1))
    c = sum(len(p) for p in preds)
    r = sum(len(t) for t in trues)
    bp = 1.0 if c > r else math.exp(1.0 - r / max(1, c))
    return bp * math.exp(sum(math.log(max(p, 1e-12)) for p in p_n) / max_n)


def toy_world(seed, vocab=60, dim=16, seq_len=12, n_victim=24, n_pub=300,
              sigma=0.0):
    """Toy seam world with a POSITION-DEPENDENT seam map. E [vocab, dim]
    random; A_p [dim, dim] random per position (non-orthogonal across
    positions — the toy analogue of a position-conditioned head); victim
    and public sequences are uniform token draws (no LM structure on
    purpose: the toy win is geometric, see the module docstring); seam
    rows h = E[tok] @ A_p + N(0, sigma^2) per element. The public split is
    the attacker-side analogue of public text: it trains the centroids and
    sets the majority control; it never touches the victim rows."""
    rng = random.Random(seed)
    E = _randn_mat(vocab, dim, rng)
    A = [_randn_mat(dim, dim, rng) for _ in range(seq_len)]
    EA = [_matmul(E, Ap) for Ap in A]
    pub = [[rng.randrange(vocab) for _ in range(seq_len)]
           for _ in range(n_pub)]
    victim = [[rng.randrange(vocab) for _ in range(seq_len)]
              for _ in range(n_victim)]

    def rows(seqs):
        out = []
        for s in seqs:
            hs = []
            for p, tok in enumerate(s):
                base = EA[p][tok]
                hs.append([v + (rng.gauss(0.0, sigma) if sigma > 0.0
                                else 0.0) for v in base])
            out.append(hs)
        return out

    return {"E": E, "A": A, "pub": pub, "victim": victim,
            "h_pub": rows(pub), "h_victim": rows(victim),
            "vocab": vocab, "dim": dim, "seq_len": seq_len, "sigma": sigma}


def toy_train_centroids(pub, h_pub, vocab, seq_len, dim):
    """The toy LEARNED inverter: nearest-centroid classification trained
    on the public split. tables[p][v] = mean public seam row at position p
    carrying token v; a position-pooled centroid backs up (p, v) pairs the
    public split never showed (rare at the default sizes), zeros as the
    last resort (never win a cosine)."""
    sums = [[None] * vocab for _ in range(seq_len)]
    cnts = [[0] * vocab for _ in range(seq_len)]
    gsum = [None] * vocab
    gcnt = [0] * vocab
    for s, hs in zip(pub, h_pub):
        for p, tok in enumerate(s):
            row = hs[p]
            if sums[p][tok] is None:
                sums[p][tok] = list(row)
            else:
                for k in range(dim):
                    sums[p][tok][k] += row[k]
            cnts[p][tok] += 1
            if gsum[tok] is None:
                gsum[tok] = list(row)
            else:
                for k in range(dim):
                    gsum[tok][k] += row[k]
            gcnt[tok] += 1
    tables = []
    for p in range(seq_len):
        cents = []
        for v in range(vocab):
            if cnts[p][v] > 0:
                cents.append([x / cnts[p][v] for x in sums[p][v]])
            elif gcnt[v] > 0:
                cents.append([x / gcnt[v] for x in gsum[v]])
            else:
                cents.append([0.0] * dim)
        tables.append(cents)
    return tables


def toy_centroid_decode(rows, tables):
    """Nearest-centroid decode of one sequence's rows [seq, dim]: the row
    at position p is classified against position p's table (cosine);
    tables are vocab-ordered so the argmax index IS the token id."""
    return [cosine_nn_decode([row], tables[p])[0]
            for p, row in enumerate(rows)]


def toy_cell(args, sigma, seed):
    """One toy cell: learned nearest-centroid decode vs the
    position-independent NN baseline over E, both scored with the shared
    plumbing (seq-accuracy, BLEU, set P/R, control, gate)."""
    world = toy_world(args.seed + 391 * seed, vocab=args.vocab,
                      dim=args.hidden, seq_len=args.seq_len,
                      n_victim=args.toy_seqs, n_pub=args.toy_pub_seqs,
                      sigma=sigma)
    tables = toy_train_centroids(world["pub"], world["h_pub"], args.vocab,
                                 args.seq_len, args.hidden)
    learned = [toy_centroid_decode(hs, tables) for hs in world["h_victim"]]
    baseline = [cosine_nn_decode(hs, world["E"]) for hs in world["h_victim"]]
    control = majority_control_token(world["pub"])
    n_seeds = len(args.seeds)
    cell = {"sigma": sigma, "seed": seed,
            **score_sequences(learned, world["victim"], control,
                              n_seeds=n_seeds),
            "bleu": round(corpus_bleu(learned, world["victim"]), 6)}
    cell["baseline_nn"] = {
        **score_sequences(baseline, world["victim"], control,
                          n_seeds=n_seeds),
        "bleu": round(corpus_bleu(baseline, world["victim"]), 6)}
    return cell


def _run_toy(args, ev):
    out = artifacts.make_artifact(
        SCHEMA,
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "vocab": args.vocab, "dim": args.hidden, "seq_len": args.seq_len,
         "toy_seqs": args.toy_seqs, "toy_pub_seqs": args.toy_pub_seqs,
         "toy_sigmas": list(args.toy_sigmas), "seeds": args.seeds,
         "learned_inverter": "nearest-centroid (per-position, public fit)",
         "baseline": "position-independent cosine-NN over E"},
        THREAT_MODEL + " Toy tier: synthetic position-dependent seam; "
        "measures decode/scoring/gate plumbing and the learned > "
        "position-independent ordering, not a real model.",
        interpretation=INTERPRETATION, evidence=ev)
    cells = []
    for sigma in args.toy_sigmas:
        for seed in args.seeds:
            cell = {"experiment": EXPERIMENT_ID,
                    "arm": "toy-nearest-centroid",
                    "submode": "learned-vs-position-independent-nn"}
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
            print(f"[vec2text-toy] sigma={sigma} seed={seed}: "
                  f"acc={cell['seq_accuracy']} baseline_nn="
                  f"{cell['baseline_nn']['seq_accuracy']} control="
                  f"{cell['control_accuracy']} bleu={cell['bleu']} "
                  f"gate={cell['gate']['verdict']}")
    for sigma in sorted(set(c["sigma"] for c in cells)):
        vals = [c for c in cells if c["sigma"] == sigma]
        out["summary"].append({
            "experiment": EXPERIMENT_ID, "sigma": sigma,
            "seq_accuracy_mean": sum(c["seq_accuracy"] for c in vals)
            / len(vals),
            "baseline_nn_accuracy_mean":
                sum(c["baseline_nn"]["seq_accuracy"] for c in vals)
            / len(vals),
            "control_accuracy_mean": sum(c["control_accuracy"] for c in vals)
            / len(vals),
            "bleu_mean": sum(c["bleu"] for c in vals) / len(vals),
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


def _load_embedding_pt(path):
    """E from a .pt: a raw [V, d] tensor or a state dict carrying a wte /
    embed_tokens weight (attack-phase baseline only)."""
    blob = torch.load(path, map_location="cpu")
    if torch.is_tensor(blob):
        if blob.dim() == 2:
            return blob.float()
        raise SystemExit(f"[{EXPERIMENT_ID}] {path}: expected a 2-D "
                         f"[V, d] embedding tensor, got shape "
                         f"{tuple(blob.shape)}")
    if isinstance(blob, dict):
        for key in ("embedding", "wte.weight", "transformer.wte.weight",
                    "embed_tokens.weight", "model.embed_tokens.weight"):
            if key in blob and torch.is_tensor(blob[key]):
                return blob[key].float()
        for k, v in blob.items():
            if torch.is_tensor(v) and v.dim() == 2 \
                    and ("wte" in k or "embed_tokens" in k):
                return v.float()
    raise SystemExit(f"[{EXPERIMENT_ID}] {path}: cannot locate an "
                     "embedding matrix (expected a [V, d] tensor or a "
                     "state dict with wte.weight / embed_tokens.weight)")


def _inverter_class():
    """The SeamInverter nn.Module, built lazily so module import and
    --help never touch torch (repo rule)."""
    import torch.nn as nn
    import torch.nn.functional as F

    class SeamInverter(nn.Module):
        """Compact transformer encoder-decoder seam inverter — the
        vec2text base model and correction model as ONE conditional net.

        encode(h, hyp): seam rows h [b, s, H] -> linear projection -> +
        position/type embeddings -> transformer encoder. When hyp (token
        ids [b, s]) is given, its embeddings join the encoder memory as a
        second typed segment — vec2text's correction conditioning.
        forward(h, dec_in, hyp): causal decoder over the token vocabulary
        cross-attending to the memory; returns logits [b, t, vocab].
        """

        def __init__(self, vocab, seam_dim, seq_len, d_model=128, nhead=4,
                     enc_layers=2, dec_layers=2, ff=512):
            super().__init__()
            self.vocab = vocab
            self.seq_len = seq_len
            self.bos = vocab
            self.tok_emb = nn.Embedding(vocab + 1, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.type_emb = nn.Embedding(2, d_model)
            self.seam_proj = nn.Linear(seam_dim, d_model)
            enc = nn.TransformerEncoderLayer(d_model, nhead, ff,
                                             batch_first=True, dropout=0.0)
            self.encoder = nn.TransformerEncoder(enc, enc_layers)
            dec = nn.TransformerDecoderLayer(d_model, nhead, ff,
                                             batch_first=True, dropout=0.0)
            self.decoder = nn.TransformerDecoder(dec, dec_layers)
            self.out = nn.Linear(d_model, vocab)
            for emb in (self.tok_emb, self.pos_emb, self.type_emb):
                nn.init.normal_(emb.weight, std=0.02)
            nn.init.normal_(self.out.weight, std=0.02)

        def encode(self, h, hyp=None):
            b, s, _ = h.shape
            pos = torch.arange(s, device=h.device).unsqueeze(0)
            mem = self.seam_proj(F.layer_norm(h, (h.shape[-1],))) \
                + self.pos_emb(pos) + self.type_emb.weight[0]
            if hyp is not None:
                hyp_mem = self.tok_emb(hyp) + self.pos_emb(pos) \
                    + self.type_emb.weight[1]
                mem = torch.cat([mem, hyp_mem], dim=1)
            return self.encoder(mem)

        def forward(self, h, dec_in, hyp=None):
            mem = self.encode(h, hyp)
            t = dec_in.shape[1]
            pos = torch.arange(t, device=h.device).unsqueeze(0)
            tgt = self.tok_emb(dec_in) + self.pos_emb(pos)
            mask = torch.full((t, t), float("-inf"),
                              device=h.device).triu(1)
            return self.out(self.decoder(tgt, mem, tgt_mask=mask))

    return SeamInverter


def _greedy_decode(model, rows, hyp=None):
    """Greedy generation of exactly rows.shape[1] tokens per sequence —
    the seam length is wire-visible, so no stopping rule is needed."""
    b, s, _ = rows.shape
    dec = torch.full((b, 1), model.bos, dtype=torch.long,
                     device=rows.device)
    for _ in range(s):
        logits = model(rows, dec, hyp)[:, -1]
        dec = torch.cat([dec, logits.argmax(-1, keepdim=True)], dim=1)
    return dec[:, 1:]


def _public_ids(args, vocab):
    """Public training windows [train_seqs, seq_len]: --train-file public
    text through the --model tokenizer (windows recycled when the file is
    short), else synthetic uniform draws — no public text assumed, the
    honest minimal grant."""
    if args.train_file:
        if not args.model:
            raise SystemExit(f"[{EXPERIMENT_ID}] --train-file needs "
                             "--model (its tokenizer)")
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise SystemExit(f"[{EXPERIMENT_ID}] --train-file requires "
                             "transformers") from None
        tok = AutoTokenizer.from_pretrained(args.model)
        with open(args.train_file) as f:
            ids = tok(f.read(), return_tensors="pt").input_ids[0]
        n_win = ids.numel() // args.seq_len
        if n_win < 1:
            raise SystemExit(f"[{EXPERIMENT_ID}] train file too small: "
                             f"{ids.numel()} tokens < seq_len "
                             f"{args.seq_len}")
        windows = ids[:n_win * args.seq_len].view(n_win, args.seq_len)
        if n_win < args.train_seqs:
            reps = (args.train_seqs + n_win - 1) // n_win
            windows = windows.repeat((reps, 1))
        return windows[:args.train_seqs]
    g = torch.Generator().manual_seed(args.seed + 5150)
    return torch.randint(0, vocab, (args.train_seqs, args.seq_len),
                         generator=g)


def _run_train(args, ev):
    if not args.inverter_pt:
        raise SystemExit(f"[{EXPERIMENT_ID}] --phase train needs "
                         "--inverter-pt (path to SAVE the trained "
                         "inverter checkpoint)")
    if not args.model:
        raise SystemExit(f"[{EXPERIMENT_ID}] --phase train needs --model "
                         "(public base weights: E + the public head for "
                         "training forwards, split at --split)")
    if args.train_steps < 1 or args.train_seqs < 1 or args.batch_size < 1:
        raise SystemExit(f"[{EXPERIMENT_ID}] --train-steps / --train-seqs "
                         "/ --batch-size must all be >= 1")
    require_torch(EXPERIMENT_ID)
    import torch.nn.functional as F

    from ..splitbench import SplitPlan, _n_layers_of, build_split
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = _load_hf_model(args.model, device)
    E = model.get_input_embeddings().weight.detach()
    split = build_split(model, SplitPlan.parse(
        args.split, n_layers=_n_layers_of(model)))

    ids = _public_ids(args, E.shape[0])
    with torch.no_grad():
        h_pub = torch.cat([
            split.head_forward(ids[i:i + 64].to(device)).cpu()
            for i in range(0, ids.shape[0], 64)])
    n, s, H = h_pub.shape
    V = E.shape[0]
    control = majority_control_token(ids.tolist())

    Inverter = _inverter_class()
    inv = Inverter(vocab=V, seam_dim=H, seq_len=s, d_model=args.d_model,
                   nhead=args.nhead, enc_layers=args.enc_layers,
                   dec_layers=args.dec_layers, ff=args.ff).to(device)
    opt = torch.optim.AdamW(inv.parameters(), lr=args.lr)
    corr_start = (args.train_steps - max(1, args.train_steps // 3)
                  if args.corrections >= 1 else args.train_steps)
    batch_g = torch.Generator().manual_seed(args.seed + 99)
    t0 = time.time()
    loss_rec = {}
    hyp = None
    for step in range(args.train_steps):
        idx = torch.randint(0, n, (args.batch_size,), generator=batch_g)
        hb = h_pub[idx].to(device)
        idb = ids[idx].to(device)
        dec_in = torch.cat(
            [torch.full((idb.shape[0], 1), inv.bos, dtype=torch.long,
                        device=device), idb[:, :-1]], dim=1)
        loss = F.cross_entropy(inv(hb, dec_in).reshape(-1, V),
                               idb.reshape(-1))
        corr_loss = None
        if step >= corr_start:
            if hyp is None or step % 4 == 0:
                with torch.no_grad():
                    hyp = _greedy_decode(inv, hb)
            corr_loss = F.cross_entropy(inv(hb, dec_in, hyp=hyp)
                                        .reshape(-1, V), idb.reshape(-1))
            loss = loss + corr_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_rec = {"step": step, "loss": round(float(loss.item()), 6)}
        if corr_loss is not None:
            loss_rec["corr_loss"] = round(float(corr_loss.item()), 6)
        if step % 50 == 0 or step == args.train_steps - 1:
            print(f"[vec2text-train] step {step + 1}/{args.train_steps} "
                  f"loss={loss_rec['loss']}"
                  + (" (correction phase)" if step >= corr_start else ""))
    wall = round(time.time() - t0, 3)

    arch = {"vocab": V, "seam_dim": H, "seq_len": s,
            "d_model": args.d_model, "nhead": args.nhead,
            "enc_layers": args.enc_layers, "dec_layers": args.dec_layers,
            "ff": args.ff}
    ckpt = {"format": CKPT_FORMAT, "arch": arch,
            "state_dict": {k: v.detach().cpu()
                           for k, v in inv.state_dict().items()},
            "control_token": control,
            "corrector_trained": bool(args.corrections >= 1),
            "train": {"model": args.model, "split": args.split,
                      "train_file": args.train_file,
                      "public_source": ("train-file" if args.train_file
                                        else "synthetic-uniform"),
                      "train_seqs": n, "train_steps": args.train_steps,
                      "corrections": args.corrections,
                      "final_loss": loss_rec, "wall_s": wall}}
    torch.save(ckpt, args.inverter_pt)

    rec = {"experiment": EXPERIMENT_ID, "phase": "train",
           "inverter_pt": args.inverter_pt, "arch": arch,
           "control_token": control,
           "corrector_trained": ckpt["corrector_trained"],
           "n_public_seqs": n, "seq_len": s, "seam_dim": H, "vocab": V,
           "final_loss": loss_rec, "wall_s": wall}
    out = artifacts.make_artifact(
        SCHEMA,
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": False,
         "phase": "train", "model": args.model, "split": args.split,
         "inverter_pt": args.inverter_pt, "train_file": args.train_file,
         "train_seqs": args.train_seqs, "train_steps": args.train_steps,
         "batch_size": args.batch_size, "lr": args.lr,
         "corrections": args.corrections, "device": device, **arch},
        THREAT_MODEL + " Train phase: the inverter is fit on PUBLIC text "
        "through the PUBLIC head; the victim is never queried.",
        interpretation=INTERPRETATION, evidence=ev)
    out["results"].append(rec)
    out["summary"].append(dict(rec))
    artifacts.append_jsonl(args.output, rec)
    if ev is not None:
        ev.event("cell_result", rec)
    artifacts.write_artifact(args.output, out, evidence=ev)
    print(f"[vec2text-train] {args.train_steps} steps in {wall}s; "
          f"inverter -> {args.inverter_pt} "
          f"(corrector_trained={ckpt['corrector_trained']})")
    return 0


def _load_scored_sequences(args):
    """Pick up to --samples scored sequences, row-aligned to canonical —
    the same conventions as seam-inversion. Training mode reads
    phase='fwd' captures; inference mode scores phase='prefill' rows.
    Returns (picked, seq_len, h_dim)."""
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


def _run_attack(args, ev):
    if not args.inverter_pt:
        raise SystemExit(f"[{EXPERIMENT_ID}] --phase attack needs "
                         "--inverter-pt (a trained inverter checkpoint "
                         "from --phase train)")
    require_torch(EXPERIMENT_ID)
    from .seam_inversion import _cosine_nn_torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.inverter_pt, map_location="cpu")
    if not isinstance(ckpt, dict) or ckpt.get("format") != CKPT_FORMAT:
        raise SystemExit(f"[{EXPERIMENT_ID}] {args.inverter_pt} is not a "
                         f"{CKPT_FORMAT} inverter checkpoint "
                         "(--phase train writes one)")
    arch = ckpt["arch"]
    Inverter = _inverter_class()
    inv = Inverter(**arch)
    inv.load_state_dict(ckpt["state_dict"])
    inv = inv.to(device).eval()
    if args.corrections >= 1 and not ckpt.get("corrector_trained"):
        raise SystemExit(f"[{EXPERIMENT_ID}] --corrections >= 1 but "
                         f"{args.inverter_pt} was trained without a "
                         "correction phase; retrain with --phase train "
                         "--corrections >= 1")
    control = ckpt.get("control_token")
    if control is None:
        raise SystemExit(f"[{EXPERIMENT_ID}] {args.inverter_pt} lacks the "
                         "attacker-side control token; retrain with this "
                         "framework's --phase train (the control is the "
                         "public-corpus majority token)")
    picked, seq_len, h_dim = _load_scored_sequences(args)
    if seq_len != arch["seq_len"]:
        raise SystemExit(f"[{EXPERIMENT_ID}] capture seq_len {seq_len} != "
                         f"the inverter's trained seq_len "
                         f"{arch['seq_len']}: retrain with --seq-len "
                         "matching the capture harness")
    if h_dim != arch["seam_dim"]:
        raise SystemExit(f"[{EXPERIMENT_ID}] capture seam dim {h_dim} != "
                         f"the inverter's seam_dim {arch['seam_dim']}: "
                         "the inverter must be trained at the capture "
                         "harness's --split")

    E = None
    baseline_note = "omitted (pass --model or --embedding-pt to add it)"
    if args.embedding_pt or args.model:
        E = _load_embedding_pt(args.embedding_pt) if args.embedding_pt \
            else _load_hf_model(args.model, device) \
            .get_input_embeddings().weight.detach().cpu()
        if E.shape[1] != h_dim:
            baseline_note = (f"omitted (dim(E) {E.shape[1]} != seam dim "
                             f"{h_dim})")
            E = None
        else:
            baseline_note = "identity-nn over E"
            E = E.to(device)

    rows = torch.stack([p["h"] for p in picked]).to(device)
    with torch.no_grad():
        pred = _greedy_decode(inv, rows)
        for _ in range(args.corrections):
            pred = _greedy_decode(inv, rows, hyp=pred)
        base_pred = None
        if E is not None:
            base_pred = _cosine_nn_torch(rows.reshape(-1, h_dim),
                                         E).reshape(len(picked), seq_len)

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
         "phase": "attack", "capture_dir": args.capture_dir,
         "canonical_pt": args.canonical_pt,
         "inverter_pt": args.inverter_pt, "arch": arch,
         "corrections": args.corrections, "samples": args.samples,
         "baseline": baseline_note, "device": device},
        THREAT_MODEL, interpretation=INTERPRETATION, evidence=ev)

    cells = []
    for key in order:
        idx = groups[key]
        cell = {"experiment": EXPERIMENT_ID, "arm": "vec2text",
                "submode": ("generate+correct" if args.corrections >= 1
                            else "generate"), "mode": args.mode,
                "step": key[0], "epoch": key[1],
                "corrections": args.corrections}
        try:
            preds = pred[idx].cpu().tolist()
            trues = [picked[i]["true"] for i in idx]
            cell.update(score_sequences(preds, trues, control))
            cell["bleu"] = round(corpus_bleu(preds, trues), 6)
            if base_pred is not None:
                bp = base_pred[idx].cpu().tolist()
                cell["baseline_nn"] = {
                    **score_sequences(bp, trues, control),
                    "bleu": round(corpus_bleu(bp, trues), 6)}
        except Exception as e:
            journal_error(args.output, EXPERIMENT_ID, cell, e)
            continue
        artifacts.append_jsonl(args.output, cell)
        if ev is not None:
            ev.event("cell_result", cell)
        out["results"].append(cell)
        cells.append(cell)
        line = (f"[vec2text-inversion] step={key[0]} corr="
                f"{args.corrections}: acc={cell['seq_accuracy']} control="
                f"{cell['control_accuracy']} bleu={cell['bleu']}")
        if "baseline_nn" in cell:
            line += f" baseline_nn={cell['baseline_nn']['seq_accuracy']}"
        print(line + f" gate={cell['gate']['verdict']}")

    if cells:
        n_corr = sum(c["n_correct"] for c in cells)
        n_tok = sum(c["n_tokens"] for c in cells)
        c_corr = sum(c["n_control_correct"] for c in cells)
        pooled_acc = n_corr / max(1, n_tok)
        pooled_ctl = c_corr / max(1, n_tok)
        all_preds = [pred[i].cpu().tolist() for i in range(len(picked))]
        all_trues = [p["true"] for p in picked]
        summary = {
            "experiment": EXPERIMENT_ID, "arm": "vec2text",
            "corrections": args.corrections,
            "n_cells": len(cells), "n_tokens": n_tok,
            "token_set_precision_mean":
                sum(c["token_set_precision"] for c in cells) / len(cells),
            "token_set_recall_mean":
                sum(c["token_set_recall"] for c in cells) / len(cells),
            "seq_accuracy": round(pooled_acc, 6),
            "control_accuracy": round(pooled_ctl, 6),
            "bleu": round(corpus_bleu(all_preds, all_trues), 6),
            "gate": gate.verdict(pooled_acc, pooled_ctl, n_tok),
            "gate_verdicts": [c["gate"]["verdict"] for c in cells]}
        if base_pred is not None:
            bp_all = base_pred.cpu().tolist()
            b_corr = sum(c["baseline_nn"]["n_correct"] for c in cells)
            summary["baseline_nn"] = {
                "seq_accuracy": round(b_corr / max(1, n_tok), 6),
                "bleu": round(corpus_bleu(bp_all, all_trues), 6)}
        out["summary"].append(summary)
    artifacts.write_artifact(args.output, out, evidence=ev)
    return 0


def run(args):
    ev = _make_evidence(args)
    if args.toy:
        if args.quick:
            args.seeds = [0]
            args.toy_sigmas = (0.0, 0.5, 2.0)
            args.toy_seqs = min(args.toy_seqs, 16)
            args.toy_pub_seqs = min(args.toy_pub_seqs, 200)
        return _run_toy(args, ev)
    if args.phase == "train":
        return _run_train(args, ev)
    if args.samples < 1:
        raise SystemExit(f"[{EXPERIMENT_ID}] --samples must be >= 1")
    if args.corrections < 0:
        raise SystemExit(f"[{EXPERIMENT_ID}] --corrections must be >= 0")
    if not args.capture_dir:
        raise SystemExit(f"[{EXPERIMENT_ID}] pass --toy, --phase train, or "
                         "--capture-dir D --canonical-pt C --inverter-pt F "
                         "for the attack phase.")
    return _run_attack(args, ev)


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import json
    import os
    import tempfile

    print("vec2text-inversion toy core (pure python):")

    check("smoothed BLEU: perfect match scores 1.0",
          abs(corpus_bleu([[1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5]]) - 1.0)
          < 1e-12)
    disjoint = corpus_bleu([[1, 1, 1, 1]], [[2, 3, 4, 5]])
    partial = corpus_bleu([[1, 2, 9, 9]], [[1, 2, 3, 4]])
    check("smoothed BLEU orders disjoint < partial < perfect",
          0.0 <= disjoint < partial < 1.0)

    args = build_parser().parse_args(["--toy"])
    args.mode = "training"
    cell0 = toy_cell(args, 0.0, 0)
    check("noiseless toy: learned centroid decode recovers >85%",
          cell0["seq_accuracy"] > 0.85)
    check("noiseless toy: learned beats the NN baseline by >20pp",
          cell0["seq_accuracy"]
          > cell0["baseline_nn"]["seq_accuracy"] + 0.2)
    check("toy cell carries control + gate + bleu + baseline fields",
          all(k in cell0 for k in
              ("control_accuracy", "gate", "bleu", "baseline_nn"))
          and cell0["gate"]["verdict"] in ("PASS", "FAIL")
          and "gate" in cell0["baseline_nn"])
    noisy = toy_cell(args, 2.0, 0)
    check("heavy noise degrades the learned decode (graceful)",
          noisy["seq_accuracy"] < cell0["seq_accuracy"])

    check("parser builds torch-less",
          build_parser().parse_args([]).phase == "attack")
    ns = build_parser().parse_args([])
    ns.mode = "training"
    try:
        run(ns)
        check("no --toy/--capture-dir is a descriptive error", False)
    except SystemExit as e:
        check("no --toy/--capture-dir is a descriptive error",
              "--capture-dir" in str(e))
    ns = build_parser().parse_args(["--phase", "train"])
    ns.mode = "training"
    try:
        run(ns)
        check("train without --inverter-pt is a descriptive error", False)
    except SystemExit as e:
        check("train without --inverter-pt is a descriptive error",
              "--inverter-pt" in str(e))
    ns = build_parser().parse_args(
        ["--capture-dir", "/nonexistent", "--canonical-pt", "/nonexistent"])
    ns.mode = "training"
    try:
        run(ns)
        check("attack without --inverter-pt is a descriptive error", False)
    except SystemExit as e:
        check("attack without --inverter-pt is a descriptive error",
              "--inverter-pt" in str(e))
    if torch is None:
        ns = build_parser().parse_args(
            ["--capture-dir", "/nonexistent", "--canonical-pt",
             "/nonexistent", "--inverter-pt", "/nonexistent"])
        ns.mode = "training"
        try:
            run(ns)
            check("real tier is torch-guarded", False)
        except SystemExit as e:
            check("real tier is torch-guarded", "requires torch" in str(e))
        ns = build_parser().parse_args(
            ["--phase", "train", "--inverter-pt", "/nonexistent",
             "--model", "/nonexistent"])
        ns.mode = "training"
        try:
            run(ns)
            check("train tier is torch-guarded", False)
        except SystemExit as e:
            check("train tier is torch-guarded", "requires torch" in str(e))

    with tempfile.TemporaryDirectory() as d:
        ns = build_parser().parse_args(
            ["--toy", "--quick", "--evidence",
             "--output", os.path.join(d, "v2t.json")])
        ns.mode = "training"
        check("toy run end to end torch-free", run(ns) == 0)
        art = json.load(open(os.path.join(d, "v2t.json")))
        check("toy artifact has sigma cells with gate verdicts",
              len(art["results"]) == 3
              and all(c["gate"]["verdict"] in ("PASS", "FAIL")
                      for c in art["results"]))
        check("toy cells carry the learned-vs-baseline pair",
              all("baseline_nn" in c and "bleu" in c
                  for c in art["results"]))
        check("journal lines match result cells",
              len(open(os.path.join(d, "v2t.json.jsonl")).read()
                  .splitlines()) == len(art["results"]))
        from ..evidence import verify_evidence
        good, problems = verify_evidence(
            os.path.join(d, "v2t.json.evidence.json"))
        check("evidence chain verifies", good and problems == [])

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
