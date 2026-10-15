#!/usr/bin/env python3
"""bisr-baseline (training + inference modes) — a scoped, increasingly
faithful adaptation of BiSR (Bidirectional Semi-white-box Reconstruction,
Chen et al., CCS 2024, arXiv:2409.00960) transplanted to the split-LLM
seam. Phase-A fidelity status (WS3): the three previously missing pieces
are now implemented IN DESIGN, clean-room from the paper (the SplitLLM
reference repo is Apache-2.0 but no code is copied); what remains
pilot-scale is enumerated at the end of this docstring.

Prospective repair (2026-09-17): --decoder gru selects the single-layer
256-unit GRU with output dropout 0.1; the legacy MLP remains the default.
BRE starts from raw SIP predictions. Exact grants use a hash-bound client
head with a separate public downstream model. For label-isolated prediction
use experiments/bisr_fidelity_predict.py; this legacy entry point also scores.
These additions do not establish published-setting fidelity.

  * SIP (paper SS3.1, learning-based stage): a trainable per-position
    decoder (layer-normed seam row + position embedding -> vocabulary
    logits) TRAINED on auxiliary text pushed through the PUBLIC bottom
    segment (the paper's frozen encoder e = f_btm-bar, semi-white-box);
    the paper's default decoder is a single-layer GRU (hidden 256) — the
    pilot decoder is a small MLP (same auto-encoder objective, CE against
    the original token ids).
  * BRE optimization arms (paper SS3.2 — the core attack): after the SIP
    initial reconstruction, --opt-arm refines it by direct optimization:
      - 'b'  backward gradient matching (SS3.2.2, Eq. 9): dummy soft
             labels y' = softmax(z/tau) are optimized so the MIMIC TOP's
             dummy gradient grad_{h*}(CE(rest_forward(h*), y')) matches
             the OBSERVED bwd seam gradient under the TAG-style combined
             distance beta*||.||_2 + (1-beta)*||.||_1; h* is the observed
             fwd seam (the paper's known-forward-activation trait), never
             optimized. Auto-regressive alignment: y'_t recovers input
             token t+1; position 0 has no label gradient and stays with
             the SIP decode (paper SS3.2.2). y' is initialized from the
             SIP output softmax, exactly as the paper prescribes.
      - 'f'  forward smashed-data matching (SS3.2.1, Eq. 7-8): a dummy
             embedding e' (continuous relaxation, init from the current
             reconstruction's tokens through emb()) is optimized so the
             PUBLIC bottom blocks above the embedding layer reproduce the
             observed seam, COSINE distance (paper's choice over L2,
             Appendix D.5); decode = per-position Euclidean nearest
             embedding row (Eq. 8).
      - 'bf' the paper's BiSR(b+f): GM first, then SM warm-started from
             the GM reconstruction. Defaults from Table 1 (GPT2-large):
             AdamW(0.9, 0.999); GM 32 iters lr 0.09 beta 0.85 tau 1.2;
             SM 800 iters lr 0.01 wd 0.01. Adaptation note: the paper
             observes x~_btm (fwd cut) and grad(x~_trk) (bwd cut) at two
             different cut layers of a 3-part split; this repository's
             harness captures BOTH directions at ONE seam, so the mimic
             top is the public segment downstream of that seam and the
             mimic bottom the segment upstream of it.
  * NaMoE (paper SS3.3, Fig. 6): a Noise-adaptive Mixture-of-Experts
    inversion model -- K experts, each noise-aware-trained on auxiliary
    seam rows perturbed at its own scale s_k (the paper's stage 1, gating
    frozen), then a gating network trained on RANDOM scales with the
    experts frozen (stage 2); expert features are combined by the gating
    weights and passed through a shared output layer (Fig. 6 left). The
    paper perturbs with embedding-d_XP / smashed-data-DP mechanisms; this
    adaptation uses additive Gaussian seam noise (this repository's
    defense-noise family) as the perturbation mechanism N_s. NaMoE feeds
    the SAME BRE pipeline (its mixture scores initialize y' and e'),
    matching the paper's usage (Fig. 9). Enabled with --namoe at train
    time; the attack phase auto-detects a NaMoE checkpoint.
  * bidirectional candidate SELECTION + forward-consistency beam
    verification (the pre-existing pilot arm): forward proposals from the
    learned inverter's scores UNION backward proposals from the bwd seam
    gradient's per-token direction signal (position zero forward-only),
    then beam_reconstruct through the public head. This is the module's
    beam-based stand-in for the paper's pipeline; the BRE arms above are
    the paper's actual second stage.
  * article-hash disjointness of the auxiliary corpus, asserted
    fail-closed against the frozen validation corpus manifest
    (calibration/train/evaluation partitions) and the pilot corpus
    (victim/eval documents).

Threat model (mirrored in every artifact): honest-but-curious observer of
the fwd seam activations h* AND the bwd seam gradients dL/dh* (training
mode; inference mode has fwd rows only — --opt-arm f is allowed, the
GM arm is refused), granted the victim's PUBLIC base weights (--model:
embedding E + public segments at --split) for auxiliary training
forwards, beam verification and the BRE mimic networks — BiSR/SIP's
semi-white-box grant. The victim's finetuned weights are never assumed
(the fine-tuning gap is part of the measurement). Canonical tokens are
evaluation truth, never an attack input (label-free attack).

CLI:

    python -m attacker --mode training --attack bisr-baseline --help
    # train (torch tier): auxiliary text -> public head -> seam rows; fit
    # the SIP inverter (disjointness asserted when --aux-file given);
    # --aux-file streams through a resume-safe shard cache (aux scale)
    $PY -m attacker --mode training --attack bisr-baseline \
        --phase train --model gpt2 --split 6 --seq-len 32 \
        --aux-file aux.jsonl --train-steps 500 \
        --inverter-pt bisr.pt --output train.json --evidence
    # train the NaMoE inverter (two-stage noise-aware, --namoe-sigmas)
    $PY -m attacker --mode training --attack bisr-baseline \
        --phase train --model gpt2 --split 6 --seq-len 32 \
        --aux-file aux.jsonl --train-steps 500 --namoe \
        --inverter-pt bisr-namoe.pt --output train.json --evidence
    # attack: seam captures + canonical -> bidirectional beam + BRE
    # optimization enhancement (paper's BiSR(b+f)) -> scoring
    $PY -m attacker --mode training --attack bisr-baseline \
        --phase attack --capture-dir CAPS --canonical-pt CAPS/canonical.pt \
        --inverter-pt bisr.pt --model gpt2 --split 6 \
        --opt-arm bf --samples 8 --output bisr.json --evidence
    # torch-free toy tier (add --toy-opt for the toy BRE/NaMoE arms)
    python -m attacker --mode training --attack bisr-baseline \
        --toy --quick --output /tmp/bisr.json

Auxiliary-data separation (fail closed): --aux-file is a JSONL of
{"id": ..., "text": ...} articles. Before any training, every article id
is checked against (a) the frozen validation corpus manifest
experiments/validation_corpus_manifest.json (all selected partitions) and
(b) the pilot corpus results/pilot-gpt2-wikitext2/inputs/pilot-corpus.jsonl
(the victim/calibration documents). ANY overlap is a descriptive error;
references that do not exist on disk are recorded as skipped, never
silently passed. With no --aux-file the inverter trains on synthetic
uniform token draws — the documented minimal grant, disjoint by
construction.

Aux-scale training (WS3 Phase-A): --aux-file is STREAMED (bounded memory:
articles tokenized in batches, windows cut incrementally) into a
resume-safe shard cache (<inverter-pt>.auxcache/ or --aux-cache): each
shard holds --aux-chunk-seqs windows of token ids + their public-head
seam rows; journal.json fingerprints the run config (aux sha256, model,
split, seq-len, aux-seqs, chunk size) and records completed shards, so an
interrupted train resumes after the last sealed shard; a fingerprint
mismatch is a descriptive error, never silent reuse. The training loop
samples batches from the shards via memory-mapped reads, so the aux
corpus is capped by DISK, not RAM. Measured host ceilings (this host,
CPU, 2026-09-16; gpt2 split 6, seq_len 32, float32 shards): tokenizer
1.12M tokens/s; public-head forward 333 windows/s (end-to-end streamed
production 281 windows/s incl. tokenization + shard writes); SipInverter
training (batch 32, d_model 128, vocab 50257, memmap sampling) 11.7
steps/s; shard cost 96.1 KiB/window. Practical ceilings: 75k windows
(≈ WikiText-2 train, 2.4M tokens) in ~5 min production + 7.2 GB disk;
1M windows (32M tokens) in ~1 h + 96 GB; 20k training steps ≈ 28 min.
The overnight-CPU ceiling is therefore DISK-bound at ~1-2M windows —
two orders above the pilot default (--aux-seqs 512) and comfortably
above the paper's aux-corpus scale (SensiReplaced-sized) for the Phase-B
runbook; larger supplies (disjoint C4/Pile slice per the WS3 risk note)
fit the same path.

Toy tier (torch-free, honest): synthetic vocab 60, d=16 world with a
POSITION-DEPENDENT seam (h = E[tok] @ A_p + noise) and a toy backward
channel (g_p = -E[label_{p+1}] @ B_p + noise, the toy TAG direction
signal). The toy SIP inverter is a per-position nearest-centroid
classifier fit on the auxiliary split; the bidirectional arm unions
forward centroid top-K/2 with backward direction top-K/2 per position and
verifies with a pure-python forward-consistency beam over the true public
maps (the white-box grant). With --toy-opt, a second toy world validates
the BRE arms with REAL optimization (analytic gradients, pure python):
the toy top at position p is the linear map C_p with the harness's
next-token CE convention, so the observed gradient is
g_p = C_p @ (softmax(h_p @ C_p) - onehot[label_{p+1}]); toy GM optimizes
soft labels y' against Eq. 9's beta*L2+(1-beta)*L1 distance, toy SM
optimizes a continuous embedding under the Eq. 7 cosine objective with
the Eq. 8 nearest-row decode. The toy forward-perturbation defense is
STRUCTURED Gaussian noise (a public per-coordinate profile at the
victim's private scale) — isotropic noise would give noise-aware
training nothing to learn (measured). Toy NaMoE then runs the paper's
two-stage recipe: stage 1 fits each expert (centroid tables) on the
clean public aux rows passed through a Wiener front-end calibrated to
its own scale (the toy form of noise-aware expert training), stage 2
trains an input-dependent gating (softmax over [1, nlev], nlev = the
row's excess squared norm — an attacker-computable noise-level estimate)
with the routing-classification form of the auto-encoder objective
(plain mixture-CE collapses to the best-on-average expert in the toy's
low-contrast regime, measured). --toy-ft-gap delta perturbs the public
maps the mimic uses (the toy fine-tuning gap). Arms reported per cell:
bisr (bidirectional beam), sip-fwd-only (learned forward argmax) and the
position-independent NN baseline; --toy-opt cells add bisr-opt-b / -f /
-bf and (with --namoe) namoe / namoe+bf. The toy validates
selection/beam/optimization/scoring/gate plumbing and the
bidirectional >= forward-only ordering under noise; real numbers come
from --capture-dir runs. Measured toy-scope caveat: the toy NaMoE
mixture tracks the noise-unaware SIP within a few points at every scale
(adaptivity), but no LARGE noise-aware margin appears — with a linear
toy world and a robust cosine decode there is little headroom; the
paper's sizable NaMoE gains (its Fig. 5/Table 5) stem from the victim
model's own adaptation to the perturbation (SS3.3's observation and the
attack-deeper strategy), a channel this toy models only observationally.

Deliberately still pilot-scale (Phase-B must close): the MLP decoder in
place of the paper's GRU-256 (Table 12 shows even a linear inverter is
strong, GRU best); Gaussian seam noise in place of d_XP/DP-Forward/NoPeek
mechanisms for NaMoE; aux corpus defaults remain small (--aux-seqs 512)
though the streaming path removes the memory ceiling; BRE hyperparameters
default to the paper's GPT2-large column but are not re-tuned per model;
no validation-gate reproduction numbers yet (Phase-B gate)."""

import argparse
from dataclasses import asdict
import json
import os
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
from .vec2text_inversion import corpus_bleu

EXPERIMENT_ID = "bisr_baseline"
MODES = ("training", "inference")
REQUIRES_LABELS = False
DESCRIPTION = ("BiSR/SIP (arXiv:2409.00960) scoped adaptation: a "
               "SIP-style learned inversion model fit on auxiliary data "
               "asserted DISJOINT from the frozen/pilot corpora (streaming, "
               "resume-safe aux-scale training), bidirectional "
               "forward+backward candidate selection with "
               "forward-consistency beam verification, the paper's BRE "
               "optimization arms (backward gradient matching Eq. 9, "
               "forward smashed-data matching Eq. 7-8, sequential b+f) and "
               "a NaMoE-style noise-adaptive mixture-of-experts inverter; "
               "clean-room from the paper, pilot-scale budgets")

SCHEMA = "dtraining.attacker.bisr_baseline.v1"
CKPT_FORMAT = "llm_attacker.bisr_inverter.v1"
CKPT_FORMAT_GRU = "llm_attacker.bisr_gru.v1"
CKPT_FORMAT_NAMOE = "llm_attacker.bisr_inverter_namoe.v1"
AUX_CACHE_FORMAT = "llm_attacker.bisr_auxcache.v1"

DEFAULT_AUX_MANIFEST = "experiments/validation_corpus_manifest.json"
DEFAULT_PILOT_CORPUS = "results/pilot-gpt2-wikitext2/inputs/pilot-corpus.jsonl"

THREAT_MODEL = (
    "honest-but-curious seam observer of fwd seam activations h* and (in "
    "training mode) the bwd seam gradients dL/dh*, granted the victim's "
    "PUBLIC base weights (--model: embedding E + public head at --split) "
    "for auxiliary training forwards and beam verification — the BiSR/SIP "
    "white-box forward-inversion grant; the victim's finetuned head is "
    "never assumed (mismatch is part of the measurement). Canonical tokens "
    "are evaluation truth, never an attack input (label-free attack). The "
    "auxiliary training corpus is asserted article-id disjoint from the "
    "frozen validation corpus and the pilot victim corpus.")

INTERPRETATION = (
    "seq_accuracy is position-aligned vs canonical, pooled per token; bleu "
    "is smoothed corpus BLEU-4 (add-one smoothing on n>=2); token-set "
    "precision/recall are orderless per-sequence set overlaps (VFLAIR "
    "convention); the control decodes the auxiliary-corpus majority token "
    "at every position (attacker-computable; stored in the checkpoint); "
    "the gate verdict is the frozen-gate Bonferroni-Wilson excess over "
    "that control with n = scored tokens. Arm 'bisr' unions learned "
    "forward proposals with backward direction proposals and verifies by "
    "forward consistency through the public head; arm 'sip-fwd-only' is "
    "the same learned forward scores without the backward channel (the "
    "paired diagnostic); 'baseline_nn' is the position-independent "
    "cosine-NN decode over E. With --opt-arm, 'bisr_opt' is the paper's "
    "BRE second stage (SS3.2): 'b' gradient matching against the observed "
    "bwd seam gradient (Eq. 9, beta*L2+(1-beta)*L1, dummy soft labels "
    "y'=softmax(z/tau) initialized from the SIP softmax, h* fixed), 'f' "
    "smashed-data matching in the continuous embedding space (Eq. 7, "
    "cosine objective; Eq. 8 nearest-row decode), 'bf' the paper's "
    "sequential BiSR(b+f); position 0 of 'b'/'bf' comes from the SIP "
    "decode (no label gradient identifies it). A NaMoE checkpoint "
    "(--namoe, paper SS3.3) replaces the single inverter with a "
    "gating-weighted mixture of noise-aware experts; its mixture scores "
    "feed every downstream arm. This remains a PILOT-SCALE adaptation: "
    "the paper's SIP decoder is a GRU-256 trained on a substantial "
    "auxiliary corpus and NaMoE perturbs with d_XP/DP mechanisms (here: "
    "Gaussian seam noise); numbers are not comparable cell-for-cell with "
    "the paper's. Toy tier is a machinery check on synthetic seams, not "
    "a security claim.")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, hidden=16)
    ap.add_argument("--phase", choices=("train", "attack"), default="attack",
                    help="BiSR stage: 'train' fits the SIP-style inverter "
                         "on auxiliary data (torch tier) and saves "
                         "--inverter-pt; 'attack' (default) loads "
                         "--inverter-pt and scores --capture-dir against "
                         "--canonical-pt. The framework's --mode "
                         "(training/inference wire surface) is separate "
                         "and unchanged")
    ap.add_argument("--capture-dir", default=None,
                    help="wire-capture directory (attack phase; training "
                         "mode needs phase='fwd' AND phase='bwd' seam "
                         "captures — the bidirectional channel; inference "
                         "mode scores phase='prefill' rows forward-only)")
    ap.add_argument("--canonical-pt", default=None,
                    help="canonical.pt ground-truth input_ids, LongTensor "
                         "[n_captured*batch, seq] row-aligned to the fwd "
                         "captures step-major (splitbench "
                         "--save-canonical)")
    ap.add_argument("--model", default=None,
                    help="HF name/path of the PUBLIC base model: train "
                         "phase supplies E + the public head for auxiliary "
                         "training forwards; attack phase supplies E + the "
                         "public head for beam verification (required)")
    ap.add_argument("--split", default="6",
                    help="seam boundary of the PUBLIC head (must match the "
                         "capture harness split; default '6')")
    ap.add_argument("--inverter-pt", default=None,
                    help="SIP-style inverter checkpoint "
                         f"({CKPT_FORMAT}): SAVED by --phase train, "
                         "LOADED by --phase attack")
    ap.add_argument("--decoder", choices=("mlp", "gru"), default="mlp",
                    help="Legacy MLP or prospective GRU-256 SIP (explicit choice)")
    ap.add_argument("--train-epochs", type=int, default=20,
                    help="Complete seeded epochs for GRU; MLP uses --train-steps")
    ap.add_argument("--knowledge", choices=("public", "exact"), default="public")
    ap.add_argument("--head-pt", default=None,
                    help="Exact GPT-2 client head; forbidden for public grant")
    ap.add_argument("--head-sha256", default=None,
                    help="Required expected digest for --head-pt")
    ap.add_argument("--aux-file", default=None,
                    help="auxiliary wikitext-2 JSONL ({\"id\", \"text\"} "
                         "articles) tokenized with the --model tokenizer "
                         "for training; article ids are asserted DISJOINT "
                         "from --aux-manifest and --pilot-corpus before "
                         "any training (default: synthetic uniform draws "
                         "-- no auxiliary text assumed)")
    ap.add_argument("--aux-manifest", default=DEFAULT_AUX_MANIFEST,
                    help="frozen validation corpus manifest whose selected "
                         "ids the auxiliary corpus must not share "
                         f"(default {DEFAULT_AUX_MANIFEST})")
    ap.add_argument("--pilot-corpus", default=DEFAULT_PILOT_CORPUS,
                    help="pilot corpus JSONL (victim/calibration "
                         "documents) the auxiliary corpus must not share "
                         f"article ids with (default {DEFAULT_PILOT_CORPUS})")
    ap.add_argument("--aux-seqs", type=int, default=512,
                    help="auxiliary training sequences (windows of "
                         "--seq-len); with --aux-file the corpus is "
                         "STREAMED into a resume-safe shard cache, so "
                         "this is capped by disk, not RAM")
    ap.add_argument("--aux-chunk-seqs", type=int, default=2048,
                    help="windows per streamed shard in the aux cache "
                         "(shard h-rows cost seq_len*H*4 bytes each)")
    ap.add_argument("--aux-cache", default=None,
                    help="aux shard cache directory (default "
                         "<inverter-pt>.auxcache); resume-safe: a config "
                         "fingerprint mismatch is a descriptive error, "
                         "never silent reuse")
    ap.add_argument("--train-steps", type=int, default=500,
                    help="optimizer steps for the SIP-style inverter "
                         "(per expert under --namoe; configurable)")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="training batch size (sequences)")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="AdamW lr for the inverter")
    ap.add_argument("--d-model", type=int, default=128,
                    help="inverter hidden width")
    ap.add_argument("--candidates", type=int, default=8,
                    help="K proposals per position (even, >= 2): forward "
                         "top K/2 union backward top K/2, forward-filled")
    ap.add_argument("--beam-width", type=int, default=2,
                    help="forward-consistency beam width")
    ap.add_argument("--samples", type=int, default=8,
                    help="cap on scored sequences (scan order)")
    ap.add_argument("--device", default=None, choices=("cpu", "cuda"),
                    help="torch device (default: cuda when available)")
    ap.add_argument("--seq-len", type=int, default=12,
                    help="toy sequence length; torch train: auxiliary "
                         "window length (must match the capture harness "
                         "seq_len)")
    ap.add_argument("--vocab", type=int, default=60, help="toy vocab size")
    ap.add_argument("--toy-seqs", type=int, default=24,
                    help="toy victim sequences per cell")
    ap.add_argument("--toy-aux-seqs", type=int, default=300,
                    help="toy auxiliary training sequences (centroid fit + "
                         "control corpus)")
    ap.add_argument("--toy-sigmas", type=float, nargs="+",
                    default=(0.0, 0.1, 0.3, 1.0, 2.0),
                    help="toy seam-noise sweep (0 = noiseless)")
    ap.add_argument("--namoe", action="store_true",
                    help="train phase: fit the NaMoE noise-adaptive "
                         "mixture-of-experts inverter (paper SS3.3): one "
                         "noise-aware expert per --namoe-sigmas scale "
                         "(stage 1, gating frozen), then gating trained "
                         "on random scales (stage 2, experts frozen). "
                         "Attack phase: auto-detected from the "
                         "checkpoint; passing --namoe against a plain "
                         "SIP checkpoint is a descriptive error")
    ap.add_argument("--namoe-sigmas", type=float, nargs="+",
                    default=(0.0, 0.1, 0.3, 1.0),
                    help="per-expert Gaussian seam-noise scales for NaMoE "
                         "(this adaptation's perturbation mechanism N_s; "
                         "the paper uses embedding-d_XP / smashed-data-DP "
                         "scales) — include 0.0 for the clean expert")
    ap.add_argument("--gating-steps", type=int, default=100,
                    help="NaMoE stage-2 gating training steps (random "
                         "noise scales, experts frozen)")
    ap.add_argument("--opt-arm", choices=("none", "b", "f", "bf"),
                    default="none",
                    help="BRE optimization-based enhancement (paper "
                         "SS3.2) on the attack phase, warm-started from "
                         "the SIP/beam reconstruction: 'b' backward "
                         "gradient matching (Eq. 9; needs the bwd seam, "
                         "training mode only), 'f' forward smashed-data "
                         "matching (Eq. 7-8; works forward-only), 'bf' "
                         "the paper's sequential BiSR(b+f)")
    ap.add_argument("--gm-iters", type=int, default=32,
                    help="gradient-matching iterations (paper Table 1 "
                         "GM-epc, GPT2-large column)")
    ap.add_argument("--gm-lr", type=float, default=0.09,
                    help="gradient-matching AdamW lr (Table 1 GM-lr)")
    ap.add_argument("--gm-beta", type=float, default=0.85,
                    help="Eq. 9 combined-distance coefficient: "
                         "beta*L2 + (1-beta)*L1 (Table 1 GM-beta)")
    ap.add_argument("--gm-tau", type=float, default=1.2,
                    help="softmax temperature of the dummy-label "
                         "parameterization y'=softmax(z/tau) (Table 1 "
                         "GM-tau)")
    ap.add_argument("--sm-iters", type=int, default=800,
                    help="smashed-data-matching iterations (Table 1 "
                         "SM-epc)")
    ap.add_argument("--sm-lr", type=float, default=0.01,
                    help="smashed-data-matching AdamW lr (Table 1 SM-lr, "
                         "GPT2-large column)")
    ap.add_argument("--sm-wd", type=float, default=0.01,
                    help="smashed-data-matching AdamW weight decay "
                         "(Table 1 SM-wd, GPT2-large column)")
    ap.add_argument("--opt-batch", type=int, default=8,
                    help="sequences optimized simultaneously by the BRE "
                         "arms (memory bound; the objective is "
                         "per-sequence exact)")
    ap.add_argument("--toy-opt", action="store_true",
                    help="toy tier: add the toy BRE optimization arms "
                         "(toy GM/SM with analytic gradients, toy NaMoE "
                         "with --namoe) on a second synthetic world")
    ap.add_argument("--toy-opt-seqs", type=int, default=8,
                    help="toy BRE victim sequences per cell")
    ap.add_argument("--toy-gm-steps", type=int, default=25,
                    help="toy gradient-matching Adam steps per position")
    ap.add_argument("--toy-sm-steps", type=int, default=60,
                    help="toy smashed-data-matching Adam steps per "
                         "position")
    ap.add_argument("--toy-ft-gap", type=float, default=0.0,
                    help="toy fine-tuning gap: std of the perturbation "
                         "between the TRUE toy maps (victim) and the "
                         "PUBLIC maps the mimic optimizes through")
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
                          node_role="attacker/bisr-baseline")


def load_article_ids(jsonl_path):
    """Article ids of a {"id": ...} JSONL corpus; duplicates are an error
    (an id collision inside one corpus makes overlap claims unauditable)."""
    from ..evidence import file_sha256
    docs = [json.loads(line) for line in open(jsonl_path)
            if line.strip()]
    ids = [d.get("id") for d in docs]
    if not ids or any(i is None for i in ids):
        raise ValueError(f"{jsonl_path}: every line must carry an 'id'")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{jsonl_path}: duplicate article ids")
    return set(ids), file_sha256(jsonl_path)


def _frozen_selected_ids(manifest_path):
    manifest = json.load(open(manifest_path))
    if manifest.get("schema") != "llm_attacker.validation_corpus.v1":
        raise ValueError(f"{manifest_path}: unexpected frozen-corpus "
                         "manifest schema")
    ids = set()
    for partition, values in manifest.get("selected_ids", {}).items():
        ids.update(values)
    if not ids:
        raise ValueError(f"{manifest_path}: no selected ids")
    return ids


def assert_aux_disjoint(aux_path, manifest_path=None, pilot_corpus_path=None):
    """Fail closed: the auxiliary training corpus must share NO article id
    with the frozen validation corpus (any partition) or the pilot victim
    corpus. Missing reference files are recorded as skipped (a caller on a
    machine without the frozen corpus cannot check against it), never
    silently treated as disjoint. Returns the audit record."""
    aux_ids, aux_sha = load_article_ids(aux_path)
    record = {"aux_corpus": str(aux_path), "aux_sha256": aux_sha,
              "aux_articles": len(aux_ids), "checked_against": [],
              "skipped_missing": [], "disjoint": True}
    for label, path, loader in (
            ("frozen_validation_manifest", manifest_path,
             _frozen_selected_ids),
            ("pilot_corpus", pilot_corpus_path, None)):
        if not path:
            continue
        if not os.path.exists(path):
            record["skipped_missing"].append(label)
            continue
        ref_ids = loader(path) if loader else load_article_ids(path)[0]
        overlap = sorted(aux_ids & ref_ids)
        if overlap:
            raise ValueError(
                f"auxiliary corpus overlaps the {label}: {len(overlap)} "
                f"shared article ids, e.g. {overlap[:3]} — BiSR's SIP "
                "inversion model must train on data DISJOINT from "
                "victim/calibration/evaluation documents")
        record["checked_against"].append(label)
    return record


def toy_world(seed, vocab=60, dim=16, seq_len=12, n_victim=24, n_aux=300,
              sigma=0.0):
    """Toy BiSR world. E [vocab, dim] random; A_p [dim, dim] per-position
    fwd seam maps (non-orthogonal across positions); B_p [dim, dim]
    per-position bwd maps. Victim and auxiliary sequences are uniform token
    draws, disjoint by construction (synthetic ids). Fwd rows
    h = E[tok] @ A_p + N(0, sigma^2); the toy backward channel emits
    g_p = -E[tok_{p+1}] @ B_p + N(0, sigma^2) — the toy TAG direction
    signal: the gradient at row p points at the LABEL of position p+1, and
    position 0 carries no label gradient."""
    rng = random.Random(seed)
    E = _randn_mat(vocab, dim, rng)
    A = [_randn_mat(dim, dim, rng) for _ in range(seq_len)]
    B = [_randn_mat(dim, dim, rng) for _ in range(seq_len)]
    EA = [_matmul(E, Ap) for Ap in A]
    EB = [_matmul(E, Bp) for Bp in B]
    aux = [[rng.randrange(vocab) for _ in range(seq_len)]
           for _ in range(n_aux)]
    victim = [[rng.randrange(vocab) for _ in range(seq_len)]
              for _ in range(n_victim)]

    def noise():
        return rng.gauss(0.0, sigma) if sigma > 0.0 else 0.0

    def fwd_rows(seqs):
        return [[[v + noise() for v in EA[p][tok]]
                 for p, tok in enumerate(s)] for s in seqs]

    def bwd_rows(seqs):
        out = []
        for s in seqs:
            gs = []
            for p in range(seq_len):
                label = s[p + 1] if p + 1 < seq_len else rng.randrange(vocab)
                gs.append([-v + noise() for v in EB[p][label]])
            out.append(gs)
        return out

    return {"E": E, "A": A, "B": B, "EA": EA, "EB": EB,
            "aux": aux, "victim": victim,
            "h_aux": fwd_rows(aux), "h_victim": fwd_rows(victim),
            "g_aux": bwd_rows(aux), "g_victim": bwd_rows(victim),
            "vocab": vocab, "dim": dim, "seq_len": seq_len, "sigma": sigma}


def toy_train_centroids(aux, h_aux, vocab, seq_len, dim):
    """The toy SIP-style LEARNED inverter: tables[p][v] = mean auxiliary
    seam row at position p carrying token v; a position-pooled centroid
    backs up unseen (p, v) pairs, zeros as the last resort (never win a
    cosine). Trained on the auxiliary split alone."""
    sums = [[None] * vocab for _ in range(seq_len)]
    cnts = [[0] * vocab for _ in range(seq_len)]
    gsum = [None] * vocab
    gcnt = [0] * vocab
    for s, hs in zip(aux, h_aux):
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


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _topk_scores(row, table, k):
    scored = sorted(range(len(table)), key=lambda v: -_cos(row, table[v]))
    return scored[:k]


def toy_candidates(world, tables, h_seq, g_seq, count):
    """Bidirectional candidate sets per position: the learned forward
    inverter's top count/2 (centroid cosine against position p's table)
    UNION the backward direction signal's top count/2 (cosine of -g_{p-1}
    against EB[p-1], the toy label gradient), forward-filled to exactly
    count. Position 0 is forward-only (no label gradient identifies it)."""
    half = max(1, count // 2)
    seq_len = len(h_seq)
    vocab = len(world["E"])
    cand = []
    for p in range(seq_len):
        fwd = _topk_scores(h_seq[p], tables[p], half)
        picked = list(dict.fromkeys(fwd))
        if p > 0:
            neg_g = [-x for x in g_seq[p - 1]]
            back = _topk_scores(neg_g, world["EB"][p - 1], half)
            picked = list(dict.fromkeys(picked + back))
        fill = _topk_scores(h_seq[p], tables[p], vocab)
        picked = list(dict.fromkeys(picked + fill))[:count]
        cand.append(picked)
    return cand


def toy_beam(world, candidates, h_seq, beam_width=2):
    """Pure-python forward-consistency beam (white-box public maps EA):
    score(prefix) = sum_{j<=p} ||EA[j][tok_j] - h_j||^2, causal so each
    extension's residual is final; finite-width pruning is approximate."""
    beam = [([], 0.0)]
    for p, cand in enumerate(candidates):
        exts = []
        for pref, res in beam:
            for v in cand:
                r = sum((a - b) ** 2
                        for a, b in zip(world["EA"][p][v], h_seq[p]))
                exts.append((pref + [v], res + r))
        exts.sort(key=lambda t: t[1])
        beam = exts[:max(1, beam_width)]
    return beam[0][0]


def toy_cell(args, sigma, seed):
    """One toy cell: bisr (bidirectional union + forward-consistency beam)
    vs sip-fwd-only (learned forward argmax) vs the position-independent NN
    baseline, all scored with the shared plumbing."""
    world = toy_world(args.seed + 619 * seed, vocab=args.vocab,
                      dim=args.hidden, seq_len=args.seq_len,
                      n_victim=args.toy_seqs, n_aux=args.toy_aux_seqs,
                      sigma=sigma)
    tables = toy_train_centroids(world["aux"], world["h_aux"], args.vocab,
                                 args.seq_len, args.hidden)
    bisr, fwd_only = [], []
    for hs, gs in zip(world["h_victim"], world["g_victim"]):
        cand = toy_candidates(world, tables, hs, gs, args.candidates)
        bisr.append(toy_beam(world, cand, hs, beam_width=args.beam_width))
        fwd_only.append([_topk_scores(hs[p], tables[p], 1)[0]
                         for p in range(args.seq_len)])
    baseline = [cosine_nn_decode(hs, world["E"]) for hs in world["h_victim"]]
    control = majority_control_token(world["aux"])
    n_seeds = len(args.seeds)
    cell = {"sigma": sigma, "seed": seed,
            "candidates": args.candidates,
            "arm": "bisr",
            **score_sequences(bisr, world["victim"], control,
                              n_seeds=n_seeds),
            "bleu": round(corpus_bleu(bisr, world["victim"]), 6)}
    cell["sip_fwd_only"] = {
        **score_sequences(fwd_only, world["victim"], control,
                          n_seeds=n_seeds),
        "bleu": round(corpus_bleu(fwd_only, world["victim"]), 6)}
    cell["baseline_nn"] = {
        **score_sequences(baseline, world["victim"], control,
                          n_seeds=n_seeds),
        "bleu": round(corpus_bleu(baseline, world["victim"]), 6)}
    return cell


def _row_mat(row, M):
    """row [r] @ M [r][c] -> [c]."""
    return [sum(row[i] * M[i][j] for i in range(len(row)))
            for j in range(len(M[0]))]


def _mat_col(M, w):
    """M [r][c] @ w [c] -> [r]."""
    return [sum(M[i][j] * w[j] for j in range(len(w)))
            for i in range(len(M))]


def _col_dot(M, j, w):
    """dot(M[:, j], w)."""
    return sum(M[i][j] * w[i] for i in range(len(w)))


def _softmax(xs):
    m = max(xs)
    ex = [2.718281828459045 ** (x - m) for x in xs]
    s = sum(ex)
    return [x / s for x in ex]


def _norm2(xs):
    return sum(x * x for x in xs) ** 0.5


def _adam_update(p, grad, m, v, t, lr, wd=0.0, betas=(0.9, 0.999),
                 eps=1e-8):
    """In-place AdamW step on parallel lists (toy sizes only)."""
    b1, b2 = betas
    for i in range(len(p)):
        m[i] = b1 * m[i] + (1 - b1) * grad[i]
        v[i] = b2 * v[i] + (1 - b2) * grad[i] * grad[i]
        mh = m[i] / (1 - b1 ** t)
        vh = v[i] / (1 - b2 ** t)
        p[i] -= lr * (mh / (vh ** 0.5 + eps) + wd * p[i])


def toy_opt_world(seed, vocab=60, dim=16, seq_len=12, n_victim=8,
                  n_aux=300, sigma=0.0, ft_gap=0.0):
    """Toy world for the BRE optimization arms. E [vocab, dim]; TRUE
    per-position bottom maps A_p (the victim's head) and top maps
    C_p [dim, vocab] (the victim's tail). The observed fwd row is
    h_p = E[tok_p] @ A_p + N(0, sigma^2 * profile_p^2) — the toy
    forward-perturbation defense: a per-coordinate noise SHAPE profile_p
    (public strategy family, drawn once per world ~ U(0.5, 2)) at the
    victim's private scale sigma; structured so that scale-matched
    noise-aware training has something to learn (paper SS3.3's
    observation is trivially false for isotropic noise + a centroid
    classifier). The observed bwd gradient is the TRUE next-token CE
    gradient at the seam, g_p = C_p @ (softmax(h_p @ C_p) -
    onehot[tok_{p+1}]) + isotropic N(0, sigma^2) for p < seq_len-1, and
    zeros at the last position (the harness drops the final logits, so
    that row carries no gradient). The attacker trains on aux rows pushed
    through the PUBLIC bottom A~_p (clean) and optimizes through the
    PUBLIC maps (A~, C~); --toy-ft-gap delta makes A~ = A + N(0, delta^2),
    C~ = C + N(0, delta^2) — the toy semi-white-box fine-tuning gap of
    paper SS2.4. Aux/victim are uniform token draws, disjoint by
    construction."""
    rng = random.Random(seed)
    E = _randn_mat(vocab, dim, rng)
    A = [_randn_mat(dim, dim, rng) for _ in range(seq_len)]
    C = [_randn_mat(dim, vocab, rng) for _ in range(seq_len)]
    profile = [[0.5 + 1.5 * rng.random() for _ in range(dim)]
               for _ in range(seq_len)]

    def perturb(maps):
        if ft_gap <= 0.0:
            return [[list(r) for r in M] for M in maps]
        return [[[x + rng.gauss(0.0, ft_gap) for x in r] for r in M]
                for M in maps]

    A_pub, C_pub = perturb(A), perturb(C)
    aux = [[rng.randrange(vocab) for _ in range(seq_len)]
           for _ in range(n_aux)]
    victim = [[rng.randrange(vocab) for _ in range(seq_len)]
              for _ in range(n_victim)]

    def fwd_noise(p):
        return [rng.gauss(0.0, sigma * profile[p][j]) if sigma > 0.0
                else 0.0 for j in range(dim)]

    def fwd_rows(seqs, maps, noisy):
        out = []
        for s in seqs:
            rows = []
            for p, tok in enumerate(s):
                base = _row_mat(E[tok], maps[p])
                rows.append([b + n for b, n in zip(base, fwd_noise(p))]
                            if noisy else base)
            out.append(rows)
        return out

    def bwd_rows(seqs, hs):
        out = []
        for s, h in zip(seqs, hs):
            gs = []
            for p in range(seq_len):
                if p + 1 >= seq_len:
                    gs.append([0.0] * dim)
                    continue
                u = _softmax(_row_mat(h[p], C[p]))
                onehot = [1.0 if v == s[p + 1] else 0.0
                          for v in range(vocab)]
                gs.append([v + (rng.gauss(0.0, sigma) if sigma > 0.0
                                else 0.0)
                           for v in _mat_col(C[p], [a - b for a, b in
                                                    zip(u, onehot)])])
            out.append(gs)
        return out

    h_victim = fwd_rows(victim, A, True)
    h_aux = fwd_rows(aux, A_pub, False)
    return {"E": E, "A": A, "C": C, "A_pub": A_pub, "C_pub": C_pub,
            "noise_profile": profile,
            "aux": aux, "victim": victim,
            "h_aux": h_aux, "h_victim": h_victim,
            "g_victim": bwd_rows(victim, h_victim),
            "vocab": vocab, "dim": dim, "seq_len": seq_len,
            "sigma": sigma, "ft_gap": ft_gap}


def _toy_centroid_scores(row, table_p):
    """Per-token cosine scores of a seam row against a centroid table."""
    return [_cos(row, table_p[v]) for v in range(len(table_p))]


def toy_gm_position(h_p, g_p, C_pub_p, z_init, iters, lr, beta, tau):
    """Toy gradient matching (paper SS3.2.2, Eq. 9) at ONE position with
    analytic gradients: y' = softmax(z/tau) is optimized so the mimic
    top's dummy gradient C~_p @ (softmax(h_p @ C~_p) - y') matches the
    observed g_p under beta*||.||_2 + (1-beta)*||.||_1. h_p is the
    observed fwd row, fixed (the known-forward-activation trait). Returns
    (argmax token of y', final matching loss)."""
    vocab = len(C_pub_p[0])
    u = _softmax(_row_mat(h_p, C_pub_p))
    z = list(z_init)
    m, v = [0.0] * vocab, [0.0] * vocab
    loss = 0.0
    for t in range(1, max(1, iters) + 1):
        y = _softmax([x / tau for x in z])
        dg = _mat_col(C_pub_p, [a - b for a, b in zip(u, y)])
        delta = [d - g for d, g in zip(dg, g_p)]
        l2 = _norm2(delta)
        loss = beta * l2 + (1.0 - beta) * sum(abs(x) for x in delta)
        d_delta = [(beta * (x / l2) if l2 > 1e-12 else 0.0)
                   + (1.0 - beta) * (1.0 if x > 0 else
                                     (-1.0 if x < 0 else 0.0))
                   for x in delta]
        d_y = [-_col_dot(C_pub_p, j, d_delta) for j in range(vocab)]
        ydot = sum(a * b for a, b in zip(y, d_y))
        grad = [yj * (dj - ydot) / tau for yj, dj in zip(y, d_y)]
        _adam_update(z, grad, m, v, t, lr)
    y = _softmax([x / tau for x in z])
    best = max(range(vocab), key=lambda j: y[j])
    return best, loss


def toy_sm_position(h_p, A_pub_p, e_init, E, iters, lr, wd):
    """Toy smashed-data matching (paper SS3.2.1, Eq. 7-8) at ONE position
    with analytic gradients: a dummy embedding e' is optimized under the
    cosine objective 1 - cos(e' @ A~_p, h_p), then decoded to the
    Euclidean-nearest embedding row (Eq. 8). Returns (token, final
    cosine distance)."""
    dim = len(e_init)
    e = list(e_init)
    m, v = [0.0] * dim, [0.0] * dim
    loss = 1.0
    for t in range(1, max(1, iters) + 1):
        w = _row_mat(e, A_pub_p)
        nw, nh = _norm2(w), _norm2(h_p)
        wdot = sum(a * b for a, b in zip(w, h_p))
        if nw < 1e-12 or nh < 1e-12:
            break
        loss = 1.0 - wdot / (nw * nh)
        dcos = [- (h_p[i] / (nw * nh)
                   - w[i] * wdot / (nw ** 3 * nh)) for i in range(dim)]
        grad = _mat_col(A_pub_p, dcos)
        _adam_update(e, grad, m, v, t, lr, wd=wd)
    dists = [sum((a - b) ** 2 for a, b in zip(ev, e)) for ev in E]
    best = min(range(len(E)), key=lambda j: dists[j])
    return best, loss


def _toy_signal_stats(world):
    """Attacker-computable calibration statistics from the CLEAN public
    aux rows: per-(position, coordinate) signal variance (for the Wiener
    experts) and mean squared row norm per position (for the gating's
    noise-level feature)."""
    seq_len, dim = world["seq_len"], world["dim"]
    n = len(world["aux"])
    sigvar = [[0.0] * dim for _ in range(seq_len)]
    norm2 = [0.0] * seq_len
    for p in range(seq_len):
        for j in range(dim):
            vals = [world["h_aux"][i][p][j] for i in range(n)]
            mu = sum(vals) / n
            sigvar[p][j] = sum((x - mu) ** 2 for x in vals) / n
        norm2[p] = sum(sum(x * x for x in world["h_aux"][i][p])
                       for i in range(n)) / n
    return sigvar, norm2


def _toy_wiener(row, p, s_k, sigvar, profile):
    """The expert's denoising front-end calibrated to its assumed scale
    s_k: coordinate-wise Wiener gain signal_var / (signal_var +
    (s_k*profile)^2); MMSE-optimal when s_k equals the victim's true
    scale (the toy form of noise-aware expertise)."""
    dim = len(row)
    return [row[j] * sigvar[p][j]
            / (sigvar[p][j] + (s_k * profile[p][j]) ** 2 + 1e-12)
            for j in range(dim)]


def _toy_expert_fit(world, scale, sigvar):
    """One NaMoE expert (toy): centroid tables fit on the CLEAN public aux
    rows passed through the expert's OWN scale-calibrated Wiener front-end
    (paper SS3.3 stage 1: each expert is trained against its perturbation
    scale N_{s'}; filtering is nonlinear per coordinate, so the per-scale
    training matters)."""
    profile = world["noise_profile"]
    filtered = [[_toy_wiener(hs[p], p, scale, sigvar, profile)
                 for p in range(world["seq_len"])]
                for hs in world["h_aux"]]
    return toy_train_centroids(world["aux"], filtered, world["vocab"],
                               world["seq_len"], world["dim"])


def _toy_expert_posterior(row, table, tau=0.1):
    """The expert's posterior over tokens at one position: softmax of its
    filtered-row cosine scores at the fixed toy score temperature tau
    (documented; cosines lie in [-1, 1] and need a scale to become a
    distribution)."""
    return _softmax([_cos(row, table[v]) / tau for v in range(len(table))])


def toy_namoe_fit(world, sigmas, gating_steps, lr, seed):
    """Toy NaMoE (paper SS3.3, Fig. 6) against the toy world's structured
    forward perturbation. Stage 1 (gating frozen): per-scale Wiener
    experts (_toy_expert_fit). Stage 2 (experts frozen): an
    INPUT-DEPENDENT gating over the experts' posterior mixture,
    w = softmax(theta @ [1, nlev]) with nlev the row's standardized
    excess squared norm over the clean-aux mean (an attacker-computable
    noise-level estimate — the toy form of the paper's gating network
    reading the smashed row). The paper trains the gating with the
    auto-encoder loss on random-scale smashed data; the toy uses the
    routing-classification form of that objective — per-row label = the
    expert whose posterior ranks the true token highest (the
    auto-encoder label, attacker-computable) — a convex surrogate that
    cannot collapse to the best-on-average expert in the toy's
    low-contrast regime (plain mixture-CE collapses there, measured).
    Returns (experts, gating, record) with gating =
    {'theta0': [K], 'theta1': [K], 'nlev_mean': ..., 'nlev_std': ...}."""
    rng = random.Random(seed)
    vocab, seq_len, dim = world["vocab"], world["seq_len"], world["dim"]
    profile = world["noise_profile"]
    sigvar, norm2 = _toy_signal_stats(world)
    experts = [_toy_expert_fit(world, s_k, sigvar) for s_k in sigmas]
    K = len(sigmas)
    nlevs = [sum(x * x for x in world["h_aux"][i][p]) - norm2[p]
             for i in range(len(world["aux"])) for p in range(seq_len)]
    nl_mu = sum(nlevs) / len(nlevs)
    nl_sd = (sum((x - nl_mu) ** 2 for x in nlevs) / len(nlevs)) ** 0.5 \
        or 1.0
    theta0, theta1 = [0.0] * K, [0.0] * K
    m0, v0 = [0.0] * K, [0.0] * K
    m1, v1 = [0.0] * K, [0.0] * K
    max_s = max(sigmas) if sigmas else 1.0
    n_aux = len(world["aux"])
    for t in range(1, max(1, gating_steps) + 1):
        i = rng.randrange(n_aux)
        s = rng.random() * max_s
        p = rng.randrange(seq_len)
        tok = world["aux"][i][p]
        row = [x + (rng.gauss(0.0, s * profile[p][j]) if s > 0.0 else 0.0)
               for j, x in enumerate(world["h_aux"][i][p])]
        nlev = (sum(x * x for x in row) - norm2[p] - nl_mu) / nl_sd
        posts = [_toy_expert_posterior(
                     _toy_wiener(row, p, sigmas[k], sigvar, profile),
                     experts[k][p])
                 for k in range(K)]
        label = max(range(K), key=lambda k: posts[k][tok])
        logits = [theta0[k] + theta1[k] * nlev for k in range(K)]
        w = _softmax(logits)
        g0 = [w[k] - (1.0 if k == label else 0.0) for k in range(K)]
        _adam_update(theta0, g0, m0, v0, t, lr)
        _adam_update(theta1, [g * nlev for g in g0], m1, v1, t, lr)
    gating = {"theta0": theta0, "theta1": theta1,
              "nlev_mean": nl_mu, "nlev_std": nl_sd,
              "sigvar": sigvar, "norm2": norm2}
    rec = {"expert_sigmas": list(sigmas), "gating_steps": gating_steps,
           "gating_lr": 0.05, "gating_objective": "routing-classification",
           "wiener_front_end": True, "toy_score_temperature": 0.1}
    return experts, gating, rec


def _toy_namoe_mixture(world, experts, gating, sigmas, row, p):
    """Per-position NaMoE mixture posterior over tokens for one observed
    row: input-dependent gating weights times per-expert posteriors."""
    profile = world["noise_profile"]
    sigvar, norm2 = gating["sigvar"], gating["norm2"]
    nlev = (sum(x * x for x in row) - norm2[p]
            - gating["nlev_mean"]) / gating["nlev_std"]
    logits = [gating["theta0"][k] + gating["theta1"][k] * nlev
              for k in range(len(sigmas))]
    w = _softmax(logits)
    posts = [_toy_expert_posterior(
                 _toy_wiener(row, p, sigmas[k], sigvar, profile),
                 experts[k][p])
             for k in range(len(sigmas))]
    mix = [sum(w[k] * posts[k][j] for k in range(len(sigmas)))
           for j in range(world["vocab"])]
    return mix, w


def _toy_gm_decode(world, tables, h_seq, g_seq, args, score_rows=None):
    """GM over positions 0..s-2 (label at p identifies token p+1);
    position 0 stays with the SIP init decode. z init = the SIP score
    vector at position p+1 (the paper's softmax-of-SIP initialization)."""
    seq_len = world["seq_len"]
    scores = score_rows or [_toy_centroid_scores(h_seq[p], tables[p])
                            for p in range(seq_len)]
    pred = [max(range(world["vocab"]),
                key=lambda j: scores[0][j])] + [0] * (seq_len - 1)
    losses = []
    for p in range(seq_len - 1):
        tok, loss = toy_gm_position(h_seq[p], g_seq[p],
                                    world["C_pub"][p], scores[p + 1],
                                    args.toy_gm_steps, args.gm_lr,
                                    args.gm_beta, args.gm_tau)
        pred[p + 1] = tok
        losses.append(loss)
    return pred, sum(losses) / max(1, len(losses))


def _toy_sm_decode(world, h_seq, init_ids, args):
    """SM over all positions, warm-started from init_ids through emb()
    (the paper's 'pass the sentence through emb() to obtain e'' init)."""
    pred, losses = [], []
    for p in range(world["seq_len"]):
        tok, loss = toy_sm_position(h_seq[p], world["A_pub"][p],
                                    world["E"][init_ids[p]], world["E"],
                                    args.toy_sm_steps, args.sm_lr,
                                    args.sm_wd)
        pred.append(tok)
        losses.append(loss)
    return pred, sum(losses) / max(1, len(losses))


def toy_opt_cell(args, sigma, seed):
    """One toy BRE cell: the paper's optimization arms (b / f / bf, plus
    the NaMoE variants with --namoe) against the SIP init and the
    majority control, scored with the shared plumbing."""
    world = toy_opt_world(args.seed + 619 * seed, vocab=args.vocab,
                          dim=args.hidden, seq_len=args.seq_len,
                          n_victim=args.toy_opt_seqs,
                          n_aux=args.toy_aux_seqs, sigma=sigma,
                          ft_gap=args.toy_ft_gap)
    tables = toy_train_centroids(world["aux"], world["h_aux"], args.vocab,
                                 args.seq_len, args.hidden)
    control = majority_control_token(world["aux"])
    n_seeds = len(args.seeds)
    sip, dec_b, dec_f, dec_bf = [], [], [], []
    gm_losses, sm_losses = [], []
    for hs, gs in zip(world["h_victim"], world["g_victim"]):
        scores = [_toy_centroid_scores(hs[p], tables[p])
                  for p in range(world["seq_len"])]
        sip.append([max(range(world["vocab"]), key=lambda j: s[j])
                    for s in scores])
        pb, lb = _toy_gm_decode(world, tables, hs, gs, args,
                                score_rows=scores)
        dec_b.append(pb)
        gm_losses.append(lb)
        pf, lf = _toy_sm_decode(world, hs, sip[-1], args)
        dec_f.append(pf)
        pbf, lbf = _toy_sm_decode(world, hs, pb, args)
        dec_bf.append(pbf)
        sm_losses.append(lbf)

    def bundle(preds):
        return {**score_sequences(preds, world["victim"], control,
                                  n_seeds=n_seeds),
                "bleu": round(corpus_bleu(preds, world["victim"]), 6)}

    cell = {"sigma": sigma, "seed": seed, "arm": "bisr-opt-bf",
            "ft_gap": args.toy_ft_gap,
            "gm": {"iters": args.toy_gm_steps, "lr": args.gm_lr,
                   "beta": args.gm_beta, "tau": args.gm_tau,
                   "final_loss_mean": round(sum(gm_losses)
                                            / max(1, len(gm_losses)), 6)},
            "sm": {"iters": args.toy_sm_steps, "lr": args.sm_lr,
                   "wd": args.sm_wd,
                   "final_loss_mean": round(sum(sm_losses)
                                            / max(1, len(sm_losses)), 6)},
            **bundle(dec_bf)}
    cell["sip_init"] = bundle(sip)
    cell["opt_b"] = bundle(dec_b)
    cell["opt_f"] = bundle(dec_f)
    if args.namoe:
        import math
        experts, gating, rec = toy_namoe_fit(world, list(args.namoe_sigmas),
                                             args.gating_steps, 0.05,
                                             args.seed + 917 * seed)
        dec_nm, dec_nm_bf = [], []
        gw_sum = [0.0] * len(args.namoe_sigmas)
        gw_n = 0
        for hs, gs in zip(world["h_victim"], world["g_victim"]):
            posts = []
            for p in range(world["seq_len"]):
                mix, w = _toy_namoe_mixture(world, experts, gating,
                                            list(args.namoe_sigmas),
                                            hs[p], p)
                posts.append(mix)
                for k in range(len(w)):
                    gw_sum[k] += w[k]
                gw_n += 1
            nm = [max(range(world["vocab"]), key=lambda j: s[j])
                  for s in posts]
            dec_nm.append(nm)
            scores = [[math.log(max(x, 1e-12)) for x in s] for s in posts]
            pb, _ = _toy_gm_decode(world, tables, hs, gs, args,
                                   score_rows=scores)
            pbf, _ = _toy_sm_decode(world, hs, pb, args)
            dec_nm_bf.append(pbf)
        rec["gating_weights_mean"] = [round(x / max(1, gw_n), 6)
                                      for x in gw_sum]
        cell["namoe"] = bundle(dec_nm)
        cell["namoe_bf"] = bundle(dec_nm_bf)
        cell["namoe_training"] = rec
    return cell


def _run_toy(args, ev):
    out = artifacts.make_artifact(
        SCHEMA,
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "adaptation": True,
         "vocab": args.vocab, "dim": args.hidden, "seq_len": args.seq_len,
         "toy_seqs": args.toy_seqs, "toy_aux_seqs": args.toy_aux_seqs,
         "toy_sigmas": list(args.toy_sigmas), "seeds": args.seeds,
         "candidates": args.candidates, "beam_width": args.beam_width,
         "toy_opt": bool(args.toy_opt),
         "toy_ft_gap": args.toy_ft_gap if args.toy_opt else None,
         "namoe": bool(args.namoe) if args.toy_opt else False,
         "learned_inverter": "nearest-centroid (per-position, aux fit)",
         "baseline": "position-independent cosine-NN over E"},
        THREAT_MODEL + " Toy tier: synthetic position-dependent seam with "
        "a toy backward channel; measures selection/beam/scoring/gate "
        "plumbing and the bidirectional >= forward-only ordering, not a "
        "real model." + (" --toy-opt adds the toy BRE optimization arms "
                         "(toy GM Eq. 9 / SM Eq. 7-8 with analytic "
                         "gradients; toy NaMoE with --namoe) on a second "
                         "synthetic world with a toy fine-tuning gap."
                         if args.toy_opt else ""),
        interpretation=INTERPRETATION, evidence=ev)
    cells = []
    opt_cells = []
    for sigma in args.toy_sigmas:
        for seed in args.seeds:
            cell = {"experiment": EXPERIMENT_ID,
                    "submode": "bidirectional-vs-forward-only"}
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
            print(f"[bisr-toy] sigma={sigma} seed={seed}: "
                  f"acc={cell['seq_accuracy']} sip_fwd_only="
                  f"{cell['sip_fwd_only']['seq_accuracy']} baseline_nn="
                  f"{cell['baseline_nn']['seq_accuracy']} control="
                  f"{cell['control_accuracy']} gate={cell['gate']['verdict']}")
            if args.toy_opt:
                ocell = {"experiment": EXPERIMENT_ID,
                         "submode": "bre-optimization-arms"}
                try:
                    ocell.update(toy_opt_cell(args, sigma, seed))
                except Exception as e:
                    journal_error(args.output, EXPERIMENT_ID, ocell, e)
                    continue
                artifacts.append_jsonl(args.output, ocell)
                if ev is not None:
                    ev.event("cell_result", ocell)
                out["results"].append(ocell)
                opt_cells.append(ocell)
                line = (f"[bisr-toy-opt] sigma={sigma} seed={seed}: "
                        f"bf={ocell['seq_accuracy']} b="
                        f"{ocell['opt_b']['seq_accuracy']} f="
                        f"{ocell['opt_f']['seq_accuracy']} sip_init="
                        f"{ocell['sip_init']['seq_accuracy']} control="
                        f"{ocell['control_accuracy']}")
                if "namoe" in ocell:
                    line += (f" namoe={ocell['namoe']['seq_accuracy']}"
                             f" namoe_bf="
                             f"{ocell['namoe_bf']['seq_accuracy']}")
                print(line)
    for sigma in sorted(set(c["sigma"] for c in cells)):
        vals = [c for c in cells if c["sigma"] == sigma]
        out["summary"].append({
            "experiment": EXPERIMENT_ID, "sigma": sigma,
            "seq_accuracy_mean": sum(c["seq_accuracy"] for c in vals)
            / len(vals),
            "sip_fwd_only_accuracy_mean":
                sum(c["sip_fwd_only"]["seq_accuracy"] for c in vals)
                / len(vals),
            "baseline_nn_accuracy_mean":
                sum(c["baseline_nn"]["seq_accuracy"] for c in vals)
                / len(vals),
            "control_accuracy_mean": sum(c["control_accuracy"] for c in vals)
            / len(vals),
            "bleu_mean": sum(c["bleu"] for c in vals) / len(vals),
            "gate_verdicts": [c["gate"]["verdict"] for c in vals],
            "n_cells": len(vals)})
    for sigma in sorted(set(c["sigma"] for c in opt_cells)):
        vals = [c for c in opt_cells if c["sigma"] == sigma]
        summ = {
            "experiment": EXPERIMENT_ID, "sigma": sigma,
            "arm": "bisr-opt-bf",
            "seq_accuracy_mean": sum(c["seq_accuracy"] for c in vals)
            / len(vals),
            "opt_b_accuracy_mean":
                sum(c["opt_b"]["seq_accuracy"] for c in vals) / len(vals),
            "opt_f_accuracy_mean":
                sum(c["opt_f"]["seq_accuracy"] for c in vals) / len(vals),
            "sip_init_accuracy_mean":
                sum(c["sip_init"]["seq_accuracy"] for c in vals)
                / len(vals),
            "control_accuracy_mean": sum(c["control_accuracy"] for c in vals)
            / len(vals),
            "gate_verdicts": [c["gate"]["verdict"] for c in vals],
            "n_cells": len(vals)}
        if any("namoe" in c for c in vals):
            with_nm = [c for c in vals if "namoe" in c]
            summ["namoe_accuracy_mean"] = \
                sum(c["namoe"]["seq_accuracy"] for c in with_nm) \
                / len(with_nm)
            summ["namoe_bf_accuracy_mean"] = \
                sum(c["namoe_bf"]["seq_accuracy"] for c in with_nm) \
                / len(with_nm)
        out["summary"].append(summ)
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


def _inverter_class():
    """The SipInverter nn.Module, built lazily so module import and --help
    never touch torch (repo rule)."""
    import torch.nn as nn

    class SipInverter(nn.Module):
        """SIP-style learned inversion model (scoped adaptation): a small
        per-position MLP mapping a (layer-normed) seam row + position
        embedding to vocabulary logits. BiSR's SIP trains a substantially
        larger inverter on a substantial auxiliary corpus; this pilot-scale
        decoder is the same IDEA at a budget this repository can afford."""

        def __init__(self, vocab, seam_dim, seq_len, d_model=128):
            super().__init__()
            self.pos_emb = nn.Embedding(seq_len, seam_dim)
            self.net = nn.Sequential(
                nn.Linear(seam_dim, d_model), nn.GELU(),
                nn.Linear(d_model, vocab))
            nn.init.normal_(self.pos_emb.weight, std=0.02)
            nn.init.normal_(self.net[2].weight, std=0.02)

        def forward(self, h):
            import torch.nn.functional as F
            b, s, H = h.shape
            pos = torch.arange(s, device=h.device).unsqueeze(0)
            x = F.layer_norm(h, (H,)) + self.pos_emb(pos)
            return self.net(x)

    return SipInverter


def _namoe_class():
    """The NaMoEInverter nn.Module, built lazily (repo torch rule)."""
    import torch.nn as nn

    class NaMoEInverter(nn.Module):
        """Noise-adaptive Mixture-of-Experts inversion model (paper SS3.3,
        Fig. 6 left): K experts, each a SipInverter-style trunk producing
        features; a per-row gating network over the (layer-normed,
        position-embedded) seam row; expert features combined by the
        gating weights and passed through the shared output layer. Stage 1
        trains each expert (+ shared embeddings/output) on its own noise
        scale with the gating frozen; stage 2 trains only the gating on
        random scales with the experts frozen (the paper's two-stage
        recipe, Fig. 6 right)."""

        def __init__(self, vocab, seam_dim, seq_len, d_model=128,
                     n_experts=4):
            super().__init__()
            self.n_experts = n_experts
            self.pos_emb = nn.Embedding(seq_len, seam_dim)
            self.experts = nn.ModuleList([
                nn.Sequential(nn.Linear(seam_dim, d_model), nn.GELU())
                for _ in range(n_experts)])
            self.gating = nn.Linear(seam_dim, n_experts)
            self.out = nn.Linear(d_model, vocab)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
            nn.init.normal_(self.out.weight, std=0.02)

        def _features(self, h):
            import torch.nn.functional as F
            b, s, H = h.shape
            pos = torch.arange(s, device=h.device).unsqueeze(0)
            x = F.layer_norm(h, (H,)) + self.pos_emb(pos)
            return x

        def forward(self, h, return_weights=False):
            import torch.nn.functional as F
            x = self._features(h)
            feats = torch.stack([e(x) for e in self.experts], -1)
            w = F.softmax(self.gating(x), dim=-1)
            mix = (feats * w.unsqueeze(-2)).sum(-1)
            logits = self.out(mix)
            return (logits, w) if return_weights else logits

        def expert_logits(self, h, k):
            """Expert k's decode through the shared output layer (stage-1
            noise-aware training of expert k)."""
            return self.out(self.experts[k](self._features(h)))

        def gating_params(self):
            return self.gating.parameters()

        def non_gating_params(self):
            for name, p in self.named_parameters():
                if not name.startswith("gating."):
                    yield p

    return NaMoEInverter


def _embed_forward(split, Z):
    """The PUBLIC bottom segment on a continuous embedding sequence Z
    [n, seq, d] — the f_btm-bar' of paper Eq. 7 (bottom blocks above the
    embedding layer). GPT-2 family: drop(Z + wpe(pos)) -> blocks[0:cut);
    Mistral/Llama family (rotary inside the blocks): Z -> blocks[0:cut).
    Mirrors each splitter's head_forward exactly."""
    if hasattr(split, "wpe"):
        from .seam_inversion import _head_embed_forward
        return _head_embed_forward(split, Z)
    return split._run_blocks(Z, split.ranges[0], None)


def _bre_gradient_match(split, h_obs, g_obs, init_scores, *, iters, lr,
                        beta, tau, opt_batch, work=None, upstream=False):
    """BRE backward enhancement (paper SS3.2.2, Eq. 9): optimize dummy
    soft labels y' = softmax(z/tau) per position so the MIMIC TOP's dummy
    gradient nabla_{h*} L_dmy(rest_forward(h*), y') matches the OBSERVED
    bwd seam gradient under beta*||.||_2 + (1-beta)*||.||_1 (the TAG-style
    combined distance). h* is the observed fwd seam, FIXED (the paper's
    known-forward-activation trait — only y' is optimized, which is what
    eliminates the sequence-order issue). y' is initialized from the SIP
    output softmax (the paper's prescription: z_0 = tau * sip_logits).
    g_obs must already be rescaled to the per-sequence mean-loss
    convention (the caller multiplies by the capture batch size).
    Auto-regressive alignment: y'_t identifies input token t+1; position
    0 has no label gradient and keeps the caller's init token. AdamW
    (0.9, 0.999), no weight decay (Table 1). Returns (ids [n, s],
    info)."""
    import torch.nn.functional as F
    n, s, H = h_obs.shape
    V = init_scores.shape[-1]
    device = h_obs.device
    out = init_scores.argmax(-1).clone()
    losses = []
    for i0 in range(0, n, opt_batch):
        h = h_obs[i0:i0 + opt_batch].detach().clone().requires_grad_(True)
        g = g_obs[i0:i0 + opt_batch].detach()
        if upstream:
            z = torch.softmax(init_scores[i0:i0 + opt_batch].detach() / tau, -1).clone()
        else:
            z = (init_scores[i0:i0 + opt_batch, 1:].detach() * tau).clone()
        z.requires_grad_(True)
        opt = torch.optim.AdamW([z], lr=lr, betas=(0.9, 0.999),
                                eps=1e-6 if upstream else 1e-8,
                                weight_decay=0.01 if upstream else 0.0)
        loss = None
        for _ in range(max(1, iters)):
            logits = split.rest_forward(h).float()
            y = torch.softmax(z, dim=-1)[:, 1:] if upstream else torch.softmax(z / tau, dim=-1)
            if upstream:
                loss_dmy = F.cross_entropy(logits[:, :-1].contiguous().view(-1, V),
                                           y.contiguous().view(-1, V))
            else:
                loss_dmy = -(y * F.log_softmax(logits[:, :-1], dim=-1)
                             ).sum(-1).mean(-1).sum()
            (dummy_g,) = torch.autograd.grad(loss_dmy, h,
                                             create_graph=True)
            diff = dummy_g - g
            loss = (sum(beta * d.square().sum() + (1.0 - beta) * d.abs().sum() for d in diff)
                    if upstream else beta * diff.norm(p=2) + (1.0 - beta) * diff.norm(p=1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if work is not None:
                work.head_forward_calls += 1
                work.head_forward_token_positions += h.shape[0] * s
                work.head_backward_calls += 1
                work.head_backward_token_positions += h.shape[0] * s
        losses.append(float(loss.detach()) if loss is not None else None)
        with torch.no_grad():
            if upstream:
                out[i0:i0 + opt_batch] = z.argmax(-1)
            else:
                dec = torch.softmax(z / tau, dim=-1).argmax(-1)
                out[i0:i0 + opt_batch, 1:] = dec
    return out, {"gm_final_loss_mean":
                 round(sum(x for x in losses if x is not None)
                       / max(1, len([x for x in losses if x is not None])),
                       6),
                 "gm_iters": iters, "gm_lr": lr, "gm_beta": beta,
                 "gm_tau": tau, "algorithm_profile": "splitllm-c376590" if upstream else "legacy-paper-variant"}


def _bre_smashed_match(split, E, h_obs, init_ids, *, iters, lr, wd,
                       opt_batch, work=None, upstream=False):
    """BRE forward enhancement (paper SS3.2.1, Eq. 7-8): optimize a dummy
    embedding e' (continuous relaxation, initialized by passing the
    current reconstruction through emb() — the paper's warm start) so the
    PUBLIC bottom blocks reproduce the observed seam under the COSINE
    objective (the paper's choice over L2, Appendix D.5), then decode to
    the per-position Euclidean-nearest embedding row (Eq. 8). AdamW
    (0.9, 0.999) with the Table-1 learning rate and weight decay. Returns
    (ids [n, s], info)."""
    import torch.nn.functional as F
    n, s, _ = h_obs.shape
    E = E.detach()
    out = init_ids.clone()
    losses = []
    for i0 in range(0, n, opt_batch):
        e = E[init_ids[i0:i0 + opt_batch]].detach().clone()
        e.requires_grad_(True)
        tgt = h_obs[i0:i0 + opt_batch].detach()
        opt = torch.optim.AdamW([e], lr=lr, betas=(0.9, 0.999),
                                eps=1e-6 if upstream else 1e-8, weight_decay=wd)
        loss = None
        for _ in range(max(1, iters)):
            h_hat = _embed_forward(split, e)
            loss = (sum(1.0 - F.cosine_similarity(x, y, dim=-1).mean() for x, y in zip(h_hat, tgt))
                    if upstream else (1.0 - F.cosine_similarity(h_hat, tgt, dim=-1)).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            if work is not None:
                work.head_forward_calls += 1
                work.head_forward_token_positions += e.shape[0] * s
                work.head_backward_calls += 1
                work.head_backward_token_positions += e.shape[0] * s
        losses.append(float(loss.detach()) if loss is not None else None)
        with torch.no_grad():
            out[i0:i0 + opt_batch] = _bre_decode(e, E, upstream=upstream)
    return out, {"sm_final_loss_mean":
                 round(sum(x for x in losses if x is not None)
                       / max(1, len([x for x in losses if x is not None])),
                       6),
                 "sm_iters": iters, "sm_lr": lr, "sm_wd": wd,
                 "algorithm_profile": "splitllm-c376590" if upstream else "legacy-paper-variant"}


def _bre_decode(embeddings, table, *, upstream=False):
    """Explicit source choice: pinned upstream dot product or legacy Euclidean."""
    if upstream:
        return torch.softmax(torch.matmul(embeddings, table.transpose(0, 1)), -1).argmax(-1)
    return torch.cdist(embeddings, table).argmin(-1)


def _bre_optimize(split, E, rows, grads, fwd_scores, init_ids, args, work):
    """The BRE second stage on the picked sequences, per --opt-arm:
    'b' = gradient matching only; 'f' = smashed-data matching from the
    SIP reconstruction; 'bf' = the paper's sequential BiSR(b+f): GM then
    SM warm-started from the GM result. grads may be None (inference
    mode) — only 'f' is allowed there (checked by the caller). Returns
    (ids [n, s], info dict)."""
    info = {"opt_arm": args.opt_arm}
    cur = init_ids
    if args.opt_arm in ("b", "bf"):
        cur, gm_info = _bre_gradient_match(
            split, rows, grads, fwd_scores, iters=args.gm_iters,
            lr=args.gm_lr, beta=args.gm_beta, tau=args.gm_tau,
            opt_batch=args.opt_batch, work=work)
        cur[:, 0] = init_ids[:, 0]
        info["gm"] = gm_info
    if args.opt_arm in ("f", "bf"):
        cur, sm_info = _bre_smashed_match(
            split, E, rows, cur, iters=args.sm_iters, lr=args.sm_lr,
            wd=args.sm_wd, opt_batch=args.opt_batch, work=work)
        info["sm"] = sm_info
    return cur, info


def _aux_cache_dir(args):
    return args.aux_cache or (args.inverter_pt + ".auxcache")


def _stream_aux_shards(args, split, device):
    """Stream --aux-file into the shard cache. Layout per shard:
    shard-%05d.ids.npy int32 [c, seq_len] + shard-%05d.h.npy float32
    [c, seq_len, H] (c = --aux-chunk-seqs, last shard may be short);
    journal.json carries the run-config fingerprint and the sealed shard
    records. RESUME-SAFE: a fingerprint match skips sealed shards and
    continues after the last one; a mismatch is a descriptive error
    (never silent reuse); unjournaled shard files are overwritten. The
    fail-closed disjointness assertion runs BEFORE any tokenization.
    Returns (shard list, disjointness, stream_record)."""
    import numpy as np
    if not args.model:
        raise SystemExit(f"[{EXPERIMENT_ID}] --aux-file needs --model "
                         "(its tokenizer)")
    disjointness = assert_aux_disjoint(args.aux_file, args.aux_manifest,
                                       args.pilot_corpus)
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit(f"[{EXPERIMENT_ID}] --aux-file requires "
                         "transformers") from None
    tok = AutoTokenizer.from_pretrained(args.model)
    cache = _aux_cache_dir(args)
    os.makedirs(cache, exist_ok=True)
    fingerprint = {"format": AUX_CACHE_FORMAT,
                   "aux_sha256": disjointness["aux_sha256"],
                   "model": args.model, "split": str(args.split),
                   "seq_len": args.seq_len, "aux_seqs": args.aux_seqs,
                   "chunk_seqs": args.aux_chunk_seqs}
    journal_path = os.path.join(cache, "journal.json")
    journal = None
    if os.path.exists(journal_path):
        journal = json.load(open(journal_path))
        if journal.get("fingerprint") != fingerprint:
            raise SystemExit(
                f"[{EXPERIMENT_ID}] aux cache fingerprint mismatch at "
                f"{cache}: the cached shards were produced for a "
                "different aux corpus/model/split/seq-len/budget — "
                "delete the cache or pass --aux-cache elsewhere (never "
                "silently reusing stale shards)")
        print(f"[bisr-train] resuming aux cache at {cache}: "
              f"{len(journal['shards'])} sealed shards")
    else:
        journal = {"fingerprint": fingerprint, "shards": [],
                   "windows_done": 0}

    def seal_journal():
        tmp = journal_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(journal, f, indent=1)
        os.replace(tmp, journal_path)

    shards = list(journal["shards"])
    done = journal["windows_done"]
    had_sealed = bool(shards)
    articles = 0
    target = args.aux_seqs
    chunk = args.aux_chunk_seqs
    t0 = time.time()
    if done < target:
        buf = torch.empty(0, dtype=torch.long)
        win_buf = []
        shard_idx = len(shards)
        articles = 0
        stop = False

        def flush():
            """Seal the current win_buf as one shard (ids + h rows)."""
            nonlocal win_buf, shard_idx, done
            if not win_buf:
                return
            ids_np = torch.stack(win_buf).numpy().astype("int32")
            with torch.no_grad():
                h_np = torch.cat([
                    split.head_forward(
                        torch.from_numpy(ids_np[i:i + 64]).long()
                        .to(device)).cpu()
                    for i in range(0, len(ids_np), 64)]).numpy()
            ids_path = os.path.join(cache, f"shard-{shard_idx:05d}.ids.npy")
            h_path = os.path.join(cache, f"shard-{shard_idx:05d}.h.npy")
            with open(ids_path + ".tmp", "wb") as fh:
                np.save(fh, ids_np)
            os.replace(ids_path + ".tmp", ids_path)
            with open(h_path + ".tmp", "wb") as fh:
                np.save(fh, h_np)
            os.replace(h_path + ".tmp", h_path)
            shards.append({"ids": os.path.basename(ids_path),
                           "h": os.path.basename(h_path),
                           "windows": int(len(ids_np))})
            done += len(ids_np)
            journal["shards"] = shards
            journal["windows_done"] = done
            seal_journal()
            print(f"[bisr-train] aux shard {shard_idx} sealed: "
                  f"{len(ids_np)} windows ({done}/{target})")
            shard_idx += 1
            win_buf = []

        def ingest(texts):
            """Tokenize one article batch and cut windows (bounded)."""
            nonlocal buf, stop
            buf = torch.cat([buf, tok("\n".join(texts),
                                      return_tensors="pt")
                             .input_ids[0].cpu()])
            while buf.numel() >= args.seq_len \
                    and done + len(win_buf) < target:
                win_buf.append(buf[:args.seq_len].clone())
                buf = buf[args.seq_len:]
                if len(win_buf) >= chunk:
                    flush()
            if done + len(win_buf) >= target:
                stop = True

        batch_texts = []
        with open(args.aux_file) as f:
            for line in f:
                if stop:
                    break
                line = line.strip()
                if not line:
                    continue
                batch_texts.append(json.loads(line)["text"])
                articles += 1
                if len(batch_texts) >= 128:
                    ingest(batch_texts)
                    batch_texts = []
            if batch_texts and not stop:
                ingest(batch_texts)
        flush()
        if done < target:
            raise SystemExit(
                f"[{EXPERIMENT_ID}] --aux-file yielded only {done} "
                f"windows of {args.seq_len} tokens (< --aux-seqs "
                f"{target}) after streaming {articles} articles; the "
                "pilot path recycled short corpora, the aux-scale path "
                "refuses silent repetition — lower --aux-seqs or supply "
                "more disjoint text (Phase-B runbook: disjoint C4/Pile "
                "slice per the WS3 risk note)")
    stream = {"cache_dir": cache, "shards": len(shards),
              "windows": done, "chunk_seqs": chunk,
              "articles_streamed": articles,
              "resumed": had_sealed,
              "wall_s": round(time.time() - t0, 3)}
    return shards, disjointness, stream


def _shard_samplers(cache, shards):
    """Memory-mapped shard readers: per-step random (shard, row) batch
    sampling without materializing the cache in RAM."""
    import numpy as np
    handles = []
    for sh in shards:
        ids = np.load(os.path.join(cache, sh["ids"]), mmap_mode="r")
        h = np.load(os.path.join(cache, sh["h"]), mmap_mode="r")
        handles.append({"ids": ids, "h": h, "n": sh["windows"]})
    total = sum(x["n"] for x in handles)
    return handles, total


def _run_train(args, ev):
    if not args.inverter_pt:
        raise SystemExit(f"[{EXPERIMENT_ID}] --phase train needs "
                         "--inverter-pt (path to SAVE the trained "
                         "inverter checkpoint)")
    if not args.model:
        raise SystemExit(f"[{EXPERIMENT_ID}] --phase train needs --model "
                         "(public base weights: E + the public head for "
                         "auxiliary training forwards, split at --split)")
    if args.train_steps < 1 or args.aux_seqs < 1 or args.batch_size < 1:
        raise SystemExit(f"[{EXPERIMENT_ID}] --train-steps / --aux-seqs "
                         "/ --batch-size must all be >= 1")
    if args.namoe and not args.namoe_sigmas:
        raise SystemExit(f"[{EXPERIMENT_ID}] --namoe needs at least one "
                         "expert scale in --namoe-sigmas")
    if args.namoe and args.gating_steps < 1:
        raise SystemExit(f"[{EXPERIMENT_ID}] --gating-steps must be >= 1")
    if args.aux_chunk_seqs < 1:
        raise SystemExit(f"[{EXPERIMENT_ID}] --aux-chunk-seqs must be >= 1")
    if args.knowledge != "public" or args.head_pt or args.head_sha256:
        raise SystemExit("SIP auxiliary training uses public weights only")
    if args.decoder == "gru" and args.namoe:
        raise SystemExit("GRU primary path excludes NaMoE")
    if args.decoder == "gru" and args.train_epochs < 1:
        raise SystemExit("GRU requires positive --train-epochs")
    require_torch(EXPERIMENT_ID)
    import torch.nn.functional as F

    from ..splitbench import SplitPlan, _n_layers_of, build_split
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        model = _load_hf_model(args.model, device)
        E = model.get_input_embeddings().weight.detach()
        split = build_split(model, SplitPlan.parse(
            args.split, n_layers=_n_layers_of(model)))
        V = E.shape[0]
        stream = None

        if args.aux_file:
            shards, disjointness, stream = _stream_aux_shards(
                args, split, device)
            handles, n = _shard_samplers(_aux_cache_dir(args), shards)
            s = args.seq_len
            H = int(handles[0]["h"].shape[-1])
            import numpy as np
            all_ids = np.concatenate(
                [np.asarray(hnd["ids"]) for hnd in handles])
            control = int(np.bincount(all_ids.ravel()).argmax())
            del all_ids

            def batch_fn(b, g):
                hb = np.empty((b, s, H), dtype="float32")
                idb = np.empty((b, s), dtype="int64")
                ks = torch.randint(0, len(handles), (b,),
                                   generator=g).tolist()
                for i, k in enumerate(ks):
                    hnd = handles[k]
                    r = int(torch.randint(0, hnd["n"], (1,),
                                          generator=g))
                    hb[i] = hnd["h"][r]
                    idb[i] = hnd["ids"][r]
                return (torch.from_numpy(idb).to(device),
                        torch.from_numpy(hb).to(device))
        else:
            g0 = torch.Generator().manual_seed(args.seed + 7331)
            ids = torch.randint(0, V, (args.aux_seqs, args.seq_len),
                                generator=g0)
            with torch.no_grad():
                h_aux = torch.cat([
                    split.head_forward(ids[i:i + 64].to(device)).cpu()
                    for i in range(0, ids.shape[0], 64)])
            n, s, H = h_aux.shape
            control = majority_control_token(ids.tolist())
            disjointness = {"aux_corpus": None,
                            "public_source": "synthetic-uniform",
                            "disjoint": "by construction (synthetic draws)",
                            "checked_against": [], "skipped_missing": []}

            def batch_fn(b, g):
                idx = torch.randint(0, n, (b,), generator=g)
                return ids[idx].to(device), h_aux[idx].to(device)

        batch_g = torch.Generator().manual_seed(args.seed + 31337)
        noise_g = torch.Generator().manual_seed(args.seed + 2718)
        t0 = time.time()
        loss_rec = {}
        train_rec = {"model": args.model, "split": args.split,
                     "aux_file": args.aux_file, "aux_seqs": n,
                     "train_steps": args.train_steps,
                     "disjointness": disjointness,
                     "pilot_scale": args.aux_file is None,
                     "aux_stream": stream}
        if args.namoe:
            sigmas = [float(x) for x in args.namoe_sigmas]
            NaMoE = _namoe_class()
            inv = NaMoE(vocab=V, seam_dim=H, seq_len=s,
                        d_model=args.d_model,
                        n_experts=len(sigmas)).to(device)
            for k, s_k in enumerate(sigmas):
                opt = torch.optim.AdamW(inv.non_gating_params(),
                                        lr=args.lr)
                for step in range(args.train_steps):
                    idb, hb = batch_fn(args.batch_size, batch_g)
                    hb = hb + torch.randn(hb.shape, generator=noise_g,
                                          device=device) * s_k
                    loss = F.cross_entropy(
                        inv.expert_logits(hb, k).reshape(-1, V),
                        idb.reshape(-1))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    loss_rec = {"stage": 1, "expert": k, "sigma": s_k,
                                "step": step,
                                "loss": round(float(loss.item()), 6)}
                    if step % 50 == 0 or step == args.train_steps - 1:
                        print(f"[bisr-train] namoe stage1 expert {k} "
                              f"(sigma={s_k}) step {step + 1}/"
                              f"{args.train_steps} loss="
                              f"{loss_rec['loss']}")
            max_s = max(sigmas)
            opt = torch.optim.AdamW(inv.gating_params(), lr=args.lr)
            for step in range(args.gating_steps):
                idb, hb = batch_fn(args.batch_size, batch_g)
                s_draw = float(torch.rand(1, generator=noise_g)) * max_s
                hb = hb + torch.randn(hb.shape, generator=noise_g,
                                      device=device) * s_draw
                loss = F.cross_entropy(inv(hb).reshape(-1, V),
                                       idb.reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_rec = {"stage": 2, "sigma_draw": round(s_draw, 6),
                            "step": step,
                            "loss": round(float(loss.item()), 6)}
                if step % 20 == 0 or step == args.gating_steps - 1:
                    print(f"[bisr-train] namoe stage2 gating step "
                          f"{step + 1}/{args.gating_steps} "
                          f"(sigma_draw={loss_rec['sigma_draw']}) "
                          f"loss={loss_rec['loss']}")
            train_rec["namoe"] = {
                "expert_sigmas": sigmas, "gating_steps": args.gating_steps,
                "noise": "additive Gaussian seam noise (this adaptation's "
                         "N_s; the paper uses embedding-d_XP / "
                         "smashed-data-DP scales)",
                "stage1_steps_per_expert": args.train_steps}
            ckpt_format = CKPT_FORMAT_NAMOE
            arch = {"vocab": V, "seam_dim": H, "seq_len": s,
                    "d_model": args.d_model, "n_experts": len(sigmas)}
        elif args.decoder == "gru":
            from ..bisr_fidelity import gru_class, train_gru
            arch = {"vocab": V, "seam_dim": H, "seq_len": s,
                    "hidden_size": 256, "dropout": 0.1}
            inv = gru_class()(**arch).to(device)

            def fetch(indices):
                if not args.aux_file:
                    return ids[indices].to(device), h_aux[indices].to(device)
                id_rows, h_rows = [], []
                for index in indices:
                    for handle in handles:
                        if index < handle["n"]:
                            id_rows.append(np.array(handle["ids"][index], copy=True))
                            h_rows.append(np.array(handle["h"][index], copy=True))
                            break
                        index -= handle["n"]
                    else:
                        raise ValueError("Auxiliary index outside sealed shards")
                return (torch.from_numpy(np.stack(id_rows)).to(device),
                        torch.from_numpy(np.stack(h_rows)).to(device))

            loss_rec = train_gru(inv, fetch, n, epochs=args.train_epochs,
                                 batch_size=args.batch_size, seed=args.seed, lr=args.lr)
            train_rec["epoch_training"] = loss_rec
            train_rec["train_steps"] = loss_rec["steps"]
            train_rec["decoder"] = "gru"
            ckpt_format = CKPT_FORMAT_GRU
        else:
            Inverter = _inverter_class()
            inv = Inverter(vocab=V, seam_dim=H, seq_len=s,
                           d_model=args.d_model).to(device)
            opt = torch.optim.AdamW(inv.parameters(), lr=args.lr)
            for step in range(args.train_steps):
                idb, hb = batch_fn(args.batch_size, batch_g)
                loss = F.cross_entropy(inv(hb).reshape(-1, V),
                                       idb.reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_rec = {"step": step,
                            "loss": round(float(loss.item()), 6)}
                if step % 50 == 0 or step == args.train_steps - 1:
                    print(f"[bisr-train] step {step + 1}/"
                          f"{args.train_steps} loss={loss_rec['loss']}")
            ckpt_format = CKPT_FORMAT
            arch = {"vocab": V, "seam_dim": H, "seq_len": s,
                    "d_model": args.d_model}
    wall = round(time.time() - t0, 3)

    train_rec["final_loss"] = loss_rec
    train_rec["wall_s"] = wall
    ckpt = {"format": ckpt_format, "arch": arch,
            "state_dict": {k: v.detach().cpu()
                           for k, v in inv.state_dict().items()},
            "control_token": control,
            "expert_sigmas": (list(args.namoe_sigmas)
                              if args.namoe else None),
            "train": train_rec}
    torch.save(ckpt, args.inverter_pt)

    rec = {"experiment": EXPERIMENT_ID, "phase": "train",
           "inverter_pt": args.inverter_pt, "arch": arch,
           "control_token": control,
           "disjointness": disjointness,
           "namoe": bool(args.namoe),
           "aux_stream": stream,
           "n_aux_seqs": n, "seq_len": s, "seam_dim": H, "vocab": V,
           "final_loss": loss_rec, "wall_s": wall}
    out = artifacts.make_artifact(
        SCHEMA,
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": False,
         "adaptation": True, "phase": "train", "model": args.model,
         "split": args.split, "inverter_pt": args.inverter_pt,
         "aux_file": args.aux_file, "aux_seqs": args.aux_seqs,
         "aux_chunk_seqs": args.aux_chunk_seqs,
         "train_steps": train_rec["train_steps"], "batch_size": args.batch_size,
         "decoder": args.decoder,
         "train_epochs": args.train_epochs if args.decoder == "gru" else None,
         "lr": args.lr, "device": device,
         "namoe": bool(args.namoe),
         "namoe_sigmas": list(args.namoe_sigmas) if args.namoe else None,
         "gating_steps": args.gating_steps if args.namoe else None,
         **arch},
        THREAT_MODEL + " Train phase: the SIP-style inverter is fit on "
        "AUXILIARY text (asserted article-disjoint from the frozen and "
        "pilot corpora) through the PUBLIC head; the victim is never "
        "queried. " + ("NaMoE two-stage noise-aware training (paper "
                       "SS3.3) with additive Gaussian seam noise as the "
                       "perturbation mechanism. " if args.namoe else "") +
        ("Aux-scale streaming path (shard cache, resume-safe)."
         if args.aux_file else "Pilot-scale synthetic aux path."),
        interpretation=INTERPRETATION, evidence=ev)
    out["results"].append(rec)
    out["summary"].append(dict(rec))
    artifacts.append_jsonl(args.output, rec)
    if ev is not None:
        ev.event("cell_result", rec)
    artifacts.write_artifact(args.output, out, evidence=ev)
    print(f"[bisr-train] {train_rec['train_steps']} steps in {wall}s; "
          f"inverter -> {args.inverter_pt} "
          f"(disjointness: {disjointness['checked_against'] or 'synthetic'})")
    return 0


def _run_attack(args, ev):
    if not args.inverter_pt:
        raise SystemExit(f"[{EXPERIMENT_ID}] --phase attack needs "
                         "--inverter-pt (a trained inverter checkpoint "
                         "from --phase train)")
    if not args.model:
        raise SystemExit(f"[{EXPERIMENT_ID}] --phase attack needs --model "
                         "(public base weights: E + the public head for "
                         "beam verification)")
    if args.candidates < 2 or args.candidates % 2:
        raise SystemExit(f"[{EXPERIMENT_ID}] --candidates must be even "
                         "and >= 2")
    if args.beam_width < 1:
        raise SystemExit(f"[{EXPERIMENT_ID}] --beam-width must be >= 1")
    if args.opt_arm in ("b", "bf") and args.mode != "training":
        raise SystemExit(f"[{EXPERIMENT_ID}] --opt-arm {args.opt_arm} "
                         "needs the bwd seam gradient (training wire "
                         "surface); inference mode allows --opt-arm f "
                         "only")
    if args.opt_batch < 1:
        raise SystemExit(f"[{EXPERIMENT_ID}] --opt-batch must be >= 1")
    if args.opt_arm != "none" and (
            args.gm_iters < 1 or args.sm_iters < 1 or args.gm_tau <= 0
            or not 0.0 <= args.gm_beta <= 1.0):
        raise SystemExit(f"[{EXPERIMENT_ID}] BRE hyperparameters: "
                         "--gm-iters/--sm-iters >= 1, --gm-tau > 0, "
                         "0 <= --gm-beta <= 1")
    require_torch(EXPERIMENT_ID)
    from ..reconstruction_baselines import (MeteredHead, backward_scores,
                                            beam_reconstruct, candidate_ids)
    from ..splitbench import SplitPlan, _n_layers_of, build_split
    from .seam_inversion import (_cosine_nn_torch, _load_scored_sequences,
                                 _load_scored_sequences_joint)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.inverter_pt, map_location="cpu")
    if not isinstance(ckpt, dict) or ckpt.get("format") not in (
            CKPT_FORMAT, CKPT_FORMAT_NAMOE, CKPT_FORMAT_GRU):
        raise SystemExit(f"[{EXPERIMENT_ID}] {args.inverter_pt} is not a "
                         f"{CKPT_FORMAT}/{CKPT_FORMAT_NAMOE} inverter "
                         "checkpoint (--phase train writes one)")
    is_namoe = ckpt["format"] == CKPT_FORMAT_NAMOE
    if args.namoe and not is_namoe:
        raise SystemExit(f"[{EXPERIMENT_ID}] --namoe asked but "
                         f"{args.inverter_pt} is a plain SIP checkpoint "
                         f"({CKPT_FORMAT}); retrain with --namoe")
    arch = ckpt["arch"]
    if is_namoe:
        NaMoE = _namoe_class()
        inv = NaMoE(**arch)
    elif ckpt["format"] == CKPT_FORMAT_GRU:
        from ..bisr_fidelity import gru_class
        inv = gru_class()(**arch)
    else:
        Inverter = _inverter_class()
        inv = Inverter(**arch)
    inv.load_state_dict(ckpt["state_dict"])
    inv = inv.to(device).eval()
    expert_sigmas = ckpt.get("expert_sigmas")
    control = ckpt.get("control_token")
    if control is None:
        raise SystemExit(f"[{EXPERIMENT_ID}] {args.inverter_pt} lacks the "
                         "attacker-side control token; retrain with this "
                         "framework's --phase train")

    bidirectional = args.mode == "training"
    if bidirectional:
        picked, seq_len, h_dim = _load_scored_sequences_joint(args)
    else:
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

    model = _load_hf_model(args.model, device)
    E = model.get_input_embeddings().weight.detach()
    grant = {"knowledge": "public", "downstream": "public"}
    if args.knowledge == "exact" or args.head_pt or args.head_sha256:
        from ..bisr_fidelity import granted_split
        split = granted_split(model, int(args.split), knowledge=args.knowledge,
                              head_path=args.head_pt, head_sha256=args.head_sha256)
        E = split.wte.weight.detach()
        grant = split.grant
    else:
        split = build_split(model, SplitPlan.parse(
            args.split, n_layers=_n_layers_of(model)))
        split.model.requires_grad_(False)
    head = MeteredHead(split)

    rows = torch.stack([p["h"] for p in picked]).to(device)
    gating_w = None
    with torch.no_grad():
        if is_namoe:
            fwd_scores, gating_w = inv(rows, return_weights=True)
        else:
            fwd_scores = inv(rows)
        if bidirectional:
            grads = torch.stack([p["g"] for p in picked]).to(device)
            bwd_scores = backward_scores(grads, E, head, control)
            cand = candidate_ids(fwd_scores, bwd_scores, args.candidates,
                                 "joint")
        else:
            grads = None
            bwd_scores = None
            cand = candidate_ids(fwd_scores, None, args.candidates,
                                 "forward")
        cand_fwd = candidate_ids(fwd_scores, None, args.candidates,
                                 "forward")
        pred, info = beam_reconstruct(head, rows, cand,
                                      beam_width=args.beam_width)
        pred_fwd, info_fwd = beam_reconstruct(head, rows, cand_fwd,
                                              beam_width=args.beam_width)
        base_pred = _cosine_nn_torch(rows.reshape(-1, h_dim),
                                     E.to(device)).reshape(len(picked),
                                                           seq_len)

    opt_pred, opt_info = None, None
    if args.opt_arm != "none":
        grads_gm = None
        if args.opt_arm in ("b", "bf"):
            batch_scale = torch.tensor(
                [p.get("batch", 1) for p in picked],
                dtype=torch.float32, device=device).reshape(-1, 1, 1)
            grads_gm = grads * batch_scale
        opt_pred, opt_info = _bre_optimize(
            split, E, rows, grads_gm, fwd_scores, fwd_scores.argmax(-1), args, head.work)

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
         "adaptation": True, "phase": "attack",
         "bidirectional": bidirectional,
         "capture_dir": args.capture_dir, "canonical_pt": args.canonical_pt,
         "inverter_pt": args.inverter_pt, "arch": arch,
         "namoe": is_namoe, "expert_sigmas": expert_sigmas,
         "model": args.model, "split": args.split,
         "candidates": args.candidates, "beam_width": args.beam_width,
         "samples": args.samples, "device": device,
         "opt_arm": args.opt_arm, "grant": grant,
         "bre_initialisation": "raw SIP; beam is separate",
         "bre": ({"gm_iters": args.gm_iters, "gm_lr": args.gm_lr,
                  "gm_beta": args.gm_beta, "gm_tau": args.gm_tau,
                  "sm_iters": args.sm_iters, "sm_lr": args.sm_lr,
                  "sm_wd": args.sm_wd, "opt_batch": args.opt_batch}
                 if args.opt_arm != "none" else None),
         "gating_weights_mean": (
             [round(float(x), 6) for x in gating_w.mean(dim=(0, 1))]
             if gating_w is not None else None)},
        (THREAT_MODEL if args.knowledge == "public" else
         "Single-seam adaptation observing forward activations and, in training, "
         "backward gradients. Exact exported client head and embeddings are "
         "granted; downstream mimic weights remain public. Canonical tokens "
         "are scoring-only. The exact head is hash-bound in config.grant.") + (" Bidirectional arm active (fwd + bwd seam)."
                        if bidirectional else
                        " Inference mode has no bwd channel; the "
                        "bidirectional arm degrades to forward-only.") +
        (" BRE optimization arm '" + args.opt_arm + "' active (paper "
         "SS3.2): the observed bwd gradient is matched through the "
         "PUBLIC mimic top (the semi-white-box fine-tuning gap, and any "
         "defense on the wire, are part of the measured mismatch)."
         if args.opt_arm != "none" else "") +
        (" NaMoE checkpoint: the forward scores are the gating-weighted "
         "expert mixture (paper SS3.3)." if is_namoe else ""),
        interpretation=INTERPRETATION, evidence=ev)

    cells = []
    for key in order:
        idx = groups[key]
        cell = {"experiment": EXPERIMENT_ID, "arm": "bisr",
                "submode": ("learned-fwd+bwd-beam" if bidirectional
                            else "learned-fwd-beam"),
                "mode": args.mode, "step": key[0], "epoch": key[1],
                "candidates": args.candidates,
                "beam_width": args.beam_width}
        try:
            preds = pred[idx].cpu().tolist()
            trues = [picked[i]["true"] for i in idx]
            cell["predictions"] = {
                "beam": preds, "sip": fwd_scores[idx].argmax(-1).cpu().tolist(),
                "sample_indices": idx,
                "bre": opt_pred[idx].cpu().tolist() if opt_pred is not None else None}
            cell.update(score_sequences(preds, trues, control))
            cell["bleu"] = round(corpus_bleu(preds, trues), 6)
            fp = pred_fwd[idx].cpu().tolist()
            cell["sip_fwd_only"] = {
                **score_sequences(fp, trues, control),
                "bleu": round(corpus_bleu(fp, trues), 6)}
            bp = base_pred[idx].cpu().tolist()
            cell["baseline_nn"] = {
                **score_sequences(bp, trues, control),
                "bleu": round(corpus_bleu(bp, trues), 6)}
            if opt_pred is not None:
                op = opt_pred[idx].cpu().tolist()
                cell["bisr_opt"] = {
                    **score_sequences(op, trues, control),
                    "bleu": round(corpus_bleu(op, trues), 6)}
                cell["opt"] = opt_info
        except Exception as e:
            journal_error(args.output, EXPERIMENT_ID, cell, e)
            continue
        artifacts.append_jsonl(args.output, cell)
        if ev is not None:
            ev.event("cell_result", cell)
        out["results"].append(cell)
        cells.append(cell)
        line = (f"[bisr-baseline] step={key[0]}: acc={cell['seq_accuracy']} "
                f"sip_fwd_only={cell['sip_fwd_only']['seq_accuracy']} "
                f"baseline_nn={cell['baseline_nn']['seq_accuracy']} "
                f"control={cell['control_accuracy']} "
                f"gate={cell['gate']['verdict']}")
        if "bisr_opt" in cell:
            line += f" bisr_opt={cell['bisr_opt']['seq_accuracy']}"
        print(line)

    if cells:
        n_corr = sum(c["n_correct"] for c in cells)
        n_tok = sum(c["n_tokens"] for c in cells)
        c_corr = sum(c["n_control_correct"] for c in cells)
        pooled_acc = n_corr / max(1, n_tok)
        pooled_ctl = c_corr / max(1, n_tok)
        all_trues = [p["true"] for p in picked]
        summary = {
            "experiment": EXPERIMENT_ID, "arm": "bisr",
            "bidirectional": bidirectional,
            "n_cells": len(cells), "n_tokens": n_tok,
            "seq_accuracy": round(pooled_acc, 6),
            "control_accuracy": round(pooled_ctl, 6),
            "bleu": round(corpus_bleu(pred.cpu().tolist(), all_trues), 6),
            "sip_fwd_only": {
                "seq_accuracy": round(
                    sum(c["sip_fwd_only"]["n_correct"] for c in cells)
                    / max(1, n_tok), 6)},
            "baseline_nn": {
                "seq_accuracy": round(
                    sum(c["baseline_nn"]["n_correct"] for c in cells)
                    / max(1, n_tok), 6)},
            "work": asdict(head.work),
            "gate": gate.verdict(pooled_acc, pooled_ctl, n_tok),
            "gate_verdicts": [c["gate"]["verdict"] for c in cells]}
        if opt_pred is not None:
            opt_corr = sum(c["bisr_opt"]["n_correct"] for c in cells
                           if "bisr_opt" in c)
            summary["bisr_opt"] = {
                "arm": args.opt_arm,
                "seq_accuracy": round(opt_corr / max(1, n_tok), 6),
                "bleu": round(corpus_bleu(opt_pred.cpu().tolist(),
                                          all_trues), 6),
                "info": opt_info}
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
            args.toy_aux_seqs = min(args.toy_aux_seqs, 200)
            args.toy_opt_seqs = min(args.toy_opt_seqs, 6)
            args.gating_steps = min(args.gating_steps, 40)
        return _run_toy(args, ev)
    if args.phase == "train":
        return _run_train(args, ev)
    if args.samples < 1:
        raise SystemExit(f"[{EXPERIMENT_ID}] --samples must be >= 1")
    if not args.capture_dir:
        raise SystemExit(f"[{EXPERIMENT_ID}] pass --toy, --phase train, or "
                         "--capture-dir D --canonical-pt C --inverter-pt F "
                         "--model M for the attack phase.")
    return _run_attack(args, ev)


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import json as _json
    import os
    import tempfile

    print("bisr-baseline toy core (pure python):")

    args = build_parser().parse_args(["--toy"])
    args.mode = "training"
    cell0 = toy_cell(args, 0.0, 0)
    check("noiseless toy: bidirectional beam recovers >85%",
          cell0["seq_accuracy"] > 0.85)
    check("noiseless toy: bidirectional >= forward-only",
          cell0["seq_accuracy"]
          >= cell0["sip_fwd_only"]["seq_accuracy"])
    check("toy cell carries control + gate + bleu + both diagnostics",
          all(k in cell0 for k in
              ("control_accuracy", "gate", "bleu", "sip_fwd_only",
               "baseline_nn"))
          and cell0["gate"]["verdict"] in ("PASS", "FAIL")
          and "gate" in cell0["sip_fwd_only"]
          and "gate" in cell0["baseline_nn"])
    noisy = toy_cell(args, 2.0, 0)
    check("heavy noise degrades the bidirectional decode (graceful)",
          noisy["seq_accuracy"] < cell0["seq_accuracy"])
    w1 = toy_world(0, vocab=60, dim=16, seq_len=12, n_victim=4, n_aux=20,
                   sigma=0.3)
    w2 = toy_world(0, vocab=60, dim=16, seq_len=12, n_victim=4, n_aux=20,
                   sigma=0.3)
    check("toy world is deterministic",
          w1["victim"] == w2["victim"] and w1["h_victim"] == w2["h_victim"]
          and w1["g_victim"] == w2["g_victim"])

    print("bisr-baseline toy BRE arms (pure python, real optimization):")
    oargs = build_parser().parse_args(["--toy", "--toy-opt",
                                       "--toy-opt-seqs", "6"])
    oargs.mode = "training"
    ow = toy_opt_world(0, vocab=60, dim=16, seq_len=12, n_victim=4,
                       n_aux=40, sigma=0.0)
    check("toy opt world is deterministic and carries the bwd channel",
          toy_opt_world(0, vocab=60, dim=16, seq_len=12, n_victim=4,
                        n_aux=40, sigma=0.0)["g_victim"]
          == ow["g_victim"]
          and len(ow["g_victim"][0]) == 12
          and all(abs(x) < 1e-12 for x in ow["g_victim"][0][-1]))
    ocell = toy_opt_cell(oargs, 0.0, 0)
    _gm_world = toy_opt_world(1234, vocab=60, dim=16, seq_len=12,
                              n_victim=8, n_aux=300, sigma=0.0)
    _gm_hit = sum(
        toy_gm_position(_gm_world["h_victim"][i][p],
                        _gm_world["g_victim"][i][p],
                        _gm_world["C_pub"][p], [0.0] * 60, 25, 0.09,
                        0.85, 1.2)[0] == _gm_world["victim"][i][p + 1]
        for i in range(3) for p in range(11))
    check("toy GM recovers the labels from a uniform init (real "
          "gradient matching, not init echo)", _gm_hit >= 31)
    check("toy opt cell: bf recovers >90% noiseless and dominates the "
          "control", ocell["seq_accuracy"] > 0.9
          and ocell["seq_accuracy"]
          > ocell["control_accuracy"] + 0.5)
    check("toy opt cell: SM does not degrade a good init",
          ocell["opt_f"]["seq_accuracy"]
          >= ocell["sip_init"]["seq_accuracy"] - 0.05)
    check("toy opt cell carries gm/sm info and per-arm bundles",
          "gm" in ocell and "sm" in ocell
          and "gate" in ocell["opt_b"] and "gate" in ocell["opt_f"])
    nargs = build_parser().parse_args(["--toy", "--toy-opt", "--namoe",
                                       "--toy-opt-seqs", "6",
                                       "--gating-steps", "40"])
    nargs.mode = "training"
    ncell = toy_opt_cell(nargs, 1.5, 0)
    check("toy NaMoE: cells carry experts/gating/mixture decodes",
          "namoe" in ncell and "namoe_bf" in ncell
          and ncell["namoe_training"]["expert_sigmas"]
          == [0.0, 0.1, 0.3, 1.0]
          and len(ncell["namoe_training"]["gating_weights_mean"]) == 4)
    check("toy NaMoE: mixture stays competitive with the noise-unaware "
          "SIP and far above the control under noise",
          ncell["namoe"]["seq_accuracy"]
          >= ncell["sip_init"]["seq_accuracy"] - 0.15
          and ncell["namoe_bf"]["seq_accuracy"]
          > ncell["control_accuracy"] + 0.3)

    with tempfile.TemporaryDirectory() as d:
        aux = os.path.join(d, "aux.jsonl")
        with open(aux, "w") as f:
            f.write(_json.dumps({"id": "aux-1", "text": "x"}) + "\n")
        manifest = os.path.join(d, "manifest.json")
        with open(manifest, "w") as f:
            _json.dump({"schema": "llm_attacker.validation_corpus.v1",
                        "selected_ids": {"evaluation": ["aux-1"]}}, f)
        try:
            assert_aux_disjoint(aux, manifest, None)
            check("contaminated auxiliary corpus is a descriptive error",
                  False)
        except ValueError as e:
            check("contaminated auxiliary corpus is a descriptive error",
                  "aux-1" in str(e) and "DISJOINT" in str(e))
        with open(manifest, "w") as f:
            _json.dump({"schema": "llm_attacker.validation_corpus.v1",
                        "selected_ids": {"evaluation": ["other-1"]}}, f)
        rec = assert_aux_disjoint(aux, manifest, None)
        check("clean auxiliary corpus passes with an audit record",
              rec["disjoint"] and "frozen_validation_manifest"
              in rec["checked_against"])
        rec = assert_aux_disjoint(aux, os.path.join(d, "missing.json"),
                                  os.path.join(d, "missing.jsonl"))
        check("missing references are recorded as skipped, not passed",
              rec["skipped_missing"] == ["frozen_validation_manifest",
                                         "pilot_corpus"])

    check("parser builds torch-less",
          build_parser().parse_args([]).phase == "attack")
    check("parser carries the BRE/NaMoE/aux-scale flags torch-less",
          build_parser().parse_args([]).opt_arm == "none"
          and build_parser().parse_args(
              ["--opt-arm", "bf", "--namoe"]).gm_beta == 0.85
          and list(build_parser().parse_args([]).namoe_sigmas)
          == [0.0, 0.1, 0.3, 1.0]
          and build_parser().parse_args([]).aux_chunk_seqs == 2048
          and build_parser().parse_args([]).sm_iters == 800
          and build_parser().parse_args(
              ["--toy-opt"]).toy_opt is True)
    ns = build_parser().parse_args(
        ["--capture-dir", "/nonexistent", "--canonical-pt", "/nonexistent",
         "--inverter-pt", "/nonexistent", "--model", "/nonexistent",
         "--opt-arm", "b"])
    ns.mode = "inference"
    try:
        run(ns)
        check("--opt-arm b on the inference surface is refused", False)
    except SystemExit as e:
        check("--opt-arm b on the inference surface is refused",
              "inference" in str(e) and "--opt-arm b" in str(e))
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
             "/nonexistent", "--inverter-pt", "/nonexistent",
             "--model", "/nonexistent"])
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
             "--output", os.path.join(d, "bisr.json")])
        ns.mode = "training"
        check("toy run end to end torch-free", run(ns) == 0)
        art = _json.load(open(os.path.join(d, "bisr.json")))
        check("toy artifact has sigma cells with gate verdicts",
              len(art["results"]) == 3
              and all(c["gate"]["verdict"] in ("PASS", "FAIL")
                      for c in art["results"]))
        check("toy cells carry the bidirectional-vs-forward pair",
              all("sip_fwd_only" in c and "baseline_nn" in c
                  for c in art["results"]))
        from ..evidence import verify_evidence
        good, problems = verify_evidence(
            os.path.join(d, "bisr.json.evidence.json"))
        check("evidence chain verifies", good and problems == [])

    with tempfile.TemporaryDirectory() as d:
        ns = build_parser().parse_args(
            ["--toy", "--quick", "--toy-opt", "--namoe", "--evidence",
             "--output", os.path.join(d, "bisr-opt.json")])
        ns.mode = "training"
        check("toy --toy-opt --namoe run end to end torch-free",
              run(ns) == 0)
        art = _json.load(open(os.path.join(d, "bisr-opt.json")))
        opt_cells = [c for c in art["results"]
                     if c.get("arm") == "bisr-opt-bf"]
        check("toy opt cells journaled with the beam cells",
              len(opt_cells) == 3 and len(art["results"]) == 6)
        check("toy opt cells carry bf/b/f/sip_init and the NaMoE arms",
              all("opt_b" in c and "opt_f" in c and "sip_init" in c
                  and "namoe" in c and "namoe_bf" in c
                  and "namoe_training" in c for c in opt_cells))
        check("toy opt summaries sit beside the beam summaries",
              any(s.get("arm") == "bisr-opt-bf" for s in art["summary"]))
        from ..evidence import verify_evidence
        good, problems = verify_evidence(
            os.path.join(d, "bisr-opt.json.evidence.json"))
        check("toy-opt evidence chain verifies", good and problems == [])

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
