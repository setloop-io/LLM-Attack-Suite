#!/usr/bin/env python3
"""splitbench — minimal split-LLM harness emitting OUR wire-capture schema.

This is the framework's own capture rig (no third-party harness): it splits
a causal LM at layer boundaries, runs a split fine-tune or split-inference
session locally, and tees every tensor crossing the seam into the repo's
standard wire_NNNNN.{pt,json} capture format (attacker/captures.py) that
every attack module consumes. One capture run feeds the whole attack
matrix (README "Efficiency design decision").

Seam-capture protocol (what attacks read):

  finetune mode (wire-training schema {session_id, mb_id, phase, step,
  epoch}): per optimizer step, one microbatch (no gradient accumulation —
  mb_id is always 0; the batch is [batch_size, seq_len]):

      h_star = head_forward(ids)
      h_wire = h_star + defense noise          (see below)
      capture phase="fwd"  = h_wire            (what the BODY receives)
      logits = rest_forward(h_wire); loss = next-token CE; loss.backward()
      capture phase="bwd"  = h_wire.grad       (what the body sends back)
      optimizer step over trainable params

  With --body-grad-capture, step 0 additionally dumps the per-parameter
  body-slice gradients (blocks after the seam + ln_f/norm + lm_head, as
  {name: grad.fp16()} in <capture_dir>/body_grads_step0.pt) — the
  batch-label-reconstruction observation surface
  (attacks/batch_label_recon.py).

  A step is captured when step % capture_every == 0; since 0 % k == 0 the
  FIRST batch of epoch 0 is always captured (label-inference attacks read
  the first training batch — hard requirement from the threat literature).
  epoch counts corpus wraps (single pass -> epoch 0 everywhere).

  inference mode (wire-inference schema {session_id, request_seq, phase,
  position, epoch}): per prompt (one request_seq each; batch_size prompts
  are served): phase="prefill" position=0 captures the whole-prompt seam
  hidden [1, P, H]; then steps greedy decode tokens, each capturing the
  single-token seam hidden [1, 1, H] at phase="decode", position=k (the
  k-th seam crossing of this request; prefill is crossing 0). Decode
  re-forwards the full prefix each step (use_cache=False): this is a
  capture rig, not a serving stack — the captured tensors are identical to
  a cached decode, the compute is not.

  epoch is 0 everywhere (no rotation ratchet in the harness; the schema
  also allows None).

Defense hook at the seam: --defense gaussian --sigma s adds N(0, s) noise
to the seam tensor before the body sees it. The capture records what the
BODY RECEIVES, i.e. POST-defense — the honest-but-curious wire observer
sees exactly the tensor that crosses. (Documented choice: pre-defense
captures would overstate the wire leak under an active defense.) sigma=0
skips the draw entirely, so gaussian@sigma=0 is bit-identical to "none".

Return-path defense hook (finetune only, composes with any --defense):
--grad-clip c clips the returned seam gradient's global L2 norm to c
(0 = off), --grad-noise s adds relative-RMS Gaussian noise (per-sequence
RMS(g) * s, the validation_controls noise_σ convention; 0 = off), and
--quant-grad quantizes the returned gradient at --quant-bits (quant
defense only). Composition order is frozen: clip -> noise -> quantize.
The bwd capture records what the VICTIM RECEIVES (post-hook) — the
return-path mirror of the forward post-defense doctrine. All-zero knobs
skip every transform, so the hook is bit-identical to off by default.
Serving has no gradient channel: infer() ignores these knobs (documented,
same as --server-attack).

The defense pack (--defense {sparsify,mid-lite,at-lite,santext,custext,
snd-lite}): honest lightweight analogues of the VFLAIR-LLM (arXiv:
2508.03097) defense suite — each is a real transform whose mechanism, and
whose gap to the full VFLAIR method, is documented below. Captures keep
recording the POST-defense tensor; per-run state (projection matrix, PCA
estimate, sanitizer) is built by the drivers via make_defense_state(cfg,
split) and stepped per microbatch.

  sparsify   (VFLAIR GradientSparsification analogue, applied to the fwd
             seam ACTIVATION — VFLAIR sparsifies gradients; same
             magnitude-pruning mechanism, different tensor): per seam row
             keep only the top-`--sparsity` fraction of entries by
             magnitude (kept = max(1, round(sparsity * d)), a per-row
             top-k, so the rule is invariant to the row's own activation
             scale); zero the rest. sparsity=1 keeps everything (== none).
  mid-lite   (MID information-bottleneck analogue — NOT learned MID: no
             mutual-information objective is optimized): project each seam
             row through a FIXED seeded random Gaussian map P [d, k]
             (entries N(0, 1/d), k = --bottleneck-dim <= d), a true
             dimensionality bottleneck. Projection-back convention: the
             body reads z = h @ P embedded back in d-space — the wire
             tensor is zeros(d) with z in the first k coordinates — so
             the attacker's wire view is the k-dim content embedded in
             d-dim space and the seam content is rank <= k by
             construction.
  at-lite    (adversarial-perturbation analogue — NOT adversarial
             training, no min-max optimization): maintain an online EMA
             estimate (momentum 0.9) of the seam covariance's top PCA
             direction v1 and shift the seam h -> h - --at-eps * v1,
             steering information away from the dominant invertible
             direction. v1 is updated with the current microbatch before
             the shift, so step 0 is already defended. at-eps=0 == none.
  santext    (SanText-style input sanitization): BEFORE embedding, each
             input token is replaced with probability --santext-p by a
             uniform draw from its --santext-k nearest embedding neighbors
             (cosine over the model's wte). canonical.pt keeps the
             ORIGINAL ids: the attacker's ground truth is the victim's
             original text while the wire carries sanitized content —
             that gap IS the defense's claim. The CE target stays the
             original next token (the utility cost shows up in the loss).
             Inference sanitizes the prompt once; decoded continuations
             are already sanitized-space tokens.
  custext    (toy-scale CusText custom-mapping analogue): like santext
             but a FIXED per-token mapping — token t maps to one member
             of its --santext-k cosine neighbors chosen by a
             (seed, token)-keyed RNG, identically at every occurrence and
             independent of batch order.
  snd-lite   (split-and-denoise analogue, REDUCED): split the seam into
             two additive shares (s1 ~ N(0, --snd-sigma^2), s2 = h - s1)
             and denoise s1 through a fixed seeded random smoothing
             matrix S (entries N(0, 1/d)): h_wire = s1 @ S + s2. Our
             threat model has ONE observed channel, so both shares are
             visible to the wire observer and the original scheme's
             separate-transport assumption is NOT modeled: the mechanism
             reduces to structured randomized smoothing of the seam.
             snd-sigma=0 == none.

  trained-bottleneck (attacker/trained_defenses.py — a LEARNED bottleneck,
             our in-repo re-implementation of the adversarial-bottleneck +
             MINE defense family, not the original defense project's): an
             MLP encoder H -> --tb-dim (GELU), minimax-trained against
             embedded token-identity / property / linear-reconstruction
             probes plus a MINE-style mutual-information penalty, in a
             PRE-PHASE on calibration data before victim finetuning
             (make_defense_state, driven below). The wire tensor follows
             the mid-lite projection-back convention (zeros(d) with z in
             the first --tb-dim coordinates) and the gradient passes
             straight through as if identity, so victim training continues
             end-to-end. The trained encoder checkpoint +
             llm_attacker.trained_bottleneck_provenance.v1 sidecar land in
             the capture dir (path recorded in the run artifact); the MINE
             value is logged as a READING, explicitly not a certificate.
             GPU-only by default; --tb-allow-cpu opts in to CPU training
             (tiny models only). This is the repo's first LEARNED
             bottleneck: mid-lite remains a fixed random projection.
  secret-mixup (attacker/trained_defenses.py — our in-repo re-implementation
             of the secret-mixup + decoy-correction defense family, not
             the original defense project's): release-time mixup
             wire = --mixup-lambda * h + (1 - lambda) * z_decoy, where
             z_decoy rows are REAL latent rows recycled from earlier
             training blocks of the same run (honest labels tracked),
             drawn per block WITHOUT replacement under CSPRNG-derived
             per-block seeds (cfg.seed ^ offset, sha256-chained). The
             pool starts EMPTY: the first block(s) go out UNDEFENDED and
             are flagged policy="undefended_empty_pool" in the
             llm_attacker.secret_mixup_provenance.v1 sidecar (saved with
             the drawn-decoy store in the capture dir at run end, path in
             the run artifact); pool capped at --mixup-pool rows, FIFO.
             Gradient: straight-through on the wire; the victim's
             returned-gradient path corrects for the decoy component via
             a parallel decoy forward ((g - (1-lambda)*g_decoy)/lambda)
             — EXACT for a linear cloud segment, APPROXIMATE under a
             nonlinear one (the defense framework's own caveat); off with
             --no-mixup-correction. No GPU pre-phase needed (no learned
             weights): utility_before/after are measured with the pool
             warmed from calibration batches. The capture converts to a
             view_privacy.v1 mixup bundle via
             `python -m attacker.trained_defenses convert-mixup`.
  replica-median (attacker/trained_defenses.py — our in-repo re-implementation
             of the K-redundant-channel + coordinate-median integrity defense
             family, not the original defense project's): --replica-count K
             (default 3; K=2 selects the disagreement/abort variant)
             INDEPENDENT body replicas — deep copies of the body slice with
             per-channel seeded gauge perturbations (per-channel seed =
             cfg.seed ^ offset ^ channel index; off with --no-replica-gauge)
             — run in parallel in-process during victim finetuning, trained
             by the shared median loss. Channel outputs are aggregated by
             COORDINATE MEDIAN (K >= 3; coordinate MEAN for K=2, whose
             no-majority policy flags an availability abort on disagreement
             above the tolerance) before the shared norm + lm_head. This is
             an INTEGRITY defense, not a seam transform: the seam tensor
             itself crosses unchanged; the per-block integrity journal
             (replica_median_journal.jsonl in the capture dir) records each
             channel's per-sample relative deviation from the group mean
             against --replica-tolerance (default 0.02: which replica
             deviated, magnitude, flagged or not). The honest-heterogeneity
             noise budget (the independent channels' own disagreement
             distribution) is measured on calibration data and logged in
             the llm_attacker.replica_median_provenance.v1 sidecar so
             attack-side tolerance matching is fair. Per captured block the
             driver writes K per-channel wires to
             <capture_dir>/channels/ch<k>/ (same wire-capture format,
             separate files) plus the aggregated median wire as the
             standard fwd capture (what the norm + lm_head consume), so
             BOTH the replica-integrity battery and the seam-reconstruction
             baselines run against the trained defender; utility_before/
             after recorded. MEMORY: K in-process body replicas multiply
             body-slice memory by ~K (plus K full-model deep copies at
             state build); at split 6 with --replica-count 3 this may
             exceed pilot GPU memory — the harness WARNS (never silently
             downsizes); lower --replica-count explicitly if needed. The
             capture converts to replica_observations.v1 / .replica_truth.v1
             via `python -m attacker.trained_defenses convert-replica`,
             which `replica-integrity --captures/--truth` consumes.

  quant      (scalar boundary quantization, the P5-style D0/D1 family —
             our in-repo executable analogue, NOT the external P5
             programme's implementation): signed symmetric per-sequence
             absmax quantization of the seam ACTIVATION to --quant-bits
             (2..16, default 8): per batch row, step = absmax(h) /
             (2^(bits-1) - 1); h_q = round(h/step).clamp(-limit, limit)
             * step — the same rule as validation_controls.
             transform_gradient's bits_N gradient condition, applied to
             the forward latent. All-zero rows stay all-zero (scale
             clamps to tiny). There is no "off" value: quantization
             always rounds (bits=16 is merely near-lossless at fp16 wire
             precision). With --quant-grad the RETURNED gradient is
             quantized at the same width (composed after --grad-clip and
             --grad-noise, see the return-path hook above). D0/D1-style
             cells: D0 = --defense quant --quant-bits 8 --quant-grad;
             D1 = --defense quant --quant-bits 4 --quant-grad
             --grad-clip 1.0 --grad-noise 0.1.

Trainable-range hook (finetune only): --trainable-range START,END freezes
every model parameter EXCEPT blocks [START, END) — frozen-backbone
segment training, the local executable analogue of a "span@start"
delegation cell (span = END - START trained at a seam placed at START via
--split START; the segment is the backbone's own blocks trained in place,
NOT a distilled surrogate replacement — see
experiments/crossed_span_feasibility.md for the exact semantics gap).
Bounds are checked against the model's block count at run time; mutually
exclusive with --lora-r > 0 (peft installs its own freeze). Default None
trains everything (unchanged historic behavior).

Malicious-server hook (finetune only): --server-attack passive-amplify
--server-attack-alpha a replaces the honest body computation inside
rest_forward with the crafted

    y = Body(h) + a * h ,        logits = lm_head(ln_f(y))

i.e. the server adds a scaled IDENTITY path around its own blocks before
the shared tail (ln_f + lm_head). The seam gradient the body sends back is
then dL/dh = a * J_lnf (p - onehot) E  +  (honest body backprop): an
a-scaled, per-row, near-isometric copy of the label/token direction that
skips the body blocks' attention mixing and MLP shear (the Boenisch/Fowl
malicious-server line transplanted to the split seam — the craft is
mathematically real: it is exactly what autograd produces for the
modified computation, not a doctored tensor). The alpha scalar is PUBLIC
(the deployed body is the attacker's own computation). Default "none"
keeps the honest rest_forward bit-identical. Inference serving has no
gradient channel, so infer() ignores this hook (documented; the attack it
feeds is attacks/malicious_server.py, training mode).

Wire format: tensors are saved fp16 CPU (halves capture size; attacks load
through surfaces.load_tensor which returns fp32). Sidecars are validated
against the surface provider (get_surface("wire-training"/"wire-inference"))
BEFORE anything is written — schema drift is a bug and raises.

Splitting: GPT-2 and the Mistral/Llama/Qwen3 family, via ARCH_REGISTRY.
For GPT-2 the 4D additive causal mask is built by hand
(torch.finfo(dtype).min on masked positions) exactly like transformers'
eager path does, and passed to each block with use_cache=False; this is
implementation-agnostic (eager adds it to the scores, sdpa accepts a
float attn_mask), so it survives transformers version drift. Mistral/
Llama blocks need more than a mask (per-block rotary position
embeddings), so MistralSplit mirrors the model's own forward exactly:
position_embeddings from model.model.rotary_emb(hidden, position_ids)
and the causal mask from transformers.masking_utils.create_causal_mask
(the verified transformers-5.15 contract; see MistralSplit's docstring).
Qwen3 shares that exact contract (its qk-norm lives INSIDE the decoder
layer, invisible at the seam), so the registry reuses the same splitter
— verified against transformers 5.15.0, see ARCH_REGISTRY.
For a 2-slice plan the "body" is blocks[boundary:] + ln_f/norm +
lm_head — the tail is folded in (SplitPlan still describes the slices).

LoRA (--lora-r > 0): peft wraps the body slice (server-side blocks only,
via layers_to_transform); peft then freezes the base weights, so the
optimizer trains exactly the body-slice adapters (client head frozen too
under peft defaults — the standard split-LoRA story). peft missing is a
descriptive error.

Usage:
    python3 -m attacker.splitbench --help                    # torch-free
    python3 -m attacker.splitbench --self-test               # torch-free
    python3 -m attacker.splitbench --config run.json [--evidence]
    python3 -m attacker.splitbench --model gpt2 --split 6 --mode finetune \
        --steps 200 --corpus-file wiki.txt --capture-dir caps/ \
        --output run.json --evidence
"""

import argparse
import dataclasses
import json
import math
import os
import random
import uuid

from . import artifacts
from .surfaces import get_surface

SCHEMA = "llm_attacker.splitbench.v1"

THREAT_MODEL = (
    "honest-but-curious wire observer on the split seam (local harness "
    "stand-in for the cloud side of a split deployment): captures record "
    "exactly what the server-side body receives (post-defense) and the "
    "gradients it returns; labeled local data, per the calibration "
    "attacks that consume wire captures")

TORCH_ERROR = (
    "[splitbench] a benchmark run requires torch (+ transformers for model "
    "loading); install the `torch` extra or use a torch venv. "
    "--help/--self-test and the pure-python core (SplitPlan, CaptureWriter, "
    "BenchConfig) work torch-less.")

_MODE_TO_WRITER = {"finetune": "training", "inference": "inference"}


class SplitPlan:
    """Layer-boundary plan for an n_layer transformer stack.

    SplitPlan.parse("6")   -> 2 slices: head[0:6] | body+tail[6:]
    SplitPlan.parse("4,8") -> 3 slices: head[0:4] | body[4:8] | tail[8:]

    boundaries are block indices where the cut happens (the boundary layer
    belongs to the next slice). n_layers may be left unbound at parse time
    and supplied later (slice_ranges(n) / the arch splitter bind it)."""

    def __init__(self, boundaries, n_layers=None):
        boundaries = tuple(boundaries)
        if not boundaries:
            raise ValueError("split plan needs at least one boundary "
                             "(e.g. '6' for a 2-slice head|body split)")
        for b in boundaries:
            if not isinstance(b, int) or isinstance(b, bool):
                raise ValueError(f"split boundary {b!r} is not an integer "
                                 "layer index")
            if b < 1:
                raise ValueError(f"split boundary {b} below 1: the head "
                                 "slice must own at least one block")
        if any(b2 <= b1 for b1, b2 in zip(boundaries, boundaries[1:])):
            raise ValueError(f"split boundaries must be strictly "
                             f"increasing, got {boundaries}")
        if n_layers is not None:
            if not isinstance(n_layers, int) or n_layers < 2:
                raise ValueError(f"n_layers must be an integer >= 2 to "
                                 f"host a split, got {n_layers!r}")
            self._check_bounds(boundaries, n_layers)
        self.boundaries = boundaries
        self.n_layers = n_layers

    @staticmethod
    def _check_bounds(boundaries, n_layers):
        for b in boundaries:
            if b > n_layers - 1:
                raise ValueError(
                    f"split boundary {b} out of range for n_layers="
                    f"{n_layers}: boundaries must lie in "
                    f"[1, {n_layers - 1}] (the tail slice must own at "
                    "least one block)")

    @classmethod
    def parse(cls, spec, n_layers=None):
        """'6' -> (6,); '4,8' -> (4, 8). Whitespace-tolerant."""
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError(f"empty split spec {spec!r}: expected a "
                             "comma-separated list of layer boundaries, "
                             "e.g. '6' or '4,8'")
        parts = []
        for piece in spec.split(","):
            piece = piece.strip()
            try:
                parts.append(int(piece))
            except ValueError:
                raise ValueError(f"invalid split boundary {piece!r} in "
                                 f"{spec!r}: expected an integer layer "
                                 "index") from None
        return cls(tuple(parts), n_layers=n_layers)

    @property
    def n_slices(self):
        return len(self.boundaries) + 1

    def slice_ranges(self, n_layers=None):
        """[(start, end), ...] block-index ranges per slice, head first.

        The head owns ranges[0]; middle slices (body) own ranges[1:-1];
        the last slice's blocks are ranges[-1] and the arch splitter folds
        ln_f + lm_head in behind them. For a 2-slice plan there are no
        middle slices and ranges[-1] IS the body+tail."""
        n = self.n_layers if n_layers is None else n_layers
        if n is None:
            raise ValueError("split plan is not bound to a layer count: "
                             "pass n_layers=... here or to "
                             "SplitPlan.parse(..., n_layers=...)")
        self._check_bounds(self.boundaries, n)
        pts = (0,) + self.boundaries + (n,)
        return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]

    def describe(self):
        names = (["head", "body+tail"] if self.n_slices == 2 else
                 ["head"] + ["body"] * (self.n_slices - 2) + ["tail"])
        if self.n_layers is None:
            cuts = ",".join(str(b) for b in self.boundaries)
            return f"{self.n_slices}-slice split at boundaries {cuts} " \
                   f"(unbound layer count)"
        parts = [f"{name}[{a}:{b}]" for name, (a, b)
                 in zip(names, self.slice_ranges())]
        return f"{self.n_slices}-slice split over {self.n_layers} " \
               f"layers: " + " | ".join(parts) + " (+ ln_f + lm_head " \
               "behind the last slice)"

    def __repr__(self):
        return f"SplitPlan(boundaries={self.boundaries}, " \
               f"n_layers={self.n_layers})"


def _parse_trainable_range(spec, n_blocks=None):
    """Parse a "START,END" trainable block range -> (start, end) tuple.

    Format-checked always; bounds-checked when n_blocks is given (the
    drivers bind the model's block count). 0 <= start < end <= n_blocks;
    end == n_blocks is legal (the range runs into the folded tail's
    blocks; ln_f/lm_head themselves stay frozen)."""
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(f"trainable_range {spec!r}: expected \"START,END\" "
                         "block indices, e.g. \"2,6\"")
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        raise ValueError(f"trainable_range {spec!r}: expected exactly two "
                         "comma-separated block indices, e.g. \"2,6\"")
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"trainable_range {spec!r}: block indices must "
                         "be integers") from None
    if start < 0:
        raise ValueError(f"trainable_range {spec!r}: START must be >= 0")
    if end <= start:
        raise ValueError(f"trainable_range {spec!r}: END must be > START "
                         f"({end} <= {start})")
    if n_blocks is not None and end > n_blocks:
        raise ValueError(f"trainable_range {spec!r}: END {end} exceeds "
                         f"the model's {n_blocks} blocks")
    return start, end


def _torch_saver(path, tensor):
    """Default .pt payload saver: lazy torch import, fp16 CPU wire format.

    The writer hands over a .tmp path and os.replace()s it into place, so
    savers must write exactly the path they are given."""
    import torch
    torch.save(tensor.detach().to(device="cpu", dtype=torch.float16), path)


_WRITER_SURFACES = {"training": "wire-training", "inference": "wire-inference"}


class CaptureWriter:
    """Writes wire_{seq:05d}.pt + wire_{seq:05d}.json pairs, crash-safe.

    Every sidecar is validated against the mode's surface provider BEFORE
    anything touches disk — drift is a bug and raises. Writes go .tmp ->
    os.replace (artifacts.py style), tensor first then sidecar, so a crash
    leaves at worst an orphan .pt that scan_captures never sees (it keys
    off wire_*.json). `saver` is the pluggable (path, tensor) payload
    writer; the default pins the fp16-CPU wire format, tests inject a fake
    to exercise the writer torch-free."""

    def __init__(self, dir, mode, session_id, saver=None):
        if mode not in _WRITER_SURFACES:
            raise ValueError(f"unknown capture mode {mode!r} (valid: "
                             f"{', '.join(sorted(_WRITER_SURFACES))})")
        self.dir = os.fspath(dir)
        os.makedirs(self.dir, exist_ok=True)
        self.mode = mode
        self.session_id = str(session_id)
        self.saver = _torch_saver if saver is None else saver
        self._provider = get_surface(_WRITER_SURFACES[mode])
        self._seq = 0
        self.count = 0
        self._closed = False

    def write_training(self, tensor, mb_id, phase, step, epoch):
        """One training-seam capture (phase 'fwd' seam hidden / 'bwd'
        seam gradient)."""
        if self.mode != "training":
            raise ValueError(f"write_training on a {self.mode!r} writer; "
                             "construct CaptureWriter(..., 'training', ...)")
        meta = {"session_id": self.session_id, "mb_id": int(mb_id),
                "phase": str(phase), "step": int(step), "epoch": epoch}
        return self._write(tensor, meta)

    def write_inference(self, tensor, request_seq, phase, position, epoch):
        """One inference-seam capture (phase 'prefill' / 'decode')."""
        if self.mode != "inference":
            raise ValueError(f"write_inference on a {self.mode!r} writer; "
                             "construct CaptureWriter(..., 'inference', ...)")
        meta = {"session_id": self.session_id,
                "request_seq": int(request_seq), "phase": str(phase),
                "position": int(position), "epoch": epoch}
        return self._write(tensor, meta)

    def _write(self, tensor, meta):
        """Validate-then-write one pair; returns the (meta, pt_path)
        record in scan_captures shape."""
        if self._closed:
            raise ValueError("capture writer is closed")
        base = os.path.join(self.dir, f"wire_{self._seq:05d}")
        meta = self._provider.validate(meta, base + ".json")
        tmp_pt = base + ".pt.tmp"
        self.saver(tmp_pt, tensor)
        os.replace(tmp_pt, base + ".pt")
        tmp_json = base + ".json.tmp"
        with open(tmp_json, "w") as f:
            json.dump(meta, f)
        os.replace(tmp_json, base + ".json")
        self._seq += 1
        self.count += 1
        return meta, base + ".pt"

    def close(self):
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


_DEFENSES = ("none", "gaussian", "sparsify", "mid-lite", "at-lite",
             "santext", "custext", "snd-lite", "trained-bottleneck",
             "secret-mixup", "replica-median", "pca-compress", "quant")


@dataclasses.dataclass
class BenchConfig:
    model: str = "gpt2"
    split: str = "6"
    mode: str = "finetune"
    steps: int = 100
    seq_len: int = 128
    batch_size: int = 4
    lr: float = 5e-5
    capture_every: int = 1
    defense: str = "none"
    sigma: float = 0.0
    sparsity: float = 0.1
    bottleneck_dim: int = 8
    at_eps: float = 0.1
    santext_p: float = 0.1
    santext_k: int = 8
    snd_sigma: float = 1.0
    tb_dim: int = 64
    tb_hidden: int = 256
    tb_probe_hidden: int = 128
    tb_minimax_steps: int = 200
    tb_mi_weight: float = 0.1
    tb_allow_cpu: bool = False
    tb_cotrain: bool = False
    tb_cotrain_interval: int = 8
    pca_dim: int = 64
    quant_bits: int = 8
    quant_grad: bool = False
    grad_clip: float = 0.0
    grad_noise: float = 0.0
    tb_checkpoint: str = None
    mixup_lambda: float = 0.75
    mixup_pool: int = 256
    mixup_correction: bool = True
    replica_count: int = 3
    replica_tolerance: float = 0.02
    replica_gauge: bool = True
    replica_gauge_scale: float = 1e-3
    lora_r: int = 0
    lora_alpha: float = 16.0
    trainable_range: str = None
    device: str = "cpu"
    dtype: str = "fp32"
    corpus_file: str = None
    dataset: str = "wikitext"
    capture_dir: str = "captures"
    output: str = None
    save_canonical: bool = False
    body_grad_capture: bool = False
    server_attack: str = "none"
    server_attack_alpha: float = 8.0
    seed: int = 1234

    @classmethod
    def from_json(cls, path):
        """Load + validate a config; unknown keys and bad values are
        descriptive ValueErrors."""
        try:
            with open(path) as f:
                raw = json.load(f)
        except OSError as e:
            raise ValueError(f"cannot read config {path}: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"config {path} is not valid JSON: {e}") from e
        if not isinstance(raw, dict):
            raise ValueError(f"config {path} must be a JSON object, got "
                             f"{type(raw).__name__}")
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - valid)
        if unknown:
            raise ValueError(f"config {path}: unknown keys {unknown} "
                             f"(valid: {sorted(valid)})")
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def validate(self):
        """Raise a descriptive ValueError on any bad field; returns self."""

        def bad(msg):
            raise ValueError(f"BenchConfig: {msg}")

        def need_int(name, lo):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < lo:
                bad(f"{name} must be an integer >= {lo}, got {v!r}")

        def need_float(name, lo, strict=False):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                bad(f"{name} must be a number, got {v!r}")
            if v < lo or (strict and v == lo):
                op = ">" if strict else ">="
                bad(f"{name} must be {op} {lo}, got {v!r}")

        if self.mode not in _MODE_TO_WRITER:
            bad(f"unknown mode {self.mode!r} (valid: "
                f"{', '.join(sorted(_MODE_TO_WRITER))})")
        if self.defense not in _DEFENSES:
            bad(f"unknown defense {self.defense!r} (valid: "
                f"{', '.join(_DEFENSES)})")
        if self.server_attack not in ("none", "passive-amplify"):
            bad(f"unknown server_attack {self.server_attack!r} (valid: "
                "none, passive-amplify)")
        if self.device not in ("cpu", "cuda"):
            bad(f"unknown device {self.device!r} (valid: cpu, cuda)")
        if self.dtype not in ("fp32", "bf16"):
            bad(f"unknown dtype {self.dtype!r} (valid: fp32, bf16)")
        for name in ("model", "split", "dataset", "capture_dir"):
            if not isinstance(getattr(self, name), str) \
                    or not getattr(self, name):
                bad(f"{name} must be a non-empty string, "
                    f"got {getattr(self, name)!r}")
        if self.corpus_file is not None and not isinstance(self.corpus_file,
                                                           str):
            bad(f"corpus_file must be a path string or null, "
                f"got {self.corpus_file!r}")
        if not isinstance(self.body_grad_capture, bool):
            bad(f"body_grad_capture must be a boolean, "
                f"got {self.body_grad_capture!r}")
        need_int("steps", 1)
        need_int("seq_len", 2)
        need_int("batch_size", 1)
        need_int("capture_every", 1)
        need_int("lora_r", 0)
        need_int("seed", 0)
        need_float("lr", 0.0, strict=True)
        need_float("sigma", 0.0)
        need_float("lora_alpha", 0.0, strict=True)
        need_float("server_attack_alpha", 0.0)
        need_float("sparsity", 0.0, strict=True)
        if isinstance(self.sparsity, (int, float)) \
                and not isinstance(self.sparsity, bool) \
                and self.sparsity > 1.0:
            bad(f"sparsity must be <= 1 (fraction KEPT), got "
                f"{self.sparsity!r}")
        need_int("bottleneck_dim", 1)
        need_float("at_eps", 0.0)
        need_float("santext_p", 0.0)
        if isinstance(self.santext_p, (int, float)) \
                and not isinstance(self.santext_p, bool) \
                and self.santext_p > 1.0:
            bad(f"santext_p must be <= 1 (a probability), got "
                f"{self.santext_p!r}")
        need_int("santext_k", 1)
        need_float("snd_sigma", 0.0)
        need_int("tb_dim", 1)
        need_int("tb_hidden", 1)
        need_int("tb_probe_hidden", 1)
        need_int("tb_minimax_steps", 1)
        need_float("tb_mi_weight", 0.0)
        if not isinstance(self.tb_allow_cpu, bool):
            bad(f"tb_allow_cpu must be a boolean, got {self.tb_allow_cpu!r}")
        if not isinstance(self.tb_cotrain, bool):
            bad(f"tb_cotrain must be a boolean, got {self.tb_cotrain!r}")
        need_int("tb_cotrain_interval", 1)
        need_int("pca_dim", 1)
        if self.tb_checkpoint is not None \
                and not isinstance(self.tb_checkpoint, str):
            bad(f"tb_checkpoint must be a path string or null, "
                f"got {self.tb_checkpoint!r}")
        need_float("mixup_lambda", 0.0, strict=True)
        if isinstance(self.mixup_lambda, (int, float)) \
                and not isinstance(self.mixup_lambda, bool) \
                and self.mixup_lambda > 1.0:
            bad(f"mixup_lambda must be <= 1 (a convex weight), got "
                f"{self.mixup_lambda!r}")
        need_int("mixup_pool", 1)
        if not isinstance(self.mixup_correction, bool):
            bad(f"mixup_correction must be a boolean, got "
                f"{self.mixup_correction!r}")
        need_int("replica_count", 2)
        need_float("replica_tolerance", 0.0, strict=True)
        if not isinstance(self.replica_gauge, bool):
            bad(f"replica_gauge must be a boolean, got "
                f"{self.replica_gauge!r}")
        need_float("replica_gauge_scale", 0.0, strict=True)
        need_int("quant_bits", 2)
        if isinstance(self.quant_bits, int) \
                and not isinstance(self.quant_bits, bool) \
                and self.quant_bits > 16:
            bad(f"quant_bits must be <= 16, got {self.quant_bits!r}")
        if not isinstance(self.quant_grad, bool):
            bad(f"quant_grad must be a boolean, got {self.quant_grad!r}")
        if self.quant_grad and self.defense != "quant":
            bad(f"quant_grad requires defense=\"quant\" (it quantizes the "
                f"returned gradient at quant_bits), got defense="
                f"{self.defense!r}")
        need_float("grad_clip", 0.0)
        need_float("grad_noise", 0.0)
        if self.trainable_range is not None:
            _parse_trainable_range(self.trainable_range)
            if self.mode != "finetune":
                bad(f"trainable_range only applies to finetune runs "
                    f"(nothing trains in inference), got mode={self.mode!r}")
            if self.lora_r > 0:
                bad("trainable_range is mutually exclusive with lora_r "
                    "(peft installs its own freeze)")
        SplitPlan.parse(self.split)
        return self


class Gpt2Split:
    """Split view of a GPT2LMHeadModel bound to a SplitPlan.

    head_forward(input_ids, attention_mask=None) -> hidden
        wte + wpe + drop + blocks[ranges[0]]
    body_forward(hidden, layer_range, attention_mask=None) -> hidden
        blocks[start:end] (identity on an empty range)
    tail_forward(hidden, attention_mask=None) -> logits
        blocks[ranges[-1]] + ln_f + lm_head — for a 2-slice plan this is
        the whole rest of the model (the tail is folded into the body)
    rest_forward(hidden, attention_mask=None) -> logits
        every middle slice + tail_forward; head_forward + rest_forward
        reproduces the unsplit model exactly (eval mode, see module doc
        for the mask choice)
    """

    def __init__(self, model, plan):
        tf = model.transformer
        n = len(tf.h)
        if plan.n_layers is not None and plan.n_layers != n:
            raise ValueError(f"split plan is bound to {plan.n_layers} "
                             f"layers but the model has {n}")
        self.model = model
        self.plan = plan
        self.ranges = plan.slice_ranges(n)
        self.wte = tf.wte
        self.wpe = tf.wpe
        self.drop = tf.drop
        self.blocks = tf.h
        self.ln_f = tf.ln_f
        self.lm_head = model.lm_head

    @property
    def device(self):
        return self.wte.weight.device

    @staticmethod
    def _causal_mask(attention_mask, batch, seq, dtype, device):
        """4D additive causal (+padding) mask, the eager-transformers
        convention: torch.finfo(dtype).min on disallowed positions. Works
        for the eager and sdpa attention implementations."""
        import torch
        minv = torch.finfo(dtype).min
        causal = torch.full((seq, seq), minv, dtype=dtype,
                            device=device).triu(1)
        mask = causal[None, None].expand(batch, 1, seq, seq)
        if attention_mask is not None:
            pad = (1.0 - attention_mask.to(dtype)) * minv
            mask = mask + pad[:, None, None, :]
        return mask

    @staticmethod
    def _hidden(out):
        """Block outputs are tuples across transformers versions; tolerate
        a bare tensor or a ModelOutput too."""
        if isinstance(out, tuple):
            return out[0]
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        return out

    def _run_blocks(self, hidden, layer_range, attention_mask):
        start, end = layer_range
        mask = self._causal_mask(attention_mask, hidden.shape[0],
                                 hidden.shape[1], hidden.dtype,
                                 hidden.device)
        for blk in self.blocks[start:end]:
            hidden = self._hidden(blk(hidden, attention_mask=mask,
                                      use_cache=False))
        return hidden

    def head_forward(self, input_ids, attention_mask=None):
        import torch
        b, s = input_ids.shape
        pos = torch.arange(0, s, dtype=torch.long,
                           device=input_ids.device).unsqueeze(0)
        hidden = self.drop(self.wte(input_ids) + self.wpe(pos))
        return self._run_blocks(hidden, self.ranges[0], attention_mask)

    def body_forward(self, hidden, layer_range, attention_mask=None):
        return self._run_blocks(hidden, layer_range, attention_mask)

    def tail_forward(self, hidden, attention_mask=None):
        hidden = self._run_blocks(hidden, self.ranges[-1], attention_mask)
        return self.lm_head(self.ln_f(hidden))

    def rest_forward(self, hidden, attention_mask=None):
        for layer_range in self.ranges[1:-1]:
            hidden = self.body_forward(hidden, layer_range, attention_mask)
        return self.tail_forward(hidden, attention_mask)

    def body_param_prefixes(self):
        """named-parameter prefixes of the server-side slice downstream
        of the seam (blocks after the head + ln_f + lm_head); matched by
        containment so peft-wrapped names classify too."""
        n = len(self.blocks)
        return tuple(f"transformer.h.{i}."
                     for i in range(self.ranges[1][0], n)) \
            + ("transformer.ln_f.", "lm_head.")


class MistralSplit:
    """Split view of a Mistral/Llama-family *ForCausalLM bound to a
    SplitPlan — the same interface and accessor protocol as Gpt2Split.

    head_forward(input_ids, attention_mask=None) -> hidden
        embed_tokens + blocks[ranges[0]] — no wpe/drop: rotary position
        embeddings are per-block (computed below), never added at input
    body_forward(hidden, layer_range, attention_mask=None) -> hidden
        blocks[start:end] (identity on an empty range)
    tail_forward(hidden, attention_mask=None) -> logits
        blocks[ranges[-1]] + norm + lm_head (tail folded in for 2 slices)
    rest_forward(hidden, attention_mask=None) -> logits
        every middle slice + tail_forward; head_forward + rest_forward
        reproduces the unsplit model exactly (verified 0.0 max-abs logit
        diff vs the unsplit forward on tiny Mistral/Llama configs, eager
        and sdpa, with and without a padding mask)

    Accessor protocol (shared with Gpt2Split so seam-head consumers such
    as seam_inversion.save_head_pt-style code keep working uniformly):
    .wte is the input-embedding module (here embed_tokens), .blocks the
    decoder stack, .ln_f the final norm (here model.model.norm; .norm is
    the family-native alias), .lm_head the unembedding.

    transformers 5.15 call contract (verified against the installed
    5.15.0 — do not guess around this):
      DecoderLayer.forward(hidden_states, attention_mask=None,
          position_ids=None, past_key_values=None, use_cache=False,
          position_embeddings=None, **kwargs) -> torch.Tensor
      i.e. a BARE tensor return (unwrapped tolerantly anyway), and
      position_embeddings is REQUIRED by the attention sublayer:
      position_embeddings = model.model.rotary_emb(hidden, position_ids)
      -> (cos, sin). The causal mask is what the model itself builds:
      transformers.masking_utils.create_causal_mask(config=...,
      inputs_embeds=hidden, attention_mask=..., past_key_values=None,
      position_ids=...) — None for sdpa on an unpadded batch (sdpa then
      takes its is_causal path), else the 4-D additive float mask. On
      older transformers without masking_utils we fall back to the
      hand-built 4-D additive mask (the Gpt2Split convention).

    Qwen3 (verified against the same installed 5.15.0):
    Qwen3DecoderLayer.forward has the IDENTICAL call contract above
    (same signature, same bare-tensor return), the core exposes the same
    embed_tokens/layers/norm/rotary_emb layout, and qk-norm lives inside
    the decoder layer (invisible at the seam) — head->rest reproduces
    the unsplit Qwen3 model with 0.0 max-abs logit diff on a tiny random
    config, so ARCH_REGISTRY registers this class under "qwen3" too.
    """

    def __init__(self, model, plan):
        core = model.model
        n = len(core.layers)
        if plan.n_layers is not None and plan.n_layers != n:
            raise ValueError(f"split plan is bound to {plan.n_layers} "
                             f"layers but the model has {n}")
        self.model = model
        self.plan = plan
        self.ranges = plan.slice_ranges(n)
        self.wte = core.embed_tokens
        self.embed_tokens = core.embed_tokens
        self.blocks = core.layers
        self.ln_f = core.norm
        self.norm = core.norm
        self.lm_head = model.lm_head
        self.rotary_emb = core.rotary_emb

    @property
    def device(self):
        return self.wte.weight.device

    @staticmethod
    def _position_ids(hidden):
        import torch
        return torch.arange(hidden.shape[1], dtype=torch.long,
                            device=hidden.device).unsqueeze(0)

    def _causal_mask(self, attention_mask, hidden, position_ids):
        """The mask the unsplit model would see (module docstring)."""
        try:
            from transformers.masking_utils import create_causal_mask
        except ImportError:
            return Gpt2Split._causal_mask(
                attention_mask, hidden.shape[0], hidden.shape[1],
                hidden.dtype, hidden.device)
        import inspect
        parameters = inspect.signature(create_causal_mask).parameters
        embedding_key = "input_embeds" if "input_embeds" in parameters else "inputs_embeds"
        kwargs = {embedding_key: hidden, "config": self.model.config,
                  "attention_mask": attention_mask, "past_key_values": None,
                  "position_ids": position_ids}
        if "cache_position" in parameters:
            kwargs["cache_position"] = position_ids[0]
        return create_causal_mask(**kwargs)

    @staticmethod
    def _hidden(out):
        """5.15 decoder layers return a bare tensor; tolerate tuples /
        ModelOutputs from other versions anyway."""
        if isinstance(out, tuple):
            return out[0]
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        return out

    def _run_blocks(self, hidden, layer_range, attention_mask):
        start, end = layer_range
        position_ids = self._position_ids(hidden)
        pos_emb = self.rotary_emb(hidden, position_ids)
        mask = self._causal_mask(attention_mask, hidden, position_ids)
        for blk in self.blocks[start:end]:
            hidden = self._hidden(blk(hidden, attention_mask=mask,
                                      position_ids=position_ids,
                                      use_cache=False,
                                      position_embeddings=pos_emb))
        return hidden

    def head_forward(self, input_ids, attention_mask=None):
        hidden = self.embed_tokens(input_ids)
        return self._run_blocks(hidden, self.ranges[0], attention_mask)

    def body_forward(self, hidden, layer_range, attention_mask=None):
        return self._run_blocks(hidden, layer_range, attention_mask)

    def tail_forward(self, hidden, attention_mask=None):
        hidden = self._run_blocks(hidden, self.ranges[-1], attention_mask)
        return self.lm_head(self.norm(hidden))

    def rest_forward(self, hidden, attention_mask=None):
        for layer_range in self.ranges[1:-1]:
            hidden = self.body_forward(hidden, layer_range, attention_mask)
        return self.tail_forward(hidden, attention_mask)

    def body_param_prefixes(self):
        """named-parameter prefixes of the server-side slice downstream
        of the seam (blocks after the head + norm + lm_head); matched by
        containment so peft-wrapped names classify too."""
        n = len(self.blocks)
        return tuple(f"model.layers.{i}."
                     for i in range(self.ranges[1][0], n)) \
            + ("model.norm.", "lm_head.")


ARCH_REGISTRY = {"gpt2": Gpt2Split,
                 "mistral": MistralSplit, "llama": MistralSplit,
                 "qwen3": MistralSplit}


def get_arch_splitter(arch):
    try:
        return ARCH_REGISTRY[arch]
    except KeyError:
        raise ValueError(
            f"unsupported architecture {arch!r}; registered: "
            f"{', '.join(sorted(ARCH_REGISTRY))}. To add one, write a "
            "splitter class with the Gpt2Split interface (head_forward / "
            "body_forward / tail_forward over the model's own modules) "
            "and register it: attacker.splitbench.ARCH_REGISTRY["
            "'<model_type>'] = MySplit") from None


def build_split(model, plan, arch=None):
    """Wrap `model` in its arch splitter; arch defaults to
    model.config.model_type."""
    if arch is None:
        arch = getattr(getattr(model, "config", None), "model_type", None)
    return get_arch_splitter(arch)(model, plan)


def _n_layers_of(model):
    """Block-stack depth across arch families: GPT-2 configs carry
    n_layer, the Mistral/Llama family num_hidden_layers."""
    cfg = getattr(model, "config", None)
    n = getattr(cfg, "n_layer", None) or getattr(cfg, "num_hidden_layers",
                                                 None)
    if n is None:
        raise ValueError(f"cannot determine the layer count of "
                         f"{type(cfg).__name__} (neither n_layer nor "
                         "num_hidden_layers)")
    return n


class SeamPcaState:
    """Online estimate of the seam covariance's top PCA direction.

    Pure python (so the torch-free test layer can exercise the math): the
    exponential-moving-average covariance lives in nested lists.
    update(rows) accumulates a batch covariance from raw rows (toy
    scale); update_cov(cov) blends a precomputed d x d covariance (the
    driver computes it torch-side on real batches and hands it over).
    top_direction() is power iteration from a deterministic start with
    sign canonicalization (the largest-magnitude component is positive),
    so equal update histories give equal directions bit for bit. Returns
    None before any update or on a degenerate zero covariance.
    """

    def __init__(self, dim, momentum=0.9, iters=128):
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise ValueError(f"SeamPcaState: dim must be an int >= 1, "
                             f"got {dim!r}")
        if not isinstance(momentum, (int, float)) \
                or isinstance(momentum, bool) \
                or not 0.0 <= momentum < 1.0:
            raise ValueError(f"SeamPcaState: momentum must be a number "
                             f"in [0, 1), got {momentum!r}")
        self.dim = dim
        self.momentum = float(momentum)
        self.iters = int(iters)
        self.cov = None
        self.n_updates = 0

    @staticmethod
    def _batch_cov(rows):
        """Centered sample covariance (/(n-1)) of float rows, dxd lists."""
        n = len(rows)
        d = len(rows[0])
        mean = [sum(r[j] for r in rows) / n for j in range(d)]
        cov = [[0.0] * d for _ in range(d)]
        for r in rows:
            dev = [r[j] - mean[j] for j in range(d)]
            for i in range(d):
                di = dev[i]
                ci = cov[i]
                for j in range(d):
                    ci[j] += di * dev[j]
        scale = 1.0 / (n - 1)
        return [[c * scale for c in row] for row in cov]

    def update(self, rows):
        """Blend one batch of raw seam rows into the EMA covariance."""
        rows = [list(map(float, r)) for r in rows]
        if len(rows) < 2:
            raise ValueError(f"SeamPcaState.update needs >= 2 rows for a "
                             f"covariance, got {len(rows)}")
        if any(len(r) != self.dim for r in rows):
            raise ValueError(f"SeamPcaState: row width must be {self.dim}")
        self.update_cov(self._batch_cov(rows))

    def update_cov(self, cov):
        """Blend a precomputed dxd covariance into the EMA estimate."""
        cov = [list(map(float, r)) for r in cov]
        if len(cov) != self.dim or any(len(r) != self.dim for r in cov):
            raise ValueError(f"SeamPcaState: covariance must be "
                             f"{self.dim}x{self.dim}")
        m = self.momentum
        self.cov = cov if self.cov is None else [
            [m * a + (1.0 - m) * b for a, b in zip(ra, rb)]
            for ra, rb in zip(self.cov, cov)]
        self.n_updates += 1

    def top_direction(self):
        """Unit top-PCA direction (list of floats) or None."""
        if self.cov is None:
            return None
        d = self.dim
        rng = random.Random(0x9C4A)
        v = [rng.gauss(0.0, 1.0) for _ in range(d)]
        for _ in range(self.iters):
            w = [sum(self.cov[i][j] * v[j] for j in range(d))
                 for i in range(d)]
            nrm = math.sqrt(sum(x * x for x in w))
            if nrm == 0.0:
                return None
            v = [x / nrm for x in w]
        i_max = max(range(d), key=lambda i: abs(v[i]))
        if v[i_max] < 0.0:
            v = [-x for x in v]
        return v


AT_MOMENTUM = 0.9


class AtLiteState:
    """Per-run at-lite defense state: the EMA top-PCA direction of the
    seam distribution (SeamPcaState) + the public shift magnitude eps.
    apply() updates the estimate with the CURRENT microbatch and returns
    the shifted tensor, so even step 0 crosses defended. The heavy
    covariance accumulation runs torch-side; the EMA + power iteration
    stay pure python (SeamPcaState)."""

    def __init__(self, eps, momentum=AT_MOMENTUM, iters=128):
        self.eps = float(eps)
        self.momentum = momentum
        self.iters = iters
        self.pca = None

    def apply(self, hidden):
        """h -> h - eps * v1 with v1 the EMA top-PCA direction (updated
        with this microbatch first). eps == 0 returns the input
        bit-identical (== no defense)."""
        import torch
        d = hidden.shape[-1]
        if self.pca is None:
            self.pca = SeamPcaState(d, momentum=self.momentum,
                                    iters=self.iters)
        x = hidden.detach().reshape(-1, d).float()
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / max(1, x.shape[0] - 1)
        self.pca.update_cov(cov.cpu().tolist())
        if self.eps == 0.0:
            return hidden
        v1 = self.pca.top_direction()
        if v1 is None:
            return hidden
        v = torch.tensor(v1, dtype=hidden.dtype, device=hidden.device)
        return hidden - self.eps * v


def cosine_neighbors(table, k):
    """k nearest neighbors by cosine similarity (self excluded) for every
    row of `table` (a sequence of float sequences) -> list of id lists,
    ties broken by lower index. O(V^2 * d) pure python: the toy-scale
    reference path (tests, tiny vocabs); the driver uses _wte_neighbor_fn
    (torch) over the model's wte."""
    rows = [list(map(float, r)) for r in table]
    norms = [math.sqrt(sum(x * x for x in r)) for r in rows]
    out = []
    for i, ri in enumerate(rows):
        sims = []
        for j, rj in enumerate(rows):
            denom = norms[i] * norms[j]
            sims.append(sum(a * b for a, b in zip(ri, rj)) / denom
                        if denom > 0.0 else 0.0)
        order = sorted(range(len(rows)), key=lambda j: (-sims[j], j))
        out.append([j for j in order if j != i][:k])
    return out


class TokenSanitizer:
    """santext / custext input sanitization, pure python + seeded.

    neighbor_fn(token_id) -> list of candidate replacement ids (the k
    nearest embedding neighbors by cosine, self excluded).

    santext: per token, with probability p replace by a uniform draw from
        the candidates. One run-level seeded RNG advanced per call, so a
        run is deterministic given the same call sequence (the drivers
        call once per microbatch).
    custext: a FIXED per-token mapping derived from a (seed, token)-keyed
        RNG at first sight of the token — order-independent: every
        occurrence of a token maps to the same replacement in every
        batch (toy-scale CusText custom-mapping analogue).
    """

    def __init__(self, mode, neighbor_fn, p, k, seed):
        if mode not in ("santext", "custext"):
            raise ValueError(f"TokenSanitizer: unknown mode {mode!r} "
                             "(valid: santext, custext)")
        self.mode = mode
        self.neighbor_fn = neighbor_fn
        self.p = float(p)
        self.k = int(k)
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self._mapping = {}

    def _custext_map(self, t):
        if t not in self._mapping:
            cand = [int(c) for c in self.neighbor_fn(t)]
            if cand:
                pick = random.Random(f"{self.seed}:{t}").randrange(
                    len(cand))
                self._mapping[t] = cand[pick]
            else:
                self._mapping[t] = t
        return self._mapping[t]

    def sanitize(self, ids2d):
        """ids2d (nested lists of ints) -> sanitized nested lists."""
        out = []
        for row in ids2d:
            orow = []
            for t in row:
                t = int(t)
                if self.mode == "custext":
                    orow.append(self._custext_map(t))
                    continue
                if self._rng.random() < self.p:
                    cand = [int(c) for c in self.neighbor_fn(t)]
                    if cand:
                        t = cand[self._rng.randrange(len(cand))]
                orow.append(t)
            out.append(orow)
        return out


def _wte_neighbor_fn(wte_weight, k):
    """Torch neighbor_fn over an embedding matrix: token_id -> its
    min(k, V-1) nearest neighbors by cosine (self excluded), cached per
    token. Lazy per-token topk keeps this cheap at real vocab sizes (no
    VxV materialization)."""
    import torch
    w = wte_weight.detach().float()
    n = w / w.norm(dim=1, keepdim=True).clamp_min(1e-12)
    k_eff = min(int(k), n.shape[0] - 1)
    cache = {}

    def fn(t):
        t = int(t)
        if t not in cache:
            sims = n @ n[t]
            sims[t] = float("-inf")
            cache[t] = [int(i) for i in
                        sims.topk(k_eff).indices.tolist()]
        return cache[t]

    return fn


_MID_SEED_XOR = 0x4D4944
_SND_SEED_XOR = 0x534E44
_TB_SEED_XOR = 0x5442
_MIX_SEED_XOR = 0x4D4958
_REPLICA_SEED_XOR = 0x524550


def _rep_config(cfg):
    """The BenchConfig -> ReplicaMedianConfig projection (seed
    decorrelated from the harness's other seeded draws, like the pack's
    XOR offsets; per-channel seed = this seed ^ channel index)."""
    from .trained_defenses import ReplicaMedianConfig
    return ReplicaMedianConfig(replica_count=cfg.replica_count,
                               tolerance=cfg.replica_tolerance,
                               gauge=cfg.replica_gauge,
                               gauge_scale=cfg.replica_gauge_scale,
                               seed=cfg.seed ^ _REPLICA_SEED_XOR)


def _build_replica_channels(split, cfg):
    """K independent body replicas: deep copies of the model, each with a
    per-channel seeded gauge perturbation of the body-slice parameters
    (per-channel seed = cfg.seed ^ _REPLICA_SEED_XOR ^ channel index), so
    the honest channels genuinely disagree. Returns (channel_forwards,
    gauge_ids, models): forwards map a seam tensor to the body-blocks
    output (the aggregation point BEFORE the shared norm + lm_head),
    gauge ids follow the replica-integrity channel-key convention."""
    import copy
    import torch
    from .attacks.replica_integrity import channel_key_id
    rc = _rep_config(cfg)
    prefixes = split.body_param_prefixes()
    n_params = sum(p.numel() for p in split.model.parameters())
    if rc.replica_count * n_params * 4 > 100_000_000 \
            or str(split.device) != "cpu":
        print(f"[splitbench] replica-median: {rc.replica_count} "
              f"in-process body replicas of a {n_params / 1e6:.1f}M-param "
              f"model on {split.device} — K deep copies + K x body-slice "
              f"memory; at split {cfg.split} this may exceed pilot GPU "
              f"memory. No automatic downsizing: lower --replica-count "
              f"explicitly if needed.")
    forwards, gauge_ids, models = [], [], []
    for k in range(rc.replica_count):
        ch_seed = rc.channel_seed(k)
        model_k = copy.deepcopy(split.model)
        if rc.gauge:
            g = torch.Generator().manual_seed(ch_seed)
            with torch.no_grad():
                for name, p in model_k.named_parameters():
                    if any(pre in name for pre in prefixes):
                        noise = torch.randn(p.shape, generator=g) \
                            .to(device=p.device, dtype=p.dtype)
                        scale = rc.gauge_scale * p.norm() \
                            / noise.norm().clamp_min(1e-8)
                        p.add_(noise * scale)
        sub = build_split(model_k, split.plan)

        def make_fwd(sub_split):
            def fwd(h):
                x = h
                for r in sub_split.ranges[1:-1]:
                    x = sub_split.body_forward(x, r, None)
                return sub_split._run_blocks(x, sub_split.ranges[-1], None)
            return fwd

        forwards.append(make_fwd(sub))
        gauge_ids.append(channel_key_id(k, ch_seed))
        models.append(model_k)
    return forwards, gauge_ids, models


def _rep_utility(split, cfg, rep, batches):
    """Held-out CE/perplexity WITHOUT (the single undefended body) vs
    WITH the K-channel median aggregation on calibration batches — the
    utility_before / utility_after convention every defense is measured
    under. Eval-only; the model's train/eval mode is restored."""
    import torch
    import torch.nn.functional as F
    from . import trained_defenses as td
    was_training = split.model.training
    split.model.eval()
    losses = {"before": [], "after": []}
    with torch.no_grad():
        for ids in batches[:2]:
            ids = ids.to(split.device)
            h = split.head_forward(ids)
            median = td.coordinate_median(
                [f(h) for f in rep.channel_forwards])
            for key, logits in (
                    ("before", _server_rest_forward(split, h, cfg)),
                    ("after", split.lm_head(split.ln_f(median)))):
                losses[key].append(float(F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                    ids[:, 1:].reshape(-1))))
    if was_training:
        split.model.train()

    def pack(xs):
        ce = sum(xs) / len(xs)
        return {"cross_entropy": ce,
                "perplexity": math.exp(min(700.0, ce))}

    return {"utility_before": pack(losses["before"]),
            "utility_after": pack(losses["after"])}


def _replica_channel_writers(dstate, writer, k_count):
    """Lazily built per-channel CaptureWriters under
    <capture_dir>/channels/ch<k>/ (same wire-capture format, separate
    files — existing wire_*.json scans of the main dir are unaffected)."""
    ws = dstate.get("replica_writers")
    if ws is None:
        ws = [CaptureWriter(os.path.join(writer.dir, "channels", f"ch{k}"),
                            writer.mode, f"{writer.session_id}-ch{k}")
              for k in range(k_count)]
        dstate["replica_writers"] = ws
    return ws


def _mix_config(cfg):
    """The BenchConfig -> MixupConfig projection (seed decorrelated from
    the harness's other seeded draws, like the pack's XOR offsets)."""
    from .trained_defenses import MixupConfig
    return MixupConfig(lam=cfg.mixup_lambda, pool_size=cfg.mixup_pool,
                       seed=cfg.seed ^ _MIX_SEED_XOR,
                       correction=cfg.mixup_correction)


def _mix_utility(split, cfg, batches):
    """Held-out CE/perplexity WITHOUT vs WITH secret-mixup on calibration
    batches — the utility_before / utility_after convention every defense
    is measured under. The "after" pass warms a throwaway SecretMixup's
    pool on the first calibration batch (which therefore goes out
    undefended, per the documented empty-pool policy) and measures the
    STEADY-STATE mixed releases on the following batches. Eval-only; the
    model's train/eval mode is restored."""
    import torch
    import torch.nn.functional as F
    from . import trained_defenses as td
    was_training = split.model.training
    split.model.eval()
    probe = td.SecretMixup(_mix_config(cfg).canonical())
    losses = {"before": [], "after": []}
    with torch.no_grad():
        for i, ids in enumerate(batches[:3]):
            ids = ids.to(split.device)
            h = split.head_forward(_apply_input_defense(ids, cfg, {}))
            wire = probe.apply(h, tokens=ids)
            if i == 0:
                continue
            for key, x in (("before", h), ("after", wire)):
                logits = _server_rest_forward(split, x, cfg)
                losses[key].append(float(F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                    ids[:, 1:].reshape(-1))))
    if was_training:
        split.model.train()

    def pack(xs):
        ce = sum(xs) / len(xs)
        return {"cross_entropy": ce,
                "perplexity": math.exp(min(700.0, ce))}

    return {"utility_before": pack(losses["before"]),
            "utility_after": pack(losses["after"])}


def _tb_config(cfg):
    """The BenchConfig -> BottleneckConfig projection (seed decorrelated
    from the harness's other seeded draws, like the pack's XOR offsets)."""
    from .trained_defenses import BottleneckConfig
    return BottleneckConfig(bottleneck_dim=cfg.tb_dim,
                            encoder_hidden=cfg.tb_hidden,
                            probe_hidden=cfg.tb_probe_hidden,
                            minimax_steps=cfg.tb_minimax_steps,
                            mi_weight=cfg.tb_mi_weight,
                            seed=cfg.seed ^ _TB_SEED_XOR,
                            allow_cpu=cfg.tb_allow_cpu)


def _tb_utility(split, cfg, tb, batches):
    """Held-out CE/perplexity of the split model WITHOUT vs WITH the
    trained encoder on calibration batches — the utility_before /
    utility_after convention every defense is measured under. Eval-only,
    RNG-free; the model's train/eval mode is restored."""
    import torch
    import torch.nn.functional as F
    was_training = split.model.training
    split.model.eval()
    losses = {"before": [], "after": []}
    with torch.no_grad():
        for ids in batches[:2]:
            ids = ids.to(split.device)
            h = split.head_forward(ids)
            for key, wire in (("before", h), ("after", tb.decode(tb.apply(h)))):
                logits = _server_rest_forward(split, wire, cfg)
                losses[key].append(float(F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                    ids[:, 1:].reshape(-1))))
    if was_training:
        split.model.train()

    def pack(xs):
        ce = sum(xs) / len(xs)
        return {"cross_entropy": ce,
                "perplexity": math.exp(min(700.0, ce))}

    return {"utility_before": pack(losses["before"]),
            "utility_after": pack(losses["after"])}


def _pca_config(cfg):
    """The BenchConfig -> PcaConfig projection (seed decorrelated like
    the pack's XOR offsets)."""
    from .trained_defenses import PcaConfig
    return PcaConfig(bottleneck_dim=cfg.pca_dim, seed=cfg.seed
                     ^ _TB_SEED_XOR)


def _pca_utility(split, cfg, pca, batches):
    """Held-out CE/perplexity WITHOUT vs WITH the pca-compress wire on
    calibration batches (the utility_before/after convention). Eval-only,
    RNG-free; train/eval mode restored."""
    import torch
    import torch.nn.functional as F
    was_training = split.model.training
    split.model.eval()
    losses = {"before": [], "after": []}
    with torch.no_grad():
        for ids in batches[:2]:
            ids = ids.to(split.device)
            h = split.head_forward(ids)
            for key, wire in (("before", h),
                              ("after", pca.decode(pca.apply(h)))):
                logits = _server_rest_forward(split, wire, cfg)
                losses[key].append(float(F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                    ids[:, 1:].reshape(-1))))
    if was_training:
        split.model.train()

    def pack(xs):
        ce = sum(xs) / len(xs)
        return {"cross_entropy": ce,
                "perplexity": math.exp(min(700.0, ce))}

    return {"utility_before": pack(losses["before"]),
            "utility_after": pack(losses["after"])}


def make_defense_state(cfg, split, calibration=None, out_dir=None):
    """Per-run state for stateful seam defenses (created per run in the
    drivers, stepped per microbatch). Returns {} for stateless defenses.
    The input-side sanitizer is built here (needs the model's wte); the
    seam-side tensor states (mid-lite projection, at-lite PCA, snd-lite
    smoothing) are built lazily on the first tensor, when the seam width
    is known — see _apply_seam_defense.

    trained-bottleneck is the exception: its encoder must be TRAINED, so
    the drivers pass `calibration` (the run's batches/prompts) and
    `out_dir` (the capture dir): the minimax pre-phase runs here, BEFORE
    victim finetuning, on head_forward activations of the calibration
    data, and the encoder checkpoint + provenance sidecar are saved into
    `out_dir` (state["tb_provenance"] records both paths; the run
    artifact picks the path up via finetune/infer stats). With
    cfg.tb_checkpoint set, the pre-phase is skipped and the checkpoint is
    reused. GPU-only by default (trained_defenses.CPU_ERROR);
    cfg.tb_allow_cpu opts in."""
    state = {}
    if cfg.defense in ("santext", "custext"):
        state["sanitizer"] = TokenSanitizer(
            cfg.defense, _wte_neighbor_fn(split.wte.weight, cfg.santext_k),
            p=cfg.santext_p, k=cfg.santext_k, seed=cfg.seed)
    if cfg.defense == "trained-bottleneck":
        from . import trained_defenses as td
        if cfg.tb_checkpoint:
            state["tb"] = td.TrainedBottleneck.load(cfg.tb_checkpoint)
            state["tb_provenance"] = {"checkpoint": cfg.tb_checkpoint,
                                      "reused": True}
            return state
        if calibration is None or out_dir is None:
            raise ValueError(
                "[splitbench] trained-bottleneck needs calibration data "
                "and an output dir for its encoder pre-phase: the "
                "finetune/infer drivers pass them to make_defense_state "
                "(or set tb_checkpoint to reuse a trained encoder)")
        import torch
        rows, tokens = [], []
        with torch.no_grad():
            for ids in calibration:
                ids = ids.to(split.device)
                h = split.head_forward(_apply_input_defense(ids, cfg, state))
                rows.append(h.reshape(-1, h.shape[-1]))
                tokens.append(ids.reshape(-1))
        tb = td.train_bottleneck(torch.cat(rows), torch.cat(tokens),
                                 _tb_config(cfg),
                                 device=str(split.device))
        tb.utility = _tb_utility(split, cfg, tb, calibration)
        state["tb"] = tb
        state["tb_provenance"] = tb.save(out_dir)
        if cfg.tb_cotrain:
            state["tb_cotrain"] = td.BottleneckCotrain(
                tb, _tb_config(cfg), split.device,
                interval=cfg.tb_cotrain_interval)
            tb.cotrain = {"enabled": True,
                          "interval": cfg.tb_cotrain_interval,
                          "updates_performed": 0}
    if cfg.defense == "pca-compress":
        from . import trained_defenses as td
        if calibration is None or out_dir is None:
            raise ValueError(
                "[splitbench] pca-compress needs calibration data and an "
                "output dir for its PCA fit: the finetune/infer drivers "
                "pass them to make_defense_state")
        import torch
        rows = []
        with torch.no_grad():
            for ids in calibration:
                ids = ids.to(split.device)
                h = split.head_forward(_apply_input_defense(ids, cfg,
                                                            state))
                rows.append(h.reshape(-1, h.shape[-1]))
        pca = td.fit_pca_compressor(torch.cat(rows), _pca_config(cfg),
                                    device=str(split.device))
        pca.utility = _pca_utility(split, cfg, pca, calibration)
        state["pca"] = pca
        state["pca_provenance"] = pca.save(out_dir)
    if cfg.defense == "secret-mixup":
        from . import trained_defenses as td
        state["mixup"] = td.SecretMixup(_mix_config(cfg).canonical())
        if calibration is not None:
            state["mixup_utility"] = _mix_utility(split, cfg, calibration)
    if cfg.defense == "replica-median":
        from . import trained_defenses as td
        forwards, gauge_ids, models = _build_replica_channels(split, cfg)
        rep = td.ReplicaMedian(_rep_config(cfg).canonical(),
                               channel_forwards=forwards,
                               gauge_ids=gauge_ids)
        state["replica"] = rep
        state["replica_models"] = models
        if calibration is not None:
            import torch
            with torch.no_grad():
                hs = [split.head_forward(
                    _apply_input_defense(ids.to(split.device), cfg, state))
                    for ids in calibration[:2]]
            rep.measure_heterogeneity(hs)
            state["replica_utility"] = _rep_utility(split, cfg, rep,
                                                    calibration)
    return state


def _apply_input_defense(ids, cfg, state=None):
    """santext/custext token-id sanitization BEFORE embedding. Returns
    the ids the head actually sees; the ORIGINAL ids stay the canonical
    ground truth (canonical.pt) and the CE target (see module docstring).
    Other defenses return the input unchanged."""
    if cfg.defense not in ("santext", "custext"):
        return ids
    if state is None or "sanitizer" not in state:
        raise ValueError("[splitbench] santext/custext need per-run "
                         "defense state: build it with "
                         "make_defense_state(cfg, split) (the finetune/"
                         "infer drivers do this)")
    import torch
    sanitized = state["sanitizer"].sanitize(ids.tolist())
    return torch.tensor(sanitized, dtype=ids.dtype, device=ids.device)


def _quantize_absmax(t, bits):
    """Signed symmetric per-sequence absmax quantization of a
    [batch, seq, hidden] tensor: limit = 2^(bits-1) - 1 levels, per-row
    step = absmax(row) / limit; round, clamp, rescale. The same rule as
    validation_controls.transform_gradient's bits_N gradient condition
    (kept local so the module imports torch-free; validation_controls
    itself imports torch at module level). All-zero rows stay all-zero
    (the scale clamps to tiny, never divides by zero)."""
    import torch
    if not isinstance(bits, int) or isinstance(bits, bool) \
            or not 2 <= bits <= 16:
        raise ValueError(f"quant bits must be an integer in 2..16, "
                         f"got {bits!r}")
    limit = 2 ** (bits - 1) - 1
    step = t.abs().amax(dim=(-2, -1), keepdim=True) / limit
    safe = step.clamp_min(torch.finfo(t.dtype).tiny)
    return (t / safe).round().clamp(-limit, limit) * step


def _apply_grad_defense(g, cfg):
    """Return-path defense hook (finetune): the gradient the VICTIM
    receives, which is what the bwd capture records (the return-path
    mirror of the post-defense forward doctrine).

    Frozen composition order: global-L2 clip to cfg.grad_clip (0 = off),
    then relative-RMS Gaussian noise of scale cfg.grad_noise (per-row
    RMS(g) * sigma, the validation_controls noise_σ convention; draws
    from the run's RNG stream like the gaussian seam noise; 0 = off),
    then per-sequence absmax quantization at cfg.quant_bits (only under
    --defense quant --quant-grad; BenchConfig.validate enforces the
    pairing). All-zero knobs return g itself — bit-identical to the
    historic raw-gradient path."""
    if cfg.grad_clip > 0:
        norm = g.norm()
        scale = (cfg.grad_clip / norm).clamp(max=1.0)
        g = g * scale
    if cfg.grad_noise > 0:
        import torch
        rms = g.square().mean(dim=(-2, -1), keepdim=True).sqrt()
        g = g + torch.randn_like(g) * rms * cfg.grad_noise
    if cfg.quant_grad:
        g = _quantize_absmax(g, cfg.quant_bits)
    return g


def _apply_seam_defense(hidden, cfg, state=None):
    """Return the tensor the BODY receives. The capture records THIS
    tensor (post-defense). sigma == 0 skips the draw, so gaussian@0 is
    bit-identical to no defense (and leaves the RNG stream untouched).

    `state` is the per-run dict from make_defense_state (the drivers pass
    it; stateful seam defenses — mid-lite, at-lite, snd-lite — lazily
    build their fixed maps/estimates in it on the first tensor so they
    persist across microbatches). Calling without state is supported for
    stateless defenses; for stateful ones each call then rebuilds its
    map (mid-lite/snd-lite: same seeded matrix, just slower) or restarts
    its estimate (at-lite: degenerate single-batch PCA — go through the
    drivers for the real defense)."""
    if cfg.defense == "gaussian":
        if cfg.sigma > 0:
            import torch
            return hidden + torch.randn_like(hidden) * cfg.sigma
        return hidden
    if cfg.defense == "quant":
        return _quantize_absmax(hidden, cfg.quant_bits)
    if cfg.defense == "sparsify":
        import torch
        d = hidden.shape[-1]
        kept = max(1, int(round(cfg.sparsity * d)))
        if kept >= d:
            return hidden
        idx = hidden.abs().topk(kept, dim=-1).indices
        mask = torch.zeros_like(hidden).scatter(-1, idx, 1.0)
        return hidden * mask
    if cfg.defense == "mid-lite":
        import torch
        d = hidden.shape[-1]
        k = cfg.bottleneck_dim
        if k > d:
            raise ValueError(f"BenchConfig: bottleneck_dim {k} exceeds "
                             f"the seam width {d} — mid-lite projects "
                             "d -> k with k <= d")
        proj = state.get("mid_proj") if state is not None else None
        if proj is None or proj.shape[0] != d:
            g = torch.Generator().manual_seed(cfg.seed ^ _MID_SEED_XOR)
            proj = torch.randn(d, k, generator=g) / math.sqrt(d)
            if state is not None:
                state["mid_proj"] = proj
        z = hidden @ proj.to(dtype=hidden.dtype, device=hidden.device)
        out = torch.zeros_like(hidden)
        out[..., :k] = z
        return out
    if cfg.defense == "at-lite":
        at = state.get("at") if state is not None else None
        if at is None:
            at = AtLiteState(cfg.at_eps)
            if state is not None:
                state["at"] = at
        return at.apply(hidden)
    if cfg.defense == "snd-lite":
        if cfg.snd_sigma == 0:
            return hidden
        import torch
        d = hidden.shape[-1]
        smooth = state.get("snd_smooth") if state is not None else None
        if smooth is None or smooth.shape[0] != d:
            g = torch.Generator().manual_seed(cfg.seed ^ _SND_SEED_XOR)
            smooth = torch.randn(d, d, generator=g) / math.sqrt(d)
            if state is not None:
                state["snd_smooth"] = smooth
        share1 = torch.randn_like(hidden) * cfg.snd_sigma
        share2 = hidden - share1
        return share1 @ smooth.to(dtype=hidden.dtype,
                                  device=hidden.device) + share2
    if cfg.defense == "trained-bottleneck":
        tb = state.get("tb") if state is not None else None
        if tb is None:
            raise ValueError(
                "[splitbench] trained-bottleneck needs its trained encoder "
                "in the per-run defense state: build it with "
                "make_defense_state(cfg, split, calibration=..., "
                "out_dir=...) (the finetune/infer drivers do this)")
        return tb.apply(hidden)
    if cfg.defense == "pca-compress":
        pca = state.get("pca") if state is not None else None
        if pca is None:
            raise ValueError(
                "[splitbench] pca-compress needs its fitted encoder in "
                "the per-run defense state: build it with "
                "make_defense_state(cfg, split, calibration=..., "
                "out_dir=...) (the finetune/infer drivers do this)")
        return pca.apply(hidden)
    if cfg.defense == "secret-mixup":
        mix = state.get("mixup") if state is not None else None
        if mix is None:
            from . import trained_defenses as td
            mix = td.SecretMixup(_mix_config(cfg).canonical())
            if state is not None:
                state["mixup"] = mix
        tokens = state.get("mixup_ids") if state is not None else None
        return mix.apply(hidden, tokens=tokens)
    if cfg.defense == "replica-median":
        return hidden
    return hidden


def _defense_config_record(cfg):
    """The active defense's knobs, for the run artifact's results block
    (the full field set is already in the artifact's config via
    dataclasses.asdict)."""
    knobs = {"none": {},
             "gaussian": {"sigma": cfg.sigma},
             "sparsify": {"sparsity": cfg.sparsity},
             "mid-lite": {"bottleneck_dim": cfg.bottleneck_dim},
             "at-lite": {"at_eps": cfg.at_eps},
             "santext": {"santext_p": cfg.santext_p,
                         "santext_k": cfg.santext_k},
             "custext": {"santext_k": cfg.santext_k},
             "snd-lite": {"snd_sigma": cfg.snd_sigma},
             "trained-bottleneck": {"bottleneck_dim": cfg.tb_dim,
                                    "encoder_hidden": cfg.tb_hidden,
                                    "probe_hidden": cfg.tb_probe_hidden,
                                    "minimax_steps": cfg.tb_minimax_steps,
                                    "mi_weight": cfg.tb_mi_weight,
                                    "cotrain": cfg.tb_cotrain,
                                    "cotrain_interval":
                                        cfg.tb_cotrain_interval},
             "pca-compress": {"bottleneck_dim": cfg.pca_dim},
             "secret-mixup": {"lambda": cfg.mixup_lambda,
                              "pool_size": cfg.mixup_pool,
                              "correction": cfg.mixup_correction},
             "replica-median": {"replica_count": cfg.replica_count,
                                "tolerance": cfg.replica_tolerance,
                                "gauge": cfg.replica_gauge,
                                "gauge_scale": cfg.replica_gauge_scale},
             "quant": {"quant_bits": cfg.quant_bits,
                       "quant_grad": cfg.quant_grad}}
    rec = {"name": cfg.defense, **knobs[cfg.defense]}
    if cfg.grad_clip > 0:
        rec["grad_clip"] = cfg.grad_clip
    if cfg.grad_noise > 0:
        rec["grad_noise"] = cfg.grad_noise
    if cfg.trainable_range is not None:
        rec["trainable_range"] = cfg.trainable_range
    return rec


BODY_GRADS_NAME = "body_grads_step0.pt"


def _rest_forward_amplified(split, hidden, alpha, attention_mask=None):
    """The malicious server's crafted body (module docstring): the honest
    middle slices, then  y = Body(h) + alpha * h  through the shared tail
    (ln_f + lm_head). Uses only the shared splitter accessor protocol
    (_run_blocks / ranges / ln_f / lm_head), so every registered arch
    splitter runs it. The alpha identity path is what makes the returned
    seam gradient carry an alpha-scaled per-row copy of the label/token
    direction (dL/dh = alpha * J_lnf (p - onehot) E + honest backprop) —
    autograd on the modified computation, not a doctored tensor."""
    h = hidden
    for layer_range in split.ranges[1:-1]:
        h = split.body_forward(h, layer_range, attention_mask)
    body_out = split._run_blocks(h, split.ranges[-1], attention_mask)
    crafted = body_out + alpha * hidden
    return split.lm_head(split.ln_f(crafted))


def _server_rest_forward(split, hidden, cfg, state=None):
    """rest_forward with the --server-attack hook applied (finetune only;
    "none" is the honest path, bit-identical)."""
    if cfg.defense == 'trained-bottleneck' and state is not None:
        hidden = state['tb'].decode(hidden)
    if cfg.defense == 'pca-compress' and state is not None:
        hidden = state['pca'].decode(hidden)
    if cfg.server_attack == "passive-amplify":
        return _rest_forward_amplified(split, hidden,
                                       cfg.server_attack_alpha)
    return split.rest_forward(hidden)


def _save_body_grads(split, capture_dir):
    """Step-0 per-parameter body-slice gradients -> body_grads_step0.pt.

    The batch-label-reconstruction observation surface: {name:
    grad.cpu().fp16()} for every body-slice parameter that received a
    gradient (split.body_param_prefixes(), matched by containment so
    peft-wrapped LoRA names classify too; under LoRA only the trainable
    body adapters carry grads, which is exactly the server-side signal).
    Tied weights (GPT-2's lm_head == wte) appear once under their
    canonical named_parameters() name — the tied alias never has its own
    entry, and the attack side matches names through the same convention.
    The body WEIGHTS are not exported: at step 0 the body is the
    pre-update body, which the attack rebuilds from --model + --split at
    the same architecture (documented in attacks/batch_label_recon.py).
    Returns the number of tensors saved."""
    import torch
    prefixes = split.body_param_prefixes()
    grads = {name: p.grad.detach().to(device="cpu", dtype=torch.float16)
             for name, p in split.model.named_parameters()
             if p.grad is not None and any(pre in name for pre in prefixes)}
    torch.save(grads, os.path.join(capture_dir, BODY_GRADS_NAME))
    return len(grads)


def finetune(split, cfg, batches, writer, on_step=None, measurement=None,
             capture_replica_input=False):
    """Split fine-tune loop with seam captures (see module docstring).

    split    Gpt2Split-like wrapper (head + rest around the seam)
    batches  list of LongTensor [batch_size, seq_len], cycled; epoch
             counts wraps
    writer   CaptureWriter(mode="training")
    on_step  optional callback({step, epoch, loss, captured}) per step
    returns  {"steps_run", "final_loss", "mean_loss"}
    """
    import torch
    import torch.nn.functional as F

    if not batches:
        raise ValueError("finetune needs at least one batch")
    torch.manual_seed(cfg.seed)
    model = split.model
    model.train()
    if cfg.trainable_range is not None:
        t_start, t_end = _parse_trainable_range(cfg.trainable_range,
                                                n_blocks=len(split.blocks))
        for p in model.parameters():
            p.requires_grad_(False)
        for blk in split.blocks[t_start:t_end]:
            for p in blk.parameters():
                p.requires_grad_(True)
    dstate = make_defense_state(cfg, split, calibration=batches,
                                out_dir=writer.dir)
    if measurement is not None:
        measurement("before", split, cfg, dstate)
    params = [p for p in model.parameters() if p.requires_grad]
    for m in dstate.get("replica_models", []):
        params += [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    n = len(batches)
    losses = []
    captured_ids = []
    for step in range(cfg.steps):
        epoch = step // n
        ids = batches[step % n].to(split.device)
        ids_in = _apply_input_defense(ids, cfg, dstate)
        dstate["mixup_ids"] = ids
        h_star = split.head_forward(ids_in)
        rep = dstate.get("replica") if cfg.defense == "replica-median" \
            else None
        capture = step % cfg.capture_every == 0
        if rep is not None:
            h_sent = h_star
            h_wire = h_star.detach().requires_grad_(True)
            median, ys, _entry = rep.forward_block(h_wire, step=step,
                                                   captured=capture)
            if cfg.server_attack == "passive-amplify":
                median = median + cfg.server_attack_alpha * h_wire
            if capture:
                ws = _replica_channel_writers(dstate, writer, rep.k)
                for k, y in enumerate(ys):
                    ws[k].write_training(y.detach(), mb_id=0, phase="fwd",
                                         step=step, epoch=epoch)
                writer.write_training((h_wire if capture_replica_input else median).detach(), mb_id=0,
                                      phase="fwd", step=step, epoch=epoch)
                captured_ids.append(ids.cpu())
            logits = split.lm_head(split.ln_f(median))
        else:
            h_sent = _apply_seam_defense(h_star, cfg, dstate)
            h_wire = h_sent.detach().requires_grad_(True)
            if capture:
                writer.write_training(h_wire, mb_id=0, phase="fwd",
                                      step=step, epoch=epoch)
                captured_ids.append(ids.cpu())
            logits = _server_rest_forward(split, h_wire, cfg, dstate)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            ids[:, 1:].reshape(-1))
        mix = dstate.get("mixup") if cfg.defense == "secret-mixup" else None
        g_decoy = None
        if mix is not None and cfg.mixup_correction \
                and mix.last_decoy is not None:
            from . import trained_defenses as td
            decoy = mix.last_decoy.detach().requires_grad_(True)
            logits_d = _server_rest_forward(split, decoy, cfg)
            adjoint = torch.autograd.grad(loss, logits,
                                          retain_graph=True)[0]
            g_decoy = torch.autograd.grad(logits_d, decoy,
                                          grad_outputs=adjoint,
                                          retain_graph=True)[0]
        loss.backward()
        g_wire = _apply_grad_defense(h_wire.grad, cfg)
        if capture:
            writer.write_training(g_wire, mb_id=0, phase="bwd",
                                  step=step, epoch=epoch)
        if h_sent.requires_grad:
            g = g_wire.detach()
            if g_decoy is not None:
                g = td.corrected_gradient(g, g_decoy, cfg.mixup_lambda)
            h_sent.backward(g)
        if cfg.body_grad_capture and step == 0:
            _save_body_grads(split, writer.dir)
        opt.step()
        opt.zero_grad()
        ct = dstate.get("tb_cotrain")
        if ct is not None and (step + 1) % ct.interval == 0:
            ct.update(h_star.detach().reshape(-1, h_star.shape[-1]),
                      ids.reshape(-1))
        losses.append(loss.item())
        if on_step is not None:
            on_step({"step": step, "epoch": epoch, "loss": losses[-1],
                     "captured": capture})
    if measurement is not None:
        measurement("after", split, cfg, dstate)
    if cfg.save_canonical and captured_ids:
        torch.save(torch.cat(captured_ids),
                   os.path.join(writer.dir, "canonical.pt"))
    stats = {"steps_run": cfg.steps, "final_loss": losses[-1],
             "mean_loss": sum(losses) / len(losses)}
    if "tb_cotrain" in dstate:
        ct = dstate["tb_cotrain"]
        tb = dstate["tb"]
        tb.cotrain = {"enabled": True, "interval": ct.interval,
                      "updates_performed": ct.updates}
        dstate["tb_provenance"] = tb.save(writer.dir)
    if "tb_provenance" in dstate:
        stats["trained_bottleneck_provenance"] = dstate["tb_provenance"]
    if "pca_provenance" in dstate:
        stats["pca_compress_provenance"] = dstate["pca_provenance"]
    if "mixup" in dstate:
        mix = dstate["mixup"]
        mix.utility = dstate.get("mixup_utility")
        stats["secret_mixup_provenance"] = mix.save(writer.dir)
    if "replica" in dstate:
        rep = dstate["replica"]
        rep.utility = dstate.get("replica_utility")
        stats["replica_median_provenance"] = rep.save(writer.dir)
    return stats


def infer(split, cfg, prompts, writer, on_request=None):
    """Split-inference loop: one prefill + `steps` greedy decode captures
    per prompt (full re-forward per token; use_cache=False — a capture
    rig, not a serving stack).

    prompts  list of LongTensor [1, P]
    writer   CaptureWriter(mode="inference")
    returns  {"steps_run" (decode tokens per request), "requests"}
    """
    import torch

    if not prompts:
        raise ValueError("infer needs at least one prompt")
    torch.manual_seed(cfg.seed)
    split.model.eval()
    dstate = make_defense_state(cfg, split, calibration=prompts,
                                out_dir=writer.dir)
    truth = []
    rep = dstate.get("replica") if cfg.defense == "replica-median" else None

    def serve(ids, request_seq, phase, position, step):
        """One served forward: returns the tensor the top consumes (the
        median aggregate for replica-median, the post-defense seam tensor
        otherwise) after writing this step's captures."""
        hidden = _apply_seam_defense(split.head_forward(ids), cfg, dstate)
        if rep is None:
            writer.write_inference(
                hidden[:, -1:, :] if phase == "decode" else hidden,
                request_seq, phase=phase, position=position, epoch=0)
            return hidden, None
        median, ys, _ = rep.forward_block(hidden, step=step, captured=True)
        ws = _replica_channel_writers(dstate, writer, rep.k)
        for k, y in enumerate(ys):
            ws[k].write_inference(y[:, -1:, :] if phase == "decode" else y,
                                  request_seq, phase=phase,
                                  position=position, epoch=0)
        wire = median[:, -1:, :] if phase == "decode" else median
        writer.write_inference(wire, request_seq, phase=phase,
                               position=position, epoch=0)
        return median, ys

    with torch.no_grad():
        for request_seq, ids in enumerate(prompts):
            truth.append({"request_seq": request_seq, "prompt_ids": ids.cpu().tolist(),
                          "generated_ids": []})
            ids = ids.to(split.device)
            ids = _apply_input_defense(ids, cfg, dstate)
            dstate["mixup_ids"] = ids
            hidden, _ = serve(ids, request_seq, "prefill", 0, step=0)
            for k in range(1, cfg.steps + 1):
                if rep is not None:
                    nxt = split.lm_head(split.ln_f(hidden))[:, -1] \
                        .argmax(-1, keepdim=True)
                else:
                    nxt = _server_rest_forward(split, hidden, cfg, dstate)[:, -1] \
                        .argmax(-1, keepdim=True)
                truth[-1]["generated_ids"].append(int(nxt.item()))
                ids = torch.cat([ids, nxt], dim=1)
                dstate["mixup_ids"] = ids
                hidden, _ = serve(ids, request_seq, "decode", k, step=k)
            if on_request is not None:
                on_request({"request_seq": request_seq,
                            "prompt_tokens": int(prompts[request_seq]
                                                 .shape[-1]),
                            "captures": cfg.steps + 1})
    if cfg.save_canonical:
        with open(os.path.join(writer.dir, "canonical.json"), "w") as f:
            json.dump(truth, f)
        if len({p.shape[-1] for p in prompts}) == 1:
            torch.save(torch.cat([p.cpu() for p in prompts]),
                       os.path.join(writer.dir, "canonical.pt"))
    out = {"steps_run": cfg.steps, "requests": len(prompts)}
    if "tb_provenance" in dstate:
        out["trained_bottleneck_provenance"] = dstate["tb_provenance"]
    if "pca_provenance" in dstate:
        out["pca_compress_provenance"] = dstate["pca_provenance"]
    if "mixup" in dstate:
        mix = dstate["mixup"]
        mix.utility = dstate.get("mixup_utility")
        out["secret_mixup_provenance"] = mix.save(writer.dir)
    if "replica" in dstate:
        rep2 = dstate["replica"]
        rep2.utility = dstate.get("replica_utility")
        out["replica_median_provenance"] = rep2.save(writer.dir)
    return out


def _load_model_and_tokenizer(cfg):
    try:
        import torch
    except ImportError:
        raise SystemExit(TORCH_ERROR) from None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise SystemExit(
            "[splitbench] model loading requires transformers "
            "(pip install transformers); the pure-python core works "
            "torch-less.") from None
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[cfg.dtype]
    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, torch_dtype=dtype, attn_implementation="eager")
    return model.to(cfg.device), tok


def _apply_lora(model, plan, cfg):
    """peft-LoRA on the server-side slice only (blocks after the first
    boundary). peft freezes the base weights: the optimizer then trains
    exactly the body-slice adapters."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise SystemExit("[splitbench] --lora-r requires peft "
                         "(pip install peft)") from None
    layers = list(range(plan.boundaries[0], model.config.n_layer))
    lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                      target_modules=["c_attn", "c_proj"],
                      layers_to_transform=layers)
    return get_peft_model(model, lcfg)


def _token_stream(cfg, tok):
    """1-D token-id LongTensor from --corpus-file (preferred) or the
    configured dataset."""
    import torch
    if cfg.corpus_file:
        if not os.path.isfile(cfg.corpus_file):
            raise SystemExit(f"[splitbench] --corpus-file not found: "
                             f"{cfg.corpus_file}")
        with open(cfg.corpus_file) as f:
            text = f.read()
    elif cfg.dataset == "wikitext":
        try:
            from datasets import load_dataset
        except ImportError:
            raise SystemExit(
                "[splitbench] --dataset wikitext requires the `datasets` "
                "package (pip install datasets) to load "
                "Salesforce/wikitext wikitext-2-raw-v1; or pass "
                "--corpus-file with local text.") from None
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                          split="train")
        text = "\n".join(t for t in ds["text"] if t.strip())
    else:
        raise SystemExit(f"[splitbench] unknown dataset {cfg.dataset!r} "
                         "(supported: wikitext; or pass --corpus-file)")
    return tok(text, return_tensors="pt").input_ids[0]


def _make_batches(ids, cfg):
    """Shuffled non-overlapping [batch_size, seq_len] windows (finetune
    stream; cycled by the driver, epoch = wraps)."""
    import torch
    n_win = ids.numel() // cfg.seq_len
    if n_win < cfg.batch_size:
        raise SystemExit(f"[splitbench] corpus too small: {ids.numel()} "
                         f"tokens -> {n_win} windows of seq_len "
                         f"{cfg.seq_len}, need at least batch_size "
                         f"{cfg.batch_size}")
    windows = ids[:n_win * cfg.seq_len].view(n_win, cfg.seq_len)
    order = torch.randperm(n_win,
                           generator=torch.Generator().manual_seed(cfg.seed))
    return [windows[order[i:i + cfg.batch_size]]
            for i in range(0, n_win - cfg.batch_size + 1, cfg.batch_size)]


def _make_prompts(ids, cfg):
    """batch_size prompts of seq_len tokens each (inference serves
    batch_size requests, one request_seq apiece)."""
    n_win = ids.numel() // cfg.seq_len
    if n_win < cfg.batch_size:
        raise SystemExit(f"[splitbench] corpus too small: {ids.numel()} "
                         f"tokens -> {n_win} windows of seq_len "
                         f"{cfg.seq_len}, need at least batch_size "
                         f"{cfg.batch_size}")
    return [ids[i * cfg.seq_len:(i + 1) * cfg.seq_len].unsqueeze(0)
            for i in range(cfg.batch_size)]


def execute(cfg, split, writer, data, evidence=None, measurement=None,
            capture_replica_input=False):
    """Drive one configured run over an already-built split + data and
    write the artifact + crash-safe journal. `data` is a batch list
    (finetune) or prompt list (inference). Returns the driver stats."""
    out = artifacts.make_artifact(
        SCHEMA, dataclasses.asdict(cfg), THREAT_MODEL,
        interpretation="results[0]: captures_written / steps_run / "
                       "final_loss (finetune) / defense; captures land in "
                       "config.capture_dir in the wire-capture schema and "
                       "feed every attack module via attacker.captures",
        evidence=evidence)

    def journal(rec):
        artifacts.append_jsonl(cfg.output, rec)
        if evidence is not None:
            evidence.event("step", rec)

    if cfg.mode == "finetune":
        stats = finetune(split, cfg, data, writer, on_step=journal,
                         measurement=measurement, capture_replica_input=capture_replica_input)
    else:
        stats = infer(split, cfg, data, writer, on_request=journal)
    rec = {"captures_written": writer.count,
           "steps_run": stats["steps_run"],
           "defense": cfg.defense,
           "sigma": cfg.sigma,
           "defense_config": _defense_config_record(cfg),
           "server_attack": cfg.server_attack,
           "split": cfg.split,
           "plan": split.plan.describe()}
    if cfg.server_attack != "none":
        rec["server_attack_alpha"] = cfg.server_attack_alpha
    if "final_loss" in stats:
        rec["final_loss"] = stats["final_loss"]
        rec["mean_loss"] = stats["mean_loss"]
    if "trained_bottleneck_provenance" in stats:
        prov = stats["trained_bottleneck_provenance"]
        rec["trained_bottleneck_provenance"] = \
            prov.get("provenance_path", prov)
    if "pca_compress_provenance" in stats:
        prov = stats["pca_compress_provenance"]
        rec["pca_compress_provenance"] = prov.get("provenance_path", prov)
    if "secret_mixup_provenance" in stats:
        prov = stats["secret_mixup_provenance"]
        rec["secret_mixup_provenance"] = prov.get("provenance_path", prov)
    if "replica_median_provenance" in stats:
        prov = stats["replica_median_provenance"]
        rec["replica_median_provenance"] = \
            prov.get("provenance_path", prov)
    if "requests" in stats:
        rec["requests"] = stats["requests"]
    out["results"].append(rec)
    out["summary"].append(dict(rec))
    artifacts.write_artifact(cfg.output, out, evidence=evidence)
    return stats


def _run(cfg, evidence=None):
    import torch

    torch.manual_seed(cfg.seed)
    model, tok = _load_model_and_tokenizer(cfg)
    plan = SplitPlan.parse(cfg.split, n_layers=_n_layers_of(model))
    split = build_split(model, plan)
    if cfg.lora_r > 0:
        model = _apply_lora(model, plan, cfg)
        split.model = model
    ids = _token_stream(cfg, tok)
    data = _make_batches(ids, cfg) if cfg.mode == "finetune" \
        else _make_prompts(ids, cfg)
    session_id = f"splitbench-{cfg.mode}-{uuid.uuid4().hex[:8]}"
    writer = CaptureWriter(cfg.capture_dir, _MODE_TO_WRITER[cfg.mode],
                           session_id)
    try:
        execute(cfg, split, writer, data, evidence=evidence)
    finally:
        writer.close()
    print(f"[splitbench] {plan.describe()}; {writer.count} captures -> "
          f"{cfg.capture_dir}")
    return 0


def run(cfg, evidence=False):
    """Full benchmark run from a BenchConfig; returns the exit code."""
    cfg.validate()
    if cfg.output:
        parent = os.path.dirname(os.path.abspath(cfg.output))
        if parent:
            os.makedirs(parent, exist_ok=True)
    if evidence:
        if not cfg.output:
            raise SystemExit("[splitbench] --evidence requires --output "
                             "(the evidence manifest is written next to "
                             "the artifact)")
        from .evidence import EvidenceLogger
        with EvidenceLogger(cfg.output,
                            node_role=f"splitbench/{cfg.mode}") as ev:
            return _run(cfg, evidence=ev)
    return _run(cfg, evidence=None)


def build_parser():
    ap = argparse.ArgumentParser(
        prog="python -m attacker.splitbench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="JSON config file (BenchConfig.from_json); the "
                         "flags below override it when set")
    ap.add_argument("--model", default=None,
                    help="HF model name or local path (default gpt2)")
    ap.add_argument("--split", default=None,
                    help="slice boundaries, e.g. '6' (head|body) or "
                         "'4,8' (head|body|tail)")
    ap.add_argument("--mode", default=None,
                    choices=sorted(_MODE_TO_WRITER),
                    help="finetune (wire-training schema) or inference "
                         "(wire-inference schema)")
    ap.add_argument("--steps", type=int, default=None,
                    help="optimizer steps (finetune) / decode tokens per "
                         "request (inference)")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="tokens per sequence window")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="sequences per microbatch (finetune) / prompts "
                         "served (inference)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--capture-every", type=int, default=None,
                    help="capture when step %% capture_every == 0 "
                         "(step 0 is always captured)")
    ap.add_argument("--defense", default=None, choices=_DEFENSES,
                    help="seam defense; captures record the post-defense "
                         "tensor (what the body receives). The pack "
                         "(sparsify/mid-lite/at-lite/santext/custext/"
                         "snd-lite) is documented in the module docstring")
    ap.add_argument("--sigma", type=float, default=None,
                    help="gaussian seam-noise std (0 = no draw, identical "
                         "to none)")
    ap.add_argument("--sparsity", type=float, default=None,
                    help="sparsify: fraction of top-magnitude seam "
                         "entries KEPT per row, in (0,1] (default 0.1; "
                         "1 = keep all, identical to none)")
    ap.add_argument("--bottleneck-dim", type=int, default=None,
                    help="mid-lite: fixed Gaussian projection width k, "
                         "<= seam width (default 8)")
    ap.add_argument("--at-eps", type=float, default=None,
                    help="at-lite: seam shift along the EMA top-PCA "
                         "direction (default 0.1; 0 = off, identical to "
                         "none)")
    ap.add_argument("--santext-p", type=float, default=None,
                    help="santext: per-token replacement probability, in "
                         "[0,1] (default 0.1)")
    ap.add_argument("--santext-k", type=int, default=None,
                    help="santext/custext: cosine-neighbor pool size "
                         "(default 8)")
    ap.add_argument("--snd-sigma", type=float, default=None,
                    help="snd-lite: additive-share scale (default 1.0; "
                         "0 = off, identical to none)")
    ap.add_argument("--tb-dim", type=int, default=None,
                    help="trained-bottleneck: learned encoder release "
                         "width k (default 64)")
    ap.add_argument("--tb-hidden", type=int, default=None,
                    help="trained-bottleneck: encoder MLP hidden width "
                         "(default 256)")
    ap.add_argument("--tb-probe-hidden", type=int, default=None,
                    help="trained-bottleneck: probe / MINE statistics-net "
                         "width (default 128)")
    ap.add_argument("--tb-minimax-steps", type=int, default=None,
                    help="trained-bottleneck: alternating probe/encoder "
                         "updates in the calibration pre-phase (default "
                         "200)")
    ap.add_argument("--tb-mi-weight", type=float, default=None,
                    help="trained-bottleneck: MINE penalty weight "
                         "(default 0.1; the reading is logged as a "
                         "diagnostic, not a certificate)")
    ap.add_argument("--tb-allow-cpu", action="store_true", default=None,
                    help="trained-bottleneck: opt in to CPU minimax "
                         "training (GPU-only by default; tiny models only)")
    ap.add_argument("--tb-checkpoint", default=None,
                    help="trained-bottleneck: reuse a trained encoder "
                         "checkpoint (.pt or its directory) instead of "
                         "running the pre-phase")
    ap.add_argument("--tb-cotrain", action="store_true", default=None,
                    help="trained-bottleneck: co-train the encoder during "
                         "victim finetuning (interleaved minimax rounds; "
                         "the encoder is NOT frozen after the pre-phase)")
    ap.add_argument("--tb-cotrain-interval", type=int, default=None,
                    help="trained-bottleneck: one co-training round every "
                         "N victim steps (default 8)")
    ap.add_argument("--pca-dim", type=int, default=None,
                    help="pca-compress: released PCA component count k, "
                         "<= seam width (default 64)")
    ap.add_argument("--mixup-lambda", type=float, default=None,
                    help="secret-mixup: convex mix weight on the real "
                         "seam row, in (0,1] (default 0.75)")
    ap.add_argument("--mixup-pool", type=int, default=None,
                    help="secret-mixup: recycled-decoy pool cap in rows "
                         "(FIFO eviction; default 256)")
    ap.add_argument("--no-mixup-correction", dest="mixup_correction",
                    action="store_false", default=None,
                    help="secret-mixup: disable the victim-side "
                         "decoy-forward return-gradient correction (on "
                         "by default; approximate under a nonlinear "
                         "cloud segment)")
    ap.add_argument("--replica-count", type=int, default=None,
                    help="replica-median: number of independent in-process "
                         "body replicas K (default 3; K=2 selects the "
                         "disagreement/abort variant). K bodies multiply "
                         "body-slice memory by ~K — the harness warns, "
                         "never silently downsizes")
    ap.add_argument("--replica-tolerance", type=float, default=None,
                    help="replica-median: per-sample relative deviation-"
                         "from-group-mean flag threshold (default 0.02)")
    ap.add_argument("--no-replica-gauge", dest="replica_gauge",
                    action="store_false", default=None,
                    help="replica-median: disable the per-channel seeded "
                         "gauge perturbation (on by default; off makes all "
                         "channels identical -> zero honest heterogeneity)")
    ap.add_argument("--replica-gauge-scale", type=float, default=None,
                    help="replica-median: relative scale of the "
                         "per-channel gauge perturbation (default 1e-3)")
    ap.add_argument("--server-attack", default=None,
                    choices=("none", "passive-amplify"),
                    help="malicious-server hook (finetune only): "
                         "passive-amplify crafts the body computation as "
                         "y = Body(h) + alpha*h through the shared tail, "
                         "so the returned seam gradient carries an "
                         "alpha-scaled per-row copy of the token signal "
                         "(see module docstring)")
    ap.add_argument("--server-attack-alpha", type=float, default=None,
                    help="public amplification scalar of the "
                         "passive-amplify craft (default 8.0)")
    ap.add_argument("--lora-r", type=int, default=None,
                    help=">0 wraps the body slice in peft LoRA adapters")
    ap.add_argument("--lora-alpha", type=float, default=None)
    ap.add_argument("--trainable-range", default=None, metavar="START,END",
                    help="frozen-backbone segment training (finetune "
                         "only): freeze every parameter except blocks "
                         "[START, END); mutually exclusive with --lora-r")
    ap.add_argument("--quant-bits", type=int, default=None,
                    help="quant: signed per-sequence absmax quantization "
                         "width, 2..16 (default 8; always rounds)")
    ap.add_argument("--quant-grad", action="store_true", default=None,
                    help="quant: also quantize the RETURNED gradient at "
                         "--quant-bits (composed after --grad-clip and "
                         "--grad-noise)")
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="return-path hook: global-L2 clip on the "
                         "returned gradient (0 = off; any defense)")
    ap.add_argument("--grad-noise", type=float, default=None,
                    help="return-path hook: relative-RMS Gaussian noise "
                         "scale on the returned gradient (0 = off; any "
                         "defense)")
    ap.add_argument("--device", default=None, choices=("cpu", "cuda"))
    ap.add_argument("--dtype", default=None, choices=("fp32", "bf16"))
    ap.add_argument("--corpus-file", default=None,
                    help="local text file (preferred over --dataset)")
    ap.add_argument("--dataset", default=None,
                    help="fallback corpus when no --corpus-file "
                         "(default wikitext = Salesforce/wikitext "
                         "wikitext-2-raw-v1, needs the datasets package)")
    ap.add_argument("--capture-dir", default=None,
                    help="where wire_NNNNN.{pt,json} pairs land")
    ap.add_argument("--output", default=None,
                    help="artifact JSON path; <output>.jsonl is the "
                         "crash-safe step journal")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save-canonical", action="store_true", default=None,
                    help="dump captured-step input_ids to "
                    "<capture-dir>/canonical.pt (ground truth for attack "
                    "scoring; row-aligned to fwd captures, step-major)")
    ap.add_argument("--body-grad-capture", action="store_true",
                    default=None,
                    help="on step 0 only, save per-parameter body-slice "
                    "gradients to <capture-dir>/body_grads_step0.pt (the "
                    "batch-label-recon observation surface)")
    ap.add_argument("--evidence", action="store_true",
                    help="seal the run with an attacker.evidence chain "
                         "(<output>.evidence.json[l]); requires --output")
    ap.add_argument("--self-test", action="store_true",
                    help="run the pure-python fixtures and exit "
                         "(torch-free)")
    return ap


def _config_from_args(args):
    cfg = BenchConfig.from_json(args.config) if args.config \
        else BenchConfig()
    for f in dataclasses.fields(BenchConfig):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    cfg.validate()
    return cfg


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    try:
        cfg = _config_from_args(args)
    except ValueError as e:
        raise SystemExit(f"[splitbench] {e}") from None
    return run(cfg, evidence=args.evidence)


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import tempfile

    print("splitbench harness (pure-python layer):")
    plan = SplitPlan.parse("6", n_layers=12)
    check("parse('6') -> 2-slice boundary at layer 6",
          plan.boundaries == (6,) and plan.n_slices == 2)
    check("2-slice ranges are head | body+tail",
          plan.slice_ranges() == [(0, 6), (6, 12)])
    plan3 = SplitPlan.parse("4,8", n_layers=12)
    check("parse('4,8') -> 3-slice ranges",
          plan3.n_slices == 3
          and plan3.slice_ranges() == [(0, 4), (4, 8), (8, 12)])
    check("whitespace-tolerant parse",
          SplitPlan.parse(" 4 , 8 ").boundaries == (4, 8))
    check("describe names the slices",
          "head[0:4]" in plan3.describe() and "tail[8:12]"
          in plan3.describe())
    for bad in ("", "x", "4,x", "4,,8", "4,4", "8,4", "0", "-1", "1.5"):
        try:
            SplitPlan.parse(bad)
            check(f"parse({bad!r}) raises", False)
        except ValueError:
            check(f"parse({bad!r}) raises", True)
    for bad in ("12", "13"):
        try:
            SplitPlan.parse(bad, n_layers=12)
            check(f"bound parse({bad!r}, 12) raises", False)
        except ValueError:
            check(f"bound parse({bad!r}, 12) raises", True)
    try:
        SplitPlan.parse("6").slice_ranges()
        check("unbound slice_ranges raises", False)
    except ValueError:
        check("unbound slice_ranges raises", True)

    check("gpt2 registered; unknown arch is a descriptive error",
          ARCH_REGISTRY.get("gpt2") is Gpt2Split)
    check("mistral + llama registry keys share the family splitter",
          ARCH_REGISTRY.get("mistral") is MistralSplit
          and ARCH_REGISTRY.get("llama") is MistralSplit)
    check("qwen3 registry key reuses the family splitter (verified 5.15)",
          ARCH_REGISTRY.get("qwen3") is MistralSplit)
    try:
        get_arch_splitter("llama-ish")
        check("unknown arch error names how to register", False)
    except ValueError as e:
        check("unknown arch error names how to register",
              "ARCH_REGISTRY" in str(e) and "llama-ish" in str(e))

    def fake_saver(path, tensor):
        with open(path, "wb") as f:
            f.write(b"FAKE:" + repr(tensor).encode())

    from .captures import scan_captures, validate_sidecar
    with tempfile.TemporaryDirectory() as d:
        with CaptureWriter(d, "training", "st", saver=fake_saver) as w:
            w.write_training(("h", 1), mb_id=0, phase="fwd", step=0,
                             epoch=0)
            w.write_training(("g", 1), mb_id=0, phase="bwd", step=0,
                             epoch=0)
            check("writer counts captures", w.count == 2)
        recs = scan_captures(d, "training")
        check("scan_captures finds the training pair",
              [(m["phase"], m["step"], m["epoch"]) for m, _ in recs]
              == [("fwd", 0, 0), ("bwd", 0, 0)])
        check("written sidecars pass validate_sidecar",
              all(validate_sidecar(m, "training") for m, _ in recs))
        check("payload went through the fake saver",
              open(recs[0][1], "rb").read().startswith(b"FAKE:"))
        check("crash-safe: no .tmp files left",
              not any(".tmp" in f for f in os.listdir(d)))
        idir = os.path.join(d, "inf")
        with CaptureWriter(idir, "inference", "si", saver=fake_saver) as w2:
            w2.write_inference(("h",), request_seq=0, phase="prefill",
                               position=0, epoch=0)
        irecs = scan_captures(idir, "inference")
        check("inference pair scans with its own schema",
              [(m["phase"], m["position"]) for m, _ in irecs]
              == [("prefill", 0)])
        try:
            w2._write(("x",), {"session_id": "si", "mb_id": 0})
            check("provider validation fires on a buggy sidecar", False)
        except ValueError:
            check("provider validation fires on a buggy sidecar", True)
        try:
            w2.write_training(("x",), mb_id=0, phase="fwd", step=0,
                              epoch=0)
            check("mode-mismatched write raises", False)
        except ValueError:
            check("mode-mismatched write raises", True)
        try:
            w2.write_inference(("x",), request_seq=0, phase="decode",
                               position=1, epoch=0)
            check("writes after close() raise", False)
        except ValueError:
            check("writes after close() raise", True)

    cfg = BenchConfig()
    check("default BenchConfig validates", cfg.validate() is cfg)
    check("body_grad_capture defaults off and validates",
          cfg.body_grad_capture is False
          and BenchConfig(body_grad_capture=True).validate()
          .body_grad_capture is True)
    try:
        BenchConfig(body_grad_capture="yes").validate()
        check("non-bool body_grad_capture raises", False)
    except ValueError:
        check("non-bool body_grad_capture raises", True)
    check("server_attack defaults off with public alpha and validates",
          cfg.server_attack == "none" and cfg.server_attack_alpha == 8.0
          and BenchConfig(server_attack="passive-amplify").validate()
          .server_attack == "passive-amplify")
    drifts = ({"mode": "train"}, {"defense": "dp"}, {"device": "tpu"},
              {"dtype": "fp16"}, {"steps": 0}, {"seq_len": 1},
              {"batch_size": 0}, {"lr": 0.0}, {"capture_every": 0},
              {"sigma": -0.1}, {"lora_r": -1}, {"lora_alpha": 0.0},
              {"split": "0"}, {"split": "4,4"},
              {"server_attack": "active-x"},
              {"server_attack_alpha": -1.0},
              {"quant_bits": 1}, {"quant_bits": 17}, {"quant_bits": 2.5},
              {"quant_grad": "yes"},
              {"defense": "none", "quant_grad": True},
              {"grad_clip": -0.1}, {"grad_noise": -0.1},
              {"trainable_range": "6,2"}, {"trainable_range": "x"},
              {"trainable_range": "2,2"}, {"trainable_range": "2,6,8"},
              {"trainable_range": "2,6", "lora_r": 4},
              {"trainable_range": "2,6", "mode": "inference"})
    for kw in drifts:
        try:
            BenchConfig(**kw).validate()
            check(f"config drift {kw} raises", False)
        except ValueError:
            check(f"config drift {kw} raises", True)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "cfg.json")
        with open(p, "w") as f:
            json.dump({"model": "gpt2", "split": "4", "steps": 10}, f)
        loaded = BenchConfig.from_json(p)
        check("from_json round-trips known keys",
              loaded.split == "4" and loaded.steps == 10)
        with open(p, "w") as f:
            json.dump({"modle": "gpt2"}, f)
        try:
            BenchConfig.from_json(p)
            check("from_json unknown key raises naming it", False)
        except ValueError as e:
            check("from_json unknown key raises naming it",
                  "modle" in str(e))

    print("defense pack (pure-python layer):")
    check("all defense names validate (none, gaussian + the 6-pack + "
          "trained-bottleneck + secret-mixup + replica-median + "
          "pca-compress + quant)",
          all(BenchConfig(defense=d).validate().defense == d
              for d in _DEFENSES) and len(_DEFENSES) == 13)
    check("pack knob defaults are the documented ones",
          cfg.sparsity == 0.1 and cfg.bottleneck_dim == 8
          and cfg.at_eps == 0.1 and cfg.santext_p == 0.1
          and cfg.santext_k == 8 and cfg.snd_sigma == 1.0)
    check("secret-mixup knob defaults are the documented ones",
          cfg.mixup_lambda == 0.75 and cfg.mixup_pool == 256
          and cfg.mixup_correction is True)
    check("replica-median knob defaults are the documented ones",
          cfg.replica_count == 3 and cfg.replica_tolerance == 0.02
          and cfg.replica_gauge is True
          and cfg.replica_gauge_scale == 1e-3)
    check("quant + return-path knob defaults are the documented ones",
          cfg.quant_bits == 8 and cfg.quant_grad is False
          and cfg.grad_clip == 0.0 and cfg.grad_noise == 0.0
          and cfg.trainable_range is None)
    check("trainable_range parses and validates",
          BenchConfig(trainable_range="2,6").validate().trainable_range
          == "2,6"
          and BenchConfig(defense="quant", quant_bits=4, quant_grad=True,
                          grad_clip=1.0, grad_noise=0.1).validate()
          .quant_bits == 4)
    pack_drifts = ({"defense": "mid"}, {"defense": "sparsifier"},
                   {"sparsity": 0.0}, {"sparsity": 1.5},
                   {"sparsity": "half"}, {"bottleneck_dim": 0},
                   {"bottleneck_dim": 2.5}, {"at_eps": -0.1},
                   {"santext_p": 1.5}, {"santext_p": -0.1},
                   {"santext_k": 0}, {"snd_sigma": -0.5},
                   {"mixup_lambda": 0.0}, {"mixup_lambda": 1.5},
                   {"mixup_lambda": "l"}, {"mixup_pool": 0},
                   {"mixup_pool": 2.5}, {"mixup_correction": "yes"},
                   {"replica_count": 1}, {"replica_count": 2.5},
                   {"replica_count": True}, {"replica_tolerance": 0.0},
                   {"replica_tolerance": -0.1}, {"replica_gauge": "yes"},
                   {"replica_gauge_scale": 0.0},
                   {"replica_gauge_scale": -1},
                   {"tb_cotrain": "yes"}, {"tb_cotrain": 1},
                   {"tb_cotrain_interval": 0}, {"tb_cotrain_interval": 2.5},
                   {"pca_dim": 0}, {"pca_dim": -2}, {"pca_dim": 2.5})
    for kw in pack_drifts:
        try:
            BenchConfig(**kw).validate()
            check(f"pack config drift {kw} raises", False)
        except ValueError:
            check(f"pack config drift {kw} raises", True)

    rows = [[2.0, 0.1], [1.9, -0.1], [2.1, 0.05],
            [-1.9, 0.1], [-2.0, -0.05], [-2.1, 0.0]]
    pca = SeamPcaState(2)
    check("PCA state: no direction before any update",
          pca.top_direction() is None)
    pca.update(rows)
    v1 = pca.top_direction()
    check("PCA state: top direction of an x-dominated cloud is +e1",
          v1 is not None and v1[0] > 0.999 and abs(v1[1]) < 0.01)
    pca2 = SeamPcaState(2)
    pca2.update(rows)
    check("PCA state: deterministic per update history",
          pca2.top_direction() == v1 and pca2.n_updates == 1)
    blend = SeamPcaState(2, momentum=0.5)
    blend.update_cov([[2.0, 0.0], [0.0, 2.0]])
    blend.update_cov([[4.0, 2.0], [2.0, 0.0]])
    check("PCA state: EMA blend is the momentum-weighted mean",
          blend.cov == [[3.0, 1.0], [1.0, 1.0]] and blend.n_updates == 2)
    zero = SeamPcaState(2)
    zero.update_cov([[0.0, 0.0], [0.0, 0.0]])
    check("PCA state: zero covariance -> no direction (degenerate)",
          zero.top_direction() is None)
    try:
        SeamPcaState(2).update([[1.0, 2.0]])
        check("PCA state: <2 rows raises", False)
    except ValueError:
        check("PCA state: <2 rows raises", True)

    table = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]
    nbrs = cosine_neighbors(table, 1)
    check("cosine_neighbors: hand-computed top-1 sets",
          nbrs == [[1], [0], [1], [2]])
    check("cosine_neighbors: k=2 ranks by cosine, self excluded",
          cosine_neighbors(table, 2)[0] == [1, 2])
    san_id = TokenSanitizer("santext", lambda t: nbrs[t], p=0.0, k=1,
                            seed=5)
    check("santext p=0 is the identity map",
          san_id.sanitize([[0, 1, 2, 3]]) == [[0, 1, 2, 3]])
    s1 = TokenSanitizer("santext", lambda t: nbrs[t], p=1.0, k=1, seed=5)
    s2 = TokenSanitizer("santext", lambda t: nbrs[t], p=1.0, k=1, seed=5)
    out1, out2 = s1.sanitize([[0, 1, 2, 3]]), s2.sanitize([[0, 1, 2, 3]])
    check("santext p=1 replaces every token with its top-1 neighbor",
          out1 == [[1, 0, 1, 2]])
    check("santext: same seed -> same stream (deterministic per run)",
          out1 == out2)
    nbrs2 = cosine_neighbors(table, 2)
    c1 = TokenSanitizer("custext", lambda t: nbrs2[t], p=0.0, k=2, seed=5)
    c2 = TokenSanitizer("custext", lambda t: nbrs2[t], p=0.0, k=2, seed=5)
    m1, m2 = c1.sanitize([[0, 1], [2, 3]]), c2.sanitize([[3, 2], [1, 0]])
    flat1 = [m1[0][0], m1[0][1], m1[1][0], m1[1][1]]
    flat2 = [m2[1][1], m2[1][0], m2[0][1], m2[0][0]]
    check("custext: fixed per-token mapping, order-independent",
          flat1 == flat2)
    check("custext: mapping stays inside the neighbor pool (never self)",
          all(flat1[t] in nbrs2[t] for t in range(4)))
    check("custext: same seed -> identical mapping across instances",
          c1.sanitize([[0, 1, 2, 3]]) ==
          TokenSanitizer("custext", lambda t: nbrs2[t], p=0.0, k=2,
                         seed=5).sanitize([[0, 1, 2, 3]]))
    try:
        TokenSanitizer("santex", lambda t: [], p=0.5, k=1, seed=0)
        check("sanitizer unknown mode raises", False)
    except ValueError:
        check("sanitizer unknown mode raises", True)

    recs = {d: _defense_config_record(BenchConfig(defense=d))
            for d in _DEFENSES}
    check("defense_config record carries the active defense's knobs",
          recs["none"] == {"name": "none"}
          and recs["gaussian"] == {"name": "gaussian", "sigma": 0.0}
          and recs["sparsify"] == {"name": "sparsify", "sparsity": 0.1}
          and recs["mid-lite"] == {"name": "mid-lite", "bottleneck_dim": 8}
          and recs["at-lite"] == {"name": "at-lite", "at_eps": 0.1}
          and recs["santext"] == {"name": "santext", "santext_p": 0.1,
                                  "santext_k": 8}
          and recs["custext"] == {"name": "custext", "santext_k": 8}
          and recs["snd-lite"] == {"name": "snd-lite", "snd_sigma": 1.0}
          and recs["secret-mixup"] == {"name": "secret-mixup",
                                       "lambda": 0.75, "pool_size": 256,
                                       "correction": True}
          and recs["replica-median"] == {"name": "replica-median",
                                         "replica_count": 3,
                                         "tolerance": 0.02,
                                         "gauge": True,
                                         "gauge_scale": 1e-3}
          and recs["pca-compress"] == {"name": "pca-compress",
                                       "bottleneck_dim": 64}
          and recs["quant"] == {"name": "quant", "quant_bits": 8,
                                "quant_grad": False})
    check("defense_config record carries active return-path knobs only",
          _defense_config_record(
              BenchConfig(defense="quant", quant_bits=4, quant_grad=True,
                          grad_clip=1.0, grad_noise=0.1,
                          trainable_range="2,6")) ==
          {"name": "quant", "quant_bits": 4, "quant_grad": True,
           "grad_clip": 1.0, "grad_noise": 0.1,
           "trainable_range": "2,6"})

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
