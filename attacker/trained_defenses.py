#!/usr/bin/env python3
"""Trained compression-bottleneck defense (the ``trained-bottleneck``
splitbench seam defense) — a LEARNED bottleneck, unlike mid-lite's fixed
random projection (attacker/defenses.py SEAM_DEFENSE_PACK).

This is our own re-implementation of the adversarial-bottleneck + MINE
defense family for the splitbench capture rig, written so the attack side
can be validated against actual trained-defender captures instead of
synthetic fixtures. It is NOT the original defense project's code or
weights.

Mechanism v2 (utility repair; distinct from historical v1)
  A public PCA encoder/decoder pair initializes a k-dimensional representation.
  A learned MLP adds a bounded residual (0.05 times calibration RMS per latent
  coordinate). The encoder minimizes normalized reconstruction distortion plus
  small bounded adversarial/MINE terms. The cloud decodes the released coordinates
  before its pretrained body; it no longer interprets padded latents as original
  hidden coordinates. This is a conservative utility-constrained compression
  baseline, not a faithful reproduction of an external defense or a privacy proof.
  The wire contains k coordinates padded to H for capture compatibility. Gradients
  follow the actual encoder and cloud decoder. The decoder is public and the
  observer can use it; it is not a secret key. MINE is only a diagnostic.
  Checkpoints carry mechanism v2 geometry. Historical v1 checkpoints retain their
  original unconstrained/identity-gradient behavior for explicit replay only.

Training happens as a PRE-PHASE on calibration data before victim
finetuning (splitbench.make_defense_state, driven by finetune/infer);
the trained encoder checkpoint lands in the run's output dir as
provenance (``trained_bottleneck_encoder.pt`` + the
``llm_attacker.trained_bottleneck_provenance.v1`` sidecar, whose path is
recorded in the run artifact).

Torch: this module imports torch-free; require_torch() runs inside every
training/application path (idiom attacker/attacks/common.py). Seeding is
torch.random.fork_rng + torch.manual_seed (idiom
attacker/attacks/representation_privacy.py): same seed + same data ->
bit-identical encoder checkpoint and captured releases, and the victim
run's RNG stream is never disturbed by the defense pre-phase.

Converter: capture_to_representation_bundle() turns a trained-bottleneck
capture directory (fwd wire captures + canonical.pt) into a
representation_privacy.v1 bundle over the REAL trained k-dim releases, so
`python -m attacker --mode ... --attack representation-privacy --bundle`
consumes them; validated with representation_privacy.validate_bundle.

Co-training (the splitbench ``tb_cotrain`` mode): the encoder is NOT
frozen after the pre-phase; every ``tb_cotrain_interval`` victim optimizer
steps the driver runs ONE interleaved minimax round on the current batch
(BottleneckCotrain: probes adversarially updated against the frozen
encoder snapshot, then the encoder minimizes normalized reconstruction
distortion through the public decoder — the utility signal — plus small
bounded adversarial/MINE terms). The encoder's weights move only in these
rounds; the victim gradient path is unchanged. Provenance records
cotrain enabled/interval/updates_performed.

PCA compression (the ``pca-compress`` splitbench seam defense): the
compression-only CONTROL for the bottleneck family. A full-SVD PCA fit on
the calibration seam activations projects H -> k (PcaConfig.
bottleneck_dim, default 64); the wire carries the k components under the
same zeros-plus-projection convention with a straight-through gradient,
and the public decoder reconstructs z @ basis.T + mean. No adversarial
training. Provenance: llm_attacker.pca_compress_provenance.v1 sidecar
(explained variance at k recorded — the key diagnostic) +
pca_compress_encoder.pt; the same capture_to_representation_bundle()
converter consumes its captures.

Secret mixup (the ``secret-mixup`` splitbench seam defense — our in-repo
re-implementation of the secret-mixup + decoy-correction defense family,
not the original defense project's): SecretMixup is a release-time
transform WITH STATE. The decoy pool recycles REAL latent rows from
earlier training blocks of the same run (with honest labels tracked);
releases are lambda*h + (1-lambda)*z_decoy (lambda default 0.75), decoys
drawn per block without replacement under CSPRNG-derived seeds. The pool
starts EMPTY: blocks released while it is empty go out UNDEFENDED and are
flagged policy="undefended_empty_pool" in the provenance sidecar (the
first block of a run is always undefended; its rows become the pool). The
victim's returned-gradient path corrects for the decoy component via a
parallel decoy forward (corrected_gradient) — EXACT for a linear cloud
segment, APPROXIMATE under a nonlinear one (the defense framework's own
caveat). Provenance: llm_attacker.secret_mixup_provenance.v1 sidecar +
the drawn-decoy store (secret_mixup_decoys.pt); the
capture_to_view_privacy_bundle() converter turns the capture into a
view_privacy.v1 mixup bundle the `view-privacy --bundle` de-mixing arm
consumes (validated with view_privacy.evaluate_bundle).

Replica median (the ``replica-median`` splitbench defense — our in-repo
re-implementation of the K-redundant-channel + coordinate-median integrity
defense family, not the original defense project's): K INDEPENDENT body
replicas (default 3; K=2 selectable for the disagreement/abort case) run
in parallel in-process during victim finetuning. Each channel has its own
seed (cfg.seed ^ seed-XOR offset ^ channel index) and its own gauge (a
per-channel seeded weight perturbation of the body slice), so honest
channels genuinely disagree; the cloud outputs are aggregated by
COORDINATE MEDIAN (K >= 3; coordinate MEAN for K=2, with a recorded
disagreement/abort policy) before the shared top (norm + lm_head).
Per sample the relative deviation of each channel from the group mean is
checked against the tolerance (default 0.02); the per-block integrity
journal records which replica deviated, the magnitude, and whether it was
flagged. The honest-heterogeneity noise budget (the independent channels'
own disagreement distribution) is measured on calibration data — or from
the run's own blocks when no calibration is available — and logged in the
llm_attacker.replica_median_provenance.v1 sidecar so attack-side tolerance
matching is fair. The driver captures K per-channel wires per block (in
<out_dir>/channels/ch<k>/, same wire-capture format, separate files) plus
the aggregated median wire as the standard fwd capture, so BOTH the
replica-integrity battery and the seam-reconstruction baselines consume
the trained defender. capture_to_replica_bundle() converts the capture
into llm_attacker.replica_observations.v1 / .replica_truth.v1 JSONs that
`replica-integrity --captures/--truth` consumes directly (validated with
validate_observations/validate_truth); the scoring truth is the per-block
coordinate GROUP MEAN of the honest channels (the scheme has no external
ground truth — documented in the converter).

CLI (stdlib-only parser; convert needs torch to load .pt payloads):
    python -m attacker.trained_defenses convert --capture-dir D \\
        [--canonical D/canonical.pt] [--dim K] --output bundle.json
    python -m attacker.trained_defenses convert-mixup --capture-dir D \\
        --output bundle.json
    python -m attacker.trained_defenses convert-replica --capture-dir D \\
        --observations obs.json --truth truth.json
"""

import argparse
import dataclasses
import hashlib
import json
import math
import os

try:
    import torch
except ImportError:
    torch = None

PROVENANCE_SCHEMA = "llm_attacker.trained_bottleneck_provenance.v1"
ENCODER_NAME = "trained_bottleneck_encoder.pt"
PROVENANCE_NAME = "trained_bottleneck_provenance.json"

MIXUP_PROVENANCE_SCHEMA = "llm_attacker.secret_mixup_provenance.v1"
MIXUP_PROVENANCE_NAME = "secret_mixup_provenance.json"
MIXUP_DECOYS_NAME = "secret_mixup_decoys.pt"

REPLICA_PROVENANCE_SCHEMA = "llm_attacker.replica_median_provenance.v1"
REPLICA_PROVENANCE_NAME = "replica_median_provenance.json"
REPLICA_JOURNAL_NAME = "replica_median_journal.jsonl"

PCA_PROVENANCE_SCHEMA = "llm_attacker.pca_compress_provenance.v1"
PCA_ENCODER_NAME = "pca_compress_encoder.pt"
PCA_PROVENANCE_NAME = "pca_compress_provenance.json"

_COTRAIN_SEED_XOR = 0x5EEDC07A

TORCH_ERROR = ("attacker.trained_defenses: the trained-bottleneck defense "
               "requires torch (the seam tensors it transforms are torch "
               "tensors). Install torch or use a torch venv; --help and "
               "the provenance/config helpers work torch-less.")

CPU_ERROR = ("attacker.trained_defenses: the trained-bottleneck encoder "
             "minimax pre-phase is GPU-only by default (CPU minimax "
             "training over real seam activations is impractically slow). "
             "Re-run on a CUDA host, or explicitly opt in to CPU training "
             "with allow_cpu=True / --tb-allow-cpu (appropriate only for "
             "tiny test-scale models and a handful of minimax steps).")


def require_torch():
    if torch is None:
        raise SystemExit(TORCH_ERROR)


@dataclasses.dataclass
class BottleneckConfig:
    """Knobs for the trained-bottleneck minimax pre-phase."""
    bottleneck_dim: int = 64
    encoder_hidden: int = 256
    probe_hidden: int = 128
    property_classes: int = 8
    minimax_steps: int = 200
    mi_weight: float = 0.1
    lr: float = 1e-3
    batch_rows: int = 1024
    seed: int = 0
    allow_cpu: bool = False

    def validate(self):
        def bad(msg):
            raise ValueError(f"BottleneckConfig: {msg}")
        for name in ("bottleneck_dim", "encoder_hidden", "probe_hidden",
                     "property_classes", "minimax_steps", "batch_rows",
                     "seed"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1 \
                    and name != "seed":
                bad(f"{name} must be an integer >= 1, got {v!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) \
                or self.seed < 0:
            bad(f"seed must be an integer >= 0, got {self.seed!r}")
        if not isinstance(self.mi_weight, (int, float)) \
                or isinstance(self.mi_weight, bool) or self.mi_weight < 0:
            bad(f"mi_weight must be a number >= 0, got {self.mi_weight!r}")
        if not isinstance(self.lr, (int, float)) \
                or isinstance(self.lr, bool) or self.lr <= 0:
            bad(f"lr must be a number > 0, got {self.lr!r}")
        if not isinstance(self.allow_cpu, bool):
            bad(f"allow_cpu must be a boolean, got {self.allow_cpu!r}")
        return self

    def canonical(self):
        return dataclasses.asdict(self)

    def config_sha256(self):
        blob = json.dumps(self.canonical(), sort_keys=True,
                          separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_nets(seam_width, vocab, cfg, device, dtype):
    """Encoder + adversarial probes + MINE statistics net."""
    encoder = torch.nn.Sequential(
        torch.nn.Linear(seam_width, cfg.encoder_hidden),
        torch.nn.GELU(),
        torch.nn.Linear(cfg.encoder_hidden, cfg.bottleneck_dim))
    token_probe = torch.nn.Sequential(
        torch.nn.Linear(cfg.bottleneck_dim, cfg.probe_hidden),
        torch.nn.GELU(),
        torch.nn.Linear(cfg.probe_hidden, vocab))
    property_probe = torch.nn.Sequential(
        torch.nn.Linear(cfg.bottleneck_dim, cfg.probe_hidden),
        torch.nn.GELU(),
        torch.nn.Linear(cfg.probe_hidden, cfg.property_classes))
    recon_probe = torch.nn.Linear(cfg.bottleneck_dim, seam_width)
    mine_net = torch.nn.Sequential(
        torch.nn.Linear(seam_width + cfg.bottleneck_dim, cfg.probe_hidden),
        torch.nn.GELU(),
        torch.nn.Linear(cfg.probe_hidden, cfg.probe_hidden),
        torch.nn.GELU(),
        torch.nn.Linear(cfg.probe_hidden, 1))
    nets = {"encoder": encoder, "token_probe": token_probe,
            "property_probe": property_probe, "recon_probe": recon_probe,
            "mine_net": mine_net}
    for net in nets.values():
        net.to(device=device, dtype=dtype)
    return nets


def _mine_reading(mine_net, h, z):
    """Donsker–Varadhan lower-bound estimate I(h; z). A READING, not a
    certificate (module docstring)."""
    joint = mine_net(torch.cat([h, z], dim=-1))
    perm = torch.randperm(h.shape[0], device=h.device)
    marginal = mine_net(torch.cat([h[perm], z], dim=-1))
    return joint.mean() - torch.logsumexp(marginal, dim=0) \
        + math.log(h.shape[0])


def train_bottleneck(hidden_rows, token_rows, cfg, device="cuda",
                     vocab=None, dtype=None):
    """Minimax-train the encoder against the embedded probes.

    hidden_rows  [N, H] seam activations (calibration, pre-finetune)
    token_rows   [N] token ids aligned to the rows
    cfg          BottleneckConfig
    device       "cuda" | "cpu" — CPU requires cfg.allow_cpu (CPU_ERROR)
    vocab        token-probe class count (default max(token)+1)

    Returns a TrainedBottleneck (encoder weights on CPU, deterministic
    under cfg.seed: fork_rng + manual_seed, so the caller's RNG stream is
    untouched).
    """
    require_torch()
    cfg.validate()
    dev = torch.device(device)
    if dev.type != "cuda" and not cfg.allow_cpu:
        raise SystemExit(CPU_ERROR)
    if hidden_rows.shape[-1] < cfg.bottleneck_dim:
        raise ValueError(f"BottleneckConfig: bottleneck_dim "
                         f"{cfg.bottleneck_dim} exceeds the seam width "
                         f"{hidden_rows.shape[-1]}")
    if hidden_rows.shape[0] != token_rows.shape[0]:
        raise ValueError("hidden_rows and token_rows must have the same "
                         "row count")
    if hidden_rows.shape[0] < 2:
        raise ValueError("need at least 2 calibration rows")
    seam_width = int(hidden_rows.shape[-1])
    if vocab is None:
        vocab = int(token_rows.max().item()) + 1
    dtype = dtype or torch.float32
    h_all = hidden_rows.detach().to(device=dev, dtype=dtype)
    t_all = token_rows.detach().to(device=dev).long()
    prop_all = t_all % cfg.property_classes
    n = h_all.shape[0]
    bs = min(cfg.batch_rows, n)
    devices = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg.seed)
        mean = h_all.mean(0)
        _, _, vh = torch.linalg.svd(h_all - mean, full_matrices=True)
        basis = vh[:cfg.bottleneck_dim].T.contiguous()
        scale = (h_all - mean).square().mean().sqrt().clamp_min(1e-6)
        nets = _build_nets(seam_width, vocab, cfg, dev, dtype)
        enc, tok_p, prop_p, rec_p, mine = (nets["encoder"],
                                           nets["token_probe"],
                                           nets["property_probe"],
                                           nets["recon_probe"],
                                           nets["mine_net"])
        torch.nn.init.zeros_(enc[-1].weight)
        torch.nn.init.zeros_(enc[-1].bias)
        def encode_rows(h):
            return (h - mean) @ basis + 0.05 * scale * torch.tanh(enc((h - mean) / scale))
        probe_params = list(tok_p.parameters()) + list(prop_p.parameters()) \
            + list(rec_p.parameters())
        probe_opt = torch.optim.Adam(probe_params, lr=cfg.lr)
        mine_opt = torch.optim.Adam(mine.parameters(), lr=cfg.lr)
        enc_opt = torch.optim.Adam(enc.parameters(), lr=cfg.lr)
        order = torch.randperm(n, device=dev)
        for step in range(cfg.minimax_steps):
            idx = order[(step * bs) % n:(step * bs) % n + bs]
            if idx.numel() < bs:
                idx = torch.cat([idx, order[:bs - idx.numel()]])
            hb, tb, pb = h_all[idx], t_all[idx], prop_all[idx]
            with torch.no_grad():
                zb = encode_rows(hb)
            probe_opt.zero_grad()
            probe_loss = torch.nn.functional.cross_entropy(tok_p(zb), tb) \
                + torch.nn.functional.cross_entropy(prop_p(zb), pb) \
                + torch.nn.functional.mse_loss(rec_p(zb), hb)
            probe_loss.backward()
            probe_opt.step()
            mine_opt.zero_grad()
            (-_mine_reading(mine, hb, zb)).backward()
            mine_opt.step()
            enc_opt.zero_grad()
            ze = encode_rows(hb)
            leak = torch.nn.functional.cross_entropy(tok_p(ze), tb) / max(1., math.log(vocab)) \
                + torch.nn.functional.cross_entropy(prop_p(ze), pb) / max(1., math.log(cfg.property_classes)) \
                + torch.nn.functional.mse_loss(rec_p(ze), hb) / scale.square()
            mi = _mine_reading(mine, hb, ze)
            reconstructed = ze @ basis.T + mean
            distortion = (reconstructed - hb).square().mean() / scale.square()
            objective = distortion - 0.001 * torch.tanh(leak) + 0.001 * cfg.mi_weight * torch.tanh(mi)
            if not torch.isfinite(objective):
                raise ValueError('non-finite bottleneck objective')
            objective.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            enc_opt.step()
        with torch.no_grad():
            z_final = encode_rows(h_all[:bs])
            mi_reading = float(_mine_reading(mine, h_all[:bs], z_final))
    state = {k: v.detach().to(device="cpu", dtype=torch.float32).clone()
             for k, v in enc.state_dict().items()}
    return TrainedBottleneck(cfg.canonical(), seam_width, vocab, state,
                             mi_reading, geometry={
                                 'mean': mean.detach().cpu(), 'basis': basis.detach().cpu(),
                                 'scale': scale.detach().cpu(), 'residual_bound': 0.05})


class TrainedBottleneck:
    """The trained encoder: wire application + checkpoint/provenance I/O.

    config/state_dict live on CPU fp32; apply() rebuilds the torch module
    lazily (per device/dtype) so the holder itself is torch-free to
    construct from disk."""

    def __init__(self, config, seam_width, vocab, state_dict, mi_reading,
                 utility=None, geometry=None):
        self.geometry = geometry
        self.config = dict(config)
        self.seam_width = int(seam_width)
        self.vocab = int(vocab)
        self.state_dict = state_dict
        self.mi_reading = float(mi_reading)
        self.utility = utility
        self.cotrain = None
        self._modules = {}

    @property
    def bottleneck_dim(self):
        return int(self.config["bottleneck_dim"])

    def _module(self, device, dtype):
        require_torch()
        key = (str(device), str(dtype))
        if key not in self._modules:
            cfg = BottleneckConfig(**self.config)
            net = torch.nn.Sequential(
                torch.nn.Linear(self.seam_width, cfg.encoder_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(cfg.encoder_hidden, cfg.bottleneck_dim))
            net.load_state_dict(self.state_dict)
            net.to(device=device, dtype=dtype).eval()
            for p in net.parameters():
                p.requires_grad_(False)
            self._modules[key] = net
        return self._modules[key]

    def encode(self, hidden):
        """Released k-vector; frozen weights, differentiable input in v2."""
        net = self._module(hidden.device, hidden.dtype)
        if self.geometry is None:
            with torch.no_grad(): return net(hidden)
        mean, basis, scale = (self.geometry[k].to(hidden) for k in ('mean','basis','scale'))
        return (hidden - mean) @ basis + self.geometry['residual_bound'] * scale * torch.tanh(net((hidden - mean) / scale))

    def decode(self, wire):
        """Cloud-side reconstruction; decoder is public, not a secret defense."""
        if self.geometry is None:
            return wire
        mean, basis = (self.geometry[k].to(wire) for k in ('mean','basis'))
        return wire[..., :self.bottleneck_dim] @ basis.T + mean

    def apply(self, hidden):
        """Padded compressed wire; true input derivatives in v2, legacy STE in v1."""
        k = self.bottleneck_dim
        if hidden.shape[-1] != self.seam_width:
            raise ValueError(f"trained-bottleneck: seam width "
                             f"{hidden.shape[-1]} != the encoder's "
                             f"{self.seam_width}")
        z = self.encode(hidden)
        wire = torch.zeros_like(hidden)
        wire[..., :k] = z
        if hidden.requires_grad and self.geometry is None:
            return hidden + (wire - hidden).detach()
        return wire


    def cotrain_module(self, device, dtype):
        """A TRAINABLE copy of the residual MLP for interleaved
        co-training (mechanism v2 only: the co-trained object is the
        bounded residual net around the fixed public PCA geometry)."""
        require_torch()
        if self.geometry is None:
            raise ValueError("trained-bottleneck co-training requires "
                             "mechanism v2 geometry (the bounded residual "
                             "around the public PCA basis); historical v1 "
                             "checkpoints are replay-only")
        cfg = BottleneckConfig(**self.config)
        net = torch.nn.Sequential(
            torch.nn.Linear(self.seam_width, cfg.encoder_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(cfg.encoder_hidden, cfg.bottleneck_dim))
        net.load_state_dict(self.state_dict)
        net.to(device=device, dtype=dtype).train()
        for p in net.parameters():
            p.requires_grad_(True)
        return net

    def sync_cotrain(self, net):
        """Write co-trained weights back into the holder and drop the
        cached eval modules so apply()/encode() pick them up."""
        self.state_dict = {k: v.detach().to(device="cpu",
                                            dtype=torch.float32).clone()
                           for k, v in net.state_dict().items()}
        self._modules.clear()


    def checkpoint_payload(self):
        return {"format": "trained_bottleneck_encoder.v2" if self.geometry is not None else "trained_bottleneck_encoder.v1",
                "geometry": self.geometry,
                "config": self.config, "seam_width": self.seam_width,
                "vocab": self.vocab, "mi_reading": self.mi_reading,
                "state_dict": self.state_dict}

    def provenance_record(self, checkpoint_path, checkpoint_sha256):
        rec = {
            "schema": PROVENANCE_SCHEMA,
            "created_by": "attacker.trained_defenses (in-repo "
                          "re-implementation of the adversarial-bottleneck "
                          "+ MINE defense family; NOT the original defense "
                          "project's code or weights)",
            "config": self.config,
            "config_sha256": BottleneckConfig(**self.config)
                .config_sha256(),
            "mechanism_version": 2 if self.geometry is not None else 1,
            "utility_constraint": "PCA reconstruction plus bounded residual; public cloud decoder" if self.geometry is not None else "legacy unconstrained",
            "objective_v2": {"normalized_distortion_weight": 1.0, "bounded_adversary_weight": 0.001,
                             "bounded_mi_weight": 0.001 * self.config['mi_weight'],
                             "latent_residual_rms_bound": 0.05} if self.geometry is not None else None,
            "seed": self.config["seed"],
            "seam_width": self.seam_width,
            "bottleneck_dim": self.bottleneck_dim,
            "vocab": self.vocab,
            "checkpoint": {"path": os.fspath(checkpoint_path),
                           "sha256": checkpoint_sha256},
            "mi_reading": {
                "value": self.mi_reading,
                "estimator": "MINE-style Donsker-Varadhan lower bound "
                             "from the defender's own statistics net",
                "certificate": False,
                "caveat": "diagnostic reading logged for provenance; NOT "
                          "a certificate of the true mutual information "
                          "and not an upper bound on attacker leakage"},
        }
        if self.cotrain is not None:
            rec["cotrain"] = self.cotrain
        if self.utility is not None:
            rec["utility"] = self.utility
        return rec

    def save(self, out_dir):
        """Write the encoder checkpoint + provenance sidecar into the
        run's output dir; returns the provenance record."""
        require_torch()
        os.makedirs(out_dir, exist_ok=True)
        ckpt = os.path.join(out_dir, ENCODER_NAME)
        torch.save(self.checkpoint_payload(), ckpt)
        rec = self.provenance_record(ckpt, _file_sha256(ckpt))
        sidecar = os.path.join(out_dir, PROVENANCE_NAME)
        with open(sidecar, "w") as f:
            json.dump(rec, f, indent=2, sort_keys=True)
        rec["provenance_path"] = sidecar
        return rec

    @classmethod
    def load(cls, path):
        """Load a checkpoint written by save() (file or its directory)."""
        require_torch()
        if os.path.isdir(path):
            path = os.path.join(path, ENCODER_NAME)
        payload = torch.load(path, weights_only=False)
        if payload.get("format") not in ("trained_bottleneck_encoder.v1", "trained_bottleneck_encoder.v2"):
            raise ValueError(f"{path}: not a "
                             "trained_bottleneck_encoder.v1 checkpoint")
        return cls(payload["config"], payload["seam_width"],
                   payload["vocab"], payload["state_dict"],
                   payload["mi_reading"], geometry=payload.get("geometry"))


class BottleneckCotrain:
    """Interleaved minimax co-training of the trained-bottleneck encoder
    DURING victim finetuning (the splitbench ``tb_cotrain`` mode).

    The frozen-encoder configuration starves victim training when the
    pre-phase ends before the victim's head moves (the defended pilot's
    utility collapse motivated this variant). Here the encoder is NOT
    frozen after the pre-phase: every ``interval`` victim optimizer steps
    the driver calls update() with the current batch's seam rows, running
    ONE interleaved minimax round:

      1) adversary step — the embedded token/property/reconstruction
         probes minimize their leakage losses against a FROZEN copy of the
         current encoder output (the adversarial direction);
      2) encoder step — the encoder minimizes the bounded v2 objective

             distortion - 0.001*tanh(leak) + 0.001*mi_weight*tanh(MINE)

         where distortion is the normalized reconstruction MSE through the
         PUBLIC decoder (the utility signal: the cloud consumes the
         decoded release, so low distortion preserves the victim's forward
         signal), leak is the probes' normalized leakage loss, and the
         MINE term is the same diagnostic reading as the pre-phase.

    The encoder's weights are updated ONLY here, never by the victim
    optimizer; the victim gradient keeps flowing through the encoder's
    input path exactly as in the frozen mode (mechanism v2 autograd), so
    the wire/capture convention is unchanged. Co-training requires
    mechanism v2 geometry.

    Determinism: probe/optimizer construction and every update run under
    torch.random.fork_rng + manual_seed(cfg.seed ^ _COTRAIN_SEED_XOR [^
    update_index]), so same (seed, data, interval) -> identical updates
    and the victim run's RNG stream is never disturbed.
    """

    def __init__(self, tb, cfg, device, dtype=None, interval=8):
        require_torch()
        cfg.validate()
        if not isinstance(interval, int) or isinstance(interval, bool) \
                or interval < 1:
            raise ValueError(f"BottleneckCotrain: interval must be an "
                             f"integer >= 1, got {interval!r}")
        self.tb = tb
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype or torch.float32
        self.interval = interval
        self.updates = 0
        self.journal = []
        dev = self.device
        devices = [dev.index if dev.index is not None
                   else torch.cuda.current_device()] \
            if dev.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(cfg.seed ^ _COTRAIN_SEED_XOR)
            geo = {k: v.to(device=dev, dtype=self.dtype)
                   for k, v in tb.geometry.items()
                   if k in ("mean", "basis", "scale")}
            geo["residual_bound"] = float(tb.geometry["residual_bound"])
            self.geometry = geo
            self.encoder = tb.cotrain_module(dev, self.dtype)
            nets = _build_nets(tb.seam_width, tb.vocab, cfg, dev,
                               self.dtype)
            self.token_probe = nets["token_probe"]
            self.property_probe = nets["property_probe"]
            self.recon_probe = nets["recon_probe"]
            self.mine_net = nets["mine_net"]
            probe_params = list(self.token_probe.parameters()) \
                + list(self.property_probe.parameters()) \
                + list(self.recon_probe.parameters())
            self.probe_opt = torch.optim.Adam(probe_params, lr=cfg.lr)
            self.mine_opt = torch.optim.Adam(self.mine_net.parameters(),
                                             lr=cfg.lr)
            self.enc_opt = torch.optim.Adam(self.encoder.parameters(),
                                            lr=cfg.lr)

    def _encode(self, h):
        geo = self.geometry
        return (h - geo["mean"]) @ geo["basis"] \
            + geo["residual_bound"] * geo["scale"] \
            * torch.tanh(self.encoder((h - geo["mean"]) / geo["scale"]))

    def update(self, hidden_rows, token_rows):
        """One interleaved minimax round on the current victim batch.

        hidden_rows  [N, H] seam activations (detached by the caller)
        token_rows   [N] token ids aligned to the rows
        Returns a small journal record; the co-trained weights are synced
        back into the holder (tb) at the end of the round."""
        require_torch()
        cfg = self.cfg
        dev = self.device
        devices = [dev.index if dev.index is not None
                   else torch.cuda.current_device()] \
            if dev.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(cfg.seed ^ _COTRAIN_SEED_XOR ^ self.updates)
            h_all = hidden_rows.detach().to(device=dev, dtype=self.dtype)
            t_all = token_rows.detach().to(device=dev).long()
            n = h_all.shape[0]
            bs = min(cfg.batch_rows, n)
            order = torch.randperm(n, device=dev)[:bs]
            hb, tb_tokens = h_all[order], t_all[order]
            prop = tb_tokens % cfg.property_classes
            geo = self.geometry
            with torch.no_grad():
                zb = self._encode(hb)
            self.probe_opt.zero_grad()
            probe_loss = torch.nn.functional.cross_entropy(
                self.token_probe(zb), tb_tokens) \
                + torch.nn.functional.cross_entropy(
                    self.property_probe(zb), prop) \
                + torch.nn.functional.mse_loss(self.recon_probe(zb), hb)
            probe_loss.backward()
            self.probe_opt.step()
            self.mine_opt.zero_grad()
            (-_mine_reading(self.mine_net, hb, zb)).backward()
            self.mine_opt.step()
            self.enc_opt.zero_grad()
            ze = self._encode(hb)
            scale = geo["scale"]
            leak = torch.nn.functional.cross_entropy(
                self.token_probe(ze), tb_tokens) \
                / max(1., math.log(self.tb.vocab)) \
                + torch.nn.functional.cross_entropy(
                    self.property_probe(ze), prop) \
                / max(1., math.log(cfg.property_classes)) \
                + torch.nn.functional.mse_loss(self.recon_probe(ze), hb) \
                / scale.square()
            mi = _mine_reading(self.mine_net, hb, ze)
            reconstructed = ze @ geo["basis"].T + geo["mean"]
            distortion = (reconstructed - hb).square().mean() \
                / scale.square()
            objective = distortion - 0.001 * torch.tanh(leak) \
                + 0.001 * cfg.mi_weight * torch.tanh(mi)
            if not torch.isfinite(objective):
                raise ValueError("non-finite cotrain bottleneck objective")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            self.enc_opt.step()
        self.updates += 1
        self.tb.sync_cotrain(self.encoder)
        rec = {"update_index": self.updates,
               "probe_loss": float(probe_loss.detach()),
               "distortion": float(distortion.detach()),
               "mi_reading": float(mi.detach())}
        self.journal.append(rec)
        return rec


def validate_provenance(rec):
    """Pure-python validation of a
    llm_attacker.trained_bottleneck_provenance.v1 record; returns it."""
    if not isinstance(rec, dict) or rec.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError(f"expected schema {PROVENANCE_SCHEMA}")
    for key in ("created_by", "config", "config_sha256", "seed",
                "seam_width", "bottleneck_dim", "vocab", "checkpoint",
                "mi_reading"):
        if key not in rec:
            raise ValueError(f"provenance missing {key!r}")
    BottleneckConfig(**rec["config"]).validate()
    if rec["config_sha256"] != \
            BottleneckConfig(**rec["config"]).config_sha256():
        raise ValueError("config_sha256 does not match config")
    if not isinstance(rec["seed"], int) or rec["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    ckpt = rec["checkpoint"]
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("path"), str) \
            or not ckpt["path"]:
        raise ValueError("checkpoint.path must be a nonempty string")
    sha = ckpt.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError("checkpoint.sha256 must be a 64-char hex digest")
    mi = rec["mi_reading"]
    if not isinstance(mi, dict) \
            or not isinstance(mi.get("value"), (int, float)) \
            or not math.isfinite(mi["value"]):
        raise ValueError("mi_reading.value must be a finite number")
    if mi.get("certificate") is not False \
            or "not a certificate" not in str(mi.get("caveat", "")).lower() \
            and "NOT a certificate" not in str(mi.get("caveat", "")):
        raise ValueError("mi_reading must be labeled a non-certificate "
                         "(certificate: false + caveat)")
    if "cotrain" in rec:
        ct = rec["cotrain"]
        if not isinstance(ct, dict) \
                or not isinstance(ct.get("enabled"), bool) \
                or not isinstance(ct.get("interval"), int) \
                or isinstance(ct.get("interval"), bool) \
                or ct["interval"] < 1 \
                or not isinstance(ct.get("updates_performed"), int) \
                or isinstance(ct.get("updates_performed"), bool) \
                or ct["updates_performed"] < 0:
            raise ValueError("cotrain must carry enabled (bool), interval "
                             "(integer >= 1) and updates_performed "
                             "(integer >= 0)")
        if ct["enabled"] and ct["updates_performed"] < 1:
            raise ValueError("an enabled cotrain record must count at "
                             "least one performed update")
    if "utility" in rec:
        for split in ("utility_before", "utility_after"):
            u = rec["utility"].get(split)
            if not isinstance(u, dict) \
                    or not isinstance(u.get("cross_entropy"), (int, float)) \
                    or not math.isfinite(u["cross_entropy"]) \
                    or not isinstance(u.get("perplexity"), (int, float)):
                raise ValueError(f"utility.{split} must carry finite "
                                 "cross_entropy and perplexity")
    return rec


def _bundle_dim(capture_dir, dim):
    if dim is not None:
        return int(dim)
    for name in (PROVENANCE_NAME, PCA_PROVENANCE_NAME):
        sidecar = os.path.join(capture_dir, name)
        if os.path.isfile(sidecar):
            with open(sidecar) as f:
                return int(json.load(f)["bottleneck_dim"])
    raise ValueError("bottleneck dim unknown: pass --dim or run the "
                     "capture with --defense trained-bottleneck or "
                     "pca-compress (their provenance sidecars record it)")


def _bundle_defender_label(capture_dir):
    if os.path.isfile(os.path.join(capture_dir, PCA_PROVENANCE_NAME)):
        return ("pca-compress (attacker.trained_defenses compression-only "
                "control; no adversarial training)")
    return ("trained-bottleneck (attacker.trained_defenses "
            "in-repo re-implementation; not the original "
            "defense project's)")


def capture_to_representation_bundle(capture_dir, canonical_path=None,
                                     dim=None, label_position=0,
                                     output=None):
    """Turn a trained-bottleneck or pca-compress splitbench capture into a
    representation_privacy.v1 bundle over the REAL trained/fit k-dim
    releases.

    One document per fwd wire capture: ``released`` is the first batch
    row's post-defense wire tensor truncated to the bottleneck
    coordinates [seq, k]; ``label`` is that row's canonical token id at
    ``label_position`` (a prespecified per-document classification
    target, per the representation_privacy contract — not full-sequence
    recovery). Documents alternate deterministically into calibration /
    evaluation in scan order. Validated with
    representation_privacy.validate_bundle before returning."""
    require_torch()
    from . import captures
    from .attacks import representation_privacy
    k = _bundle_dim(capture_dir, dim)
    records = [(m, p) for m, p in captures.scan_captures(capture_dir,
                                                         "training")
               if m["phase"] == "fwd"]
    if len(records) < 8:
        raise ValueError(f"need >= 8 fwd captures for the bundle's >= 4 "
                         f"calibration / >= 4 evaluation documents, got "
                         f"{len(records)} in {capture_dir}")
    canonical_path = canonical_path or os.path.join(capture_dir,
                                                    "canonical.pt")
    if not os.path.isfile(canonical_path):
        raise ValueError(f"canonical token ids not found: {canonical_path} "
                         "(re-run the capture with --save-canonical)")
    canonical = torch.load(canonical_path, weights_only=True)
    docs, row = [], 0
    for meta, pt in records:
        wire = captures.load_tensor(pt).float()
        if wire.dim() == 2:
            wire = wire.unsqueeze(0)
        b = wire.shape[0]
        released = wire[0][:, :k].tolist()
        label = int(canonical[row, label_position])
        docs.append({"document_id": f"{meta['session_id']}-"
                                    f"step{meta['step']:05d}",
                     "released": released, "label": label})
        row += b
    calibration = docs[0::2]
    evaluation = docs[1::2]
    bundle = {
        "schema": "representation_privacy.v1",
        "provenance": {
            "source": f"splitbench bottleneck-family capture "
                      f"{os.fspath(capture_dir)}",
            "defender": _bundle_defender_label(capture_dir),
            "release_stage": "post-defense seam wire (fwd), first batch "
                             "row, bottleneck coordinates"},
        "calibration": calibration,
        "evaluation": evaluation,
    }
    representation_privacy.validate_bundle(bundle)
    if output:
        with open(output, "w") as f:
            json.dump(bundle, f, indent=2, sort_keys=True)
    return bundle


@dataclasses.dataclass
class PcaConfig:
    """Knobs for the pca-compress defense (fit-only; no training)."""
    bottleneck_dim: int = 64
    seed: int = 0

    def validate(self):
        def bad(msg):
            raise ValueError(f"PcaConfig: {msg}")
        if not isinstance(self.bottleneck_dim, int) \
                or isinstance(self.bottleneck_dim, bool) \
                or self.bottleneck_dim < 1:
            bad(f"bottleneck_dim must be an integer >= 1, got "
                f"{self.bottleneck_dim!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) \
                or self.seed < 0:
            bad(f"seed must be an integer >= 0, got {self.seed!r}")
        return self

    def canonical(self):
        return dataclasses.asdict(self)

    def config_sha256(self):
        blob = json.dumps(self.canonical(), sort_keys=True,
                          separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()


def fit_pca_compressor(hidden_rows, cfg, device="cuda"):
    """Fit the pca-compress encoder on calibration seam activations.

    hidden_rows  [N, H] seam activations (calibration, pre-finetune)
    cfg          PcaConfig
    device       "cuda" | "cpu" — PCA is a plain SVD, so CPU is fine
                 (no GPU-only guard, unlike the minimax pre-phase)

    The fit is a full SVD of the centered calibration rows in float64
    (deterministic: same rows -> bit-identical basis), cast to fp32 for
    storage. Returns a PcaCompress holder with the explained-variance
    spectrum — the key diagnostic recorded in the provenance sidecar."""
    require_torch()
    cfg.validate()
    if hidden_rows.shape[-1] < cfg.bottleneck_dim:
        raise ValueError(f"PcaConfig: bottleneck_dim "
                         f"{cfg.bottleneck_dim} exceeds the seam width "
                         f"{hidden_rows.shape[-1]}")
    if hidden_rows.shape[0] < 2:
        raise ValueError("need at least 2 calibration rows")
    dev = torch.device(device)
    h = hidden_rows.detach().to(device=dev, dtype=torch.float64)
    mean = h.mean(0)
    _, s, vh = torch.linalg.svd(h - mean, full_matrices=True)
    k = cfg.bottleneck_dim
    basis = vh[:k].T.contiguous()
    total = s.square().sum().clamp_min(1e-12)
    spectrum = (s.square() / total).to(torch.float32).cpu()
    explained_at_k = float(spectrum[:k].sum())
    return PcaCompress(cfg.canonical(), int(h.shape[-1]),
                       mean.to(torch.float32).cpu(),
                       basis.to(torch.float32).cpu(),
                       spectrum.tolist(), explained_at_k)


class PcaCompress:
    """The fitted PCA encoder: wire application + checkpoint/provenance.

    apply() emits the zeros-plus-projection wire (k components in the
    first k coordinates, zeros elsewhere) with a straight-through
    gradient, the same release convention as mid-lite and the historical
    trained-bottleneck v1; decode() is the PUBLIC cloud-side
    reconstruction z @ basis.T + mean (not a secret key)."""

    def __init__(self, config, seam_width, mean, basis,
                 explained_variance_ratio, explained_variance_at_k,
                 utility=None):
        self.config = dict(config)
        PcaConfig(**self.config).validate()
        self.seam_width = int(seam_width)
        self.mean = mean
        self.basis = basis
        self.explained_variance_ratio = [float(x) for x in
                                         explained_variance_ratio]
        self.explained_variance_at_k = float(explained_variance_at_k)
        self.utility = utility

    @property
    def bottleneck_dim(self):
        return int(self.config["bottleneck_dim"])

    def encode(self, hidden):
        """Released k-vector (linear; input-differentiable)."""
        return (hidden - self.mean.to(hidden)) @ self.basis.to(hidden)

    def decode(self, wire):
        """Cloud-side reconstruction; the decoder is public."""
        basis = self.basis.to(wire)
        return wire[..., :self.bottleneck_dim] @ basis.T \
            + self.mean.to(wire)

    def apply(self, hidden):
        """Padded compressed wire, straight-through gradient."""
        if hidden.shape[-1] != self.seam_width:
            raise ValueError(f"pca-compress: seam width "
                             f"{hidden.shape[-1]} != the encoder's "
                             f"{self.seam_width}")
        k = self.bottleneck_dim
        z = self.encode(hidden)
        wire = torch.zeros_like(hidden)
        wire[..., :k] = z
        if hidden.requires_grad:
            return hidden + (wire - hidden).detach()
        return wire


    def checkpoint_payload(self):
        return {"format": "pca_compress_encoder.v1",
                "config": self.config, "seam_width": self.seam_width,
                "mean": self.mean, "basis": self.basis,
                "explained_variance_ratio": self.explained_variance_ratio,
                "explained_variance_at_k": self.explained_variance_at_k}

    def provenance_record(self, checkpoint_path, checkpoint_sha256):
        rec = {
            "schema": PCA_PROVENANCE_SCHEMA,
            "created_by": "attacker.trained_defenses (in-repo "
                          "compression-only control for the "
                          "trained-bottleneck family; no adversarial "
                          "training)",
            "training": "none (no adversarial training): plain PCA fit "
                        "on the public calibration partition — the "
                        "compression-only control; it isolates what the "
                        "bottleneck's adversarial training adds beyond "
                        "compression",
            "config": self.config,
            "config_sha256": PcaConfig(**self.config).config_sha256(),
            "seed": self.config["seed"],
            "seam_width": self.seam_width,
            "bottleneck_dim": self.bottleneck_dim,
            "explained_variance_at_k": self.explained_variance_at_k,
            "explained_variance_ratio": self.explained_variance_ratio,
            "checkpoint": {"path": os.fspath(checkpoint_path),
                           "sha256": checkpoint_sha256},
        }
        if self.utility is not None:
            rec["utility"] = self.utility
        return rec

    def save(self, out_dir):
        """Write the encoder checkpoint + provenance sidecar into the
        run's output dir; returns the provenance record."""
        require_torch()
        os.makedirs(out_dir, exist_ok=True)
        ckpt = os.path.join(out_dir, PCA_ENCODER_NAME)
        torch.save(self.checkpoint_payload(), ckpt)
        rec = self.provenance_record(ckpt, _file_sha256(ckpt))
        sidecar = os.path.join(out_dir, PCA_PROVENANCE_NAME)
        with open(sidecar, "w") as f:
            json.dump(rec, f, indent=2, sort_keys=True)
        rec["provenance_path"] = sidecar
        return rec

    @classmethod
    def load(cls, path):
        """Load a checkpoint written by save() (file or its directory)."""
        require_torch()
        if os.path.isdir(path):
            path = os.path.join(path, PCA_ENCODER_NAME)
        payload = torch.load(path, weights_only=False)
        if payload.get("format") != "pca_compress_encoder.v1":
            raise ValueError(f"{path}: not a pca_compress_encoder.v1 "
                             "checkpoint")
        return cls(payload["config"], payload["seam_width"],
                   payload["mean"], payload["basis"],
                   payload["explained_variance_ratio"],
                   payload["explained_variance_at_k"])


def validate_pca_provenance(rec):
    """Pure-python validation of a
    llm_attacker.pca_compress_provenance.v1 record; returns it."""
    if not isinstance(rec, dict) or rec.get("schema") != \
            PCA_PROVENANCE_SCHEMA:
        raise ValueError(f"expected schema {PCA_PROVENANCE_SCHEMA}")
    for key in ("created_by", "training", "config", "config_sha256",
                "seed", "seam_width", "bottleneck_dim",
                "explained_variance_at_k", "explained_variance_ratio",
                "checkpoint"):
        if key not in rec:
            raise ValueError(f"provenance missing {key!r}")
    PcaConfig(**rec["config"]).validate()
    if rec["config_sha256"] != PcaConfig(**rec["config"]).config_sha256():
        raise ValueError("config_sha256 does not match config")
    if "no adversarial training" not in str(rec["training"]) \
            and "none" not in str(rec["training"]).lower():
        raise ValueError("training must record the compression-only "
                         "(no adversarial training) control status")
    if not isinstance(rec["seed"], int) or rec["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    k = rec["bottleneck_dim"]
    if not isinstance(rec["seam_width"], int) \
            or not isinstance(k, int) or not 1 <= k <= rec["seam_width"]:
        raise ValueError("need 1 <= bottleneck_dim <= seam_width")
    ev = rec["explained_variance_at_k"]
    if not isinstance(ev, (int, float)) or not math.isfinite(ev) \
            or not -1e-6 <= ev <= 1.0 + 1e-6:
        raise ValueError("explained_variance_at_k must be a finite "
                         "fraction in [0, 1]")
    spectrum = rec["explained_variance_ratio"]
    if not isinstance(spectrum, list) \
            or len(spectrum) != rec["seam_width"] \
            or any(not isinstance(x, (int, float))
                   or not math.isfinite(x) or x < -1e-9 for x in spectrum):
        raise ValueError("explained_variance_ratio must be one "
                         "nonnegative finite fraction per seam coordinate")
    if abs(sum(spectrum[:k]) - ev) > 1e-4:
        raise ValueError("explained_variance_at_k must equal the sum of "
                         "the first k spectrum entries")
    if sum(spectrum) > 1.0 + 1e-4:
        raise ValueError("the explained-variance spectrum cannot sum "
                         "above 1")
    ckpt = rec["checkpoint"]
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("path"), str) \
            or not ckpt["path"]:
        raise ValueError("checkpoint.path must be a nonempty string")
    sha = ckpt.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError("checkpoint.sha256 must be a 64-char hex digest")
    if "utility" in rec:
        for split in ("utility_before", "utility_after"):
            u = rec["utility"].get(split)
            if not isinstance(u, dict) \
                    or not isinstance(u.get("cross_entropy"), (int, float)) \
                    or not math.isfinite(u["cross_entropy"]) \
                    or not isinstance(u.get("perplexity"), (int, float)):
                raise ValueError(f"utility.{split} must carry finite "
                                 "cross_entropy and perplexity")
    return rec


@dataclasses.dataclass
class MixupConfig:
    """Knobs for the secret-mixup release transform."""
    lam: float = 0.75
    pool_size: int = 256
    seed: int = 0
    correction: bool = True

    def validate(self):
        def bad(msg):
            raise ValueError(f"MixupConfig: {msg}")
        if not isinstance(self.lam, (int, float)) \
                or isinstance(self.lam, bool) \
                or not 0 < float(self.lam) <= 1:
            bad(f"lam must be a number in (0, 1], got {self.lam!r}")
        if not isinstance(self.pool_size, int) \
                or isinstance(self.pool_size, bool) or self.pool_size < 1:
            bad(f"pool_size must be an integer >= 1, got {self.pool_size!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) \
                or self.seed < 0:
            bad(f"seed must be an integer >= 0, got {self.seed!r}")
        if not isinstance(self.correction, bool):
            bad(f"correction must be a boolean, got {self.correction!r}")
        return self

    def canonical(self):
        return {"lambda": float(self.lam), "pool_size": self.pool_size,
                "seed": self.seed, "correction": self.correction}

    def config_sha256(self):
        blob = json.dumps(self.canonical(), sort_keys=True,
                          separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()


def corrected_gradient(g_wire, g_decoy, lam):
    """The victim's return-path correction for the mixup release.

        g_corr = (g_wire - (1 - lambda) * g_decoy) / lambda

    g_wire is the server's returned gradient at the MIXED wire point;
    g_decoy is the gradient of the same loss adjoint through the parallel
    decoy-only forward. Under a LINEAR cloud segment F(x) = A x this is
    EXACT: g_wire = g_decoy = A^T v, so g_corr = A^T v — precisely the
    gradient the victim would have received over an undefended seam. Under
    a NONLINEAR cloud segment the Jacobian at the mixed point differs from
    the Jacobian at the decoy point and the subtraction is only
    APPROXIMATE — this mirrors the defense framework's own caveat; the
    residual is the correction error, not a falsification of the defense.
    """
    return (g_wire - (1.0 - lam) * g_decoy) / lam


class SecretMixup:
    """Release-time mixup transform with a recycled-decoy pool (stateful).

    Decoy pool: REAL latent rows recycled from earlier training blocks of
    the SAME run (pre-mix seam rows, detached to CPU) with their honest
    token labels tracked alongside. Per block, decoy rows are drawn from
    the pool WITHOUT replacement (one permutation cycle; if the block has
    more rows than the pool the permutation is tiled — each pool row is
    drawn at most once per cycle) under a torch.Generator seeded from
    sha256("secret-mixup:<seed>:<block>") — CSPRNG-seeded, deterministic
    per (config seed, block index), independent of the caller's RNG stream.

    Release:  wire_i = lambda * h_i + (1 - lambda) * z_decoy(i).

    Empty-pool policy: the pool starts EMPTY. Blocks released while the
    pool is empty go out UNDEFENDED (the release is the raw seam tensor)
    and are flagged per block with policy="undefended_empty_pool" in the
    provenance sidecar; the block's rows still recycle into the pool, so
    the first block of a run is always undefended and later blocks are
    mixed. Gradient: straight-through (out = h + (wire - h).detach()),
    the same wire-quantization convention as trained-bottleneck; the
    decoy-forward return-path correction happens in the driver
    (corrected_gradient above). The current block's rows become decoys
    only for LATER blocks (never their own).
    """

    def __init__(self, config, utility=None):
        if isinstance(config, MixupConfig):
            config = config.canonical()
        self.config = dict(config)
        MixupConfig(lam=self.config["lambda"],
                    pool_size=self.config["pool_size"],
                    seed=self.config["seed"],
                    correction=self.config["correction"]).validate()
        self.utility = utility
        self.pool = []
        self.pool_labels = []
        self.blocks = []
        self.last_decoy = None
        self._decoy_store = {}

    @property
    def lam(self):
        return float(self.config["lambda"])

    def _draw_seed(self, block):
        blob = f"secret-mixup:{self.config['seed']}:{block}".encode()
        return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")

    def apply(self, hidden, tokens=None):
        """The POST-defense wire tensor for one block (straight-through
        gradient). `tokens` (optional, same leading shape) supplies the
        honest labels recycled into the pool with this block's rows."""
        require_torch()
        cfg = MixupConfig(lam=self.config["lambda"],
                          pool_size=self.config["pool_size"],
                          seed=self.config["seed"],
                          correction=self.config["correction"])
        block = len(self.blocks)
        lam = cfg.lam
        flat = hidden.reshape(-1, hidden.shape[-1])
        n = int(flat.shape[0])
        pool_before = len(self.pool)
        self.last_decoy = None
        if pool_before == 0:
            policy = "undefended_empty_pool"
            mixed = flat
            draw_seed, indices, decoy_labels = None, [], []
        else:
            policy = "mixed"
            draw_seed = self._draw_seed(block)
            g = torch.Generator().manual_seed(draw_seed)
            perm = torch.randperm(pool_before, generator=g)
            reps = -(-n // pool_before)
            idx = perm.repeat(reps)[:n]
            z = torch.stack([self.pool[int(i)] for i in idx]) \
                .to(device=hidden.device, dtype=hidden.dtype)
            self.last_decoy = z.reshape(hidden.shape)
            self._decoy_store[block] = z.detach().to(
                device="cpu", dtype=torch.float32).clone()
            mixed = lam * flat + (1.0 - lam) * z
            indices = [int(i) for i in idx]
            decoy_labels = [self.pool_labels[int(i)] for i in idx]
        labels = [None] * n
        if tokens is not None:
            labels = [int(t) for t in tokens.reshape(-1).tolist()]
        for i in range(n):
            self.pool.append(flat[i].detach().to(
                device="cpu", dtype=torch.float32).clone())
            self.pool_labels.append(labels[i])
        overflow = len(self.pool) - cfg.pool_size
        if overflow > 0:
            del self.pool[:overflow]
            del self.pool_labels[:overflow]
        self.blocks.append({
            "block_index": block, "policy": policy, "n_rows": n,
            "pool_size_before": pool_before,
            "pool_size_after": len(self.pool), "draw_seed": draw_seed,
            "decoy_pool_indices": indices,
            "honest_labels": labels,
            "decoy_honest_labels": decoy_labels})
        out = mixed.reshape(hidden.shape)
        if hidden.requires_grad:
            return hidden + (out - hidden).detach()
        return out


    def provenance_record(self, decoys_path=None, decoys_sha256=None):
        rec = {
            "schema": MIXUP_PROVENANCE_SCHEMA,
            "created_by": "attacker.trained_defenses (in-repo "
                          "re-implementation of the secret-mixup + decoy "
                          "correction defense family; NOT the original "
                          "defense project's code)",
            "config": self.config,
            "config_sha256": MixupConfig(
                lam=self.config["lambda"],
                pool_size=self.config["pool_size"],
                seed=self.config["seed"],
                correction=self.config["correction"]).config_sha256(),
            "lambda": self.lam,
            "seed": self.config["seed"],
            "correction_mode": "decoy_forward_subtract"
                               if self.config["correction"] else "off",
            "correction_caveat": "the decoy-forward subtraction is exact "
                                 "for a linear cloud segment and only "
                                 "APPROXIMATE under a nonlinear one (the "
                                 "defense framework's own caveat); the "
                                 "residual is correction error, not a "
                                 "certificate",
            "empty_pool_policy": "release_undefended",
            "pool_final_size": len(self.pool),
            "blocks": self.blocks,
        }
        if decoys_path is not None:
            rec["decoy_store"] = {"path": os.fspath(decoys_path),
                                  "sha256": decoys_sha256}
        if self.utility is not None:
            rec["utility"] = self.utility
        return rec

    def save(self, out_dir):
        """Write the decoy store (drawn decoy rows per block, for exact
        converter de-mixing) + the provenance sidecar; returns the record."""
        require_torch()
        os.makedirs(out_dir, exist_ok=True)
        decoys = os.path.join(out_dir, MIXUP_DECOYS_NAME)
        torch.save({"format": "secret_mixup_decoys.v1",
                    "blocks": self._decoy_store}, decoys)
        rec = self.provenance_record(decoys, _file_sha256(decoys))
        sidecar = os.path.join(out_dir, MIXUP_PROVENANCE_NAME)
        with open(sidecar, "w") as f:
            json.dump(rec, f, indent=2, sort_keys=True)
        rec["provenance_path"] = sidecar
        return rec


def validate_mixup_provenance(rec):
    """Pure-python validation of a
    llm_attacker.secret_mixup_provenance.v1 record; returns it."""
    if not isinstance(rec, dict) \
            or rec.get("schema") != MIXUP_PROVENANCE_SCHEMA:
        raise ValueError(f"expected schema {MIXUP_PROVENANCE_SCHEMA}")
    for key in ("created_by", "config", "config_sha256", "lambda", "seed",
                "correction_mode", "correction_caveat", "empty_pool_policy",
                "pool_final_size", "blocks"):
        if key not in rec:
            raise ValueError(f"provenance missing {key!r}")
    cfg = rec["config"]
    MixupConfig(lam=cfg["lambda"], pool_size=cfg["pool_size"],
                seed=cfg["seed"], correction=cfg["correction"]).validate()
    if rec["config_sha256"] != MixupConfig(
            lam=cfg["lambda"], pool_size=cfg["pool_size"], seed=cfg["seed"],
            correction=cfg["correction"]).config_sha256():
        raise ValueError("config_sha256 does not match config")
    if rec["lambda"] != cfg["lambda"]:
        raise ValueError("top-level lambda must match config.lambda")
    if not isinstance(rec["seed"], int) or rec["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if rec["correction_mode"] not in ("decoy_forward_subtract", "off"):
        raise ValueError("unknown correction_mode")
    if "approximate" not in str(rec["correction_caveat"]).lower():
        raise ValueError("correction_caveat must name the nonlinearity "
                         "approximation")
    if rec["empty_pool_policy"] != "release_undefended":
        raise ValueError("empty_pool_policy must be 'release_undefended'")
    if not isinstance(rec["pool_final_size"], int) \
            or rec["pool_final_size"] < 0:
        raise ValueError("pool_final_size must be a nonnegative integer")
    if not isinstance(rec["blocks"], list):
        raise ValueError("blocks must be a list")
    for i, b in enumerate(rec["blocks"]):
        if not isinstance(b, dict) or b.get("block_index") != i:
            raise ValueError("blocks must be indexed 0..n-1 in order")
        if b.get("policy") not in ("mixed", "undefended_empty_pool"):
            raise ValueError(f"block {i}: unknown policy")
        for k in ("n_rows", "pool_size_before", "pool_size_after"):
            if not isinstance(b.get(k), int) or b[k] < 0:
                raise ValueError(f"block {i}: {k} must be a nonnegative "
                                 "integer")
        if b["policy"] == "mixed":
            if not isinstance(b.get("draw_seed"), int) or b["draw_seed"] < 0:
                raise ValueError(f"block {i}: mixed blocks need a draw_seed")
            idx = b.get("decoy_pool_indices")
            if not isinstance(idx, list) or len(idx) != b["n_rows"] \
                    or any(not isinstance(j, int)
                           or not 0 <= j < b["pool_size_before"]
                           for j in idx):
                raise ValueError(f"block {i}: decoy_pool_indices must be "
                                 "valid pool row indices, one per row")
            first = idx[:min(len(idx), b["pool_size_before"])]
            if len(set(first)) != len(first):
                raise ValueError(f"block {i}: decoy draws repeat within a "
                                 "permutation cycle (sampling must be "
                                 "without replacement)")
        else:
            if b.get("draw_seed") is not None \
                    or b.get("decoy_pool_indices") != []:
                raise ValueError(f"block {i}: undefended blocks carry no "
                                 "draw seed / decoy indices")
        if not isinstance(b.get("honest_labels"), list) \
                or len(b["honest_labels"]) != b["n_rows"]:
            raise ValueError(f"block {i}: honest_labels must track one "
                             "label per released row")
        if not isinstance(b.get("decoy_honest_labels"), list):
            raise ValueError(f"block {i}: decoy_honest_labels must be a "
                             "list")
    if "decoy_store" in rec:
        ds = rec["decoy_store"]
        sha = ds.get("sha256") if isinstance(ds, dict) else None
        if not isinstance(ds.get("path"), str) or not ds["path"] \
                or not isinstance(sha, str) or len(sha) != 64:
            raise ValueError("decoy_store needs a path and a 64-char "
                             "sha256")
    if "utility" in rec:
        for split in ("utility_before", "utility_after"):
            u = rec["utility"].get(split)
            if not isinstance(u, dict) \
                    or not isinstance(u.get("cross_entropy"), (int, float)) \
                    or not math.isfinite(u["cross_entropy"]) \
                    or not isinstance(u.get("perplexity"), (int, float)):
                raise ValueError(f"utility.{split} must carry finite "
                                 "cross_entropy and perplexity")
    return rec


def capture_to_view_privacy_bundle(capture_dir, output=None):
    """Turn a secret-mixup splitbench finetune capture into a
    view_privacy.v1 mixup bundle over the REAL recycled-decoy releases.

    Documents: the finetune driver cycles its batch list, so the same
    batch slot recurs every epoch — each slot is one target with R
    repeated releases (one per epoch), exactly the repeated-release
    structure the mixup de-mixing arm assumes. ``released`` rows are the
    captured post-defense wire tensors (first batch row, flattened);
    ``truth`` is the EXACT per-release de-mix against the provenance
    decoy store (the mix is linear, so (wire - (1-lambda)*z)/lambda
    recovers the pre-mix seam row up to fp16 capture rounding), averaged
    per slot. ``calibration_mean`` is the attacker-known mean of the
    per-block decoy draws; ``oracle_pool_mean`` is the exact pool mean
    (a labeled oracle arm); ``corrected_outputs``/``clean_outputs`` pair
    the defender's own correction with the recovered clean rows. Validated
    with view_privacy.evaluate_bundle before returning."""
    require_torch()
    from . import captures
    from .attacks import view_privacy
    sidecar = os.path.join(capture_dir, MIXUP_PROVENANCE_NAME)
    if not os.path.isfile(sidecar):
        raise ValueError(f"no {MIXUP_PROVENANCE_NAME} in {capture_dir}: "
                         "re-run the capture with --defense secret-mixup")
    with open(sidecar) as f:
        prov = validate_mixup_provenance(json.load(f))
    lam = float(prov["lambda"])
    decoys_path = os.path.join(capture_dir, MIXUP_DECOYS_NAME)
    payload = torch.load(decoys_path, weights_only=False)
    if payload.get("format") != "secret_mixup_decoys.v1":
        raise ValueError(f"{decoys_path}: not a secret_mixup_decoys.v1 "
                         "store")
    store = {int(k): v for k, v in payload["blocks"].items()}
    records = [(m, p) for m, p in captures.scan_captures(capture_dir,
                                                         "training")
               if m["phase"] == "fwd"]
    steps = [m["step"] for m, _ in records]
    if steps != list(range(len(steps))):
        raise ValueError("the mixup converter needs a contiguous "
                         "capture-every-1 finetune capture (step k <-> "
                         "release block k)")
    epoch0 = [m["step"] for m, _ in records if m["epoch"] == 0]
    if not epoch0:
        raise ValueError("no epoch-0 captures found")
    steps_per_epoch = max(epoch0) + 1
    slots = {}
    calib_samples = []
    all_decoy_rows = []
    for meta, pt in records:
        block = meta["step"]
        if block not in store:
            continue
        wire = captures.load_tensor(pt).double()
        if wire.dim() == 2:
            wire = wire.unsqueeze(0)
        seq = wire.shape[1]
        z = store[block].double()[:seq]
        released = wire[0].reshape(-1)
        truth = (released - (1.0 - lam) * z.reshape(-1)) / lam
        slot = meta["step"] % steps_per_epoch
        ent = slots.setdefault(slot, {"releases": [], "truth": []})
        ent["releases"].append(released.tolist())
        ent["truth"].append(truth.tolist())
        calib_samples.append(z.reshape(-1).tolist())
        all_decoy_rows.append(store[block].double())
    if len(slots) < 2 or not calib_samples:
        raise ValueError(f"need >= 2 mixed target slots for the mixup "
                         f"bundle, got {len(slots)} in {capture_dir}")
    max_r = max(len(e["releases"]) for e in slots.values())
    slots = {s: e for s, e in slots.items()
             if len(e["releases"]) == max_r}
    if len(slots) < 2:
        raise ValueError(f"need >= 2 fully-repeated target slots, got "
                         f"{len(slots)} in {capture_dir}")

    def vec_mean(rows):
        return [sum(col) / len(col) for col in zip(*rows)]

    targets = sorted(slots)
    releases = [slots[s]["releases"] for s in targets]
    truth = [vec_mean(slots[s]["truth"]) for s in targets]
    corrected = [[row for row in slots[s]["truth"]] for s in targets]
    calibration_mean = vec_mean(calib_samples)
    pool_mean_row = torch.cat(all_decoy_rows).mean(dim=0)
    oracle = pool_mean_row.repeat(seq).tolist()
    lo = max(1e-6, round(lam - 0.1, 4))
    hi = min(1.0, round(lam + 0.1, 4))
    bundle = {
        "schema": "view_privacy.v1",
        "case": "mixup",
        "data": {
            "releases": releases,
            "truth": truth,
            "lambda": lam,
            "assumed_lambdas": sorted({lo, lam, hi}),
            "calibration_mean": calibration_mean,
            "calibration_samples": calib_samples,
            "oracle_pool_mean": oracle,
            "corrected_outputs": corrected,
            "clean_outputs": truth,
        },
    }
    if not view_privacy.evaluate_bundle(bundle):
        raise ValueError("converted bundle failed the view_privacy "
                         "mixup evaluation gate")
    if output:
        with open(output, "w") as f:
            json.dump(bundle, f)
    return bundle


@dataclasses.dataclass
class ReplicaMedianConfig:
    """Knobs for the replica-median integrity defense."""
    replica_count: int = 3
    tolerance: float = 0.02
    gauge: bool = True
    gauge_scale: float = 1e-3
    seed: int = 0

    def validate(self):
        def bad(msg):
            raise ValueError(f"ReplicaMedianConfig: {msg}")
        if not isinstance(self.replica_count, int) \
                or isinstance(self.replica_count, bool) \
                or self.replica_count < 2:
            bad(f"replica_count must be an integer >= 2, got "
                f"{self.replica_count!r}")
        for name in ("tolerance", "gauge_scale"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) \
                    or not math.isfinite(v) or v <= 0:
                bad(f"{name} must be a finite number > 0, got {v!r}")
        if not isinstance(self.gauge, bool):
            bad(f"gauge must be a boolean, got {self.gauge!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) \
                or self.seed < 0:
            bad(f"seed must be an integer >= 0, got {self.seed!r}")
        return self

    def canonical(self):
        return {"replica_count": self.replica_count,
                "tolerance": float(self.tolerance), "gauge": self.gauge,
                "gauge_scale": float(self.gauge_scale), "seed": self.seed}

    def config_sha256(self):
        blob = json.dumps(self.canonical(), sort_keys=True,
                          separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def channel_seed(self, index):
        """Per-channel seed: master seed XOR the channel index (the
        documented cfg.seed ^ offset ^ channel_offset convention)."""
        return self.seed ^ index


def coordinate_median(ys):
    """Coordinate median of a list of equal-shape tensors (K >= 3). For
    K=2 the aggregation rule is the coordinate MEAN (the K=2 policy: no
    majority exists, so disagreement can only abort, never outvote)."""
    require_torch()
    if len(ys) < 2:
        raise ValueError("need at least two channel outputs")
    stacked = torch.stack(list(ys))
    if stacked.shape[0] == 2:
        return stacked.mean(dim=0)
    return stacked.median(dim=0).values


def _rel_deviations(ys):
    """Per-channel, per-row RELATIVE deviation from the group mean:
    ||y_k[r] - mean[r]|| / (||mean[r]|| + eps) over flattened rows.
    Returns (devs [K, n_rows], group_mean)."""
    flats = [y.detach().reshape(-1, y.shape[-1]).double() for y in ys]
    mean = torch.stack(flats).mean(dim=0)
    denom = mean.norm(dim=-1) + 1e-8
    devs = torch.stack([(f - mean).norm(dim=-1) / denom for f in flats])
    return devs, mean


class ReplicaMedian:
    """K in-process body replicas + coordinate-median aggregation (stateful).

    The driver (splitbench) builds the K independent channel forwards
    (deep-copied body slices with per-channel seeded gauges, trained in
    parallel by the shared median loss) and hands them in as callables
    h -> body output; this class owns the aggregation rule, the per-block
    integrity journal, and provenance. Gradient: the median is a plain
    autograd expression over the channel outputs, so the median element
    per coordinate receives the top's gradient — the replicas train in
    parallel in-process, and the victim head receives the aggregated
    gradient through the seam leaf.

    K=2 policy ("coordinate_mean_k2"): no honest majority exists, so the
    channels are averaged and a per-row relative disagreement above the
    tolerance raises the journal's disagreement_abort flag — an
    AVAILABILITY abort (recorded, not raised), mirroring the
    replica-integrity battery's K=2 abort semantics.
    """

    def __init__(self, config, channel_forwards=None, gauge_ids=None,
                 utility=None):
        if isinstance(config, ReplicaMedianConfig):
            config = config.canonical()
        self.config = dict(config)
        cfg = ReplicaMedianConfig(**self.config).validate()
        self.channel_forwards = list(channel_forwards or [])
        if self.channel_forwards \
                and len(self.channel_forwards) != cfg.replica_count:
            raise ValueError(f"replica-median: {len(self.channel_forwards)} "
                             f"channel forwards for K={cfg.replica_count}")
        self.channel_seeds = [cfg.channel_seed(k)
                              for k in range(cfg.replica_count)]
        self.gauge_ids = list(gauge_ids or
                              [f"channel-{k}" for k in
                               range(cfg.replica_count)])
        self.utility = utility
        self.journal = []
        self.heterogeneity = None

    @property
    def k(self):
        return int(self.config["replica_count"])

    @property
    def aggregation_rule(self):
        return "coordinate_median" if self.k >= 3 else "coordinate_mean_k2"

    def forward_block(self, hidden, step=None, captured=False):
        """One defended block: K channel forwards -> aggregated wire +
        journal entry. Returns (aggregate, channel_outputs, entry); the
        aggregate stays in the autograd graph, the journal entry is
        computed from detached values."""
        require_torch()
        if len(self.channel_forwards) != self.k:
            raise ValueError(
                "replica-median: the per-run defense state carries no "
                "channel forwards: build it with make_defense_state(cfg, "
                "split, ...) (the finetune/infer drivers do this)")
        ys = [f(hidden) for f in self.channel_forwards]
        agg = coordinate_median(ys)
        devs, _mean = _rel_deviations(ys)
        tol = float(self.config["tolerance"])
        entry = {"block_index": len(self.journal), "step": step,
                 "captured": bool(captured),
                 "n_rows": int(devs.shape[1]),
                 "aggregation_rule": self.aggregation_rule,
                 "channels": []}
        for k in range(self.k):
            d = devs[k]
            n_flag = int((d > tol).sum().item())
            entry["channels"].append({
                "channel": k, "gauge_id": self.gauge_ids[k],
                "max_rel_dev": float(d.max().item()),
                "mean_rel_dev": float(d.mean().item()),
                "n_flagged": n_flag, "flagged": n_flag > 0})
        if self.k == 2:
            d = devs.max(dim=0).values
            entry["k2_disagreement"] = {
                "max_rel": float(d.max().item()),
                "abort": bool((d > tol).any().item()),
                "policy": "availability_abort_flagged"}
        self.journal.append(entry)
        return agg, ys, entry

    def measure_heterogeneity(self, hidden_batches):
        """The honest-heterogeneity noise budget: the independent
        channels' OWN disagreement distribution over calibration seam
        tensors (no training, no grad). Recorded so attack-side tolerance
        matching is fair (the tolerance must sit above honest noise)."""
        require_torch()
        dev_rows = []
        with torch.no_grad():
            for h in hidden_batches:
                ys = [f(h) for f in self.channel_forwards]
                devs, _ = _rel_deviations(ys)
                dev_rows.append(devs)
        if not dev_rows:
            raise ValueError("measure_heterogeneity needs >= 1 batch")
        d = torch.cat(dev_rows, dim=1)
        flat = d.reshape(-1)
        srt = flat.sort().values
        p95 = float(srt[min(len(srt) - 1,
                            int(math.ceil(0.95 * len(srt))) - 1)].item())
        self.heterogeneity = {
            "source": "calibration_pre_phase",
            "n_rows": int(d.shape[1]),
            "per_channel_mean_rel_dev":
                [float(x) for x in d.mean(dim=1).tolist()],
            "mean_rel_dev": float(flat.mean().item()),
            "p95_rel_dev": p95,
            "max_rel_dev": float(flat.max().item()),
            "tolerance": float(self.config["tolerance"]),
            "note": "the independent channels' honest disagreement "
                    "distribution; the attack-side tolerance must sit "
                    "above this budget for a fair comparison"}
        return self.heterogeneity

    def _heterogeneity_from_journal(self):
        """Fallback budget when no calibration pre-phase ran (e.g.
        experiment.py's utility()): measured from the run's own blocks."""
        vals = [c["max_rel_dev"] for e in self.journal
                for c in e["channels"]]
        if not vals:
            return {"source": "run_blocks", "n_rows": 0,
                    "per_channel_mean_rel_dev": [], "mean_rel_dev": 0.0,
                    "p95_rel_dev": 0.0, "max_rel_dev": 0.0,
                    "tolerance": float(self.config["tolerance"]),
                    "note": "no blocks recorded"}
        srt = sorted(vals)
        per_k = [sum(e["channels"][k]["mean_rel_dev"]
                     for e in self.journal) / len(self.journal)
                 for k in range(self.k)]
        return {"source": "run_blocks",
                "n_rows": sum(e["n_rows"] for e in self.journal),
                "per_channel_mean_rel_dev": per_k,
                "mean_rel_dev": sum(vals) / len(vals),
                "p95_rel_dev": srt[min(len(srt) - 1,
                                       int(math.ceil(0.95 * len(srt))) - 1)],
                "max_rel_dev": srt[-1],
                "tolerance": float(self.config["tolerance"]),
                "note": "no calibration pre-phase; budget measured from "
                        "the run's own blocks"}


    def provenance_record(self, journal_path=None, journal_sha256=None):
        rec = {
            "schema": REPLICA_PROVENANCE_SCHEMA,
            "created_by": "attacker.trained_defenses (in-repo "
                          "re-implementation of the K-redundant-channel + "
                          "coordinate-median integrity defense family; NOT "
                          "the original defense project's code)",
            "config": self.config,
            "config_sha256":
                ReplicaMedianConfig(**self.config).config_sha256(),
            "replica_count": self.k,
            "channel_seeds": self.channel_seeds,
            "gauge_ids": self.gauge_ids,
            "aggregation_rule": self.aggregation_rule,
            "tolerance": float(self.config["tolerance"]),
            "k2_policy": ("availability_abort_flagged" if self.k == 2
                          else None),
            "honest_heterogeneity": self.heterogeneity
                                    or self._heterogeneity_from_journal(),
            "deployment": "in-process channels (one host, one process); "
                          "cross-host deployment remains open "
                          "(attacker/remote.py + nodes.json)",
        }
        if journal_path is not None:
            rec["journal"] = {"path": os.fspath(journal_path),
                              "sha256": journal_sha256}
        if self.utility is not None:
            rec["utility"] = self.utility
        return rec

    def save(self, out_dir):
        """Write the per-block integrity journal (JSONL) + the provenance
        sidecar into the run's output dir; returns the record."""
        os.makedirs(out_dir, exist_ok=True)
        jpath = os.path.join(out_dir, REPLICA_JOURNAL_NAME)
        with open(jpath, "w") as f:
            for entry in self.journal:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
        rec = self.provenance_record(jpath, _file_sha256(jpath))
        sidecar = os.path.join(out_dir, REPLICA_PROVENANCE_NAME)
        with open(sidecar, "w") as f:
            json.dump(rec, f, indent=2, sort_keys=True)
        rec["provenance_path"] = sidecar
        return rec


def validate_replica_provenance(rec):
    """Pure-python validation of a
    llm_attacker.replica_median_provenance.v1 record; returns it."""
    if not isinstance(rec, dict) \
            or rec.get("schema") != REPLICA_PROVENANCE_SCHEMA:
        raise ValueError(f"expected schema {REPLICA_PROVENANCE_SCHEMA}")
    for key in ("created_by", "config", "config_sha256", "replica_count",
                "channel_seeds", "gauge_ids", "aggregation_rule",
                "tolerance", "k2_policy", "honest_heterogeneity",
                "deployment"):
        if key not in rec:
            raise ValueError(f"provenance missing {key!r}")
    cfg = ReplicaMedianConfig(**rec["config"]).validate()
    if rec["config_sha256"] != cfg.config_sha256():
        raise ValueError("config_sha256 does not match config")
    if rec["replica_count"] != cfg.replica_count:
        raise ValueError("replica_count must match config.replica_count")
    seeds = rec["channel_seeds"]
    if not isinstance(seeds, list) or len(seeds) != cfg.replica_count \
            or any(not isinstance(s, int) or s < 0 for s in seeds) \
            or seeds != [cfg.channel_seed(k)
                         for k in range(cfg.replica_count)]:
        raise ValueError("channel_seeds must be the per-channel "
                         "seed ^ index projections")
    ids = rec["gauge_ids"]
    if not isinstance(ids, list) or len(ids) != cfg.replica_count \
            or any(not isinstance(g, str) or not g for g in ids) \
            or len(set(ids)) != len(ids):
        raise ValueError("gauge_ids must be distinct nonempty strings, "
                         "one per channel")
    rule = rec["aggregation_rule"]
    expect = "coordinate_median" if cfg.replica_count >= 3 \
        else "coordinate_mean_k2"
    if rule != expect:
        raise ValueError(f"aggregation_rule must be {expect!r} for "
                         f"K={cfg.replica_count}")
    if not isinstance(rec["tolerance"], (int, float)) \
            or not math.isfinite(rec["tolerance"]) or rec["tolerance"] <= 0:
        raise ValueError("tolerance must be a finite number > 0")
    if rec["tolerance"] != float(cfg.tolerance):
        raise ValueError("top-level tolerance must match config.tolerance")
    if rec["k2_policy"] != ("availability_abort_flagged"
                            if cfg.replica_count == 2 else None):
        raise ValueError("k2_policy must be 'availability_abort_flagged' "
                         "for K=2 and null otherwise")
    hh = rec["honest_heterogeneity"]
    if not isinstance(hh, dict) \
            or hh.get("source") not in ("calibration_pre_phase",
                                        "run_blocks"):
        raise ValueError("honest_heterogeneity needs a known source")
    for key in ("mean_rel_dev", "p95_rel_dev", "max_rel_dev"):
        v = hh.get(key)
        if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f"honest_heterogeneity.{key} must be a "
                             "finite number >= 0")
    if hh["p95_rel_dev"] > hh["max_rel_dev"] \
            or hh["mean_rel_dev"] > hh["max_rel_dev"]:
        raise ValueError("heterogeneity stats are inconsistent "
                         "(mean/p95 exceed max)")
    if not isinstance(hh.get("per_channel_mean_rel_dev"), list) \
            or hh["source"] == "calibration_pre_phase" and \
            len(hh["per_channel_mean_rel_dev"]) != cfg.replica_count:
        raise ValueError("honest_heterogeneity.per_channel_mean_rel_dev "
                         "must list one value per channel")
    if not isinstance(rec["deployment"], str) \
            or "in-process" not in rec["deployment"]:
        raise ValueError("deployment must state the in-process channel "
                         "caveat")
    if "journal" in rec:
        j = rec["journal"]
        sha = j.get("sha256") if isinstance(j, dict) else None
        if not isinstance(j.get("path"), str) or not j["path"] \
                or not isinstance(sha, str) or len(sha) != 64:
            raise ValueError("journal needs a path and a 64-char sha256")
    if "utility" in rec:
        for split in ("utility_before", "utility_after"):
            u = rec["utility"].get(split)
            if not isinstance(u, dict) \
                    or not isinstance(u.get("cross_entropy"), (int, float)) \
                    or not math.isfinite(u["cross_entropy"]) \
                    or not isinstance(u.get("perplexity"), (int, float)):
                raise ValueError(f"utility.{split} must carry finite "
                                 "cross_entropy and perplexity")
    return rec


def capture_to_replica_bundle(capture_dir, observations=None, truth=None,
                              max_blocks=None):
    """Turn a replica-median splitbench capture into the
    llm_attacker.replica_observations.v1 / .replica_truth.v1 JSONs that
    `replica-integrity --captures/--truth` consumes DIRECTLY against the
    trained channels.

    One sample per CAPTURED block (journal entries with captured=true,
    paired in scan order with each channel's wire captures under
    <capture_dir>/channels/ch<k>/): ``replicas`` are the K per-channel
    wire tensors (first batch row, flattened), ``key_ids`` the per-channel
    gauge ids from the provenance sidecar. ``truth`` is the per-block
    coordinate GROUP MEAN of the K honest channels — the scheme has no
    external ground truth (the aggregated median is the defender's own
    output, which would trivialize scoring); the clean arm's median-vs-
    mean gap is exactly the honest heterogeneity the sidecar budgets.
    Validated with replica_integrity.validate_observations /
    validate_truth before returning. Returns (observations, truth)."""
    require_torch()
    from . import captures
    from .attacks import replica_integrity
    sidecar = os.path.join(capture_dir, REPLICA_PROVENANCE_NAME)
    if not os.path.isfile(sidecar):
        raise ValueError(f"no {REPLICA_PROVENANCE_NAME} in {capture_dir}: "
                         "re-run the capture with --defense replica-median")
    with open(sidecar) as f:
        prov = validate_replica_provenance(json.load(f))
    k = prov["replica_count"]
    jpath = os.path.join(capture_dir, REPLICA_JOURNAL_NAME)
    blocks = []
    with open(jpath) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("captured"):
                blocks.append(entry)
    if max_blocks is not None:
        blocks = blocks[:int(max_blocks)]
    if not blocks:
        raise ValueError(f"no captured blocks in {jpath}")
    samples = [{"id": f"block{e['block_index']:05d}",
                "replicas": [None] * k,
                "key_ids": list(prov["gauge_ids"])} for e in blocks]
    for ch in range(k):
        cdir = os.path.join(capture_dir, "channels", f"ch{ch}")
        recs = []
        if os.path.isdir(cdir):
            recs = [p for _m, p in captures.scan_captures(cdir, "training")]
            if not recs:
                recs = [p for _m, p in
                        captures.scan_captures(cdir, "inference")]
        if len(recs) < len(blocks):
            raise ValueError(f"channel ch{ch} has {len(recs)} captures "
                             f"for {len(blocks)} captured blocks")
        for slot, pt in zip(samples, recs):
            wire = captures.load_tensor(pt).double()
            if wire.dim() == 2:
                wire = wire.unsqueeze(0)
            slot["replicas"][ch] = wire[0].reshape(-1).tolist()
    targets = {}
    for slot in samples:
        if any(r is None for r in slot["replicas"]):
            raise ValueError("incomplete per-channel captures for a "
                             "captured block")
        targets[slot["id"]] = [sum(col) / k
                               for col in zip(*slot["replicas"])]
    obs = {"schema": "llm_attacker.replica_observations.v1",
           "samples": samples}
    scoring = {"schema": "llm_attacker.replica_truth.v1",
               "targets": targets}
    replica_integrity.validate_observations(obs)
    replica_integrity.validate_truth(scoring)
    if observations is not None:
        with open(observations, "w") as f:
            json.dump(obs, f)
    if truth is not None:
        with open(truth, "w") as f:
            json.dump(scoring, f)
    return obs, scoring


def build_parser():
    ap = argparse.ArgumentParser(
        prog="python -m attacker.trained_defenses", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    cv = sub.add_parser("convert", help="trained-bottleneck capture dir -> "
                                        "representation_privacy.v1 bundle")
    cv.add_argument("--capture-dir", required=True)
    cv.add_argument("--canonical", default=None,
                    help="canonical.pt path (default "
                         "<capture-dir>/canonical.pt)")
    cv.add_argument("--dim", type=int, default=None,
                    help="bottleneck width k (default: read the capture's "
                         "provenance sidecar)")
    cv.add_argument("--label-position", type=int, default=0,
                    help="token position of the per-document label "
                         "(default 0)")
    cv.add_argument("--output", required=True)
    cm = sub.add_parser("convert-mixup",
                        help="secret-mixup capture dir -> view_privacy.v1 "
                             "mixup bundle")
    cm.add_argument("--capture-dir", required=True)
    cm.add_argument("--output", required=True)
    cr = sub.add_parser("convert-replica",
                        help="replica-median capture dir -> "
                             "llm_attacker.replica_observations.v1 + "
                             ".replica_truth.v1 JSONs for "
                             "replica-integrity --captures/--truth")
    cr.add_argument("--capture-dir", required=True)
    cr.add_argument("--observations", required=True)
    cr.add_argument("--truth", required=True)
    cr.add_argument("--max-blocks", type=int, default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "convert-replica":
        obs, scoring = capture_to_replica_bundle(
            args.capture_dir, observations=args.observations,
            truth=args.truth, max_blocks=args.max_blocks)
        print(json.dumps({"observations_schema": obs["schema"],
                          "truth_schema": scoring["schema"],
                          "samples": len(obs["samples"]),
                          "observations": args.observations,
                          "truth": args.truth}, indent=2))
        return 0
    if args.command == "convert-mixup":
        bundle = capture_to_view_privacy_bundle(args.capture_dir,
                                                output=args.output)
        print(json.dumps({"schema": bundle["schema"],
                          "case": bundle["case"],
                          "targets": len(bundle["data"]["releases"]),
                          "output": args.output}, indent=2))
        return 0
    if args.command == "convert":
        bundle = capture_to_representation_bundle(
            args.capture_dir, canonical_path=args.canonical, dim=args.dim,
            label_position=args.label_position, output=args.output)
        print(json.dumps({"schema": bundle["schema"],
                          "calibration": len(bundle["calibration"]),
                          "evaluation": len(bundle["evaluation"]),
                          "output": args.output}, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
