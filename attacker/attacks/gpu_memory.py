#!/usr/bin/env python3
"""Citation scope (reference audit): these allocator and stream-contention probes
do not implement LeftoverLocals/CVE-2023-4969 (cross-process GPU local
memory) or GPU.zip (graphics compression). The toy allocator is an
analogy, not a model of the CVE's underlying mechanism.
See docs/reference-audit-2026-09-22.md.

gpu-memory — allocator-residue and contention probes on the
Phase B systems/hardware seam.

Two arms:

  allocator-residue   probe the GPU allocator state between victim kernels:
                      latent rows the victim wrote and freed WITHOUT
                      scrubbing are recoverable by the next same-size
                      allocation. Implementation tiers, in order:
                      (a) pynvml     — memory-usage series (per-process when
                                       the driver exposes it): is victim
                                       allocator activity observable at all;
                      (b) torch-cuda — memory_allocated/memory_reserved
                                       deltas around a victim workload, plus
                                       an uninitialized-allocation read
                                       probe: allocate WITHOUT zeroing right
                                       after the victim frees and count
                                       recoverable bytes / decoded rows;
                      (c) sim        — pure-python simulated GPU buffer pool
                                       (the --toy tier): victim writes
                                       latent rows, frees, attacker
                                       reallocates and reads the residue.
  scrub-check         validate the defense claim "we zero buffers on free":
                      the victim writes a canary pattern, frees, the attacker
                      re-allocates and counts surviving canary bytes. The
                      scrubbed variant must show ~0 recovery — that is the
                      defense-validated arm.
  um-timing           ALLOCATION-FREE contention channel for unified-memory
                      hardware (GB10-class: CPU and GPU share one memory
                      controller): a co-tenant process measures memory-
                      bandwidth contention signatures (repeated pointer-
                      chase/read benchmarks) correlated with the victim's
                      GPU kernel schedule — when the victim runs large
                      kernels, the attacker's fixed benchmark stalls.
                      Tiers: (sim) a simulated two-process contention world
                      with a planted on/off victim schedule -> schedule
                      recovery AUC vs a shuffled-label control (the --toy
                      tier); (torch-cuda) a real measurement harness —
                      the victim is a loop of large GPU matmuls with a
                      planted on/off pattern on a dedicated CUDA stream,
                      the attacker a timing sampler on the default stream
                      (a SINGLE-PROCESS approximation of co-tenancy: two
                      streams in one process standing in for two processes
                      sharing the unified memory controller — documented,
                      not a multi-process claim).

Label-free (the attacker only observes allocator state / reclaimed bytes /
its own benchmark timings). Modes: training and inference — any split
phase whose boundary tensors transit GPU memory the attacker can
subsequently allocate, or whose kernel schedule contends with the
attacker's benchmarks on a shared memory controller.

Usage:
    python -m attacker --mode training --attack gpu-memory --help
    python -m attacker --mode inference --attack gpu-memory --toy --quick \
        --output /tmp/gpumem.json
    python -m attacker --mode training --attack gpu-memory \
        --arm scrub-check --toy --output /tmp/scrub.json
    python -m attacker --mode inference --attack gpu-memory \
        --arm um-timing --toy --output /tmp/umt.json
    python -m attacker --mode training --attack gpu-memory \
        --arm um-timing --slots 64 --output umt.json   # needs torch+CUDA
    python -m attacker --mode training --attack gpu-memory \
        --evidence --output /tmp/gpumem.json        # real tiers: pynvml/torch
"""

import argparse
import random
import struct

try:
    import torch
except ImportError:
    torch = None

from .. import artifacts, gate, hwprobe
from .common import add_common_args, journal_error
from .lira_membership import rank_auc

EXPERIMENT_ID = "gpu_memory_residue"
MODES = ("training", "inference")
REQUIRES_LABELS = False
DESCRIPTION = ("Phase B: allocator-residue leftover-GPU-"
               "memory attack — allocator-residue probing between victim "
               "kernels + scrub-check validation of zero-on-free defenses "
               "+ um-timing: unified-memory (GB10-class) bandwidth-"
               "contention channel recovering the victim's kernel schedule")

ARMS = ("allocator-residue", "scrub-check", "um-timing")

CANARY = bytes.fromhex("deadbeefcafebabe")


class ToyGPUPool:
    """Simulated GPU allocator: one bytearray arena, LIFO free list.

    free() returns the region to the free list; the NEXT same-size alloc()
    reuses the most recently freed block (the allocator hot path). Only
    free() with scrub_on_free=True zeroes the region — everything else
    leaves residue in place, modelling allocator reuse, which is distinct from
    the GPU local-memory mechanism of CVE-2023-4969."""

    def __init__(self, nbytes, scrub_on_free=False):
        self.arena = bytearray(nbytes)
        self.free_list = []
        self.high_water = 0
        self.scrub_on_free = scrub_on_free

    def alloc(self, n):
        for i in range(len(self.free_list) - 1, -1, -1):
            off, sz = self.free_list[i]
            if sz >= n:
                self.free_list.pop(i)
                if sz > n:
                    self.free_list.append((off + n, sz - n))
                return off
        off = self.high_water
        if off + n > len(self.arena):
            raise MemoryError(f"toy pool exhausted ({off + n} > "
                              f"{len(self.arena)} bytes)")
        self.high_water = off + n
        return off

    def free(self, off, n):
        if self.scrub_on_free:
            self.arena[off:off + n] = b"\x00" * n
        self.free_list.append((off, n))

    def write(self, off, data):
        self.arena[off:off + len(data)] = data

    def read(self, off, n):
        return bytes(self.arena[off:off + n])


def make_codebook(vocab, dim, seed):
    """Toy latent-row codebook: vocab deterministic gaussian rows of dim
    floats (the 'model' the victim's rows come from; public to the
    attacker)."""
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 4.0) for _ in range(dim)] for _ in range(vocab)]


def encode_row(vec):
    return struct.pack("<%df" % len(vec), *vec)


def decode_rows(blob, dim):
    """bytes -> list of float rows (tuples), len(blob)//(4*dim) rows."""
    per = 4 * dim
    n = len(blob) // per
    vals = struct.unpack("<%df" % (n * dim), blob[:n * per])
    return [vals[i * dim:(i + 1) * dim] for i in range(n)]


def nearest_codebook(row, codebook):
    best, best_d = 0, None
    for v, vec in enumerate(codebook):
        d = 0.0
        for a, b in zip(row, vec):
            d += (a - b) * (a - b)
        if best_d is None or d < best_d:
            best, best_d = v, d
    return best


def sim_allocator_residue_cell(rows, dim, vocab, scrubbed, seed):
    """One toy allocator-residue cell: victim writes `rows` latent rows and
    frees (scrubbing iff `scrubbed`); the attacker reallocates the hot block
    and reads residue. A control region the victim never touched gives the
    chance-level reference. Returns the per-cell journal record."""
    rng = random.Random(seed)
    codebook = make_codebook(vocab, dim, seed + 1)
    tokens = [rng.randrange(vocab) for _ in range(rows)]
    blob = b"".join(encode_row(codebook[t]) for t in tokens)
    n = len(blob)
    pool = ToyGPUPool(4 * n + 4096, scrub_on_free=scrubbed)
    ctl_off = pool.alloc(n)
    vic_off = pool.alloc(n)
    pool.write(vic_off, blob)
    pool.free(vic_off, n)
    atk_off = pool.alloc(n)
    residue = pool.read(atk_off, n)
    control = pool.read(ctl_off, n)

    victim_nonzero = sum(1 for b in blob if b != 0)
    recoverable = sum(1 for vb, rb in zip(blob, residue)
                      if vb != 0 and rb == vb)
    ctl_recoverable = sum(1 for vb, cb in zip(blob, control)
                          if vb != 0 and cb == vb)
    dec_rows = decode_rows(residue, dim)
    ctl_rows = decode_rows(control, dim)
    acc = sum(1 for r, t in zip(dec_rows, tokens)
              if nearest_codebook(r, codebook) == t) / max(1, rows)
    ctl_acc = sum(1 for r, t in zip(ctl_rows, tokens)
                  if nearest_codebook(r, codebook) == t) / max(1, rows)
    return {"tier": "sim", "arm": "allocator-residue",
            "defense": "scrubbed" if scrubbed else "unscrubbed",
            "rows": rows, "row_bytes": 4 * dim,
            "bytes_recoverable": recoverable,
            "recovery_rate": recoverable / max(1, victim_nonzero),
            "byte_recovery": recoverable / n,
            "control_byte_recovery": ctl_recoverable / n,
            "fraction_nonzero": sum(1 for b in residue if b != 0) / n,
            "token_recovery": acc, "control_token_recovery": ctl_acc,
            "n_eval_bytes": n}


def sim_scrub_check_cell(rows, row_bytes, scrubbed, seed):
    """One toy scrub-check cell: victim writes the canary, frees (scrubbing
    iff `scrubbed`), attacker reallocates and counts surviving canary
    bytes. The scrubbed cell must show ~0 recovery (defense validated)."""
    n = rows * row_bytes
    blob = (CANARY * (n // len(CANARY) + 1))[:n]
    pool = ToyGPUPool(4 * n + 4096, scrub_on_free=scrubbed)
    ctl_off = pool.alloc(n)
    vic_off = pool.alloc(n)
    pool.write(vic_off, blob)
    pool.free(vic_off, n)
    atk_off = pool.alloc(n)
    residue = pool.read(atk_off, n)
    control = pool.read(ctl_off, n)
    found = sum(1 for cb, rb in zip(blob, residue) if rb == cb)
    ctl_found = sum(1 for cb, rb in zip(blob, control) if rb == cb)
    return {"tier": "sim", "arm": "scrub-check",
            "defense": "scrubbed" if scrubbed else "unscrubbed",
            "rows": rows, "row_bytes": row_bytes,
            "canary_bytes": n, "canary_found": found,
            "canary_recovery": found / n, "control_canary_recovery":
            ctl_found / n, "n_eval_bytes": n}


def plant_schedule(slots, on_fraction, seed):
    """Planted victim kernel schedule: Bernoulli(on_fraction) on/off per
    slot, forced to at least 2 on and 2 off slots so the AUC contrast is
    always defined."""
    rng = random.Random(seed)
    sched = [rng.random() < on_fraction for _ in range(slots)]
    if sum(sched) < 2:
        sched[0] = sched[1] = True
    if sum(sched) > slots - 2:
        sched[-1] = sched[-2] = False
    return sched


def sim_um_timing_cell(slots, on_fraction, base_ms, penalty_ms, noise_ms,
                       seed):
    """One toy um-timing cell: a simulated two-process contention world.
    The victim's planted on/off kernel schedule multiplies the memory-
    bandwidth pressure; the co-tenant attacker times a fixed pointer-
    chase/read benchmark every slot (base + penalty*on + noise). Recovery
    = rank AUC of the latency series against the planted schedule;
    control = the same series against a SHUFFLED schedule (attacker-
    computable null, ~0.5)."""
    schedule = plant_schedule(slots, on_fraction, seed)
    rng = random.Random(seed + 1)
    lat = [max(0.01, rng.gauss(base_ms + (penalty_ms if on else 0.0),
                               noise_ms))
           for on in schedule]
    on_lat = [t for t, on in zip(lat, schedule) if on]
    off_lat = [t for t, on in zip(lat, schedule) if not on]
    shuffled = list(schedule)
    random.Random(seed + 2).shuffle(shuffled)
    ctl_auc = rank_auc([t for t, on in zip(lat, shuffled) if on],
                       [t for t, on in zip(lat, shuffled) if not on])
    return {"tier": "sim", "arm": "um-timing", "defense": "n/a",
            "slots": slots, "on_slots": sum(schedule),
            "mean_latency_on_ms": sum(on_lat) / len(on_lat),
            "mean_latency_off_ms": sum(off_lat) / len(off_lat),
            "latency_delta_ms": sum(on_lat) / len(on_lat)
            - sum(off_lat) / len(off_lat),
            "schedule_recovery_auc": rank_auc(on_lat, off_lat),
            "control_auc": ctl_auc,
            "n_eval_slots": min(len(on_lat), len(off_lat)),
            "note": "simulated two-process contention on a shared "
                    "memory controller (GB10-class unified memory); the "
                    "attacker never allocates victim memory — it only "
                    "times its own benchmark (allocation-free channel)"}


def pynvml_residue_cell(args):
    """Tier (a): passive allocator observability via pynvml — a memory-usage
    series (per-process when the driver exposes it). No allocation control,
    so no recovery metrics: this cell answers 'is victim allocator activity
    visible between kernels at all'. Unified-memory devices (GB10) do not
    implement nvmlDeviceGetMemoryInfo; the series then falls back to the
    driver memory-utilization percentage."""
    series = hwprobe.nvml_memory_series(args.device_index, args.samples)
    used = [u for u in series["used_bytes"] if u is not None]
    pct = [p for p in series.get("mem_util_pct", []) if p is not None]
    rec = {"tier": "pynvml", "arm": "allocator-residue",
           "device_index": series["device_index"],
           "samples": len(series["used_bytes"]),
           "per_process_attribution": any(p is not None
                                          for p in series["per_process"])}
    if used:
        rec.update({"used_bytes_min": min(used), "used_bytes_max": max(used),
                    "used_bytes_delta": max(used) - min(used)})
    elif pct:
        rec.update({"mem_util_pct_min": min(pct), "mem_util_pct_max": max(pct),
                    "mem_util_pct_delta": max(pct) - min(pct),
                    "fallback": "nvmlDeviceGetMemoryInfo not implemented on "
                                "this device (unified memory); reporting "
                                "driver memory-utilization % instead"})
    else:
        raise RuntimeError("pynvml: no memory observable implemented on "
                           f"device {series['device_index']} (neither "
                           "GetMemoryInfo nor GetUtilizationRates)")
    rec["note"] = ("observability probe — nonzero used-memory delta on an "
                   "idle attacker process means victim allocator activity "
                   "is visible between kernels")
    return rec


def _torch_allocator_residue_cell(args, seed):
    """Tier (b) allocator-residue: memory_allocated/memory_reserved deltas
    around the victim workload + an uninitialized-allocation read probe
    (torch.empty reuses the just-freed block WITHOUT zeroing)."""
    dev = torch.device(f"cuda:{args.device_index}")
    g = torch.Generator().manual_seed(seed)
    codebook = torch.randn(args.vocab, args.row_dim, generator=g) * 4.0
    tokens = torch.randint(args.vocab, (args.rows,), generator=g)
    reserved0 = torch.cuda.memory_reserved(dev)
    ctl = torch.empty((args.rows, args.row_dim), device=dev)
    vic = codebook.to(dev)[tokens].clone()
    victim_cpu = vic.cpu()
    torch.cuda.synchronize(dev)
    allocated_live = torch.cuda.memory_allocated(dev)
    if args.scrubbed:
        vic.zero_()
    del vic
    torch.cuda.synchronize(dev)
    allocated_after_free = torch.cuda.memory_allocated(dev)
    probe = torch.empty((args.rows, args.row_dim), device=dev)
    torch.cuda.synchronize(dev)
    residue = probe.cpu()
    pred = torch.cdist(residue.float(), codebook.float()).argmin(1)
    ctl_pred = torch.cdist(ctl.cpu().float(), codebook.float()).argmin(1)
    n_bytes = args.rows * 4 * args.row_dim
    recoverable = int(((residue == victim_cpu)
                       & (victim_cpu != 0)).sum().item()) * 4
    ctl_recoverable = int(((ctl.cpu() == victim_cpu)
                           & (victim_cpu != 0)).sum().item()) * 4
    return {"tier": "torch-cuda", "arm": "allocator-residue",
            "defense": "scrubbed" if args.scrubbed else "unscrubbed",
            "seed": seed, "rows": args.rows, "row_bytes": 4 * args.row_dim,
            "memory_allocated_live": int(allocated_live),
            "memory_allocated_after_free": int(allocated_after_free),
            "memory_reserved": int(torch.cuda.memory_reserved(dev)),
            "memory_reserved_baseline": int(reserved0),
            "bytes_recoverable": recoverable,
            "byte_recovery": recoverable / n_bytes,
            "control_byte_recovery": ctl_recoverable / n_bytes,
            "fraction_nonzero": (residue != 0).float().mean().item(),
            "token_recovery": (pred == tokens).float().mean().item(),
            "control_token_recovery": (ctl_pred == tokens)
            .float().mean().item(),
            "n_eval_bytes": n_bytes}


def _torch_scrub_check_cell(args, seed):
    """Tier (b) scrub-check: victim fills a canary tensor, frees (zeroing
    iff --scrubbed), attacker torch.empty()s the same shape and counts
    surviving canary elements."""
    dev = torch.device(f"cuda:{args.device_index}")
    vic = torch.full((args.rows, args.row_dim), 1337.0, device=dev)
    torch.cuda.synchronize(dev)
    if args.scrubbed:
        vic.zero_()
    del vic
    torch.cuda.synchronize(dev)
    probe = torch.empty((args.rows, args.row_dim), device=dev)
    torch.cuda.synchronize(dev)
    n = args.rows * args.row_dim
    found = int((probe == 1337.0).sum().item())
    return {"tier": "torch-cuda", "arm": "scrub-check",
            "defense": "scrubbed" if args.scrubbed else "unscrubbed",
            "seed": seed, "rows": args.rows, "row_bytes": 4 * args.row_dim,
            "canary_bytes": n * 4, "canary_float32_values": n, "canary_found": found,
            "canary_recovery": found / n, "control_canary_recovery": 0.0,
            "n_eval_bytes": n}


def _torch_um_timing_cell(args, seed):
    """Tier (b) um-timing: real contention measurement harness for
    unified-memory hosts. The victim is a loop of large GPU matmuls with
    a planted on/off pattern on a DEDICATED CUDA stream; the attacker is
    a timing sampler (fixed device-to-device copy benchmark) on the
    default stream — a SINGLE-PROCESS approximation of co-tenancy: two
    streams in one process standing in for two processes sharing the
    unified memory controller. Recovery = rank AUC of the per-slot
    benchmark latency against the planted pattern; control = the same
    series against shuffled labels."""
    import time
    dev = torch.device(f"cuda:{args.device_index}")
    g = torch.Generator().manual_seed(seed)
    schedule = (torch.rand(args.slots, generator=g)
                < args.on_fraction).tolist()
    if sum(schedule) < 2:
        schedule[0] = schedule[1] = True
    if sum(schedule) > args.slots - 2:
        schedule[-1] = schedule[-2] = False
    a = torch.randn(args.matmul_dim, args.matmul_dim, device=dev)
    b = torch.randn(args.matmul_dim, args.matmul_dim, device=dev)
    n_elems = args.bench_mb * 1024 * 1024 // 4
    bench = torch.randn(n_elems, device=dev)
    scratch = torch.empty_like(bench)
    vstream = torch.cuda.Stream(dev)
    lat = []
    for on in schedule:
        if on:
            with torch.cuda.stream(vstream):
                for _ in range(args.victim_iters):
                    a = a @ b
        t0 = time.perf_counter()
        for _ in range(args.bench_iters):
            scratch.copy_(bench)
        torch.cuda.synchronize(dev)
        lat.append((time.perf_counter() - t0) * 1000.0)
        torch.cuda.synchronize(dev)
    on_lat = [t for t, on in zip(lat, schedule) if on]
    off_lat = [t for t, on in zip(lat, schedule) if not on]
    shuffled = list(schedule)
    random.Random(seed + 2).shuffle(shuffled)
    ctl_auc = rank_auc([t for t, on in zip(lat, shuffled) if on],
                       [t for t, on in zip(lat, shuffled) if not on])
    return {"tier": "torch-cuda", "arm": "um-timing", "defense": "n/a",
            "seed": seed, "slots": args.slots, "on_slots": sum(schedule),
            "matmul_dim": args.matmul_dim, "bench_mb": args.bench_mb,
            "mean_latency_on_ms": sum(on_lat) / len(on_lat),
            "mean_latency_off_ms": sum(off_lat) / len(off_lat),
            "latency_delta_ms": sum(on_lat) / len(on_lat)
            - sum(off_lat) / len(off_lat),
            "schedule_recovery_auc": rank_auc(on_lat, off_lat),
            "control_auc": ctl_auc,
            "n_eval_slots": min(len(on_lat), len(off_lat)),
            "note": "single-process approximation of co-tenancy: victim "
                    "matmul bursts on a dedicated CUDA stream, attacker "
                    "copy benchmark on the default stream; on GB10-class "
                    "unified memory the two share one memory controller. "
                    "A production co-tenant would be a separate process — "
                    "the contention physics is the same, the isolation "
                    "boundary is not claimed here"}


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--arm", choices=ARMS, nargs="+", default=list(ARMS),
                    help="attack arm(s) to run (default: both)")
    ap.add_argument("--scrubbed", dest="scrubbed", action="store_true",
                    default=False,
                    help="victim zeroes buffers on free (the defense claim "
                         "under test); the toy scrub-check arm always runs "
                         "BOTH defense variants side by side")
    ap.add_argument("--no-scrubbed", dest="scrubbed", action="store_false",
                    help="victim frees without scrubbing (default)")
    ap.add_argument("--rows", type=int, default=256,
                    help="victim latent rows per cell (toy / torch tiers)")
    ap.add_argument("--row-dim", type=int, default=16,
                    help="floats per latent row (4 bytes each)")
    ap.add_argument("--vocab", type=int, default=64,
                    help="toy codebook size (token alphabet)")
    ap.add_argument("--samples", type=int, default=20,
                    help="pynvml memory-series samples (real tier)")
    ap.add_argument("--device-index", type=int, default=0,
                    help="GPU device index for the real tiers")
    ap.add_argument("--slots", type=int, default=64,
                    help="um-timing: schedule slots per cell")
    ap.add_argument("--on-fraction", type=float, default=0.5,
                    help="um-timing: fraction of slots the victim is busy")
    ap.add_argument("--base-ms", type=float, default=5.0,
                    help="um-timing toy: contention-free benchmark latency")
    ap.add_argument("--penalty-ms", type=float, default=8.0,
                    help="um-timing toy: added latency when the victim's "
                         "kernels contend")
    ap.add_argument("--noise-ms", type=float, default=1.0,
                    help="um-timing toy: benchmark noise stddev")
    ap.add_argument("--matmul-dim", type=int, default=2048,
                    help="um-timing torch-cuda tier: victim matmul size")
    ap.add_argument("--bench-mb", type=int, default=64,
                    help="um-timing torch-cuda tier: attacker benchmark "
                         "buffer MiB")
    ap.add_argument("--bench-iters", type=int, default=4,
                    help="um-timing torch-cuda tier: benchmark copies per "
                         "slot")
    ap.add_argument("--victim-iters", type=int, default=8,
                    help="um-timing torch-cuda tier: victim matmuls per "
                         "busy slot")
    ap.add_argument("--evidence", action="store_true",
                    help="wrap the run in an EvidenceLogger (requires "
                         "--output): the artifact write is recorded as a "
                         "chained event and the evidence manifest is "
                         "finalized alongside")
    return ap


def _gate_cell(rec, n_seeds):
    """Attach the frozen-gate verdict to a recovery cell and derive the
    human channel label. The residue arms are scored at the BYTE level
    (the honest probe unit for memory residue; token-row decode accuracy
    stays as the interpretive metric). The um-timing arm is scored on the
    schedule-recovery AUC vs the shuffled-label control, n = the smaller
    on/off slot class (the membership-metric convention, cf.
    lira_membership). FAIL = recovery above control beyond the gate =
    the channel is OPEN."""
    if rec["arm"] == "allocator-residue":
        recovery, control = rec["byte_recovery"], rec["control_byte_recovery"]
    elif rec["arm"] == "scrub-check":
        recovery, control = rec["canary_recovery"], rec["control_canary_recovery"]
    else:
        g = gate.verdict(rec["schedule_recovery_auc"], rec["control_auc"],
                         rec["n_eval_slots"], n_seeds=n_seeds)
        rec["gate"] = g
        rec["contention_channel"] = "open" if g["verdict"] == "FAIL" \
            else ("closed" if g["verdict"] == "PASS" else "unmeasured")
        return rec
    g = gate.verdict(recovery, control, rec["n_eval_bytes"], n_seeds=n_seeds)
    rec["gate"] = g
    rec["residue_channel"] = "open" if g["verdict"] == "FAIL" else ("closed" if g["verdict"] == "PASS" else "unmeasured")
    rec["defense_validated"] = g["verdict"] == "PASS"
    return rec


def run_toy(args, evidence=None):
    if args.quick:
        args.rows = min(args.rows, 128)
        args.slots = min(args.slots, 32)
        args.seeds = args.seeds[:2]
    out = artifacts.make_artifact(
        "dtraining.attacker.gpu_memory.v1",
        {"attack": EXPERIMENT_ID, "mode": getattr(args, "mode", None),
         "toy": True, "arms": args.arm, "scrubbed": args.scrubbed,
         "rows": args.rows, "row_dim": args.row_dim, "vocab": args.vocab,
         "slots": args.slots, "on_fraction": args.on_fraction,
         "seeds": args.seeds, "seed": args.seed},
        "honest-but-curious co-resident attacker on the GPU host: after the "
        "victim's kernels free their boundary/latent buffers, the attacker "
        "allocates same-size blocks WITHOUT zeroing and reads what the "
        "victim left behind in the allocator; on "
        "unified-memory hardware (GB10-class) a co-tenant additionally "
        "reads the victim's kernel schedule off memory-bandwidth "
        "contention (um-timing, allocation-free). Label-free. "
        "Toy tier: a simulated GPU buffer pool with LIFO block reuse and "
        "a simulated two-process contention world — machinery checks of "
        "the metrics and gate plumbing, not security claims about any "
        "specific allocator or memory controller.",
        interpretation="gate FAIL / residue_channel=open: leftover victim "
                       "bytes are recoverable (token rows decodable above "
                       "the never-written control). The scrub-check arm "
                       "validates the zero-on-free defense claim: the "
                       "scrubbed variant must sit at ~0 recovery "
                       "(defense_validated=true). gate FAIL / "
                       "contention_channel=open: the victim's kernel "
                       "schedule is recoverable from the attacker's own "
                       "benchmark latencies (AUC above the shuffled-label "
                       "control).",
        evidence=evidence)
    for arm in args.arm:
        per_defense = {}
        for seed in args.seeds:
            if arm == "scrub-check":
                variants = (False, True)
            elif arm == "um-timing":
                variants = (None,)
            else:
                variants = (args.scrubbed,)
            for scrubbed in variants:
                cell = {"experiment": EXPERIMENT_ID, "seed": seed}
                try:
                    if arm == "allocator-residue":
                        rec = sim_allocator_residue_cell(
                            args.rows, args.row_dim, args.vocab, scrubbed,
                            args.seed + 1000 * seed)
                    elif arm == "scrub-check":
                        rec = sim_scrub_check_cell(args.rows,
                                                   4 * args.row_dim,
                                                   scrubbed,
                                                   args.seed + 1000 * seed)
                    else:
                        rec = sim_um_timing_cell(
                            args.slots, args.on_fraction, args.base_ms,
                            args.penalty_ms, args.noise_ms,
                            args.seed + 1000 * seed)
                except Exception as e:
                    journal_error(args.output, EXPERIMENT_ID,
                                  {**cell, "arm": arm,
                                   "defense": ("n/a" if scrubbed is None
                                               else "scrubbed" if scrubbed
                                               else "unscrubbed")}, e)
                    continue
                rec = {**cell, **rec}
                _gate_cell(rec, n_seeds=len(args.seeds))
                artifacts.append_jsonl(args.output, rec)
                out["results"].append(rec)
                per_defense.setdefault(rec["defense"], []).append(rec)
                metric = (rec["token_recovery"] if arm == "allocator-residue"
                          else rec["canary_recovery"] if arm == "scrub-check"
                          else rec["schedule_recovery_auc"])
                channel = (rec["contention_channel"] if arm == "um-timing"
                           else rec["residue_channel"])
                print(f"[gpumem] arm={arm} defense={rec['defense']} "
                      f"seed={seed}: recovery={metric:.4f} "
                      f"channel={channel}")
        for defense, recs in sorted(per_defense.items()):
            if arm == "allocator-residue":
                mean_key = "token_recovery_mean"
                vals = [r["token_recovery"] for r in recs]
            elif arm == "scrub-check":
                mean_key = "canary_recovery_mean"
                vals = [r["canary_recovery"] for r in recs]
            else:
                mean_key = "schedule_recovery_auc_mean"
                vals = [r["schedule_recovery_auc"] for r in recs]
            row = {"experiment": EXPERIMENT_ID, "arm": arm, "tier": "sim",
                   "defense": defense, mean_key: sum(vals) / len(vals),
                   "n_seeds": len(recs),
                   "gate_verdict": recs[-1]["gate"]["verdict"]}
            if arm == "um-timing":
                row["contention_channel"] = recs[-1]["contention_channel"]
            else:
                row["residue_channel"] = recs[-1]["residue_channel"]
                row["defense_validated"] = recs[-1]["defense_validated"]
            out["summary"].append(row)
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def run_real(args, evidence=None):
    tiers = hwprobe.detect_tiers()
    if not tiers:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] no hardware probe tier available on this "
            "host (need pynvml or torch+CUDA; see attacker/hwprobe.py). "
            "Use --toy for the pure-python simulation tier.")
    if "um-timing" in args.arm and "torch-cuda" not in tiers:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] um-timing needs torch+CUDA: the "
            "contention harness launches victim matmuls and a benchmark "
            "sampler on separate CUDA streams (GB10-class unified-memory "
            "hosts). No CUDA device is visible on this host. Use --toy "
            "for the simulated two-process contention tier.")
    out = artifacts.make_artifact(
        "dtraining.attacker.gpu_memory.v1",
        {"attack": EXPERIMENT_ID, "mode": getattr(args, "mode", None),
         "toy": False, "arms": args.arm, "scrubbed": args.scrubbed,
         "tiers": tiers, "rows": args.rows, "row_dim": args.row_dim,
         "slots": args.slots, "on_fraction": args.on_fraction,
         "device_index": args.device_index},
        "leftover-memory attack against real GPU allocator state "
        "(allocator-residue probe), tiered: pynvml observability "
        "series, then torch-cuda allocator deltas + uninitialized-"
        "allocation read probes; plus the um-timing unified-memory "
        "contention harness (torch-cuda only: victim matmul bursts on a "
        "dedicated CUDA stream, attacker copy benchmark on the default "
        "stream — a single-process approximation of co-tenancy). "
        "Label-free.",
        interpretation="residue_channel=open means victim bytes survive "
                       "free and are recoverable by the next allocation; "
                       "the scrubbed scrub-check cell must show ~0 canary "
                       "recovery (defense validated). "
                       "contention_channel=open means the victim's kernel "
                       "schedule is recoverable from the attacker's own "
                       "benchmark latencies (AUC above the shuffled-label "
                       "control).",
        evidence=evidence)
    for arm in args.arm:
        for seed in args.seeds:
            for tier in tiers:
                if arm == "um-timing" and tier != "torch-cuda":
                    continue
                cell = {"experiment": EXPERIMENT_ID, "arm": arm,
                        "seed": seed, "tier": tier}
                try:
                    if tier == "pynvml":
                        if arm != "allocator-residue":
                            continue
                        rec = pynvml_residue_cell(args)
                    elif arm == "allocator-residue":
                        rec = _torch_allocator_residue_cell(args, seed)
                    elif arm == "scrub-check":
                        rec = _torch_scrub_check_cell(args, seed)
                    else:
                        rec = _torch_um_timing_cell(args, seed)
                except Exception as e:
                    journal_error(args.output, EXPERIMENT_ID, cell, e)
                    out["results"].append({**cell, "error": str(e)})
                    continue
                rec = {**cell, **rec}
                if "token_recovery" in rec or "canary_recovery" in rec \
                        or "schedule_recovery_auc" in rec:
                    rec["gate"] = {"verdict": "UNMEASURED", "sampling_unit": "allocation_or_timing_trial",
                                   "reason": "Bytes within one allocation and timing slots are dependent; repeated independent trials required"}
                    rec["defense_validated"] = False
                    rec["evidence_tier"] = "hardware_descriptive"
                    rec["residue_channel"] = "unmeasured"
                    rec["contention_channel"] = "unmeasured"
                artifacts.append_jsonl(args.output, rec)
                out["results"].append(rec)
                print(f"[gpumem] tier={tier} arm={arm} seed={seed}: "
                      + ", ".join(f"{k}={v}" for k, v in rec.items()
                                  if k.endswith("recovery")
                                  or k.endswith("auc")
                                  or k == "used_bytes_delta"))
    by_tier_arm = {}
    for rec in out["results"]:
        if "error" in rec:
            continue
        key = (rec["tier"], rec["arm"], rec.get("defense", "n/a"))
        by_tier_arm.setdefault(key, []).append(rec)
    for (tier, arm, defense), recs in sorted(by_tier_arm.items()):
        row = {"experiment": EXPERIMENT_ID, "tier": tier, "arm": arm,
               "defense": defense, "n_seeds": len(recs)}
        if "token_recovery" in recs[-1]:
            row["token_recovery_mean"] = sum(
                r["token_recovery"] for r in recs) / len(recs)
            row["residue_channel"] = recs[-1]["residue_channel"]
        if "canary_recovery" in recs[-1]:
            row["canary_recovery_mean"] = sum(
                r["canary_recovery"] for r in recs) / len(recs)
            row["defense_validated"] = recs[-1]["defense_validated"]
        if "schedule_recovery_auc" in recs[-1]:
            row["schedule_recovery_auc_mean"] = sum(
                r["schedule_recovery_auc"] for r in recs) / len(recs)
            row["control_auc_mean"] = sum(
                r["control_auc"] for r in recs) / len(recs)
            row["contention_channel"] = recs[-1]["contention_channel"]
        if "used_bytes_delta" in recs[-1]:
            row["used_bytes_delta_max"] = max(r["used_bytes_delta"]
                                              for r in recs)
        out["summary"].append(row)
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def _run(args, evidence=None):
    if args.toy:
        return run_toy(args, evidence=evidence)
    return run_real(args, evidence=evidence)


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

    print("gpu-memory leftover-memory attack (pure python):")

    pool = ToyGPUPool(1024)
    a = pool.alloc(128)
    pool.write(a, b"\x41" * 128)
    pool.free(a, 128)
    b = pool.alloc(128)
    check("sim allocator reuses the hot freed block (LIFO)", a == b)
    check("unscrubbed residue is readable after free",
          pool.read(b, 128) == b"\x41" * 128)
    pool2 = ToyGPUPool(1024, scrub_on_free=True)
    c = pool2.alloc(128)
    pool2.write(c, b"\x41" * 128)
    pool2.free(c, 128)
    check("scrubbed free zeroes the region",
          pool2.read(c, 128) == b"\x00" * 128)

    cb = make_codebook(8, 4, 7)
    check("codebook encode/decode round-trips at float32 precision",
          encode_row(list(decode_rows(encode_row(cb[3]), 4)[0]))
          == encode_row(cb[3]))
    check("nearest-codebook decode of a round-tripped row is exact",
          all(nearest_codebook(decode_rows(encode_row(cb[v]), 4)[0], cb) == v
              for v in range(8)))

    rec = sim_allocator_residue_cell(rows=64, dim=8, vocab=16,
                                     scrubbed=False, seed=1)
    check("unscrubbed residue recovers every decodable row",
          rec["token_recovery"] == 1.0 and rec["recovery_rate"] == 1.0)
    check("never-written control sits at chance",
          rec["control_token_recovery"] <= 0.25)
    rec_s = sim_allocator_residue_cell(rows=64, dim=8, vocab=16,
                                       scrubbed=True, seed=1)
    check("scrubbed free leaves ~no recoverable rows",
          rec_s["token_recovery"] <= 0.25 and rec_s["fraction_nonzero"] == 0.0)

    can_u = sim_scrub_check_cell(rows=32, row_bytes=32, scrubbed=False, seed=1)
    can_s = sim_scrub_check_cell(rows=32, row_bytes=32, scrubbed=True, seed=1)
    check("unscrubbed canary fully recoverable",
          can_u["canary_recovery"] == 1.0)
    check("scrubbed canary ~0 recovery (defense validated)",
          can_s["canary_recovery"] == 0.0)

    sched = plant_schedule(64, 0.5, 1)
    check("planted schedule keeps the AUC contrast defined",
          2 <= sum(sched) <= 62)
    sched_edge = plant_schedule(8, 0.0, 1)
    check("degenerate on-fraction is forced to >= 2 on and >= 2 off",
          2 <= sum(sched_edge) <= 6)
    um = sim_um_timing_cell(64, 0.5, 5.0, 8.0, 1.0, 1)
    check("um-timing: planted schedule recovered at ceiling AUC",
          um["schedule_recovery_auc"] >= 0.99
          and um["latency_delta_ms"] > 5.0)
    check("um-timing: shuffled-label control sits at chance",
          0.3 <= um["control_auc"] <= 0.7)
    check("um-timing: the planted cell FAILs the gate (channel open)",
          gate.verdict(um["schedule_recovery_auc"], um["control_auc"],
                       um["n_eval_slots"])["verdict"] == "FAIL")
    um_quiet = sim_um_timing_cell(40000, 0.5, 5.0, 0.0, 1.0, 1)
    check("um-timing: no contention -> AUC at chance (null world)",
          0.45 <= um_quiet["schedule_recovery_auc"] <= 0.55)
    check("um-timing: the null cell PASSes the gate at floor-permitting "
          "n (channel closed; small-n toys cannot PASS by construction — "
          "the frozen gate's CI half-width exceeds 1pp below ~16k slots)",
          gate.verdict(um_quiet["schedule_recovery_auc"],
                       um_quiet["control_auc"],
                       um_quiet["n_eval_slots"])["verdict"] == "INSUFFICIENT_EVIDENCE")

    g_open = gate.verdict(1.0, 0.02, 8192)
    g_shut = gate.verdict(0.02, 0.02, 8192)
    check("gate: full recovery is a break (FAIL = channel open)",
          g_open["verdict"] == "FAIL")
    check("gate: recovery at control passes (channel closed)",
          g_shut["verdict"] == "INSUFFICIENT_EVIDENCE")

    check("hwprobe tier detection never raises (torch-free host)",
          isinstance(hwprobe.detect_tiers(), list))
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
