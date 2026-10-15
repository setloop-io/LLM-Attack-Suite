"""Replica aggregation engineering battery, not a BFT or cryptographic proof.

--captures JSON: {schema: llm_attacker.replica_observations.v1, samples:
[{id: str, replicas: [[finite numbers], ...], key_ids: [str, ...]}]}.
--truth separate scoring JSON: {schema: llm_attacker.replica_truth.v1,
 targets: {sample_id: [finite numbers]}}. IDs denote key equivalence only;
actual key material is neither required nor recovered. Fixture replay is
not a live deployment attack. Malicious replacements use observed vectors,
never scoring targets. Coordinate medians require an honest majority merely
to remain in honest coordinate ranges; they need not recover exact truth.

Live multi-channel instantiation (--live): samples are produced by
independently seeded channels (spawn_channels), each with its own gauge
(random.Random(seed + 104729*index)) and a distinct derived key id, so the
live-* arms run collusion/union attacks against independently generated
channel outputs rather than in-process clones. Channels run either
in-process (--channel-runner inproc, the deterministic CI path) or as
spawned worker subprocesses (--channel-runner subprocess, default) that
keep per-channel drift state alive across requests. This is an engineering
audit of the aggregation policy under realistic channel independence, not
a falsification of any deployed defense. The future cross-host integration
point is attacker/remote.py together with attacker/nodes.json (channel
workers dispatched to configured nodes); that deployment integration is
explicitly out of scope here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys

from .. import artifacts
from ..evidence import EvidenceLogger, file_sha256
from .common import add_common_args

EXPERIMENT_ID = "replica_integrity"
MODES = ("training", "inference")
REQUIRES_LABELS = False
DESCRIPTION = "Replica collusion, K=2 abort, honest heterogeneity, and shared-key disclosure blast radius; synthetic or fixture replay"
BASE_ARMS = ("clean", "one-malicious", "two-colluding", "k2-disagreement",
             "honest-heterogeneity", "shared-key-disclosure")
LIVE_ARMS = tuple("live-" + a for a in BASE_ARMS)
ARMS = BASE_ARMS + LIVE_ARMS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHANNEL_SEED_STRIDE = 104729
_NOISE_BOUND = 2e-3
_DRIFT_STEP = 1e-3
_DRIFT_BOUND = 3e-3


def _vector(value):
    if (not isinstance(value, list) or not value or
        any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError("vectors must be nonempty lists of finite numbers")
    return [float(x) for x in value]


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_observations(observations):
    if (not isinstance(observations, dict) or set(observations) != {"schema", "samples"}
        or observations["schema"] != "llm_attacker.replica_observations.v1"
        or not isinstance(observations["samples"], list) or not observations["samples"]):
        raise ValueError("invalid replica observation schema")
    return observations


def validate_truth(scoring):
    if (not isinstance(scoring, dict) or set(scoring) != {"schema", "targets"}
        or scoring["schema"] != "llm_attacker.replica_truth.v1"
        or not isinstance(scoring["targets"], dict)):
        raise ValueError("invalid replica truth schema")
    return scoring


def load_inputs(captures, truth):
    observations = validate_observations(
        json.loads(Path(captures).read_text(), object_pairs_hook=_strict_object))
    scoring = validate_truth(
        json.loads(Path(truth).read_text(), object_pairs_hook=_strict_object))
    samples, seen = [], set()
    for row in observations["samples"]:
        if not isinstance(row, dict) or set(row) != {"id", "replicas", "key_ids"}:
            raise ValueError("observation must contain id, replicas, key_ids only")
        sid = row["id"]
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("sample IDs must be unique nonempty strings")
        seen.add(sid)
        if not isinstance(row["replicas"], list) or len(row["replicas"]) < 2:
            raise ValueError("at least two replica vectors required")
        reps = [_vector(x) for x in row["replicas"]]
        keys = row["key_ids"]
        if (not isinstance(keys, list) or len(keys) != len(reps)
            or any(not isinstance(x, str) or not x for x in keys)):
            raise ValueError("key_ids must identify every replica")
        if sid not in scoring["targets"]:
            raise ValueError("missing scoring target")
        target = _vector(scoring["targets"][sid])
        if any(len(x) != len(target) for x in reps):
            raise ValueError("replica/target dimensions differ")
        samples.append({"id": sid, "replicas": reps, "key_ids": keys})
    if set(scoring["targets"]) != seen:
        raise ValueError("truth IDs must match observation IDs exactly")
    return samples, {sid: _vector(scoring["targets"][sid]) for sid in seen}


def _latent_target(vec_seed, dim):
    """Latent target shared by all channels for one sample."""
    rng = random.Random(vec_seed)
    return [rng.uniform(-1, 1) for _ in range(dim)]


def channel_key_id(index, seed):
    """Distinct per-channel key identity derived from the gauge seed."""
    gauge_seed = seed + _CHANNEL_SEED_STRIDE * index
    digest = hashlib.sha256(repr(gauge_seed).encode()).hexdigest()[:8]
    return f"channel-{index}-{digest}"


class _ChannelState:
    """One independently seeded channel: own gauge, own key, drifting bias."""

    def __init__(self, index, dim, seed):
        if dim < 1:
            raise ValueError("channel dim must be positive")
        self.index = index
        self.dim = dim
        self.gauge = random.Random(seed + _CHANNEL_SEED_STRIDE * index)
        self.key_id = channel_key_id(index, seed)
        self.drift = [0.0] * dim

    def respond(self, vec_seed, steps=1):
        if steps < 1:
            raise ValueError("steps must be positive")
        for _ in range(steps):
            self.drift = [max(-_DRIFT_BOUND, min(_DRIFT_BOUND,
                          d + self.gauge.uniform(-_DRIFT_STEP, _DRIFT_STEP)))
                          for d in self.drift]
        noise = [self.gauge.uniform(-_NOISE_BOUND, _NOISE_BOUND)
                 for _ in range(self.dim)]
        return [t + d + n for t, d, n in
                zip(_latent_target(vec_seed, self.dim), self.drift, noise)]

    def close(self):
        pass


class _SubprocessChannel:
    """Channel living in a spawned worker process; state persists across
    requests, so per-channel drift genuinely accumulates out of process."""

    def __init__(self, index, dim, seed):
        self.index = index
        self.dim = dim
        self.key_id = channel_key_id(index, seed)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "attacker.attacks.replica_integrity",
             "--channel-worker", "--channel-index", str(index),
             "--hidden", str(dim), "--seed", str(seed)],
            cwd=str(_REPO_ROOT), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True)

    def respond(self, vec_seed, steps=1):
        request = json.dumps({"vec_seed": vec_seed, "steps": steps})
        self.proc.stdin.write(request + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"channel worker {self.index} exited")
        vec = _vector(json.loads(line))
        if len(vec) != self.dim:
            raise RuntimeError(f"channel worker {self.index} width mismatch")
        return vec

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
            self.proc.wait(timeout=10)


def spawn_channels(n, dim, seed, runner="subprocess"):
    """n independent seeded channels sharing only the latent targets."""
    if n < 2:
        raise ValueError("at least two channels required")
    if runner == "inproc":
        return [_ChannelState(i, dim, seed) for i in range(n)]
    if runner == "subprocess":
        return [_SubprocessChannel(i, dim, seed) for i in range(n)]
    raise ValueError("unknown channel runner")


def _channel_worker(args):
    """Hidden worker mode: one JSON request per stdin line, one JSON vector
    per stdout line; gauge and drift state evolve across requests."""
    if args.channel_index is None or args.channel_index < 0:
        raise ValueError("--channel-worker requires --channel-index >= 0")
    state = _ChannelState(args.channel_index, args.hidden, args.seed)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line, object_pairs_hook=_strict_object)
        if not isinstance(req, dict) or set(req) != {"vec_seed", "steps"}:
            raise ValueError("worker requests must contain vec_seed, steps only")
        vec = state.respond(int(req["vec_seed"]), int(req["steps"]))
        sys.stdout.write(json.dumps(vec) + "\n")
        sys.stdout.flush()
    return 0


def _live_samples(args):
    """Samples from spawned independent channels; targets stay latent."""
    n_channels = args.channels if args.channels is not None else args.replicas
    if n_channels < 3 or args.samples < 1 or args.hidden < 1:
        raise ValueError("live requires >=3 channels, positive samples and hidden")
    channels = spawn_channels(n_channels, args.hidden, args.seed,
                              args.channel_runner)
    samples, targets = [], {}
    try:
        count = min(args.samples, 4) if args.quick else args.samples
        for i in range(count):
            sid = str(i)
            vec_seed = args.seed * 1000003 + i
            samples.append({"id": sid,
                            "replicas": [ch.respond(vec_seed, 1)
                                         for ch in channels],
                            "key_ids": [ch.key_id for ch in channels]})
            targets[sid] = _latent_target(vec_seed, args.hidden)
    finally:
        for ch in channels:
            ch.close()
    return samples, targets


def aggregate(replicas, tolerance):
    """Explicit modeled policy: strict-majority support near coordinate median.

    With K=2 both replicas must agree within tolerance (L-infinity).
    Alarm reports *any* disagreement; acceptance and alarm are separate.
    This is an experimental policy, not a claimed implementation of a target.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    reps = [_vector(x) for x in replicas]
    if len(reps) < 2 or len({len(x) for x in reps}) != 1:
        raise ValueError("need at least two equal-width replicas")
    median = [statistics.median(v) for v in zip(*reps)]
    disagreement = max(max(v) - min(v) for v in zip(*reps)) > tolerance
    support = sum(max(abs(a-b) for a, b in zip(x, median)) <= tolerance for x in reps)
    accepted = not disagreement if len(reps) == 2 else support >= len(reps)//2 + 1
    return {"prediction": median, "accepted": accepted, "alarm": disagreement,
            "support": support, "required_support": len(reps)//2 + 1}


def perturb(sample, arm, shift, tolerance):
    """Attack has observations and supplied key identities only, no truth."""
    replicas = [list(v) for v in sample["replicas"]]
    count = 0
    if arm in ("one-malicious", "two-colluding"):
        count = 1 if arm == "one-malicious" else 2
        if len(replicas) < 3:
            raise ValueError(f"{arm} requires at least three replicas")
        replacement = [x + shift for x in replicas[-1]]
        replicas[-count:] = [list(replacement) for _ in range(count)]
    elif arm == "k2-disagreement":
        replicas = replicas[:2]
        replicas[1] = [x + shift for x in replicas[1]]
        count = 1
    elif arm == "honest-heterogeneity":
        center = replicas[0]
        replicas = [[x-tolerance*.4 for x in center],
                    [x+tolerance*.4 for x in center],
                    [x+tolerance*.4 for x in center]]
        count = 1
    elif arm not in ("clean", "shared-key-disclosure"):
        raise ValueError("unknown arm")
    return replicas, count


def evaluate(sample, target, arm, shift, tolerance, error_tolerance):
    replicas, count = perturb(sample, arm, shift, tolerance)
    decision = aggregate(replicas, tolerance)
    error = max(abs(a-b) for a, b in zip(decision["prediction"], target))
    result = {"sample_id": sample["id"], "arm": arm, "replica_count": len(replicas),
              "attacker_count": count, "honest_majority": count < len(replicas)/2,
              **decision, "max_absolute_error": error,
              "accepted_error": decision["accepted"] and error > error_tolerance,
              "availability_abort": not decision["accepted"],
              "false_positive": decision["alarm"] if count == 0 else None,
              "false_abort": not decision["accepted"] if count == 0 else None}
    if arm == "shared-key-disclosure":
        ids = sample["key_ids"]
        result.update(disclosed_replica_index=0,
                      compromised_replicas=sum(x == ids[0] for x in ids),
                      independent_key_counterfactual_compromised_replicas=1,
                      primitive_broken=False)
    return result


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap, hidden=8)
    ap.add_argument("--captures")
    ap.add_argument("--truth")
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(BASE_ARMS))
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--shift", type=float, default=1.)
    ap.add_argument("--tolerance", type=float, default=.1)
    ap.add_argument("--error-tolerance", type=float, default=.01)
    ap.add_argument("--evidence", action="store_true")
    ap.add_argument("--live", action="store_true",
                    help="sample from spawned independent channels instead of the toy clone generator")
    ap.add_argument("--channel-runner", choices=("inproc", "subprocess"),
                    default="subprocess",
                    help="inproc is the deterministic CI path; subprocess spawns worker processes")
    ap.add_argument("--channels", type=int, default=None,
                    help="number of live channels (default: --replicas)")
    ap.add_argument("--channel-worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--channel-index", type=int, default=None, help=argparse.SUPPRESS)
    return ap


def run(args):
    for name in ("shift", "tolerance", "error_tolerance"):
        v = getattr(args, name)
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.toy == bool(args.captures) or bool(args.captures) != bool(args.truth):
        raise ValueError("select --toy or both --captures and --truth")
    if args.live and not args.toy:
        raise ValueError("--live requires --toy (fixture replay is unchanged)")
    live_arms = [a for a in args.arms if a.startswith("live-")]
    if live_arms and not (args.toy and args.live):
        raise ValueError("live-* arms require --toy --live")
    if args.live and args.channels is not None and args.channels < 3:
        raise ValueError("--channels must be at least 3")
    if getattr(args, "mode", "inference") not in MODES:
        raise ValueError("unsupported mode")
    if args.evidence and not args.output:
        raise ValueError("--evidence requires --output")
    if args.output and any(Path(str(args.output)+s).exists() for s in ("", ".jsonl", ".evidence.json", ".evidence.jsonl")):
        raise ValueError("output must be fresh")
    if args.toy:
        if args.replicas < 3 or args.samples < 1 or args.hidden < 1:
            raise ValueError("toy requires >=3 replicas, positive samples and hidden")
        if args.live:
            samples, targets = _live_samples(args)
        else:
            rng = random.Random(args.seed)
            samples, targets = [], {}
            for i in range(min(args.samples, 4) if args.quick else args.samples):
                vec = [rng.uniform(-1, 1) for _ in range(args.hidden)]
                sid = str(i)
                samples.append({"id": sid, "replicas": [vec[:] for _ in range(args.replicas)],
                                "key_ids": ["shared-assumed-key"] * args.replicas})
                targets[sid] = vec
    else:
        samples, targets = load_inputs(args.captures, args.truth)
        if "honest-heterogeneity" in args.arms:
            raise ValueError("honest-heterogeneity is a constructed toy case; select replay arms explicitly")
    rows = []
    for arm in args.arms:
        base_arm = arm[5:] if arm.startswith("live-") else arm
        for s in samples:
            row = evaluate(s, targets[s["id"]], base_arm, args.shift,
                           args.tolerance, args.error_tolerance)
            row["arm"] = arm
            rows.append(row)
    out = artifacts.make_artifact("llm_attacker.replica_integrity.v1", vars(args),
        {"adversary": "active replica participant, optionally colluding; disclosed-key holder",
         "reads": "replica vectors; key equivalence identities", "truth": "scoring only"})
    out.update(results=rows,
        evidence_tier=("live-channel-engineering" if args.live else
                       "synthetic-engineering" if args.toy else "fixture-replay-engineering"),
        limitations=["No live injector or deployment-level BFT guarantee.",
                      "Coordinate median only bounds honest coordinate ranges with an honest majority.",
                      "Quorum acceptance can coexist with a disagreement alarm.",
                      "Shared key identity models conditional disclosure, not a cryptographic break."])
    if not args.toy:
        out["input_sha256"] = {"captures": file_sha256(args.captures), "truth": file_sha256(args.truth)}
    for arm in args.arms:
        group = [r for r in rows if r["arm"] == arm]
        summary = {"arm": arm, "samples": len(group)}
        for key in ("accepted_error", "availability_abort", "alarm", "false_positive", "false_abort"):
            values = [r[key] for r in group if r[key] is not None]
            summary[key+"_rate"] = sum(values)/len(values) if values else None
        out["summary"].append(summary)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with EvidenceLogger(args.output, node_role=EXPERIMENT_ID,
                            extra={"evidence_tier": out["evidence_tier"]}) as ev:
            for row in rows:
                artifacts.append_jsonl(args.output, row)
                ev.event("cell", row)
            artifacts.write_artifact(args.output, out, evidence=ev)
    print(json.dumps(out["summary"], indent=2))
    return 0


def self_test():
    assert aggregate([[0.], [0.], [1.]], .1)["prediction"] == [0.]
    assert aggregate([[0.], [1.], [1.]], .1)["prediction"] == [1.]
    assert not aggregate([[0.], [1.]], .1)["accepted"]
    keys = {channel_key_id(i, 7) for i in range(4)}
    assert len(keys) == 4 and all(k.startswith(f"channel-{i}-")
                                  for i, k in enumerate(sorted(keys)))
    args = build_parser().parse_args(["--toy", "--live", "--quick",
                                      "--channel-runner", "inproc",
                                      "--arms", "live-clean",
                                      "live-two-colluding",
                                      "live-shared-key-disclosure"])
    assert run(args) == 0
    pair = spawn_channels(2, 3, 11, "subprocess")
    ch = pair[0]
    try:
        first, second = ch.respond(5, 1), ch.respond(6, 1)
        other = pair[1].respond(5, 1)
    finally:
        for c in pair:
            c.close()
    assert len(first) == len(second) == 3 and first != second
    assert first != other and ch.key_id != pair[1].key_id
    assert ch.key_id == channel_key_id(0, 11)
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.channel_worker:
        return _channel_worker(args)
    return run(args)


if __name__ == "__main__":
    if "--channel-worker" in sys.argv:
        raise SystemExit(main())
    raise SystemExit(self_test())
