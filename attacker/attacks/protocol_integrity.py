#!/usr/bin/env python3
"""protocol-integrity — active protocol-level attacks on the bidirectional
split link (THREAT_MODEL_REVIEW P0): replay, reorder, duplicate, drop,
cross-session-swap, epoch-desync, downgrade, tensor-tamper, oversize and
truncate against a simulated split endpoint. Where the seam-inversion
family tests the CONFIDENTIALITY of tensors crossing the link, this attack
tests the INTEGRITY and AVAILABILITY of the split protocol itself.
Label-free; applies to training (activation/gradient frames with epochs and
key rotation) and inference (prefill/decode frames).

Attacker model: active inline network attacker / malicious relay /
compromised proxy. The attacker observes and modifies frames but does NOT
possess endpoint keys. Per-session, per-epoch auth keys stay with the
endpoints: the toy world derives them deterministically from the run seed
and uses them in memory only — never written into journals or artifacts;
journal mode takes them via a separate --session-keys file.

PROTOCOL-FRAMES JOURNAL SCHEMA (llm_attacker.protocol_frames.v1)
=================================================================
A wire_shape.v1-style MITM journal extended with frame fields. One JSON
object per line, exactly these keys:

    {"ts": 1712345678.123, "bytes": 4096,          # wire_shape.v1 fields:
                                                   # ts = observation time,
                                                   # bytes = payload bytes
                                                   # ACTUALLY delivered
     "frame_id": 0, "session_id": "s0", "epoch": 0, "seq": 3,
     "dir": "c2s",            # c2s | s2c
     "phase": "prefill",      # prefill/decode (inference), fwd/bwd (training)
     "dtype": "float16", "shape": [2, 64],
     "declared_bytes": 256,   # the authenticated length prefix
     "payload_hash": "ab12..",  # hex; unkeyed hash of the payload content
     "auth_tag": "cd34..",    # hex; HMAC-style tag over the frame fields
                              # with the per-session, per-epoch endpoint key
     "validate_mode": "strict",  # negotiated validation mode; PRE-AUTH like
                              # a TLS ClientHello, so NOT covered by auth_tag
     "value_marks": []        # what a payload scanner would find in the
                              # delivered bytes: subset of {"nan", "inf",
                              # "extreme"}; stands in for the payload itself,
                              # which never enters a journal (hashes only)
    }

Schema drift (missing/extra keys, bad types, non-hex hashes) is a
descriptive ValueError naming the file and line — never a silent skip. The
schema is registered as the "protocol-frames" SurfaceProvider
(attacker.surfaces) at module import.

DEFENSES (the defended world enables all; the vulnerable world none):
  auth                verify auth_tag with the session/epoch key (unknown
                      sessions cannot authenticate)
  session_binding     the frame's session must match the connection's
  monotonic_seq       seq must strictly increase within the current epoch
                      (seq resets at epoch rollover, as keys rotate)
  epoch_check         epoch must not roll back
  size_cap            declared/delivered bytes above the negotiated cap fail
                      closed BEFORE allocation (checked first)
  validate_dtype_shape  framing validation: known dtype, positive shape,
                      declared == dtype itemsize * prod(shape), and
                      delivered == declared (truncation)
  validate_values     payload scan: NaN/Inf/extreme markers fail closed
Checks run in a fixed order (cheap policy/size gates before crypto) and
rejection reasons come from a small constant set (REJECTION_REASONS) that
carries no secret-dependent detail.

ARMS (mutators over a captured stream; same seeded mutation in both worlds):
  replay              resend a valid frame later in the same session/epoch —
                      the genuine tag passes auth; only monotonic_seq catches
  reorder             rotate a window of frames out of order
  duplicate           inject an immediate copy of a frame
  drop                omit a frame (availability attack; see limitations)
  cross-session-swap  inject a genuine frame from another tenant/session
                      (falls back to a fabricated foreign session when the
                      journal has only one)
  epoch-desync        re-inject a genuine pre-rollover frame after the epoch
                      advanced (rollback), or flip the epoch field
  downgrade           rewrite the pre-auth validate_mode to a weaker mode
  tensor-tamper       variants: perturb (scan-invisible content change with
                      re-hashed unkeyed payload_hash), extreme values,
                      NaN/Inf markers, invalid dtype, invalid shape
  oversize            layout-consistent frame beyond the negotiated cap
  truncate            deliver fewer bytes than the authenticated prefix

WORLDS: "vulnerable" (all defenses off — accepts every mutation; the
attack-works baseline that keeps the controls honest) and "defended" (all
on). --world both (default) runs the pair; a negative-control cell (clean
journal through the DEFENDED endpoint) is always added and must show zero
false rejections, while a benign control session runs concurrently in every
cell and its rejection rate must stay 0.

METRICS per cell: undetected mutation rate, detection counts, mean frames
to detection (mttd), fail-open rate (defended), crash/hang rate (0.0 — the
simulator models accept/reject only), benign-session rejection rate,
cross-session acceptance rate (swap arm), false rejection rate (control).

LIMITATIONS / EXCLUDED VARIANTS:
  - crash/hang/retry-storm are reported as 0.0 placeholders, not measured:
    the simulator models accept/reject decisions only.
  - drop is undetected BY CONSTRUCTION in both worlds: monotonic_seq
    tolerates gaps, and liveness/gap tracking is an excluded variant.
  - two bounding worlds only; per-defense ablation grids are excluded (in
    the full defended profile the auth envelope catches most field flips
    before the inner validators, which are defense-in-depth).
  - payloads are never modeled: tamper is represented by scanner marks and
    a re-hashed unkeyed payload_hash; the keyed auth_tag stays unforgeable.
  - no live injector ships here: the real-target path consumes a MITM
    journal (--journal) plus endpoint keys (--session-keys); driving a live
    relay is the integration point named in the --session-keys error.

Usage:
    python -m attacker --mode inference --attack protocol-integrity --help
    python -m attacker --mode training --attack protocol-integrity \
        --toy --quick --output /tmp/pi.json
    python -m attacker --mode inference --attack protocol-integrity \
        --journal captures/frames.jsonl --session-keys captures/keys.json \
        --output pi.json
"""

import argparse
import hashlib
import hmac
import json
import math
import os
import random
import statistics

from .. import artifacts, threat_model
from ..surfaces import SurfaceProvider, register_surface
from .common import add_common_args, journal_error

EXPERIMENT_ID = "protocol_integrity"
MODES = ("training", "inference")
REQUIRES_LABELS = False
DESCRIPTION = ("active inline/relay attacks on the split protocol itself "
               "(replay, reorder, swap, epoch-desync, downgrade, tamper, "
               "oversize, truncate): integrity/availability of the frame "
               "stream, not tensor confidentiality")

SCHEMA_VERSION = "llm_attacker.protocol_frames.v1"
ARTIFACT_SCHEMA = "dtraining.attacker.protocol_integrity.v1"
DIRS = ("c2s", "s2c")
DTYPES = {"float16": 2, "bfloat16": 2, "float32": 4, "int8": 1, "int32": 4}
AUTH_FIELDS = ("frame_id", "session_id", "epoch", "seq", "dir", "phase",
               "dtype", "shape", "declared_bytes", "payload_hash")
ARMS = ("replay", "reorder", "duplicate", "drop", "cross-session-swap",
        "epoch-desync", "downgrade", "tensor-tamper", "oversize", "truncate")
TAMPER_VARIANTS = ("perturb", "extreme", "naninf", "dtype", "shape")
WORLDS = ("vulnerable", "defended")
NEGATIVE_CONTROL = "negative-control"
REJECTION_REASONS = frozenset({
    "auth-failure", "session-mismatch", "seq-not-monotonic", "epoch-rollback",
    "downgrade-refused", "size-cap-exceeded", "dtype-shape-invalid",
    "truncated", "value-invalid"})


def content_hash(session_id, epoch, seq, declared_bytes, marks, note=""):
    """Unkeyed hash of the (never journaled) payload content. The attacker
    can recompute it over tampered content — integrity rests on the keyed
    auth_tag that commits to payload_hash."""
    blob = json.dumps([session_id, epoch, seq, declared_bytes, sorted(marks),
                       note], separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def compute_auth_tag(key_hex, frame):
    """HMAC-SHA256 over the canonical AUTH_FIELDS of the frame."""
    msg = json.dumps({k: frame[k] for k in AUTH_FIELDS},
                     sort_keys=True, separators=(",", ":"))
    return hmac.new(bytes.fromhex(key_hex), msg.encode(),
                    hashlib.sha256).hexdigest()


def _session_key(seed, session_id, epoch):
    """Deterministic per-session, per-epoch toy key (in memory only)."""
    return hashlib.sha256(
        f"protocol-integrity/session-key:{seed}:{session_id}:{epoch}"
        .encode()).hexdigest()


def _is_hex(s):
    return isinstance(s, str) and bool(s) and all(
        c in "0123456789abcdef" for c in s.lower())


class ProtocolFramesProvider(SurfaceProvider):
    """The protocol-frames journal as an observation surface: records of
    {ts, bytes, frame_id, session_id, epoch, seq, dir, phase, dtype, shape,
    declared_bytes, payload_hash, auth_tag, validate_mode, value_marks}
    (schema llm_attacker.protocol_frames.v1). rows(mode, source=<journal
    path>) yields ([ts, bytes], metadata) pairs with metadata the full
    normalized record. Pure python, torch-free."""

    name = "protocol-frames"
    schema = {"ts": float, "bytes": int, "frame_id": int, "session_id": str,
              "epoch": int, "seq": int, "dir": str, "phase": str,
              "dtype": str, "shape": list, "declared_bytes": int,
              "payload_hash": str, "auth_tag": str, "validate_mode": str,
              "value_marks": list}
    modes = ("training", "inference")

    def validate(self, meta, path="<record>"):
        """Key-set validation plus per-field type checks. Note that dtype,
        shape, declared_bytes and tags are checked for TYPE only: invalid
        values are attack input and belong to the endpoint, not the parser.
        ValueError naming `path` on drift."""
        keys = set(meta)
        if keys != set(self.schema):
            raise ValueError(
                f"{path}: record keys {sorted(keys)} != surface "
                f"'protocol-frames' schema {sorted(self.schema)} "
                f"({SCHEMA_VERSION})")

        def bad(field, want):
            raise ValueError(
                f"{path}: {field} must be {want}, got {meta[field]!r}")

        ts = meta["ts"]
        if isinstance(ts, bool) or not isinstance(ts, (int, float)):
            bad("ts", "a number")
        for f in ("bytes", "frame_id", "epoch", "seq", "declared_bytes"):
            v = meta[f]
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                bad(f, "an int >= 0")
        for f in ("session_id", "phase", "dtype", "validate_mode"):
            if not isinstance(meta[f], str) or not meta[f]:
                bad(f, "a non-empty str")
        if meta["dir"] not in DIRS:
            bad("dir", f"one of {DIRS}")
        shape = meta["shape"]
        if not isinstance(shape, list) or any(
                isinstance(x, bool) or not isinstance(x, int) for x in shape):
            bad("shape", "a list of int")
        for f in ("payload_hash", "auth_tag"):
            if not _is_hex(meta[f]):
                bad(f, "a non-empty hex str")
        marks = meta["value_marks"]
        if not isinstance(marks, list) \
                or not all(isinstance(m, str) for m in marks):
            bad("value_marks", "a list of str")
        return {"ts": float(ts), "bytes": meta["bytes"],
                "frame_id": meta["frame_id"], "session_id": meta["session_id"],
                "epoch": meta["epoch"], "seq": meta["seq"], "dir": meta["dir"],
                "phase": meta["phase"], "dtype": meta["dtype"],
                "shape": list(shape), "declared_bytes": meta["declared_bytes"],
                "payload_hash": meta["payload_hash"],
                "auth_tag": meta["auth_tag"],
                "validate_mode": meta["validate_mode"],
                "value_marks": list(marks)}

    def rows(self, mode=None, source=None):
        self.check_mode(mode)
        if source is None:
            raise ValueError("surface 'protocol-frames' needs source=<frame "
                             f"journal path ({SCHEMA_VERSION})")

        def gen():
            with open(source) as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"{source}:{lineno}: invalid JSON "
                                         f"({e})") from None
                    meta = self.validate(rec, f"{source}:{lineno}")
                    yield [meta["ts"], meta["bytes"]], meta
        return gen()


register_surface(ProtocolFramesProvider())


def read_journal(path):
    """Parse a protocol-frames journal -> [records sorted by ts]. Records
    are validated through the ProtocolFramesProvider (schema drift is a
    descriptive ValueError naming file and line)."""
    if not os.path.isfile(path):
        raise ValueError(f"{path}: cannot read protocol-frames journal "
                         "(no such file)")
    provider = ProtocolFramesProvider()
    records = [meta for _, meta in provider.rows(source=path)]
    if not records:
        raise ValueError(f"{path}: empty protocol-frames journal")
    records.sort(key=lambda r: r["ts"])
    return records


def make_journal(seed, n_sessions, frames_per_session, mode="inference",
                 hidden=64):
    """A valid interleaved frame stream for n_sessions concurrent sessions.
    Each session sends frames_per_session frames (alternating c2s/s2c
    phases), with an epoch rollover — and key rotation + seq reset — at the
    midpoint so rollback arms have genuine old-epoch frames to replay.
    Returns (records, keys) with keys = {session_id: {epoch_str: hex}};
    keys stay in memory and are never journaled."""
    rng = random.Random(seed)
    sids = [f"s{si}" for si in range(n_sessions)]
    keys = {sid: {str(e): _session_key(seed, sid, e) for e in (0, 1)}
            for sid in sids}
    records = []
    t = 1_700_000_000.0
    for f in range(frames_per_session):
        epoch = 0 if f < frames_per_session // 2 else 1
        seq = f if epoch == 0 else f - frames_per_session // 2
        direction = "c2s" if f % 2 == 0 else "s2c"
        if mode == "training":
            phase = "fwd" if direction == "c2s" else "bwd"
        else:
            phase = "prefill" if direction == "c2s" else "decode"
        for sid in sids:
            shape = [rng.randint(1, 4), hidden]
            declared = DTYPES["float16"] * math.prod(shape)
            frame = {"frame_id": -1,
                     "session_id": sid, "epoch": epoch, "seq": seq,
                     "dir": direction, "phase": phase, "dtype": "float16",
                     "shape": shape, "declared_bytes": declared,
                     "payload_hash": content_hash(sid, epoch, seq, declared,
                                                  []),
                     "validate_mode": "strict"}
            frame["auth_tag"] = compute_auth_tag(keys[sid][str(epoch)], frame)
            t += rng.uniform(0.004, 0.02)
            records.append({"ts": round(t, 6), "bytes": declared,
                            "value_marks": [], **frame})
    records.sort(key=lambda r: r["ts"])
    for i, r in enumerate(records):
        r["frame_id"] = i
        r["auth_tag"] = compute_auth_tag(keys[r["session_id"]][str(r["epoch"])],
                                         r)
    return records, keys


def defense_profile(world, size_cap):
    """The two bounding worlds. defended: every check on, cap enforced.
    vulnerable: everything off — accepts every frame."""
    if world == "defended":
        return {"auth": True, "session_binding": True, "monotonic_seq": True,
                "epoch_check": True, "size_cap": size_cap,
                "validate_dtype_shape": True, "validate_values": True}
    if world == "vulnerable":
        return {"auth": False, "session_binding": False,
                "monotonic_seq": False, "epoch_check": False,
                "size_cap": None, "validate_dtype_shape": False,
                "validate_values": False}
    raise ValueError(f"unknown world {world!r} (expected one of {WORLDS})")


class ProtocolEndpoint:
    """Simulated split-protocol endpoint with a configurable defense profile
    {"auth", "session_binding", "monotonic_seq", "epoch_check": bool,
     "size_cap": int|None, "validate_dtype_shape", "validate_values": bool}.

    accept(frame, ctx) -> (accepted, reason): ctx carries the connection's
    session_id; reason is "ok" or one of REJECTION_REASONS (a small constant
    set — rejections must not leak secret-dependent detail). Rejected frames
    never mutate per-session state, so an attack on one session cannot
    disturb another. Checks run cheap-policy-first: size cap and the
    negotiation floor precede authentication (fail closed before allocation
    or crypto)."""

    def __init__(self, defenses, session_keys, policy_mode="strict"):
        self.defenses = dict(defenses)
        self.policy_mode = policy_mode
        self.sessions = {sid: {"keys": dict(ek), "last_seq": -1, "epoch": 0}
                         for sid, ek in session_keys.items()}

    def accept(self, frame, ctx):
        d = self.defenses
        sid = frame["session_id"]
        nbytes = frame.get("bytes", frame["declared_bytes"])
        if d["size_cap"] is not None and (frame["declared_bytes"]
                                          > d["size_cap"]
                                          or nbytes > d["size_cap"]):
            return False, "size-cap-exceeded"
        if d["auth"] and frame["validate_mode"] != self.policy_mode:
            return False, "downgrade-refused"
        sess = self.sessions.get(sid)
        if d["auth"]:
            if sess is None:
                return False, "auth-failure"
            key = sess["keys"].get(str(frame["epoch"]))
            if key is None or not hmac.compare_digest(
                    compute_auth_tag(key, frame), str(frame["auth_tag"])):
                return False, "auth-failure"
        if d["session_binding"] and (sess is None
                                     or sid != ctx.get("session_id")):
            return False, "session-mismatch"
        if sess is None:
            sess = self.sessions.setdefault(
                sid, {"keys": {}, "last_seq": -1, "epoch": 0})
        if d["epoch_check"] and frame["epoch"] < sess["epoch"]:
            return False, "epoch-rollback"
        if d["monotonic_seq"] and frame["epoch"] == sess["epoch"] \
                and frame["seq"] <= sess["last_seq"]:
            return False, "seq-not-monotonic"
        if d["validate_dtype_shape"]:
            itemsize = DTYPES.get(frame["dtype"])
            shape = frame["shape"]
            if itemsize is None or not shape or any(
                    not isinstance(x, int) or x <= 0 for x in shape):
                return False, "dtype-shape-invalid"
            if itemsize * math.prod(shape) != frame["declared_bytes"]:
                return False, "dtype-shape-invalid"
            if nbytes != frame["declared_bytes"]:
                return False, "truncated"
        if d["validate_values"] and frame["value_marks"]:
            return False, "value-invalid"
        if frame["epoch"] > sess["epoch"]:
            sess["epoch"] = frame["epoch"]
            sess["last_seq"] = -1
        sess["last_seq"] = max(sess["last_seq"], frame["seq"])
        return True, "ok"


def _copy(r):
    return {**r, "shape": list(r["shape"]), "value_marks": list(r["value_marks"])}


def _mut_replay(V, donor, rng, size_cap, hidden):
    """Resend a valid frame later in the SAME session/epoch: the genuine tag
    passes auth; only monotonic_seq can catch it."""
    half = len(V) // 2
    j = rng.randrange(max(1, half - 3))
    at = min(j + 3, half - 1)
    dup = _copy(V[j])
    dup["_pi_mark"] = "mutated"
    return V[:at] + [dup] + V[at:], 1, 0


def _mut_reorder(V, donor, rng, size_cap, hidden):
    """Rotate a window of same-epoch frames one step: the leading (highest-
    seq) frame is legitimately accepted (gaps allowed), the rest arrive out
    of order."""
    half = len(V) // 2
    w = min(4, half)
    a = rng.randrange(half - w + 1) if half > w else 0
    window = V[a:a + w]
    for r in window:
        r["_pi_mark"] = "mutated"
    return V[:a] + [window[-1]] + window[:-1] + V[a + w:], w, 0


def _mut_duplicate(V, donor, rng, size_cap, hidden):
    """Inject an immediate copy of a frame (same seq arrives twice)."""
    j = rng.randrange(len(V))
    dup = _copy(V[j])
    dup["_pi_mark"] = "mutated"
    return V[:j + 1] + [dup] + V[j + 1:], 1, 0


def _mut_drop(V, donor, rng, size_cap, hidden):
    """Omit a frame. Nothing arrives to reject: undetected by construction
    (liveness/gap tracking is an excluded variant)."""
    j = rng.randrange(len(V) // 2)
    return V[:j] + V[j + 1:], 1, 0


def _mut_cross_session_swap(V, donor, rng, size_cap, hidden):
    """Substitute a genuine frame captured from another tenant/session; with
    a single-session journal, fabricate a foreign session instead."""
    at = max(1, len(V) // 4)
    if donor:
        fr = _copy(donor[rng.randrange(len(donor))])
    else:
        fr = _copy(V[rng.randrange(len(V))])
        fr["session_id"] = "foreign-session"
        fr["auth_tag"] = hashlib.sha256(b"forged-tag").hexdigest()
    fr["_pi_mark"] = "foreign"
    return V[:at] + [fr] + V[at:], 1, 1


def _mut_epoch_desync(V, donor, rng, size_cap, hidden):
    """Re-inject a genuine pre-rollover frame after the epoch advanced (its
    tag is valid under the old epoch key; only the epoch check catches the
    rollback). Short streams fall back to flipping the epoch field."""
    half = len(V) // 2
    if half >= 2:
        old = _copy(V[rng.randrange(half)])
        old["_pi_mark"] = "mutated"
        at = min(half + 1, len(V))
        return V[:at] + [old] + V[at:], 1, 0
    fr = _copy(V[-1])
    fr["epoch"] = max(0, fr["epoch"] - 1)
    fr["_pi_mark"] = "mutated"
    return V + [fr], 1, 0


def _mut_downgrade(V, donor, rng, size_cap, hidden):
    """Rewrite the pre-auth validate_mode of a run of frames to a weaker
    mode — the tag still verifies (the field is outside the envelope), so
    only the endpoint's negotiation floor refuses it."""
    a = max(0, len(V) // 4 - 1)
    n = min(3, len(V) - a)
    for i in range(a, a + n):
        V[i]["validate_mode"] = "legacy"
        V[i]["_pi_mark"] = "mutated"
    return list(V), n, 0


def _tamper_frame(r, variant):
    """One tensor-tamper variant applied to a frame copy in place. Content
    changes re-hash the unkeyed payload_hash (the attacker knows the
    algorithm) but cannot fix the keyed auth_tag."""
    if variant == "perturb":
        r["payload_hash"] = content_hash(r["session_id"], r["epoch"],
                                         r["seq"], r["declared_bytes"],
                                         r["value_marks"], note="perturbed")
    elif variant == "extreme":
        r["value_marks"] = ["extreme"]
        r["payload_hash"] = content_hash(r["session_id"], r["epoch"],
                                         r["seq"], r["declared_bytes"],
                                         ["extreme"])
    elif variant == "naninf":
        r["value_marks"] = ["nan"]
        r["payload_hash"] = content_hash(r["session_id"], r["epoch"],
                                         r["seq"], r["declared_bytes"],
                                         ["nan"])
    elif variant == "dtype":
        r["dtype"] = "float9"
    elif variant == "shape":
        r["shape"] = [0] + list(r["shape"][1:])
    else:
        raise ValueError(f"unknown tensor-tamper variant {variant!r}")


def _mut_tensor_tamper(V, donor, rng, size_cap, hidden):
    variant = TAMPER_VARIANTS[rng.randrange(len(TAMPER_VARIANTS))]
    half = len(V) // 2
    a = rng.randrange(max(1, half - 1))
    n = min(2, len(V) - a)
    for i in range(a, a + n):
        _tamper_frame(V[i], variant)
        V[i]["_pi_mark"] = "mutated"
    return list(V), n, 0


def _mut_oversize(V, donor, rng, size_cap, hidden):
    """Inflate a frame past the negotiated cap, keeping dtype/shape/declared
    internally consistent — only the size cap catches it (pre-allocation)."""
    cap = size_cap or (1 << 20)
    j = rng.randrange(len(V))
    r = _copy(V[j])
    itemsize = DTYPES[r["dtype"]]
    rows = cap // (itemsize * r["shape"][-1]) + 1
    r["shape"] = [rows, r["shape"][-1]]
    r["declared_bytes"] = itemsize * math.prod(r["shape"])
    r["bytes"] = r["declared_bytes"]
    r["payload_hash"] = content_hash(r["session_id"], r["epoch"], r["seq"],
                                     r["declared_bytes"], [])
    r["_pi_mark"] = "mutated"
    return V[:j] + [r] + V[j + 1:], 1, 0


def _mut_truncate(V, donor, rng, size_cap, hidden):
    """Deliver fewer bytes than the authenticated length prefix declares
    (the prefix itself is untouched, so auth passes; framing catches it)."""
    k = min(2, len(V))
    for i in rng.sample(range(len(V)), k=k):
        V[i]["bytes"] = V[i]["declared_bytes"] // 2
        V[i]["_pi_mark"] = "mutated"
    return list(V), k, 0


MUTATORS = {"replay": _mut_replay, "reorder": _mut_reorder,
            "duplicate": _mut_duplicate, "drop": _mut_drop,
            "cross-session-swap": _mut_cross_session_swap,
            "epoch-desync": _mut_epoch_desync, "downgrade": _mut_downgrade,
            "tensor-tamper": _mut_tensor_tamper, "oversize": _mut_oversize,
            "truncate": _mut_truncate}


def _retime(stream, original_ts):
    """Make list order the arrival order without disturbing the global
    interleave: reuse the victim session's original (sorted) timestamps,
    extending past the end with the median gap when the stream grew."""
    original_ts = sorted(original_ts)
    gap = statistics.median([b - a for a, b in
                             zip(original_ts, original_ts[1:])]) \
        if len(original_ts) > 1 else 0.01
    for i, r in enumerate(stream):
        r["ts"] = (original_ts[i] if i < len(original_ts)
                   else round(original_ts[-1]
                              + (i - len(original_ts) + 1) * gap, 6))


def simulate(streams, keys, profile):
    """Feed the per-connection streams (merged in ts order) through a fresh
    ProtocolEndpoint. Returns [(record, accepted, reason)] in arrival
    order."""
    endpoint = ProtocolEndpoint(profile, keys)
    merged = [r for rs in streams.values() for r in rs]
    merged.sort(key=lambda r: r["ts"])
    return [(r, *endpoint.accept(r, {"session_id": r["_pi_conn"]}))
            for r in merged]


def run_cell(arm, world, records, keys, args, seed):
    """One cell: mutate the victim session's stream with `arm` (or not, for
    the negative control), simulate the endpoint in `world`, and reduce to
    metrics. The benign control session always runs concurrently and
    unmutated. Deterministic: the mutator rng derives from args.seed/seed/
    arm only, so both worlds see the same mutation."""
    rng = random.Random(args.seed + 1000 * seed + 17 * (
        ARMS.index(arm) + 1 if arm in ARMS else len(ARMS) + 1))
    by_session = {}
    for r in records:
        c = _copy(r)
        c["_pi_conn"] = r["session_id"]
        c["_pi_mark"] = None
        by_session.setdefault(r["session_id"], []).append(c)
    for rs in by_session.values():
        rs.sort(key=lambda r: r["ts"])
    sessions = sorted(by_session)
    control = sessions[-1]
    victim = sessions[seed % max(1, len(sessions) - 1)]
    donor = next((s for s in sessions if s not in (victim, control)), None)
    if donor is None and control != victim:
        donor = control
    profile = defense_profile(world, args.size_cap)
    if arm == NEGATIVE_CONTROL:
        mutated, n_events, n_foreign = list(by_session[victim]), 0, 0
    else:
        mutated, n_events, n_foreign = MUTATORS[arm](
            by_session[victim], by_session.get(donor) or [], rng,
            args.size_cap, args.hidden)
    for r in mutated:
        r["_pi_conn"] = victim
    _retime(mutated, [r["ts"] for r in by_session[victim]])
    sim = simulate({**{s: rs for s, rs in by_session.items() if s != victim},
                    victim: mutated}, keys, profile)

    vic = [(r, ok, why) for r, ok, why in sim if r["_pi_conn"] == victim]
    marked = [(i, r, ok, why) for i, (r, ok, why) in enumerate(vic)
              if r["_pi_mark"]]
    n_detected = sum(1 for _, _, ok, _ in marked if not ok)
    if marked:
        undetected = (len(marked) - n_detected) / len(marked)
    elif n_events:
        undetected = 1.0
    else:
        undetected = None
    rej_pos = [i for i, _, ok, _ in marked if not ok]
    benign = [(r, ok) for r, ok, _ in sim if r["_pi_conn"] == control]
    n_benign_rej = sum(1 for _, ok in benign if not ok)
    rec = {"experiment": EXPERIMENT_ID, "arm": arm, "world": world,
           "seed": seed, "victim_session": victim,
           "control_session": control, "n_frames": len(sim),
           "n_events": n_events, "n_detected": n_detected,
           "n_undetected": (len(marked) - n_detected) if marked
           else (n_events if n_events else 0),
           "undetected_mutation_rate": undetected,
           "mttd_frames": (min(rej_pos)
                           - min(i for i, _, _, _ in marked) + 1)
           if rej_pos else None,
           "fail_open_rate": undetected if world == "defended" else None,
           "crash_hang_rate": 0.0,
           "n_benign": len(benign), "n_benign_rejected": n_benign_rej,
           "benign_rejection_rate": (n_benign_rej / len(benign))
           if benign else None,
           "rejection_reasons": sorted({why for _, _, ok, why in marked
                                        if not ok})}
    if n_foreign:
        foreign = [(r, ok) for _, r, ok, _ in marked
                   if r["_pi_mark"] == "foreign"]
        rec["n_foreign"] = len(foreign)
        rec["cross_session_acceptance_rate"] = (
            sum(1 for _, ok in foreign if ok) / len(foreign))
    if arm == NEGATIVE_CONTROL:
        n_rej = sum(1 for _, ok, _ in sim if not ok)
        rec["n_rejected"] = n_rej
        rec["false_rejection_rate"] = n_rej / len(sim) if sim else None
    return rec


def _make_evidence(args):
    """An EvidenceLogger when --evidence and --output are set (None
    otherwise, with the repo-convention warning)."""
    if not getattr(args, "evidence", False):
        return None
    if not args.output:
        print(f"[{EXPERIMENT_ID}] --evidence needs --output; continuing "
              "without an evidence chain")
        return None
    from ..evidence import EvidenceLogger
    return EvidenceLogger(args.output, node_role=f"attacker/{EXPERIMENT_ID}")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--evidence", action="store_true",
                    help="write a tamper-evident evidence chain alongside "
                         "the artifact (attacker.evidence)")
    ap.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS),
                    help="mutation arm(s) to run (default: all)")
    ap.add_argument("--world", choices=("both",) + WORLDS, default="both",
                    help="endpoint world(s) to simulate (default: both — "
                         "vulnerable is the attack-works baseline, defended "
                         "the expected-secure world; the negative control "
                         "always runs defended)")
    ap.add_argument("--journal", default=None,
                    help=f"MITM protocol-frames journal to analyze "
                         f"({SCHEMA_VERSION})")
    ap.add_argument("--session-keys", default=None,
                    help="journal mode: JSON {session_id: {epoch: key hex}} "
                         "so the defended world can verify auth tags. Used "
                         "in memory only, never written into the artifact")
    ap.add_argument("--n-sessions", type=int, default=4,
                    help="toy: concurrent sessions (the last is the benign "
                         "control)")
    ap.add_argument("--frames-per-session", type=int, default=24,
                    help="toy: frames per session (epoch rollover + key "
                         "rotation at the midpoint)")
    ap.add_argument("--size-cap", type=int, default=1 << 20,
                    help="negotiated per-frame byte cap the defended world "
                         "enforces")
    return ap


def _threat_model_doc(args):
    return threat_model.skeleton(
        summary="active inline network attacker / malicious relay / "
                "compromised proxy on the split client<->server frame path: "
                "observes and mutates protocol frames (replay, reorder, "
                "duplicate, drop, cross-session swap, epoch desync, "
                "downgrade, tensor tamper, oversize, truncate) without "
                "possessing endpoint keys. Tests the integrity and "
                "availability of the split protocol itself, not the "
                "confidentiality of the tensors crossing it.",
        scope={"use_case": "split-training", "deployment": "multi-tenant",
               "in_scope": ["split-training activation/gradient frame "
                            "streams (epochs, key rotation)",
                            "split-inference prefill/decode frame streams",
                            "the bidirectional client<->server protocol: "
                            "framing, negotiation, session/epoch state"],
               "out_of_scope": ["payload confidentiality (seam-inversion "
                                "family)",
                                "endpoint key theft / insiders holding keys",
                                "crash/hang modeling (the simulator reports "
                                "accept/reject only)",
                                "drop detection via liveness/gap tracking"]},
        actor={"role": "network", "intent": "active",
               "persistence": "session",
               "vantage": ["inline on the frame path",
                           "malicious relay or compromised proxy"],
               "authorization": "none: no endpoint keys, no tenant "
                                "credentials",
               "privileges": [],
               "knowledge": ["the framing schema " + SCHEMA_VERSION,
                             "per-frame metadata incl. the unkeyed payload "
                             "hashes"],
               "capabilities": ["observe frames in both directions",
                                "modify, inject, reorder, duplicate and drop "
                                "frames",
                                "capture frames across sessions/epochs for "
                                "replay"],
               "resources": ["a captured frame journal or a synthetic "
                             "stream"],
               "collusion": []},
        target={"assets": ["protocol integrity", "session/epoch state "
                           "consistency", "endpoint availability for benign "
                           "tenants", "tensor payload integrity"],
                "components": ["split client endpoint",
                               "split server endpoint",
                               "relay/proxy segment"],
                "data_stores": [],
                "data_flows": ["client->server activation/prefill frames",
                               "server->client gradient/decode frames"],
                "trust_boundaries": ["the network link between split "
                                     "endpoints (S10 protocol/control-plane "
                                     "metadata; S7 traffic shape)"],
                "entry_points": ["frames traversing the relay"],
                "exit_points": ["endpoint frame intake"]},
        scenario={"preconditions": ["attacker is inline on the frame path",
                                    "endpoint keys are uncompromised",
                                    "sessions negotiate validate_mode=strict"
                                    ],
                  "lifecycle_stage": ["deployment", "operations"],
                  "security_properties": ["integrity", "availability",
                                          "authenticity"],
                  "attack_steps": list(ARMS),
                  "technical_impact": ["state divergence via accepted "
                                       "replays/reorderings",
                                       "cross-tenant frame injection",
                                       "availability loss via drop/oversize/"
                                       "truncation"],
                  "business_impact": ["corrupted training runs or served "
                                      "outputs",
                                      "tenant-isolation and availability "
                                      "findings"]},
        controls={"assumed": ["endpoints hold per-session per-epoch keys",
                              "no endpoint compromise"],
                  "tested": ["frame authentication (keyed auth_tag)",
                             "session binding", "monotonic sequence numbers",
                             "epoch rollback checks", "negotiated size caps",
                             "dtype/shape/framing validation",
                             "payload value scanning (NaN/Inf/extreme)",
                             "negotiation downgrade policy floor"],
                  "bypassed": [],
                  "detective": ["rejection reasons drawn from a small "
                                "constant set (no secret-dependent detail)"],
                  "response": ["fail-closed rejection before allocation or "
                               "model execution"],
                  "residual_risk": "", "owner": "", "status": "open"},
        mappings={"owasp_llm_2026": ["LLM02", "LLM05", "LLM06"],
                  "owasp_agentic_2026": [], "mitre_atlas": [],
                  "nist_ai_100_2": [],
                  "cwe": ["CWE-294", "CWE-345"]},
        evaluation={"success_metrics": ["undetected mutation rate per arm",
                                        "cross-session acceptance rate",
                                        "mean frames to detection",
                                        "benign-session rejection rate "
                                        "(must stay 0)",
                                        "false rejection rate on clean "
                                        "traffic (must be 0)"],
                    "baseline": "the vulnerable world (all defenses off) "
                                "accepts every mutation; the defended world "
                                "rejects every mutated frame class except "
                                "drop (no liveness model)",
                    "budget": {"toy_sessions": args.n_sessions,
                               "toy_frames_per_session":
                               args.frames_per_session,
                               "seeds": list(args.seeds)},
                    "limitations": ["simulator models accept/reject only — "
                                    "crash/hang/retry-storm reported as 0.0",
                                    "drop undetected by construction (no "
                                    "liveness/gap tracking)",
                                    "two bounding worlds; per-defense "
                                    "ablation grids excluded",
                                    "payloads never modeled (scanner marks "
                                    "+ unkeyed hashes only)"],
                    "coverage_status": "covered"})


def _make_artifact(args, source, extra_config=None):
    config = {"attack": EXPERIMENT_ID, "mode": getattr(args, "mode", None),
              "source": source, "arms": args.arms, "world": args.world,
              "journal_schema": SCHEMA_VERSION, "seed": args.seed,
              "size_cap": args.size_cap,
              "defended_profile": defense_profile("defended", args.size_cap),
              "vulnerable_profile": defense_profile("vulnerable",
                                                    args.size_cap)}
    if extra_config:
        config.update(extra_config)
    return artifacts.make_artifact(
        ARTIFACT_SCHEMA, config, _threat_model_doc(args),
        interpretation="Expected secure behavior: frame authentication + "
        "session binding reject modified or foreign frames ('auth-failure'/"
        "'session-mismatch'); monotonic sequence + epoch checks reject "
        "replay and rollback; invalid sizes/shapes/dtypes/values and "
        "truncated deliveries fail closed BEFORE allocation or model "
        "execution; downgrade requests below the negotiated validation "
        "floor are refused; rejection reasons come from the small constant "
        "REJECTION_REASONS set and leak no secret-dependent detail; the "
        "endpoint stays available to unaffected sessions (benign control "
        "rejection rate must stay 0). The vulnerable world (all defenses "
        "off) must accept every mutation — the attack-works baseline that "
        "keeps the controls non-vacuous — and the negative control (clean "
        "journal through the defended endpoint) must show 0 false "
        "rejections. drop is undetected in BOTH worlds by construction: "
        "detection needs liveness/gap tracking, an excluded variant.")


def _summary_row(arm, world, recs):
    row = {"experiment": EXPERIMENT_ID, "arm": arm, "world": world,
           "n_seeds": len(recs)}
    rates = [r["undetected_mutation_rate"] for r in recs
             if r["undetected_mutation_rate"] is not None]
    if rates:
        row["undetected_mutation_rate_mean"] = statistics.fmean(rates)
    mttds = [r["mttd_frames"] for r in recs if r["mttd_frames"] is not None]
    if mttds:
        row["mttd_frames_mean"] = statistics.fmean(mttds)
    brates = [r["benign_rejection_rate"] for r in recs
              if r["benign_rejection_rate"] is not None]
    if brates:
        row["benign_rejection_rate_mean"] = statistics.fmean(brates)
    if arm == NEGATIVE_CONTROL:
        row["false_rejection_rate_mean"] = statistics.fmean(
            r["false_rejection_rate"] for r in recs)
    return row


def _progress(rec):
    rate = rec["undetected_mutation_rate"]
    mttd = rec["mttd_frames"]
    line = (f"[{EXPERIMENT_ID}] {rec['arm']} world={rec['world']} "
            f"seed={rec['seed']}: "
            f"undetected={f'{rate:.3f}' if rate is not None else '-'} "
            f"detected={rec['n_detected']}/{rec['n_events']} "
            f"mttd={mttd if mttd is not None else '-'} "
            f"benign_rej={rec['benign_rejection_rate']:.3f}")
    if rec["arm"] == NEGATIVE_CONTROL:
        line += f" false_rej={rec['false_rejection_rate']:.3f}"
    if "cross_session_acceptance_rate" in rec:
        line += f" xsess_accept={rec['cross_session_acceptance_rate']:.3f}"
    return line


def _run_worlds(args, out, seeds, records_of):
    """Shared cell loop: for each seed, every (arm, world) cell plus the
    mandatory defended negative control; failed cells are journaled with an
    error field, never raised."""
    worlds = list(WORLDS) if args.world == "both" else [args.world]
    per_key = {}
    for seed in seeds:
        records, keys = records_of(seed)
        cells = ([(arm, w) for w in worlds for arm in args.arms]
                 + [(NEGATIVE_CONTROL, "defended")])
        for arm, w in cells:
            try:
                rec = run_cell(arm, w, records, keys, args, seed)
            except Exception as e:
                journal_error(args.output, EXPERIMENT_ID,
                              {"arm": arm, "world": w, "seed": seed}, e)
                continue
            artifacts.append_jsonl(args.output, rec)
            out["results"].append(rec)
            per_key.setdefault((arm, w), []).append(rec)
            print(_progress(rec))
    for (arm, w), recs in sorted(per_key.items()):
        out["summary"].append(_summary_row(arm, w, recs))
    artifacts.write_artifact(args.output, out,
                         evidence=_make_evidence(args))
    return 0


def run_toy(args):
    if args.quick:
        args.seeds = args.seeds[:2]
        args.n_sessions = min(args.n_sessions, 3)
        args.frames_per_session = min(args.frames_per_session, 12)
    args.n_sessions = max(2, args.n_sessions)
    args.frames_per_session = max(8, args.frames_per_session)
    mode = getattr(args, "mode", None)
    mode = mode if mode in MODES else "inference"
    out = _make_artifact(args, "toy",
                         {"toy": True, "session_kind": mode,
                          "n_sessions": args.n_sessions,
                          "frames_per_session": args.frames_per_session,
                          "seeds": args.seeds})

    def records_of(seed):
        return make_journal(args.seed + 1000 * seed, args.n_sessions,
                            args.frames_per_session, mode=mode,
                            hidden=args.hidden)
    return _run_worlds(args, out, args.seeds, records_of)


def _load_session_keys(args, records):
    if not args.session_keys:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] journal mode needs the endpoints' "
            "per-session keys so the defended world (and the mandatory "
            "negative control) can verify auth tags: pass --session-keys "
            "FILE (JSON {session_id: {epoch: key hex}}). Keys are used in "
            "memory only and never written into the artifact. Integration "
            "point for a live driver: the sibling MITM proxy emitting a "
            f"{SCHEMA_VERSION} journal plus this key file; no live injector "
            "ships in this module.")
    if not os.path.isfile(args.session_keys):
        raise SystemExit(f"[{EXPERIMENT_ID}] --session-keys file not found: "
                         f"{args.session_keys}")
    try:
        with open(args.session_keys) as f:
            keys = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"{args.session_keys}: invalid JSON ({e})") from None
    if not isinstance(keys, dict) or not all(
            isinstance(v, dict) for v in keys.values()):
        raise ValueError(f"{args.session_keys}: expected a JSON object "
                         "{session_id: {epoch: key hex}}")
    missing = sorted({r["session_id"] for r in records} - set(keys))
    if missing:
        raise ValueError(f"{args.session_keys}: no key for journal "
                         f"session(s) {missing}")
    return keys


def run_journal(args):
    records = read_journal(args.journal)
    keys = _load_session_keys(args, records)
    out = _make_artifact(args, "journal", {"journal": args.journal})
    return _run_worlds(args, out, [0], lambda seed: (records, keys))


def run(args):
    if args.journal:
        return run_journal(args)
    if args.toy:
        return run_toy(args)
    raise SystemExit(f"[{EXPERIMENT_ID}] pick an input path: --journal FILE "
                     f"(MITM protocol-frames journal, schema "
                     f"{SCHEMA_VERSION}; needs --session-keys) or --toy "
                     "(synthetic world with known ground truth).")


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import types

    print("protocol-integrity frame simulation (pure python):")

    from ..surfaces import get_surface, list_surfaces
    prov = get_surface("protocol-frames")
    check("protocol-frames surface registered at import",
          "protocol-frames" in list_surfaces()
          and isinstance(prov, ProtocolFramesProvider))

    records, keys = make_journal(7, 3, 12, hidden=8)
    check("toy journal records validate against the surface schema",
          all(prov.validate(r) for r in records))
    check("journal generation is deterministic",
          make_journal(7, 3, 12, hidden=8)[0] == records)
    try:
        prov.validate({"ts": 1.0, "bytes": 4})
        check("schema drift rejected", False)
    except ValueError:
        check("schema drift rejected", True)
    try:
        prov.validate({**records[0], "auth_tag": "zz"})
        check("non-hex auth_tag rejected", False)
    except ValueError:
        check("non-hex auth_tag rejected", True)

    args = types.SimpleNamespace(seed=99, size_cap=1 << 16, hidden=8)
    defended = defense_profile("defended", args.size_cap)
    vulnerable = defense_profile("vulnerable", args.size_cap)

    by_session = {}
    for r in records:
        by_session.setdefault(r["session_id"], []).append(
            {**r, "_pi_conn": r["session_id"]})
    clean = simulate(by_session, keys, defended)
    check("negative control: defended endpoint accepts the clean journal",
          all(ok_ for _, ok_, _ in clean))
    reasons_seen = {why for _, ok_, why in clean if not ok_}

    expected = {"replay": "seq-not-monotonic",
                "duplicate": "seq-not-monotonic",
                "cross-session-swap": "session-mismatch",
                "epoch-desync": "epoch-rollback",
                "downgrade": "downgrade-refused",
                "oversize": "size-cap-exceeded",
                "truncate": "truncated",
                "tensor-tamper": "auth-failure"}
    for i, (arm, why) in enumerate(sorted(expected.items())):
        cell = run_cell(arm, "defended", records, keys, args, seed=i)
        check(f"defended world detects {arm} as {why}",
              cell["n_detected"] >= 1
              and cell["rejection_reasons"] == [why])
        reasons_seen.update(cell["rejection_reasons"])
        vcell = run_cell(arm, "vulnerable", records, keys, args, seed=i)
        check(f"vulnerable world accepts the {arm} mutation",
              vcell["n_detected"] == 0
              and vcell["undetected_mutation_rate"] == 1.0)

    rcell = run_cell("reorder", "defended", records, keys, args, seed=0)
    check("reorder partially detected (leading gap frame accepted)",
          0 < rcell["undetected_mutation_rate"] < 1
          and rcell["rejection_reasons"] == ["seq-not-monotonic"])
    dcell = run_cell("drop", "defended", records, keys, args, seed=0)
    check("drop undetected by construction in the defended world",
          dcell["n_detected"] == 0
          and dcell["undetected_mutation_rate"] == 1.0)

    for v in TAMPER_VARIANTS:
        fr = _copy(records[3])
        _tamper_frame(fr, v)
        ok_, why = ProtocolEndpoint(defended, keys).accept(
            fr, {"session_id": fr["session_id"]})
        check(f"tensor-tamper variant {v} rejected at auth",
              not ok_ and why == "auth-failure")

    inner = dict(defended, auth=False)
    fr = _copy(records[3])
    fr["value_marks"] = ["inf"]
    check("NaN/Inf marker fails closed without auth",
          ProtocolEndpoint(inner, keys).accept(
              fr, {"session_id": fr["session_id"]}) == (False, "value-invalid"))
    fr = _copy(records[3])
    fr["dtype"] = "float9"
    check("invalid dtype fails closed without auth",
          ProtocolEndpoint(inner, keys).accept(
              fr, {"session_id": fr["session_id"]})
          == (False, "dtype-shape-invalid"))

    check("rejection reasons stay within the constant set",
          reasons_seen <= REJECTION_REASONS)
    check("benign control session never rejected under attack",
          all(run_cell(a, w, records, keys, args, seed=1)
              ["benign_rejection_rate"] == 0.0
              for a in ARMS for w in WORLDS))
    check("run_cell is deterministic",
          run_cell("replay", "defended", records, keys, args, seed=2)
          == run_cell("replay", "defended", records, keys, args, seed=2))
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
