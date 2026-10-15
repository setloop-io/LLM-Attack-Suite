#!/usr/bin/env python3
"""key-session-abuse — complete lifecycle abuse of seam-defense secrets
(THREAT_MODEL_REVIEW P1). stale-key covers one stale epoch key and
seed-disclosure covers one disclosed init seed; this module tests the WHOLE
lifecycle of the secrets a seam defense (rotated seam tensors, per-session
framing, epoch ratchets) depends on: derivation, binding, nonces, retention,
negotiation, logging, and destruction.

The toy tier is a SIMULATOR, not cryptography: all key material is derived
with a hashlib.sha256-based KDF explicitly labeled TOY-KDF. It models the
RELATIONS a real schedule must enforce (one-wayness, uniqueness, binding,
monotonicity, destruction-on-advance) so the defended/vulnerable contrast
is about protocol state machines, never about sha256 strength.

THE SIMULATED SCHEDULE (TOY-KDF)
================================
    master seed --(vulnerable)--> epoch_key[0] = KDF("epoch", master, 0)
                                  epoch_key[t] = KDF("roll", epoch_key[t-1])
                                  (a FORWARD-COMPUTABLE chain, every key
                                  retained — this is WHY the world is weak)
    master seed --(defended)----> chain[0] = KDF("bootstrap", master);
                                  chain[t+1] = KDF("ratchet", chain[t]);
                                  epoch_key[t] = KDF("epoch-key", chain[t]);
                                  master destroyed after bootstrap, old
                                  chain state and old epoch keys destroyed
                                  on advance (one-way ratchet, forward
                                  secrecy; PCS re-keying is out of scope)
    session_key = KDF("session", epoch_key, session_id)   (both worlds)
    keystream   = KDF("stream", session_key, nonce, ctr)  (both worlds)
    frame       = ct = pt XOR keystream; the wire shows
                  {epoch, session_id, nonce, ct} — ground-truth pt stays
                  inside the simulator for scoring.

DEFENSE FLAGS (KeyStore defenses dict): session_binding, epoch_monotonic,
ratchet_forward (old state destroyed), nonce_uniqueness, key_log_scrubbing,
negotiation_auth. --world vulnerable runs with all six off (attacks must
succeed), --world defended with all six on (every attack must be REJECTED —
the defended cells are the mandatory negative control), default "both"
because the contrast IS the result. key-in-logs additionally emits an
isolated "control" cell: a benign log that never contained key material
must scan clean in every world.

ARMS (each journaled as its own per-(arm, world, seed) cell)
============================================================
  nonce-reuse            NoncePool reuse across sessions/epochs; two-time-pad
                         check ct1^ct2 == pt1^pt2 on same-scope collisions and
                         cross-key stream identity on cross-scope collisions.
  session-binding        present a valid (insider-held) session key under a
                         FOREIGN session id; own-session presentation is the
                         sanity control and must stay accepted in both worlds.
  epoch-rollback         replay a stale epoch key after the world advanced;
                         acceptance rate + detection/revocation delay in ops.
  master-seed-exposure   disclose the master seed from the store; count how
                         many epochs/sessions fall (the blast radius).
  forward-secrecy        compromise the epoch-t key; measure recovery of
                         frames from epochs < t (past) and > t (future).
  key-in-logs            scan a fixture log + crashdump + env snapshot for
                         derived key material; benign-log isolated control.
  negotiation-downgrade  unauthenticated request for weaker defense
                         parameters; an authenticated no-op renegotiation is
                         the control and must stay accepted.
  collusion              stale-key holder + passive network observer combine
                         an old epoch key with observed frames to recover
                         CURRENT-session content; a random-key label-free
                         control bounds the recovery floor.

EXPECTED SECURE BEHAVIOR (the interpretation): reuse, rollback, and
downgrade are REJECTED; one disclosure compromises exactly one scope; old
ratchet states are unrecoverable; no key material appears in logs, dumps,
or env; collusion yields nothing beyond the stale epoch.

REAL-TARGET INTEGRATION POINT
=============================
key-in-logs is a real scanner: `--scan-log FILE --known-keys FILE` scans a
real log / crashdump / env snapshot for known key material (known-keys:
one bare 64-hex key or `name=hex` per line; malformed lines and missing
files are descriptive errors naming the file). The remaining arms target a
live seam-defense driver through the KeyStore/NoncePool/RatchetState
interface below — wire the real schedule's present/verify/negotiate paths
into these classes; the toy tier validates the state machines.

Usage:
    python -m attacker --mode training --attack key-session-abuse --help
    python -m attacker --mode inference --attack key-session-abuse \
        --toy --quick --output /tmp/ksa.json
    python -m attacker --mode inference --attack key-session-abuse \
        --scan-log /var/log/seam.log --known-keys keys.txt --output k.json

LIMITATIONS / EXCLUDED VARIANTS: TOY-KDF is a simulation of key-schedule
RELATIONS, not real cryptography — no claim about sha256, AEAD, or DH.
Post-compromise security (healing via fresh per-epoch entropy) is excluded:
the defended ratchet limits PAST recovery only. Multi-device key sync,
hardware-backed key storage, and threshold/HSM custody are out of scope.
The vulnerable world models a forward-computable chain with full retention;
real weak schedules may fail differently. Ground-truth plaintexts are used
for scoring inside the simulator only and are never journaled.
"""

import argparse
import hashlib
import json
import os
import random
import statistics

from .. import artifacts, threat_model
from .common import add_common_args, journal_error

EXPERIMENT_ID = "key_session_abuse"
MODES = ("training", "inference")
REQUIRES_LABELS = "insider (protocol-state)"
DESCRIPTION = ("complete lifecycle abuse of seam-defense secrets: nonce "
               "reuse, session binding, epoch rollback, master-seed blast "
               "radius, forward secrecy, key-in-logs, negotiation "
               "downgrade, stale-key+observer collusion")

SCHEMA_VERSION = "dtraining.attacker.key_session_abuse.v1"
KDF_ID = "TOY-KDF-sha256 (key-schedule RELATION simulator; NOT real crypto)"

DEFENSES = ("session_binding", "epoch_monotonic", "ratchet_forward",
            "nonce_uniqueness", "key_log_scrubbing", "negotiation_auth")

ARMS = ("nonce-reuse", "session-binding", "epoch-rollback",
        "master-seed-exposure", "forward-secrecy", "key-in-logs",
        "negotiation-downgrade", "collusion")

WORLDS = ("vulnerable", "defended")
VULNERABLE = {d: False for d in DEFENSES}
DEFENDED = {d: True for d in DEFENSES}

MSG_BYTES = 24
KNOWN_KEY_HEX = 64


def kdf(label, *parts):
    """TOY-KDF: sha256 over a domain label and length-prefixed parts.
    Deterministic; used to MODEL key-schedule relations — never real
    crypto (KDF_ID)."""
    h = hashlib.sha256()
    h.update(b"TOY-KDF\x00")
    h.update(label.encode())
    h.update(b"\x00")
    for p in parts:
        if isinstance(p, str):
            p = p.encode()
        elif isinstance(p, int):
            p = p.to_bytes(8, "big")
        h.update(len(p).to_bytes(4, "big"))
        h.update(p)
    return h.digest()


def keystream(key, nonce, nbytes):
    """Toy keystream: KDF counter expansion. ct = pt XOR keystream."""
    out, ctr = b"", 0
    while len(out) < nbytes:
        out += kdf("stream", key, nonce, ctr)
        ctr += 1
    return out[:nbytes]


def xor_bytes(a, b):
    return bytes(x ^ y for x, y in zip(a, b))


def stream_identity(key_a, nonce_a, key_b, nonce_b, nbytes=MSG_BYTES):
    """Fraction of identical bytes between two keystreams: 1.0 when key AND
    nonce collide (two-time pad), ~= 1/256 otherwise."""
    sa, sb = keystream(key_a, nonce_a, nbytes), keystream(key_b, nonce_b, nbytes)
    return sum(x == y for x, y in zip(sa, sb)) / nbytes


class NoncePool:
    """Per-world nonce issuer. nonce_uniqueness on: globally unique
    KDF-derived nonces with a seen-registry that REJECTS forced reissue.
    Off: counter mod nonce_space — reuse across sessions and epochs by
    construction (the vulnerable world)."""

    def __init__(self, seed_material, uniqueness, nonce_space=4):
        self.uniqueness = uniqueness
        self.nonce_space = nonce_space
        self._secret = kdf("noncepool", seed_material)
        self._ctr = 0
        self._seen = set()
        self.rejected_reuse_attempts = 0

    def issue(self):
        if self.uniqueness:
            nonce = kdf("nonce", self._secret, self._ctr)[:16]
            self._ctr += 1
            if nonce in self._seen:
                self.rejected_reuse_attempts += 1
                return self.issue()
            self._seen.add(nonce)
            return nonce
        nonce = (self._ctr % self.nonce_space).to_bytes(2, "big")
        self._ctr += 1
        return nonce

    def force_reissue(self, nonce):
        """Attacker attempts to make the pool reissue a known nonce.
        Returns True when the pool complied (vulnerable), False when the
        reuse was rejected (defended)."""
        if self.uniqueness and nonce in self._seen:
            self.rejected_reuse_attempts += 1
            return False
        self._seen.add(nonce)
        return True


class RatchetState:
    """One-way chain state: advance() rolls forward and DESTROYS the old
    state (ratchet_forward); expose() returns the live state."""

    def __init__(self, master_seed):
        self._state = kdf("bootstrap", master_seed)
        self.epoch = 0
        self.destroyed_states = 0

    def expose(self):
        return self._state

    def advance(self):
        self._state = kdf("ratchet", self._state)
        self.epoch += 1
        self.destroyed_states += 1
        return self._state


class KeyStore:
    """Simulated seam-defense key store + schedule under a defenses dict.
    Holds (or destroys) the master seed, the epoch-key chain, retained
    epoch keys, the negotiation key, and the epoch-monotonic watermark."""

    def __init__(self, master_seed, defenses):
        self.defenses = dict(defenses)
        self.negotiation_key = kdf("negotiation", master_seed)
        self.epoch = 0
        self.watermark = 0
        self.detections = []
        if self.defenses["ratchet_forward"]:
            self.ratchet = RatchetState(master_seed)
            self.master = None
            self._epoch_key = kdf("epoch-key", self.ratchet.expose())
            self.retained = {}
        else:
            self.ratchet = None
            self.master = master_seed
            self._epoch_key = kdf("epoch", master_seed, 0)
            self.retained = {0: self._epoch_key}

    def advance_epoch(self):
        self.epoch += 1
        if self.ratchet is not None:
            self._epoch_key = kdf("epoch-key", self.ratchet.advance())
        else:
            self._epoch_key = kdf("roll", self._epoch_key)
            self.retained[self.epoch] = self._epoch_key
        if self.defenses["epoch_monotonic"]:
            self.watermark = self.epoch

    def epoch_key(self):
        return self._epoch_key

    def session_key(self, session_id, epoch_key=None):
        return kdf("session", epoch_key or self._epoch_key, session_id)

    def derive_all_from_master(self, master, n_epochs):
        """Blast radius of a master-seed disclosure: walk the chain the
        world's derivation rule defines. Returns [epoch_key per epoch]."""
        keys = [kdf("epoch", master, 0)]
        for _ in range(1, n_epochs):
            keys.append(kdf("roll", keys[-1]))
        return keys

    def present_epoch_key(self, key, claimed_epoch):
        """An epoch key is presented for use while the store sits at
        self.epoch. epoch_monotonic rejects anything below the watermark
        (and records the detection op); a ratcheted store accepts only the
        live key; the vulnerable store accepts any retained key."""
        op = len(self.detections)
        if self.defenses["epoch_monotonic"] and claimed_epoch < self.watermark:
            self.detections.append(op)
            return False
        if self.ratchet is not None:
            ok = key == self._epoch_key
        else:
            ok = key in set(self.retained.values())
        if not ok:
            self.detections.append(op)
        return ok

    def verify_bound_key(self, claimed_session, presented_key, tag, nonce):
        """Frame authentication. Vulnerable: the tag only has to match the
        PRESENTED key (no binding). Defended: the presented key must equal
        the key bound to the claimed session id."""
        if tag != kdf("frametag", presented_key, nonce):
            return False
        if self.defenses["session_binding"]:
            return presented_key == self.session_key(claimed_session)
        return True

    def renegotiate(self, params, auth_tag=None):
        """Defense-parameter negotiation. negotiation_auth requires a tag
        keyed by the negotiation key over the canonical params; without a
        valid tag the request is rejected. Returns True when applied."""
        if self.defenses["negotiation_auth"]:
            expected = kdf("neg-auth", self.negotiation_key,
                           json.dumps(params, sort_keys=True))
            if auth_tag != expected:
                return False
        return True

    def disclose_master(self):
        """Insider protocol-state disclosure of the store: the master seed
        is present only when it was never destroyed."""
        return self.master


def simulate_world(world, seed, n_epochs, n_sessions, n_frames,
                   nonce_space=4):
    """Run one world end to end. Returns {"store", "pool", "frames",
    "pocket", "master", "n_epochs", ...}. frames: wire records {epoch,
    session_id, nonce, ct} plus simulator-only "pt" ground truth. pocket:
    material an insider legitimately obtained earlier (stale epoch key, a
    mid-life epoch key, a current victim session key)."""
    defenses = VULNERABLE if world == "vulnerable" else DEFENDED
    rng = random.Random(seed)
    master = rng.getrandbits(256).to_bytes(32, "big")
    store = KeyStore(master, defenses)
    pool = NoncePool(master, defenses["nonce_uniqueness"],
                     nonce_space=nonce_space)
    frames, pocket = [], {}
    mid_epoch = n_epochs // 2
    for t in range(n_epochs):
        if t:
            store.advance_epoch()
        ek = store.epoch_key()
        if t == 0:
            pocket["stale_epoch_key"] = ek
        if t == mid_epoch:
            pocket["mid_epoch"] = t
            pocket["mid_epoch_key"] = ek
        for s in range(n_sessions):
            sid = f"tenant-{s}"
            sk = store.session_key(sid, ek)
            if t == n_epochs - 1 and s == 0:
                pocket["victim_session"] = sid
                pocket["victim_session_key"] = sk
            for i in range(n_frames):
                nonce = pool.issue()
                pt = kdf("msg", master, sid, t, i)[:MSG_BYTES]
                frames.append({"epoch": t, "session_id": sid, "nonce": nonce,
                               "ct": xor_bytes(pt, keystream(sk, nonce,
                                                             MSG_BYTES)),
                               "pt": pt})
    pocket["first_nonce"] = frames[0]["nonce"]
    return {"world": world, "store": store, "pool": pool, "frames": frames,
            "pocket": pocket, "master": master, "n_epochs": n_epochs,
            "n_sessions": n_sessions, "n_frames": n_frames}


def _recover_fraction(frames, key_of):
    """Fraction of frames whose decryption under key_of(frame) reproduces
    the ground-truth plaintext (simulator-side scoring)."""
    if not frames:
        return 0.0
    ok = 0
    for f in frames:
        key = key_of(f)
        if key is not None and xor_bytes(
                f["ct"], keystream(key, f["nonce"], MSG_BYTES)) == f["pt"]:
            ok += 1
    return ok / len(frames)


def arm_nonce_reuse(sim):
    store, pool, frames = sim["store"], sim["pool"], sim["frames"]
    by_scope = {}
    for f in frames:
        by_scope.setdefault((f["epoch"], f["session_id"]), []).append(f)
    same_pairs = cross_pairs = xor_leak = 0
    same_ids, cross_ids = [], []
    for recs in by_scope.values():
        seen = {}
        for f in recs:
            if f["nonce"] in seen:
                g = seen[f["nonce"]]
                same_pairs += 1
                same_ids.append(stream_identity(
                    _scope_key(sim, g), g["nonce"],
                    _scope_key(sim, f), f["nonce"]))
                if xor_bytes(g["ct"], f["ct"]) == xor_bytes(g["pt"], f["pt"]):
                    xor_leak += 1
            else:
                seen[f["nonce"]] = f
    by_nonce = {}
    for f in frames:
        by_nonce.setdefault(f["nonce"], []).append(f)
    for recs in by_nonce.values():
        first = recs[0]
        scope0 = (first["epoch"], first["session_id"])
        others = [f for f in recs[1:]
                  if (f["epoch"], f["session_id"]) != scope0]
        if others:
            cross_pairs += len(others)
            cross_ids.append(stream_identity(
                _scope_key(sim, first), first["nonce"],
                _scope_key(sim, others[0]), others[0]["nonce"]))
    forced_ok = pool.force_reissue(sim["pocket"]["first_nonce"])
    succeeded = same_pairs > 0 or (forced_ok and not pool.uniqueness)
    return {"n_frames": len(frames), "n_scopes": len(by_scope),
            "same_scope_reuse_pairs": same_pairs,
            "same_scope_stream_identity": _mean(same_ids),
            "two_time_pad_xor_leak_pairs": xor_leak,
            "cross_scope_reuse_pairs": cross_pairs,
            "cross_scope_stream_identity": _mean(cross_ids),
            "forced_reuse_accepted": bool(forced_ok),
            "rejected_reuse_attempts": pool.rejected_reuse_attempts,
            "attack_succeeded": bool(succeeded)}


def _scope_key(sim, frame):
    return sim["store"].session_key(frame["session_id"], _epoch_key_of(sim, frame))


def _epoch_key_of(sim, frame):
    """Simulator-side truth: the epoch key a frame was sealed under."""
    t = frame["epoch"]
    store = sim["store"]
    if store.ratchet is not None:
        chain = kdf("bootstrap", sim["master"])
        for _ in range(t):
            chain = kdf("ratchet", chain)
        return kdf("epoch-key", chain)
    return store.derive_all_from_master(sim["master"], t + 1)[-1]


def arm_session_binding(sim):
    store, pocket = sim["store"], sim["pocket"]
    rng_nonce = kdf("probe", sim["master"], "binding")[:16]
    victim_key, victim_sid = pocket["victim_session_key"], pocket["victim_session"]
    foreign = sorted({f["session_id"] for f in sim["frames"]}
                     - {victim_sid}) or ["tenant-foreign"]
    accepted = 0
    for sid in foreign:
        tag = kdf("frametag", victim_key, rng_nonce)
        accepted += store.verify_bound_key(sid, victim_key, tag, rng_nonce)
    own_tag = kdf("frametag", victim_key, rng_nonce)
    own_ok = store.verify_bound_key(victim_sid, victim_key, own_tag, rng_nonce)
    rate = accepted / len(foreign)
    return {"foreign_presentations": len(foreign), "foreign_accepted": accepted,
            "foreign_acceptance_rate": round(rate, 6),
            "own_presentations": 1, "own_accepted": int(own_ok),
            "own_acceptance_rate": float(own_ok),
            "attack_succeeded": accepted > 0}


def arm_epoch_rollback(sim):
    store, pocket = sim["store"], sim["pocket"]
    stale, current = pocket["stale_epoch_key"], store.epoch
    ops_budget = 8
    accepted = 0
    detect_delay = None
    for op in range(ops_budget):
        store.detections.clear()
        if store.present_epoch_key(stale, claimed_epoch=0):
            accepted += 1
        elif store.detections and detect_delay is None:
            detect_delay = op
    return {"stale_epoch": 0, "current_epoch": current,
            "rollback_attempts": ops_budget, "rollback_accepted": accepted,
            "rollback_acceptance_rate": round(accepted / ops_budget, 6),
            "stale_state_available": 0 in store.retained,
            "detection_delay_ops": detect_delay,
            "ops_budget": ops_budget,
            "attack_succeeded": accepted > 0}


def arm_master_seed_exposure(sim):
    store = sim["store"]
    master = store.disclose_master()
    n_epochs, n_sessions = sim["n_epochs"], sim["n_sessions"]
    if master is None:
        return {"master_present": False, "epochs_compromised": 0,
                "sessions_compromised": 0, "total_epochs": n_epochs,
                "total_sessions": n_epochs * n_sessions,
                "blast_radius": 0.0, "one_disclosure_one_scope": True,
                "attack_succeeded": False}
    keys = store.derive_all_from_master(master, n_epochs)
    sessions = n_epochs * n_sessions
    return {"master_present": True, "epochs_compromised": len(keys),
            "sessions_compromised": sessions, "total_epochs": n_epochs,
            "total_sessions": sessions, "blast_radius": 1.0,
            "one_disclosure_one_scope": sessions <= 1,
            "attack_succeeded": True}


def arm_forward_secrecy(sim):
    store, pocket, frames = sim["store"], sim["pocket"], sim["frames"]
    t = pocket["mid_epoch"]
    compromised = pocket["mid_epoch_key"]
    past = [f for f in frames if f["epoch"] < t]
    future = [f for f in frames if f["epoch"] > t]
    past_keys = dict(store.retained)
    past_rec = _recover_fraction(
        past, lambda f: store.session_key(f["session_id"],
                                          past_keys[f["epoch"]])
        if f["epoch"] in past_keys else None)
    rolled, key = {}, compromised
    for e in range(t + 1, sim["n_epochs"]):
        key = kdf("roll", key)
        rolled[e] = key
    future_rec = _recover_fraction(
        future, lambda f: store.session_key(f["session_id"], rolled[f["epoch"]])
        if f["epoch"] in rolled else None)
    return {"compromised_epoch": t, "n_epochs": sim["n_epochs"],
            "past_frames": len(past), "future_frames": len(future),
            "past_recovery_fraction": round(past_rec, 6),
            "future_recovery_fraction": round(future_rec, 6),
            "old_state_retained": bool(past_keys),
            "attack_succeeded": past_rec > 0.0 or future_rec > 0.0}


def _log_fixture(sim, scrub):
    """A fixture app log + crashdump + env snapshot. scrub=key_log_scrubbing
    replaces every key-material interpolation with a one-way marker."""
    store = sim["store"]
    master_hex = sim["master"].hex()
    ek0 = _epoch_key_of(sim, {"epoch": 0}).hex()
    sk = store.session_key("tenant-0", _epoch_key_of(sim, {"epoch": 0})).hex()

    def show(key_hex):
        return ("[scrubbed:" + kdf("scrub", bytes.fromhex(key_hex)).hex()[:8]
                + "]") if scrub else key_hex

    log = "\n".join([
        "INFO seam defense bootstrap complete",
        f"INFO loaded master seed {show(master_hex)}",
        f"INFO epoch 0 key {show(ek0)} active",
        "INFO rotation ok, frames flowing",
    ])
    dump = "\n".join([
        "CRASH frame verifier: assertion failed",
        f"context epoch_key={show(ek0)}",
        f"context session_key={show(sk)} tenant=tenant-0",
    ])
    env = "\n".join([
        "PATH=/usr/bin",
        f"SEAM_MASTER_SEED={show(master_hex)}",
        f"SEAM_SESSION_KEY={show(sk)}",
    ])
    material = {"master_seed": master_hex, "epoch0_key": ek0,
                "session0_key": sk}
    return {"app.log": log, "crashdump.txt": dump, "env.snapshot": env}, material


def scan_for_keys(texts, material):
    """Scan {filename: text} for each named key's hex (full or 16-char
    prefix). Returns (n_leaks, sorted leaked 'file:item' names)."""
    leaks = []
    for fname, text in texts.items():
        for item, key_hex in sorted(material.items()):
            if key_hex in text or key_hex[:16] in text:
                leaks.append(f"{fname}:{item}")
    return len(leaks), sorted(leaks)


def arm_key_in_logs(sim):
    scrub = sim["store"].defenses["key_log_scrubbing"]
    texts, material = _log_fixture(sim, scrub)
    n_leaks, leaked = scan_for_keys(texts, material)
    markers = sum(t.count("[scrubbed:") for t in texts.values())
    return {"files_scanned": sorted(texts), "n_key_items": len(material),
            "n_leaks": n_leaks, "leaked_items": leaked,
            "leak_rate": round(n_leaks / (len(texts) * len(material)), 6),
            "scrub_markers": markers,
            "attack_succeeded": n_leaks > 0}


def arm_key_in_logs_control(sim):
    """Isolated control: a benign log that NEVER contained key material
    must scan clean regardless of the world (vacuity guard)."""
    texts = {"benign.log": "INFO all nominal\nINFO frames flowing\n",
             "benign.env": "PATH=/usr/bin\nHOME=/home/svc\n"}
    _, material = _log_fixture(sim, scrub=True)
    n_leaks, leaked = scan_for_keys(texts, material)
    return {"files_scanned": sorted(texts), "n_key_items": len(material),
            "n_leaks": n_leaks, "leaked_items": leaked, "leak_rate": 0.0,
            "scrub_markers": 0, "attack_succeeded": n_leaks > 0}


def arm_negotiation_downgrade(sim):
    store = sim["store"]
    weaker = {"nonce_uniqueness": False, "session_binding": False,
              "epoch_monotonic": False}
    accepted = store.renegotiate(weaker, auth_tag=None)
    good_tag = kdf("neg-auth", store.negotiation_key,
                   json.dumps({"epoch_length": 16}, sort_keys=True))
    auth_ok = store.renegotiate({"epoch_length": 16}, auth_tag=good_tag)
    post_reuse = sim["pool"].force_reissue(sim["pocket"]["first_nonce"]) \
        if accepted else False
    return {"downgrade_accepted": bool(accepted),
            "weakened_params": sorted(weaker),
            "post_downgrade_reuse_success": bool(post_reuse),
            "authenticated_renegotiation_accepted": bool(auth_ok),
            "attack_succeeded": bool(accepted and post_reuse)}


def arm_collusion(sim):
    store, pocket, frames = sim["store"], sim["pocket"], sim["frames"]
    stale, n_epochs = pocket["stale_epoch_key"], sim["n_epochs"]
    stale_frames = [f for f in frames if f["epoch"] == 0]
    current_frames = [f for f in frames if f["epoch"] == n_epochs - 1]
    stale_rec = _recover_fraction(
        stale_frames, lambda f: store.session_key(f["session_id"], stale))
    rolled = stale
    for _ in range(n_epochs - 1):
        rolled = kdf("roll", rolled)
    current_rec = _recover_fraction(
        current_frames, lambda f: store.session_key(f["session_id"], rolled))
    control_key = kdf("control", sim["master"], "random-guess")
    control_rec = _recover_fraction(
        current_frames, lambda f: store.session_key(f["session_id"],
                                                    control_key))
    return {"stale_epoch": 0, "current_epoch": n_epochs - 1,
            "frames_observed": len(frames),
            "stale_epoch_recovery": round(stale_rec, 6),
            "current_session_recovery": round(current_rec, 6),
            "control_recovery": round(control_rec, 6),
            "beyond_stale_epoch": current_rec > control_rec + 0.5,
            "attack_succeeded": current_rec > control_rec + 0.5}


ARM_FUNCS = {"nonce-reuse": arm_nonce_reuse,
             "session-binding": arm_session_binding,
             "epoch-rollback": arm_epoch_rollback,
             "master-seed-exposure": arm_master_seed_exposure,
             "forward-secrecy": arm_forward_secrecy,
             "key-in-logs": arm_key_in_logs,
             "negotiation-downgrade": arm_negotiation_downgrade,
             "collusion": arm_collusion}

_HEADLINE = {"nonce-reuse": "same_scope_reuse_pairs",
             "session-binding": "foreign_acceptance_rate",
             "epoch-rollback": "rollback_acceptance_rate",
             "master-seed-exposure": "sessions_compromised",
             "forward-secrecy": "future_recovery_fraction",
             "key-in-logs": "n_leaks",
             "negotiation-downgrade": "downgrade_accepted",
             "collusion": "current_session_recovery"}


def _mean(xs):
    return round(statistics.fmean(xs), 6) if xs else None


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
                    help="lifecycle arm(s) to run (default: all eight)")
    ap.add_argument("--world", choices=(*WORLDS, "both"), default="both",
                    help="vulnerable (defenses off) / defended (all on — "
                         "the negative control) / both (default: the "
                         "contrast IS the result)")
    ap.add_argument("--epochs", type=int, default=4,
                    help="toy: epochs per world")
    ap.add_argument("--sessions", type=int, default=3,
                    help="toy: concurrent tenant sessions per epoch")
    ap.add_argument("--frames", type=int, default=8,
                    help="toy: frames per session per epoch")
    ap.add_argument("--nonce-space", type=int, default=4,
                    help="toy: vulnerable-world nonce period (smaller = "
                         "more reuse)")
    ap.add_argument("--scan-log", default=None,
                    help="REAL integration point: log / crashdump / env "
                         "snapshot to scan for known key material "
                         "(requires --known-keys)")
    ap.add_argument("--known-keys", default=None,
                    help="key-material list for --scan-log: one bare "
                         f"{KNOWN_KEY_HEX}-hex key or name=hex per line")
    return ap


def _threat_model_doc():
    return threat_model.skeleton(
        summary="complete lifecycle abuse of seam-defense secrets "
                "(S10): a malicious insider with protocol-state access "
                "abuses derivation, binding, nonces, retention, "
                "negotiation, logging, and destruction of the seam "
                "key schedule across epochs",
        scope={"use_case": "split-training", "deployment": "multi-tenant",
               "in_scope": ["seam-defense key schedule (master seed, epoch "
                            "keys, session keys, nonces, ratchet state)",
                            "session/epoch binding and negotiation state",
                            "logs, crash dumps, and env snapshots that may "
                            "carry key material"],
               "out_of_scope": ["TOY-KDF cryptographic strength (simulated "
                                "relations only)",
                                "post-compromise security / healing via "
                                "fresh entropy",
                                "HSM/threshold key custody, multi-device "
                                "sync"]},
        actor={"role": "insider", "intent": "malicious",
               "persistence": "cross-epoch",
               "vantage": ["protocol-state access to the seam-defense "
                           "control plane", "passive wire observation "
                           "(collusion arm)"],
               "authorization": "none beyond stale/incidentally disclosed "
                                "protocol state",
               "knowledge": ["the TOY-KDF derivation rule", "a stale epoch "
                             "key or one disclosed store state"],
               "capabilities": ["present keys under foreign session ids",
                                "replay stale epoch keys",
                                "request unauthenticated renegotiation",
                                "read logs/crashdumps/env snapshots",
                                "observe wire frames passively"],
               "collusion": ["stale-key holder + passive network observer"]},
        target={"assets": ["master seed", "epoch keys", "session keys",
                           "nonces/IVs", "ratchet state"],
                "components": ["KeyStore", "NoncePool", "RatchetState",
                               "negotiation handler", "frame verifier"],
                "data_stores": ["app log", "crash dump", "process env"],
                "trust_boundaries": ["client<->cloud seam control plane",
                                     "tenant session boundary",
                                     "epoch boundary"],
                "entry_points": ["frame presentation", "renegotiation "
                                 "endpoint", "telemetry/log writers"]},
        scenario={"preconditions": ["a seam defense derives per-epoch and "
                                    "per-session keys from a master seed",
                                    "the attacker holds stale or disclosed "
                                    "protocol state"],
                  "lifecycle_stage": ["deployment", "operations"],
                  "security_properties": ["confidentiality", "authenticity"],
                  "attack_steps": [a for a in ARMS],
                  "technical_impact": ["cross-session impersonation",
                                       "epoch rollback",
                                       "full-schedule blast radius from one "
                                       "disclosure",
                                       "current-session content recovery "
                                       "via collusion"]},
        controls={"assumed": ["a seam defense with epoch rotation exists"],
                  "tested": list(DEFENSES),
                  "detective": ["reuse/rollback rejection + detection delay "
                                "in ops"],
                  "response": ["key destruction on advance "
                               "(ratchet_forward)", "log scrubbing"],
                  "status": "open"},
        mappings={"owasp_llm_2026": ["LLM02", "LLM04", "LLM08"]},
        evaluation={"success_metrics": [
                        "sessions/epochs compromised from one disclosure",
                        "past/future recovery fraction after compromise",
                        "reuse and rollback acceptance rates",
                        "detection/revocation delay in ops",
                        "recovery above the label-free control"],
                    "baseline": "defended world (all six mitigations on) + "
                                "isolated benign-log / random-key controls",
                    "budget": {"ops": 8, "worlds": 2},
                    "limitations": ["TOY-KDF is a relation simulator, not "
                                    "crypto", "PCS excluded",
                                    "ground-truth plaintexts used for "
                                    "simulator-side scoring only"],
                    "coverage_status": "covered"})


def _make_artifact(args, source, extra_config=None):
    config = {"attack": EXPERIMENT_ID, "mode": getattr(args, "mode", None),
              "source": source, "arms": args.arms, "world": args.world,
              "kdf": KDF_ID, "seed": args.seed}
    if extra_config:
        config.update(extra_config)
    return artifacts.make_artifact(
        SCHEMA_VERSION, config, _threat_model_doc(),
        interpretation="EXPECTED SECURE BEHAVIOR: reuse/rollback/downgrade "
                       "rejected (defended cells all attack_succeeded="
                       "false); one disclosure compromises exactly one "
                       "scope; old ratchet states unrecoverable "
                       "(forward-secrecy past=future=0); no key material "
                       "in logs/dumps/env; collusion yields nothing "
                       "beyond the stale epoch (current recovery == "
                       "label-free control). The vulnerable world must "
                       "show the opposite everywhere, else the control "
                       "is vacuous.")


def run_toy(args):
    if args.quick:
        args.seeds = args.seeds[:1]
        args.epochs = min(args.epochs, 3)
        args.sessions = min(args.sessions, 2)
        args.frames = min(args.frames, 6)
    worlds = WORLDS if args.world == "both" else (args.world,)
    out = _make_artifact(args, "toy", {
        "toy": True, "defenses": {"vulnerable": VULNERABLE,
                                  "defended": DEFENDED},
        "epochs": args.epochs, "sessions": args.sessions,
        "frames": args.frames, "nonce_space": args.nonce_space,
        "seeds": args.seeds})
    per_cell = {}
    for seed in args.seeds:
        for world in worlds:
            sim = simulate_world(world, args.seed + 1000 * seed,
                                 args.epochs, args.sessions, args.frames,
                                 nonce_space=args.nonce_space)
            for arm in args.arms:
                cell = {"arm": arm, "world": world, "seed": seed}
                try:
                    rec = {"experiment": EXPERIMENT_ID, **cell,
                           "role": ("attack" if world == "vulnerable"
                                    else "negative-control"),
                           **ARM_FUNCS[arm](sim)}
                except Exception as e:
                    journal_error(args.output, EXPERIMENT_ID, cell, e)
                    continue
                artifacts.append_jsonl(args.output, rec)
                out["results"].append(rec)
                per_cell.setdefault((arm, world), []).append(rec)
                print(f"[{EXPERIMENT_ID}] {arm} {world} seed={seed}: "
                      f"attack_succeeded={rec['attack_succeeded']} "
                      f"{_HEADLINE[arm]}={rec[_HEADLINE[arm]]}")
                if arm == "key-in-logs":
                    try:
                        rec = {"experiment": EXPERIMENT_ID, "arm": arm,
                               "world": "control", "seed": seed,
                               "role": "isolated-control",
                               **arm_key_in_logs_control(sim)}
                    except Exception as e:
                        journal_error(args.output, EXPERIMENT_ID,
                                      {"arm": arm, "world": "control",
                                       "seed": seed}, e)
                        continue
                    artifacts.append_jsonl(args.output, rec)
                    out["results"].append(rec)
                    per_cell.setdefault((arm, "control"), []).append(rec)
                    print(f"[{EXPERIMENT_ID}] {arm} control seed={seed}: "
                          f"n_leaks={rec['n_leaks']} (benign log)")
    for (arm, world), recs in sorted(per_cell.items()):
        row = {"experiment": EXPERIMENT_ID, "arm": arm, "world": world,
               "n_seeds": len({r["seed"] for r in recs}),
               "attack_success_rate": round(statistics.fmean(
                   float(r["attack_succeeded"]) for r in recs), 6)}
        metric = _HEADLINE[arm]
        vals = [r[metric] for r in recs
                if isinstance(r.get(metric), (int, float))]
        if vals:
            row[f"{metric}_mean"] = round(statistics.fmean(vals), 6)
        out["summary"].append(row)
    artifacts.write_artifact(args.output, out,
                         evidence=_make_evidence(args))
    return 0


def read_known_keys(path):
    """Parse a known-keys file: one bare 64-hex key or name=hex per line;
    blank/# lines skipped. Malformed lines are a descriptive ValueError
    naming file and line."""
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError as e:
        raise SystemExit(f"[{EXPERIMENT_ID}] --known-keys: cannot read "
                         f"{path}: {e}")
    material = {}
    for lineno, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, hexpart = line.partition("=")
        if not sep:
            name, hexpart = f"key{len(material)}", name
        hexpart = hexpart.strip().lower()
        if len(hexpart) != KNOWN_KEY_HEX or \
                any(c not in "0123456789abcdef" for c in hexpart):
            raise ValueError(
                f"{path}:{lineno}: expected a {KNOWN_KEY_HEX}-hex key "
                f"(optionally name=hex), got {line!r}")
        material[name.strip()] = hexpart
    if not material:
        raise ValueError(f"{path}: no key material found "
                         f"(one {KNOWN_KEY_HEX}-hex key per line)")
    return material


def run_scan(args):
    """--scan-log FILE --known-keys FILE: scan a real log / crashdump /
    env snapshot for known key material (the key-in-logs arm, live)."""
    if not args.known_keys:
        raise SystemExit(f"[{EXPERIMENT_ID}] --scan-log requires "
                         "--known-keys FILE (one 64-hex key or name=hex "
                         "per line)")
    material = read_known_keys(args.known_keys)
    try:
        with open(args.scan_log) as f:
            text = f.read()
    except OSError as e:
        raise SystemExit(f"[{EXPERIMENT_ID}] --scan-log: cannot read "
                         f"{args.scan_log}: {e}")
    n_leaks, leaked = scan_for_keys({os.path.basename(args.scan_log): text},
                                    material)
    out = _make_artifact(args, "scan-log",
                         {"scan_log": args.scan_log,
                          "known_keys": args.known_keys})
    rec = {"experiment": EXPERIMENT_ID, "arm": "key-in-logs",
           "world": "real", "role": "attack",
           "files_scanned": [args.scan_log], "n_key_items": len(material),
           "n_leaks": n_leaks, "leaked_items": leaked,
           "leak_rate": round(n_leaks / max(1, len(material)), 6),
           "attack_succeeded": n_leaks > 0}
    artifacts.append_jsonl(args.output, rec)
    out["results"].append(rec)
    out["summary"].append({"experiment": EXPERIMENT_ID,
                           "arm": "key-in-logs", "world": "real",
                           "n_leaks": n_leaks,
                           "attack_succeeded": rec["attack_succeeded"]})
    artifacts.write_artifact(args.output, out,
                         evidence=_make_evidence(args))
    print(f"[{EXPERIMENT_ID}] key-in-logs real: scanned {args.scan_log} "
          f"for {len(material)} item(s): n_leaks={n_leaks} {leaked}")
    return 0


def run(args):
    if getattr(args, "scan_log", None):
        return run_scan(args)
    if not args.toy:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] pick an input path: --toy (simulated "
            "vulnerable/defended worlds) or --scan-log FILE --known-keys "
            "FILE (real scanner). Integration point for the live "
            "lifecycle arms: wire a real seam-defense driver into the "
            "KeyStore/NoncePool/RatchetState presentation, verification, "
            "and negotiation surfaces (see module docstring).")
    return run_toy(args)


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("key-session-abuse lifecycle simulator (pure python):")

    zero = b"\x00" * 32
    check("TOY-KDF frozen vectors: vulnerable chain of the zero master",
          kdf("epoch", zero, 0).hex()
          == "964470c47ab78de66233c46f6ac82022ca2dbb840f0bf58ed2cc02ba604e7c6a"
          and kdf("roll", kdf("epoch", zero, 0)).hex()
          == "65f0af81f8e22a72fe9a3cf480545d928cb35dea6c3b259f383ccf68072fdc68")
    check("TOY-KDF frozen vectors: defended ratchet of the zero master",
          kdf("bootstrap", zero).hex()
          == "094b3c0492dd240a68d2bfa2fe4317c91dd8a1f3ec439e7330f328b06b61a41d"
          and kdf("epoch-key", kdf("bootstrap", zero)).hex()
          == "609cff85aa5270fd1e46ade8c5f489dce5af4a15c7d619c667e5e35e2f54a616")
    check("one-way: epoch key never rolls back to the master that made it",
          kdf("roll", kdf("epoch", zero, 0)) != zero
          and kdf("epoch", kdf("epoch", zero, 0), 0) != kdf("epoch", zero, 1))

    sim_v = simulate_world("vulnerable", seed=7, n_epochs=4, n_sessions=3,
                           n_frames=8)
    sim_d = simulate_world("defended", seed=7, n_epochs=4, n_sessions=3,
                           n_frames=8)

    check("vulnerable world: master seed retained; defended: destroyed",
          sim_v["store"].disclose_master() is not None
          and sim_d["store"].disclose_master() is None)
    check("vulnerable world retains every epoch key; defended retains none",
          len(sim_v["store"].retained) == 4
          and sim_d["store"].retained == {})

    r = arm_nonce_reuse(sim_v)
    check("vulnerable nonce-reuse: same-scope reuse with two-time pad",
          r["same_scope_reuse_pairs"] > 0
          and r["two_time_pad_xor_leak_pairs"] == r["same_scope_reuse_pairs"]
          and r["same_scope_stream_identity"] == 1.0
          and r["attack_succeeded"])
    r = arm_nonce_reuse(sim_d)
    check("defended nonce-reuse: no reuse, forced reissue rejected",
          r["same_scope_reuse_pairs"] == 0 and not r["forced_reuse_accepted"]
          and r["rejected_reuse_attempts"] > 0 and not r["attack_succeeded"])

    rv, rd = arm_session_binding(sim_v), arm_session_binding(sim_d)
    check("session-binding: foreign accepted only when vulnerable; own "
          "accepted in both worlds (control)",
          rv["foreign_acceptance_rate"] == 1.0
          and rd["foreign_acceptance_rate"] == 0.0
          and rv["own_acceptance_rate"] == 1.0
          and rd["own_acceptance_rate"] == 1.0)

    rv, rd = arm_epoch_rollback(sim_v), arm_epoch_rollback(sim_d)
    check("epoch-rollback: vulnerable accepts and never detects; defended "
          "rejects at op 0",
          rv["rollback_acceptance_rate"] == 1.0
          and rv["detection_delay_ops"] is None
          and rd["rollback_acceptance_rate"] == 0.0
          and rd["detection_delay_ops"] == 0)

    rv, rd = arm_master_seed_exposure(sim_v), arm_master_seed_exposure(sim_d)
    check("master-seed-exposure: full blast radius vs zero scopes",
          rv["sessions_compromised"] == 12 and rv["blast_radius"] == 1.0
          and rd["sessions_compromised"] == 0 and not rd["master_present"])

    rv, rd = arm_forward_secrecy(sim_v), arm_forward_secrecy(sim_d)
    check("forward-secrecy: vulnerable recovers past AND future; defended "
          "recovers neither",
          rv["past_recovery_fraction"] == 1.0
          and rv["future_recovery_fraction"] == 1.0
          and rd["past_recovery_fraction"] == 0.0
          and rd["future_recovery_fraction"] == 0.0)

    rv, rd = arm_key_in_logs(sim_v), arm_key_in_logs(sim_d)
    rc = arm_key_in_logs_control(sim_v)
    check("key-in-logs: vulnerable fixture leaks, defended scrubbed, "
          "benign control clean",
          rv["n_leaks"] > 0 and rd["n_leaks"] == 0
          and rd["scrub_markers"] > 0 and rc["n_leaks"] == 0)

    rv, rd = arm_negotiation_downgrade(sim_v), arm_negotiation_downgrade(sim_d)
    check("negotiation-downgrade: unauthenticated accepted only when "
          "vulnerable; authenticated control accepted in both",
          rv["downgrade_accepted"] and rv["post_downgrade_reuse_success"]
          and not rd["downgrade_accepted"]
          and rv["authenticated_renegotiation_accepted"]
          and rd["authenticated_renegotiation_accepted"])

    rv, rd = arm_collusion(sim_v), arm_collusion(sim_d)
    check("collusion: stale epoch opens in both worlds; current session "
          "only when vulnerable; random-key control at floor",
          rv["stale_epoch_recovery"] == 1.0 and rd["stale_epoch_recovery"] == 1.0
          and rv["current_session_recovery"] == 1.0
          and rd["current_session_recovery"] == 0.0
          and rv["control_recovery"] == 0.0 and rd["control_recovery"] == 0.0
          and rv["beyond_stale_epoch"] and not rd["beyond_stale_epoch"])

    check("threat model is a valid v2 document",
          threat_model.validate(_threat_model_doc()) is not None)

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
