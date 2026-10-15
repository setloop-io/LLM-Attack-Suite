#!/usr/bin/env python3
"""Formal evidence chain for attack artifacts (pure stdlib, torch-free).

Every attack artifact can be accompanied by a tamper-evident evidence
trail:

    <artifact>.evidence.jsonl   human-readable event stream, one JSON
                                object per line; each event carries the
                                sha256 of the previous event (hash chain)
    <artifact>.evidence.json    run manifest (the "EvidenceRecord"):
                                run_id, timestamps, argv, host/platform,
                                package versions, git state, whitelisted
                                env, node role, the artifact's own sha256,
                                and the final chain_hash

Tamper evidence: event i's hash commits to its full contents plus event
i-1's hash, so editing, deleting, or reordering any historical event line
breaks every subsequent link and the manifest's chain_hash.
verify_evidence() re-derives the whole chain for auditors.

Usage:

    from attacker.evidence import EvidenceLogger, verify_evidence

    with EvidenceLogger("out/er1.json", node_role="toula/gpu") as ev:
        ev.event("cell_result", {"cell": 1, "top1": 3.2})
        ...                    # write the artifact itself
        ev.finalize("out/er1.json")   # or let __exit__ finalize

    ok, problems = verify_evidence("out/er1.json.evidence.json")

Design rules (repo conventions): stdlib only, never raises on provenance
collection (best-effort, cf. artifacts.light_provenance), events flushed
after every append so a hard crash cannot take recorded events down with
the process (cf. artifacts.append_jsonl).
"""

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

__all__ = ["EvidenceLogger", "verify_evidence", "file_sha256",
           "ENV_WHITELIST", "GENESIS_HASH", "self_test"]

SCHEMA = "llm_attacker.evidence.v2"

GENESIS_HASH = "0" * 64

ENV_WHITELIST = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "CONTAINER_IMAGE",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)

_EVIDENCE_JSONL = ".evidence.jsonl"
_EVIDENCE_JSON = ".evidence.json"


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(obj):
    """Deterministic JSON encoding used for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path):
    """sha256 hex digest of a file's contents (public helper)."""
    return _sha256_file(path)


def _jsonable(payload):
    """Coerce a payload to something JSON-serializable, never raising."""
    try:
        json.dumps(payload)
        return payload if payload is not None else {}
    except (TypeError, ValueError):
        return {"_unserializable": repr(payload)}


def _installed_packages():
    """importlib.metadata package versions, best-effort (never raises)."""
    try:
        from importlib.metadata import distributions
        out = {}
        for dist in distributions():
            name = dist.metadata.get("Name")
            if name:
                out.setdefault(name.lower(), dist.version)
        return dict(sorted(out.items()))
    except Exception:
        return {}


def _git_state():
    """Commit + dirty flag if inside a git repo, best-effort (no hard dep)."""
    state = {"commit": "unknown", "dirty": None}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        commit = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        if commit:
            state["commit"] = commit
            porcelain = subprocess.run(
                ["git", "-c", "safe.directory=*", "-C", root,
                 "status", "--porcelain"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            state["dirty"] = bool(porcelain)
    except Exception:
        pass
    from pathlib import Path
    source = {str(p.relative_to(root)): _sha256_file(p)
              for p in sorted(Path(root, "attacker").rglob("*.py"))}
    state["source_sha256"] = _sha256_text(_canonical(source))
    return state


def _whitelisted_env():
    return {k: os.environ[k] for k in ENV_WHITELIST if k in os.environ}


class EvidenceLogger:
    """Tamper-evident evidence logger bound to one artifact path.

    path          artifact being produced; events stream to
                  <path>.evidence.jsonl, the manifest lands at
                  <path>.evidence.json
    node_role     free-form node label (e.g. "toula/gpu")
    extra         extra JSON-serializable dict merged into the manifest
    """

    def __init__(self, path, node_role=None, extra=None):
        self.path = str(path)
        self.events_path = self.path + _EVIDENCE_JSONL
        self.manifest_path = self.path + _EVIDENCE_JSON
        self.run_id = str(uuid.uuid4())
        self.node_role = node_role
        self.extra = dict(extra) if extra else {}
        self._start_utc = _utc_now()
        self._prev_hash = GENESIS_HASH
        self._count = 0
        self._finalized = False
        parent = os.path.dirname(os.path.abspath(self.events_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fh = open(self.events_path, "w", encoding="utf-8")
        self.event("run_start", {"run_id": self.run_id,
                                 "node_role": self.node_role})

    def event(self, kind, payload=None):
        """Append one chained event; returns the recorded event dict."""
        if self._finalized:
            raise ValueError("evidence logger already finalized")
        rec = {"seq": self._count,
               "ts_utc": _utc_now(),
               "run_id": self.run_id,
               "kind": str(kind),
               "payload": _jsonable(payload),
               "prev_hash": self._prev_hash}
        rec["hash"] = _sha256_text(_canonical(rec))
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()
        self._prev_hash = rec["hash"]
        self._count += 1
        return rec

    def finalize(self, artifact_path=None):
        """Seal the chain and write the run manifest. Idempotent."""
        if self._finalized:
            return self.manifest_path
        artifact_path = str(artifact_path) if artifact_path is not None \
            else self.path
        end_utc = _utc_now()
        self.event("run_end", {"end_utc": end_utc})
        artifact = {"path": artifact_path}
        if os.path.isfile(artifact_path):
            artifact["sha256"] = _sha256_file(artifact_path)
            artifact["size_bytes"] = os.path.getsize(artifact_path)
        else:
            artifact["sha256"] = None
            artifact["error"] = "artifact missing at finalize"
        manifest = {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "node_role": self.node_role,
            "start_utc": self._start_utc,
            "end_utc": end_utc,
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "packages": _installed_packages(),
            "git": _git_state(),
            "env": _whitelisted_env(),
            "extra": self.extra,
            "events_path": self.events_path,
            "event_count": self._count,
            "chain_hash": self._prev_hash,
            "artifact": artifact,
        }
        manifest["manifest_sha256"] = _sha256_text(_canonical(manifest))
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, self.manifest_path)
        self._finalized = True
        self._fh.close()
        return self.manifest_path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is not None and not self._finalized:
                self.event("error", {"error": f"{exc_type.__name__}: {exc}"})
            if not self._finalized:
                self.finalize()
            elif not self._fh.closed:
                self._fh.close()
        except Exception:
            pass
        return False


def _resolve(path, manifest_path, suffix, replace_with=""):
    """Resolve a manifest-recorded path, falling back to the sibling of the
    manifest (runs verified in place after a directory move). The sibling
    is `manifest_path` with `suffix` swapped for `replace_with` — e.g. the
    events log is the manifest's `.evidence.json` swapped to
    `.evidence.jsonl`, the artifact is the manifest minus `.evidence.json`.
    """
    if manifest_path.endswith(suffix):
        candidate = manifest_path[:-len(suffix)] + replace_with
        if os.path.isfile(candidate):
            return candidate
    if path and os.path.isfile(path):
        return path
    return path


def verify_evidence(manifest_path, expected_manifest_sha256=None):
    """Re-derive the evidence chain for auditors.

    Returns (ok, problems): ok is True iff no problem was found; problems
    is a human-readable list of every integrity violation detected.
    """
    problems = []
    manifest_path = str(manifest_path)
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        return False, [f"manifest unreadable: {e}"]

    if not isinstance(manifest, dict) or not isinstance(manifest.get("artifact"), dict):
        return False, ["manifest and artifact must be JSON objects"]
    digest = _sha256_text(_canonical({k: v for k, v in manifest.items() if k != "manifest_sha256"}))
    if digest != manifest.get("manifest_sha256"):
        problems.append("manifest hash mismatch (metadata tampered?)")
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
        problems.append("manifest does not match trusted external digest")
    for field in ("schema", "run_id", "chain_hash", "event_count",
                  "events_path", "artifact"):
        if field not in manifest:
            problems.append(f"manifest missing field: {field}")
    if problems:
        return False, problems
    if manifest["schema"] != SCHEMA:
        problems.append(f"unknown schema: {manifest['schema']}")

    events_path = _resolve(manifest.get("events_path"), manifest_path,
                           _EVIDENCE_JSON, _EVIDENCE_JSONL)
    events = []
    if not events_path or not os.path.isfile(events_path):
        problems.append(f"events file missing: {events_path}")
    else:
        with open(events_path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        problems.append(f"event line {lineno} is not an object")
                        continue
                    events.append(event)
                except json.JSONDecodeError as e:
                    problems.append(f"event line {lineno} is not JSON: {e}")

    prev = GENESIS_HASH
    for i, ev in enumerate(events):
        if ev.get("seq") != i:
            problems.append(f"event {i}: seq discontinuity "
                            f"(got {ev.get('seq')})")
        if ev.get("run_id") != manifest["run_id"]:
            problems.append(f"event {i}: run_id mismatch")
        if ev.get("prev_hash") != prev:
            problems.append(f"event {i}: prev_hash link broken")
        recorded = ev.get("hash")
        body = {k: v for k, v in ev.items() if k != "hash"}
        if recorded != _sha256_text(_canonical(body)):
            problems.append(f"event {i}: hash mismatch (content tampered?)")
        prev = recorded or prev
    if events and events[-1].get("kind") != "run_end":
        problems.append("event stream not sealed (last event is not run_end)")
    if len(events) != manifest["event_count"]:
        problems.append(f"event_count mismatch: manifest says "
                        f"{manifest['event_count']}, stream has {len(events)}")
    if prev != manifest["chain_hash"]:
        problems.append("chain_hash mismatch: manifest does not match the "
                        "re-derived event chain")

    artifact = manifest.get("artifact") or {}
    recorded_sha = artifact.get("sha256")
    if recorded_sha is None:
        problems.append("artifact was missing at finalize time")
    else:
        artifact_path = _resolve(artifact.get("path"), manifest_path,
                                 _EVIDENCE_JSON)
        if not artifact_path or not os.path.isfile(artifact_path):
            problems.append(f"artifact missing: {artifact.get('path')}")
        elif _sha256_file(artifact_path) != recorded_sha:
            problems.append("artifact sha256 mismatch (artifact tampered?)")

    return (not problems), problems


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import tempfile
    print("evidence chain (pure python):")
    with tempfile.TemporaryDirectory() as d:
        art = os.path.join(d, "cell.json")
        with open(art, "w") as f:
            json.dump({"schema": "selftest", "results": []}, f)

        ev = EvidenceLogger(art, node_role="selftest/cpu",
                            extra={"experiment": "selftest"})
        ev.event("cell_result", {"cell": 1, "top1": 3.2})
        ev.event("artifact_write", {"path": art})
        manifest_path = ev.finalize(art)
        check("manifest and event stream written",
              os.path.isfile(manifest_path) and
              os.path.isfile(art + ".evidence.jsonl"))
        manifest = json.load(open(manifest_path))
        check("manifest carries run identity fields",
              all(k in manifest for k in
                  ("run_id", "start_utc", "end_utc", "argv", "hostname",
                   "platform", "python_version", "cpu_count", "packages",
                   "git", "env", "node_role", "chain_hash", "artifact")))
        check("node role + extra recorded",
              manifest["node_role"] == "selftest/cpu" and
              manifest["extra"]["experiment"] == "selftest")
        check("artifact sha256 recorded",
              manifest["artifact"]["sha256"] == _sha256_file(art))
        check("env capture is whitelisted only",
              all(k in ENV_WHITELIST for k in manifest["env"]))

        good, problems = verify_evidence(manifest_path)
        check("verify_evidence accepts the untampered chain",
              good and problems == [])

        lines = open(art + ".evidence.jsonl").read().splitlines()
        victim = json.loads(lines[1])
        victim["payload"]["top1"] = 99.9
        lines[1] = json.dumps(victim)
        with open(art + ".evidence.jsonl", "w") as f:
            f.write("\n".join(lines) + "\n")
        bad, problems = verify_evidence(manifest_path)
        check("verify_evidence detects a tampered event line",
              not bad and any("tampered" in p or "chain_hash" in p
                              for p in problems))

        ev2 = EvidenceLogger(os.path.join(d, "b.json"))
        ev2.event("cell_result", {"cell": 1})
        mp2 = ev2.finalize()
        m2 = json.load(open(mp2))
        m2["chain_hash"] = GENESIS_HASH
        with open(mp2, "w") as f:
            json.dump(m2, f)
        bad2, _ = verify_evidence(mp2)
        check("verify_evidence detects a tampered manifest", not bad2)

        with EvidenceLogger(os.path.join(d, "c.json")) as ev3:
            ev3.event("cell_result", {"cell": 1})
        check("context manager finalizes on exit",
              os.path.isfile(ev3.manifest_path))
        m3 = json.load(open(ev3.manifest_path))
        check("context-manager manifest is sealed",
              m3["event_count"] == len(open(
                  ev3.events_path).read().strip().splitlines()))
        ok3, problems3 = verify_evidence(ev3.manifest_path)
        check("only the missing artifact is flagged",
              not ok3 and problems3 == ["artifact was missing at finalize "
                                        "time"])

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
