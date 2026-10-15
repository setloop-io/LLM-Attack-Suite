#!/usr/bin/env python3
"""supply-chain — AI-artifact registry verification under attack
(THREAT_MODEL_REVIEW P0, OWASP LLM04/LLM05, surface S13). Determines
whether AI artifacts and extension points are identified, verified,
promoted, loaded, and audited safely: base models/checkpoints, tokenizers
and configs, LoRA adapters + quantization artifacts, training/eval/RAG
datasets, attack plugins/surface providers, python packages/container
images, capture tensors/serialized bundles.

A fixture registry lives on disk (toy mode builds it under --work-dir or a
fresh temp dir; --registry verifies a real one):

    <registry>/registry.json            registry index + policy
                                        (llm_attacker.ai_registry.v1):
                                        artifacts list, AI-BOM, and a
                                        policy of pinned digests, min
                                        versions, tokenizer/adapter
                                        bindings, plugin allowlist,
                                        approved dependencies
    <registry>/environment.json         installed package set
                                        (llm_attacker.ai_environment.v1)
    <registry>/<name>/manifest.json     signed artifact manifest
                                        (llm_attacker.ai_artifact_manifest.v1):
                                        name+version+purpose+tenant ->
                                        payload sha256, a TOY HMAC
                                        signature (hashlib/hmac,
                                        labelled TOY — a stand-in for real
                                        PKI, NOT a real signature scheme),
                                        provenance (builder, source repo,
                                        timestamp), purpose bindings, and
                                        an AI-BOM entry list
    <registry>/<name>/payload.json      the artifact bytes (small JSON
                                        stand-ins for real checkpoints)
    <registry>/archive/toy-model-v1/    an older, still validly signed
                                        version (the rollback bait)

The Verifier runs an ordered pipeline of stages:

    manifest-parse -> provenance -> signature -> pinned-digest ->
    version-monotonic -> binding -> plugin-allowlist ->
    safe-serialization -> (dependency-manifest, environment-level) -> load

and FAILS CLOSED at the first violated stage, before the payload bytes are
ever parsed: pinned-digest only hashes bytes, safe-serialization is a
bytes-level gate of a restricted JSON loader that refuses pickle magic
(0x80 protocol bytes), __reduce__/__reduce_ex__ markers, and executable
magics BEFORE any json.loads. Nothing is ever unpickled — the pickle
fixture is INERT bytes (magic + markers only, not a valid pickle).

Arms (each runs against the ENFORCING verifier — every attack must be
rejected — and against a PERMISSIVE verifier that accepts everything, the
vulnerable baseline proving the attack actually lands):

  artifact-substitution  same expected name+version, different bytes
                         (wholesale payload replacement; no signing key)
                         -> rejected at pinned-digest, BEFORE parsing
  digest-mismatch        pinned artifact mutated in place (bit flip)
                         -> rejected at pinned-digest
  unsigned-promotion     v3 promotion with no provenance, no signature
                         -> rejected at provenance
  rollback               archive v1, validly signed, self-consistent
                         digest, but older than the pinned v2
                         -> rejected at version-monotonic
  tokenizer-drift        validly signed v2 tokenizer: changed vocab /
                         normalization / chat template, same model
                         identity (models a compromised build pipeline)
                         -> rejected at binding
  adapter-confusion      validly signed v2 adapter bound to the wrong
                         (model, tenant) -> rejected at binding
  plugin-origin          two variants: unapproved plugin name; approved
                         name with a conflicting origin (name-squatting)
                         -> rejected at plugin-allowlist
  unsafe-serialization   validly signed v2 dataset whose payload is the
                         INERT pickle-magic fixture (insider pipeline)
                         -> rejected at safe-serialization BEFORE parse
  dependency-drift       installed set vs approved manifest (version
                         bump + unapproved extra package)
                         -> rejected at dependency-manifest

Metrics per arm: verification bypass rate (permissive ~1.0, enforcing
0.0), % of loaded assets represented in the registry AI-BOM, provenance
completeness of loaded assets, time-to-detect (pipeline stage index of
the rejection; lower = earlier; substitution must trip pinned-digest,
i.e. before the load stage), and the failure-stage distribution.
Negative control: the untouched registry verifies cleanly under the
enforcing verifier (0 false rejections).

Real-target path (fully implemented, torch-free): --registry DIR verifies
an on-disk registry with the enforcing verifier and reports
accepted/rejected artifacts, the environment diff, AI-BOM coverage and
provenance completeness; --manifest FILE overrides the registry policy
with an approved manifest (llm_attacker.ai_approved_manifest.v1). Schema
drift anywhere is a descriptive ValueError naming the file. The "load"
stage only ever json.loads a payload that passed the gate — wiring real
deserializers (safetensors/GGUF/torch.load) behind it is the documented
integration point; verifying real binary checkpoints means hashing +
gate-scanning here, never parsing.

Limitations / excluded variants: the TOY HMAC is not real PKI (no key
management, no transparency log); arms that need a validly signed
malicious artifact (tokenizer-drift, adapter-confusion, plugin-origin,
unsafe-serialization) assume a compromised or insider build pipeline,
arms that test the digest/provenance gates do not; reproducible-build
attestation, container image layer scanning, and real SBOM
(CycloneDX/SPDX) import are out of scope — the AI-BOM here is the toy
registry list; the permissive verifier's "load" is simulated (a real
permissive stack would execute the payload; we only record that it
would).

Usage:
    python -m attacker --mode inference --attack supply-chain --help
    python -m attacker --mode inference --attack supply-chain --toy \
        --quick --output /tmp/sc.json
    python -m attacker --mode training --attack supply-chain \
        --registry /path/to/registry --output sc.json
"""

import argparse
import hashlib
import hmac
import json
import os
import random
import shutil
import statistics
import tempfile

from .. import artifacts, threat_model
from .common import add_common_args, journal_error

EXPERIMENT_ID = "supply_chain"
MODES = ("training", "inference")
REQUIRES_LABELS = False
DESCRIPTION = ("AI-artifact supply-chain verification under attack: "
               "substitution, tamper, unsigned promotion, rollback, "
               "tokenizer/adapter drift, plugin origin, unsafe "
               "serialization, dependency drift vs an enforcing and a "
               "permissive registry verifier (OWASP LLM04/LLM05, S13)")

REGISTRY_SCHEMA = "llm_attacker.ai_registry.v1"
MANIFEST_SCHEMA = "llm_attacker.ai_artifact_manifest.v1"
APPROVED_MANIFEST_SCHEMA = "llm_attacker.ai_approved_manifest.v1"
ENVIRONMENT_SCHEMA = "llm_attacker.ai_environment.v1"

TOY_SIGNING_KEY = b"toy-registry-signing-key-NOT-A-REAL-KEY"
SIGNATURE_SCHEME = "toy-hmac-sha256 (TOY stand-in for PKI — not a real signature)"

PURPOSES = ("model", "tokenizer", "adapter", "dataset", "plugin",
            "package", "capture")
SIGNED_FIELDS = ("name", "version", "purpose", "tenant", "payload",
                 "provenance", "bindings", "ai_bom")

STAGES = ("manifest-parse", "provenance", "signature", "pinned-digest",
          "version-monotonic", "binding", "plugin-allowlist",
          "safe-serialization", "dependency-manifest", "load")
LOAD_STAGE_INDEX = STAGES.index("load")

ARMS = ("artifact-substitution", "digest-mismatch", "unsigned-promotion",
        "rollback", "tokenizer-drift", "adapter-confusion",
        "plugin-origin", "unsafe-serialization", "dependency-drift")

EXPECTED_STAGE = {
    "artifact-substitution": "pinned-digest",
    "digest-mismatch": "pinned-digest",
    "unsigned-promotion": "provenance",
    "rollback": "version-monotonic",
    "tokenizer-drift": "binding",
    "adapter-confusion": "binding",
    "plugin-origin": "plugin-allowlist",
    "unsafe-serialization": "safe-serialization",
    "dependency-drift": "dependency-manifest",
}

ARM_INFO = {
    "artifact-substitution": "wholesale payload replacement under the "
                             "same expected name+version; no signing key "
                             "needed — the registry pin anchors the bytes",
    "digest-mismatch": "in-place bit flip of a pinned artifact; the "
                       "manifest still claims the original digest",
    "unsigned-promotion": "v3 promotion candidate with provenance and "
                          "signature stripped",
    "rollback": "archived v1, validly signed and self-consistent, served "
                "in place of the pinned v2 (monotonic-version policy)",
    "tokenizer-drift": "validly signed v2 tokenizer (compromised build "
                       "pipeline): vocab/normalization/chat-template "
                       "changed, model identity preserved",
    "adapter-confusion": "validly signed v2 adapter bound to the wrong "
                         "(model, tenant) — a cross-tenant artifact "
                         "offered to tenant-a's serving stack",
    "plugin-origin": "unapproved plugin name, plus an allowlisted name "
                     "with a conflicting origin (name-squatting)",
    "unsafe-serialization": "validly signed v2 dataset whose payload is "
                            "the INERT pickle-magic fixture — the "
                            "restricted loader refuses it BEFORE parsing; "
                            "never unpickled",
    "dependency-drift": "installed environment vs the approved dependency "
                        "manifest: version bump + unapproved extra package",
}

TOY_EPOCH = 1_700_000_000
TOY_BASE_VOCAB = ["<pad>", "<unk>", "<bos>", "<eos>"]
TOY_APPROVED_DEPS = {"numpy": "1.26.4", "tokenizers": "0.20.0",
                     "safetensors": "0.4.5"}

INERT_PICKLE_FIXTURE = (b"\x80\x04INERT-FIXTURE - not a valid pickle; "
                        b"carries __reduce__ marker bytes so the scanner "
                        b"can prove untrusted parsing cannot execute")


def canonical_json(obj):
    """Deterministic byte encoding used for digests/signatures/payloads."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def sign_manifest(manifest):
    """TOY HMAC signature over the signed manifest fields (see
    TOY_SIGNING_KEY — a labelled stand-in for real PKI, nothing more)."""
    body = {k: manifest.get(k) for k in SIGNED_FIELDS}
    return hmac.new(TOY_SIGNING_KEY, canonical_json(body),
                    hashlib.sha256).hexdigest()


def _is_sha256(s):
    return (isinstance(s, str) and len(s) == 64
            and all(c in "0123456789abcdef" for c in s))


def _is_non_empty_str(s):
    return isinstance(s, str) and bool(s)


def validate_manifest(doc, path="<manifest>"):
    """Schema-validate one manifest.json; returns a normalized copy (with
    bindings/ai_bom filled). Any drift is a ValueError naming `path`."""
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: manifest must be a JSON object "
                         f"(schema {MANIFEST_SCHEMA})")
    if doc.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{path}: schema must be {MANIFEST_SCHEMA!r}, "
                         f"got {doc.get('schema')!r}")
    unknown = sorted(set(doc) - {"schema", "signature", *SIGNED_FIELDS})
    if unknown:
        raise ValueError(f"{path}: unknown manifest keys {unknown}")
    for field in ("name", "tenant"):
        if not _is_non_empty_str(doc.get(field)):
            raise ValueError(f"{path}: {field} must be a non-empty str, "
                             f"got {doc.get(field)!r}")
    v = doc.get("version")
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ValueError(f"{path}: version must be an int >= 0, got {v!r}")
    if doc.get("purpose") not in PURPOSES:
        raise ValueError(f"{path}: purpose must be one of "
                         f"{list(PURPOSES)}, got {doc.get('purpose')!r}")
    payload = doc.get("payload")
    if not isinstance(payload, dict) \
            or not _is_non_empty_str(payload.get("file")) \
            or not _is_sha256(payload.get("sha256")):
        raise ValueError(f"{path}: payload must be {{\"file\": <str>, "
                         f"\"sha256\": <64-hex>}}, got {payload!r}")
    sig = doc.get("signature")
    if not isinstance(sig, str) or (sig and not _is_sha256(sig)):
        raise ValueError(f"{path}: signature must be '' (unsigned) or a "
                         f"64-hex {SIGNATURE_SCHEME} digest, got {sig!r}")
    prov = doc.get("provenance")
    if not isinstance(prov, dict):
        raise ValueError(f"{path}: provenance must be an object "
                         f"(possibly empty = missing), got {prov!r}")
    for field in ("builder", "source_repo"):
        if field in prov and not _is_non_empty_str(prov[field]):
            raise ValueError(f"{path}: provenance.{field} must be a "
                             f"non-empty str, got {prov[field]!r}")
    if "timestamp" in prov and (isinstance(prov["timestamp"], bool)
                                or not isinstance(prov["timestamp"],
                                                  (int, float))):
        raise ValueError(f"{path}: provenance.timestamp must be a number, "
                         f"got {prov['timestamp']!r}")
    bindings = doc.get("bindings", {})
    if not isinstance(bindings, dict):
        raise ValueError(f"{path}: bindings must be an object, "
                         f"got {bindings!r}")
    bom = doc.get("ai_bom", [])
    if not isinstance(bom, list):
        raise ValueError(f"{path}: ai_bom must be a list, got {bom!r}")
    for i, entry in enumerate(bom):
        if not isinstance(entry, dict) \
                or not _is_non_empty_str(entry.get("name")) \
                or not _is_sha256(entry.get("sha256")):
            raise ValueError(f"{path}: ai_bom[{i}] must be {{\"name\": "
                             f"<str>, \"version\": <int>, \"sha256\": "
                             f"<64-hex>}}, got {entry!r}")
    out = dict(doc)
    out["bindings"] = dict(bindings)
    out["ai_bom"] = [dict(e) for e in bom]
    return out


def load_manifest(path):
    """Read + validate a manifest.json; ValueError naming the file on any
    drift (missing file, bad JSON, schema drift)."""
    if not os.path.isfile(path):
        raise ValueError(f"{path}: no such manifest file")
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"{path}: cannot parse manifest JSON: {e}")
    return validate_manifest(doc, path)


def _validate_pins(pins, path):
    if not isinstance(pins, dict):
        raise ValueError(f"{path}: policy.pins must be an object, "
                         f"got {pins!r}")
    for name, pin in pins.items():
        if not _is_non_empty_str(name) or not isinstance(pin, dict) \
                or isinstance(pin.get("version"), bool) \
                or not isinstance(pin.get("version"), int) \
                or not _is_sha256(pin.get("sha256")):
            raise ValueError(f"{path}: policy.pins[{name!r}] must be "
                             f"{{\"version\": <int>, \"sha256\": "
                             f"<64-hex>}}, got {pin!r}")
    return dict(pins)


def _validate_bindings_map(binds, path, field):
    if not isinstance(binds, dict):
        raise ValueError(f"{path}: policy.{field} must be an object, "
                         f"got {binds!r}")
    for name, b in binds.items():
        if not isinstance(b, dict):
            raise ValueError(f"{path}: policy.{field}[{name!r}] must "
                             f"be an object, got {b!r}")
    return {k: dict(v) for k, v in binds.items()}


def _validate_allowed_plugins(plugins, path):
    if not isinstance(plugins, list) or any(
            not isinstance(p, dict)
            or not _is_non_empty_str(p.get("name"))
            or not _is_non_empty_str(p.get("origin")) for p in plugins):
        raise ValueError(f"{path}: policy.allowed_plugins must be a list "
                         f"of {{\"name\": <str>, \"origin\": <str>}}, "
                         f"got {plugins!r}")
    return [dict(p) for p in plugins]


def _validate_approved_deps(deps, path):
    if not isinstance(deps, dict) or any(
            not _is_non_empty_str(k) or not _is_non_empty_str(v)
            for k, v in deps.items()):
        raise ValueError(f"{path}: policy.approved_dependencies must be "
                         f"an object of name -> version str, got {deps!r}")
    return dict(deps)


_POLICY_KEYS = ("require_provenance", "pins", "tokenizer_bindings",
                "adapter_bindings", "allowed_plugins",
                "approved_dependencies")


def _validate_policy(policy, path):
    """Validate the registry policy block; missing sections default to
    empty (the verifier then simply has nothing to enforce there), wrong
    types are a ValueError naming the file."""
    if not isinstance(policy, dict):
        raise ValueError(f"{path}: policy must be an object, "
                         f"got {policy!r}")
    out = {"require_provenance": True, "pins": {},
           "tokenizer_bindings": {}, "adapter_bindings": {},
           "allowed_plugins": [], "approved_dependencies": {}}
    rp = policy.get("require_provenance", True)
    if not isinstance(rp, bool):
        raise ValueError(f"{path}: policy.require_provenance must be a "
                         f"bool, got {rp!r}")
    out["require_provenance"] = rp
    out["pins"] = _validate_pins(policy.get("pins", {}), path)
    for field in ("tokenizer_bindings", "adapter_bindings"):
        out[field] = _validate_bindings_map(policy.get(field, {}), path,
                                            field)
    out["allowed_plugins"] = _validate_allowed_plugins(
        policy.get("allowed_plugins", []), path)
    out["approved_dependencies"] = _validate_approved_deps(
        policy.get("approved_dependencies", {}), path)
    return out


def _validate_policy_override(doc, path):
    """Validate an approved-manifest override: only the policy sections
    PRESENT in the file are validated and returned (verify() updates the
    registry policy with exactly those — absent sections keep the
    registry's own values)."""
    unknown = sorted(set(doc) - {"schema", *_POLICY_KEYS})
    if unknown:
        raise ValueError(f"{path}: unknown approved-manifest keys "
                         f"{unknown} (allowed: {sorted(_POLICY_KEYS)})")
    out = {}
    if "require_provenance" in doc:
        rp = doc["require_provenance"]
        if not isinstance(rp, bool):
            raise ValueError(f"{path}: require_provenance must be a "
                             f"bool, got {rp!r}")
        out["require_provenance"] = rp
    if "pins" in doc:
        out["pins"] = _validate_pins(doc["pins"], path)
    for field in ("tokenizer_bindings", "adapter_bindings"):
        if field in doc:
            out[field] = _validate_bindings_map(doc[field], path, field)
    if "allowed_plugins" in doc:
        out["allowed_plugins"] = _validate_allowed_plugins(
            doc["allowed_plugins"], path)
    if "approved_dependencies" in doc:
        out["approved_dependencies"] = _validate_approved_deps(
            doc["approved_dependencies"], path)
    return out


def load_environment(path):
    """Read + validate environment.json -> {"installed": {name: ver}}.
    None when the file is absent; ValueError naming it on drift."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"{path}: cannot parse environment JSON: {e}")
    if not isinstance(doc, dict) or doc.get("schema") != ENVIRONMENT_SCHEMA:
        raise ValueError(f"{path}: schema must be {ENVIRONMENT_SCHEMA!r}, "
                         f"got {None if not isinstance(doc, dict) else doc.get('schema')!r}")
    installed = doc.get("installed")
    if not isinstance(installed, dict) or any(
            not _is_non_empty_str(k) or not _is_non_empty_str(v)
            for k, v in installed.items()):
        raise ValueError(f"{path}: installed must be an object of "
                         f"name -> version str, got {installed!r}")
    return {"installed": dict(installed)}


def load_registry(registry_dir):
    """Read + validate a registry dir -> {"dir", "artifacts", "policy",
    "ai_bom", "environment"}. ValueError naming the file on any drift."""
    rpath = os.path.join(registry_dir, "registry.json")
    if not os.path.isfile(rpath):
        raise ValueError(f"{rpath}: no such registry index file")
    try:
        with open(rpath) as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"{rpath}: cannot parse registry JSON: {e}")
    if not isinstance(doc, dict) or doc.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(f"{rpath}: schema must be {REGISTRY_SCHEMA!r}, "
                         f"got {None if not isinstance(doc, dict) else doc.get('schema')!r}")
    names = doc.get("artifacts")
    if not isinstance(names, list) \
            or any(not _is_non_empty_str(n) for n in names):
        raise ValueError(f"{rpath}: artifacts must be a list of artifact "
                         f"dir names, got {names!r}")
    bom = doc.get("ai_bom", [])
    if not isinstance(bom, list) or any(
            not isinstance(e, dict) or not _is_non_empty_str(e.get("name"))
            or not _is_sha256(e.get("sha256")) for e in bom):
        raise ValueError(f"{rpath}: ai_bom must be a list of {{\"name\": "
                         f"<str>, \"version\": <int>, \"sha256\": "
                         f"<64-hex>}}, got {bom!r}")
    return {"dir": registry_dir, "artifacts": list(names),
            "policy": _validate_policy(doc.get("policy", {}), rpath),
            "ai_bom": [dict(e) for e in bom],
            "environment": load_environment(
                os.path.join(registry_dir, "environment.json"))}


def load_approved_manifest(path):
    """--manifest override: an approved manifest (schema
    llm_attacker.ai_approved_manifest.v1). Only the policy sections
    present in the file override the registry's (verify() updates the
    registry policy with exactly those). ValueError naming the file on
    drift."""
    if not os.path.isfile(path):
        raise ValueError(f"{path}: no such approved-manifest file")
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"{path}: cannot parse approved-manifest JSON: {e}")
    if not isinstance(doc, dict) \
            or doc.get("schema") != APPROVED_MANIFEST_SCHEMA:
        raise ValueError(f"{path}: schema must be "
                         f"{APPROVED_MANIFEST_SCHEMA!r}, got "
                         f"{None if not isinstance(doc, dict) else doc.get('schema')!r}")
    return _validate_policy_override(doc, path)


_PICKLE_MARKERS = (b"__reduce__", b"__reduce_ex__", b"c__builtin__",
                   b"cbuiltins", b"cos\nsystem", b"cposix\nsystem")
_EXECUTABLE_MAGICS = ((b"\x7fELF", "ELF executable"),
                      (b"MZ", "PE/MZ executable"),
                      (b"#!", "shebang script"))


def restricted_load_gate(data, path="<payload>"):
    """Bytes-level gate of the restricted JSON loader. Returns a rejection
    reason str, or None when the bytes may be parsed as JSON. Detects
    pickle magic (0x80 + protocol byte), __reduce__/global-opcode markers,
    and executable magics — and refuses — BEFORE anything is parsed. It
    never deserializes; it only scans bytes."""
    if len(data) >= 2 and data[0] == 0x80 and 0 <= data[1] <= 9:
        return (f"{path}: pickle protocol magic (0x80 followed by "
                f"protocol byte {data[1]}) — restricted loader refuses "
                f"before parsing")
    head = data[:4096]
    for marker in _PICKLE_MARKERS:
        if marker in head:
            return (f"{path}: pickle marker {marker!r} — restricted "
                    f"loader refuses before parsing")
    for magic, label in _EXECUTABLE_MAGICS:
        if data[:len(magic)] == magic:
            return (f"{path}: {label} magic — restricted loader refuses "
                    f"before parsing")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return f"{path}: not UTF-8 — only JSON payloads are loadable"
    return None


def provenance_complete(manifest):
    prov = manifest.get("provenance") or {}
    return all(_is_non_empty_str(prov.get(k))
               for k in ("builder", "source_repo")) \
        and isinstance(prov.get("timestamp"), (int, float)) \
        and not isinstance(prov.get("timestamp"), bool)


class Verifier:
    """The registry verifier. mode="enforcing" runs the full stage
    pipeline and fails closed at the first violated stage (the rejection
    is recorded BEFORE any load); mode="permissive" accepts everything —
    the vulnerable baseline proving each arm's attack actually lands when
    verification is absent."""

    def __init__(self, policy, mode="enforcing"):
        if mode not in ("enforcing", "permissive"):
            raise ValueError(f"verifier mode must be enforcing|permissive, "
                             f"got {mode!r}")
        self.policy = policy
        self.mode = mode

    def verify_environment(self, installed):
        """Diff the installed package set against the approved dependency
        manifest. Enforcing: any drift rejects at dependency-manifest."""
        if self.mode == "permissive":
            return {"ok": True, "loaded": True, "name": "<environment>",
                    "failure_stage": None, "stage_index": None,
                    "reason": None, "checks": [
                        {"stage": "dependency-manifest", "ok": True,
                         "skipped": True}]}
        approved = self.policy.get("approved_dependencies", {})
        diffs = []
        for name in sorted(installed):
            if name not in approved:
                diffs.append(f"unapproved package {name}=={installed[name]}")
            elif approved[name] != installed[name]:
                diffs.append(f"version drift: {name} installed "
                             f"{installed[name]} != approved "
                             f"{approved[name]}")
        for name in sorted(approved):
            if name not in installed:
                diffs.append(f"approved package {name} missing from the "
                             f"installed set")
        checks = [{"stage": "dependency-manifest", "ok": not diffs,
                   "reason": "; ".join(diffs) if diffs else None}]
        if diffs:
            return {"ok": False, "loaded": False, "name": "<environment>",
                    "failure_stage": "dependency-manifest",
                    "stage_index": STAGES.index("dependency-manifest"),
                    "reason": "; ".join(diffs), "checks": checks}
        return {"ok": True, "loaded": True, "name": "<environment>",
                "failure_stage": None, "stage_index": None, "reason": None,
                "checks": checks}

    def verify_artifact(self, adir, display=None):
        """Run the stage pipeline over one artifact directory. Returns a
        dict with ok/loaded, the failure stage + index (time-to-detect),
        and per-stage checks. Enforcing fails closed at the first violated
        stage; permissive accepts everything (vulnerable baseline).
        `display` is the path used in rejection reasons (defaults to
        `adir`); callers pass a registry-relative path so reports are
        machine-independent and deterministic."""
        if self.mode == "permissive":
            return self._verify_permissive(adir)
        if display is None:
            display = adir
        checks = []

        def fail(stage, reason):
            checks.append({"stage": stage, "ok": False, "reason": reason})
            return {"ok": False, "loaded": False,
                    "failure_stage": stage, "stage_index": STAGES.index(stage),
                    "reason": reason, "checks": checks}

        def passed(stage, detail=None):
            entry = {"stage": stage, "ok": True}
            if detail:
                entry["detail"] = detail
            checks.append(entry)

        base = {"name": None, "version": None, "purpose": None,
                "payload_sha256": None, "provenance_complete": False}

        mpath = os.path.join(adir, "manifest.json")
        dman = os.path.join(display, "manifest.json")
        try:
            manifest = load_manifest(mpath)
        except ValueError as e:
            reason = str(e)
            if mpath != dman and reason.startswith(mpath):
                reason = dman + reason[len(mpath):]
            return {**base, **fail("manifest-parse", reason)}
        passed("manifest-parse")

        if self.policy.get("require_provenance", True) \
                and not provenance_complete(manifest):
            return {**base, "name": manifest["name"],
                    "version": manifest["version"],
                    "purpose": manifest["purpose"],
                    **fail("provenance",
                           f"{dman}: promotion without complete "
                           f"provenance (builder/source_repo/timestamp "
                           f"required)")}
        passed("provenance")

        expected_sig = sign_manifest(manifest)
        if not hmac.compare_digest(expected_sig,
                                   manifest.get("signature") or ""):
            return {**base, "name": manifest["name"],
                    "version": manifest["version"],
                    "purpose": manifest["purpose"],
                    "provenance_complete": provenance_complete(manifest),
                    **fail("signature",
                           f"{dman}: invalid {SIGNATURE_SCHEME} — the "
                           f"manifest was tampered with after signing or "
                           f"never signed")}
        passed("signature")

        ppath = os.path.join(adir, manifest["payload"]["file"])
        dpay = os.path.join(display, manifest["payload"]["file"])
        try:
            with open(ppath, "rb") as f:
                payload = f.read()
        except OSError as e:
            return {**base, "name": manifest["name"],
                    "version": manifest["version"],
                    "purpose": manifest["purpose"],
                    "provenance_complete": True,
                    **fail("pinned-digest",
                           f"{dpay}: cannot read payload: {e}")}
        digest_actual = sha256_hex(payload)
        base["payload_sha256"] = digest_actual
        digest_problems = []
        if digest_actual != manifest["payload"]["sha256"]:
            digest_problems.append(
                f"payload sha256 != manifest digest — bytes tampered "
                f"with after signing ({dpay})")
        pin = self.policy.get("pins", {}).get(manifest["name"])
        if pin and manifest["version"] == pin["version"] \
                and digest_actual != pin["sha256"]:
            digest_problems.append(
                f"payload sha256 != registry pin for "
                f"{manifest['name']} v{pin['version']} — artifact "
                f"substitution under the expected name")
        if digest_problems:
            return {**base, "name": manifest["name"],
                    "version": manifest["version"],
                    "purpose": manifest["purpose"],
                    "provenance_complete": True,
                    **fail("pinned-digest", "; ".join(digest_problems))}
        passed("pinned-digest")

        if pin and manifest["version"] < pin["version"]:
            return {**base, "name": manifest["name"],
                    "version": manifest["version"],
                    "purpose": manifest["purpose"],
                    "provenance_complete": True,
                    **fail("version-monotonic",
                           f"rollback: {manifest['name']} v"
                           f"{manifest['version']} is older than the "
                           f"pinned v{pin['version']} — serving it would "
                           f"re-introduce superseded vulnerabilities")}
        passed("version-monotonic")

        binding_problem = self._check_binding(manifest)
        if binding_problem:
            return {**base, "name": manifest["name"],
                    "version": manifest["version"],
                    "purpose": manifest["purpose"],
                    "provenance_complete": True,
                    **fail("binding", binding_problem)}
        passed("binding")

        plugin_problem = self._check_plugin(manifest)
        if plugin_problem:
            return {**base, "name": manifest["name"],
                    "version": manifest["version"],
                    "purpose": manifest["purpose"],
                    "provenance_complete": True,
                    **fail("plugin-allowlist", plugin_problem)}
        passed("plugin-allowlist")

        gate = restricted_load_gate(payload, dpay)
        if gate:
            return {**base, "name": manifest["name"],
                    "version": manifest["version"],
                    "purpose": manifest["purpose"],
                    "provenance_complete": True,
                    **fail("safe-serialization", gate)}
        passed("safe-serialization")

        try:
            json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as e:
            return {**base, "name": manifest["name"],
                    "version": manifest["version"],
                    "purpose": manifest["purpose"],
                    "provenance_complete": True,
                    **fail("load", f"{dpay}: payload is not valid JSON "
                                   f"for the restricted loader: {e}")}
        passed("load")
        return {**base, "name": manifest["name"],
                "version": manifest["version"],
                "purpose": manifest["purpose"],
                "provenance_complete": True,
                "ok": True, "loaded": True, "failure_stage": None,
                "stage_index": None, "reason": None, "checks": checks}

    def _check_binding(self, manifest):
        purpose, bindings = manifest["purpose"], manifest["bindings"]
        if purpose == "tokenizer":
            model = bindings.get("model")
            entry = self.policy.get("tokenizer_bindings", {}).get(model)
            if entry is None:
                return (f"tokenizer binds unregistered model {model!r} — "
                        f"no tokenizer binding policy for it")
            problems = []
            for field in ("vocab_sha256", "chat_template_sha256"):
                if bindings.get(field) != entry.get(field):
                    problems.append(
                        f"{field} {bindings.get(field)!r} != the hash "
                        f"bound to model {model!r} — tokenizer drift "
                        f"(vocab/normalization/chat-template changed under "
                        f"the same model identity)")
            return "; ".join(problems) or None
        if purpose == "adapter":
            entry = self.policy.get("adapter_bindings", {}).get(
                manifest["name"])
            if entry is None:
                return (f"adapter {manifest['name']!r} has no approved "
                        f"(model, tenant) binding in this registry")
            got = {"model": bindings.get("model"),
                   "tenant": bindings.get("tenant")}
            if got != {"model": entry.get("model"),
                       "tenant": entry.get("tenant")}:
                return (f"adapter-confusion: {manifest['name']!r} is "
                        f"bound to {got} but this registry approves it "
                        f"only for (model={entry.get('model')!r}, "
                        f"tenant={entry.get('tenant')!r})")
            return None
        if purpose == "model":
            entry = self.policy.get("tokenizer_bindings", {}).get(
                manifest["name"])
            if entry is not None \
                    and bindings.get("tokenizer") != entry.get("tokenizer"):
                return (f"model {manifest['name']!r} binds tokenizer "
                        f"{bindings.get('tokenizer')!r}, policy binds "
                        f"{entry.get('tokenizer')!r}")
        return None

    def _check_plugin(self, manifest):
        if manifest["purpose"] != "plugin":
            return None
        origin = (manifest.get("provenance") or {}).get("source_repo")
        allowed = self.policy.get("allowed_plugins", [])
        for a in allowed:
            if a["name"] == manifest["name"]:
                if a["origin"] == origin:
                    return None
                return (f"plugin name-conflict: {manifest['name']!r} is "
                        f"allowlisted from origin {a['origin']!r} but "
                        f"this artifact claims {origin!r} "
                        f"(name-squatting)")
        return (f"plugin {manifest['name']!r} (origin {origin!r}) is not "
                f"in the registry allowlist")

    def _verify_permissive(self, adir):
        """The vulnerable baseline: every stage skipped, everything loads.
        Reads the manifest leniently (metrics only); the 'load' itself is
        simulated — payloads are never parsed or executed here either."""
        info = {"name": os.path.basename(adir.rstrip(os.sep)),
                "version": None, "purpose": None, "payload_sha256": None,
                "provenance_complete": False}
        try:
            with open(os.path.join(adir, "manifest.json")) as f:
                doc = json.load(f)
            if isinstance(doc, dict):
                info["name"] = doc.get("name") or info["name"]
                info["version"] = doc.get("version")
                info["purpose"] = doc.get("purpose")
                info["provenance_complete"] = provenance_complete(doc)
                p = doc.get("payload")
                if isinstance(p, dict) and isinstance(p.get("file"), str):
                    with open(os.path.join(adir, p["file"]), "rb") as f:
                        info["payload_sha256"] = sha256_hex(f.read())
        except (OSError, json.JSONDecodeError):
            pass
        return {**info, "ok": True, "loaded": True, "failure_stage": None,
                "stage_index": None, "reason": None,
                "checks": [{"stage": s, "ok": True, "skipped": True}
                           for s in STAGES if s != "dependency-manifest"]}


def ai_bom_represented(result, ai_bom):
    """Is the verified artifact (name, version, actual payload digest)
    represented in the registry AI-BOM?"""
    if result.get("payload_sha256") is None:
        return None
    return any(e.get("name") == result.get("name")
               and e.get("version") == result.get("version")
               and e.get("sha256") == result.get("payload_sha256")
               for e in ai_bom)


def verify(registry_dir, manifest_override=None):
    """Verify a real on-disk registry with the enforcing verifier (the
    --registry path, fully implemented). Returns a report dict; schema
    drift anywhere is a descriptive ValueError naming the file."""
    reg = load_registry(registry_dir)
    policy = dict(reg["policy"])
    if manifest_override is not None:
        policy.update(load_approved_manifest(manifest_override))
    verifier = Verifier(policy, mode="enforcing")
    results = []
    for name in reg["artifacts"]:
        res = verifier.verify_artifact(os.path.join(registry_dir, name),
                                       display=name)
        res = dict(res)
        res["ai_bom_represented"] = ai_bom_represented(res, reg["ai_bom"])
        results.append(res)
    environment = None
    if reg["environment"] is not None:
        environment = verifier.verify_environment(
            reg["environment"]["installed"])
    loaded = [r for r in results if r["loaded"]]
    cov = (statistics.fmean(1.0 if r["ai_bom_represented"] else 0.0
                            for r in loaded) if loaded else None)
    prov = (statistics.fmean(1.0 if r["provenance_complete"] else 0.0
                             for r in loaded) if loaded else None)
    n_rejected = sum(1 for r in results if not r["ok"]) \
        + (0 if environment is None or environment["ok"] else 1)
    return {"registry": os.path.abspath(registry_dir),
            "n_artifacts": len(results), "artifacts": results,
            "environment": environment, "n_rejected": n_rejected,
            "ai_bom_coverage_loaded": cov,
            "provenance_completeness_loaded": prov}


def _toy_provenance(repo="repo:llm-attacker/toy-registry", ts=TOY_EPOCH):
    return {"builder": "toy-builder", "source_repo": repo, "timestamp": ts}


def _toy_weights(rng, n, tag):
    return {"format": "toy-json-weights", "tag": tag,
            "weights": [round(rng.uniform(-1.0, 1.0), 6) for _ in range(n)]}


def make_manifest(name, version, purpose, tenant, payload_bytes,
                  provenance, bindings=None, ai_bom=None,
                  payload_file="payload.json"):
    """Build a signed manifest dict (TOY signature — see TOY_SIGNING_KEY)."""
    doc = {"schema": MANIFEST_SCHEMA, "name": name, "version": version,
           "purpose": purpose, "tenant": tenant,
           "payload": {"file": payload_file,
                       "sha256": sha256_hex(payload_bytes)},
           "provenance": dict(provenance), "bindings": dict(bindings or {}),
           "ai_bom": [dict(e) for e in (ai_bom or [])]}
    doc["signature"] = sign_manifest(doc)
    return doc


def _write_artifact_dir(adir, manifest, payload_bytes):
    os.makedirs(adir, exist_ok=True)
    with open(os.path.join(adir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    with open(os.path.join(adir, manifest["payload"]["file"]), "wb") as f:
        f.write(payload_bytes)


def _toy_tokenizer_payload(rng, extra_tokens=(), normalization="lowercase",
                           template="{{ bos_token }}{{ prompt }}"):
    vocab = TOY_BASE_VOCAB + [f"tok_{rng.randrange(10 ** 6):06d}"
                              for _ in range(8)] + list(extra_tokens)
    return {"vocab": vocab, "normalization": normalization,
            "chat_template": template}


def _tokenizer_hashes(payload):
    return {"vocab_sha256": sha256_hex(canonical_json(payload["vocab"])),
            "chat_template_sha256": sha256_hex(
                payload["chat_template"].encode())}


def make_toy_registry(root, seed=0):
    """Build the pristine fixture registry on disk (deterministic in
    `seed`): 5 current artifacts (model v2, tokenizer v1, adapter v1,
    plugin v1, dataset v1), an archived model v1 (the rollback bait), the
    registry index + policy (pins, bindings, plugin allowlist, approved
    dependencies), the AI-BOM, and a matching environment.json."""
    rng = random.Random(seed)

    model_v1 = canonical_json(_toy_weights(rng, 16, "toy-model-v1"))
    model_v2 = canonical_json(_toy_weights(rng, 16, "toy-model-v2"))
    tok_payload = _toy_tokenizer_payload(rng)
    tok = canonical_json(tok_payload)
    tok_hashes = _tokenizer_hashes(tok_payload)
    adapter = canonical_json({"format": "toy-lora", "rank": 4,
                              "a": [round(rng.uniform(-1, 1), 6)
                                    for _ in range(8)],
                              "b": [round(rng.uniform(-1, 1), 6)
                                    for _ in range(8)]})
    plugin = canonical_json({"entry_point": "toy_plugin:register",
                             "permitted_surfaces": ["toy"],
                             "code_sha256": sha256_hex(b"toy-plugin-code")})
    dataset = canonical_json({"format": "toy-json-rows",
                              "rows": [{"id": i, "text": f"toy row {i}"}
                                       for i in range(4)]})
    digests = {"toy-model": sha256_hex(model_v2),
               "toy-tokenizer": sha256_hex(tok),
               "toy-adapter": sha256_hex(adapter),
               "toy-plugin": sha256_hex(plugin),
               "toy-dataset": sha256_hex(dataset)}

    def bom_entry(name, version):
        return {"name": name, "version": version, "sha256": digests[name]}

    manifests = {
        "toy-model": make_manifest(
            "toy-model", 2, "model", "tenant-a", model_v2,
            _toy_provenance(), bindings={"tokenizer": "toy-tokenizer"},
            ai_bom=[bom_entry("toy-tokenizer", 1),
                    bom_entry("toy-dataset", 1)]),
        "toy-tokenizer": make_manifest(
            "toy-tokenizer", 1, "tokenizer", "tenant-a", tok,
            _toy_provenance(),
            bindings={"model": "toy-model", **tok_hashes}),
        "toy-adapter": make_manifest(
            "toy-adapter", 1, "adapter", "tenant-a", adapter,
            _toy_provenance(),
            bindings={"model": "toy-model", "tenant": "tenant-a"},
            ai_bom=[bom_entry("toy-model", 2)]),
        "toy-plugin": make_manifest(
            "toy-plugin", 1, "plugin", "tenant-a", plugin,
            _toy_provenance(repo="repo:llm-attacker/toy-plugins"),
            bindings={"host": "llm-attacker"},
            ai_bom=[bom_entry("toy-model", 2)]),
        "toy-dataset": make_manifest(
            "toy-dataset", 1, "dataset", "tenant-a", dataset,
            _toy_provenance()),
    }
    payloads = {"toy-model": model_v2, "toy-tokenizer": tok,
                "toy-adapter": adapter, "toy-plugin": plugin,
                "toy-dataset": dataset}
    for name in manifests:
        _write_artifact_dir(os.path.join(root, name), manifests[name],
                            payloads[name])

    v1_man = make_manifest(
        "toy-model", 1, "model", "tenant-a", model_v1,
        _toy_provenance(ts=TOY_EPOCH - 86400),
        bindings={"tokenizer": "toy-tokenizer"})
    _write_artifact_dir(os.path.join(root, "archive", "toy-model-v1"),
                        v1_man, model_v1)

    policy = {"require_provenance": True,
              "pins": {name: {"version": manifests[name]["version"],
                              "sha256": digests[name]}
                       for name in manifests},
              "tokenizer_bindings": {"toy-model": {
                  "tokenizer": "toy-tokenizer", **tok_hashes}},
              "adapter_bindings": {"toy-adapter": {
                  "model": "toy-model", "tenant": "tenant-a"}},
              "allowed_plugins": [{"name": "toy-plugin",
                                   "origin": "repo:llm-attacker/"
                                              "toy-plugins"}],
              "approved_dependencies": dict(TOY_APPROVED_DEPS)}
    registry = {"schema": REGISTRY_SCHEMA,
                "artifacts": sorted(manifests),
                "ai_bom": [{"purpose": manifests[name]["purpose"],
                            **bom_entry(name, manifests[name]["version"])}
                           for name in sorted(manifests)],
                "policy": policy}
    with open(os.path.join(root, "registry.json"), "w") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
    with open(os.path.join(root, "environment.json"), "w") as f:
        json.dump({"schema": ENVIRONMENT_SCHEMA,
                   "installed": dict(TOY_APPROVED_DEPS)}, f, indent=2,
                  sort_keys=True)
    return {"digests": digests, "tokenizer_hashes": tok_hashes,
            "artifacts": sorted(manifests)}


def apply_arm(root, arm, seed):
    """Apply one attack arm to the registry copy at `root`. Returns a list
    of {"target": <artifact dir relative to root, or None for the
    environment>, "variant": <str>} — the artifacts the verifier is asked
    to load. Deterministic in (seed, arm)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r} (arms: {list(ARMS)})")
    rng = random.Random(10_000 * seed + ARMS.index(arm))

    if arm == "artifact-substitution":
        payload = canonical_json(_toy_weights(
            rng, 16, "attacker-substituted-weights"))
        with open(os.path.join(root, "toy-model", "payload.json"),
                  "wb") as f:
            f.write(payload)
        return [{"target": "toy-model", "variant": "wholesale-replacement"}]

    if arm == "digest-mismatch":
        p = os.path.join(root, "toy-model", "payload.json")
        with open(p, "rb") as f:
            data = bytearray(f.read())
        data[17] ^= 0x01
        with open(p, "wb") as f:
            f.write(bytes(data))
        return [{"target": "toy-model", "variant": "in-place-bit-flip"}]

    if arm == "unsigned-promotion":
        payload = canonical_json(_toy_weights(rng, 16, "toy-model-v3"))
        man = make_manifest("toy-model", 3, "model", "tenant-a", payload,
                            provenance={},
                            bindings={"tokenizer": "toy-tokenizer"})
        man["signature"] = ""
        _write_artifact_dir(os.path.join(root, "candidate-toy-model-v3"),
                            man, payload)
        return [{"target": "candidate-toy-model-v3",
                 "variant": "no-provenance-no-signature"}]

    if arm == "rollback":
        return [{"target": os.path.join("archive", "toy-model-v1"),
                 "variant": "serve-archived-v1"}]

    if arm == "tokenizer-drift":
        drifted = _toy_tokenizer_payload(
            rng, extra_tokens=["<|drifted|>"], normalization="nop",
            template="{{ evil_prefix }}{{ prompt }}")
        payload = canonical_json(drifted)
        man = make_manifest(
            "toy-tokenizer", 2, "tokenizer", "tenant-a", payload,
            _toy_provenance(),
            bindings={"model": "toy-model", **_tokenizer_hashes(drifted)})
        _write_artifact_dir(os.path.join(root,
                                         "candidate-toy-tokenizer-v2"),
                            man, payload)
        return [{"target": "candidate-toy-tokenizer-v2",
                 "variant": "vocab+normalization+template-drift"}]

    if arm == "adapter-confusion":
        payload = canonical_json({"format": "toy-lora", "rank": 4,
                                  "a": [round(rng.uniform(-1, 1), 6)
                                        for _ in range(8)],
                                  "b": [round(rng.uniform(-1, 1), 6)
                                        for _ in range(8)]})
        man = make_manifest(
            "toy-adapter", 2, "adapter", "tenant-b", payload,
            _toy_provenance(),
            bindings={"model": "toy-model-b", "tenant": "tenant-b"})
        _write_artifact_dir(os.path.join(root, "candidate-toy-adapter-v2"),
                            man, payload)
        return [{"target": "candidate-toy-adapter-v2",
                 "variant": "wrong-model-wrong-tenant"}]

    if arm == "plugin-origin":
        targets = []
        payload_evil = canonical_json({"entry_point": "evil:register",
                                       "code_sha256": sha256_hex(b"evil")})
        man_evil = make_manifest(
            "evil-plugin", 1, "plugin", "tenant-a", payload_evil,
            _toy_provenance(), bindings={"host": "llm-attacker"})
        _write_artifact_dir(os.path.join(root, "candidate-evil-plugin"),
                            man_evil, payload_evil)
        targets.append({"target": "candidate-evil-plugin",
                        "variant": "unapproved-name"})
        payload_squat = canonical_json(
            {"entry_point": "toy_plugin:register_evil",
             "code_sha256": sha256_hex(b"squat")})
        man_squat = make_manifest(
            "toy-plugin", 2, "plugin", "tenant-a", payload_squat,
            _toy_provenance(repo="repo:evil-fork/toy-plugins"),
            bindings={"host": "llm-attacker"})
        _write_artifact_dir(os.path.join(root, "candidate-toy-plugin-v2"),
                            man_squat, payload_squat)
        targets.append({"target": "candidate-toy-plugin-v2",
                        "variant": "name-conflict-origin-squat"})
        return targets

    if arm == "unsafe-serialization":
        man = make_manifest("toy-dataset", 2, "dataset", "tenant-a",
                            INERT_PICKLE_FIXTURE, _toy_provenance())
        _write_artifact_dir(os.path.join(root, "candidate-toy-dataset-v2"),
                            man, INERT_PICKLE_FIXTURE)
        return [{"target": "candidate-toy-dataset-v2",
                 "variant": "inert-pickle-magic-payload"}]

    epath = os.path.join(root, "environment.json")
    with open(epath) as f:
        env = json.load(f)
    env["installed"]["numpy"] = "1.26.5"
    env["installed"]["evilpkg"] = "9.9.9"
    with open(epath, "w") as f:
        json.dump(env, f, indent=2, sort_keys=True)
    return [{"target": None, "variant": "installed-set-drift"}]


def _target_result(verifier, root, reg, target):
    """Verify one target; returns (name, result-dict) with the AI-BOM
    representation metric attached."""
    if target["target"] is None:
        res = verifier.verify_environment(
            (reg["environment"] or {"installed": {}})["installed"])
        return "<environment>", {**res, "ai_bom_represented": None,
                                 "provenance_complete": None}
    res = verifier.verify_artifact(os.path.join(root, target["target"]),
                                   display=target["target"])
    return res["name"] or target["target"], {
        **res, "ai_bom_represented": ai_bom_represented(res, reg["ai_bom"])}


def _arm_cell(root, reg, arm, targets, mode, seed):
    """One cell = one arm x one verifier mode x one seed."""
    verifier = Verifier(reg["policy"], mode=mode)
    details = []
    for t in targets:
        name, res = _target_result(verifier, root, reg, t)
        details.append({
            "target": name, "variant": t.get("variant"),
            "rejected": not res["ok"], "loaded": res["loaded"],
            "failure_stage": res["failure_stage"],
            "stage_index": res["stage_index"], "reason": res["reason"],
            "ai_bom_represented": res["ai_bom_represented"],
            "provenance_complete": res["provenance_complete"]})
    n = len(details)
    n_bypassed = sum(1 for d in details if not d["rejected"])
    rejected = [d for d in details if d["rejected"]]
    first = rejected[0] if rejected else None
    loaded = [d for d in details if d["loaded"]
              and d["ai_bom_represented"] is not None]
    expected = EXPECTED_STAGE[arm]
    return {
        "experiment": EXPERIMENT_ID, "arm": arm, "seed": seed,
        "verifier": mode,
        "role": "enforcing" if mode == "enforcing"
                else "vulnerable-baseline",
        "n_targets": n, "targets": details,
        "bypass_rate": round(n_bypassed / n, 6),
        "rejected_all": n_bypassed == 0,
        "first_failure_stage": first["failure_stage"] if first else None,
        "first_failure_stage_index": first["stage_index"] if first else None,
        "n_stages_total": len(STAGES),
        "expected_stage": expected,
        "detected_at_expected_stage": all(
            d["failure_stage"] == expected for d in rejected),
        "detected_before_load": all(
            d["stage_index"] is not None and d["stage_index"]
            < LOAD_STAGE_INDEX for d in rejected),
        "ai_bom_coverage_loaded": (
            round(statistics.fmean(
                1.0 if d["ai_bom_represented"] else 0.0
                for d in loaded), 6) if loaded else None),
        "provenance_completeness_loaded": (
            round(statistics.fmean(
                1.0 if d["provenance_complete"] else 0.0
                for d in loaded), 6) if loaded else None),
        "provenance_completeness_targets": round(statistics.fmean(
            1.0 if d["provenance_complete"] else 0.0
            for d in details), 6),
        "note": ARM_INFO[arm]}


def _negative_control_cell(root, seed):
    """The untouched registry must verify cleanly (0 false rejections)."""
    report = verify(root)
    false_rejections = [
        {"target": r["name"], "stage": r["failure_stage"],
         "reason": r["reason"]}
        for r in report["artifacts"] if not r["ok"]]
    env = report["environment"]
    if env is not None and not env["ok"]:
        false_rejections.append({"target": "<environment>",
                                 "stage": env["failure_stage"],
                                 "reason": env["reason"]})
    return {"experiment": EXPERIMENT_ID, "arm": "negative-control",
            "seed": seed, "verifier": "enforcing",
            "role": "negative-control",
            "n_artifacts": report["n_artifacts"],
            "environment_checked": env is not None,
            "n_false_rejections": len(false_rejections),
            "false_rejections": false_rejections,
            "ai_bom_coverage_loaded": report["ai_bom_coverage_loaded"],
            "provenance_completeness_loaded":
                report["provenance_completeness_loaded"]}


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS),
                    help="attack arm(s) to run (default: all)")
    ap.add_argument("--registry", default=None,
                    help=f"verify a real on-disk registry dir "
                         f"(registry.json {REGISTRY_SCHEMA} + one "
                         f"manifest.json {MANIFEST_SCHEMA} per artifact)")
    ap.add_argument("--manifest", default=None,
                    help=f"approved-manifest policy override "
                         f"({APPROVED_MANIFEST_SCHEMA}) applied over the "
                         f"registry's embedded policy")
    ap.add_argument("--work-dir", default=None,
                    help="toy: build the fixture registries here (default: "
                         "a fresh temp dir)")
    ap.add_argument("--evidence", action="store_true",
                    help="wrap the run in an EvidenceLogger (requires "
                         "--output): the artifact write is recorded as a "
                         "chained event and the evidence manifest is "
                         "finalized alongside")
    return ap


def _threat_model_doc():
    return threat_model.normalize({
        "schema": threat_model.SCHEMA_V2,
        "summary": "malicious supply-chain actor (compromised registry, "
                   "distribution channel, or build pipeline) substitutes, "
                   "tampers with, rolls back, or confuses AI artifacts and "
                   "extension points — models, tokenizers, adapters, "
                   "datasets, plugins, packages — to get unverified bytes "
                   "loaded into training/inference serving",
        "scope": {"use_case": "split-training",
                  "deployment": "multi-tenant",
                  "in_scope": ["AI-artifact registry verification for "
                               "split-training and split-inference stacks",
                               "promotion, rollback, binding, plugin, "
                               "serialization, and dependency gates"],
                  "out_of_scope": ["registry-server protocols and real "
                                   "PKI/transparency logs",
                                   "reproducible-build attestation",
                                   "container image layer scanning",
                                   "real SBOM (CycloneDX/SPDX) import"]},
        "actor": {"role": "insider", "intent": "malicious",
                  "vantage": ["write access to a registry/distribution "
                              "channel or build pipeline",
                              "promotion path for new artifact versions"],
                  "authorization": "fixture registries and evaluator-owned "
                                   "--registry dirs only",
                  "privileges": ["artifact bytes/manifest replacement",
                                 "for the signed-malicious arms: the toy "
                                 "pipeline key (models a compromised "
                                 "builder)"],
                  "knowledge": ["registry layout, expected artifact names "
                                "and versions",
                                "tokenizer/adapter binding conventions"],
                  "capabilities": ["artifact substitution and in-place "
                                   "tamper", "unsigned promotion",
                                   "rollback to archived versions",
                                   "tokenizer/adapter identity confusion",
                                   "plugin name-squatting",
                                   "unsafe-serialization payloads",
                                   "dependency drift"],
                  "resources": ["fixture registries on local disk"],
                  "persistence": "cross-session"},
        "target": {"assets": ["base models/checkpoints",
                              "tokenizers/configs",
                              "LoRA adapters + quantization artifacts",
                              "training/eval/RAG datasets",
                              "attack plugins/surface providers",
                              "python packages/container images",
                              "capture tensors/serialized bundles"],
                   "components": ["artifact registry index + policy",
                                  "signed manifests", "AI-BOM",
                                  "restricted serialization loader",
                                  "dependency manifest"],
                   "data_stores": ["registry directory", "archive dir",
                                   "environment.json"],
                   "data_flows": ["builder -> registry -> promotion -> "
                                  "serving load"],
                   "trust_boundaries": ["registry -> serving stack",
                                        "tenant -> shared registry",
                                        "untrusted bytes -> parser"],
                   "entry_points": ["--toy fixture registry",
                                    "--registry real registry dir"],
                   "exit_points": ["verification report (accept/reject + "
                                   "stage)"]},
        "scenario": {"preconditions": ["registry on disk with signed "
                                       "manifests and a policy",
                                       "a verifier that enforces (or, "
                                       "baseline, skips) the policy"],
                     "lifecycle_stage": ["supply-chain", "deployment"],
                     "security_properties": ["integrity", "authenticity"],
                     "attack_steps": ["build pristine registry",
                                      "apply one arm's mutation",
                                      "verify with enforcing + permissive "
                                      "verifiers",
                                      "record bypass rate, detection "
                                      "stage, AI-BOM/provenance metrics"],
                     "technical_impact": ["unverified or rolled-back "
                                          "artifacts loaded into serving",
                                          "untrusted bytes reaching a "
                                          "deserializer",
                                          "unapproved plugins/dependencies "
                                          "executing"],
                     "business_impact": ["supply-chain compromise of "
                                         "training/inference fleets",
                                         "cross-tenant model/adapter "
                                         "confusion",
                                         "audit failure"]},
        "controls": {"assumed": ["pinned digests + (toy) signatures",
                                 "provenance required for promotion",
                                 "monotonic versions", "tokenizer/adapter "
                                 "bindings", "plugin allowlist",
                                 "restricted serialization loader",
                                 "approved dependency manifest"],
                     "tested": list(ARMS),
                     "bypassed": ["the permissive verifier (vulnerable "
                                  "baseline) bypasses all of them by "
                                  "design"],
                     "detective": ["stage-indexed verification report",
                                   "per-cell journal"],
                     "status": "open"},
        "mappings": {"owasp_llm_2026": ["LLM04", "LLM05"],
                     "cwe": ["CWE-494", "CWE-502"]},
        "evaluation": {"success_metrics": ["verification bypass rate per "
                                           "arm (enforcing 0.0, "
                                           "permissive ~1.0)",
                                           "% loaded assets in the AI-BOM",
                                           "provenance completeness of "
                                           "loaded assets",
                                           "time-to-detect (stage index)",
                                           "failure-stage distribution"],
                       "baseline": "negative control: the untouched "
                                   "registry verifies cleanly (0 false "
                                   "rejections); permissive verifier "
                                   "bypasses every arm (attack lands)",
                       "budget": {"toy": "5 pinned artifacts + 1 archive "
                                         "+ per-arm candidates; a few KB "
                                         "of JSON",
                                  "real": "--registry verifies in place, "
                                          "read-only"},
                       "limitations": ["TOY HMAC is not real PKI",
                                       "JSON stand-in payloads; real "
                                       "deserializers are the integration "
                                       "point behind the load stage",
                                       "inert pickle fixture is never "
                                       "unpickled; permissive loads are "
                                       "simulated",
                                       "no reproducible-build attestation "
                                       "or SBOM import"],
                       "coverage_status": "covered"}})


def _make_artifact(args, source, extra_config=None, evidence=None):
    config = {"attack": EXPERIMENT_ID, "mode": getattr(args, "mode", None),
              "source": source, "arms": getattr(args, "arms", None),
              "seed": args.seed, "registry_schema": REGISTRY_SCHEMA,
              "manifest_schema": MANIFEST_SCHEMA,
              "signature_scheme": SIGNATURE_SCHEME,
              "stages": list(STAGES)}
    if extra_config:
        config.update(extra_config)
    return artifacts.make_artifact(
        "dtraining.attacker.supply_chain.v1", config,
        _threat_model_doc(),
        interpretation="secure behavior: every attacked cell is rejected "
                       "by the enforcing verifier (bypass_rate 0.0) at the "
                       "arm's expected stage, BEFORE the load stage "
                       "(substitution/tamper at pinned-digest — the bytes "
                       "are hashed, never parsed); the permissive baseline "
                       "bypasses every arm (~1.0, proving the attacks "
                       "land); the untouched registry verifies cleanly "
                       "(0 false rejections). Loaded permissive assets "
                       "drop out of the AI-BOM and provenance records — "
                       "that gap is the audit signal.",
        evidence=evidence)


def run_toy(args, evidence=None):
    if args.quick:
        args.seeds = args.seeds[:1]
    out = _make_artifact(args, "toy", {
        "toy": True, "seeds": args.seeds,
        "work_dir": args.work_dir or "<tempdir>"}, evidence=evidence)
    work = args.work_dir or tempfile.mkdtemp(prefix=EXPERIMENT_ID + "_")
    per_arm = {arm: {"enforcing": [], "permissive": []}
               for arm in args.arms}
    nc_cells = []
    for seed in args.seeds:
        fseed = args.seed + 1000 * seed
        pristine = os.path.join(work, f"seed-{seed}", "pristine")
        make_toy_registry(pristine, fseed)
        try:
            nc = _negative_control_cell(pristine, seed)
        except Exception as e:
            journal_error(args.output, EXPERIMENT_ID,
                          {"arm": "negative-control", "seed": seed}, e)
        else:
            nc_cells.append(nc)
            artifacts.append_jsonl(args.output, nc)
            out["results"].append(nc)
            print(f"[{EXPERIMENT_ID}] negative-control seed={seed}: "
                  f"artifacts={nc['n_artifacts']} "
                  f"false_rejections={nc['n_false_rejections']} "
                  f"ai_bom_coverage={nc['ai_bom_coverage_loaded']}")
        for arm in args.arms:
            aroot = os.path.join(work, f"seed-{seed}", f"attack-{arm}")
            shutil.copytree(pristine, aroot)
            try:
                targets = apply_arm(aroot, arm, fseed)
                areg = load_registry(aroot)
            except Exception as e:
                journal_error(args.output, EXPERIMENT_ID,
                              {"arm": arm, "seed": seed}, e)
                continue
            for mode in ("enforcing", "permissive"):
                try:
                    rec = _arm_cell(aroot, areg, arm, targets, mode, seed)
                except Exception as e:
                    journal_error(args.output, EXPERIMENT_ID,
                                  {"arm": arm, "seed": seed,
                                   "verifier": mode}, e)
                    continue
                per_arm[arm][mode].append(rec)
                artifacts.append_jsonl(args.output, rec)
                out["results"].append(rec)
                stage = rec["first_failure_stage"] or "none (loaded)"
                print(f"[{EXPERIMENT_ID}] {arm} seed={seed} "
                      f"verifier={mode}: bypass_rate="
                      f"{rec['bypass_rate']:.2f} stage={stage} "
                      f"({rec['role']})")
    for arm in args.arms:
        enf, per = per_arm[arm]["enforcing"], per_arm[arm]["permissive"]
        row = {"experiment": EXPERIMENT_ID, "arm": arm,
               "n_seeds": len(enf)}
        if enf:
            row["bypass_rate_enforcing"] = round(statistics.fmean(
                r["bypass_rate"] for r in enf), 6)
            row["enforce_stage_distribution"] = {
                s: sum(1 for r in enf if r["first_failure_stage"] == s)
                for s in STAGES
                if any(r["first_failure_stage"] == s for r in enf)}
            idx = [r["first_failure_stage_index"] for r in enf
                   if r["first_failure_stage_index"] is not None]
            row["time_to_detect_stage_index_mean"] = (
                round(statistics.fmean(idx), 3) if idx else None)
            row["detected_at_expected_stage"] = all(
                r["detected_at_expected_stage"] for r in enf)
            row["detected_before_load"] = all(
                r["detected_before_load"] for r in enf)
        if per:
            row["bypass_rate_permissive"] = round(statistics.fmean(
                r["bypass_rate"] for r in per), 6)
            loaded_cov = [r["ai_bom_coverage_loaded"] for r in per
                          if r["ai_bom_coverage_loaded"] is not None]
            row["ai_bom_coverage_loaded_permissive"] = (
                round(statistics.fmean(loaded_cov), 6)
                if loaded_cov else None)
            loaded_prov = [r["provenance_completeness_loaded"] for r in per
                           if r["provenance_completeness_loaded"] is not None]
            row["provenance_completeness_loaded_permissive"] = (
                round(statistics.fmean(loaded_prov), 6)
                if loaded_prov else None)
        out["summary"].append(row)
    out["summary"].append({
        "experiment": EXPERIMENT_ID, "arm": "all",
        "n_seeds": len(args.seeds),
        "negative_control_false_rejections": sum(
            c["n_false_rejections"] for c in nc_cells),
        "ai_bom_coverage_loaded_control": (
            nc_cells[-1]["ai_bom_coverage_loaded"] if nc_cells else None),
        "provenance_completeness_loaded_control": (
            nc_cells[-1]["provenance_completeness_loaded"]
            if nc_cells else None)})
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def run_registry(args, evidence=None):
    try:
        report = verify(args.registry, manifest_override=args.manifest)
    except ValueError as e:
        raise SystemExit(f"[{EXPERIMENT_ID}] {e}")
    out = _make_artifact(args, "registry", {
        "registry": args.registry, "manifest_override": args.manifest},
        evidence=evidence)
    n_loaded_in_bom = 0
    for res in report["artifacts"]:
        rec = {"experiment": EXPERIMENT_ID, "arm": "registry-verify",
               "role": "real-registry", "target": res["name"],
               "version": res["version"], "purpose": res["purpose"],
               "rejected": not res["ok"], "loaded": res["loaded"],
               "failure_stage": res["failure_stage"],
               "stage_index": res["stage_index"], "reason": res["reason"],
               "detected_before_load": (
                   res["stage_index"] is not None
                   and res["stage_index"] < LOAD_STAGE_INDEX),
               "ai_bom_represented": res["ai_bom_represented"],
               "provenance_complete": res["provenance_complete"]}
        artifacts.append_jsonl(args.output, rec)
        out["results"].append(rec)
        if rec["loaded"] and rec["ai_bom_represented"]:
            n_loaded_in_bom += 1
        print(f"[{EXPERIMENT_ID}] registry {res['name']} "
              f"v{res['version']} ({res['purpose']}): "
              + (f"REJECTED at {res['failure_stage']} — {res['reason']}"
                 if not res["ok"] else "verified"))
    env = report["environment"]
    if env is not None:
        rec = {"experiment": EXPERIMENT_ID, "arm": "registry-verify",
               "role": "real-registry", "target": "<environment>",
               "rejected": not env["ok"], "loaded": env["loaded"],
               "failure_stage": env["failure_stage"],
               "stage_index": env["stage_index"], "reason": env["reason"],
               "detected_before_load": (
                   env["stage_index"] is not None
                   and env["stage_index"] < LOAD_STAGE_INDEX)}
        artifacts.append_jsonl(args.output, rec)
        out["results"].append(rec)
        print(f"[{EXPERIMENT_ID}] registry <environment>: "
              + (f"REJECTED at {env['failure_stage']} — {env['reason']}"
                 if not env["ok"] else "verified"))
    out["summary"].append({
        "experiment": EXPERIMENT_ID, "arm": "registry-verify",
        "registry": report["registry"],
        "n_artifacts": report["n_artifacts"],
        "n_rejected": report["n_rejected"],
        "ai_bom_coverage_loaded": report["ai_bom_coverage_loaded"],
        "provenance_completeness_loaded":
            report["provenance_completeness_loaded"]})
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def _run(args, evidence=None):
    if args.toy:
        return run_toy(args, evidence=evidence)
    if args.registry:
        return run_registry(args, evidence=evidence)
    raise SystemExit(f"[{EXPERIMENT_ID}] pick an input path: --toy "
                     f"(fixture registry with ground truth) or --registry "
                     f"DIR (verify a real on-disk registry, schema "
                     f"{REGISTRY_SCHEMA}; --manifest FILE overrides the "
                     f"policy).")


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

    print("supply-chain registry verification (pure python):")

    man = make_manifest("t", 1, "model", "tenant-a", b"{}",
                        _toy_provenance())
    check("TOY signature round-trips", sign_manifest(man)
          == man["signature"])
    tampered = dict(man, version=2)
    check("tampered manifest no longer verifies",
          sign_manifest(tampered) != man["signature"])

    check("inert pickle fixture refused BEFORE parsing",
          restricted_load_gate(INERT_PICKLE_FIXTURE) is not None)
    check("protocol-0 magic refused",
          restricted_load_gate(b"\x80\x00}") is not None)
    check("__reduce__ marker refused",
          restricted_load_gate(b'{"x": "__reduce__"}') is not None)
    check("ELF magic refused",
          restricted_load_gate(b"\x7fELF....") is not None)
    check("non-UTF8 refused",
          restricted_load_gate(b"\xff\xfe{}") is not None)
    check("clean JSON admitted by the gate",
          restricted_load_gate(b'{"weights": [1, 2]}') is None)

    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, "pristine")
        make_toy_registry(root, 42)
        report = verify(root)
        check("pristine registry verifies cleanly (negative control: 0 "
              "false rejections)", report["n_rejected"] == 0)
        check("pristine registry: all 5 artifacts + environment loaded",
              len(report["artifacts"]) == 5
              and all(r["loaded"] for r in report["artifacts"])
              and report["environment"]["ok"])
        check("pristine registry: AI-BOM coverage and provenance "
              "completeness of loaded assets are 1.0",
              report["ai_bom_coverage_loaded"] == 1.0
              and report["provenance_completeness_loaded"] == 1.0)

        root2 = os.path.join(d, "pristine2")
        make_toy_registry(root2, 42)
        check("toy registry build is deterministic",
              open(os.path.join(root, "registry.json"), "rb").read()
              == open(os.path.join(root2, "registry.json"), "rb").read()
              and open(os.path.join(root, "toy-model", "manifest.json"),
                       "rb").read()
              == open(os.path.join(root2, "toy-model", "manifest.json"),
                      "rb").read())

        for arm in ARMS:
            aroot = os.path.join(d, f"arm-{arm}")
            shutil.copytree(root, aroot)
            targets = apply_arm(aroot, arm, 42)
            reg = load_registry(aroot)
            enf = _arm_cell(aroot, reg, arm, targets, "enforcing", 0)
            per = _arm_cell(aroot, reg, arm, targets, "permissive", 0)
            check(f"{arm}: enforcing rejects all targets at "
                  f"{EXPECTED_STAGE[arm]} (before load)",
                  enf["rejected_all"]
                  and enf["detected_at_expected_stage"]
                  and enf["detected_before_load"])
            check(f"{arm}: permissive baseline is bypassed (attack "
                  f"lands)", per["bypass_rate"] == 1.0)
        sub = _arm_cell(os.path.join(d, "arm-artifact-substitution"),
                        load_registry(os.path.join(
                            d, "arm-artifact-substitution")),
                        "artifact-substitution",
                        [{"target": "toy-model",
                          "variant": "wholesale-replacement"}],
                        "enforcing", 0)
        check("substitution is caught at the digest stage, BEFORE the "
              "bytes are parsed",
              sub["first_failure_stage"] == "pinned-digest"
              and sub["first_failure_stage_index"]
              < STAGES.index("safe-serialization") < LOAD_STAGE_INDEX)
        check("permissive world loads assets the AI-BOM does not "
              "represent (audit gap is measurable)",
              _arm_cell(os.path.join(d, "arm-artifact-substitution"),
                        load_registry(os.path.join(
                            d, "arm-artifact-substitution")),
                        "artifact-substitution",
                        [{"target": "toy-model",
                          "variant": "wholesale-replacement"}],
                        "permissive", 0)["ai_bom_coverage_loaded"] == 0.0)

        bad = os.path.join(d, "badreg")
        os.makedirs(bad)
        with open(os.path.join(bad, "registry.json"), "w") as f:
            json.dump({"schema": "wrong.v0"}, f)
        try:
            verify(bad)
            check("registry schema drift rejected naming the file", False)
        except ValueError as e:
            check("registry schema drift rejected naming the file",
                  "registry.json" in str(e))
        try:
            verify(os.path.join(d, "missing-registry"))
            check("missing registry rejected naming the file", False)
        except ValueError as e:
            check("missing registry rejected naming the file",
                  "registry.json" in str(e))
        badman = os.path.join(d, "badman.json")
        with open(badman, "w") as f:
            f.write("{not json")
        try:
            load_manifest(badman)
            check("malformed manifest rejected naming the file", False)
        except ValueError as e:
            check("malformed manifest rejected naming the file",
                  "badman.json" in str(e))

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
