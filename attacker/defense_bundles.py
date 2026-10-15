"""Deployed-defense bundle adapter contract (llm_attacker.defense_bundle.v1).

This module is THE adapter contract between an external split-training
defense framework's real exported artifacts and the existing attack
batteries. It is an ENVELOPE, not a parallel format: the payload schemas
(representation_privacy.v1, view_privacy.v1,
llm_attacker.replica_observations.v1, llm_attacker.replica_truth.v1, ...)
stay untouched and keep their own frozen validators; the envelope only
adds provenance, component resolution (inline JSON or bundle-relative
paths with optional sha256 pinning), and dispatch to those validators.

Envelope:
    {schema: "llm_attacker.defense_bundle.v1",
     provenance: {framework, version, exported_at, contact?},
     components: [{id, kind, inline?|path?, sha256?}]}

Each component carries exactly one payload: ``inline`` (a JSON object) or
``path`` (relative to the bundle directory). ``sha256`` optionally pins the
payload bytes (canonical JSON for inline payloads, file bytes for paths).

Standing caveat: adapter acceptance means format conformance only;
fixture-level positives do not falsify deployed defenses.

CLI (stdlib-only):
    python -m attacker.defense_bundles validate BUNDLE.json
    python -m attacker.defense_bundles materialize BUNDLE.json --out dir/
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from .evidence import file_sha256

BUNDLE_SCHEMA = "llm_attacker.defense_bundle.v1"
MANIFEST_SCHEMA = "llm_attacker.defense_bundle_manifest.v1"

KINDS = (
    "representation_privacy.v1",
    "view_privacy.v1",
    "llm_attacker.replica_observations.v1",
    "llm_attacker.replica_truth.v1",
    "phase_manifest.v1",
    "kdf_metadata.v1",
    "decoy_pool.v1",
    "view_layout.v1",
)

_TARGETS = {
    "representation_privacy.v1": ("representation-privacy", "--bundle"),
    "view_privacy.v1": ("view-privacy", "--bundle"),
    "llm_attacker.replica_observations.v1": ("replica-integrity", "--captures"),
    "llm_attacker.replica_truth.v1": ("replica-integrity", "--truth"),
}


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonempty_str(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _check_sha256(component, digest, ident):
    expected = component.get("sha256")
    if expected is None:
        return
    if not isinstance(expected, str) or len(expected) != 64 or any(
            c not in "0123456789abcdefABCDEF" for c in expected):
        raise ValueError(f"component {ident!r}: sha256 must be a 64-char hex digest")
    if expected.lower() != digest:
        raise ValueError(
            f"component {ident!r}: sha256 mismatch (expected {expected.lower()}, got {digest})")


def _finite_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _validate_kdf_metadata(payload):
    if (not isinstance(payload, dict) or payload.get("schema") != "kdf_metadata.v1"
        or not set(payload) <= {"schema", "algorithm", "domain", "master_bits", "epoch"}):
        raise ValueError("invalid kdf_metadata.v1 schema")
    _nonempty_str(payload.get("algorithm"), "kdf_metadata.algorithm")
    _nonempty_str(payload.get("domain"), "kdf_metadata.domain")
    bits = payload.get("master_bits")
    if not isinstance(bits, int) or isinstance(bits, bool) or bits < 1:
        raise ValueError("kdf_metadata.master_bits must be a positive integer")
    epoch = payload.get("epoch")
    if epoch is not None and (not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0):
        raise ValueError("kdf_metadata.epoch must be a nonnegative integer")
    return payload


def _validate_decoy_pool(payload):
    if (not isinstance(payload, dict) or payload.get("schema") != "decoy_pool.v1"
        or not set(payload) <= {"schema", "pool", "honest_labels", "recycled_from"}):
        raise ValueError("invalid decoy_pool.v1 schema")
    pool = payload.get("pool")
    if (not isinstance(pool, list) or not pool
        or any(not isinstance(row, list) or not row
               or not all(_finite_number(v) for v in row) for row in pool)):
        raise ValueError("decoy_pool.pool must be a nonempty list of finite numeric rows")
    labels = payload.get("honest_labels")
    if (not isinstance(labels, list) or len(labels) != len(pool)
        or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in labels)):
        raise ValueError("decoy_pool.honest_labels must be nonnegative integers matching pool rows")
    if "recycled_from" in payload:
        _nonempty_str(payload["recycled_from"], "decoy_pool.recycled_from")
    return payload


def _validate_view_layout(payload):
    if (not isinstance(payload, dict) or payload.get("schema") != "view_layout.v1"
        or set(payload) != {"schema", "n_views", "coords"}):
        raise ValueError("invalid view_layout.v1 schema")
    n_views, coords = payload["n_views"], payload["coords"]
    if not isinstance(n_views, int) or isinstance(n_views, bool) or n_views < 1:
        raise ValueError("view_layout.n_views must be a positive integer")
    if (not isinstance(coords, list) or len(coords) != n_views
        or any(not isinstance(view, list) or not view
               or any(not isinstance(j, int) or isinstance(j, bool) or j < 0 for j in view)
               or len(set(view)) != len(view) for view in coords)):
        raise ValueError("view_layout.coords must be n_views nonempty lists of unique nonnegative integers")
    return payload


def _validate_phase_manifest(payload):
    if (not isinstance(payload, dict) or payload.get("schema") != "phase_manifest.v1"
        or set(payload) != {"schema", "phase_provenance", "phase_records"}):
        raise ValueError("invalid phase_manifest.v1 schema")
    if not isinstance(payload["phase_provenance"], dict):
        raise ValueError("phase_manifest.phase_provenance must be an object")
    from .attacks import representation_privacy
    representation_privacy.validate_phase({
        "phase_provenance": payload["phase_provenance"],
        "phase_records": payload["phase_records"],
        "calibration": []})
    return payload


def validate_component(kind, payload):
    """Dispatch one payload to the existing validator for its kind."""
    if kind == "representation_privacy.v1":
        from .attacks import representation_privacy
        representation_privacy.validate_bundle(payload)
        return payload
    if kind == "view_privacy.v1":
        from .attacks import view_privacy
        view_privacy.evaluate_bundle(payload)
        return payload
    if kind == "llm_attacker.replica_observations.v1":
        from .attacks import replica_integrity
        return replica_integrity.validate_observations(payload)
    if kind == "llm_attacker.replica_truth.v1":
        from .attacks import replica_integrity
        return replica_integrity.validate_truth(payload)
    if kind == "phase_manifest.v1":
        return _validate_phase_manifest(payload)
    if kind == "kdf_metadata.v1":
        return _validate_kdf_metadata(payload)
    if kind == "decoy_pool.v1":
        return _validate_decoy_pool(payload)
    if kind == "view_layout.v1":
        return _validate_view_layout(payload)
    raise ValueError(f"unknown component kind: {kind!r}")


def _resolve_component(component, bundle_dir):
    ident = component["id"]
    has_inline, has_path = "inline" in component, "path" in component
    if has_inline == has_path:
        raise ValueError(f"component {ident!r}: supply exactly one of inline or path")
    if has_inline:
        payload = component["inline"]
        if not isinstance(payload, dict):
            raise ValueError(f"component {ident!r}: inline payload must be a JSON object")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        _check_sha256(component, hashlib.sha256(canonical).hexdigest(), ident)
        return payload
    rel = component["path"]
    _nonempty_str(rel, f"component {ident!r}: path")
    candidate = (bundle_dir / rel).resolve()
    if Path(rel).is_absolute() or bundle_dir not in candidate.parents and candidate != bundle_dir:
        raise ValueError(f"component {ident!r}: path must stay inside the bundle directory")
    if not candidate.is_file():
        raise ValueError(f"component {ident!r}: payload file not found: {rel}")
    _check_sha256(component, file_sha256(candidate), ident)
    payload = json.loads(candidate.read_text(), object_pairs_hook=_strict_object)
    if not isinstance(payload, dict):
        raise ValueError(f"component {ident!r}: payload file must contain a JSON object")
    return payload


def load_bundle(path):
    """Strict-parse and fully validate a defense bundle envelope."""
    bundle_path = Path(path)
    if not bundle_path.is_file():
        raise ValueError(f"bundle file not found: {path}")
    bundle = json.loads(bundle_path.read_text(), object_pairs_hook=_strict_object)
    if not isinstance(bundle, dict):
        raise ValueError("defense bundle must be a JSON object")
    if bundle.get("schema") != BUNDLE_SCHEMA:
        raise ValueError(f"unsupported defense bundle schema (expected {BUNDLE_SCHEMA})")
    provenance = bundle.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("defense bundle requires a provenance object")
    if not set(provenance) <= {"framework", "version", "exported_at", "contact"}:
        raise ValueError("provenance carries unknown fields")
    for key in ("framework", "version", "exported_at"):
        _nonempty_str(provenance.get(key), f"provenance.{key}")
    if "contact" in provenance:
        _nonempty_str(provenance["contact"], "provenance.contact")
    components = bundle.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("defense bundle requires a nonempty components list")
    bundle_dir = bundle_path.resolve().parent
    seen, resolved = set(), []
    for component in components:
        if not isinstance(component, dict):
            raise ValueError("each component must be an object")
        if not set(component) <= {"id", "kind", "inline", "path", "sha256"}:
            raise ValueError("component carries unknown fields")
        ident = component.get("id")
        _nonempty_str(ident, "component id")
        if ident in seen:
            raise ValueError(f"duplicate component id: {ident!r}")
        seen.add(ident)
        kind = component.get("kind")
        if kind not in KINDS:
            raise ValueError(f"unknown component kind: {kind!r}")
        entry = {"id": ident, "kind": kind,
                 "payload": validate_component(kind, _resolve_component(component, bundle_dir))}
        if "path" in component:
            entry["path"] = component["path"]
        resolved.append(entry)
    return {"schema": bundle["schema"], "provenance": provenance,
            "components": resolved,
            "bundle_sha256": file_sha256(bundle_path)}


_SUFFIXES = {
    "representation_privacy.v1": ".representation_privacy.json",
    "view_privacy.v1": ".view_privacy.json",
    "llm_attacker.replica_observations.v1": ".captures.json",
    "llm_attacker.replica_truth.v1": ".truth.json",
}


def materialize(bundle_path, out_dir):
    """Write each component as the exact file its target battery consumes.

    Returns a manifest mapping component id -> target attack + flags.
    """
    bundle = load_bundle(bundle_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    components = {}
    for entry in bundle["components"]:
        ident, kind, payload = entry["id"], entry["kind"], entry["payload"]
        suffix = _SUFFIXES.get(kind, f".{kind.removesuffix('.v1')}.json")
        target_path = out / f"{ident}{suffix}"
        target_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        target, flag = _TARGETS.get(kind, ("metadata", None))
        components[ident] = {"kind": kind, "file": str(target_path), "target": target,
                             "flags": [flag, str(target_path)] if flag else [],
                             "sha256": file_sha256(target_path)}
    manifest = {"schema": MANIFEST_SCHEMA, "bundle": str(bundle_path),
                "bundle_sha256": bundle["bundle_sha256"],
                "provenance": bundle["provenance"], "components": components,
                "caveat": "adapter acceptance means format conformance only; "
                          "fixture-level positives do not falsify deployed defenses"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="validate a defense bundle envelope").add_argument("bundle")
    mat = sub.add_parser("materialize", help="materialize battery-consumable files")
    mat.add_argument("bundle")
    mat.add_argument("--out", required=True)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        bundle = load_bundle(args.bundle)
        print(json.dumps({"schema": bundle["schema"],
                          "components": [c["id"] for c in bundle["components"]],
                          "bundle_sha256": bundle["bundle_sha256"]}, indent=2))
        return 0
    manifest = materialize(args.bundle, args.out)
    print(json.dumps({k: v["target"] for k, v in manifest["components"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
