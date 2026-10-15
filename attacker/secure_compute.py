#!/usr/bin/env python3
"""Shared confidential-compute execution contract and admission gate.

This module is deliberately vendor-neutral.  It produces a finite case plan,
can drive an evaluator-supplied JSON-lines process adapter, validates the
returned records strictly, and applies zero-tolerance admission rules.  A
vendor adapter is the only component that should contain cloud credentials or
SDK calls.

The contract keeps trusted diagnostic observations distinct from the external
attacker view.  It never treats an unavailable host trace as a pass.
"""

import hashlib
import json
import math
import os
import shlex
import stat
import subprocess

from .surfaces import SurfaceProvider, register_surface

CASE_SCHEMA = "llm_attacker.secure_compute_case.v1"
RESULT_SCHEMA = "llm_attacker.secure_compute_result.v1"
PLAN_SCHEMA = "llm_attacker.secure_compute_plan.v1"
GATE_SCHEMA = "llm_attacker.secure_compute_gate.v1"
MANIFEST_SCHEMA = "llm_attacker.secure_compute_deployment.v1"
EVIDENCE_MANIFEST_SCHEMA = "llm_attacker.secure_compute_evidence_manifest.v1"

PROFILES = ("tee", "fhe", "hybrid")
VIEWS = ("external_attacker", "trusted_diagnostic")
RESULT_STATUSES = ("observed", "unavailable", "error")

ATTESTATION_ARMS = (
    "valid", "wrong-measurement", "modified-quote", "stale-collateral",
    "replayed-nonce", "wrong-machine-relay", "endpoint-key-substitution",
    "channel-binding-mismatch",
)
PROTOCOL_ARMS = (
    "clean-control", "replay", "reorder", "duplicate", "truncate",
    "cross-session-swap",
    "downgrade", "tamper", "stale-step-result",
    "request-result-substitution",
)
COMPUTATION_ARMS = (
    "correct-result-control", "stale-weights", "substituted-weights",
    "skipped-operator",
    "scaled-gradient", "zeroed-gradient", "replayed-output",
    "incorrect-result",
)
BOUNDARY_PHASES = (
    "forward", "backward", "update", "cpu-gpu-staging", "ipc", "vram",
    "logs", "checkpoints", "dumps", "environment", "telemetry",
)
HOST_TRACE_ARMS = (
    "packet-shape", "timing", "performance-counters", "access-trace",
    "crash-restart",
)
FHE_ARMS = (
    "malformed-ciphertext", "modified-ciphertext", "replayed-ciphertext",
    "wrong-context", "extra-slot", "cross-tenant-context",
    "production-feedback", "leaky-positive-control",
)

CASE_KEYS = {
    "schema", "case_id", "category", "arm", "profile", "expected",
    "view", "request_id", "session_id", "step_id", "canary_id",
    "required", "payload", "plan_sha256",
}
RESULT_KEYS = {
    "schema", "case_id", "case_sha256", "plan_sha256",
    "deployment_manifest_sha256", "evidence_manifest_sha256",
    "accepted", "secret_released",
    "canary_observed", "signal_observed", "status", "error_code",
    "request_id", "session_id", "step_id", "evidence", "details",
}

EVIDENCE_FIELDS = {
    "attestation-key-release": ("quote", "collateral", "firmware",
                                "verifier_policy", "nonce", "endpoint_key",
                                "decision"),
    "protocol-integrity": ("transcript", "mutation", "decision"),
    "malicious-computation": ("request", "mutation", "result", "decision"),
    "boundary-canary": ("observation", "canary_manifest"),
    "confidential-host-trace": ("trace", "collection_policy"),
    "fhe-feedback-oracle": ("ciphertext", "context", "feedback", "decision"),
}

MANIFEST_SECTIONS = {
    "adapter": ("provider", "target_factory", "implementation_revision",
                "command_argv", "credential_mode"),
    "hardware": ("cpu_tee", "gpu", "cc_mode", "debug_enabled", "firmware",
                 "vbios", "driver", "tee_module", "protected_memory"),
    "trust": ("trusted_components", "adversary_roles", "out_of_scope"),
    "attestation": ("verifier", "verifier_policy", "collateral_as_of_utc",
                    "allowed_measurements", "key_release_service"),
    "workload": ("revision", "config_sha256", "corpus_sha256", "seed_list"),
    "protected_phases": ("ingress", "tokenisation", "forward", "return",
                         "backward", "update", "optimizer", "checkpointing",
                         "logging", "crash_restart", "egress"),
    "leakage_policy": ("message_length", "timing", "failures",
                       "resource_counters", "outputs", "model_release"),
    "export_policy": ("checkpoints", "captures", "diagnostics", "raw_secrets"),
    "gates": ("valid_release_required", "invalid_release_tolerance",
              "external_canary_tolerance", "mutation_accept_tolerance"),
    "budget": ("currency", "max_cost", "max_wall_minutes"),
}


def _case(profile, category, arm, expected, *, view="external_attacker",
          required=True, index=0, canary_id=None, payload=None):
    case_id = f"{category}:{arm}:{view}"
    return {
        "schema": CASE_SCHEMA,
        "case_id": case_id,
        "category": category,
        "arm": arm,
        "profile": profile,
        "expected": expected,
        "view": view,
        "request_id": f"req-{index:04d}",
        "session_id": f"session-{index // 8:03d}",
        "step_id": index,
        "canary_id": canary_id,
        "required": bool(required),
        "payload": dict(payload or {}),
        "plan_sha256": None,
    }


def build_plan(profile, require_host_traces=False,
               deployment_manifest_sha256=None):
    """Return the minimum common TEE/FHE audit plan.

    The plan contains no canary values or secrets; the vendor adapter creates
    and retains those locally.  Host traces are explicit but optional unless
    ``require_host_traces`` is set, so inaccessible mechanisms become
    UNASSESSED rather than fictitious passes.
    """
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")
    if deployment_manifest_sha256 is not None and not _is_sha256(
            deployment_manifest_sha256):
        raise ValueError("deployment_manifest_sha256 must be 64 lowercase hex")
    cases = []

    def add(category, arm, expected, **kw):
        payload = dict(kw.pop("payload", {}) or {})
        payload["deployment_manifest_sha256"] = deployment_manifest_sha256
        cases.append(_case(profile, category, arm, expected,
                           index=len(cases), payload=payload, **kw))

    for arm in ATTESTATION_ARMS:
        add("attestation-key-release", arm,
            "allow-release" if arm == "valid" else "deny-release")
    for arm in PROTOCOL_ARMS:
        add("protocol-integrity", arm,
            "accept" if arm == "clean-control" else "reject")
    for arm in COMPUTATION_ARMS:
        add("malicious-computation", arm,
            "accept" if arm == "correct-result-control" else "reject")
    for phase in BOUNDARY_PHASES:
        cid = f"canary-{phase}"
        add("boundary-canary", phase, "no-canary",
            view="external_attacker", canary_id=cid)
        add("boundary-canary", phase, "observe-canary",
            view="trusted_diagnostic", canary_id=cid)
    for arm in HOST_TRACE_ARMS:
        add("confidential-host-trace", arm, "observe-or-declare",
            required=require_host_traces)
    if profile in ("fhe", "hybrid"):
        for arm in FHE_ARMS:
            expected = "signal" if arm == "leaky-positive-control" else (
                "no-signal" if arm == "production-feedback" else "reject")
            add("fhe-feedback-oracle", arm, expected)

    plan = {
        "schema": PLAN_SCHEMA,
        "profile": profile,
        "gate_schema": GATE_SCHEMA,
        "require_host_traces": bool(require_host_traces),
        "deployment_manifest_sha256": deployment_manifest_sha256,
        "cases": cases,
    }
    plan["sha256"] = plan_digest(plan)
    for case in plan["cases"]:
        case["plan_sha256"] = plan["sha256"]
    return plan


def plan_digest(plan):
    body = {k: v for k, v in plan.items() if k != "sha256"}
    body["cases"] = [
        {**case, "plan_sha256": None} for case in body.get("cases", [])
    ]
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def case_digest(case):
    """Bind a result to the complete input case, including deployment hash."""
    validate_case(case)
    raw = json.dumps(case, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def expected_case_contract(category, arm, view):
    """Return the only valid expectation for a secure-compute coordinate."""
    if category == "attestation-key-release" and arm in ATTESTATION_ARMS:
        return "allow-release" if arm == "valid" else "deny-release"
    if category == "protocol-integrity" and arm in PROTOCOL_ARMS:
        return "accept" if arm == "clean-control" else "reject"
    if category == "malicious-computation" and arm in COMPUTATION_ARMS:
        return "accept" if arm == "correct-result-control" else "reject"
    if category == "boundary-canary" and arm in BOUNDARY_PHASES:
        if view not in VIEWS:
            raise ValueError(f"invalid boundary view {view!r}")
        return ("observe-canary" if view == "trusted_diagnostic"
                else "no-canary")
    if category == "confidential-host-trace" and arm in HOST_TRACE_ARMS:
        return "observe-or-declare"
    if category == "fhe-feedback-oracle" and arm in FHE_ARMS:
        if arm == "leaky-positive-control":
            return "signal"
        if arm == "production-feedback":
            return "no-signal"
        return "reject"
    raise ValueError(f"unknown secure-compute category/arm "
                     f"{category!r}/{arm!r}")


def _expected_case_ids(profile):
    ids = {f"attestation-key-release:{arm}:external_attacker"
           for arm in ATTESTATION_ARMS}
    ids |= {f"protocol-integrity:{arm}:external_attacker"
            for arm in PROTOCOL_ARMS}
    ids |= {f"malicious-computation:{arm}:external_attacker"
            for arm in COMPUTATION_ARMS}
    ids |= {f"boundary-canary:{phase}:{view}"
            for phase in BOUNDARY_PHASES for view in VIEWS}
    ids |= {f"confidential-host-trace:{arm}:external_attacker"
            for arm in HOST_TRACE_ARMS}
    if profile in ("fhe", "hybrid"):
        ids |= {f"fhe-feedback-oracle:{arm}:external_attacker"
                for arm in FHE_ARMS}
    return ids


def validate_plan(plan, path="<plan>"):
    plan_keys = {"schema", "profile", "gate_schema", "require_host_traces",
                 "deployment_manifest_sha256", "cases", "sha256"}
    if not isinstance(plan, dict) or set(plan) != plan_keys \
            or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"{path}: expected schema {PLAN_SCHEMA}")
    if plan.get("profile") not in PROFILES:
        raise ValueError(f"{path}: invalid profile {plan.get('profile')!r}")
    if plan.get("gate_schema") != GATE_SCHEMA:
        raise ValueError(f"{path}: gate_schema must be {GATE_SCHEMA}")
    if not isinstance(plan.get("require_host_traces"), bool):
        raise ValueError(f"{path}: require_host_traces must be bool")
    manifest = plan.get("deployment_manifest_sha256")
    if manifest is not None and not _is_sha256(manifest):
        raise ValueError(f"{path}: invalid deployment_manifest_sha256")
    cases = plan.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path}: cases must be a non-empty list")
    if not _is_sha256(plan.get("sha256")):
        raise ValueError(f"{path}: sha256 must be 64 lowercase hex")
    seen = set()
    requests = set()
    for i, case in enumerate(cases):
        validate_case(case, f"{path}:cases[{i}]")
        if case["case_id"] in seen:
            raise ValueError(f"{path}: duplicate case_id {case['case_id']!r}")
        seen.add(case["case_id"])
        if case["request_id"] in requests:
            raise ValueError(f"{path}: duplicate request_id {case['request_id']!r}")
        requests.add(case["request_id"])
        if case["profile"] != plan["profile"]:
            raise ValueError(f"{path}: case profile does not match plan")
        if case["plan_sha256"] != plan["sha256"]:
            raise ValueError(f"{path}: case plan_sha256 does not match plan")
        if case["payload"].get("deployment_manifest_sha256") != manifest:
            raise ValueError(f"{path}: case deployment manifest binding mismatch")
        should_require = (plan["require_host_traces"]
                          if case["category"] == "confidential-host-trace"
                          else True)
        if case["required"] is not should_require:
            raise ValueError(f"{path}: case required flag contradicts plan")
        if case["step_id"] != i:
            raise ValueError(f"{path}: step_id must equal canonical case order")
    expected_ids = _expected_case_ids(plan["profile"])
    if seen != expected_ids:
        raise ValueError(f"{path}: case inventory mismatch; missing "
                         f"{sorted(expected_ids - seen)}, extra "
                         f"{sorted(seen - expected_ids)}")
    if plan.get("sha256") != plan_digest(plan):
        raise ValueError(f"{path}: sha256 does not bind the plan contents")
    return plan


def validate_case(case, path="<case>"):
    if not isinstance(case, dict) or set(case) != CASE_KEYS:
        got = sorted(case) if isinstance(case, dict) else type(case).__name__
        raise ValueError(f"{path}: keys {got!r} != {sorted(CASE_KEYS)!r}")
    if case["schema"] != CASE_SCHEMA:
        raise ValueError(f"{path}: schema must be {CASE_SCHEMA}")
    for key in ("case_id", "category", "arm", "expected", "request_id",
                "session_id"):
        if not isinstance(case[key], str) or not case[key]:
            raise ValueError(f"{path}: {key} must be a non-empty str")
    if case["profile"] not in PROFILES or case["view"] not in VIEWS:
        raise ValueError(f"{path}: invalid profile or view")
    if not _is_sha256(case["plan_sha256"]):
        raise ValueError(f"{path}: plan_sha256 must be 64 lowercase hex")
    if not isinstance(case["step_id"], int) or case["step_id"] < 0:
        raise ValueError(f"{path}: step_id must be a non-negative int")
    if case["canary_id"] is not None and not isinstance(case["canary_id"], str):
        raise ValueError(f"{path}: canary_id must be str or null")
    if not isinstance(case["required"], bool) or not isinstance(case["payload"], dict):
        raise ValueError(f"{path}: required must be bool and payload an object")
    expected = expected_case_contract(case["category"], case["arm"],
                                      case["view"])
    if case["expected"] != expected:
        raise ValueError(f"{path}: expectation must be {expected!r}")
    canonical_id = f"{case['category']}:{case['arm']}:{case['view']}"
    if case["case_id"] != canonical_id:
        raise ValueError(f"{path}: case_id must be {canonical_id!r}")
    if case["category"] == "boundary-canary":
        if case["canary_id"] != f"canary-{case['arm']}":
            raise ValueError(f"{path}: invalid boundary canary binding")
    elif case["canary_id"] is not None:
        raise ValueError(f"{path}: only boundary cases may carry canary_id")
    if case["category"] != "boundary-canary" \
            and case["view"] != "external_attacker":
        raise ValueError(f"{path}: only boundary cases use trusted_diagnostic")
    if case["category"] == "fhe-feedback-oracle" \
            and case["profile"] not in ("fhe", "hybrid"):
        raise ValueError(f"{path}: FHE case requires fhe or hybrid profile")
    manifest = case["payload"].get("deployment_manifest_sha256")
    if manifest is not None and not _is_sha256(manifest):
        raise ValueError(f"{path}: invalid deployment manifest binding")
    return case


def validate_result(result, path="<result>"):
    if not isinstance(result, dict) or set(result) != RESULT_KEYS:
        got = sorted(result) if isinstance(result, dict) else type(result).__name__
        raise ValueError(f"{path}: keys {got!r} != {sorted(RESULT_KEYS)!r}")
    if result["schema"] != RESULT_SCHEMA:
        raise ValueError(f"{path}: schema must be {RESULT_SCHEMA}")
    if not isinstance(result["case_id"], str) or not result["case_id"]:
        raise ValueError(f"{path}: case_id must be a non-empty str")
    if not _is_sha256(result["case_sha256"]):
        raise ValueError(f"{path}: case_sha256 must be 64 lowercase hex")
    if not _is_sha256(result["plan_sha256"]):
        raise ValueError(f"{path}: plan_sha256 must be 64 lowercase hex")
    manifest = result["deployment_manifest_sha256"]
    if manifest is not None and not _is_sha256(manifest):
        raise ValueError(f"{path}: invalid deployment_manifest_sha256")
    if not _is_sha256(result["evidence_manifest_sha256"]):
        raise ValueError(f"{path}: evidence_manifest_sha256 must be 64 lowercase hex")
    for key in ("accepted", "secret_released", "canary_observed",
                "signal_observed"):
        if not isinstance(result[key], bool):
            raise ValueError(f"{path}: {key} must be bool")
    if result["status"] not in RESULT_STATUSES:
        raise ValueError(f"{path}: status must be one of {RESULT_STATUSES}")
    if result["error_code"] is not None and not isinstance(result["error_code"], str):
        raise ValueError(f"{path}: error_code must be str or null")
    for key in ("request_id", "session_id"):
        if not isinstance(result[key], str) or not result[key]:
            raise ValueError(f"{path}: {key} must be a non-empty str")
    if not isinstance(result["step_id"], int) or result["step_id"] < 0:
        raise ValueError(f"{path}: step_id must be a non-negative int")
    if not isinstance(result["evidence"], dict):
        raise ValueError(f"{path}: evidence must be an object")
    for key, digest in result["evidence"].items():
        if not isinstance(key, str) or not key or not _is_sha256(digest):
            raise ValueError(f"{path}: evidence must map non-empty names to "
                             "64-character lowercase SHA-256 digests")
    if not isinstance(result["details"], dict):
        raise ValueError(f"{path}: details must be an object")
    return result


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def file_sha256(path):
    h = hashlib.sha256()
    descriptor = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{path}: must be a regular non-symlink file")
        with os.fdopen(descriptor, "rb") as f:
            descriptor = None
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
            after = os.fstat(f.fileno())
        if ((before.st_dev, before.st_ino, before.st_size,
             before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_size,
                    after.st_mtime_ns, after.st_ctime_ns)):
            raise ValueError(f"{path}: file changed while it was hashed")
    except OSError as exc:
        raise ValueError(f"{path}: cannot hash file: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return h.hexdigest()


def _read_json_bound(path, expected_sha256=None, label="JSON file"):
    """Read one regular JSON file and prove the parsed bytes match its digest."""
    descriptor = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > 64 * 1024 * 1024:
            raise ValueError(f"{label} must be a bounded regular non-symlink file")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            raw = stream.read()
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if ((before.st_dev, before.st_ino, before.st_size,
         before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size,
                after.st_mtime_ns, after.st_ctime_ns)):
        raise ValueError(f"{label} changed while it was read")
    actual = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"{label} digest mismatch")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    return value, actual


def _bundle_path(root, relative, label):
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path must be non-empty str")
    normalized = relative.replace("\\", "/")
    parts = normalized.split("/")
    if normalized.startswith("/") or ".." in parts or "" in parts:
        raise ValueError(f"unsafe {label} path {relative!r}")
    root = os.path.realpath(root)
    path = os.path.realpath(os.path.join(root, *parts))
    try:
        inside = os.path.commonpath((root, path)) == root
    except ValueError:
        inside = False
    if not inside:
        raise ValueError(f"{label} path escapes evidence bundle")
    return path


def verify_result_evidence_bundle(result, evidence_bundle_dir):
    """Resolve and verify a result's per-case manifest and evidence files."""
    validate_result(result)
    relative = result["details"].get("evidence_manifest")
    manifest_path = _bundle_path(
        evidence_bundle_dir, relative, "evidence manifest")
    manifest, _ = _read_json_bound(
        manifest_path, result["evidence_manifest_sha256"],
        "evidence manifest")
    manifest_keys = {"schema", "category", "case_id", "case_sha256",
                     "plan_sha256", "deployment_manifest_sha256",
                     "evidence", "files"}
    if not isinstance(manifest, dict) or set(manifest) != manifest_keys \
            or manifest.get("schema") != EVIDENCE_MANIFEST_SCHEMA:
        raise ValueError("invalid secure-compute evidence manifest schema")
    for key in ("case_id", "case_sha256", "plan_sha256",
                "deployment_manifest_sha256", "evidence"):
        if manifest.get(key) != result.get(key):
            raise ValueError(f"evidence manifest {key} binding mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(result["evidence"]):
        raise ValueError("evidence manifest files do not match result fields")
    category = result["case_id"].split(":", 1)[0]
    if manifest.get("category") != category:
        raise ValueError("evidence manifest category binding mismatch")
    for field, evidence_relative in files.items():
        evidence_path = _bundle_path(
            evidence_bundle_dir, evidence_relative,
            f"evidence {field!r}")
        if file_sha256(evidence_path) != result["evidence"][field]:
            raise ValueError(f"evidence digest mismatch for {field!r}")
    return True


def load_deployment_manifest(path, expected_profile=None):
    """Load and validate the frozen JSON deployment contract."""
    doc, _ = _read_json_bound(path, label="deployment manifest")
    if not isinstance(doc, dict) or doc.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{path}: expected schema {MANIFEST_SCHEMA}")
    expected_top_level = {
        "schema", "ready_for_execution", "deployment_id", "profile",
        *MANIFEST_SECTIONS, "stop_rules",
    }
    if set(doc) != expected_top_level:
        raise ValueError(f"{path}: deployment manifest fields differ")
    if doc.get("ready_for_execution") is not True:
        raise ValueError(f"{path}: ready_for_execution must be true after review")
    if not isinstance(doc.get("deployment_id"), str) or not doc["deployment_id"]:
        raise ValueError(f"{path}: deployment_id must be a non-empty str")
    if doc.get("profile") not in PROFILES:
        raise ValueError(f"{path}: profile must be one of {PROFILES}")
    if expected_profile is not None and doc["profile"] != expected_profile:
        raise ValueError(f"{path}: profile {doc['profile']!r} does not match "
                         f"requested {expected_profile!r}")
    for section, fields in MANIFEST_SECTIONS.items():
        value = doc.get(section)
        if not isinstance(value, dict) or set(value) != set(fields):
            raise ValueError(f"{path}: {section} must be an object")
    for key in ("provider", "target_factory", "implementation_revision",
                "credential_mode"):
        value = doc["adapter"][key]
        if not isinstance(value, str) or not value:
            raise ValueError(f"{path}: adapter.{key} must be a non-empty str")
    command = doc["adapter"]["command_argv"]
    if not isinstance(command, list) or not command \
            or not all(isinstance(item, str) and item for item in command):
        raise ValueError(
            f"{path}: adapter.command_argv must be a non-empty list of str")
    for key in ("trusted_components", "adversary_roles"):
        value = doc["trust"][key]
        if not isinstance(value, list) or not value \
                or not all(isinstance(x, str) and x for x in value):
            raise ValueError(f"{path}: trust.{key} must be a non-empty list of str")
    if doc["hardware"]["debug_enabled"] is not False:
        raise ValueError(f"{path}: hardware.debug_enabled must be false")
    for key in ("config_sha256", "corpus_sha256"):
        if not _is_sha256(doc["workload"][key]):
            raise ValueError(f"{path}: workload.{key} must be 64 lowercase hex")
    seeds = doc["workload"]["seed_list"]
    if not isinstance(seeds, list) or not seeds \
            or not all(isinstance(seed, int) and seed >= 0 for seed in seeds):
        raise ValueError(f"{path}: workload.seed_list must be non-empty "
                         "non-negative integers")
    measurements = doc["attestation"]["allowed_measurements"]
    if not isinstance(measurements, list) or not measurements:
        raise ValueError(f"{path}: attestation.allowed_measurements must be non-empty")
    phases = doc["protected_phases"]
    if not all(isinstance(phases[key], bool)
               for key in MANIFEST_SECTIONS["protected_phases"]):
        raise ValueError(f"{path}: every protected_phases value must be bool")
    for key in ("invalid_release_tolerance", "external_canary_tolerance",
                "mutation_accept_tolerance"):
        if doc["gates"][key] != 0:
            raise ValueError(f"{path}: gates.{key} must be exactly zero")
    if doc["gates"]["valid_release_required"] is not True:
        raise ValueError(f"{path}: gates.valid_release_required must be true")
    budget = doc["budget"]
    if (isinstance(budget["max_cost"], bool)
            or not isinstance(budget["max_cost"], (int, float))
            or not math.isfinite(float(budget["max_cost"]))
            or budget["max_cost"] < 0):
        raise ValueError(f"{path}: budget.max_cost must be non-negative")
    if (isinstance(budget["max_wall_minutes"], bool)
            or not isinstance(budget["max_wall_minutes"], (int, float))
            or not math.isfinite(float(budget["max_wall_minutes"]))
            or budget["max_wall_minutes"] <= 0):
        raise ValueError(f"{path}: budget.max_wall_minutes must be positive")
    if not isinstance(doc.get("stop_rules"), list) or not doc["stop_rules"]:
        raise ValueError(f"{path}: stop_rules must be a non-empty list")
    return doc


def read_jsonl(path):
    rows = []
    try:
        with open(path) as f:
            lines = list(f)
    except OSError as exc:
        raise ValueError(f"{path}: cannot read: {exc}") from exc
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        rows.append(validate_result(row, f"{path}:{lineno}"))
    if not rows:
        raise ValueError(f"{path}: no result records")
    return rows


def run_process_driver(command, plan, timeout=900):
    """Run an explicit JSONL adapter command without a shell.

    The complete plan is supplied on stdin (one case per line); stdout must
    contain one strict result record per line.  Stderr is retained as a
    diagnostic string but is never parsed as evidence.
    """
    validate_plan(plan)
    argv = shlex.split(command)
    if not argv:
        raise ValueError("driver command is empty")
    payload = "".join(json.dumps(c, sort_keys=True) + "\n"
                      for c in plan["cases"])
    try:
        proc = subprocess.run(argv, input=payload, text=True,
                              capture_output=True, timeout=timeout,
                              check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"secure-compute driver failed: {exc}") from exc
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-1:] or ["no stderr"]
        raise RuntimeError(f"secure-compute driver exited {proc.returncode}: "
                           f"{tail[0]}")
    results = []
    for lineno, line in enumerate(proc.stdout.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"driver stdout:{lineno}: invalid JSON: {exc}") from exc
        results.append(validate_result(row, f"driver stdout:{lineno}"))
    if not results:
        raise ValueError("driver produced no result records")
    return results, proc.stderr


def _grade_case(case, result):
    if result["case_sha256"] != case_digest(case):
        return "FAIL", "result is not bound to the complete input case"
    if result["plan_sha256"] != case["plan_sha256"]:
        return "FAIL", "result plan binding mismatch"
    if result["deployment_manifest_sha256"] != \
            case["payload"].get("deployment_manifest_sha256"):
        return "FAIL", "result deployment manifest binding mismatch"
    if result["request_id"] != case["request_id"] \
            or result["session_id"] != case["session_id"] \
            or result["step_id"] != case["step_id"]:
        return "FAIL", "request/session/step binding mismatch"
    if result["status"] == "error":
        return "FAIL", "adapter reported error"
    if result["status"] == "unavailable":
        if case["category"] == "confidential-host-trace":
            required_evidence = ("collection_policy", "unavailability")
            if set(result["evidence"]) != set(required_evidence):
                return "FAIL", ("evidence fields must be exactly "
                                f"{sorted(required_evidence)}")
        return ("FAIL" if case["required"] else "UNASSESSED",
                "required observation unavailable" if case["required"]
                else "observation unavailable and explicitly unassessed")
    required_evidence = EVIDENCE_FIELDS[case["category"]]
    if set(result["evidence"]) != set(required_evidence):
        return "FAIL", ("evidence fields must be exactly "
                        f"{sorted(required_evidence)}")
    expected = case["expected"]
    if expected == "allow-release":
        ok = result["accepted"] and result["secret_released"]
    elif expected == "accept":
        ok = result["accepted"] and not result["secret_released"]
    elif expected == "deny-release":
        ok = not result["accepted"] and not result["secret_released"]
    elif expected == "reject":
        ok = not result["accepted"] and not result["secret_released"]
    elif expected == "no-canary":
        ok = not result["canary_observed"]
    elif expected == "observe-canary":
        ok = result["canary_observed"]
    elif expected == "observe-or-declare":
        ok = True
    elif expected == "signal":
        ok = result["signal_observed"]
    elif expected == "no-signal":
        ok = not result["signal_observed"]
    else:
        return "FAIL", f"unknown expectation {expected!r}"
    return ("PASS", "expectation satisfied") if ok else (
        "FAIL", f"expectation {expected!r} violated")


def grade(plan, results, evidence_bundle_dir=None,
          require_evidence_bundle=False):
    """Apply the exact v1 admission gate; missing/extra/duplicate rows fail."""
    validate_plan(plan)
    by_id = {}
    duplicates = []
    for i, result in enumerate(results):
        validate_result(result, f"results[{i}]")
        if result["case_id"] in by_id:
            duplicates.append(result["case_id"])
        by_id[result["case_id"]] = result
    planned = {c["case_id"] for c in plan["cases"]}
    extras = sorted(set(by_id) - planned)
    missing = sorted(planned - set(by_id))
    cells = []
    for case in plan["cases"]:
        result = by_id.get(case["case_id"])
        if result is None:
            state, reason = "FAIL", "missing result"
        else:
            if require_evidence_bundle and evidence_bundle_dir is None:
                state, reason = "FAIL", "evidence bundle directory is required"
            elif evidence_bundle_dir is not None:
                try:
                    verify_result_evidence_bundle(result, evidence_bundle_dir)
                except ValueError as exc:
                    state, reason = "FAIL", f"evidence bundle: {exc}"
                else:
                    state, reason = _grade_case(case, result)
            else:
                state, reason = _grade_case(case, result)
        cells.append({"case_id": case["case_id"],
                      "category": case["category"], "arm": case["arm"],
                      "view": case["view"], "required": case["required"],
                      "verdict": state, "reason": reason})
    if extras:
        cells.append({"case_id": "<extra-results>", "category": "schema",
                      "arm": "extra", "view": "external_attacker",
                      "required": True, "verdict": "FAIL",
                      "reason": f"unplanned result ids: {extras}"})
    if duplicates:
        cells.append({"case_id": "<duplicate-results>", "category": "schema",
                      "arm": "duplicate", "view": "external_attacker",
                      "required": True, "verdict": "FAIL",
                      "reason": f"duplicate result ids: {sorted(set(duplicates))}"})
    counts = {state: sum(c["verdict"] == state for c in cells)
              for state in ("PASS", "FAIL", "UNASSESSED")}
    overall = "FAIL" if counts["FAIL"] else (
        "UNASSESSED" if counts["UNASSESSED"] else "PASS")
    invalid_release_count = sum(
        bool(by_id[c["case_id"]]["secret_released"])
        for c in plan["cases"]
        if c["expected"] == "deny-release" and c["case_id"] in by_id)
    external_canary_exposures = sum(
        bool(by_id[c["case_id"]]["canary_observed"])
        for c in plan["cases"]
        if c["expected"] == "no-canary" and c["case_id"] in by_id)
    return {"schema": GATE_SCHEMA, "overall": overall, "counts": counts,
            "invalid_secret_releases": invalid_release_count,
            "external_canary_exposures": external_canary_exposures,
            "missing_case_ids": missing, "extra_case_ids": extras,
            "duplicate_case_ids": sorted(set(duplicates)), "cells": cells}


def make_toy_results(plan, defended=True, unavailable_host=True):
    """Deterministic local fixture.  It is a machinery check, not evidence."""
    out = []
    for case in plan["cases"]:
        expected = case["expected"]
        unavailable = (case["category"] == "confidential-host-trace"
                       and unavailable_host)
        secure = defended
        fields = (("collection_policy", "unavailability") if unavailable
                  else EVIDENCE_FIELDS[case["category"]])
        evidence = {name: hashlib.sha256(
            f"toy:{case['case_id']}:{name}".encode()).hexdigest()
                    for name in fields}
        synthetic_files = {
            name: f"synthetic/{case['step_id']:04d}/{name}.json"
            for name in fields
        }
        evidence_manifest = {
            "schema": EVIDENCE_MANIFEST_SCHEMA,
            "category": case["category"],
            "case_id": case["case_id"],
            "case_sha256": case_digest(case),
            "plan_sha256": case["plan_sha256"],
            "deployment_manifest_sha256":
                case["payload"].get("deployment_manifest_sha256"),
            "evidence": evidence,
            "files": synthetic_files,
        }
        evidence_manifest_sha256 = hashlib.sha256(json.dumps(
            evidence_manifest, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        out.append({
            "schema": RESULT_SCHEMA,
            "case_id": case["case_id"],
            "case_sha256": case_digest(case),
            "plan_sha256": case["plan_sha256"],
            "deployment_manifest_sha256":
                case["payload"].get("deployment_manifest_sha256"),
            "evidence_manifest_sha256": evidence_manifest_sha256,
            "accepted": (expected in ("allow-release", "accept"))
                        if secure else True,
            "secret_released": (expected == "allow-release") if secure else (
                case["category"] == "attestation-key-release"),
            "canary_observed": (expected == "observe-canary") if secure else (
                case["category"] == "boundary-canary"),
            "signal_observed": (expected == "signal") if secure else True,
            "status": "unavailable" if unavailable else "observed",
            "error_code": "UNAVAILABLE_IN_FIXTURE" if unavailable else None,
            "request_id": case["request_id"],
            "session_id": case["session_id"],
            "step_id": case["step_id"],
            "evidence": evidence,
            "details": {"toy": True, "defended": defended,
                        "evidence_manifest":
                            f"synthetic/{case['step_id']:04d}/manifest.json"},
        })
    return out


def write_plan(path, plan):
    validate_plan(plan)
    target = os.fspath(path)
    descriptor = None
    created = False
    try:
        descriptor = os.open(
            target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = None
            json.dump(plan, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                os.unlink(target)
            except OSError:
                pass
        raise


class SecureComputeResultProvider(SurfaceProvider):
    name = "secure-compute-results"
    schema = {key: None for key in RESULT_KEYS}
    modes = ("training", "inference")

    def validate(self, meta, path="<result>"):
        return validate_result(meta, path)

    def rows(self, mode=None, source=None):
        self.check_mode(mode)
        if source is None:
            raise ValueError("surface 'secure-compute-results' needs source=<jsonl>")
        rows = read_jsonl(source)
        return (([int(r["accepted"]), int(r["secret_released"]),
                  int(r["canary_observed"]), int(r["signal_observed"])], r)
                for r in rows)


register_surface(SecureComputeResultProvider())


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = bool(cond) and ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("secure-compute plan/driver/gate (pure python):")
    tee = build_plan("tee")
    fhe = build_plan("fhe")
    check("plans validate and are digest-bound",
          validate_plan(tee) is tee and validate_plan(fhe) is fhe)
    check("FHE adds feedback/oracle cases",
          len(fhe["cases"]) == len(tee["cases"]) + len(FHE_ARMS))
    secure_gate = grade(tee, make_toy_results(tee))
    check("secure fixture is explicit UNASSESSED for inaccessible host traces",
          secure_gate["overall"] == "UNASSESSED"
          and secure_gate["invalid_secret_releases"] == 0)
    required = build_plan("tee", require_host_traces=True)
    check("required unavailable traces fail closed",
          grade(required, make_toy_results(required))["overall"] == "FAIL")
    bad = make_toy_results(tee, unavailable_host=False)
    bad[1]["secret_released"] = True
    check("one invalid release fails with zero tolerance",
          grade(tee, bad)["overall"] == "FAIL")
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
