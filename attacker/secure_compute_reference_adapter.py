#!/usr/bin/env python3
"""Deterministic, local reference adapter for the secure-compute contract.

The adapter is intentionally hardware-free.  It implements the same JSONL
process boundary a future vendor adapter must implement, but drives a small
deterministic target abstraction with two explicit fixture policies:

``defended``
    Enforces attestation, protocol and computation binding, boundary
    confinement, ciphertext validation, and production-feedback blinding.

``vulnerable``
    Deliberately disables those protections so every negative case is a
    positive sensitivity control.  Positive controls (valid release, trusted
    canaries, host observations, and the leaky FHE oracle) remain functional.

Neither fixture contacts a network or cloud service.  Every result is backed
by category-specific JSON evidence files; the result carries their SHA-256
digests and relative paths, never raw keys, canary values, plaintext, or
trusted tensors.
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from . import secure_compute as contract
from . import secure_compute_process_adapter as process_adapter


ADAPTER_SCHEMA = "llm_attacker.secure_compute_reference_adapter.v1"
EVIDENCE_SCHEMA = "llm_attacker.secure_compute_reference_evidence.v1"
FIXTURES = ("defended", "vulnerable")
HOST_TRACE_MODES = ("observed", "unavailable")


@dataclass(frozen=True)
class FixturePolicy:
    """Protection switches for one deterministic local target."""

    name: str
    enforce_attestation: bool
    enforce_protocol_binding: bool
    verify_computation: bool
    confine_boundary: bool
    reject_bad_ciphertexts: bool
    blind_production_feedback: bool


FIXTURE_POLICIES = {
    "defended": FixturePolicy(
        name="defended",
        enforce_attestation=True,
        enforce_protocol_binding=True,
        verify_computation=True,
        confine_boundary=True,
        reject_bad_ciphertexts=True,
        blind_production_feedback=True,
    ),
    "vulnerable": FixturePolicy(
        name="vulnerable",
        enforce_attestation=False,
        enforce_protocol_binding=False,
        verify_computation=False,
        confine_boundary=False,
        reject_bad_ciphertexts=False,
        blind_production_feedback=False,
    ),
}


def expected_contract(case):
    """Derive the plan expectation from category/arm/view, never vice versa."""
    return contract.expected_case_contract(
        case["category"], case["arm"], case["view"])


def validate_reference_case(case, path="<case>"):
    """Validate both the generic schema and the reference arm contract."""
    contract.validate_case(case, path)
    expected = expected_contract(case)
    if case["expected"] != expected:
        raise ValueError(
            f"{path}: expected {case['expected']!r} does not match the "
            f"reference contract {expected!r}")
    canonical_id = f"{case['category']}:{case['arm']}:{case['view']}"
    if case["case_id"] != canonical_id:
        raise ValueError(f"{path}: case_id must be {canonical_id!r}")
    if case["category"] == "boundary-canary":
        if not case["canary_id"]:
            raise ValueError(f"{path}: boundary case needs canary_id")
    elif case["canary_id"] is not None:
        raise ValueError(f"{path}: only boundary cases may carry canary_id")
    if case["category"] == "fhe-feedback-oracle" \
            and case["profile"] not in ("fhe", "hybrid"):
        raise ValueError(f"{path}: FHE cases require fhe or hybrid profile")
    if case["category"] != "boundary-canary" \
            and case["view"] != "external_attacker":
        raise ValueError(f"{path}: only boundary cases use trusted_diagnostic")
    return case


def fixture_expected_verdict(case, fixture):
    """Qualification oracle for the two deliberate fixture policies."""
    if fixture not in FIXTURE_POLICIES:
        raise ValueError(f"fixture must be one of {FIXTURES}")
    if fixture == "defended":
        return "PASS"
    return ("FAIL" if expected_contract(case) in
            ("deny-release", "reject", "no-canary", "no-signal")
            else "PASS")


def _canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) +
            "\n").encode("utf-8")


def _digest(value):
    if not isinstance(value, bytes):
        value = str(value).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _safe_component(value):
    value = "".join(c if c.isalnum() or c in "-_" else "-"
                    for c in str(value)).strip("-")
    if not value:
        raise ValueError("empty evidence path component")
    return value


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class DeterministicLocalTarget(process_adapter.SecureComputeTarget):
    """No-hardware target that emits reproducible outcomes and evidence."""

    def __init__(self, fixture, evidence_dir, host_traces="observed"):
        if fixture not in FIXTURE_POLICIES:
            raise ValueError(f"fixture must be one of {FIXTURES}")
        if host_traces not in HOST_TRACE_MODES:
            raise ValueError(f"host_traces must be one of {HOST_TRACE_MODES}")
        self.policy = FIXTURE_POLICIES[fixture]
        self.evidence_dir = Path(evidence_dir)
        self.host_traces = host_traces

    def _token(self, case, label):
        material = {
            "fixture": self.policy.name,
            "case_id": case["case_id"],
            "request_id": case["request_id"],
            "session_id": case["session_id"],
            "step_id": case["step_id"],
            "label": label,
            "deployment_manifest_sha256":
                case["payload"].get("deployment_manifest_sha256"),
        }
        return _digest(_canonical_bytes(material))

    def _outcome(self, case):
        category = case["category"]
        arm = case["arm"]
        outcome = {
            "accepted": False,
            "secret_released": False,
            "canary_observed": False,
            "signal_observed": False,
            "status": "observed",
            "error_code": None,
        }
        if category == "attestation-key-release":
            allowed = arm == "valid" or not self.policy.enforce_attestation
            outcome["accepted"] = allowed
            outcome["secret_released"] = allowed
            if not allowed:
                outcome["error_code"] = "ATTESTATION_POLICY_REJECTED"
        elif category == "protocol-integrity":
            outcome["accepted"] = (arm == "clean-control" or
                                   not self.policy.enforce_protocol_binding)
            if not outcome["accepted"]:
                outcome["error_code"] = "PROTOCOL_MUTATION_REJECTED"
        elif category == "malicious-computation":
            outcome["accepted"] = (arm == "correct-result-control" or
                                   not self.policy.verify_computation)
            if not outcome["accepted"]:
                outcome["error_code"] = "COMPUTATION_RESULT_REJECTED"
        elif category == "boundary-canary":
            outcome["accepted"] = True
            outcome["canary_observed"] = (
                case["view"] == "trusted_diagnostic" or
                not self.policy.confine_boundary)
        elif category == "confidential-host-trace":
            if self.host_traces == "unavailable":
                outcome["status"] = "unavailable"
                outcome["error_code"] = "HOST_TRACE_UNAVAILABLE_LOCAL_FIXTURE"
            else:
                outcome["accepted"] = True
                outcome["signal_observed"] = True
        elif category == "fhe-feedback-oracle":
            if arm == "leaky-positive-control":
                outcome["accepted"] = True
                outcome["signal_observed"] = True
            elif arm == "production-feedback":
                outcome["accepted"] = True
                outcome["signal_observed"] = (
                    not self.policy.blind_production_feedback)
            else:
                outcome["accepted"] = (
                    not self.policy.reject_bad_ciphertexts)
                if not outcome["accepted"]:
                    outcome["error_code"] = "CIPHERTEXT_REJECTED"
        else:
            raise ValueError(f"unsupported category {category!r}")
        return outcome

    def _evidence_records(self, case, outcome):
        category = case["category"]
        arm = case["arm"]
        decision = {
            "accepted": outcome["accepted"],
            "secret_released": outcome["secret_released"],
            "status": outcome["status"],
            "error_code": outcome["error_code"],
        }
        if category == "attestation-key-release":
            records = {
                "quote": {"quote_id": self._token(case, "quote"),
                          "mutation": arm, "synthetic": True},
                "collateral": {"collateral_id": self._token(case, "collateral"),
                               "fresh": arm != "stale-collateral"},
                "firmware": {"measurement_id": self._token(case, "firmware"),
                             "allowed": arm not in
                             ("wrong-measurement", "modified-quote")},
                "verifier_policy": {
                    "policy_id": self._token(case, "verifier-policy"),
                    "enforced": self.policy.enforce_attestation},
                "nonce": {"nonce_id": self._token(case, "nonce"),
                          "fresh": arm != "replayed-nonce"},
                "endpoint_key": {
                    "endpoint_key_id": self._token(case, "endpoint-key"),
                    "bound": arm not in ("endpoint-key-substitution",
                                         "channel-binding-mismatch",
                                         "wrong-machine-relay")},
                "decision": decision,
            }
        elif category == "protocol-integrity":
            records = {
                "transcript": {
                    "transcript_id": self._token(case, "transcript"),
                    "request_id": case["request_id"],
                    "session_id": case["session_id"],
                    "step_id": case["step_id"]},
                "mutation": {"name": arm,
                             "is_control": arm == "clean-control",
                             "detected": (arm != "clean-control" and
                                          not outcome["accepted"])},
                "decision": decision,
            }
        elif category == "malicious-computation":
            records = {
                "request": {"request_digest": self._token(case, "request"),
                            "step_id": case["step_id"]},
                "mutation": {"name": arm,
                             "is_control": arm == "correct-result-control",
                             "detected": (arm != "correct-result-control" and
                                          not outcome["accepted"])},
                "result": {"result_digest": self._token(case, "result"),
                           "bound": (arm == "correct-result-control" or
                                     not outcome["accepted"])},
                "decision": decision,
            }
        elif category == "boundary-canary":
            records = {
                "observation": {
                    "observation_id": self._token(case, "observation"),
                    "phase": arm, "view": case["view"],
                    "canary_observed": outcome["canary_observed"]},
                "canary_manifest": {
                    "canary_id": case["canary_id"],
                    "canary_value_sha256": self._token(case, "canary-value"),
                    "raw_value_exported": False},
            }
        elif category == "confidential-host-trace":
            records = {
                "collection_policy": {
                    "trace_kind": arm,
                    "mode": self.host_traces,
                    "collector_id": self._token(case, "collector")},
            }
            if outcome["status"] == "unavailable":
                records["unavailability"] = {
                    "error_code": outcome["error_code"],
                    "declared": True,
                }
            else:
                records["trace"] = {
                    "trace_digest": self._token(case, "trace"),
                    "signal_observed": outcome["signal_observed"],
                    "raw_trace_exported": False,
                }
        elif category == "fhe-feedback-oracle":
            records = {
                "ciphertext": {
                    "ciphertext_id": self._token(case, "ciphertext"),
                    "mutation": arm, "raw_ciphertext_exported": False},
                "context": {
                    "context_id": self._token(case, "context"),
                    "profile": case["profile"]},
                "feedback": {
                    "feedback_class": ("signal" if outcome["signal_observed"]
                                       else "constant"),
                    "signal_observed": outcome["signal_observed"]},
                "decision": decision,
            }
        else:
            raise ValueError(f"unsupported category {category!r}")
        return records

    def _write_evidence(self, case, records):
        expected_fields = (set(("collection_policy", "unavailability"))
                           if case["category"] == "confidential-host-trace"
                           and self.host_traces == "unavailable"
                           else set(contract.EVIDENCE_FIELDS[case["category"]]))
        if set(records) != expected_fields:
            raise RuntimeError(
                f"internal evidence mismatch for {case['case_id']}: "
                f"{sorted(records)} != {sorted(expected_fields)}")
        case_dir = Path(self.policy.name) / _safe_component(case["category"]) / (
            f"{case['step_id']:04d}-{_safe_component(case['arm'])}-"
            f"{_safe_component(case['view'])}")
        digests = {}
        relative_paths = {}
        for field in sorted(records):
            relative = case_dir / f"{_safe_component(field)}.json"
            destination = self.evidence_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            document = {
                "schema": EVIDENCE_SCHEMA,
                "adapter_schema": ADAPTER_SCHEMA,
                "fixture": self.policy.name,
                "category": case["category"],
                "artifact": field,
                "binding": {
                    "case_id": case["case_id"],
                    "case_sha256": contract.case_digest(case),
                    "plan_sha256": case["plan_sha256"],
                    "profile": case["profile"],
                    "view": case["view"],
                    "request_id": case["request_id"],
                    "session_id": case["session_id"],
                    "step_id": case["step_id"],
                    "deployment_manifest_sha256":
                        case["payload"].get("deployment_manifest_sha256"),
                },
                "record": records[field],
            }
            raw = _canonical_bytes(document)
            temporary = destination.with_name(
                destination.name + f".{os.getpid()}.tmp")
            with open(temporary, "wb") as handle:
                handle.write(raw)
            os.replace(temporary, destination)
            digests[field] = _digest(raw)
            relative_paths[field] = relative.as_posix()
        manifest_relative = case_dir / "manifest.json"
        manifest_destination = self.evidence_dir / manifest_relative
        manifest_document = {
            "schema": contract.EVIDENCE_MANIFEST_SCHEMA,
            "category": case["category"],
            "case_id": case["case_id"],
            "case_sha256": contract.case_digest(case),
            "plan_sha256": case["plan_sha256"],
            "deployment_manifest_sha256":
                case["payload"].get("deployment_manifest_sha256"),
            "evidence": digests,
            "files": relative_paths,
        }
        manifest_raw = _canonical_bytes(manifest_document)
        manifest_temporary = manifest_destination.with_name(
            manifest_destination.name + f".{os.getpid()}.tmp")
        with open(manifest_temporary, "wb") as handle:
            handle.write(manifest_raw)
        os.replace(manifest_temporary, manifest_destination)
        return (digests, relative_paths, _digest(manifest_raw),
                manifest_relative.as_posix())

    def execute(self, case):
        validate_reference_case(case)
        outcome = self._outcome(case)
        (evidence, evidence_files, evidence_manifest_sha256,
         evidence_manifest) = self._write_evidence(
             case, self._evidence_records(case, outcome))
        result = {
            "schema": contract.RESULT_SCHEMA,
            "case_id": case["case_id"],
            "case_sha256": contract.case_digest(case),
            "plan_sha256": case["plan_sha256"],
            "deployment_manifest_sha256":
                case["payload"].get("deployment_manifest_sha256"),
            "evidence_manifest_sha256": evidence_manifest_sha256,
            **outcome,
            "request_id": case["request_id"],
            "session_id": case["session_id"],
            "step_id": case["step_id"],
            "evidence": evidence,
            "details": {
                "reference_adapter_schema": ADAPTER_SCHEMA,
                "fixture": self.policy.name,
                "local_only": True,
                "evidence_files": evidence_files,
                "evidence_manifest": evidence_manifest,
            },
        }
        return contract.validate_result(result)


def verify_result_evidence(result, evidence_dir):
    """Verify every reference-adapter evidence path and digest fail-closed."""
    contract.verify_result_evidence_bundle(result, evidence_dir)
    details = result["details"]
    if details.get("reference_adapter_schema") != ADAPTER_SCHEMA:
        raise ValueError("result was not emitted by the reference adapter")
    paths = details.get("evidence_files")
    if not isinstance(paths, dict) or set(paths) != set(result["evidence"]):
        raise ValueError("evidence_files must map every and only evidence digest")
    root = Path(evidence_dir)
    for field, relative in paths.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"evidence path for {field!r} must be non-empty str")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"unsafe evidence path for {field!r}: {relative!r}")
        path = root.joinpath(*pure.parts)
        try:
            actual = _file_sha256(path)
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot verify evidence {relative!r}: {exc}") from exc
        if actual != result["evidence"][field]:
            raise ValueError(f"evidence digest mismatch for {field!r}")
        if document.get("schema") != EVIDENCE_SCHEMA \
                or document.get("artifact") != field \
                or document.get("binding", {}).get("case_id") != result["case_id"]:
            raise ValueError(f"evidence binding mismatch for {field!r}")
    return True


def serve(target, input_stream=None, output_stream=None):
    """Serve the strict one-case/one-result JSONL process contract."""
    return process_adapter.serve(
        target, input_stream=input_stream, output_stream=output_stream,
        case_validator=validate_reference_case)


def create_runtime_target(runtime):
    """Construct the local target through the production factory interface.

    A real Azure/GCP module exposes the same ``factory(runtime)`` shape but
    returns its provider target.  This qualification factory remains local,
    deterministic, credential-free, and cloud-SDK-free.
    """
    if not isinstance(runtime, process_adapter.AdapterRuntime):
        raise TypeError("runtime must be AdapterRuntime")
    return DeterministicLocalTarget("defended", runtime.evidence_dir)


def build_parser():
    parser = argparse.ArgumentParser(
        description=("Local-only reference JSONL adapter for the LLM-Attacker "
                     "secure-compute contract."))
    parser.add_argument("--fixture", choices=FIXTURES, required=True,
                        help="explicit defended or deliberately vulnerable world")
    parser.add_argument("--evidence-dir", required=True,
                        help="directory for category-specific local evidence JSON")
    parser.add_argument("--host-traces", choices=HOST_TRACE_MODES,
                        default="observed",
                        help="emit deterministic observations or explicit "
                             "unavailability (default: observed)")
    return parser


def self_test():
    import tempfile

    plan = contract.build_plan("hybrid", require_host_traces=True)
    with tempfile.TemporaryDirectory() as directory:
        defended = DeterministicLocalTarget(
            "defended", Path(directory) / "defended")
        vulnerable = DeterministicLocalTarget(
            "vulnerable", Path(directory) / "vulnerable")
        defended_rows = [defended.execute(case) for case in plan["cases"]]
        vulnerable_rows = [vulnerable.execute(case) for case in plan["cases"]]
        ok = contract.grade(plan, defended_rows)["overall"] == "PASS"
        gate = contract.grade(plan, vulnerable_rows)
        by_id = {cell["case_id"]: cell for cell in gate["cells"]
                 if cell["case_id"] in {c["case_id"] for c in plan["cases"]}}
        ok = ok and all(
            by_id[case["case_id"]]["verdict"] ==
            fixture_expected_verdict(case, "vulnerable")
            for case in plan["cases"])
        ok = ok and all(verify_result_evidence(
            row, Path(directory) / "defended") for row in defended_rows)
    print("secure-compute reference adapter (local only):")
    print(f"  [{'PASS' if ok else 'FAIL'}] defended/vulnerable qualification")
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        target = DeterministicLocalTarget(
            args.fixture, args.evidence_dir, args.host_traces)
        serve(target)
    except Exception as exc:
        print(f"[secure-compute-reference-adapter] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
