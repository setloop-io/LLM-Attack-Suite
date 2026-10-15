#!/usr/bin/env python3
"""Provider-neutral process host for secure-compute target adapters.

This module is the executable boundary between the LLM-Attacker audit and a
deployment-specific Azure, GCP, or other confidential-compute wrapper.  The
provider wrapper supplies one factory with this signature::

    def create_target(runtime: AdapterRuntime) -> SecureComputeTarget: ...

The returned target executes one validated case at a time.  This host owns the
strict JSONL framing, verifies that every case is bound to the frozen deployment
manifest, validates every returned binding, and resolves every evidence digest
before it releases a result on stdout.  Provider SDKs, credentials, verifier
calls, key release, mutation injection, and evidence acquisition remain in the
provider wrapper.

Configuration may be supplied as flags or through the four non-secret control
environment variables declared below.  Provider-native credential variables or
workload identity are intentionally neither interpreted nor logged here.
"""

import argparse
import importlib
import json
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from . import secure_compute as contract

if __name__ == "__main__":
    sys.modules.setdefault(
        "attacker.secure_compute_process_adapter", sys.modules[__name__])


ENV_TARGET_FACTORY = "LLM_ATTACKER_CC_TARGET_FACTORY"
ENV_PROVIDER = "LLM_ATTACKER_CC_PROVIDER"
ENV_DEPLOYMENT_MANIFEST = "LLM_ATTACKER_CC_DEPLOYMENT_MANIFEST"
ENV_EVIDENCE_DIR = "LLM_ATTACKER_CC_EVIDENCE_DIR"

CONTROL_ENVIRONMENT = (
    ENV_TARGET_FACTORY,
    ENV_PROVIDER,
    ENV_DEPLOYMENT_MANIFEST,
    ENV_EVIDENCE_DIR,
)

_PROVIDER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class AdapterRuntime:
    """Frozen, non-secret context passed to a provider target factory."""

    provider: str
    target_factory: str
    evidence_dir: Path
    deployment_manifest: Path
    deployment_manifest_sha256: str
    deployment: dict

    @classmethod
    def load(cls, provider, target_factory, evidence_dir,
             deployment_manifest):
        if not isinstance(provider, str) or not _PROVIDER_RE.fullmatch(provider):
            raise ValueError(
                "provider must match [A-Za-z0-9][A-Za-z0-9._-]*")
        _split_factory_spec(target_factory)
        if not isinstance(evidence_dir, str) or not evidence_dir.strip():
            raise ValueError("evidence directory must be a non-empty path")
        if not isinstance(deployment_manifest, str) \
                or not deployment_manifest.strip():
            raise ValueError("deployment manifest must be a non-empty path")

        manifest_path = Path(deployment_manifest).expanduser().resolve()
        evidence_path = Path(evidence_dir).expanduser().resolve()
        deployment = contract.load_deployment_manifest(str(manifest_path))
        adapter = deployment["adapter"]
        if adapter["provider"] != provider:
            raise ValueError(
                f"provider {provider!r} does not match deployment manifest "
                f"adapter.provider {adapter['provider']!r}")
        if adapter["target_factory"] != target_factory:
            raise ValueError(
                "target factory does not match deployment manifest "
                "adapter.target_factory")
        evidence_path.mkdir(parents=True, exist_ok=True)
        return cls(
            provider=provider,
            target_factory=target_factory,
            evidence_dir=evidence_path,
            deployment_manifest=manifest_path,
            deployment_manifest_sha256=contract.file_sha256(str(manifest_path)),
            deployment=deployment,
        )

    def validate_case(self, case, path="<case>"):
        contract.validate_case(case, path)
        if case["profile"] != self.deployment["profile"]:
            raise ValueError(
                f"{path}: profile does not match deployment manifest")
        if case["payload"].get("deployment_manifest_sha256") != \
                self.deployment_manifest_sha256:
            raise ValueError(
                f"{path}: deployment manifest digest binding mismatch")
        return case


class SecureComputeTarget(ABC):
    """Interface implemented by an Azure/GCP/provider target wrapper."""

    @abstractmethod
    def execute(self, case):
        """Execute one case and return a strict secure-compute result."""


def _split_factory_spec(spec):
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise ValueError("target factory must be formatted as module:callable")
    module_name, callable_path = spec.split(":", 1)
    if not module_name or not callable_path \
            or any(not part.isidentifier()
                   for part in module_name.split(".") + callable_path.split(".")):
        raise ValueError("target factory must be formatted as module:callable")
    return module_name, callable_path


def load_target(spec, runtime):
    """Load the manifest-bound provider factory and construct its target."""
    module_name, callable_path = _split_factory_spec(spec)
    try:
        value = importlib.import_module(module_name)
        for component in callable_path.split("."):
            value = getattr(value, component)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"cannot load target factory {spec!r}: {exc}") from exc
    if not callable(value):
        raise ValueError(f"target factory {spec!r} is not callable")
    target = value(runtime)
    if not callable(getattr(target, "execute", None)):
        raise ValueError(
            f"target factory {spec!r} must return an object implementing "
            "SecureComputeTarget.execute(case)")
    return target


def _validate_result_binding(case, result, path):
    contract.validate_result(result, path)
    expected = {
        "case_id": case["case_id"],
        "case_sha256": contract.case_digest(case),
        "plan_sha256": case["plan_sha256"],
        "deployment_manifest_sha256":
            case["payload"].get("deployment_manifest_sha256"),
        "request_id": case["request_id"],
        "session_id": case["session_id"],
        "step_id": case["step_id"],
    }
    for key, value in expected.items():
        if result[key] != value:
            raise ValueError(f"{path}: result {key} binding mismatch")
    return result


def serve(target, runtime=None, input_stream=None, output_stream=None,
          case_validator=None):
    """Serve the strict one-case/one-result JSONL contract.

    When ``runtime`` is present (the production/provider path), the host also
    enforces deployment/profile binding and verifies the complete evidence
    bundle before emitting each result.  ``case_validator`` may add stricter
    fixture-only semantics without weakening the generic contract.
    """
    if not callable(getattr(target, "execute", None)):
        raise ValueError("target must implement SecureComputeTarget.execute(case)")
    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    case_validator = contract.validate_case if case_validator is None \
        else case_validator
    count = 0
    for lineno, line in enumerate(input_stream, 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"stdin:{lineno}: invalid JSON: {exc}") from exc
        path = f"stdin:{lineno}"
        case = case_validator(case, path)
        if runtime is not None:
            runtime.validate_case(case, path)
        result = _validate_result_binding(
            case, target.execute(case), f"stdout:{lineno}")
        if runtime is not None:
            contract.verify_result_evidence_bundle(
                result, str(runtime.evidence_dir))
        output_stream.write(json.dumps(
            result, sort_keys=True, separators=(",", ":")) + "\n")
        output_stream.flush()
        count += 1
    if count == 0:
        raise ValueError("stdin contained no secure-compute cases")
    return count


def _default(environ, key):
    value = environ.get(key)
    return value if value else None


def build_parser(environ=None):
    environ = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(
        description=("Provider-neutral secure-compute JSONL adapter host. "
                     "The target factory is frozen in the deployment manifest."))
    parser.add_argument(
        "--target-factory", default=_default(environ, ENV_TARGET_FACTORY),
        help=f"provider factory module:callable (or ${ENV_TARGET_FACTORY})")
    parser.add_argument(
        "--provider", default=_default(environ, ENV_PROVIDER),
        help=f"manifest-bound provider identifier (or ${ENV_PROVIDER})")
    parser.add_argument(
        "--deployment-manifest",
        default=_default(environ, ENV_DEPLOYMENT_MANIFEST),
        help=f"frozen deployment JSON (or ${ENV_DEPLOYMENT_MANIFEST})")
    parser.add_argument(
        "--evidence-dir", default=_default(environ, ENV_EVIDENCE_DIR),
        help=f"access-controlled evidence root (or ${ENV_EVIDENCE_DIR})")
    return parser


def main(argv=None, environ=None):
    parser = build_parser(environ)
    args = parser.parse_args(argv)
    missing = [name for name, value in (
        ("--target-factory", args.target_factory),
        ("--provider", args.provider),
        ("--deployment-manifest", args.deployment_manifest),
        ("--evidence-dir", args.evidence_dir),
    ) if not value]
    if missing:
        parser.error("missing required provider controls: " + ", ".join(missing))
    try:
        runtime = AdapterRuntime.load(
            args.provider, args.target_factory, args.evidence_dir,
            args.deployment_manifest)
        target = load_target(args.target_factory, runtime)
        serve(target, runtime=runtime)
    except Exception as exc:
        print(f"[secure-compute-process-adapter] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
