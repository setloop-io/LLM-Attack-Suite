#!/usr/bin/env python3
"""One finite, cross-paper TEE/FHE/hybrid pre-cloud security audit.

The attack emits a digest-bound plan and either grades an existing JSONL
result transcript or drives an evaluator-supplied process adapter.  The adapter
receives one ``llm_attacker.secure_compute_case.v1`` object per stdin line and
must return one ``llm_attacker.secure_compute_result.v1`` object per stdout
line.  This makes vendor SDK code replaceable while the cases and gate stay
identical across deployments and papers.

No cloud target is contacted unless the evaluator explicitly supplies
``--driver-command``.  ``--toy`` is only a local machinery check.
"""

import argparse
import json

from .. import artifacts, secure_compute, threat_model
from .common import add_common_args

EXPERIMENT_ID = "secure_compute_audit"
MODES = ("training", "inference")
REQUIRES_LABELS = "canaries and trusted diagnostic controls"
DESCRIPTION = ("digest-bound confidential-compute admission battery: "
               "attestation/key release, live protocol mutation, malicious "
               "computation, dual-view boundary canaries, host traces, and "
               "conditional FHE feedback/oracle checks")

ARTIFACT_SCHEMA = "dtraining.attacker.secure_compute_audit.v1"


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--profile", choices=secure_compute.PROFILES, default="tee",
                    help="deployment profile; fhe/hybrid add ciphertext and "
                         "feedback-oracle cases")
    ap.add_argument("--results", default=None,
                    help="offline strict result JSONL from a vendor adapter")
    ap.add_argument("--driver-command", default=None,
                    help="explicit process adapter command; plan JSONL is sent "
                         "on stdin and strict result JSONL is read from stdout "
                         "(executed without a shell)")
    ap.add_argument("--driver-timeout", type=int, default=900,
                    help="whole adapter timeout in seconds (default: 900)")
    ap.add_argument("--plan-output", default=None,
                    help="write the digest-bound case plan as JSON")
    ap.add_argument("--deployment-manifest", default=None,
                    help="frozen deployment-contract JSON to bind by "
                         "SHA-256; mandatory for real execution or grading")
    ap.add_argument("--evidence-bundle-dir", default=None,
                    help="local root containing the per-case evidence manifests "
                         "and files; mandatory for real execution or grading")
    ap.add_argument("--plan-only", action="store_true",
                    help="write --plan-output and stop without execution")
    ap.add_argument("--require-host-traces", action="store_true",
                    help="make every host-trace observation mandatory; by "
                         "default unavailable traces remain UNASSESSED")
    ap.add_argument("--evidence", action="store_true",
                    help="write the framework evidence chain beside --output")
    return ap


def _threat_model(profile):
    return threat_model.skeleton(
        summary=("malicious cloud operator, host/hypervisor administrator, "
                 "network relay, or accelerator attempts to obtain protected "
                 "training/inference state, trigger secret release for an "
                 "invalid confidential-compute identity, or substitute "
                 "protocol/computation results. Trusted diagnostic plaintext "
                 "is evaluation truth and is never attacker-visible."),
        scope={
            "use_case": "split-training",
            "deployment": "multi-tenant",
            "in_scope": ["attestation and key release", "bidirectional live protocol",
                         "remote computation/result binding", "CPU/GPU and persistent boundaries",
                         "host-visible traces", "FHE feedback surface when applicable"],
            "out_of_scope": ["physical probing unless the deployment exposes a qualified adapter",
                              "unreported microarchitectural channels", "claims about other hardware"],
        },
        actor={
            "role": "cloud",
            "intent": "malicious", "persistence": "cross-session",
            "vantage": ["host OS or hypervisor", "network path", "remote accelerator"],
            "authorization": "platform administration but no valid tenant secret",
            "capabilities": ["replay, reorder, substitute and corrupt traffic",
                             "present invalid attestation evidence",
                             "alter weights, operators, gradients and results",
                             "observe declared host-visible side channels"],
        },
        target={
            "assets": ["tenant plaintext", "keys", "model and optimizer state",
                       "training/inference integrity"],
            "components": ["attester", "verifier", "key-release service",
                           "secure channel", "remote compute backend"],
            "data_flows": ["forward", "backward", "update", "checkpoint and telemetry"],
            "trust_boundaries": ["external_attacker versus trusted_diagnostic views"],
        },
        scenario={
            "lifecycle_stage": ["training", "inference", "deployment", "operations"],
            "security_properties": ["confidentiality", "integrity", "authenticity",
                                    "authorization"],
            "attack_steps": ["execute every digest-bound case once through the target adapter",
                             "bind each response to request/session/step",
                             "apply the exact fail-closed admission gate"],
            "technical_impact": ["secret release", "plaintext exposure", "accepted incorrect result"],
        },
        controls={
            "assumed": ["vendor adapter preserves case identifiers and emits no raw secret"],
            "tested": ["attestation policy", "channel/result binding", "boundary confinement",
                       "ciphertext rejection and feedback blinding when applicable"],
            "detective": ["independent canaries", "trusted positive controls"],
            "response": ["fail admission; do not launch scientific or paid cohort"],
            "residual_risk": "UNASSESSED traces remain named; a pass is scoped to the tested platform.",
            "owner": "experiment owner", "status": "mitigated",
        },
        mappings={"owasp_llm_2026": ["LLM02", "LLM05", "LLM06", "LLM08"],
                  "nist_ai_100_2": ["evasion", "privacy", "poisoning"]},
        evaluation={
            "success_metrics": ["valid release succeeds", "zero invalid secret releases",
                                "zero accepted protocol/computation mutations",
                                "zero external canary exposures", "trusted canaries observed"],
            "baseline": "intentionally leaky FHE oracle and trusted canary controls",
            "budget": {"logical_cases": "finite plan; one result per case"},
            "limitations": ["toy results are machinery checks only",
                            "adapter quality and platform coverage bound conclusions"],
            "coverage_status": "partial",
        })


def _evidence(args):
    if not args.evidence:
        return None
    if not args.output:
        print(f"[{EXPERIMENT_ID}] --evidence needs --output; continuing without it")
        return None
    from ..evidence import EvidenceLogger
    return EvidenceLogger(args.output, node_role=f"attacker/{EXPERIMENT_ID}")


def _summary(gate):
    categories = sorted({c["category"] for c in gate["cells"]})
    out = []
    for category in categories:
        cells = [c for c in gate["cells"] if c["category"] == category]
        counts = {state: sum(c["verdict"] == state for c in cells)
                  for state in ("PASS", "FAIL", "UNASSESSED")}
        verdict = "FAIL" if counts["FAIL"] else (
            "UNASSESSED" if counts["UNASSESSED"] else "PASS")
        out.append({"experiment": EXPERIMENT_ID, "category": category,
                    "verdict": verdict, **counts})
    return out


def run(args):
    if args.driver_timeout <= 0:
        raise SystemExit(f"[{EXPERIMENT_ID}] --driver-timeout must be positive")
    chosen = sum(bool(x) for x in (args.toy, args.results, args.driver_command))
    if not args.plan_only and chosen != 1:
        raise SystemExit(f"[{EXPERIMENT_ID}] choose exactly one of --toy, "
                         "--results, or --driver-command")
    real_mode = bool(args.results or args.driver_command)
    if real_mode and not args.deployment_manifest:
        raise SystemExit(f"[{EXPERIMENT_ID}] real execution/grading requires "
                         "--deployment-manifest")
    if real_mode and not args.evidence_bundle_dir:
        raise SystemExit(f"[{EXPERIMENT_ID}] real execution/grading requires "
                         "--evidence-bundle-dir")
    if args.deployment_manifest:
        secure_compute.load_deployment_manifest(
            args.deployment_manifest, expected_profile=args.profile)
        manifest_sha256 = secure_compute.file_sha256(args.deployment_manifest)
    else:
        manifest_sha256 = None
    plan = secure_compute.build_plan(
        args.profile, args.require_host_traces, manifest_sha256)
    if args.plan_output:
        secure_compute.write_plan(args.plan_output, plan)
        print(f"[plan] wrote {args.plan_output} ({plan['sha256']})")
    if args.plan_only:
        if not args.plan_output:
            raise SystemExit(f"[{EXPERIMENT_ID}] --plan-only needs --plan-output")
        return 0
    driver_stderr_lines = 0
    if args.toy:
        results = secure_compute.make_toy_results(plan)
        source = "toy"
    elif args.results:
        results = secure_compute.read_jsonl(args.results)
        source = "results-jsonl"
    else:
        results, stderr = secure_compute.run_process_driver(
            args.driver_command, plan, args.driver_timeout)
        driver_stderr_lines = len(stderr.splitlines())
        source = "process-driver"

    gate = secure_compute.grade(
        plan, results, evidence_bundle_dir=args.evidence_bundle_dir,
        require_evidence_bundle=real_mode)
    ev = _evidence(args)
    out = artifacts.make_artifact(
        ARTIFACT_SCHEMA,
        {"attack": EXPERIMENT_ID, "mode": getattr(args, "mode", None),
         "profile": args.profile, "source": source, "toy": bool(args.toy),
         "plan_schema": secure_compute.PLAN_SCHEMA,
         "plan_sha256": plan["sha256"],
         "deployment_manifest_sha256": manifest_sha256,
         "evidence_bundle_verified": bool(real_mode),
         "require_host_traces": bool(args.require_host_traces),
         "driver_stderr_lines": driver_stderr_lines},
        _threat_model(args.profile),
        interpretation=("PASS is scoped to this exact plan and target. FAIL blocks "
                        "cloud science. UNASSESSED is preserved and is never converted "
                        "to a pass. Toy artifacts are not scientific evidence."),
        evidence=ev)
    out["plan"] = plan
    out["results"] = results
    out["gate"] = gate
    out["summary"] = _summary(gate)
    for result in results:
        artifacts.append_jsonl(args.output, {"experiment": EXPERIMENT_ID, **result})
    artifacts.write_artifact(args.output, out, evidence=ev)
    print(f"[{EXPERIMENT_ID}] gate={gate['overall']} "
          f"pass={gate['counts']['PASS']} fail={gate['counts']['FAIL']} "
          f"unassessed={gate['counts']['UNASSESSED']}")
    return (1 if gate["overall"] == "FAIL" else
            2 if gate["overall"] == "UNASSESSED" else 0)


def self_test():
    ok = secure_compute.self_test() == 0
    ap = build_parser()
    args = ap.parse_args(["--toy", "--quick"])
    args.mode = "training"
    ok = (run(args) == 2) and ok
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    ns = build_parser().parse_args()
    ns.mode = None
    raise SystemExit(run(ns))
