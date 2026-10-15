#!/usr/bin/env python3
"""Coverage manifest (THREAT_MODEL_REVIEW Release 1, item 3).

Adding attacks without changing coverage reporting lets incomplete
assessments look comprehensive.  This module makes coverage a first-class
result: given a set of run artifacts, it reports every threat in the
framework's threat universe as exactly one of

    COVERED       every attack mapped to the threat produced at least one
                  valid (non-error) result
    PARTIAL       some, but not all, mapped attacks produced valid results
                  (the untested ones are named, not silent)
    UNMEASURED    applicable, but no mapped attack produced a valid result
    OUT_OF_SCOPE  explicitly inapplicable, with a documented reason

UNMEASURED and OUT_OF_SCOPE are never conflated.  The overall summary
reports the number and severity of unmeasured applicable risks alongside
the measured ones.

Threat universe: the framework's surface model (S1-S14,
README.md), the OWASP Top 10 for LLM Applications 2026
(LLM01-LLM10), and the OWASP Top 10 for Agentic Applications 2026
(ASI01-ASI10) — the last declared OUT_OF_SCOPE by default because the
framework has no agentic target surface (no tools, memory, or inter-agent
channels); pass ``applicable_agentic=True`` to make them applicable.

Attack -> threat mapping is derived from attacker.attack_metadata, so a
new attack joins the manifest the day its metadata lands.  An artifact
"measures" attack A when at least one result/summary record carries A's
EXPERIMENT_ID without an ``error`` field.

CLI:  python -m attacker --coverage ART1.json [ART2.json ...] [--output M]

Pure stdlib; torch-free (repo rule).
"""

import argparse
import json
import sys

from . import attack_metadata

SCHEMA = "llm_attacker.coverage.v1"
STATUSES = ("COVERED", "PARTIAL", "UNMEASURED", "OUT_OF_SCOPE")

OWASP_LLM_NAMES = {
    "LLM01": "Prompt Injection",
    "LLM02": "Sensitive Information Disclosure",
    "LLM03": "Excessive Agency",
    "LLM04": "Supply Chain",
    "LLM05": "Data and Model Poisoning",
    "LLM06": "Unbounded Consumption",
    "LLM07": "Misinformation",
    "LLM08": "Hidden Context Exposure",
    "LLM09": "Vector and Embedding Weaknesses",
    "LLM10": "Improper Output Handling",
}

OWASP_AGENTIC_NAMES = {
    "ASI01": "Agent Goal Hijack", "ASI02": "Tool Misuse and Exploitation",
    "ASI03": "Identity and Privilege Abuse", "ASI04": "Agentic Supply Chain",
    "ASI05": "Unexpected Code Execution", "ASI06": "Memory and Context Poisoning",
    "ASI07": "Insecure Inter-Agent Communication", "ASI08": "Cascading Failures",
    "ASI09": "Human-Agent Trust Exploitation", "ASI10": "Rogue Agents",
}

DEFAULT_OUT_OF_SCOPE = {
    **{f"ASI{i:02d}": "no agentic target surface: the framework evaluates "
       "split/distributed LLMs without tools, memory, or inter-agent "
       "channels (declare applicable only for agentic targets)"
       for i in range(1, 11)},
}


def threat_universe(applicable_agentic=False, extra_out_of_scope=None):
    """{threat_id: {"name", "kind", "out_of_scope": reason-or-None}} —
    the full list of threats the manifest reports on."""
    out = {}
    for sid, sname in sorted(attack_metadata.SURFACES.items()):
        out[sid] = {"name": sname, "kind": "surface", "out_of_scope": None}
    for rid, rname in sorted(OWASP_LLM_NAMES.items()):
        out[rid] = {"name": rname, "kind": "owasp-llm-2026",
                    "out_of_scope": None}
    for aid, aname in sorted(OWASP_AGENTIC_NAMES.items()):
        out[aid] = {"name": aname, "kind": "owasp-agentic-2026",
                    "out_of_scope": None if applicable_agentic
                    else DEFAULT_OUT_OF_SCOPE[aid]}
    for tid, reason in (extra_out_of_scope or {}).items():
        if tid in out:
            out[tid]["out_of_scope"] = reason
        else:
            out[tid] = {"name": tid, "kind": "custom",
                        "out_of_scope": reason}
    return out


def mapped_attacks(threat_id):
    """Attack names mapped to a threat via attack_metadata (surfaces map
    by S-id, OWASP LLM risks by LLM-id, OWASP agentic risks by ASI-id;
    custom threats map none until a metadata entry says so)."""
    if threat_id in attack_metadata.SURFACES:
        return attack_metadata.attacks_for_surface(threat_id)
    if threat_id in OWASP_LLM_NAMES:
        return attack_metadata.attacks_for_owasp(threat_id)
    if threat_id in OWASP_AGENTIC_NAMES:
        return attack_metadata.attacks_for_agentic(threat_id)
    return []


COMPLETENESS = {
    "tool_misuse": {
        "arms": ("dangerous-parameters", "scope-confusion",
                 "side-effect-amplification", "authorized-task"),
        "worlds": ("vulnerable", "defended"),
        "metrics": ("trials", "total_violations", "violation_rate",
                    "wilson_upper"),
    },
    "agent_goal_hijack": {
        "arms": ("tool-output-hijack", "retrieved-document-hijack",
                 "benign-task"),
        "worlds": ("vulnerable", "defended", "none", "provenance-only"),
        "metrics": ("trials", "total_violations", "violation_rate",
                    "wilson_upper", "violations", "false_blocks"),
    },
    "human_agent_trust": {
        "arms": ("summary-mismatch", "hidden-action-detail", "benign-task"),
        "worlds": ("vulnerable", "defended", "none",
                   "display-integrity-only"),
        "metrics": ("trials", "total_violations", "violation_rate",
                    "wilson_upper", "violations", "false_blocks"),
    },
    "memory_context_poisoning": {
        "arms": ("cross-session-memory", "delayed-benign-then-trigger",
                 "benign-task"),
        "worlds": ("vulnerable", "defended", "none", "memory-provenance-only",
                   "instruction-provenance-only"),
        "metrics": ("trials", "total_violations", "violation_rate",
                    "wilson_upper", "violations", "false_blocks"),
    },
    "agentic_supply_chain": {
        "arms": ("compromised-component", "unsigned-default-binding",
                 "benign-task"),
        "worlds": ("vulnerable", "defended", "none",
                   "component-provenance-only", "instruction-provenance-only"),
        "metrics": ("trials", "total_violations", "violation_rate",
                    "wilson_upper", "violations", "false_blocks"),
    },
    "identity_privilege_abuse": {
        "arms": ("confused-deputy", "role-token-forgery",
                 "lateral-session-use", "authorized-task"),
        "worlds": ("vulnerable", "defended", "none",
                   "principal-binding-only", "instruction-provenance-only"),
        "metrics": ("trials", "total_violations", "violation_rate",
                    "wilson_upper", "violations", "false_blocks"),
    },
    "cascading_failure": {
        "arms": ("broken-chain-redirect", "benign-task"),
        "worlds": ("vulnerable", "defended", "none",
                   "cascade-revalidation-only", "instruction-provenance-only",
                   "parameter-typecheck-only", "tenant-containment-only",
                   "component-provenance-only"),
        "metrics": ("trials", "total_violations", "violation_rate",
                    "wilson_upper", "violations", "false_blocks"),
    },
    "unexpected_code_execution": {
        "arms": ("direct-injection", "document-injection", "approval-swap",
                 "authorized-exec"),
        "worlds": ("vulnerable", "defended", "none", "exec-gate-only",
                   "instruction-provenance-only", "parameter-typecheck-only",
                   "tenant-containment-only", "component-provenance-only",
                   "principal-binding-only"),
        "metrics": ("trials", "total_violations", "violation_rate",
                    "wilson_upper", "violations", "false_blocks"),
    },
    "rogue_agent": {
        "arms": ("unauthorized-replication", "goal-self-modification",
                 "persistence-reload", "attested-lifecycle"),
        "worlds": ("vulnerable", "defended", "none",
                   "lifecycle-attestation-only",
                   "instruction-provenance-only", "memory-provenance-only",
                   "parameter-typecheck-only", "tenant-containment-only",
                   "principal-binding-only"),
        "metrics": ("trials", "total_violations", "violation_rate",
                    "wilson_upper", "violations", "false_blocks"),
    },
    "insecure_agent_communication": {
        "arms": ("sender-spoofing", "tamper-in-transit", "replay",
                 "authentic-relay"),
        "worlds": ("vulnerable", "defended", "none",
                   "message-authentication-only",
                   "instruction-provenance-only", "memory-provenance-only",
                   "component-provenance-only", "cascade-revalidation-only",
                   "principal-binding-only", "parameter-typecheck-only",
                   "tenant-containment-only"),
        "metrics": ("trials", "total_violations", "violation_rate",
                    "wilson_upper", "violations", "false_blocks"),
    },
    "cross_agent_deputy": {
        "arms": ("forwarded-deputy-request", "sanctioned-peer-request"),
        "worlds": ("vulnerable", "defended", "none",
                   "principal-binding-only", "message-authentication-only",
                   "instruction-provenance-only", "memory-provenance-only",
                   "component-provenance-only", "cascade-revalidation-only",
                   "tenant-containment-only", "parameter-typecheck-only"),
        "metrics": ("trials", "total_violations", "violation_rate",
                    "wilson_upper", "violations", "false_blocks"),
    },
}


def _integrity_completeness():
    """The seven integrity/availability experiments' completeness maps (seed
    G2/G3), keyed on the RAW summary-row coordinates: `world` absent -> None
    (resource/poisoning/supply-chain), `arm` absent -> None (malicious-server).
    Their topology is NON-Cartesian (protocol-integrity has 21 cells, not
    11x2; key-session 17, not 8x3) and their metric keys differ per cell, so
    the rectangular {"arms","worlds","metrics"} form cannot express them —
    hence the explicit {(arm, world): (metric_keys,)} map. Verdict-blind: it
    checks cell + key PRESENCE, never a value (calibration Appendix A)."""
    out = {}
    pi = {(a, w): ("undetected_mutation_rate_mean",)
          for a in ("cross-session-swap", "downgrade", "drop", "duplicate",
                    "epoch-desync", "oversize", "reorder", "replay",
                    "tensor-tamper", "truncate")
          for w in ("defended", "vulnerable")}
    pi[("negative-control", "defended")] = ("false_rejection_rate_mean",)
    out["protocol_integrity"] = {"cells": pi}
    out["malicious_client"] = {"cells": {
        (a, w): ("attack_success_rate",)
        for a in ("budget-extraction", "collusion", "crafted-extraction",
                  "gradient-poisoning", "instability-bomb",
                  "neuron-observability")
        for w in ("defended", "isolated", "vulnerable")}}
    ks = {(a, w): ("attack_success_rate",)
          for a in ("collusion", "epoch-rollback", "forward-secrecy",
                    "master-seed-exposure", "negotiation-downgrade",
                    "nonce-reuse", "session-binding")
          for w in ("defended", "vulnerable")}
    for w in ("control", "defended", "vulnerable"):
        ks[("key-in-logs", w)] = ("attack_success_rate",)
    out["key_session_abuse"] = {"cells": ks}
    sc = {(a, None): ("bypass_rate_enforcing", "bypass_rate_permissive")
          for a in ("artifact-substitution", "digest-mismatch",
                    "unsigned-promotion", "rollback", "tokenizer-drift",
                    "adapter-confusion", "plugin-origin",
                    "unsafe-serialization", "dependency-drift")}
    sc[("all", None)] = ("negative_control_false_rejections",)
    out["supply_chain"] = {"cells": sc}
    rex = {(a, None): ("amplification_vs_control_mean",)
           for a in ("batch-fragmentation", "cache-thrash",
                     "decode-amplification", "long-context", "memory-pressure",
                     "protocol-oversize", "retry-storm",
                     "tokenizer-amplification")}
    rex[("benign-control", None)] = ("amplification_parity_mean",)
    rex[("training-skew", None)] = ("cost_amplification_mean",)
    out["resource_exhaustion"] = {"cells": rex}
    pb = {(a, None): ("asr_undefended_mean_max_budget", "asr_defended_mean",
                      "random_corruption_asr_mean")
          for a in ("adapter-backdoor", "checkpoint-backdoor", "clean-label",
                    "dirty-label", "gradient-sign", "model-replacement",
                    "rag-poison", "trigger-backdoor")}
    pb[("zero-poison", None)] = ("asr_mean",)
    out["poisoning_backdoor"] = {"cells": pb}
    out["malicious_server_steering"] = {"cells": {
        (None, "honest"): ("seq_accuracy_mean", "control_accuracy_mean"),
        (None, "crafted"): ("seq_accuracy_mean", "control_accuracy_mean")}}
    return out


COMPLETENESS.update(_integrity_completeness())


def _privacy_completeness():
    """The split-privacy.v2 required-attack completeness maps (seed Lane C),
    keyed on the RAW summary-row coordinates a real toy run emits. Like the
    integrity maps this is verdict-blind (cell + metric-key PRESENCE, never a
    value) and OUTCOME-INDEPENDENT. Two shapes are used: the non-Cartesian
    {(arm, world): (keys,)} form where arms carry heterogeneous metrics or an
    axis is absent (-> None), and the rectangular {"arms","worlds","metrics"}
    form where a genuine arm x world rectangle carries one uniform metric set.
    gradient-inversion (dlgpp, torch-only) and membership (ea4, external
    module) are NOT gated: neither runs offline, so no honest cell set can be
    verified — they stay ungated pending a runnable tier."""
    out = {}
    out["seam_inversion"] = {"cells": {
        (None, None): ("seq_accuracy_mean", "control_accuracy_mean")}}
    for exp in ("norm_based_scoring", "lira_membership"):
        out[exp] = {"cells": {(None, "null"): ("auc_mean",),
                              (None, "separable"): ("auc_mean",)}}
    out["gpu_memory_residue"] = {"cells": {
        ("allocator-residue", None): ("token_recovery_mean",),
        ("scrub-check", None): ("canary_recovery_mean",),
        ("um-timing", None): ("schedule_recovery_auc_mean",)}}
    out["wire_shape_analysis"] = {"cells": {
        ("length-leak", None): ("mae_tokens_mean", "rank_corr_mean"),
        ("phase-detect", None): ("f1_mean", "label_accuracy_mean")}}
    out["multi_tenant_isolation"] = {
        "arms": ("session-collision", "continuous-batch-mixup",
                 "error-path-bleed", "kv-content-canary",
                 "prefix-cache-isolation", "response-misattribution",
                 "allocator-reuse"),
        "worlds": ("vulnerable", "defended", "isolated"),
        "metrics": ("canary_recovery_exact_mean",
                    "isolation_failures_per_million_mean")}
    rsc = {"contention-fingerprinting": ("auc_mean", "tpr_at_1pct_fpr_mean"),
           "expert-map-extraction": ("fidelity_mean", "lift_pp_mean"),
           "length-domain-inference": ("domain_accuracy_mean",
                                       "mean_domain_auc_mean"),
           "linkability": ("auc_mean", "tpr_at_1pct_fpr_mean"),
           "property-classification": ("auc_mean", "tpr_at_1pct_fpr_mean")}
    out["router_side_channel"] = {"cells": {
        (arm, world): keys for arm, keys in rsc.items()
        for world in ("informative", "null")}}
    out["artifact_telemetry_leak"] = {"cells": {
        ("store-scan", None): ("n_hits_total",
                               "unauthorized_readability_total")}}
    return out


def _genai_completeness():
    """The genai-app.v1 required-attack completeness maps (seed Lane C). These
    modules emit the arm x world rectangle in their RESULT rows (the summary is
    a single aggregate row), so the guard keys on the per-cell result records —
    the same shape the four agentic entries use. Verdict-blind and
    outcome-independent. multi_tenant_isolation + artifact_telemetry_leak are
    shared with split-privacy.v2 and gated in _privacy_completeness."""
    out = {}
    out["prompt_injection"] = {
        "arms": ("direct-override", "indirect-document",
                 "cross-session-persistence", "payload-splitting",
                 "encoded-unicode", "adaptive"),
        "worlds": ("vulnerable", "defended"),
        "metrics": ("trials", "total_violations", "violation_rate",
                    "wilson_upper")}
    out["rag_vector_abuse"] = {
        "arms": ("cross-tenant-retrieval", "namespace-bypass",
                 "poisoned-ingestion", "embedding-inversion",
                 "membership-linkability", "stale-retention",
                 "retrieval-flooding", "citation-substitution"),
        "worlds": ("vulnerable", "defended"),
        "metrics": ("rate", "trials", "wilson_upper")}
    out["hidden_context_exposure"] = {
        "arms": ("direct-extraction", "behavioral-inference",
                 "prefix-continuation", "encoded-extraction",
                 "multi-turn-accumulation"),
        "worlds": ("vulnerable", "defended", "empty"),
        "metrics": ("exact_recovery", "semantic")}
    out["output_sink_injection"] = {
        "arms": ("html-xss", "sql-injection", "shell-command",
                 "template-ssti", "url-ssrf", "file-path",
                 "generated-code", "structured-output", "adaptive"),
        "worlds": ("vulnerable", "defended"),
        "metrics": ("trials", "effect_rate", "wilson_upper")}
    mii = {"grounded-qa": ("accuracy",),
           "citation-fidelity": ("entailment_rate",),
           "adversarial-premise": ("premise_acceptance_rate",),
           "fabricated-sources": ("fabricated_citation_rate",),
           "calibration": ("ece",),
           "contradiction": ("contradiction_rate",),
           "temporal-staleness": ("stale_answer_rate",),
           "downstream-decision": ("action_harm_rate",)}
    out["misinformation_integrity"] = {"cells": {
        (arm, world): keys for arm, keys in mii.items()
        for world in ("vulnerable", "defended")}}
    return out


COMPLETENESS.update(_privacy_completeness())
COMPLETENESS.update(_genai_completeness())


def _cells_complete(records, req):
    """True iff `records` holds a non-error result cell for every required
    cell with all its required metric keys present. Verdict-blind.

    Two entry shapes are accepted (additive; seed G2):
      - explicit {"cells": {(arm, world): (metric_keys,)}} — a NON-Cartesian
        map with per-cell metric keys (the seven integrity entries);
      - rectangular {"arms","worlds","metrics"} — need = arms x worlds with one
        metric set for every cell (the four agentic entries, unchanged)."""
    if "cells" in req:
        need = set(req["cells"])
        have = set()
        for rec in records:
            if not isinstance(rec, dict) or "error" in rec:
                continue
            metrics = req["cells"].get((rec.get("arm"), rec.get("world")))
            if metrics is not None and all(k in rec for k in metrics):
                have.add((rec.get("arm"), rec.get("world")))
        return need <= have
    have = set()
    for rec in records:
        if not isinstance(rec, dict) or "error" in rec:
            continue
        cell = (rec.get("arm"), rec.get("world"))
        if cell[0] in req["arms"] and cell[1] in req["worlds"] \
                and all(k in rec for k in req["metrics"]):
            have.add(cell)
    need = {(a, w) for a in req["arms"] for w in req["worlds"]}
    return need <= have


def experiments_of(artifact):
    """The set of EXPERIMENT_IDs with at least one valid (non-error)
    result in one artifact dict."""
    good = set()
    records = list(artifact.get("results") or [])
    records += list(artifact.get("summary") or [])
    for rec in records:
        if isinstance(rec, dict) and "experiment" in rec \
                and "error" not in rec:
            good.add(rec["experiment"])
    for exp, req in COMPLETENESS.items():
        if exp in good and not _cells_complete(records, req):
            good.discard(exp)
    return good


def load_artifacts(paths):
    """Read artifact JSONs; a missing/unparseable file is a descriptive
    ValueError (fail-closed: never silently drop evidence)."""
    arts = []
    for p in paths:
        try:
            with open(p) as f:
                arts.append(json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"coverage: cannot read artifact {p!r}: {e}") \
                from e
    return arts


def _attack_experiment_ids():
    """EXPERIMENT_ID -> attack registry name (lazy import: the registry
    validates all built-ins at import; torch-free by repo rule)."""
    from .attacks import ATTACKS, load
    out = {}
    for name in ATTACKS:
        try:
            out[getattr(load(name), "EXPERIMENT_ID", None)] = name
        except Exception:
            continue
    out.pop(None, None)
    return out


def build_manifest(artifacts, applicable_agentic=False,
                   extra_out_of_scope=None, universe=None):
    """The coverage manifest for a list of artifact dicts (or paths)."""
    if artifacts and all(isinstance(a, str) for a in artifacts):
        artifacts = load_artifacts(artifacts)
    exp_to_attack = _attack_experiment_ids()
    measured_attacks = set()
    unknown_experiments = set()
    for art in artifacts:
        for exp in experiments_of(art):
            if exp in exp_to_attack:
                measured_attacks.add(exp_to_attack[exp])
            else:
                unknown_experiments.add(exp)

    threats = {}
    universe = universe if universe is not None else threat_universe(
        applicable_agentic=applicable_agentic,
        extra_out_of_scope=extra_out_of_scope)
    for tid, spec in sorted(universe.items()):
        mapped = mapped_attacks(tid)
        measured = sorted(m for m in mapped if m in measured_attacks)
        if spec["out_of_scope"]:
            status = "OUT_OF_SCOPE"
        elif not mapped:
            status = "UNMEASURED"
        elif set(measured) == set(mapped):
            status = "COVERED"
        elif measured:
            status = "PARTIAL"
        else:
            status = "UNMEASURED"
        threats[tid] = {"name": spec["name"], "kind": spec["kind"],
                        "status": status, "mapped_attacks": sorted(mapped),
                        "measured_attacks": measured,
                        "untested_attacks": sorted(set(mapped)
                                                   - set(measured))}
        if spec["out_of_scope"]:
            threats[tid]["out_of_scope_reason"] = spec["out_of_scope"]
    counts = {s: sum(1 for t in threats.values() if t["status"] == s)
              for s in STATUSES}
    return {"schema": SCHEMA,
            "coverage_kind": "execution_only",
            "assurance": "Execution coverage does not establish target-specific empirical security or cross-target coverage.",
            "threats": threats,
            "summary": {
                **counts,
                "applicable": counts["COVERED"] + counts["PARTIAL"]
                + counts["UNMEASURED"],
                "unmeasured_applicable": sorted(
                    t for t, v in threats.items()
                    if v["status"] == "UNMEASURED"
                    and not v.get("out_of_scope_reason")),
            },
            "attacks_measured": sorted(measured_attacks),
            "unknown_experiments": sorted(unknown_experiments),
            "artifact_count": len(artifacts)}


def render_manifest_md(manifest):
    """Human-readable coverage table (markdown)."""
    lines = ["# Coverage manifest", "",
             f"schema: `{manifest['schema']}` — artifacts: "
             f"{manifest['artifact_count']}", "",
             "| threat | name | status | measured | untested |", 
             "|---|---|---|---|---|"]
    for tid, t in sorted(manifest["threats"].items()):
        note = ", ".join(t["measured_attacks"]) or "—"
        gaps = ", ".join(t["untested_attacks"]) or "—"
        if t["status"] == "OUT_OF_SCOPE":
            gaps = t.get("out_of_scope_reason", "")[:60] or "—"
        lines.append(f"| {tid} | {t['name']} | {t['status']} | {note} | "
                     f"{gaps} |")
    s = manifest["summary"]
    lines += ["",
              f"**Summary:** COVERED {s['COVERED']}, PARTIAL {s['PARTIAL']}, "
              f"UNMEASURED {s['UNMEASURED']} (of which applicable: "
              f"{len(s['unmeasured_applicable'])}), OUT_OF_SCOPE "
              f"{s['OUT_OF_SCOPE']}"]
    if s["unmeasured_applicable"]:
        lines.append(f"**Unmeasured applicable risks:** "
                     + ", ".join(s["unmeasured_applicable"]))
    if manifest["unknown_experiments"]:
        lines.append(f"**Artifacts with unmapped experiment IDs:** "
                     + ", ".join(manifest["unknown_experiments"]))
    return "\n".join(lines) + "\n"


def build_parser():
    ap = argparse.ArgumentParser(
        prog="python -m attacker --coverage",
        description=__doc__)
    ap.add_argument("artifacts", nargs="*", help="attack artifact JSONs")
    ap.add_argument("--output", default=None, help="manifest JSON path")
    ap.add_argument("--md", default=None, help="markdown report path")
    ap.add_argument("--applicable-agentic", action="store_true",
                    help="treat OWASP agentic risks as applicable (agentic "
                         "targets only)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = build_manifest(load_artifacts(args.artifacts),
                              applicable_agentic=args.applicable_agentic)
    text = render_manifest_md(manifest)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[coverage] wrote {args.output}")
    if args.md:
        with open(args.md, "w") as f:
            f.write(text)
        print(f"[coverage] wrote {args.md}")
    return 0


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("coverage manifest (pure python):")
    art = {"results": [{"experiment": "wire_shape_analysis",
                        "arm": "length-leak", "mae_tokens_mean": 1.0,
                        "rank_corr_mean": 0.9},
                       {"experiment": "wire_shape_analysis",
                        "arm": "phase-detect", "f1_mean": 0.8,
                        "label_accuracy_mean": 0.7},
                       {"experiment": "gpu_memory_residue", "error": "boom"}],
           "summary": []}
    manifest = build_manifest([art])
    check("manifest carries the schema and all statuses",
          manifest["schema"] == SCHEMA
          and set(manifest["summary"]) >= set(STATUSES))
    check("a valid result marks its attack measured",
          "wire-shape" in manifest["attacks_measured"])
    check("an error-only result does not count as measured",
          "gpu-memory" not in manifest["attacks_measured"])
    s7 = manifest["threats"]["S7"]
    check("surface threat reflects measured/mapped attacks",
          s7["status"] in ("COVERED", "PARTIAL")
          and "wire-shape" in s7["measured_attacks"])
    check("agentic risks default to OUT_OF_SCOPE with a reason",
          manifest["threats"]["ASI01"]["status"] == "OUT_OF_SCOPE"
          and manifest["threats"]["ASI01"].get("out_of_scope_reason"))
    m2 = build_manifest([art], applicable_agentic=True)
    check("agentic risks become applicable on demand (never conflated)",
          m2["threats"]["ASI01"]["status"] == "UNMEASURED"
          and "out_of_scope_reason" not in m2["threats"]["ASI01"])
    empty = build_manifest([])
    check("no artifacts => every applicable threat UNMEASURED",
          empty["threats"]["LLM02"]["status"] == "UNMEASURED"
          and empty["summary"]["COVERED"] == 0)
    check("markdown renders the table and the summary",
          "| threat |" in render_manifest_md(manifest)
          and "Summary" in render_manifest_md(manifest))

    def _cell(arm, world):
        return {"experiment": "tool_misuse", "arm": arm, "world": world,
                "trials": 2, "total_violations": 0, "violation_rate": 0.0,
                "wilson_upper": 0.5}
    control_only = {"results": [_cell("authorized-task", "vulnerable"),
                               _cell("authorized-task", "defended")],
                    "summary": []}
    m3 = build_manifest([control_only], applicable_agentic=True)
    check("C5: control-only artifact leaves ASI02 UNMEASURED",
          m3["threats"]["ASI02"]["status"] == "UNMEASURED"
          and "tool-misuse" not in m3["attacks_measured"])
    check("C5: it leaves LLM03 and S14 UNMEASURED too (not just ASI02)",
          m3["threats"]["LLM03"]["status"] == "UNMEASURED"
          and m3["threats"]["S14"]["status"] == "UNMEASURED")

    full = {"results": [_cell(a, w)
                        for a in ("dangerous-parameters", "scope-confusion",
                                  "side-effect-amplification",
                                  "authorized-task")
                        for w in ("vulnerable", "defended")], "summary": []}
    m4 = build_manifest([full], applicable_agentic=True)
    check("C5: a complete tool-misuse artifact marks ASI02 COVERED",
          m4["threats"]["ASI02"]["status"] == "COVERED"
          and m4["threats"]["ASI02"]["mapped_attacks"] == ["tool-misuse"])
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
