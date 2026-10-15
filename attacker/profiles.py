#!/usr/bin/env python3
"""Benchmark profiles (THREAT_MODEL_REVIEW §"Benchmark and reporting
changes").

Adding attacks without changing coverage reporting lets incomplete
assessments look comprehensive.  The benchmark is therefore divided into
explicit profiles; each profile reports every applicable threat as
COVERED / PARTIAL / UNMEASURED / OUT_OF_SCOPE (attacker/coverage.py
semantics — UNMEASURED and OUT_OF_SCOPE are never conflated), and the
overall result shows the number of unmeasured applicable risks alongside
the measured coverage.

Profiles (versioned; changing the threat or attack set is a new version,
never a silent edit):

  benchmark.split-privacy.v2
      the S1-S9 privacy families + multi-tenant isolation + artifact and
      telemetry leakage + router side channel + privacy composition as
      the final composite cell.
  benchmark.split-integrity-availability.v1
      protocol integrity, malicious client and server, poisoning and
      backdoor, resource exhaustion, key/session abuse.
  benchmark.genai-app.v1
      the OWASP LLM01-LLM10 manifest: prompt injection, RAG/vector abuse,
      hidden context, output sinks, misinformation, application data
      exposure, and tenant-isolation prerequisites.
  benchmark.agentic.v1
      OWASP ASI01-ASI10.  Created only when tool, memory, identity, and
      action surfaces exist; with no agentic target surface every risk is
      OUT_OF_SCOPE with the documented reason (never UNMEASURED).
  benchmark.agentic.v2 / v3 / v4 / v5 / v6 / v7 / v8 / v9 / v10 / v11 / v12
      v2 adds tool-misuse (ASI02) as a required attack; v3 additionally
      requires agent-goal-hijack (ASI01); v4 additionally requires
      human-agent-trust (ASI09); v5 additionally requires
      memory-context-poisoning (ASI06); v6 additionally requires
      agentic-supply-chain (ASI04); v7 additionally requires
      cascading-failure (ASI08); v8 additionally requires
      identity-privilege-abuse (ASI03); v9 additionally requires
      unexpected-code-execution (ASI05) on top of the union of v7+v8;
      v10 additionally requires rogue-agent (ASI10) on top of v9;
      v11 additionally requires insecure-agent-communication (ASI07) on
      top of v10 — with v11 the ASI01-ASI10 set is COMPLETE (all ten
      agentic modules required); v12 additionally requires
      cross-agent-deputy (the Confused-Deputy-style cross-agent scenario,
      LLM03 + S14) on top of v11 — the ASI threat set is unchanged.
      Each is a NEW version, never an edit of a prior one (the versioning
      rule): each threat's status still derives from the global manifest.

CLI:  python -m attacker --profiles ART1.json [ART2.json ...]
            [--profile NAME] [--output P.json] [--applicable-agentic]

Pure stdlib; torch-free (repo rule).
"""

import argparse
import json

from . import coverage

SCHEMA = "llm_attacker.profile_report.v1"

PROFILES = {
    "benchmark.split-privacy.v2": {
        "description": "S1-S9 privacy families + tenant isolation + "
                       "artifact/telemetry leakage + router side channel "
                       "+ privacy composition (final composite cell)",
        "threats": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
                    "S11", "S12", "LLM02", "LLM09"],
        "required_attacks": [
            "seam-inversion", "label-scoring", "gradient-inversion",
            "gpu-memory", "wire-shape", "membership", "lira-membership",
            "multi-tenant-isolation", "artifact-telemetry-leak",
            "router-side-channel",
        ],
        "final_composite_cell": "privacy-composition",
    },
    "benchmark.split-integrity-availability.v1": {
        "description": "protocol integrity, malicious client and server, "
                       "poisoning/backdoor, resource exhaustion, "
                       "key/session abuse",
        "threats": ["S10", "S13", "LLM04", "LLM05", "LLM06"],
        "required_attacks": [
            "protocol-integrity", "malicious-client", "malicious-server",
            "poisoning-backdoor", "resource-exhaustion",
            "key-session-abuse", "supply-chain",
        ],
    },
    "benchmark.genai-app.v1": {
        "description": "OWASP LLM01-LLM10 manifest: prompt injection, "
                       "RAG/vector abuse, hidden context, output sinks, "
                       "misinformation, application data exposure, "
                       "tenant-isolation prerequisites",
        "threats": [f"LLM{i:02d}" for i in range(1, 11)] + ["S14"],
        "required_attacks": [
            "prompt-injection", "rag-vector-abuse",
            "hidden-context-exposure", "output-sink-injection",
            "misinformation-integrity", "multi-tenant-isolation",
            "artifact-telemetry-leak",
        ],
    },
    "benchmark.agentic.v1": {
        "description": "OWASP ASI01-ASI10; applicable only when tool, "
                       "memory, identity, and action surfaces exist "
                       "(a safe sandbox and deterministic side-effect "
                       "policy are required)",
        "threats": [f"ASI{i:02d}" for i in range(1, 11)],
        "required_attacks": [],
        "agentic": True,
    },
    "benchmark.agentic.v2": {
        "description": "OWASP ASI01-ASI10 with the first agentic module "
                       "(tool-misuse / ASI02) as a required attack; "
                       "applicable only for agentic targets "
                       "(pass --applicable-agentic). Additive over v1: it "
                       "changes only the required_attacks MEASURED list — "
                       "ASI02 status derives from the global manifest, not "
                       "from this profile.",
        "threats": [f"ASI{i:02d}" for i in range(1, 11)],
        "required_attacks": ["tool-misuse"],
        "agentic": True,
    },
    "benchmark.agentic.v3": {
        "description": "OWASP ASI01-ASI10 with the first TWO agentic modules "
                       "(tool-misuse / ASI02 AND agent-goal-hijack / ASI01) as "
                       "required attacks; applicable only for agentic targets "
                       "(pass --applicable-agentic). A NEW version (never a "
                       "mutation of v2, which means 'tool-misuse required'): "
                       "adding agent-goal-hijack is a versioned change. Each "
                       "attack's threat status still derives from the global "
                       "manifest, not from this profile.",
        "threats": [f"ASI{i:02d}" for i in range(1, 11)],
        "required_attacks": ["tool-misuse", "agent-goal-hijack"],
        "agentic": True,
    },
    "benchmark.agentic.v4": {
        "description": "OWASP ASI01-ASI10 with the first THREE agentic modules "
                       "(tool-misuse / ASI02, agent-goal-hijack / ASI01 AND "
                       "human-agent-trust / ASI09) as required attacks; "
                       "applicable only for agentic targets (pass "
                       "--applicable-agentic). A NEW version (never a mutation "
                       "of v3, which means 'tool-misuse + agent-goal-hijack "
                       "required'): adding human-agent-trust is a versioned "
                       "change. Each attack's threat status still derives from "
                       "the global manifest, not from this profile.",
        "threats": [f"ASI{i:02d}" for i in range(1, 11)],
        "required_attacks": ["tool-misuse", "agent-goal-hijack",
                             "human-agent-trust"],
        "agentic": True,
    },
    "benchmark.agentic.v5": {
        "description": "OWASP ASI01-ASI10 with the first FOUR agentic modules "
                       "(tool-misuse / ASI02, agent-goal-hijack / ASI01, "
                       "human-agent-trust / ASI09 AND memory-context-poisoning "
                       "/ ASI06) as required attacks; applicable only for "
                       "agentic targets (pass --applicable-agentic). A NEW "
                       "version (never a mutation of v4, which means "
                       "'tool-misuse + agent-goal-hijack + human-agent-trust "
                       "required'): adding memory-context-poisoning is a "
                       "versioned change. Each attack's threat status still "
                       "derives from the global manifest, not from this "
                       "profile.",
        "threats": [f"ASI{i:02d}" for i in range(1, 11)],
        "required_attacks": ["tool-misuse", "agent-goal-hijack",
                             "human-agent-trust", "memory-context-poisoning"],
        "agentic": True,
    },
    "benchmark.agentic.v6": {
        "description": "OWASP ASI01-ASI10 with the first FIVE agentic modules "
                       "(tool-misuse / ASI02, agent-goal-hijack / ASI01, "
                       "human-agent-trust / ASI09, memory-context-poisoning / "
                       "ASI06 AND agentic-supply-chain / ASI04) as required "
                       "attacks; applicable only for agentic targets (pass "
                       "--applicable-agentic). A NEW version (never a mutation "
                       "of v5, which means 'tool-misuse + agent-goal-hijack + "
                       "human-agent-trust + memory-context-poisoning "
                       "required'): adding agentic-supply-chain is a versioned "
                       "change. Each attack's threat status still derives from "
                       "the global manifest, not from this profile.",
        "threats": [f"ASI{i:02d}" for i in range(1, 11)],
        "required_attacks": ["tool-misuse", "agent-goal-hijack",
                             "human-agent-trust", "memory-context-poisoning",
                             "agentic-supply-chain"],
        "agentic": True,
    },
    "benchmark.agentic.v8": {
        "description": "OWASP ASI01-ASI10 with the first SIX agentic modules "
                       "(tool-misuse / ASI02, agent-goal-hijack / ASI01, "
                       "human-agent-trust / ASI09, memory-context-poisoning / "
                       "ASI06, agentic-supply-chain / ASI04 AND "
                       "identity-privilege-abuse / ASI03) as required attacks; "
                       "applicable only for agentic targets (pass "
                       "--applicable-agentic). A NEW version (never a mutation "
                       "of v6, which means 'the first five agentic modules "
                       "required'): adding identity-privilege-abuse is a "
                       "versioned change (v8 was reserved ahead of v7 landing "
                       "— versions are reserved, not landed in order). Each "
                       "attack's threat status still derives from the global "
                       "manifest, not from this profile.",
        "threats": [f"ASI{i:02d}" for i in range(1, 11)],
        "required_attacks": ["tool-misuse", "agent-goal-hijack",
                             "human-agent-trust", "memory-context-poisoning",
                             "agentic-supply-chain",
                             "identity-privilege-abuse"],
        "agentic": True,
    },
    "benchmark.agentic.v7": {
        "description": "OWASP ASI01-ASI10 with the first SIX agentic modules "
                       "(tool-misuse / ASI02, agent-goal-hijack / ASI01, "
                       "human-agent-trust / ASI09, memory-context-poisoning / "
                       "ASI06, agentic-supply-chain / ASI04 AND cascading-failure "
                       "/ ASI08) as required attacks; applicable only for agentic "
                       "targets (pass --applicable-agentic). A NEW version (never "
                       "a mutation of v6, which means 'tool-misuse + "
                       "agent-goal-hijack + human-agent-trust + "
                       "memory-context-poisoning + agentic-supply-chain "
                       "required'): adding cascading-failure is a versioned "
                       "change. Each attack's threat status still derives from the "
                       "global manifest, not from this profile.",
        "threats": [f"ASI{i:02d}" for i in range(1, 11)],
        "required_attacks": ["tool-misuse", "agent-goal-hijack",
                             "human-agent-trust", "memory-context-poisoning",
                             "agentic-supply-chain", "cascading-failure"],
        "agentic": True,
    },
    "benchmark.agentic.v9": {
        "description": "OWASP ASI01-ASI10 with the first EIGHT agentic modules "
                       "(tool-misuse / ASI02, agent-goal-hijack / ASI01, "
                       "human-agent-trust / ASI09, memory-context-poisoning / "
                       "ASI06, agentic-supply-chain / ASI04, cascading-failure "
                       "/ ASI08, identity-privilege-abuse / ASI03 AND "
                       "unexpected-code-execution / ASI05) as required attacks; "
                       "applicable only for agentic targets (pass "
                       "--applicable-agentic). A NEW version (never a mutation "
                       "of v7 or v8, which each mean 'their own six modules "
                       "required'): adding unexpected-code-execution on top of "
                       "the union of v7+v8 is a versioned change. Each attack's "
                       "threat status still derives from the global manifest, "
                       "not from this profile.",
        "threats": [f"ASI{i:02d}" for i in range(1, 11)],
        "required_attacks": ["tool-misuse", "agent-goal-hijack",
                             "human-agent-trust", "memory-context-poisoning",
                             "agentic-supply-chain", "cascading-failure",
                             "identity-privilege-abuse",
                             "unexpected-code-execution"],
        "agentic": True,
    },
    "benchmark.agentic.v10": {
        "description": "OWASP ASI01-ASI10 with the first NINE agentic modules "
                       "(tool-misuse / ASI02, agent-goal-hijack / ASI01, "
                       "human-agent-trust / ASI09, memory-context-poisoning / "
                       "ASI06, agentic-supply-chain / ASI04, cascading-failure "
                       "/ ASI08, identity-privilege-abuse / ASI03, "
                       "unexpected-code-execution / ASI05 AND rogue-agent / "
                       "ASI10) as required attacks; applicable only for "
                       "agentic targets (pass --applicable-agentic). A NEW "
                       "version (never a mutation of v9, which means 'the "
                       "first eight agentic modules required'): adding "
                       "rogue-agent is a versioned change. Each attack's "
                       "threat status still derives from the global manifest, "
                       "not from this profile.",
        "threats": [f"ASI{i:02d}" for i in range(1, 11)],
        "required_attacks": ["tool-misuse", "agent-goal-hijack",
                             "human-agent-trust", "memory-context-poisoning",
                             "agentic-supply-chain", "cascading-failure",
                             "identity-privilege-abuse",
                             "unexpected-code-execution", "rogue-agent"],
        "agentic": True,
    },
    "benchmark.agentic.v11": {
        "description": "OWASP ASI01-ASI10 with ALL TEN agentic modules "
                       "(tool-misuse / ASI02, agent-goal-hijack / ASI01, "
                       "human-agent-trust / ASI09, memory-context-poisoning "
                       "/ ASI06, agentic-supply-chain / ASI04, "
                       "cascading-failure / ASI08, identity-privilege-abuse "
                       "/ ASI03, unexpected-code-execution / ASI05, "
                       "rogue-agent / ASI10 AND insecure-agent-communication "
                       "/ ASI07) as required attacks — the ASI01-ASI10 set is "
                       "COMPLETE; applicable only for agentic targets (pass "
                       "--applicable-agentic). A NEW version (never a "
                       "mutation of v10, which means 'the first nine agentic "
                       "modules required'): adding insecure-agent-"
                       "communication is a versioned change. Each attack's "
                       "threat status still derives from the global "
                       "manifest, not from this profile.",
        "threats": [f"ASI{i:02d}" for i in range(1, 11)],
        "required_attacks": ["tool-misuse", "agent-goal-hijack",
                             "human-agent-trust", "memory-context-poisoning",
                             "agentic-supply-chain", "cascading-failure",
                             "identity-privilege-abuse",
                             "unexpected-code-execution", "rogue-agent",
                             "insecure-agent-communication"],
        "agentic": True,
    },
    "benchmark.agentic.v12": {
        "description": "OWASP ASI01-ASI10 with ALL TEN agentic modules "
                       "(the complete v11 set) PLUS cross-agent-deputy — "
                       "the Confused-Deputy-style cross-agent scenario (an "
                       "AUTHENTIC peer message on the inter-agent bus "
                       "forwards a request to exercise the deputy's HELD "
                       "refund authority, one the session's authenticated "
                       "principal lacks; principal_binding is the "
                       "load-bearing control) — as required attacks; "
                       "applicable only for agentic targets (pass "
                       "--applicable-agentic). A NEW version (never a "
                       "mutation of v11, which means 'the complete ten "
                       "modules required'): adding cross-agent-deputy is a "
                       "versioned change. The scenario maps to LLM03 + S14 "
                       "(deliberately no closed-vocab ASI id — see its "
                       "attack_metadata entry), so the ASI01-ASI10 threat "
                       "set is unchanged; each attack's threat status still "
                       "derives from the global manifest, not from this "
                       "profile.",
        "threats": [f"ASI{i:02d}" for i in range(1, 11)],
        "required_attacks": ["tool-misuse", "agent-goal-hijack",
                             "human-agent-trust", "memory-context-poisoning",
                             "agentic-supply-chain", "cascading-failure",
                             "identity-privilege-abuse",
                             "unexpected-code-execution", "rogue-agent",
                             "insecure-agent-communication",
                             "cross-agent-deputy"],
        "agentic": True,
    },
}


def evaluate_profile(name, artifacts, applicable_agentic=False):
    """Evaluate one profile against run artifacts -> the profile report.

    Per threat: the coverage status restricted to the profile's threat
    set.  Per required attack: MEASURED / UNMEASURED.  The overall grade
    lists unmeasured applicable risks explicitly (they are never counted
    as covered, and never silently dropped)."""
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r} "
                         f"(known: {sorted(PROFILES)})")
    spec = PROFILES[name]
    manifest = coverage.build_manifest(
        artifacts, applicable_agentic=applicable_agentic)
    threats = {t: manifest["threats"][t] for t in spec["threats"]
               if t in manifest["threats"]}
    measured = set(manifest["attacks_measured"])
    attacks = {a: ("MEASURED" if a in measured else "UNMEASURED")
               for a in spec["required_attacks"]}
    final_cell = spec.get("final_composite_cell")
    if final_cell:
        attacks[final_cell + " (final composite)"] = (
            "MEASURED" if final_cell in measured else "UNMEASURED")
    counts = {s: sum(1 for t in threats.values() if t["status"] == s)
              for s in coverage.STATUSES}
    unmeasured = sorted(t for t, v in threats.items()
                        if v["status"] == "UNMEASURED")
    return {"schema": SCHEMA, "profile": name,
            "description": spec["description"],
            "threats": threats, "attacks": attacks,
            "summary": {**counts,
                        "applicable": counts["COVERED"] + counts["PARTIAL"]
                        + counts["UNMEASURED"],
                        "unmeasured_applicable": unmeasured,
                        "attacks_measured": sum(
                            1 for v in attacks.values()
                            if v == "MEASURED"),
                        "attacks_total": len(attacks)}}


def evaluate_all(artifacts, applicable_agentic=False):
    return {name: evaluate_profile(name, artifacts,
                                   applicable_agentic=applicable_agentic)
            for name in PROFILES}


def render_profile_md(report):
    """Human-readable profile report (markdown)."""
    s = report["summary"]
    lines = [f"# {report['profile']}", "", report["description"], "",
             "| threat | status | measured attacks |", "|---|---|---|"]
    for tid, t in sorted(report["threats"].items()):
        lines.append(f"| {tid} | {t['status']} | "
                     f"{', '.join(t['measured_attacks']) or '—'} |")
    lines += ["", f"threats: COVERED {s['COVERED']}, PARTIAL "
              f"{s['PARTIAL']}, UNMEASURED {s['UNMEASURED']}, "
              f"OUT_OF_SCOPE {s['OUT_OF_SCOPE']}",
              f"attacks measured: {s['attacks_measured']}/"
              f"{s['attacks_total']}"]
    if s["unmeasured_applicable"]:
        lines.append(f"**unmeasured applicable risks:** "
                     + ", ".join(s["unmeasured_applicable"]))
    return "\n".join(lines) + "\n"


def build_parser():
    ap = argparse.ArgumentParser(
        prog="python -m attacker --profiles", description=__doc__)
    ap.add_argument("artifacts", nargs="*", help="attack artifact JSONs")
    ap.add_argument("--profile", default=None,
                    choices=sorted(PROFILES), help="one profile only")
    ap.add_argument("--output", default=None, help="profile report JSON")
    ap.add_argument("--md", default=None, help="markdown report path")
    ap.add_argument("--applicable-agentic", action="store_true",
                    help="agentic targets: treat ASI01-ASI10 as applicable")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    arts = coverage.load_artifacts(args.artifacts)
    if args.profile:
        reports = {args.profile: evaluate_profile(
            args.profile, arts, applicable_agentic=args.applicable_agentic)}
    else:
        reports = evaluate_all(arts,
                               applicable_agentic=args.applicable_agentic)
    md = "\n".join(render_profile_md(r) for r in reports.values())
    print(md)
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"schema": SCHEMA, "profiles": reports}, f, indent=2)
        print(f"[profiles] wrote {args.output}")
    if args.md:
        with open(args.md, "w") as f:
            f.write(md)
        print(f"[profiles] wrote {args.md}")
    return 0


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("benchmark profiles (pure python):")
    check("fifteen versioned profiles exist (agentic.v12 added, additive)",
          set(PROFILES) == {"benchmark.split-privacy.v2",
                            "benchmark.split-integrity-availability.v1",
                            "benchmark.genai-app.v1",
                            "benchmark.agentic.v1",
                            "benchmark.agentic.v2",
                            "benchmark.agentic.v3",
                            "benchmark.agentic.v4",
                            "benchmark.agentic.v5",
                            "benchmark.agentic.v6",
                            "benchmark.agentic.v7",
                            "benchmark.agentic.v8",
                            "benchmark.agentic.v9",
                            "benchmark.agentic.v10",
                            "benchmark.agentic.v11",
                            "benchmark.agentic.v12"})
    check("profile threat sets reference the coverage universe",
          all(t in coverage.threat_universe(applicable_agentic=True)
              for spec in PROFILES.values() for t in spec["threats"]))
    art = {"results": [{"experiment": "wire_shape_analysis",
                        "arm": "length-leak", "mae_tokens_mean": 1.0,
                        "rank_corr_mean": 0.9},
                       {"experiment": "wire_shape_analysis",
                        "arm": "phase-detect", "f1_mean": 0.8,
                        "label_accuracy_mean": 0.7}],
           "summary": []}
    rep = evaluate_profile("benchmark.split-privacy.v2", [art])
    check("a measured attack shows up in its profile",
          rep["attacks"]["wire-shape"] == "MEASURED"
          and rep["attacks"]["seam-inversion"] == "UNMEASURED")
    check("unmeasured applicable risks are listed, never dropped",
          "S1" in rep["summary"]["unmeasured_applicable"])
    ag = evaluate_profile("benchmark.agentic.v1", [])
    check("agentic profile is OUT_OF_SCOPE by default (not UNMEASURED)",
          all(t["status"] == "OUT_OF_SCOPE"
              for t in ag["threats"].values())
          and ag["summary"]["unmeasured_applicable"] == [])
    ag2 = evaluate_profile("benchmark.agentic.v1", [],
                           applicable_agentic=True)
    check("applicable agentic risks become UNMEASURED on demand",
          all(t["status"] == "UNMEASURED"
              for t in ag2["threats"].values()))
    try:
        evaluate_profile("benchmark.nope.v9", [])
        check("unknown profile is a descriptive error", False)
    except ValueError as e:
        check("unknown profile is a descriptive error",
              "nope" in str(e))
    md = render_profile_md(rep)
    check("markdown renders the profile table", "| threat |" in md)
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
