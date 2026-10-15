#!/usr/bin/env python3
"""Structured threat-model schema (llm_attacker.threat_model.v2).

The historical artifact field ``threat_model`` is a free-form string.  That
makes coverage unauditable: two artifacts cannot be compared, and the
benchmark cannot tell "we thought about it" apart from "we measured it".
This module defines a versioned, machine-checkable schema (THREAT_MODEL
REVIEW §"Threat-model schema recommendation") while keeping the legacy
string accepted — ``normalize`` upgrades it to a v2 document whose
``summary`` carries the original text.

A v2 document:

    {"schema": "llm_attacker.threat_model.v2",
     "summary": "short human-readable description",
     "scope":     {"use_case": ..., "deployment": ...,
                   "in_scope": [...], "out_of_scope": [...]},
     "actor":     {"role": ..., "vantage": [...], "intent": ...,
                   "authorization": ..., "privileges": [...],
                   "knowledge": [...], "capabilities": [...],
                   "resources": [...], "persistence": ...,
                   "collusion": [...]},
     "target":    {"assets": [...], "components": [...],
                   "data_stores": [...], "data_flows": [...],
                   "trust_boundaries": [...], "entry_points": [...],
                   "exit_points": [...]},
     "scenario":  {"preconditions": [...], "lifecycle_stage": [...],
                   "security_properties": [...], "attack_steps": [...],
                   "technical_impact": [...], "business_impact": [...]},
     "controls":  {"assumed": [...], "tested": [...], "bypassed": [...],
                   "detective": [...], "response": [...],
                   "residual_risk": "", "owner": "", "status": ...},
     "mappings":  {"owasp_llm_2026": [...], "owasp_agentic_2026": [...],
                   "mitre_atlas": [...], "nist_ai_100_2": [...],
                   "cwe": [...]},
     "evaluation": {"success_metrics": [...], "baseline": "",
                    "budget": {...}, "limitations": [...],
                    "coverage_status": ...}}

Validation is strict and descriptive (repo doctrine): unknown keys, wrong
types, and values outside the enumerated sets raise a ValueError naming
the exact path — never a silent skip.  Only ``schema`` and ``summary``
are required; every other field defaults to an empty list / empty string
so partial documents normalize to complete ones.

Pure stdlib; torch-free (repo rule).
"""

SCHEMA_V2 = "llm_attacker.threat_model.v2"
LEGACY_SCHEMAS = ("llm_attacker.threat_model.v1",)

USE_CASES = ("split-training", "split-inference", "genai-app", "agentic")
DEPLOYMENTS = ("single-tenant", "multi-tenant", "federated", "edge-cloud")
ACTOR_ROLES = ("cloud", "client", "co-tenant", "network", "insider",
               "API user")
INTENTS = ("passive", "active", "malicious")
PERSISTENCE = ("single-request", "session", "cross-session", "cross-epoch")
CONTROL_STATUSES = ("open", "mitigated", "accepted", "transferred")
COVERAGE_STATUSES = ("covered", "partial", "unmeasured", "out-of-scope")

SECTIONS = {
    "scope": {"use_case": ("enum", USE_CASES),
              "deployment": ("enum", DEPLOYMENTS),
              "in_scope": "strlist", "out_of_scope": "strlist"},
    "actor": {"role": ("enum", ACTOR_ROLES), "vantage": "strlist",
              "intent": ("enum", INTENTS), "authorization": "str",
              "privileges": "strlist", "knowledge": "strlist",
              "capabilities": "strlist", "resources": "strlist",
              "persistence": ("enum", PERSISTENCE), "collusion": "strlist"},
    "target": {"assets": "strlist", "components": "strlist",
               "data_stores": "strlist", "data_flows": "strlist",
               "trust_boundaries": "strlist", "entry_points": "strlist",
               "exit_points": "strlist"},
    "scenario": {"preconditions": "strlist", "lifecycle_stage": "strlist",
                 "security_properties": "strlist", "attack_steps": "strlist",
                 "technical_impact": "strlist", "business_impact": "strlist"},
    "controls": {"assumed": "strlist", "tested": "strlist",
                 "bypassed": "strlist", "detective": "strlist",
                 "response": "strlist", "residual_risk": "str",
                 "owner": "str",
                 "status": ("enum", CONTROL_STATUSES)},
    "mappings": {"owasp_llm_2026": "strlist", "owasp_agentic_2026": "strlist",
                 "mitre_atlas": "strlist", "nist_ai_100_2": "strlist",
                 "cwe": "strlist"},
    "evaluation": {"success_metrics": "strlist", "baseline": "str",
                   "budget": "dict", "limitations": "strlist",
                   "coverage_status": ("enum", COVERAGE_STATUSES)},
}

_TOP_KEYS = ("schema", "summary", *SECTIONS)


def _default_section(spec):
    out = {}
    for field, kind in spec.items():
        if isinstance(kind, tuple):
            out[field] = None
        elif kind == "strlist":
            out[field] = []
        elif kind == "dict":
            out[field] = {}
        else:
            out[field] = ""
    return out


def skeleton(summary="", **overrides):
    """A complete v2 document with defaults; ``overrides`` are deep-merged
    per section (``actor={"intent": "passive"}``)."""
    doc = {"schema": SCHEMA_V2, "summary": summary}
    for section, spec in SECTIONS.items():
        doc[section] = _default_section(spec)
    for key, value in overrides.items():
        if key not in SECTIONS or not isinstance(value, dict):
            raise ValueError(f"threat-model skeleton override {key!r} must be "
                             f"a section dict (sections: {sorted(SECTIONS)})")
        doc[key].update(value)
    return doc


def is_legacy(value):
    """True when `value` is a legacy free-form threat-model string."""
    return isinstance(value, str)


def check(value):
    """Return the list of schema violations (empty list == valid)."""
    problems = []
    if not isinstance(value, dict):
        problems.append(f"threat_model must be an object (schema "
                        f"{SCHEMA_V2}) or a legacy string, got "
                        f"{type(value).__name__}")
        return problems
    schema = value.get("schema")
    if schema != SCHEMA_V2:
        problems.append(f"schema must be {SCHEMA_V2!r}, got {schema!r}")
        return problems
    unknown = sorted(set(value) - set(_TOP_KEYS))
    if unknown:
        problems.append(f"unknown top-level keys {unknown} "
                        f"(allowed: {sorted(_TOP_KEYS)})")
    if not isinstance(value.get("summary"), str) or not value.get("summary"):
        problems.append("summary must be a non-empty str")
    for section, spec in SECTIONS.items():
        sec = value.get(section, {})
        if not isinstance(sec, dict):
            problems.append(f"{section} must be an object")
            continue
        extra = sorted(set(sec) - set(spec))
        if extra:
            problems.append(f"{section}: unknown fields {extra} "
                            f"(allowed: {sorted(spec)})")
        for field, kind in spec.items():
            if field not in sec:
                continue
            v = sec[field]
            path = f"{section}.{field}"
            if isinstance(kind, tuple):
                if v is not None and v not in kind[1]:
                    problems.append(f"{path} must be one of {list(kind[1])}, "
                                    f"got {v!r}")
            elif kind == "str":
                if not isinstance(v, str):
                    problems.append(f"{path} must be a str, got "
                                    f"{type(v).__name__}")
            elif kind == "dict":
                if not isinstance(v, dict):
                    problems.append(f"{path} must be an object, got "
                                    f"{type(v).__name__}")
            else:
                if not (isinstance(v, list)
                        and all(isinstance(x, str) for x in v)):
                    problems.append(f"{path} must be a list of str")
    return problems


def validate(value):
    """Validate a v2 document; raise a descriptive ValueError on any
    violation; return the document unchanged on success."""
    problems = check(value)
    if problems:
        raise ValueError(f"invalid threat model ({SCHEMA_V2}): "
                         + "; ".join(problems))
    return value


def normalize(value):
    """Normalize any accepted threat-model value to a complete v2 document.

    A legacy string becomes ``skeleton(summary=<the string>)`` (transition
    period, THREAT_MODEL_REVIEW §schema); a v2 dict is validated and its
    missing fields filled with defaults.  Anything else is a descriptive
    ValueError.
    """
    if isinstance(value, str):
        return skeleton(summary=value)
    problems = check(value)
    if problems:
        raise ValueError(f"invalid threat model ({SCHEMA_V2}): "
                         + "; ".join(problems))
    doc = skeleton(summary=value["summary"])
    for key in _TOP_KEYS:
        if key in ("schema", "summary"):
            continue
        if key in value:
            doc[key].update(value[key])
    return doc


def self_test():
    ok = True

    def check_(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("structured threat-model schema (pure python):")
    doc = skeleton("self-test threat model",
                   actor={"intent": "passive", "role": "cloud"})
    check_("skeleton is complete and valid", validate(doc) is doc)
    check_("all sections present with defaults",
           sorted(doc) == sorted(_TOP_KEYS)
           and doc["controls"]["status"] is None
           and doc["mappings"]["owasp_llm_2026"] == [])

    legacy = normalize("passive cloud observing rotated seam tensors")
    check_("legacy string normalizes to summary",
           legacy["schema"] == SCHEMA_V2
           and legacy["summary"]
           == "passive cloud observing rotated seam tensors")
    check_("normalized legacy validates", validate(legacy) is legacy)

    partial = {"schema": SCHEMA_V2, "summary": "x",
               "actor": {"intent": "malicious"}}
    full = normalize(partial)
    check_("partial v2 keeps supplied fields and fills defaults",
           full["actor"]["intent"] == "malicious"
           and full["target"]["assets"] == [])

    try:
        validate({"schema": SCHEMA_V2, "summary": "x",
                  "actor": {"intent": "sneaky"}})
        check_("bad enum value is a descriptive error", False)
    except ValueError as e:
        check_("bad enum value is a descriptive error",
               "actor.intent" in str(e) and "sneaky" in str(e))
    try:
        validate({"schema": SCHEMA_V2, "summary": "x", "bogus": {}})
        check_("unknown top-level key rejected", False)
    except ValueError as e:
        check_("unknown top-level key rejected", "bogus" in str(e))
    try:
        normalize({"schema": "wrong.v0", "summary": "x"})
        check_("wrong schema rejected", False)
    except ValueError as e:
        check_("wrong schema rejected", SCHEMA_V2 in str(e))
    try:
        normalize(42)
        check_("non-dict non-str rejected", False)
    except ValueError:
        check_("non-dict non-str rejected", True)
    try:
        validate({"schema": SCHEMA_V2})
        check_("missing summary rejected", False)
    except ValueError as e:
        check_("missing summary rejected", "summary" in str(e))
    check_("is_legacy distinguishes str from v2 dict",
           is_legacy("text") and not is_legacy(doc))
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
