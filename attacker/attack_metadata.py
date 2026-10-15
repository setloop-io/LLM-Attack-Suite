#!/usr/bin/env python3
"""Required attack metadata (THREAT_MODEL_REVIEW Release 1, item 2).

The attack contract (``attacker.attacks.validate_contract``) enforces
``MODES`` and ``REQUIRES_LABELS`` but says nothing machine-checkable about
the documented threat axes: actor intent (passive/active/malicious),
persistence (per-/cross-epoch ...), observation surface, security
property, lifecycle stage, and framework mappings (OWASP / NIST / ATLAS).
This registry makes those coordinates first-class, one entry per attack,
validated as a whole by ``validate_registry`` (CI-enforced in tests and in
the registry self-test).

The registry is deliberately a single declarative table — metadata is a
claim about the attack, reviewed with the same doctrine as the frozen
gate: changing an entry is an explicit, reviewable edit.

Surfaces extend the paper's S1-S9 model with the stores/control-plane
surfaces the review found unmodeled (S10-S14); see README.md.

Pure stdlib; torch-free (repo rule).
"""

from . import threat_model

SCHEMA = "llm_attacker.attack_metadata.v2"

SURFACES = {
    "S1": "seam activations, forward (h* crossing to the body)",
    "S2": "seam gradients, backward (dL/dh* returned by the body)",
    "S3": "body weights (held by the cloud)",
    "S4": "outputs / logits / answers",
    "S5": "GPU memory residue (freed buffers, allocator state)",
    "S6": "cache / timing (KV/prefix cache, contention, acceptance)",
    "S7": "traffic shape (packet sizes/timing on the link)",
    "S8": "API reasoning blobs (encrypted CoT envelopes)",
    "S9": "labels / properties",
    "S10": "protocol and control-plane metadata (negotiation, epochs, keys)",
    "S11": "persistent stores and telemetry (captures, logs, checkpoints, "
           "crash dumps, backups)",
    "S12": "MoE router and scheduling metadata",
    "S13": "AI artifacts and supply chain (models, tokenizers, adapters, "
           "datasets, plugins)",
    "S14": "application context (prompts, system instructions, RAG "
           "documents, tool schemas, memory)",
    "S15": "attestation evidence, verifier policy, endpoint binding and "
           "key-release decisions",
    "S16": "confidential-host traces (page/access events, counters, crash "
           "and restart observations)",
    "S17": "FHE ciphertext layout, context, errors, status, timing and "
           "verification feedback",
    "S18": "remote computation and result binding (weights, operators, "
           "gradients, request/step identity)",
}

SECURITY_PROPERTIES = ("confidentiality", "integrity", "availability",
                       "authenticity", "authorization", "privacy", "safety")

LIFECYCLE_STAGES = ("training", "fine-tuning", "inference", "deployment",
                    "operations", "supply-chain")

OWASP_LLM_2026 = tuple(f"LLM{i:02d}" for i in range(1, 11))

OWASP_AGENTIC_2026 = tuple(f"ASI{i:02d}" for i in range(1, 11))

_FIELDS = {"actor_intent": threat_model.INTENTS,
           "persistence": threat_model.PERSISTENCE,
           "surfaces": tuple(SURFACES),
           "security_properties": SECURITY_PROPERTIES,
           "lifecycle_stages": LIFECYCLE_STAGES,
           "owasp_llm_2026": OWASP_LLM_2026}

ATTACK_METADATA = {
    "accumulation": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    "known-prefix": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02", "LLM08")},
    "stale-key": {
        "actor_intent": "malicious", "persistence": "cross-epoch",
        "surfaces": ("S1", "S10"), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference", "operations"),
        "owasp_llm_2026": ("LLM02",)},
    "seed-disclosure": {
        "actor_intent": "malicious", "persistence": "cross-epoch",
        "surfaces": ("S10",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "deployment"),
        "owasp_llm_2026": ("LLM02", "LLM08")},
    "sharded": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    "wire-eval": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1", "S2"), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    "alignment-search": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    "subspace-joint": {
        "actor_intent": "passive", "persistence": "cross-epoch",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    "leak-accumulation": {
        "actor_intent": "malicious", "persistence": "cross-epoch",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference", "operations"),
        "owasp_llm_2026": ("LLM02",)},
    "max-effort": {
        "actor_intent": "passive", "persistence": "cross-session",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    "gradient-inversion": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S2",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "fine-tuning"),
        "owasp_llm_2026": ("LLM02",)},
    "membership": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1",), "security_properties": ("privacy",),
        "lifecycle_stages": ("training",),
        "owasp_llm_2026": ("LLM02",)},
    "lira-membership": {
        "actor_intent": "passive", "persistence": "cross-session",
        "surfaces": ("S1",), "security_properties": ("privacy",),
        "lifecycle_stages": ("training",),
        "owasp_llm_2026": ("LLM02",)},
    "label-scoring": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S2", "S9"), "security_properties": ("privacy",),
        "lifecycle_stages": ("training", "fine-tuning"),
        "owasp_llm_2026": ("LLM02",)},
    "batch-label-recon": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S2", "S9"), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training",),
        "owasp_llm_2026": ("LLM02",)},
    "malicious-server": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S2", "S9"),
        "security_properties": ("confidentiality", "integrity"),
        "lifecycle_stages": ("training",),
        "owasp_llm_2026": ("LLM02", "LLM05")},
    "active-cloud": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S1", "S2"),
        "security_properties": ("confidentiality", "integrity"),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02", "LLM05")},
    "output-inversion": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S4",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM02",)},
    "kv-timing": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S6",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM02",)},
    "reasoning-replay": {
        "actor_intent": "active", "persistence": "cross-session",
        "surfaces": ("S8",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM02", "LLM08")},
    "trace-inversion": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S8", "S4"), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM02", "LLM08")},
    "latent-probe": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training",),
        "owasp_llm_2026": ("LLM02", "LLM09")},
    "latent-sensitivity": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training",),
        "owasp_llm_2026": ("LLM02", "LLM09")},
    "latent-matching": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training",),
        "owasp_llm_2026": ("LLM02", "LLM09")},
    "seam-inversion": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1", "S2"),
        "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    "vec2text-inversion": {
        "actor_intent": "passive", "persistence": "cross-session",
        "surfaces": ("S1", "S4"),
        "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    "body-extraction": {
        "actor_intent": "active", "persistence": "cross-session",
        "surfaces": ("S3",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM02",)},
    "gpu-memory": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S5",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference", "operations"),
        "owasp_llm_2026": ("LLM02",)},
    "wire-shape": {
        "actor_intent": "passive", "persistence": "cross-session",
        "surfaces": ("S7",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    "remote-fetch": {
        "actor_intent": "passive", "persistence": "single-request",
        "surfaces": ("S11",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("operations",),
        "owasp_llm_2026": ("LLM02",)},
    "mitm-proxy": {
        "actor_intent": "active", "persistence": "session",
        "surfaces": ("S7", "S10"),
        "security_properties": ("confidentiality", "integrity"),
        "lifecycle_stages": ("training", "inference", "deployment"),
        "owasp_llm_2026": ("LLM02",)},
    "ica-bss": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    "protocol-integrity": {
        "actor_intent": "active", "persistence": "session",
        "surfaces": ("S10", "S7"),
        "security_properties": ("integrity", "availability", "authenticity"),
        "lifecycle_stages": ("deployment", "operations"),
        "owasp_llm_2026": ("LLM02", "LLM05", "LLM06")},
    "multi-tenant-isolation": {
        "actor_intent": "active", "persistence": "session",
        "surfaces": ("S4", "S5", "S6"),
        "security_properties": ("confidentiality", "authorization"),
        "lifecycle_stages": ("inference", "operations"),
        "owasp_llm_2026": ("LLM02", "LLM06", "LLM08", "LLM09")},
    "artifact-telemetry-leak": {
        "actor_intent": "passive", "persistence": "cross-session",
        "surfaces": ("S11",),
        "security_properties": ("confidentiality", "privacy"),
        "lifecycle_stages": ("deployment", "operations"),
        "owasp_llm_2026": ("LLM02", "LLM08")},
    "key-session-abuse": {
        "actor_intent": "malicious", "persistence": "cross-epoch",
        "surfaces": ("S10",),
        "security_properties": ("confidentiality", "authenticity"),
        "lifecycle_stages": ("deployment", "operations"),
        "owasp_llm_2026": ("LLM02", "LLM04", "LLM08")},
    "secure-compute-audit": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S7", "S10", "S11", "S15", "S16", "S17", "S18"),
        "security_properties": ("confidentiality", "integrity",
                                "authenticity", "authorization"),
        "lifecycle_stages": ("training", "inference", "deployment", "operations"),
        "owasp_llm_2026": ("LLM02", "LLM05", "LLM06", "LLM08")},
    "resource-exhaustion": {
        "actor_intent": "active", "persistence": "session",
        "surfaces": ("S6", "S10"),
        "security_properties": ("availability",),
        "lifecycle_stages": ("inference", "operations"),
        "owasp_llm_2026": ("LLM06",)},
    "poisoning-backdoor": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S13", "S2"),
        "security_properties": ("integrity", "safety"),
        "lifecycle_stages": ("training", "fine-tuning"),
        "owasp_llm_2026": ("LLM04", "LLM05")},
    "malicious-client": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S1", "S2", "S3"),
        "security_properties": ("confidentiality", "integrity",
                                "availability"),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02", "LLM05", "LLM06")},
    "supply-chain": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S13",),
        "security_properties": ("integrity", "authenticity"),
        "lifecycle_stages": ("supply-chain", "deployment"),
        "owasp_llm_2026": ("LLM04", "LLM05")},
    "router-side-channel": {
        "actor_intent": "passive", "persistence": "cross-session",
        "surfaces": ("S12",),
        "security_properties": ("confidentiality", "privacy"),
        "lifecycle_stages": ("inference", "operations"),
        "owasp_llm_2026": ("LLM02", "LLM09")},
    "privacy-composition": {
        "actor_intent": "passive", "persistence": "cross-epoch",
        "surfaces": ("S1", "S2", "S6", "S7"),
        "security_properties": ("privacy", "confidentiality"),
        "lifecycle_stages": ("training", "inference", "operations"),
        "owasp_llm_2026": ("LLM02",)},
    "prompt-injection": {
        "actor_intent": "active", "persistence": "cross-session",
        "surfaces": ("S14",),
        "security_properties": ("integrity", "authorization",
                                "confidentiality"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM01", "LLM02", "LLM03", "LLM08")},
    "hidden-context-exposure": {
        "actor_intent": "active", "persistence": "cross-session",
        "surfaces": ("S14", "S4"),
        "security_properties": ("confidentiality",),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM08", "LLM02")},
    "rag-vector-abuse": {
        "actor_intent": "active", "persistence": "cross-session",
        "surfaces": ("S14", "S9"),
        "security_properties": ("confidentiality", "integrity",
                                "authorization"),
        "lifecycle_stages": ("inference", "operations"),
        "owasp_llm_2026": ("LLM01", "LLM02", "LLM05", "LLM09")},
    "output-sink-injection": {
        "actor_intent": "active", "persistence": "session",
        "surfaces": ("S14", "S4"),
        "security_properties": ("integrity", "safety", "authorization"),
        "lifecycle_stages": ("inference", "deployment"),
        "owasp_llm_2026": ("LLM10",)},
    "misinformation-integrity": {
        "actor_intent": "passive", "persistence": "single-request",
        "surfaces": ("S4", "S14"),
        "security_properties": ("safety", "integrity"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM07",)},
    "tool-misuse": {
        "actor_intent": "malicious", "persistence": "session",
        "surfaces": ("S14",),
        "security_properties": ("authorization", "integrity", "safety"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM03",),
        "owasp_agentic_2026": ("ASI02",)},
    "agent-goal-hijack": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S14",),
        "security_properties": ("integrity", "authorization", "safety"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM01",),
        "owasp_agentic_2026": ("ASI01",)},
    "human-agent-trust": {
        "actor_intent": "malicious", "persistence": "session",
        "surfaces": ("S14",),
        "security_properties": ("integrity", "authorization", "safety"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM07",),
        "owasp_agentic_2026": ("ASI09",),
        "cwe": ("CWE-451",)},
    "memory-context-poisoning": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S14",),
        "security_properties": ("integrity", "authorization", "safety"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM01",),
        "owasp_agentic_2026": ("ASI06",),
        "cwe": ("CWE-501",)},
    "agentic-supply-chain": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S14",),
        "security_properties": ("integrity", "authorization", "safety"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM01",),
        "owasp_agentic_2026": ("ASI04",),
        "cwe": ("CWE-1357",)},
    "identity-privilege-abuse": {
        "actor_intent": "malicious", "persistence": "session",
        "surfaces": ("S14",),
        "security_properties": ("authorization", "integrity", "safety"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM03",),
        "owasp_agentic_2026": ("ASI03",),
        "cwe": ("CWE-863",)},
    "cascading-failure": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S14",),
        "security_properties": ("integrity", "authorization", "safety"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM01",),
        "owasp_agentic_2026": ("ASI08",),
        "cwe": ("CWE-345",)},
    "unexpected-code-execution": {
        "actor_intent": "malicious", "persistence": "session",
        "surfaces": ("S14",),
        "security_properties": ("authorization", "integrity", "safety"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM03",),
        "owasp_agentic_2026": ("ASI05",),
        "cwe": ("CWE-94",)},
    "rogue-agent": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S14",),
        "security_properties": ("authorization", "integrity", "safety"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM03",),
        "owasp_agentic_2026": ("ASI10",),
        "cwe": ("CWE-693",)},
    "insecure-agent-communication": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S14",),
        "security_properties": ("authenticity", "authorization", "integrity"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM03",),
        "owasp_agentic_2026": ("ASI07",),
        "cwe": ("CWE-294",)},
    "cross-agent-deputy": {
        "actor_intent": "malicious", "persistence": "session",
        "surfaces": ("S14",),
        "security_properties": ("authorization", "integrity", "safety"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM03",),
        "cwe": ("CWE-441",)},
    "replica-integrity": {
        "actor_intent": "malicious", "persistence": "session",
        "surfaces": ("S1", "S2", "S10"),
        "security_properties": ("integrity", "availability", "confidentiality"),
        "lifecycle_stages": ("training", "inference"), "owasp_llm_2026": ("LLM02", "LLM05")},
    "view-privacy": {
        "actor_intent": "active", "persistence": "cross-epoch",
        "surfaces": ("S1", "S4", "S10"),
        "security_properties": ("confidentiality", "privacy"),
        "lifecycle_stages": ("training", "inference"), "owasp_llm_2026": ("LLM02",)},
    "representation-privacy": {
        "actor_intent": "passive", "persistence": "cross-session",
        "surfaces": ("S1", "S9"), "security_properties": ("confidentiality", "privacy"),
        "lifecycle_stages": ("training", "fine-tuning", "inference"), "owasp_llm_2026": ("LLM02",)},
    "gradient-hardening": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1", "S2"), "security_properties": ("confidentiality", "privacy"),
        "lifecycle_stages": ("training", "fine-tuning"), "owasp_llm_2026": ("LLM02",)},
    "two-seam-delegation": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1", "S2"),
        "security_properties": ("confidentiality", "privacy"),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    "bisr-baseline": {
        "actor_intent": "passive", "persistence": "cross-session",
        "surfaces": ("S1", "S2"),
        "security_properties": ("confidentiality", "privacy"),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
}

_EXTRA_MAPPINGS = ("mitre_atlas", "nist_ai_100_2", "cwe", "owasp_agentic_2026")

_CLOSED_EXTRA_MAPPINGS = {"owasp_agentic_2026": OWASP_AGENTIC_2026}


def metadata_for(name):
    """The metadata dict for attack `name`; KeyError naming the registry
    when the attack has none (a registry-completeness bug — CI-enforced)."""
    try:
        return ATTACK_METADATA[name]
    except KeyError:
        raise KeyError(f"attack {name!r} has no entry in "
                       f"attacker.attack_metadata.ATTACK_METADATA; add one "
                       f"(THREAT_MODEL_REVIEW Release 1 item 2)") from None


def validate_metadata(name, md):
    """Raise a descriptive ValueError when metadata `md` for attack `name`
    violates the metadata schema; return `md` on success."""
    problems = []
    if not isinstance(md, dict):
        raise ValueError(f"attack {name!r}: metadata must be a dict")
    unknown = sorted(set(md) - (set(_FIELDS) | set(_EXTRA_MAPPINGS)))
    if unknown:
        problems.append(f"unknown fields {unknown}")
    for field, vocab in _FIELDS.items():
        if field not in md:
            problems.append(f"missing required field {field!r}")
            continue
        value = md[field]
        if field in ("actor_intent", "persistence"):
            if value not in vocab:
                problems.append(f"{field} must be one of {list(vocab)}, "
                                f"got {value!r}")
        else:
            if not (isinstance(value, (tuple, list)) and value
                    and all(isinstance(x, str) for x in value)):
                problems.append(f"{field} must be a non-empty list of str")
                continue
            bad = sorted(set(value) - set(vocab))
            if bad:
                problems.append(f"{field} values {bad} not in "
                                f"{sorted(vocab)}")
    for field in _EXTRA_MAPPINGS:
        if field in md and not (isinstance(md[field], (tuple, list))
                                and all(isinstance(x, str)
                                        for x in md[field])):
            problems.append(f"{field} must be a list of str")
    for field, vocab in _CLOSED_EXTRA_MAPPINGS.items():
        if field in md and isinstance(md[field], (tuple, list)):
            bad = sorted(set(md[field]) - set(vocab))
            if bad:
                problems.append(f"{field} values {bad} not in "
                                f"{sorted(vocab)}")
    if problems:
        raise ValueError(f"attack {name!r} metadata invalid: "
                         + "; ".join(problems))
    return md


def validate_registry(names=None):
    """Validate every entry (and, when `names` is given — e.g. the attack
    registry keys — that each name has metadata and no entry is orphaned).
    Returns the sorted list of covered attack names; raises ValueError with
    all problems listed."""
    problems = []
    for name, md in ATTACK_METADATA.items():
        try:
            validate_metadata(name, md)
        except ValueError as e:
            problems.append(str(e))
    if names is not None:
        missing = sorted(set(names) - set(ATTACK_METADATA))
        orphaned = sorted(set(ATTACK_METADATA) - set(names))
        if missing:
            problems.append(f"attacks without metadata: {missing}")
        if orphaned:
            problems.append(f"metadata entries for unknown attacks: "
                            f"{orphaned}")
    if problems:
        raise ValueError("attack metadata registry invalid: "
                         + "; ".join(problems))
    return sorted(ATTACK_METADATA)


def attacks_for_owasp(risk):
    """Attack names whose metadata maps to OWASP LLM 2026 `risk`
    (e.g. "LLM02"), sorted."""
    if risk not in OWASP_LLM_2026:
        raise ValueError(f"unknown OWASP LLM 2026 risk {risk!r} "
                         f"(known: {list(OWASP_LLM_2026)})")
    return sorted(n for n, md in ATTACK_METADATA.items()
                  if risk in md.get("owasp_llm_2026", ()))


def attacks_for_agentic(risk):
    """Attack names whose metadata maps to OWASP Agentic 2026 `risk`
    (e.g. "ASI02"), sorted. Mirrors attacks_for_owasp for the closed
    agentic vocabulary."""
    if risk not in OWASP_AGENTIC_2026:
        raise ValueError(f"unknown OWASP Agentic 2026 risk {risk!r} "
                         f"(known: {list(OWASP_AGENTIC_2026)})")
    return sorted(n for n, md in ATTACK_METADATA.items()
                  if risk in md.get("owasp_agentic_2026", ()))


def attacks_for_surface(surface):
    """Attack names observing `surface` (e.g. "S1"), sorted."""
    if surface not in SURFACES:
        raise ValueError(f"unknown surface {surface!r} "
                         f"(known: {sorted(SURFACES)})")
    return sorted(n for n, md in ATTACK_METADATA.items()
                  if surface in md.get("surfaces", ()))


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    def _closed_md(**kw):
        base = {"actor_intent": "passive", "persistence": "session",
                "surfaces": ("S1",), "security_properties": ("confidentiality",),
                "lifecycle_stages": ("training",), "owasp_llm_2026": ("LLM02",)}
        base.update(kw)
        return base

    print("attack metadata registry (pure python):")
    try:
        covered = validate_registry()
        check("every registry entry is valid", bool(covered))
    except ValueError as e:
        check("every registry entry is valid", False)
        print(f"      {e}")
    try:
        validate_metadata("fake", {"actor_intent": "sneaky"})
        check("bad intent is a descriptive error", False)
    except ValueError as e:
        check("bad intent is a descriptive error",
              "fake" in str(e) and "actor_intent" in str(e))
    try:
        validate_metadata("fake", {
            "actor_intent": "passive", "persistence": "session",
            "surfaces": ("S99",), "security_properties": ("confidentiality",),
            "lifecycle_stages": ("training",), "owasp_llm_2026": ("LLM02",)})
        check("unknown surface rejected", False)
    except ValueError as e:
        check("unknown surface rejected", "S99" in str(e))
    try:
        validate_registry(["accumulation", "no-such-attack"])
        check("metadata gap vs registry is a descriptive error", False)
    except ValueError as e:
        check("metadata gap vs registry is a descriptive error",
              "no-such-attack" in str(e))
    check("attacks_for_owasp inverts the mapping",
          "accumulation" in attacks_for_owasp("LLM02"))
    check("attacks_for_surface inverts the mapping",
          "wire-shape" in attacks_for_surface("S7"))
    check("schema is v2 (closed agentic mapping added)",
          SCHEMA == "llm_attacker.attack_metadata.v2")
    try:
        validate_metadata("fake", _closed_md(owasp_agentic_2026=("ASI99",)))
        check("out-of-vocab agentic id rejected", False)
    except ValueError as e:
        check("out-of-vocab agentic id rejected", "ASI99" in str(e))
    check("valid agentic id accepted",
          validate_metadata("fake", _closed_md(
              owasp_agentic_2026=("ASI02",))) is not None)
    check("attacks_for_agentic inverts the closed mapping",
          "tool-misuse" in attacks_for_agentic("ASI02"))
    try:
        attacks_for_agentic("ASI99")
        check("attacks_for_agentic rejects an unknown risk", False)
    except ValueError as e:
        check("attacks_for_agentic rejects an unknown risk", "ASI99" in str(e))
    try:
        metadata_for("no-such-attack")
        check("metadata_for unknown attack raises naming the registry", False)
    except KeyError as e:
        check("metadata_for unknown attack raises naming the registry",
              "attack_metadata" in str(e))
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
