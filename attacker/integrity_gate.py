#!/usr/bin/env python3
"""integrity-gate.v1 — the scoring gate for the split integrity/availability
battery (Lane A, calibration README.md §12).

The frozen confidentiality gate (attacker/gate.py) scores a recovery *excess*
in percentage points. An undetected-mutation *rate*, a DoS amplification
*ratio*, and a supply-chain verification-bypass *count* are not that shape.
This module is the versioned scoring config for those metrics — same doctrine
as the frozen gate: the constants below are the CLAIM. Changing any threshold
is a NEW gate version, never a silent edit; retarget by constructing an
`IntegrityGate`, never by mutating `DEFAULT_INTEGRITY_GATE`.

The locked decisions (calibration §12, D1-D7):

  D1  amplification (ratio) threshold 1.5x — retry-storm (1.400244x) is CLOSED.
  D4  every cell carries a role (baseline / defense / control / informational);
      a gradeable in-axis baseline must be OPEN and control CLOSED.
  D5  a `count` scale (threshold 0) so supply-chain false-rejection binds.
  D7  ratio contract `value <= threshold`; scored field `integrity_value`
      (NEVER a name containing `excess` — scorecard._excess substring-matches).

Verdict vocabulary: OPEN (attack lands past the threshold), CLOSED (held),
UNMEASURED (fail-closed: absent/insufficient n_seeds, a non-finite or
out-of-domain value, no matching rule, an out-of-profile-axis cell, or an
inherently non-gradeable cell). Fail-closed is the whole point: `NaN > thr`
is False, so a naive compare grades NaN CLOSED — the silent false negative
this gate exists to prevent.

Pure python (math/hashlib/json), torch-free.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass

GATE_VERSION = "integrity-gate.v1"

SCALES = ("rate", "ratio", "count", "pp")
ROLES = ("baseline", "defense", "control", "informational")


@dataclass(frozen=True)
class MetricRule:
    """One normalized cell's scoring rule. Exactly one rule matches a cell,
    keyed on (experiment, arm, world). Its single scored result rolls into
    EVERY axis in `axes` (a tuple; a cell may span 1-3 axes). `control_scope`
    records where the reported control lives (same_row / sibling_world /
    sibling_arm / const / frozen_gate) — the grade itself is ABSOLUTE
    (value vs threshold), never a subtraction (calibration §3). A `threshold`
    of None marks an inherently non-gradeable cell (malicious-server, routed
    to the frozen gate / a pp-scale aggregate) → always UNMEASURED."""
    rule_id: str
    experiment: str
    arm: object
    world: object
    axes: tuple
    signal_key: str
    control_scope: str
    control_key: object
    scale: str
    direction: str
    threshold: object
    role: str


@dataclass(frozen=True)
class IntegrityGate:
    """One named scoring gate: the version, the seed-count floor, and the
    frozen rule table. Defaults are integrity-gate.v1."""
    version: str = GATE_VERSION
    n_seeds_required: int = 3
    rules: tuple = ()


def _rule(rid, exp, arm, world, axes, signal, scope, ckey, scale, thr, role):
    return MetricRule(rid, exp, arm, world, tuple(axes), signal, scope, ckey,
                      scale, "higher_worse", thr, role)


def _protocol_rules():
    exp = "protocol_integrity"
    arms = ("cross-session-swap", "downgrade", "drop", "duplicate",
            "epoch-desync", "oversize", "reorder", "replay", "tensor-tamper",
            "truncate")
    out = []
    for arm in arms:
        axes = ("S10", "LLM06") if arm == "oversize" else ("S10",)
        out.append(_rule(f"pi:{arm}:defended", exp, arm, "defended", axes,
                         "undetected_mutation_rate_mean", "const", None,
                         "rate", 0.05, "defense"))
        out.append(_rule(f"pi:{arm}:vulnerable", exp, arm, "vulnerable", axes,
                         "undetected_mutation_rate_mean", "const", None,
                         "rate", 0.05, "baseline"))
    out.append(_rule("pi:negative-control:defended", exp, "negative-control",
                     "defended", ("S10",), "false_rejection_rate_mean",
                     "const", None, "rate", 0.05, "control"))
    return out


def _malicious_client_rules():
    exp = "malicious_client"
    axis_of = {"gradient-poisoning": ("LLM05",), "instability-bomb": ("LLM06",)}
    arms = ("budget-extraction", "collusion", "crafted-extraction",
            "gradient-poisoning", "instability-bomb", "neuron-observability")
    role_of = {"defended": "defense", "isolated": "control",
               "vulnerable": "baseline"}
    out = []
    for arm in arms:
        axes = axis_of.get(arm, ())
        for world in ("defended", "isolated", "vulnerable"):
            out.append(_rule(f"mc:{arm}:{world}", exp, arm, world, axes,
                             "attack_success_rate", "sibling_world",
                             "isolated", "rate", 0.05, role_of[world]))
    return out


def _key_session_rules():
    exp = "key_session_abuse"
    two_world = ("collusion", "epoch-rollback", "forward-secrecy",
                 "master-seed-exposure", "negotiation-downgrade",
                 "nonce-reuse", "session-binding")
    out = []
    for arm in two_world:
        out.append(_rule(f"ks:{arm}:defended", exp, arm, "defended", ("S10",),
                         "attack_success_rate", "sibling_world", "defended",
                         "rate", 0.05, "defense"))
        out.append(_rule(f"ks:{arm}:vulnerable", exp, arm, "vulnerable",
                         ("S10",), "attack_success_rate", "sibling_world",
                         "defended", "rate", 0.05, "baseline"))
    for world, role in (("control", "control"), ("defended", "defense"),
                        ("vulnerable", "baseline")):
        out.append(_rule(f"ks:key-in-logs:{world}", exp, "key-in-logs", world,
                         ("S10",), "attack_success_rate", "sibling_world",
                         "defended", "rate", 0.05, role))
    return out


def _supply_chain_rules():
    exp = "supply_chain"
    arms = ("artifact-substitution", "digest-mismatch", "unsigned-promotion",
            "rollback", "tokenizer-drift", "adapter-confusion", "plugin-origin",
            "unsafe-serialization", "dependency-drift")
    out = []
    for arm in arms:
        out.append(_rule(f"sc:{arm}:permissive", exp, arm, "permissive",
                         ("S13", "LLM04"), "bypass_rate_permissive",
                         "same_row", "bypass_rate_permissive", "rate", 0.05,
                         "baseline"))
        out.append(_rule(f"sc:{arm}:enforcing", exp, arm, "enforcing",
                         ("S13", "LLM04"), "bypass_rate_enforcing", "same_row",
                         "bypass_rate_permissive", "rate", 0.05, "defense"))
    out.append(_rule("sc:all:negative-control", exp, "all", "negative-control",
                     ("S13", "LLM04"), "negative_control_false_rejections",
                     "const", None, "count", 0.0, "control"))
    return out


def _resource_rules():
    exp = "resource_exhaustion"
    amp = ("batch-fragmentation", "cache-thrash", "decode-amplification",
           "long-context", "memory-pressure", "protocol-oversize",
           "retry-storm", "tokenizer-amplification")
    out = []
    for arm in amp:
        axes = ("LLM06", "S10") if arm == "protocol-oversize" else ("LLM06",)
        out.append(_rule(f"rex:{arm}", exp, arm, None, axes,
                         "amplification_vs_control_mean", "const", None,
                         "ratio", 1.5, "defense"))
    out.append(_rule("rex:benign-control", exp, "benign-control", None,
                     ("LLM06",), "amplification_parity_mean", "const", None,
                     "ratio", 1.5, "control"))
    out.append(_rule("rex:training-skew", exp, "training-skew", None,
                     ("LLM06",), "cost_amplification_mean", "const", None,
                     "ratio", 1.5, "defense"))
    return out


def _poisoning_rules():
    exp = "poisoning_backdoor"
    axis_of = {"adapter-backdoor": ("S13", "LLM04", "LLM05"),
               "checkpoint-backdoor": ("S13", "LLM04", "LLM05"),
               "rag-poison": ("LLM05", "S13")}
    arms = ("adapter-backdoor", "checkpoint-backdoor", "clean-label",
            "dirty-label", "gradient-sign", "model-replacement", "rag-poison",
            "trigger-backdoor")
    world_sig = {"undefended": ("asr_undefended_mean_max_budget", "baseline"),
                 "defended": ("asr_defended_mean", "defense"),
                 "random-corruption": ("random_corruption_asr_mean", "control")}
    out = []
    for arm in arms:
        axes = axis_of.get(arm, ("LLM05",))
        for world, (sig, role) in world_sig.items():
            out.append(_rule(f"pb:{arm}:{world}", exp, arm, world, axes, sig,
                             "sibling_world", "random-corruption", "rate", 0.1,
                             role))
    out.append(_rule("pb:zero-poison", exp, "zero-poison", "zero-poison",
                     ("LLM05",), "asr_mean", "const", None, "rate", 0.1,
                     "control"))
    return out


def _malicious_server_rules():
    exp = "malicious_server_steering"
    return [
        _rule("ms:honest", exp, "bwd-direction", "honest", ("LLM05",),
              "seq_accuracy_mean", "frozen_gate", "control_accuracy_mean",
              "rate", None, "control"),
        _rule("ms:crafted", exp, "bwd-direction", "crafted", ("LLM05",),
              "seq_accuracy_mean", "frozen_gate", "control_accuracy_mean",
              "rate", None, "baseline"),
        _rule("ms:aggregate", exp, "bwd-direction", None, ("LLM05",),
              "crafted_minus_honest_pp", "const", None, "pp", None,
              "informational"),
    ]


def _build_rules():
    out = (_protocol_rules() + _malicious_client_rules() + _key_session_rules()
           + _supply_chain_rules() + _resource_rules() + _poisoning_rules()
           + _malicious_server_rules())
    seen = {}
    for r in out:
        key = (r.experiment, r.arm, r.world)
        if key in seen:
            raise ValueError(f"duplicate rule coordinate {key}")
        seen[key] = r
    return tuple(out)


RULES = _build_rules()
DEFAULT_INTEGRITY_GATE = IntegrityGate(rules=RULES)


def match_rule(cell, gate):
    """The single rule for a cell, keyed on (experiment, arm, world). None on
    zero matches; ValueError on two or more (a rule-table defect)."""
    hits = [r for r in gate.rules
            if r.experiment == cell.get("experiment")
            and r.arm == cell.get("arm") and r.world == cell.get("world")]
    if len(hits) > 1:
        raise ValueError(
            f"integrity-gate: {len(hits)} rules match cell "
            f"{(cell.get('experiment'), cell.get('arm'), cell.get('world'))} "
            f"(exactly one required)")
    return hits[0] if hits else None


def _num(v):
    """A finite real (not a bool — isinstance(True, int) is True)."""
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v))


def _in_domain(value, scale):
    if scale == "rate":
        return 0.0 <= value <= 1.0
    if scale == "ratio":
        return value >= 0.0
    if scale == "count":
        return value >= 0.0
    return False


def _seeds_ok(cell, gate):
    n = cell.get("n_seeds")
    return isinstance(n, int) and not isinstance(n, bool) \
        and n >= gate.n_seeds_required


def _unmeasured(value, threshold, role, reason):
    return {"integrity_value": value, "integrity_threshold": threshold,
            "integrity_severity": None, "role": role, "verdict": "UNMEASURED",
            "reason": reason}


def score_cell(cell, sibling_cells, gate):
    """Score one normalized cell against the gate — the scoring hop. Returns
    {integrity_value, integrity_threshold, integrity_severity, role, verdict,
    reason}. Fail-closed on every uncertainty (never CLOSED-by-default).

    `sibling_cells` is accepted for control-scope reporting; the grade is
    absolute (value vs threshold) and does not subtract a sibling."""
    rule = match_rule(cell, gate)
    if rule is None:
        return _unmeasured(None, None, None,
                           "no rule matches (experiment, arm, world)")
    role, thr, scale = rule.role, rule.threshold, rule.scale
    raw = cell.get(rule.signal_key)
    if thr is None:
        why = ("non-gradeable cross-world aggregate (pp scale, no per-cell "
               "backing, no n_seeds)" if scale == "pp" else
               "routed to frozen-gate.v2; a small-n frozen-gate verdict is an "
               "artifact — UNMEASURED until the real tier")
        return _unmeasured(raw if _num(raw) else None, None, role, why)
    if not _seeds_ok(gate=gate, cell=cell):
        return _unmeasured(None, thr, role,
                           f"n_seeds {cell.get('n_seeds')!r} absent, "
                           f"non-integral, or < {gate.n_seeds_required}")
    if not _num(raw):
        return _unmeasured(None, thr, role,
                           f"value {raw!r} for {rule.signal_key} is not a "
                           "finite real (fail-closed)")
    if not _in_domain(raw, scale):
        return _unmeasured(None, thr, role,
                           f"value {raw!r} out of domain for scale {scale!r}")
    if scale == "count":
        severity, verdict = float(raw), ("OPEN" if raw > thr else "CLOSED")
    else:
        severity, verdict = raw / thr, ("OPEN" if raw > thr else "CLOSED")
    if not rule.axes:
        return {"integrity_value": raw, "integrity_threshold": thr,
                "integrity_severity": severity, "role": role,
                "verdict": "UNMEASURED",
                "reason": "no axis in the profile threat set"}
    return {"integrity_value": raw, "integrity_threshold": thr,
            "integrity_severity": severity, "role": role, "verdict": verdict,
            "reason": ""}


def gate_config(gate):
    """A canonical JSON-able dict of a gate (rules sorted by rule_id)."""
    rules = []
    for r in sorted(gate.rules, key=lambda r: r.rule_id):
        d = asdict(r)
        d["axes"] = list(d["axes"])
        rules.append(d)
    return {"version": gate.version, "n_seeds_required": gate.n_seeds_required,
            "rules": rules}


def gate_digest(gate):
    """sha256 of the canonical gate config — the tamper seal on the index."""
    blob = json.dumps(gate_config(gate), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def gate_from_config(cfg):
    """Reconstruct an IntegrityGate from a serialized gate_config()."""
    rules = tuple(
        MetricRule(rule_id=r["rule_id"], experiment=r["experiment"],
                   arm=r["arm"], world=r["world"], axes=tuple(r["axes"]),
                   signal_key=r["signal_key"], control_scope=r["control_scope"],
                   control_key=r["control_key"], scale=r["scale"],
                   direction=r["direction"], threshold=r["threshold"],
                   role=r["role"])
        for r in cfg["rules"])
    return IntegrityGate(version=cfg["version"],
                         n_seeds_required=cfg["n_seeds_required"], rules=rules)


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("integrity-gate.v1 (pure python):")
    g = DEFAULT_INTEGRITY_GATE
    check("113 rules, one per normalized cell (calibration Appendix A)",
          len(g.rules) == 113)
    coords = {(r.experiment, r.arm, r.world) for r in g.rules}
    check("every rule coordinate is unique", len(coords) == 113)
    axis_cells = {}
    for r in g.rules:
        for ax in r.axes:
            axis_cells[ax] = axis_cells.get(ax, 0) + 1
    check("per-axis cell counts match calibration §4 "
          "(S10 39, S13 28, LLM04 25, LLM05 31, LLM06 15)",
          axis_cells == {"S10": 39, "S13": 28, "LLM04": 25, "LLM05": 31,
                         "LLM06": 15})

    def cell(exp, arm, world, seeds=3, **metrics):
        return {"experiment": exp, "arm": arm, "world": world,
                "n_seeds": seeds, **metrics}

    fixed = cell("resource_exhaustion", "retry-storm", None,
                 amplification_vs_control_mean=1.400244)
    lo = IntegrityGate(rules=(_rule("t", "resource_exhaustion", "retry-storm",
                                    None, ("LLM06",),
                                    "amplification_vs_control_mean", "const",
                                    None, "ratio", 1.25, "defense"),))
    hi = IntegrityGate(rules=(_rule("t", "resource_exhaustion", "retry-storm",
                                    None, ("LLM06",),
                                    "amplification_vs_control_mean", "const",
                                    None, "ratio", 1.5, "defense"),))
    check("(a) same cell flips OPEN<->CLOSED on threshold alone (1.25 vs 1.5)",
          score_cell(fixed, [], lo)["verdict"] == "OPEN"
          and score_cell(fixed, [], hi)["verdict"] == "CLOSED")

    check("(b) zero rule matches -> UNMEASURED",
          score_cell(cell("nope", "x", "y", undetected_mutation_rate_mean=1.0),
                     [], g)["verdict"] == "UNMEASURED")
    dup = IntegrityGate(rules=g.rules + (g.rules[0],))
    r0 = g.rules[0]
    try:
        score_cell(cell(r0.experiment, r0.arm, r0.world,
                        **{r0.signal_key: 1.0}), [], dup)
        check("(b) two rule matches -> error", False)
    except ValueError:
        check("(b) two rule matches -> error", True)

    base = dict(experiment="protocol_integrity", arm="drop", world="defended")
    fc = [("NaN", {**base, "n_seeds": 3, "undetected_mutation_rate_mean":
                   float("nan")}),
          ("+inf", {**base, "n_seeds": 3, "undetected_mutation_rate_mean":
                    float("inf")}),
          ("-inf", {**base, "n_seeds": 3, "undetected_mutation_rate_mean":
                    float("-inf")}),
          ("True", {**base, "n_seeds": 3, "undetected_mutation_rate_mean":
                    True}),
          ("string", {**base, "n_seeds": 3, "undetected_mutation_rate_mean":
                      "1.0"}),
          ("negative rate", {**base, "n_seeds": 3,
                             "undetected_mutation_rate_mean": -0.1}),
          ("rate > 1", {**base, "n_seeds": 3,
                        "undetected_mutation_rate_mean": 1.5}),
          ("n_seeds=2", {**base, "n_seeds": 2,
                         "undetected_mutation_rate_mean": 1.0}),
          ("n_seeds absent", {**base, "undetected_mutation_rate_mean": 1.0}),
          ("n_seeds=2.5", {**base, "n_seeds": 2.5,
                           "undetected_mutation_rate_mean": 1.0})]
    check("(c) fail-closed: NaN/inf/bool/str/out-of-domain/bad-seeds all "
          "UNMEASURED, never CLOSED",
          all(score_cell(c, [], g)["verdict"] == "UNMEASURED"
              for _, c in fc))

    def rex(arm, key, v):
        return score_cell(cell("resource_exhaustion", arm, None, **{key: v}),
                          [], g)
    check("(d) 23.254993 OPEN, 1.400244 CLOSED (retry-storm, D1), 1.018812 "
          "CLOSED",
          rex("protocol-oversize", "amplification_vs_control_mean",
              23.254993)["verdict"] == "OPEN"
          and rex("retry-storm", "amplification_vs_control_mean",
                  1.400244)["verdict"] == "CLOSED"
          and rex("benign-control", "amplification_parity_mean",
                  1.018812)["verdict"] == "CLOSED")

    from .scorecard import _excess
    scored = rex("protocol-oversize", "amplification_vs_control_mean",
                 23.254993)
    check("(e) scored field is integrity_value (scorecard._excess blind)",
          "integrity_value" in scored and _excess(scored) == (None, None))

    r = match_rule(cell("protocol_integrity", "oversize", "vulnerable"), g)
    ov = score_cell(cell("protocol_integrity", "oversize", "vulnerable",
                         undetected_mutation_rate_mean=1.0), [], g)
    check("(f) protocol-oversize spans S10 AND LLM06 with one scored value",
          r.axes == ("S10", "LLM06") and ov["verdict"] == "OPEN"
          and ov["integrity_value"] == 1.0)

    z = score_cell(cell("supply_chain", "all", "negative-control",
                        negative_control_false_rejections=0), [], g)
    one = score_cell(cell("supply_chain", "all", "negative-control",
                          negative_control_false_rejections=1), [], g)
    check("(g) count scale: 0 CLOSED (severity 0.0, no 0/0), 1 OPEN",
          z["verdict"] == "CLOSED" and z["integrity_severity"] == 0.0
          and one["verdict"] == "OPEN" and one["integrity_severity"] == 1.0)

    noaxis = score_cell(cell("malicious_client", "budget-extraction",
                             "vulnerable", attack_success_rate=0.5), [], g)
    check("out-of-profile-axis cell is UNMEASURED with the stated reason",
          noaxis["verdict"] == "UNMEASURED"
          and noaxis["reason"] == "no axis in the profile threat set")
    ms = score_cell(cell("malicious_server_steering", "bwd-direction",
                         "honest", seq_accuracy_mean=0.0297), [], g)
    check("malicious-server honest world is UNMEASURED (frozen-gate routing)",
          ms["verdict"] == "UNMEASURED" and ms["role"] == "control")

    cfg = gate_config(g)
    check("gate reconstructs from its canonical config (round-trip)",
          gate_digest(gate_from_config(cfg)) == gate_digest(g))
    tampered = json.loads(json.dumps(cfg))
    tampered["rules"][0]["threshold"] = 999.0
    check("a hand-edited threshold changes the digest (tamper seal)",
          gate_digest(gate_from_config(tampered)) != gate_digest(g))

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
