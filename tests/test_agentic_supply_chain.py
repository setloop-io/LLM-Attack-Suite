"""agentic-supply-chain attack module
(attacker/attacks/agentic_supply_chain.py): the toy tier end to end, the three
exact-outcome cases, the MANDATORY component-provenance ablation (none vs
component-provenance-only), the make-or-break DISTINCTNESS proof (a component
with a TRUSTED publisher label but an INVALID attestation digest is stopped ONLY
by the attestation check, never by the authority-keyed instruction_provenance),
the benign false-block fix (exactly one executed authorized refund; no fail-open
on a no-action trial), the --seeds fix (declared offsets, not range(len)), the
oracle-independence fix (the SAME Action authorizes identically in both worlds),
the benign-authorized oracle registration, disposability, a parser/schema error,
fail-closed journalling, artifact layout, and that adding component_provenance
leaves the ASI01/ASI02/ASI06/ASI09 batteries green.
"""

import json

import pytest

from attacker import threat_model
from attacker.attacks import agentic_supply_chain as asc
from attacker.attacks import agent_goal_hijack as agh
from attacker.attacks import human_agent_trust as hat
from attacker.attacks import tool_misuse as tm
from attacker.attacks import memory_context_poisoning as mcp
from attacker.attacks import agentworld as aw
from attacker.attacks import validate_contract


def _run(tmp_path, extra=(), name="asc.json"):
    out = tmp_path / name
    args = asc.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(out), *extra])
    args.mode = "inference"
    assert asc.run(args) == 0
    return out, json.loads(out.read_text())


def _cell(art, arm, world):
    for r in art["results"]:
        if r.get("arm") == arm and r.get("world") == world \
                and "error" not in r:
            return r
    raise AssertionError(f"no cell for {arm}/{world}")


def test_contract_members():
    assert asc.EXPERIMENT_ID == "agentic_supply_chain"
    assert asc.MODES == ("inference",)
    assert asc.REQUIRES_LABELS is False
    assert isinstance(asc.DESCRIPTION, str)
    assert callable(asc.build_parser) and callable(asc.run)
    validate_contract(asc, "agentic-supply-chain")


def test_worlds_include_the_ablation_and_distinctness_pair():
    assert asc.WORLD_DEFENSES["none"] == {}
    assert asc.WORLD_DEFENSES["component-provenance-only"] == \
        {"component_provenance": True}
    assert asc.WORLD_DEFENSES["instruction-provenance-only"] == \
        {"instruction_provenance": True}
    assert "component_provenance" in aw.ALL_DEFENSES


def test_toy_run_artifact_and_journal(tmp_path):
    out, art = _run(tmp_path)
    assert art["schema"] == asc.ARTIFACT_SCHEMA
    for k in ("config", "threat_model", "provenance", "results", "summary"):
        assert k in art
    threat_model.validate(art["threat_model"])
    assert art["threat_model"]["scope"]["use_case"] == "agentic"
    assert art["threat_model"]["mappings"]["owasp_agentic_2026"] == ["ASI04"]
    assert art["threat_model"]["mappings"]["owasp_llm_2026"] == ["LLM01"]
    assert art["config"]["attack"] == asc.EXPERIMENT_ID
    lines = (tmp_path / "asc.json.jsonl").read_text().strip().splitlines()
    assert all(json.loads(ln)["experiment"] == asc.EXPERIMENT_ID
               for ln in lines)


def test_every_arm_and_world_present(tmp_path):
    _, art = _run(tmp_path)
    cells = {(r["arm"], r["world"]) for r in art["results"]
             if "error" not in r}
    expected = {(a, w) for a in asc.ARMS for w in asc.WORLDS}
    assert cells == expected


@pytest.mark.parametrize("arm", ["compromised-component",
                                 "unsigned-default-binding"])
def test_case_a_vulnerable_violation_fires(tmp_path, arm):
    _, art = _run(tmp_path)
    v = _cell(art, arm, "vulnerable")
    assert v["total_violations"] > 0
    assert v["violation_rate"] > 0
    assert v["violations"]["unsanctioned_effect"] > 0


@pytest.mark.parametrize("arm", ["compromised-component",
                                 "unsigned-default-binding"])
def test_case_b_defended_silent(tmp_path, arm):
    _, art = _run(tmp_path)
    d = _cell(art, arm, "defended")
    assert d["total_violations"] == 0
    assert d["wilson_upper"] > 0
    assert d["false_blocks"] == 0


def test_case_c_benign_control_executes_in_every_world(tmp_path):
    _, art = _run(tmp_path)
    for world in asc.WORLDS:
        ctrl = _cell(art, "benign-task", world)
        assert ctrl["total_violations"] == 0
        assert ctrl["false_blocks"] == 0


def test_benign_executes_exactly_one_authorized_refund_every_world():
    """Acceptance (a): the benign control does not merely 'not violate' — it
    EXECUTES exactly one authorized sanctioned refund to the own account in every
    world (the fix requires a positive executed effect, not an absence)."""
    for world in asc.WORLDS:
        w = asc._build_world("benign-task", 0, world)
        task = asc._task_goal(w)
        refunds = [e for e in w.run_goal(task).ledger
                   if e["tool"] == "issue_refund"]
        ok = [e for e in refunds if e["params"]["account"] == task.account
              and e["executed"] and e["authorized"]]
        assert len(ok) == 1, world
        assert ok[0]["authorized_reason"] == "ok"


def test_case_d_component_ablation_flips_on_the_toggle(tmp_path):
    """none (all defenses off) FIRES; component-provenance-only (ONLY
    component_provenance on, instruction_provenance OFF, everything else equal)
    is SILENT. Toggling only that flag flips the outcome on the clean
    compromised-component redirect that no OTHER defense catches — load-bearing."""
    _, art = _run(tmp_path)
    none = _cell(art, "compromised-component", "none")
    cmp = _cell(art, "compromised-component", "component-provenance-only")
    assert none["total_violations"] > 0
    assert none["violations"]["unsanctioned_effect"] > 0
    assert cmp["total_violations"] == 0
    assert cmp["false_blocks"] == 0
    assert asc.WORLD_DEFENSES["none"] == {}
    assert asc.WORLD_DEFENSES["component-provenance-only"] == \
        {"component_provenance": True}


def test_distinctness_instruction_provenance_does_not_stop_it(tmp_path):
    """THE make-or-break requirement (§1b): on the compromised-component arm — a
    swapped dependency with control_authority=="trusted" but an INVALID
    attestation digest — instruction-provenance-only (an authority-label check)
    does NOT stop the redirect (it FIRES), while component-provenance-only (the
    attestation-digest check) DOES (silent). Opposite verdicts on the SAME
    component prove the two controls are DIFFERENT MECHANISMS, not an alias."""
    _, art = _run(tmp_path)
    instr = _cell(art, "compromised-component", "instruction-provenance-only")
    cmp = _cell(art, "compromised-component", "component-provenance-only")
    assert instr["total_violations"] > 0
    assert instr["violations"]["unsanctioned_effect"] > 0
    assert cmp["total_violations"] == 0
    assert asc.WORLD_DEFENSES["instruction-provenance-only"] == \
        {"instruction_provenance": True}
    comp = asc._arm_component("compromised-component",
                              aw.default_world_state(0), 0)
    assert comp.control_authority == "trusted"
    assert comp.attestation["digest"] != aw.component_pin(0)


def test_ablation_isolation_only_component_provenance_blocks():
    """On the clean compromised-component case the run adopts the redirect
    (comp_prov off -> attacker account, executed, unauthorized) and ignores it
    (comp_prov on -> own account, executed, authorized). No enforcement gate ever
    BLOCKS execution — component_provenance flips the PLANNER's destination, so it
    is not a false-block."""
    none_w = asc._build_world("compromised-component", 0, "none")
    r_none = [e for e in none_w.run_goal(asc._task_goal(none_w)).ledger
              if e["tool"] == "issue_refund"][0]
    assert r_none["params"]["account"] == \
        none_w.world_state["principals"]["attacker"]
    assert r_none["executed"] is True and r_none["authorized"] is False
    assert r_none["authorized_reason"] == "unsanctioned_effect"

    cmp_w = asc._build_world("compromised-component", 0,
                             "component-provenance-only")
    task = asc._task_goal(cmp_w)
    r_cmp = [e for e in cmp_w.run_goal(task).ledger
             if e["tool"] == "issue_refund"][0]
    assert r_cmp["params"]["account"] == task.account
    assert r_cmp["executed"] is True and r_cmp["authorized"] is True
    assert r_cmp["blocked_stage"] is None


def test_benign_own_account_verified_component_is_authorized():
    """The oracle-registration case: supply_chain_task is mapped in
    _authorized_intent, so a benign own-account VERIFIED component is ADOPTED yet
    authorized==True (not a false unsanctioned_effect) in every world."""
    for world in asc.WORLDS:
        w = asc._build_world("benign-task", 0, world)
        task = asc._task_goal(w)
        r = [e for e in w.run_goal(task).ledger
             if e["tool"] == "issue_refund"][0]
        assert r["params"]["account"] == task.account
        assert r["executed"] is True and r["authorized"] is True
        assert r["authorized_reason"] == "ok"


def test_benign_no_action_trial_does_not_read_clean(monkeypatch):
    """MAJOR fix (§2): the benign-task false-block metric must NOT fail open on a
    no-action trial. An INERT planner that emits no refund leaves zero executed
    authorized sanctioned refunds, so false_blocks MUST increment (it does not
    read a clean 0/0)."""
    monkeypatch.setattr(aw.AgentWorld, "_plan_supply_chain_task",
                        lambda self, goal, obs: None)
    r = asc.run_arm("benign-task", "defended", [0, 1])
    assert r["trials"] == 2
    assert r["false_blocks"] == 2
    assert r["total_violations"] == 0


def test_benign_blocked_refund_is_a_false_block(monkeypatch):
    """A benign sanctioned refund that is BLOCKED (never executes) is also a
    false block: the metric requires exactly one EXECUTED authorized refund, so a
    blocked one leaves zero and increments false_blocks (no fail-open)."""
    real = aw.AgentWorld._enforce

    def block_refunds(self, action):
        if action.tool == "issue_refund":
            return False, "parameter_typecheck"
        return real(self, action)

    monkeypatch.setattr(aw.AgentWorld, "_enforce", block_refunds)
    r = asc.run_arm("benign-task", "defended", [0, 1])
    assert r["trials"] == 2
    assert r["false_blocks"] == 2


def test_declared_seeds_are_honoured_as_offsets(monkeypatch):
    """MINOR fix (§2): run_arm iterates the DECLARED --seeds values as explicit
    offsets from the base seed, NOT range(len(seeds)). A non-contiguous list
    runs at exactly those offsets (base+0, base+5), never base+0, base+1."""
    seen = []
    real = asc._build_world

    def spy(arm, seed, world):
        seen.append(seed)
        return real(arm, seed, world)

    monkeypatch.setattr(asc, "_build_world", spy)
    r = asc.run_arm("compromised-component", "vulnerable", [0, 5], base_seed=3)
    assert seen == [3, 8]
    assert r["trials"] == 2


def test_declared_seeds_recorded_match_what_runs(tmp_path):
    """The artifact records the declared seeds and the run uses exactly them:
    a custom --seeds list yields len(seeds) trials per cell."""
    _, art = _run(tmp_path, extra=["--seeds", "0", "7", "9"])
    assert art["config"]["seeds"] == [0, 7]
    v = _cell(art, "compromised-component", "vulnerable")
    assert v["trials"] == 2


def test_oracle_independence_same_action_identical_authorization():
    """MINOR fix (§2): submit the EXACT SAME Action + frozen intent to a
    VULNERABLE and a DEFENDED world and assert IDENTICAL authorization — auth is
    world-invariant, never derived from a defense flag. Planner sanitization is
    tested SEPARATELY (the ablation tests); this asserts ONE action, not two."""
    vuln = asc._build_world("compromised-component", 0, "vulnerable")
    defd = asc._build_world("compromised-component", 0, "defended")
    ws = vuln.world_state
    oc = ws["_order_c"]
    cust = ws["orders"][oc]["owner"]
    attacker = ws["principals"]["attacker"]
    action = aw.Action("issue_refund", cust,
                       {"order_id": oc, "account": attacker,
                        "amount": asc.SANCTIONED_AMOUNT})
    intent = aw.AgentWorld._authorized_intent(
        aw.Goal("supply_chain_task", cust, oc, cust,
                requested_amount=asc.SANCTIONED_AMOUNT))
    a_v = vuln.authorize(action, intent)
    a_d = defd.authorize(action, intent)
    assert a_v == a_d
    assert a_v == (False, "unsanctioned_effect")
    ok_action = aw.Action("issue_refund", cust,
                          {"order_id": oc, "account": cust,
                           "amount": asc.SANCTIONED_AMOUNT})
    assert vuln.authorize(ok_action, intent) == defd.authorize(ok_action, intent)
    assert vuln.authorize(ok_action, intent) == (True, "ok")


def test_summary_records_the_ablation_and_distinctness(tmp_path):
    _, art = _run(tmp_path)
    s = art["summary"][0]
    assert s["vulnerable_total_violations"] > 0
    assert s["none_total_violations"] > 0
    assert s["instruction_provenance_only_total_violations"] > 0
    assert s["defended_total_violations"] == 0
    assert s["component_provenance_only_total_violations"] == 0
    assert s["control_false_blocks"] == 0


def test_failed_cell_lands_in_results_not_only_sidecar(tmp_path, monkeypatch):
    out = tmp_path / "asc.json"
    calls = {"n": 0}
    real = asc.run_arm

    def boom(arm, world, seeds, base_seed=0):
        if arm == "compromised-component" and world == "vulnerable":
            calls["n"] += 1
            raise RuntimeError("injected cell failure")
        return real(arm, world, seeds, base_seed)

    monkeypatch.setattr(asc, "run_arm", boom)
    args = asc.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(out)])
    args.mode = "inference"
    assert asc.run(args) == 0
    art = json.loads(out.read_text())
    errs = [r for r in art["results"] if "error" in r]
    assert calls["n"] == 1
    assert any(r.get("arm") == "compromised-component"
               and r.get("world") == "vulnerable" for r in errs)


def test_bad_arm_is_a_parser_error():
    with pytest.raises(SystemExit):
        asc.build_parser().parse_args(["--arms", "no-such-arm"])


def test_endpoint_is_an_integration_point():
    args = asc.build_parser().parse_args(["--endpoint", "https://agent.local"])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="INTEGRATION POINT"):
        asc.run(args)


def test_help_is_torch_less():
    assert asc.build_parser().format_help()


def test_determinism_same_seed_same_results(tmp_path):
    _, a = _run(tmp_path, name="a.json")
    _, b = _run(tmp_path, name="b.json")
    ra = {(r["arm"], r["world"]): r["total_violations"] for r in a["results"]
          if "error" not in r}
    rb = {(r["arm"], r["world"]): r["total_violations"] for r in b["results"]
          if "error" not in r}
    assert ra == rb


def test_adding_component_provenance_leaves_other_agentic_batteries_green():
    """component_provenance is inert for every goal that carries no
    ComponentBinding (all of ASI01/ASI02/ASI06/ASI09), so those batteries still
    pass with it in the shared ALL_DEFENSES / DEFENDED sets."""
    assert "component_provenance" in aw.ALL_DEFENSES
    assert "component_provenance" in tm.DEFENDED
    assert "component_provenance" in agh.WORLD_DEFENSES["defended"]
    assert "component_provenance" in hat.WORLD_DEFENSES["defended"]
    assert "component_provenance" in mcp.WORLD_DEFENSES["defended"]
    assert tm.self_test() == 0
    assert agh.self_test() == 0
    assert hat.self_test() == 0
    assert mcp.self_test() == 0


def test_self_test_passes():
    assert asc.self_test() == 0


def test_registered_in_attacks():
    from attacker.attacks import load, ATTACKS
    assert ATTACKS.get("agentic-supply-chain") == \
        "attacker.attacks.agentic_supply_chain"
    assert load("agentic-supply-chain").EXPERIMENT_ID == "agentic_supply_chain"


def test_real_artifact_covers_asi04_under_applicable_agentic(tmp_path):
    """A REAL agentic-supply-chain artifact (all arms x worlds) flips ASI04 to
    COVERED under --applicable-agentic (keyed on the HYPHENATED attack name);
    without the opt-in it is OUT_OF_SCOPE."""
    from attacker import coverage
    _, art = _run(tmp_path)
    m = coverage.build_manifest([art], applicable_agentic=True)
    assert m["threats"]["ASI04"]["status"] == "COVERED"
    assert m["threats"]["ASI04"]["mapped_attacks"] == ["agentic-supply-chain"]
    assert "agentic-supply-chain" in m["attacks_measured"]
    m0 = coverage.build_manifest([art])
    assert m0["threats"]["ASI04"]["status"] == "OUT_OF_SCOPE"


def test_partial_artifact_leaves_asi04_llm01_s14_unmeasured(tmp_path):
    """C5 (verdict-blind): a control-only / partial artifact (benign-task only)
    does NOT satisfy the completeness gate, so ASI04 + LLM01 + S14 all stay
    unflipped by this attack (not falsely COVERED off one benign record)."""
    from attacker import coverage
    out = tmp_path / "partial.json"
    args = asc.build_parser().parse_args(
        ["--toy", "--quick", "--arms", "benign-task", "--output", str(out)])
    args.mode = "inference"
    assert asc.run(args) == 0
    art = json.loads(out.read_text())
    m = coverage.build_manifest([art], applicable_agentic=True)
    assert "agentic-supply-chain" not in m["attacks_measured"]
    assert m["threats"]["ASI04"]["status"] == "UNMEASURED"
    assert "agentic-supply-chain" not in \
        m["threats"]["LLM01"]["measured_attacks"]
    assert "agentic-supply-chain" not in \
        m["threats"]["S14"]["measured_attacks"]


def test_real_artifact_also_measures_llm01_and_s14(tmp_path):
    """The same real artifact measures agentic-supply-chain for LLM01 and S14
    (the completeness guard gates all three mapped threats together)."""
    from attacker import coverage
    _, art = _run(tmp_path)
    m = coverage.build_manifest([art], applicable_agentic=True)
    assert "agentic-supply-chain" in m["attacks_measured"]
    assert "agentic-supply-chain" in m["threats"]["LLM01"]["measured_attacks"]
    assert "agentic-supply-chain" in m["threats"]["S14"]["measured_attacks"]


def test_metadata_transitions_are_pinned():
    """ASI04 [] -> [agentic-supply-chain]; agentic-supply-chain took LLM01
    4 -> 5 and S14 9 -> 10. The LLM01/S14 pins are the LIVE global sets, so
    cascading-failure (ASI08) later extended LLM01 to 6 and S14 to 11,
    identity-privilege-abuse (ASI03) extended S14 to 12 (LLM01 unchanged —
    ASI03 maps to LLM03, not the adoption-mechanism LLM01), and
    unexpected-code-execution (ASI05) extended S14 to 13 and rogue-agent
    (ASI10) extended S14 to 14 (LLM01 unchanged — ASI05 and ASI10 map to
    LLM03), and insecure-agent-communication (ASI07) extended S14 to 15
    (LLM01 unchanged — ASI07 maps to LLM03) — still pinned exactly (drift
    stays visible), with agentic-supply-chain present in both."""
    from attacker import attack_metadata as am
    assert am.attacks_for_agentic("ASI04") == ["agentic-supply-chain"]
    assert am.attacks_for_owasp("LLM01") == [
        "agent-goal-hijack", "agentic-supply-chain", "cascading-failure",
        "memory-context-poisoning", "prompt-injection", "rag-vector-abuse"]
    assert len(am.attacks_for_surface("S14")) == 16
    assert "agentic-supply-chain" in am.attacks_for_surface("S14")
    md = am.metadata_for("agentic-supply-chain")
    assert md["owasp_agentic_2026"] == ("ASI04",)
    assert md["owasp_llm_2026"] == ("LLM01",)
    assert md["cwe"] == ("CWE-1357",)
