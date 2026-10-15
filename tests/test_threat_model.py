"""Structured threat-model schema (attacker/threat_model.py): v2 validation,
legacy normalization, enumerated vocabularies, and skeleton defaults."""

import pytest

from attacker import threat_model as tm


def test_skeleton_is_complete_and_valid():
    doc = tm.skeleton("unit-test threat model")
    assert tm.validate(doc) is doc
    assert doc["schema"] == tm.SCHEMA_V2
    assert set(doc) == {"schema", "summary", *tm.SECTIONS}
    for section, spec in tm.SECTIONS.items():
        assert set(doc[section]) == set(spec)


def test_skeleton_section_overrides_merge():
    doc = tm.skeleton("x", actor={"intent": "active", "role": "network"},
                      evaluation={"coverage_status": "partial"})
    assert doc["actor"]["intent"] == "active"
    assert doc["actor"]["persistence"] is None
    assert doc["evaluation"]["coverage_status"] == "partial"
    with pytest.raises(ValueError, match="bogus"):
        tm.skeleton("x", bogus={"a": 1})


def test_enumerated_vocabularies_are_closed_sets():
    for bad in ({"actor": {"intent": "sneaky"}},
                {"actor": {"role": "superuser"}},
                {"actor": {"persistence": "forever"}},
                {"scope": {"use_case": "webapp"}},
                {"scope": {"deployment": "cloud"}},
                {"controls": {"status": "fixed"}},
                {"evaluation": {"coverage_status": "done"}}):
        with pytest.raises(ValueError):
            tm.validate(tm.skeleton("x", **bad))
    for good in ({"actor": {"intent": "malicious", "role": "insider",
                            "persistence": "cross-epoch"}},
                 {"scope": {"use_case": "split-training",
                            "deployment": "multi-tenant"}},
                 {"controls": {"status": "mitigated"}},
                 {"evaluation": {"coverage_status": "out-of-scope"}}):
        tm.validate(tm.skeleton("x", **good))


def test_type_checks_per_field_kind():
    doc = tm.skeleton("x")
    doc["target"]["assets"] = "not-a-list"
    with pytest.raises(ValueError, match="target.assets"):
        tm.validate(doc)
    doc = tm.skeleton("x")
    doc["target"]["assets"] = ["ok", 3]
    with pytest.raises(ValueError, match="target.assets"):
        tm.validate(doc)
    doc = tm.skeleton("x")
    doc["evaluation"]["budget"] = ["not", "a", "dict"]
    with pytest.raises(ValueError, match="evaluation.budget"):
        tm.validate(doc)
    doc = tm.skeleton("x")
    doc["controls"]["owner"] = 7
    with pytest.raises(ValueError, match="controls.owner"):
        tm.validate(doc)


def test_unknown_keys_rejected_at_both_levels():
    with pytest.raises(ValueError, match="nope"):
        tm.validate(tm.skeleton("x", **{}) | {"nope": 1})
    doc = tm.skeleton("x")
    doc["actor"]["eyes"] = 2
    with pytest.raises(ValueError, match="actor"):
        tm.validate(doc)


def test_summary_and_schema_required():
    with pytest.raises(ValueError, match="summary"):
        tm.validate({"schema": tm.SCHEMA_V2})
    with pytest.raises(ValueError, match="summary"):
        tm.validate(tm.skeleton(""))
    with pytest.raises(ValueError, match="schema"):
        tm.validate({"summary": "x", "schema": "other.v9"})


def test_legacy_string_normalizes_to_summary():
    doc = tm.normalize("honest-but-curious cloud, per-epoch rotation")
    assert tm.is_legacy("honest-but-curious cloud, per-epoch rotation")
    assert not tm.is_legacy(doc)
    assert doc["schema"] == tm.SCHEMA_V2
    assert doc["summary"] == "honest-but-curious cloud, per-epoch rotation"
    tm.validate(doc)


def test_normalize_partial_v2_fills_defaults_and_keeps_fields():
    doc = tm.normalize({"schema": tm.SCHEMA_V2, "summary": "s",
                        "actor": {"intent": "passive"},
                        "mappings": {"owasp_llm_2026": ["LLM02"]}})
    assert doc["actor"]["intent"] == "passive"
    assert doc["mappings"]["owasp_llm_2026"] == ["LLM02"]
    assert doc["scenario"]["attack_steps"] == []
    assert doc["controls"]["residual_risk"] == ""


def test_normalize_rejects_garbage():
    for bad in (42, None, ["x"], {"summary": "no schema"},
                {"schema": tm.SCHEMA_V2, "summary": "s", "actor": "cloud"}):
        with pytest.raises(ValueError):
            tm.normalize(bad)


def test_check_reports_problems_without_raising():
    doc = tm.skeleton("x")
    doc["actor"]["intent"] = "sneaky"
    doc["extra"] = True
    problems = tm.check(doc)
    assert any("actor.intent" in p for p in problems)
    assert any("extra" in p for p in problems)
    assert tm.check(tm.skeleton("ok")) == []


def test_self_test_passes():
    assert tm.self_test() == 0
