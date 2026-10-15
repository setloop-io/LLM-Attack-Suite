"""Reference defenses (attacker/defenses.py).

Two layers: the fast torch-free layer (scenario description, config
validation, registry errors, torch-missing behavior) runs everywhere; the
BDD layer (bundle generation, the 4-defenses x latent-probe matrix) is
marked slow and skips when torch (or the defense SDK) is absent."""

import importlib.util
import json

import pytest

from attacker import defenses
from attacker.defenses import (DEFENSES, SHAPE_DEFAULTS,
                               describe_scenarios, make_defense,
                               run_scenarios, validate_defense_config)


def test_describe_scenarios_is_the_living_matrix():
    scenarios = describe_scenarios()
    assert [s["defense"] for s in scenarios] == \
        ["plain-split", "rotation", "dp-noise", "composite"]
    assert all(s["attack"] == "latent-probe" for s in scenarios)
    assert [s["expected_verdict"] for s in scenarios] == \
        ["FAIL", "FAIL", "FAIL", "PASS"]
    for s in scenarios:
        assert s["why"] and s["description"]
        for key in SHAPE_DEFAULTS:
            assert key in s["config"]
    assert describe_scenarios() == scenarios
    scenarios[0]["config"]["hidden"] = -1
    assert describe_scenarios()[0]["config"]["hidden"] == \
        SHAPE_DEFAULTS["hidden"]


def test_scenario_configs_validate_clean():
    for s in describe_scenarios():
        assert validate_defense_config(s["defense"], s["config"]) == []


def test_validate_defense_config_rejections():
    assert "unknown defense" in validate_defense_config("nope", {})[0]
    assert validate_defense_config("dp-noise", {"sgima": 1.0})
    assert validate_defense_config("dp-noise", {"sigma": -1.0})
    assert validate_defense_config("dp-noise", {"sigma": "loud"})
    assert validate_defense_config("composite", {"permute": 1})
    assert validate_defense_config("rotation", {"hidden": 2})
    assert validate_defense_config("rotation", {"tokens": 1})
    assert validate_defense_config("rotation", {"vocab": True})
    assert validate_defense_config("plain-split",
                                   {"signal_scale": 0.0})
    assert validate_defense_config("dp-noise", {"sigma": 0.5}) == []
    assert validate_defense_config("plain-split", {"sigma": 0.5})
    assert validate_defense_config("rotation", {"permute": True})


def test_make_defense_registry():
    d = make_defense("dp-noise", sigma=0.5, hidden=16)
    assert d.name == "dp-noise"
    assert d.config["sigma"] == 0.5 and d.config["hidden"] == 16
    assert d.config["tokens"] == SHAPE_DEFAULTS["tokens"]
    assert d.expected_verdict == "FAIL"
    with pytest.raises(ValueError, match="unknown defense 'nope'"):
        make_defense("nope")
    with pytest.raises(ValueError, match="invalid config"):
        make_defense("rotation", hidden=1)


def test_torchfree_host_behavior_is_explicit():
    if defenses.torch is not None:
        pytest.skip("torch host: the torch-free guard is covered by the "
                    "module self-test's torch branch")
    with pytest.raises(SystemExit, match="requires torch"):
        make_defense("rotation").generate(0)
    with pytest.raises(SystemExit, match="requires torch"):
        run_scenarios("/tmp/never_written")


def test_module_self_test_passes():
    assert defenses.self_test() == 0


TINY = {"hidden": 16, "tokens": 4, "train_blocks": 8, "eval_blocks": 4,
        "vocab": 4, "signal_scale": 4.0}


@pytest.mark.slow
def test_bundle_generation_schema_and_rotation_invariants():
    torch = pytest.importorskip("torch")
    for name in DEFENSES:
        bundle = make_defense(name, **TINY).generate(0)
        assert bundle["schema"] == defenses.SCHEMA
        assert bundle["defense"] == name
        for wire_key, tok_key, blocks in (
                ("train_wire", "train_tokens", TINY["train_blocks"]),
                ("eval_wire", "eval_tokens", TINY["eval_blocks"])):
            wire, tokens = bundle[wire_key], bundle[tok_key]
            assert wire.shape == (blocks, TINY["tokens"], TINY["hidden"])
            assert tokens.shape == (blocks, TINY["tokens"])
            assert wire.dtype == torch.float32
        assert bundle["train_wire"].shape[1:] == \
            bundle["eval_wire"].shape[1:]

    plain = make_defense("plain-split", **TINY).generate(0)
    rot = make_defense("rotation", **TINY).generate(0)
    assert torch.allclose(plain["eval_wire"].norm(dim=-1),
                          rot["eval_wire"].norm(dim=-1), atol=1e-5)
    g_plain = torch.einsum("btd,bsd->bts", plain["eval_wire"],
                           plain["eval_wire"])
    g_rot = torch.einsum("btd,bsd->bts", rot["eval_wire"],
                         rot["eval_wire"])
    assert torch.allclose(g_plain, g_rot, atol=1e-4)
    assert not torch.allclose(plain["eval_wire"], rot["eval_wire"])
    means = torch.stack([plain["train_wire"].reshape(-1, TINY["hidden"])
                         [plain["train_tokens"].reshape(-1) == v].mean(0)
                         for v in range(TINY["vocab"])])
    flat = plain["eval_wire"].reshape(-1, TINY["hidden"])
    pred = torch.cdist(flat, means).argmin(1)
    acc = (pred == plain["eval_tokens"].reshape(-1)).float().mean()
    assert acc > 0.8


@pytest.mark.slow
def test_order_ground_truth_inverts_the_recorded_permutation():
    """The train_order/eval_order fields are per-frame permutations that
    genuinely invert the release map's permutation (composite), and the
    identity when the defense does not permute (plain-split)."""
    torch = pytest.importorskip("torch")
    arange = torch.arange(TINY["tokens"])
    plain = make_defense("plain-split", **TINY).generate(0)
    comp = make_defense("composite", sigma=0.0, **TINY).generate(0)
    for bundle in (plain, comp):
        for key in ("train_order", "eval_order"):
            blocks = TINY["train_blocks"] if "train" in key \
                else TINY["eval_blocks"]
            assert bundle[key].shape == (blocks, TINY["tokens"])
            assert bundle[key].dtype == torch.long
            assert torch.all(bundle[key].sort(-1).values == arange)
    assert torch.all(plain["eval_order"] == arange)
    assert not torch.all(comp["eval_order"] == arange)
    assert torch.allclose(
        torch.gather(plain["eval_wire"].norm(dim=-1), 1, comp["eval_order"]),
        comp["eval_wire"].norm(dim=-1), atol=1e-5)


@pytest.mark.slow
def test_run_scenarios_end_to_end(tmp_path):
    pytest.importorskip("torch")
    if importlib.util.find_spec("privacy_runtime") is None:
        pytest.skip("privacy_runtime (defense SDK) not importable on "
                    "this host")
    rows = run_scenarios(str(tmp_path), shape=TINY, epochs=1, restarts=1,
                         timeout=600)
    assert len(rows) == 4
    scorecard = json.load(open(tmp_path / "scorecard.json"))
    assert scorecard["schema"] == defenses.SCORECARD_SCHEMA
    for row in rows:
        assert row["status"] == "ok", row.get("stderr_tail")
        assert row["verdict"] in ("PASS", "FAIL")
        assert row["expectation_met"] is not None
    plain = [r for r in rows if r["defense"] == "plain-split"][0]
    assert plain["verdict"] == "FAIL"
    assert plain["excess_pp"] > 1.0
    composite = [r for r in rows if r["defense"] == "composite"][0]
    assert "at_floor" in composite and "excess_pp" in composite
