"""um-timing (gpu-memory unified-memory contention arm): GB10-class
allocation-free co-tenant channel — toy tier end-to-end (planted schedule
recovery AUC vs shuffled control, gate, journal/artifact/summary), the
combined-arm plumbing, and the descriptive SystemExit contract for
non-toy runs without CUDA. The torch-CUDA measurement harness self-skips
without a visible CUDA device."""

import json

import pytest

from attacker import hwprobe
from attacker.attacks import gpu_memory
from attacker.attacks.lira_membership import rank_auc


def _run_toy(tmp_path, *extra, mode="inference", name="um.json"):
    args = gpu_memory.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(tmp_path / name), *extra])
    args.mode = mode
    assert gpu_memory.run(args) == 0
    art = json.loads((tmp_path / name).read_text())
    journal = [json.loads(l) for l in
               (tmp_path / (name + ".jsonl")).read_text().splitlines()]
    return art, journal


def test_arm_registered_and_help_torchless():
    assert "um-timing" in gpu_memory.ARMS
    ap = gpu_memory.build_parser()
    with pytest.raises(SystemExit) as exc:
        ap.parse_args(["--help"])
    assert exc.value.code == 0
    args = ap.parse_args(["--arm", "um-timing"])
    assert args.arm == ["um-timing"]
    assert args.slots == 64 and args.on_fraction == 0.5


def test_self_test_passes():
    assert gpu_memory.self_test() == 0


def test_sim_cell_planted_vs_control():
    rec = gpu_memory.sim_um_timing_cell(64, 0.5, 5.0, 8.0, 1.0, 1)
    assert rec["schedule_recovery_auc"] >= 0.99
    assert 0.3 <= rec["control_auc"] <= 0.7
    assert rec["latency_delta_ms"] > 5.0
    assert rec["n_eval_slots"] == min(rec["on_slots"],
                                      rec["slots"] - rec["on_slots"])
    quiet = gpu_memory.sim_um_timing_cell(64, 0.5, 5.0, 0.0, 1.0, 1)
    assert 0.3 <= quiet["schedule_recovery_auc"] <= 0.7


def test_toy_end_to_end(tmp_path):
    art, journal = _run_toy(tmp_path, "--arm", "um-timing")
    recs = [r for r in art["results"] if r["arm"] == "um-timing"]
    assert len(recs) == len(journal) == 2
    for r in recs:
        assert r["tier"] == "sim" and r["defense"] == "n/a"
        assert r["schedule_recovery_auc"] >= 0.99
        assert 0.3 <= r["control_auc"] <= 0.7
        assert r["gate"]["verdict"] == "FAIL"
        assert r["contention_channel"] == "open"
        assert "residue_channel" not in r
        assert "defense_validated" not in r
        assert r["gate"]["n"] == r["n_eval_slots"]
    (summ,) = [s for s in art["summary"] if s["arm"] == "um-timing"]
    assert summ["schedule_recovery_auc_mean"] >= 0.99
    assert summ["gate_verdict"] == "FAIL"
    assert summ["contention_channel"] == "open"
    assert "defense_validated" not in summ


def test_combined_arms_share_the_plumbing(tmp_path):
    """All three arms in one run: the new arm must not disturb the
    existing residue arms' cells, defenses, or summary rows."""
    art, journal = _run_toy(tmp_path, "--arm", "allocator-residue",
                            "um-timing")
    arms = {r["arm"] for r in art["results"]}
    assert arms == {"allocator-residue", "um-timing"}
    assert len(journal) == len(art["results"])
    for r in art["results"]:
        if r["arm"] == "allocator-residue":
            assert r["token_recovery"] == 1.0
            assert r["residue_channel"] == "open"
            assert r["defense_validated"] is False
        else:
            assert r["contention_channel"] == "open"
    summ_arms = {s["arm"] for s in art["summary"]}
    assert summ_arms == {"allocator-residue", "um-timing"}


def test_evidence_chain(tmp_path):
    from attacker.evidence import verify_evidence
    out = tmp_path / "ev.json"
    args = gpu_memory.build_parser().parse_args(
        ["--toy", "--quick", "--arm", "um-timing", "--evidence",
         "--output", str(out)])
    args.mode = "inference"
    assert gpu_memory.run(args) == 0
    ok, problems = verify_evidence(str(out) + ".evidence.json")
    assert ok and problems == []


def test_non_toy_without_cuda_is_a_descriptive_systemexit():
    """House style: non-toy um-timing never raises a raw error without
    CUDA — it is a descriptive SystemExit naming the tier and --toy."""
    if hwprobe.torch_cuda_available():
        pytest.skip("CUDA host: the real measurement harness runs here")
    args = gpu_memory.build_parser().parse_args(["--arm", "um-timing"])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="--toy"):
        gpu_memory.run(args)


def test_non_toy_um_timing_with_only_pynvml_is_descriptive(monkeypatch):
    """A pynvml-only host can run the residue arms but um-timing needs
    kernel-launch control: its own descriptive SystemExit."""
    monkeypatch.setattr(hwprobe, "detect_tiers", lambda: ["pynvml"])
    args = gpu_memory.build_parser().parse_args(["--arm", "um-timing"])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="torch\\+CUDA"):
        gpu_memory.run(args)


def test_rank_auc_convention_matches_lira():
    """The AUC helper is the shared lira one (tie-averaged Mann-Whitney)."""
    assert rank_auc([2.0, 3.0], [0.0, 1.0]) == 1.0
    assert rank_auc([1.0, 1.0], [1.0, 1.0]) == 0.5
