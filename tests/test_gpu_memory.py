"""gpu-memory (LeftoverLocals-style, CVE-2023-4969): toy end-to-end runs,
scrub-check defense validation, gate verdicts, evidence chain, dispatch.
All torch-free (the sim tier)."""

import importlib
import json
import os

import pytest

from attacker import hwprobe
from attacker.attacks import gpu_memory
from attacker.evidence import verify_evidence


@pytest.fixture
def registered():
    """Dispatch through the real registry without editing ATTACKS."""
    from attacker import attacks
    mod = importlib.import_module("attacker.attacks.gpu_memory")
    registered_here = "gpu-memory" not in attacks.ATTACKS
    if registered_here:
        attacks.register_attack("gpu-memory", mod)
    yield mod
    if registered_here:
        attacks.ATTACKS.pop("gpu-memory", None)


def _run_toy(tmp_path, *extra, mode="training"):
    args = gpu_memory.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(tmp_path / "gm.json"), *extra])
    args.mode = mode
    assert gpu_memory.run(args) == 0
    out = tmp_path / "gm.json"
    art = json.loads(out.read_text())
    journal = [json.loads(l) for l in
               (tmp_path / "gm.json.jsonl").read_text().splitlines()]
    return art, journal


def test_contract_and_torchless_help():
    from attacker.attacks import validate_contract
    assert validate_contract(gpu_memory) is gpu_memory
    assert gpu_memory.MODES == ("training", "inference")
    assert gpu_memory.REQUIRES_LABELS is False
    ap = gpu_memory.build_parser()
    with pytest.raises(SystemExit) as exc:
        ap.parse_args(["--help"])
    assert exc.value.code == 0


def test_self_test_passes():
    assert gpu_memory.self_test() == 0


def test_dispatch_toy_end_to_end(registered, tmp_path):
    from attacker.__main__ import main
    out = tmp_path / "disp.json"
    rc = main(["--mode", "inference", "--attack", "gpu-memory",
               "--toy", "--quick", "--output", str(out)])
    assert rc == 0
    art = json.loads(out.read_text())
    assert art["schema"] == "dtraining.attacker.gpu_memory.v1"
    assert art["config"]["attack"] == gpu_memory.EXPERIMENT_ID
    assert art["config"]["mode"] == "inference"
    assert art["results"] and art["summary"]
    journal = (tmp_path / "disp.json.jsonl").read_text().strip().splitlines()
    assert len(journal) == len(art["results"])
    assert all(json.loads(l)["experiment"] == gpu_memory.EXPERIMENT_ID
               for l in journal)


def test_allocator_residue_unscrubbed_recovers(tmp_path):
    art, journal = _run_toy(tmp_path, "--arm", "allocator-residue")
    recs = [r for r in art["results"] if r["arm"] == "allocator-residue"]
    assert recs and all(r["defense"] == "unscrubbed" for r in recs)
    for r in recs:
        assert r["tier"] == "sim"
        assert r["token_recovery"] == 1.0
        assert r["recovery_rate"] == 1.0
        assert r["fraction_nonzero"] > 0.9
        assert r["control_token_recovery"] <= 0.2
        assert r["gate"]["verdict"] == "FAIL"
        assert r["residue_channel"] == "open"
        assert r["defense_validated"] is False
    summ = [s for s in art["summary"] if s["arm"] == "allocator-residue"]
    assert summ[0]["token_recovery_mean"] == 1.0
    assert summ[0]["gate_verdict"] == "FAIL"


def test_allocator_residue_scrubbed_holds(tmp_path):
    art, _ = _run_toy(tmp_path, "--arm", "allocator-residue", "--scrubbed")
    recs = art["results"]
    assert all(r["defense"] == "scrubbed" for r in recs)
    for r in recs:
        assert r["token_recovery"] <= 0.2
        assert r["fraction_nonzero"] == 0.0
        assert r["bytes_recoverable"] == 0
        assert r["gate"]["verdict"] == "INSUFFICIENT_EVIDENCE"
        assert r["residue_channel"] == "unmeasured"
        assert r["defense_validated"] is False


def test_scrub_check_arm_defense_contrast(tmp_path):
    """The defense-claim arm: scrubbed must show ~0 recovery, unscrubbed
    must show full canary recovery — one run carries both."""
    art, journal = _run_toy(tmp_path, "--arm", "scrub-check")
    by_defense = {}
    for r in art["results"]:
        assert r["arm"] == "scrub-check"
        by_defense.setdefault(r["defense"], []).append(r)
    assert set(by_defense) == {"scrubbed", "unscrubbed"}
    for r in by_defense["unscrubbed"]:
        assert r["canary_recovery"] == 1.0
        assert r["gate"]["verdict"] == "FAIL"
        assert r["defense_validated"] is False
    for r in by_defense["scrubbed"]:
        assert r["canary_recovery"] == 0.0
        assert r["canary_found"] == 0
        assert r["gate"]["verdict"] == "INSUFFICIENT_EVIDENCE"
        assert r["defense_validated"] is False
    summ = {(s["arm"], s["defense"]): s for s in art["summary"]}
    assert summ[("scrub-check", "scrubbed")]["canary_recovery_mean"] == 0.0
    assert summ[("scrub-check", "unscrubbed")]["canary_recovery_mean"] == 1.0


def test_evidence_chain(tmp_path):
    out = tmp_path / "ev.json"
    args = gpu_memory.build_parser().parse_args(
        ["--toy", "--quick", "--arm", "scrub-check", "--evidence",
         "--output", str(out)])
    args.mode = "training"
    assert gpu_memory.run(args) == 0
    manifest = str(out) + ".evidence.json"
    assert os.path.isfile(manifest)
    assert os.path.isfile(str(out) + ".evidence.jsonl")
    ok, problems = verify_evidence(manifest)
    assert ok and problems == []
    m = json.loads(open(manifest).read())
    assert m["node_role"] == f"attack/{gpu_memory.EXPERIMENT_ID}"
    assert m["artifact"]["sha256"]


def test_evidence_requires_output():
    args = gpu_memory.build_parser().parse_args(
        ["--toy", "--quick", "--evidence"])
    args.mode = "training"
    with pytest.raises(SystemExit, match="--output"):
        gpu_memory.run(args)


def test_real_tiers_need_hardware_else_descriptive_error():
    if hwprobe.detect_tiers():
        pytest.skip("GPU host: real probe tiers are available here")
    args = gpu_memory.build_parser().parse_args([])
    args.mode = "training"
    with pytest.raises(SystemExit, match="--toy"):
        gpu_memory.run(args)


def test_hwprobe_self_test():
    assert hwprobe.self_test() == 0


class _FakeNVMLError(Exception):
    pass


class _FakeNVML:
    """pynvml stand-in for a unified-memory device: GetMemoryInfo raises
    NotSupported, utilization + process queries work (measured on GB10)."""
    NVMLError = _FakeNVMLError

    @staticmethod
    def nvmlInit():
        pass

    @staticmethod
    def nvmlShutdown():
        pass

    @staticmethod
    def nvmlDeviceGetHandleByIndex(i):
        return object()

    @staticmethod
    def nvmlDeviceGetCount():
        return 1

    @staticmethod
    def nvmlDeviceGetMemoryInfo(handle):
        raise _FakeNVMLError("Not Supported")

    @staticmethod
    def nvmlDeviceGetUtilizationRates(handle):
        import types
        return types.SimpleNamespace(gpu=3, memory=7)

    @staticmethod
    def nvmlDeviceGetComputeRunningProcesses(handle):
        import types
        return [types.SimpleNamespace(pid=42, usedGpuMemory=1 << 20)]


def test_nvml_series_gb10_fallback(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "pynvml", _FakeNVML)
    series = hwprobe.nvml_memory_series(0, samples=2, interval_s=0)
    assert series["used_bytes"] == [None, None]
    assert series["mem_util_pct"] == [7, 7]
    assert series["per_process"][0] == {42: 1 << 20}


def test_pynvml_residue_cell_gb10_fallback(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "pynvml", _FakeNVML)
    args = gpu_memory.build_parser().parse_args(["--arm", "allocator-residue"])
    rec = gpu_memory.pynvml_residue_cell(args)
    assert rec["mem_util_pct_delta"] == 0
    assert "unified memory" in rec["fallback"]
    assert rec["per_process_attribution"] is True
    assert "used_bytes_delta" not in rec
