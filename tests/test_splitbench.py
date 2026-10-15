"""splitbench tests.

Pure-python layer (torch-free, runs everywhere): SplitPlan parse and
validation matrix, CaptureWriter with a fake saver (schema validation
through the real surface providers, crash-safety), BenchConfig from_json +
validation drift, the CLI --help subprocess, and self_test().

Torch layer (self-skips without torch/transformers, marked slow): a tiny
randomly-initialized GPT2LMHeadModel on CPU checks the seam-capture
protocol end to end — finetune fwd/bwd capture counts and shapes, exact
split math, inference prefill/decode positions, gaussian-defense
semantics, and the run artifact + journal + evidence chain.
"""

import json
import math
import os
import subprocess
import sys

import pytest

from attacker import captures, splitbench
from attacker.splitbench import (ARCH_REGISTRY, BenchConfig, CaptureWriter,
                                 SplitPlan)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_splitplan_two_slice():
    plan = SplitPlan.parse("6", n_layers=12)
    assert plan.boundaries == (6,)
    assert plan.n_slices == 2
    assert plan.slice_ranges() == [(0, 6), (6, 12)]
    assert "6" in plan.describe()


def test_splitplan_three_slice():
    plan = SplitPlan.parse("4,8", n_layers=12)
    assert plan.boundaries == (4, 8)
    assert plan.n_slices == 3
    assert plan.slice_ranges() == [(0, 4), (4, 8), (8, 12)]
    desc = plan.describe()
    assert "head[0:4]" in desc and "tail[8:12]" in desc


def test_splitplan_whitespace_tolerant():
    assert SplitPlan.parse(" 4 , 8 ").boundaries == (4, 8)


def test_splitplan_unbound_describes_and_late_binds():
    plan = SplitPlan.parse("6")
    assert plan.n_slices == 2 and plan.n_layers is None
    assert "unbound" in plan.describe()
    with pytest.raises(ValueError, match="not bound"):
        plan.slice_ranges()
    assert plan.slice_ranges(12) == [(0, 6), (6, 12)]


@pytest.mark.parametrize("spec", [
    "", " ", "x", "4,x", "4,,8", ",4", "4,", "4,4", "8,4", "0", "-1",
    "1.5", "0,6",
])
def test_splitplan_malformed_specs_raise(spec):
    with pytest.raises(ValueError):
        SplitPlan.parse(spec)


@pytest.mark.parametrize("spec", ["12", "13", "6,12"])
def test_splitplan_out_of_range_boundaries_raise(spec):
    with pytest.raises(ValueError, match="out of range"):
        SplitPlan.parse(spec, n_layers=12)


def test_splitplan_needs_at_least_two_layers():
    with pytest.raises(ValueError):
        SplitPlan.parse("1", n_layers=1)


def test_splitplan_late_bind_range_check():
    assert SplitPlan.parse("11").slice_ranges(12) == [(0, 11), (11, 12)]
    with pytest.raises(ValueError, match="out of range"):
        SplitPlan.parse("12").slice_ranges(12)


def test_arch_registry():
    assert ARCH_REGISTRY["gpt2"] is splitbench.Gpt2Split
    with pytest.raises(ValueError) as ei:
        splitbench.get_arch_splitter("no-such-arch")
    assert "no-such-arch" in str(ei.value) and "ARCH_REGISTRY" in str(ei.value)


def fake_saver(path, tensor):
    with open(path, "wb") as f:
        f.write(b"FAKE:" + repr(tensor).encode())


def test_writer_training_pairs_scan_and_validate(tmp_path):
    d = str(tmp_path)
    with CaptureWriter(d, "training", "sess-t", saver=fake_saver) as w:
        w.write_training(("h", 0), mb_id=0, phase="fwd", step=0, epoch=0)
        w.write_training(("g", 0), mb_id=0, phase="bwd", step=0, epoch=0)
        w.write_training(("h", 1), mb_id=0, phase="fwd", step=1, epoch=0)
        assert w.count == 3
    assert os.path.isfile(os.path.join(d, "wire_00000.pt"))
    assert os.path.isfile(os.path.join(d, "wire_00000.json"))
    assert os.path.isfile(os.path.join(d, "wire_00002.json"))
    recs = captures.scan_captures(d, "training")
    assert [(m["phase"], m["step"], m["mb_id"], m["epoch"], m["session_id"])
            for m, _ in recs] == [("fwd", 0, 0, 0, "sess-t"),
                                  ("bwd", 0, 0, 0, "sess-t"),
                                  ("fwd", 1, 0, 0, "sess-t")]
    for meta, _ in recs:
        assert captures.validate_sidecar(meta, "training") is meta
    with open(recs[0][1], "rb") as f:
        assert f.read().startswith(b"FAKE:")


def test_writer_inference_pairs_scan_and_validate(tmp_path):
    d = str(tmp_path)
    with CaptureWriter(d, "inference", "sess-i", saver=fake_saver) as w:
        w.write_inference(("h",), request_seq=0, phase="prefill",
                          position=0, epoch=0)
        w.write_inference(("t",), request_seq=0, phase="decode",
                          position=1, epoch=0)
    recs = captures.scan_captures(d, "inference")
    assert [(m["phase"], m["position"], m["request_seq"]) for m, _ in recs] \
        == [("prefill", 0, 0), ("decode", 1, 0)]
    for meta, _ in recs:
        assert captures.validate_sidecar(meta, "inference") is meta


def test_writer_is_crash_safe(tmp_path):
    with CaptureWriter(str(tmp_path), "training", "s",
                       saver=fake_saver) as w:
        w.write_training(("h",), mb_id=0, phase="fwd", step=0, epoch=0)
    assert not any(".tmp" in name for name in os.listdir(tmp_path))


def test_writer_provider_validation_fires_on_drift(tmp_path):
    w = CaptureWriter(str(tmp_path), "training", "s", saver=fake_saver)
    with pytest.raises(ValueError):
        w._write(("x",), {"session_id": "s", "mb_id": 0})
    assert w.count == 0
    assert os.listdir(tmp_path) == []


def test_writer_mode_mismatch_and_close_raise(tmp_path):
    w = CaptureWriter(str(tmp_path), "inference", "s", saver=fake_saver)
    with pytest.raises(ValueError, match="training"):
        w.write_training(("x",), mb_id=0, phase="fwd", step=0, epoch=0)
    w.close()
    with pytest.raises(ValueError, match="closed"):
        w.write_inference(("x",), request_seq=0, phase="decode",
                          position=1, epoch=0)
    with pytest.raises(ValueError, match="unknown capture mode"):
        CaptureWriter(str(tmp_path), "bogus", "s", saver=fake_saver)


def test_writer_epoch_none_accepted(tmp_path):
    d = str(tmp_path)
    with CaptureWriter(d, "training", "s", saver=fake_saver) as w:
        w.write_training(("h",), mb_id=0, phase="fwd", step=0, epoch=None)
    recs = captures.scan_captures(d, "training")
    assert recs[0][0]["epoch"] is None


def test_benchconfig_defaults_validate():
    assert BenchConfig().validate().mode == "finetune"


def test_benchconfig_from_json_roundtrip(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"model": "gpt2", "split": "4,8",
                             "mode": "inference", "steps": 10,
                             "sigma": 0.5, "defense": "gaussian"}))
    cfg = BenchConfig.from_json(str(p))
    assert cfg.split == "4,8" and cfg.mode == "inference"
    assert cfg.steps == 10 and cfg.sigma == 0.5


@pytest.mark.parametrize("raw", [
    {"modle": "gpt2"},
    {"split": "4", "splits": 2},
])
def test_benchconfig_from_json_unknown_keys_raise(tmp_path, raw):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unknown"):
        BenchConfig.from_json(str(p))


def test_benchconfig_from_json_bad_file(tmp_path):
    with pytest.raises(ValueError, match="cannot read config"):
        BenchConfig.from_json(str(tmp_path / "nope.json"))
    p = tmp_path / "cfg.json"
    p.write_text("[1, 2]")
    with pytest.raises(ValueError, match="JSON object"):
        BenchConfig.from_json(str(p))
    p.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        BenchConfig.from_json(str(p))


@pytest.mark.parametrize("kw", [
    {"mode": "train"}, {"mode": "serve"},
    {"defense": "dp"}, {"device": "tpu"}, {"dtype": "fp16"},
    {"steps": 0}, {"steps": -3}, {"steps": "10"},
    {"seq_len": 1}, {"batch_size": 0}, {"capture_every": 0},
    {"lr": 0.0}, {"lr": -1.0}, {"sigma": -0.1},
    {"lora_r": -1}, {"lora_alpha": 0.0},
    {"split": "0"}, {"split": "4,4"}, {"split": ""},
    {"model": ""}, {"capture_dir": ""}, {"corpus_file": 3},
])
def test_benchconfig_validation_drift_raises(kw):
    with pytest.raises(ValueError):
        BenchConfig(**kw).validate()


def test_benchconfig_gaussian_sigma_zero_is_legal():
    assert BenchConfig(defense="gaussian", sigma=0.0).validate()


def test_cli_help_torchless():
    r = subprocess.run([sys.executable, "-m", "attacker.splitbench",
                        "--help"], cwd=ROOT, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    assert "splitbench" in r.stdout and "--capture-every" in r.stdout


def test_cli_config_drift_is_a_clean_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"mode": "train"}))
    r = subprocess.run([sys.executable, "-m", "attacker.splitbench",
                        "--config", str(p)], cwd=ROOT, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode != 0
    assert "unknown mode" in (r.stderr + r.stdout)


def test_self_test_returns_zero(capsys):
    assert splitbench.self_test() == 0
    assert "SELF-TEST PASSED" in capsys.readouterr().out


def test_run_creates_missing_output_parent_dir(tmp_path, monkeypatch):
    """run() must create the artifact's parent dir before EvidenceLogger
    (or the artifact writer) touches it — regression for the toula smoke
    FileNotFoundError on a fresh capture dir."""
    seen = {}

    def fake_run(cfg, evidence=None):
        seen["dir_exists"] = os.path.isdir(os.path.dirname(cfg.output))
        return 0

    monkeypatch.setattr(splitbench, "_run", fake_run)
    out = tmp_path / "fresh" / "nested" / "run.json"
    cfg = splitbench.BenchConfig(output=str(out), capture_dir=str(tmp_path / "cap"))
    assert splitbench.run(cfg, evidence=True) == 0
    assert seen["dir_exists"] is True


@pytest.fixture
def tiny():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=4, n_embd=32, n_head=4, vocab_size=99,
                     n_positions=64, bos_token_id=0, eos_token_id=98,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return torch, model


def _split(model, spec="2"):
    plan = SplitPlan.parse(spec, n_layers=model.config.n_layer)
    return splitbench.build_split(model, plan)


def _batches(torch, n=3, batch=2, seq=8):
    g = torch.Generator().manual_seed(99)
    return [torch.randint(0, 99, (batch, seq), generator=g)
            for _ in range(n)]


@pytest.mark.slow
def test_torch_finetune_seam_captures(tiny, tmp_path):
    torch, model = tiny
    split = _split(model)
    writer = CaptureWriter(str(tmp_path), "training", "t0")
    cfg = BenchConfig(steps=3, seq_len=8, batch_size=2, capture_every=1,
                      lr=1e-4)
    stats = splitbench.finetune(split, cfg, _batches(torch), writer)
    assert writer.count == 6
    recs = captures.scan_captures(str(tmp_path), "training")
    assert [(m["phase"], m["step"], m["epoch"]) for m, _ in recs] == [
        ("fwd", 0, 0), ("bwd", 0, 0),
        ("fwd", 1, 0), ("bwd", 1, 0),
        ("fwd", 2, 0), ("bwd", 2, 0)]
    fwd = captures.load_tensor(recs[0][1])
    bwd = captures.load_tensor(recs[1][1])
    assert fwd.shape == bwd.shape == (2, 8, 32)
    assert math.isfinite(stats["final_loss"])


@pytest.mark.slow
def test_torch_finetune_saves_canonical(tiny, tmp_path):
    torch, model = tiny
    split = _split(model)
    writer = CaptureWriter(str(tmp_path), "training", "t0")
    cfg = BenchConfig(steps=3, seq_len=8, batch_size=2, capture_every=1,
                      lr=1e-4, save_canonical=True)
    splitbench.finetune(split, cfg, _batches(torch), writer)
    canonical = torch.load(tmp_path / "canonical.pt", weights_only=True)
    assert canonical.shape == (3 * 2, 8)
    assert canonical.dtype == torch.long

    d2 = tmp_path / "every2"
    writer2 = CaptureWriter(str(d2), "training", "t1")
    cfg2 = BenchConfig(steps=3, seq_len=8, batch_size=2, capture_every=2,
                       lr=1e-4)
    splitbench.finetune(_split(model), cfg2, _batches(torch), writer2)
    steps = [m["step"] for m, _ in captures.scan_captures(str(d2),
                                                          "training")
             if m["phase"] == "fwd"]
    assert steps == [0, 2]


@pytest.mark.slow
def test_torch_split_is_exact(tiny):
    torch, model = tiny
    split = _split(model, "2")
    ids = torch.randint(0, 99, (2, 8))
    with torch.no_grad():
        ref = model(ids).logits
        out = split.rest_forward(split.head_forward(ids))
    assert torch.allclose(ref, out, atol=1e-4)


@pytest.mark.slow
def test_torch_inference_seam_captures(tiny, tmp_path):
    torch, model = tiny
    split = _split(model)
    writer = CaptureWriter(str(tmp_path), "inference", "s0")
    prompts = [torch.randint(0, 99, (1, 6),
                             generator=torch.Generator().manual_seed(5))]
    stats = splitbench.infer(split, BenchConfig(mode="inference", steps=4),
                             prompts, writer)
    assert stats["requests"] == 1
    recs = captures.scan_captures(str(tmp_path), "inference")
    assert [(m["phase"], m["position"], m["request_seq"]) for m, _ in recs] \
        == [("prefill", 0, 0), ("decode", 1, 0), ("decode", 2, 0),
            ("decode", 3, 0), ("decode", 4, 0)]
    assert captures.load_tensor(recs[0][1]).shape == (1, 6, 32)
    assert all(captures.load_tensor(pt).shape == (1, 1, 32)
               for _, pt in recs[1:])


@pytest.mark.slow
def test_torch_gaussian_defense_semantics(tiny, tmp_path):
    torch, _ = tiny

    def first_fwd(defense, sigma, sub):
        torch.manual_seed(0)
        from transformers import GPT2Config, GPT2LMHeadModel
        cfg_m = GPT2Config(n_layer=4, n_embd=32, n_head=4, vocab_size=99,
                           n_positions=64, bos_token_id=0, eos_token_id=98,
                           resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
        split = _split(GPT2LMHeadModel(cfg_m).eval())
        writer = CaptureWriter(str(tmp_path / sub), "training", "s")
        cfg = BenchConfig(steps=1, seq_len=8, batch_size=2, defense=defense,
                          sigma=sigma, seed=7)
        splitbench.finetune(split, cfg, _batches(torch, n=1), writer)
        return captures.load_tensor(str(tmp_path / sub / "wire_00000.pt"))

    none = first_fwd("none", 0.0, "none")
    noisy = first_fwd("gaussian", 0.5, "noisy")
    zero = first_fwd("gaussian", 0.0, "zero")
    assert not torch.equal(none, noisy)
    assert torch.equal(none, zero)


@pytest.mark.slow
def test_torch_run_artifact_journal_and_evidence(tiny, tmp_path):
    torch, model = tiny
    split = _split(model)
    out = tmp_path / "run.json"
    cfg = BenchConfig(split="2", steps=2, seq_len=8, batch_size=2, lr=1e-4,
                      capture_dir=str(tmp_path / "caps"), output=str(out),
                      seed=11)
    writer = CaptureWriter(cfg.capture_dir, "training", "art")
    from attacker.evidence import EvidenceLogger, verify_evidence
    with EvidenceLogger(cfg.output, node_role="splitbench/test") as ev:
        stats = splitbench.execute(cfg, split, writer, _batches(torch, n=2),
                                   evidence=ev)
    assert stats["steps_run"] == 2 and writer.count == 4
    art = json.loads(out.read_text())
    assert art["schema"] == "llm_attacker.splitbench.v1"
    assert art["config"]["split"] == "2"
    rec = art["results"][0]
    assert rec["captures_written"] == 4 and rec["steps_run"] == 2
    assert math.isfinite(rec["final_loss"])
    assert art["summary"][0]["defense"] == "none"
    lines = (tmp_path / "run.json.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["step"] == 0
    ok, problems = verify_evidence(str(out) + ".evidence.json")
    assert ok, problems
