"""quant seam defense + return-path hook + trainable-range tests.

Torch-free layer (runs everywhere): BenchConfig validation matrix for the
new knobs, _parse_trainable_range, and _defense_config_record.

Torch layer (slow, self-skips without torch/transformers; tiny random
GPT-2 2L/32/99 on CPU, the test_defense_pack.py fixture pattern):
(a) _quantize_absmax hand-computed math (the validation_controls bits_N
    rule: signed symmetric per-sequence absmax, limit = 2^(bits-1)-1),
(b) _apply_grad_defense semantics: identity at defaults, global-L2 clip,
    quant grid, frozen clip -> noise -> quantize composition order,
(c) finetune integration: quant changes the fwd wire vs "none" at the
    same seed/data with finite loss; the bwd capture is the POST-hook
    gradient (on the per-row absmax grid, differs from the none run,
    reproducible across identical runs),
(d) trainable_range freezes everything except blocks [START, END).
"""

import json
import math

import pytest

from attacker import captures, splitbench


@pytest.mark.parametrize("kw", [
    {"defense": "quant"},
    {"defense": "quant", "quant_bits": 2},
    {"defense": "quant", "quant_bits": 16, "quant_grad": True},
    {"defense": "quant", "quant_bits": 4, "quant_grad": True,
     "grad_clip": 1.0, "grad_noise": 0.1},
    {"grad_clip": 1.0},
    {"grad_noise": 0.1},
    {"defense": "gaussian", "sigma": 0.3, "grad_clip": 0.5,
     "grad_noise": 0.1},
    {"trainable_range": "0,12"},
    {"trainable_range": "2,6"},
])
def test_quant_config_valid(kw):
    cfg = splitbench.BenchConfig(**kw)
    assert cfg.validate() is cfg


@pytest.mark.parametrize("kw", [
    {"defense": "quant", "quant_bits": 1},
    {"defense": "quant", "quant_bits": 17},
    {"quant_bits": 2.5},
    {"quant_bits": True},
    {"quant_grad": "yes"},
    {"defense": "none", "quant_grad": True},
    {"defense": "gaussian", "quant_grad": True},
    {"grad_clip": -0.1},
    {"grad_noise": -0.1},
    {"grad_noise": "x"},
    {"trainable_range": ""},
    {"trainable_range": "6"},
    {"trainable_range": "2,6,8"},
    {"trainable_range": "6,2"},
    {"trainable_range": "2,2"},
    {"trainable_range": "-1,2"},
    {"trainable_range": "a,b"},
    {"trainable_range": "2,6", "lora_r": 4},
    {"trainable_range": "2,6", "mode": "inference"},
])
def test_quant_config_drift_raises(kw):
    with pytest.raises(ValueError):
        splitbench.BenchConfig(**kw).validate()


def test_quant_knob_defaults():
    cfg = splitbench.BenchConfig()
    assert (cfg.quant_bits, cfg.quant_grad, cfg.grad_clip, cfg.grad_noise,
            cfg.trainable_range) == (8, False, 0.0, 0.0, None)


def test_parse_trainable_range():
    assert splitbench._parse_trainable_range("2,6") == (2, 6)
    assert splitbench._parse_trainable_range(" 0 , 12 ") == (0, 12)
    assert splitbench._parse_trainable_range("10,12", n_blocks=12) == (10, 12)
    with pytest.raises(ValueError, match="exceeds"):
        splitbench._parse_trainable_range("2,13", n_blocks=12)


def test_defense_config_record_quant():
    assert splitbench._defense_config_record(
        splitbench.BenchConfig(defense="quant")) == \
        {"name": "quant", "quant_bits": 8, "quant_grad": False}
    assert splitbench._defense_config_record(
        splitbench.BenchConfig(defense="quant", quant_bits=4,
                               quant_grad=True, grad_clip=1.0,
                               grad_noise=0.1, trainable_range="2,6")) == \
        {"name": "quant", "quant_bits": 4, "quant_grad": True,
         "grad_clip": 1.0, "grad_noise": 0.1, "trainable_range": "2,6"}
    assert splitbench._defense_config_record(
        splitbench.BenchConfig(defense="none")) == {"name": "none"}
    assert splitbench._defense_config_record(
        splitbench.BenchConfig(defense="gaussian", sigma=0.3)) == \
        {"name": "gaussian", "sigma": 0.3}
    assert splitbench._defense_config_record(
        splitbench.BenchConfig(defense="gaussian", grad_clip=1.0)) == \
        {"name": "gaussian", "sigma": 0.0, "grad_clip": 1.0}


def _tiny_model(torch, seed=0):
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(seed)
    cfg = GPT2Config(n_layer=2, n_embd=32, n_head=4, vocab_size=99,
                     n_positions=64, bos_token_id=0, eos_token_id=98,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    return GPT2LMHeadModel(cfg).eval()


@pytest.fixture
def tiny():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return torch


def _split(model):
    plan = splitbench.SplitPlan.parse("1", n_layers=model.config.n_layer)
    return splitbench.build_split(model, plan)


def _batches(torch, n=3, batch=2, seq=8):
    g = torch.Generator().manual_seed(99)
    return [torch.randint(0, 99, (batch, seq), generator=g)
            for _ in range(n)]


def _run(torch, tmp_path, name, defense, steps=3, **kw):
    """3-step split finetune of a fresh tiny model; returns (stats,
    per-step losses). Identical weights/data/seed across calls."""
    model = _tiny_model(torch)
    split = _split(model)
    writer = splitbench.CaptureWriter(str(tmp_path / name), "training",
                                      name)
    cfg = splitbench.BenchConfig(steps=steps, seq_len=8, batch_size=2,
                                 lr=1e-4, defense=defense, seed=7, **kw)
    losses = []
    stats = splitbench.finetune(split, cfg, _batches(torch, n=steps),
                                writer,
                                on_step=lambda r: losses.append(r["loss"]))
    return stats, losses


@pytest.mark.slow
def test_torch_quantize_absmax_hand_computed(tiny):
    torch = tiny
    t = torch.tensor([[[1.0, -2.0, 0.9, -0.9]]])
    out = splitbench._quantize_absmax(t, 2)
    assert torch.equal(out, torch.tensor([[[0.0, -2.0, 0.0, -0.0]]]))
    out3 = splitbench._quantize_absmax(t, 3)
    exp3 = torch.tensor([[[2.0, -3.0, 1.0, -1.0]]]) * (2.0 / 3.0)
    assert torch.allclose(out3, exp3, atol=1e-6)
    t2 = torch.tensor([[[1.0, -2.0]], [[4.0, -8.0]]])
    o2 = splitbench._quantize_absmax(t2, 2)
    assert torch.equal(o2[0], torch.tensor([[0.0, -2.0]]))
    assert torch.equal(o2[1], torch.tensor([[0.0, -8.0]]))
    z = torch.zeros(1, 2, 4)
    assert torch.equal(splitbench._quantize_absmax(z, 4), z)
    with pytest.raises(ValueError, match="2..16"):
        splitbench._quantize_absmax(t, 17)


@pytest.mark.slow
def test_torch_grad_hook_identity_at_defaults(tiny):
    torch = tiny
    g = torch.randn(2, 3, 32, generator=torch.Generator().manual_seed(5))
    out = splitbench._apply_grad_defense(g, splitbench.BenchConfig())
    assert out is g


@pytest.mark.slow
def test_torch_grad_hook_clip_semantics(tiny):
    torch = tiny
    g = torch.randn(2, 3, 32, generator=torch.Generator().manual_seed(5))
    cfg = splitbench.BenchConfig(grad_clip=0.5)
    out = splitbench._apply_grad_defense(g.clone(), cfg)
    assert math.isclose(out.norm().item(), 0.5, rel_tol=1e-5)
    cfg2 = splitbench.BenchConfig(grad_clip=1e6)
    g2 = g.clone()
    assert torch.equal(splitbench._apply_grad_defense(g2, cfg2), g2)
    z = torch.zeros(1, 2, 4)
    assert torch.equal(
        splitbench._apply_grad_defense(
            z, splitbench.BenchConfig(grad_clip=1.0)), z)


@pytest.mark.slow
def test_torch_grad_hook_composition_order(tiny):
    torch = tiny
    g = torch.randn(1, 4, 32, generator=torch.Generator().manual_seed(11))
    cfg = splitbench.BenchConfig(defense="quant", quant_bits=3,
                                 quant_grad=True, grad_clip=0.25)
    out = splitbench._apply_grad_defense(g.clone(), cfg)
    clipped = g * min(1.0, 0.25 / g.norm().item())
    ref = splitbench._quantize_absmax(clipped, 3)
    assert torch.allclose(out, ref, atol=1e-6)
    limit = 2 ** (3 - 1) - 1
    step = out.abs().amax(dim=(-2, -1), keepdim=True) / limit
    levels = out / step.clamp_min(1e-12)
    assert (levels - levels.round()).abs().max().item() < 1e-4
    assert levels.abs().max().item() == pytest.approx(limit, abs=1e-3)


@pytest.mark.slow
def test_torch_quant_changes_wire_and_keeps_loss_finite(tiny, tmp_path):
    torch = tiny
    _, none_losses = _run(torch, tmp_path, "none", "none")
    _, losses = _run(torch, tmp_path, "quant", "quant", quant_bits=8)
    none_fwd = captures.load_tensor(
        str(tmp_path / "none" / "wire_00000.pt"))
    fwd = captures.load_tensor(str(tmp_path / "quant" / "wire_00000.pt"))
    assert fwd.shape == none_fwd.shape
    assert not torch.equal(none_fwd, fwd)
    assert len(losses) == 3
    assert all(math.isfinite(x) for x in losses + none_losses)


def _bwd_capture(tmp_path, name):
    """The step-0 bwd capture tensor of a run dir (located via sidecars,
    not file numbering)."""
    recs = captures.scan_captures(str(tmp_path / name), "training")
    bwd = [p for m, p in recs if m["phase"] == "bwd" and m["step"] == 0]
    assert len(bwd) == 1
    return captures.load_tensor(bwd[0])


@pytest.mark.slow
def test_torch_d1_cell_bwd_capture_is_post_hook(tiny, tmp_path):
    torch = tiny
    d1 = {"quant_bits": 4, "quant_grad": True, "grad_clip": 1.0,
          "grad_noise": 0.1}
    _run(torch, tmp_path, "none", "none", steps=1)
    _run(torch, tmp_path, "d1", "quant", steps=1, **d1)
    _run(torch, tmp_path, "d1again", "quant", steps=1, **d1)
    none_bwd = _bwd_capture(tmp_path, "none")
    d1_bwd = _bwd_capture(tmp_path, "d1")
    assert not torch.equal(none_bwd, d1_bwd)
    assert torch.equal(d1_bwd, _bwd_capture(tmp_path, "d1again"))
    limit = 2 ** (4 - 1) - 1
    step = d1_bwd.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    levels = d1_bwd / (step / limit)
    assert (levels - levels.round()).abs().max().item() < 0.02
    assert levels.abs().max().item() == pytest.approx(limit, abs=0.05)


@pytest.mark.slow
def test_torch_grad_hook_off_is_bit_identical_bwd(tiny, tmp_path):
    torch = tiny
    _run(torch, tmp_path, "a", "none", steps=1)
    _run(torch, tmp_path, "b", "none", steps=1, grad_clip=0.0,
         grad_noise=0.0)
    assert torch.equal(_bwd_capture(tmp_path, "a"), _bwd_capture(tmp_path, "b"))


@pytest.mark.slow
def test_torch_quant_run_artifact_records_defense_config(tiny, tmp_path):
    torch = tiny
    model = _tiny_model(torch)
    split = _split(model)
    out = tmp_path / "run.json"
    cfg = splitbench.BenchConfig(split="1", steps=2, seq_len=8,
                                 batch_size=2, lr=1e-4, defense="quant",
                                 quant_bits=4, quant_grad=True,
                                 grad_clip=1.0, grad_noise=0.1,
                                 capture_dir=str(tmp_path / "caps"),
                                 output=str(out), seed=11)
    writer = splitbench.CaptureWriter(cfg.capture_dir, "training", "art")
    splitbench.execute(cfg, split, writer, _batches(torch, n=2))
    art = json.loads(out.read_text())
    rec = art["results"][0]
    assert rec["defense"] == "quant"
    assert rec["defense_config"] == {"name": "quant", "quant_bits": 4,
                                     "quant_grad": True, "grad_clip": 1.0,
                                     "grad_noise": 0.1}
    assert art["config"]["quant_bits"] == 4


@pytest.mark.slow
def test_torch_trainable_range_freezes_outside_segment(tiny, tmp_path):
    torch = tiny
    model = _tiny_model(torch)
    split = _split(model)
    writer = splitbench.CaptureWriter(str(tmp_path / "tr"), "training",
                                      "tr")
    cfg = splitbench.BenchConfig(steps=3, seq_len=8, batch_size=2,
                                 lr=1e-3, defense="none", seed=7,
                                 trainable_range="1,2")
    splitbench.finetune(split, cfg, _batches(torch, n=3), writer)
    ref = _tiny_model(torch)
    tf = model.transformer
    assert torch.equal(tf.wte.weight, ref.transformer.wte.weight)
    for p, q in zip(tf.h[0].parameters(), ref.transformer.h[0].parameters()):
        assert torch.equal(p, q)
    assert torch.equal(tf.ln_f.weight, ref.transformer.ln_f.weight)
    changed = any(not torch.equal(p, q) for p, q in
                  zip(tf.h[1].parameters(),
                      ref.transformer.h[1].parameters()))
    assert changed


@pytest.mark.slow
def test_torch_trainable_range_bounds_checked_at_run(tiny, tmp_path):
    torch = tiny
    model = _tiny_model(torch)
    split = _split(model)
    writer = splitbench.CaptureWriter(str(tmp_path / "oob"), "training",
                                      "oob")
    cfg = splitbench.BenchConfig(steps=1, seq_len=8, batch_size=2,
                                 lr=1e-4, defense="none", seed=7,
                                 trainable_range="1,3")
    with pytest.raises(ValueError, match="exceeds"):
        splitbench.finetune(split, cfg, _batches(torch, n=1), writer)
