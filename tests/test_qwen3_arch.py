"""splitbench Qwen3 arch tests + the bf16 end-to-end dtype check.

Torch-free layer (runs everywhere): ARCH_REGISTRY exposes "qwen3" (the
Mistral family splitter — verified against transformers 5.15.0:
Qwen3DecoderLayer.forward has the identical call contract, qk-norm lives
inside the decoder layer), arch dispatch on model_type "qwen3", and the
CLI --help stays torch-less.

Torch layer (self-skips without torch/transformers, marked slow): a tiny
randomly-initialized Qwen3 (hidden 32, heads 4, kv_heads 2, layers 4,
vocab 99) on CPU checks head->rest split exactness against the unsplit
model (sdpa and eager+padding, allclose 1e-4), finetune seam captures +
canonical.pt, and inference prefill/decode captures.

The bf16 end-to-end check (tiny GPT-2, dtype bf16: finetune runs, wire
captures land fp16 on disk, loss finite) lives in this file because
tests/test_splitbench.py is owned elsewhere — it exercises the splitbench
dtype path, not Qwen3.
"""

import math
import os
import subprocess
import sys

import pytest

from attacker import captures, splitbench
from attacker.splitbench import (ARCH_REGISTRY, BenchConfig, CaptureWriter,
                                 MistralSplit, SplitPlan)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_registry_exposes_qwen3_key():
    assert ARCH_REGISTRY["qwen3"] is MistralSplit


def test_arch_dispatch_error_lists_qwen3():
    with pytest.raises(ValueError) as ei:
        splitbench.get_arch_splitter("qwen3-ish")
    msg = str(ei.value)
    assert "qwen3-ish" in msg and "ARCH_REGISTRY" in msg
    assert "qwen3" in msg


def test_build_split_dispatches_on_qwen3_model_type():
    class FakeCfg:
        model_type = "qwen3"
        n_layer = None
        num_hidden_layers = None

    class FakePlan:
        n_layers = None

        def slice_ranges(self, n):
            return [(0, 1), (1, 2)]

    import types
    core = types.SimpleNamespace(
        embed_tokens=types.SimpleNamespace(weight=None), layers=[0, 0],
        norm="norm", rotary_emb="rot")
    model = types.SimpleNamespace(config=FakeCfg(), model=core,
                                  lm_head="head")
    split = splitbench.build_split(model, FakePlan())
    assert isinstance(split, MistralSplit)
    assert split.ln_f == "norm" and split.norm == "norm"


def test_cli_help_torchless():
    r = subprocess.run([sys.executable, "-m", "attacker.splitbench",
                        "--help"], cwd=ROOT, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    assert "--capture-every" in r.stdout


HIDDEN = 32


@pytest.fixture
def tiny():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(hidden_size=HIDDEN, intermediate_size=64,
                      num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=99,
                      max_position_embeddings=64)
    model = Qwen3ForCausalLM(cfg)
    model.eval()
    return torch, model


def _split(model, spec="2"):
    plan = SplitPlan.parse(spec, n_layers=model.config.num_hidden_layers)
    return splitbench.build_split(model, plan)


def _batches(torch, n=2, batch=2, seq=8):
    g = torch.Generator().manual_seed(99)
    return [torch.randint(0, 99, (batch, seq), generator=g)
            for _ in range(n)]


@pytest.mark.slow
def test_torch_qwen3_split_is_exact(tiny):
    torch, model = tiny
    assert model.config.model_type == "qwen3"
    split = _split(model, "2")
    ids = torch.randint(0, 99, (2, 8))
    with torch.no_grad():
        ref = model(ids).logits
        out = split.rest_forward(split.head_forward(ids))
    assert torch.allclose(ref, out, atol=1e-4)


@pytest.mark.slow
def test_torch_qwen3_split_is_exact_eager_with_padding(tiny):
    torch, model = tiny
    model.config._attn_implementation = "eager"
    split = _split(model, "2")
    ids = torch.randint(0, 99, (2, 8))
    am = torch.ones(2, 8, dtype=torch.long)
    am[1, 6:] = 0
    with torch.no_grad():
        ref = model(ids, attention_mask=am).logits
        out = split.rest_forward(split.head_forward(ids, am), am)
    assert torch.allclose(ref, out, atol=1e-4)


@pytest.mark.slow
def test_torch_qwen3_finetune_captures_and_canonical(tiny, tmp_path):
    torch, model = tiny
    split = _split(model)
    writer = CaptureWriter(str(tmp_path), "training", "q0")
    cfg = BenchConfig(steps=2, seq_len=8, batch_size=2, capture_every=1,
                      lr=1e-4, save_canonical=True)
    stats = splitbench.finetune(split, cfg, _batches(torch), writer)
    writer.close()
    assert writer.count == 4
    recs = captures.scan_captures(str(tmp_path), "training")
    assert [(m["phase"], m["step"], m["epoch"]) for m, _ in recs] == [
        ("fwd", 0, 0), ("bwd", 0, 0), ("fwd", 1, 0), ("bwd", 1, 0)]
    assert captures.load_tensor(recs[0][1]).shape == (2, 8, HIDDEN)
    canonical = torch.load(tmp_path / "canonical.pt", weights_only=True)
    assert canonical.shape == (2 * 2, 8) and canonical.dtype == torch.long
    assert math.isfinite(stats["final_loss"])


@pytest.mark.slow
def test_torch_qwen3_inference_seam_captures(tiny, tmp_path):
    torch, model = tiny
    split = _split(model)
    writer = CaptureWriter(str(tmp_path), "inference", "qi")
    prompts = [torch.randint(0, 99, (1, 6),
                             generator=torch.Generator().manual_seed(5))]
    stats = splitbench.infer(split, BenchConfig(mode="inference", steps=2),
                             prompts, writer)
    writer.close()
    assert stats["requests"] == 1
    recs = captures.scan_captures(str(tmp_path), "inference")
    assert [(m["phase"], m["position"], m["request_seq"]) for m, _ in recs] \
        == [("prefill", 0, 0), ("decode", 1, 0), ("decode", 2, 0)]
    assert captures.load_tensor(recs[0][1]).shape == (1, 6, HIDDEN)
    assert all(captures.load_tensor(pt).shape == (1, 1, HIDDEN)
               for _, pt in recs[1:])


@pytest.mark.slow
def test_torch_bf16_finetune_end_to_end(tmp_path):
    """dtype bf16: finetune runs, captures land fp16 on disk (the wire
    convention — _torch_saver casts), canonical stays long, loss finite."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=4, n_embd=32, n_head=4, vocab_size=99,
                     n_positions=64, bos_token_id=0, eos_token_id=98,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = GPT2LMHeadModel(cfg).to(torch.bfloat16)
    model.train()
    plan = SplitPlan.parse("2", n_layers=model.config.n_layer)
    split = splitbench.build_split(model, plan)
    writer = CaptureWriter(str(tmp_path), "training", "bf16")
    bench = BenchConfig(steps=2, seq_len=8, batch_size=2, capture_every=1,
                        lr=1e-4, dtype="bf16", save_canonical=True)
    stats = splitbench.finetune(split, bench, _batches(torch), writer)
    writer.close()
    assert writer.count == 4
    assert math.isfinite(stats["final_loss"])
    assert math.isfinite(stats["mean_loss"])
    recs = captures.scan_captures(str(tmp_path), "training")
    for meta, pt in recs:
        on_disk = torch.load(pt, weights_only=True)
        assert on_disk.dtype == torch.float16
        assert on_disk.shape == (2, 8, 32)
        assert math.isfinite(on_disk.float().abs().max().item())
    canonical = torch.load(tmp_path / "canonical.pt", weights_only=True)
    assert canonical.dtype == torch.long and canonical.shape == (2 * 2, 8)
