"""splitbench Mistral/Llama-family arch + body-grad capture tests.

Torch-free layer (runs everywhere): ARCH_REGISTRY exposes "mistral" and
"llama" (shared family splitter), the arch-dispatch error names how to
register, the CLI --help stays torch-less (including --body-grad-capture),
and the body_grad_capture config plumbing validates torch-free.

Torch layer (self-skips without torch/transformers, marked slow): a tiny
randomly-initialized Mistral (hidden 32, heads 4, kv_heads 2, layers 4,
vocab 99) on CPU checks head->rest split exactness against the unsplit
model (sdpa and eager, with and without a padding mask), the 3-slice
plan, finetune seam captures + canonical.pt + body_grads_step0.pt, and
inference prefill/decode captures.
"""

import json
import os
import subprocess
import sys

import pytest

from attacker import captures, splitbench
from attacker.splitbench import (ARCH_REGISTRY, BenchConfig, CaptureWriter,
                                 MistralSplit, SplitPlan)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_registry_exposes_mistral_and_llama_keys():
    assert ARCH_REGISTRY["mistral"] is MistralSplit
    assert ARCH_REGISTRY["llama"] is MistralSplit


def test_arch_dispatch_error_names_how_to_register():
    with pytest.raises(ValueError) as ei:
        splitbench.get_arch_splitter("qwen-ish")
    msg = str(ei.value)
    assert "qwen-ish" in msg and "ARCH_REGISTRY" in msg
    assert "register" in msg
    assert "mistral" in msg and "llama" in msg and "gpt2" in msg


def test_build_split_dispatches_on_model_type():
    class FakeCfg:
        model_type = "mistral"
        n_layer = None
        num_hidden_layers = None

    class FakeModel:
        config = FakeCfg()

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


def test_body_grad_capture_config_plumbing(tmp_path):
    assert BenchConfig().body_grad_capture is False
    assert BenchConfig(body_grad_capture=True).validate() \
        .body_grad_capture is True
    with pytest.raises(ValueError, match="body_grad_capture"):
        BenchConfig(body_grad_capture="yes").validate()
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"body_grad_capture": True, "steps": 2}))
    assert BenchConfig.from_json(str(p)).body_grad_capture is True
    args = splitbench.build_parser().parse_args(["--body-grad-capture"])
    assert splitbench._config_from_args(args).body_grad_capture is True
    args = splitbench.build_parser().parse_args([])
    assert splitbench._config_from_args(args).body_grad_capture is False


def test_cli_help_torchless_lists_new_flag():
    r = subprocess.run([sys.executable, "-m", "attacker.splitbench",
                        "--help"], cwd=ROOT, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    assert "--body-grad-capture" in r.stdout
    assert "--capture-every" in r.stdout


def test_self_test_still_green(capsys):
    assert splitbench.self_test() == 0
    assert "SELF-TEST PASSED" in capsys.readouterr().out


HIDDEN = 32


@pytest.fixture
def tiny():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import MistralConfig, MistralForCausalLM
    torch.manual_seed(0)
    cfg = MistralConfig(hidden_size=HIDDEN, intermediate_size=64,
                        num_hidden_layers=4, num_attention_heads=4,
                        num_key_value_heads=2, vocab_size=99,
                        max_position_embeddings=64)
    model = MistralForCausalLM(cfg)
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
def test_torch_split_is_exact(tiny):
    torch, model = tiny
    split = _split(model, "2")
    ids = torch.randint(0, 99, (2, 8))
    with torch.no_grad():
        ref = model(ids).logits
        out = split.rest_forward(split.head_forward(ids))
    assert torch.allclose(ref, out, atol=1e-4)


@pytest.mark.slow
def test_torch_split_is_exact_eager_with_padding(tiny):
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
def test_torch_three_slice_plan_is_exact(tiny):
    torch, model = tiny
    split = _split(model, "1,3")
    assert split.ranges == [(0, 1), (1, 3), (3, 4)]
    ids = torch.randint(0, 99, (2, 8))
    with torch.no_grad():
        ref = model(ids).logits
        h = split.head_forward(ids)
        h = split.body_forward(h, split.ranges[1])
        out = split.tail_forward(h)
    assert torch.allclose(ref, out, atol=1e-4)


@pytest.mark.slow
def test_torch_finetune_captures_canonical_and_body_grads(tiny, tmp_path):
    torch, model = tiny
    split = _split(model)
    writer = CaptureWriter(str(tmp_path), "training", "m0")
    cfg = BenchConfig(steps=2, seq_len=8, batch_size=2, capture_every=1,
                      lr=1e-4, save_canonical=True, body_grad_capture=True)
    stats = splitbench.finetune(split, cfg, _batches(torch), writer)
    writer.close()
    assert writer.count == 4
    recs = captures.scan_captures(str(tmp_path), "training")
    assert [(m["phase"], m["step"], m["epoch"]) for m, _ in recs] == [
        ("fwd", 0, 0), ("bwd", 0, 0), ("fwd", 1, 0), ("bwd", 1, 0)]
    assert captures.load_tensor(recs[0][1]).shape == (2, 8, HIDDEN)
    canonical = torch.load(tmp_path / "canonical.pt", weights_only=True)
    assert canonical.shape == (2 * 2, 8) and canonical.dtype == torch.long
    path = tmp_path / splitbench.BODY_GRADS_NAME
    assert path.is_file()
    grads = torch.load(path, weights_only=True)
    assert grads and all(v.dtype == torch.float16 for v in grads.values())
    prefixes = split.body_param_prefixes()
    expected = {name for name, p in model.named_parameters()
                if any(pre in name for pre in prefixes)}
    assert set(grads) == expected
    assert any(k.startswith("model.layers.2.") for k in grads)
    assert "model.norm.weight" in grads
    assert not any(k.startswith("model.layers.0.") for k in grads)
    import math
    assert math.isfinite(stats["final_loss"])


@pytest.mark.slow
def test_torch_body_grad_capture_defaults_off(tiny, tmp_path):
    torch, model = tiny
    writer = CaptureWriter(str(tmp_path), "training", "m1")
    cfg = BenchConfig(steps=1, seq_len=8, batch_size=2, lr=1e-4)
    splitbench.finetune(_split(model), cfg, _batches(torch, n=1), writer)
    writer.close()
    assert not (tmp_path / splitbench.BODY_GRADS_NAME).exists()


@pytest.mark.slow
def test_torch_inference_seam_captures(tiny, tmp_path):
    torch, model = tiny
    split = _split(model)
    writer = CaptureWriter(str(tmp_path), "inference", "mi")
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
