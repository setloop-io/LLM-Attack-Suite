"""Trained compression-bottleneck defense tests (Phase 1a).

Torch-free layer (runs everywhere): BenchConfig validation matrix for the
tb_* knobs, _defense_config_record, BottleneckConfig validation +
config hash, provenance-sidecar schema validation (incl. the
MI-reading-must-be-a-non-certificate requirement), converter argument
errors, and the CPU guard's SystemExit on a torch host without CUDA.

Torch layer (slow, self-skips without torch/transformers; tiny random
GPT-2 2L/32/99 on CPU with allow_cpu): determinism (same seed -> identical
encoder checkpoint hash and identical captured releases), the
straight-through gradient (grad flows as identity, shape preserved), the
provenance sidecar's completeness against the run output, the converter
round-trip into representation_privacy.validate_bundle, utility
before/after fields, and the run artifact's defense_config + provenance
path.

Phase 1b adds the secret-mixup layer: mixup release math (rows are the
stated convex combination of the pool contents), sampling without
replacement + honest-label tracking, the empty-pool undefended policy and
pool FIFO cap, determinism (same seed -> identical releases + provenance),
the llm_attacker.secret_mixup_provenance.v1 schema + drift matrix, the
decoy-forward correction's exactness under a linear cloud segment
(shapes/finiteness), splitbench wiring (provenance sidecar + decoy store +
utility fields + run artifact), and the converter round-trip into the
strict view_privacy.v1 mixup gate.
"""

import hashlib
import json
import math
import os

import pytest

from attacker import splitbench
from attacker import trained_defenses as td


def test_config_accepts_trained_bottleneck():
    cfg = splitbench.BenchConfig(defense="trained-bottleneck")
    assert cfg.validate() is cfg
    assert (cfg.tb_dim, cfg.tb_hidden, cfg.tb_probe_hidden,
            cfg.tb_minimax_steps, cfg.tb_mi_weight, cfg.tb_allow_cpu,
            cfg.tb_checkpoint) == (64, 256, 128, 200, 0.1, False, None)


@pytest.mark.parametrize("kw", [
    {"tb_dim": 0}, {"tb_dim": -1}, {"tb_dim": 2.5}, {"tb_dim": True},
    {"tb_hidden": 0}, {"tb_probe_hidden": -4},
    {"tb_minimax_steps": 0}, {"tb_minimax_steps": "x"},
    {"tb_mi_weight": -0.1}, {"tb_mi_weight": "w"},
    {"tb_allow_cpu": "yes"}, {"tb_allow_cpu": 1},
    {"tb_checkpoint": 3},
])
def test_config_tb_drift_raises(kw):
    with pytest.raises(ValueError):
        splitbench.BenchConfig(defense="trained-bottleneck",
                               **kw).validate()


def test_defense_registry_includes_trained_bottleneck():
    assert "trained-bottleneck" in splitbench._DEFENSES
    for d in ("none", "gaussian", "sparsify", "mid-lite", "at-lite",
              "santext", "custext", "snd-lite"):
        assert d in splitbench._DEFENSES


def test_defense_config_record_tb_knobs():
    rec = splitbench._defense_config_record(
        splitbench.BenchConfig(defense="trained-bottleneck", tb_dim=16,
                               tb_minimax_steps=5))
    assert rec["name"] == "trained-bottleneck"
    assert rec["bottleneck_dim"] == 16 and rec["minimax_steps"] == 5
    for d in splitbench._DEFENSES:
        assert splitbench._defense_config_record(
            splitbench.BenchConfig(defense=d))["name"] == d


def test_bottleneck_config_validation_and_hash():
    cfg = td.BottleneckConfig(bottleneck_dim=16, minimax_steps=3, seed=7)
    assert cfg.validate() is cfg
    assert cfg.config_sha256() == \
        td.BottleneckConfig(**cfg.canonical()).config_sha256()
    assert cfg.config_sha256() != \
        td.BottleneckConfig(**{**cfg.canonical(), "seed": 8}).config_sha256()
    with pytest.raises(ValueError):
        td.BottleneckConfig(bottleneck_dim=0).validate()
    with pytest.raises(ValueError):
        td.BottleneckConfig(mi_weight=-1).validate()
    with pytest.raises(ValueError):
        td.BottleneckConfig(allow_cpu="yes").validate()


def _prov_record(**over):
    cfg = td.BottleneckConfig(bottleneck_dim=16, minimax_steps=3, seed=7)
    tb = td.TrainedBottleneck(cfg.canonical(), seam_width=32, vocab=99,
                              state_dict={}, mi_reading=1.25)
    tb.utility = {
        "utility_before": {"cross_entropy": 4.5, "perplexity": 90.0},
        "utility_after": {"cross_entropy": 4.7, "perplexity": 110.0}}
    rec = tb.provenance_record("/tmp/x/" + td.ENCODER_NAME, "a" * 64)
    rec.update(over)
    return rec


def test_provenance_schema_complete():
    rec = td.validate_provenance(_prov_record())
    assert rec["schema"] == td.PROVENANCE_SCHEMA
    assert rec["config_sha256"] and rec["seed"] == 7
    assert rec["checkpoint"]["path"].endswith(td.ENCODER_NAME)
    mi = rec["mi_reading"]
    assert mi["certificate"] is False
    assert isinstance(mi["value"], float)
    assert "certificate" in mi["caveat"].lower()
    assert "re-implementation" in rec["created_by"]


@pytest.mark.parametrize("mut", [
    lambda r: r.update(schema="representation_privacy.v1"),
    lambda r: r.pop("checkpoint"),
    lambda r: r.update(config_sha256="0" * 64),
    lambda r: r["mi_reading"].update(certificate=True),
    lambda r: r["mi_reading"].update(caveat="a bound on the true MI"),
    lambda r: r["mi_reading"].update(value=float("nan")),
    lambda r: r["utility"].update(utility_after={"cross_entropy": "x"}),
])
def test_provenance_drift_raises(mut):
    rec = _prov_record()
    mut(rec)
    with pytest.raises(ValueError):
        td.validate_provenance(rec)


def test_converter_requires_enough_captures(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match=">= 8 fwd captures"):
        td.capture_to_representation_bundle(str(tmp_path), dim=4)


def test_module_imports_torch_free():
    import sys
    src = open(os.path.join(os.path.dirname(td.__file__),
                            "trained_defenses.py")).read()
    assert "try:\n    import torch" in src
    assert td.PROVENANCE_SCHEMA == \
        "llm_attacker.trained_bottleneck_provenance.v1"
    assert td.MIXUP_PROVENANCE_SCHEMA == \
        "llm_attacker.secret_mixup_provenance.v1"
    assert sys


def test_config_accepts_secret_mixup():
    cfg = splitbench.BenchConfig(defense="secret-mixup")
    assert cfg.validate() is cfg
    assert (cfg.mixup_lambda, cfg.mixup_pool, cfg.mixup_correction) == \
        (0.75, 256, True)


@pytest.mark.parametrize("kw", [
    {"mixup_lambda": 0.0}, {"mixup_lambda": -0.1}, {"mixup_lambda": 1.5},
    {"mixup_lambda": True}, {"mixup_lambda": "l"},
    {"mixup_pool": 0}, {"mixup_pool": -3}, {"mixup_pool": 2.5},
    {"mixup_correction": "yes"}, {"mixup_correction": 1},
])
def test_config_mixup_drift_raises(kw):
    with pytest.raises(ValueError):
        splitbench.BenchConfig(defense="secret-mixup", **kw).validate()


def test_defense_registry_includes_secret_mixup():
    assert "secret-mixup" in splitbench._DEFENSES
    assert len(splitbench._DEFENSES) == 13
    for d in ("none", "gaussian", "sparsify", "mid-lite", "at-lite",
              "santext", "custext", "snd-lite", "trained-bottleneck"):
        assert d in splitbench._DEFENSES


def test_defense_config_record_mixup_knobs():
    rec = splitbench._defense_config_record(
        splitbench.BenchConfig(defense="secret-mixup", mixup_lambda=0.5,
                               mixup_pool=32, mixup_correction=False))
    assert rec == {"name": "secret-mixup", "lambda": 0.5, "pool_size": 32,
                   "correction": False}
    for d in splitbench._DEFENSES:
        assert splitbench._defense_config_record(
            splitbench.BenchConfig(defense=d))["name"] == d


def test_mixup_config_validation_and_hash():
    cfg = td.MixupConfig(lam=0.5, pool_size=8, seed=3, correction=False)
    assert cfg.validate() is cfg
    assert cfg.canonical() == {"lambda": 0.5, "pool_size": 8, "seed": 3,
                               "correction": False}
    assert cfg.config_sha256() == \
        td.MixupConfig(lam=0.5, pool_size=8, seed=3,
                       correction=False).config_sha256()
    assert cfg.config_sha256() != \
        td.MixupConfig(lam=0.5, pool_size=8, seed=4,
                       correction=False).config_sha256()
    with pytest.raises(ValueError):
        td.MixupConfig(lam=0.0).validate()
    with pytest.raises(ValueError):
        td.MixupConfig(lam=1.5).validate()
    with pytest.raises(ValueError):
        td.MixupConfig(pool_size=0).validate()
    with pytest.raises(ValueError):
        td.MixupConfig(correction="yes").validate()


def _mix_prov_record(**over):
    mix = td.SecretMixup(td.MixupConfig(pool_size=8, seed=3).canonical())
    mix.utility = {
        "utility_before": {"cross_entropy": 4.5, "perplexity": 90.0},
        "utility_after": {"cross_entropy": 4.7, "perplexity": 110.0}}
    mix.blocks = [
        {"block_index": 0, "policy": "undefended_empty_pool", "n_rows": 4,
         "pool_size_before": 0, "pool_size_after": 4, "draw_seed": None,
         "decoy_pool_indices": [], "honest_labels": [1, 2, 3, 4],
         "decoy_honest_labels": []},
        {"block_index": 1, "policy": "mixed", "n_rows": 4,
         "pool_size_before": 4, "pool_size_after": 8,
         "draw_seed": mix._draw_seed(1), "decoy_pool_indices": [3, 1, 0, 2],
         "honest_labels": [5, 6, 7, 8],
         "decoy_honest_labels": [4, 2, 1, 3]}]
    mix.pool = [None] * 8
    rec = mix.provenance_record("/tmp/x/" + td.MIXUP_DECOYS_NAME, "b" * 64)
    mix.pool = []
    rec.update(over)
    return rec


def test_mixup_provenance_schema_complete():
    rec = td.validate_mixup_provenance(_mix_prov_record())
    assert rec["schema"] == td.MIXUP_PROVENANCE_SCHEMA
    assert rec["lambda"] == 0.75 and rec["seed"] == 3
    assert rec["correction_mode"] == "decoy_forward_subtract"
    assert "approximate" in rec["correction_caveat"].lower()
    assert rec["empty_pool_policy"] == "release_undefended"
    assert rec["blocks"][0]["policy"] == "undefended_empty_pool"
    assert rec["blocks"][1]["draw_seed"] is not None
    assert "re-implementation" in rec["created_by"]


@pytest.mark.parametrize("mut", [
    lambda r: r.update(schema="view_privacy.v1"),
    lambda r: r.pop("blocks"),
    lambda r: r.update(config_sha256="0" * 64),
    lambda r: r.update(correction_mode="magic"),
    lambda r: r.update(correction_caveat="exact always"),
    lambda r: r.update(empty_pool_policy="hold"),
    lambda r: r["blocks"][1].update(decoy_pool_indices=[1, 1, 0, 2]),
    lambda r: r["blocks"][1].update(draw_seed=None),
    lambda r: r["blocks"][0].update(draw_seed=5),
    lambda r: r["blocks"][1].update(honest_labels=[1]),
    lambda r: r.update(utility={"utility_after": {"cross_entropy": "x"}}),
])
def test_mixup_provenance_drift_raises(mut):
    rec = _mix_prov_record()
    mut(rec)
    with pytest.raises(ValueError):
        td.validate_mixup_provenance(rec)


def test_mixup_converter_requires_sidecar(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="secret-mixup"):
        td.capture_to_view_privacy_bundle(str(tmp_path))


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


TB_KW = dict(tb_dim=8, tb_hidden=16, tb_probe_hidden=16, tb_minimax_steps=4,
             tb_allow_cpu=True)


def _run(torch, tmp_path, name, steps=3, save_canonical=False):
    """Tiny split finetune under --defense trained-bottleneck; identical
    weights/data/seed across calls."""
    model = _tiny_model(torch)
    split = _split(model)
    writer = splitbench.CaptureWriter(str(tmp_path / name), "training",
                                      name)
    cfg = splitbench.BenchConfig(steps=steps, seq_len=8, batch_size=2,
                                 lr=1e-4, defense="trained-bottleneck",
                                 seed=7, save_canonical=save_canonical,
                                 **TB_KW)
    stats = splitbench.finetune(split, cfg, _batches(torch, n=steps),
                                writer)
    return stats


def test_torch_cpu_guard_raises_without_allow_cpu(tiny):
    torch = tiny
    if torch.cuda.is_available():
        pytest.skip("CPU guard test needs a CUDA-less view")
    h = torch.randn(16, 32)
    t = torch.randint(0, 99, (16,))
    cfg = td.BottleneckConfig(bottleneck_dim=8, minimax_steps=1)
    with pytest.raises(SystemExit, match="GPU-only"):
        td.train_bottleneck(h, t, cfg, device="cpu")
    cfg2 = splitbench.BenchConfig(defense="trained-bottleneck", **TB_KW)
    with pytest.raises(ValueError, match="make_defense_state"):
        splitbench._apply_seam_defense(torch.randn(1, 2, 32), cfg2, {})


@pytest.mark.slow
def test_torch_straight_through_gradient(tiny):
    torch = tiny
    g = torch.Generator().manual_seed(5)
    h_cal = torch.randn(32, 32, generator=g)
    t_cal = torch.randint(0, 99, (32,), generator=g)
    cfg = td.BottleneckConfig(bottleneck_dim=8, encoder_hidden=16,
                              probe_hidden=16, minimax_steps=3, seed=7,
                              allow_cpu=True)
    tb = td.train_bottleneck(h_cal, t_cal, cfg, device="cpu")
    h = torch.randn(2, 5, 32, generator=g, requires_grad=True)
    out = tb.apply(h)
    assert out.shape == h.shape
    assert out[..., 8:].abs().max().item() == 0.0
    assert torch.allclose(out[..., :8], tb.encode(h.detach()),
                          atol=1e-5)
    upstream = torch.randn(2, 5, 32, generator=g)
    out.backward(upstream)
    assert h.grad is not None
    assert torch.isfinite(h.grad).all()
    assert not torch.allclose(h.grad, upstream, atol=1e-6)


@pytest.mark.slow
def test_torch_determinism_same_seed(tiny, tmp_path):
    torch = tiny
    from attacker import captures
    _run(torch, tmp_path, "r1")
    _run(torch, tmp_path, "r2")

    def sha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    c1 = sha(tmp_path / "r1" / td.ENCODER_NAME)
    c2 = sha(tmp_path / "r2" / td.ENCODER_NAME)
    assert c1 == c2
    a = captures.load_tensor(str(tmp_path / "r1" / "wire_00000.pt"))
    b = captures.load_tensor(str(tmp_path / "r2" / "wire_00000.pt"))
    assert torch.equal(a, b)
    model = _tiny_model(torch)
    split = _split(model)
    writer = splitbench.CaptureWriter(str(tmp_path / "none"), "training",
                                      "none")
    cfg = splitbench.BenchConfig(steps=1, seq_len=8, batch_size=2,
                                 lr=1e-4, defense="none", seed=7)
    splitbench.finetune(split, cfg, _batches(torch, n=1), writer)
    none_fwd = captures.load_tensor(str(tmp_path / "none"
                                        / "wire_00000.pt"))
    assert not torch.equal(none_fwd, a)


@pytest.mark.slow
def test_torch_provenance_sidecar_and_utility(tiny, tmp_path):
    stats = _run(tiny, tmp_path, "prov")
    prov_path = tmp_path / "prov" / td.PROVENANCE_NAME
    assert prov_path.is_file()
    rec = td.validate_provenance(json.loads(prov_path.read_text()))
    ckpt = rec["checkpoint"]
    assert os.path.isfile(ckpt["path"])
    assert hashlib.sha256(open(ckpt["path"], "rb").read()).hexdigest() \
        == ckpt["sha256"]
    assert math.isfinite(rec["mi_reading"]["value"])
    assert rec["mi_reading"]["certificate"] is False
    for split in ("utility_before", "utility_after"):
        u = rec["utility"][split]
        assert math.isfinite(u["cross_entropy"])
        assert u["perplexity"] > 1.0
    assert stats["trained_bottleneck_provenance"]["provenance_path"] \
        == str(prov_path)


@pytest.mark.slow
def test_torch_converter_round_trip_validate_bundle(tiny, tmp_path):
    _run(tiny, tmp_path, "conv", steps=8, save_canonical=True)
    from attacker.attacks import representation_privacy
    out = tmp_path / "bundle.json"
    bundle = td.capture_to_representation_bundle(
        str(tmp_path / "conv"), output=str(out))
    shape = representation_privacy.validate_bundle(
        json.loads(out.read_text()))
    assert shape == (8, 8)
    assert len(bundle["calibration"]) == 4
    assert len(bundle["evaluation"]) == 4
    row = bundle["calibration"][0]
    assert row["label"] >= 0 and len(row["released"][0]) == 8
    assert "trained-bottleneck" in bundle["provenance"]["defender"]


@pytest.mark.slow
def test_torch_run_artifact_records_provenance(tiny, tmp_path):
    model = _tiny_model(tiny)
    split = _split(model)
    out = tmp_path / "run.json"
    cfg = splitbench.BenchConfig(split="1", steps=2, seq_len=8,
                                 batch_size=2, lr=1e-4,
                                 defense="trained-bottleneck",
                                 capture_dir=str(tmp_path / "caps"),
                                 output=str(out), seed=11, **TB_KW)
    writer = splitbench.CaptureWriter(cfg.capture_dir, "training", "art")
    splitbench.execute(cfg, split, writer, _batches(tiny, n=2))
    art = json.loads(out.read_text())
    rec = art["results"][0]
    assert rec["defense"] == "trained-bottleneck"
    assert rec["defense_config"]["name"] == "trained-bottleneck"
    assert rec["defense_config"]["bottleneck_dim"] == 8
    prov = rec["trained_bottleneck_provenance"]
    assert str(prov).endswith(td.PROVENANCE_NAME)
    assert os.path.isfile(prov)
    assert math.isfinite(rec["final_loss"])


MIX_KW = dict(mixup_lambda=0.75, mixup_pool=64)


def _mix(torch, lam=0.75, pool_size=8, seed=3, correction=True):
    return td.SecretMixup(td.MixupConfig(lam=lam, pool_size=pool_size,
                                         seed=seed,
                                         correction=correction).canonical())


@pytest.mark.slow
def test_torch_mixup_release_math_and_no_replacement(tiny):
    torch = tiny
    g = torch.Generator().manual_seed(5)
    mix = _mix(torch, pool_size=8, seed=3)
    h0 = torch.randn(4, 8, generator=g)
    t0 = torch.arange(4)
    w0 = mix.apply(h0, tokens=t0)
    assert torch.equal(w0, h0)
    assert mix.blocks[0]["policy"] == "undefended_empty_pool"
    assert mix.blocks[0]["honest_labels"] == [0, 1, 2, 3]
    h1 = torch.randn(4, 8, generator=g)
    t1 = torch.arange(10, 14)
    w1 = mix.apply(h1, tokens=t1)
    seed = mix._draw_seed(1)
    perm = torch.randperm(4, generator=torch.Generator().manual_seed(seed))
    z = h0[perm]
    assert torch.allclose(w1, 0.75 * h1 + 0.25 * z, atol=1e-6)
    assert len(set(mix.blocks[1]["decoy_pool_indices"])) == 4
    assert mix.blocks[1]["decoy_honest_labels"] == \
        [int(t0[i]) for i in perm.tolist()]
    assert mix.last_decoy is not None and torch.allclose(mix.last_decoy, z)


@pytest.mark.slow
def test_torch_mixup_empty_pool_policy_and_pool_cap(tiny):
    torch = tiny
    mix = _mix(torch, pool_size=3, seed=1)
    h = torch.randn(5, 4)
    w = mix.apply(h)
    assert torch.equal(w, h)
    assert mix.blocks[0]["pool_size_after"] == 3
    assert mix.pool_labels == [None, None, None]
    w2 = mix.apply(torch.randn(7, 4))
    idx = mix.blocks[1]["decoy_pool_indices"]
    assert len(idx) == 7 and len(set(idx[:3])) == 3
    assert w2.shape == (7, 4)


@pytest.mark.slow
def test_torch_mixup_determinism_same_seed(tiny):
    torch = tiny
    g = torch.Generator().manual_seed(9)
    blocks = [torch.randn(4, 8, generator=g) for _ in range(4)]
    toks = [torch.randint(0, 99, (4,), generator=g) for _ in range(4)]
    outs = []
    for _ in range(2):
        mix = _mix(torch, pool_size=8, seed=7)
        outs.append([mix.apply(h, tokens=t) for h, t in zip(blocks, toks)])
        prov = mix.provenance_record()
    for a, b in zip(*outs):
        assert torch.equal(a, b)
    p1 = td.SecretMixup(td.MixupConfig(pool_size=8, seed=7).canonical())
    p2 = td.SecretMixup(td.MixupConfig(pool_size=8, seed=7).canonical())
    for h, t in zip(blocks, toks):
        p1.apply(h, tokens=t)
        p2.apply(h, tokens=t)
    assert p1.provenance_record() == p2.provenance_record()
    p3 = td.SecretMixup(td.MixupConfig(pool_size=8, seed=8).canonical())
    for h, t in zip(blocks, toks):
        p3.apply(h, tokens=t)
    assert p1.provenance_record()["blocks"][1]["draw_seed"] != \
        p3.provenance_record()["blocks"][1]["draw_seed"]


@pytest.mark.slow
def test_torch_mixup_correction_gradient_sanity(tiny):
    torch = tiny
    g = torch.Generator().manual_seed(11)
    lam = 0.75
    a = torch.randn(8, 8, generator=g)
    h = torch.randn(4, 8, generator=g)
    mix = _mix(torch, pool_size=8, seed=2)
    mix.apply(torch.randn(4, 8, generator=g))
    h_req = h.clone().requires_grad_(True)
    wire = mix.apply(h_req)
    v = torch.randn(4, 8, generator=g)
    y_wire = wire @ a
    decoy = mix.last_decoy.detach().clone().requires_grad_(True)
    y_decoy = decoy @ a
    gw, = torch.autograd.grad(y_wire, wire, grad_outputs=v)
    gd, = torch.autograd.grad(y_decoy, decoy, grad_outputs=v)
    g_corr = td.corrected_gradient(gw, gd, lam)
    assert g_corr.shape == gw.shape and torch.isfinite(g_corr).all()
    wire.backward(g_corr)
    assert torch.allclose(h_req.grad, v @ a.t(), atol=1e-5)


def _mix_run(torch, tmp_path, name, steps=8, n_batches=4, **kw):
    model = _tiny_model(torch)
    split = _split(model)
    writer = splitbench.CaptureWriter(str(tmp_path / name), "training",
                                      name)
    cfg = splitbench.BenchConfig(steps=steps, seq_len=8, batch_size=2,
                                 lr=1e-4, defense="secret-mixup", seed=7,
                                 **{**MIX_KW, **kw})
    stats = splitbench.finetune(split, cfg, _batches(torch, n=n_batches),
                                writer)
    return stats


@pytest.mark.slow
def test_torch_mixup_splitbench_run_provenance_utility(tiny, tmp_path):
    torch = tiny
    from attacker import captures
    stats = _mix_run(torch, tmp_path, "mix")
    prov_path = tmp_path / "mix" / td.MIXUP_PROVENANCE_NAME
    assert prov_path.is_file()
    rec = td.validate_mixup_provenance(json.loads(prov_path.read_text()))
    assert rec["lambda"] == 0.75
    assert rec["blocks"][0]["policy"] == "undefended_empty_pool"
    assert all(b["policy"] == "mixed" for b in rec["blocks"][1:])
    assert all(b["draw_seed"] is not None for b in rec["blocks"][1:])
    assert all(len(b["honest_labels"]) == 16 for b in rec["blocks"])
    ds = rec["decoy_store"]
    assert os.path.isfile(ds["path"])
    assert hashlib.sha256(open(ds["path"], "rb").read()).hexdigest() \
        == ds["sha256"]
    for s in ("utility_before", "utility_after"):
        u = rec["utility"][s]
        assert math.isfinite(u["cross_entropy"]) and u["perplexity"] > 1.0
    assert stats["secret_mixup_provenance"]["provenance_path"] \
        == str(prov_path)
    fwd0 = captures.load_tensor(str(tmp_path / "mix" / "wire_00000.pt"))
    fwd1 = captures.load_tensor(str(tmp_path / "mix" / "wire_00002.pt"))
    bwd1 = captures.load_tensor(str(tmp_path / "mix" / "wire_00003.pt"))
    assert not torch.equal(fwd0, fwd1)
    assert bwd1.shape == fwd1.shape and torch.isfinite(bwd1.float()).all()
    assert math.isfinite(stats["final_loss"])


@pytest.mark.slow
def test_torch_mixup_splitbench_determinism(tiny, tmp_path):
    torch = tiny
    from attacker import captures
    _mix_run(torch, tmp_path, "m1", steps=3, n_batches=3)
    _mix_run(torch, tmp_path, "m2", steps=3, n_batches=3)
    for f in ("wire_00000.pt", "wire_00002.pt", "wire_00004.pt"):
        a = captures.load_tensor(str(tmp_path / "m1" / f))
        b = captures.load_tensor(str(tmp_path / "m2" / f))
        assert torch.equal(a, b)
    p1 = json.loads((tmp_path / "m1" / td.MIXUP_PROVENANCE_NAME).read_text())
    p2 = json.loads((tmp_path / "m2" / td.MIXUP_PROVENANCE_NAME).read_text())
    for r in (p1, p2):
        r["decoy_store"]["path"] = "<path>"
    assert p1 == p2


@pytest.mark.slow
def test_torch_mixup_converter_round_trip_view_privacy(tiny, tmp_path):
    torch = tiny
    _mix_run(torch, tmp_path, "conv", steps=8, n_batches=4)
    from attacker.attacks import view_privacy
    out = tmp_path / "bundle.json"
    bundle = td.capture_to_view_privacy_bundle(str(tmp_path / "conv"),
                                               output=str(out))
    loaded = json.loads(out.read_text())
    assert set(loaded) == {"schema", "case", "data"}
    rows = view_privacy.evaluate_bundle(loaded)
    assert rows and all(r["case"] == "mixup" for r in rows)
    data = loaded["data"]
    assert len(data["releases"]) == 3
    assert all(len(r) == 2 for r in data["releases"])
    assert data["lambda"] == 0.75
    corr = [r for r in rows if r["arm"] == "nonlinear_corrected_output"]
    assert corr and max(r["mse"] for r in corr) < 1e-6
    def mse(arm, count):
        hit = [r for r in rows if r["arm"] == arm
               and r["releases"] == count
               and r.get("assumed_lambda") == 0.75]
        return hit[0]["mse"] if hit else None
    assert mse("oracle_pool_mean", 2) <= mse("naive_zero_mean", 2)


@pytest.mark.slow
def test_torch_mixup_run_artifact_records_provenance(tiny, tmp_path):
    model = _tiny_model(tiny)
    split = _split(model)
    out = tmp_path / "run.json"
    cfg = splitbench.BenchConfig(split="1", steps=2, seq_len=8,
                                 batch_size=2, lr=1e-4,
                                 defense="secret-mixup",
                                 capture_dir=str(tmp_path / "caps"),
                                 output=str(out), seed=11, **MIX_KW)
    writer = splitbench.CaptureWriter(cfg.capture_dir, "training", "art")
    splitbench.execute(cfg, split, writer, _batches(tiny, n=2))
    rec = json.loads(out.read_text())["results"][0]
    assert rec["defense"] == "secret-mixup"
    assert rec["defense_config"] == {"name": "secret-mixup", "lambda": 0.75,
                                     "pool_size": 64, "correction": True}
    prov = rec["secret_mixup_provenance"]
    assert str(prov).endswith(td.MIXUP_PROVENANCE_NAME)
    assert os.path.isfile(prov)
    assert math.isfinite(rec["final_loss"])


def test_config_accepts_replica_median():
    cfg = splitbench.BenchConfig(defense="replica-median")
    assert cfg.validate() is cfg
    assert (cfg.replica_count, cfg.replica_tolerance, cfg.replica_gauge,
            cfg.replica_gauge_scale) == (3, 0.02, True, 1e-3)


@pytest.mark.parametrize("kw", [
    {"replica_count": 1}, {"replica_count": 0}, {"replica_count": 2.5},
    {"replica_count": True}, {"replica_tolerance": 0.0},
    {"replica_tolerance": -0.1}, {"replica_tolerance": "t"},
    {"replica_gauge": "yes"}, {"replica_gauge": 1},
    {"replica_gauge_scale": 0.0}, {"replica_gauge_scale": -1e-3},
])
def test_config_replica_drift_raises(kw):
    with pytest.raises(ValueError):
        splitbench.BenchConfig(defense="replica-median", **kw).validate()


def test_defense_registry_includes_replica_median():
    assert "replica-median" in splitbench._DEFENSES
    assert len(splitbench._DEFENSES) == 13
    for d in ("none", "gaussian", "sparsify", "mid-lite", "at-lite",
              "santext", "custext", "snd-lite", "trained-bottleneck",
              "secret-mixup", "pca-compress"):
        assert d in splitbench._DEFENSES


def test_defense_config_record_replica_knobs():
    rec = splitbench._defense_config_record(
        splitbench.BenchConfig(defense="replica-median", replica_count=2,
                               replica_tolerance=0.05,
                               replica_gauge=False))
    assert rec == {"name": "replica-median", "replica_count": 2,
                   "tolerance": 0.05, "gauge": False, "gauge_scale": 1e-3}
    for d in splitbench._DEFENSES:
        assert splitbench._defense_config_record(
            splitbench.BenchConfig(defense=d))["name"] == d


def test_replica_config_validation_and_hash():
    cfg = td.ReplicaMedianConfig(replica_count=2, tolerance=0.05, seed=9)
    assert cfg.validate() is cfg
    assert cfg.channel_seed(0) == 9 and cfg.channel_seed(1) == 8
    assert cfg.channel_seed(0) != cfg.channel_seed(1)
    assert cfg.config_sha256() == \
        td.ReplicaMedianConfig(**cfg.canonical()).config_sha256()
    assert cfg.config_sha256() != \
        td.ReplicaMedianConfig(**{**cfg.canonical(),
                                  "seed": 10}).config_sha256()
    with pytest.raises(ValueError):
        td.ReplicaMedianConfig(replica_count=1).validate()
    with pytest.raises(ValueError):
        td.ReplicaMedianConfig(tolerance=0.0).validate()
    with pytest.raises(ValueError):
        td.ReplicaMedianConfig(gauge="yes").validate()
    with pytest.raises(ValueError):
        td.ReplicaMedianConfig(gauge_scale=-1).validate()


def _rep_prov_record(**over):
    rep = td.ReplicaMedian(
        td.ReplicaMedianConfig(seed=5).canonical(),
        gauge_ids=["ga", "gb", "gc"])
    rep.heterogeneity = {
        "source": "calibration_pre_phase", "n_rows": 32,
        "per_channel_mean_rel_dev": [1e-3, 1.1e-3, 0.9e-3],
        "mean_rel_dev": 1e-3, "p95_rel_dev": 2e-3, "max_rel_dev": 3e-3,
        "tolerance": 0.02, "note": "n"}
    rep.utility = {
        "utility_before": {"cross_entropy": 4.5, "perplexity": 90.0},
        "utility_after": {"cross_entropy": 4.6, "perplexity": 100.0}}
    rec = rep.provenance_record("/tmp/x/" + td.REPLICA_JOURNAL_NAME,
                                "c" * 64)
    rec.update(over)
    return rec


def test_replica_provenance_schema_complete():
    rec = td.validate_replica_provenance(_rep_prov_record())
    assert rec["schema"] == td.REPLICA_PROVENANCE_SCHEMA
    assert rec["replica_count"] == 3 and len(rec["channel_seeds"]) == 3
    assert rec["channel_seeds"] == [5, 4, 7]
    assert rec["aggregation_rule"] == "coordinate_median"
    assert rec["k2_policy"] is None
    assert rec["honest_heterogeneity"]["source"] == "calibration_pre_phase"
    assert "in-process" in rec["deployment"]
    assert "re-implementation" in rec["created_by"]
    k2 = td.ReplicaMedian(td.ReplicaMedianConfig(replica_count=2,
                                                 seed=5).canonical(),
                          gauge_ids=["ga", "gb"])
    rec2 = td.validate_replica_provenance(k2.provenance_record())
    assert rec2["aggregation_rule"] == "coordinate_mean_k2"
    assert rec2["k2_policy"] == "availability_abort_flagged"
    assert rec2["honest_heterogeneity"]["source"] == "run_blocks"


@pytest.mark.parametrize("mut", [
    lambda r: r.update(schema="view_privacy.v1"),
    lambda r: r.pop("honest_heterogeneity"),
    lambda r: r.update(config_sha256="0" * 64),
    lambda r: r.update(channel_seeds=[5, 4, 999]),
    lambda r: r.update(gauge_ids=["ga", "ga", "gc"]),
    lambda r: r.update(aggregation_rule="coordinate_mean_k2"),
    lambda r: r.update(tolerance=0.05),
    lambda r: r.update(k2_policy="availability_abort_flagged"),
    lambda r: r["honest_heterogeneity"].update(source="moon"),
    lambda r: r["honest_heterogeneity"].update(p95_rel_dev=9.0),
    lambda r: r.update(journal={"path": "", "sha256": "x"}),
    lambda r: r.update(deployment="cross-host channels"),
    lambda r: r.update(utility={"utility_after": {"cross_entropy": "x"}}),
])
def test_replica_provenance_drift_raises(mut):
    rec = _rep_prov_record()
    mut(rec)
    with pytest.raises(ValueError):
        td.validate_replica_provenance(rec)


def test_replica_converter_requires_sidecar(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="replica-median"):
        td.capture_to_replica_bundle(str(tmp_path))


def _lin_rep(torch, scales, tol=0.02, seed=5):
    cfg = td.ReplicaMedianConfig(replica_count=len(scales), tolerance=tol,
                                 seed=seed)
    return td.ReplicaMedian(
        cfg.canonical(),
        channel_forwards=[(lambda s: (lambda x: x * s))(s)
                          for s in scales],
        gauge_ids=[f"g{i}" for i in range(len(scales))])


@pytest.mark.slow
def test_torch_coordinate_median_correctness(tiny):
    torch = tiny
    import statistics
    g = torch.Generator().manual_seed(3)
    ys = [torch.randn(4, 6, generator=g) for _ in range(3)]
    med = td.coordinate_median(ys)
    exp = torch.tensor([[statistics.median([ys[k][i, j].item()
                                            for k in range(3)])
                         for j in range(6)] for i in range(4)])
    assert torch.allclose(med, exp)
    a, b = torch.randn(3, 5, generator=g), torch.randn(3, 5, generator=g)
    assert torch.allclose(td.coordinate_median([a, b]), (a + b) / 2)
    with pytest.raises(ValueError):
        td.coordinate_median([a])


@pytest.mark.slow
def test_torch_median_aggregation_and_deviation_flagging(tiny):
    torch = tiny
    g = torch.Generator().manual_seed(5)
    h = torch.randn(3, 8, generator=g)
    rep = _lin_rep(torch, [1.0, 1.001, 0.999])
    agg, ys, entry = rep.forward_block(h, step=0, captured=True)
    assert torch.allclose(agg, h)
    assert not any(c["flagged"] for c in entry["channels"])
    assert all(c["max_rel_dev"] < 0.02 for c in entry["channels"])
    assert entry["aggregation_rule"] == "coordinate_median"
    rep2 = _lin_rep(torch, [1.0, 1.0, 1.05])
    agg2, _ys2, e2 = rep2.forward_block(h, step=0, captured=True)
    flags = [c["flagged"] for c in e2["channels"]]
    assert flags == [False, False, True]
    assert e2["channels"][2]["n_flagged"] == e2["n_rows"]
    assert e2["channels"][2]["max_rel_dev"] > 0.02
    assert torch.allclose(agg2, h)


@pytest.mark.slow
def test_torch_k2_disagreement_policy(tiny):
    torch = tiny
    h = torch.randn(2, 6, generator=torch.Generator().manual_seed(7))
    rep = _lin_rep(torch, [1.0, 1.001])
    agg, _ys, e = rep.forward_block(h, step=0, captured=True)
    assert e["aggregation_rule"] == "coordinate_mean_k2"
    assert torch.allclose(agg, h * 1.0005, atol=1e-6)
    assert e["k2_disagreement"]["abort"] is False
    rep2 = _lin_rep(torch, [1.0, 1.1])
    _a2, _y2, e2 = rep2.forward_block(h, step=0, captured=True)
    assert e2["k2_disagreement"]["abort"] is True
    assert e2["k2_disagreement"]["policy"] == "availability_abort_flagged"
    assert e2["k2_disagreement"]["max_rel"] > 0.02


@pytest.mark.slow
def test_torch_replica_journal_determinism(tiny):
    torch = tiny
    g = torch.Generator().manual_seed(11)
    blocks = [torch.randn(2, 6, generator=g) for _ in range(3)]
    journals = []
    for _ in range(2):
        rep = _lin_rep(torch, [1.0, 1.001, 0.999], seed=9)
        for i, b in enumerate(blocks):
            rep.forward_block(b, step=i, captured=True)
        journals.append(rep.journal)
    assert journals[0] == journals[1]
    assert [e["block_index"] for e in journals[0]] == [0, 1, 2]


REP_KW = dict(replica_count=3, replica_tolerance=0.02)


def _rep_run(torch, tmp_path, name, steps=4, n_batches=4, **kw):
    model = _tiny_model(torch)
    split = _split(model)
    writer = splitbench.CaptureWriter(str(tmp_path / name), "training",
                                      name)
    cfg = splitbench.BenchConfig(steps=steps, seq_len=8, batch_size=2,
                                 lr=1e-4, defense="replica-median", seed=7,
                                 **{**REP_KW, **kw})
    stats = splitbench.finetune(split, cfg, _batches(torch, n=n_batches),
                                writer)
    return stats


@pytest.mark.slow
def test_torch_replica_channel_independence(tiny, tmp_path):
    torch = tiny
    model = _tiny_model(torch)
    split = _split(model)
    cfg = splitbench.BenchConfig(defense="replica-median", seed=7, **REP_KW)
    state = splitbench.make_defense_state(cfg, split)
    rep = state["replica"]
    assert len(rep.channel_forwards) == 3
    assert len(set(rep.channel_seeds)) == 3
    assert len(set(rep.gauge_ids)) == 3
    h = torch.randn(2, 8, 32, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ys = [f(h) for f in rep.channel_forwards]
    assert not torch.equal(ys[0], ys[1])
    assert not torch.equal(ys[1], ys[2])
    assert rep.heterogeneity is None
    model2 = _tiny_model(torch)
    split2 = _split(model2)
    state2 = splitbench.make_defense_state(cfg, split2)
    with torch.no_grad():
        ys2 = [f(h) for f in state2["replica"].channel_forwards]
    for a, b in zip(ys, ys2):
        assert torch.equal(a, b)
    cfg_off = splitbench.BenchConfig(defense="replica-median", seed=7,
                                     replica_gauge=False)
    state_off = splitbench.make_defense_state(cfg_off, _split(
        _tiny_model(torch)))
    with torch.no_grad():
        zo = [f(h) for f in state_off["replica"].channel_forwards]
    assert torch.equal(zo[0], zo[1]) and torch.equal(zo[1], zo[2])
    assert splitbench._apply_seam_defense(h, cfg, state) is h


@pytest.mark.slow
def test_torch_replica_run_captures_journal_provenance(tiny, tmp_path):
    torch = tiny
    from attacker import captures
    stats = _rep_run(torch, tmp_path, "rep")
    d = tmp_path / "rep"
    prov_path = d / td.REPLICA_PROVENANCE_NAME
    rec = td.validate_replica_provenance(json.loads(prov_path.read_text()))
    assert rec["replica_count"] == 3
    assert rec["honest_heterogeneity"]["source"] == "calibration_pre_phase"
    assert rec["honest_heterogeneity"]["max_rel_dev"] < 0.02
    for s in ("utility_before", "utility_after"):
        u = rec["utility"][s]
        assert math.isfinite(u["cross_entropy"]) and u["perplexity"] > 1.0
    assert stats["replica_median_provenance"]["provenance_path"] \
        == str(prov_path)
    jpath = d / td.REPLICA_JOURNAL_NAME
    assert hashlib.sha256(jpath.read_bytes()).hexdigest() \
        == rec["journal"]["sha256"]
    entries = [json.loads(l) for l in jpath.read_text().splitlines()]
    assert [e["block_index"] for e in entries] == list(range(4))
    assert all(len(e["channels"]) == 3 for e in entries)
    assert not any(c["flagged"] for c in entries[0]["channels"])
    assert all("max_rel_dev" in c and "flagged" in c
               for e in entries for c in e["channels"])
    ch_wires = []
    for k in range(3):
        recs = captures.scan_captures(str(d / "channels" / f"ch{k}"),
                                      "training")
        assert [m["phase"] for m, _ in recs] == ["fwd"] * 4
        ch_wires.append([captures.load_tensor(p).float() for _, p in recs])
    main = captures.scan_captures(str(d), "training")
    fwd = [captures.load_tensor(p).float() for m, p in main
           if m["phase"] == "fwd"]
    assert len(fwd) == 4
    med = torch.stack([ch_wires[k][0] for k in range(3)]).median(dim=0) \
        .values
    assert torch.allclose(fwd[0], med, atol=2e-3)
    assert not torch.equal(ch_wires[0][0], ch_wires[1][0])
    assert math.isfinite(stats["final_loss"])


@pytest.mark.slow
def test_torch_replica_run_determinism(tiny, tmp_path):
    torch = tiny
    from attacker import captures
    _rep_run(torch, tmp_path, "r1", steps=3, n_batches=3)
    _rep_run(torch, tmp_path, "r2", steps=3, n_batches=3)
    for f in ("wire_00000.pt", "wire_00002.pt"):
        a = captures.load_tensor(str(tmp_path / "r1" / f))
        b = captures.load_tensor(str(tmp_path / "r2" / f))
        assert torch.equal(a, b)
        for k in range(3):
            ca = captures.load_tensor(str(tmp_path / "r1" / "channels"
                                          / f"ch{k}" / f))
            cb = captures.load_tensor(str(tmp_path / "r2" / "channels"
                                          / f"ch{k}" / f))
            assert torch.equal(ca, cb)
    j1 = (tmp_path / "r1" / td.REPLICA_JOURNAL_NAME).read_text()
    j2 = (tmp_path / "r2" / td.REPLICA_JOURNAL_NAME).read_text()
    assert j1 == j2
    p1 = json.loads((tmp_path / "r1" / td.REPLICA_PROVENANCE_NAME)
                    .read_text())
    p2 = json.loads((tmp_path / "r2" / td.REPLICA_PROVENANCE_NAME)
                    .read_text())
    for r in (p1, p2):
        r["journal"]["path"] = "<path>"
    assert p1 == p2


@pytest.mark.slow
def test_torch_replica_k2_run(tiny, tmp_path):
    torch = tiny
    _rep_run(torch, tmp_path, "k2", steps=2, n_batches=2, replica_count=2)
    d = tmp_path / "k2"
    rec = td.validate_replica_provenance(
        json.loads((d / td.REPLICA_PROVENANCE_NAME).read_text()))
    assert rec["replica_count"] == 2
    assert rec["aggregation_rule"] == "coordinate_mean_k2"
    assert rec["k2_policy"] == "availability_abort_flagged"
    entries = [json.loads(l) for l in
               (d / td.REPLICA_JOURNAL_NAME).read_text().splitlines()]
    assert all("k2_disagreement" in e for e in entries)
    assert all(e["k2_disagreement"]["abort"] is False for e in entries)
    main = [m for m, _p in
            __import__("attacker.captures", fromlist=["x"])
            .scan_captures(str(d), "training") if m["phase"] == "fwd"]
    assert len(main) == 2


@pytest.mark.slow
def test_torch_replica_converter_round_trip_and_attack(tiny, tmp_path):
    torch = tiny
    _rep_run(torch, tmp_path, "conv", steps=4, n_batches=4)
    from attacker.attacks import replica_integrity
    obs_p, truth_p = tmp_path / "obs.json", tmp_path / "truth.json"
    obs, scoring = td.capture_to_replica_bundle(
        str(tmp_path / "conv"), observations=str(obs_p),
        truth=str(truth_p))
    samples, targets = replica_integrity.load_inputs(str(obs_p),
                                                     str(truth_p))
    assert len(samples) == 4 and len(samples[0]["replicas"]) == 3
    assert len(set(samples[0]["key_ids"])) == 3
    assert len(samples[0]["replicas"][0]) == 8 * 32
    for arm in ("clean", "one-malicious"):
        rows = [replica_integrity.evaluate(s, targets[s["id"]], arm, 1.0,
                                           0.1, 0.01) for s in samples]
        assert all(r["accepted"] for r in rows), arm
        assert all(not r["availability_abort"] for r in rows), arm
    assert obs["schema"] == "llm_attacker.replica_observations.v1"
    assert scoring["schema"] == "llm_attacker.replica_truth.v1"


@pytest.mark.slow
def test_torch_replica_run_artifact_records_provenance(tiny, tmp_path):
    model = _tiny_model(tiny)
    split = _split(model)
    out = tmp_path / "run.json"
    cfg = splitbench.BenchConfig(split="1", steps=2, seq_len=8,
                                 batch_size=2, lr=1e-4,
                                 defense="replica-median",
                                 capture_dir=str(tmp_path / "caps"),
                                 output=str(out), seed=11, **REP_KW)
    writer = splitbench.CaptureWriter(cfg.capture_dir, "training", "art")
    splitbench.execute(cfg, split, writer, _batches(tiny, n=2))
    rec = json.loads(out.read_text())["results"][0]
    assert rec["defense"] == "replica-median"
    assert rec["defense_config"] == {"name": "replica-median",
                                     "replica_count": 3,
                                     "tolerance": 0.02, "gauge": True,
                                     "gauge_scale": 1e-3}
    prov = rec["replica_median_provenance"]
    assert str(prov).endswith(td.REPLICA_PROVENANCE_NAME)
    assert os.path.isfile(prov)
    assert math.isfinite(rec["final_loss"])


def test_v2_decoder_roundtrip_and_bounded_calibration_distortion(tmp_path):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(144)
    h = torch.randn(24, 8, generator=g) * 3 + 12
    cfg = td.BottleneckConfig(bottleneck_dim=4, encoder_hidden=8,
                              probe_hidden=8, minimax_steps=3, allow_cpu=True)
    tb = td.train_bottleneck(h, torch.arange(24) % 5, cfg, device='cpu')
    mean, basis, scale = (tb.geometry[k] for k in ('mean','basis','scale'))
    projected = (h-mean) @ basis @ basis.T + mean
    reconstructed = tb.decode(tb.apply(h))
    excess = ((reconstructed-h).square().sum(-1) - (projected-h).square().sum(-1))
    assert (excess <= cfg.bottleneck_dim * (.05*scale)**2 + 1e-4).all()
    tb.save(tmp_path)
    loaded = td.TrainedBottleneck.load(tmp_path)
    assert loaded.checkpoint_payload()['format']=='trained_bottleneck_encoder.v2'
    assert torch.equal(loaded.decode(loaded.apply(h)), reconstructed)
    assert torch.isfinite(reconstructed).all()
    assert (reconstructed-h).square().mean() < (tb.apply(h)-h).square().mean()


def test_config_accepts_pca_compress():
    cfg = splitbench.BenchConfig(defense="pca-compress")
    assert cfg.validate() is cfg
    assert cfg.pca_dim == 64
    assert (cfg.tb_cotrain, cfg.tb_cotrain_interval) == (False, 8)
    assert splitbench.BenchConfig(defense="trained-bottleneck",
                                  tb_cotrain=True,
                                  tb_cotrain_interval=4).validate()


@pytest.mark.parametrize("kw", [
    {"pca_dim": 0}, {"pca_dim": -1}, {"pca_dim": 2.5}, {"pca_dim": True},
    {"tb_cotrain": "yes"}, {"tb_cotrain": 1},
    {"tb_cotrain_interval": 0}, {"tb_cotrain_interval": -2},
    {"tb_cotrain_interval": 1.5},
])
def test_config_sweep_knob_drift_raises(kw):
    with pytest.raises(ValueError):
        splitbench.BenchConfig(defense="pca-compress", **kw).validate()
    with pytest.raises(ValueError):
        splitbench.BenchConfig(defense="trained-bottleneck",
                               **kw).validate()


def test_defense_registry_includes_pca_compress():
    assert "pca-compress" in splitbench._DEFENSES
    assert len(splitbench._DEFENSES) == 13


def test_defense_config_record_sweep_knobs():
    rec = splitbench._defense_config_record(
        splitbench.BenchConfig(defense="pca-compress", pca_dim=16))
    assert rec == {"name": "pca-compress", "bottleneck_dim": 16}
    rec = splitbench._defense_config_record(
        splitbench.BenchConfig(defense="trained-bottleneck",
                               tb_cotrain=True, tb_cotrain_interval=4))
    assert rec["cotrain"] is True and rec["cotrain_interval"] == 4
    rec = splitbench._defense_config_record(
        splitbench.BenchConfig(defense="trained-bottleneck"))
    assert rec["cotrain"] is False
    for k in (64, 128, 256):
        assert splitbench.BenchConfig(defense="trained-bottleneck",
                                      tb_dim=k).validate().tb_dim == k
        assert splitbench.BenchConfig(defense="pca-compress",
                                      pca_dim=k).validate().pca_dim == k


def test_pca_config_validation_and_hash():
    cfg = td.PcaConfig(bottleneck_dim=16, seed=7)
    assert cfg.validate() is cfg
    assert cfg.config_sha256() == \
        td.PcaConfig(**cfg.canonical()).config_sha256()
    assert cfg.config_sha256() != \
        td.PcaConfig(bottleneck_dim=16, seed=8).config_sha256()
    with pytest.raises(ValueError):
        td.PcaConfig(bottleneck_dim=0).validate()
    with pytest.raises(ValueError):
        td.PcaConfig(seed=-1).validate()


def _pca_prov_record(**over):
    pca = td.PcaCompress(td.PcaConfig(bottleneck_dim=2, seed=7).canonical(),
                         seam_width=4, mean=None, basis=None,
                         explained_variance_ratio=[0.5, 0.25, 0.15, 0.1],
                         explained_variance_at_k=0.75)
    pca.utility = {
        "utility_before": {"cross_entropy": 4.5, "perplexity": 90.0},
        "utility_after": {"cross_entropy": 4.6, "perplexity": 99.0}}
    rec = pca.provenance_record("/tmp/x/" + td.PCA_ENCODER_NAME, "c" * 64)
    rec.update(over)
    return rec


def test_pca_provenance_schema_complete():
    rec = td.validate_pca_provenance(_pca_prov_record())
    assert rec["schema"] == td.PCA_PROVENANCE_SCHEMA
    assert rec["explained_variance_at_k"] == 0.75
    assert len(rec["explained_variance_ratio"]) == 4
    assert rec["seed"] == 7 and rec["bottleneck_dim"] == 2
    assert rec["checkpoint"]["path"].endswith(td.PCA_ENCODER_NAME)
    assert "no adversarial training" in rec["training"]


@pytest.mark.parametrize("mut", [
    lambda r: r.update(schema="representation_privacy.v1"),
    lambda r: r.pop("explained_variance_at_k"),
    lambda r: r.update(config_sha256="0" * 64),
    lambda r: r.update(explained_variance_at_k=1.5),
    lambda r: r.update(explained_variance_at_k=float("nan")),
    lambda r: r.update(explained_variance_at_k=0.5),
    lambda r: r.update(explained_variance_ratio=[0.5, 0.25]),
    lambda r: r.update(training="adversarial minimax"),
    lambda r: r["checkpoint"].update(sha256="xyz"),
    lambda r: r["utility"].update(utility_after={"cross_entropy": "x"}),
])
def test_pca_provenance_drift_raises(mut):
    rec = _pca_prov_record()
    mut(rec)
    with pytest.raises(ValueError):
        td.validate_pca_provenance(rec)


def test_tb_provenance_cotrain_block_validation():
    rec = _prov_record()
    rec["cotrain"] = {"enabled": True, "interval": 8,
                      "updates_performed": 6}
    td.validate_provenance(rec)
    for bad in ({"enabled": "yes", "interval": 8, "updates_performed": 6},
                {"enabled": True, "interval": 0, "updates_performed": 6},
                {"enabled": True, "interval": 8, "updates_performed": 0},
                {"enabled": True, "interval": 8,
                 "updates_performed": -1}):
        rec2 = _prov_record()
        rec2["cotrain"] = bad
        with pytest.raises(ValueError):
            td.validate_provenance(rec2)


PCA_KW = dict(pca_dim=8)


def _pca_run(torch, tmp_path, name, steps=3, save_canonical=False):
    model = _tiny_model(torch)
    split = _split(model)
    writer = splitbench.CaptureWriter(str(tmp_path / name), "training",
                                      name)
    cfg = splitbench.BenchConfig(steps=steps, seq_len=8, batch_size=2,
                                 lr=1e-4, defense="pca-compress",
                                 seed=7, save_canonical=save_canonical,
                                 **PCA_KW)
    return splitbench.finetune(split, cfg, _batches(torch, n=steps),
                               writer)


@pytest.mark.slow
def test_torch_pca_math_matches_svd_reference(tiny):
    torch = tiny
    g = torch.Generator().manual_seed(5)
    rows = torch.randn(64, 32, generator=g)
    cfg = td.PcaConfig(bottleneck_dim=8, seed=3)
    pca = td.fit_pca_compressor(rows, cfg, device="cpu")
    mean = rows.double().mean(0)
    _, s, vh = torch.linalg.svd(rows.double() - mean,
                                full_matrices=True)
    for j in range(8):
        col = pca.basis[:, j].double()
        ref = vh[j]
        assert torch.allclose(col, ref, atol=1e-5) or \
            torch.allclose(col, -ref, atol=1e-5)
    ev = (s.square() / s.square().sum())
    assert abs(pca.explained_variance_at_k
               - float(ev[:8].sum())) < 1e-5
    assert len(pca.explained_variance_ratio) == 32
    h = torch.randn(4, 32, generator=g)
    resid = (pca.decode(pca.apply(h)) - (h - pca.mean)).square().sum() \
        / (h - pca.mean).square().sum()
    assert resid.item() >= 0.0
    pca2 = td.fit_pca_compressor(rows.clone(), cfg, device="cpu")
    assert torch.equal(pca.basis, pca2.basis) \
        and torch.equal(pca.mean, pca2.mean)


@pytest.mark.slow
def test_torch_pca_apply_straight_through(tiny):
    torch = tiny
    g = torch.Generator().manual_seed(7)
    rows = torch.randn(32, 32, generator=g)
    pca = td.fit_pca_compressor(rows, td.PcaConfig(bottleneck_dim=8),
                                device="cpu")
    h = torch.randn(2, 5, 32, generator=g, requires_grad=True)
    out = pca.apply(h)
    assert out.shape == h.shape
    assert out[..., 8:].abs().max().item() == 0.0
    assert torch.allclose(out[..., :8], pca.encode(h.detach()),
                          atol=1e-5)
    upstream = torch.randn(2, 5, 32, generator=g)
    out.backward(upstream)
    assert torch.allclose(h.grad, upstream)


@pytest.mark.slow
def test_torch_pca_splitbench_run_and_converter(tiny, tmp_path):
    torch = tiny
    stats = _pca_run(torch, tmp_path, "pca", steps=8, save_canonical=True)
    prov_path = tmp_path / "pca" / td.PCA_PROVENANCE_NAME
    assert prov_path.is_file()
    rec = td.validate_pca_provenance(json.loads(prov_path.read_text()))
    assert rec["bottleneck_dim"] == 8
    assert 0.0 <= rec["explained_variance_at_k"] <= 1.0
    for s in ("utility_before", "utility_after"):
        u = rec["utility"][s]
        assert math.isfinite(u["cross_entropy"]) and u["perplexity"] > 1.0
    assert stats["pca_compress_provenance"]["provenance_path"] \
        == str(prov_path)
    assert math.isfinite(stats["final_loss"])
    from attacker.attacks import representation_privacy
    out = tmp_path / "bundle.json"
    bundle = td.capture_to_representation_bundle(str(tmp_path / "pca"),
                                                 output=str(out))
    shape = representation_privacy.validate_bundle(
        json.loads(out.read_text()))
    assert shape == (8, 8)
    assert "pca-compress" in bundle["provenance"]["defender"]
    model = _tiny_model(torch)
    split = _split(model)
    cfg = splitbench.BenchConfig(defense="pca-compress", **PCA_KW)
    with pytest.raises(ValueError, match="calibration"):
        splitbench.make_defense_state(cfg, split)


@pytest.mark.slow
def test_torch_cotrain_updates_weights_and_is_deterministic(tiny,
                                                            tmp_path):
    torch = tiny
    kw = dict(TB_KW, tb_cotrain=True, tb_cotrain_interval=2)

    def run(name):
        model = _tiny_model(torch)
        split = _split(model)
        writer = splitbench.CaptureWriter(str(tmp_path / name),
                                          "training", name)
        cfg = splitbench.BenchConfig(steps=6, seq_len=8, batch_size=2,
                                     lr=1e-4, defense="trained-bottleneck",
                                     seed=7, save_canonical=True, **kw)
        return splitbench.finetune(split, cfg, _batches(torch, n=6),
                                   writer)

    s1, s2 = run("ct1"), run("ct2")
    from attacker import captures

    def ckpt(name):
        return torch.load(tmp_path / name / td.ENCODER_NAME,
                          weights_only=False)["state_dict"]

    c1, c2 = ckpt("ct1"), ckpt("ct2")
    assert all(torch.equal(c1[k], c2[k]) for k in c1)
    for f in ("wire_00000.pt", "wire_00002.pt"):
        assert torch.equal(captures.load_tensor(str(tmp_path / "ct1" / f)),
                           captures.load_tensor(str(tmp_path / "ct2" / f)))
    _run(torch, tmp_path, "fr", steps=6)
    frozen = ckpt("fr")
    assert any(not torch.equal(c1[k], frozen[k]) for k in c1)
    rec = td.validate_provenance(json.loads(
        (tmp_path / "ct1" / td.PROVENANCE_NAME).read_text()))
    assert rec["cotrain"] == {"enabled": True, "interval": 2,
                              "updates_performed": 3}
    assert math.isfinite(s1["final_loss"]) and s1["final_loss"] == \
        s2["final_loss"]


@pytest.mark.slow
def test_torch_cotrain_gradient_flow_preserved(tiny):
    torch = tiny
    g = torch.Generator().manual_seed(5)
    h_cal = torch.randn(32, 32, generator=g)
    t_cal = torch.randint(0, 99, (32,), generator=g)
    cfg = td.BottleneckConfig(bottleneck_dim=8, encoder_hidden=16,
                              probe_hidden=16, minimax_steps=3, seed=7,
                              allow_cpu=True)
    tb = td.train_bottleneck(h_cal, t_cal, cfg, device="cpu")
    ct = td.BottleneckCotrain(tb, cfg, "cpu", interval=2)
    before = {k: v.clone() for k, v in tb.state_dict.items()}
    rec = ct.update(h_cal, torch.randint(0, 99, (32,), generator=g))
    assert rec["update_index"] == 1
    assert any(not torch.equal(before[k], tb.state_dict[k])
               for k in before)
    h = torch.randn(2, 5, 32, generator=g, requires_grad=True)
    out = tb.apply(h)
    assert out.shape == h.shape and out[..., 8:].abs().max().item() == 0.0
    out.backward(torch.randn(2, 5, 32, generator=g))
    assert h.grad is not None and torch.isfinite(h.grad).all()
    torch.manual_seed(123)
    expected = torch.randn(4)
    torch.manual_seed(123)
    ct.update(h_cal, torch.randint(0, 99, (32,), generator=g))
    assert torch.equal(torch.randn(4), expected)


@pytest.mark.slow
@pytest.mark.parametrize("k", [8, 16, 32])
def test_torch_tb_dim_values_train(tiny, k):
    torch = tiny
    g = torch.Generator().manual_seed(5)
    h_cal = torch.randn(32, 32, generator=g)
    t_cal = torch.randint(0, 99, (32,), generator=g)
    cfg = td.BottleneckConfig(bottleneck_dim=k, encoder_hidden=16,
                              probe_hidden=16, minimax_steps=1, seed=7,
                              allow_cpu=True)
    tb = td.train_bottleneck(h_cal, t_cal, cfg, device="cpu")
    assert tb.bottleneck_dim == k
    assert tb.apply(torch.randn(2, 4, 32, generator=g)).shape == (2, 4, 32)
    with pytest.raises(ValueError, match="exceeds the seam width"):
        td.train_bottleneck(h_cal, t_cal,
                            td.BottleneckConfig(bottleneck_dim=64,
                                                minimax_steps=1,
                                                allow_cpu=True),
                            device="cpu")
