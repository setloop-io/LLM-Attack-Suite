"""VFLAIR-analogue seam-defense pack tests.

Torch-free layer (runs everywhere): BenchConfig validation matrix for the
pack knobs, the defenses.py pack registration (describe_scenarios /
validate_defense_config), SeamPcaState math on tiny pure-python inputs,
cosine_neighbors, TokenSanitizer determinism with a fake embedding table,
and _defense_config_record.

Torch layer (slow, self-skips without torch/transformers; tiny random
GPT-2 2L/32/99 on CPU): each pack defense (a) changes the captured fwd
seam tensor vs "none" at the same seed/data and (b) keeps the loss finite
over 3 finetune steps; plus the defense-specific semantics:
(c) sparsify keeps EXACTLY the top-k% per row,
(d) santext's canonical.pt holds the ORIGINAL ids while the captured
    seam differs,
(e) mid-lite seam content is rank-limited (numerical rank <= k+1),
(f) at-lite's second-step perturbation correlates with the first step's
    top-PCA direction.
"""

import json
import math

import pytest

from attacker import captures, defenses, splitbench


@pytest.mark.parametrize("kw", [
    {"defense": "sparsify"},
    {"defense": "sparsify", "sparsity": 1.0},
    {"defense": "sparsify", "sparsity": 0.01},
    {"defense": "mid-lite", "bottleneck_dim": 1},
    {"defense": "at-lite", "at_eps": 0.0},
    {"defense": "santext", "santext_p": 0.0, "santext_k": 1},
    {"defense": "santext", "santext_p": 1.0},
    {"defense": "custext"},
    {"defense": "snd-lite", "snd_sigma": 0.0},
    {"defense": "gaussian", "sigma": 0.3},
])
def test_pack_config_valid(kw):
    cfg = splitbench.BenchConfig(**kw)
    assert cfg.validate() is cfg


@pytest.mark.parametrize("kw", [
    {"defense": "vflair-mid"}, {"defense": "MID"}, {"defense": "sp"},
    {"sparsity": 0.0}, {"sparsity": -0.1}, {"sparsity": 1.5},
    {"sparsity": "0.5"}, {"sparsity": True},
    {"bottleneck_dim": 0}, {"bottleneck_dim": -3},
    {"bottleneck_dim": 2.5}, {"bottleneck_dim": True},
    {"at_eps": -0.1}, {"at_eps": "x"},
    {"santext_p": -0.1}, {"santext_p": 1.5}, {"santext_p": "p"},
    {"santext_k": 0}, {"santext_k": -1}, {"santext_k": 1.5},
    {"snd_sigma": -0.5}, {"snd_sigma": "s"},
])
def test_pack_config_drift_raises(kw):
    with pytest.raises(ValueError):
        splitbench.BenchConfig(**kw).validate()


def test_pack_knob_defaults():
    cfg = splitbench.BenchConfig()
    assert (cfg.sparsity, cfg.bottleneck_dim, cfg.at_eps, cfg.santext_p,
            cfg.santext_k, cfg.snd_sigma) == (0.1, 8, 0.1, 0.1, 8, 1.0)


def test_defenses_py_default_matrix_unchanged():
    scenarios = defenses.describe_scenarios()
    assert [s["defense"] for s in scenarios] == \
        ["plain-split", "rotation", "dp-noise", "composite"]
    assert defenses.describe_scenarios() == scenarios


def test_defenses_py_pack_appended_and_validates():
    pack = defenses.describe_scenarios(include_seam_pack=True)
    assert len(pack) == 10
    assert pack[:4] == defenses.describe_scenarios()
    assert [s["defense"] for s in pack[4:]] == \
        ["sparsify", "mid-lite", "at-lite", "santext", "custext",
         "snd-lite"]
    for s in pack[4:]:
        assert s["attack"] == "seam-inversion"
        assert s["surface"] == "splitbench-seam"
        assert s["expected_verdict"] is None
        assert s["why"] and s["description"]
        assert defenses.validate_defense_config(s["defense"],
                                                s["config"]) == []
    pack[4]["config"]["sparsity"] = -1
    assert defenses.describe_scenarios(include_seam_pack=True)[4] \
        ["config"]["sparsity"] == 0.1


@pytest.mark.parametrize("name,cfg", [
    ("sparsify", {"sprsity": 0.2}),
    ("sparsify", {"sparsity": 0.0}),
    ("sparsify", {"sparsity": 1.5}),
    ("sparsify", {"sparsity": True}),
    ("sparsify", {"sigma": 0.1}),
    ("mid-lite", {"bottleneck_dim": 0}),
    ("mid-lite", {"bottleneck_dim": 1.5}),
    ("mid-lite", {"bottleneck_dim": True}),
    ("at-lite", {"at_eps": -0.1}),
    ("santext", {"santext_p": 1.5}),
    ("santext", {"santext_k": 0}),
    ("custext", {"santext_k": -2}),
    ("custext", {"santext_p": 0.5}),
    ("snd-lite", {"snd_sigma": -1.0}),
    ("snd-lite", {"snd_sigma": "loud"}),
])
def test_defenses_py_pack_validation_rejections(name, cfg):
    assert defenses.validate_defense_config(name, cfg)


def test_defenses_py_pack_validation_accepts_partials():
    assert defenses.validate_defense_config("sparsify",
                                            {"sparsity": 0.3}) == []
    assert defenses.validate_defense_config("santext",
                                            {"santext_k": 4}) == []
    assert defenses.validate_defense_config("custext", {}) == []


def test_defenses_py_unknown_names():
    assert "unknown defense" in \
        defenses.validate_defense_config("nope", {})[0]
    assert "seam pack" in defenses.validate_defense_config("nope", {})[0]
    assert "unknown seam defense" in \
        defenses.validate_seam_defense_config("mid", {})[0]


def test_seam_pca_state_top_direction_math():
    rows = [[0.05 * ((-1) ** i), float(i - 5)] for i in range(11)]
    st = splitbench.SeamPcaState(2)
    assert st.top_direction() is None
    st.update(rows)
    v = st.top_direction()
    assert v[1] > 0.999 and abs(v[0]) < 0.01
    assert abs(sum(x * x for x in v) - 1.0) < 1e-9
    assert st.n_updates == 1


def test_seam_pca_state_determinism_and_ema_blend():
    rows = [[2.0, 0.1], [1.9, -0.1], [2.1, 0.05], [-2.0, 0.0]]
    a, b = splitbench.SeamPcaState(2), splitbench.SeamPcaState(2)
    a.update(rows)
    b.update(rows)
    assert a.top_direction() == b.top_direction()
    blend = splitbench.SeamPcaState(2, momentum=0.75)
    blend.update_cov([[4.0, 0.0], [0.0, 1.0]])
    blend.update_cov([[0.0, 0.0], [0.0, 3.0]])
    assert blend.cov == [[3.0, 0.0], [0.0, 1.5]]
    assert blend.n_updates == 2
    zero = splitbench.SeamPcaState(2)
    zero.update_cov([[0.0, 0.0], [0.0, 0.0]])
    assert zero.top_direction() is None


def test_seam_pca_state_validation():
    with pytest.raises(ValueError, match="dim"):
        splitbench.SeamPcaState(0)
    with pytest.raises(ValueError, match="momentum"):
        splitbench.SeamPcaState(2, momentum=1.0)
    with pytest.raises(ValueError, match=">= 2 rows"):
        splitbench.SeamPcaState(2).update([[1.0, 2.0]])
    with pytest.raises(ValueError, match="width"):
        splitbench.SeamPcaState(2).update([[1.0], [2.0]])
    with pytest.raises(ValueError, match="2x2"):
        splitbench.SeamPcaState(2).update_cov([[1.0, 0.0, 0.0]])


FAKE_TABLE = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]


def _fake_neighbors(k=3):
    return splitbench.cosine_neighbors(FAKE_TABLE, k)


def test_cosine_neighbors_hand_computed():
    assert splitbench.cosine_neighbors(FAKE_TABLE, 1) == \
        [[1], [0], [1], [2]]
    n2 = splitbench.cosine_neighbors(FAKE_TABLE, 2)
    assert n2[0] == [1, 2] and n2[3] == [2, 1]
    assert splitbench.cosine_neighbors(FAKE_TABLE, 99)[0] == [1, 2, 3]


def test_token_sanitizer_santext_determinism_fake_table():
    nbrs = _fake_neighbors()
    ids = [[0, 1, 2, 3], [3, 2, 1, 0]]
    a = splitbench.TokenSanitizer("santext", lambda t: nbrs[t], p=0.5,
                                  k=3, seed=11)
    b = splitbench.TokenSanitizer("santext", lambda t: nbrs[t], p=0.5,
                                  k=3, seed=11)
    out_a, out_b = a.sanitize(ids), b.sanitize(ids)
    assert out_a == out_b
    for r, orow in zip(ids, out_a):
        for t, o in zip(r, orow):
            assert o == t or o in nbrs[t]
    z = splitbench.TokenSanitizer("santext", lambda t: nbrs[t], p=0.0,
                                  k=3, seed=11)
    assert z.sanitize(ids) == ids
    one = splitbench.TokenSanitizer("santext", lambda t: nbrs[t], p=1.0,
                                    k=3, seed=11)
    out1 = one.sanitize(ids)
    assert all(o != t and o in nbrs[t]
               for r, orow in zip(ids, out1) for t, o in zip(r, orow))
    other = splitbench.TokenSanitizer("santext", lambda t: nbrs[t],
                                      p=1.0, k=3, seed=12)
    assert other.sanitize(ids) != out1


def test_token_sanitizer_custext_fixed_mapping_fake_table():
    nbrs = _fake_neighbors()
    a = splitbench.TokenSanitizer("custext", lambda t: nbrs[t], p=0.0,
                                  k=3, seed=11)
    b = splitbench.TokenSanitizer("custext", lambda t: nbrs[t], p=0.0,
                                  k=3, seed=11)
    oa = a.sanitize([[0, 1], [2, 3]])
    ob = b.sanitize([[3, 2], [1, 0]])
    mapping = [oa[0][0], oa[0][1], oa[1][0], oa[1][1]]
    assert mapping == [ob[1][1], ob[1][0], ob[0][1], ob[0][0]]
    assert all(mapping[t] != t and mapping[t] in nbrs[t]
               for t in range(4))
    assert a.sanitize([[3, 0, 2, 1]]) == \
        [[mapping[3], mapping[0], mapping[2], mapping[1]]]
    fresh = splitbench.TokenSanitizer("custext", lambda t: nbrs[t],
                                      p=0.0, k=3, seed=11)
    assert fresh.sanitize([[0, 1, 2, 3]]) == [mapping]


def test_token_sanitizer_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        splitbench.TokenSanitizer("santex", lambda t: [], p=0.5, k=1,
                                  seed=0)


def test_defense_config_record_knobs():
    assert splitbench._defense_config_record(
        splitbench.BenchConfig(defense="santext", santext_p=0.3,
                               santext_k=5)) == \
        {"name": "santext", "santext_p": 0.3, "santext_k": 5}
    assert splitbench._defense_config_record(
        splitbench.BenchConfig(defense="mid-lite", bottleneck_dim=3)) == \
        {"name": "mid-lite", "bottleneck_dim": 3}
    for d in splitbench._DEFENSES:
        assert splitbench._defense_config_record(
            splitbench.BenchConfig(defense=d))["name"] == d


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


PACK_RUNS = [
    ("sparsify", {"sparsity": 0.5}),
    ("mid-lite", {"bottleneck_dim": 8}),
    ("at-lite", {"at_eps": 0.5}),
    ("santext", {"santext_p": 1.0, "santext_k": 3}),
    ("custext", {"santext_k": 3}),
    ("snd-lite", {"snd_sigma": 0.5}),
]


@pytest.mark.slow
@pytest.mark.parametrize("defense,kw", PACK_RUNS)
def test_torch_pack_changes_wire_and_keeps_loss_finite(tiny, tmp_path,
                                                       defense, kw):
    torch = tiny
    _, none_losses = _run(torch, tmp_path, "none", "none")
    _, losses = _run(torch, tmp_path, defense, defense, **kw)
    none_fwd = captures.load_tensor(
        str(tmp_path / "none" / "wire_00000.pt"))
    fwd = captures.load_tensor(str(tmp_path / defense / "wire_00000.pt"))
    assert fwd.shape == none_fwd.shape
    assert not torch.equal(none_fwd, fwd)
    assert len(losses) == 3
    assert all(math.isfinite(x) for x in losses)
    assert all(math.isfinite(x) for x in none_losses)


@pytest.mark.slow
def test_torch_sparsify_keeps_exactly_topk_per_row(tiny):
    torch = tiny
    g = torch.Generator().manual_seed(3)
    hidden = torch.randn(2, 7, 32, generator=g)
    cfg = splitbench.BenchConfig(defense="sparsify", sparsity=0.25)
    out = splitbench._apply_seam_defense(hidden, cfg)
    kept = max(1, round(0.25 * 32))
    mask = out != 0
    assert (mask.sum(dim=-1) == kept).all()
    assert torch.equal(out[mask], hidden[mask])
    kept_min = hidden.abs().masked_fill(~mask, float("inf")).amin(-1)
    zeroed_max = hidden.abs().masked_fill(mask, 0.0).amax(-1)
    assert (kept_min >= zeroed_max).all()
    cfg1 = splitbench.BenchConfig(defense="sparsify", sparsity=1.0)
    assert torch.equal(splitbench._apply_seam_defense(hidden, cfg1),
                       hidden)


@pytest.mark.slow
def test_torch_santext_canonical_keeps_original_ids(tiny, tmp_path):
    torch = tiny
    batches = _batches(torch, n=3)
    model = _tiny_model(torch)
    split = _split(model)
    writer = splitbench.CaptureWriter(str(tmp_path / "san"), "training",
                                      "san")
    cfg = splitbench.BenchConfig(steps=3, seq_len=8, batch_size=2,
                                 lr=1e-4, defense="santext",
                                 santext_p=1.0, santext_k=3, seed=7,
                                 save_canonical=True)
    splitbench.finetune(split, cfg, batches, writer)
    canonical = torch.load(tmp_path / "san" / "canonical.pt",
                           weights_only=True)
    assert canonical.dtype == torch.long
    assert torch.equal(canonical, torch.cat(batches))
    _run(torch, tmp_path, "none2", "none")
    none_fwd = captures.load_tensor(
        str(tmp_path / "none2" / "wire_00000.pt"))
    san_fwd = captures.load_tensor(str(tmp_path / "san" / "wire_00000.pt"))
    assert not torch.equal(none_fwd, san_fwd)


@pytest.mark.slow
def test_torch_mid_lite_wire_is_rank_limited(tiny, tmp_path):
    torch = tiny
    _run(torch, tmp_path, "mid", "mid-lite", bottleneck_dim=4)
    fwd = captures.load_tensor(str(tmp_path / "mid" / "wire_00000.pt"))
    assert fwd.shape == (2, 8, 32)
    flat = fwd.reshape(-1, 32).float()
    assert int(torch.linalg.matrix_rank(flat)) <= 4 + 1
    assert fwd[..., 4:].abs().max().item() == 0.0


@pytest.mark.slow
def test_torch_mid_lite_projection_is_seeded_reproducible(tiny, tmp_path):
    torch = tiny
    _run(torch, tmp_path, "m1", "mid-lite", steps=1, bottleneck_dim=8)
    _run(torch, tmp_path, "m2", "mid-lite", steps=1, bottleneck_dim=8)
    a = captures.load_tensor(str(tmp_path / "m1" / "wire_00000.pt"))
    b = captures.load_tensor(str(tmp_path / "m2" / "wire_00000.pt"))
    assert torch.equal(a, b)


@pytest.mark.slow
def test_torch_mid_lite_wider_than_seam_raises(tiny):
    torch = tiny
    hidden = torch.randn(1, 2, 32)
    cfg = splitbench.BenchConfig(defense="mid-lite", bottleneck_dim=64)
    with pytest.raises(ValueError, match="bottleneck_dim"):
        splitbench._apply_seam_defense(hidden, cfg)


@pytest.mark.slow
def test_torch_at_lite_perturbation_tracks_top_pca(tiny):
    torch = tiny
    model = _tiny_model(torch)
    split = _split(model)
    batches = _batches(torch, n=2)
    with torch.no_grad():
        h0 = split.head_forward(batches[0])
        h1 = split.head_forward(batches[1])
    cfg = splitbench.BenchConfig(defense="at-lite", at_eps=0.5)
    state = {}
    w0 = splitbench._apply_seam_defense(h0, cfg, state)
    w1 = splitbench._apply_seam_defense(h1, cfg, state)
    p0, p1 = h0 - w0, h1 - w1
    assert torch.allclose(p0, p0[0:1, 0:1].expand_as(p0))
    v0 = p0[0, 0] / 0.5
    v1 = p1[0, 0] / 0.5
    assert math.isclose(v0.norm().item(), 1.0, abs_tol=1e-4)
    ref = splitbench.SeamPcaState(32)
    ref.update(h0.reshape(-1, 32).tolist())
    ref_v = torch.tensor(ref.top_direction(), dtype=torch.float32)
    assert torch.dot(v0, ref_v).item() > 0.999
    assert torch.dot(v0, v1).item() > 0.5
    assert state["at"].pca.n_updates == 2


@pytest.mark.slow
def test_torch_at_lite_eps_zero_is_identity(tiny):
    torch = tiny
    hidden = torch.randn(2, 3, 32,
                         generator=torch.Generator().manual_seed(4))
    cfg = splitbench.BenchConfig(defense="at-lite", at_eps=0.0)
    state = {}
    out = splitbench._apply_seam_defense(hidden, cfg, state)
    assert torch.equal(out, hidden)
    assert state["at"].pca.n_updates == 1


@pytest.mark.slow
def test_torch_pack_run_artifact_records_defense_config(tiny, tmp_path):
    torch = tiny
    model = _tiny_model(torch)
    split = _split(model)
    out = tmp_path / "run.json"
    cfg = splitbench.BenchConfig(split="1", steps=2, seq_len=8,
                                 batch_size=2, lr=1e-4, defense="sparsify",
                                 sparsity=0.5,
                                 capture_dir=str(tmp_path / "caps"),
                                 output=str(out), seed=11)
    writer = splitbench.CaptureWriter(cfg.capture_dir, "training", "art")
    splitbench.execute(cfg, split, writer, _batches(torch, n=2))
    art = json.loads(out.read_text())
    rec = art["results"][0]
    assert rec["defense"] == "sparsify"
    assert rec["defense_config"] == {"name": "sparsify", "sparsity": 0.5}
    assert art["config"]["sparsity"] == 0.5
    assert math.isfinite(rec["final_loss"])
