"""attacker.checkpoint tests + the seam-inversion --resume reference
pattern (the kill simulation lives here because tests/test_seam_inversion.py
is owned elsewhere; the checkpointer tests are torch-free).

Torch-free layer (runs everywhere): save/resume roundtrip, cadence gating
(step 0 always checkpointed), latest-step selection over mtime, keep-K
pruning, crash-safety (an orphan .tmp from a killed write is invisible to
latest() and cleaned by the next save), empty-dir resume -> None, one
named stream per subdirectory, pluggable saver/loader, validation drift,
and the torch-free-only edges (a .pt checkpoint without torch is a
descriptive error; a non-JSON-serializable state without torch is too).

Torch layer (self-skips without torch/transformers, marked slow): the
default .pt format round-trips tensor state, and the seam-inversion
vmi-head optimizer loop resumes mid-loop after a simulated kill — its
final state equals the uninterrupted run exactly.
"""

import importlib.util
import json
import os
import types

import pytest

from attacker.checkpoint import Checkpointer, self_test

TORCH_PRESENT = importlib.util.find_spec("torch") is not None


def test_save_resume_roundtrip(tmp_path):
    ck = Checkpointer(str(tmp_path), "vmi", 1)
    ck.save(0, {"step": 0, "loss": 1.5})
    ck.save(1, {"step": 1, "loss": 1.25})
    assert ck.latest() == (1, {"step": 1, "loss": 1.25})
    assert ck.resume() == {"step": 1, "loss": 1.25}


def test_checkpoint_layout_is_dir_name_ckpt_step(tmp_path):
    ck = Checkpointer(str(tmp_path), "cell-3", 1)
    path = ck.save(7, {"x": 1})
    assert ck.dir == os.path.join(str(tmp_path), "cell-3")
    assert os.path.basename(path) == f"ckpt_000007{ck.ext}"
    assert os.path.isfile(path)


def test_named_streams_share_a_root_without_colliding(tmp_path):
    a = Checkpointer(str(tmp_path), "cell-a", 1)
    b = Checkpointer(str(tmp_path), "cell-b", 1)
    a.save(0, {"who": "a"})
    b.save(9, {"who": "b"})
    assert a.latest() == (0, {"who": "a"})
    assert b.latest() == (9, {"who": "b"})


def test_cadence_gating_and_step_zero_always_saved(tmp_path):
    ck = Checkpointer(str(tmp_path), "gated", 3)
    assert ck.save(1, {"x": 1}) is None
    assert ck.save(2, {"x": 2}) is None
    assert ck.latest() is None
    assert ck.save(0, {"x": 0}) is not None
    assert ck.save(4, {"x": 4}) is None
    assert ck.save(3, {"x": 3}) is not None
    assert ck.latest() == (3, {"x": 3})


def test_latest_prefers_highest_step_over_newest_mtime(tmp_path):
    ck = Checkpointer(str(tmp_path), "mtime", 1, keep=5)
    ck.save(9, {"step": 9})
    ck.save(3, {"step": 3})
    assert ck.latest() == (9, {"step": 9})


def test_keep_prunes_to_last_k(tmp_path):
    ck = Checkpointer(str(tmp_path), "prune", 1, keep=2)
    for s in range(5):
        ck.save(s, {"step": s})
    assert sorted(os.listdir(ck.dir)) == [f"ckpt_000003{ck.ext}",
                                          f"ckpt_000004{ck.ext}"]
    assert ck.latest() == (4, {"step": 4})


def test_empty_dir_resume_returns_none(tmp_path):
    ck = Checkpointer(str(tmp_path), "empty", 1)
    assert ck.latest() is None
    assert ck.resume() is None


def test_crash_safety_orphan_tmp_invisible_and_cleaned(tmp_path):
    ck = Checkpointer(str(tmp_path), "crash", 1)
    ck.save(1, {"ok": True})
    orphan = os.path.join(ck.dir, f"ckpt_000009{ck.ext}.tmp")
    with open(orphan, "w") as f:
        f.write("{torn")
    assert ck.latest() == (1, {"ok": True})
    ck.save(2, {"ok": 2})
    assert not any(f.endswith(".tmp") for f in os.listdir(ck.dir))
    assert ck.latest() == (2, {"ok": 2})


def test_save_leaves_no_tmp_files(tmp_path):
    ck = Checkpointer(str(tmp_path), "clean", 1)
    ck.save(0, {"x": 0})
    assert not any(".tmp" in f for f in os.listdir(ck.dir))


def test_forced_json_format_is_plain_json_on_disk(tmp_path):
    ck = Checkpointer(str(tmp_path), "json", 1, ext=".json")
    path = ck.save(0, {"nested": {"a": [1, 2, 3]}})
    with open(path) as f:
        assert json.load(f) == {"nested": {"a": [1, 2, 3]}}


def test_pluggable_saver_and_loader(tmp_path):
    def fake_saver(path, state):
        with open(path, "wb") as f:
            f.write(b"FAKE:" + repr(state).encode())

    def fake_loader(path):
        with open(path, "rb") as f:
            return f.read().decode()

    ck = Checkpointer(str(tmp_path), "fake", 1, saver=fake_saver,
                      loader=fake_loader)
    ck.save(0, ("z", 1))
    with open(os.path.join(ck.dir, f"ckpt_000000{ck.ext}"), "rb") as f:
        assert f.read().startswith(b"FAKE:")
    assert ck.resume().startswith("FAKE:")


@pytest.mark.parametrize("bad", ["", "a/b", "..", "a\\b", "."])
def test_name_must_be_a_single_path_component(tmp_path, bad):
    with pytest.raises(ValueError, match="name"):
        Checkpointer(str(tmp_path), bad, 1)


@pytest.mark.parametrize("kw", [{"every": 0}, {"every": -1}, {"every": 1.5},
                                {"every": True}])
def test_bad_cadence_raises(tmp_path, kw):
    with pytest.raises(ValueError, match="every"):
        Checkpointer(str(tmp_path), "x", **kw)


def test_bad_keep_and_ext_and_step_raise(tmp_path):
    with pytest.raises(ValueError, match="keep"):
        Checkpointer(str(tmp_path), "x", 1, keep=0)
    with pytest.raises(ValueError, match="ext"):
        Checkpointer(str(tmp_path), "x", 1, ext=".bin")
    ck = Checkpointer(str(tmp_path), "x", 1)
    for bad_step in (-1, 1.5, True, "3"):
        with pytest.raises(ValueError, match="step"):
            ck.save(bad_step, {})


def test_default_extension_follows_torch_availability(tmp_path):
    ck = Checkpointer(str(tmp_path), "e", 1)
    assert ck.ext == (".pt" if TORCH_PRESENT else ".json")


@pytest.mark.skipif(TORCH_PRESENT, reason="torch-free only edge")
def test_pt_checkpoint_without_torch_is_a_descriptive_error(tmp_path):
    ck = Checkpointer(str(tmp_path), "pt", 1)
    with open(os.path.join(ck.dir, "ckpt_000007.pt"), "wb") as f:
        f.write(b"not-a-real-payload")
    with pytest.raises(ValueError, match="torch"):
        ck.latest()


@pytest.mark.skipif(TORCH_PRESENT, reason="torch-free only edge")
def test_non_json_state_without_torch_is_a_descriptive_error(tmp_path):
    ck = Checkpointer(str(tmp_path), "json", 1)
    with pytest.raises(ValueError, match="JSON-serializable"):
        ck.save(0, {"tensor": object()})


def test_module_self_test_passes(capsys):
    assert self_test() == 0
    assert "SELF-TEST PASSED" in capsys.readouterr().out


@pytest.fixture
def tiny_head():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel
    from attacker import splitbench
    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=2, n_embd=32, n_head=4, vocab_size=99,
                     n_positions=64, bos_token_id=0, eos_token_id=98,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = GPT2LMHeadModel(cfg).eval()
    plan = splitbench.SplitPlan.parse("1", n_layers=2)
    split = splitbench.build_split(model, plan)
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(0, 99, (2, 8), generator=g)
    with torch.no_grad():
        target = split.head_forward(ids).detach()
    E = model.get_input_embeddings().weight.detach()
    return torch, split, target, E


@pytest.mark.slow
def test_torch_default_pt_roundtrip_tensor_state(tmp_path):
    torch = pytest.importorskip("torch")
    ck = Checkpointer(str(tmp_path), "pt", 1)
    payload = {"step": 3, "Z": torch.arange(6, dtype=torch.float32) * 0.5,
               "opt": {"state": {0: {"exp_avg": torch.ones(4)}}}}
    path = ck.save(3, payload)
    assert path.endswith(".pt")
    step, loaded = ck.latest()
    assert step == 3
    assert torch.equal(loaded["Z"], payload["Z"])
    assert torch.equal(loaded["opt"]["state"][0]["exp_avg"], torch.ones(4))


def _vmi_run(torch, split, target, E, steps, seed, ckpt=None):
    """One vmi-head optimizer run (or resumed segment) on the tiny head."""
    from attacker.attacks import seam_inversion as si
    torch.manual_seed(seed)
    args = types.SimpleNamespace(lr=0.05, steps=steps)
    return si._optimize_vmi_head(split, target, E, args, None,
                                 checkpointer=ckpt)


@pytest.mark.slow
def test_torch_seam_inversion_resume_matches_uninterrupted(tiny_head,
                                                           tmp_path):
    """Kill simulation: 10-step run == 5 steps, 'kill', resume to 10 —
    bitwise-identical final decode and loss (Adam moments are checkpointed
    too, so the resumed segment replays the same fp op sequence)."""
    torch, split, target, E = tiny_head
    ckdir = str(tmp_path / "ck")

    pred_ref, loss_ref = _vmi_run(torch, split, target, E, steps=10,
                                  seed=42)

    ck = Checkpointer(ckdir, "vmi-step0", 1)
    _vmi_run(torch, split, target, E, steps=5, seed=42, ckpt=ck)
    assert ck.latest()[0] == 4

    ck2 = Checkpointer(ckdir, "vmi-step0", 1)
    pred_res, loss_res = _vmi_run(torch, split, target, E, steps=10,
                                  seed=999, ckpt=ck2)
    assert torch.equal(pred_res, pred_ref)
    assert loss_res == pytest.approx(loss_ref, abs=1e-9)
    assert sorted(os.listdir(ck2.dir)) == ["ckpt_000008.pt",
                                           "ckpt_000009.pt"]


@pytest.mark.slow
def test_torch_seam_inversion_resume_past_the_end_decodes(tiny_head,
                                                          tmp_path):
    """Resume with args.steps <= checkpointed step: the loop is already
    done — decode the restored Z, no steps rerun, loss is None."""
    torch, split, target, E = tiny_head
    ck = Checkpointer(str(tmp_path / "ck"), "vmi-step0", 1)
    pred_a, loss_a = _vmi_run(torch, split, target, E, steps=4, seed=42,
                              ckpt=ck)
    assert loss_a is not None and ck.latest()[0] == 3
    ck2 = Checkpointer(ck.root, "vmi-step0", 1)
    pred_b, loss_b = _vmi_run(torch, split, target, E, steps=4, seed=999,
                              ckpt=ck2)
    assert loss_b is None
    assert torch.equal(pred_b, pred_a)
