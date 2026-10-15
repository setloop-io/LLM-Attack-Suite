"""seam-inversion joint arm: fwd seam h* + bwd seam g* TOGETHER — bwd
per-token direction candidates (tag-style) re-ranked by
forward-consistency with the captured h* (causal beam search over the
public head).

Torch-free layer: the complementary-rank grid toy world (fwd and bwd each
rank-deficient in complementary directions — singles at the 1/grid
ambiguity floor, joint recovers; the honesty centerpiece), parser/guards,
journal/artifact/evidence. Torch layer (slow): real splitbench captures
from a tiny random GPT-2 in the constructed regime where BOTH singles are
degraded (dense-vocab distorting body — the plain tiny world saturates
both singles at 1.0, measured, so the ordering is asserted here per the
task's documented escape hatch).
"""

import json

import pytest

from attacker.attacks import seam_inversion as si


def parse(*argv):
    args = si.build_parser().parse_args(list(argv))
    args.mode = "training"
    return args


def test_parser_joint_flags_torchless():
    args = si.build_parser().parse_args([])
    assert "joint" in si.ARMS
    assert args.arm == "vmi"
    assert args.joint_candidates == 8
    assert args.joint_beam == 4
    assert args.joint_grid == 4
    assert si.build_parser().parse_args(["--arm", "joint"]).arm == "joint"
    with pytest.raises(SystemExit):
        si.build_parser().parse_args(["--help"])
    with pytest.raises(SystemExit, match="--joint-candidates"):
        si.run(parse("--toy", "--arm", "joint", "--joint-candidates", "0"))


def test_joint_toy_world_deterministic_and_shaped():
    w1 = si.toy_joint_world(0, grid=4, dim=16, n_seq=4, seq_len=6)
    w2 = si.toy_joint_world(0, grid=4, dim=16, n_seq=4, seq_len=6)
    assert w1["victim"] == w2["victim"] and w1["E"] == w2["E"]
    assert len(w1["E"]) == 16 and len(w1["E"][0]) == 16
    assert w1["m"] == 8 and w1["grid"] == 4
    with pytest.raises(ValueError, match="even"):
        si.toy_joint_world(0, grid=4, dim=15)


def test_joint_toy_singles_at_floor_joint_recovers():
    args = parse("--toy", "--arm", "joint", "--seeds", "0", "1", "2")
    means = {}
    for variant in ("fwd-only", "bwd-only", "joint"):
        accs = [si.toy_joint_cell(args, variant, s)["seq_accuracy"]
                for s in (0, 1, 2)]
        means[variant] = sum(accs) / 3
    assert means["joint"] > 0.95
    assert means["fwd-only"] < 0.6 and means["bwd-only"] < 0.6
    assert means["joint"] > means["fwd-only"] + 0.4
    assert means["joint"] > means["bwd-only"] + 0.4
    cell = si.toy_joint_cell(args, "joint", 0)
    for key in ("seq_accuracy", "control_accuracy", "token_set_precision",
                "token_set_recall", "gate", "variant", "grid"):
        assert key in cell
    with pytest.raises(ValueError, match="variant"):
        si.toy_joint_cell(args, "sideways", 0)


def test_joint_toy_run_journals_artifact_and_evidence(tmp_path):
    from attacker.evidence import verify_evidence
    out = str(tmp_path / "sij.json")
    assert si.run(parse("--toy", "--arm", "joint", "--quick", "--evidence",
                        "--output", out)) == 0
    art = json.load(open(out))
    assert art["schema"] == si.SCHEMA
    assert art["config"]["arm"] == "joint"
    cells = art["results"]
    assert sorted(c["variant"] for c in cells) == \
        ["bwd-only", "fwd-only", "joint"]
    summary = art["summary"][0]
    assert summary["joint_minus_fwd_pp"] > 40.0
    assert summary["joint_minus_bwd_pp"] > 40.0
    journal = [json.loads(l) for l in open(out + ".jsonl")]
    assert len(journal) == len(cells)
    ok, problems = verify_evidence(out + ".evidence.json")
    assert ok and problems == []


def test_joint_real_path_errors_are_descriptive_torchless():
    with pytest.raises(SystemExit, match="--capture-dir"):
        si.run(parse("--arm", "joint"))
    if si.torch is None:
        with pytest.raises(SystemExit, match="requires torch"):
            si.run(parse("--arm", "joint", "--capture-dir", "/nonexistent",
                         "--canonical-pt", "/nonexistent", "--model", "x"))


def test_module_self_test_passes():
    assert si.self_test() == 0


VOCAB = 4000


@pytest.fixture
def tiny_run(tmp_path):
    """4L is unnecessary — the 2L dense-vocab distorting body (beta=8
    output-projection shear, gamma=4 peaky attention) already degrades
    both singles: measured fwd-linearization 0.312, bwd-direction 0.929,
    joint 1.000. (The plain 2L/32/99 world saturates BOTH singles at
    1.000 — measured on the committed fixtures — so the ordering is
    asserted in this constructed noisier variant, the task's documented
    escape hatch.)"""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel
    from attacker import splitbench
    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=2, n_embd=32, n_head=4, vocab_size=VOCAB,
                     n_positions=64, bos_token_id=0, eos_token_id=VOCAB - 1,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = GPT2LMHeadModel(cfg)
    with torch.no_grad():
        for blk in model.transformer.h:
            blk.attn.c_proj.weight.mul_(8.0)
            blk.mlp.c_proj.weight.mul_(8.0)
            blk.attn.c_attn.weight.mul_(4.0)
    model_dir = tmp_path / "distortgpt2"
    model.save_pretrained(model_dir)
    plan = splitbench.SplitPlan.parse("1", n_layers=2)
    split = splitbench.build_split(model, plan)
    cap_dir = tmp_path / "caps"
    writer = splitbench.CaptureWriter(str(cap_dir), "training", "t-jt")
    g = torch.Generator().manual_seed(11)
    batches = [torch.randint(0, VOCAB, (2, 8), generator=g)
               for _ in range(2)]
    bench = splitbench.BenchConfig(steps=2, seq_len=8, batch_size=2,
                                   lr=1e-4, capture_every=1,
                                   save_canonical=True)
    splitbench.finetune(split, bench, batches, writer)
    writer.close()
    assert (cap_dir / "canonical.pt").is_file()
    return torch, model, str(cap_dir), str(model_dir)


@pytest.mark.slow
def test_torch_joint_beats_both_singles_in_constructed_regime(
        tiny_run, tmp_path):
    _, _, cap_dir, model_dir = tiny_run
    out = str(tmp_path / "joint.json")
    args = parse("--arm", "joint", "--capture-dir", cap_dir,
                 "--canonical-pt", cap_dir + "/canonical.pt",
                 "--model", model_dir, "--split", "1",
                 "--calibration-seqs", "64", "--samples", "4",
                 "--evidence", "--output", out)
    assert si.run(args) == 0
    art = json.load(open(out))
    cells = art["results"]
    assert len(cells) == 2
    for c in cells:
        assert c["arm"] == "joint" and c["submode"] == "joint-beam"
        for key in ("seq_accuracy", "control_accuracy", "gate",
                    "fwd_linearization_seq_accuracy",
                    "bwd_direction_seq_accuracy",
                    "joint_minus_fwd_pp", "joint_minus_bwd_pp"):
            assert key in c
    summary = art["summary"][0]
    joint_acc = summary["seq_accuracy"]
    fwd_acc = summary["fwd_linearization_seq_accuracy_mean"]
    bwd_acc = summary["bwd_direction_seq_accuracy_mean"]
    assert fwd_acc < 0.6
    assert bwd_acc < 0.95
    assert joint_acc >= 0.95
    assert summary["joint_minus_fwd_pp_mean"] > 30.0
    assert summary["joint_minus_bwd_pp_mean"] > 5.0
    assert joint_acc > bwd_acc > fwd_acc
    assert summary["gate"]["verdict"] == "FAIL"
    journal = [json.loads(l) for l in open(out + ".jsonl")]
    assert len(journal) == len(cells)
    from attacker.evidence import verify_evidence
    ok, problems = verify_evidence(out + ".evidence.json")
    assert ok and problems == []


@pytest.mark.slow
def test_torch_joint_requires_training_mode_and_head(tiny_run):
    _, _, cap_dir, model_dir = tiny_run
    args = parse("--arm", "joint", "--capture-dir", cap_dir,
                 "--canonical-pt", cap_dir + "/canonical.pt",
                 "--model", model_dir, "--split", "1", "--samples", "2")
    args.mode = "inference"
    with pytest.raises(SystemExit, match="training mode only"):
        si.run(args)
