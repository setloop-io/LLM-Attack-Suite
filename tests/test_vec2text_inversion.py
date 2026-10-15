"""vec2text-inversion: torch-free toy tier (nearest-centroid learned
inverter vs position-independent NN baseline on a synthetic
position-dependent seam; BLEU/scoring/gate plumbing) plus the slow torch
layer — a tiny random GPT-2 captured through splitbench, an inverter
trained 200 steps on public draws, then attacked with corrections=0/1."""

import json

import pytest

from attacker import artifacts
from attacker.attacks import vec2text_inversion as vi


def parse(*argv):
    args = vi.build_parser().parse_args(list(argv))
    args.mode = "training"
    return args


def test_parser_builds_torchless():
    args = vi.build_parser().parse_args([])
    assert args.phase == "attack"
    assert args.corrections == 0
    assert args.split == "6"
    assert args.train_steps == 2000 and args.samples == 8
    assert args.d_model == 128 and args.enc_layers == 2 and args.dec_layers == 2
    assert args.evidence is False
    with pytest.raises(SystemExit):
        vi.build_parser().parse_args(["--help"])
    with pytest.raises(SystemExit):
        vi.build_parser().parse_args(["--phase", "nope"])


def test_module_contract_and_register_dispatch():
    from attacker import attacks
    attacks.validate_contract(vi, "vec2text-inversion")
    assert vi.EXPERIMENT_ID == "vec2text_inversion"
    assert set(vi.MODES) == {"training", "inference"}
    assert vi.REQUIRES_LABELS is False
    if "vec2text-inversion" not in attacks.ATTACKS:
        attacks.register_attack("vec2text-inversion", vi)
    assert attacks.load("vec2text-inversion") is vi
    assert "vec2text-inversion" in attacks.for_mode("training")
    assert "vec2text-inversion" in attacks.for_mode("inference")


def test_bleu_fixtures():
    assert vi.corpus_bleu([[1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5]]) == \
        pytest.approx(1.0)
    assert vi.corpus_bleu([[1, 2]], [[1, 3]]) == pytest.approx(0.25 ** 0.25)
    disjoint = vi.corpus_bleu([[1, 1, 1, 1]], [[2, 3, 4, 5]])
    partial = vi.corpus_bleu([[1, 2, 9, 9]], [[1, 2, 3, 4]])
    assert 0.0 <= disjoint < partial < 1.0
    with pytest.raises(ValueError):
        vi.corpus_bleu([[1]], [[1]], max_n=0)


def test_toy_world_deterministic_and_shaped():
    w1 = vi.toy_world(0, vocab=60, dim=16, seq_len=12, n_victim=4,
                      n_pub=20, sigma=0.3)
    w2 = vi.toy_world(0, vocab=60, dim=16, seq_len=12, n_victim=4,
                      n_pub=20, sigma=0.3)
    assert w1["victim"] == w2["victim"] and w1["h_victim"] == w2["h_victim"]
    assert len(w1["h_victim"]) == 4 and len(w1["h_victim"][0]) == 12
    assert len(w1["h_victim"][0][0]) == 16 and len(w1["E"]) == 60
    assert len(w1["A"]) == 12


def test_toy_learned_beats_nn_baseline():
    args = parse("--toy")
    for seed in (0, 1, 2):
        cell = vi.toy_cell(args, 0.0, seed)
        assert cell["seq_accuracy"] > 0.85
        assert cell["seq_accuracy"] \
            > cell["baseline_nn"]["seq_accuracy"] + 0.2
        assert cell["gate"]["verdict"] == "FAIL"
        assert cell["bleu"] > 0.7


def test_toy_noise_degrades_gracefully():
    args = parse("--toy", "--seeds", "0")
    accs = [vi.toy_cell(args, s, 0)["seq_accuracy"]
            for s in (0.0, 0.3, 1.0, 2.0, 4.0)]
    assert accs[0] > 0.85
    assert accs[-1] < accs[0]
    for prev, cur in zip(accs, accs[1:]):
        assert cur <= prev + 0.15


def test_toy_run_journals_artifact_and_evidence(tmp_path):
    from attacker.evidence import verify_evidence
    out = str(tmp_path / "v2t.json")
    assert vi.run(parse("--toy", "--quick", "--evidence",
                        "--output", out)) == 0
    art = json.load(open(out))
    assert art["schema"] == vi.SCHEMA
    assert len(art["results"]) == 3
    for cell in art["results"]:
        for key in ("seq_accuracy", "control_accuracy",
                    "token_set_precision", "token_set_recall", "bleu",
                    "baseline_nn", "gate"):
            assert key in cell
        assert "gate" in cell["baseline_nn"]
    journal = [json.loads(l) for l in open(out + ".jsonl")]
    assert [j["sigma"] for j in journal] == [0.0, 0.5, 2.0]
    assert art["summary"] and "baseline_nn_accuracy_mean" in art["summary"][0]
    ok, problems = verify_evidence(out + ".evidence.json")
    assert ok and problems == []
    with pytest.raises(SystemExit, match="--evidence requires --output"):
        vi.run(parse("--toy", "--quick", "--evidence"))


def test_real_path_errors_are_descriptive_torchless():
    with pytest.raises(SystemExit, match="--capture-dir"):
        vi.run(parse())
    with pytest.raises(SystemExit, match="--inverter-pt"):
        vi.run(parse("--phase", "train"))
    with pytest.raises(SystemExit, match="--inverter-pt"):
        vi.run(parse("--capture-dir", "/nonexistent",
                     "--canonical-pt", "/nonexistent"))
    if vi.torch is None:
        with pytest.raises(SystemExit, match="requires torch"):
            vi.run(parse("--capture-dir", "/nonexistent",
                         "--canonical-pt", "/nonexistent",
                         "--inverter-pt", "/nonexistent"))
        with pytest.raises(SystemExit, match="requires torch"):
            vi.run(parse("--phase", "train", "--inverter-pt", "/x.pt",
                         "--model", "/nonexistent"))


def test_module_self_test_passes():
    assert vi.self_test() == 0


VOCAB = 99


@pytest.fixture
def tiny_run(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel
    from attacker import splitbench
    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=2, n_embd=32, n_head=4, vocab_size=VOCAB,
                     n_positions=64, bos_token_id=0, eos_token_id=98,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = GPT2LMHeadModel(cfg)
    model_dir = tmp_path / "tinygpt2"
    model.save_pretrained(model_dir)
    plan = splitbench.SplitPlan.parse("1", n_layers=2)
    split = splitbench.build_split(model, plan)
    cap_dir = tmp_path / "caps"
    writer = splitbench.CaptureWriter(str(cap_dir), "training", "t-v2t")
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


def _train_args(model_dir, inv_pt, *extra, corrections=1):
    return parse("--phase", "train", "--model", model_dir, "--split", "1",
                 "--seq-len", "8", "--train-seqs", "512",
                 "--train-steps", "200", "--corrections", str(corrections),
                 "--inverter-pt", inv_pt, *extra)


def _attack_args(cap_dir, inv_pt, *extra):
    return parse("--phase", "attack", "--capture-dir", cap_dir,
                 "--canonical-pt", cap_dir + "/canonical.pt",
                 "--inverter-pt", inv_pt, "--samples", "4", *extra)


@pytest.mark.slow
def test_torch_train_checkpoint_format(tiny_run, tmp_path):
    torch, _, _, model_dir = tiny_run
    inv_pt = str(tmp_path / "inv.pt")
    out = str(tmp_path / "train.json")
    assert vi.run(_train_args(model_dir, inv_pt, "--output", out)) == 0
    ckpt = torch.load(inv_pt, map_location="cpu")
    assert ckpt["format"] == vi.CKPT_FORMAT
    assert ckpt["corrector_trained"] is True
    assert isinstance(ckpt["control_token"], int)
    assert ckpt["arch"]["seq_len"] == 8 and ckpt["arch"]["seam_dim"] == 32
    assert ckpt["arch"]["vocab"] == VOCAB
    art = json.load(open(out))
    assert art["results"][0]["phase"] == "train"
    assert art["results"][0]["final_loss"]["step"] == 199


@pytest.mark.slow
def test_torch_attack_beats_control_and_baseline(tiny_run, tmp_path):
    _, _, cap_dir, model_dir = tiny_run
    inv_pt = str(tmp_path / "inv.pt")
    assert vi.run(_train_args(model_dir, inv_pt)) == 0

    out0 = str(tmp_path / "atk0.json")
    args = _attack_args(cap_dir, inv_pt, "--corrections", "0",
                        "--model", model_dir, "--output", out0, "--evidence")
    assert vi.run(args) == 0
    art = json.load(open(out0))
    cells = art["results"]
    assert len(cells) == 2
    assert all(c["submode"] == "generate" for c in cells)
    summary = art["summary"][0]
    assert summary["n_tokens"] == 4 * 8
    assert summary["seq_accuracy"] >= summary["control_accuracy"] + 0.5
    assert summary["seq_accuracy"] > 0.8
    assert summary["bleu"] > 0.5
    assert summary["gate"]["verdict"] == "FAIL"
    assert 0.0 <= summary["baseline_nn"]["seq_accuracy"] <= 1.0
    assert "bleu" in summary["baseline_nn"]
    from attacker.evidence import verify_evidence
    ok, problems = verify_evidence(out0 + ".evidence.json")
    assert ok and problems == []

    out1 = str(tmp_path / "atk1.json")
    args = _attack_args(cap_dir, inv_pt, "--corrections", "1",
                        "--output", out1)
    assert vi.run(args) == 0
    art1 = json.load(open(out1))
    assert all(c["submode"] == "generate+correct" for c in art1["results"])
    assert art1["summary"][0]["seq_accuracy"] >= summary["seq_accuracy"]


@pytest.mark.slow
def test_torch_corrections_on_base_only_checkpoint_is_descriptive(
        tiny_run, tmp_path):
    _, _, cap_dir, model_dir = tiny_run
    inv_pt = str(tmp_path / "inv_base.pt")
    assert vi.run(_train_args(model_dir, inv_pt, corrections=0)) == 0
    import torch
    ckpt = torch.load(inv_pt, map_location="cpu")
    assert ckpt["corrector_trained"] is False
    with pytest.raises(SystemExit, match="correction phase"):
        vi.run(_attack_args(cap_dir, inv_pt, "--corrections", "1"))
