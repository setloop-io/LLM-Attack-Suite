"""batch-label-recon: torch-free toy tier (closed-form least-squares label
recovery on a synthetic KNOWN linear body; planted vs null cells,
decode/scoring/gate plumbing) plus the slow torch layer — real splitbench
captures from a tiny random GPT-2 (finetune with --body-grad-capture)
attacked by the gradient-matching real tier."""

import json
import math

import pytest

from attacker import artifacts
from attacker.attacks import batch_label_recon as blr


def parse(*argv):
    args = blr.build_parser().parse_args(list(argv))
    args.mode = "training"
    return args


def test_parser_builds_torchless():
    args = blr.build_parser().parse_args([])
    assert args.split == "6"
    assert args.steps == 200 and args.seam_weight == 1.0
    assert args.evidence is False
    assert args.body_grads_pt is None and args.body_state_pt is None
    with pytest.raises(SystemExit):
        blr.build_parser().parse_args(["--help"])


def test_module_contract_and_register_dispatch():
    from attacker import attacks
    attacks.validate_contract(blr, "batch-label-recon")
    assert blr.EXPERIMENT_ID == "batch_label_reconstruction"
    assert blr.MODES == ("training",)
    assert blr.REQUIRES_LABELS is False
    wired = "batch-label-recon" in attacks.ATTACKS
    if not wired:
        attacks.register_attack("batch-label-recon", blr)
    try:
        assert attacks.load("batch-label-recon") is blr
        assert "batch-label-recon" in attacks.for_mode("training")
        assert "batch-label-recon" not in attacks.for_mode("inference")
    finally:
        if not wired:
            attacks.ATTACKS.pop("batch-label-recon", None)


def test_toy_world_deterministic_and_shaped():
    w1 = blr.toy_world(0, vocab=12, dim=16, n_seq=4, seq_len=4)
    w2 = blr.toy_world(0, vocab=12, dim=16, n_seq=4, seq_len=4)
    assert w1["labels"] == w2["labels"] and w1["GW"] == w2["GW"]
    assert len(w1["GW"]) == 12 and len(w1["GW"][0]) == 16
    assert len(w1["X"]) == 16 and len(w1["P"]) == 16
    for row in w1["P"]:
        assert sum(row) == pytest.approx(1.0)


def test_least_squares_fixtures():
    X = blr._gauss_solve([[2.0, 1.0], [1.0, 3.0]], [[1.0], [2.0]])
    assert X[0][0] == pytest.approx(0.2) and X[1][0] == pytest.approx(0.6)
    with pytest.raises(ValueError, match="singular"):
        blr._gauss_solve([[1.0, 2.0], [2.0, 4.0]], [[1.0], [2.0]])
    world = blr.toy_world(7, vocab=6, dim=8, n_seq=2, seq_len=4)
    preds, _ = blr.toy_recover(world["W"], world["X"], world["P"],
                               world["GW"])
    assert preds == [t for seq in world["labels"] for t in seq]


def test_scoring_fixtures():
    p, r = blr.set_precision_recall([1, 2, 3], [2, 3, 4])
    assert p == pytest.approx(2 / 3) and r == pytest.approx(2 / 3)
    acc, nc, nt = blr.seq_accuracy([[1, 2], [3, 4]], [[1, 9], [3, 4]])
    assert (acc, nc, nt) == (0.75, 3, 4)
    with pytest.raises(ValueError, match="length mismatch"):
        blr.seq_accuracy([[1, 2]], [[1, 2, 3]])
    assert blr.bag_mode_control([[5, 5, 7], [5, 7]]) == 5
    with pytest.raises(ValueError, match="empty"):
        blr.bag_mode_control([[]])
    cell = blr.score_batch([[1, 2]], [[1, 9]], control_tok=1)
    for key in ("token_set_precision", "token_set_recall", "seq_accuracy",
                "control_accuracy", "n_tokens", "gate"):
        assert key in cell
    assert cell["gate"]["verdict"] in ("PASS", "FAIL")
    assert cell["gate"]["n"] == 2


def test_toy_planted_high_recovery():
    args = parse("--toy")
    for seed in (0, 1, 2):
        cell = blr.toy_cell(args, "planted", seed)
        assert cell["seq_accuracy"] > 0.95
        assert cell["token_set_recall"] > 0.95
        assert cell["gate"]["verdict"] == "FAIL"


def test_toy_null_sits_at_control():
    args = parse("--toy")
    null = blr.toy_cell(args, "null", 0)
    planted = blr.toy_cell(args, "planted", 0)
    assert null["seq_accuracy"] < 0.5
    assert abs(null["seq_accuracy"] - null["control_accuracy"]) < 0.25
    assert null["gate"]["at_floor"]
    assert not planted["gate"]["at_floor"]
    assert null["gate"]["excess_pp"] < planted["gate"]["excess_pp"]


def test_toy_run_journals_artifact_and_evidence(tmp_path):
    from attacker.evidence import verify_evidence
    out = str(tmp_path / "blr.json")
    assert blr.run(parse("--toy", "--quick", "--evidence",
                         "--output", out)) == 0
    art = json.load(open(out))
    assert art["schema"] == blr.SCHEMA
    assert len(art["results"]) == 2
    assert {c["source"] for c in art["results"]} == {"planted", "null"}
    for cell in art["results"]:
        for key in ("seq_accuracy", "control_accuracy",
                    "token_set_precision", "token_set_recall", "gate"):
            assert key in cell
    journal = [json.loads(l) for l in open(out + ".jsonl")]
    assert [j["source"] for j in journal] == ["planted", "null"]
    assert art["summary"] and "seq_accuracy_mean" in art["summary"][0]
    ok, problems = verify_evidence(out + ".evidence.json")
    assert ok and problems == []
    with pytest.raises(SystemExit, match="--evidence requires --output"):
        blr.run(parse("--toy", "--quick", "--evidence"))


def test_real_path_errors_are_descriptive_torchless():
    with pytest.raises(SystemExit, match="--capture-dir"):
        blr.run(parse())
    with pytest.raises(SystemExit, match="--model"):
        blr.run(parse("--capture-dir", "/nonexistent",
                      "--canonical-pt", "/nonexistent"))
    if blr.torch is None:
        with pytest.raises(SystemExit, match="requires torch"):
            blr.run(parse("--capture-dir", "/nonexistent",
                          "--canonical-pt", "/nonexistent",
                          "--model", "x"))


def test_module_self_test_passes():
    assert blr.self_test() == 0


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
    writer = splitbench.CaptureWriter(str(cap_dir), "training", "t-blr")
    g = torch.Generator().manual_seed(11)
    batches = [torch.randint(0, VOCAB, (2, 8), generator=g)
               for _ in range(2)]
    bench = splitbench.BenchConfig(steps=2, seq_len=8, batch_size=2,
                                   lr=1e-4, capture_every=1,
                                   save_canonical=True,
                                   body_grad_capture=True)
    splitbench.finetune(split, bench, batches, writer)
    writer.close()
    assert (cap_dir / "canonical.pt").is_file()
    assert (cap_dir / "body_grads_step0.pt").is_file()
    return torch, str(cap_dir), str(model_dir)


def _real_args(cap_dir, model_dir, *extra):
    return parse("--capture-dir", cap_dir, "--canonical-pt",
                 cap_dir + "/canonical.pt", "--model", model_dir,
                 "--split", "1", *extra)


@pytest.mark.slow
def test_torch_grad_match_beats_control_in_30_steps(tiny_run, tmp_path):
    _, cap_dir, model_dir = tiny_run
    out = str(tmp_path / "blr.json")
    args = _real_args(cap_dir, model_dir, "--steps", "30",
                      "--evidence", "--output", out)
    assert blr.run(args) == 0
    art = json.load(open(out))
    assert art["config"]["n_body_grads_matched"] > 0
    cells = art["results"]
    assert len(cells) == 1
    cell = cells[0]
    assert cell["submode"] == "soft-label-dlg"
    assert math.isfinite(cell["final_match_loss"])
    assert cell["n_tokens"] == 2 * 7
    assert cell["seq_accuracy"] >= cell["control_accuracy"] + 0.1
    assert cell["token_set_recall"] > cell["control_accuracy"]
    assert cell["gate"]["verdict"] == "FAIL"
    summary = art["summary"][0]
    assert summary["n_cells"] == 1
    assert summary["seq_accuracy"] == cell["seq_accuracy"]
    journal = [json.loads(l) for l in open(out + ".jsonl")]
    assert len(journal) == 1 and journal[0]["step"] == 0
    from attacker.evidence import verify_evidence
    ok, problems = verify_evidence(out + ".evidence.json")
    assert ok and problems == []


@pytest.mark.slow
def test_torch_name_mismatch_is_a_clean_error(tiny_run, tmp_path):
    torch, cap_dir, _ = tiny_run
    pytest.importorskip("transformers")
    from transformers import MistralConfig, MistralForCausalLM
    other = MistralForCausalLM(MistralConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB,
        max_position_embeddings=64))
    other_dir = tmp_path / "tinymistral"
    other.save_pretrained(other_dir)
    out = str(tmp_path / "blr_mm.json")
    args = _real_args(cap_dir, str(other_dir), "--steps", "1",
                      "--output", out)
    with pytest.raises(SystemExit, match="SAME architecture"):
        blr.run(args)
