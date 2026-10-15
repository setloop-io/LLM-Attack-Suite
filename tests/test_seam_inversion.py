"""seam-inversion: torch-free toy tier (ridge linearization on a synthetic
KNOWN-map seam, decode/scoring/gate plumbing) plus the slow torch layer —
real splitbench captures from a tiny random GPT-2 attacked by the vmi
linearization arm, the wbi relaxation arm, and the vmi --head-pt arm."""

import json

import pytest

from attacker import artifacts
from attacker.attacks import seam_inversion as si


def parse(*argv):
    args = si.build_parser().parse_args(list(argv))
    args.mode = "training"
    return args


def test_parser_builds_torchless():
    args = si.build_parser().parse_args([])
    assert args.arm == "vmi"
    assert args.linearization is False
    assert args.split == "6"
    assert args.steps == 200 and args.samples == 8
    assert args.evidence is False
    with pytest.raises(SystemExit):
        si.build_parser().parse_args(["--help"])
    with pytest.raises(SystemExit):
        si.build_parser().parse_args(["--arm", "nope"])


def test_module_contract_and_register_dispatch():
    from attacker import attacks
    attacks.validate_contract(si, "seam-inversion")
    assert si.EXPERIMENT_ID == "seam_inversion"
    assert set(si.MODES) == {"training", "inference"}
    assert si.REQUIRES_LABELS is False
    if "seam-inversion" not in attacks.ATTACKS:
        attacks.register_attack("seam-inversion", si)
    assert attacks.load("seam-inversion") is si
    assert "seam-inversion" in attacks.for_mode("training")
    assert "seam-inversion" in attacks.for_mode("inference")


def test_toy_world_deterministic_and_shaped():
    w1 = si.toy_world(0, vocab=40, dim=8, n_seq=4, seq_len=6, sigma=0.1)
    w2 = si.toy_world(0, vocab=40, dim=8, n_seq=4, seq_len=6, sigma=0.1)
    assert w1["victim"] == w2["victim"] and w1["h"] == w2["h"]
    assert len(w1["h"]) == 4 and len(w1["h"][0]) == 6
    assert len(w1["h"][0][0]) == 8 and len(w1["E"]) == 40


def test_ridge_solve_and_decode_fixtures():
    X = si._gauss_solve([[2.0, 1.0], [1.0, 3.0]], [[1.0], [2.0]])
    assert X[0][0] == pytest.approx(0.2) and X[1][0] == pytest.approx(0.6)
    with pytest.raises(ValueError, match="singular"):
        si._gauss_solve([[1.0, 2.0], [2.0, 4.0]], [[1.0], [2.0]])
    import random
    rng = random.Random(3)
    A = si._randn_mat(4, 4, rng)
    D = si.ridge_decode_map(A, 1e-9)
    e = [0.3, -0.7, 1.1, 0.2]
    e_hat = si._matmul([si._matmul([e], A)[0]], D)[0]
    assert max(abs(a - b) for a, b in zip(e, e_hat)) < 1e-6
    assert si.cosine_nn_decode([[0.9, 0.1], [0.0, 0.0]],
                               [[1.0, 0.0], [0.0, 1.0]]) == [0, 0]


def test_scoring_fixtures():
    p, r = si.set_precision_recall([1, 2, 3], [2, 3, 4])
    assert p == pytest.approx(2 / 3) and r == pytest.approx(2 / 3)
    acc, nc, nt = si.seq_accuracy([[1, 2], [3, 4]], [[1, 9], [3, 4]])
    assert (acc, nc, nt) == (0.75, 3, 4)
    with pytest.raises(ValueError, match="length mismatch"):
        si.seq_accuracy([[1, 2]], [[1, 2, 3]])
    assert si.majority_control_token([[5, 5, 7], [5, 7]]) == 5
    cell = si.score_sequences([[1, 2]], [[1, 9]], control_tok=1)
    for key in ("token_set_precision", "token_set_recall", "seq_accuracy",
                "control_accuracy", "n_tokens", "gate"):
        assert key in cell
    assert cell["gate"]["verdict"] in ("PASS", "FAIL")
    assert cell["gate"]["n"] == 2


def test_toy_high_recovery_at_zero_noise():
    args = parse("--toy")
    for seed in (0, 1, 2):
        cell = si.toy_cell(args, 0.0, seed)
        assert cell["seq_accuracy"] > 0.90
        assert cell["token_set_precision"] > 0.90
        assert cell["gate"]["verdict"] == "FAIL"


def test_toy_degradation_monotone_ish_under_growing_noise():
    args = parse("--toy", "--seeds", "0")
    sigmas = (0.0, 0.05, 0.2, 0.5, 1.0, 2.0)
    accs = [si.toy_cell(args, s, 0)["seq_accuracy"] for s in sigmas]
    assert accs[0] > 0.90
    assert accs[-1] < accs[0]
    for prev, cur in zip(accs, accs[1:]):
        assert cur <= prev + 0.15
    assert accs[-1] < 0.9


def test_toy_run_journals_artifact_and_evidence(tmp_path):
    from attacker.evidence import verify_evidence
    out = str(tmp_path / "si.json")
    assert si.run(parse("--toy", "--quick", "--evidence",
                        "--output", out)) == 0
    art = json.load(open(out))
    assert art["schema"] == si.SCHEMA
    assert len(art["results"]) == 3
    for cell in art["results"]:
        for key in ("seq_accuracy", "control_accuracy",
                    "token_set_precision", "token_set_recall", "gate"):
            assert key in cell
    journal = [json.loads(l) for l in open(out + ".jsonl")]
    assert [j["sigma"] for j in journal] == [0.0, 0.5, 1.0]
    assert art["summary"] and "seq_accuracy_mean" in art["summary"][0]
    ok, problems = verify_evidence(out + ".evidence.json")
    assert ok and problems == []
    with pytest.raises(SystemExit, match="--evidence requires --output"):
        si.run(parse("--toy", "--quick", "--evidence"))


def test_real_path_errors_are_descriptive_torchless():
    with pytest.raises(SystemExit, match="--capture-dir"):
        si.run(parse())
    if si.torch is None:
        with pytest.raises(SystemExit, match="requires torch"):
            si.run(parse("--capture-dir", "/nonexistent",
                         "--canonical-pt", "/nonexistent", "--model", "x"))


def test_module_self_test_passes():
    assert si.self_test() == 0


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
    writer = splitbench.CaptureWriter(str(cap_dir), "training", "t-si")
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


def _real_args(cap_dir, model_dir, *extra):
    return parse("--capture-dir", cap_dir, "--canonical-pt",
                 cap_dir + "/canonical.pt", "--model", model_dir,
                 "--split", "1", "--calibration-seqs", "64",
                 "--samples", "4", *extra)


@pytest.mark.slow
def test_torch_vmi_linearization_beats_control(tiny_run, tmp_path):
    _, _, cap_dir, model_dir = tiny_run
    out = str(tmp_path / "vmi.json")
    args = _real_args(cap_dir, model_dir, "--arm", "vmi", "--output", out,
                      "--seeds", "11", "22", "33", "44")
    assert si.run(args) == 0
    art = json.load(open(out))
    cells = art["results"]
    assert len(cells) == 2
    assert all(c["submode"] == "vmi-linearization" for c in cells)
    summary = art["summary"][0]
    assert summary["arm"] == "vmi" and summary["n_tokens"] == 4 * 8
    assert summary["seq_accuracy"] >= summary["control_accuracy"] + 0.2
    assert summary["seq_accuracy"] > 0.3
    assert summary["gate"]["verdict"] == "FAIL"
    assert all(c["gate"]["n_seeds"] == 4 for c in cells)
    assert all(c["gate"]["seeds_sufficient"] is True for c in cells)
    assert summary["gate"]["n_seeds"] == 4
    assert summary["gate"]["seeds_sufficient"] is True


@pytest.mark.slow
def test_torch_wbi_runs_and_scores_finite(tiny_run, tmp_path):
    _, _, cap_dir, model_dir = tiny_run
    out = str(tmp_path / "wbi.json")
    args = _real_args(cap_dir, model_dir, "--arm", "wbi", "--steps", "20",
                      "--output", out)
    assert si.run(args) == 0
    art = json.load(open(out))
    cells = art["results"]
    assert len(cells) == 2
    import math
    for c in cells:
        assert c["submode"] == "wbi-map"
        assert math.isfinite(c["final_objective"])
        assert 0.0 <= c["seq_accuracy"] <= 1.0
        assert math.isfinite(c["gate"]["excess_pp"])


@pytest.mark.slow
def test_torch_vmi_head_pt_roundtrip(tiny_run, tmp_path):
    torch, model, cap_dir, _ = tiny_run
    head_pt = str(tmp_path / "head.pt")
    si.save_head_pt(head_pt, model, boundary=1)
    out = str(tmp_path / "vmi_head.json")
    args = parse("--capture-dir", cap_dir, "--canonical-pt",
                 cap_dir + "/canonical.pt", "--head-pt", head_pt,
                 "--arm", "vmi", "--steps", "60", "--samples", "4",
                 "--calibration-seqs", "64", "--output", out)
    assert si.run(args) == 0
    art = json.load(open(out))
    cells = art["results"]
    assert len(cells) == 2
    assert all(c["submode"] == "vmi-head" for c in cells)
    summary = art["summary"][0]
    import math
    assert math.isfinite(cells[0]["final_objective"])
    assert summary["seq_accuracy"] >= summary["control_accuracy"] + 0.2
