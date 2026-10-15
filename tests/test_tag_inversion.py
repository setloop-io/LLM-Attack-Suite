"""gradient-inversion tag arm (TAG/LAMP transformer-specific gradient
inversion): cosine per-token gradient-direction scoring + a light
combinatorial assignment, instead of DLG-era free L2 optimization.

Torch-free layer: the exact linear toy world (orthogonal seam map + tied
unembedding), candidate/decode/scoring/gate plumbing, journal/artifact/
evidence, and the descriptive torch-less guards. Torch layer (slow): real
splitbench bwd captures from a tiny random GPT-2 attacked by the tag arm
— recovery above the majority control and >= the dlgpp-style activation-
match optimization baseline at equal budget."""

import json

import pytest

from attacker.attacks import gradient_inversion as gi


def parse(*argv):
    args = gi.build_parser().parse_args(list(argv))
    args.mode = "training"
    return args


def test_parser_builds_torchless_with_tag_flags():
    args = gi.build_parser().parse_args([])
    assert "tag" in gi.ARMS
    assert args.arm == "dlgpp"
    assert args.assignment == "greedy"
    assert args.tag_temp == 8.0
    assert args.toy_seqs == 8
    assert args.split == "6"
    with pytest.raises(SystemExit):
        gi.build_parser().parse_args(["--help"])
    with pytest.raises(SystemExit):
        gi.build_parser().parse_args(["--arm", "nope"])
    with pytest.raises(SystemExit):
        gi.build_parser().parse_args(["--assignment", "nope"])
    assert gi.build_parser().parse_args(["--arm", "tag"]).arm == "tag"
    for a in ("greedy", "hungarian"):
        assert gi.build_parser().parse_args(["--assignment", a]).assignment \
            == a


def test_tag_arm_in_registry_dispatch():
    from attacker import attacks
    attacks.validate_contract(gi, "gradient-inversion")
    if "gradient-inversion" not in attacks.ATTACKS:
        attacks.register_attack("gradient-inversion", gi)
    assert attacks.load("gradient-inversion") is gi


def test_existing_arms_preserved():
    assert set(gi.ARMS) == {"dlgpp", "idlg", "prior-arm", "tag", "all"}
    with pytest.raises(SystemExit, match="real-capture path"):
        gi.run(parse("--arm", "idlg"))
    with pytest.raises(SystemExit, match="real-capture path"):
        gi.run(parse("--arm", "prior-arm"))


def test_tag_world_deterministic_and_shaped():
    w1 = gi.toy_tag_world(0, vocab=16, dim=8, seq_len=4, n_seq=3)
    w2 = gi.toy_tag_world(0, vocab=16, dim=8, seq_len=4, n_seq=3)
    assert [s["ids"] for s in w1["seqs"]] == [s["ids"] for s in w2["seqs"]]
    assert len(w1["seqs"]) == 3
    assert len(w1["seqs"][0]["g"]) == 3
    assert len(w1["seqs"][0]["g"][0]) == 8
    assert len(w1["U"]) == 16 and len(w1["U"][0]) == 8


def test_tag_orthogonal_map_is_isometry():
    import math
    A = gi._orthogonal_mat(8, __import__("random").Random(0))
    for i in range(8):
        for j in range(8):
            dot = sum(A[k][i] * A[k][j] for k in range(8))
            assert dot == pytest.approx(1.0 if i == j else 0.0, abs=1e-9)


def test_tag_candidates_peak_at_label_exact():
    w = gi.toy_tag_world(7, vocab=16, dim=32, seq_len=6, n_seq=4)
    for s in w["seqs"]:
        C = gi.tag_candidates(w, s["g"])
        assert len(C) == 5 and len(C[0]) == 16
        for i, row in enumerate(C):
            assert max(range(16), key=lambda v: row[v]) == s["labels"][i]


def test_hungarian_and_greedy_decode():
    C = [[0.1, 0.9, 0.2], [0.8, 0.3, 0.4]]
    assert gi._tag_decode(C, "greedy") == [1, 0]
    assert gi._tag_decode(C, "hungarian") == [1, 0]
    cost = [[-1.0 if j == i else 0.0 for j in range(4)] for i in range(3)]
    assert gi._hungarian_assign(cost) == [0, 1, 2]


def test_tag_scoring_fixtures():
    p, r = gi._set_precision_recall([1, 2, 3], [2, 3, 4])
    assert p == pytest.approx(2 / 3) and r == pytest.approx(2 / 3)
    acc, nc, nt = gi._seq_accuracy([[1, 2], [3, 4]], [[1, 9], [3, 4]])
    assert (acc, nc, nt) == (0.75, 3, 4)
    assert gi._majority_control_token([[5, 5, 7], [5, 7]]) == 5
    cell = gi._score_tag([[1, 2]], [[1, 9]], control_tok=1)
    for key in ("token_set_precision", "token_set_recall", "seq_accuracy",
                    "control_accuracy", "n_tokens", "gate"):
        assert key in cell
    assert cell["gate"]["verdict"] in ("PASS", "FAIL")


def test_tag_toy_planted_high_null_at_control():
    args = parse("--toy", "--arm", "tag", "--vocab", "32", "--hidden", "16",
                 "--seq-len", "6", "--toy-seqs", "8")
    for seed in (0, 1, 2):
        planted = gi.toy_tag_cell(args, "planted", seed)
        null = gi.toy_tag_cell(args, "null", seed)
        assert planted["seq_accuracy"] > 0.90
        assert planted["gate"]["verdict"] == "FAIL"
        assert abs(null["seq_accuracy"] - null["control_accuracy"]) < 0.05


def test_tag_toy_hungarian_underperforms_greedy_on_repeats():
    args = parse("--toy", "--arm", "tag", "--vocab", "16", "--hidden", "16",
                 "--seq-len", "8", "--toy-seqs", "8")
    greedy = sum(gi.toy_tag_cell(args, "planted", s)["seq_accuracy"]
                 for s in (0, 1, 2)) / 3
    hung = sum(gi.toy_tag_cell(parse("--toy", "--arm", "tag", "--vocab", "16",
                                     "--hidden", "16", "--seq-len", "8",
                                     "--toy-seqs", "8",
                                     "--assignment", "hungarian"),
                               "planted", s)["seq_accuracy"]
               for s in (0, 1, 2)) / 3
    assert greedy >= hung


def test_tag_toy_run_journals_artifact_and_evidence(tmp_path):
    from attacker.evidence import verify_evidence
    out = str(tmp_path / "tag.json")
    assert gi.run(parse("--toy", "--arm", "tag", "--quick", "--evidence",
                        "--output", out)) == 0
    art = json.load(open(out))
    assert len(art["results"]) == 4
    for cell in art["results"]:
        assert cell["arm"] == "tag"
        for key in ("seq_accuracy", "control_accuracy",
                    "token_set_precision", "token_set_recall", "gate",
                    "source", "assignment"):
            assert key in cell
    assert sorted(c["source"] for c in art["results"]) == \
        ["null", "null", "planted", "planted"]
    journal = [json.loads(l) for l in open(out + ".jsonl")]
    assert len(journal) == 4
    ok, problems = verify_evidence(out + ".evidence.json")
    assert ok and problems == []
    with pytest.raises(SystemExit, match="--evidence requires --output"):
        gi.run(parse("--toy", "--arm", "tag", "--quick", "--evidence"))


def test_tag_real_path_errors_are_descriptive_torchless():
    with pytest.raises(SystemExit, match="--capture-dir"):
        gi.run(parse("--arm", "tag"))
    if gi.torch is None:
        with pytest.raises(SystemExit, match="requires torch"):
            gi.run(parse("--arm", "tag", "--capture-dir", "/nonexistent",
                         "--canonical-pt", "/nonexistent", "--model", "x"))


def test_module_self_test_passes():
    assert gi.self_test() == 0


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
    writer = splitbench.CaptureWriter(str(cap_dir), "training", "t-tag")
    g = torch.Generator().manual_seed(11)
    batches = [torch.randint(0, VOCAB, (2, 8), generator=g)
               for _ in range(2)]
    bench = splitbench.BenchConfig(steps=2, seq_len=8, batch_size=2,
                                   lr=1e-4, capture_every=1,
                                   save_canonical=True)
    splitbench.finetune(split, bench, batches, writer)
    writer.close()
    assert (cap_dir / "canonical.pt").is_file()
    return str(cap_dir), str(model_dir)


def _tag_real_args(cap_dir, model_dir, *extra):
    return parse("--arm", "tag", "--capture-dir", cap_dir, "--canonical-pt",
                 cap_dir + "/canonical.pt", "--model", model_dir,
                 "--split", "1", "--samples", "4", *extra)


@pytest.mark.slow
def test_torch_tag_greedy_above_control_and_beats_dlgpp_baseline(
        tiny_run, tmp_path):
    cap_dir, model_dir = tiny_run
    out = str(tmp_path / "tag.json")
    args = _tag_real_args(cap_dir, model_dir, "--assignment", "greedy",
                          "--rounds", "60", "--restarts", "2",
                          "--output", out)
    assert gi.run(args) == 0
    art = json.load(open(out))
    cells = art["results"]
    assert len(cells) == 2
    for c in cells:
        assert c["arm"] == "tag" and c["submode"] == "real-greedy"
        for key in ("seq_accuracy", "control_accuracy",
                    "token_set_precision", "token_set_recall", "gate",
                    "dlgpp_baseline_seq_accuracy", "improvement_pp"):
            assert key in c
        assert c["seq_accuracy"] >= c["control_accuracy"] + 0.2
        assert c["seq_accuracy"] >= c["dlgpp_baseline_seq_accuracy"]
        assert c["gate"]["verdict"] == "FAIL"
    summary = art["summary"][0]
    assert summary["arm"] == "tag" and summary["n_tokens"] == 4 * 7
    assert summary["seq_accuracy"] >= summary["control_accuracy"] + 0.2
    assert summary["seq_accuracy"] >= \
        summary["dlgpp_baseline_seq_accuracy_mean"]
    assert summary["seq_accuracy"] > 0.9


@pytest.mark.slow
def test_torch_tag_hungarian_and_evidence(tiny_run, tmp_path):
    from attacker.evidence import verify_evidence
    cap_dir, model_dir = tiny_run
    out = str(tmp_path / "tag_h.json")
    args = _tag_real_args(cap_dir, model_dir, "--assignment", "hungarian",
                          "--rounds", "60", "--restarts", "2", "--evidence",
                          "--output", out)
    assert gi.run(args) == 0
    art = json.load(open(out))
    cells = art["results"]
    assert len(cells) == 2
    for c in cells:
        assert c["submode"] == "real-hungarian"
        assert c["seq_accuracy"] >= c["control_accuracy"] + 0.2
        assert c["gate"]["verdict"] == "FAIL"
    journal = [json.loads(l) for l in open(out + ".jsonl")]
    assert len(journal) == 2
    ok, problems = verify_evidence(out + ".evidence.json")
    assert ok and problems == []


@pytest.mark.slow
def test_torch_tag_budget_free_advantage(tiny_run, tmp_path):
    cap_dir, model_dir = tiny_run
    out = str(tmp_path / "tag_budget.json")
    args = _tag_real_args(cap_dir, model_dir, "--assignment", "greedy",
                          "--rounds", "30", "--restarts", "1",
                          "--output", out)
    assert gi.run(args) == 0
    summary = json.load(open(out))["summary"][0]
    assert summary["seq_accuracy"] > 0.9
    assert summary["seq_accuracy"] >= \
        summary["dlgpp_baseline_seq_accuracy_mean"]
