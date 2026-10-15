"""malicious-server (Boenisch/Fowl line for split training): the cloud
crafts the body computation (splitbench --server-attack passive-amplify,
y = Body(h) + alpha*h) so the returned seam gradient leaks the client
batch far beyond the honest-but-curious gradient.

Torch-free layer: the consistent linear mixing toy world (honest at the
majority control, crafted ~1.0), parser/registry plumbing,
journal/artifact/evidence, guards. Torch layer (slow): real splitbench
captures through the actual --server-attack hook on a dense-vocab
DISTORTING tiny world (the regime where the honest direction decode
genuinely degrades — the plain tiny world is documented to leak fully
even to the honest server, so no gap can live there).
"""

import json

import pytest

from attacker.attacks import malicious_server as ms


def parse(*argv):
    args = ms.build_parser().parse_args(list(argv))
    args.mode = "training"
    return args


def test_parser_builds_torchless():
    args = ms.build_parser().parse_args([])
    assert args.assignment == "greedy"
    assert args.server_attack_alpha == 8.0
    assert args.toy_mix == 1.0 and args.tag_temp == 8.0
    assert args.split == "6" and args.samples == 8
    with pytest.raises(SystemExit):
        ms.build_parser().parse_args(["--help"])
    with pytest.raises(SystemExit):
        ms.build_parser().parse_args(["--assignment", "nope"])


def test_module_contract_and_register_dispatch():
    from attacker import attacks
    attacks.validate_contract(ms, "malicious-server")
    assert ms.EXPERIMENT_ID == "malicious_server_steering"
    assert ms.MODES == ("training",)
    assert ms.REQUIRES_LABELS is False
    already = "malicious-server" in attacks.ATTACKS
    if not already:
        attacks.register_attack("malicious-server", ms)
    try:
        assert attacks.load("malicious-server") is ms
        assert "malicious-server" in attacks.for_mode("training")
        assert "malicious-server" not in attacks.for_mode("inference")
    finally:
        if not already:
            attacks.ATTACKS.pop("malicious-server", None)


def test_toy_world_deterministic_and_shaped():
    w1 = ms.toy_ms_world(0, vocab=16, dim=16, seq_len=6, n_seq=3)
    w2 = ms.toy_ms_world(0, vocab=16, dim=16, seq_len=6, n_seq=3)
    assert w1["seqs"] == w2["seqs"] and w1["U"] == w2["U"]
    assert len(w1["seqs"]) == 3 and len(w1["seqs"][0]) == 6
    assert len(w1["U"]) == 16 and len(w1["U"][0]) == 16
    assert len(w1["M"]) == 16


def test_toy_gradients_are_consistent_forward_rows():
    w = ms.toy_ms_world(7, vocab=16, dim=16, seq_len=6, n_seq=2)
    gh = ms.toy_ms_grads(w, w["seqs"][0], crafted=False)
    gc = ms.toy_ms_grads(w, w["seqs"][0], crafted=True)
    assert len(gh) == 5 and len(gh[0]) == 16
    assert gh != gc
    C = ms.toy_ms_candidates(w, gc)
    assert len(C) == 5 and len(C[0]) == 16
    assert all(-1.0 - 1e-9 <= c <= 1.0 + 1e-9 for row in C for c in row)


def test_toy_crafted_beats_honest_by_the_published_margin():
    args = parse("--toy", "--vocab", "32", "--hidden", "32",
                 "--seq-len", "8", "--toy-seqs", "8")
    honest, crafted = [], []
    for seed in (0, 1, 2):
        h = ms.toy_ms_cell(args, "honest", seed)
        c = ms.toy_ms_cell(args, "crafted", seed)
        honest.append(h["seq_accuracy"])
        crafted.append(c["seq_accuracy"])
        assert h["seq_accuracy"] < h["control_accuracy"] + 0.10
        assert c["seq_accuracy"] > 0.90
        for key in ("seq_accuracy", "control_accuracy",
                    "token_set_precision", "token_set_recall", "gate",
                    "world"):
            assert key in h and key in c
    assert sum(crafted) / 3 - sum(honest) / 3 > 0.8
    with pytest.raises(ValueError, match="world"):
        ms.toy_ms_cell(args, "sideways", 0)


def test_toy_run_journals_artifact_and_evidence(tmp_path):
    from attacker.evidence import verify_evidence
    out = str(tmp_path / "ms.json")
    assert ms.run(parse("--toy", "--quick", "--evidence",
                        "--output", out)) == 0
    art = json.load(open(out))
    assert art["schema"] == ms.SCHEMA
    cells = art["results"]
    assert sorted(c["world"] for c in cells) == \
        ["crafted", "crafted", "honest", "honest"]
    assert all(c["gate"]["verdict"] in ("PASS", "FAIL") for c in cells)
    margin = [s for s in art["summary"] if "crafted_minus_honest_pp" in s]
    assert margin and margin[0]["crafted_minus_honest_pp"] > 80.0
    journal = [json.loads(l) for l in open(out + ".jsonl")]
    assert len(journal) == len(cells)
    ok, problems = verify_evidence(out + ".evidence.json")
    assert ok and problems == []
    with pytest.raises(SystemExit, match="--evidence requires --output"):
        ms.run(parse("--toy", "--quick", "--evidence"))


def test_real_path_errors_are_descriptive_torchless():
    with pytest.raises(SystemExit) as exc:
        ms.run(parse())
    assert "requires torch" in str(exc.value) \
        or "--capture-dir" in str(exc.value)
    if ms.torch is None:
        with pytest.raises(SystemExit, match="requires torch"):
            ms.run(parse("--capture-dir", "/nonexistent",
                         "--canonical-pt", "/nonexistent",
                         "--crafted-dir", "/nonexistent",
                         "--crafted-canonical-pt", "/nonexistent",
                         "--model", "x"))


def test_module_self_test_passes():
    assert ms.self_test() == 0


def test_toy_summary_n_seeds_is_the_distinct_recorded_seed_count(tmp_path):
    out = str(tmp_path / "ms.json")
    assert ms.run(parse("--toy", "--seeds", "0", "0", "1",
                        "--output", out)) == 0
    art = json.loads((tmp_path / "ms.json").read_text())
    by = {}
    for c in art["results"]:
        by.setdefault(c["world"], []).append(c)
    rows = [r for r in art["summary"] if "world" in r]
    assert {r["world"] for r in rows} == set(ms.WORLDS)
    for row in rows:
        cells = by[row["world"]]
        assert row["n_seeds"] == len({c["seed"] for c in cells}) == 2
        assert row["n_cells"] == len(cells) == 3
        assert {c["gate"]["n_seeds"] for c in cells} == {3}


def test_toy_cross_world_aggregate_row_carries_no_n_seeds(tmp_path):
    """crafted_minus_honest_pp is a difference of two per-world means with
    no per-cell records of its own: a non-gradeable aggregate, deliberately
    without n_seeds so a consumer cannot mistake it for a scored cell."""
    out = str(tmp_path / "ms.json")
    assert ms.run(parse("--toy", "--quick", "--output", out)) == 0
    art = json.loads((tmp_path / "ms.json").read_text())
    agg = [r for r in art["summary"] if "crafted_minus_honest_pp" in r]
    assert len(agg) == 1
    assert set(agg[0]) == {"experiment", "crafted_minus_honest_pp"}


def _make_model(torch, transformers, vocab, n_layer, beta, gamma, seed=0):
    """The dense-vocab DISTORTING tiny world: amplified attention output /
    MLP projections (beta — the trained/deep body's direction shear
    stand-in) and sharpened attention scores (gamma — peaky cross-position
    mixing stand-in) at a realistic vocab density."""
    torch.manual_seed(seed)
    cfg = transformers.GPT2Config(
        n_layer=n_layer, n_embd=32, n_head=4, vocab_size=vocab,
        n_positions=64, bos_token_id=0, eos_token_id=vocab - 1,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = transformers.GPT2LMHeadModel(cfg)
    with torch.no_grad():
        for blk in model.transformer.h:
            blk.attn.c_proj.weight.mul_(beta)
            blk.mlp.c_proj.weight.mul_(beta)
            blk.attn.c_attn.weight.mul_(gamma)
    return model


def _capture(torch, splitbench, model, cap_dir, batches, server_attack,
             alpha=8.0):
    plan = splitbench.SplitPlan.parse("1", n_layers=model.config.n_layer)
    split = splitbench.build_split(model, plan)
    writer = splitbench.CaptureWriter(str(cap_dir), "training",
                                      f"t-ms-{server_attack}")
    bench = splitbench.BenchConfig(steps=2, seq_len=8, batch_size=2,
                                   lr=1e-4, capture_every=1,
                                   server_attack=server_attack,
                                   server_attack_alpha=alpha,
                                   save_canonical=True)
    splitbench.finetune(split, bench, batches, writer)
    writer.close()
    return str(cap_dir)


@pytest.fixture
def distort_run(tmp_path):
    """Honest vs crafted captures of the dense-vocab distorting world
    (4L/32/V=32000, beta=16, gamma=8 — measured: honest 0.429, crafted
    1.000)."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from attacker import splitbench
    model = _make_model(torch, transformers, 32000, 4, 16.0, 8.0)
    model_dir = tmp_path / "distortgpt2"
    model.save_pretrained(model_dir)
    g = torch.Generator().manual_seed(11)
    batches = [torch.randint(0, 32000, (2, 8), generator=g)
               for _ in range(2)]
    honest_dir = _capture(torch, splitbench, model, tmp_path / "caps_h",
                          batches, "none")
    crafted_dir = _capture(torch, splitbench, model, tmp_path / "caps_c",
                           batches, "passive-amplify")
    return str(honest_dir), str(crafted_dir), str(model_dir)


def _ms_args(honest_dir, crafted_dir, model_dir, *extra):
    return parse("--capture-dir", honest_dir, "--canonical-pt",
                 honest_dir + "/canonical.pt", "--crafted-dir",
                 crafted_dir, "--crafted-canonical-pt",
                 crafted_dir + "/canonical.pt", "--model", model_dir,
                 "--split", "1", "--samples", "4", *extra)


@pytest.mark.slow
def test_torch_crafted_world_dramatically_outperforms_honest(
        distort_run, tmp_path):
    honest_dir, crafted_dir, model_dir = distort_run
    out = str(tmp_path / "ms.json")
    assert ms.run(_ms_args(honest_dir, crafted_dir, model_dir,
                           "--evidence", "--output", out)) == 0
    art = json.load(open(out))
    cells = art["results"]
    assert sorted(set(c["world"] for c in cells)) == ["crafted", "honest"]
    assert all(c["submode"] == "real-greedy" for c in cells)
    summary = {s["world"]: s for s in art["summary"] if "world" in s}
    honest_acc = summary["honest"]["seq_accuracy"]
    crafted_acc = summary["crafted"]["seq_accuracy"]
    margin = [s for s in art["summary"]
              if "crafted_minus_honest_pp" in s][0]
    assert honest_acc < 0.75
    assert crafted_acc >= 0.95
    assert margin["crafted_minus_honest_pp"] > 30.0
    assert summary["crafted"]["gate"]["verdict"] == "FAIL"
    assert summary["honest"]["n_tokens"] == summary["crafted"]["n_tokens"]
    journal = [json.loads(l) for l in open(out + ".jsonl")]
    assert len(journal) == len(cells)
    from attacker.evidence import verify_evidence
    ok, problems = verify_evidence(out + ".evidence.json")
    assert ok and problems == []


@pytest.mark.slow
def test_torch_plain_tiny_world_saturates_even_honest(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from attacker import splitbench
    model = _make_model(torch, transformers, 99, 2, 1.0, 1.0)
    model_dir = tmp_path / "tinygpt2"
    model.save_pretrained(model_dir)
    g = torch.Generator().manual_seed(11)
    batches = [torch.randint(0, 99, (2, 8), generator=g) for _ in range(2)]
    honest_dir = _capture(torch, splitbench, model, tmp_path / "caps_h",
                          batches, "none")
    out = str(tmp_path / "ms_tiny.json")
    assert ms.run(_ms_args(honest_dir, honest_dir, str(model_dir),
                           "--output", out)) == 0
    summary = {s["world"]: s for s in json.load(open(out))["summary"]
               if "world" in s}
    assert summary["honest"]["seq_accuracy"] >= 0.9


@pytest.mark.slow
def test_torch_server_attack_hook_changes_the_wire(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from attacker import captures, splitbench
    model = _make_model(torch, transformers, 99, 2, 1.0, 1.0)
    g = torch.Generator().manual_seed(11)
    batches = [torch.randint(0, 99, (2, 8), generator=g) for _ in range(1)]
    honest_dir = _capture(torch, splitbench, model, tmp_path / "h",
                          batches, "none")
    crafted_dir = _capture(torch, splitbench, model, tmp_path / "c",
                           batches, "passive-amplify")
    recs_h = captures.scan_captures(honest_dir, "training")
    recs_c = captures.scan_captures(crafted_dir, "training")
    g_h = captures.load_tensor([p for m, p in recs_h
                                if m["phase"] == "bwd"][0])
    g_c = captures.load_tensor([p for m, p in recs_c
                                if m["phase"] == "bwd"][0])
    assert not torch.allclose(g_h, g_c)
    with pytest.raises(ValueError, match="server_attack"):
        splitbench.BenchConfig(server_attack="active-x").validate()
    with pytest.raises(ValueError, match="server_attack_alpha"):
        splitbench.BenchConfig(server_attack_alpha=-1.0).validate()
    assert splitbench.BenchConfig(
        server_attack="passive-amplify").validate()
