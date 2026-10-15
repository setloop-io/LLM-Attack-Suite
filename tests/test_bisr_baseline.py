"""bisr-baseline: contract/registry/metadata, torch-free parser + toy tier
(bidirectional candidate union + forward-consistency beam vs learned
forward-only vs position-independent NN on a synthetic position-dependent
seam with a toy backward channel), the toy BRE optimization arms (toy
gradient matching Eq. 9 / smashed-data matching Eq. 7-8 with analytic
gradients, toy NaMoE two-stage training), the fail-closed auxiliary-corpus
disjointness assertion, determinism, and self_test; plus the slow torch
layer — a tiny random GPT-2 captured through splitbench, a SIP-style
inverter trained 100 steps on synthetic draws, then attacked; the
aux-scale streaming shard cache (resume-safe, fingerprint-pinned); the
NaMoE checkpoint round-trip; and the BRE --opt-arm attack arms."""

import json

import pytest

from attacker import artifacts
from attacker.attacks import bisr_baseline as bb


def parse(*argv):
    args = bb.build_parser().parse_args(list(argv))
    args.mode = "training"
    return args


def test_module_contract_and_register_dispatch():
    from attacker import attacks
    attacks.validate_contract(bb, "bisr-baseline")
    assert bb.EXPERIMENT_ID == "bisr_baseline"
    assert set(bb.MODES) == {"training", "inference"}
    assert bb.REQUIRES_LABELS is False
    assert "adaptation" in bb.DESCRIPTION
    if "bisr-baseline" not in attacks.ATTACKS:
        attacks.register_attack("bisr-baseline", bb)
    assert attacks.load("bisr-baseline") is bb
    assert "bisr-baseline" in attacks.for_mode("training")
    assert "bisr-baseline" in attacks.for_mode("inference")


def test_metadata_entry_matches_vocabulary():
    from attacker import attack_metadata as am
    md = am.metadata_for("bisr-baseline")
    assert md["surfaces"] == ("S1", "S2")
    assert set(md["lifecycle_stages"]) == {"training", "inference"}
    am.validate_metadata("bisr-baseline", md)


def test_parser_builds_torchless():
    args = bb.build_parser().parse_args([])
    assert args.phase == "attack"
    assert args.candidates == 8 and args.beam_width == 2
    assert args.split == "6"
    assert args.train_steps == 500 and args.aux_seqs == 512
    assert args.evidence is False
    with pytest.raises(SystemExit):
        bb.build_parser().parse_args(["--help"])
    with pytest.raises(SystemExit):
        bb.build_parser().parse_args(["--phase", "nope"])


def test_toy_world_deterministic_and_shaped():
    w1 = bb.toy_world(0, vocab=60, dim=16, seq_len=12, n_victim=4,
                      n_aux=20, sigma=0.3)
    w2 = bb.toy_world(0, vocab=60, dim=16, seq_len=12, n_victim=4,
                      n_aux=20, sigma=0.3)
    assert w1["victim"] == w2["victim"]
    assert w1["h_victim"] == w2["h_victim"]
    assert w1["g_victim"] == w2["g_victim"]
    assert len(w1["h_victim"]) == 4 and len(w1["h_victim"][0]) == 12
    assert len(w1["g_victim"][0]) == 12
    assert len(w1["A"]) == 12 and len(w1["B"]) == 12


def test_toy_bidirectional_selection_and_beam():
    args = parse("--toy")
    for seed in (0, 1, 2):
        cell = bb.toy_cell(args, 0.0, seed)
        assert cell["seq_accuracy"] > 0.85
        assert cell["seq_accuracy"] >= cell["sip_fwd_only"]["seq_accuracy"]
        assert cell["seq_accuracy"] \
            > cell["baseline_nn"]["seq_accuracy"] + 0.2
        assert cell["gate"]["verdict"] == "FAIL"
        assert cell["bleu"] > 0.7
        assert "gate" in cell["sip_fwd_only"] and "gate" in cell["baseline_nn"]


def test_toy_candidates_union_and_position_zero_forward_only():
    world = bb.toy_world(0, vocab=30, dim=8, seq_len=6, n_victim=1,
                         n_aux=40, sigma=0.0)
    tables = bb.toy_train_centroids(world["aux"], world["h_aux"], 30, 6, 8)
    cand = bb.toy_candidates(world, tables, world["h_victim"][0],
                             world["g_victim"][0], 4)
    assert len(cand) == 6
    assert all(len(set(c)) == 4 for c in cand)
    true = world["victim"][0]
    assert all(true[p] in cand[p] for p in range(6))


def test_toy_noise_degrades_gracefully():
    args = parse("--toy", "--seeds", "0")
    accs = [bb.toy_cell(args, s, 0)["sip_fwd_only"]["seq_accuracy"]
            for s in (0.0, 0.3, 1.0, 2.0, 4.0)]
    assert accs[0] > 0.8
    assert accs[-1] <= accs[0]


def test_toy_run_journals_artifact_and_evidence(tmp_path):
    from attacker.evidence import verify_evidence
    out = str(tmp_path / "bisr.json")
    assert bb.run(parse("--toy", "--quick", "--evidence",
                        "--output", out)) == 0
    art = json.load(open(out))
    assert art["schema"] == bb.SCHEMA
    assert art["config"]["adaptation"] is True
    assert len(art["results"]) == 3
    for cell in art["results"]:
        for key in ("seq_accuracy", "control_accuracy",
                    "token_set_precision", "token_set_recall", "bleu",
                    "sip_fwd_only", "baseline_nn", "gate"):
            assert key in cell
    journal = [json.loads(l) for l in open(out + ".jsonl")]
    assert [j["sigma"] for j in journal] == [0.0, 0.5, 2.0]
    assert art["summary"] \
        and "sip_fwd_only_accuracy_mean" in art["summary"][0]
    ok, problems = verify_evidence(out + ".evidence.json")
    assert ok and problems == []
    with pytest.raises(SystemExit, match="--evidence requires --output"):
        bb.run(parse("--toy", "--quick", "--evidence"))


def _write_aux(tmp_path, ids):
    p = tmp_path / "aux.jsonl"
    p.write_text("".join(json.dumps({"id": i, "text": "hello world"})
                         + "\n" for i in ids))
    return str(p)


def _write_manifest(tmp_path, ids):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(
        {"schema": "llm_attacker.validation_corpus.v1",
         "selected_ids": {"calibration": ids[:1], "evaluation": ids[1:]}}))
    return str(p)


def test_disjointness_assertion_fires_on_contamination(tmp_path):
    aux = _write_aux(tmp_path, ["aux-a", "aux-b"])
    manifest = _write_manifest(tmp_path, ["clean-1", "aux-b"])
    with pytest.raises(ValueError, match="aux-b"):
        bb.assert_aux_disjoint(aux, manifest, None)


def test_disjointness_passes_clean_and_records_audit(tmp_path):
    aux = _write_aux(tmp_path, ["aux-a", "aux-b"])
    manifest = _write_manifest(tmp_path, ["clean-1", "clean-2"])
    rec = bb.assert_aux_disjoint(aux, manifest, None)
    assert rec["disjoint"] is True
    assert rec["aux_articles"] == 2
    assert "frozen_validation_manifest" in rec["checked_against"]


def test_disjointness_missing_references_are_skipped_not_passed(tmp_path):
    aux = _write_aux(tmp_path, ["aux-a"])
    rec = bb.assert_aux_disjoint(aux, str(tmp_path / "nope.json"),
                                 str(tmp_path / "nope.jsonl"))
    assert rec["skipped_missing"] == ["frozen_validation_manifest",
                                      "pilot_corpus"]
    assert rec["checked_against"] == []


def test_disjointness_checks_the_pilot_corpus_too(tmp_path):
    aux = _write_aux(tmp_path, ["aux-a"])
    pilot = _write_aux(tmp_path, ["aux-a"])
    manifest = _write_manifest(tmp_path, ["clean-1"])
    with pytest.raises(ValueError, match="pilot_corpus"):
        bb.assert_aux_disjoint(aux, manifest, pilot)


def test_load_article_ids_rejects_duplicates_and_missing_ids(tmp_path):
    dup = _write_aux(tmp_path, ["x", "x"])
    with pytest.raises(ValueError, match="duplicate"):
        bb.load_article_ids(dup)
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"text": "no id"}) + "\n")
    with pytest.raises(ValueError, match="id"):
        bb.load_article_ids(str(p))


def test_real_path_errors_are_descriptive_torchless():
    with pytest.raises(SystemExit, match="--capture-dir"):
        bb.run(parse())
    with pytest.raises(SystemExit, match="--inverter-pt"):
        bb.run(parse("--phase", "train"))
    with pytest.raises(SystemExit, match="--inverter-pt"):
        bb.run(parse("--capture-dir", "/nonexistent",
                     "--canonical-pt", "/nonexistent"))
    with pytest.raises(SystemExit, match="--model"):
        bb.run(parse("--capture-dir", "/nonexistent",
                     "--canonical-pt", "/nonexistent",
                     "--inverter-pt", "/nonexistent"))
    with pytest.raises(SystemExit, match="--candidates"):
        bb.run(parse("--capture-dir", "/nonexistent",
                     "--canonical-pt", "/nonexistent",
                     "--inverter-pt", "/nonexistent",
                     "--model", "/nonexistent", "--candidates", "3"))
    if bb.torch is None:
        with pytest.raises(SystemExit, match="requires torch"):
            bb.run(parse("--capture-dir", "/nonexistent",
                         "--canonical-pt", "/nonexistent",
                         "--inverter-pt", "/nonexistent",
                         "--model", "/nonexistent"))
        with pytest.raises(SystemExit, match="requires torch"):
            bb.run(parse("--phase", "train", "--inverter-pt", "/x.pt",
                         "--model", "/nonexistent"))


def test_module_self_test_passes():
    assert bb.self_test() == 0


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
    writer = splitbench.CaptureWriter(str(cap_dir), "training", "t-bisr")
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


def _train_args(model_dir, inv_pt, *extra):
    return parse("--phase", "train", "--model", model_dir, "--split", "1",
                 "--seq-len", "8", "--aux-seqs", "256",
                 "--train-steps", "100",
                 "--inverter-pt", inv_pt, *extra)


def _attack_args(cap_dir, inv_pt, model_dir, *extra):
    return parse("--phase", "attack", "--capture-dir", cap_dir,
                 "--canonical-pt", cap_dir + "/canonical.pt",
                 "--inverter-pt", inv_pt, "--model", model_dir,
                 "--split", "1", "--samples", "4", *extra)


@pytest.mark.slow
def test_torch_train_checkpoint_format_and_disjointness(tiny_run, tmp_path):
    torch, _, _, model_dir = tiny_run
    inv_pt = str(tmp_path / "bisr.pt")
    out = str(tmp_path / "train.json")
    assert bb.run(_train_args(model_dir, inv_pt, "--output", out)) == 0
    ckpt = torch.load(inv_pt, map_location="cpu")
    assert ckpt["format"] == bb.CKPT_FORMAT
    assert isinstance(ckpt["control_token"], int)
    assert ckpt["arch"]["seq_len"] == 8 and ckpt["arch"]["seam_dim"] == 32
    assert ckpt["arch"]["vocab"] == VOCAB
    assert ckpt["train"]["disjointness"]["public_source"] == \
        "synthetic-uniform"
    art = json.load(open(out))
    assert art["results"][0]["phase"] == "train"
    assert art["results"][0]["final_loss"]["step"] == 99


@pytest.mark.slow
def test_torch_attack_bidirectional_beats_control(tiny_run, tmp_path):
    torch, _, cap_dir, model_dir = tiny_run
    inv_pt = str(tmp_path / "bisr.pt")
    assert bb.run(_train_args(model_dir, inv_pt)) == 0

    out = str(tmp_path / "atk.json")
    args = _attack_args(cap_dir, inv_pt, model_dir, "--output", out,
                        "--evidence")
    assert bb.run(args) == 0
    art = json.load(open(out))
    cells = art["results"]
    assert len(cells) == 2
    assert all(c["submode"] == "learned-fwd+bwd-beam" for c in cells)
    assert all(c["arm"] == "bisr" for c in cells)
    summary = art["summary"][0]
    assert summary["bidirectional"] is True
    assert summary["n_tokens"] == 4 * 8
    assert summary["seq_accuracy"] >= summary["control_accuracy"] + 0.5
    assert summary["gate"]["verdict"] == "FAIL"
    assert "sip_fwd_only" in summary and "baseline_nn" in summary
    assert summary["work"]["head_forward_calls"] > 0
    from attacker.evidence import verify_evidence
    ok, problems = verify_evidence(out + ".evidence.json")
    assert ok and problems == []


@pytest.mark.slow
def test_torch_attack_rejects_mismatched_checkpoint(tiny_run, tmp_path):
    _, _, cap_dir, model_dir = tiny_run
    inv_pt = str(tmp_path / "bisr.pt")
    assert bb.run(_train_args(model_dir, inv_pt)) == 0
    other = str(tmp_path / "other.pt")
    assert bb.run(parse("--phase", "train", "--model", model_dir,
                        "--split", "1", "--seq-len", "16",
                        "--aux-seqs", "8", "--train-steps", "1",
                        "--inverter-pt", other)) == 0
    with pytest.raises(SystemExit, match="seq_len"):
        bb.run(_attack_args(cap_dir, other, model_dir))


def test_parser_bre_namoe_auxscale_flags_torchless():
    args = bb.build_parser().parse_args([])
    assert args.opt_arm == "none"
    assert args.gm_iters == 32 and args.gm_lr == 0.09
    assert args.gm_beta == 0.85 and args.gm_tau == 1.2
    assert args.sm_iters == 800 and args.sm_lr == 0.01 and args.sm_wd == 0.01
    assert args.opt_batch == 8
    assert args.namoe is False
    assert list(args.namoe_sigmas) == [0.0, 0.1, 0.3, 1.0]
    assert args.gating_steps == 100
    assert args.aux_chunk_seqs == 2048 and args.aux_cache is None
    assert args.toy_opt is False and args.toy_ft_gap == 0.0
    args = bb.build_parser().parse_args(
        ["--opt-arm", "bf", "--namoe", "--namoe-sigmas", "0.0", "0.5",
         "--gating-steps", "7", "--aux-chunk-seqs", "16", "--toy-opt",
         "--toy-ft-gap", "0.05"])
    assert args.opt_arm == "bf" and args.namoe is True
    assert list(args.namoe_sigmas) == [0.0, 0.5] and args.gating_steps == 7
    assert args.aux_chunk_seqs == 16 and args.toy_opt is True
    assert args.toy_ft_gap == 0.05
    with pytest.raises(SystemExit):
        bb.build_parser().parse_args(["--opt-arm", "fb"])


def test_opt_arm_inference_refusal_is_descriptive_torchless():
    for arm in ("b", "bf"):
        args = parse("--capture-dir", "/nonexistent",
                     "--canonical-pt", "/nonexistent",
                     "--inverter-pt", "/nonexistent",
                     "--model", "/nonexistent", "--opt-arm", arm)
        args.mode = "inference"
        with pytest.raises(SystemExit, match="--opt-arm"):
            bb.run(args)
    args = parse("--capture-dir", "/nonexistent",
                 "--canonical-pt", "/nonexistent",
                 "--inverter-pt", "/nonexistent",
                 "--model", "/nonexistent", "--opt-arm", "f")
    args.mode = "inference"
    try:
        bb.run(args)
        raised = None
    except SystemExit as e:
        raised = str(e)
    except FileNotFoundError:
        raised = None
    assert raised is None or "--opt-arm" not in raised


def test_registry_still_one_module_entry():
    from attacker import attacks
    assert "bisr-baseline" in attacks.ATTACKS
    assert not any(n.startswith("bisr-") and n != "bisr-baseline"
                   for n in attacks.ATTACKS)


def test_toy_opt_world_deterministic_and_bwd_convention():
    w1 = bb.toy_opt_world(0, vocab=60, dim=16, seq_len=12, n_victim=4,
                          n_aux=40, sigma=0.3, ft_gap=0.05)
    w2 = bb.toy_opt_world(0, vocab=60, dim=16, seq_len=12, n_victim=4,
                          n_aux=40, sigma=0.3, ft_gap=0.05)
    assert w1["victim"] == w2["victim"]
    assert w1["h_victim"] == w2["h_victim"]
    assert w1["g_victim"] == w2["g_victim"]
    assert all(x == 0.0 for x in w1["g_victim"][0][-1])
    assert w1["A_pub"] != w1["A"] and w1["ft_gap"] == 0.05
    w0 = bb.toy_opt_world(0, vocab=60, dim=16, seq_len=12, n_victim=4,
                          n_aux=40, sigma=0.0, ft_gap=0.0)
    assert w0["A_pub"] == w0["A"] and w0["C_pub"] == w0["C"]


def test_toy_gm_recovers_labels_from_uniform_init():
    world = bb.toy_opt_world(1234, vocab=60, dim=16, seq_len=12,
                             n_victim=8, n_aux=300, sigma=0.0)
    hits = 0
    for i in range(3):
        for p in range(11):
            tok, _ = bb.toy_gm_position(world["h_victim"][i][p],
                                        world["g_victim"][i][p],
                                        world["C_pub"][p], [0.0] * 60,
                                        25, 0.09, 0.85, 1.2)
            hits += int(tok == world["victim"][i][p + 1])
    assert hits >= 31


def test_toy_sm_stays_at_a_good_init_and_decodes_nearest_row():
    world = bb.toy_opt_world(1234, vocab=60, dim=16, seq_len=12,
                             n_victim=4, n_aux=100, sigma=0.0)
    true = world["victim"][0]
    for p in (0, 5, 11):
        tok, loss = bb.toy_sm_position(world["h_victim"][0][p],
                                       world["A_pub"][p],
                                       list(world["E"][true[p]]),
                                       world["E"], 60, 0.01, 0.01)
        assert tok == true[p] and loss < 0.05


def test_toy_opt_cell_contrasts():
    args = parse("--toy", "--toy-opt", "--toy-opt-seqs", "8")
    cell = bb.toy_opt_cell(args, 0.0, 0)
    assert cell["arm"] == "bisr-opt-bf"
    assert cell["seq_accuracy"] > 0.9
    assert cell["seq_accuracy"] > cell["control_accuracy"] + 0.5
    assert cell["opt_f"]["seq_accuracy"] \
        >= cell["sip_init"]["seq_accuracy"] - 0.05
    assert cell["gate"]["verdict"] == "FAIL"
    for arm in ("opt_b", "opt_f", "sip_init"):
        assert "gate" in cell[arm] and "bleu" in cell[arm]
    assert cell["gm"]["iters"] == args.toy_gm_steps
    assert cell["sm"]["iters"] == args.toy_sm_steps
    noisy = bb.toy_opt_cell(args, 3.0, 0)
    assert noisy["seq_accuracy"] < cell["seq_accuracy"]
    assert noisy["seq_accuracy"] > noisy["control_accuracy"] + 0.2


def test_toy_namoe_flag_behavior():
    args = parse("--toy", "--toy-opt", "--namoe", "--toy-opt-seqs", "8",
                 "--gating-steps", "60")
    cell = bb.toy_opt_cell(args, 1.5, 0)
    assert "namoe" in cell and "namoe_bf" in cell
    rec = cell["namoe_training"]
    assert rec["expert_sigmas"] == [0.0, 0.1, 0.3, 1.0]
    assert rec["gating_steps"] == 60
    assert len(rec["gating_weights_mean"]) == 4
    assert all(0.0 <= w <= 1.0 for w in rec["gating_weights_mean"])
    assert cell["namoe"]["seq_accuracy"] \
        >= cell["sip_init"]["seq_accuracy"] - 0.15
    assert cell["namoe_bf"]["seq_accuracy"] \
        > cell["control_accuracy"] + 0.3
    plain = bb.toy_opt_cell(
        parse("--toy", "--toy-opt", "--toy-opt-seqs", "8"), 1.5, 0)
    assert "namoe" not in plain and "namoe_training" not in plain


def test_toy_opt_run_journals_artifact_and_evidence(tmp_path):
    from attacker.evidence import verify_evidence
    out = str(tmp_path / "bisr-opt.json")
    rc = bb.run(parse("--toy", "--quick", "--toy-opt", "--namoe",
                      "--evidence", "--output", out))
    assert rc == 0
    art = json.load(open(out))
    assert art["config"]["toy_opt"] is True
    assert art["config"]["namoe"] is True
    beam_cells = [c for c in art["results"] if c.get("arm") == "bisr"]
    opt_cells = [c for c in art["results"]
                 if c.get("arm") == "bisr-opt-bf"]
    assert len(beam_cells) == 3 and len(opt_cells) == 3
    for c in opt_cells:
        for key in ("seq_accuracy", "control_accuracy", "opt_b", "opt_f",
                    "sip_init", "namoe", "namoe_bf", "namoe_training",
                    "gm", "sm", "gate", "bleu"):
            assert key in c, key
    assert any(s.get("arm") == "bisr-opt-bf" for s in art["summary"])
    journal = [json.loads(l) for l in open(out + ".jsonl")]
    assert len(journal) == 6
    ok, problems = verify_evidence(out + ".evidence.json")
    assert ok and problems == []


@pytest.fixture
def tiny_aux_run(tiny_run, tmp_path):
    """tiny_run plus an offline word-level tokenizer (ids < VOCAB) and a
    disjoint aux JSONL, for the --aux-file streaming path."""
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    torch, model, cap_dir, model_dir = tiny_run
    vocab = {f"t{i}": i for i in range(VOCAB)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="t0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok).save_pretrained(
        model_dir)
    aux = tmp_path / "aux.jsonl"
    aux.write_text("".join(
        json.dumps({"id": f"aux-{i}",
                    "text": " ".join(f"t{(i * 7 + j) % VOCAB}"
                                     for j in range(60))}) + "\n"
        for i in range(40)))
    return torch, model, cap_dir, model_dir, str(aux)


def _stream_train_args(model_dir, aux, inv_pt, aux_seqs="24", *extra):
    return parse("--phase", "train", "--model", model_dir, "--split", "1",
                 "--seq-len", "8", "--aux-file", aux,
                 "--aux-seqs", aux_seqs, "--aux-chunk-seqs", "8",
                 "--train-steps", "10",
                 "--aux-manifest", "/nonexistent/manifest.json",
                 "--pilot-corpus", "/nonexistent/pilot.jsonl",
                 "--inverter-pt", inv_pt, *extra)


@pytest.mark.slow
def test_torch_streaming_aux_shards_resume_and_fingerprint(tiny_aux_run,
                                                           tmp_path):
    _, _, _, model_dir, aux = tiny_aux_run
    inv_pt = str(tmp_path / "bisr.pt")
    out = str(tmp_path / "train.json")
    assert bb.run(_stream_train_args(model_dir, aux, inv_pt,
                                     "24", "--output", out)) == 0
    cache = inv_pt + ".auxcache"
    journal = json.load(open(cache + "/journal.json"))
    assert journal["windows_done"] == 24
    assert len(journal["shards"]) == 3
    rec = json.load(open(out))["results"][0]
    assert rec["aux_stream"]["resumed"] is False
    assert rec["aux_stream"]["windows"] == 24
    assert rec["namoe"] is False
    out2 = str(tmp_path / "train2.json")
    assert bb.run(_stream_train_args(model_dir, aux, inv_pt,
                                     "24", "--output", out2)) == 0
    rec2 = json.load(open(out2))["results"][0]
    assert rec2["aux_stream"]["resumed"] is True
    assert len(json.load(open(cache + "/journal.json"))["shards"]) == 3
    with pytest.raises(SystemExit, match="fingerprint"):
        bb.run(_stream_train_args(model_dir, aux, inv_pt, "32"))


@pytest.mark.slow
def test_torch_streaming_aux_disjointness_fires(tiny_aux_run, tmp_path):
    _, _, _, model_dir, aux = tiny_aux_run
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(
        {"schema": "llm_attacker.validation_corpus.v1",
         "selected_ids": {"evaluation": ["aux-3"]}}))
    with pytest.raises(ValueError, match="aux-3"):
        bb.run(parse("--phase", "train", "--model", model_dir, "--split",
                     "1", "--seq-len", "8", "--aux-file", aux,
                     "--aux-seqs", "8", "--aux-chunk-seqs", "8",
                     "--train-steps", "1",
                     "--aux-manifest", str(manifest),
                     "--pilot-corpus", "/nonexistent.jsonl",
                     "--inverter-pt", str(tmp_path / "x.pt")))


@pytest.mark.slow
def test_torch_namoe_checkpoint_and_attack(tiny_run, tmp_path):
    torch, _, cap_dir, model_dir = tiny_run
    nm_pt = str(tmp_path / "namoe.pt")
    assert bb.run(parse("--phase", "train", "--model", model_dir,
                        "--split", "1", "--seq-len", "8", "--aux-seqs",
                        "64", "--train-steps", "10", "--namoe",
                        "--namoe-sigmas", "0.0", "0.5",
                        "--gating-steps", "5",
                        "--inverter-pt", nm_pt)) == 0
    ckpt = torch.load(nm_pt, map_location="cpu")
    assert ckpt["format"] == bb.CKPT_FORMAT_NAMOE
    assert ckpt["expert_sigmas"] == [0.0, 0.5]
    assert ckpt["arch"]["n_experts"] == 2
    assert ckpt["train"]["namoe"]["expert_sigmas"] == [0.0, 0.5]
    out = str(tmp_path / "atk.json")
    assert bb.run(_attack_args(cap_dir, nm_pt, model_dir, "--output", out,
                               "--opt-arm", "f", "--sm-iters", "3")) == 0
    art = json.load(open(out))
    assert art["config"]["namoe"] is True
    assert art["config"]["expert_sigmas"] == [0.0, 0.5]
    assert len(art["config"]["gating_weights_mean"]) == 2
    assert "bisr_opt" in art["summary"][0]
    plain_pt = str(tmp_path / "plain.pt")
    assert bb.run(_train_args(model_dir, plain_pt)) == 0
    with pytest.raises(SystemExit, match="plain SIP"):
        bb.run(_attack_args(cap_dir, plain_pt, model_dir, "--namoe"))


@pytest.mark.slow
def test_torch_opt_arm_bf_attack(tiny_run, tmp_path):
    _, _, cap_dir, model_dir = tiny_run
    inv_pt = str(tmp_path / "bisr.pt")
    assert bb.run(_train_args(model_dir, inv_pt)) == 0
    out = str(tmp_path / "atk.json")
    assert bb.run(_attack_args(cap_dir, inv_pt, model_dir, "--output",
                               out, "--evidence", "--opt-arm", "bf",
                               "--gm-iters", "5", "--sm-iters", "5")) == 0
    art = json.load(open(out))
    assert art["config"]["opt_arm"] == "bf"
    assert art["config"]["bre"]["gm_beta"] == 0.85
    cells = art["results"]
    assert all("bisr_opt" in c and "opt" in c for c in cells)
    assert cells[0]["opt"]["gm"]["gm_iters"] == 5
    assert cells[0]["opt"]["sm"]["sm_iters"] == 5
    summary = art["summary"][0]
    opt = summary["bisr_opt"]
    assert opt["arm"] == "bf"
    assert opt["seq_accuracy"] >= summary["control_accuracy"] + 0.5
    assert "gm" in opt["info"] and "sm" in opt["info"]
    from attacker.evidence import verify_evidence
    ok, problems = verify_evidence(out + ".evidence.json")
    assert ok and problems == []


@pytest.mark.slow
def test_torch_opt_arm_b_only_and_f_only(tiny_run, tmp_path):
    _, _, cap_dir, model_dir = tiny_run
    inv_pt = str(tmp_path / "bisr.pt")
    assert bb.run(_train_args(model_dir, inv_pt)) == 0
    out = str(tmp_path / "atkb.json")
    assert bb.run(_attack_args(cap_dir, inv_pt, model_dir, "--output",
                               out, "--opt-arm", "b",
                               "--gm-iters", "3")) == 0
    info_b = json.load(open(out))["summary"][0]["bisr_opt"]["info"]
    assert "gm" in info_b and "sm" not in info_b
    out = str(tmp_path / "atkf.json")
    assert bb.run(_attack_args(cap_dir, inv_pt, model_dir, "--output",
                               out, "--opt-arm", "f",
                               "--sm-iters", "3")) == 0
    info_f = json.load(open(out))["summary"][0]["bisr_opt"]["info"]
    assert "sm" in info_f and "gm" not in info_f


@pytest.fixture
def tiny_infer_run(tmp_path):
    """A tiny random GPT-2 with INFERENCE-mode (prefill) captures, for
    the forward-only surface (--opt-arm f allowed, b/bf refused)."""
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
    cap_dir = tmp_path / "capsinf"
    writer = splitbench.CaptureWriter(str(cap_dir), "inference", "t-inf")
    g = torch.Generator().manual_seed(5)
    prompts = [torch.randint(0, VOCAB, (1, 8), generator=g)
               for _ in range(3)]
    bench = splitbench.BenchConfig(steps=2, seq_len=8, batch_size=1,
                                   lr=1e-4, capture_every=1,
                                   save_canonical=True)
    splitbench.infer(split, bench, prompts, writer)
    writer.close()
    assert (cap_dir / "canonical.pt").is_file()
    return torch, model, str(cap_dir), str(model_dir)


@pytest.mark.slow
def test_torch_opt_arm_inference_surface(tiny_infer_run, tmp_path):
    _, _, cap_dir, model_dir = tiny_infer_run
    inv_pt = str(tmp_path / "bisr.pt")
    assert bb.run(_train_args(model_dir, inv_pt)) == 0
    args = _attack_args(cap_dir, inv_pt, model_dir, "--opt-arm", "f",
                        "--sm-iters", "3", "--samples", "3",
                        "--output", str(tmp_path / "inf.json"))
    args.mode = "inference"
    assert bb.run(args) == 0
    art = json.load(open(str(tmp_path / "inf.json")))
    summary = art["summary"][0]
    assert summary["bidirectional"] is False
    assert "bisr_opt" in summary
    assert summary["bisr_opt"]["arm"] == "f"
    assert "sm" in summary["bisr_opt"]["info"]
    for arm in ("b", "bf"):
        args = _attack_args(cap_dir, inv_pt, model_dir, "--opt-arm", arm)
        args.mode = "inference"
        with pytest.raises(SystemExit, match="--opt-arm"):
            bb.run(args)
