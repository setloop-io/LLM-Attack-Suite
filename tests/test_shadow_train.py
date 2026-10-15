"""shadow-train: the internal shadow-model trainer that generates LIRA /
private-phase reference statistics. Corpus generation, config and the CLI
parser are torch-free; every training test is torch-gated per repo
convention (pytest.importorskip)."""

import dataclasses
import json

import pytest

from attacker import shadow_train as st
from attacker.attacks import lira_membership as lm
from attacker.attacks import representation_privacy as rp


def small_cfg(**over):
    base = dict(n_shadows=4, steps=200, hidden=32, dim=16, n_examples=48,
                n_public=64, seed=7)
    base.update(over)
    return st.ShadowConfig(**base)


def test_config_defaults_and_parser_torchfree():
    cfg = st.ShadowConfig()
    assert (cfg.n_shadows, cfg.steps, cfg.lr, cfg.hidden, cfg.dim,
            cfg.vocab, cfg.seq_len, cfg.n_examples, cfg.n_public,
            cfg.seed, cfg.arch) == (4, 200, 0.01, 64, 32, 64, 16, 64, 256,
                                    1234, "mlp")
    args = st.build_parser().parse_args(["--output", "x.jsonl"])
    assert (args.n_shadows, args.steps, args.seed) == (4, 200, 1234)
    assert args.phase_bundle is None


def test_corpus_deterministic_half_members_with_signal():
    a = st.make_synthetic_corpus(st.ShadowConfig(), 5)
    b = st.make_synthetic_corpus(st.ShadowConfig(), 5)
    assert a == b
    assert len(a["ids"]) == 64 and len(a["members"]) == 32
    assert all(len(row) == 16 and all(0 <= t < 64 for t in row)
               for row in a["ids"])
    assert set(a["labels"]) == {0, 1}
    c = st.make_synthetic_corpus(st.ShadowConfig(), 6)
    assert c["ids"] != a["ids"]


def test_unknown_arch_is_a_descriptive_error():
    pytest.importorskip("torch")
    cfg = dataclasses.replace(st.ShadowConfig(), arch="transformer")
    with pytest.raises(SystemExit, match="unknown arch"):
        st._build_model(cfg)


def test_records_determinism_byte_identical():
    pytest.importorskip("torch")
    cfg = small_cfg()
    first = [json.dumps(r, sort_keys=True) for r in st.build_lira_records(cfg)]
    second = [json.dumps(r, sort_keys=True) for r in st.build_lira_records(cfg)]
    assert first == second


def test_records_schema_roundtrip_through_load_records(tmp_path):
    pytest.importorskip("torch")
    cfg = small_cfg()
    records = st.build_lira_records(cfg)
    path = tmp_path / "recs.jsonl"
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    loaded = lm.load_records([str(path)])
    assert len(loaded) == cfg.n_examples
    assert sum(r["split"] == "in" for r in loaded) == cfg.n_examples // 2
    members = [r for r in loaded if r["split"] == "in"]
    assert all(len(r["ref_in"]) >= 2 and len(r["ref_out"]) >= 2
               for r in members)


def test_planted_signal_lira_per_example_auc():
    pytest.importorskip("torch")
    cfg = small_cfg(n_shadows=4)
    scored = lm.score_records(st.build_lira_records(cfg))
    ins = [r for r in scored if r["split"] == "in"]
    assert all(r["model"] == "per_example" for r in ins)
    auc = lm.rank_auc([r["llr"] for r in ins],
                      [r["llr"] for r in scored if r["split"] == "out"])
    assert auc > 0.8


def test_rmia_path_runs_every_record_has_two_ref_out():
    pytest.importorskip("torch")
    records = st.build_lira_records(small_cfg())
    assert all(len(r["ref_out"]) >= 2 for r in records)
    scored = lm.rmia_scores(records)
    assert all(0.0 <= r["rmia"] <= 1.0 for r in scored)
    ins = [r["rmia"] for r in scored if r["split"] == "in"]
    outs = [r["rmia"] for r in scored if r["split"] == "out"]
    assert lm.rank_auc(ins, outs) > 0.6
    assert sum(ins) / len(ins) > sum(outs) / len(outs)


def test_phase_bundle_passes_validation_and_audits():
    pytest.importorskip("torch")
    bundle = st.build_phase_bundle(small_cfg())
    rp.validate_bundle(bundle)
    rows = rp.validate_phase(bundle)
    assert bundle["phase_provenance"]["references_attacker_owned"] is True
    assert bundle["phase_provenance"]["statistic"] == \
        "negative_log_likelihood"
    assert len(bundle["phase_provenance"]["reference_models"]) >= 2
    assert all(len(r["ref_in"]) >= 2 and len(r["ref_out"]) >= 2
               for r in rows)
    audit = rp.phase_audit(bundle, "training")
    for rec in audit["results"]:
        if rec["method"] == "lira_per_example":
            assert rec["auc"] > 0.8


def test_online_end_to_end(tmp_path):
    pytest.importorskip("torch")
    out = str(tmp_path / "online.json")
    args = lm.build_parser().parse_args(
        ["--online", "--n-shadows", "4", "--shadow-steps", "40",
         "--output", out])
    args.mode = "training"
    assert lm.run(args) == 0
    artifact = json.load(open(out))
    assert artifact["config"]["n_shadows"] == 4
    assert artifact["config"]["shadow_steps"] == 40
    rec = artifact["results"][0]
    assert rec["world"] == "online_shadow"
    assert rec["gate"]["verdict"] in ("PASS", "FAIL", "INSUFFICIENT_EVIDENCE")
    assert len(open(out + ".jsonl").read().splitlines()) == 1


def test_online_requires_two_shadows():
    pytest.importorskip("torch")
    args = lm.build_parser().parse_args(["--online", "--n-shadows", "1"])
    args.mode = "training"
    with pytest.raises(SystemExit, match="--n-shadows must be >=2"):
        lm.run(args)


def test_shadow_toy_end_to_end(tmp_path):
    pytest.importorskip("torch")
    out = tmp_path / "audit.json"
    args = rp.build_parser().parse_args(
        ["--shadow-toy", "--output", str(out), "--arms", "bottleneck",
         "private_phase"])
    args.mode = "training"
    assert rp.run(args) == 0
    artifact = json.loads(out.read_text())
    assert artifact["config"]["shadow_toy"] is True
    prov = artifact["phase_provenance"]
    assert prov["references_attacker_owned"] is True
    assert len(prov["reference_models"]) >= 2
    assert set(artifact["results"]["arms"]) == {"bottleneck",
                                                "private_phase"}


def test_cli_main_writes_records_and_bundle(tmp_path):
    pytest.importorskip("torch")
    recs = tmp_path / "records.jsonl"
    bundle = tmp_path / "bundle.json"
    assert st.main(["--output", str(recs), "--phase-bundle", str(bundle),
                    "--n-shadows", "4", "--steps", "20", "--seed", "3"]) == 0
    loaded = lm.load_records([str(recs)])
    assert len(loaded) == st.ShadowConfig.n_examples
    rp.validate_phase(json.loads(bundle.read_text()))


def _tiny_lm_dir(tmp_path, vocab=48):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.GPT2Config(vocab_size=vocab, n_embd=16, n_layer=2,
                                  n_head=2, n_positions=16, n_ctx=16)
    model = transformers.GPT2LMHeadModel(cfg)
    path = tmp_path / "tiny-gpt2"
    model.save_pretrained(path)
    return str(path)


def _ids_corpus(tmp_path, n=12, seq_len=8, vocab=48):
    import random as _r
    rng = _r.Random(5)
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(
        json.dumps({"id": f"doc-{i:03d}",
                    "ids": [rng.randrange(vocab) for _ in range(seq_len)]})
        + "\n" for i in range(n)))
    return str(path)


def test_load_lm_corpus_ids_torchfree(tmp_path):
    path = _ids_corpus(tmp_path)
    rows = st.load_lm_corpus(path, 8)
    assert len(rows) == 12 and all(len(r["ids"]) == 8 for r in rows)
    assert rows[0]["id"] == "doc-000"
    with pytest.raises(SystemExit, match="tokens < seq_len"):
        st.load_lm_corpus(path, 16)


def test_lm_shadow_records_schema_and_scoring(tmp_path):
    pytest.importorskip("torch")
    model_path = _tiny_lm_dir(tmp_path)
    corpus_path = _ids_corpus(tmp_path)
    cfg = st.ShadowConfig(n_shadows=4, steps=3, seed=7, seq_len=8)
    rows = st.load_lm_corpus(corpus_path, 8)
    records = st.build_lira_records_lm(cfg, rows, model_path)
    assert len(records) == 12
    assert sum(r["split"] == "in" for r in records) == 6
    assert all(r["example_id"].startswith("doc-") for r in records)
    scored = lm.score_records(records)
    metrics = lm.lira_metrics(scored)
    assert 0.0 <= metrics["auc"] <= 1.0
    assert metrics["n_in"] == 6 and metrics["n_out"] == 6


def test_cli_lm_path_requires_model(tmp_path):
    pytest.importorskip("torch")
    corpus_path = _ids_corpus(tmp_path)
    with pytest.raises(SystemExit, match="--corpus-jsonl requires --model"):
        st.main(["--output", str(tmp_path / "r.jsonl"),
                 "--corpus-jsonl", corpus_path])
