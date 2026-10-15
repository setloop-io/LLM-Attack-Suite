"""Regression checks for release-blocking scientific and execution defects."""
import json
import pytest
from attacker import gate, scorecard
from attacker.evidence import EvidenceLogger, verify_evidence


def test_gate_does_not_pass_missing_or_insufficient_seeds():
    for seeds in (None, 1, 2):
        assert gate.verdict(.04, .04, 8192, n_seeds=seeds)["verdict"] == "INSUFFICIENT_EVIDENCE"
    for rate in (float("nan"), float("inf"), -1, 2):
        with pytest.raises(ValueError): gate.verdict(rate, .04, 8192)


def test_scorecard_selects_upper_bound_and_deduplicates(tmp_path):
    art = {"schema": "test", "config": {}, "results": [],
           "summary": [{"raw_excess_pp": -1, "upper95_excess_over_majority_pp": 3, "n_seeds": 3}]}
    path = tmp_path/"a.json"; path.write_text(json.dumps(art))
    copy = tmp_path/"b.json"; copy.write_text(json.dumps(art))
    card = scorecard.build_scorecard([path, copy, path])
    assert card["summary"]["overall"] == "FAIL"
    assert card["summary"]["n_artifacts"] == 1
    assert card["worst_cell"]["excess_pp"] == 3


def test_result_gate_and_error_cannot_be_hidden_by_summary(tmp_path):
    art = {"schema": "test", "config": {},
           "results": [{"gate": gate.verdict(.2, .04, 8192)}],
           "summary": [{"upper95_excess_over_majority_pp": .2, "n_seeds": 3}]}
    path = tmp_path/"a.json"; path.write_text(json.dumps(art))
    assert scorecard.build_scorecard([path])["summary"]["overall"] == "FAIL"
    art["results"] = [{"error": "OOM"}]
    path.write_text(json.dumps(art))
    assert scorecard.build_scorecard([path])["summary"]["overall"] == "INCOMPLETE"


def test_manifest_metadata_is_bound_and_external_digest_supported(tmp_path):
    path = tmp_path/"a.json"; path.write_text('{}')
    with EvidenceLogger(path) as ev: ev.event("test", {})
    manifest_path = str(path)+".evidence.json"
    manifest = json.loads(open(manifest_path).read())
    assert verify_evidence(manifest_path, manifest["manifest_sha256"])[0]
    manifest["hostname"] = "fabricated"
    open(manifest_path, "w").write(json.dumps(manifest))
    assert not verify_evidence(manifest_path)[0]


def test_training_updates_client_and_inference_records_original_truth(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from attacker import splitbench as sb
    torch.manual_seed(23)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=32, n_layer=2,
            n_embd=16, n_head=2, n_positions=32, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.))
    split = sb.build_split(model, sb.SplitPlan.parse("1", n_layers=2))
    parameter = model.transformer.h[0].mlp.c_fc.weight
    before = parameter.detach().clone()
    ids = torch.tensor([[1,2,3,4]])
    cfg = sb.BenchConfig(steps=1, split="1", save_canonical=True, lr=.01)
    with sb.CaptureWriter(str(tmp_path/"train"), "training", "test") as w:
        sb.finetune(split, cfg, [ids], w)
    assert not torch.equal(before, parameter)
    with sb.CaptureWriter(str(tmp_path/"infer"), "inference", "test") as w:
        sb.infer(split, cfg, [ids], w)
    truth = json.loads((tmp_path/"infer"/"canonical.json").read_text())
    assert truth[0]["prompt_ids"] == ids.tolist()
    assert len(truth[0]["generated_ids"]) == 1


def test_experiment_document_partition_and_uncertainty(tmp_path):
    from attacker.experiment import partition, load_documents, interval
    docs = [{"id": str(i), "text": f"document {i}"} for i in range(12)]
    groups = partition(docs, 42)
    assert sum(map(len, groups.values())) == len(docs)
    assert len({d["id"] for part in groups.values() for d in part}) == len(docs)
    assert interval([1,2])["lower"] is None
    if pytest.importorskip("scipy"):
        estimate = interval([1,2,3])
        assert estimate["lower"] < 2 < estimate["upper"]
    path = tmp_path/"docs.jsonl"
    path.write_text('\n'.join(json.dumps(d) for d in docs + [docs[0]]))
    with pytest.raises(ValueError, match="duplicate"): load_documents(path)


def test_merge_preserves_failure_in_any_environment():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("release_suite", Path(__file__).with_name("run_suite.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    records = [("a", {"tests": [{"nodeid": "x", "outcome": "failed"}]}),
               ("b", {"tests": [{"nodeid": "x", "outcome": "passed"}]})]
    assert mod.merge(records, [])["failed"] == ["x"]
    assert mod.merge([], [])["failed"]


def test_experiment_full_smoke_and_component_tamper(tmp_path):
    pytest.importorskip("torch"); pytest.importorskip("transformers"); pytest.importorskip("scipy")
    from attacker.experiment import main, verify_run
    folder = tmp_path/"experiment"
    assert main(["--smoke", "--output", str(folder), "--seeds", "7", "--train-steps", "2",
                 "--attack-steps", "2"]) == 0
    report = json.loads((folder/"experiment.json").read_text())
    assert report["status"] == "COMPLETE"
    assert len(report["results"]) == 10
    assert {c["mode"] for c in report["results"]} == {"training", "inference"}
    assert all(c["training"]["client_parameter_change_l2"] > 0 for c in report["results"])
    assert all(c["attack_minus_control_pp"]["status"] == "INSUFFICIENT_EVIDENCE" for c in report["summary"])
    assert all(c["attack_config"]["seeds"] == [7] for c in report["results"])
    assert verify_run(folder)[0]
    (folder/"calibration.pt").write_bytes(b"tampered")
    assert not verify_run(folder)[0]


def test_relocated_evidence_verifies_local_copy_not_original(tmp_path):
    import shutil
    origin = tmp_path/"origin"; origin.mkdir()
    artifact = origin/"run.json"; artifact.write_text('{}')
    with EvidenceLogger(artifact): pass
    moved = tmp_path/"moved"; shutil.copytree(origin, moved)
    (moved/"run.json").write_text('{"tampered": true}')
    assert not verify_evidence(moved/"run.json.evidence.json")[0]
    assert verify_evidence(origin/"run.json.evidence.json")[0]
