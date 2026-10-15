"""body-extraction (query-driven body-slice distillation, model-IP theft):
torch-free toy tier (closed-form student on a KNOWN linear body; planted
vs untrained control; weightsim non-identifiability), dispatch, journal/
artifact/evidence plumbing — plus the slow torch layer (tiny random GPT-2
local victim sim: victim body = blocks after the seam, student = reinit
slice, Adam distillation beats the untrained control)."""

import importlib
import json
import os

import pytest

from attacker.attacks import body_extraction as be
from attacker.evidence import verify_evidence


@pytest.fixture
def registered():
    """Dispatch through the real registry without editing ATTACKS."""
    from attacker import attacks
    mod = importlib.import_module("attacker.attacks.body_extraction")
    registered_here = "body-extraction" not in attacks.ATTACKS
    if registered_here:
        attacks.register_attack("body-extraction", mod)
    yield mod
    if registered_here:
        attacks.ATTACKS.pop("body-extraction", None)


def _run_toy(tmp_path, *extra, mode="training", name="be.json"):
    args = be.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(tmp_path / name), *extra])
    args.mode = mode
    assert be.run(args) == 0
    art = json.loads((tmp_path / name).read_text())
    journal = [json.loads(l) for l in
               (tmp_path / (name + ".jsonl")).read_text().splitlines()]
    return art, journal


def test_contract_and_torchless_help():
    from attacker.attacks import validate_contract
    assert validate_contract(be) is be
    assert be.EXPERIMENT_ID == "body_slice_extraction"
    assert be.MODES == ("training", "inference")
    assert be.REQUIRES_LABELS is False
    ap = be.build_parser()
    with pytest.raises(SystemExit) as exc:
        ap.parse_args(["--help"])
    assert exc.value.code == 0


def test_self_test_passes():
    assert be.self_test() == 0


def test_dispatch_toy_end_to_end(registered, tmp_path):
    from attacker.__main__ import main
    out = tmp_path / "disp.json"
    rc = main(["--mode", "inference", "--attack", "body-extraction",
               "--toy", "--quick", "--output", str(out)])
    assert rc == 0
    art = json.loads(out.read_text())
    assert art["schema"] == be.SCHEMA
    assert art["config"]["attack"] == be.EXPERIMENT_ID
    assert art["config"]["mode"] == "inference"
    assert art["results"] and art["summary"]
    journal = (tmp_path / "disp.json.jsonl").read_text().strip().splitlines()
    assert len(journal) == len(art["results"])
    assert all(json.loads(l)["experiment"] == be.EXPERIMENT_ID
               for l in journal)


def test_toy_distill_planted_vs_control(tmp_path):
    art, journal = _run_toy(tmp_path, "--arm", "distill")
    recs = [r for r in art["results"] if r["arm"] == "distill"]
    assert len(recs) == len(journal) == 2
    for r in recs:
        assert r["tier"] == "sim" and r["student"] == "closed-form"
        assert r["r2"] > 0.99
        assert r["logit_agreement"] == 1.0
        assert r["element_match"] > 0.99
        assert r["control_logit_agreement"] <= 0.25
        assert r["control_element_match"] < 0.2
        assert r["gate"]["verdict"] == "FAIL"
        assert r["extraction_channel"] == "open"
        assert r["gate"]["n"] == r["n_eval_elements"]
    summ = [s for s in art["summary"] if s["arm"] == "distill"]
    assert summ[0]["r2_mean"] > 0.99
    assert summ[0]["gate_verdict"] == "FAIL"
    assert summ[0]["control_logit_agreement_mean"] <= 0.25


def test_toy_weightsim_nonidentifiability(tmp_path):
    """The honesty arm: the student matches the victim FUNCTIONALLY while
    its per-layer weights drift — functional equivalence is the honest
    extraction metric, weight recovery is not claimed."""
    art, _ = _run_toy(tmp_path, "--arm", "weightsim")
    recs = [r for r in art["results"] if r["arm"] == "weightsim"]
    assert recs
    for r in recs:
        assert r["functional_r2"] > 0.99
        assert r["logit_agreement"] == 1.0
        assert r["per_layer_output_drift"][-1] < 0.01
        assert r["per_layer_weight_drift"][0] > 0.2
        assert r["per_layer_weight_drift"][1] > 0.2
        assert r["control_final_output_drift"] > 0.5
        assert "cryptanalytic" in r["note"]
        assert r["gate"]["verdict"] == "FAIL"
        assert r["extraction_channel"] == "open"


def test_toy_rank_limited_student(tmp_path):
    art, _ = _run_toy(tmp_path, "--arm", "distill", "--student-dim", "32",
                    "--steps", "15")
    (rec,) = [r for r in art["results"] if r["seed"] == 0]
    assert rec["student"] == "rank-32-als"
    assert rec["student_dim"] == 32
    assert rec["r2"] > rec["control_r2"]


def test_toy_deterministic(tmp_path):
    art1, _ = _run_toy(tmp_path, "--arm", "distill", name="a.json")
    art2, _ = _run_toy(tmp_path, "--arm", "distill", name="b.json")
    assert [r["r2"] for r in art1["results"]] \
        == [r["r2"] for r in art2["results"]]


def test_evidence_chain(tmp_path):
    out = tmp_path / "ev.json"
    args = be.build_parser().parse_args(
        ["--toy", "--quick", "--arm", "weightsim", "--evidence",
         "--output", str(out)])
    args.mode = "training"
    assert be.run(args) == 0
    manifest = str(out) + ".evidence.json"
    assert os.path.isfile(manifest)
    ok, problems = verify_evidence(manifest)
    assert ok and problems == []
    m = json.loads(open(manifest).read())
    assert m["node_role"] == f"attack/{be.EXPERIMENT_ID}"


def test_evidence_requires_output():
    args = be.build_parser().parse_args(["--toy", "--quick", "--evidence"])
    args.mode = "training"
    with pytest.raises(SystemExit, match="--output"):
        be.run(args)


def test_no_tier_is_descriptive():
    args = be.build_parser().parse_args([])
    args.mode = "training"
    with pytest.raises(SystemExit, match="--model"):
        be.run(args)


def test_real_tier_requires_torch():
    if be.torch is not None:
        pytest.skip("torch present: require_torch passes here")
    args = be.build_parser().parse_args(["--model", "/nonexistent"])
    args.mode = "training"
    with pytest.raises(SystemExit, match="requires torch"):
        be.run(args)


VOCAB = 99


@pytest.fixture
def tiny_model(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=3, n_embd=32, n_head=4, vocab_size=VOCAB,
                     n_positions=64, bos_token_id=0, eos_token_id=98,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model_dir = tmp_path / "tinygpt2"
    GPT2LMHeadModel(cfg).save_pretrained(model_dir)
    return str(model_dir)


def _real_args(model_dir, out, *extra):
    args = be.build_parser().parse_args(
        ["--model", model_dir, "--split", "1", "--queries", "192",
         "--steps", "100", "--seq-len", "8", "--seeds", "0",
         "--output", str(out), *extra])
    args.mode = "training"
    return args


@pytest.mark.slow
def test_torch_distill_beats_untrained_control(tiny_model, tmp_path):
    out = tmp_path / "be_torch.json"
    args = _real_args(tiny_model, out, "--arm", "distill", "--evidence")
    assert be.run(args) == 0
    art = json.loads(out.read_text())
    (rec,) = art["results"]
    assert rec["tier"] == "local-sim" and rec["query_source"] == "random+head"
    assert rec["body_layers"] == [1, 3]
    assert rec["r2"] > rec["control_r2"] + 0.3
    assert rec["logit_agreement"] > rec["control_logit_agreement"] + 0.2
    assert rec["element_match"] > rec["control_element_match"]
    assert rec["gate"]["verdict"] == "FAIL"
    assert rec["extraction_channel"] == "open"
    assert rec["gate"]["n"] == rec["n_eval_elements"]
    summ = art["summary"][0]
    assert summ["arm"] == "distill" and summ["gate_verdict"] == "FAIL"
    journal = [json.loads(l) for l in open(str(out) + ".jsonl")]
    assert len(journal) == 1 and journal[0]["arm"] == "distill"
    ok, problems = verify_evidence(str(out) + ".evidence.json")
    assert ok and problems == []


@pytest.mark.slow
def test_torch_weightsim_per_layer_drift(tiny_model, tmp_path):
    out = tmp_path / "be_ws.json"
    args = _real_args(tiny_model, out, "--arm", "weightsim")
    assert be.run(args) == 0
    (rec,) = json.loads(out.read_text())["results"]
    assert len(rec["per_layer_output_drift"]) == 2
    assert len(rec["per_layer_weight_drift"]) == 2
    assert rec["per_layer_output_drift"][-1] \
        < rec["control_per_layer_output_drift"][-1]
    assert rec["functional_equivalence"] is True
    assert rec["per_layer_weight_drift"][0] > 0.3
    assert "NOT claimed" in rec["note"]
    assert rec["gate"]["verdict"] == "FAIL"


@pytest.mark.slow
def test_torch_capture_dir_query_source(tiny_model, tmp_path):
    """Offline probe rows: splitbench wire captures supply the QUERY
    distribution (responses still come from the local victim sim)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2LMHeadModel
    from attacker import splitbench
    torch.manual_seed(0)
    model = GPT2LMHeadModel.from_pretrained(tiny_model)
    plan = splitbench.SplitPlan.parse("1", n_layers=3)
    split = splitbench.build_split(model, plan)
    cap_dir = tmp_path / "caps"
    writer = splitbench.CaptureWriter(str(cap_dir), "training", "t-be")
    g = torch.Generator().manual_seed(11)
    batches = [torch.randint(0, VOCAB, (2, 8), generator=g)
               for _ in range(2)]
    bench = splitbench.BenchConfig(steps=2, seq_len=8, batch_size=2,
                                   lr=1e-4, capture_every=1)
    splitbench.finetune(split, bench, batches, writer)
    writer.close()
    out = tmp_path / "be_cap.json"
    args = _real_args(tiny_model, out, "--arm", "distill", "--steps", "150",
                      "--capture-dir", str(cap_dir))
    assert be.run(args) == 0
    (rec,) = json.loads(out.read_text())["results"]
    assert rec["query_source"] == "capture-dir"
    assert rec["queries"] == 2 * 2 * 8
    assert rec["r2"] > rec["control_r2"] + 0.3
    assert rec["logit_agreement"] > rec["control_logit_agreement"] + 0.2
