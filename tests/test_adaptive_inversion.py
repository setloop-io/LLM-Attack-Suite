"""seam-inversion adaptive (defense-aware) arms: --adaptive noise-whitening
models the PUBLIC gaussian seam defense inside the inversion (Tramèr
doctrine).

Torch-free layer: the adaptive toy tier (plain noise-naive ridge vs the
noise-marginalized Wiener map at matched sigma, full gate fields),
parser/flag plumbing, journal/artifact/evidence, and the descriptive
guards. Torch layer (slow): real splitbench gaussian-defense captures from
a tiny random GPT-2 — the linearization arm's noise-augmented ridge
strictly beats the plain fit where the defense bites (sigma = 2x the
measured activation std; at sigma = std/2 the plain arm is NOT degraded,
measured 1.000, so there is nothing to gain — the task-suggested sigma is
documented to saturate), and the vmi-head noise-marginalized objective
runs at parity with the plain objective (same population minimizer for a
single capture — the module docstring's candid note).
"""

import json
import math

import pytest

from attacker.attacks import seam_inversion as si


def parse(*argv):
    args = si.build_parser().parse_args(list(argv))
    args.mode = "training"
    return args


def test_parser_adaptive_flags_torchless():
    args = si.build_parser().parse_args([])
    assert args.adaptive == "none"
    assert args.noise_samples == 8
    assert args.defense_sigma == 0.0
    a = si.build_parser().parse_args(
        ["--adaptive", "noise-whitening", "--noise-samples", "16",
         "--defense-sigma", "0.5"])
    assert a.adaptive == "noise-whitening"
    assert a.noise_samples == 16 and a.defense_sigma == 0.5
    with pytest.raises(SystemExit):
        si.build_parser().parse_args(["--help"])
    with pytest.raises(SystemExit):
        si.build_parser().parse_args(["--adaptive", "x-ray"])
    with pytest.raises(SystemExit, match="--noise-samples"):
        si.run(parse("--toy", "--adaptive", "noise-whitening",
                     "--noise-samples", "0"))


def test_adaptive_toy_ordering_at_matched_sigma():
    args = parse("--toy", "--seeds", "0", "1", "2")
    for sigma, min_margin in ((0.2, 0.0), (0.5, 0.02), (1.0, 0.10)):
        plains, adapts = [], []
        for seed in (0, 1, 2):
            plains.append(si.toy_cell_adaptive(args, sigma, seed, "plain")
                          ["seq_accuracy"])
            adapts.append(si.toy_cell_adaptive(args, sigma, seed,
                                               "adaptive")
                          ["seq_accuracy"])
        plain_m = sum(plains) / 3
        adapt_m = sum(adapts) / 3
        assert adapt_m >= plain_m + min_margin, (sigma, plain_m, adapt_m)
        if sigma >= 0.5:
            assert plain_m < 0.95


def test_adaptive_toy_cell_fields():
    args = parse("--toy", "--seeds", "0")
    plain = si.toy_cell_adaptive(args, 0.5, 0, "plain")
    adapt = si.toy_cell_adaptive(args, 0.5, 0, "adaptive")
    assert plain["variant"] == "plain" and adapt["variant"] == "adaptive"
    assert plain["ridge_lambda_effective"] == si.TOY_RIDGE_LAMBDA
    assert adapt["ridge_lambda_effective"] == pytest.approx(
        si.TOY_RIDGE_LAMBDA + 0.25)
    for cell in (plain, adapt):
        for key in ("seq_accuracy", "control_accuracy",
                    "token_set_precision", "token_set_recall", "gate",
                    "sigma", "seed"):
            assert key in cell
        assert cell["gate"]["verdict"] in ("PASS", "FAIL")
    with pytest.raises(ValueError, match="variant"):
        si.toy_cell_adaptive(args, 0.5, 0, "sideways")


def test_adaptive_toy_run_journals_artifact_and_evidence(tmp_path):
    from attacker.evidence import verify_evidence
    out = str(tmp_path / "sia.json")
    assert si.run(parse("--toy", "--adaptive", "noise-whitening", "--quick",
                        "--evidence", "--output", out)) == 0
    art = json.load(open(out))
    assert art["schema"] == si.SCHEMA
    assert art["config"]["adaptive"] == "noise-whitening"
    cells = art["results"]
    assert len(cells) == 6
    assert sorted(c["variant"] for c in cells) == \
        ["adaptive"] * 3 + ["plain"] * 3
    for summary in art["summary"]:
        assert "adaptive_margin_pp" in summary
    margins = {s["sigma"]: s["adaptive_margin_pp"] for s in art["summary"]}
    assert margins[1.0] > 10.0
    journal = [json.loads(l) for l in open(out + ".jsonl")]
    assert len(journal) == len(cells)
    ok, problems = verify_evidence(out + ".evidence.json")
    assert ok and problems == []


def test_plain_toy_path_unchanged_by_default(tmp_path):
    out = str(tmp_path / "si.json")
    assert si.run(parse("--toy", "--quick", "--output", out)) == 0
    art = json.load(open(out))
    assert len(art["results"]) == 3
    assert all("variant" not in c for c in art["results"])
    assert art["results"][0]["submode"] == "known-map-linearization"


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
    g = torch.Generator().manual_seed(11)
    batches = [torch.randint(0, VOCAB, (2, 8), generator=g)
               for _ in range(2)]
    with torch.no_grad():
        act_std = split.head_forward(batches[0]).std().item()

    def capture(sigma, sub):
        writer = splitbench.CaptureWriter(str(tmp_path / sub), "training",
                                          "t-ad")
        bench = splitbench.BenchConfig(steps=2, seq_len=8, batch_size=2,
                                       lr=1e-4, capture_every=1,
                                       defense="gaussian", sigma=sigma,
                                       save_canonical=True)
        splitbench.finetune(splitbench.build_split(model, plan), bench,
                            batches, writer)
        writer.close()
        return str(tmp_path / sub)

    cap_half = capture(act_std / 2, "caps_half")
    cap_double = capture(2 * act_std, "caps_double")
    return torch, model, cap_half, cap_double, str(model_dir), act_std


def _run_arm(cap_dir, model_dir, sigma, adaptive, *extra):
    argv = ["--capture-dir", cap_dir, "--canonical-pt",
            cap_dir + "/canonical.pt", "--model", model_dir,
            "--split", "1", "--calibration-seqs", "64", "--samples", "4",
            "--adaptive", adaptive]
    if adaptive != "none":
        argv += ["--defense-sigma", str(sigma)]
    return parse(*(argv + list(extra)))


@pytest.mark.slow
def test_torch_linearization_adaptive_beats_plain(tiny_run, tmp_path):
    _, _, _, cap_double, model_dir, act_std = tiny_run
    sigma = 2 * act_std
    plain_out = str(tmp_path / "plain.json")
    adapt_out = str(tmp_path / "adapt.json")
    assert si.run(_run_arm(cap_double, model_dir, sigma, "none",
                           "--output", plain_out)) == 0
    assert si.run(_run_arm(cap_double, model_dir, sigma, "noise-whitening",
                           "--output", adapt_out)) == 0
    plain = json.load(open(plain_out))
    adapt = json.load(open(adapt_out))
    assert plain["summary"][0]["submode"] == "vmi-linearization"
    assert adapt["summary"][0]["submode"] == \
        "vmi-linearization+noise-whitening"
    p_acc = plain["summary"][0]["seq_accuracy"]
    a_acc = adapt["summary"][0]["seq_accuracy"]
    assert p_acc < 0.5
    assert a_acc > p_acc + 0.05
    assert a_acc > adapt["summary"][0]["control_accuracy"] + 0.1


@pytest.mark.slow
def test_torch_task_suggested_sigma_saturates_plain(tiny_run, tmp_path):
    _, _, cap_half, _, model_dir, _ = tiny_run
    out = str(tmp_path / "half.json")
    assert si.run(_run_arm(cap_half, model_dir, 0.0, "none",
                           "--output", out)) == 0
    summary = json.load(open(out))["summary"][0]
    assert summary["seq_accuracy"] > 0.9


@pytest.mark.slow
def test_torch_vmi_head_adaptive_parity_with_plain(tiny_run, tmp_path):
    torch, model, _, cap_double, model_dir, act_std = tiny_run
    sigma = 2 * act_std
    head_pt = str(tmp_path / "head.pt")
    si.save_head_pt(head_pt, model, boundary=1)

    def run_vmi(adaptive, out):
        argv = ["--capture-dir", cap_double, "--canonical-pt",
                cap_double + "/canonical.pt", "--head-pt", head_pt,
                "--arm", "vmi", "--steps", "60", "--samples", "4",
                "--calibration-seqs", "64", "--adaptive", adaptive,
                "--output", out]
        if adaptive != "none":
            argv += ["--defense-sigma", str(sigma)]
        assert si.run(parse(*argv)) == 0
        return json.load(open(out))

    plain = run_vmi("none", str(tmp_path / "vp.json"))
    adapt = run_vmi("noise-whitening", str(tmp_path / "va.json"))
    p_acc = plain["summary"][0]["seq_accuracy"]
    a_acc = adapt["summary"][0]["seq_accuracy"]
    assert plain["summary"][0]["submode"] == "vmi-head"
    assert adapt["summary"][0]["submode"] == "vmi-head+noise-whitening"
    for art in (plain, adapt):
        assert math.isfinite(art["results"][0]["final_objective"])
    assert a_acc >= p_acc - 0.06
    assert a_acc > adapt["summary"][0]["control_accuracy"]
