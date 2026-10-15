"""label-scoring (norm-based property inference): torch-free toy tier
(two-mixture norms with a planted gap, null-world honesty arm, property
labeling) plus the slow torch layer — real-tier per-sequence norm
computation over synthetic bwd .pt captures in the wire-training schema."""

import json
import math

import pytest

from attacker.attacks import label_scoring as ls


def parse(*argv):
    args = ls.build_parser().parse_args(list(argv))
    args.mode = "training"
    return args


def test_parser_builds_torchless():
    args = ls.build_parser().parse_args([])
    assert args.property is None and args.toy_gap == 4.0
    assert args.evidence is False
    with pytest.raises(SystemExit):
        ls.build_parser().parse_args(["--help"])


def test_module_contract_and_register_dispatch():
    from attacker import attacks
    attacks.validate_contract(ls, "label-scoring")
    assert ls.EXPERIMENT_ID == "norm_based_scoring"
    assert ls.MODES == ("training",)
    assert ls.REQUIRES_LABELS is False
    if "label-scoring" not in attacks.ATTACKS:
        attacks.register_attack("label-scoring", ls)
    assert attacks.load("label-scoring") is ls
    assert "label-scoring" in attacks.for_mode("training")
    assert "label-scoring" not in attacks.for_mode("inference")


def test_parse_property_and_label_rows():
    assert ls.parse_property("contains-token:220") == ("contains-token", 220)
    for bad in ("contains-token", "has-token:1", "contains-token:x",
                "contains-token:-1", "", 42):
        with pytest.raises(ValueError):
            ls.parse_property(bad)
    assert ls.label_rows([[1, 2, 3], [4, 5], [2, 7]],
                         ("contains-token", 2)) == [True, False, True]
    assert ls.label_rows([[7], [7]], ("contains-token", 0)) \
        == [False, False]


def test_sequence_norm_math():
    assert ls.sequence_norm([[3.0, 4.0], [0.0, 0.0], [1.0, 2.0]]) \
        == pytest.approx(5.0 + math.sqrt(5.0))
    assert ls.sequence_norm([]) == 0.0


def test_toy_separable_and_null_worlds():
    present, absent = ls.make_toy_norms("separable", 5000, 0, 4.0)
    m = ls.norm_metrics(present, absent)
    assert m["auc"] > 0.99
    assert m["gate"]["verdict"] == "FAIL"
    assert m["tpr_at_fpr"]["0.01"]["tpr"] > 0.5
    present, absent = ls.make_toy_norms("null", 100000, 1, 0.0)
    m = ls.norm_metrics(present, absent)
    assert abs(m["auc"] - 0.5) < 0.01
    assert m["gate"]["verdict"] == "INSUFFICIENT_EVIDENCE"
    with pytest.raises(ValueError, match=">=1 present"):
        ls.norm_metrics([], [1.0])


def test_toy_run_journals_artifact_and_evidence(tmp_path):
    from attacker.evidence import verify_evidence
    out = str(tmp_path / "ls.json")
    assert ls.run(parse("--toy", "--quick", "--evidence",
                        "--output", out)) == 0
    art = json.load(open(out))
    assert art["schema"] == ls.SCHEMA
    assert sorted(c["world"] for c in art["results"]) == ["null", "separable"]
    for cell in art["results"]:
        for key in ("auc", "tpr_at_fpr", "gate", "n_present", "n_absent"):
            assert key in cell
    journal = [json.loads(l) for l in open(out + ".jsonl")]
    assert len(journal) == len(art["results"])
    ok, problems = verify_evidence(out + ".evidence.json")
    assert ok and problems == []


def test_real_path_errors_are_descriptive_torchless():
    with pytest.raises(SystemExit, match="--capture-dir"):
        ls.run(parse())
    with pytest.raises(SystemExit, match="contains-token"):
        ls.run(parse("--toy", "--property", "bogus"))
    if ls.torch is None:
        with pytest.raises(SystemExit, match="requires torch"):
            ls.run(parse("--capture-dir", "/nonexistent"))
        with pytest.raises(SystemExit, match="requires torch"):
            ls.run(parse("--capture-dir", "/nonexistent",
                         "--property", "contains-token:7"))


def test_module_self_test_passes():
    assert ls.self_test() == 0


@pytest.fixture
def planted_captures(tmp_path):
    torch = pytest.importorskip("torch")
    from attacker import splitbench
    cap_dir = tmp_path / "caps"
    writer = splitbench.CaptureWriter(str(cap_dir), "training", "t-ls")
    g = torch.Generator().manual_seed(5)
    n_steps, batch, seq, hdim = 6, 4, 5, 8
    ids = torch.randint(0, 99, (n_steps * batch, seq), generator=g)
    ids[1::2] = torch.randint(0, 99, ids[1::2].shape, generator=g)
    ids[1::2][ids[1::2] == 7] = 8
    ids[0::2, 0] = 7
    labels = [7 in row.tolist() for row in ids]
    assert any(labels) and not all(labels)
    for step in range(n_steps):
        grads = 0.01 * torch.randn(batch, seq, hdim, generator=g)
        for r in range(batch):
            if labels[step * batch + r]:
                grads[r, 0, 0] += 1.0
        writer.write_training(grads, mb_id=0, phase="bwd", step=step,
                              epoch=0)
    writer.close()
    torch.save(ids, cap_dir / "canonical.pt")
    return torch, str(cap_dir)


@pytest.mark.slow
def test_torch_sequence_norms_match_pure_python(planted_captures):
    torch, cap_dir = planted_captures
    from attacker import captures
    meta, pt = captures.scan_captures(cap_dir, "training")[0]
    t = captures.load_tensor(pt)
    norms = ls._sequence_norms_torch(t)
    assert len(norms) == t.shape[0]
    for r in range(t.shape[0]):
        ref = ls.sequence_norm(t[r].tolist())
        assert norms[r] == pytest.approx(ref, rel=1e-3)


@pytest.mark.slow
def test_torch_real_tier_property_inference(planted_captures, tmp_path):
    _, cap_dir = planted_captures
    out = str(tmp_path / "prop.json")
    args = parse("--capture-dir", cap_dir, "--canonical-pt",
                 cap_dir + "/canonical.pt", "--property",
                 "contains-token:7", "--output", out)
    assert ls.run(args) == 0
    art = json.load(open(out))
    info = [c for c in art["results"] if c["kind"] == "norm_stats"]
    score = [c for c in art["results"] if c["kind"] == "property_score"]
    assert len(info) == 6
    assert all(c["n_seq"] == 4 for c in info)
    assert len(score) == 1
    cell = score[0]
    assert cell["n_present"] == 12 and cell["n_absent"] == 12
    assert cell["auc"] == pytest.approx(1.0)
    assert cell["gate"]["verdict"] == "FAIL"
    journal = [json.loads(l) for l in open(out + ".jsonl")]
    assert len(journal) == 7


@pytest.mark.slow
def test_torch_real_tier_without_property_is_info_only(planted_captures,
                                                       tmp_path):
    _, cap_dir = planted_captures
    out = str(tmp_path / "info.json")
    assert ls.run(parse("--capture-dir", cap_dir, "--output", out)) == 0
    art = json.load(open(out))
    assert all(c["kind"] == "norm_stats" for c in art["results"])
    assert not any("gate" in c for c in art["results"])
    stats = [s for s in art["summary"] if s["kind"] == "norm_stats"]
    assert stats and stats[0]["n_seq"] == 6 * 4
    assert stats[0]["norm_max"] > 10 * stats[0]["norm_mean"] or \
        stats[0]["norm_max"] > 0.5
