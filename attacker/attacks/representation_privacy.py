"""Fresh-probe audits of released representations, not defender reproductions.

JSON contract: schema='representation_privacy.v1', provenance={source,defender,
release_stage}, calibration/evaluation lists of {document_id, released:[T,D],
label:int, optional membership:0|1}. Label is one prespecified per-document
classification target (such as token identity at a fixed position or property),
not full-sequence recovery. Membership must be supplied for every document or
omitted globally. Calibration labels are attacker-known; evaluation labels
are scoring truth only. Optional phase_records require the separately
validated phase_provenance contract (see validate_phase).
"""
import argparse
import json
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

from .. import artifacts
from .common import add_common_args
from .lira_membership import rank_auc, _mean_var, _log_gauss, _log_rho

EXPERIMENT_ID = "representation_privacy"
MODES = ("training", "inference")
REQUIRES_LABELS = True
DESCRIPTION = "Fresh coordinate/norm/Gram probes and provenance-gated private-phase membership"
ARMS = ("bottleneck", "gram_flattening", "private_phase")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--bundle", help="released-feature JSON with document-disjoint calibration/evaluation")
    ap.add_argument("--shadow-toy", action="store_true",
                    help="build the bundle from the internal shadow-model "
                         "trainer (attacker/shadow_train.py synthetic "
                         "ensemble) instead of --toy/--bundle; requires "
                         "torch (--toy stays numpy-only)")
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=["bottleneck", "gram_flattening"])
    ap.add_argument("--evidence", action="store_true")
    ap.add_argument("--nonlinear-probe", action="store_true", help="also fit a fixed 64-unit, 300-step CPU MLP on public calibration only; requires torch")
    return ap


def require_numpy():
    if np is None:
        raise RuntimeError("representation-privacy requires numpy for numeric audits; install numpy")


def _finite(value, name, ndim=None):
    require_numpy()
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must contain numeric values")
    arr = raw.astype(float)
    if not np.isfinite(arr).all() or (ndim is not None and arr.ndim != ndim):
        raise ValueError(f"{name} must be finite with dimension {ndim}")
    return arr


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_bundle(bundle):
    if bundle.get("schema") != "representation_privacy.v1":
        raise ValueError("unsupported released-feature schema")
    provenance = bundle.get("provenance", {})
    for key in ("source", "defender", "release_stage"):
        if not isinstance(provenance.get(key), str) or not provenance[key].strip():
            raise ValueError(f"missing provenance.{key}")
    seen, shape, membership_presence = set(), None, set()
    for partition in ("calibration", "evaluation"):
        rows = bundle.get(partition)
        if not isinstance(rows, list) or len(rows) < 4:
            raise ValueError(f"{partition} requires >=4 documents")
        for row in rows:
            ident = row.get("document_id")
            if not isinstance(ident, str) or not ident or ident in seen:
                raise ValueError("document IDs must be unique and calibration/evaluation disjoint")
            seen.add(ident)
            arr = _finite(row.get("released"), "released", 2)
            if 0 in arr.shape or (shape is not None and arr.shape != shape):
                raise ValueError("released feature dimensions must match and be nonempty")
            shape = arr.shape
            if not _integer(row.get("label")):
                raise ValueError("label must be a nonnegative integer")
            membership_presence.add("membership" in row)
            if "membership" in row and (not _integer(row["membership"]) or row["membership"] not in (0, 1)):
                raise ValueError("membership must be 0 or 1")
    if len(membership_presence) != 1:
        raise ValueError("membership must be present on all rows or omitted globally")
    if True in membership_presence:
        for partition in ("calibration", "evaluation"):
            if {r["membership"] for r in bundle[partition]} != {0, 1}:
                raise ValueError("each partition needs both membership classes")
    return shape


def features(rows, kind):
    """Only released values enter feature construction; metadata is excluded."""
    require_numpy()
    x = np.asarray([r["released"] for r in rows], dtype=float)
    if kind == "coordinate":
        return x.reshape(len(x), -1)
    with np.errstate(over="ignore", invalid="ignore"):
        gram = x @ x.transpose(0, 2, 1)
    if not np.isfinite(gram).all():
        raise ValueError("released features overflow Gram calculation")
    if kind == "norm":
        return np.sqrt(np.maximum(0, np.diagonal(gram, axis1=1, axis2=2)))
    if kind == "gram":
        a, b = np.triu_indices(x.shape[1])
        return gram[:, a, b]
    raise ValueError("unknown feature family")


def fit_predict(train_x, train_y, eval_x, ridge=1.0):
    """Fixed ridge classifier: preprocessing/classes fit on calibration only."""
    require_numpy()
    classes = np.unique(train_y)
    mean, scale = train_x.mean(0), train_x.std(0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    a = np.column_stack((np.ones(len(train_x)), (train_x - mean) / scale))
    b = np.column_stack((np.ones(len(eval_x)), (eval_x - mean) / scale))
    targets = np.asarray(train_y)[:, None] == classes[None, :]
    weights = a.T @ np.linalg.solve(a @ a.T + ridge * np.eye(len(a)), targets)
    scores = b @ weights
    return classes[scores.argmax(1)], scores, classes


def mlp_fit_predict(train_x, train_y, eval_x, seed=0):
    """Fresh nonlinear decoder, fixed hyperparameters, no evaluation tuning.

    One ReLU hidden layer (64 units), full-batch Adam lr=.01, 300 updates,
    CPU float32. Standardization and class vocabulary use calibration only.
    Torch is optional and loaded only when this explicit probe is requested.
    """
    require_numpy()
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("--nonlinear-probe requires torch") from error
    classes, encoded = np.unique(train_y, return_inverse=True)
    mean, scale = train_x.mean(0), train_x.std(0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    a = torch.tensor((train_x - mean) / scale, dtype=torch.float32)
    b = torch.tensor((eval_x - mean) / scale, dtype=torch.float32)
    y = torch.tensor(encoded, dtype=torch.long)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = torch.nn.Sequential(torch.nn.Linear(a.shape[1], 64),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(64, len(classes)))
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        for _ in range(300):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(a), y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            scores = model(b).numpy()
    if not np.isfinite(scores).all():
        raise ValueError("non-finite nonlinear probe scores")
    return classes[scores.argmax(1)], scores, classes


def probe_audit(bundle, seed=0, nonlinear=False):
    shape = validate_bundle(bundle)
    train, evaluation = bundle["calibration"], bundle["evaluation"]
    predictions = {}
    for kind in ("coordinate", "norm", "gram") + (("coordinate_mlp",) if nonlinear else ()):
        feature_kind = "coordinate" if kind == "coordinate_mlp" else kind
        a, b = features(train, feature_kind), features(evaluation, feature_kind)
        for target in (("label", "membership") if "membership" in train[0] else ("label",)):
            y = np.asarray([r[target] for r in train])
            for control in ("fitted", "shuffled_calibration_labels"):
                labels = y if control == "fitted" else np.random.default_rng(seed).permutation(y)
                pred, scores, classes = (mlp_fit_predict(a, labels, b, seed) if kind == "coordinate_mlp" else fit_predict(a, labels, b))
                predictions[(kind, target, control)] = (pred, scores, classes)
    results = []
    for (kind, target, control), (pred, scores, classes) in predictions.items():
        truth = np.asarray([r[target] for r in evaluation])
        majority = max(set(r[target] for r in train), key=lambda c: sum(r[target] == c for r in train))
        rec = {"family": kind, "target": target, "control": control,
               "accuracy": float(np.mean(pred == truth)),
               "calibration_majority_accuracy": float(np.mean(truth == majority)),
               "unseen_evaluation_classes": int(sum(v not in classes for v in truth)),
               "predictions": pred.tolist()}
        if target == "membership":
            score = scores[:, list(classes).index(1)] - scores[:, list(classes).index(0)]
            rec["auc"] = rank_auc(score[truth == 1].tolist(), score[truth == 0].tolist())
            rec["scores"] = score.tolist()
        results.append(rec)
    return {"feature_shape": list(shape), "calibration_documents": len(train),
            "evaluation_documents": len(evaluation), "results": results,
            "interpretation": "prespecified per-document label classification, not full-sequence recovery; fresh fixed ridge probes plus optional calibration-only MLP; no MINE certificate or trained-defender replication",
            "nonlinear_probe": {"enabled": nonlinear, "hidden_units": 64, "steps": 300, "optimizer": "Adam", "learning_rate": .01, "seed": seed},
            "uncertainty": "descriptive held-out metrics; no population confidence claim"}


def validate_phase(bundle):
    """Require phase labels and attacker-owned references, never global truth fit.

    phase_provenance: public_dataset, private_dataset, target_model,
    reference_models (>=2 IDs), references_attacker_owned=true,
    statistic='negative_log_likelihood'. Each phase_record: document_id,
    phase={public_only,private_training_member,never_seen}, public_exposed:bool,
    statistic, ref_in/ref_out (>=2 finite NLLs). Phase truth is scoring metadata.
    Public-exposure strata must each contain members and matched nonmembers.
    """
    provenance = bundle.get("phase_provenance", {})
    for key in ("public_dataset", "private_dataset", "target_model"):
        if not isinstance(provenance.get(key), str) or not provenance[key].strip():
            raise ValueError(f"missing phase provenance {key}")
    refs = provenance.get("reference_models")
    if not isinstance(refs, list) or len(refs) < 2 or any(not isinstance(r, str) or not r for r in refs) or len(set(refs)) != len(refs):
        raise ValueError("phase reference_models requires >=2 distinct IDs")
    if provenance.get("references_attacker_owned") is not True or provenance.get("statistic") != "negative_log_likelihood":
        raise ValueError("phase requires attacker-owned references and negative_log_likelihood statistic")
    rows, seen = bundle.get("phase_records", []), set()
    if not rows:
        raise ValueError("phase_records required")
    forbidden = {r["document_id"] for r in bundle.get("calibration", [])}
    for row in rows:
        ident = row.get("document_id")
        if not isinstance(ident, str) or not ident or ident in seen or ident in forbidden:
            raise ValueError("phase evaluation IDs must be unique and disjoint from calibration")
        seen.add(ident)
        phase, exposure = row.get("phase"), row.get("public_exposed")
        if phase not in ("public_only", "private_training_member", "never_seen") or not isinstance(exposure, bool):
            raise ValueError("explicit phase and public_exposed required")
        if (phase == "public_only" and not exposure) or (phase == "never_seen" and exposure):
            raise ValueError("phase contradicts public exposure")
        _finite(row.get("statistic"), "statistic", 0)
        for key in ("ref_in", "ref_out"):
            ref = _finite(row.get(key), key, 1)
            if len(ref) < 2:
                raise ValueError("phase requires >=2 per-example IN and OUT reference statistics")
    for exposure in {r["public_exposed"] for r in rows}:
        selected = [r for r in rows if r["public_exposed"] == exposure]
        if {r["phase"] == "private_training_member" for r in selected} != {True, False}:
            raise ValueError("phase comparisons require public-exposure-matched members/nonmembers")
    return rows


def phase_audit(bundle, mode):
    if mode != "training":
        raise ValueError("private_phase arm is training-only")
    rows = validate_phase(bundle)
    llr, relative = [], []
    for row in rows:
        mi, vi = _mean_var(row["ref_in"])
        mo, vo = _mean_var(row["ref_out"])
        llr.append(_log_gauss(row["statistic"], mi, vi) - _log_gauss(row["statistic"], mo, vo))
        relative.append(_log_rho(row["statistic"], row["ref_out"]))
    results = []
    for exposure in sorted({r["public_exposed"] for r in rows}):
        for method, scores in (("lira_per_example", llr), ("rmia_relative_likelihood", relative)):
            pos = [s for r, s in zip(rows, scores) if r["public_exposed"] == exposure and r["phase"] == "private_training_member"]
            neg = [s for r, s in zip(rows, scores) if r["public_exposed"] == exposure and r["phase"] != "private_training_member"]
            results.append({"method": method, "public_exposed": exposure,
                            "members": len(pos), "nonmembers": len(neg), "auc": rank_auc(pos, neg)})
    return {"results": results, "interpretation": "per-example LiRA and RMIA relative-likelihood ranking; no RMIA population-quantile claim or reproduced published effect; defender v9.3 +1.49pp is an upper-confidence excess over label-free majority, not a membership result"}


def make_fixture(seed=0, signal=True, flattened=False):
    require_numpy()
    rng = np.random.default_rng(seed)
    bundle = {"schema": "representation_privacy.v1", "provenance": {
        "source": "synthetic fixture", "defender": "synthetic 64d release, no trained defender", "release_stage": "synthetic"}}
    for part in ("calibration", "evaluation"):
        rows = []
        for i in range(80):
            label = i % 2
            x = rng.normal(0, .03, size=(1, 64))
            if signal:
                x[0, 0] += 2 * label - 1
            if flattened:
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            rows.append({"document_id": f"{part}-{i}", "released": x.tolist(), "label": label, "membership": label})
        bundle[part] = rows
    bundle["phase_provenance"] = {"public_dataset": "synthetic-public", "private_dataset": "synthetic-private", "target_model": "synthetic", "reference_models": ["shadow-1", "shadow-2"], "references_attacker_owned": True, "statistic": "negative_log_likelihood"}
    bundle["phase_records"] = [{"document_id": f"phase-{i}", "phase": "private_training_member" if i % 2 else ("public_only" if i < 8 else "never_seen"), "public_exposed": i < 8, "statistic": (1.0 if i % 2 else 3.0) if signal else 2.0, "ref_in": [.9, 1.1], "ref_out": [2.9, 3.1]} for i in range(16)]
    return bundle


def run(args):
    from ..evidence import EvidenceLogger, file_sha256
    if args.evidence and not args.output:
        raise ValueError("--evidence requires --output")
    shadow_toy = getattr(args, "shadow_toy", False)
    if sum([bool(args.toy), bool(args.bundle), bool(shadow_toy)]) != 1:
        raise ValueError("supply exactly one of --toy, --bundle or --shadow-toy")
    require_numpy()
    if args.output:
        output = Path(args.output)
        collisions = [Path(str(output) + suffix) for suffix in ("", ".jsonl", ".evidence.json", ".evidence.jsonl")]
        if any(path.exists() for path in collisions):
            raise ValueError("output and evidence paths must be fresh; refusing to overwrite")
        output.parent.mkdir(parents=True, exist_ok=True)
    if shadow_toy:
        from .common import require_torch
        require_torch(EXPERIMENT_ID)
        from ..shadow_train import ShadowConfig, build_phase_bundle
        bundle = build_phase_bundle(ShadowConfig(seed=args.seed))
    else:
        bundle = make_fixture(args.seed) if args.toy else json.loads(Path(args.bundle).read_text())
    mode = getattr(args, "mode", "training")
    if "private_phase" in args.arms and mode != "training":
        raise ValueError("private_phase arm is training-only")
    result = {"arms": {}, "support": "synthetic machinery" if (args.toy or shadow_toy) else "released-feature adapter; provenance supplied by operator"}
    for arm in args.arms:
        result["arms"][arm] = phase_audit(bundle, mode) if arm == "private_phase" else probe_audit(bundle, args.seed, getattr(args, "nonlinear_probe", False))
    artifact = artifacts.make_artifact("dtraining.attacker.representation_privacy.v1", {"attack": EXPERIMENT_ID, "mode": mode, "toy": args.toy, "shadow_toy": shadow_toy, "arms": args.arms, "nonlinear_probe": getattr(args, "nonlinear_probe", False)}, "attacker-known labeled calibration releases; disjoint evaluation truth is used only after predictions; phase references attacker-owned")
    artifact["results"] = result
    artifact["source_provenance"] = bundle.get("provenance")
    artifact["phase_provenance"] = bundle.get("phase_provenance")
    if args.bundle:
        artifact["input_sha256"] = file_sha256(args.bundle)
    if args.evidence:
        with EvidenceLogger(args.output, node_role=f"attacker/{EXPERIMENT_ID}") as logger:
            logger.event("input_contract", {"sha256": artifact.get("input_sha256"), "toy": args.toy, "label_roles": "calibration=attacker-known; evaluation=scoring-only"})
            artifacts.write_artifact(args.output, artifact)
    else:
        artifacts.write_artifact(args.output, artifact)
    return 0


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import os
    import tempfile

    print("representation-privacy audits (numpy-only fixtures):")
    check("parser builds torch-less", build_parser() is not None)
    positive = probe_audit(make_fixture())
    null = probe_audit(make_fixture(signal=False))
    fitted = next(r for r in positive["results"]
                  if r["family"] == "coordinate" and r["target"] == "membership"
                  and r["control"] == "fitted")
    null_fitted = next(r for r in null["results"]
                       if r["family"] == "coordinate"
                       and r["target"] == "membership"
                       and r["control"] == "fitted")
    check("fixture membership probe separates; null world is honest",
          fitted["auc"] > .95 and .3 < null_fitted["auc"] < .7)
    check("phase fixture: all strata/methods at AUC 1; null at 0.5",
          all(r["auc"] == 1
              for r in phase_audit(make_fixture(), "training")["results"])
          and all(r["auc"] == .5
                  for r in phase_audit(make_fixture(signal=False),
                                       "training")["results"]))
    from . import common as _common
    saved_torch = _common.torch
    _common.torch = None
    try:
        run(build_parser().parse_args(["--shadow-toy"]))
        check("--shadow-toy raises the require_torch SystemExit torch-less",
              False)
    except SystemExit as e:
        check("--shadow-toy raises the require_torch SystemExit torch-less",
              "requires torch" in str(e))
    finally:
        _common.torch = saved_torch
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "audit.json")
        ns = build_parser().parse_args(["--toy", "--output", out,
                                        "--arms", "bottleneck",
                                        "private_phase"])
        ns.mode = "training"
        check("toy run end to end (numpy-only)", run(ns) == 0)
        art = json.load(open(out))
        check("toy artifact carries arms and phase provenance",
              set(art["results"]["arms"]) == {"bottleneck", "private_phase"}
              and art["phase_provenance"]["references_attacker_owned"]
              is True)

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
