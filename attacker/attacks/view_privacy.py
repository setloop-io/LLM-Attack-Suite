"""Observation-based view/privacy batteries; synthetic or strict JSON arrays.

JSON schema ``view_privacy.v1``: {schema, case, data}. Cases:
* mixup: releases[N][R][D], truth[N][D], lambda, calibration_mean[D],
  optional assumed_lambdas[], calibration_samples[C][D], oracle_pool_mean[D],
  corrected_outputs[N][R][D], clean_outputs[N][D].
* views: train[N][D], train_labels[N], evaluation[M][D], eval_labels[M],
  views[K][] coordinate indices. Enumerates every nonempty coalition (K <= 8).
* tripwire: timestamps[N], epochs[N], releases[N][D], truth_by_epoch{str:[D]},
  window, max_count. Each epoch is a separate coordinate/key domain.

Truth and evaluation labels are scoring-only. Calibration is explicitly
attacker-known. Coordinate views are NOT threshold secret sharing. Tripwire
models count monitors, not an arbitrary adaptive controller. Mixup averaging
requires repeated releases of the SAME target; unknown decoy mean makes its
intercept nonidentifiable. This is not an attack on a cryptographic primitive.
"""
import argparse
import itertools
import json
import math
import random
from pathlib import Path
from statistics import mean

from .. import artifacts
from ..evidence import EvidenceLogger, file_sha256
from .common import add_common_args

EXPERIMENT_ID = "view_privacy"
MODES = ("training", "inference")
REQUIRES_LABELS = "Public auxiliary labels for property probes; calibration mean for demixing; evaluation truth for scoring only"
DESCRIPTION = "Mixup de-mixing, coordinate-view collusion, and epoch-local tripwire accumulation"
SCHEMA_VERSION = "view_privacy.v1"


def _number(x):
    return isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)


def _matrix(x, name):
    if not isinstance(x, list) or not x or not isinstance(x[0], list) or not x[0]:
        raise ValueError(name + " must be a nonempty matrix")
    d = len(x[0])
    if any(not isinstance(row, list) or len(row) != d or not all(_number(v) for v in row) for row in x):
        raise ValueError(name + " must be rectangular finite numeric arrays")
    return len(x), d


def _keys(data, required, optional=()):
    if not isinstance(data, dict) or not set(required) <= data.keys() or set(data) - set(required) - set(optional):
        raise ValueError("Missing or unknown data fields")


def _mse(a, b):
    return mean((x-y)**2 for ar, br in zip(a, b) for x, y in zip(ar, br))


def _vector_mean(rows):
    return [mean(col) for col in zip(*rows)]


def demix(releases, assumed_lambda, calibration_mean):
    """Estimator deliberately has no truth argument."""
    return [[(v-(1-assumed_lambda)*mu)/assumed_lambda
             for v, mu in zip(_vector_mean(rows), calibration_mean)] for rows in releases]


def evaluate_mixup(data):
    _keys(data, ("releases", "truth", "lambda", "calibration_mean"),
          ("assumed_lambdas", "corrected_outputs", "clean_outputs", "calibration_samples", "oracle_pool_mean"))
    n, d = _matrix(data["truth"], "truth")
    releases = data["releases"]
    if not isinstance(releases, list) or len(releases) != n:
        raise ValueError("releases must contain one repeated-release matrix per target")
    shapes = [_matrix(rows, "releases") for rows in releases]
    if len(set(shapes)) != 1 or shapes[0][1] != d:
        raise ValueError("release shapes do not match truth")
    lam = data["lambda"]
    assumptions = data.get("assumed_lambdas", [lam])
    if not _number(lam) or not 0 < lam <= 1 or not isinstance(assumptions, list) or not assumptions or any(not _number(v) or not 0 < v <= 1 for v in assumptions):
        raise ValueError("lambdas must be finite in (0,1]")
    mu = data["calibration_mean"]
    if not isinstance(mu, list) or len(mu) != d or not all(_number(v) for v in mu):
        raise ValueError("calibration_mean shape/value invalid")
    calibration_source = "supplied_mean_provenance_unverified"
    if "calibration_samples" in data:
        c, cd = _matrix(data["calibration_samples"], "calibration_samples")
        if cd != d or any(abs(a-b) > 1e-10 for a, b in zip(mu, _vector_mean(data["calibration_samples"]))):
            raise ValueError("calibration_samples must match calibration_mean")
        calibration_source = "finite_auxiliary_samples"
    oracle = data.get("oracle_pool_mean")
    if "oracle_pool_mean" in data and (not isinstance(oracle, list) or len(oracle) != d or not all(_number(v) for v in oracle)):
        raise ValueError("oracle_pool_mean shape/value invalid")
    correction = "corrected_outputs" in data
    if correction != ("clean_outputs" in data):
        raise ValueError("corrected_outputs requires paired clean_outputs")
    if correction:
        if _matrix(data["clean_outputs"], "clean_outputs") != (n, d) or not isinstance(data["corrected_outputs"], list) or len(data["corrected_outputs"]) != n or any(_matrix(x, "corrected_outputs") != shapes[0] for x in data["corrected_outputs"]):
            raise ValueError("correction output shapes must match")
    r = shapes[0][0]
    budgets = sorted(set([1, min(4, r), min(16, r), r]))
    out = []
    for count in budgets:
        observed = [rows[:count] for rows in releases]
        for assumed in assumptions:
            centers = [("calibrated_mean", mu), ("naive_zero_mean", [0.0]*d)]
            if oracle is not None:
                centers.append(("oracle_pool_mean", oracle))
            for name, center in centers:
                prediction = demix(observed, assumed, center)
                sem2 = None if count < 2 else mean(
                    sum((row[j]-mean(z[j] for z in rows))**2 for row in rows)
                    / (count*(count-1)*assumed**2)
                    for rows in observed for j in range(d))
                out.append(dict(case="mixup", arm=name, releases=count,
                                assumed_lambda=assumed, mse=_mse(prediction, data["truth"]),
                                conditional_mean_variance=sem2,
                                calibration_source="oracle_entire_pool" if name == "oracle_pool_mean" else calibration_source if name == "calibrated_mean" else "zero_mean_assumption",
                                calibration_sample_count=len(data.get("calibration_samples", [])) if name == "calibrated_mean" else None))
        out.append(dict(case="mixup", arm="no_observation_calibration_mean", releases=count,
                        mse=_mse([mu]*n, data["truth"])))
        if correction:
            out.append(dict(case="mixup", arm="nonlinear_corrected_output", releases=count,
                            mse=_mse([_vector_mean(rows[:count]) for rows in data["corrected_outputs"]], data["clean_outputs"])))
    return out


def centroid_predictions(train, labels, evaluation, coords):
    """Auxiliary-label centroid classifier; evaluation labels never enter."""
    classes = sorted(set(labels))
    centers = {c: _vector_mean([[row[j] for j in coords] for row, label in zip(train, labels) if label == c]) for c in classes}
    return [min(classes, key=lambda c: sum((row[j]-centers[c][k])**2 for k, j in enumerate(coords))) for row in evaluation]


def evaluate_views(data):
    _keys(data, ("train", "train_labels", "evaluation", "eval_labels", "views"))
    n, d = _matrix(data["train"], "train")
    m, ed = _matrix(data["evaluation"], "evaluation")
    if ed != d:
        raise ValueError("train/evaluation dimensions differ")
    for key, size in (("train_labels", n), ("eval_labels", m)):
        if not isinstance(data[key], list) or len(data[key]) != size or any(type(x) is not int for x in data[key]):
            raise ValueError("labels must be integer arrays matching rows")
    if set(data["eval_labels"]) - set(data["train_labels"]):
        raise ValueError("evaluation contains classes missing from calibration")
    views = data["views"]
    if not isinstance(views, list) or not 1 <= len(views) <= 8 or any(not isinstance(v, list) or not v or any(type(j) is not int or not 0 <= j < d for j in v) or len(set(v)) != len(v) for v in views):
        raise ValueError("views need 1..8 nonempty unique valid coordinate lists")
    mu = _vector_mean(data["train"])
    labels = data["train_labels"]
    majority = min(set(labels), key=lambda c: (-labels.count(c), c))
    control = mean(y == majority for y in data["eval_labels"])
    out = [dict(case="views", arm="no_observation", coalition=[], observed_fraction=0,
                reconstruction_mse=_mse([mu]*m, data["evaluation"]), property_accuracy=control)]
    for count in range(1, len(views)+1):
        for coalition in itertools.combinations(range(len(views)), count):
            coords = sorted(set(j for i in coalition for j in views[i]))
            prediction = [[row[j] if j in coords else mu[j] for j in range(d)] for row in data["evaluation"]]
            guesses = centroid_predictions(data["train"], labels, data["evaluation"], coords)
            out.append(dict(case="views", arm="single_view" if count == 1 else "colluding_union",
                            coalition=list(coalition), observed_fraction=len(coords)/d,
                            reconstruction_mse=_mse(prediction, data["evaluation"]),
                            property_accuracy=mean(a == b for a, b in zip(guesses, data["eval_labels"])),
                            property_excess_over_majority=mean(a == b for a, b in zip(guesses, data["eval_labels"]))-control))
    covered = set(j for view in views for j in view)
    full_coalitions = [row for row in out if row["observed_fraction"] == 1]
    min_full = min((len(row["coalition"]) for row in full_coalitions), default=None)
    for row in out:
        row.update(num_views=len(views), view_coordinates=views, dimension=d,
                   minimum_coalition_full_coordinate_coverage=min_full,
                   layout="disjoint_coordinate_fragments" if sum(map(len, views)) == len(covered) else "overlapping_coordinate_views")
    return out


def monitor_acceptance(timestamps, window, max_count, kind):
    """Reject excess requests; rejected arrivals still count toward monitoring."""
    accepted, history = [], []
    for t in timestamps:
        if kind == "sliding":
            history = [s for s in history if s > t-window]
        elif kind == "fixed":
            history = [s for s in history if math.floor(s/window) == math.floor(t/window)]
        else:
            raise ValueError("unknown monitor")
        accepted.append(len(history) < max_count)
        history.append(t)
    return accepted


def evaluate_tripwire(data):
    _keys(data, ("timestamps", "epochs", "releases", "truth_by_epoch", "window", "max_count"))
    n, d = _matrix(data["releases"], "releases")
    times, epochs = data["timestamps"], data["epochs"]
    if not isinstance(times, list) or len(times) != n or any(not _number(t) or t < 0 for t in times) or times != sorted(times):
        raise ValueError("timestamps must be finite sorted nonnegative arrivals")
    if not isinstance(epochs, list) or len(epochs) != n or any(type(e) is not int or e < 0 for e in epochs) or epochs != sorted(epochs):
        raise ValueError("epochs must be monotonically increasing integer key domains")
    truth = data["truth_by_epoch"]
    if not isinstance(truth, dict) or set(truth) != {str(e) for e in epochs} or any(not isinstance(row, list) or len(row) != d or not all(_number(x) for x in row) for row in truth.values()):
        raise ValueError("truth_by_epoch must exactly match epoch coordinate domains")
    if not _number(data["window"]) or data["window"] <= 0 or type(data["max_count"]) is not int or data["max_count"] < 1:
        raise ValueError("invalid monitor budget")
    out = []
    for kind in ("fixed", "sliding"):
        accepted = monitor_acceptance(times, data["window"], data["max_count"], kind)
        for epoch in sorted(set(epochs)):
            rows = [row for row, e, yes in zip(data["releases"], epochs, accepted) if e == epoch and yes]
            out.append(dict(case="tripwire", monitor=kind, epoch=epoch, accepted=len(rows),
                            rejected=sum(e == epoch and not yes for e, yes in zip(epochs, accepted)),
                            no_observation_zero_mse=_mse([[0.0]*d], [truth[str(epoch)]]),
                            first_release_mse=_mse(rows[:1], [truth[str(epoch)]]) if rows else None,
                            accumulated_mse=_mse([_vector_mean(rows)], [truth[str(epoch)]]) if rows else None,
                            pooling_scope="same_epoch_only"))
    return out


def evaluate_bundle(bundle):
    if not isinstance(bundle, dict) or set(bundle) != {"schema", "case", "data"} or bundle["schema"] != SCHEMA_VERSION:
        raise ValueError("Expected exact view_privacy.v1 bundle schema/case/data")
    evaluators = {"mixup": evaluate_mixup, "views": evaluate_views, "tripwire": evaluate_tripwire}
    if not isinstance(bundle["case"], str) or bundle["case"] not in evaluators:
        raise ValueError("unknown case")
    return evaluators[bundle["case"]](bundle["data"])


def synthetic_bundles(seed=1234, quick=False):
    rng = random.Random(seed)
    n, d, r = (12, 4, 32) if quick else (64, 16, 128)
    pool = [[rng.gauss(2, 1) for _ in range(d)] for _ in range(16)]
    oracle_mu, lam = _vector_mean(pool), 0.75
    calibration_samples = [list(rng.choice(pool)) for _ in range(64)]
    mu = _vector_mean(calibration_samples)
    targets = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]
    releases, corrected = [], []
    for target in targets:
        rs, cs = [], []
        for _ in range(r):
            decoy = rng.choice(pool)
            mixed = [lam*x+(1-lam)*z for x, z in zip(target, decoy)]
            rs.append(mixed)
            cs.append([(math.tanh(y)-(1-lam)*math.tanh(z))/lam for y, z in zip(mixed, decoy)])
        releases.append(rs)
        corrected.append(cs)
    yield dict(schema=SCHEMA_VERSION, case="mixup", data=dict(releases=releases, truth=targets,
               calibration_mean=mu, calibration_samples=calibration_samples, oracle_pool_mean=oracle_mu, **{"lambda": lam}, assumed_lambdas=[0.65, 0.75, 0.85],
               corrected_outputs=corrected, clean_outputs=[[math.tanh(x) for x in t] for t in targets]))
    labels = [i % 2 for i in range(4*n)]
    rows = [[rng.gauss((2*y-1)*1.5 if j < d//2 else 0, 1) for j in range(d)] for y in labels]
    yield dict(schema=SCHEMA_VERSION, case="views", data=dict(train=rows[:2*n], train_labels=labels[:2*n],
               evaluation=rows[2*n:], eval_labels=labels[2*n:], views=[list(range(d//2)), list(range(d//2, d))]))
    yield dict(schema=SCHEMA_VERSION, case="views", data=dict(train=rows[:2*n], train_labels=labels[:2*n],
               evaluation=rows[2*n:], eval_labels=labels[2*n:],
               views=[[j for j in range(d) if j % 3 != missing] for missing in range(3)]))
    epoch_truth = {str(i): [rng.gauss(0, 2) for _ in range(d)] for i in range(2)}
    for spacing in (0.1, 2.6):
        yield dict(schema=SCHEMA_VERSION, case="tripwire", data=dict(timestamps=[i*spacing for i in range(2*r)],
                   epochs=[i//r for i in range(2*r)], releases=[[x+rng.gauss(0, 2) for x in epoch_truth[str(i//r)]] for i in range(2*r)],
                   truth_by_epoch=epoch_truth, window=10, max_count=4))


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--bundle", help="Strict JSON observation bundle")
    return ap


def run(args):
    if args.output and any(Path(args.output + suffix).exists() for suffix in ("", ".jsonl", ".evidence.json", ".evidence.jsonl")):
        raise ValueError("Output and evidence paths must be fresh")
    if bool(args.toy) == bool(args.bundle):
        raise ValueError("Choose exactly one of --toy or --bundle")
    source = {"kind": "synthetic", "seed": args.seed}
    if args.bundle:
        source = {"kind": "supplied_arrays", "sha256": file_sha256(args.bundle)}
        bundles = [json.loads(Path(args.bundle).read_text())]
    else:
        bundles = list(synthetic_bundles(args.seed, args.quick))
    results = []
    for i, bundle in enumerate(bundles):
        results.extend(dict(experiment=EXPERIMENT_ID, bundle_index=i, **row) for row in evaluate_bundle(bundle))
    ev = EvidenceLogger(args.output, extra=source) if args.output else None
    out = artifacts.make_artifact(SCHEMA_VERSION, vars(args),
          "Repeated-target mixup observer; coordinate-view collusion requires multiple servers; count-monitor observer accumulates within one key epoch only.",
          interpretation="Synthetic machinery check or supplied-array evaluation; no trained-defense or cryptographic-security certificate. Mixup SEM is conditional on a fixed estimated mean and excludes calibration uncertainty; synthetic main calibration uses 64 independent pool draws, exact pool mean is a separately labeled oracle arm. Nonlinear correction uses supplied paired endpoint outputs.", evidence=ev)
    out["results"], out["source"] = results, source
    for row in results:
        artifacts.append_jsonl(args.output, row)
    artifacts.write_artifact(args.output, out, evidence=ev)
    print(json.dumps({"experiment": EXPERIMENT_ID, "cells": len(results), "source": source}))
    return 0


def self_test():
    try:
        assert all(evaluate_bundle(b) for b in synthetic_bundles(1234, True))
    except Exception:
        return 1
    return 0
