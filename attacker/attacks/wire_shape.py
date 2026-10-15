#!/usr/bin/env python3
"""wire-shape — passive traffic-shape analysis of the split link (the
general traffic-analysis setting; NetEcho, arXiv:2510.25472, is related
work, not the implemented conversation-recovery algorithm): packet sizes and timestamps on the client<->server link
leak per-message token counts, prefill/decode/training phase boundaries,
and sequence lengths — no payload content needed. Label-free; applies to
both modes (inference serving traffic and split-training activation/
gradient traffic).

WIRE-SHAPE JOURNAL SCHEMA (llm_attacker.wire_shape.v1)
======================================================
Exactly what the sibling MITM proxy emits. One JSON object per line:

    {"ts": 1712345678.123, "bytes": 4096, "dir": "c2s", "session_id": "s0"}

    ts          float, epoch seconds of the observed record (required)
    bytes       int >= 0, payload byte count of the record (required)
    dir         "c2s" (client->server) | "s2c" (server->client) (required)
    session_id  str, OPTIONAL; absent normalizes to None (records with no
                session_id are analyzed as one default session)

Schema drift (missing/extra keys, bad dir, non-numeric ts/bytes) is a
descriptive ValueError naming the file and line — never a silent skip.
The schema is also registered as the "wire-shape" SurfaceProvider
(attacker.surfaces) at module import, so future attacks consume the
journal through the provider registry: rows(mode, source=<journal path>)
yields ([ts, bytes], metadata) pairs.

Arms:

  length-leak   recover per-message token counts from byte sizes. A
                message = a maximal run of same-direction records inside
                one detected phase segment. tokens ~= bytes /
                bytes-per-token, with the model either SUPPLIED
                (--bytes-per-token) or LEARNED (toy mode: fit on the first
                half of messages, evaluated on the held-out half).
                Reports MAE (tokens) and Spearman rank correlation vs
                ground truth.
  phase-detect  segment prefill / decode / training phase boundaries from
                inter-record timing gaps (boundary = gap above
                --gap-ms, default 3x the median gap). Reports boundary
                precision/recall/F1 vs ground truth in toy mode
                (a predicted boundary matches truth within
                --boundary-tol records) plus heuristic segment labels.

Usage:
    python -m attacker --mode inference --attack wire-shape --help
    python -m attacker --mode inference --attack wire-shape --toy --quick \
        --output /tmp/ws.json
    python -m attacker --mode training --attack wire-shape \
        --journal captures/mitm.jsonl --bytes-per-token 256.0 --output ws.json
"""

import argparse
import json
import math
import random
import statistics

from .. import artifacts
from ..surfaces import SurfaceProvider, register_surface
from .common import add_common_args, journal_error

EXPERIMENT_ID = "wire_shape_analysis"
MODES = ("training", "inference")
REQUIRES_LABELS = False
DESCRIPTION = ("passive traffic-shape analysis of the split link (NetEcho "
               "line): packet sizes/timestamps leak per-message token "
               "counts and prefill/decode/training phase boundaries")

SCHEMA_VERSION = "llm_attacker.wire_shape.v1"
DIRS = ("c2s", "s2c")
ARMS = ("length-leak", "phase-detect")


class WireShapeProvider(SurfaceProvider):
    """The wire-shape journal as an observation surface: records of
    {ts, bytes, dir, session_id?} (schema llm_attacker.wire_shape.v1).
    rows(mode, source=<journal path>) yields ([ts, bytes], metadata) pairs
    with metadata the full normalized record. Pure python, torch-free."""

    name = "wire-shape"
    schema = {"ts": float, "bytes": int, "dir": str,
              "session_id": (str, type(None))}
    modes = ("training", "inference")

    def validate(self, meta, path="<record>"):
        """Key-set validation with the optional session_id normalized to
        None; dir must be c2s|s2c; ts/bytes numeric. ValueError on drift."""
        keys = set(meta)
        if keys == {"ts", "bytes", "dir"}:
            meta = {**meta, "session_id": None}
        elif keys != set(self.schema):
            raise ValueError(
                f"{path}: record keys {sorted(keys)} != surface "
                f"'wire-shape' schema {sorted(self.schema)} "
                f"(session_id optional; schema {SCHEMA_VERSION})")
        if meta["dir"] not in DIRS:
            raise ValueError(f"{path}: dir {meta['dir']!r} not in {DIRS}")
        ts, nbytes = meta["ts"], meta["bytes"]
        if isinstance(ts, bool) or not isinstance(ts, (int, float)):
            raise ValueError(f"{path}: ts must be a number, got {ts!r}")
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) \
                or nbytes < 0:
            raise ValueError(f"{path}: bytes must be an int >= 0, "
                             f"got {nbytes!r}")
        return {"ts": float(ts), "bytes": int(nbytes), "dir": meta["dir"],
                "session_id": meta["session_id"]}

    def rows(self, mode=None, source=None):
        self.check_mode(mode)
        if source is None:
            raise ValueError("surface 'wire-shape' needs source=<wire-shape "
                             f"journal path ({SCHEMA_VERSION})")
        def gen():
            with open(source) as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    meta = self.validate(json.loads(line),
                                         f"{source}:{lineno}")
                    yield [meta["ts"], meta["bytes"]], meta
        return gen()


register_surface(WireShapeProvider())


def read_journal(path):
    """Parse a wire-shape journal -> {session_key: [records sorted by ts]}.
    Records are validated through the WireShapeProvider (schema drift is a
    descriptive ValueError naming file and line)."""
    provider = WireShapeProvider()
    sessions = {}
    for _, meta in provider.rows(source=path):
        sessions.setdefault(meta["session_id"] or "<default>", []).append(meta)
    for records in sessions.values():
        records.sort(key=lambda r: r["ts"])
    return sessions


def segment_boundaries(records, gap_ms=None):
    """Detect phase boundaries from inter-record timing gaps. Returns
    (boundary_indices, threshold_seconds): record i is a boundary when the
    gap BEFORE it exceeds the threshold. Records must be ts-sorted."""
    if len(records) < 2:
        return [], float("nan")
    gaps = [records[i + 1]["ts"] - records[i]["ts"]
            for i in range(len(records) - 1)]
    thr = (gap_ms / 1000.0 if gap_ms is not None
           else 3.0 * statistics.median(gaps))
    return [i + 1 for i, g in enumerate(gaps) if g > thr], thr


def boundary_f1(true_bounds, pred_bounds, tol=1):
    """Boundary F1 with greedy one-to-one matching within tol records."""
    true_bounds, pred_bounds = sorted(true_bounds), sorted(pred_bounds)
    matched_true = set()
    tp = 0
    for p in pred_bounds:
        for ti, t in enumerate(true_bounds):
            if ti not in matched_true and abs(p - t) <= tol:
                matched_true.add(ti)
                tp += 1
                break
    precision = tp / len(pred_bounds) if pred_bounds \
        else (1.0 if not true_bounds else 0.0)
    recall = tp / len(true_bounds) if true_bounds \
        else (1.0 if not pred_bounds else 0.0)
    f1 = (2.0 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "n_true": len(true_bounds), "n_pred": len(pred_bounds),
            "n_matched": tp}


def extract_messages(records, boundaries=()):
    """Per-message units: maximal same-direction runs, also split at phase
    boundaries. Returns [{"dir", "start", "end", "bytes", "n_records"}]."""
    bset = set(boundaries)
    msgs = []
    for i, r in enumerate(records):
        if msgs and i not in bset and msgs[-1]["dir"] == r["dir"]:
            last = msgs[-1]
            last["end"] = i + 1
            last["bytes"] += r["bytes"]
            last["n_records"] += 1
        else:
            msgs.append({"dir": r["dir"], "start": i, "end": i + 1,
                         "bytes": r["bytes"], "n_records": 1})
    return msgs


def label_segments(records, boundaries, big_bytes=1024.0):
    """Heuristic phase labels for the segments between boundaries:
    s2c-dominated small-record segments -> decode; small c2s bursts ->
    prefill; large-record segments (activation/gradient tensors) ->
    training; mixed -> mixed. Info metric, not gate-scored."""
    bounds = sorted(set(boundaries) | {0, len(records)})
    out = []
    for s, e in zip(bounds, bounds[1:]):
        seg = records[s:e]
        if not seg:
            continue
        dirs = {r["dir"] for r in seg}
        mean_bytes = statistics.fmean(r["bytes"] for r in seg)
        if mean_bytes >= big_bytes:
            label = "training"
        elif dirs == {"s2c"} and len(seg) >= 4:
            label = "decode"
        elif dirs == {"c2s"}:
            label = "prefill"
        elif dirs == {"s2c"}:
            label = "decode"
        else:
            label = "mixed"
        out.append({"start": s, "end": e, "label": label,
                    "n_records": len(seg)})
    return out


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a, b):
    """Spearman rank correlation (average ranks on ties); nan for n < 2
    or a constant input."""
    if len(a) != len(b):
        raise ValueError("spearman: inputs differ in length")
    n = len(a)
    if n < 2:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    if va <= 0.0 or vb <= 0.0:
        return float("nan")
    return cov / math.sqrt(va * vb)


def length_leak(messages, truth=None, bytes_per_token=None):
    """Estimate per-message token counts from byte sizes.

    messages: extract_messages output; truth: optional {(start, end):
    tokens} ground-truth map (toy mode). bytes_per_token: supplied model;
    when None and truth is given, the model is LEARNED on the first half of
    matched messages and evaluated on the held-out half. Returns a dict of
    estimates (+ MAE / Spearman when truth is available)."""
    matched = [(m, truth[(m["start"], m["end"])]) for m in messages
               if truth and (m["start"], m["end"]) in truth] if truth \
        else [(m, None) for m in messages]
    learned = bytes_per_token is None
    if learned:
        fit = matched[:len(matched) // 2]
        eval_msgs = matched[len(matched) // 2:]
        if truth and fit:
            bytes_per_token = (sum(m["bytes"] for m, _ in fit)
                               / max(1, sum(t for _, t in fit)))
    else:
        eval_msgs = matched
    if not bytes_per_token or bytes_per_token <= 0:
        bytes_per_token = 4.0
    pairs = [[t, m["bytes"] / bytes_per_token] for m, t in eval_msgs]
    rec = {"bytes_per_token": bytes_per_token,
           "bytes_per_token_learned": learned,
           "n_messages": len(eval_msgs),
           "est_tokens": [round(p[1], 3) for p in pairs]}
    if truth:
        rec["true_tokens"] = [t for _, t in eval_msgs]
        rec["mae_tokens"] = (statistics.fmean(
            abs(t - e) for t, e in pairs) if pairs else float("nan"))
        rec["rank_corr"] = spearman([t for t, _ in pairs],
                                    [e for _, e in pairs])
    return rec


def make_toy_session(session_type, turns, seed):
    """Synthetic wire-shape session with ground truth. Inference turns:
    one c2s prefill record (prompt upload), a prefill-compute gap, then one
    s2c record per decoded token; a long gap closes the turn. Training
    steps: a c2s activation burst (fwd), a phase gap, an s2c gradient burst
    (bwd), a long step gap; payloads are activation-sized (x64 bytes per
    token-row). Returns (records, truth) with truth = {"boundaries",
    "segments", "messages"} — boundaries are record indices where a new
    phase starts (never 0), segments carry phase labels, messages carry
    per-message token counts."""
    rng = random.Random(seed)
    scale = 64 if session_type == "training" else 1
    sid = f"toy-{session_type}-{seed}"
    records, boundaries, segments, messages = [], [], [], []
    t = 1_700_000_000.0

    def emit(direction, tokens, gap_ms):
        nonlocal t
        nbytes = sum(rng.randint(3, 5) for _ in range(tokens)) * scale
        records.append({"ts": round(t, 6), "bytes": nbytes, "dir": direction,
                        "session_id": sid})
        t += gap_ms / 1000.0

    def open_segment(label):
        start = len(records)
        if start:
            boundaries.append(start)
        segments.append({"start": start, "label": label})
        messages.append({"start": start, "tokens": 0})
        return start

    def close_segment(start, label):
        segments[-1]["end"] = len(records)
        messages[-1]["end"] = len(records)

    for _ in range(turns):
        if session_type == "training":
            k = rng.randint(2, 4)
            s = open_segment("training")
            for _ in range(k):
                tok = rng.randint(16, 64)
                messages[-1]["tokens"] += tok
                emit("c2s", tok, rng.uniform(8.0, 25.0))
            close_segment(s, "training")
            t += rng.uniform(0.06, 0.12)
            s = open_segment("training")
            for _ in range(k):
                tok = rng.randint(16, 64)
                messages[-1]["tokens"] += tok
                emit("s2c", tok, rng.uniform(8.0, 25.0))
            close_segment(s, "training")
            t += rng.uniform(0.15, 0.5)
        else:
            s = open_segment("prefill")
            tok = rng.randint(8, 64)
            messages[-1]["tokens"] += tok
            emit("c2s", tok, rng.uniform(40.0, 60.0))
            close_segment(s, "prefill")
            s = open_segment("decode")
            for _ in range(rng.randint(12, 48)):
                messages[-1]["tokens"] += 1
                emit("s2c", 1, rng.uniform(5.0, 15.0))
            close_segment(s, "decode")
            t += rng.uniform(0.2, 0.5)
    return records, {"boundaries": sorted(boundaries), "segments": segments,
                     "messages": messages, "session_id": sid}


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--arm", choices=ARMS, nargs="+", default=list(ARMS),
                    help="analysis arm(s) to run (default: both)")
    ap.add_argument("--journal", default=None,
                    help=f"wire-shape journal to analyze ({SCHEMA_VERSION})")
    ap.add_argument("--bytes-per-token", type=float, default=None,
                    help="bytes-per-token model for length-leak; when "
                         "omitted, toy mode LEARNS it on the first half of "
                         "messages (held-out eval); journal mode falls "
                         "back to 4.0 with a note")
    ap.add_argument("--gap-ms", type=float, default=None,
                    help="phase-boundary gap threshold in ms (default: 3x "
                         "the median inter-record gap)")
    ap.add_argument("--boundary-tol", type=int, default=1,
                    help="a predicted boundary matches truth within this "
                         "many records")
    ap.add_argument("--turns", type=int, default=12,
                    help="toy: conversation turns / training steps")
    ap.add_argument("--session-type", choices=["inference", "training"],
                    default=None,
                    help="toy session shape (default: derived from --mode)")
    ap.add_argument("--evidence", action="store_true",
                    help="wrap the run in an EvidenceLogger (requires "
                         "--output): the artifact write is recorded as a "
                         "chained event and the evidence manifest is "
                         "finalized alongside")
    return ap


def _make_artifact(args, source, extra_config=None, evidence=None):
    config = {"attack": EXPERIMENT_ID, "mode": getattr(args, "mode", None),
              "source": source, "arms": args.arm,
              "bytes_per_token": args.bytes_per_token,
              "gap_ms": args.gap_ms, "boundary_tol": args.boundary_tol,
              "journal_schema": SCHEMA_VERSION, "seed": args.seed}
    if extra_config:
        config.update(extra_config)
    return artifacts.make_artifact(
        "dtraining.attacker.wire_shape.v1", config,
        "passive network observer of the split client<->server link "
        "(e.g. the sibling MITM proxy): sees only per-record sizes, "
        "timestamps, and directions — no payloads — and recovers "
        "per-message token counts and phase boundaries. Label-free.",
        interpretation="length-leak: low MAE / high rank correlation means "
                       "message sizes expose token counts; phase-detect: "
                       "high boundary F1 means timing gaps expose prefill/"
                       "decode/training structure. Both are passive-"
                       "observer leaks, independent of payload encryption.",
        evidence=evidence)


def _truth_maps(truth):
    msg_truth = {(m["start"], m["end"]): m["tokens"]
                 for m in truth["messages"]}
    seg_truth = {(s["start"], s["end"]): s["label"]
                 for s in truth["segments"]}
    return msg_truth, seg_truth


def _length_leak_cell(records, truth, args, seed):
    bounds, thr = segment_boundaries(records, args.gap_ms)
    msgs = extract_messages(records, bounds)
    msg_truth, _ = _truth_maps(truth) if truth else (None, None)
    rec = {"experiment": EXPERIMENT_ID, "arm": "length-leak", "seed": seed,
           "n_records": len(records),
           **length_leak(msgs, truth=msg_truth,
                         bytes_per_token=args.bytes_per_token)}
    if not args.bytes_per_token and not truth:
        rec["note"] = ("no --bytes-per-token supplied and no ground truth "
                       "to learn from; fell back to 4.0 bytes/token")
    return rec


def _phase_detect_cell(records, truth, args, seed):
    bounds, thr = segment_boundaries(records, args.gap_ms)
    labels = label_segments(records, bounds)
    rec = {"experiment": EXPERIMENT_ID, "arm": "phase-detect", "seed": seed,
           "n_records": len(records),
           "gap_threshold_ms": (thr * 1000.0 if thr == thr else None),
           "n_boundaries_pred": len(bounds),
           "segment_labels": {l: sum(1 for s in labels if s["label"] == l)
                              for l in sorted({s["label"]
                                               for s in labels})}}
    if truth:
        scores = boundary_f1(truth["boundaries"], bounds,
                             tol=args.boundary_tol)
        _, seg_truth = _truth_maps(truth)
        pred_map = {(s["start"], s["end"]): s["label"] for s in labels}
        n_match = sum(1 for k, lab in seg_truth.items()
                      if pred_map.get(k) == lab)
        rec.update(scores)
        rec["label_accuracy"] = (n_match / len(seg_truth)
                                 if seg_truth else float("nan"))
    return rec


def run_toy(args, evidence=None):
    if args.quick:
        args.turns = min(args.turns, 6)
        args.seeds = args.seeds[:2]
    session_type = args.session_type or (
        getattr(args, "mode", None) if getattr(args, "mode", None)
        in ("training", "inference") else "inference")
    out = _make_artifact(args, "toy", {
        "toy": True, "session_type": session_type, "turns": args.turns,
        "seeds": args.seeds}, evidence=evidence)
    per_arm = {}
    for seed in args.seeds:
        records, truth = make_toy_session(session_type, args.turns,
                                          args.seed + 1000 * seed)
        for arm in args.arm:
            try:
                rec = (_length_leak_cell(records, truth, args, seed)
                       if arm == "length-leak"
                       else _phase_detect_cell(records, truth, args, seed))
            except Exception as e:
                journal_error(args.output, EXPERIMENT_ID,
                              {"arm": arm, "seed": seed}, e)
                continue
            rec["session_type"] = session_type
            artifacts.append_jsonl(args.output, rec)
            out["results"].append(rec)
            per_arm.setdefault(arm, []).append(rec)
            if arm == "length-leak":
                print(f"[wire-shape] length-leak seed={seed}: "
                      f"MAE={rec['mae_tokens']:.3f} tokens "
                      f"rho={rec['rank_corr']:.4f} "
                      f"(bpt={rec['bytes_per_token']:.3f}, "
                      f"learned={rec['bytes_per_token_learned']})")
            else:
                print(f"[wire-shape] phase-detect seed={seed}: "
                      f"F1={rec['f1']:.4f} (P={rec['precision']:.3f} "
                      f"R={rec['recall']:.3f}) "
                      f"labels={rec['segment_labels']}")
    for arm, recs in sorted(per_arm.items()):
        row = {"experiment": EXPERIMENT_ID, "arm": arm,
               "session_type": session_type, "n_seeds": len(recs)}
        if arm == "length-leak":
            row["mae_tokens_mean"] = statistics.fmean(
                r["mae_tokens"] for r in recs)
            row["rank_corr_mean"] = statistics.fmean(
                r["rank_corr"] for r in recs)
        else:
            row["f1_mean"] = statistics.fmean(r["f1"] for r in recs)
            row["precision_mean"] = statistics.fmean(
                r["precision"] for r in recs)
            row["recall_mean"] = statistics.fmean(r["recall"] for r in recs)
            row["label_accuracy_mean"] = statistics.fmean(
                r["label_accuracy"] for r in recs)
        out["summary"].append(row)
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def run_journal(args, evidence=None):
    out = _make_artifact(args, "journal", {"journal": args.journal},
                         evidence=evidence)
    sessions = read_journal(args.journal)
    per_arm = {}
    for skey, records in sorted(sessions.items()):
        for arm in args.arm:
            try:
                rec = (_length_leak_cell(records, None, args, None)
                       if arm == "length-leak"
                       else _phase_detect_cell(records, None, args, None))
            except Exception as e:
                journal_error(args.output, EXPERIMENT_ID,
                              {"arm": arm, "session_id": skey}, e)
                continue
            rec["session_id"] = skey
            del rec["seed"]
            artifacts.append_jsonl(args.output, rec)
            out["results"].append(rec)
            per_arm.setdefault(arm, []).append(rec)
            print(f"[wire-shape] {arm} session={skey}: "
                  f"{rec['n_records']} records, "
                  + (f"est_tokens={rec['est_tokens']}"
                     if arm == "length-leak" else
                     f"boundaries={rec['n_boundaries_pred']} "
                     f"labels={rec['segment_labels']}"))
    for arm, recs in sorted(per_arm.items()):
        row = {"experiment": EXPERIMENT_ID, "arm": arm,
               "n_sessions": len(recs),
               "n_records_total": sum(r["n_records"] for r in recs)}
        if arm == "length-leak":
            row["n_messages_total"] = sum(r["n_messages"] for r in recs)
        else:
            row["n_boundaries_total"] = sum(r["n_boundaries_pred"]
                                            for r in recs)
        out["summary"].append(row)
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def _run(args, evidence=None):
    if args.journal:
        return run_journal(args, evidence=evidence)
    if args.toy:
        return run_toy(args, evidence=evidence)
    raise SystemExit(f"[{EXPERIMENT_ID}] pick an input path: --journal "
                     f"FILE (wire-shape journal, schema {SCHEMA_VERSION}) "
                     "or --toy (synthetic session with ground truth).")


def run(args):
    if getattr(args, "evidence", False):
        if not args.output:
            raise SystemExit(f"[{EXPERIMENT_ID}] --evidence requires "
                             "--output (the evidence manifest is written "
                             "next to the artifact)")
        from ..evidence import EvidenceLogger
        with EvidenceLogger(args.output,
                            node_role=f"attack/{EXPERIMENT_ID}") as ev:
            return _run(args, evidence=ev)
    return _run(args, evidence=None)


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("wire-shape traffic analysis (pure python):")

    from ..surfaces import get_surface, list_surfaces
    prov = get_surface("wire-shape")
    check("wire-shape surface registered at import",
          "wire-shape" in list_surfaces() and isinstance(
              prov, WireShapeProvider))
    check("missing session_id normalizes to None",
          prov.validate({"ts": 1.0, "bytes": 4, "dir": "c2s"})
          ["session_id"] is None)
    try:
        prov.validate({"ts": 1.0, "bytes": 4, "dir": "sideways"})
        check("bad dir rejected", False)
    except ValueError:
        check("bad dir rejected", True)
    try:
        prov.validate({"ts": 1.0, "bytes": 4})
        check("schema drift rejected", False)
    except ValueError:
        check("schema drift rejected", True)

    records, truth = make_toy_session("inference", turns=6, seed=3)
    check("toy session records match the journal schema",
          all(prov.validate(r) for r in records))
    bounds, thr = segment_boundaries(records)
    scores = boundary_f1(truth["boundaries"], bounds, tol=1)
    check("clean toy session: boundaries detected exactly (F1 = 1.0)",
          scores["f1"] == 1.0 and thr == thr)
    msgs = extract_messages(records, bounds)
    ll = length_leak(msgs, truth={(m["start"], m["end"]): m["tokens"]
                                  for m in truth["messages"]})
    check("clean toy session: learned-bpt MAE below 2 tokens",
          ll["mae_tokens"] < 2.0 and ll["bytes_per_token_learned"])
    check("clean toy session: rank correlation >= 0.9",
          ll["rank_corr"] >= 0.9)
    labels = label_segments(records, bounds)
    pred_map = {(s["start"], s["end"]): s["label"] for s in labels}
    check("clean toy session: heuristic phase labels all correct",
          all(pred_map.get((s["start"], s["end"])) == s["label"]
              for s in truth["segments"]))

    tr_records, tr_truth = make_toy_session("training", turns=4, seed=5)
    tr_bounds, _ = segment_boundaries(tr_records)
    check("training toy session: boundary F1 = 1.0",
          boundary_f1(tr_truth["boundaries"], tr_bounds, tol=1)["f1"] == 1.0)
    check("training segments labeled 'training'",
          all(s["label"] == "training"
              for s in label_segments(tr_records, tr_bounds)))

    check("spearman anchors",
          spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
          and spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0)
    check("boundary_f1 anchors: empty/empty is perfect, disjoint is 0",
          boundary_f1([], [])["f1"] == 1.0
          and boundary_f1([10], [500], tol=1)["f1"] == 0.0)
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
