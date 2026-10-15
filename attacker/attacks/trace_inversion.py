#!/usr/bin/env python3
"""Citation scope: a synthetic arithmetic retrieval/composition study inspired by
Zhang, Morris and Shmatikov, "How to Steal Reasoning Without Reasoning
Traces" (https://arxiv.org/abs/2603.07267). It does not train the paper's
LLM trace-inversion or distillation models. The local search, calibration
and metrics are adaptations. See docs/reference-audit-2026-09-22.md.

trace-inversion (inference mode) — synthesize a victim's hidden reasoning
trace from (problem x, compressed summary b*, final answer y), after
arXiv:2603.07267 (Trace Inversion), strengthened per
README.md:

  1. verifier-guided search: k candidate traces per input, hard-filtered by
     answer-consistency (recomputed answer == observed y), ranked by
     summary-coverage (--candidates; 1 = a single local candidate);
  2. one critique-and-revise refinement pass on answer-inconsistent drafts
     (--refine);
  3. correctness-filtered distillation: the capability student learns only
     from answer-consistent synthesized traces;
  4. style-transfer calibration: --calibrate-with consumes reasoning-replay's
     decoded genuine traces and learns the victim's true template
     parameters (opening/step/final/closing slots) within this toy world;
  5. metrics that survive review: TF1 + ROUGE-L-ish LCS F1 vs the genuine
     traces (the toy world has ground truth — use it), answer-consistency,
     summary-coverage, and a capability-excess score (student trained on
     synthesized traces vs the answer-only control) gated with
     attacker.gate.verdict.

TOY WORLD (documented, honest, torch-free): synthetic arithmetic multi-step
problems "Compute a op1 b op2 c." (a, b, c in 2..9, ops in {+,-,*}). The
"victim" is a small-model stand-in that evaluates STRICTLY LEFT-TO-RIGHT
(many small models do) and narrates two templated steps, a compressed
summary b* (opening + final line; intermediate steps dropped), and the
final answer y. The attacker sees (x, b*, y) in the `summary` setting and
(x, y) in `no-summary`. The inversion machinery is real: the surrogate
corpus comes from a DIFFERENT problem family (two-operand problems, a
different template, standard operator precedence — the surrogate-style gap
is genuine), retrieval indexes it by problem signature, and the synthesizer
substitutes the parsed values and RE-DERIVES the intermediate steps by
actually computing the arithmetic — so a synthesized trace is genuinely
consistent exactly when the attacker's evaluation-order hypothesis matches
the victim's convention. What the attacker cannot know a priori: (a) the
victim's template style, (b) the victim's evaluation-order convention.
Search + verification recovers (b) from y; calibration on replay-decoded
genuine traces recovers (a).

Usage:
    python -m attacker --mode inference --attack trace-inversion --help
    python -m attacker --mode inference --attack trace-inversion --toy \
        --quick --output /tmp/ti.json
    python -m attacker --mode inference --attack trace-inversion --toy \
        --candidates 4 --refine --calibrate-with replay.json.decoded.jsonl \
        --evidence --output ti.json
"""

import argparse
import json
import math
import random
import re
from collections import Counter

from .. import artifacts, gate
from .common import add_common_args, journal_error

EXPERIMENT_ID = "trace_inversion"
MODES = ("inference",)
REQUIRES_LABELS = False
DESCRIPTION = ("synthesize hidden reasoning traces from (problem, summary, "
               "answer) via retrieval+composition with verifier-guided "
               "search, refinement, replay-calibrated style transfer, and "
               "a gated capability-excess student study (Trace Inversion "
               "line, arXiv:2603.07267)")

SCHEMA = "dtraining.attacker.trace_inversion.v1"

OPS = ("+", "-", "*")
ORDER_LTR = "left-to-right"
ORDER_PRECEDENCE = "precedence"
ORDERS = (ORDER_PRECEDENCE, ORDER_LTR)
SETTINGS = ("summary", "no-summary")

VICTIM_STYLE = {
    "name": "victim",
    "opening": "Let's compute {a} {op1} {b} {op2} {c}.",
    "step": "Step {i}: {verb} {u} and {v} to get {r}.",
    "verbs": {"+": "add", "-": "subtract", "*": "multiply"},
    "final": "The final answer is {y}.",
    "closing": "So the result is {y}.",
}

SURROGATE_STYLE = {
    "name": "surrogate",
    "opening": "Compute {a} {op1} {b} {op2} {c}.",
    "step": "Work: {u} {op} {v} = {r}.",
    "verbs": None,
    "final": "Answer: {y}.",
    "closing": None,
}

SURROGATE_TERSE = {**SURROGATE_STYLE, "name": "surrogate-terse",
                   "opening": None}

NOTES = (
    "Toy world: the victim evaluates strictly left-to-right and narrates "
    "templated steps; the surrogate corpus is a different problem family "
    "(two-operand, precedence convention, different template). The "
    "synthesizer RE-COMPUTES the arithmetic, so answer-consistency is "
    "genuine. Calibration learns only template slots (opening/step/final/"
    "closing phrasing) from replay-decoded traces, never evaluation order — "
    "order is recovered by verifier-guided search against the observed "
    "answer. All numbers are machinery checks of the combined design, not "
    "claims about production models.")


def _apply(op, u, v):
    return {"+": u + v, "-": u - v, "*": u * v}[op]


def make_problem(rng, lo=2, hi=9):
    return {"a": rng.randint(lo, hi), "b": rng.randint(lo, hi),
            "c": rng.randint(lo, hi),
            "op1": rng.choice(OPS), "op2": rng.choice(OPS)}


def problem_text(p):
    return f"Compute {p['a']} {p['op1']} {p['b']} {p['op2']} {p['c']}."


_PROBLEM_RE = re.compile(
    r"(-?\d+)\s*([+\-*])\s*(-?\d+)\s*([+\-*])\s*(-?\d+)")


def parse_problem(text):
    """Parse "Compute a op1 b op2 c." back into a problem dict (None on
    failure — a genuine attacker-side limitation)."""
    m = _PROBLEM_RE.search(text or "")
    if not m:
        return None
    a, op1, b, op2, c = m.groups()
    return {"a": int(a), "op1": op1, "b": int(b), "op2": op2, "c": int(c)}


def compute_steps(p, order):
    """[(u, op, v, r), ...] in evaluation order under the given convention.
    The conventions differ exactly when op2 == '*' and op1 != '*'."""
    if order == ORDER_PRECEDENCE and p["op2"] == "*" and p["op1"] != "*":
        r1 = p["b"] * p["c"]
        return [(p["b"], "*", p["c"], r1),
                (p["a"], p["op1"], r1, _apply(p["op1"], p["a"], r1))]
    r1 = _apply(p["op1"], p["a"], p["b"])
    return [(p["a"], p["op1"], p["b"], r1),
            (r1, p["op2"], p["c"], _apply(p["op2"], r1, p["c"]))]


def final_answer(p, order):
    return compute_steps(p, order)[-1][3]


def render_step(step, i, style):
    u, op, v, r = step
    if style.get("verbs"):
        return style["step"].format(i=i, verb=style["verbs"][op],
                                    u=u, v=v, r=r)
    return style["step"].format(i=i, op=op, u=u, v=v, r=r)


def render_trace(p, steps, style):
    """Full templated trace: opening, steps, final, optional closing."""
    lines = []
    y = steps[-1][3]
    if style.get("opening"):
        lines.append(style["opening"].format(**p))
    lines.extend(render_step(s, i, style) for i, s in enumerate(steps, 1))
    if style.get("final"):
        lines.append(style["final"].format(y=y))
    if style.get("closing"):
        lines.append(style["closing"].format(y=y))
    return " ".join(lines)


def victim_example(rng):
    """One toy victim rollout: {"problem", "x", "trace", "summary",
    "answer"}. The victim evaluates LEFT-TO-RIGHT; the summary b* keeps the
    opening and final lines and drops the intermediate steps."""
    p = make_problem(rng)
    steps = compute_steps(p, ORDER_LTR)
    y = steps[-1][3]
    opening = VICTIM_STYLE["opening"].format(**p)
    final = VICTIM_STYLE["final"].format(y=y)
    return {"problem": p, "x": problem_text(p),
            "trace": render_trace(p, steps, VICTIM_STYLE),
            "summary": f"{opening} {final}", "answer": str(y)}


def make_surrogate_corpus(rng, n=40):
    """Traces from a DIFFERENT problem family: two-operand problems under
    standard precedence, narrated in the surrogate templates. Returns
    [{"signature": frozenset(ops), "magnitude": int, "style": dict,
      "text": str}] — the retrieval index the attacker actually has."""
    corpus = []
    for _ in range(n):
        u, v = rng.randint(2, 99), rng.randint(2, 99)
        op = rng.choice(OPS)
        r = _apply(op, u, v)
        style = SURROGATE_STYLE if rng.random() < 0.5 else SURROGATE_TERSE
        parts = []
        if style["opening"]:
            parts.append(f"Compute {u} {op} {v}.")
        parts.append(style["step"].format(i=1, op=op, u=u, v=v, r=r))
        parts.append(style["final"].format(y=r))
        corpus.append({"signature": frozenset([op]),
                       "magnitude": u + v, "style": style,
                       "text": " ".join(parts)})
    return corpus


def retrieve(corpus, problem, tie_break=0):
    """Nearest surrogate trace by problem signature: maximal operator
    overlap, then closest operand magnitude, then a deterministic tie-break
    (candidate search perturbs the tie-break to vary template slots)."""
    ops = {problem["op1"], problem["op2"]}
    magnitude = problem["a"] + problem["b"] + problem["c"]
    keyed = sorted(
        corpus,
        key=lambda e: (-len(ops & e["signature"]),
                       abs(e["magnitude"] - magnitude),
                       (corpus.index(e) + tie_break) % len(corpus)))
    return keyed[0]


def _frac(texts, pred):
    hits = sum(1 for t in texts if pred(t))
    return hits / max(1, len(texts))


def learn_style(texts):
    """Learn victim-style template parameters from genuine (replay-decoded)
    traces: majority vote per slot over the candidate phrasings. Robust to
    token-level decode corruption because every slot is detected by
    redundant markers. Returns (style, support) with per-slot detection
    fractions for auditability."""
    texts = [t for t in texts if t and t.split()]
    support = {
        "n_traces": len(texts),
        "lets_opening": _frac(texts, lambda t: "Let's compute" in t),
        "numbered_steps": _frac(texts, lambda t:
                                bool(re.search(r"Step\s+\d+\s*:", t))),
        "work_steps": _frac(texts, lambda t: "Work:" in t),
        "final_answer_is": _frac(texts, lambda t: "final answer is" in t),
        "closing_line": _frac(texts, lambda t: "So the result is" in t),
        "verbal_ops": _frac(texts, lambda t: any(
            w in t for w in ("add", "subtract", "multiply"))),
    }
    numbered = support["numbered_steps"] >= support["work_steps"]
    verbal = support["verbal_ops"] >= 0.5
    style = {
        "name": "calibrated",
        "opening": (VICTIM_STYLE["opening"]
                    if support["lets_opening"] >= 0.5
                    else SURROGATE_STYLE["opening"]),
        "step": (VICTIM_STYLE["step"] if (numbered and verbal)
                 else SURROGATE_STYLE["step"]),
        "verbs": (dict(VICTIM_STYLE["verbs"]) if (numbered and verbal)
                  else None),
        "final": (VICTIM_STYLE["final"]
                  if support["final_answer_is"] >= 0.5
                  else SURROGATE_STYLE["final"]),
        "closing": (VICTIM_STYLE["closing"]
                    if support["closing_line"] >= 0.35 else None),
    }
    return style, support


def token_f1(pred, true):
    """Token-multiset F1 (TF1)."""
    pt, tt = pred.split(), true.split()
    if not pt or not tt:
        return 0.0
    common = sum((Counter(pt) & Counter(tt)).values())
    p, r = common / len(pt), common / len(tt)
    return 2.0 * p * r / (p + r) if p + r else 0.0


def _lcs_len(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(pred, true):
    """ROUGE-L-ish: LCS-based F1 over token sequences."""
    pt, tt = pred.split(), true.split()
    if not pt or not tt:
        return 0.0
    lcs = _lcs_len(pt, tt)
    p, r = lcs / len(pt), lcs / len(tt)
    return 2.0 * p * r / (p + r) if p + r else 0.0


def summary_coverage(summary, trace_text):
    """Fraction of b* content tokens (multiset) present in the trace."""
    if not summary:
        return None
    want, have = Counter(summary.split()), Counter(trace_text.split())
    total = sum(want.values())
    return (sum((want & have).values()) / total) if total else None


def _split_sentences(text):
    return [s.strip().rstrip(".") + "." for s in (text or "").split(". ")
            if s.strip()]


def synthesize(x_text, summary, answer, style, order, use_summary=True):
    """Compose one candidate trace: parse x, re-derive the steps by actually
    computing under the order hypothesis, render in the given style, and —
    in the summary setting — copy the observed b* lines verbatim (they are
    genuine victim text)."""
    p = parse_problem(x_text)
    if p is None:
        return {"trace": "", "steps": None, "problem": None, "order": order,
                "recomputed": None, "consistent": False,
                "coverage": summary_coverage(summary, "") if use_summary
                else None,
                "error": "unparseable problem text"}
    steps = compute_steps(p, order)
    recomputed = steps[-1][3]
    consistent = str(recomputed) == str(answer).strip()
    if use_summary and summary:
        sents = _split_sentences(summary)
        opening = sents[0] if sents else None
        final = sents[-1] if len(sents) >= 2 else None
    else:
        opening = final = None
    lines = []
    if opening:
        lines.append(opening)
    elif style.get("opening"):
        lines.append(style["opening"].format(**p))
    lines.extend(render_step(s, i, style) for i, s in enumerate(steps, 1))
    if final:
        lines.append(final)
    elif style.get("final"):
        lines.append(style["final"].format(y=recomputed))
    if style.get("closing"):
        lines.append(style["closing"].format(y=recomputed))
    trace = " ".join(lines)
    return {"trace": trace, "steps": steps, "problem": p, "order": order,
            "style": style, "recomputed": recomputed,
            "consistent": consistent,
            "coverage": summary_coverage(summary, trace)}


def style_variant(style):
    """The alternate template slot filling for candidate search."""
    v = dict(style)
    v["name"] = style.get("name", "style") + "-variant"
    v["closing"] = None if style.get("closing") else "So the result is {y}."
    return v


def synthesize_candidates(x_text, summary, answer, style, k,
                          use_summary=True, corpus=None):
    """Up to k candidates, varying the evaluation-order hypothesis and the
    template slots (uncalibrated: retrieval tie-breaks over the surrogate
    corpus; calibrated: the learned style and its variant)."""
    k = max(1, int(k))
    candidates = []
    for i in range(k):
        order = ORDERS[i % len(ORDERS)]
        if corpus is not None:
            cand_style = retrieve(corpus, parse_problem(x_text)
                                  or {"op1": "+", "op2": "+", "a": 0,
                                      "b": 0, "c": 0},
                                  tie_break=i // len(ORDERS))["style"]
        else:
            cand_style = (style if (i // len(ORDERS)) % 2 == 0
                          else style_variant(style))
        cand = synthesize(x_text, summary, answer, cand_style, order,
                          use_summary)
        cand["cand_index"] = i
        candidates.append(cand)
    return candidates


def best_of_k(candidates):
    """Verifier-guided ranking: answer-consistency is a hard filter
    (recomputed == observed y), then maximal summary-coverage, then lowest
    candidate index."""
    consistent = [c for c in candidates if c["consistent"]]
    pool = consistent or candidates
    best = max(pool, key=lambda c: (
        c["coverage"] if c["coverage"] is not None else 0.0,
        -c.get("cand_index", 0)))
    return best


def refine(best, x_text, summary, answer, use_summary=True):
    """One critique-and-revise iteration: the draft's recomputed answer
    contradicts the observed y, so re-derive the trace under the untried
    evaluation order (re-rendered in the draft's own style). Returns
    (possibly-new best, refined_flag)."""
    if best["consistent"] or not best.get("steps"):
        return best, False
    alt = ORDER_LTR if best["order"] == ORDER_PRECEDENCE \
        else ORDER_PRECEDENCE
    cand = synthesize(x_text, summary, answer,
                      best.get("style", SURROGATE_STYLE), alt, use_summary)
    cand["cand_index"] = None
    cand["refined_from"] = best["order"]
    if cand["consistent"]:
        return cand, True
    return best, False


def train_student(train_syntheses):
    """Learn the evaluation-order convention from ANSWER-CONSISTENT
    synthesized traces only (correctness-filtered distillation): whichever
    operand pair the consistent traces evaluate first is the transferred
    procedure."""
    votes = {ORDER_LTR: 0, ORDER_PRECEDENCE: 0}
    used = 0
    for s in train_syntheses:
        if not s["consistent"] or not s.get("steps"):
            continue
        p, first = s["problem"], s["steps"][0]
        u, _, v, _ = first
        if u == p["b"] and v == p["c"] and p["op1"] != "*":
            votes[ORDER_PRECEDENCE] += 1
        else:
            votes[ORDER_LTR] += 1
        used += 1
    order = ORDER_LTR if votes[ORDER_LTR] >= votes[ORDER_PRECEDENCE] \
        else ORDER_PRECEDENCE
    return {"order": order, "n_traces_used": used,
            "votes": dict(votes)}


def student_answer(model, problem):
    return final_answer(problem, model["order"])


def train_answer_only(pairs):
    """The answer-only control: exact (x -> y) memorization with a global
    majority fallback — no procedure transfers."""
    table = {}
    for x, y in pairs:
        table[x] = y
    majority = Counter(y for _, y in pairs).most_common(1)[0][0] \
        if pairs else None
    return {"table": table, "majority": majority}


def control_answer(model, x_text):
    return model["table"].get(x_text, model["majority"])


def make_toy_world(seed, n_train, n_eval, n_heldout):
    rng = random.Random(seed)
    return {"train": [victim_example(rng) for _ in range(n_train)],
            "eval": [victim_example(rng) for _ in range(n_eval)],
            "heldout": [victim_example(rng) for _ in range(n_heldout)]}


def _solve_one(ex, style, k, use_summary, corpus, do_refine):
    cands = synthesize_candidates(ex["x"], ex["summary"], ex["answer"],
                                  style, k, use_summary, corpus=corpus)
    best = best_of_k(cands)
    refined = False
    if do_refine:
        best, refined = refine(best, ex["x"], ex["summary"], ex["answer"],
                               use_summary)
    return best, refined


def run_cell(world, *, setting, calibrated, k, do_refine, seed, style,
             corpus, n_seeds):
    use_summary = setting == "summary"
    train_syn = [_solve_one(ex, style, k, use_summary, corpus, do_refine)[0]
                 for ex in world["train"]]
    eval_pairs = [_solve_one(ex, style, k, use_summary, corpus, do_refine)
                  for ex in world["eval"]]
    eval_syn = [b for b, _ in eval_pairs]
    n_refined = sum(1 for _, r in eval_pairs if r)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    tf1 = mean([token_f1(s["trace"], ex["trace"])
                for s, ex in zip(eval_syn, world["eval"])])
    rouge = mean([rouge_l_f1(s["trace"], ex["trace"])
                  for s, ex in zip(eval_syn, world["eval"])])
    consistency = mean([1.0 if s["consistent"] else 0.0
                        for s in eval_syn])
    coverage = mean([s["coverage"] for s in eval_syn])

    model = train_student(train_syn)
    control = train_answer_only([(ex["x"], ex["answer"])
                                 for ex in world["train"]])
    student_hits = control_hits = 0
    for ex in world["heldout"]:
        p = parse_problem(ex["x"])
        if p is None:
            continue
        if str(student_answer(model, p)) == ex["answer"]:
            student_hits += 1
        if str(control_answer(control, ex["x"])) == ex["answer"]:
            control_hits += 1
    n_held = len(world["heldout"])
    student_acc = student_hits / max(1, n_held)
    control_acc = control_hits / max(1, n_held)
    verdict = gate.verdict(student_acc, control_acc, n_held,
                           n_seeds=n_seeds)
    return {"experiment": EXPERIMENT_ID, "setting": setting,
            "calibrated": calibrated, "candidates": k, "refined": do_refine,
            "seed": seed, "n_train": len(train_syn), "n_eval": len(eval_syn),
            "n_refined": n_refined,
            "tf1": round(tf1, 4), "rouge_l": round(rouge, 4),
            "answer_consistency": round(consistency, 4),
            "summary_coverage": (round(coverage, 4)
                                 if coverage is not None else None),
            "student": {"accuracy": round(student_acc, 4),
                        "order_learned": model["order"],
                        "n_traces_used": model["n_traces_used"]},
            "control": {"accuracy": round(control_acc, 4)},
            "capability_excess_pp": verdict["excess_pp"],
            "gate": verdict}


def load_decoded(path):
    """reasoning-replay decoded-traces JSONL -> list of decoded texts."""
    texts = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"[{EXPERIMENT_ID}] {path}:{lineno}: not "
                                 f"JSON: {e}")
            if not isinstance(rec.get("decoded_text"), str):
                raise SystemExit(
                    f"[{EXPERIMENT_ID}] {path}:{lineno}: missing "
                    "'decoded_text' (expected reasoning-replay's "
                    "<output>.decoded.jsonl schema)")
            texts.append(rec["decoded_text"])
    return texts


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--setting", choices=SETTINGS, nargs="+",
                    default=list(SETTINGS),
                    help="observation setting: (x, b*, y) vs (x, y) "
                         "(default: both)")
    ap.add_argument("--candidates", type=int, default=1,
                    help="verifier-guided search width; 1 = the paper's "
                         "single decode. When > 1, the k=1 baseline cell is "
                         "also emitted for the uplift measurement")
    ap.add_argument("--refine", action="store_true",
                    help="one critique-and-revise pass on answer-"
                         "inconsistent drafts (re-derive under the untried "
                         "evaluation order)")
    ap.add_argument("--calibrate-with", default=None, metavar="JSONL",
                    help="reasoning-replay decoded traces "
                         "(<output>.decoded.jsonl): learn victim-style "
                         "template parameters and add calibrated cells")
    ap.add_argument("--n-problems", type=int, default=24,
                    help="toy problems per split (train/eval; heldout is "
                         "1.5x, rounded)")
    ap.add_argument("--evidence", action="store_true",
                    help="seal the run with an attacker.evidence chain "
                         "(<output>.evidence.json[l]); requires --output")
    return ap


def _run(args, evidence=None):
    if not args.toy:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] a live victim integration is deliberately "
            "not wired in this torch-free tier: real inversion needs a "
            "victim API producing (x, b*, y) triples plus a GPU node for "
            "the learned inversion model (see docs/REASONING_THEFT_MODULES."
            "md, module 2 item 6). Use --toy for the documented end-to-end "
            "machinery check (retrieval+composition, search, refinement, "
            "calibration, capability-excess gating all run for real), "
            "optionally with --calibrate-with <replay decoded.jsonl>.")
    n = args.n_problems
    if args.quick:
        args.seeds = args.seeds[:1]
        n = min(n, 12)
    decoded = None
    if args.calibrate_with:
        decoded = load_decoded(args.calibrate_with)
    out = artifacts.make_artifact(
        SCHEMA,
        {"attack": EXPERIMENT_ID, "mode": getattr(args, "mode", None),
         "toy": True, "settings": args.setting,
         "candidates": args.candidates, "refine": args.refine,
         "calibrate_with": args.calibrate_with, "n_problems": n,
         "seeds": args.seeds},
        "label-free black-box observer of a reasoning API: sees the problem "
        "x, a compressed trace summary b* (summary setting) or nothing "
        "(no-summary), and the final answer y; synthesizes the hidden "
        "trace by retrieval+composition over a style-shifted surrogate "
        "corpus. Toy victim evaluates strictly left-to-right; ground-truth "
        "traces score TF1/ROUGE. Calibration traces come from "
        "reasoning-replay decodes of the SAME toy victim (the combined "
        "design).",
        interpretation="TF1 / ROUGE-L vs genuine traces (toy ground "
                       "truth); answer-consistency = recomputed answer == "
                       "observed y; capability-excess = student accuracy "
                       "over the answer-only memorization control, gated "
                       "(excess > +1.0 pp FAILs = capability transfers). "
                       + NOTES,
        evidence=evidence)
    cells = []
    for seed in args.seeds:
        world = make_toy_world(args.seed + 1000 * seed, n, n,
                               max(6, int(1.5 * n)))
        corpus = make_surrogate_corpus(random.Random(args.seed + 7 * seed))
        learned = None
        if decoded:
            learned, support = learn_style(decoded)
            out["config"]["calibration_support"] = support
        for setting in args.setting:
            calibrated_opts = [False, True] if learned else [False]
            for calibrated in calibrated_opts:
                style = learned if calibrated else SURROGATE_STYLE
                korpus = None if calibrated else corpus
                for k in sorted({1, args.candidates}):
                    for do_refine in ((False, True) if args.refine
                                      else (False,)):
                        cell_id = {"setting": setting,
                                   "calibrated": calibrated,
                                   "candidates": k, "refined": do_refine,
                                   "seed": seed}
                        try:
                            rec = run_cell(world, setting=setting,
                                           calibrated=calibrated, k=k,
                                           do_refine=do_refine, seed=seed,
                                           style=style, corpus=korpus,
                                           n_seeds=len(args.seeds))
                        except Exception as e:
                            journal_error(args.output, EXPERIMENT_ID,
                                          cell_id, e)
                            continue
                        cells.append(rec)
                        artifacts.append_jsonl(args.output, rec)
                        if evidence is not None:
                            evidence.event("cell_result", rec)
                        out["results"].append(rec)
                        print(f"[trace-inversion] {setting} seed={seed} "
                              f"calibrated={calibrated} k={k} "
                              f"refined={do_refine}: tf1={rec['tf1']} "
                              f"rougeL={rec['rouge_l']} "
                              f"consistency={rec['answer_consistency']} "
                              f"excess={rec['capability_excess_pp']:+.1f}pp "
                              f"gate={rec['gate']['verdict']}")
    by = {}
    for rec in cells:
        key = (rec["setting"], rec["calibrated"], rec["candidates"],
               rec["refined"])
        by.setdefault(key, []).append(rec)
    for (setting, calibrated, k, do_refine), recs in sorted(by.items()):
        out["summary"].append({
            "experiment": EXPERIMENT_ID, "setting": setting,
            "calibrated": calibrated, "candidates": k,
            "refined": do_refine, "n_seeds": len(recs),
            "tf1_mean": sum(r["tf1"] for r in recs) / len(recs),
            "rouge_l_mean": sum(r["rouge_l"] for r in recs) / len(recs),
            "answer_consistency_mean":
                sum(r["answer_consistency"] for r in recs) / len(recs),
            "capability_excess_pp_mean":
                sum(r["capability_excess_pp"] for r in recs) / len(recs),
            "gate_verdicts": [r["gate"]["verdict"] for r in recs]})
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


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

    import os
    import tempfile

    print("trace-inversion core (pure python):")

    rng = random.Random(0)
    p = {"a": 3, "op1": "+", "b": 4, "op2": "*", "c": 5}
    check("problem text round-trips through the parser",
          parse_problem(problem_text(p)) == p)
    check("the two order conventions differ exactly on a+b*c",
          final_answer(p, ORDER_LTR) == 35
          and final_answer(p, ORDER_PRECEDENCE) == 23)
    p2 = {"a": 3, "op1": "*", "b": 4, "op2": "-", "c": 5}
    check("op1='*' makes the conventions coincide",
          final_answer(p2, ORDER_LTR) == final_answer(p2, ORDER_PRECEDENCE)
          == 7)

    ex = victim_example(random.Random(7))
    check("victim example is self-consistent",
          str(final_answer(ex["problem"], ORDER_LTR)) == ex["answer"]
          and ex["answer"] in ex["trace"]
          and ex["answer"] in ex["summary"])
    check("summary drops the intermediate steps",
          "Step" not in ex["summary"] and "Step" in ex["trace"])

    check("token_f1 anchors",
          token_f1("a b c", "a b c") == 1.0
          and token_f1("a b", "c d") == 0.0
          and 0.0 < token_f1("a b c", "a b d") < 1.0)
    check("rouge_l is order-sensitive where token_f1 is not",
          rouge_l_f1("a b c", "a b c") == 1.0
          and rouge_l_f1("a b", "c d") == 0.0
          and rouge_l_f1("b a", "a b") < token_f1("b a", "a b"))

    corpus = make_surrogate_corpus(random.Random(1), n=20)
    check("surrogate corpus is the other family (2-operand, precedence)",
          all(len(e["signature"]) == 1 for e in corpus))
    check("retrieval returns a corpus entry with a style",
          retrieve(corpus, p)["style"]["step"].startswith("Work:"))

    x = problem_text(p)
    summary = f"{VICTIM_STYLE['opening'].format(**p)} " \
              f"{VICTIM_STYLE['final'].format(y=35)}"
    c1 = best_of_k(synthesize_candidates(x, summary, "35", SURROGATE_STYLE,
                                         1, True, corpus=corpus))
    c4 = best_of_k(synthesize_candidates(x, summary, "35", SURROGATE_STYLE,
                                         4, True, corpus=corpus))
    check("k=1 default hypothesis is answer-inconsistent on a+b*c",
          not c1["consistent"] and c1["recomputed"] == 23)
    check("best-of-4 verifier search finds the LTR-consistent candidate",
          c4["consistent"] and c4["recomputed"] == 35
          and c4["order"] == ORDER_LTR)
    fixed, did = refine(c1, x, summary, "35", True)
    check("refinement repairs the inconsistent draft", did
          and fixed["consistent"] and fixed["order"] == ORDER_LTR)

    check("summary coverage is exact when b* lines are copied",
          summary_coverage(summary, c4["trace"]) == 1.0)

    style, support = learn_style([ex["trace"] for ex in
                                  (victim_example(random.Random(i))
                                   for i in range(8))])
    check("calibration learns the victim template from genuine traces",
          style["opening"] == VICTIM_STYLE["opening"]
          and style["step"] == VICTIM_STYLE["step"]
          and style["final"] == VICTIM_STYLE["final"]
          and style["closing"] == VICTIM_STYLE["closing"]
          and support["n_traces"] == 8)
    s_style, _ = learn_style([e["text"] for e in corpus])
    check("calibration learns the surrogate template from surrogate text",
          s_style["step"] == SURROGATE_STYLE["step"]
          and s_style["closing"] is None)

    world = make_toy_world(11, 12, 12, 18)
    cell_uncal = run_cell(world, setting="summary", calibrated=False, k=1,
                          do_refine=False, seed=0, style=SURROGATE_STYLE,
                          corpus=corpus, n_seeds=1)
    cell_cal = run_cell(world, setting="summary", calibrated=True, k=1,
                        do_refine=False, seed=0, style=style, corpus=None,
                        n_seeds=1)
    check("calibrated inversion beats surrogate-style TF1",
          cell_cal["tf1"] > cell_uncal["tf1"] + 0.05)
    check("uncalibrated k=1 consistency reflects the order gap",
          0.5 < cell_uncal["answer_consistency"] < 1.0)
    check("student transfers the LTR procedure; control stays near zero",
          cell_uncal["student"]["accuracy"] > 0.9
          and cell_uncal["control"]["accuracy"] < 0.2)
    check("capability-excess carries a gate verdict",
          cell_uncal["gate"]["verdict"] in ("PASS", "FAIL")
          and cell_uncal["capability_excess_pp"] > 1.0)

    with tempfile.TemporaryDirectory() as d:
        ns = build_parser().parse_args(
            ["--toy", "--quick", "--candidates", "4", "--refine",
             "--output", os.path.join(d, "ti.json")])
        ns.mode = "inference"
        check("toy run end to end torch-free", run(ns) == 0)
        art = json.load(open(os.path.join(d, "ti.json")))
        check("artifact has both settings x candidate widths x refine arms",
              len(art["results"]) == 2 * 2 * 2
              and all(r["gate"]["verdict"] in ("PASS", "FAIL")
                      for r in art["results"]))

    try:
        ns = build_parser().parse_args([])
        run(ns)
        check("non-toy run raises the documented integration-point error",
              False)
    except SystemExit as e:
        check("non-toy run raises the documented integration-point error",
              "--toy" in str(e))

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
