#!/usr/bin/env python3
"""api-transcript observation surface (schema llm_attacker.api_transcript.v1).

The reasoning-theft modules (reasoning-replay, trace-inversion) observe a
different surface than the split-training wire captures: provider API
transcripts in which assistant turns may carry OPAQUE REASONING BLOBS (the
provider-side encrypted thinking blocks of arXiv:2603.07267 /
stolen-thoughts.com). One JSON object per line:

    {"session_id": str, "turn": int >= 0, "role": "user"|"assistant",
     "text": str, "blobs": [str, ...],          # opaque provider blobs, possibly empty
     "usage": {"thinking_tokens": int | null},  # billed reasoning tokens
     "ts": float epoch seconds}

`usage` may carry extra provider-specific keys; `thinking_tokens` is
required (null when the provider bills no reasoning for the turn).
Schema drift (missing/extra top-level keys, bad role, non-string blobs,
missing thinking_tokens, non-numeric ts) is a descriptive ValueError
naming file and line — never a silent skip.

The schema is registered as the "api-transcript" SurfaceProvider
(attacker.surfaces) at module import, so attacks consume journals through
the provider registry: rows(mode, source=<journal path>) yields
(text, metadata) pairs with metadata the full normalized record. Third
parties replay their own logged sessions through the battery by writing
this schema; attacker.mitm.extract_blobs harvests candidate blobs from
MITM payload sidecars into it.

Pure stdlib, torch-free (repo rule).
"""

import json
import time

from .surfaces import SurfaceProvider, register_surface

SCHEMA_VERSION = "llm_attacker.api_transcript.v1"
ROLES = ("user", "assistant")


class ApiTranscriptProvider(SurfaceProvider):
    """The api-transcript journal as an observation surface.
    rows(mode, source=<journal path>) yields (text, metadata) pairs with
    metadata the full normalized record. Pure python, torch-free."""

    name = "api-transcript"
    schema = {"session_id": str, "turn": int, "role": str, "text": str,
              "blobs": list, "usage": dict, "ts": (int, float)}
    modes = ("inference",)

    def validate(self, meta, path="<record>"):
        """Key-set validation plus per-field type checks; ValueError on
        drift, naming the record's origin."""
        keys = set(meta)
        if keys != set(self.schema):
            raise ValueError(
                f"{path}: record keys {sorted(keys)} != surface "
                f"'api-transcript' schema {sorted(self.schema)} "
                f"(schema {SCHEMA_VERSION})")
        sid = meta["session_id"]
        if not isinstance(sid, str) or not sid:
            raise ValueError(f"{path}: session_id must be a non-empty str, "
                             f"got {sid!r}")
        turn = meta["turn"]
        if isinstance(turn, bool) or not isinstance(turn, int) or turn < 0:
            raise ValueError(f"{path}: turn must be an int >= 0, got {turn!r}")
        if meta["role"] not in ROLES:
            raise ValueError(f"{path}: role {meta['role']!r} not in {ROLES}")
        if not isinstance(meta["text"], str):
            raise ValueError(f"{path}: text must be a str, "
                             f"got {type(meta['text']).__name__}")
        blobs = meta["blobs"]
        if not isinstance(blobs, list) \
                or any(not isinstance(b, str) for b in blobs):
            raise ValueError(f"{path}: blobs must be a list of str "
                             f"(opaque provider blob strings), got {blobs!r}")
        usage = meta["usage"]
        if not isinstance(usage, dict) or "thinking_tokens" not in usage:
            raise ValueError(f"{path}: usage must be a dict carrying "
                             f"'thinking_tokens' (int | null), got {usage!r}")
        tt = usage["thinking_tokens"]
        if tt is not None and (isinstance(tt, bool) or not isinstance(tt, int)
                               or tt < 0):
            raise ValueError(f"{path}: usage.thinking_tokens must be an "
                             f"int >= 0 or null, got {tt!r}")
        ts = meta["ts"]
        if isinstance(ts, bool) or not isinstance(ts, (int, float)):
            raise ValueError(f"{path}: ts must be a number, got {ts!r}")
        return {"session_id": sid, "turn": turn, "role": meta["role"],
                "text": meta["text"], "blobs": list(blobs),
                "usage": dict(usage), "ts": float(ts)}

    def rows(self, mode=None, source=None):
        self.check_mode(mode)
        if source is None:
            raise ValueError("surface 'api-transcript' needs source=<api-"
                             f"transcript journal path ({SCHEMA_VERSION})")

        def gen():
            with open(source) as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"{source}:{lineno}: not JSON: {e}")
                    meta = self.validate(rec, f"{source}:{lineno}")
                    yield meta["text"], meta
        return gen()


register_surface(ApiTranscriptProvider())


def make_record(session_id, turn, role, text, blobs=(), thinking_tokens=None,
                ts=None, usage_extra=None):
    """Build one normalized schema record (validated; raises ValueError on
    drift). ts defaults to now; usage_extra merges provider-specific usage
    fields."""
    usage = {"thinking_tokens": thinking_tokens}
    if usage_extra:
        usage.update(usage_extra)
    return ApiTranscriptProvider().validate(
        {"session_id": session_id, "turn": turn, "role": role, "text": text,
         "blobs": list(blobs), "usage": usage,
         "ts": time.time() if ts is None else ts}, "<make_record>")


def load_transcripts(path):
    """Read + validate an api-transcript journal into a list of normalized
    records, sorted per session by turn. Schema drift is a descriptive
    ValueError naming file and line."""
    provider = ApiTranscriptProvider()
    records = [meta for _, meta in provider.rows(source=path)]
    records.sort(key=lambda r: (r["session_id"], r["turn"]))
    return records


def extract_blobs(transcripts):
    """Flatten transcript records into one record per carried blob:

        {"session_id", "turn", "blob_index", "blob", "thinking_tokens",
         "text", "ts"}

    `text` is the assistant turn's visible text (the final answer the blob
    is supposed to have produced — the answer-consistency reference);
    `thinking_tokens` is the billed reasoning-token count (the round-trip
    billing reference). Turns without blobs contribute nothing."""
    out = []
    for rec in transcripts:
        for i, blob in enumerate(rec["blobs"]):
            out.append({"session_id": rec["session_id"], "turn": rec["turn"],
                        "blob_index": i, "blob": blob,
                        "thinking_tokens": rec["usage"]["thinking_tokens"],
                        "text": rec["text"], "ts": rec["ts"]})
    return out


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import os
    import tempfile

    print("api-transcript surface (pure python):")

    from .surfaces import get_surface, list_surfaces
    prov = get_surface("api-transcript")
    check("api-transcript surface registered at import",
          "api-transcript" in list_surfaces()
          and isinstance(prov, ApiTranscriptProvider))
    check("provider serves inference mode only", prov.modes == ("inference",))

    good = make_record("s0", 0, "user", "Compute 1 + 2.", ts=1700000000)
    check("a conforming record validates and normalizes ts",
          good["ts"] == 1700000000.0 and good["blobs"] == []
          and good["usage"]["thinking_tokens"] is None)
    asst = make_record("s0", 1, "assistant", "The answer is 3.",
                       blobs=["RTBLOB1.x.y"], thinking_tokens=12,
                       ts=1700000001)
    check("assistant record with a blob validates",
          asst["blobs"] == ["RTBLOB1.x.y"]
          and asst["usage"]["thinking_tokens"] == 12)

    for name, bad in (
            ("extra key rejected", {**good, "extra": 1}),
            ("missing key rejected",
             {k: v for k, v in good.items() if k != "blobs"}),
            ("bad role rejected", {**good, "role": "system"}),
            ("non-str blob rejected", {**good, "blobs": [7]}),
            ("usage without thinking_tokens rejected",
             {**good, "usage": {"completion_tokens": 3}}),
            ("bad ts rejected", {**good, "ts": "noon"}),
            ("bad turn rejected", {**good, "turn": -1})):
        try:
            prov.validate(bad, "<fixture>")
            check(name, False)
        except ValueError:
            check(name, True)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.jsonl")
        with open(path, "w") as f:
            for rec in (asst, good):
                f.write(json.dumps(rec) + "\n")
        loaded = load_transcripts(path)
        check("loader round-trips and sorts by turn",
              [r["turn"] for r in loaded] == [0, 1])
        flat = extract_blobs(loaded)
        check("extract_blobs flattens one record per blob",
              len(flat) == 1 and flat[0]["blob"] == "RTBLOB1.x.y"
              and flat[0]["thinking_tokens"] == 12
              and flat[0]["text"] == "The answer is 3.")
        drift = os.path.join(d, "drift.jsonl")
        with open(drift, "w") as f:
            f.write(json.dumps({"session_id": "s", "turn": 0}) + "\n")
        try:
            load_transcripts(drift)
            check("loader drift error names file and line", False)
        except ValueError as e:
            check("loader drift error names file and line",
                  f"{drift}:1" in str(e))
        try:
            list(prov.rows(mode="training", source=path))
            check("mode mismatch raises", False)
        except ValueError:
            check("mode mismatch raises", True)

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
