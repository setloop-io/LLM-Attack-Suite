#!/usr/bin/env python3
"""multi-tenant-isolation — cross-tenant isolation tests for a shared LLM
serving stack (THREAT_MODEL_REVIEW P0): do batching, routing, caches,
memory reuse, or error/cancel paths let one tenant infer or receive
another tenant's information?

Attacker model: a low-privilege AUTHENTICATED tenant (no administrative
access, no direct access to victim identifiers) sending chosen requests
concurrently with a cooperating victim fixture. Label-free; applies to
inference serving and to multi-tenant (fine-)training serving — both
share the batcher / cache / allocator components under test.

TOY WORLDS (--toy, deterministic, seeded, torch-free)
=====================================================
ToyServingStack simulates the components the arms exercise:

  sessions   session table mapping session_id -> tenant + cached context
  cache      prefix cache keyed by prefix hash; the cache_tenant_binding
             defense keys entries by (tenant, hash) so a hit can never
             cross a tenant / authorization boundary
  batcher    continuous-batcher (row, sequence) slots; batch_row_binding
             namespaces slots per tenant
  kv         KV-content rows (the same row space the batcher serves)
  buffers    decode/response buffers; scrub_on_cancel zeroes a cancelled
             request's residual state
  streams    streaming response multiplexer keyed by request id;
             stream_id_binding namespaces ids per tenant
  allocator  block allocator; allocator_zeroing wipes freed blocks

Worlds (--world):

  vulnerable   every defense off — shared state leaks across tenants.
  defended     every defense on, plus strict session resolution (exact id
               + owning-tenant match; part of the defended posture, not a
               separate knob). Must match the isolated control.
  isolated     NEGATIVE CONTROL, always run alongside: physically
               isolated per-tenant stacks — no sharing exists, so every
               metric must sit at baseline (canary recovery 0, membership
               AUC ~ 0.5, misattribution 0, 0 failures per million).

ARMS (--arms; default all seven)
================================
  session-collision        malformed / reused / truncated / extended
                           session ids adopt a victim session (and its
                           cached context) when the table is not bound
  continuous-batch-mixup   synchronized same-shape requests alias the
                           shared (row, sequence) slots, so the attacker
                           reads the victim's just-written row content
  prefix-cache-isolation   victim populates the prefix cache; the
                           attacker probes member / nonmember candidates —
                           cross-tenant hits leak membership (timing
                           scores -> AUC + TPR at low FPR) AND content
  kv-content-canary        victim canaries written to KV rows, then
                           direct attacker row probes over guessable ids
  error-path-bleed         forced cancel / timeout mid-decode leaves
                           partial victim content in the response buffer;
                           the attacker's retry reads the residue
  response-misattribution  recycled request ids on the streaming mux
                           deliver the victim's chunks to the attacker
  allocator-reuse          freed victim blocks are handed to the
                           attacker with their content intact (the
                           simulated analogue of gpu-memory's residue)

METRICS (per cell)
==================
  canary_recovery_exact / canary_recovery_fuzzy
      fraction of leak opportunities containing a COMPLETE victim canary
      / mean best token-overlap fraction vs any canary
  cache_membership_auc / cache_tpr_at_1pct_fpr (prefix-cache-isolation)
      Mann-Whitney AUC and TPR at FPR <= 1% over seeded timing scores,
      pure python
  misattribution_rate (response-misattribution)
      fraction of attacker-received chunks whose origin tenant is not
      the attacker (incorrect response/request association)
  isolation_failures_per_million
      cross-tenant leak events per request, extrapolated to 1e6

SERVING JOURNAL SCHEMA (llm_attacker.multi_tenant.v1)
=====================================================
The real-target integration point: point --journal at a serving-gateway
journal. One JSON object per line:

    {"ts": 1712345678.123, "tenant_hint": "t-7", "session_id": "s-9",
     "request_id": "r-42", "phase": "prefill", "cache_hit": false,
     "bytes": 4096}

    ts           float, epoch seconds (required)
    request_id   non-empty str, the multiplex/correlation id (required)
    phase        "prefill" | "decode" | "cancel" | "retry" (required)
    cache_hit    bool (required)
    bytes        int >= 0, request payload bytes (required)
    tenant_hint  str, optional* — tenant identity as observed by the
                 gateway (log/metric label; never a secret)
    session_id   str, optional* — session identifier
    * at least one of tenant_hint / session_id is required; both allowed

Schema drift (bad JSON, missing/extra keys, wrong types, unknown phase)
is a descriptive ValueError naming the file and line — never a silent
skip. Journal mode measures the arms observable from this schema:
session-collision (one session_id under multiple tenant_hints),
prefix-cache-isolation (cache hits on shapes the tenant never sent —
heuristically attributed to another tenant's cache population), and
response-misattribution (one request_id under multiple tenants). The
remaining arms need active probing and run in the toy tier only.

EXPECTED SECURE BEHAVIOR (interpretation)
=========================================
Tenant identity is bound to session, cache entry, batch row, stream, and
response; cache hits never bypass authorization; cancel/retry paths scrub
tenant state; freed memory is zeroed before reuse. The defended world
must match the physically-isolated control within tolerance (here:
exactly — same seeded draws, no leak channel), and the vulnerable world
must show the attack working (otherwise the control is vacuous).

LIMITATIONS / EXCLUDED VARIANTS
===============================
  - The toy tier is a SIMULATOR of serving-stack component semantics, not
    a live server driver; the live integration point is the --journal
    analyzer above (emit the schema from the gateway's access log).
  - Journal cross-tenant hit attribution is a heuristic (first-seen
    shape); a first-request hit can have benign explanations (shared
    system prompt warmed by the operator) and is reported as a signal to
    investigate, not proof of isolation failure.
  - Covert timing/contention channels, QoS/fairness, and rate-limit arms
    are excluded: kv-timing covers the cache timing channel in depth and
    resource-exhaustion (THREAT_MODEL_REVIEW) covers availability.
  - allocator-reuse is the simulated analogue of gpu-memory's hardware
    residue probe, not a replacement for it.
  - MoE router contention is excluded (router-side-channel).

Usage:
    python -m attacker --mode inference --attack multi-tenant-isolation \
        --help
    python -m attacker --mode inference --attack multi-tenant-isolation \
        --toy --quick --output /tmp/mti.json
    python -m attacker --mode inference --attack multi-tenant-isolation \
        --journal serving.jsonl --output mti_journal.json
"""

import argparse
import hashlib
import json
import random
import statistics

from .. import artifacts, threat_model
from .common import add_common_args, journal_error

EXPERIMENT_ID = "multi_tenant_isolation"
DISPLAY = "multi-tenant-isolation"
MODES = ("training", "inference")
REQUIRES_LABELS = False
DESCRIPTION = ("cross-tenant isolation tests for a shared serving stack: "
               "sessions, continuous batching, prefix cache, KV store, "
               "cancel/error paths, streaming mux, and allocator reuse, "
               "against a physically-isolated negative control")

SCHEMA_VERSION = "dtraining.attacker.multi_tenant_isolation.v1"
JOURNAL_SCHEMA = "llm_attacker.multi_tenant.v1"
JOURNAL_PHASES = ("prefill", "decode", "cancel", "retry")
JOURNAL_REQUIRED = ("ts", "request_id", "phase", "cache_hit", "bytes")
JOURNAL_IDENT = ("tenant_hint", "session_id")

DEFENSES = ("cache_tenant_binding", "batch_row_binding", "scrub_on_cancel",
            "stream_id_binding", "allocator_zeroing")
WORLD_FLAGS = {
    "vulnerable": {k: False for k in DEFENSES},
    "defended": {k: True for k in DEFENSES},
    "isolated": {k: True for k in DEFENSES},
}

ARMS = ("session-collision", "continuous-batch-mixup",
        "prefix-cache-isolation", "kv-content-canary", "error-path-bleed",
        "response-misattribution", "allocator-reuse")
JOURNAL_ARMS = ("session-collision", "prefix-cache-isolation",
                "response-misattribution")
PRIMARY_METRIC = {
    "session-collision": "canary_recovery_exact",
    "continuous-batch-mixup": "canary_recovery_exact",
    "prefix-cache-isolation": "cache_membership_auc",
    "kv-content-canary": "canary_recovery_exact",
    "error-path-bleed": "canary_recovery_fuzzy",
    "response-misattribution": "misattribution_rate",
    "allocator-reuse": "canary_recovery_exact"}

VICTIM = "tenant-victim"
ATTACKER = "tenant-attacker"


def make_canary(rng, index):
    """A unique, unambiguous victim canary: 6 tokens (CANARY tag, index,
    four random words). Deterministic under the cell's seeded rng."""
    return ["CANARY", f"{index:04d}"] + [f"{rng.getrandbits(32):08x}"
                                         for _ in range(4)]


def _prefix_hash(tokens):
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()[:16]


def rank_auc(pos, neg):
    """Mann-Whitney AUC = P(pos > neg), ties count 1/2. No distributional
    assumption on the scores; nan when a class is empty."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else (0.5 if p == q else 0.0)
    return wins / (len(pos) * len(neg))


def tpr_at_fpr(pos, neg, max_fpr=0.01):
    """TPR at the loosest threshold whose FPR stays <= max_fpr (predict
    member iff score >= threshold). 0.0 when no threshold keeps FPR low
    enough; nan when a class is empty."""
    if not pos or not neg:
        return float("nan")
    best = 0.0
    for thr in sorted(set(pos) | set(neg), reverse=True):
        if sum(1 for q in neg if q >= thr) / len(neg) > max_fpr:
            continue
        best = max(best, sum(1 for p in pos if p >= thr) / len(pos))
    return best


def canary_scores(observations, canaries):
    """(exact, fuzzy) victim-canary recovery over leak opportunities.
    exact: fraction of observations containing a COMPLETE canary; fuzzy:
    mean best token-overlap fraction vs any canary."""
    if not observations or not canaries:
        return 0.0, 0.0
    csets = [set(c) for c in canaries]
    exact = fuzzy = 0.0
    for ob in observations:
        oset = set(ob)
        exact += 1.0 if any(cs <= oset for cs in csets) else 0.0
        fuzzy += max((len(oset & cs) / len(cs) for cs in csets), default=0.0)
    return exact / len(observations), fuzzy / len(observations)


def _leak_events(observations, canaries):
    """Observations containing at least one victim canary token."""
    toks = {t for can in canaries for t in can}
    return sum(1 for ob in observations if set(ob) & toks)


def _fpm(failures, n_requests):
    """Isolation failures per million requests (extrapolated rate)."""
    return {"n_requests": n_requests,
            "isolation_failures": failures,
            "isolation_failures_per_million":
                failures / max(1, n_requests) * 1e6}


class ToyServingStack:
    """Deterministic, seeded simulation of the serving components under
    test (module docstring): session table, prefix cache, continuous
    batcher / KV rows, response buffers, streaming mux, allocator.

    world="vulnerable": every defense off — shared state leaks across
    tenants. world="defended": every defense on plus strict session
    resolution. world="isolated": the negative control (physically
    isolated per-tenant stacks — modeled as the defended posture with no
    shared infrastructure at all; metrics must sit at baseline)."""

    def __init__(self, world="vulnerable", seed=0):
        if world not in WORLD_FLAGS:
            raise ValueError(f"unknown world {world!r} "
                             f"(worlds: {sorted(WORLD_FLAGS)})")
        self.world = world
        self.defenses = dict(WORLD_FLAGS[world])
        self.session_strict = world != "vulnerable"
        self.sessions = {}
        self._last_session = None
        self.cache = {}
        self.rows = {}
        self.buffers = {}
        self.streams = {}
        self._blocks = []
        self._free = []

    def open_session(self, session_id, tenant, context):
        key = (tenant, session_id) if self.session_strict else session_id
        self.sessions[key] = {"owner": tenant, "context": list(context)}
        self._last_session = key

    def resolve_session(self, session_id, tenant):
        """Resolve a presented session id to a session. Vulnerable: the
        table is keyed by session_id alone and resolves by prefix match,
        so a reused / truncated / extended victim id ADOPTS the victim
        session (and "" adopts the most recent one). Defended/isolated:
        exact id + owning-tenant match, else a fresh own-tenant session —
        a foreign id never adopts tenant state."""
        if self.session_strict:
            key = (tenant, session_id)
            if key not in self.sessions:
                self.open_session(session_id, tenant, [])
            return self.sessions[key]
        if session_id in self.sessions:
            return self.sessions[session_id]
        if not self.sessions:
            self.open_session(session_id, tenant, [])
            return self.sessions[session_id]
        if session_id == "":
            return self.sessions[self._last_session]
        for key in self.sessions:
            if key.startswith(session_id) or session_id.startswith(key):
                return self.sessions[key]
        self.open_session(session_id, tenant, [])
        return self.sessions[session_id]

    def cache_put(self, tenant, prefix_tokens, content):
        key = self._cache_key(tenant, prefix_tokens)
        self.cache[key] = {"owner": tenant, "content": list(content)}

    def cache_lookup(self, tenant, prefix_tokens):
        """-> (hit, entry|None). Vulnerable: keyed by the prefix hash
        alone, so a hit can return ANOTHER tenant's entry — the
        authorization bypass this arm measures. Defended/isolated:
        (tenant, hash); hits never cross tenants."""
        entry = self.cache.get(self._cache_key(tenant, prefix_tokens))
        return (entry is not None), entry

    def _cache_key(self, tenant, prefix_tokens):
        h = _prefix_hash(prefix_tokens)
        return (tenant, h) if self.defenses["cache_tenant_binding"] else h

    def _row_key(self, tenant, shape):
        return (tenant, shape) if self.defenses["batch_row_binding"] \
            else shape

    def batch_step(self, tenant, shape, content):
        """One batcher step: the request is routed to the (row, sequence)
        slot for its shape and reads the slot's CURRENT content before
        its own write lands (prefill reuses resident KV state).
        Vulnerable: slots are shared — a synchronized same-shape request
        from another tenant reads the previous tenant's content.
        Defended/isolated: slots are bound per tenant."""
        key = self._row_key(tenant, shape)
        served = list(self.rows.get(key, {}).get("content", []))
        self.rows[key] = {"owner": tenant, "content": list(content)}
        return served

    def kv_write(self, tenant, row, content):
        self.rows[self._row_key(tenant, row)] = \
            {"owner": tenant, "content": list(content)}

    def kv_read(self, tenant, row):
        """Direct row probe. Vulnerable: rows are addressable by id alone.
        Defended/isolated: rows are bound to the owning tenant — a
        foreign row id resolves to the attacker's own (empty) row."""
        entry = self.rows.get(self._row_key(tenant, row))
        return list(entry["content"]) if entry else []

    def cancel_decode(self, tenant, buffer_id, partial_content):
        """A cancelled/timed-out decode. scrub_on_cancel zeroes the
        response buffer; otherwise the partial victim content remains as
        residual state for the next request landing on the buffer."""
        self.buffers[buffer_id] = ([] if self.defenses["scrub_on_cancel"]
                                   else list(partial_content))

    def read_buffer(self, tenant, buffer_id):
        return list(self.buffers.get(buffer_id, []))

    def stream_publish(self, tenant, request_id, tokens):
        key = (tenant, request_id) if self.defenses["stream_id_binding"] \
            else request_id
        entry = self.streams.setdefault(key, {"owner": tenant, "chunks": []})
        entry["chunks"].extend((tenant, t) for t in tokens)

    def stream_consume(self, tenant, request_id):
        """-> [(origin_tenant, token), ...]. Vulnerable: the mux is keyed
        by request_id alone — consuming a recycled/foreign id delivers
        the ORIGINAL owner's chunks. Defended/isolated: (tenant,
        request_id) — a foreign id is just the attacker's empty stream."""
        key = (tenant, request_id) if self.defenses["stream_id_binding"] \
            else request_id
        return list(self.streams.get(key, {"chunks": []})["chunks"])

    def alloc_block(self, tenant):
        """Hand out a block. Freed blocks are reused LIFO; with
        allocator_zeroing a freed block is wiped, otherwise its content
        persists as residue for the next tenant."""
        if self._free:
            return self._free.pop()
        self._blocks.append([])
        return len(self._blocks) - 1

    def write_block(self, idx, tokens):
        self._blocks[idx] = list(tokens)

    def free_block(self, idx):
        if self.defenses["allocator_zeroing"]:
            self._blocks[idx] = []
        self._free.append(idx)

    def read_block(self, idx):
        return list(self._blocks[idx])


def _cell_session_collision(stack, rng, n):
    """Victim opens a session holding canary context; the attacker
    presents malformed/reused/truncated/extended variants of its id."""
    n_ctx = max(2, n // 8)
    canaries = [make_canary(rng, i) for i in range(n_ctx)]
    ctx = [t for can in canaries for t in can]
    sid = f"s-{rng.getrandbits(48):012x}"
    stack.open_session(sid, VICTIM, ctx)
    variants = [sid, sid[:max(4, len(sid) // 2)], sid[:8], "", sid + "-x"]
    obs = [stack.resolve_session(v, ATTACKER)["context"] for v in variants]
    exact, fuzzy = canary_scores(obs, canaries)
    n_requests = n_ctx + len(variants)
    return {"n_session_variants": len(variants),
            "canary_recovery_exact": exact,
            "canary_recovery_fuzzy": fuzzy,
            **_fpm(_leak_events(obs, canaries), n_requests)}


def _cell_batch_mixup(stack, rng, n):
    """Synchronized same-shape victim/attacker requests alias the shared
    (row, sequence) slots; the attacker reads the victim's row content."""
    pairs = max(2, n // 4)
    canaries = [make_canary(rng, i) for i in range(pairs)]
    obs = []
    for i in range(pairs):
        sync = rng.random() < 0.8
        shape = ("seq", rng.choice((128, 256)))
        a_shape = shape if sync else ("seq", 1024)
        stack.batch_step(VICTIM, shape, canaries[i])
        obs.append(stack.batch_step(ATTACKER, a_shape,
                                    [f"attacker-probe-{i}"]))
    exact, fuzzy = canary_scores(obs, canaries)
    return {"n_synchronized_pairs": pairs,
            "canary_recovery_exact": exact,
            "canary_recovery_fuzzy": fuzzy,
            **_fpm(_leak_events(obs, canaries), 2 * pairs)}


def _cell_prefix_cache(stack, rng, n):
    """Victim populates the prefix cache; the attacker probes member and
    nonmember candidates. Cross-tenant hits leak membership (timing
    scores -> AUC / TPR@1%FPR) AND the cached canary content."""
    members = max(8, n // 6)
    nonmembers = members * 3
    canaries, member_prefixes = [], []
    for i in range(members):
        toks = [f"sys-{rng.getrandbits(32):08x}", "shared-boilerplate",
                f"u-{i}"]
        can = make_canary(rng, i)
        canaries.append(can)
        member_prefixes.append(toks)
        stack.cache_put(VICTIM, toks, can)
    probes = [(p, True) for p in member_prefixes]
    probes += [([f"sys-{rng.getrandbits(32):08x}", "candidate",
                 f"c-{j}"], False) for j in range(nonmembers)]
    scores, labels, obs = [], [], []
    cross_hits = 0
    for toks, is_member in probes:
        hit, entry = stack.cache_lookup(ATTACKER, toks)
        lat = rng.gauss(8.0, 1.5) if hit else rng.gauss(40.0, 3.0)
        scores.append(-lat)
        labels.append(is_member)
        obs.append(entry["content"] if hit else [])
        if hit and entry["owner"] != ATTACKER:
            cross_hits += 1
    pos = [s for s, m in zip(scores, labels) if m]
    neg = [s for s, m in zip(scores, labels) if not m]
    exact, fuzzy = canary_scores(obs, canaries)
    return {"n_probes": len(probes), "n_member_probes": members,
            "n_nonmember_probes": nonmembers,
            "cross_tenant_hits": cross_hits,
            "cache_membership_auc": rank_auc(pos, neg),
            "cache_tpr_at_1pct_fpr": tpr_at_fpr(pos, neg, max_fpr=0.01),
            "canary_recovery_exact": exact,
            "canary_recovery_fuzzy": fuzzy,
            **_fpm(cross_hits, members + len(probes))}


def _cell_kv_canary(stack, rng, n):
    """Victim canaries in KV rows, then attacker row probes over the
    guessable (sequential) row id space."""
    k = max(2, n // 8)
    canaries = [make_canary(rng, i) for i in range(k)]
    for i, can in enumerate(canaries):
        stack.kv_write(VICTIM, i, can)
    obs = [stack.kv_read(ATTACKER, i) for i in range(k + 2)]
    exact, fuzzy = canary_scores(obs, canaries)
    return {"n_rows_written": k, "n_rows_probed": k + 2,
            "canary_recovery_exact": exact,
            "canary_recovery_fuzzy": fuzzy,
            **_fpm(_leak_events(obs, canaries), k + k + 2)}


def _cell_error_path_bleed(stack, rng, n):
    """Forced cancel/timeout mid-decode leaves partial victim content in
    the response buffer; the attacker's retry reads the residue."""
    trials = max(2, n // 8)
    canaries, obs = [], []
    for i in range(trials):
        can = make_canary(rng, i)
        canaries.append(can)
        frac = rng.uniform(0.4, 1.0)
        partial = can[:max(1, int(len(can) * frac))]
        stack.cancel_decode(VICTIM, i % 2, partial)
        obs.append(stack.read_buffer(ATTACKER, i % 2))
    exact, fuzzy = canary_scores(obs, canaries)
    return {"n_cancellations": trials,
            "canary_recovery_exact": exact,
            "canary_recovery_fuzzy": fuzzy,
            **_fpm(_leak_events(obs, canaries), 2 * trials)}


def _cell_misattribution(stack, rng, n):
    """Recycled request ids on the streaming mux: the attacker consumes a
    request id the victim is streaming under and receives its chunks."""
    streams = max(2, n // 8)
    canaries = [make_canary(rng, i) for i in range(streams)]
    obs, recv = [], []
    for i in range(streams):
        rid = f"r-{i % max(2, streams // 2)}"
        stack.stream_publish(VICTIM, rid, canaries[i])
        got = stack.stream_consume(ATTACKER, rid)
        recv.extend(got)
        obs.append([t for _, t in got])
    mis = [c for c in recv if c[0] != ATTACKER]
    exact, fuzzy = canary_scores(obs, canaries)
    return {"n_streams": streams,
            "n_chunks_received": len(recv),
            "n_chunks_misattributed": len(mis),
            "misattribution_rate": len(mis) / len(recv) if recv else 0.0,
            "canary_recovery_exact": exact,
            "canary_recovery_fuzzy": fuzzy,
            **_fpm(len(mis), 2 * streams)}


def _cell_allocator_reuse(stack, rng, n):
    """Victim frees blocks holding canaries; the attacker's allocations
    reuse them — residue unless the allocator zeroes on free."""
    k = max(2, n // 8)
    canaries = [make_canary(rng, i) for i in range(k)]
    obs = []
    for i in range(k):
        b = stack.alloc_block(VICTIM)
        stack.write_block(b, canaries[i])
        stack.free_block(b)
        obs.append(stack.read_block(stack.alloc_block(ATTACKER)))
    exact, fuzzy = canary_scores(obs, canaries)
    return {"n_blocks_recycled": k,
            "canary_recovery_exact": exact,
            "canary_recovery_fuzzy": fuzzy,
            **_fpm(_leak_events(obs, canaries), 2 * k)}


ARM_FUNCS = {
    "session-collision": _cell_session_collision,
    "continuous-batch-mixup": _cell_batch_mixup,
    "prefix-cache-isolation": _cell_prefix_cache,
    "kv-content-canary": _cell_kv_canary,
    "error-path-bleed": _cell_error_path_bleed,
    "response-misattribution": _cell_misattribution,
    "allocator-reuse": _cell_allocator_reuse,
}


def run_arm_cell(arm, world, seed, n_requests):
    """One (arm, world, seed) cell. The rng is seeded per (seed, arm) —
    NOT per world — so the defended world and the isolated control draw
    identical noise and must match exactly."""
    if arm not in ARM_FUNCS:
        raise ValueError(f"unknown arm {arm!r} (arms: {sorted(ARM_FUNCS)})")
    rng = random.Random(f"{EXPERIMENT_ID}:{seed}:{arm}")
    stack = ToyServingStack(world)
    rec = ARM_FUNCS[arm](stack, rng, n_requests)
    rec.update({"experiment": EXPERIMENT_ID, "arm": arm, "world": world,
                "seed": seed, "control": world == "isolated"})
    return rec


def _validate_journal_record(rec, path, lineno):
    where = f"{path}:{lineno}"
    if not isinstance(rec, dict):
        raise ValueError(f"{where}: record is not a JSON object; expected "
                         f"schema {JOURNAL_SCHEMA}")
    keys = set(rec)
    missing = [k for k in JOURNAL_REQUIRED if k not in keys]
    extra = sorted(keys - set(JOURNAL_REQUIRED) - set(JOURNAL_IDENT))
    if missing or extra:
        raise ValueError(f"{where}: keys {sorted(keys)}: missing {missing}, "
                         f"extra {extra}; expected schema {JOURNAL_SCHEMA} "
                         f"(required {sorted(JOURNAL_REQUIRED)} + at least "
                         f"one of {list(JOURNAL_IDENT)})")
    if not any(rec.get(k) is not None for k in JOURNAL_IDENT):
        raise ValueError(f"{where}: need at least one of "
                         f"{list(JOURNAL_IDENT)}; schema {JOURNAL_SCHEMA}")
    for k in JOURNAL_IDENT:
        if k in rec and (not isinstance(rec[k], str) or not rec[k]):
            raise ValueError(f"{where}: {k} must be a non-empty str, got "
                             f"{rec[k]!r}; schema {JOURNAL_SCHEMA}")
    ts = rec["ts"]
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        raise ValueError(f"{where}: ts must be a number, got {ts!r}; "
                         f"schema {JOURNAL_SCHEMA}")
    if not isinstance(rec["request_id"], str) or not rec["request_id"]:
        raise ValueError(f"{where}: request_id must be a non-empty str, "
                         f"got {rec['request_id']!r}; schema {JOURNAL_SCHEMA}")
    if rec["phase"] not in JOURNAL_PHASES:
        raise ValueError(f"{where}: phase {rec['phase']!r} not in "
                         f"{JOURNAL_PHASES}; schema {JOURNAL_SCHEMA}")
    if not isinstance(rec["cache_hit"], bool):
        raise ValueError(f"{where}: cache_hit must be a bool, got "
                         f"{rec['cache_hit']!r}; schema {JOURNAL_SCHEMA}")
    nbytes = rec["bytes"]
    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
        raise ValueError(f"{where}: bytes must be an int >= 0, got "
                         f"{nbytes!r}; schema {JOURNAL_SCHEMA}")
    return {"ts": float(ts), "tenant_hint": rec.get("tenant_hint"),
            "session_id": rec.get("session_id"),
            "request_id": rec["request_id"], "phase": rec["phase"],
            "cache_hit": rec["cache_hit"], "bytes": int(nbytes)}


def read_journal(path):
    """Parse a llm_attacker.multi_tenant.v1 serving journal. Schema drift
    is a descriptive ValueError naming the file and line."""
    try:
        f = open(path)
    except OSError as e:
        raise ValueError(f"{path}: cannot read journal "
                         f"({e.strerror or e}); expected schema "
                         f"{JOURNAL_SCHEMA}") from None
    records = []
    with f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: not JSON ({e}); "
                                 f"expected schema {JOURNAL_SCHEMA}") from None
            records.append(_validate_journal_record(rec, path, lineno))
    if not records:
        raise ValueError(f"{path}: no records; expected schema "
                         f"{JOURNAL_SCHEMA}")
    return records


def _journal_session_collision(records, path):
    """One session_id observed under multiple tenant_hints = the session
    table (or the gateway's session routing) mixes tenants."""
    by_session = {}
    for r in records:
        if r["session_id"] is None:
            continue
        by_session.setdefault(r["session_id"], set()).add(
            r["tenant_hint"] or "<unknown>")
    colliding = {s: sorted(t) for s, t in by_session.items() if len(t) > 1}
    rec = {"experiment": EXPERIMENT_ID, "arm": "session-collision",
           "source": "journal", "journal": path, "control": False,
           "n_records": len(records), "n_sessions": len(by_session),
           "n_colliding_sessions": len(colliding),
           "session_collision_rate":
               len(colliding) / max(1, len(by_session)),
           "colliding_sessions": sorted(colliding)[:20],
           **_fpm(len(colliding), len(records))}
    if not by_session:
        rec["note"] = ("journal carries no session_id fields; session "
                       "collisions are not measurable from tenant_hint "
                       "alone")
    return rec


def _journal_prefix_cache(records, path):
    """A cache hit on a byte-shape the tenant never sent is attributed to
    ANOTHER tenant's cache population (heuristic — see docstring)."""
    seen, per_tenant = {}, {}
    hits = cross = 0
    for r in sorted(records, key=lambda r: r["ts"]):
        tenant = r["tenant_hint"] or r["session_id"] or "<unknown>"
        own = seen.setdefault(tenant, set())
        if not r["cache_hit"]:
            own.add(r["bytes"])
            continue
        hits += 1
        if r["bytes"] not in own:
            cross += 1
            per_tenant[tenant] = per_tenant.get(tenant, 0) + 1
        own.add(r["bytes"])
    return {"experiment": EXPERIMENT_ID, "arm": "prefix-cache-isolation",
            "source": "journal", "journal": path, "control": False,
            "n_records": len(records), "n_cache_hits": hits,
            "cross_tenant_attributed_hits": cross,
            "cross_tenant_hit_rate": cross / max(1, hits),
            "per_tenant": per_tenant,
            "note": "hit on a shape the tenant never sent is "
                    "heuristically attributed to another tenant's cache "
                    "population (a signal to investigate, not proof)",
            **_fpm(cross, len(records))}


def _journal_misattribution(records, path):
    """One request_id observed under multiple tenants = the streaming mux
    / correlation layer delivers one tenant's stream to another."""
    by_req = {}
    for r in records:
        by_req.setdefault(r["request_id"], set()).add(
            r["tenant_hint"] or r["session_id"] or "<unknown>")
    shared = {q: sorted(t) for q, t in by_req.items() if len(t) > 1}
    return {"experiment": EXPERIMENT_ID, "arm": "response-misattribution",
            "source": "journal", "journal": path, "control": False,
            "n_records": len(records), "n_request_ids": len(by_req),
            "n_cross_tenant_request_ids": len(shared),
            "misattribution_rate": len(shared) / max(1, len(by_req)),
            "cross_tenant_request_ids": sorted(shared)[:20],
            **_fpm(len(shared), len(records))}


JOURNAL_FUNCS = {
    "session-collision": _journal_session_collision,
    "prefix-cache-isolation": _journal_prefix_cache,
    "response-misattribution": _journal_misattribution,
}


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS),
                    help="attack arm(s) to run (default: all seven)")
    ap.add_argument("--world", choices=("vulnerable", "defended"),
                    default=None,
                    help="toy: which posture world to run (default: both); "
                         "the physically-isolated negative control ALWAYS "
                         "runs alongside")
    ap.add_argument("--requests", type=int, default=96,
                    help="toy: per-arm request budget (scale knob)")
    ap.add_argument("--journal", default=None,
                    help=f"serving journal to analyze ({JOURNAL_SCHEMA}); "
                         f"measurable arms: {', '.join(JOURNAL_ARMS)}")
    ap.add_argument("--evidence", action="store_true",
                    help="wrap the run in an EvidenceLogger (requires "
                         "--output): the artifact write is recorded as a "
                         "chained event and the evidence manifest is "
                         "finalized alongside")
    return ap


def _threat_model_doc():
    return threat_model.validate(threat_model.skeleton(
        "active low-privilege authenticated tenant on a shared "
        "multi-tenant LLM serving stack probing session, batch, cache, "
        "stream, error-path, and allocator isolation against a "
        "cooperating victim fixture",
        scope={"use_case": "split-inference", "deployment": "multi-tenant",
               "in_scope": ["shared serving stack components: session "
                            "table, continuous batcher, prefix cache, KV "
                            "store, response buffers, streaming mux, "
                            "allocator",
                            "serving journals emitted by the gateway"],
               "out_of_scope": ["covert timing/contention channels "
                                "(kv-timing)",
                                "availability/fairness arms "
                                "(resource-exhaustion)",
                                "hardware memory residue (gpu-memory)",
                                "MoE router contention "
                                "(router-side-channel)"]},
        actor={"role": "co-tenant",
               "vantage": ["API client", "co-tenant"],
               "intent": "active",
               "authorization": "authorized-red-team",
               "privileges": ["low-privilege authenticated tenant"],
               "knowledge": ["client-visible request/session id schemes",
                             "public serving-stack architecture"],
               "capabilities": ["send chosen requests concurrently with "
                                "the victim fixture",
                                "choose session/request identifiers "
                                "within the client API",
                                "force cancel/timeout/retry of its own "
                                "requests"],
               "resources": ["one tenant API key",
                             "a normal request-rate budget"],
               "persistence": "session",
               "collusion": ["cooperating victim fixture (the evaluation "
                             "harness)"]},
        target={"assets": ["victim prompts and responses",
                           "KV/prefix-cache contents",
                           "session state and cached context",
                           "residual decode buffers and freed blocks"],
                "components": ["session table", "continuous batcher",
                               "prefix cache", "KV-content store",
                               "cancel/error paths", "streaming response "
                               "multiplexer", "block allocator"],
                "data_stores": ["prefix cache", "KV store",
                                "response buffers", "allocator free list"],
                "data_flows": ["request -> session -> batch row -> "
                               "decode -> stream"],
                "trust_boundaries": ["tenant boundary inside the shared "
                                     "serving stack"],
                "entry_points": ["inference / fine-tuning API"],
                "exit_points": ["API responses and streams",
                                "cache-hit observable behavior"]},
        scenario={"preconditions": ["multi-tenant serving stack shared "
                                    "by attacker and victim",
                                    "attacker holds a valid low-"
                                    "privilege tenant credential"],
                  "lifecycle_stage": ["inference", "operations"],
                  "security_properties": ["confidentiality",
                                          "authorization"],
                  "attack_steps": ["plant victim canaries through the "
                                   "cooperating fixture",
                                   "probe shared components with chosen "
                                   "requests / ids / shapes",
                                   "read victim state out of responses, "
                                   "cache behavior, buffers, or blocks"],
                  "technical_impact": ["cross-tenant information "
                                       "disclosure",
                                       "cache hits bypassing "
                                       "authorization",
                                       "responses delivered to the "
                                       "wrong tenant"],
                  "business_impact": ["tenant data breach on shared "
                                      "infrastructure"]},
        controls={"assumed": ["tenants share batcher/cache/allocator "
                              "infrastructure"],
                  "tested": ["cache tenant binding", "batch row binding",
                             "scrub on cancel", "stream id binding",
                             "allocator zeroing",
                             "strict session resolution"],
                  "status": "open",
                  "residual_risk": "channels not simulated here (covert "
                                   "timing, hardware residue, router "
                                   "metadata) are covered by sibling "
                                   "attacks"},
        mappings={"owasp_llm_2026": ["LLM02", "LLM06", "LLM08", "LLM09"],
                  "cwe": ["CWE-668", "CWE-200"]},
        evaluation={"success_metrics": ["canary recovery exact/fuzzy",
                                        "cache-membership AUC and TPR at "
                                        "1% FPR",
                                        "incorrect response/request "
                                        "association rate",
                                        "isolation failures per million "
                                        "requests"],
                    "baseline": "physically-isolated per-tenant control "
                                "world (canary recovery 0, AUC ~ 0.5, "
                                "misattribution 0)",
                    "budget": {"requests_per_arm": "config.requests",
                               "administrative_access": "none",
                               "victim_access": "cooperating fixture "
                                                "only"},
                    "limitations": ["simulated stack; live integration "
                                    "via the serving journal analyzer",
                                    "journal hit attribution is "
                                    "heuristic"],
                    "coverage_status": "covered"}))


def _make_artifact(args, source, extra_config=None, evidence=None):
    config = {"attack": EXPERIMENT_ID, "mode": getattr(args, "mode", None),
              "source": source, "arms": args.arms,
              "requests": args.requests, "seed": args.seed,
              "journal_schema": JOURNAL_SCHEMA,
              "defenses": {w: dict(f) for w, f in WORLD_FLAGS.items()},
              "control_world": "isolated"}
    if extra_config:
        config.update(extra_config)
    return artifacts.make_artifact(
        SCHEMA_VERSION, config, _threat_model_doc(),
        interpretation="tenant identity must be bound to session, cache "
                       "entry, batch row, stream, and response; cache "
                       "hits must never bypass authorization; cancel/"
                       "retry paths must scrub tenant state. The defended "
                       "world must match the physically-isolated control "
                       "(canary recovery 0, AUC ~ 0.5, misattribution 0, "
                       "0 failures per million); the vulnerable world "
                       "must show the attack working, otherwise the "
                       "control is vacuous.",
        evidence=evidence)


def _progress(rec):
    parts = []
    if "canary_recovery_exact" in rec:
        parts.append(f"canary={rec['canary_recovery_exact']:.3f}/"
                     f"{rec['canary_recovery_fuzzy']:.3f}")
    if "cache_membership_auc" in rec:
        parts.append(f"AUC={rec['cache_membership_auc']:.4f} "
                     f"TPR@1%FPR={rec['cache_tpr_at_1pct_fpr']:.3f}")
    if "misattribution_rate" in rec:
        parts.append(f"misattr={rec['misattribution_rate']:.3f}")
    parts.append(f"fpm={rec['isolation_failures_per_million']:.0f}")
    return (f"[{DISPLAY}] {rec.get('world', 'journal')}/{rec['arm']} "
            f"seed={rec.get('seed', '-')}: " + " ".join(parts))


def run_toy(args, evidence=None):
    if args.quick:
        args.seeds = args.seeds[:1]
        args.requests = min(args.requests, 32)
    worlds = [args.world] if args.world else ["vulnerable", "defended"]
    worlds.append("isolated")
    out = _make_artifact(args, "toy",
                         {"toy": True, "worlds": worlds,
                          "seeds": args.seeds}, evidence=evidence)
    per = {}
    for seed in args.seeds:
        for world in worlds:
            for arm in args.arms:
                try:
                    rec = run_arm_cell(arm, world, seed, args.requests)
                except Exception as e:
                    journal_error(args.output, EXPERIMENT_ID,
                                  {"arm": arm, "world": world,
                                   "seed": seed}, e)
                    continue
                artifacts.append_jsonl(args.output, rec)
                out["results"].append(rec)
                per.setdefault((arm, world), []).append(rec)
                print(_progress(rec))
    means = {}
    for (arm, world), recs in sorted(per.items()):
        row = {"experiment": EXPERIMENT_ID, "arm": arm, "world": world,
               "control": world == "isolated", "n_seeds": len(recs)}
        for key in ("canary_recovery_exact", "canary_recovery_fuzzy",
                    "cache_membership_auc", "cache_tpr_at_1pct_fpr",
                    "misattribution_rate",
                    "isolation_failures_per_million"):
            vals = [r[key] for r in recs
                    if key in r and r[key] == r[key]]
            if vals:
                row[key + "_mean"] = statistics.fmean(vals)
                means[(arm, world, key)] = row[key + "_mean"]
        out["summary"].append(row)
    for arm in args.arms:
        key = PRIMARY_METRIC[arm]
        if (arm, "isolated", key) not in means:
            continue
        row = {"experiment": EXPERIMENT_ID, "arm": arm,
               "check": "negative-control", "metric": key,
               "isolated_control": means[(arm, "isolated", key)]}
        if (arm, "vulnerable", key) in means:
            row["vulnerable"] = means[(arm, "vulnerable", key)]
            row["attack_signal"] = row["vulnerable"] - \
                row["isolated_control"]
        if (arm, "defended", key) in means:
            row["defended"] = means[(arm, "defended", key)]
            row["defended_minus_control"] = row["defended"] - \
                row["isolated_control"]
        out["summary"].append(row)
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def run_journal(args, evidence=None):
    out = _make_artifact(args, "journal", {"journal": args.journal},
                         evidence=evidence)
    try:
        records = read_journal(args.journal)
    except ValueError as e:
        raise SystemExit(f"[{DISPLAY}] {e}") from None
    for arm in args.arms:
        if arm not in JOURNAL_ARMS:
            print(f"[{DISPLAY}] {arm}: not measurable from a "
                  f"{JOURNAL_SCHEMA} journal (needs active probing / the "
                  f"toy stack); skipped")
            continue
        try:
            rec = JOURNAL_FUNCS[arm](records, args.journal)
        except Exception as e:
            journal_error(args.output, EXPERIMENT_ID, {"arm": arm}, e)
            continue
        artifacts.append_jsonl(args.output, rec)
        out["results"].append(rec)
        print(_progress(rec))
    per_arm = {}
    for rec in out["results"]:
        per_arm.setdefault(rec["arm"], []).append(rec)
    for arm, recs in sorted(per_arm.items()):
        out["summary"].append(
            {"experiment": EXPERIMENT_ID, "arm": arm, "source": "journal",
             "n_records": sum(r["n_records"] for r in recs),
             "isolation_failures":
                 sum(r["isolation_failures"] for r in recs)})
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def _run(args, evidence=None):
    if args.journal:
        return run_journal(args, evidence=evidence)
    if args.toy:
        return run_toy(args, evidence=evidence)
    raise SystemExit(f"[{DISPLAY}] pick an input path: --toy (simulated "
                     f"stack with ground truth and a physically-isolated "
                     f"negative control) or --journal FILE (real serving "
                     f"journal, schema {JOURNAL_SCHEMA}).")


def run(args):
    if getattr(args, "evidence", False):
        if not args.output:
            raise SystemExit(f"[{DISPLAY}] --evidence requires --output "
                             "(the evidence manifest is written next to "
                             "the artifact)")
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

    print("multi-tenant isolation (pure python):")
    check("world flags: vulnerable all off / defended+isolated all on",
          not any(WORLD_FLAGS["vulnerable"].values())
          and all(WORLD_FLAGS["defended"].values())
          and all(WORLD_FLAGS["isolated"].values()))

    can = make_canary(random.Random(7), 3)
    check("canary: 6 tokens, CANARY-tagged, index-encoded",
          len(can) == 6 and can[0] == "CANARY" and can[1] == "0003")
    check("rank_auc anchors (separated / reversed / tied)",
          rank_auc([3.0, 4.0], [1.0, 2.0]) == 1.0
          and rank_auc([1.0, 2.0], [3.0, 4.0]) == 0.0
          and rank_auc([1.0, 1.0], [1.0, 1.0]) == 0.5)
    check("tpr_at_fpr anchors (separated / indistinguishable)",
          tpr_at_fpr([3.0, 4.0], [1.0, 2.0], 0.01) == 1.0
          and tpr_at_fpr([1.0, 1.0], [1.0, 1.0], 0.01) == 0.0)
    check("canary_scores anchors (exact vs partial overlap)",
          canary_scores([["CANARY", "0000", "a", "b", "c", "d"], ["x"]],
                        [["CANARY", "0000", "a", "b", "c", "d"]])
          == (0.5, 0.5))

    v = ToyServingStack("vulnerable")
    v.open_session("s-abc", VICTIM, ["CANARY", "00"])
    check("vulnerable: a truncated foreign session id adopts the victim "
          "session",
          v.resolve_session("s-ab", ATTACKER)["owner"] == VICTIM)
    d = ToyServingStack("defended")
    d.open_session("s-abc", VICTIM, ["CANARY", "00"])
    fresh = d.resolve_session("s-abc", ATTACKER)
    check("defended: a foreign session id never adopts tenant state",
          fresh["owner"] == ATTACKER and fresh["context"] == [])
    v.cache_put(VICTIM, ["p"], ["CANARY"])
    hit, entry = v.cache_lookup(ATTACKER, ["p"])
    check("vulnerable: cache hit crosses the tenant boundary",
          hit and entry["owner"] == VICTIM)
    d.cache_put(VICTIM, ["p"], ["CANARY"])
    check("defended: cache hits never bypass authorization",
          not d.cache_lookup(ATTACKER, ["p"])[0])

    for arm in ARMS:
        rv = run_arm_cell(arm, "vulnerable", 0, 48)
        check(f"{arm}: vulnerable world leaks (failures > 0)",
              rv["isolation_failures"] > 0)
        rc = run_arm_cell(arm, "isolated", 0, 48)
        check(f"{arm}: isolated control at baseline (0 failures)",
              rc["isolation_failures"] == 0
              and rc["canary_recovery_exact"] == 0.0)
    rd = run_arm_cell("prefix-cache-isolation", "defended", 0, 48)
    check("defended cache arm: membership channel closed "
          "(TPR@1%FPR = 0)", rd["cache_tpr_at_1pct_fpr"] == 0.0)

    import os
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        jf = os.path.join(td, "j.jsonl")
        with open(jf, "w") as f:
            f.write(json.dumps({"ts": 1.0, "tenant_hint": "a",
                                "request_id": "r1", "phase": "prefill",
                                "cache_hit": False, "bytes": 8}) + "\n")
            f.write(json.dumps({"ts": 2.0, "tenant_hint": "a",
                                "request_id": "r2", "phase": "sideways",
                                "cache_hit": True, "bytes": 8}) + "\n")
        try:
            read_journal(jf)
            check("journal schema drift raises", False)
        except ValueError as e:
            check("journal schema drift is a descriptive ValueError "
                  "naming file+line", f"{jf}:2" in str(e)
                  and "sideways" in str(e))
        try:
            read_journal(os.path.join(td, "missing.jsonl"))
            check("missing journal raises", False)
        except ValueError as e:
            check("missing journal is a descriptive ValueError naming "
                  "the file", "missing.jsonl" in str(e))
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
