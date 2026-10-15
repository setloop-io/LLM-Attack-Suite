#!/usr/bin/env python3
"""artifact-telemetry-leak — canary sweep of persistent and operational
data stores (THREAT_MODEL_REVIEW P0). Seam defenses (key rotation,
encryption, minimal disclosure) protect tensors in flight, but prompts,
labels, credentials, tenant IDs, and key material still land in captures,
logs, checkpoints, artifacts/sidecars, crash dumps, tmp files, and
backups. This attack plants unique synthetic canaries, exercises
successful/failed/cancelled/crashed workflows, and scans evaluator-
authorized store roots for exact, base64, hex, truncated, and token-split
canary variants — plus file-permission (world/group-readable) and
retention (past the configured deletion deadline) audits. Label-free;
applies to training and inference operations alike. Passive, files-only
vantage: the scanner reads ONLY inside the explicitly passed roots
(SAFETY-CRITICAL) and never copies file content into the report — hits
carry sha256 + byte offsets + permission/retention metadata only.

CANARY SPEC FILE (llm_attacker.canary_specs.v1)
===============================================
--canary-file accepts {"schema": ..., "canaries": [...]} or a bare JSON
list; each entry is {"value": "<secret string>", "id": "<optional>",
"data_class": "<optional>"} — value is a required non-empty str, id
defaults to "canary-<i>", data_class to "unclassified". Malformed files
are a descriptive ValueError naming the file and entry — never a silent
skip.

Variants searched per canary (generated values look like
"CANARY-<class>-<hexid>"):

    exact        the literal string (logs, canonical-label files, dumps)
    base64/hex   encoded forms (evidence sidecars, serialized state)
    truncated    "CANARY-<class>" — the id field truncated away, as in
                 fixed-width metrics labels
    token_split  the "-"-separated tokens in order on one line
                 ("tenant CANARY tenant_ids ab12cd34 revoked"); lines
                 containing the exact form are excluded to avoid
                 double-counting

TOY WORLDS (make_fixture_store(root, seed, redacted))
=====================================================
toy-leaking   a realistic store tree (captures/ logs/ checkpoints/
              artifacts/ crashes/ tmp/ backups/) written by four toy
              workflows (success, failed, cancelled, crashed) with
              canaries planted: an exact prompt canary in the failed-
              workflow log line, an exact label canary in the canonical-
              label file (world-readable, 0o444) AND in a 72h-old backup
              (two copies), a base64 credential canary in an evidence
              sidecar, a truncated tenant-ID canary in a metrics label,
              a token-split tenant-ID canary in the cancelled-workflow
              note, and exact key material in the crash dump.
toy-redacted  NEGATIVE CONTROL: the same workflows routed through
              redact_text (every variant of every canary removed),
              owner-only permissions (0o600), and the expired backup
              deleted per the retention policy. Expected: zero hits,
              zero flagged permissions, zero retention violations —
              redaction precision/recall 1.0 on the fixture pair. The
              leaking world MUST show hits, else the control is vacuous.

Arms: store-scan (one journaled cell per scanned store root) and
redaction-check (the toy fixture pair: leaking found vs redacted clean;
journaled as a result cell).

Usage:
    python -m attacker --mode inference --attack artifact-telemetry-leak \
        --help
    python -m attacker --mode training --attack artifact-telemetry-leak \
        --toy --quick --output /tmp/atl.json
    python -m attacker --mode inference --attack artifact-telemetry-leak \
        --roots /srv/split/captures /var/log/split-llm \
        --canary-file canaries.json --deletion-deadline 48 \
        --output atl.json

INTEGRATION POINT: there is deliberately no live-service driver — the
real-target path is batch scanning of evaluator-authorized --roots
(refuses "/" and $HOME itself with a descriptive SystemExit) with the
canaries the evaluator planted supplied via --canary-file. Mount object
stores / pull log-aggregator exports into a local root first.

Metrics (per store cell): canary exposure rate by data class
(exposure_by_class), unauthorized readability count (world- or group-
readable files), retention-beyond-deadline count, copies per canary, and
the full hit list (store, data class, relative access path, variants,
offsets, sha256, mode, retention hint). The redaction-check cell reports
redaction precision (planted occurrences / redactor actions: replacements
+ deletions) and recall (1 - surviving planted variants / planted
variants) on the fixture pair.

EXPECTED SECURE BEHAVIOR: no canary variant in any store; no world- or
group-readable store files; deletion deadline respected; redaction
recall = 1.0 on the fixture pair.

LIMITATIONS / EXCLUDED VARIANTS:
- raw-byte substring matching only: compressed (gzip/zip), encrypted, or
  serialized-tensor containers (.pt/.safetensors) are scanned as bytes,
  never unpacked or decoded — a canary inside compression is invisible
  to this tier;
- files > 64 MiB get the permission/retention audit but no content scan
  (listed under skipped_large_files); non-regular files (FIFOs, sockets,
  device nodes) are never opened (skipped_special_files); symlinks are
  not followed outside the passed roots;
- token-split matching is single-line and order-preserving;
- object stores are covered only via mounted filesystems — no S3/bucket
  API metadata listing, no log-aggregator APIs;
- the redacted toy world models the defended deployment as a bundle
  (redaction + owner-only permissions + deletion respected), not each
  defense in isolation.
"""

import argparse
import base64
import hashlib
import json
import os
import random
import stat
import tempfile
import time

from .. import artifacts, threat_model
from .common import add_common_args, journal_error

EXPERIMENT_ID = "artifact_telemetry_leak"
MODES = ("training", "inference")
REQUIRES_LABELS = False
DESCRIPTION = ("canary sweep of persistent/operational stores (captures, "
               "logs, checkpoints, artifacts, crashes, tmp, backups): "
               "finds exact/base64/hex/truncated/token-split leaks of "
               "planted prompt, label, credential, tenant-ID, and key "
               "canaries, plus world/group-readable permission and "
               "retention-past-deadline audits")

SCHEMA_VERSION = "llm_attacker.artifact_telemetry_leak.v1"
CANARY_FILE_SCHEMA = "llm_attacker.canary_specs.v1"
DATA_CLASSES = ("prompts", "labels", "credentials", "tenant_ids",
                "key_material")
STORES = ("captures", "logs", "checkpoints", "artifacts", "crashes",
          "tmp", "backups")
ARMS = ("store-scan", "redaction-check")
REDACTION_MARKER = "[REDACTED]"
TOY_NOW = 1_700_000_000.0
MAX_OFFSETS = 16
MAX_FILE_BYTES = 64 * 1024 * 1024


def make_canaries(seed, classes=DATA_CLASSES):
    """One synthetic canary per data class: {"id", "data_class", "value"}
    with value "CANARY-<class>-<8 hex chars>". Deterministic in `seed`."""
    rng = random.Random(seed)
    out = []
    for cls in classes:
        hexid = "".join(rng.choice("0123456789abcdef") for _ in range(8))
        out.append({"id": f"canary-{cls}", "data_class": cls,
                    "value": f"CANARY-{cls}-{hexid}"})
    return out


def canary_variants(canary):
    """The searchable forms of one canary (see the module docstring)."""
    raw = canary["value"].encode()
    tokens = canary["value"].split("-")
    return {"exact": canary["value"],
            "base64": base64.b64encode(raw).decode(),
            "hex": raw.hex(),
            "truncated": "-".join(tokens[:2]),
            "tokens": tokens}


def redact_text(text, canaries):
    """Remove every variant of every canary from `text` (longest forms
    first so a truncated prefix never clobbers an exact match; the final
    token — the hex id — is removed to kill token-split survivals).
    Returns (redacted_text, n_substitutions)."""
    forms = []
    for c in canaries:
        v = canary_variants(c)
        forms += [v["exact"], v["base64"], v["hex"], v["truncated"],
                  v["tokens"][-1]]
    n_subs = 0
    for form in sorted(set(forms), key=len, reverse=True):
        if not form:
            continue
        k = text.count(form)
        if k:
            text = text.replace(form, REDACTION_MARKER)
            n_subs += k
    return text, n_subs


def load_canaries(path):
    """Load real canary specs from --canary-file
    (schema llm_attacker.canary_specs.v1 or a bare list). Descriptive
    ValueError naming the file on any drift."""
    try:
        with open(path) as f:
            text = f.read()
    except OSError as e:
        raise ValueError(f"canary file {path!r}: cannot read: {e}")
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"canary file {path!r}: invalid JSON: {e}")
    if isinstance(doc, dict):
        specs = doc.get("canaries")
    elif isinstance(doc, list):
        specs = doc
    else:
        specs = None
    if not isinstance(specs, list) or not specs:
        raise ValueError(f"canary file {path!r}: expected a non-empty list "
                         f"or an object with a 'canaries' list "
                         f"(schema {CANARY_FILE_SCHEMA})")
    out = []
    for i, spec in enumerate(specs):
        where = f"{path!r} entry {i}"
        if not isinstance(spec, dict):
            raise ValueError(f"canary file {where}: must be an object with "
                             f"a 'value' string")
        unknown = sorted(set(spec) - {"id", "data_class", "value"})
        if unknown:
            raise ValueError(f"canary file {where}: unknown keys {unknown}")
        value = spec.get("value")
        if not isinstance(value, str) or not value:
            raise ValueError(f"canary file {where}: 'value' must be a "
                             f"non-empty str")
        for opt in ("id", "data_class"):
            if opt in spec and not isinstance(spec[opt], str):
                raise ValueError(f"canary file {where}: {opt!r} must be a str")
        out.append({"id": spec.get("id") or f"canary-{i}",
                    "data_class": spec.get("data_class") or "unclassified",
                    "value": value})
    return out


def make_fixture_store(root, seed, redacted=False, canaries=None,
                       now=TOY_NOW):
    """Write the toy store tree under `root` (see the module docstring).
    redacted=True builds the defended NEGATIVE-CONTROL world: all content
    routed through redact_text, owner-only permissions, and the expired
    backup deleted per policy. Returns metadata: {"root", "redacted",
    "planted": [{canary_id, data_class, variant, store_path, occurrences}],
    "n_redactions", "n_deleted"}. Deterministic: content, permissions, and
    mtimes are all pinned (mtimes relative to `now`)."""
    canaries = canaries or make_canaries(seed)
    by_class = {c["data_class"]: c for c in canaries}
    planted, n_redactions, n_deleted = [], 0, 0
    for d in STORES:
        os.makedirs(os.path.join(root, d), exist_ok=True)

    def emit(rel, content, mode=0o600, age_hours=1.0):
        nonlocal n_redactions
        if redacted:
            content, k = redact_text(content, canaries)
            n_redactions += k
            mode = 0o600
        path = os.path.join(root, rel)
        with open(path, "w") as f:
            f.write(content)
        os.chmod(path, mode)
        mtime = now - age_hours * 3600.0
        os.utime(path, (mtime, mtime))

    def plant(cls, variant, rel):
        planted.append({"canary_id": by_class[cls]["id"], "data_class": cls,
                        "variant": variant, "store_path": rel,
                        "occurrences": 1})

    lines = ["2026-01-01T00:00:00Z workflow wf-001 tenant=acme status=start",
             "2026-01-01T00:00:01Z workflow wf-001 status=success",
             "2026-01-01T00:00:02Z workflow wf-002 tenant=acme status=start"]
    if "prompts" in by_class:
        lines.append("2026-01-01T00:00:03Z workflow wf-002 status=failed "
                     f"error=seam_timeout prompt={by_class['prompts']['value']}")
        plant("prompts", "exact", "logs/app.log")
    else:
        lines.append("2026-01-01T00:00:03Z workflow wf-002 status=failed "
                     "error=seam_timeout")
    if "tenant_ids" in by_class:
        tok = by_class["tenant_ids"]["value"].split("-")
        lines.append("2026-01-01T00:00:04Z workflow wf-003 status=cancelled "
                     f'note="tenant {tok[0]} {tok[1]} {tok[2]} revoked '
                     'mid-run"')
        plant("tenant_ids", "token_split", "logs/app.log")
    else:
        lines.append("2026-01-01T00:00:04Z workflow wf-003 status=cancelled")
    lines.append("2026-01-01T00:00:05Z workflow wf-004 status=crashed "
                 "dump=crashes/crash_0007.dump")
    emit("logs/app.log", "\n".join(lines) + "\n")

    prom = ["attacker_workflows_total{workflow=\"wf-001\",status=\"success\"} 1"]
    if "tenant_ids" in by_class:
        trunc = canary_variants(by_class["tenant_ids"])["truncated"]
        prom.append("attacker_workflows_total{workflow=\"wf-002\","
                    f"status=\"failed\",tenant=\"{trunc}\"}} 1")
        plant("tenant_ids", "truncated", "logs/metrics.prom")
    else:
        prom.append("attacker_workflows_total{workflow=\"wf-002\","
                    "status=\"failed\"} 1")
    emit("logs/metrics.prom", "\n".join(prom) + "\n")

    label_rows = [{"seq": 0, "split": "victim", "label": "benign-label"}]
    if "labels" in by_class:
        label_rows.append({"seq": 1, "split": "victim",
                           "label": by_class["labels"]["value"]})
        plant("labels", "exact", "captures/labels.jsonl")
    emit("captures/labels.jsonl",
         "".join(json.dumps(r, sort_keys=True) + "\n" for r in label_rows),
         mode=0o444)
    emit("captures/wire_0000.jsonl",
         '{"bytes": 128, "dir": "c2s", "ts": 1700000000.0}\n'
         '{"bytes": 96, "dir": "s2c", "ts": 1700000000.1}\n')

    emit("checkpoints/ckpt_epoch3.json", json.dumps(
        {"epoch": 3, "global_step": 1200, "optimizer": "adamw",
         "adapter": None,
         "rng_sha256": hashlib.sha256(f"{seed}-ckpt".encode()).hexdigest()},
        sort_keys=True) + "\n")
    emit("artifacts/report.json", json.dumps(
        {"cancelled": 1, "crashed": 1, "failed": 1, "ok": 1,
         "workflows": 4}, sort_keys=True) + "\n")

    sidecar = {"artifact": "artifacts/report.json",
               "schema": "llm_attacker.evidence.v1"}
    if "credentials" in by_class:
        sidecar["seal"] = canary_variants(
            by_class["credentials"])["base64"]
        plant("credentials", "base64", "artifacts/evidence.sidecar.json")
    else:
        sidecar["seal"] = base64.b64encode(b"benign-seal").decode()
    emit("artifacts/evidence.sidecar.json",
         json.dumps(sidecar, sort_keys=True) + "\n")

    noise = hashlib.sha256(f"{seed}-dump".encode()).hexdigest()
    dump = ["CRASH workflow wf-004 signal=SIGSEGV",
            f"stack: 0xDEADBEEF 0x{noise[:16].upper()}"]
    if "key_material" in by_class:
        dump.append(f"reason: unwrap failed key={by_class['key_material']['value']}")
        plant("key_material", "exact", "crashes/crash_0007.dump")
    else:
        dump.append("reason: unwrap failed")
    dump.append(f"hexcontext: {noise}")
    emit("crashes/crash_0007.dump", "\n".join(dump) + "\n")

    emit("tmp/scratch.tmp", "scratch pad\n")

    if "labels" in by_class:
        if redacted:
            n_deleted += 1
        else:
            emit("backups/run_0042.bak.jsonl",
                 json.dumps({"seq": 1, "split": "victim",
                             "label": by_class["labels"]["value"]},
                            sort_keys=True) + "\n",
                 age_hours=72.0)
            plant("labels", "exact", "backups/run_0042.bak.jsonl")
    return {"root": root, "redacted": redacted, "planted": planted,
            "n_redactions": n_redactions, "n_deleted": n_deleted}


def validate_roots(roots):
    """SAFETY-CRITICAL: the scanner reads only inside explicitly passed,
    existing directories; "/" and $HOME itself are refused (pass the
    store directories under them instead). Descriptive ValueError."""
    if not roots:
        raise ValueError("no scan roots given (--roots PATH...)")
    home = os.path.realpath(os.path.expanduser("~"))
    fsroot = os.path.realpath("/")
    out = []
    for r in roots:
        resolved = os.path.realpath(os.path.expanduser(r))
        if resolved == fsroot:
            raise ValueError(f"refusing to scan the filesystem root '/' "
                             f"itself — pass explicit evaluator-authorized "
                             f"store directories instead")
        if resolved == home:
            raise ValueError(f"refusing to scan $HOME itself ({home}) — "
                             f"pass explicit evaluator-authorized store "
                             f"directories under it instead")
        if not os.path.isdir(resolved):
            raise ValueError(f"scan root {r!r}: not found or not a directory")
        out.append(resolved)
    return out


def _find_offsets(text, needle):
    """All byte-offsets of `needle` in `text` (empty needle -> [])."""
    if not needle:
        return []
    offs, start = [], 0
    while True:
        i = text.find(needle, start)
        if i < 0:
            return offs
        offs.append(i)
        start = i + 1


def _token_split_offsets(text, tokens, exact):
    """Offsets of the token-split variant: the "-"-separated canary tokens
    in order on one line; lines containing the exact form are skipped."""
    if len(tokens) < 2:
        return []
    offs, pos = [], 0
    for line in text.splitlines(keepends=True):
        if exact and exact in line:
            pos += len(line)
            continue
        cur, first, ok = 0, None, True
        for tok in tokens:
            j = line.find(tok, cur)
            if j < 0:
                ok = False
                break
            if first is None:
                first = j
            cur = j + len(tok)
        if ok and first is not None:
            offs.append(pos + first)
        pos += len(line)
    return offs


def scan_store(root, canaries, deletion_deadline_hours=24.0, now=None,
               label=None):
    """Scan one store root for canary variants + permission/retention
    hygiene. Returns a dict with n_files, hits (sha256 + offsets + mode +
    retention hint — never file content), exposure_by_class,
    copies_per_canary, and the audit counts/lists. Reads only regular
    files <= MAX_FILE_BYTES under `root`; never follows symlinks out."""
    now = time.time() if now is None else float(now)
    label = label or os.path.basename(os.path.normpath(os.path.abspath(root)))
    prepared = [(c, canary_variants(c)) for c in canaries]
    hits, skipped_large, skipped_special, unreadable = [], [], [], []
    world_files, group_files, overdue_files = [], [], []
    n_files = n_scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root)
            try:
                st = os.stat(path)
            except OSError:
                continue
            n_files += 1
            mode = stat.S_IMODE(st.st_mode)
            world = bool(mode & stat.S_IROTH)
            group = bool(mode & stat.S_IRGRP)
            if world:
                world_files.append(rel)
            if group:
                group_files.append(rel)
            if deletion_deadline_hours is None:
                retention = "no-deadline"
            else:
                age_h = (now - st.st_mtime) / 3600.0
                retention = ("beyond-deadline"
                             if age_h > deletion_deadline_hours
                             else "within-deadline")
                if retention == "beyond-deadline":
                    overdue_files.append(rel)
            if not stat.S_ISREG(st.st_mode):
                skipped_special.append(rel)
                continue
            if st.st_size > MAX_FILE_BYTES:
                skipped_large.append({"path": rel, "bytes": st.st_size})
                continue
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except OSError:
                unreadable.append(rel)
                continue
            n_scanned += 1
            text = data.decode("utf-8", "replace")
            sha = hashlib.sha256(data).hexdigest()
            for canary, var in prepared:
                occ = [("exact", o) for o in _find_offsets(text, var["exact"])]
                exact_set = {o for _, o in occ}
                occ += [("base64", o)
                        for o in _find_offsets(text, var["base64"])]
                occ += [("hex", o)
                        for o in _find_offsets(text, var["hex"])]
                occ += [("truncated", o)
                        for o in _find_offsets(text, var["truncated"])
                        if o not in exact_set]
                occ += [("token_split", o)
                        for o in _token_split_offsets(
                            text, var["tokens"], var["exact"])]
                if not occ:
                    continue
                occ.sort(key=lambda kv: kv[1])
                hits.append({"store": label, "canary_id": canary["id"],
                             "data_class": canary["data_class"],
                             "path": rel,
                             "variants": sorted({k for k, _ in occ}),
                             "n_occurrences": len(occ),
                             "offsets": [o for _, o in occ[:MAX_OFFSETS]],
                             "offsets_capped": len(occ) > MAX_OFFSETS,
                             "sha256": sha, "mode_octal": oct(mode),
                             "world_readable": world,
                             "group_readable": group,
                             "retention": retention})
    by_class = {}
    for c in canaries:
        by_class.setdefault(c["data_class"], {"exposed": 0, "total": 0})
        by_class[c["data_class"]]["total"] += 1
    copies = {}
    for h in hits:
        copies.setdefault(h["canary_id"], set()).add(h["path"])
    exposed_ids = {h["canary_id"] for h in hits}
    for c in canaries:
        if c["id"] in exposed_ids:
            by_class[c["data_class"]]["exposed"] += 1
    for cls, rec in by_class.items():
        rec["rate"] = rec["exposed"] / rec["total"] if rec["total"] else 0.0
    flagged = sorted(set(world_files) | set(group_files))
    return {"store": label, "n_files": n_files,
            "n_files_content_scanned": n_scanned,
            "skipped_large_files": skipped_large,
            "skipped_special_files": skipped_special,
            "unreadable_files": unreadable,
            "n_hits": len(hits),
            "classes_exposed": sorted({h["data_class"] for h in hits}),
            "exposure_by_class": by_class,
            "copies_per_canary": {k: len(v)
                                  for k, v in sorted(copies.items())},
            "unauthorized_readability_count": len(flagged),
            "world_readable_files": world_files,
            "group_readable_files": group_files,
            "deletion_deadline_hours": deletion_deadline_hours,
            "retention_beyond_deadline_count": len(overdue_files),
            "retention_beyond_deadline_files": overdue_files,
            "hits": hits}


def scan_stores(roots, canaries, deletion_deadline_hours=24.0, now=None):
    """Scan each evaluator-authorized root (basename becomes the store
    label); one scan dict per root, aligned with `roots`. Roots are
    validated first (refuses "/" and $HOME itself)."""
    return [scan_store(r, canaries, deletion_deadline_hours, now=now)
            for r in validate_roots(roots)]


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--roots", nargs="+", default=None,
                    help="evaluator-authorized store roots to scan "
                         "(refuses '/' and $HOME itself). SAFETY-CRITICAL: "
                         "only these directories are read")
    ap.add_argument("--canary-file", default=None,
                    help=f"real canary specs ({CANARY_FILE_SCHEMA}; a JSON "
                         "list of {value, id?, data_class?}); with --roots "
                         "supply the canaries the evaluator planted")
    ap.add_argument("--deletion-deadline", type=float, default=24.0,
                    metavar="HOURS",
                    help="retention check: files older than this are "
                         "flagged beyond-deadline (default: 24; <=0 "
                         "disables the check)")
    ap.add_argument("--work-dir", default=None,
                    help="toy: build the fixture stores here (default: a "
                         "fresh temp dir)")
    ap.add_argument("--evidence", action="store_true",
                    help="wrap the run in an EvidenceLogger (requires "
                         "--output): the artifact write is recorded as a "
                         "chained event and the evidence manifest is "
                         "finalized alongside")
    return ap


def _threat_model_doc():
    return threat_model.normalize({
        "schema": threat_model.SCHEMA_V2,
        "summary": "passive, files-only actor with read access to "
                   "persistent/operational stores (captures, logs, "
                   "checkpoints, artifacts/sidecars, crash dumps, tmp, "
                   "backups) hunting planted canaries that stand in for "
                   "prompts, labels, credentials, tenant IDs, and key "
                   "material — a path seam defenses never see",
        "scope": {"use_case": "split-training",
                  "deployment": "multi-tenant",
                  "in_scope": ["split-training and split-inference "
                               "operational stores on shared filesystems",
                               "evaluator-authorized scan roots only"],
                  "out_of_scope": ["compressed/encrypted container "
                                   "internals", "object-store APIs",
                                   "log-aggregator APIs", "live services"]},
        "actor": {"role": "co-tenant", "intent": "passive",
                  "vantage": ["read access to persistent store roots",
                              "shared filesystem / mounted object-store "
                              "metadata"],
                  "authorization": "evaluator-authorized roots only; "
                                   "scanner refuses '/' and $HOME itself",
                  "privileges": ["filesystem read within the passed roots"],
                  "knowledge": ["store layout conventions",
                                "canary values planted by the evaluator"],
                  "capabilities": ["raw-byte substring scan for exact/"
                                   "base64/hex/truncated/token-split "
                                   "variants", "permission-bit and mtime "
                                   "audits"],
                  "resources": ["read-only filesystem access"],
                  "persistence": "cross-session"},
        "target": {"assets": ["prompts", "canonical labels", "credentials",
                              "tenant identifiers", "key material"],
                   "components": ["capture pipeline", "logging/telemetry",
                                  "checkpointing", "evidence sidecars",
                                  "crash handling", "backup/retention"],
                   "data_stores": list(STORES),
                   "data_flows": ["workflow -> store emission",
                                  "store -> backup/retention"],
                   "trust_boundaries": ["process -> persistent store",
                                        "tenant -> shared filesystem"],
                   "entry_points": ["--roots scan"],
                   "exit_points": ["per-hit records (sha256 + offsets only)"]},
        "scenario": {"preconditions": ["evaluator plants canaries and "
                                       "exercises workflows",
                                       "stores readable by the scanner"],
                     "lifecycle_stage": ["deployment", "operations"],
                     "security_properties": ["confidentiality", "privacy"],
                     "attack_steps": ["plant canaries per data class",
                                      "exercise success/failed/cancelled/"
                                      "crashed workflows",
                                      "scan authorized roots for variants",
                                      "audit permissions and retention"],
                     "technical_impact": ["sensitive data recoverable from "
                                          "operational stores despite seam "
                                          "defenses"],
                     "business_impact": ["cross-tenant disclosure",
                                         "credential compromise",
                                         "retention/compliance violations"]},
        "controls": {"assumed": ["emission-time redaction",
                                 "retention/deletion policy",
                                 "least-privilege file permissions"],
                     "tested": ["redactor completeness (fixture pair)",
                                "permission audit", "retention audit"],
                     "bypassed": ["seam encryption/rotation (S1/S2 defenses "
                                  "do not cover stores at rest)"],
                     "detective": ["this scanner"],
                     "status": "open"},
        "mappings": {"owasp_llm_2026": ["LLM02", "LLM08"],
                     "cwe": ["CWE-532", "CWE-200"]},
        "evaluation": {"success_metrics": ["canary exposure rate by store "
                                           "and data class",
                                           "unauthorized readability count",
                                           "retention-beyond-deadline count",
                                           "redaction precision/recall",
                                           "copies per canary"],
                       "baseline": "toy-redacted world (negative control): "
                                   "zero hits, owner-only permissions, "
                                   "deletion respected",
                       "budget": {"reads": "regular files <= 64 MiB under "
                                           "--roots only",
                                  "toy": "5 canary classes x 7 placements"},
                       "limitations": ["raw-byte matching only — no "
                                       "decompression/decryption",
                                       "single-line token-split matching",
                                       "mounted filesystems only"],
                       "coverage_status": "covered"}})


def _make_artifact(args, source, extra_config=None, evidence=None):
    config = {"attack": EXPERIMENT_ID, "mode": getattr(args, "mode", None),
              "source": source, "seed": args.seed,
              "canary_file": args.canary_file,
              "deletion_deadline_hours": args.deletion_deadline}
    if extra_config:
        config.update(extra_config)
    return artifacts.make_artifact(
        "dtraining.attacker.artifact_telemetry_leak.v1", config,
        _threat_model_doc(),
        interpretation="secure behavior: no canary variant in any store, "
                       "no world/group-readable store files, deletion "
                       "deadline respected, redaction recall = 1.0 on the "
                       "fixture pair — and the leaking toy world MUST show "
                       "hits, else the negative control is vacuous. "
                       "Exposure is reported per store and data class with "
                       "hashes + offsets only (never file content).",
        evidence=evidence)


def _deadline(args):
    d = args.deletion_deadline
    return d if d and d > 0 else None


def _redaction_check(meta_leak, meta_red, scan_leak, scan_red, seed):
    """Fixture-pair redaction metrics: precision = planted occurrences /
    redactor actions (replacements + policy deletions); recall = 1 -
    surviving planted variants / planted variants; scanner_fixture_recall
    = planted canaries actually found in the leaking world (the control is
    vacuous without it)."""
    planted_pairs = {(p["canary_id"], p["variant"])
                     for p in meta_leak["planted"]}
    planted_occ = sum(p["occurrences"] for p in meta_leak["planted"])
    survivors = {(h["canary_id"], v) for h in scan_red["hits"]
                 for v in h["variants"]} & planted_pairs
    recall = (1.0 - len(survivors) / len(planted_pairs)
              if planted_pairs else 1.0)
    handled = meta_red["n_redactions"] + meta_red["n_deleted"]
    precision = (min(1.0, planted_occ / handled) if handled
                 else (1.0 if planted_occ == 0 else 0.0))
    planted_ids = {p["canary_id"] for p in meta_leak["planted"]}
    found_ids = {h["canary_id"] for h in scan_leak["hits"]}
    scanner_recall = (len(found_ids & planted_ids) / len(planted_ids)
                      if planted_ids else 1.0)
    return {"experiment": EXPERIMENT_ID, "arm": "redaction-check",
            "seed": seed, "planted_variants": len(planted_pairs),
            "planted_occurrences": planted_occ,
            "n_redactions": meta_red["n_redactions"],
            "n_deleted": meta_red["n_deleted"],
            "leaking_hits": scan_leak["n_hits"],
            "redacted_hits": scan_red["n_hits"],
            "surviving_variants": sorted(f"{a}:{b}" for a, b in survivors),
            "redaction_precision": precision,
            "redaction_recall": recall,
            "scanner_fixture_recall": scanner_recall}


def run_toy(args, evidence=None):
    if args.quick:
        args.seeds = args.seeds[:1]
    canned = None
    if args.canary_file:
        try:
            canned = load_canaries(args.canary_file)
        except ValueError as e:
            raise SystemExit(f"[{EXPERIMENT_ID}] {e}")
    deadline = _deadline(args)
    out = _make_artifact(args, "toy", {
        "toy": True, "seeds": args.seeds,
        "workflows": ["success", "failed", "cancelled", "crashed"],
        "canary_source": args.canary_file or "generated"},
        evidence=evidence)
    work = args.work_dir or tempfile.mkdtemp(prefix=EXPERIMENT_ID + "_")
    for seed in args.seeds:
        fseed = args.seed + 1000 * seed
        canaries = canned or make_canaries(fseed)
        metas, scans = {}, {}
        for label, redacted, role in (
                ("toy-leaking", False, "vulnerable-world"),
                ("toy-redacted", True, "negative-control")):
            root = os.path.join(work, f"seed-{seed}", label)
            metas[label] = make_fixture_store(
                root, fseed, redacted=redacted, canaries=canaries,
                now=TOY_NOW)
            try:
                scan = scan_store(root, canaries, deadline, now=TOY_NOW,
                                  label=label)
            except Exception as e:
                journal_error(args.output, EXPERIMENT_ID,
                              {"arm": "store-scan", "store": label,
                               "seed": seed}, e)
                continue
            rec = {"experiment": EXPERIMENT_ID, "arm": "store-scan",
                   "seed": seed, "role": role, **scan}
            scans[label] = rec
            artifacts.append_jsonl(args.output, rec)
            out["results"].append(rec)
            print(f"[{EXPERIMENT_ID}] store={label} seed={seed}: "
                  f"files={rec['n_files']} hits={rec['n_hits']} "
                  f"classes={rec['classes_exposed']} "
                  f"perms_flagged={rec['unauthorized_readability_count']} "
                  f"retention_flagged="
                  f"{rec['retention_beyond_deadline_count']} ({role})")
        if "toy-leaking" in scans and "toy-redacted" in scans:
            pair = _redaction_check(metas["toy-leaking"],
                                    metas["toy-redacted"],
                                    scans["toy-leaking"],
                                    scans["toy-redacted"], seed)
            artifacts.append_jsonl(args.output, pair)
            out["results"].append(pair)
            print(f"[{EXPERIMENT_ID}] redaction-check seed={seed}: "
                  f"precision={pair['redaction_precision']:.3f} "
                  f"recall={pair['redaction_recall']:.3f} "
                  f"(leaking hits={pair['leaking_hits']}, "
                  f"redacted hits={pair['redacted_hits']})")
    store_recs = [r for r in out["results"] if r.get("arm") == "store-scan"]
    pair_recs = [r for r in out["results"]
                 if r.get("arm") == "redaction-check"]
    row = {"experiment": EXPERIMENT_ID, "arm": "store-scan",
           "n_seeds": len(args.seeds), "n_stores": len(store_recs),
           "n_hits_total": sum(r["n_hits"] for r in store_recs),
           "classes_exposed_overall": sorted(
               {c for r in store_recs for c in r["classes_exposed"]}),
           "unauthorized_readability_total": sum(
               r["unauthorized_readability_count"] for r in store_recs),
           "retention_beyond_deadline_total": sum(
               r["retention_beyond_deadline_count"] for r in store_recs)}
    if pair_recs:
        row["redaction_precision_min"] = min(
            r["redaction_precision"] for r in pair_recs)
        row["redaction_recall_min"] = min(
            r["redaction_recall"] for r in pair_recs)
        row["scanner_fixture_recall_min"] = min(
            r["scanner_fixture_recall"] for r in pair_recs)
    out["summary"].append(row)
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def run_roots(args, evidence=None):
    try:
        roots = validate_roots(args.roots)
        canaries = (load_canaries(args.canary_file) if args.canary_file
                    else make_canaries(args.seed))
    except ValueError as e:
        raise SystemExit(f"[{EXPERIMENT_ID}] {e}")
    deadline = _deadline(args)
    out = _make_artifact(args, "roots", {
        "roots": roots, "n_canaries": len(canaries),
        "canary_source": args.canary_file or "generated"}, evidence=evidence)
    totals = {"hits": 0, "perms": 0, "overdue": 0}
    classes = set()
    for root in roots:
        try:
            scan = scan_store(root, canaries, deadline)
        except Exception as e:
            journal_error(args.output, EXPERIMENT_ID,
                          {"arm": "store-scan", "root": root}, e)
            continue
        rec = {"experiment": EXPERIMENT_ID, "arm": "store-scan",
               "role": "evaluator-authorized-store", "root": root, **scan}
        artifacts.append_jsonl(args.output, rec)
        out["results"].append(rec)
        totals["hits"] += rec["n_hits"]
        totals["perms"] += rec["unauthorized_readability_count"]
        totals["overdue"] += rec["retention_beyond_deadline_count"]
        classes.update(rec["classes_exposed"])
        print(f"[{EXPERIMENT_ID}] store={rec['store']} root={root}: "
              f"files={rec['n_files']} hits={rec['n_hits']} "
              f"classes={rec['classes_exposed']} "
              f"perms_flagged={rec['unauthorized_readability_count']} "
              f"retention_flagged={rec['retention_beyond_deadline_count']}")
    out["summary"].append({
        "experiment": EXPERIMENT_ID, "arm": "store-scan",
        "n_stores": len(roots), "n_hits_total": totals["hits"],
        "classes_exposed_overall": sorted(classes),
        "unauthorized_readability_total": totals["perms"],
        "retention_beyond_deadline_total": totals["overdue"]})
    artifacts.write_artifact(args.output, out, evidence=evidence)
    return 0


def _run(args, evidence=None):
    if args.toy:
        return run_toy(args, evidence=evidence)
    if args.roots:
        return run_roots(args, evidence=evidence)
    raise SystemExit(f"[{EXPERIMENT_ID}] pick an input path: --toy "
                     f"(synthetic fixture pair with ground truth) or "
                     f"--roots PATH... (evaluator-authorized stores; "
                     f"--canary-file supplies the planted canaries).")


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

    print("artifact-telemetry-leak store scanner (pure python):")

    c1 = make_canaries(7)
    check("canary generation is deterministic", c1 == make_canaries(7))
    check("different seeds give different canaries", c1 != make_canaries(8))
    check("one canary per data class",
          sorted(c["data_class"] for c in c1) == sorted(DATA_CLASSES))
    v = canary_variants(c1[0])
    check("base64/hex variants round-trip",
          base64.b64decode(v["base64"]).decode() == c1[0]["value"]
          and bytes.fromhex(v["hex"]).decode() == c1[0]["value"])
    check("truncated variant is the class prefix",
          v["truncated"] == "-".join(c1[0]["value"].split("-")[:2]))

    import tempfile as _tempfile
    with _tempfile.TemporaryDirectory() as d:
        canaries = make_canaries(11)
        root_l = os.path.join(d, "toy-leaking")
        root_r = os.path.join(d, "toy-redacted")
        meta_l = make_fixture_store(root_l, 11, redacted=False,
                                    canaries=canaries)
        meta_r = make_fixture_store(root_r, 11, redacted=True,
                                    canaries=canaries)
        scans = scan_stores([root_l, root_r], canaries, 24.0, now=TOY_NOW)
        sl, sr = scans
        check("leaking fixture exposes every data class",
              sorted(sl["classes_exposed"]) == sorted(DATA_CLASSES))
        variants_found = {v for h in sl["hits"] for v in h["variants"]}
        check("exact/base64/truncated/token-split variants all detected",
              {"exact", "base64", "truncated", "token_split"}
              <= variants_found)
        check("label canary has two copies (capture + backup)",
              sl["copies_per_canary"].get("canary-labels") == 2)
        check("world-readable canonical-label file flagged",
              sl["unauthorized_readability_count"] >= 1
              and sl["world_readable_files"])
        check("72h-old backup flagged beyond the 24h deadline",
              sl["retention_beyond_deadline_count"] == 1
              and sl["retention_beyond_deadline_files"]
              == ["backups/run_0042.bak.jsonl"])
        check("redacted fixture is clean (negative control)",
              sr["n_hits"] == 0
              and sr["unauthorized_readability_count"] == 0
              and sr["retention_beyond_deadline_count"] == 0)
        check("redactor handled every planted occurrence",
              meta_r["n_redactions"] + meta_r["n_deleted"]
              == sum(p["occurrences"] for p in meta_l["planted"]))
        pair = _redaction_check(meta_l, meta_r, sl, sr, 0)
        check("fixture pair: redaction precision == recall == 1.0",
              pair["redaction_precision"] == 1.0
              and pair["redaction_recall"] == 1.0)
        check("scanner fixture recall is 1.0 (control not vacuous)",
              pair["scanner_fixture_recall"] == 1.0)

        for bad in ("/", os.path.expanduser("~")):
            try:
                validate_roots([bad])
                check(f"refuses to scan {bad!r} itself", False)
            except ValueError as e:
                check(f"refuses to scan {bad!r} itself", "refusing" in str(e))
        try:
            validate_roots([os.path.join(d, "missing-dir")])
            check("missing scan root rejected", False)
        except ValueError as e:
            check("missing scan root rejected", "missing-dir" in str(e))

        bad = os.path.join(d, "bad.json")
        with open(bad, "w") as f:
            f.write("{not json")
        try:
            load_canaries(bad)
            check("malformed canary file rejected", False)
        except ValueError as e:
            check("malformed canary file rejected", "bad.json" in str(e))
        try:
            load_canaries(os.path.join(d, "nope.json"))
            check("missing canary file rejected", False)
        except ValueError as e:
            check("missing canary file rejected", "nope.json" in str(e))
        good = os.path.join(d, "canaries.json")
        with open(good, "w") as f:
            json.dump({"schema": CANARY_FILE_SCHEMA, "canaries": [
                {"data_class": "credentials",
                 "value": "CANARY-credentials-deadbeef"}]}, f)
        loaded = load_canaries(good)
        check("canary file loads with generated ids",
              loaded[0]["id"] == "canary-0"
              and loaded[0]["value"] == "CANARY-credentials-deadbeef")

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
