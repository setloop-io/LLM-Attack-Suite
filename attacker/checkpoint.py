#!/usr/bin/env python3
"""checkpoint — generic, torch-free resumable state for attack loops.

Long optimization loops (seam-inversion's vmi optimizer, gradient-inversion
arms, future 7B+ cells) need kill-and-restart safety. This module is the
framework's small, dependency-free take on checkpointing, in the spirit of
splitbench's CaptureWriter: a pluggable saver, crash-safe tmp+replace
writes, and zero torch at import time.

Layout: a Checkpointer owns one NAMED checkpoint stream under a directory:

    Checkpointer(dir, name, every)  ->  dir/name/ckpt_{step:06d}.{ext}

`name` is a single path component — one stream per subdirectory, so
several loops (e.g. one per attack cell) share `dir` without colliding.
`every` is the cadence: save(step, state) is a no-op unless
step % every == 0 (so step 0 is always checkpointed — the repo's
capture-every convention).

Format: ext defaults to ".pt" when torch is importable (torch.save handles
tensor-bearing state dicts) else ".json" (state must then be
JSON-serializable). Pass ext= to force a format, and saver= / loader= to
plug a custom (path, state) writer / path -> state reader — the same
injection pattern as CaptureWriter, so tests exercise the machinery
torch-free with a fake saver. The default saver/loader are
extension-driven; .pt loads with weights_only=True (checkpoints are the
attacker's own loop state — tensors + primitives; plug loader= for exotic
payloads). A .pt checkpoint on a torch-free host is a descriptive error
pointing at loader=.

Crash-safety: every write goes <file>.tmp -> os.replace (artifacts.py
style), so latest() only ever sees complete checkpoints. An orphan .tmp
from a killed write is invisible to latest() and cleaned on the next save.
Only the last `keep` checkpoints are retained (default 2).

API:

    ck = Checkpointer(dir, name, every, keep=2)
    path  = ck.save(step, state)   # None when step % every != 0
    found = ck.latest()            # (step, payload) | None
    state = ck.resume()            # payload | None (latest()'s payload)

Pure python: the module imports and self-tests with NO torch installed.
"""

import json
import os
import re

_PATTERN = re.compile(r"^ckpt_(\d{6})\.(json|pt)$")
_EXTS = (".json", ".pt")


def _torch_importable():
    try:
        import torch
    except ImportError:
        return False
    return True


def _default_saver(path, state):
    """Extension-driven default payload writer: .pt -> torch.save (lazy
    import — the framework's torch-guard rule), else json.dump.

    The checkpointer hands over a .tmp path and os.replace()s it into
    place, so savers must write exactly the path they are given."""
    if path.endswith(".pt.tmp") or path.endswith(".pt"):
        try:
            import torch
        except ImportError:
            raise ValueError(
                f"checkpoint {path} needs torch for the .pt format; run "
                "on a torch host, force ext='.json', or plug a custom "
                "saver=") from None
        torch.save(state, path)
        return
    try:
        with open(path, "w") as f:
            json.dump(state, f)
    except TypeError as e:
        raise ValueError(
            f"checkpoint state is not JSON-serializable ({e}); torch "
            "was not importable for the .pt format — run on a torch "
            "host or plug a custom saver=") from None


def _default_loader(path):
    """Extension-driven default reader (see _default_saver). .pt loads
    with weights_only=True: checkpoints are the attacker's own loop state
    (tensors + primitives); plug loader= for exotic payloads."""
    if path.endswith(".pt"):
        try:
            import torch
        except ImportError:
            raise ValueError(
                f"checkpoint {path} is a torch .pt payload but torch is "
                "not importable here; resume on a torch host or plug a "
                "custom loader=") from None
        return torch.load(path, map_location="cpu", weights_only=True)
    with open(path) as f:
        return json.load(f)


def _need_int(name, value, lo):
    if not isinstance(value, int) or isinstance(value, bool) or value < lo:
        raise ValueError(f"Checkpointer: {name} must be an integer "
                         f">= {lo}, got {value!r}")


def _check_name(name):
    if not isinstance(name, str) or not name:
        raise ValueError(f"Checkpointer: name must be a non-empty string, "
                         f"got {name!r}")
    if name in (".", "..") or os.path.basename(name) != name \
            or "/" in name or "\\" in name:
        raise ValueError(f"Checkpointer: name {name!r} must be a single "
                         "path component (one checkpoint stream per "
                         "subdirectory)")


class Checkpointer:
    """Crash-safe resumable state for one optimization loop (module doc).

    save(step, state) writes dir/name/ckpt_{step:06d}.{ext} when
    step % every == 0 (else returns None without touching disk), tmp ->
    os.replace, then prunes to the last `keep` checkpoints. latest()
    returns (step, payload) of the newest complete checkpoint — highest
    step wins; a same-step .pt/.json pair (torch availability changed
    between runs) resolves to .pt, the richer format. resume() is
    latest()'s payload (None on an empty stream)."""

    def __init__(self, dir, name, every, keep=2, saver=None, loader=None,
                 ext=None):
        _check_name(name)
        _need_int("every", every, 1)
        _need_int("keep", keep, 1)
        if ext is None:
            ext = ".pt" if _torch_importable() else ".json"
        if ext not in _EXTS:
            raise ValueError(f"Checkpointer: ext must be one of "
                             f"{_EXTS}, got {ext!r}")
        self.root = os.fspath(dir)
        self.name = name
        self.dir = os.path.join(self.root, name)
        os.makedirs(self.dir, exist_ok=True)
        self.every = every
        self.keep = keep
        self.ext = ext
        self.saver = _default_saver if saver is None else saver
        self.loader = _default_loader if loader is None else loader

    def _scan(self):
        """(step, filename) of every COMPLETE checkpoint, step-sorted."""
        out = []
        for fname in os.listdir(self.dir):
            m = _PATTERN.match(fname)
            if m:
                out.append((int(m.group(1)), fname))
        return sorted(out)

    def save(self, step, state):
        """Checkpoint `state` at `step`; returns the path, or None when
        the cadence gates this step out."""
        _need_int("step", step, 0)
        if step % self.every != 0:
            return None
        path = os.path.join(self.dir, f"ckpt_{step:06d}{self.ext}")
        tmp = path + ".tmp"
        self.saver(tmp, state)
        os.replace(tmp, path)
        self._prune()
        return path

    def _prune(self):
        """Keep the last `keep` checkpoints; drop orphan .tmp files."""
        for _, fname in self._scan()[:-self.keep]:
            os.remove(os.path.join(self.dir, fname))
        for fname in os.listdir(self.dir):
            if fname.startswith("ckpt_") and fname.endswith(".tmp"):
                os.remove(os.path.join(self.dir, fname))

    def latest(self):
        """(step, payload) of the newest complete checkpoint, or None."""
        found = self._scan()
        if not found:
            return None
        step, fname = max(found,
                          key=lambda item: (item[0],
                                            item[1].endswith(".pt")))
        return step, self.loader(os.path.join(self.dir, fname))

    def resume(self):
        """The newest checkpoint's payload, or None on an empty stream."""
        found = self.latest()
        return None if found is None else found[1]

    def __repr__(self):
        return (f"Checkpointer(dir={self.dir!r}, every={self.every}, "
                f"keep={self.keep}, ext={self.ext!r})")


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import tempfile

    print("resumable checkpoint streams (pure python):")

    with tempfile.TemporaryDirectory() as d:
        ck = Checkpointer(d, "vmi", 1)
        check("empty stream: latest()/resume() -> None",
              ck.latest() is None and ck.resume() is None)
        ck.save(0, {"step": 0, "loss": 1.5})
        ck.save(1, {"step": 1, "loss": 1.25})
        check("save/resume roundtrip via latest()",
              ck.latest() == (1, {"step": 1, "loss": 1.25}))
        check("resume() returns the payload",
              ck.resume() == {"step": 1, "loss": 1.25})
        check("checkpoints live under dir/name as ckpt_{step:06d}",
              os.path.basename(ck.dir) == "vmi" and sorted(
                  os.listdir(ck.dir)) ==
              [f"ckpt_000000{ck.ext}", f"ckpt_000001{ck.ext}"])

    with tempfile.TemporaryDirectory() as d:
        ck = Checkpointer(d, "gated", 3)
        check("save off-cadence is a no-op",
              ck.save(1, {"x": 1}) is None
              and ck.save(2, {"x": 2}) is None and ck.latest() is None)
        p0 = ck.save(0, {"x": 0})
        p3 = ck.save(3, {"x": 3})
        check("cadence hits step 0 and step % every == 0",
              p0 is not None and p3 is not None
              and ck.latest() == (3, {"x": 3}))

    with tempfile.TemporaryDirectory() as d:
        ck = Checkpointer(d, "prune", 1, keep=2)
        for s in range(5):
            ck.save(s, {"step": s})
        check("keeps only the last K checkpoints",
              sorted(os.listdir(ck.dir)) ==
              [f"ckpt_000003{ck.ext}", f"ckpt_000004{ck.ext}"])
        check("latest survives pruning", ck.latest() == (4, {"step": 4}))

    with tempfile.TemporaryDirectory() as d:
        ck = Checkpointer(d, "mtime", 1, keep=5)
        ck.save(9, {"step": 9})
        ck.save(3, {"step": 3})
        check("latest picks the highest step, not the newest mtime",
              ck.latest() == (9, {"step": 9}))

    with tempfile.TemporaryDirectory() as d:
        ck = Checkpointer(d, "crash", 1)
        ck.save(1, {"ok": True})
        orphan = os.path.join(ck.dir, f"ckpt_000009{ck.ext}.tmp")
        with open(orphan, "w") as f:
            f.write("{torn")
        check("an orphan .tmp from a killed write is invisible to latest()",
              ck.latest() == (1, {"ok": True}))
        ck.save(2, {"ok": 2})
        check("the next save cleans orphan .tmp files",
              not any(f.endswith(".tmp") for f in os.listdir(ck.dir)))

    with tempfile.TemporaryDirectory() as d:
        def fake_saver(path, state):
            with open(path, "wb") as f:
                f.write(b"FAKE:" + repr(state).encode())

        def fake_loader(path):
            with open(path, "rb") as f:
                return f.read().decode()

        ck = Checkpointer(d, "fake", 1, saver=fake_saver,
                          loader=fake_loader)
        ck.save(0, ("z", 1))
        check("pluggable saver/loader round-trip (CaptureWriter pattern)",
              ck.resume().startswith("FAKE:"))

    for bad in ("", "a/b", "..", "a\\b"):
        try:
            Checkpointer(tempfile.mkdtemp(), bad, 1)
            check(f"bad name {bad!r} raises", False)
        except ValueError:
            check(f"bad name {bad!r} raises", True)
    for kw in ({"every": 0}, {"every": 1, "keep": 0}, {"every": 1.5}):
        try:
            Checkpointer(tempfile.mkdtemp(), "x", **kw)
            check(f"bad cadence/keep {kw} raises", False)
        except ValueError:
            check(f"bad cadence/keep {kw} raises", True)
    with tempfile.TemporaryDirectory() as d:
        ck = Checkpointer(d, "steps", 1)
        for bad_step in (-1, 1.5, True):
            try:
                ck.save(bad_step, {})
                check(f"bad step {bad_step!r} raises", False)
            except ValueError:
                check(f"bad step {bad_step!r} raises", True)
        try:
            Checkpointer(d, "y", 1, ext=".bin")
            check("unknown ext raises", False)
        except ValueError:
            check("unknown ext raises", True)

    check("default ext follows torch availability",
          Checkpointer(tempfile.mkdtemp(), "e", 1).ext ==
          (".pt" if _torch_importable() else ".json"))

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
