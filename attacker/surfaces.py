#!/usr/bin/env python3
"""Observation-surface providers: one registry for everything the attacker
can observe.

A provider adapts one observation surface (wire captures, hardware
counters, timing channels, ...) to the framework. The contract is
structural — subclassing SurfaceProvider is convenient but any object with
the right attributes can be registered:

  name     str, unique registry key ("wire-training", ...)
  schema   dict of required metadata keys -> advisory python type (or None);
           key-set equality is what validation enforces
  modes    tuple of framework modes ("training" / "inference") served
  rows(mode=None, source=None)
           yields (tensor_or_array, metadata_dict); torch-only providers
           raise a descriptive error when torch is absent
  validate(meta, path) -> normalized metadata dict, raising ValueError on
           schema drift (the base class implements key-set validation)

Registry (keyed by name):
  register_surface(provider, replace=False)
  get_surface(name)
  list_surfaces()

The two historical wire-capture schemas (split-training and
split-inference sidecars) are built-in providers registered at import;
attacker/captures.py keeps its public functions as thin wrappers over
them. Third parties register their own adapter instead of forking:

  class VRAMSurface(SurfaceProvider):
      name = "vram-residue"
      schema = {"device": str, "offset": int, "epoch": None}
      modes = ("training", "inference")
      def rows(self, mode=None, source=None):
          self.check_mode(mode)
          ...
  register_surface(VRAMSurface())
"""

import glob
import json
import os

try:
    import torch
except ImportError:
    torch = None


class SurfaceProvider:
    """Base class documenting the provider contract."""

    name = ""
    schema = {}
    modes = ()

    def check_mode(self, mode):
        """Raise ValueError if mode is set and this surface doesn't serve it."""
        if self.modes and mode is not None and mode not in self.modes:
            raise ValueError(
                f"surface '{self.name}' does not serve mode {mode!r} "
                f"(modes: {', '.join(self.modes)})")

    def validate(self, meta, path="<metadata>"):
        """Key-set validation: raise ValueError on schema drift."""
        keys = set(meta)
        want = set(self.schema)
        if keys != want:
            raise ValueError(f"{path}: metadata keys {sorted(keys)} != "
                             f"surface '{self.name}' schema {sorted(want)}")
        return meta

    def rows(self, mode=None, source=None):
        raise NotImplementedError(
            f"surface '{self.name}' does not implement rows()")


def load_tensor(pt_path):
    """Load one capture tensor as float32 CPU. Torch-only."""
    if torch is None:
        raise RuntimeError("torch is required to load .pt captures")
    return torch.load(pt_path, map_location="cpu").float()


class WireCaptureProvider(SurfaceProvider):
    """A directory of wire_NNNN.pt tensors + wire_NNNN.json sidecars, as
    written by the split-training / split-inference capture hooks. Loading
    tensors needs torch; schema/scan logic is pure python (repo rule)."""

    def __init__(self, name, mode, schema, sort_key, legacy_schema=None,
                 normalize=None):
        self.name = name
        self.mode = mode
        self.modes = (mode,)
        self.schema = dict(schema)
        self.sort_key = sort_key
        self.legacy_schema = set(legacy_schema) if legacy_schema else None
        self._normalize = normalize

    def validate(self, meta, path="<sidecar>"):
        """Raise ValueError on schema drift; legacy sidecars (pre-ratchet,
        no 'epoch') are accepted and normalized."""
        keys = set(meta)
        if self.legacy_schema is not None and keys == self.legacy_schema:
            return self._normalize(dict(meta))
        want = set(self.schema)
        if keys != want:
            raise ValueError(f"{path}: sidecar keys {sorted(keys)} != "
                             f"{self.mode} schema {sorted(want)}")
        return meta

    def scan(self, capture_dir):
        """[(meta, pt_path), ...] sorted by the surface's alignment key.
        Pure python: sidecars are validated, tensors are never read."""
        records = []
        for jf in sorted(glob.glob(os.path.join(capture_dir, "wire_*.json"))):
            pt = jf[:-len(".json")] + ".pt"
            if not os.path.exists(pt):
                continue
            with open(jf) as f:
                meta = self.validate(json.load(f), jf)
            records.append((meta, pt))
        records.sort(key=lambda r: self.sort_key(r[0]))
        return records

    def load(self, pt_path):
        return load_tensor(pt_path)

    def rows(self, mode=None, source=None):
        self.check_mode(mode)
        if source is None:
            raise ValueError(f"surface '{self.name}' needs "
                             f"source=<wire capture dir>")
        return ((self.load(pt), meta) for meta, pt in self.scan(source))


_REGISTRY = {}


def register_surface(provider, replace=False):
    """Register a surface provider by name. A name collision is a
    descriptive error unless replace=True (never silent overrides)."""
    name = getattr(provider, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"surface provider {provider!r} has no usable "
                         f"`name` (non-empty str required)")
    schema = getattr(provider, "schema", None)
    if not isinstance(schema, dict) or not schema:
        raise ValueError(f"surface '{name}': `schema` must be a non-empty "
                         f"dict of required metadata keys")
    if not callable(getattr(provider, "rows", None)):
        raise ValueError(f"surface '{name}': `rows(mode)` must be callable")
    if name in _REGISTRY and not replace:
        raise ValueError(f"surface '{name}' is already registered "
                         f"({_REGISTRY[name]!r}); pass replace=True to "
                         f"override")
    _REGISTRY[name] = provider
    return provider


def get_surface(name):
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown surface {name!r}; registered: "
                       f"{', '.join(list_surfaces()) or '<none>'}") from None


def list_surfaces():
    return sorted(_REGISTRY)


register_surface(WireCaptureProvider(
    "wire-training", "training",
    {"session_id": str, "mb_id": int, "phase": str, "step": int,
     "epoch": (int, type(None))},
    sort_key=lambda m: (m.get("step", 0), m.get("mb_id", 0))))
register_surface(WireCaptureProvider(
    "wire-inference", "inference",
    {"session_id": str, "request_seq": int, "phase": str, "position": int,
     "epoch": (int, type(None))},
    sort_key=lambda m: (m.get("session_id", ""), m.get("request_seq", 0),
                        m.get("position", 0)),
    legacy_schema={"session_id", "request_seq", "phase", "position"},
    normalize=lambda m: {**m, "epoch": None}))


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("observation-surface registry (pure python):")
    check("built-in wire surfaces registered at import",
          {"wire-inference", "wire-training"} <= set(list_surfaces()))
    check("wire-training schema is the historical sidecar key set",
          set(get_surface("wire-training").schema)
          == {"session_id", "mb_id", "phase", "step", "epoch"})
    check("wire-inference schema is the historical sidecar key set",
          set(get_surface("wire-inference").schema)
          == {"session_id", "request_seq", "phase", "position", "epoch"})

    tr = get_surface("wire-training")
    check("provider validate passes a conforming sidecar",
          tr.validate({"session_id": "s", "mb_id": 0, "phase": "fwd",
                       "step": 0, "epoch": None})["epoch"] is None)
    try:
        tr.validate({"session_id": "s", "mb_id": 0})
        check("provider validate raises on schema drift", False)
    except ValueError:
        check("provider validate raises on schema drift", True)
    inf = get_surface("wire-inference")
    check("legacy inference sidecar normalized to epoch=None",
          inf.validate({"session_id": "s", "request_seq": 1,
                        "phase": "prefill", "position": 0})["epoch"] is None)
    try:
        tr.rows(mode="inference", source="/nonexistent")
        check("mode mismatch raises", False)
    except ValueError:
        check("mode mismatch raises", True)

    class ToySurface(SurfaceProvider):
        name = "selftest-toy"
        schema = {"frame_id": int, "epoch": None}
        modes = ("training",)

        def rows(self, mode=None, source=None):
            self.check_mode(mode)
            return iter([([1.0, 2.0], {"frame_id": 0, "epoch": 0}),
                         ([3.0, 4.0], {"frame_id": 1, "epoch": 0})])

    try:
        toy = register_surface(ToySurface())
        check("third-party provider registers", "selftest-toy" in list_surfaces())
        check("get_surface returns the registered object",
              get_surface("selftest-toy") is toy)
        check("rows(mode) yields (array, metadata) pairs",
              [(list(a), m["frame_id"]) for a, m in toy.rows(mode="training")]
              == [([1.0, 2.0], 0), ([3.0, 4.0], 1)])
        try:
            register_surface(ToySurface())
            check("name collision is a descriptive error", False)
        except ValueError as e:
            check("name collision is a descriptive error",
                  "selftest-toy" in str(e))
        check("replace=True overrides explicitly",
              register_surface(ToySurface(), replace=True) is not toy)
        try:
            get_surface("no-such-surface")
            check("unknown surface raises KeyError naming the registry", False)
        except KeyError as e:
            check("unknown surface raises KeyError naming the registry",
                  "selftest-toy" in str(e))
    finally:
        _REGISTRY.pop("selftest-toy", None)
    check("registry left pristine after self-test",
          "selftest-toy" not in list_surfaces())

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
