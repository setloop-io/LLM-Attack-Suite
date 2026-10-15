"""Surface provider registry + captures.py wrapper compatibility."""

import json

import pytest

from attacker import captures, surfaces
from attacker.surfaces import (SurfaceProvider, get_surface, list_surfaces,
                               register_surface)


def test_builtin_wire_surfaces_registered():
    assert {"wire-inference", "wire-training"} <= set(list_surfaces())
    assert set(get_surface("wire-training").schema) == {
        "session_id", "mb_id", "phase", "step", "epoch"}
    assert set(get_surface("wire-inference").schema) == {
        "session_id", "request_seq", "phase", "position", "epoch"}


def test_captures_constants_derive_from_providers():
    assert captures.TRAINING_SIDECAR_KEYS == set(
        get_surface("wire-training").schema)
    assert captures.INFERENCE_SIDECAR_KEYS == set(
        get_surface("wire-inference").schema)
    assert captures.INFERENCE_SIDECAR_KEYS_LEGACY == {
        "session_id", "request_seq", "phase", "position"}
    assert captures.SCHEMAS["training"] is captures.TRAINING_SIDECAR_KEYS
    assert captures.schema_keys("inference") == set(
        get_surface("wire-inference").schema)


def test_validate_sidecar_semantics_unchanged():
    tr = {"session_id": "s", "mb_id": 3, "phase": "fwd", "step": 2,
          "epoch": 1}
    assert captures.validate_sidecar(dict(tr), "training") == tr
    legacy = {"session_id": "s", "request_seq": 1, "phase": "prefill",
              "position": 0}
    assert captures.validate_sidecar(
        dict(legacy), "inference")["epoch"] is None
    with pytest.raises(ValueError):
        captures.validate_sidecar({"session_id": "s", "mb_id": 0},
                                  "training")
    with pytest.raises(KeyError):
        captures.validate_sidecar(dict(tr), "sideways")


def _write_wire_dir(d, metas):
    for i, m in enumerate(metas):
        (d / f"wire_{i:04d}.json").write_text(json.dumps(m))
        (d / f"wire_{i:04d}.pt").write_bytes(b"\x00")


def test_scan_captures_orders_by_alignment_key_not_filename(tmp_path):
    _write_wire_dir(tmp_path, [
        {"session_id": "s0", "mb_id": 1, "phase": "fwd", "step": 0,
         "epoch": 0},
        {"session_id": "s0", "mb_id": 0, "phase": "fwd", "step": 0,
         "epoch": 0},
        {"session_id": "s0", "mb_id": 0, "phase": "bwd", "step": 1,
         "epoch": 1},
    ])
    recs = captures.scan_captures(str(tmp_path), "training")
    assert [(r[0]["step"], r[0]["mb_id"]) for r in recs] == [
        (0, 0), (0, 1), (1, 0)]
    grp = captures.group_by_epoch(recs, phase="fwd")
    assert sorted(grp.keys()) == [0] and len(grp[0]) == 2


def test_scan_captures_raises_on_drift_inside_directory(tmp_path):
    _write_wire_dir(tmp_path, [{"session_id": "s0", "mb_id": 0}])
    with pytest.raises(ValueError):
        captures.scan_captures(str(tmp_path), "training")


class ToySurface(SurfaceProvider):
    name = "pytest-toy"
    schema = {"frame_id": int, "epoch": None}
    modes = ("training",)

    def rows(self, mode=None, source=None):
        self.check_mode(mode)
        return iter([([1.0, 2.0], {"frame_id": 0, "epoch": 0}),
                     ([3.0, 4.0], {"frame_id": 1, "epoch": 0})])


@pytest.fixture
def toy_surface():
    toy = register_surface(ToySurface())
    yield toy
    surfaces._REGISTRY.pop(ToySurface.name, None)


def test_third_party_provider_registration(toy_surface):
    assert ToySurface.name in list_surfaces()
    assert get_surface(ToySurface.name) is toy_surface
    rows = list(toy_surface.rows(mode="training"))
    assert [m["frame_id"] for _, m in rows] == [0, 1]
    assert rows[0][0] == [1.0, 2.0]
    with pytest.raises(ValueError):
        toy_surface.rows(mode="inference")


def test_register_surface_collision_and_replace(toy_surface):
    with pytest.raises(ValueError, match="pytest-toy"):
        register_surface(ToySurface())
    replacement = register_surface(ToySurface(), replace=True)
    assert get_surface(ToySurface.name) is replacement


def test_register_surface_rejects_incomplete_providers():
    with pytest.raises(ValueError):
        register_surface(SurfaceProvider())
    with pytest.raises(ValueError):
        register_surface(type("P", (), {"name": "x", "schema": {}})())


def test_get_surface_unknown_names_registered():
    with pytest.raises(KeyError, match="wire-training"):
        get_surface("no-such-surface")


def test_wire_provider_rows_require_source():
    with pytest.raises(ValueError, match="source"):
        get_surface("wire-training").rows(mode="training")
