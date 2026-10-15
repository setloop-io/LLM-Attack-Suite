"""captures (attacker/captures.py): the pinned sidecar schemas, validate /
scan / group pure-python core, malformed-record handling, and the torch-only
tensor paths' fail-closed behavior when torch is absent."""

import json
import runpy
import warnings

import pytest

from attacker import captures
from attacker.captures import (INFERENCE_SIDECAR_KEYS,
                               INFERENCE_SIDECAR_KEYS_LEGACY,
                               TRAINING_SIDECAR_KEYS, group_by_epoch,
                               load_epoch_rows, load_tensor, rows_2d,
                               scan_captures, schema_keys, validate_sidecar)


def _write_capture(capture_dir, idx, meta, pt_bytes=b"\x00"):
    pt = capture_dir / f"wire_{idx:04d}.pt"
    pt.write_bytes(pt_bytes)
    (capture_dir / f"wire_{idx:04d}.json").write_text(json.dumps(meta))
    return str(pt)


def test_sidecar_key_sets_are_the_pinned_schemas():
    assert TRAINING_SIDECAR_KEYS == {"session_id", "mb_id", "phase", "step",
                                     "epoch"}
    assert INFERENCE_SIDECAR_KEYS == {"session_id", "request_seq", "phase",
                                      "position", "epoch"}
    assert INFERENCE_SIDECAR_KEYS_LEGACY == INFERENCE_SIDECAR_KEYS - {"epoch"}
    assert schema_keys("training") is TRAINING_SIDECAR_KEYS
    assert schema_keys("inference") is INFERENCE_SIDECAR_KEYS
    with pytest.raises(KeyError):
        schema_keys("no-such-mode")


def test_schemas_round_trip_through_json_unchanged():
    tr = {"session_id": "s", "mb_id": 3, "phase": "fwd", "step": 2,
          "epoch": 1}
    inf = {"session_id": "s", "request_seq": 1, "phase": "decode",
           "position": 5, "epoch": 0}
    for meta in (tr, inf):
        assert json.loads(json.dumps(meta)) == meta


def test_validate_sidecar_accepts_conforming_records():
    tr = {"session_id": "s", "mb_id": 3, "phase": "fwd", "step": 2,
          "epoch": 1}
    assert validate_sidecar(dict(tr), "training") == tr
    none_epoch = {**tr, "epoch": None}
    assert validate_sidecar(dict(none_epoch), "training") == none_epoch
    inf = {"session_id": "s", "request_seq": 1, "phase": "decode",
           "position": 5, "epoch": 0}
    assert validate_sidecar(dict(inf), "inference") == inf


def test_validate_sidecar_normalizes_legacy_inference_to_epoch_none():
    legacy = {"session_id": "s", "request_seq": 1, "phase": "prefill",
              "position": 0}
    out = validate_sidecar(dict(legacy), "inference")
    assert out == {**legacy, "epoch": None}
    assert "epoch" not in legacy


def test_validate_sidecar_raises_on_schema_drift():
    base = {"session_id": "s", "mb_id": 0, "phase": "fwd", "step": 0,
            "epoch": 0}
    with pytest.raises(ValueError):
        validate_sidecar({k: v for k, v in base.items() if k != "step"},
                         "training")
    with pytest.raises(ValueError):
        validate_sidecar({**base, "surprise": 1}, "training")
    with pytest.raises(ValueError):
        validate_sidecar({k: v for k, v in base.items() if k != "epoch"},
                         "training")
    with pytest.raises(KeyError):
        validate_sidecar(dict(base), "bogus-mode")
    with pytest.raises(ValueError, match="sidecar keys"):
        validate_sidecar({"session_id": "s"}, "training",
                         path="wire_0007.json")


def test_scan_orders_training_by_step_then_mb_not_filename(tmp_path):
    metas = [
        {"session_id": "s0", "mb_id": 1, "phase": "fwd", "step": 0,
         "epoch": 0},
        {"session_id": "s0", "mb_id": 0, "phase": "bwd", "step": 1,
         "epoch": 1},
        {"session_id": "s0", "mb_id": 0, "phase": "fwd", "step": 0,
         "epoch": 0},
    ]
    for i, m in enumerate(metas):
        _write_capture(tmp_path, i, m)
    recs = scan_captures(str(tmp_path), "training")
    assert [(r[0]["step"], r[0]["mb_id"]) for r in recs] == [(0, 0), (0, 1),
                                                             (1, 0)]
    assert all(r[1].endswith(".pt") for r in recs)


def test_scan_orders_inference_by_session_seq_position(tmp_path):
    metas = [
        {"session_id": "s1", "request_seq": 0, "phase": "decode",
         "position": 1, "epoch": 0},
        {"session_id": "s0", "request_seq": 2, "phase": "decode",
         "position": 0, "epoch": 0},
        {"session_id": "s0", "request_seq": 1, "phase": "prefill",
         "position": 0, "epoch": 0},
    ]
    for i, m in enumerate(metas):
        _write_capture(tmp_path, i, m)
    recs = scan_captures(str(tmp_path), "inference")
    got = [(r[0]["session_id"], r[0]["request_seq"], r[0]["position"])
           for r in recs]
    assert got == [("s0", 1, 0), ("s0", 2, 0), ("s1", 0, 1)]


def test_scan_normalizes_legacy_inference_sidecars(tmp_path):
    _write_capture(tmp_path, 0, {"session_id": "s", "request_seq": 0,
                                 "phase": "prefill", "position": 0})
    recs = scan_captures(str(tmp_path), "inference")
    assert recs[0][0]["epoch"] is None


def test_scan_skips_orphan_sidecars_and_non_wire_files(tmp_path):
    _write_capture(tmp_path, 0, {"session_id": "s", "mb_id": 0,
                                 "phase": "fwd", "step": 0, "epoch": 0})
    (tmp_path / "wire_0001.json").write_text(json.dumps(
        {"session_id": "s", "mb_id": 1, "phase": "fwd", "step": 0,
         "epoch": 0}))
    (tmp_path / "notes.json").write_text("{}")
    (tmp_path / "wire_0002.txt").write_text("not a capture")
    recs = scan_captures(str(tmp_path), "training")
    assert len(recs) == 1 and recs[0][0]["mb_id"] == 0


def test_scan_raises_on_malformed_records(tmp_path):
    _write_capture(tmp_path, 0, {"session_id": "s", "mb_id": 0})
    with pytest.raises(ValueError, match="wire_0000"):
        scan_captures(str(tmp_path), "training")


def test_scan_raises_on_unparseable_sidecar(tmp_path):
    (tmp_path / "wire_0000.pt").write_bytes(b"\x00")
    (tmp_path / "wire_0000.json").write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        scan_captures(str(tmp_path), "training")


def test_scan_empty_dir_and_bad_mode(tmp_path):
    assert scan_captures(str(tmp_path), "training") == []
    with pytest.raises(KeyError):
        scan_captures(str(tmp_path), "bogus-mode")


def _scan_training(tmp_path):
    metas = [
        {"session_id": "s0", "mb_id": 0, "phase": "fwd", "step": 0,
         "epoch": 0},
        {"session_id": "s0", "mb_id": 1, "phase": "fwd", "step": 0,
         "epoch": 0},
        {"session_id": "s0", "mb_id": 0, "phase": "bwd", "step": 1,
         "epoch": 1},
        {"session_id": "s0", "mb_id": 0, "phase": "fwd", "step": 1,
         "epoch": 1},
    ]
    for i, m in enumerate(metas):
        _write_capture(tmp_path, i, m)
    return scan_captures(str(tmp_path), "training")


def test_group_by_epoch_groups_and_filters(tmp_path):
    recs = _scan_training(tmp_path)
    grp = group_by_epoch(recs)
    assert sorted(grp) == [0, 1]
    assert [m["mb_id"] for m, _ in grp[0]] == [0, 1]
    assert [m["phase"] for m, _ in grp[1]] == ["bwd", "fwd"]
    fwd = group_by_epoch(recs, phase="fwd")
    assert sorted(fwd) == [0, 1]
    assert all(m["phase"] == "fwd" for items in fwd.values() for m, _ in items)
    assert group_by_epoch(recs, phase="no-such-phase") == {}


def test_group_by_epoch_buckets_legacy_none_epoch():
    legacy_meta = {"session_id": "s", "request_seq": 0, "phase": "prefill",
                   "position": 0, "epoch": None}
    grp = group_by_epoch([(legacy_meta, "wire_0000.pt")])
    assert list(grp) == [None]
    assert grp[None][0][0] is legacy_meta


class _FakeTensor:
    """Duck-typed tensor double: only the shape machinery rows_2d touches."""

    def __init__(self, dims):
        self.shape = tuple(dims)

    def dim(self):
        return len(self.shape)

    def unsqueeze(self, axis):
        assert axis == 0
        return _FakeTensor((1, *self.shape))

    def reshape(self, *shape):
        return _FakeTensor(shape)


def test_rows_2d_flattens_to_row_matrix():
    assert rows_2d(_FakeTensor([7])).shape == (1, 7)
    t = _FakeTensor([5, 7])
    assert rows_2d(t) is t
    assert rows_2d(_FakeTensor([2, 3, 7])).shape == (-1, 7)


def test_load_tensor_requires_torch():
    if captures.torch is not None:
        pytest.skip("torch installed — torch-absent behavior not applicable")
    with pytest.raises(RuntimeError, match="torch is required"):
        load_tensor("wire_0000.pt")


def test_load_epoch_rows_fails_closed_without_torch(tmp_path):
    if captures.torch is not None:
        pytest.skip("torch installed — torch-absent behavior not applicable")
    recs = _scan_training(tmp_path)
    with pytest.raises(RuntimeError, match="torch is required"):
        load_epoch_rows(recs, phase="fwd")


def test_self_test_passes():
    assert captures.self_test() == 0


def test_module_runs_standalone_torchless():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("attacker.captures", run_name="__main__")
    assert exc.value.code == 0
