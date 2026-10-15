"""synthetic (attacker/synthetic.py): the toy-world generator is torch-only
by design ("Synthetic toy scenario generator (torch-only)"). Pinned torch-free:
the module imports (repo rule), make_toy_world refuses without torch, and
write_toy_captures' pure-python machinery — sidecar schema, wire_NNNN naming,
fwd+bwd phase writing, epoch partitioning — verified end-to-end by stubbing
only the tensor-serialization seam (torch.save) and scanning the result back
through attacker.captures (writer and loader cannot drift apart)."""

import os

import pytest

from attacker import captures, synthetic


def test_module_imports_torch_less():
    if synthetic.torch is not None:
        pytest.skip("torch installed — torch-absent behavior not applicable")
    assert callable(synthetic.make_toy_world)
    assert callable(synthetic.write_toy_captures)


def test_make_toy_world_requires_torch():
    if synthetic.torch is not None:
        pytest.skip("torch installed — torch-absent behavior not applicable")
    with pytest.raises(AttributeError):
        synthetic.make_toy_world()


class _FakeRows:
    """Duck-typed stand-in for world['public_h'] rows (shape + slicing)."""
    shape = (2048, 64)

    def __getitem__(self, _sl):
        return self


class _FakeWireRows:
    def float(self):
        return self


def _toy_world(n_epochs=2):
    return {"public_h": _FakeRows(), "n_epochs": n_epochs,
            "wire": lambda h, t: _FakeWireRows()}


class _FakeTorch:
    """Only the serialization seam: wire tensors become placeholder bytes
    (scan_captures never reads .pt payloads, by design)."""

    @staticmethod
    def save(_obj, path):
        with open(path, "wb") as f:
            f.write(b"\x00")


def test_write_toy_captures_fails_closed_without_torch(tmp_path):
    if synthetic.torch is not None:
        pytest.skip("torch installed — torch-absent behavior not applicable")
    capture_dir = tmp_path / "caps"
    with pytest.raises(AttributeError):
        synthetic.write_toy_captures(_toy_world(), str(capture_dir),
                                     "training")
    assert list(capture_dir.iterdir()) == []


def test_write_toy_captures_training_schema_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(synthetic, "torch", _FakeTorch)
    capture_dir = str(tmp_path / "caps")
    records = synthetic.write_toy_captures(_toy_world(n_epochs=2),
                                           capture_dir, "training",
                                           rows_per_epoch=1)
    assert len(records) == 4
    assert sorted(m["phase"] for m, _ in records) == ["bwd", "bwd",
                                                      "fwd", "fwd"]
    for (meta, pt), idx in zip(records, range(4)):
        assert pt == os.path.join(capture_dir, f"wire_{idx:04d}.pt")
        assert os.path.isfile(pt) and os.path.isfile(pt[:-3] + ".json")
        assert meta["session_id"] == "toy"
        assert (meta["mb_id"], meta["step"], meta["epoch"]) == (
            (0, 0, 0) if idx < 2 else (1, 1, 1))
        captures.validate_sidecar(dict(meta), "training", path=pt)
    scanned = captures.scan_captures(capture_dir, "training")
    assert [m["phase"] for m, _ in scanned] == ["fwd", "bwd", "fwd", "bwd"]
    by_epoch = captures.group_by_epoch(scanned, phase="fwd")
    assert sorted(by_epoch) == [0, 1]
    assert all(len(items) == 1 for items in by_epoch.values())


def test_write_toy_captures_inference_schema_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(synthetic, "torch", _FakeTorch)
    capture_dir = str(tmp_path / "caps")
    records = synthetic.write_toy_captures(_toy_world(n_epochs=3),
                                           capture_dir, "inference",
                                           rows_per_epoch=1)
    assert [m["phase"] for m, _ in records] == ["decode"] * 3
    for meta, _ in records:
        captures.validate_sidecar(dict(meta), "inference")
        assert (meta["request_seq"], meta["position"]) == (
            meta["epoch"], meta["epoch"])
    scanned = captures.scan_captures(capture_dir, "inference")
    assert [m["epoch"] for m, _ in scanned] == [0, 1, 2]


def test_write_toy_captures_explicit_phase_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(synthetic, "torch", _FakeTorch)
    records = synthetic.write_toy_captures(_toy_world(n_epochs=2),
                                           str(tmp_path / "caps"), "training",
                                           rows_per_epoch=1, phase="bwd")
    assert [m["phase"] for m, _ in records] == ["bwd", "bwd"]
