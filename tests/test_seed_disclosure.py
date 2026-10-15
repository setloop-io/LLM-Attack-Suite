"""seed-disclosure attack module (attacker/attacks/seed_disclosure.py):
the presence-vs-impact measurement end to end — digest verification
(MATCH / NO-MATCH), probe-lift over a random-init control (planted vs
null world), seed precedence, fail-closed ValueErrors, the JSON CLI path,
and the torch-only .pt bundle guard.

This module is genuinely torch-free for JSON bundles, so the whole
behavioral tier runs here; only the torch-bundle load branch (lines
105-106) stays uncovered without torch.
"""

import hashlib
import json

import pytest

from attacker import attacks
from attacker.attacks import seed_disclosure as sd


def _parse(*argv):
    return sd.build_parser().parse_args(list(argv))


def _bundle_file(tmp_path, bundle, name="bundle.json"):
    p = tmp_path / name
    p.write_text(json.dumps(bundle))
    return str(p)


def _enc_from_seed(seed):
    """Deterministic 'reconstructed encoder' stand-in: bytes derived from
    the disclosed seed, like a seeded init reconstruction."""
    return hashlib.sha256(f"encoder-init:{seed}".encode()).digest()


def test_contract_and_registry():
    assert sd.EXPERIMENT_ID == "seed_disclosure"
    assert sd.MODES == ("training",)
    assert "insider" in sd.REQUIRES_LABELS
    assert isinstance(sd.DESCRIPTION, str)
    attacks.validate_contract(sd, "seed-disclosure")
    assert attacks.ATTACKS.get("seed-disclosure") == \
        "attacker.attacks.seed_disclosure"
    assert attacks.load("seed-disclosure") is sd


def test_bytes_handles_every_branch():
    assert sd._bytes(b"ab") == b"ab"
    assert sd._bytes(bytearray(b"ab")) == b"ab"
    assert sd._bytes("ab") == b"ab"

    class Detachable:
        def detach(self): return self
        def cpu(self): return self
        def contiguous(self): return self
        def numpy(self): return self
        def tobytes(self): return b"\x01\x02"

    class RawBuffer:
        def tobytes(self): return b"\x03\x04"

    assert sd._bytes(Detachable()) == b"\x01\x02"
    assert sd._bytes(RawBuffer()) == b"\x03\x04"
    assert sd._bytes({"b": 1, "a": 2}) == sd._bytes({"a": 2, "b": 1})
    assert sd._bytes({"a": 2, "b": 1}) == b'{"a":2,"b":1}'


def test_digest_is_sha256_of_canonical_bytes():
    assert sd._digest(b"encoder-from-seed") == \
        hashlib.sha256(b"encoder-from-seed").hexdigest()
    assert sd._digest("encoder-from-seed") == sd._digest(b"encoder-from-seed")


def test_match_world_presence_and_impact():
    """The protocol discloses seed 7; reconstruction from that seed matches
    the released digest (presence), and the probe trained on the
    reconstructed encoder beats the random-init control (impact)."""
    enc = _enc_from_seed(7)
    r = sd.run_measurement({
        "disclosed_seed": 7,
        "reconstruct_encoder": _enc_from_seed,
        "released_init_digest": "sha256:" + sd._digest(enc).upper(),
        "run_inversion_probe": lambda e: 0.4 if e == enc else 0.1,
        "random_init_encoder": b"random-init",
    })
    assert r["init_match"] is True and r["verdict"] == "MATCH"
    assert r["seed"] == 7
    assert r["init_digest"] == sd._digest(enc)
    assert r["known_init_probe"] == 0.4 and r["random_init_probe"] == 0.1
    assert r["recovery_delta"] == pytest.approx(0.3)


def test_null_world_probe_at_control():
    """Null world: the disclosed seed does NOT reconstruct the released
    init (digest mismatch) and the probe shows no lift over control."""
    r = sd.run_measurement({
        "disclosed_seed": 999,
        "reconstruct_encoder": _enc_from_seed,
        "released_init_digest": sd._digest(_enc_from_seed(7)),
        "run_inversion_probe": lambda e: 0.1,
        "random_init": b"random-init",
    })
    assert r["init_match"] is False and r["verdict"] == "NO-MATCH"
    assert r["recovery_delta"] == pytest.approx(0.0)


def test_precomputed_probe_scores_path():
    enc = b"encoder-from-seed"
    r = sd.run_measurement({"disclosed_seed": 7, "initial_encoder": enc,
                            "released_init_digest": sd._digest(enc),
                            "known_init_probe": 0.4,
                            "random_init_probe": 0.1})
    assert r["verdict"] == "MATCH"
    assert r["recovery_delta"] == pytest.approx(0.3)


def test_seed_precedence_override_beats_bundle_and_metadata():
    enc = _enc_from_seed(3)
    bundle = {"disclosed_seed": 1, "metadata": {"disclosed_seed": 2,
                                                "seed": 0},
              "initial_encoder": enc,
              "released_init_digest": sd._digest(enc),
              "known_init_probe": 0.2, "random_init_probe": 0.2}
    assert sd.run_measurement(bundle, disclosed_seed=3)["seed"] == 3
    assert sd.run_measurement(bundle)["seed"] == 1
    bundle2 = {"metadata": {"disclosed_seed": 2, "seed": 0},
               "initial_encoder": enc,
               "released_init_digest": sd._digest(enc),
               "known_init_probe": 0.2, "random_init_probe": 0.2}
    assert sd.run_measurement(bundle2)["seed"] == 2
    bundle3 = {"metadata": {"seed": 0}, "initial_encoder": enc,
               "released_init_digest": sd._digest(enc),
               "known_init_probe": 0.2, "random_init_probe": 0.2}
    assert sd.run_measurement(bundle3)["seed"] == 0


def test_metadata_init_digest_fallback():
    enc = b"enc"
    r = sd.run_measurement({"disclosed_seed": 1, "initial_encoder": enc,
                            "metadata": {"init_digest": sd._digest(enc)},
                            "known_init_probe": 0.5,
                            "random_init_probe": 0.5})
    assert r["init_match"] is True


def test_missing_seed_encoder_digest_and_control_all_raise():
    enc = b"enc"
    with pytest.raises(ValueError, match="no disclosed seed"):
        sd.run_measurement({"initial_encoder": enc,
                            "released_init_digest": sd._digest(enc)})
    with pytest.raises(ValueError, match="reconstruct_encoder"):
        sd.run_measurement({"disclosed_seed": 1,
                            "released_init_digest": "x"})
    with pytest.raises(ValueError, match="no released_init_digest"):
        sd.run_measurement({"disclosed_seed": 1, "initial_encoder": enc})
    with pytest.raises(ValueError, match="random_init"):
        sd.run_measurement({"disclosed_seed": 1, "initial_encoder": enc,
                            "released_init_digest": sd._digest(enc),
                            "run_inversion_probe": lambda e: 0.1})


def test_run_json_bundle_end_to_end(tmp_path, capsys):
    enc = _enc_from_seed(7)
    bundle = {"disclosed_seed": 7, "initial_encoder": "enc-from-seed-7",
              "released_init_digest": sd._digest("enc-from-seed-7"),
              "known_init_probe": 0.4, "random_init_probe": 0.1}
    out = tmp_path / "sd.json"
    args = _parse("--bundle", _bundle_file(tmp_path, bundle),
                  "--output", str(out))
    assert sd.run(args) == 0
    art = json.loads(out.read_text())
    (res,) = art["results"]
    assert res["experiment"] == sd.EXPERIMENT_ID
    assert res["verdict"] == "MATCH"
    assert res["recovery_delta"] == pytest.approx(0.3)
    printed = json.loads(capsys.readouterr().out)
    assert printed == res


def test_run_without_output_still_prints(tmp_path, capsys):
    bundle = {"disclosed_seed": 1, "initial_encoder": "e",
              "released_init_digest": sd._digest("e"),
              "known_init_probe": 0.2, "random_init_probe": 0.2}
    args = _parse("--bundle", _bundle_file(tmp_path, bundle))
    assert sd.run(args) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "MATCH"


def test_pt_bundle_fails_closed_without_torch(tmp_path):
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("the .pt guard only fires when torch is absent")
    p = tmp_path / "bundle.pt"
    p.write_bytes(b"not-really-a-torch-bundle")
    with pytest.raises(SystemExit, match="require torch"):
        sd.run(_parse("--bundle", str(p)))


def test_self_test_passes():
    assert sd.self_test() == 0
