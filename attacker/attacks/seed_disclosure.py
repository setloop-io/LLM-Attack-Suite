"""Insider attack for a protocol-disclosed initialization seed.

The arm deliberately separates *presence* from *impact*: ``init_match`` is
the security verdict, while ``recovery_delta`` compares a probe trained from
the reconstructed initial encoder with a random-init control.  Bundles may
provide ``reconstruct_encoder(seed)`` and ``run_inversion_probe(encoder)``
hooks, or precomputed ``known_init_probe`` / ``random_init_probe`` scores.
"""

import argparse
import hashlib
import json
from pathlib import Path

from .common import add_common_args

EXPERIMENT_ID = "seed_disclosure"
MODES = ("training",)
REQUIRES_LABELS = "insider (protocol-disclosed init seed)"
DESCRIPTION = ("protocol-state disclosure: reconstruct the initial encoder "
               "from a disclosed seed, verify the released digest, and "
               "measure inversion-probe lift over random initialization")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--bundle", required=True,
                    help="JSON bundle or torch bundle containing the seed")
    ap.add_argument("--disclosed-seed", default=None,
                    help="override the seed in bundle metadata")
    return ap


def _bytes(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode()
    if hasattr(value, "detach"):
        return value.detach().cpu().contiguous().numpy().tobytes()
    if hasattr(value, "tobytes"):
        return value.tobytes()
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=str).encode()


def _digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def run_measurement(bundle, disclosed_seed=None):
    """Run the seed-disclosure presence and exploit-value checks.

    Required bundle fields are ``released_init_digest`` and either
    ``reconstruct_encoder`` or ``initial_encoder``.  The latter is useful for
    capture manifests that already materialize the reconstruction.  Probe
    hooks return a scalar recovery score (percentage or fraction; the delta
    is reported in the same units).
    """
    metadata = bundle.get("metadata", {})
    seed = disclosed_seed if disclosed_seed is not None else bundle.get(
        "disclosed_seed", metadata.get("disclosed_seed", metadata.get("seed")))
    if seed is None:
        raise ValueError("bundle has no disclosed seed")
    if "reconstruct_encoder" in bundle:
        encoder = bundle["reconstruct_encoder"](seed)
    elif "initial_encoder" in bundle:
        encoder = bundle["initial_encoder"]
    else:
        raise ValueError("bundle needs reconstruct_encoder or initial_encoder")
    expected = bundle.get("released_init_digest", metadata.get("init_digest"))
    if not expected:
        raise ValueError("bundle has no released_init_digest")
    actual = _digest(encoder)
    match = actual.lower() == str(expected).lower().removeprefix("sha256:")

    probe = bundle.get("run_inversion_probe")
    if probe:
        known = float(probe(encoder))
        control_encoder = bundle.get("random_init_encoder")
        if control_encoder is None and "random_init" in bundle:
            control_encoder = bundle["random_init"]
        if control_encoder is None:
            raise ValueError("probe bundle needs random_init_encoder/random_init")
        control = float(probe(control_encoder))
    else:
        known = float(bundle["known_init_probe"])
        control = float(bundle["random_init_probe"])
    return {"experiment": EXPERIMENT_ID, "seed": seed,
            "init_digest": actual, "released_init_digest": str(expected),
            "init_match": match, "verdict": "MATCH" if match else "NO-MATCH",
            "known_init_probe": known, "random_init_probe": control,
            "recovery_delta": known - control}


def run(args):
    path = Path(args.bundle)
    if path.suffix in (".pt", ".pth"):
        try:
            import torch
            bundle = torch.load(path, map_location="cpu", weights_only=False)
        except ImportError as exc:
            raise SystemExit("seed-disclosure .pt bundles require torch") from exc
    else:
        bundle = json.loads(path.read_text())
    result = run_measurement(bundle, args.disclosed_seed)
    if args.output:
        Path(args.output).write_text(json.dumps({"results": [result]}, indent=2))
    print(json.dumps(result, sort_keys=True))
    return 0


def self_test():
    enc = b"encoder-from-seed"
    result = run_measurement({"disclosed_seed": 7, "initial_encoder": enc,
                              "released_init_digest": _digest(enc),
                              "known_init_probe": 0.4,
                              "random_init_probe": 0.1})
    ok = (result["verdict"] == "MATCH"
          and abs(result["recovery_delta"] - 0.3) < 1e-12)
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1
