#!/usr/bin/env python3
"""Hardware-probe tier detection for the Phase B systems/hardware seam.

Pure python, torch-free import (repo rule): pynvml / torch are only touched
inside the probe functions, never at module import. Phase B attacks
(attacker/attacks/gpu_memory.py, ...) degrade through an ordered tier list:

  1. "pynvml"     NVIDIA Management Library bindings — memory-usage series,
                  per-process attribution when the driver exposes it
  2. "torch-cuda" torch.cuda allocator counters + uninitialized-allocation
                  read probes (requires torch AND a visible CUDA device)
  3. "sim"        pure-python simulated buffer pool (the --toy tier; always
                  available, a machinery check, not a security claim)

detect_tiers() reports which REAL tiers this host serves; "sim" is always
available and therefore never listed.
"""

TIERS = ("pynvml", "torch-cuda")


def pynvml_available():
    """True when pynvml imports and initializes against a live driver."""
    try:
        import pynvml
        try:
            pynvml.nvmlInit()
        except Exception:
            return False
        try:
            return pynvml.nvmlDeviceGetCount() > 0
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except ImportError:
        return False


def torch_cuda_available():
    """True when torch imports and a CUDA device is visible."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def detect_tiers():
    """Ordered list of available REAL probe tiers (subset of TIERS)."""
    tiers = []
    if pynvml_available():
        tiers.append("pynvml")
    if torch_cuda_available():
        tiers.append("torch-cuda")
    return tiers


def nvml_memory_series(device_index=0, samples=20, interval_s=0.05):
    """Sample used-memory bytes via pynvml; per-process attribution when the
    driver exposes it. Returns {"used_bytes": [...], "mem_util_pct": [...],
    "per_process": [...]}; entries are None where the device does not
    implement the query (unified-memory parts such as GB10 raise
    NVMLError_NotSupported for nvmlDeviceGetMemoryInfo — the utilization
    percentage and per-process queries still work there).
    Raises ImportError/RuntimeError with a descriptive message when the
    tier is unavailable — callers degrade, never crash."""
    try:
        import pynvml
    except ImportError as e:
        raise ImportError("hwprobe: pynvml tier requested but pynvml is not "
                          "installed (pip install pynvml / the 'gpu' extra)") \
            from e
    import time
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        used, mem_pct, per_proc = [], [], []
        for _ in range(samples):
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used.append(int(info.used))
            except Exception:
                used.append(None)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_pct.append(int(util.memory))
            except Exception:
                mem_pct.append(None)
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                per_proc.append({int(p.pid): int(p.usedGpuMemory or 0)
                                 for p in procs})
            except Exception:
                per_proc.append(None)
            time.sleep(interval_s)
        return {"device_index": device_index, "used_bytes": used,
                "mem_util_pct": mem_pct, "per_process": per_proc}
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("hardware probe tier detection (pure python):")
    tiers = detect_tiers()
    check("detect_tiers returns a subset of the known tiers",
          isinstance(tiers, list) and all(t in TIERS for t in tiers))
    check("tier helpers never raise on a tier-less host",
          pynvml_available() in (True, False)
          and torch_cuda_available() in (True, False))
    check("pynvml/torch-cuda flags agree with detect_tiers",
          ("pynvml" in tiers) == pynvml_available()
          and ("torch-cuda" in tiers) == torch_cuda_available())
    check("'sim' tier is implicit (never listed as a real tier)",
          "sim" not in tiers)
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
