# Reproduce the frozen validation without Git history

The v1.0 release adds an independently usable source archive and reproduction
helper. The existing selection, original `reproduce.py`, `run/` records and
`MANIFEST.sha256` are preserved byte-for-byte. Their historical commit and tag
names identify the source originally executed; the old Git objects are no
longer needed when using the helper below.

`source-snapshot.tar.gz` contains exactly the 123 source files used by the
original validation. It was exported from historical source commit
`6c071a68ae5134522d0a4cd3d7d0fa67acabdfa8`, with the same source roots as the
original `run/run.json`. All extracted files match `run/source-files.sha256`.
The uncompressed Git archive SHA-256 is also identical to the digest recorded
by the original run:

```text
1d25108661b4daa68211d785b9a7b0b39527762c118d592eece48a2421acf088
```

The gzip archive SHA-256 is:

```text
c3dd27e4d1ddf9af69519c6f82f9ff102fd8331bf3143261a561372d3b9040dc
```

`SOURCE-SNAPSHOT.json` records the export and verification details.
`SNAPSHOT.sha256` covers these new additions separately from the unchanged
historical manifest. Retain its digest with the release's independent manifest.

## Verify and run

From this directory, check both the historical records and the new additions:

```sh
shasum -a 256 -c MANIFEST.sha256
shasum -a 256 -c SNAPSHOT.sha256
python reproduce_from_snapshot.py --verify-only
```

Use a Python environment containing the original run's dependencies:
Python 3.14.6, pytest 9.1.1, NumPy 2.5.3, PyTorch 2.14.0 and
Transformers 4.57.6. The helper does not install software or download models.
The full installed distribution inventory remains in `run/packages.json`.
Then run with a new output directory:

```sh
python reproduce_from_snapshot.py \
  --output /tmp/paper-b-measurement-independent-run
```

The helper uses `source-snapshot.tar.gz` next to itself by default; `--snapshot`
can point to a separate copy. It verifies the archive, original selection and
source-manifest digests, rejects unsafe archive members, verifies all 123
extracted source files and runs the same 25 selected functions / 37 cases with
the original offline CPU environment settings. No repository path or Git
command is required. The result retains the historical source commit and tag
as provenance and adds the source archive path and digest. Existing output
directories are refused. A temporary source directory is retained for inspection.

## v1.0 verification

The new helper was executed with the recorded dependencies on 15 October 2026:
**37 passed, zero failed, zero skipped**, with identical category counts
(MV1–MV7: 3, 4, 4, 2, 3, 6, 15). All source files remained unchanged. A modified
archive was rejected before extraction or test execution. All 14 pre-existing
record files in this supplement remained unchanged.

`SNAPSHOT-VERIFICATION.json` records these checks. The fresh run is retained at
`build/release-v1.0/validation-reproduction/` in the preparation workspace.
These are repeated fixed-case measurement checks, not new leakage experiments
or validation of all attack modules. The inherited `torch-free` marker in pytest
output remains a historical reporting label; several fixtures use PyTorch.
