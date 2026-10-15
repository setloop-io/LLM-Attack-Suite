#!/usr/bin/env python3
"""python -m attacker — unified attacker framework dispatcher.

    python -m attacker --list-attacks [--mode training]
    python -m attacker --self-test
    python -m attacker --mode MODE --attack NAME [attack flags...]
    python -m attacker --mode MODE --attack NAME --help   (torch-less)
    python -m attacker --scorecard ART.json [ART2.json ...] [--output SC.json]
    python -m attacker --coverage ART.json [ART2.json ...] [--output M.json]
    python -m attacker --report ART.json [ART2.json ...] [--stdout]
    python -m attacker --benchmark quick|full --target-config C --workdir D
    python -m attacker --benchmark-grade WORKDIR       (re-grade, torch-free)
    python -m attacker --recon --model M | --node N | --endpoint URL

Mode selects the wire-capture schema (attacker.captures.SCHEMAS) and the
available attack set: training adds gradient / membership / active-update
attacks, inference adds serving / output-side attacks.

--scorecard aggregates attack artifact JSONs into one scorecard + text
report (attacker/scorecard.py; exit 1 when any cell FAILs the gate).

--report renders markdown execution reports for artifacts (attacker/
report.py; evidence integrity included when the run was sealed). It also
piggybacks on attack runs: `--attack NAME ... --output out.json --report`
writes out.json.report.md after a successful run.

--benchmark runs the standard environment battery (attacker/benchmark.py,
benchmark.v1) against a target config and writes the environment report
card; --benchmark-grade re-grades an existing workdir torch-free.

--recon fingerprints a model / fleet node / inference endpoint before
attack selection (attacker/recon.py; all probes timeout-guarded).

Every attack module is imported lazily and MUST keep --help torch-less
(repo rule, CI-enforced): torch may only be touched inside run().
"""

import argparse
import sys

from .attacks import ATTACKS, for_mode, load

MODES = ("training", "inference")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="python -m attacker",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=MODES, default=None,
                    help="surface the framework operates on: selects the "
                         "capture schema and the available attack set")
    ap.add_argument("--attack", choices=sorted(ATTACKS), default=None,
                    help="attack to run (remaining args go to its parser)")
    ap.add_argument("--list-attacks", action="store_true",
                    help="list attacks (optionally filtered by --mode)")
    ap.add_argument("--self-test", action="store_true",
                    help="run every module's pure-python fixture checks "
                         "(torch sections run when torch is present)")
    return ap


def self_test():
    ok = True
    modules = ["attacker.solve_primitives", "attacker.dtw", "attacker.ica",
               "attacker.captures", "attacker.surfaces", "attacker.artifacts",
               "attacker.checkpoint", "attacker.evidence", "attacker.gate",
               "attacker.scorecard", "attacker.integrity_gate",
               "attacker.threat_model", "attacker.attack_metadata",
               "attacker.coverage", "attacker.profiles",
               "attacker.hwprobe", "attacker.remote", "attacker.mitm",
               "attacker.defenses", "attacker.transcripts",
               "attacker.splitbench", "attacker.report",
               "attacker.benchmark", "attacker.benchmark_integrity",
               "attacker.recon", "attacker.agent_harness",
               "attacker.secure_compute",
               "attacker.secure_compute_process_adapter",
               "attacker.secure_compute_reference_adapter",
               "attacker.attacks"]
    from importlib import import_module
    targets = [(m, import_module(m)) for m in modules]
    targets += [(getattr(load(n), "__name__", n), load(n))
                for n in sorted(ATTACKS)]
    for modname, mod in targets:
        fn = getattr(mod, "self_test", None)
        if fn is None:
            continue
        print(f"\n== {modname} ==")
        try:
            ok = (fn() == 0) and ok
        except Exception as e:
            msg = str(e)
            if isinstance(e, RuntimeError) and "requires" in msg and "install" in msg:
                print(f"  [SKIP] {modname}.self_test skipped: {e}")
                continue
            print(f"  [FAIL] {modname}.self_test raised: {e}")
            ok = False
    print("\nFRAMEWORK SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def list_attacks(mode):
    for name in sorted(ATTACKS):
        mod = load(name)
        if mode and mode not in mod.MODES:
            continue
        labels = ("labeled" if mod.REQUIRES_LABELS is True else
                  "label-free" if mod.REQUIRES_LABELS is False
                  else str(mod.REQUIRES_LABELS))
        print(f"  {name:20s} [{mod.EXPERIMENT_ID}] "
              f"modes={','.join(mod.MODES)} labels={labels}")
        print(f"      {mod.DESCRIPTION}")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    for flag, modname, fn in (("--scorecard", "attacker.scorecard", "main"),
                              ("--benchmark-grade", "attacker.benchmark",
                               "grade_main"),
                              ("--benchmark-integrity-grade",
                               "attacker.benchmark_integrity", "grade_main"),
                              ("--benchmark-integrity",
                               "attacker.benchmark_integrity", "main"),
                              ("--benchmark", "attacker.benchmark", "main"),
                              ("--coverage", "attacker.coverage", "main"),
                              ("--profiles", "attacker.profiles", "main"),
                              ("--recon", "attacker.recon", "main")):
        if flag in argv:
            from importlib import import_module
            return getattr(import_module(modname), fn)(
                argv[argv.index(flag) + 1:])
    if "--report" in argv and "--attack" not in argv:
        from .report import main as report_main
        return report_main(argv[argv.index("--report") + 1:])
    ap = build_parser()
    args, rest = ap.parse_known_args(argv)

    if args.self_test:
        return self_test()
    if args.list_attacks or (args.attack is None):
        if args.attack is None and not args.list_attacks:
            ap.print_help()
            print()
        return list_attacks(args.mode)

    mod = load(args.attack)
    if args.mode and args.mode not in mod.MODES:
        print(f"error: attack '{args.attack}' does not apply to mode "
              f"'{args.mode}' (modes: {', '.join(mod.MODES)})",
              file=sys.stderr)
        return 2
    want_report = "--report" in rest
    if want_report:
        rest = [a for a in rest if a != "--report"]
    sub = mod.build_parser()
    sub.prog = f"python -m attacker --mode {args.mode or '<mode>'} " \
               f"--attack {args.attack}"
    sub_args = sub.parse_args(rest)
    sub_args.mode = args.mode or (mod.MODES[0] if len(mod.MODES) == 1
                                  else None)
    if sub_args.mode is None:
        print("error: --mode is required for this attack "
              f"(modes: {', '.join(mod.MODES)})", file=sys.stderr)
        return 2
    rc = mod.run(sub_args)
    if want_report and rc == 0 and getattr(sub_args, "output", None):
        try:
            from .report import write_report
            write_report(sub_args.output)
        except Exception as e:
            print(f"[report] post-run report failed: {e}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
