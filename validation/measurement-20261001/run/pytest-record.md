# Test run — GREEN

- env: python 3.14.6 (torch present), 2026-10-01T18:59:26Z
- totals: 37 passed, 0 failed, 0 skipped of 37

## Per tier

| tier | passed | failed | skipped |
|---|---|---|---|
| torch-free | 37 | 0 | 0 |

## Slowest 10

- 5.8749s `tests/test_bisr_fidelity.py::test_prediction_process_never_loads_scorer_truth` (passed)
- 1.2628s `tests/test_scorecard.py::test_v3_pass_with_backing_control_stays_pass` (passed)
- 0.1387s `tests/test_validation_protocol.py::test_repeated_documents_do_not_create_independent_victim_runs` (passed)
- 0.1093s `tests/test_validation_protocol.py::test_complete_synthetic_report_rescores_successfully` (passed)
- 0.0532s `tests/test_validation_protocol.py::test_report_fails_closed_even_when_mutated_report_is_sealed[work-matched prefix work]` (passed)
- 0.053s `tests/test_validation_protocol.py::test_report_fails_closed_even_when_mutated_report_is_sealed[missing-missing cells]` (passed)
- 0.0521s `tests/test_validation_protocol.py::test_report_fails_closed_even_when_mutated_report_is_sealed[setup-setup matrix]` (passed)
- 0.052s `tests/test_validation_protocol.py::test_report_fails_closed_even_when_mutated_report_is_sealed[duplicate-duplicate or unexpected]` (passed)
- 0.0516s `tests/test_validation_protocol.py::test_report_fails_closed_even_when_mutated_report_is_sealed[path-prediction path/hash]` (passed)
- 0.0514s `tests/test_validation_protocol.py::test_report_fails_closed_even_when_mutated_report_is_sealed[victim-victim matrix]` (passed)
