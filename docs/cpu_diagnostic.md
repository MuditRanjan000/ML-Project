# CPU Diagnostic Scope

`scripts/cpu_diagnostic.py` is a diagnostic-only, local CPU smoke test. It uses
ResNet-18, Recipe A, seed 42, two epochs, 512 frozen training-split samples,
128 frozen validation-split samples, and 128 clean CIFAR-100 test samples.

It verifies the persisted frozen split, validation-selected `best_model.pt`
metadata, and disk-backed clean prediction records. It does **not** load
CIFAR-100-C, create synthetic corruption records, or generate the canonical
`runs`, `corruption_cells`, and `reliability_bins` tables. The missing full
tables are therefore an expected clean-only diagnostic limitation, not an
official experiment result or completed evaluation run.

All diagnostic outputs belong under `results/diagnostics/` and remain outside
Git. Official 12-run evaluation continues to use the contract evaluator and
requires all 75 real CIFAR-100-C cells.
