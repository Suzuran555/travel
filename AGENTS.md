# Local Development Conventions

## UrbanTrip Validation

When the user asks to "run tests", "test", or "validate" an UrbanTrip change,
that means a full relevant testcase run followed by the official evaluation
(`eval_tpc.py`). Unit tests, `py_compile`, and `git diff --check` are sanity
checks only. They must not be described as the requested test run or as final
validation.

For full UrbanTrip runs, activate the project environment first:

```bash
conda activate chinatravel
```

Do not modify evaluation code when running or comparing evaluations. Preserve
immutable result snapshots before a rerun so timeout or implementation
comparisons use per-case results from the correct run.
