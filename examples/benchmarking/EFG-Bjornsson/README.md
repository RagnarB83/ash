# EFG benchmark example

This example runs ASH's bundled `EFG-Bjornsson` property set through the native
`run_benchmark` workflow. It demonstrates how the reference data, ORCA
single-points, parsed electric field gradients, and benchmark error summary fit
together.

## Requirements

- ASH installed from this checkout
- ORCA installed and available through ASH settings or the `ORCADIR` environment
  variable

## Run

From the repository root:

```sh
python examples/benchmarking/EFG-Bjornsson/run_benchmark.py
```

The script uses `BP86 def2-SVP` and one core as a compact example setup. Change
the ORCA method in the script for your study. The benchmark writes calculations
and a result summary under `benchmarks_calcs/` in the current working directory.
