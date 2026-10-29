# Paired benchmarks for the standalone runtime

The scripts in this directory measure the standalone runtime against a baseline
pair style. They are maintenance tooling for validating and publishing the
runtime; deployment is described in [the standalone guide](../README.md).

`run_benchmark.py` consumes a JSON configuration with `baseline`, `candidate`,
and `cases`. Each target supplies `lmp`, optional `plugin`, its `pair` command, an
optional argv `launcher` and environment overrides. Each case supplies an input
template, fixed warmup/steps, optional head label, input files for hashing and
template variables. Paths resolve relative to the configuration.

`prepare_cases.py --models MODEL_DIRECTORY --output NEW_DIRECTORY` creates
deterministic 256/2048/16384-atom synthetic binary structures, model
copies/hashes, static-force input and CPU/CUDA configuration examples. Optional
`--real-xyz FILE` adds the first InSe extxyz frame with a recorded rigid rotation
and source hash. These static timings do not measure production MD stability.

```json
{
  "repetitions": 5,
  "baseline": {
    "lmp": "/path/to/lmp", "plugin": "/path/to/baseline.so",
    "pair": "pair_style mlip/native device cpu precision fp64 pottype mtp_m",
    "env": {"OMP_NUM_THREADS": "1"}
  },
  "candidate": {
    "lmp": "/path/to/lmp", "plugin": "/tmp/mtp-cpu/mtp_standalone_plugin.so",
    "pair": "pair_style mlip/opt", "env": {"OMP_NUM_THREADS": "1"}
  },
  "cases": [{
    "name": "q2-2048", "head": "mtp_m_q2", "input": "in.example",
    "warmup": 100, "steps": 1000,
    "files": ["/path/to/model.mtp", "/path/to/structure.data"],
    "variables": {"MODEL": "/path/to/model.mtp", "DATA": "/path/to/structure.data", "ELEMENTS": "Al Ni"}
  }]
}
```

From the exported package root, run
`python3 standalone/benchmarks/run_benchmark.py --config config.json --work RESULTS`.
Inside this checkout use `LAMMPS/USER-MLIP/standalone/benchmarks/run_benchmark.py`.

The runner alternates baseline/candidate order, uses fresh processes and keeps
at least five paired samples, loop-time medians/ranges, binary/plugin/input
hashes and raw logs. It also reports the geometric mean speedup and a
deterministic paired-bootstrap 95% interval; an interval spanning 1 means the
direction is unresolved. Freeze pilot-selected parameters before collection, keep
hardware allocation and CPU binding equal, and benchmark representative real
models separately from these tiny correctness fixtures. CPU MTP-M uses an
actual supporting baseline, such as native IMR-MLP; old CPU `mlip/opt` cannot serve
as a nonlinear baseline. GPU uses the frozen supporting gpudirect version.

To isolate host radial arithmetic, compile `radial_micro.cpp` with
the same optimization flags for both versions (paths below start at the package
root):

```sh
c++ -O3 -DNDEBUG -std=c++17 -I mlip-core \
  standalone/benchmarks/radial_micro.cpp -o radial_micro
./radial_micro chebyshev 8 10000000
# Override MTP_RADIAL_HEADER with a quoted frozen header path and add
# -I mlip-core when building a baseline header outside the source directory.
```

The probe reports nanoseconds per evaluation and an observable checksum. Its
PolyB matrices are synthetic contraction inputs; native numerical validation is
provided by the radial oracle tests. Microbenchmarks do not replace paired
LAMMPS throughput measurements. Keep compilation and correctness jobs out of
timing runs, retain raw samples, and report uncertainty and unavailable devices
instead of treating noisy or missing measurements as a pass.
