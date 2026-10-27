#!/bin/sh
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: GPL-2.0-only
# Adapted for the IMR-MLP public source layout, 2026-09-22.

set -eu

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 LMP MTP_WITH_SPECIES_ZERO [MPIEXEC]" >&2
  exit 2
fi

lmp=$1
potential=$2
mpiexec_command=${3-}
test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/mlip-transport-test.XXXXXX")
trap 'rm -rf -- "$work"' EXIT HUP INT TERM

run_case()
{
  prefix=$1
  style=$2
  threads=$3
  shift 3
  "$@" "$lmp" -log "$work/$prefix.log" -screen none \
    -var style "$style" -var threads "$threads" -var potential "$potential" \
    -var dump_file "$work/$prefix.dump" -in "$test_dir/in.transport"
}

run_case serial mlip/transport 1 env OMP_NUM_THREADS=1
run_case omp mlip/transport/omp 2 env OMP_NUM_THREADS=2 OMP_PROC_BIND=false
prefixes="$work/serial $work/omp"

if [ -n "$mpiexec_command" ]; then
  # The MPI launcher may contain its own flags, so it is intentionally split.
  # shellcheck disable=SC2086
  run_case mpi mlip/transport 1 $mpiexec_command -np 2 env OMP_NUM_THREADS=1
  prefixes="$prefixes $work/mpi"
fi

# shellcheck disable=SC2086
python3 "$test_dir/verify_lammps.py" $prefixes
