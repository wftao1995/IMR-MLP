#!/bin/sh
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: GPL-2.0-only
# Adapted for the IMR-MLP public source layout, 2026-09-22.

set -eu

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 LMP" >&2
  exit 2
fi
lmp=$1
test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/mlip-spline-test.XXXXXX")
trap 'rm -rf -- "$work"' EXIT HUP INT TERM

OMP_NUM_THREADS=1 "$lmp" -log "$work/spline.log" -screen none \
  -var style mlip/transport/spline -var threads 1 \
  -var potential "$test_dir/simple_spline.mtp" \
  -var dump_file "$work/spline.dump" -in "$test_dir/in.transport"
python3 "$test_dir/verify_lammps.py" "$work/spline"
