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
work=$(mktemp -d "${TMPDIR:-/tmp}/mlip-pairwise-heat.XXXXXX")
trap 'rm -rf -- "$work"' EXIT HUP INT TERM

OMP_NUM_THREADS=1 "$lmp" -log "$work/pairwise.log" -screen none \
  -var potential "$test_dir/simple_spline.mtp" \
  -in "$test_dir/in.pairwise_heat"
python3 "$test_dir/verify_pairwise_heat.py" "$work/pairwise.log"
