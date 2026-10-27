#!/bin/sh
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: GPL-2.0-only
# Adapted for the IMR-MLP public source layout, 2026-09-22.

set -eu

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 TRANSPORT_LMP MTP_WITH_SPECIES_ZERO [BASELINE_STYLE] [BASELINE_LMP]" >&2
  exit 2
fi
lmp=$1
potential=$2
baseline=${3-mlip/opt}
baseline_lmp=${4-$lmp}
test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/mlip-pair-parity.XXXXXX")
trap 'rm -rf -- "$work"' EXIT HUP INT TERM

for entry in "baseline:$baseline:$baseline_lmp" "transport:mlip/transport:$lmp"; do
  name=${entry%%:*}
  remainder=${entry#*:}
  style=${remainder%%:*}
  executable=${remainder#*:}
  OMP_NUM_THREADS=1 "$executable" -log "$work/$name.log" -screen none \
    -var style "$style" -var potential "$potential" \
    -var dump_file "$work/$name.dump" -in "$test_dir/in.pair_parity"
done

python3 "$test_dir/verify_pair_parity.py" \
  "$work/baseline.log" "$work/baseline.dump" \
  "$work/transport.log" "$work/transport.dump"
