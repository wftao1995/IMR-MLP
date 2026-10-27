#!/bin/sh
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: GPL-2.0-only
# Adapted for the IMR-MLP public source layout, 2026-09-22.

set -eu

usage()
{
  echo "Usage: $0 [--with-omp] LAMMPS_SRC" >&2
  echo "       $0 --uninstall LAMMPS_SRC" >&2
  exit 2
}

mode=install
with_omp=no
case "${1-}" in
  --with-omp) with_omp=yes; shift ;;
  --uninstall) mode=uninstall; shift ;;
esac
[ "$#" -eq 1 ] || usage

package_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
target=$1
[ -f "$target/atom.h" ] && [ -f "$target/pair.h" ] || {
  echo "ERROR: '$target' is not a LAMMPS src directory" >&2
  exit 1
}
target=$(CDPATH= cd -- "$target" && pwd)
manifest="$target/.mlip_transport_manifest"

base_files="
mlip_pair_metadata.h
mtp_metadata.h
picojson.h
mlip_transport_provider.h
mlip_transport_tally.h
radial_basis_transport.h
pair_mlip_transport.h
pair_mlip_transport.cpp
pair_mlip_transport_spline.h
pair_mlip_transport_spline.cpp
compute_mlip_heat_flux.h
compute_mlip_heat_flux.cpp
compute_mlip_heat_current.h
compute_mlip_heat_current.cpp"
omp_files="pair_mlip_transport_omp.h pair_mlip_transport_omp.cpp"

hash_file()
{
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

if [ "$mode" = uninstall ]; then
  [ -f "$manifest" ] || {
    echo "ERROR: no transport manifest in '$target'" >&2
    exit 1
  }
  while read -r expected file; do
    [ -n "$file" ] || continue
    case "$file" in */*|.*) echo "ERROR: unsafe manifest entry '$file'" >&2; exit 1;; esac
    [ -f "$target/$file" ] || {
      echo "ERROR: installed file '$file' is missing; refusing partial uninstall" >&2
      exit 1
    }
    actual=$(hash_file "$target/$file")
    [ "$actual" = "$expected" ] || {
      echo "ERROR: '$file' was modified; refusing to remove it" >&2
      exit 1
    }
  done < "$manifest"
  while read -r expected file; do
    [ -n "$file" ] && rm -f -- "$target/$file"
  done < "$manifest"
  rm -f -- "$manifest"
  echo "Removed the MTP transport interface from $target"
  exit 0
fi

[ ! -e "$manifest" ] || {
  echo "ERROR: transport interface is already installed in '$target'" >&2
  exit 1
}
if [ "$with_omp" = yes ] &&
   [ ! -f "$target/thr_omp.h" ] && [ ! -f "$target/OPENMP/thr_omp.h" ]; then
  echo "ERROR: --with-omp requires the LAMMPS OPENMP package sources" >&2
  exit 1
fi

files=$base_files
[ "$with_omp" = no ] || files="$files $omp_files"
source_file()
{
  case "$1" in
    mlip_pair_metadata.h) printf '%s' "$package_dir/../$1" ;;
    mtp_metadata.h|picojson.h) printf '%s' "$package_dir/../../../tools/$1" ;;
    *) printf '%s' "$package_dir/$1" ;;
  esac
}
for file in $files; do
  [ -f "$(source_file "$file")" ] || {
    echo "ERROR: package file '$file' is missing" >&2
    exit 1
  }
  [ ! -e "$target/$file" ] || {
    echo "ERROR: '$target/$file' already exists; nothing was overwritten" >&2
    exit 1
  }
done

: > "$manifest"
for file in $files; do
  cp -- "$(source_file "$file")" "$target/$file"
  printf '%s %s\n' "$(hash_file "$target/$file")" "$file" >> "$manifest"
done

echo "Installed the MTP transport interface in $target"
if [ "$with_omp" = yes ]; then
  echo "Included pair_style mlip/transport/omp (build LAMMPS with PKG_OPENMP=on)."
fi
echo "Reconfigure/rebuild LAMMPS so its source glob and style registries are regenerated."
