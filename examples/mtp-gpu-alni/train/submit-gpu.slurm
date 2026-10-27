#!/bin/bash
#SBATCH --job-name=mtp-fit-gpu
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=mtp-fit-gpu-%j.out

set -euo pipefail

: "${GPMAIN:?Set GPMAIN to the BUILD_MODE=cuda imrmlp-cu executable}"
: "${MTP_POTENTIAL:?Set MTP_POTENTIAL to a classic linear MTP file}"

GPMAIN_PATH="$(command -v "$GPMAIN")"
POTENTIAL_PATH="$(realpath "$MTP_POTENTIAL")"

EXAMPLE_DIR="${MTP_EXAMPLE_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ ! -f "$EXAMPLE_DIR/index.map" ]]; then
    echo "Submit from the mtp-gpu-alni directory or set MTP_EXAMPLE_DIR" >&2
    exit 2
fi
RUN_ROOT="${MTP_RUN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}/mtp-gpu-runs}"
RUN_DIR="${RUN_ROOT}/train-${SLURM_JOB_ID:-manual}"

mkdir -p "$RUN_DIR"
cp "$EXAMPLE_DIR/POSCAR.L12-Ni3Al-displaced" "$EXAMPLE_DIR/index.map" "$RUN_DIR/"
ln -s "$POTENTIAL_PATH" "$RUN_DIR/potential.mtp"
cd "$RUN_DIR"

"$GPMAIN_PATH" calc_efs \
    --input=POSCAR.L12-Ni3Al-displaced \
    --output=train-reference.cfg \
    --input-format=vasp \
    --output-format=mlpcfg \
    --index=index.map \
    --potfile=potential.mtp \
    --pottype=mtp \
    --device=cuda \
    --gpu-id=0

python3 "$EXAMPLE_DIR/train/make_synthetic_training_set.py" \
    train-reference.cfg train.cfg

"$GPMAIN_PATH" trainnlpot \
    --settings="$EXAMPLE_DIR/train/fit-gpu.ini" \
    --do-lbfgs 2>&1 | tee train.log
