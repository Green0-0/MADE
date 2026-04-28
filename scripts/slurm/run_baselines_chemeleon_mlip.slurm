#!/bin/bash
#SBATCH --job-name=made_chem_mlip
#SBATCH --output=slurm-%j-chem-mlip.out
#SBATCH --error=slurm-%j-chem-mlip.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a5000:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source .venv/bin/activate

if ! command -v curl >/dev/null 2>&1; then
  echo "curl not found in PATH" >&2
  exit 1
fi

export CHEMELEON_CKPT_DIR="${HOME}/.cache/chemeleon_dng/ckpts"
CHEMELEON_CACHE="$(dirname "${CHEMELEON_CKPT_DIR}")"
mkdir -p "${CHEMELEON_CKPT_DIR}"

CHEMELEON_TAR_URL="${CHEMELEON_TAR_URL:-https://figshare.com/ndownloader/files/54966305}"
CHEMELEON_TAR_FILE="${CHEMELEON_CKPT_DIR}/checkpoints.tar.gz"

if [[ -n "${CHEMELEON_CKPT_PATH:-}" && -f "${CHEMELEON_CKPT_PATH}" ]]; then
  echo "Using pre-downloaded Chemeleon checkpoint: ${CHEMELEON_CKPT_PATH}"
elif [[ -n "$(ls -A "${CHEMELEON_CKPT_DIR}" 2>/dev/null)" ]]; then
  echo "Using existing Chemeleon checkpoints in ${CHEMELEON_CKPT_DIR}"
else
  if [[ ! -s "${CHEMELEON_TAR_FILE}" ]]; then
    echo "Downloading Chemeleon checkpoints..."
    curl -L --retry 5 --retry-delay 5 --fail -o "${CHEMELEON_TAR_FILE}" "${CHEMELEON_TAR_URL}"
  fi

  if [[ ! -s "${CHEMELEON_TAR_FILE}" ]]; then
    echo "Chemeleon download failed or empty. Set CHEMELEON_TAR_URL or CHEMELEON_CKPT_PATH." >&2
    exit 1
  fi

  echo "Extracting Chemeleon checkpoints..."
  tar -xzf "${CHEMELEON_TAR_FILE}" -C "${CHEMELEON_CACHE}"
fi

uv run scripts/run_baseline_experiments.py \
  --agent-configs chemeleon_mlip_ranking_chain_filter \
  --systems-file ./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json \
  --max-systems 10 \
  --budget 50 \
  --num-episodes 3 \
  --use-wandb
