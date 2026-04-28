#!/bin/bash
#SBATCH --job-name=check_chemeleon
#SBATCH --output=slurm-%j-chemeleon-check.out
#SBATCH --error=slurm-%j-chemeleon-check.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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

CHEMELEON_TAR_URL="https://figshare.com/ndownloader/files/39307959"
CHEMELEON_TAR_FILE="${CHEMELEON_CKPT_DIR}/checkpoints.tar.gz"

if [[ ! -s "${CHEMELEON_TAR_FILE}" ]]; then
  echo "Downloading Chemeleon checkpoints..."
  curl -L --retry 5 --retry-delay 5 --fail -o "${CHEMELEON_TAR_FILE}" "${CHEMELEON_TAR_URL}"
fi

if [[ -z "$(ls -A "${CHEMELEON_CKPT_DIR}" 2>/dev/null)" ]]; then
  echo "Extracting Chemeleon checkpoints..."
  tar -xzf "${CHEMELEON_TAR_FILE}" -C "${CHEMELEON_CACHE}"
fi

python - <<'PY'
import inspect
import os
from pathlib import Path

from chemeleon_dng import sample
from chemeleon_dng.diffusion.diffusion_module import DiffusionModule
from chemeleon_dng.download_util import get_checkpoint_path

ckpt_dir = os.environ.get("CHEMELEON_CKPT_DIR")
kwargs = {}
if ckpt_dir:
    params = inspect.signature(get_checkpoint_path).parameters
    if "ckpt_dir" in params:
        kwargs["ckpt_dir"] = ckpt_dir

ckpt_path = get_checkpoint_path("csp", sample.DEFAULT_MODEL_PATH, **kwargs)
ckpt_path = Path(ckpt_path)
print(f"Checkpoint path: {ckpt_path}")
print(f"Checkpoint exists: {ckpt_path.exists()}")

dm = DiffusionModule.load_from_checkpoint(ckpt_path, map_location="cpu")
print("Chemeleon checkpoint load OK")
PY
