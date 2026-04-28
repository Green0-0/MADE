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

export CHEMELEON_CKPT_DIR="${HOME}/.cache/chemeleon_dng/ckpts"
export CHEMELEON_HF_REPO="${CHEMELEON_HF_REPO:-G-reen/chemeleon_reupload}"
export CHEMELEON_HF_DIR="${CHEMELEON_HF_DIR:-${HOME}/.cache/chemeleon_dng/hf}"

if [[ -n "${CHEMELEON_CKPT_PATH:-}" && -f "${CHEMELEON_CKPT_PATH}" ]]; then
  echo "Using pre-downloaded Chemeleon checkpoint: ${CHEMELEON_CKPT_PATH}"
elif [[ -n "$(ls -A "${CHEMELEON_CKPT_DIR}" 2>/dev/null)" ]]; then
  echo "Using existing Chemeleon checkpoints in ${CHEMELEON_CKPT_DIR}"
else
  if ! python - <<'PY'
try:
    import huggingface_hub  # noqa: F401
except Exception:
    raise SystemExit(1)
PY
  then
    uv pip install "huggingface_hub>=0.24.0"
  fi

  CHEMELEON_ENV_FILE="${CHEMELEON_HF_DIR}/chemeleon_ckpt_env.sh"
  mkdir -p "${CHEMELEON_HF_DIR}"
  python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = os.environ["CHEMELEON_HF_REPO"]
local_dir = os.environ["CHEMELEON_HF_DIR"]
env_file = Path(os.environ["CHEMELEON_ENV_FILE"]) if "CHEMELEON_ENV_FILE" in os.environ else Path(local_dir) / "chemeleon_ckpt_env.sh"

snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)

base = Path(local_dir)
patterns = ["**/*.ckpt", "**/*.pt", "**/*.pth"]
matches = []
for pattern in patterns:
    matches.extend(base.glob(pattern))

if not matches:
    raise RuntimeError(f"No checkpoint files found in {base}")

ckpt_path = sorted(matches, key=lambda p: p.stat().st_size, reverse=True)[0]
env_file.write_text(f'export CHEMELEON_CKPT_PATH="{ckpt_path}"\n')
print(f"Selected checkpoint: {ckpt_path}")
PY

  source "${CHEMELEON_ENV_FILE}"
fi

python - <<'PY'
import inspect
import os
from pathlib import Path

from chemeleon_dng import sample
from chemeleon_dng.diffusion.diffusion_module import DiffusionModule
from chemeleon_dng.download_util import get_checkpoint_path

ckpt_path_override = os.environ.get("CHEMELEON_CKPT_PATH")
if ckpt_path_override:
    ckpt_path = Path(ckpt_path_override)
else:
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
