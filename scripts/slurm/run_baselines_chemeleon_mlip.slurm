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

uv run scripts/run_baseline_experiments.py \
  --agent-configs chemeleon_mlip_ranking_chain_filter \
  --systems-file ./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json \
  --max-systems 10 \
  --budget 50 \
  --num-episodes 3 \
  --use-wandb
