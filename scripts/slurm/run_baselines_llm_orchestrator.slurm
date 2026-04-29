#!/bin/bash
#SBATCH --job-name=made_llm_orch
#SBATCH --output=slurm-%j-llm-orch.out
#SBATCH --error=slurm-%j-llm-orch.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:a6000:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source .venv/bin/activate

export CHEMELEON_CKPT_DIR="${HOME}/.cache/chemeleon_dng/ckpts"
export CHEMELEON_HF_REPO="${CHEMELEON_HF_REPO:-G-reen/chemeleon_reupload}"
export CHEMELEON_HF_DIR="${CHEMELEON_HF_DIR:-${HOME}/.cache/chemeleon_dng/hf}"
export CHEMELEON_TASK="${CHEMELEON_TASK:-csp}"
find "${CHEMELEON_CKPT_DIR}" -name "*.tar.gz" -size 0 -print -delete 2>/dev/null || true

find_chemeleon_ckpt() {
  python - <<'PY'
import os
from pathlib import Path
import sys

base = Path(os.environ["CHEMELEON_CKPT_DIR"])
task = os.environ.get("CHEMELEON_TASK", "csp").lower()
patterns = ["**/*.ckpt", "**/*.pt", "**/*.pth"]
matches = []
for pattern in patterns:
    matches.extend(base.glob(pattern))

matches = [m for m in matches if m.is_file()]
task_matches = [m for m in matches if task in m.name.lower()]
selected = task_matches or matches
if not selected:
    sys.exit(1)

ckpt_path = sorted(selected, key=lambda p: p.stat().st_size, reverse=True)[0]
print(ckpt_path)
PY
}

if [[ -n "${CHEMELEON_CKPT_PATH:-}" && -f "${CHEMELEON_CKPT_PATH}" ]]; then
  echo "Using pre-downloaded Chemeleon checkpoint: ${CHEMELEON_CKPT_PATH}"
elif CHEMELEON_CKPT_PATH=$(find_chemeleon_ckpt); then
  export CHEMELEON_CKPT_PATH
  echo "Using existing Chemeleon checkpoint: ${CHEMELEON_CKPT_PATH}"
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
  CHEMELEON_CKPT_PATH=$(python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = os.environ["CHEMELEON_HF_REPO"]
local_dir = os.environ["CHEMELEON_HF_DIR"]
task = os.environ.get("CHEMELEON_TASK", "csp").lower()
env_file = Path(os.environ["CHEMELEON_ENV_FILE"]) if "CHEMELEON_ENV_FILE" in os.environ else Path(local_dir) / "chemeleon_ckpt_env.sh"

snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)

base = Path(local_dir)
patterns = ["**/*.ckpt", "**/*.pt", "**/*.pth"]
matches = []
for pattern in patterns:
    matches.extend(base.glob(pattern))

if not matches:
    raise RuntimeError(f"No checkpoint files found in {base}")

task_matches = [m for m in matches if task in m.name.lower()]
selected = task_matches or matches
ckpt_path = sorted(selected, key=lambda p: p.stat().st_size, reverse=True)[0]
env_file.write_text(f'export CHEMELEON_CKPT_PATH="{ckpt_path}"\n')
print(ckpt_path)
PY
  )

  export CHEMELEON_CKPT_PATH
  echo "Selected checkpoint from HF: ${CHEMELEON_CKPT_PATH}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found in PATH" >&2
  exit 1
fi

VLLM_VENV_PATH="${VLLM_VENV_PATH:-${PWD}/.vllm}"
VLLM_PYTHON="${VLLM_VENV_PATH}/bin/python"

if [[ ! -x "${VLLM_PYTHON}" ]]; then
  echo "vLLM venv not found: ${VLLM_VENV_PATH}" >&2
  exit 1
fi

"${VLLM_PYTHON}" - <<'PY'
import importlib
import sys

try:
    importlib.import_module("vllm")
except Exception as exc:
    print(f"vllm import failed in vLLM venv: {exc}", file=sys.stderr)
    raise
PY

export MODEL_NAME="QuantTrio/Qwen3.5-9B-AWQ"
export VLLM_HOST="127.0.0.1"
export VLLM_PORT="8000"
export OPENAI_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"

cleanup_vllm() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup_vllm EXIT

"${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --model "${MODEL_NAME}" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 &
VLLM_PID=$!

python - <<'PY'
import json
import time
import urllib.request

url = "http://127.0.0.1:8000/v1/models"
deadline = time.time() + 600
last_error = None

while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("data"):
                    print("vLLM ready")
                    raise SystemExit(0)
    except Exception as exc:
        last_error = exc
        time.sleep(2)

raise SystemExit(f"vLLM did not become ready: {last_error}")
PY

if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
  echo "vLLM process exited before baseline run" >&2
  exit 1
fi

uv run scripts/run_baseline_experiments.py \
  --agent-configs llm_react_orchestrator \
  --systems-file ./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json \
  --max-systems 10 \
  --budget 50 \
  --num-episodes 3 \
  --use-wandb \
  ++agent.llm_config.model="openai/QuantTrio/Qwen3.5-9B-AWQ" \
  ++agent.llm_config.base_url="${OPENAI_BASE_URL}"
