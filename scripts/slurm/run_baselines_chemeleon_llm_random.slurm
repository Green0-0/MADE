#!/bin/bash
#SBATCH --job-name=made_chem_llm_rand
#SBATCH --output=slurm-%j-chem-llm-rand.out
#SBATCH --error=slurm-%j-chem-llm-rand.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:a6000:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source .venv/bin/activate

find "${HOME}/.cache" -name "checkpoints.tar.gz" -size 0 -print -delete || true

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found in PATH" >&2
  exit 1
fi

python - <<'PY'
import importlib
import sys

try:
    importlib.import_module("vllm")
except Exception as exc:
    print(f"vllm import failed: {exc}", file=sys.stderr)
    raise
PY

export MODEL_NAME="Qwen/Qwen3.5-27B-GPTQ-Int4"
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

python -m vllm.entrypoints.openai.api_server \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --model "${MODEL_NAME}" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 &
VLLM_PID=$!

python - <<'PY'
import json
import time
import urllib.request

url = "http://127.0.0.1:8000/v1/models"
deadline = time.time() + 300
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
  --agent-configs chemeleon_llm_planner \
  --systems-file ./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json \
  --max-systems 10 \
  --budget 50 \
  --num-episodes 3 \
  --use-wandb \
  ++agent.planner.llm_config.model="openai/Qwen/Qwen3.5-27B-GPTQ-Int4" \
  ++agent.planner.llm_config.base_url="${OPENAI_BASE_URL}"
