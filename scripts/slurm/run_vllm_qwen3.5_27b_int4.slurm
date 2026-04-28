#!/bin/bash
#SBATCH --job-name=vllm_qwen35_27b_int4
#SBATCH --output=slurm-%j-vllm.out
#SBATCH --error=slurm-%j-vllm.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:a6000:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source .venv/bin/activate

export MODEL_NAME="Qwen/Qwen3.5-27B-GPTQ-Int4"
export VLLM_HOST="127.0.0.1"
export VLLM_PORT="8000"
export OPENAI_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

python -m vllm.entrypoints.openai.api_server \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --model "${MODEL_NAME}" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096
