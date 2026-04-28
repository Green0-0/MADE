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

export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"

uv run scripts/run_baseline_experiments.py \
  --agent-configs chemeleon_llm_planner \
  --systems-file ./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json \
  --max-systems 10 \
  --budget 50 \
  --num-episodes 3 \
  --use-wandb \
  ++agent.planner.llm_config.model="openai/Qwen/Qwen3.5-27B-GPTQ-Int4" \
  ++agent.planner.llm_config.base_url="${OPENAI_BASE_URL}"
