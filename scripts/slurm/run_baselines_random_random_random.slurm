#!/bin/bash
#SBATCH --job-name=made_rand_rand_rand
#SBATCH --output=slurm-%j-rand-rand-rand.out
#SBATCH --error=slurm-%j-rand-rand-rand.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:a6000:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source .venv/bin/activate

uv run scripts/run_baseline_experiments.py \
  --agent-configs random_generator_baseline \
  --systems-file ./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json \
  --max-systems 10 \
  --budget 50 \
  --num-episodes 3 \
  --use-wandb
