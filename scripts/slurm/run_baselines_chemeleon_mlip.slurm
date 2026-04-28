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

uv run scripts/run_baseline_experiments.py \
  --agent-configs chemeleon_mlip_ranking_chain_filter \
  --systems-file ./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json \
  --max-systems 10 \
  --budget 50 \
  --num-episodes 3 \
  --use-wandb
