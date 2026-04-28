# Slurm scripts

These scripts run the baseline experiments from the paper on Slurm with a single A6000.

Common settings:
- LLM jobs: 1x A6000 GPU, 128GB RAM, 12 CPU cores
- Non-LLM jobs: 1x A5000 GPU, 32-64GB RAM, 8 CPU cores

Suggested usage:

1) Run baselines (one job per script):

```bash
sbatch scripts/slurm/run_baselines_chemeleon_mlip.slurm
sbatch scripts/slurm/run_baselines_llm_orchestrator.slurm
sbatch scripts/slurm/run_baselines_chemeleon_llm_random.slurm
sbatch scripts/slurm/run_baselines_random_random_random.slurm
```

Notes:
- If your cluster uses a different venv path, edit `source .venv/bin/activate`.
- The scripts `cd` to `${SLURM_SUBMIT_DIR}` so they run from the repo root.
- Chemeleon checkpoints are pre-downloaded from Figshare to `${HOME}/.cache/chemeleon_dng`.
- Override the download URL by setting `CHEMELEON_TAR_URL` or bypass download entirely with `CHEMELEON_CKPT_PATH`.
- The LLM scripts start and stop their own vLLM instance.
- The vLLM server listens on `127.0.0.1:8000` by default; adjust `OPENAI_BASE_URL`
  if your endpoint is different.
- The model id is set to `openai/Qwen/Qwen3.5-27B-GPTQ-Int4` so DSPy routes to the
  OpenAI-compatible server.
