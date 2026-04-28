# Slurm scripts

These scripts run the baseline experiments from the paper on Slurm with a single A6000.

Common settings:
- 1x A6000 GPU
- 128GB RAM
- 12 CPU cores

Suggested usage:

1) Start vLLM for the Qwen model (keep running in a separate job or tmux):

```bash
sbatch scripts/slurm/run_vllm_qwen3.5_27b_int4.slurm
```

2) Run baselines (one job per script):

```bash
sbatch scripts/slurm/run_baselines_chemeleon_mlip.slurm
sbatch scripts/slurm/run_baselines_llm_orchestrator.slurm
sbatch scripts/slurm/run_baselines_chemeleon_llm_random.slurm
sbatch scripts/slurm/run_baselines_random_random_random.slurm
```

Notes:
- If your cluster uses a different venv path, edit `source .venv/bin/activate`.
- The scripts `cd` to `${SLURM_SUBMIT_DIR}` so they run from the repo root.
- The vLLM server listens on `127.0.0.1:8000` by default; adjust `OPENAI_BASE_URL`
  in the LLM scripts if your endpoint is different.
- The model id is set to `openai/Qwen/Qwen3.5-27B-GPTQ-Int4` so DSPy routes to the
  OpenAI-compatible server.
