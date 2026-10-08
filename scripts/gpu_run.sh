#!/bin/bash
# Run a python script on one GH200 (interactive partition). Usage: scripts/gpu_run.sh <minutes> <script.py> [args]
MIN=${1:-20}; shift
cd /u/yzhang116/field_level_PNG
srun --account=bdne-dtai-gh --partition=ghx4-interactive --gpus=1 --cpus-per-task=8 --mem=64g --time=00:${MIN}:00 \
  bash -c "source env/setup_env.sh; unset JAX_PLATFORMS; python $*"
