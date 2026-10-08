#!/bin/bash
# Usage: scripts/sbatch_gpu.sh <name> <hours> <command...>   (one GH200, batch partition)
NAME=$1; HOURS=$2; shift 2
sbatch --job-name=$NAME --account=bdne-dtai-gh --partition=ghx4 --gpus-per-node=1 --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=120g \
  --time=${HOURS}:00:00 --no-requeue -o /u/yzhang116/field_level_PNG/logs/%x.o%j -e /u/yzhang116/field_level_PNG/logs/%x.o%j \
  --wrap="cd /u/yzhang116/field_level_PNG && source env/setup_env.sh && unset JAX_PLATFORMS && $*"
