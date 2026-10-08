# Source this before running anything in this repo (Delta AI).
#   source env/setup_env.sh
# Uses the user's existing `pm` conda env (jax 0.6.2 + CUDA12 plugin, discodj 0.0.2).
module load cudatoolkit/25.5_12.9 2>/dev/null
export LD_LIBRARY_PATH=/sw/user/gsl/lib:$LD_LIBRARY_PATH   # discodj_native needs libgsl.so.28
export PATH=/u/yzhang116/.conda/envs/pm/bin:$PATH
# On login nodes there is no GPU; silence the CUDA plugin error:
if ! nvidia-smi -L >/dev/null 2>&1; then export JAX_PLATFORMS=cpu; fi
export DESI_DR1_DIR=/work/hdd/bdne/yzhang116/desi_dr1
# Login nodes cap the number of threads per user (cgroup pids.max=1000); keep BLAS/XLA modest there.
if [ -z "$SLURM_JOB_ID" ]; then
  export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
  export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4 ${XLA_FLAGS}"
fi
