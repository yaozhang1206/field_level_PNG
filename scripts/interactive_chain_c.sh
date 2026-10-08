cd /u/yzhang116/field_level_PNG && source env/setup_env.sh && unset JAX_PLATFORMS
echo "##### marginal 160 QSO+LRG (resume)"; python scripts/run_fisher_marginal.py --res 160 --nreal 1 --probes 12 --tracers QSO+LRG --tag marginal_res160_p1.6 --resume
echo "##### linear model 80 QSO+LRG"; python scripts/run_fisher_marginal.py --res 80 --nreal 1 --probes 24 --tracers QSO+LRG --linear --tag marginal_res80_p1.6_linear --resume
echo "##### done"
