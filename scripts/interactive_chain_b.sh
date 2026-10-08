cd /u/yzhang116/field_level_PNG && source env/setup_env.sh && unset JAX_PLATFORMS
echo "##### extend probes 80 QSO+LRG"; python scripts/run_fisher_marginal.py --res 80 --nreal 1 --probes 24 --tracers QSO+LRG --tag marginal_res80_p1.6 --resume
echo "##### extend probes 80 single"; python scripts/run_fisher_marginal.py --res 80 --nreal 1 --probes 24 --tracers QSO,LRG --tag marginal_res80_p1.6_single --resume
echo "##### extend probes 80 p1.0"; python scripts/run_fisher_marginal.py --res 80 --nreal 1 --probes 16 --p_qso 1.0 --tracers QSO+LRG --resume
echo "##### second realisation 80 QSO+LRG"; python scripts/run_fisher_marginal.py --res 80 --nreal 2 --probes 12 --tracers QSO+LRG --tag marginal_res80_p1.6 --resume
echo "##### marginal 160 QSO+LRG"; python scripts/run_fisher_marginal.py --res 160 --nreal 1 --probes 8 --tracers QSO+LRG --tag marginal_res160_p1.6 --resume
echo "##### done"
