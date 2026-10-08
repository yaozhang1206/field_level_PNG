cd /u/yzhang116/field_level_PNG && source env/setup_env.sh && unset JAX_PLATFORMS
echo "##### marginal 80 QSO+LRG"; python scripts/run_fisher_marginal.py --res 80 --nreal 1 --probes 6 --tracers QSO+LRG --tag marginal_res80_p1.6
echo "##### schur 80 p1.6";  python scripts/run_fisher_field.py --res 80 --nreal 4 --tracers QSO+LRG,QSO,LRG
echo "##### marginal 80 QSO, LRG"; python scripts/run_fisher_marginal.py --res 80 --nreal 1 --probes 6 --tracers QSO,LRG --tag marginal_res80_p1.6_single
echo "##### marginal 80 p1.0"; python scripts/run_fisher_marginal.py --res 80 --nreal 1 --probes 6 --p_qso 1.0 --tracers QSO+LRG
echo "##### done"
