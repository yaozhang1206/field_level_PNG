cd /u/yzhang116/field_level_PNG && source env/setup_env.sh && unset JAX_PLATFORMS
echo "##### test_forward_model"; python test/test_forward_model.py
echo "##### test_fisher_field_linear"; NREAL=2 python test/test_fisher_field_linear.py
