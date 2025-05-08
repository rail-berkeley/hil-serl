XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 /home/qirico/miniconda3/envs/hilserl3/bin/python ../../train_rlif.py "$@" \
    --exp_name=franka_sim \
    --checkpoint_path=debug_soft_cl_cta5_utd1_nostateconstraint \
    --demo_path=/home/qirico/Desktop/All-Weird/Human-Interventions/jax-hitl-hil-serl/examples/experiments/franka_sim/demo_data/franka_sim_20_demos_2025-04-25_21-16-29.pkl \
    --demo_path=/home/qirico/Desktop/All-Weird/Human-Interventions/jax-hitl-hil-serl/examples/experiments/franka_sim/demo_data/franka_sim_30_demos_2025-04-27_21-13-05.pkl \
    --learner \
    --method=soft_cl \


# XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 /home/qirico/miniconda3/envs/hilserl3/bin/python ../../train_rlif.py "$@" \
#     --exp_name=franka_sim \
#     --checkpoint_path=debug_soft_cl_cta2_state_fixed_3 \
#     --demo_path=/home/qirico/Desktop/All-Weird/Human-Interventions/jax-hitl-hil-serl/examples/experiments/franka_sim/demo_data/franka_sim_20_fixed.pkl \
#     --learner \
#     --method=soft_cl \
#     --state_based \
