# sudo /home/qirico/miniconda3/envs/hilserl/bin/python ../../../record_success_fail.py --exp_name=cube_reach3
# sudo /home/qirico/miniconda3/envs/hilserl/bin/python ../../../train_reward_classifier.py --exp_name=cube_reach3
# sudo /home/qirico/miniconda3/envs/hilserl/bin/python ../../../record_demos.py --exp_name=cube_reach3 --successes_needed=20

# sudo /home/qirico/miniconda3/envs/hilserl/bin/python ../../../train_bc.py \
#     --exp_name=cube_reach3 \
#     --bc_checkpoint_path=debug_bc \
#     --train_steps=100000

# sudo /home/qirico/miniconda3/envs/hilserl/bin/python ../../../train_bc.py \
#     --exp_name=cube_reach3 \
#     --bc_checkpoint_path=/home/qirico/Desktop/All-Weird/Human-Interventions/jax-hitl-hil-serl/examples/experiments/cube_reach3/side_only/debug_bc/checkpoint_99990 \
#     --eval_n_trajs=20

