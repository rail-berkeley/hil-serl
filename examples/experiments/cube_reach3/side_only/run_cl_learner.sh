export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
sudo /home/qirico/miniconda3/envs/hilserl/bin/python ../../../train_rlif.py "$@" \
    --exp_name=cube_reach3 \
    --checkpoint_path=debug_cl_1 \
    --demo_path=/home/qirico/Desktop/All-Weird/Human-Interventions/jax-hitl-hil-serl/examples/experiments/cube_reach3/side_only/demo_data/cube_reach3_20_demos_2025-04-16_12-07-35.pkl \
    --learner \
    --save_video \
    --method=cl \
