export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
sudo /home/qirico/miniconda3/envs/hilserl3/bin/python ../../../train_rlif.py "$@" \
    --exp_name=cube_reach3 \
    --checkpoint_path=debug_rlif_2 \
    --demo_path=/home/qirico/Desktop/All-Weird/Human-Interventions/jax-hitl-hil-serl/examples/experiments/cube_reach3/side_only/demo_data/cube_reach3_20_demos_2025-04-22_20-19-06.pkl \
    --learner \
    --save_video \
    --method=rlif \
