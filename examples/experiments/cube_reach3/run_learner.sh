export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
/home/qirico/miniconda3/envs/hilserl/bin/python ../../train_rlif.py "$@" \
    --exp_name=cube_reach3 \
    --checkpoint_path=../../experiments/cube_reach3/debug_rlif \
    --demo_path=/home/jacob/Desktop/rico/jax-hitl-hil-serl/examples/experiments/cube_reach2/demo_data/cube_reach2_20_demos_2025-04-04_10-06-29.pkl \
    --learner \
    --save_video \