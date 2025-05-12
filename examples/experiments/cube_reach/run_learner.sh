export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
/home/jacob/miniforge3/envs/serl2/bin/python ../../train_rlpd.py "$@" \
    --exp_name=cube_reach \
    --checkpoint_path=../../experiments/cube_reach/debug \
    --demo_path=/home/jacob/hil-serl/examples/experiments/cube_reach/demo_data/cube_reach_20_demos_2025-03-30_15-16-49.pkl \
    --learner \
    --save_video \