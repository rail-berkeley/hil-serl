export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
/home/jacob/miniforge3/envs/serl2/bin/python ../../train_rlif.py "$@" \
    --exp_name=cube_reach2 \
    --checkpoint_path=../../experiments/cube_reach2/debug_rlif \
    --actor \
    --eval_checkpoint_step=100000 \
    --eval_n_trajs=10 \
