export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
/home/qirico/miniconda3/envs/hilserl/bin/python ../../train_rlif.py "$@" \
    --exp_name=cube_reach3 \
    --checkpoint_path=../../experiments/cube_reach3/debug_rlif \
    --actor \
    --eval_checkpoint_step=100000 \
    --eval_n_trajs=10 \
