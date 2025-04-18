export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
sudo /home/qirico/miniconda3/envs/hilserl/bin/python ../../../train_rlpd.py "$@" \
    --exp_name=cube_reach3 \
    --checkpoint_path=debug_rlif \
    --actor \
    --eval_checkpoint_step=40000 \
    --eval_n_trajs=20 \
