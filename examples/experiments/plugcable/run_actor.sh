export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
export WANDB_ENTITY=724183484zz@gmail.com
export WANDB_API_KEY=cbacbd4d69837b2f36b650e8682607b8910fa76c
python ../../train_rlpd.py "$@" \
    --exp_name=plug_cable \
    --checkpoint_path=first_cable\
    --demo_path=... \
    --actor \
    --eval_checkpoint_step=45000 \
    --eval_n_trajs=100 \


