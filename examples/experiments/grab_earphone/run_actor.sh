export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
export WANDB_ENTITY=724183484zz@gmail.com
export WANDB_API_KEY=cbacbd4d69837b2f36b650e8682607b8910fa76c
python ../../train_rlpd.py "$@" \
    --exp_name=grab_earphone \
    --checkpoint_path=first_change_camera_position \
    --demo_path=... \
    --actor \
    --eval_checkpoint_step=36000 \
    --eval_n_trajs=10 \


