export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
export WANDB_ENTITY=zjq19990202@gmail.com
export WANDB_API_KEY=0a88034024ec899c79a9809b84c3a4f01c8b191d
python ../../train_rlpd.py "$@" \
    --exp_name=plug_usb \
    --checkpoint_path=first_plug\
    --demo_path=... \
    --actor \
    # --eval_checkpoint_step=63000 \
    # --eval_n_trajs=100 \

    

