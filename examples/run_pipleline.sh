export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8 && \
export WANDB_ENTITY=zjq19990202@gmail.com
export WANDB_API_KEY=0a88034024ec899c79a9809b84c3a4f01c8b191d
python ./rl_pipleline.py "$@" \
    --exp_name=plug_cable --exp_name=plug_usb \
    --checkpoint_path=thirdplug --checkpoint_path=secondplug \
    --demo_path=... \
    --actor \
    --eval_checkpoint_step=64000 --eval_checkpoint_step=50000 \
    --eval_n_trajs=10


